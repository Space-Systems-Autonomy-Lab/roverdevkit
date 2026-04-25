"""Train and score the Week-6 baseline surrogate matrix on a Parquet dataset.

Single canonical entry point for the Week-6 §6 step-4 acceptance run:
fit Ridge / RF / XGBoost per target, the joint MLP across all primary
targets, and LogReg / XGBoost feasibility classifiers; then score them
on the held-out test split (with a per-scenario-family breakdown) and
run the registry-rover Layer-1 sanity check.

Outputs (under ``--out-dir``):

- ``metrics_long.parquet`` — tidy long-format frame
  ``(algorithm, target, split, scenario_family, metric, value)``.
- ``acceptance_gate.csv`` — one row per ``(algorithm, target)`` with
  the plan's threshold, observed value, and pass/fail.
- ``registry_sanity.csv`` — predictions for Pragyan / Yutu-2 /
  Sojourner vs. the deterministic evaluator (Layer-1 truth).
- ``fit_seconds.csv`` — per-fit wall-clock for the writeup.

Examples
--------
::

    # Full 40k v1 acceptance run
    python scripts/run_baselines.py \\
        --dataset data/analytical/lhs_v1.parquet \\
        --out-dir reports/week6_baselines_v1

    # Fast pilot smoke (skip MLP, smaller forest)
    python scripts/run_baselines.py \\
        --dataset data/analytical/lhs_pilot.parquet \\
        --out-dir reports/week6_baselines_pilot \\
        --no-mlp
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from roverdevkit.surrogate.baselines import (
    acceptance_gate,
    evaluate_baselines,
    fit_baselines,
    predict_for_registry_rovers,
)
from roverdevkit.surrogate.dataset import read_parquet


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the Parquet dataset produced by scripts/build_dataset.py.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for the output reports (created if missing).",
    )
    p.add_argument("--seed", type=int, default=42, help="Estimator random_state.")
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Plumbed through to RF / XGBoost. -1 uses all cores.",
    )
    p.add_argument(
        "--no-mlp",
        action="store_true",
        help="Skip fitting the joint MLP. Useful for fast smokes.",
    )
    p.add_argument(
        "--no-registry-check",
        action="store_true",
        help="Skip the registry-rover Layer-1 sanity check.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("run_baselines")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log.info("loading dataset from %s", args.dataset)
    df = read_parquet(args.dataset)
    log.info(
        "loaded %d rows x %d cols; splits: %s",
        len(df),
        len(df.columns),
        df["split"].value_counts().to_dict() if "split" in df.columns else {},
    )

    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] == "val"]
    df_test = df[df["split"] == "test"]
    log.info("train=%d val=%d test=%d", len(df_train), len(df_val), len(df_test))

    # ----- fit ---------------------------------------------------------------
    t_fit = time.perf_counter()
    fitted = fit_baselines(
        df_train,
        fit_mlp=not args.no_mlp,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        verbose=True,
    )
    fit_elapsed = time.perf_counter() - t_fit
    log.info("fit complete in %.1f s", fit_elapsed)

    # ----- evaluate (val + test, with per-scenario-family breakdown) --------
    log.info("scoring val and test splits...")
    t_eval = time.perf_counter()
    val_metrics = evaluate_baselines(fitted, df_val, split_label="val")
    test_metrics = evaluate_baselines(fitted, df_test, split_label="test")
    train_metrics = evaluate_baselines(fitted, df_train, split_label="train")
    metrics = pd.concat([train_metrics, val_metrics, test_metrics], ignore_index=True)
    log.info("scoring done in %.1f s; %d metric rows", time.perf_counter() - t_eval, len(metrics))

    metrics_path = args.out_dir / "metrics_long.parquet"
    metrics.to_parquet(metrics_path, index=False)
    log.info("wrote %s (%d rows)", metrics_path, len(metrics))

    # ----- acceptance gate (test, overall) ----------------------------------
    gate = acceptance_gate(metrics, split="test", family="__all__")
    gate_path = args.out_dir / "acceptance_gate.csv"
    gate.to_csv(gate_path, index=False)
    log.info("wrote %s; passing rows: %d/%d", gate_path, int(gate["passes"].sum()), len(gate))
    print("\n=== Acceptance gate (test split, all families) ===", flush=True)
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(gate.to_string(index=False))

    # ----- compact summary table per (algorithm, target) on test ------------
    test_overall = metrics.query("split == 'test' and scenario_family == '__all__'")
    pivot = (
        test_overall.pivot_table(
            index=["algorithm", "target"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["target", "algorithm"])
    )
    pivot_path = args.out_dir / "test_summary.csv"
    pivot.to_csv(pivot_path, index=False)
    log.info("wrote %s", pivot_path)
    print("\n=== Per-(algorithm, target) test metrics ===", flush=True)
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(pivot.to_string(index=False))

    # ----- per-scenario breakdown on the primary metrics --------------------
    fam_rows = metrics.query(
        "split == 'test' and scenario_family != '__all__' and metric in ('r2', 'auc')"
    )
    fam_pivot = (
        fam_rows.pivot_table(
            index=["algorithm", "target", "metric"],
            columns="scenario_family",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["target", "metric", "algorithm"])
    )
    fam_pivot_path = args.out_dir / "test_per_family.csv"
    fam_pivot.to_csv(fam_pivot_path, index=False)
    log.info("wrote %s", fam_pivot_path)

    # ----- fit-time table ---------------------------------------------------
    fit_rows = [
        {"algorithm": k[0], "target": k[1], "fit_seconds": v} for k, v in fitted.fit_seconds.items()
    ]
    fit_df = pd.DataFrame(fit_rows).sort_values(["algorithm", "target"])
    fit_path = args.out_dir / "fit_seconds.csv"
    fit_df.to_csv(fit_path, index=False)
    log.info("wrote %s (%.1f s wall-clock total fit)", fit_path, fit_elapsed)

    # ----- registry rover Layer-1 sanity ------------------------------------
    if not args.no_registry_check:
        log.info("running registry-rover sanity check...")
        try:
            sanity = predict_for_registry_rovers(fitted)
            sanity_path = args.out_dir / "registry_sanity.csv"
            sanity.to_csv(sanity_path, index=False)
            log.info("wrote %s (%d rows)", sanity_path, len(sanity))
            # Compact view: per-rover R²-like sanity (median |abs_error| across algos)
            print("\n=== Registry-rover Layer-1 sanity ===", flush=True)
            summary = (
                sanity.assign(abs_pct=lambda d: 100 * d["rel_error"].abs())
                .groupby(["rover", "target"])["abs_pct"]
                .median()
                .unstack("target")
            )
            with pd.option_context("display.max_columns", None, "display.width", 160):
                print("Median |relative error| (%) across algorithms:")
                print(summary.round(2).to_string())
        except Exception as exc:  # pragma: no cover — diagnostic, not fatal
            log.warning("registry-rover sanity check failed: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
