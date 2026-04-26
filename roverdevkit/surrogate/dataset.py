"""Parallel Phase-2 dataset builder.

Takes a list of :class:`LHSSample` from :mod:`.sampling`, runs
:func:`roverdevkit.mission.evaluator.evaluate_verbose` on each
(optionally in parallel), flattens the results into a pandas DataFrame
with a stable column schema, and writes Parquet.

Column schema (documented in ``data/analytical/SCHEMA.md``):

- ``sample_index`` / ``split`` / ``stratum_id`` / ``fidelity`` /
  ``status`` — dataset metadata.
- ``design_*`` — 12 design-vector inputs.
- ``scenario_*`` — scenario inputs (family + jittered mission params +
  jittered Bekker soil parameters).
- ``range_km`` / ``energy_margin_pct`` / ``energy_margin_raw_pct`` /
  ``slope_capability_deg`` / ``total_mass_kg`` / ``peak_motor_torque_nm``
  / ``sinkage_max_m`` / ``motor_torque_ok`` — :class:`MissionMetrics`
  targets the surrogate predicts.
- ``stat_*`` — aggregate statistics (mean / p95 / max / final) from the
  :class:`TraverseLog` time series for the Week-7.5 SCM-correction
  gate and surrogate diagnostics.

Thermal scope (v2)
------------------
The system-level evaluator still computes ``thermal_survival`` as a
diagnostic but the surrogate **does not** consume or predict it.
Rationale: the current mass model treats RHU power and MLI quality as
free, so thermal_survival reduces to a near-trivial gate ("did you add
an RHU?") with no design trade-off. Including it as a target would
add a degenerate column and dilute the headline R²/AUC. The Pragyan
vs. Yutu-2 thermal distinction is preserved in the Week-5 validation
harness (``roverdevkit.validation.rover_registry``); a future mass-model
upgrade that charges RHU/MLI mass will let thermal re-enter the
surrogate as a real Pareto trade.

Failure handling
----------------
If ``evaluate_verbose`` raises, the row is kept with ``status`` set to
the exception class name and all numeric columns set to NaN. The full
exception message is logged to stderr. This lets dataset builds
complete even when a handful of pathological LHS corners trip the
physics, at the cost of a small NaN fraction the baseline trainer
later filters out.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from roverdevkit.mission.evaluator import DetailedEvaluation, evaluate_verbose
from roverdevkit.mission.traverse_sim import TraverseLog
from roverdevkit.schema import DesignVector, MissionMetrics, MissionScenario
from roverdevkit.surrogate.sampling import LHSSample

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "v4"
"""Bump when the column schema or training distribution changes so
downstream code can detect stale Parquet files. Written into Parquet
file-level metadata.

History
-------
- v1 (Week 6 step 1, retired): included ``thermal_survival`` as a
  feasibility target.
- v2 (Week 6 step 2): drop ``thermal_survival`` from the schema; the
  evaluator still computes it but the surrogate does not consume it.
  See module docstring for the scoping rationale.
- v3 (Week 6 step 4 follow-up, 2026-04-25): widened LHS bounds on
  ``wheel_width_m`` (0.15 -> 0.20), ``grouser_height_m`` (0.012 ->
  0.020), and ``chassis_mass_kg`` (35 -> 50) so the flown / design-
  target lunar micro-rovers in
  :mod:`roverdevkit.validation.rover_registry` (Yutu-2, MoonRanger,
  Rashid-1) sit inside the surrogate's training support rather than
  at corner points. Column schema is byte-identical to v2; the bump
  signals the changed training distribution so a v2-trained surrogate
  isn't silently reused on a v3 dataset.
- v4 (Week 8 step 1, 2026-04-26): rebuild with the trained wheel-level
  SCM correction (``use_scm_correction=True``) composed into the
  analytical evaluator. Column schema is byte-identical to v3; the
  bump signals the corrected mobility physics so a v3-trained
  surrogate isn't silently reused on v4 data. Promotion was gated
  by the Week-7.5 sign-flip gate (``reports/week7_5_gate``) and the
  Week-7.7 BW-vs-SCM-direct bake-off (``reports/week7_7_bakeoff``)."""

DEFAULT_FIDELITY = "analytical"


# ---------------------------------------------------------------------------
# Per-worker correction cache (loaded once per pool worker, not per sample)
# ---------------------------------------------------------------------------

_WORKER_USE_SCM_CORRECTION: bool = False
_WORKER_CORRECTION: Any = None  # WheelLevelCorrection | None, set in _init_worker


def _init_worker(use_scm_correction: bool) -> None:
    """Pool initializer: cache the SCM correction artifact per worker.

    Loaded once per spawned process so each ``_evaluate_sample`` call
    skips the joblib disk read. Falls back gracefully (warn-once,
    BW-only) if the artifact is missing — same contract as
    :func:`evaluate_verbose`.
    """
    global _WORKER_USE_SCM_CORRECTION, _WORKER_CORRECTION
    _WORKER_USE_SCM_CORRECTION = use_scm_correction
    if use_scm_correction:
        from roverdevkit.terramechanics.correction_model import (
            DEFAULT_CORRECTION_PATH,
            load_correction_or_none,
        )

        _WORKER_CORRECTION = load_correction_or_none(DEFAULT_CORRECTION_PATH, on_missing="warn")
    else:
        _WORKER_CORRECTION = None


# ---------------------------------------------------------------------------
# Row flattening
# ---------------------------------------------------------------------------


def _flatten_design(design: DesignVector) -> dict[str, Any]:
    return {
        "design_wheel_radius_m": design.wheel_radius_m,
        "design_wheel_width_m": design.wheel_width_m,
        "design_grouser_height_m": design.grouser_height_m,
        "design_grouser_count": int(design.grouser_count),
        "design_n_wheels": int(design.n_wheels),
        "design_chassis_mass_kg": design.chassis_mass_kg,
        "design_wheelbase_m": design.wheelbase_m,
        "design_solar_area_m2": design.solar_area_m2,
        "design_battery_capacity_wh": design.battery_capacity_wh,
        "design_avionics_power_w": design.avionics_power_w,
        "design_nominal_speed_mps": design.nominal_speed_mps,
        "design_drive_duty_cycle": design.drive_duty_cycle,
    }


def _flatten_scenario(scenario: MissionScenario, sample: LHSSample) -> dict[str, Any]:
    # The scenario's soil_simulant name is the *family nominal*; the
    # actual jittered Bekker parameters the evaluator used come from
    # ``sample.soil`` and are recorded as scenario_soil_* below.
    return {
        "scenario_family": sample.scenario_family,
        "scenario_name": scenario.name,
        "scenario_latitude_deg": scenario.latitude_deg,
        "scenario_traverse_distance_m": scenario.traverse_distance_m,
        "scenario_terrain_class": scenario.terrain_class,
        "scenario_soil_simulant": scenario.soil_simulant,
        "scenario_mission_duration_earth_days": scenario.mission_duration_earth_days,
        "scenario_max_slope_deg": scenario.max_slope_deg,
        "scenario_sun_geometry": scenario.sun_geometry,
        "scenario_soil_n": sample.soil.n,
        "scenario_soil_k_c": sample.soil.k_c,
        "scenario_soil_k_phi": sample.soil.k_phi,
        "scenario_soil_cohesion_kpa": sample.soil.cohesion_kpa,
        "scenario_soil_friction_angle_deg": sample.soil.friction_angle_deg,
        "scenario_soil_shear_modulus_k_m": sample.soil.shear_modulus_k_m,
    }


def _flatten_metrics(metrics: MissionMetrics) -> dict[str, Any]:
    # ``metrics.thermal_survival`` is intentionally not flattened into
    # the dataset; see the module-level "Thermal scope" docstring.
    return {
        "range_km": metrics.range_km,
        "energy_margin_pct": metrics.energy_margin_pct,
        "energy_margin_raw_pct": metrics.energy_margin_raw_pct,
        "slope_capability_deg": metrics.slope_capability_deg,
        "total_mass_kg": metrics.total_mass_kg,
        "peak_motor_torque_nm": metrics.peak_motor_torque_nm,
        "sinkage_max_m": metrics.sinkage_max_m,
        "motor_torque_ok": bool(metrics.motor_torque_ok),
    }


def _array_stats(arr: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, p95, max) of an array. NaN-safe on empty."""
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    return (
        float(np.mean(arr)),
        float(np.percentile(arr, 95.0)),
        float(np.max(arr)),
    )


def _flatten_log_stats(log: TraverseLog) -> dict[str, Any]:
    p_in_mean, p_in_p95, p_in_max = _array_stats(log.power_in_w)
    p_out_mean, p_out_p95, p_out_max = _array_stats(log.power_out_w)
    p_mob_mean, p_mob_p95, p_mob_max = _array_stats(log.mobility_power_w)
    slip_mean, slip_p95, slip_max = _array_stats(np.abs(log.slip))
    sink_mean, sink_p95, _ = _array_stats(log.sinkage_m)
    tq_mean, tq_p95, _ = _array_stats(np.abs(log.wheel_torque_nm))
    sun_mean, _, sun_max = _array_stats(log.sun_elevation_deg)
    soc_final = float(log.state_of_charge[-1]) if log.state_of_charge.size else float("nan")
    soc_min = float(np.min(log.state_of_charge)) if log.state_of_charge.size else float("nan")
    return {
        "stat_power_in_mean_w": p_in_mean,
        "stat_power_in_p95_w": p_in_p95,
        "stat_power_in_max_w": p_in_max,
        "stat_power_out_mean_w": p_out_mean,
        "stat_power_out_p95_w": p_out_p95,
        "stat_power_out_max_w": p_out_max,
        "stat_mobility_power_mean_w": p_mob_mean,
        "stat_mobility_power_p95_w": p_mob_p95,
        "stat_mobility_power_max_w": p_mob_max,
        "stat_slip_mean": slip_mean,
        "stat_slip_p95": slip_p95,
        "stat_slip_max": slip_max,
        "stat_sinkage_mean_m": sink_mean,
        "stat_sinkage_p95_m": sink_p95,
        "stat_wheel_torque_mean_nm": tq_mean,
        "stat_wheel_torque_p95_nm": tq_p95,
        "stat_sun_elevation_mean_deg": sun_mean,
        "stat_sun_elevation_max_deg": sun_max,
        "stat_soc_final": soc_final,
        "stat_soc_min": soc_min,
        "stat_rover_stalled": bool(log.rover_stalled),
        "stat_battery_floored": bool(log.battery_floored),
        "stat_reached_distance": bool(log.reached_distance),
        "stat_terminated_reason": log.terminated_reason,
    }


# The full list of numeric-metric/stat columns. Used to populate NaN
# values on failed rows and as the canonical output-column set.
_NUMERIC_METRIC_COLS: tuple[str, ...] = (
    "range_km",
    "energy_margin_pct",
    "energy_margin_raw_pct",
    "slope_capability_deg",
    "total_mass_kg",
    "peak_motor_torque_nm",
    "sinkage_max_m",
)

_BOOL_METRIC_COLS: tuple[str, ...] = ("motor_torque_ok",)

_STAT_NUMERIC_COLS: tuple[str, ...] = (
    "stat_power_in_mean_w",
    "stat_power_in_p95_w",
    "stat_power_in_max_w",
    "stat_power_out_mean_w",
    "stat_power_out_p95_w",
    "stat_power_out_max_w",
    "stat_mobility_power_mean_w",
    "stat_mobility_power_p95_w",
    "stat_mobility_power_max_w",
    "stat_slip_mean",
    "stat_slip_p95",
    "stat_slip_max",
    "stat_sinkage_mean_m",
    "stat_sinkage_p95_m",
    "stat_wheel_torque_mean_nm",
    "stat_wheel_torque_p95_nm",
    "stat_sun_elevation_mean_deg",
    "stat_sun_elevation_max_deg",
    "stat_soc_final",
    "stat_soc_min",
)

_STAT_BOOL_COLS: tuple[str, ...] = (
    "stat_rover_stalled",
    "stat_battery_floored",
    "stat_reached_distance",
)


def _nan_outputs() -> dict[str, Any]:
    """Build the output columns dict for a failed evaluation."""
    out: dict[str, Any] = {col: float("nan") for col in _NUMERIC_METRIC_COLS}
    out.update({col: False for col in _BOOL_METRIC_COLS})
    out.update({col: float("nan") for col in _STAT_NUMERIC_COLS})
    out.update({col: False for col in _STAT_BOOL_COLS})
    out["stat_terminated_reason"] = "evaluator_error"
    return out


# ---------------------------------------------------------------------------
# Per-sample worker (must be module-level and picklable for multiprocessing)
# ---------------------------------------------------------------------------


def _evaluate_sample(sample: LHSSample) -> dict[str, Any]:
    """Run one sample through the evaluator and return a flattened row.

    Catches all exceptions from the physics layer and records them as
    ``status`` rather than failing the whole batch.
    """
    row: dict[str, Any] = {
        "sample_index": sample.sample_index,
        "split": sample.split,
        "stratum_id": sample.stratum_id,
        "fidelity": DEFAULT_FIDELITY,
    }
    row.update(_flatten_design(sample.design))
    row.update(_flatten_scenario(sample.scenario, sample))

    try:
        result: DetailedEvaluation = evaluate_verbose(
            sample.design,
            sample.scenario,
            soil_override=sample.soil,
            correction=_WORKER_CORRECTION,
        )
        row.update(_flatten_metrics(result.metrics))
        row.update(_flatten_log_stats(result.log))
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 -- catch-all is intentional; see docstring
        logger.warning(
            "evaluate_verbose failed on sample %d (%s): %s",
            sample.sample_index,
            sample.scenario_family,
            exc,
        )
        row.update(_nan_outputs())
        row["status"] = type(exc).__name__
    return row


# ---------------------------------------------------------------------------
# Dataset-level metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetMetadata:
    """Human- and machine-readable metadata written to Parquet file footer.

    Attributes are stringified into the Parquet's ``schema.metadata``
    dict so ``pq.read_metadata(path).metadata`` recovers the build
    provenance without deserialising the entire file.
    """

    schema_version: str = SCHEMA_VERSION
    sampler_seed: int = 0
    n_per_scenario: int = 0
    scenario_families: tuple[str, ...] = field(default_factory=tuple)
    val_frac: float = 0.1
    test_frac: float = 0.1
    fidelity: str = DEFAULT_FIDELITY
    evaluator_version: str = "0.1.0"
    use_scm_correction: bool = False
    built_at_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    notes: str = ""

    def to_parquet_metadata(self) -> dict[bytes, bytes]:
        return {
            b"schema_version": self.schema_version.encode(),
            b"sampler_seed": str(self.sampler_seed).encode(),
            b"n_per_scenario": str(self.n_per_scenario).encode(),
            b"scenario_families": ",".join(self.scenario_families).encode(),
            b"val_frac": str(self.val_frac).encode(),
            b"test_frac": str(self.test_frac).encode(),
            b"fidelity": self.fidelity.encode(),
            b"evaluator_version": self.evaluator_version.encode(),
            b"use_scm_correction": str(self.use_scm_correction).encode(),
            b"built_at_utc": self.built_at_utc.encode(),
            b"notes": self.notes.encode(),
        }


# ---------------------------------------------------------------------------
# Public builder API
# ---------------------------------------------------------------------------


def build_dataset(
    samples: Sequence[LHSSample] | Iterable[LHSSample],
    *,
    n_workers: int | None = None,
    chunksize: int = 32,
    progress: bool = True,
    use_scm_correction: bool = False,
) -> pd.DataFrame:
    """Evaluate ``samples`` in parallel and return a flattened DataFrame.

    Parameters
    ----------
    samples
        Iterable from :func:`.sampling.generate_samples`. Materialised
        to a list internally so the total count is known for the
        progress bar.
    n_workers
        Worker-process count. ``None`` (default) uses
        ``os.cpu_count() - 1`` capped at 1. Pass ``1`` for serial
        execution (easier to debug / useful for small smoke tests).
    chunksize
        ``multiprocessing.Pool.imap_unordered`` chunk size. Larger
        values reduce IPC overhead but increase tail latency.
    progress
        If True and ``tqdm`` is importable, display a progress bar.

    Returns
    -------
    pandas.DataFrame
        One row per sample, ordered by ``sample_index``. Failed rows
        are preserved with NaN numeric columns and ``status`` set to
        the exception class name.
    """
    sample_list: list[LHSSample] = list(samples)
    if not sample_list:
        raise ValueError("No samples supplied to build_dataset.")

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    _iter: Iterable[dict[str, Any]]
    if n_workers == 1:
        # Serial path: initialise the worker cache in-process once.
        _init_worker(use_scm_correction)
        _iter = (_evaluate_sample(s) for s in sample_list)
    else:
        pool = mp.get_context("spawn").Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(use_scm_correction,),
        )
        _iter = pool.imap_unordered(_evaluate_sample, sample_list, chunksize=chunksize)

    rows: list[dict[str, Any]] = []
    wrapped = _maybe_wrap_progress(_iter, total=len(sample_list), enabled=progress)
    try:
        for row in wrapped:
            rows.append(row)
    finally:
        if n_workers != 1:
            pool.close()
            pool.join()

    rows.sort(key=lambda r: r["sample_index"])
    df = pd.DataFrame(rows)
    return _coerce_dtypes(df)


def _maybe_wrap_progress(
    it: Iterable[dict[str, Any]],
    *,
    total: int,
    enabled: bool,
) -> Iterable[dict[str, Any]]:
    if not enabled:
        return it
    try:
        from tqdm import tqdm
    except ImportError:
        return it
    return tqdm(it, total=total, desc="evaluate", unit="sample")


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Set stable column dtypes independent of row order / NaN locations."""
    if "split" in df:
        df["split"] = df["split"].astype("category")
    if "scenario_family" in df:
        df["scenario_family"] = df["scenario_family"].astype("category")
    if "scenario_name" in df:
        df["scenario_name"] = df["scenario_name"].astype("category")
    if "scenario_terrain_class" in df:
        df["scenario_terrain_class"] = df["scenario_terrain_class"].astype("category")
    if "scenario_soil_simulant" in df:
        df["scenario_soil_simulant"] = df["scenario_soil_simulant"].astype("category")
    if "scenario_sun_geometry" in df:
        df["scenario_sun_geometry"] = df["scenario_sun_geometry"].astype("category")
    if "fidelity" in df:
        df["fidelity"] = df["fidelity"].astype("category")
    if "status" in df:
        df["status"] = df["status"].astype("category")
    if "stat_terminated_reason" in df:
        df["stat_terminated_reason"] = df["stat_terminated_reason"].astype("category")
    return df


def write_parquet(
    df: pd.DataFrame,
    path: str | Path,
    *,
    metadata: DatasetMetadata | None = None,
    compression: str = "zstd",
) -> Path:
    """Write ``df`` to Parquet with dataset-level metadata.

    Uses zstd compression by default (smaller & faster than snappy on
    tabular numeric data).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if metadata is not None:
        existing = dict(table.schema.metadata or {})
        existing.update(metadata.to_parquet_metadata())
        table = table.replace_schema_metadata(existing)
    pq.write_table(table, out, compression=compression)
    return out


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Load a dataset Parquet back into a DataFrame with categorical dtypes."""
    df = pq.read_table(path).to_pandas()
    return _coerce_dtypes(df)


def read_parquet_metadata(path: str | Path) -> dict[str, str]:
    """Return the string-valued file-level metadata dict."""
    md = pq.read_metadata(path).metadata or {}
    return {k.decode(): v.decode() for k, v in md.items()}


# ---------------------------------------------------------------------------
# Convenience end-to-end helper
# ---------------------------------------------------------------------------


def build_and_write(
    samples: Sequence[LHSSample],
    out_path: str | Path,
    *,
    metadata: DatasetMetadata | None = None,
    build_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Run the full build → write pipeline in one call. Returns (df, path)."""
    build_kwargs = build_kwargs or {}
    df = build_dataset(samples, **build_kwargs)
    path = write_parquet(df, out_path, metadata=metadata)
    return df, path


__all__ = [
    "DEFAULT_FIDELITY",
    "SCHEMA_VERSION",
    "DatasetMetadata",
    "build_and_write",
    "build_dataset",
    "read_parquet",
    "read_parquet_metadata",
    "write_parquet",
]
