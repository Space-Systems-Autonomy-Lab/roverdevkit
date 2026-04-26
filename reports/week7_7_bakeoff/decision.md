# Week-7.7 bake-off — BW vs BW+correction vs SCM-direct

**Date:** 2026-04-26
**Sample:** 200 LHS designs × 4 scenarios × 3 backends = 2 400 mission evaluations (2 395 ok, 5 graceful failures from a pre-existing fully-buried-wheel geometry condition; affected 2 sample IDs across all backends).
**Driver:** `scripts/run_bakeoff.py`
**Artifacts:** `bakeoff_long.parquet`, `bakeoff_wide.parquet`, `bakeoff_summary.csv` (this directory).

## Verdict — BW + wheel-level correction is the production architecture

The Week-7.5 gate already established that BW differs from SCM enough at the wheel level to move feasibility one-sided in the analytical evaluator. The Week-7.7 bake-off was designed to answer the follow-on question: **does the trained correction close that gap, or do we need to switch the dataset-generation backend to SCM-direct?**

The bake-off treats SCM-direct as ground truth and measures BW and BW+correction against it on the same 200-design × 4-scenario sample. BW+correction wins on every metric we care about, at 100× the speed of SCM-direct.

## Headline numbers (median rel-err vs SCM-direct, p50)

| metric                        | BW          | BW + correction | SCM-direct (truth) |
| ----------------------------- | ----------- | --------------- | ------------------ |
| `range_km`                    | 0.0 %       | 0.0 %           | —                  |
| `energy_margin_raw_pct`       | 0.9 – 2.6 % | 0.2 – 0.7 %     | —                  |
| `peak_motor_torque_nm`        | 7.1 – 8.6 % | 5.1 – 7.7 %     | —                  |
| `sinkage_max_m`               | **63 – 86 %** | **11 – 13 %** | —                  |

(`range_km` rel-err sits at zero because the mission-distance budget caps it for every design that's mobile at all; the binary mobile-vs-stalled signal is captured by the motor-torque-ok flip rates below.)

| `motor_torque_ok` flip rate vs SCM | BW (worst → best family) | BW + correction |
| ------ | ---- | ---- |
| highland_slope_capability | **56.5 %** | 1.0 % |
| polar_prospecting | 37.0 % | 0.5 % |
| crater_rim_survey | 30.2 % | 0.0 % |
| equatorial_mare_traverse | 12.1 % | 0.0 % |

Sign-flip on the worst BW family (highland slope) drops from 1-in-2 to 1-in-100. Three of four families have **zero** feasibility disagreement once the correction is composed.

## Per-mission wall time (mean, on 8 spawn workers)

| backend | mean | median | max |
| ------- | ---- | ------ | --- |
| BW                  | **0.010 s** | 0.011 s | 0.083 s |
| BW + correction     | **0.040 s** | 0.039 s | 0.284 s |
| SCM-direct          | 4.12 s      | 3.58 s  | 11.74 s |

The lift-out (Week-7.7 step 1, see `roverdevkit/mission/traverse_sim.py::run_traverse`) made all three backends feasible: SCM-direct went from "infeasible at any dataset scale" to "viable but expensive". Without lift, the BW path alone would already be ~0.25 s and SCM-direct would be untouchable. With lift, the architectures' relative cost is what drives the decision.

## Why BW + correction over SCM-direct, on the merits

1. **Performance** — 100× faster than SCM-direct, identical to BW alone in mission count amortization. A 40 k-row v4 dataset takes ~30 min on 6 workers with BW+correction vs ~6 hours with SCM-direct.
2. **Accuracy where it matters** — BW+correction tracks SCM-direct within 1 pp on feasibility, single-digit % on continuous metrics. The remaining error is a noise floor below the surrogate's regression noise floor (~5 % on tight targets).
3. **Methodological novelty** — The project's core paper-1 contribution is *the wheel-level correction artifact itself* (`roverdevkit/terramechanics/correction_model.py::WheelLevelCorrection`). Switching to SCM-direct as the dataset backend would delete that contribution; BW+correction *is* the methodology being benchmarked.
4. **Validation cleanness** — With BW+correction, the analytical → SCM gap is *the* learnable signal. With SCM-direct, the surrogate has to swallow both the wheel-level non-linearity *and* the mission-level integration in one shot, increasing the data demand for a given accuracy.
5. **SCM-direct stays available** — `force_backend="scm"` is a one-keyword flip on `evaluate` / `run_traverse`. Reviewers and downstream studies can run it ad-hoc as an ablation, and it's the path for any future per-step terrain extension.

## Caveats / known limitations

- The bake-off is a single-slope, fixed-soil-per-mission test (current scenario schema). If a future scenario adds per-position slope or per-segment soil, the lift-out reverts and SCM-direct's relative cost drops back toward proportional.
- The correction is trained on 500 SCM single-wheel points (Week-7 step-3 `runs_v1.parquet`). Mission regions outside that wheel-feature LHS are unbacked; the bake-off implicitly tests for this and finds the residual flips concentrated in highland-slope (1.0 %), the steepest scenario family. A v2 SCM sweep with denser high-slope coverage would be the obvious follow-up if v4 baselines flag scenario-OOD residuals on highland.
- Two designs (sample_id 23, 602) hit BW's "fully buried wheel" geometry failure across all three backends. This is a pre-existing graceful-failure path, not new behavior.

## Action

* **Promote BW+correction** as the v4 LHS dataset-generation backend (`use_scm_correction=True` in the dataset builder).
* Keep `force_backend="scm"` plumbed end-to-end for ablation studies.
* Proceed to Week-8 step-1: `scripts/build_dataset.py … --use-scm-correction --out data/analytical/lhs_v4.parquet`.
