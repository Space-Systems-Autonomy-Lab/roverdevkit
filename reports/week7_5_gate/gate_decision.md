# Week-7.5 gate decision

**Date:** 2026-04-25
**Sweep:** `data/scm/gate_eval_v1.parquet` — 200 paired evaluations
(50 designs × 4 scenario families × {BW, BW+correction})
**Correction artifact:** `data/scm/correction_v1.joblib`
**Gate criterion (project_plan.md §6 W7.5):**
median `|Δrange|/range > 10%` *or* feasibility/sign-flip rate `> 5%`
within any scenario family.

## Verdict: **GATE FIRES** — promote the wheel-level correction

The Bekker-Wong analytical evaluator and the BW + wheel-level-correction
evaluator disagree on the **feasibility frontier** of every scenario family
by more than the 5 % threshold, so the analytical-only surrogate would
mis-classify a substantial fraction of marginal designs. The wheel-level
correction stays composed inside `roverdevkit/mission/traverse_sim.py`
and the Phase-2 LHS dataset will be regenerated with
`use_scm_correction=True` (see project_plan.md Week-8 ordering).

## Evidence

### Per-family gate summary (`gate_summary.csv`)

| family | n | range_med_rel_err | feas_flip | em_sign_flip | range_gate | sign_gate |
|---|---|---|---|---|---|---|
| crater_rim_survey | 50 | 0.00 | 30.0% | 6.0% | False | **True** |
| equatorial_mare_traverse | 50 | 0.00 | 12.0% | 0.0% | False | **True** |
| highland_slope_capability | 50 | 0.00 | 68.0% | 2.0% | False | **True** |
| polar_prospecting | 50 | 0.00 | 34.0% | 2.0% | False | **True** |

### Direction of feasibility flips (`feasibility_direction.csv`)

| family | both_feas | bw_only_feas | scm_only_feas | neither_feas |
|---|---|---|---|---|
| crater_rim_survey | 34 | **0** | 15 | 1 |
| equatorial_mare_traverse | 44 | **0** | 6 | 0 |
| highland_slope_capability | 16 | **0** | 34 | 0 |
| polar_prospecting | 32 | **0** | 17 | 1 |

The flips are **entirely one-sided**: in every family, zero designs are
BW-only-feasible. BW is systematically more conservative than SCM at
the feasibility boundary, especially on highland slopes (68 % of the
50 designs flip from BW-stalled to SCM-mobile). This is consistent with
the Week-7 step-3 wheel-level finding that BW under-predicts drawbar
pull and over-predicts stall in high-slip / loose-soil regimes.

### Spearman ρ on mobility metrics (`spearman_mobility.csv`)

| family | sinkage_max | peak_torque | energy_margin_raw_pct |
|---|---|---|---|
| crater_rim_survey | 0.975 | 0.954 | 0.946 |
| equatorial_mare_traverse | 0.962 | 0.967 | 0.991 |
| highland_slope_capability | 0.950 | 0.957 | 0.827 |
| polar_prospecting | 0.957 | 0.947 | 0.870 |

Continuous-metric rankings are well preserved (ρ = 0.83-0.99) — the
correction shifts the feasibility frontier but does not reorder the
designs that survive in both modes. This is why `range_med_rel_err`
on the both-mobile subset is exactly zero (designs that complete the
mission saturate at the scenario `traverse_distance_m`); the gate
fires on the qualitative feasibility flip rather than on a quantitative
range delta.

## Action

1. **Composition layer is now production.** `traverse_sim.run_traverse`
   accepts a `correction` parameter; the evaluator opt-in is
   `use_scm_correction=True` (with graceful fallback to BW if the
   joblib is missing). No further code changes needed for the surrogate
   pipeline — the Week-8 LHS regeneration just sets the flag.
2. **No mission-level correction surrogate needed.** The wheel-level
   correction model is the runtime composition layer, applied at every
   evaluator invocation. A separate `Δmetric = f(design, scenario)`
   surrogate (the §6 W7.5 alternate path) is *not* commissioned because
   the wheel-level correction passes its own gate (Week-7 step-4 R² =
   0.91-0.96 on test) and because the gate-firing signal here is
   feasibility, not metric calibration — feasibility is exactly what
   the wheel-level correction shifts.
3. **Week-8 ordering update.** Regenerate `data/analytical/lhs_v3.parquet`
   with `use_scm_correction=True` before re-fitting baselines. The
   v4 dataset becomes the new training corpus for the production
   surrogate.

## Files

- `gate_summary.csv` — per-family thresholds and gate flags (this report's headline table)
- `feasibility_direction.csv` — one-sided flip breakdown (BW-conservative bias)
- `spearman_mobility.csv` — rank stability on continuous mobility metrics
- `data/scm/gate_eval_v1.parquet` — paired BW/SCM metrics, 200 rows, all headline fields
- `data/scm/correction_v1.joblib` — the wheel-level correction artifact (Week-7 step-4)
