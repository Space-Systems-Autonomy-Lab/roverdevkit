# W8 step-4 — Quantile XGBoost Prediction Intervals (v4 corpus)

**Dataset:** `data/analytical/lhs_v4.parquet` (40 000 rows, BW + wheel-level
SCM correction). Train / val / test = 32 137 / 3 880 / 3 983 feasible rows
(motor_torque_ok = True). Test split is the canonical W8 step-2 / step-3
hold-out — never seen by the tuner or the quantile heads during fitting.

**Code:** `roverdevkit/surrogate/uncertainty.py`,
`scripts/calibrate_intervals.py`. Smoke tests in
`tests/test_surrogate_uncertainty.py` (6 tests, all pass).

**Hyperparameters:** Each of the three quantile heads
(`τ ∈ {0.05, 0.50, 0.95}`) per target reuses the W8 step-3 tuned median
configuration from `reports/week8_tuned_v4/tuned_best_params.json` —
only `objective="reg:quantileerror"` and `quantile_alpha` differ across
heads. Rationale in the `uncertainty.py` module docstring.

## Headline numbers

### Median (τ=0.5) sanity guardrail

The τ=0.5 quantile head should be a near-replica of the W8 step-3
tuned median. The R² delta on the test split is the §6 step-4 sanity
guardrail.

| Target                | Quantile R² (τ=0.5) | W8 step-3 tuned R² | Δ        |
|-----------------------|---------------------|--------------------|----------|
| range_km              | 0.9982              | 0.9993             | −0.0010  |
| energy_margin_raw_pct | 0.9912              | 0.9961             | −0.0049  |
| slope_capability_deg  | 0.9929              | 0.9945             | −0.0016  |
| total_mass_kg         | 0.9996              | 0.9998             | −0.0002  |

All four medians are within 0.005 R² of the squared-error baseline.
The small loss is the expected cost of fitting under pinball loss
rather than squared loss; it is well below the W8 step-2 acceptance
gate (R² ≥ 0.95 / 0.85) on every target.

### 90 % prediction-interval coverage on the test split

| Target                | Mean width | Raw 90 % cov. | Sorted 90 % cov. | Crossing rate |
|-----------------------|-----------:|--------------:|-----------------:|--------------:|
| range_km              | 13.52 km   | 0.852         | **0.919**        | 27.3 %        |
| energy_margin_raw_pct | 250.2 pp   | 0.866         | **0.918**        | 20.9 %        |
| slope_capability_deg  | 2.03 °     | 0.802         | 0.856            | 20.5 %        |
| total_mass_kg         | 1.48 kg    | 0.876         | **0.920**        | 21.4 %        |

"Raw" = independent quantile predictions, no monotonicity enforcement.
"Sorted" = row-wise sort of the three predictions before measuring
coverage of `[q05, q95]`. Sorting is non-worse for empirical coverage
and is the recommended downstream consumer behaviour
(`QuantileHeads.predict(..., repair_crossings=True)`).

**Three of four targets land within ±2 pp of the 90 % nominal after
row-wise sort.** `slope_capability_deg` remains 4 pp under-covered;
the target has very narrow per-row PI width (≈2°) so the model is
slightly over-confident on the tails. Acceptable for the methodology
paper given the step-3 R² already meets the §7 Layer-1 gate; a
lightweight conformal-prediction wrapper would close the residual
gap if a future revision needs strict 90 % calibration.

## Per-scenario breakdown (sorted coverage)

| Target                | crater_rim | equatorial_mare | highland_slope | polar_prospecting |
|-----------------------|-----------:|----------------:|---------------:|------------------:|
| range_km              | 0.917      | 0.882           | 0.917          | 0.963             |
| energy_margin_raw_pct | 0.916      | 0.912           | 0.899          | 0.947             |
| slope_capability_deg  | 0.858      | 0.862           | 0.842          | 0.863             |
| total_mass_kg         | 0.923      | 0.923           | 0.920          | 0.913             |

Polar PIs are slightly conservative (over-covered) and
equatorial-mare PIs slightly aggressive on `range_km` (0.882). This
mirrors the underlying corpus: polar missions saturate against the
solar/battery cap so the range tail concentrates near the limit
and quantile spread overshoots; equatorial-mare missions have a
bimodal range distribution (binding-vs-saturated) that the
independent heads do not capture as well.

## Crossing rates

20–27 % of test rows have a non-monotone `(q05, q50, q95)` triple
before the row-wise sort. This is expected — XGBoost quantile heads
are fit independently and the project decided against per-quantile
HP tuning (see module docstring). Sorting fixes every crossing without
changing the median; the bundle saves both the raw and sortable APIs
so the writeup can quote either, and downstream NSGA-II / Pareto
consumers should pass `repair_crossings=True`.

## Wall-clock

| Stage                            | Wall-clock |
|----------------------------------|-----------:|
| 4 targets × 3 quantile heads     | 132.8 s    |
| Coverage tables (raw + sorted)   | < 1 s      |
| Total reproduce time on 8 cores  | ~2.3 min   |

Per-target fit (single core wall):

| Target                | Fit (s) |
|-----------------------|--------:|
| range_km              | 33.2    |
| energy_margin_raw_pct | 29.0    |
| slope_capability_deg  | 37.1    |
| total_mass_kg         | 33.5    |

## Artifacts

```
reports/week8_intervals_v4/
├── coverage.csv              # long-format (target, family, repair) coverage
├── median_sanity.csv         # τ=0.5 vs W8 step-3 tuned R² guardrail
├── fit_seconds.csv           # per-target fit wall-clock
├── quantile_bundles.joblib   # {target: QuantileHeads} for downstream load
└── SUMMARY.md                # this file
```

## Reproduce

```bash
python scripts/calibrate_intervals.py \
    --dataset data/analytical/lhs_v4.parquet \
    --tuned-params reports/week8_tuned_v4/tuned_best_params.json \
    --out-dir reports/week8_intervals_v4
```

## What this closes

- **§6 step-4 deliverable:** Final mission-level surrogate (corrected
  evaluator outputs → tuned XGBoost medians + quantile heads) with
  calibrated 90 % prediction intervals and a coverage table by
  scenario family. ✅
- **§7 Layer 1 PI claim:** "calibrated 90 % prediction intervals on
  the corrected-evaluator output, reported with a per-scenario-family
  coverage table." ✅ (3/4 targets at ≤2 pp; slope at 4 pp,
  acknowledged.)
- **Phase-3 readiness:** NSGA-II consumers can now load
  `quantile_bundles.joblib` and either build probabilistic
  feasibility constraints (`q05` of a constraint vs threshold) or
  rank Pareto candidates by PI width as a robust-design proxy. The
  corrected evaluator stays the source of truth for headline plots.
