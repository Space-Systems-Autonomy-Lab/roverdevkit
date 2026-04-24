"""ML surrogate layer over the mission evaluator.

The surrogate learns the same capability-envelope mapping the evaluator
computes (``design -> MissionMetrics``), including both the clipped
``energy_margin_pct`` and its unclipped training companion
``energy_margin_raw_pct``. Operational-utilisation queries are a
downstream wrapper on top of this layer, not a separate surrogate.

- :mod:`.features` — feature engineering (dimensionless groups, physics-informed
  transforms).
- :mod:`.models` — XGBoost (primary), sklearn baselines, and optional PyTorch
  NN ensemble. Includes the multi-fidelity composition.
- :mod:`.train` — train/val/test split, cross-validation, Optuna tuning.
- :mod:`.uncertainty` — calibrated prediction intervals (quantile regression
  for trees; MC-dropout or deep ensembles for NNs).

Target accuracy (project_plan.md §7 Layer 1):
    R² > 0.95 for range_km and energy_margin_pct; R² > 0.85 for slope.
"""
