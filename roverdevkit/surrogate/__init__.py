"""ML surrogate layer over the mission evaluator.

The surrogate learns the same capability-envelope mapping the evaluator
computes (``design -> MissionMetrics``), including both the clipped
``energy_margin_pct`` and its unclipped training companion
``energy_margin_raw_pct``. Operational-utilisation queries are a
downstream wrapper on top of this layer, not a separate surrogate.

- :mod:`.sampling` — stratified Latin-Hypercube sampler over the 12-D
  design space × 4 scenario families with jittered scenario/soil
  parameters.
- :mod:`.dataset` — parallel dataset builder + Parquet I/O. Consumes
  ``LHSSample`` from :mod:`.sampling` and produces a flat-schema
  DataFrame of evaluator outputs plus aggregate traverse-log
  statistics.
- :mod:`.features` — feature engineering (dimensionless groups,
  physics-informed transforms). Week 6.
- :mod:`.models` — XGBoost (primary), sklearn baselines, and optional
  PyTorch NN ensemble. Includes the multi-fidelity composition
  ``final = analytical_surrogate + correction_surrogate`` (Week 7).
- :mod:`.train` — cross-validation, Optuna tuning. Week 6-7.
- :mod:`.uncertainty` — calibrated prediction intervals.

Target accuracy (project_plan.md §7 Layer 1):
    R² > 0.95 for range_km and energy_margin_pct; R² > 0.85 for slope.
"""

from roverdevkit.surrogate.baselines import (
    ACCEPTANCE_GATES,
    CLASSIFIER_ALGORITHMS,
    JOINT_MLP_NAME,
    REGRESSION_ALGORITHMS,
    FittedBaselines,
    acceptance_gate,
    evaluate_baselines,
    fit_baselines,
    predict_for_registry_rovers,
)
from roverdevkit.surrogate.dataset import (
    DatasetMetadata,
    build_and_write,
    build_dataset,
    read_parquet,
    read_parquet_metadata,
    write_parquet,
)
from roverdevkit.surrogate.sampling import (
    FAMILIES,
    LHSSample,
    ScenarioFamily,
    generate_samples,
)

__all__ = [
    "ACCEPTANCE_GATES",
    "CLASSIFIER_ALGORITHMS",
    "FAMILIES",
    "DatasetMetadata",
    "FittedBaselines",
    "JOINT_MLP_NAME",
    "LHSSample",
    "REGRESSION_ALGORITHMS",
    "ScenarioFamily",
    "acceptance_gate",
    "build_and_write",
    "build_dataset",
    "evaluate_baselines",
    "fit_baselines",
    "generate_samples",
    "predict_for_registry_rovers",
    "read_parquet",
    "read_parquet_metadata",
    "write_parquet",
]
