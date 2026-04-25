"""Feature-matrix construction and column inventories for the surrogate.

This module is the single source of truth for **which columns the
baseline and multi-fidelity surrogates train on** (project_plan.md §6).
It mirrors the flat Parquet schema emitted by
:mod:`roverdevkit.surrogate.dataset` (see ``data/analytical/SCHEMA.md``)
and intentionally takes no ML-library dependency so the mission
evaluator can import it transitively without pulling XGBoost / sklearn.

Columns
-------
Inputs (39 columns):

- :data:`DESIGN_FEATURE_COLUMNS` (12) — the raw design vector.
- :data:`SCENARIO_NUMERIC_COLUMNS` (9) — continuous scenario + soil
  parameters (latitude, mission duration, max slope, Bekker n / k_c /
  k_phi / cohesion / friction / shear modulus).
- :data:`SCENARIO_CATEGORICAL_COLUMNS` (4) — scenario-family discrete
  features. Kept as pandas ``category`` dtype so XGBoost can consume
  them natively via ``enable_categorical=True`` without one-hot
  blow-up.

Targets:

- :data:`REGRESSION_TARGETS` — the 7 numeric mission metrics. The
  primary ones (range, raw energy margin, slope, total mass) are what
  the Week-6 accuracy table reports on; the others are secondary
  diagnostics.
- :data:`CLASSIFICATION_TARGETS` — the single ``motor_torque_ok``
  feasibility flag (a real Bekker-Wong outcome that depends on grouser
  geometry, soil shear parameters, slope, and mass).

Why no thermal target
---------------------
``thermal_survival`` was dropped from the surrogate schema in v2 (see
``data/analytical/SCHEMA.md``): with the current mass model RHU power
and MLI quality are free, so thermal reduces to a near-trivial gate
without a real design trade-off. The system-level evaluator still
computes it as a diagnostic; a future mass-model upgrade that charges
RHU/MLI mass would restore thermal as a learnable Pareto target.

Engineered features (``add_engineered_features``) are deferred to
Week 7: base numeric + categorical columns are sufficient for the
Week-6 XGBoost baseline and the multi-fidelity composition, and
adding engineered features pre-baseline would confound the
"did-features-help?" ablation.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Input columns
# ---------------------------------------------------------------------------

DESIGN_FEATURE_COLUMNS: list[str] = [
    "design_wheel_radius_m",
    "design_wheel_width_m",
    "design_grouser_height_m",
    "design_grouser_count",
    "design_n_wheels",
    "design_chassis_mass_kg",
    "design_wheelbase_m",
    "design_solar_area_m2",
    "design_battery_capacity_wh",
    "design_avionics_power_w",
    "design_nominal_speed_mps",
    "design_drive_duty_cycle",
]
"""12-D design vector, prefixed to match the Parquet schema."""

SCENARIO_NUMERIC_COLUMNS: list[str] = [
    "scenario_latitude_deg",
    "scenario_mission_duration_earth_days",
    "scenario_max_slope_deg",
    "scenario_soil_n",
    "scenario_soil_k_c",
    "scenario_soil_k_phi",
    "scenario_soil_cohesion_kpa",
    "scenario_soil_friction_angle_deg",
    "scenario_soil_shear_modulus_k_m",
]
"""Continuous scenario + jittered Bekker-soil inputs (9 columns).

``scenario_traverse_distance_m`` is intentionally excluded: it is
family-fixed (non-binding) and would otherwise leak the scenario
identity into a supposedly continuous feature."""

SCENARIO_CATEGORICAL_COLUMNS: list[str] = [
    "scenario_family",
    "scenario_terrain_class",
    "scenario_soil_simulant",
    "scenario_sun_geometry",
]
"""Scenario-family categorical inputs (4 columns). Keep as pandas
``category`` dtype and let XGBoost handle them natively via
``enable_categorical=True`` rather than one-hot encoding."""

INPUT_COLUMNS: list[str] = (
    DESIGN_FEATURE_COLUMNS + SCENARIO_NUMERIC_COLUMNS + SCENARIO_CATEGORICAL_COLUMNS
)
"""Concatenated input column list used by :func:`build_feature_matrix`."""


# ---------------------------------------------------------------------------
# Target columns
# ---------------------------------------------------------------------------

REGRESSION_TARGETS: list[str] = [
    "range_km",
    "energy_margin_pct",
    "energy_margin_raw_pct",
    "slope_capability_deg",
    "total_mass_kg",
    "peak_motor_torque_nm",
    "sinkage_max_m",
]
"""All numeric mission-metric targets."""

PRIMARY_REGRESSION_TARGETS: list[str] = [
    "range_km",
    "energy_margin_raw_pct",
    "slope_capability_deg",
    "total_mass_kg",
]
"""Subset reported in the Week-6 accuracy gate (project_plan.md §7 L1)."""

CLASSIFICATION_TARGETS: list[str] = ["motor_torque_ok"]
"""Single binary feasibility target.

``motor_torque_ok`` captures whether the rover can generate sufficient
drawbar pull to climb the scenario's worst-case slope under
Bekker-Wong terramechanics with the sampled (jittered) soil
parameters. This is a real physics-driven binary outcome; learning it
exercises the surrogate's joint understanding of grouser geometry,
soil shear parameters, mass, and slope."""

FEASIBILITY_COLUMN: str = "motor_torque_ok"
"""Alias for the single feasibility column. Kept as a constant so
downstream baselines / classifiers reference one canonical name even
if the underlying definition changes (e.g. when thermal is restored as
a real trade-off in a future mass-model upgrade)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model-input columns in canonical order.

    The returned DataFrame is a shallow copy of ``df`` restricted to
    :data:`INPUT_COLUMNS`; categorical dtypes are preserved for direct
    XGBoost ``enable_categorical=True`` consumption. Callers should
    filter to ``status == 'ok'`` rows (see :func:`valid_rows`) before
    passing to the trainer -- rows where the evaluator raised have NaN
    targets and this helper does **not** drop them.

    Raises
    ------
    KeyError
        If any required column is missing, with the full missing list
        in the message so a stale Parquet file is easy to diagnose.
    """
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"missing required input columns: {missing}. "
            "Check SCHEMA_VERSION on the source Parquet."
        )
    return df.loc[:, INPUT_COLUMNS].copy()


def valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rows where the evaluator succeeded.

    Drops any row with ``status != 'ok'`` (or missing ``status``). Also
    drops rows where primary regression targets are NaN, which can
    happen if a physics sub-model silently returns non-finite values.
    """
    if "status" in df.columns:
        mask = df["status"].astype(str) == "ok"
    else:
        mask = pd.Series(True, index=df.index)
    for col in PRIMARY_REGRESSION_TARGETS:
        if col in df.columns:
            mask &= df[col].notna()
    return df.loc[mask].copy()


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered feature columns appended.

    Placeholder: the Week-6 baseline uses only raw + categorical
    features so any Week-7 engineered-feature lift is cleanly
    attributable in the ablation.
    """
    raise NotImplementedError(
        "Engineered features land in Week 7 after the baseline accuracy "
        "table is frozen (project_plan.md §6)."
    )


__all__ = [
    "CLASSIFICATION_TARGETS",
    "DESIGN_FEATURE_COLUMNS",
    "FEASIBILITY_COLUMN",
    "INPUT_COLUMNS",
    "PRIMARY_REGRESSION_TARGETS",
    "REGRESSION_TARGETS",
    "SCENARIO_CATEGORICAL_COLUMNS",
    "SCENARIO_NUMERIC_COLUMNS",
    "add_engineered_features",
    "build_feature_matrix",
    "valid_rows",
]
