"""Physics-informed feature engineering for the surrogate.

Hand-engineered features that significantly help tree-based models on
terramechanics-like inputs:

- ``z_over_R``  — dimensionless sinkage.
- ``R_times_W`` — effective contact area scale.
- ``grouser_volume_fraction`` — grouser height × count / wheel circumference.
- ``specific_power`` — solar area / total mass.
- ``energy_per_km`` — battery capacity / traverse distance target.
- ``slip_x_slope`` — interaction feature for slope climbing.

These are populated iteratively in Weeks 6–7, driven by whichever features
most improve held-out accuracy. Keep this module free of ML-library imports
so that the mission evaluator itself doesn't take a dependency on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered feature columns appended."""
    raise NotImplementedError("Implement in Week 6 per project_plan.md §6.")


FEATURE_COLUMNS: list[str] = [
    "wheel_radius_m",
    "wheel_width_m",
    "grouser_height_m",
    "grouser_count",
    "n_wheels",
    "chassis_mass_kg",
    "wheelbase_m",
    "solar_area_m2",
    "battery_capacity_wh",
    "avionics_power_w",
    "nominal_speed_mps",
    "drive_duty_cycle",
]
"""Base 12-D design-vector feature list, before engineered additions."""


TARGET_COLUMNS: list[str] = [
    "range_km",
    "energy_margin_pct",
    "slope_capability_deg",
    "total_mass_kg",
]
"""Multi-output regression targets."""
