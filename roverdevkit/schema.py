"""Shared data schemas for design vectors, scenarios, and mission metrics.

These are the canonical types that flow between the mission evaluator,
surrogate, and tradespace layers. Using Pydantic gives us validation at the
boundaries (e.g. reject a wheel radius outside the design-space bounds) and
free JSON/YAML serialization for scenario config files.

Design-variable ranges follow :file:`project_plan.md` §3.1.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Design vector
# ---------------------------------------------------------------------------


class DesignVector(BaseModel):
    """A single point in the 12-dimensional rover design space.

    Units are SI unless otherwise noted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Mobility
    wheel_radius_m: float = Field(ge=0.05, le=0.20, description="Wheel radius R")
    wheel_width_m: float = Field(ge=0.03, le=0.15, description="Wheel width W")
    grouser_height_m: float = Field(ge=0.0, le=0.012, description="Grouser height h_g")
    grouser_count: int = Field(ge=0, le=24, description="Number of grousers N_g")
    n_wheels: Literal[4, 6] = Field(description="Wheel count N_w")

    # Chassis
    chassis_mass_kg: float = Field(ge=3.0, le=35.0, description="Dry chassis mass m_c")
    wheelbase_m: float = Field(ge=0.3, le=1.2, description="Wheelbase L_wb")

    # Power
    solar_area_m2: float = Field(ge=0.1, le=1.5, description="Solar array area A_s")
    battery_capacity_wh: float = Field(ge=20.0, le=500.0, description="Battery capacity C_b")
    avionics_power_w: float = Field(ge=5.0, le=40.0, description="Continuous avionics draw P_a")

    # Operations
    nominal_speed_mps: float = Field(ge=0.01, le=0.10, description="Nominal drive speed v")
    drive_duty_cycle: float = Field(ge=0.1, le=0.6, description="Fraction of mission-day driving δ")


# ---------------------------------------------------------------------------
# Mission scenario
# ---------------------------------------------------------------------------


TerrainClass = Literal["mare_nominal", "mare_loose", "highland_dense", "polar_regolith"]
ScenarioName = Literal[
    "equatorial_mare_traverse",
    "polar_prospecting",
    "highland_slope_capability",
    "crater_rim_survey",
]


class MissionScenario(BaseModel):
    """Fixed mission context against which a design is evaluated.

    Scenarios are typically loaded from YAML in
    :mod:`roverdevkit.mission.scenarios`. The four canonical scenarios
    (``ScenarioName``) are what the tradespace optimiser sweeps in
    Phase 3; validation scenarios (Week 5 rover-comparison harness)
    reuse the same schema with descriptive names, so ``name`` is a free
    string rather than the Literal. Invalid values never reach the
    optimiser because that path goes through ``load_scenario()``, which
    takes a ``ScenarioName`` Literal.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    latitude_deg: float = Field(ge=-90.0, le=90.0)
    traverse_distance_m: float = Field(gt=0.0)
    terrain_class: TerrainClass
    soil_simulant: str = Field(
        description="Key into data/soil_simulants.csv, e.g. 'Apollo_regolith_nominal'."
    )
    mission_duration_earth_days: float = Field(gt=0.0)
    max_slope_deg: float = Field(ge=0.0, le=35.0, default=15.0)
    sun_geometry: Literal["continuous", "diurnal", "polar_intermittent"] = "diurnal"


# ---------------------------------------------------------------------------
# Mission metrics (evaluator output)
# ---------------------------------------------------------------------------


class MissionMetrics(BaseModel):
    """Mission-level outputs of the evaluator or surrogate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Primary metrics
    range_km: float
    energy_margin_pct: float
    slope_capability_deg: float

    # Unclipped energy-balance signal for the surrogate. Defined as
    # ``(E_generated - E_consumed) / E_consumed * 100``, integrated over
    # the whole traverse. Unlike ``energy_margin_pct`` (SOC-based, clipped
    # at 0-100), this one is unbounded on both sides: negative means the
    # rover consumed more than it generated, >100 means surplus exceeded
    # consumption. Kept as a separate field so Week-6 LHS surrogates see
    # a smooth target; reporting gates keep using the clipped version.
    energy_margin_raw_pct: float = 0.0

    # Secondary metrics
    total_mass_kg: float
    peak_motor_torque_nm: float
    sinkage_max_m: float

    # Constraint flags
    thermal_survival: bool
    motor_torque_ok: bool

    # Optional uncertainty, populated by the surrogate layer
    range_km_std: float | None = None
    energy_margin_pct_std: float | None = None
    slope_capability_deg_std: float | None = None
