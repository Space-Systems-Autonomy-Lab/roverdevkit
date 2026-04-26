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
    wheel_width_m: float = Field(
        ge=0.03,
        le=0.20,
        description=(
            "Wheel width W. Upper bound 0.20 m covers the heavier "
            "lunar-class micro-rovers (Yutu-2-class wheels are 0.15 m; "
            "Lunokhod-class is 0.20 m). Widened from 0.15 in the v3 "
            "LHS bounds widening to admit more representative validation "
            "rovers as in-distribution points."
        ),
    )
    grouser_height_m: float = Field(
        ge=0.0,
        le=0.020,
        description=(
            "Grouser height h_g. Upper bound 20 mm covers published lunar "
            "micro-rover wheels (Rashid-1 flew 15 mm, Yutu-class wheels "
            "use ~12 mm). The LHS sampler in surrogate.sampling currently "
            "draws to 12 mm only; widening it to the schema ceiling is a "
            "dataset-regen task tracked in the project log."
        ),
    )
    grouser_count: int = Field(ge=0, le=24, description="Number of grousers N_g")
    n_wheels: Literal[4, 6] = Field(description="Wheel count N_w")

    # Chassis
    chassis_mass_kg: float = Field(
        ge=3.0,
        le=50.0,
        description=(
            "Dry chassis mass m_c. Upper bound 50 kg widened from 35 kg "
            "in the v3 LHS bounds widening so the heavier flown lunar "
            "micro-rovers (Yutu-class, ~30-40 kg ex-payload) sit inside "
            "the surrogate's training support rather than at a corner. "
            "Floor 3 kg keeps the design space anchored to actual "
            "micro-rover scale."
        ),
    )
    wheelbase_m: float = Field(ge=0.3, le=1.2, description="Wheelbase L_wb")

    # Power
    solar_area_m2: float = Field(ge=0.1, le=1.5, description="Solar array area A_s")
    battery_capacity_wh: float = Field(ge=20.0, le=500.0, description="Battery capacity C_b")
    avionics_power_w: float = Field(ge=5.0, le=40.0, description="Continuous avionics draw P_a")

    # Operations
    nominal_speed_mps: float = Field(ge=0.01, le=0.10, description="Nominal drive speed v")
    drive_duty_cycle: float = Field(
        ge=0.02,
        le=0.6,
        description=(
            "Designed drive duty cycle δ: fraction of the mission the rover "
            "is commanded to drive. The hardware (battery, thermal, avionics) "
            "is sized to sustain this duty. Floor 0.02 captures real-ops "
            "regimes (Pragyan ~0.02, Yutu-2 ~0.015, Sojourner ~0.01); ceiling "
            "0.6 captures continuous-drive reference designs. Mission-ops "
            "utilisation *below* designed duty is a post-hoc query, not a "
            "design variable."
        ),
    )


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
    """Mission-level outputs of the evaluator or surrogate.

    All fields describe the **capability envelope** of a design under its
    own stated ``drive_duty_cycle`` and the scenario's mission window --
    i.e. what the hardware could deliver if the ops schedule commanded
    it to drive at the designed duty throughout. Real missions typically
    command a fraction of the designed duty (Pragyan ~0.02, Yutu-2 ~0.015,
    Sojourner ~0.01) for commanding / thermal-window / science-campaign
    reasons; the "range at operational utilisation u" can be recovered
    from ``range_km * u / drive_duty_cycle`` where ``u <= drive_duty_cycle``.
    Phase-3 downstream ops-planning lives on top of this capability layer,
    not inside it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Primary metrics - all are capability-at-designed-duty
    range_km: float  # "range capability" over scenario duration at designed δ
    energy_margin_pct: float  # SOC-based, clipped 0-100; reporting metric
    slope_capability_deg: float  # max climbable slope on this soil

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
