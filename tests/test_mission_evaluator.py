"""End-to-end mission-evaluator integration tests.

Smoke tests in Week 4: the pipeline runs end-to-end on every canonical
scenario for a Rashid-like design and returns finite, in-range metrics.

The Week 5 acceptance test -- "loaded with Yutu-2 / Pragyan / Rashid
parameters, does the evaluator predict daily traverse distance and
power profile in the right order of magnitude?" -- lives in a separate
notebook per project_plan.md §6 W5.
"""

from __future__ import annotations

import math

import pytest

from roverdevkit.mission.evaluator import evaluate, range_at_utilisation
from roverdevkit.mission.scenarios import list_scenarios, load_scenario
from roverdevkit.schema import DesignVector, MissionMetrics, MissionScenario


@pytest.mark.integration
def test_evaluator_returns_mission_metrics(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    metrics = evaluate(rashid_like_design, equatorial_scenario)
    assert isinstance(metrics, MissionMetrics)


@pytest.mark.integration
def test_mass_in_micro_rover_class(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    metrics = evaluate(rashid_like_design, equatorial_scenario)
    # Rashid was ~10 kg; the design vector yields a bottom-up estimate
    # in the 5-50 kg micro-rover class (project_plan.md §1).
    assert 5.0 <= metrics.total_mass_kg <= 50.0


@pytest.mark.integration
def test_all_metrics_are_finite(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    m = evaluate(rashid_like_design, equatorial_scenario)
    for value in (
        m.range_km,
        m.energy_margin_pct,
        m.slope_capability_deg,
        m.total_mass_kg,
        m.peak_motor_torque_nm,
        m.sinkage_max_m,
    ):
        assert math.isfinite(value)
        assert value >= 0.0


@pytest.mark.integration
def test_range_bounded_by_traverse_distance(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    m = evaluate(rashid_like_design, equatorial_scenario)
    # range_km cannot exceed traverse_distance_m/1000 -- the sim caps it.
    assert m.range_km <= equatorial_scenario.traverse_distance_m / 1000.0 + 1e-9


@pytest.mark.integration
def test_slope_capability_within_schema_bounds(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    m = evaluate(rashid_like_design, equatorial_scenario)
    # schema allows 0-35 deg
    assert 0.0 <= m.slope_capability_deg <= 35.0


@pytest.mark.integration
@pytest.mark.parametrize("name", sorted(list_scenarios()))
def test_evaluator_runs_on_every_scenario(rashid_like_design: DesignVector, name: str) -> None:
    scenario = load_scenario(name)
    metrics = evaluate(rashid_like_design, scenario)
    assert metrics.total_mass_kg > 0.0
    assert metrics.range_km >= 0.0


@pytest.mark.integration
def test_bigger_battery_gives_higher_or_equal_energy_margin(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    baseline = evaluate(rashid_like_design, equatorial_scenario)
    bigger_battery = rashid_like_design.model_copy(update={"battery_capacity_wh": 400.0})
    upgraded = evaluate(bigger_battery, equatorial_scenario)
    # A bigger battery cannot worsen energy margin on the same scenario.
    assert upgraded.energy_margin_pct >= baseline.energy_margin_pct - 1e-6


@pytest.mark.integration
def test_denser_soil_boosts_slope_capability(
    rashid_like_design: DesignVector,
) -> None:
    def make(soil: str, slope: float) -> MissionScenario:
        return MissionScenario(
            name="highland_slope_capability",
            latitude_deg=10.0,
            traverse_distance_m=500.0,
            terrain_class="highland_dense",
            soil_simulant=soil,
            mission_duration_earth_days=5.0,
            max_slope_deg=slope,
        )

    loose = evaluate(rashid_like_design, make("Apollo_regolith_loose", 10.0))
    dense = evaluate(rashid_like_design, make("Apollo_regolith_dense", 10.0))
    assert dense.slope_capability_deg > loose.slope_capability_deg


@pytest.mark.integration
def test_scm_correction_not_yet_wired(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    with pytest.raises(NotImplementedError, match="SCM correction"):
        evaluate(rashid_like_design, equatorial_scenario, use_scm_correction=True)


# ---------------------------------------------------------------------------
# Capability-envelope vs operational-utilisation rescaling
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_range_at_utilisation_matches_capability_at_designed_duty(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    """Passing ``u = drive_duty_cycle`` reproduces the capability range."""
    metrics = evaluate(rashid_like_design, equatorial_scenario)
    rescaled = range_at_utilisation(
        metrics, rashid_like_design, rashid_like_design.drive_duty_cycle
    )
    assert math.isclose(rescaled, metrics.range_km, rel_tol=1e-9)


@pytest.mark.integration
def test_range_at_utilisation_scales_linearly_with_duty(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    """Half the operational duty cycle -> half the rescaled range."""
    metrics = evaluate(rashid_like_design, equatorial_scenario)
    half_duty = 0.5 * rashid_like_design.drive_duty_cycle
    rescaled = range_at_utilisation(metrics, rashid_like_design, half_duty)
    assert math.isclose(rescaled, 0.5 * metrics.range_km, rel_tol=1e-9)


@pytest.mark.integration
def test_range_at_utilisation_rejects_over_duty(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    """Passing ``u`` above designed duty is a coding error (hardware not sized)."""
    metrics = evaluate(rashid_like_design, equatorial_scenario)
    with pytest.raises(ValueError, match="exceeds designed duty"):
        range_at_utilisation(metrics, rashid_like_design, rashid_like_design.drive_duty_cycle + 0.1)


def test_range_at_utilisation_rejects_negative() -> None:
    metrics = MissionMetrics(
        range_km=5.0,
        energy_margin_pct=50.0,
        slope_capability_deg=10.0,
        total_mass_kg=15.0,
        peak_motor_torque_nm=5.0,
        sinkage_max_m=0.01,
        thermal_survival=True,
        motor_torque_ok=True,
    )
    design = DesignVector(
        wheel_radius_m=0.12,
        wheel_width_m=0.08,
        grouser_height_m=0.008,
        grouser_count=12,
        n_wheels=6,
        chassis_mass_kg=12.0,
        wheelbase_m=0.6,
        solar_area_m2=0.4,
        battery_capacity_wh=100.0,
        avionics_power_w=15.0,
        nominal_speed_mps=0.02,
        drive_duty_cycle=0.15,
    )
    with pytest.raises(ValueError, match=">= 0"):
        range_at_utilisation(metrics, design, -0.01)
