"""Week-5 real-rover validation gate (project_plan.md §6 W5).

This is **the** Week-5 CI gate. The plan calls out Week 5 as the critical
pre-ML check: if the evaluator can't reproduce real rover behaviour, we
fix the evaluator before touching the surrogate layer. The tests here
encode the acceptance criteria that let that gate fail loudly in CI.

See :mod:`roverdevkit.validation.rover_comparison` for the scoring
definitions. The gate hits five criteria per rover (range feasibility,
range sanity ceiling, thermal survival match, motor/traversal ok, peak
solar in band); this file adds finer-grained per-rover tests so that
when the gate fires, the failure message points at a specific criterion
rather than a generic aggregate.
"""

from __future__ import annotations

import pytest

from roverdevkit.validation.rover_comparison import (
    acceptance_gate,
    compare_all,
    compare_one,
)
from roverdevkit.validation.rover_registry import (
    registry,
    registry_by_name,
    truth_by_rover,
)

REGISTERED_ROVERS = [e.rover_name for e in registry()]


# ---------------------------------------------------------------------------
# Aggregate gate
# ---------------------------------------------------------------------------


def test_acceptance_gate_passes_for_full_registry() -> None:
    """The Week-5 gate: every registered rover passes every criterion."""
    summary = compare_all()
    acceptance_gate(summary)
    assert summary.all_pass
    assert summary.n_pass == len(REGISTERED_ROVERS)


def test_comparison_summary_is_deterministic_across_runs() -> None:
    """Two back-to-back runs must produce identical scoring.

    If this ever flakes, either (a) `evaluate` has an unseeded random
    source, or (b) the traverse sim has a floating-point sensitivity
    we haven't documented. Both are bugs.
    """
    first = compare_all()
    second = compare_all()
    for a, b in zip(first.results, second.results, strict=True):
        assert a.range_m_predicted == pytest.approx(b.range_m_predicted)
        assert a.peak_solar_power_w_predicted == pytest.approx(b.peak_solar_power_w_predicted)
        assert a.metrics.thermal_survival == b.metrics.thermal_survival
        assert a.passes == b.passes


# ---------------------------------------------------------------------------
# Per-rover per-criterion tests (fail messages are self-diagnosing)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rover_name", REGISTERED_ROVERS)
def test_range_is_feasible_vs_published(rover_name: str) -> None:
    """Predicted range >= published low-band: sim must claim the rover
    *could* at least reach what it actually flew."""
    result = compare_one(registry_by_name(rover_name))
    assert result.range_feasible, (
        f"{rover_name}: predicted range {result.range_m_predicted:.1f} m is "
        f"below the published low bound {result.truth.traverse_m_low:.1f} m."
    )


@pytest.mark.parametrize("rover_name", REGISTERED_ROVERS)
def test_range_below_sanity_ceiling(rover_name: str) -> None:
    """Predicted range <= 10 x published high band: catch pathological
    over-prediction (e.g. a broken stall detector)."""
    result = compare_one(registry_by_name(rover_name))
    assert result.range_below_sanity_ceiling, (
        f"{rover_name}: predicted range {result.range_m_predicted:.1f} m "
        f"exceeds 10x the published high bound "
        f"{result.truth.traverse_m_high:.1f} m."
    )


@pytest.mark.parametrize("rover_name", REGISTERED_ROVERS)
def test_thermal_survival_matches_published(rover_name: str) -> None:
    """Sim's hot+cold steady-state survival prediction matches reality.

    Pragyan's published False (died in lunar night) is the strongest
    signal in the set - it validates the sink-temp + RHU-carrying
    logic.
    """
    result = compare_one(registry_by_name(rover_name))
    assert result.thermal_matches, (
        f"{rover_name}: thermal prediction {result.metrics.thermal_survival} "
        f"!= published {result.truth.thermal_survival_published}."
    )


@pytest.mark.parametrize("rover_name", REGISTERED_ROVERS)
def test_motor_and_traversal_not_stalled(rover_name: str) -> None:
    """Motor torque within envelope and rover didn't stall on scenario slope."""
    result = compare_one(registry_by_name(rover_name))
    assert result.motor_and_traversal_ok, (
        f"{rover_name}: motor_torque_ok is False or rover stalled on the "
        f"scenario's {registry_by_name(rover_name).scenario.max_slope_deg:.0f} "
        f"deg typical-ops slope."
    )


@pytest.mark.parametrize("rover_name", REGISTERED_ROVERS)
def test_peak_solar_power_in_published_band(rover_name: str) -> None:
    """Predicted peak solar power sits inside the published low/high band."""
    result = compare_one(registry_by_name(rover_name))
    assert result.peak_solar_in_band, (
        f"{rover_name}: predicted peak solar "
        f"{result.peak_solar_power_w_predicted:.1f} W is outside the "
        f"published band [{result.truth.peak_solar_power_w_low:.1f}, "
        f"{result.truth.peak_solar_power_w_high:.1f}] W."
    )


# ---------------------------------------------------------------------------
# Sensitivity: "right direction when parameters change" (plan §6 W5)
# ---------------------------------------------------------------------------


def test_larger_battery_never_reduces_energy_margin() -> None:
    """Monotonic: doubling the battery should not decrease energy margin."""
    from dataclasses import replace

    from roverdevkit.mission.evaluator import evaluate

    base = registry_by_name("Pragyan")
    baseline = evaluate(
        base.design,
        base.scenario,
        gravity_m_per_s2=base.gravity_m_per_s2,
        thermal_architecture=base.thermal_architecture,
    )
    larger = base.design.model_copy(
        update={"battery_capacity_wh": min(500.0, base.design.battery_capacity_wh * 2.0)}
    )
    bigger_battery = evaluate(
        larger,
        base.scenario,
        gravity_m_per_s2=base.gravity_m_per_s2,
        thermal_architecture=base.thermal_architecture,
    )
    # Use model_copy to sanity-check replace helper is unused.
    _ = replace
    assert bigger_battery.energy_margin_pct >= baseline.energy_margin_pct - 1e-6


def test_polar_latitude_reduces_peak_solar_power() -> None:
    """Yutu-2 at 45 deg latitude must see higher peak solar than at 85 deg."""
    from roverdevkit.power.solar import panel_power_w, sun_elevation_deg

    yutu2 = registry_by_name("Yutu-2")

    peak_mid = panel_power_w(
        panel_area_m2=yutu2.design.solar_area_m2,
        panel_efficiency=yutu2.panel_efficiency,
        sun_elevation_deg=sun_elevation_deg(45.5, lunar_hour_angle_deg=0.0),
        dust_degradation_factor=yutu2.panel_dust_factor,
    )
    peak_polar = panel_power_w(
        panel_area_m2=yutu2.design.solar_area_m2,
        panel_efficiency=yutu2.panel_efficiency,
        sun_elevation_deg=sun_elevation_deg(-85.0, lunar_hour_angle_deg=0.0),
        dust_degradation_factor=yutu2.panel_dust_factor,
    )
    assert peak_mid > peak_polar


# ---------------------------------------------------------------------------
# Truth-table sanity: catches CSV drift
# ---------------------------------------------------------------------------


def test_every_registered_rover_has_a_truth_row() -> None:
    for entry in registry():
        truth = truth_by_rover(entry.rover_name)
        assert truth.rover_name == entry.rover_name
        assert truth.scenario_name == entry.scenario.name


def test_published_traverse_bands_are_valid() -> None:
    """Low <= published <= high for every row."""
    for entry in registry():
        t = truth_by_rover(entry.rover_name)
        assert t.traverse_m_low <= t.traverse_m_published <= t.traverse_m_high
        assert (
            t.peak_solar_power_w_low <= t.peak_solar_power_w_published <= t.peak_solar_power_w_high
        )
