"""Cross-scenario ranking + one-at-a-time sensitivity tests.

Tests the "is the evaluator right in the right *direction* when
parameters change?" requirement from project_plan.md §6 W5.

Split into two families:
1. Ranking invariants - qualitative comparisons between hand-crafted
   archetype designs on canonical scenarios.
2. Sensitivity invariants - signs of metric deltas under single-variable
   perturbations.

These are intentionally direction-only tests. Magnitudes would couple
to specific parameter values and become brittle to calibration.
"""

from __future__ import annotations

from roverdevkit.validation.cross_scenario import (
    one_at_a_time_sensitivity,
    rank_archetypes,
)

# ---------------------------------------------------------------------------
# Ranking invariants
# ---------------------------------------------------------------------------


def test_slope_climber_has_highest_slope_capability_everywhere() -> None:
    """On every canonical scenario, the slope-specialty archetype must
    out-climb the other two. This validates that the slope-capability
    calculation is sensitive to the expected design levers (big wheels,
    many wheels, deep grousers)."""
    rankings = rank_archetypes()
    for scen, rank in rankings.items():
        slope_cap = {name: m.slope_capability_deg for name, m in rank.per_archetype.items()}
        assert slope_cap["slope_climber"] >= slope_cap["large_traverser"], scen
        assert slope_cap["slope_climber"] >= slope_cap["polar_survivor"], scen
        assert rank.slope_capability_winner == "slope_climber", scen


def test_large_traverser_has_highest_or_tied_range_in_benign_scenarios() -> None:
    """On scenarios where range is not catastrophically capped, the
    fast/big/duty-heavy archetype should tie or beat the slow archetype."""
    rankings = rank_archetypes()
    eq_rank = rankings["equatorial_mare_traverse"]
    range_km = {name: m.range_km for name, m in eq_rank.per_archetype.items()}
    assert range_km["large_traverser"] >= range_km["polar_survivor"]


def test_every_canonical_scenario_evaluates_all_archetypes() -> None:
    """Guard against silent evaluator crashes - must get 3 metrics per scenario."""
    rankings = rank_archetypes()
    assert len(rankings) == 4
    for rank in rankings.values():
        assert set(rank.per_archetype.keys()) == {
            "large_traverser",
            "polar_survivor",
            "slope_climber",
        }


def test_total_mass_is_consistent_across_scenarios() -> None:
    """Total mass depends only on the design vector, not the scenario.

    A surprising cross-scenario mass variance would signal that the
    evaluator is coupling mass to mission duration or terrain - a bug.
    """
    rankings = rank_archetypes()
    masses_per_archetype: dict[str, set[float]] = {}
    for rank in rankings.values():
        for name, m in rank.per_archetype.items():
            masses_per_archetype.setdefault(name, set()).add(round(m.total_mass_kg, 6))
    for name, masses in masses_per_archetype.items():
        assert len(masses) == 1, f"{name}: total_mass_kg varies across scenarios {masses}"


# ---------------------------------------------------------------------------
# Sensitivity invariants (one-at-a-time)
# ---------------------------------------------------------------------------


def _sens_by_var() -> dict[str, object]:
    """Convenience: sensitivity entries keyed by variable name."""
    return {e.variable: e for e in one_at_a_time_sensitivity()}


def test_more_avionics_power_lowers_energy_margin() -> None:
    """avionics_power_w is a pure parasitic load; margin must drop."""
    entry = _sens_by_var()["avionics_power_w"]
    assert entry.delta_energy_margin_pct < 0.0  # type: ignore[attr-defined]


def test_more_chassis_mass_increases_total_mass() -> None:
    """chassis_mass_kg is part of total_mass_kg directly plus propagates
    through the mass-model fixed point."""
    entry = _sens_by_var()["chassis_mass_kg"]
    assert entry.delta_total_mass_kg > 0.0  # type: ignore[attr-defined]


def test_bigger_wheel_radius_raises_slope_capability() -> None:
    """Bigger wheels -> more contact area -> more DP on a given soil.

    The wheel_radius lever is the single largest slope-capability
    modifier in the design space."""
    entry = _sens_by_var()["wheel_radius_m"]
    assert entry.delta_slope_capability_deg > 0.5  # type: ignore[attr-defined]


def test_faster_speed_increases_range_when_not_cap_bound() -> None:
    """In the distance-unlimited sensitivity scenario, doubling the
    nominal speed must grow the predicted range."""
    entry = _sens_by_var()["nominal_speed_mps"]
    assert entry.delta_range_km > 0.0  # type: ignore[attr-defined]


def test_higher_duty_cycle_increases_range_when_not_cap_bound() -> None:
    """duty cycle multiplies forward progress per step."""
    entry = _sens_by_var()["drive_duty_cycle"]
    assert entry.delta_range_km > 0.0  # type: ignore[attr-defined]


def test_bigger_solar_area_never_decreases_energy_margin() -> None:
    """More solar in -> margin same or higher. Small bumps may saturate
    a high-margin baseline; the monotonicity direction must hold."""
    entry = _sens_by_var()["solar_area_m2"]
    assert entry.delta_energy_margin_pct >= -1e-6  # type: ignore[attr-defined]


def test_bigger_battery_never_decreases_energy_margin() -> None:
    """More stored energy can only help energy-margin, all else equal."""
    entry = _sens_by_var()["battery_capacity_wh"]
    assert entry.delta_energy_margin_pct >= -1e-6  # type: ignore[attr-defined]
