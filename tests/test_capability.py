"""Tests for the max-climbable-slope helper."""

from __future__ import annotations

import pytest

from roverdevkit.mission.capability import (
    DEFAULT_LUNAR_GRAVITY_M_PER_S2,
    SLOPE_SEARCH_UPPER_DEG,
    max_climbable_slope_deg,
)
from roverdevkit.terramechanics.bekker_wong import WheelGeometry
from roverdevkit.terramechanics.soils import get_soil_parameters


@pytest.fixture
def micro_wheel() -> WheelGeometry:
    return WheelGeometry(radius_m=0.10, width_m=0.06)


def test_max_slope_is_in_valid_range(micro_wheel: WheelGeometry) -> None:
    soil = get_soil_parameters("Apollo_regolith_nominal")
    slope = max_climbable_slope_deg(micro_wheel, soil, total_mass_kg=15.0, n_wheels=4)
    assert 0.0 <= slope <= SLOPE_SEARCH_UPPER_DEG


def test_softer_soil_gives_lower_slope_capability(micro_wheel: WheelGeometry) -> None:
    dense = get_soil_parameters("Apollo_regolith_dense")
    loose = get_soil_parameters("Apollo_regolith_loose")
    dense_slope = max_climbable_slope_deg(micro_wheel, dense, 15.0, 4)
    loose_slope = max_climbable_slope_deg(micro_wheel, loose, 15.0, 4)
    assert dense_slope > loose_slope


def test_heavier_rover_sinks_more_and_climbs_less_or_equal(
    micro_wheel: WheelGeometry,
) -> None:
    soil = get_soil_parameters("Apollo_regolith_loose")
    light = max_climbable_slope_deg(micro_wheel, soil, 10.0, 4)
    heavy = max_climbable_slope_deg(micro_wheel, soil, 40.0, 4)
    # In general heavier vehicle on soft soil climbs less (more sinkage,
    # higher resistance). Allow equality because the cap at 35 deg can
    # saturate both.
    assert light >= heavy


def test_larger_wheel_climbs_at_least_as_well() -> None:
    soil = get_soil_parameters("Apollo_regolith_nominal")
    small = WheelGeometry(radius_m=0.06, width_m=0.05)
    large = WheelGeometry(radius_m=0.15, width_m=0.10)
    small_slope = max_climbable_slope_deg(small, soil, 15.0, 4)
    large_slope = max_climbable_slope_deg(large, soil, 15.0, 4)
    assert large_slope >= small_slope


def test_rejects_invalid_mass(micro_wheel: WheelGeometry) -> None:
    soil = get_soil_parameters("Apollo_regolith_nominal")
    with pytest.raises(ValueError, match="total_mass_kg"):
        max_climbable_slope_deg(micro_wheel, soil, 0.0, 4)


def test_cap_returned_when_rover_exceeds_schema_bound(
    micro_wheel: WheelGeometry,
) -> None:
    # On very dense soil, a light rover with big wheels easily exceeds
    # 35 deg; the helper must return the schema cap rather than the
    # unbounded physical slope.
    soil = get_soil_parameters("Apollo_regolith_dense")
    big_wheel = WheelGeometry(radius_m=0.18, width_m=0.12)
    slope = max_climbable_slope_deg(big_wheel, soil, total_mass_kg=5.0, n_wheels=4)
    assert slope == pytest.approx(SLOPE_SEARCH_UPPER_DEG)


def test_lunar_gravity_constant_matches_mass_model() -> None:
    # Guardrail: the lunar-gravity default must match the mass model's
    # constant so the two never drift apart.
    from roverdevkit.mass.parametric_mers import MassModelParams

    mass_g = MassModelParams().gravity_moon_m_per_s2
    assert mass_g == pytest.approx(DEFAULT_LUNAR_GRAVITY_M_PER_S2)
