"""Tests for the lumped-parameter thermal survival check."""

from __future__ import annotations

import math

import pytest

from roverdevkit.power.thermal import (
    STEFAN_BOLTZMANN_W_PER_M2_K4,
    ThermalArchitecture,
    default_architecture_for_design,
    evaluate_thermal,
    survives_mission,
)


@pytest.fixture
def nominal_arch() -> ThermalArchitecture:
    return ThermalArchitecture(surface_area_m2=0.1)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_construction_validates_positive_area() -> None:
    with pytest.raises(ValueError, match="surface_area_m2"):
        ThermalArchitecture(surface_area_m2=0.0)


def test_construction_validates_emissivity_range() -> None:
    with pytest.raises(ValueError, match="emissivity"):
        ThermalArchitecture(surface_area_m2=0.1, emissivity=1.5)


def test_construction_validates_op_temp_order() -> None:
    with pytest.raises(ValueError, match="min_operating_temp_c"):
        ThermalArchitecture(
            surface_area_m2=0.1,
            min_operating_temp_c=50.0,
            max_operating_temp_c=-30.0,
        )


# ---------------------------------------------------------------------------
# Physics: hot case
# ---------------------------------------------------------------------------


def test_cold_case_equals_sink_with_no_internal_power() -> None:
    # With no sun, no hibernation load, and no RHU, equilibrium must equal
    # the cold-case sink temperature.
    arch = ThermalArchitecture(
        surface_area_m2=0.1,
        absorptivity=0.0,
        hibernation_power_w=0.0,
        rhu_power_w=0.0,
    )
    result = evaluate_thermal(arch, avionics_power_w=0.0, latitude_deg=0.0)
    assert result.lunar_night_temp_c == pytest.approx(
        arch.sink_temp_lunar_night_k - 273.15, abs=0.1
    )


def test_hot_case_is_hotter_at_equator_than_at_pole(
    nominal_arch: ThermalArchitecture,
) -> None:
    equator = evaluate_thermal(nominal_arch, avionics_power_w=10.0, latitude_deg=0.0)
    pole = evaluate_thermal(nominal_arch, avionics_power_w=10.0, latitude_deg=85.0)
    assert equator.peak_sun_temp_c > pole.peak_sun_temp_c


def test_hot_case_temp_increases_with_avionics_power(
    nominal_arch: ThermalArchitecture,
) -> None:
    low = evaluate_thermal(nominal_arch, avionics_power_w=5.0, latitude_deg=20.0)
    high = evaluate_thermal(nominal_arch, avionics_power_w=30.0, latitude_deg=20.0)
    assert high.peak_sun_temp_c > low.peak_sun_temp_c


# ---------------------------------------------------------------------------
# Physics: cold case
# ---------------------------------------------------------------------------


def test_cold_case_temp_increases_with_rhu_power() -> None:
    warm = ThermalArchitecture(surface_area_m2=0.1, rhu_power_w=10.0)
    cold = ThermalArchitecture(surface_area_m2=0.1, rhu_power_w=0.0)
    warm_r = evaluate_thermal(warm, avionics_power_w=15.0, latitude_deg=20.0)
    cold_r = evaluate_thermal(cold, avionics_power_w=15.0, latitude_deg=20.0)
    assert warm_r.lunar_night_temp_c > cold_r.lunar_night_temp_c


def test_cold_case_does_not_depend_on_operating_avionics_power(
    nominal_arch: ThermalArchitecture,
) -> None:
    low = evaluate_thermal(nominal_arch, avionics_power_w=5.0, latitude_deg=20.0)
    high = evaluate_thermal(nominal_arch, avionics_power_w=30.0, latitude_deg=20.0)
    assert low.lunar_night_temp_c == pytest.approx(high.lunar_night_temp_c, abs=1e-6)


# ---------------------------------------------------------------------------
# Survival flag
# ---------------------------------------------------------------------------


def test_survive_is_true_for_well_balanced_rover() -> None:
    # OSR-like coating (low alpha, high eps), modest RHU, and enough
    # hibernation draw keep the cold case above -30 C without frying
    # the rover at noon. A realistic passing design.
    arch = ThermalArchitecture(
        surface_area_m2=0.1,
        absorptivity=0.15,
        emissivity=0.9,
        rhu_power_w=15.0,
        hibernation_power_w=5.0,
    )
    result = evaluate_thermal(arch, avionics_power_w=15.0, latitude_deg=20.0)
    assert arch.min_operating_temp_c <= result.lunar_night_temp_c
    assert result.peak_sun_temp_c <= arch.max_operating_temp_c
    assert survives_mission(arch, avionics_power_w=15.0, latitude_deg=20.0) is True


def test_survive_is_false_for_unheated_rover_in_lunar_night() -> None:
    arch = ThermalArchitecture(
        surface_area_m2=0.3,
        rhu_power_w=0.0,
        hibernation_power_w=0.1,
    )
    # No internal power, no sun: T ≈ sink (100 K = -173 C), fails cold limit.
    result = evaluate_thermal(arch, avionics_power_w=15.0, latitude_deg=20.0)
    assert result.lunar_night_temp_c < arch.min_operating_temp_c
    assert not result.survives


def test_survive_is_false_if_overheating_in_hot_case() -> None:
    # Very absorptive, tiny area, small emissivity -> overheats fast.
    arch = ThermalArchitecture(
        surface_area_m2=0.02,
        absorptivity=1.0,
        emissivity=0.3,
        rhu_power_w=20.0,
        hibernation_power_w=5.0,
    )
    result = evaluate_thermal(arch, avionics_power_w=30.0, latitude_deg=0.0)
    assert result.peak_sun_temp_c > arch.max_operating_temp_c


# ---------------------------------------------------------------------------
# Sanity: radiative balance closes
# ---------------------------------------------------------------------------


def test_radiative_balance_closes(nominal_arch: ThermalArchitecture) -> None:
    # Plug the output back into Q_in = Q_out and check to 0.1 W.
    result = evaluate_thermal(nominal_arch, avionics_power_w=15.0, latitude_deg=20.0)
    t_hot_k = result.peak_sun_temp_c + 273.15
    q_out = (
        nominal_arch.emissivity
        * STEFAN_BOLTZMANN_W_PER_M2_K4
        * nominal_arch.surface_area_m2
        * (t_hot_k**4 - nominal_arch.sink_temp_peak_sun_k**4)
    )
    # Reconstruct Q_in independently:
    elevation_factor = math.cos(math.radians(20.0))
    sunlit_area = nominal_arch.surface_area_m2 * nominal_arch.solar_projected_area_fraction
    q_solar = nominal_arch.absorptivity * 1361.0 * elevation_factor * sunlit_area
    q_in = q_solar + 15.0 + nominal_arch.rhu_power_w
    assert q_out == pytest.approx(q_in, rel=1e-6)


def test_default_architecture_for_design_returns_valid_arch() -> None:
    arch = default_architecture_for_design(surface_area_m2=0.05, rhu_power_w=5.0)
    assert arch.surface_area_m2 == pytest.approx(0.05)
    assert arch.rhu_power_w == pytest.approx(5.0)
    assert arch.hibernation_power_w > 0.0
