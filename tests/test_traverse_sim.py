"""Tests for the time-stepped traverse simulator."""

from __future__ import annotations

import numpy as np
import pytest

from roverdevkit.mission.scenarios import load_scenario
from roverdevkit.mission.traverse_sim import TraverseLog, run_traverse
from roverdevkit.schema import DesignVector, MissionScenario
from roverdevkit.terramechanics.soils import get_soil_parameters


@pytest.fixture
def soil_nominal():
    return get_soil_parameters("Apollo_regolith_nominal")


@pytest.fixture
def equatorial(rashid_like_design: DesignVector):
    # use an easier slope so the micro-rover isn't stalled every test
    return load_scenario("equatorial_mare_traverse").model_copy(update={"max_slope_deg": 5.0})


# ---------------------------------------------------------------------------
# Structural / smoke tests
# ---------------------------------------------------------------------------


def test_traverse_returns_full_log_for_full_duration(
    rashid_like_design: DesignVector,
    equatorial: MissionScenario,
    soil_nominal,
) -> None:
    log = run_traverse(rashid_like_design, equatorial, soil_nominal, total_mass_kg=15.0)
    assert isinstance(log, TraverseLog)
    # Default dt_s = 3600, duration = 14 days => 337 steps inclusive.
    n = len(log.t_s)
    assert n > 200
    for arr in (
        log.position_m,
        log.state_of_charge,
        log.power_in_w,
        log.power_out_w,
        log.mobility_power_w,
        log.slip,
        log.sinkage_m,
        log.wheel_torque_nm,
        log.sun_elevation_deg,
    ):
        assert len(arr) == n
    # Always runs to the end of the mission duration:
    assert log.t_s[-1] == pytest.approx(
        equatorial.mission_duration_earth_days * 24 * 3600.0, rel=1e-6
    )


def test_time_array_is_monotonic(
    rashid_like_design: DesignVector,
    equatorial: MissionScenario,
    soil_nominal,
) -> None:
    log = run_traverse(rashid_like_design, equatorial, soil_nominal, 15.0)
    assert np.all(np.diff(log.t_s) > 0.0)


def test_position_is_non_decreasing(
    rashid_like_design: DesignVector,
    equatorial: MissionScenario,
    soil_nominal,
) -> None:
    log = run_traverse(rashid_like_design, equatorial, soil_nominal, 15.0)
    assert np.all(np.diff(log.position_m) >= -1e-9)


def test_soc_stays_within_physical_bounds(
    rashid_like_design: DesignVector,
    equatorial: MissionScenario,
    soil_nominal,
) -> None:
    log = run_traverse(rashid_like_design, equatorial, soil_nominal, 15.0)
    assert np.all(log.state_of_charge >= 0.0)
    assert np.all(log.state_of_charge <= 1.0)
    # Floor is the default 0.15:
    assert np.all(log.state_of_charge >= 0.15 - 1e-6)


def test_solar_power_is_zero_during_night(
    rashid_like_design: DesignVector,
    equatorial: MissionScenario,
    soil_nominal,
) -> None:
    log = run_traverse(rashid_like_design, equatorial, soil_nominal, 15.0)
    night_mask = log.sun_elevation_deg <= 0.0
    assert np.all(log.power_in_w[night_mask] <= 1e-9)


# ---------------------------------------------------------------------------
# Physics checks
# ---------------------------------------------------------------------------


def test_position_caps_at_traverse_distance(rashid_like_design: DesignVector, soil_nominal) -> None:
    # Short traverse + plenty of time so the rover will finish.
    scenario = MissionScenario(
        name="crater_rim_survey",
        latitude_deg=0.0,
        traverse_distance_m=200.0,
        terrain_class="mare_nominal",
        soil_simulant="Apollo_regolith_nominal",
        mission_duration_earth_days=7.0,
        max_slope_deg=0.0,
    )
    log = run_traverse(rashid_like_design, scenario, soil_nominal, 12.0)
    assert log.reached_distance
    assert log.position_m[-1] == pytest.approx(scenario.traverse_distance_m, rel=1e-3)


def test_steeper_slope_draws_more_mobility_power(
    rashid_like_design: DesignVector,
    soil_nominal,
) -> None:
    def make(slope: float) -> MissionScenario:
        return MissionScenario(
            name="equatorial_mare_traverse",
            latitude_deg=10.0,
            traverse_distance_m=500.0,
            terrain_class="mare_nominal",
            soil_simulant="Apollo_regolith_nominal",
            mission_duration_earth_days=5.0,
            max_slope_deg=slope,
        )

    flat = run_traverse(rashid_like_design, make(0.0), soil_nominal, 15.0)
    steep = run_traverse(rashid_like_design, make(10.0), soil_nominal, 15.0)
    # Average mobility power is higher when climbing than on flat.
    flat_avg = float(np.mean(flat.mobility_power_w))
    steep_avg = float(np.mean(steep.mobility_power_w))
    assert steep_avg > flat_avg


def test_bigger_solar_area_charges_battery_more(
    rashid_like_design: DesignVector,
    equatorial: MissionScenario,
    soil_nominal,
) -> None:
    bigger = rashid_like_design.model_copy(update={"solar_area_m2": 1.0})
    a = run_traverse(rashid_like_design, equatorial, soil_nominal, 15.0)
    b = run_traverse(bigger, equatorial, soil_nominal, 15.0)
    # Integrated energy in is higher for the bigger panel:
    assert float(np.sum(b.power_in_w)) > float(np.sum(a.power_in_w))


def test_underpowered_rover_eventually_floors_battery(soil_nominal) -> None:
    # Tiny solar, hefty avionics, long mission -> battery drains and floors.
    design = DesignVector(
        wheel_radius_m=0.10,
        wheel_width_m=0.06,
        grouser_height_m=0.005,
        grouser_count=12,
        n_wheels=4,
        chassis_mass_kg=6.0,
        wheelbase_m=0.35,
        solar_area_m2=0.1,
        battery_capacity_wh=20.0,
        avionics_power_w=40.0,
        nominal_speed_mps=0.03,
        drive_duty_cycle=0.6,
    )
    scenario = MissionScenario(
        name="equatorial_mare_traverse",
        latitude_deg=89.0,  # low sun elevation -> less solar
        traverse_distance_m=500.0,
        terrain_class="mare_nominal",
        soil_simulant="Apollo_regolith_nominal",
        mission_duration_earth_days=14.0,
        max_slope_deg=0.0,
    )
    log = run_traverse(design, scenario, soil_nominal, 10.0)
    assert log.battery_floored


def test_unclimbable_slope_records_stall(rashid_like_design: DesignVector, soil_nominal) -> None:
    # Soft soil + steep slope -> rover stalls.
    scenario = MissionScenario(
        name="highland_slope_capability",
        latitude_deg=10.0,
        traverse_distance_m=200.0,
        terrain_class="highland_dense",
        soil_simulant="Apollo_regolith_nominal",
        mission_duration_earth_days=3.0,
        max_slope_deg=30.0,
    )
    loose = get_soil_parameters("Apollo_regolith_loose")
    log = run_traverse(rashid_like_design, scenario, loose, 20.0)
    assert log.rover_stalled


def test_log_flags_are_all_bools(
    rashid_like_design: DesignVector,
    equatorial: MissionScenario,
    soil_nominal,
) -> None:
    log = run_traverse(rashid_like_design, equatorial, soil_nominal, 15.0)
    assert isinstance(log.battery_floored, bool)
    assert isinstance(log.rover_stalled, bool)
    assert isinstance(log.reached_distance, bool)
    assert isinstance(log.terminated_reason, str)
    assert log.terminated_reason  # non-empty
