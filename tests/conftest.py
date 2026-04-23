"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from roverdevkit.schema import DesignVector, MissionScenario


@pytest.fixture
def rashid_like_design() -> DesignVector:
    """A Rashid-like design vector for tests and worked examples.

    Numbers chosen to match published Rashid specs where available
    (see data/published_rovers.csv) and reasonable defaults otherwise.
    """
    return DesignVector(
        wheel_radius_m=0.1,
        wheel_width_m=0.06,
        grouser_height_m=0.005,
        grouser_count=12,
        n_wheels=4,
        chassis_mass_kg=6.0,
        wheelbase_m=0.35,
        solar_area_m2=0.4,
        battery_capacity_wh=100.0,
        avionics_power_w=15.0,
        nominal_speed_mps=0.03,
        drive_duty_cycle=0.3,
    )


@pytest.fixture
def equatorial_scenario() -> MissionScenario:
    return MissionScenario(
        name="equatorial_mare_traverse",
        latitude_deg=20.2,
        traverse_distance_m=5000.0,
        terrain_class="mare_nominal",
        soil_simulant="Apollo_regolith_nominal",
        mission_duration_earth_days=14.0,
        max_slope_deg=15.0,
        sun_geometry="diurnal",
    )
