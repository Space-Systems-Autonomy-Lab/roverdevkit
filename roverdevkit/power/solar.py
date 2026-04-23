"""Lunar solar geometry and panel power model.

Computes instantaneous power generation given latitude, time of lunar day,
panel area, tilt, efficiency, and a dust-degradation factor.

References:
    Larson & Wertz, *Space Mission Analysis and Design* (SMAD), 3rd ed.
    Patel, M. R., *Spacecraft Power Systems*, CRC Press.

For lunar sun geometry we use NASA's published lunar ephemeris tables;
full SPICE is not needed for a tradespace-level tool (project_plan.md §4).

Validation (Week 2): cross-check noon power for Yutu-2 latitude against
published Yutu-2 power profile numbers.
"""

from __future__ import annotations

SOLAR_CONSTANT_AU_1_W_PER_M2 = 1361.0
"""Mean solar irradiance at 1 AU, W/m²."""


def sun_elevation_deg(latitude_deg: float, lunar_hour_angle_deg: float) -> float:
    """Sun elevation above the local horizontal plane on the Moon.

    Parameters
    ----------
    latitude_deg
        Selenographic latitude of the rover.
    lunar_hour_angle_deg
        Sun hour angle; 0° at local lunar noon.
    """
    raise NotImplementedError("Implement in Week 2 per project_plan.md §6.")


def panel_power_w(
    panel_area_m2: float,
    panel_efficiency: float,
    sun_elevation_deg: float,
    panel_tilt_deg: float = 0.0,
    dust_degradation_factor: float = 1.0,
) -> float:
    """Instantaneous electrical power output from the solar array.

    Returns zero when the sun is below the horizon.
    """
    raise NotImplementedError("Implement in Week 2 per project_plan.md §6.")
