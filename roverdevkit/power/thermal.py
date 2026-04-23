"""Lumped-parameter thermal survival check.

Binary pass/fail constraint: does a lumped-parameter model of the avionics
enclosure stay within survivability limits (a) during peak sun and (b) during
lunar night? Inputs: avionics heat load, optional RHU power, surface
absorptivity/emissivity, insulation conductance.

Thermal architecture is deliberately *not* a continuous design variable in
v1 — adding it is a week-12 stretch goal (project_plan.md §3.1).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThermalArchitecture:
    """Lumped-parameter thermal model for the avionics box."""

    surface_area_m2: float
    absorptivity: float = 0.3
    emissivity: float = 0.85
    insulation_ua_w_per_k: float = 0.5
    """External UA (conductance × area), W/K."""

    rhu_power_w: float = 0.0

    min_operating_temp_c: float = -30.0
    max_operating_temp_c: float = 50.0


def survives_mission(
    architecture: ThermalArchitecture,
    avionics_power_w: float,
    latitude_deg: float,
) -> bool:
    """Binary survival check for peak-sun and lunar-night extremes."""
    raise NotImplementedError("Implement in Week 4 per project_plan.md §6.")
