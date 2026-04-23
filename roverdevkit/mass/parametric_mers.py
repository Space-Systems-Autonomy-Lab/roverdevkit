"""Parametric mass-estimating relationships (MERs) for subsystem masses.

Fitted in Week 3 from :file:`data/published_rovers.csv`.

**Caveat — small-n fit.** The training set is approximately eight real
flown/prototype lunar micro-rovers. These MERs are intended to capture the
*trends* of the tradespace ("bigger wheels weigh more in a known way"),
not to predict subsystem mass to ±1 %. The Week 6 surrogate can't be more
accurate on mass than the MERs that fed it; we acknowledge this explicitly
in the paper (§7 Discussion).

Each MER has the form::

    m_subsystem = a * (primary_driver)^b + c

with coefficients fitted by least squares on log-log axes where appropriate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MassBreakdown:
    """Subsystem-level mass breakdown (kg). Sum equals total_kg."""

    chassis_kg: float
    wheels_kg: float
    motors_and_drives_kg: float
    solar_panels_kg: float
    battery_kg: float
    harness_kg: float
    avionics_kg: float
    thermal_kg: float
    margin_kg: float

    @property
    def total_kg(self) -> float:
        return (
            self.chassis_kg
            + self.wheels_kg
            + self.motors_and_drives_kg
            + self.solar_panels_kg
            + self.battery_kg
            + self.harness_kg
            + self.avionics_kg
            + self.thermal_kg
            + self.margin_kg
        )


def estimate_mass(
    wheel_radius_m: float,
    wheel_width_m: float,
    n_wheels: int,
    chassis_mass_kg: float,
    solar_area_m2: float,
    battery_capacity_wh: float,
    avionics_power_w: float,
) -> MassBreakdown:
    """Estimate a full subsystem-level mass breakdown from design variables."""
    raise NotImplementedError("Implement in Week 3 per project_plan.md §6.")
