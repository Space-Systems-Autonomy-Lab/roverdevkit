"""Analytical terramechanics: Bekker-Wong pressure-sinkage + Janosi-Hanamoto shear.

Implements single-wheel drawbar pull, sinkage, and driving torque as a
function of wheel geometry, vertical load, slip, and soil parameters.

Primary reference:
    Wong, J. Y. *Theory of Ground Vehicles*, 4th ed., Wiley, 2008 — chapters 2–4.

Validation data (to be populated in Weeks 1–2):
    - Ding et al. 2011, IEEE T-RO — single-wheel lunar-rover experiments.
    - Iizuka & Kubota 2011 — grousered wheel experiments.
    - Wong — worked examples from the textbook (used as unit-test ground truth).

Design goal: this module should be pure Python + NumPy, have no hard
dependencies beyond the scientific stack, and run a single wheel evaluation
in well under 1 ms so that 50k+ mission evaluations are practical.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SoilParameters:
    """Bekker / Mohr-Coulomb soil parameters.

    Sourced from :file:`data/soil_simulants.csv`.
    """

    n: float
    """Sinkage exponent (dimensionless)."""

    k_c: float
    """Cohesive modulus, kN/m^(n+1)."""

    k_phi: float
    """Frictional modulus, kN/m^(n+2)."""

    cohesion_kpa: float
    """Soil cohesion c, kPa."""

    friction_angle_deg: float
    """Internal friction angle φ, degrees."""

    shear_modulus_k_m: float = 0.018
    """Janosi-Hanamoto shear-deformation modulus, meters. Default from Wong."""


@dataclass(frozen=True)
class WheelGeometry:
    """Rigid-wheel geometry for a single wheel."""

    radius_m: float
    width_m: float
    grouser_height_m: float = 0.0
    grouser_count: int = 0


@dataclass(frozen=True)
class WheelForces:
    """Steady-state per-wheel outputs of the Bekker-Wong model."""

    drawbar_pull_n: float
    driving_torque_nm: float
    sinkage_m: float
    rolling_resistance_n: float
    slip: float


def single_wheel_forces(
    wheel: WheelGeometry,
    soil: SoilParameters,
    vertical_load_n: float,
    slip: float,
) -> WheelForces:
    """Compute steady-state drawbar pull, torque, and sinkage for one wheel.

    Parameters
    ----------
    wheel
        Wheel geometry.
    soil
        Soil parameters.
    vertical_load_n
        Normal load on the wheel, in newtons (lunar gravity already applied
        by the caller).
    slip
        Longitudinal slip ratio, in [-1, 1]. Positive for driving.

    Returns
    -------
    WheelForces
        Per-wheel forces and kinematic quantities.
    """
    raise NotImplementedError("Implement in Week 1 per project_plan.md §6.")
