"""Analytical terramechanics: Bekker-Wong pressure-sinkage + Janosi-Hanamoto shear.

Implements single-wheel drawbar pull, sinkage, and driving torque as a
function of wheel geometry, vertical load, slip, and soil parameters.

Primary reference:
    Wong, J. Y. *Theory of Ground Vehicles*, 4th ed., Wiley, 2008 — chapters 2--4.
    Following equation numbering from ch. 2 (pressure-sinkage), ch. 4
    (wheel-soil interaction), and Wong & Reece (1967).

Assumptions baked in for the rigid-wheel model:

- Entry angle θ₁ measured from the wheel centre, positive forward.
- Exit angle θ₂ = 0 (Wong's standard rigid-wheel assumption).
- Transition angle for peak radial stress θ_m = (c₁ + c₂·|s|)·θ₁
  with c₁ = 0.4, c₂ = 0.2 (Wong 2008; Ding 2011 report very similar
  values from single-wheel experiments).
- Soil parameters from Bekker + Mohr-Coulomb.
- Grousers are ignored in the analytical path; their contribution is
  picked up by the PyChrono SCM correction layer (Path 2).

Validation roadmap:
    - Ding et al. 2011, IEEE T-RO — single-wheel lunar-rover experiments.
    - Iizuka & Kubota 2011 — grousered wheel experiments.
    - Wong — worked examples from the textbook (used as unit-test ground truth).

Design goal: pure Python + NumPy + SciPy, no hard dependencies beyond the
scientific stack, < 1 ms per ``single_wheel_forces`` call so that 50k+
mission evaluations are practical on a laptop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

# Stress-distribution coefficients (Wong 2008, ch. 4).
# Ding 2011 fits give (c1, c2) ∈ (0.18–0.43, 0.09–0.25) depending on soil;
# Wong's standard values below are a reasonable single-value default for
# the analytical path. The Path-2 correction layer absorbs residuals.
_C1_THETA_M: float = 0.4
_C2_THETA_M: float = 0.2

# Trapezoidal integration grid density. 100 points is well inside the
# regime where integration error is much smaller than model-form error
# (Bekker-Wong itself is a ±15–30 % model). Profiled at ~0.3 ms per call.
_N_QUAD: int = 100


# ---------------------------------------------------------------------------
# Data classes (frozen — safe to hash / cache)
# ---------------------------------------------------------------------------


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
    """Janosi-Hanamoto shear-deformation modulus K, meters. Default from Wong."""


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
    entry_angle_rad: float
    """Entry angle θ₁ that satisfies vertical force balance."""


# ---------------------------------------------------------------------------
# Internal helpers (all SI once inside)
# ---------------------------------------------------------------------------


def _effective_modulus_pa_per_m_n(soil: SoilParameters, width_m: float) -> float:
    """Combined modulus (k_c/b + k_phi) converted to SI: Pa/m^n.

    Bekker's pressure-sinkage law is::

        p(z) = (k_c/b + k_phi) z^n

    With ``k_c`` in kN/m^(n+1) and ``k_phi`` in kN/m^(n+2), the combination
    has units kN/m^(n+2). Multiply by 10³ to get Pa/m^n.
    """
    return (soil.k_c / width_m + soil.k_phi) * 1000.0


def _integrate_forces(
    theta_1: float,
    wheel: WheelGeometry,
    soil: SoilParameters,
    slip: float,
) -> tuple[float, float, float]:
    """Return ``(W, DP, T)`` in SI for a given entry angle θ₁.

    ``W`` is the integrated vertical force, ``DP`` the drawbar pull, ``T``
    the driving torque about the wheel axis.
    """
    if theta_1 <= 0.0:
        return 0.0, 0.0, 0.0

    radius_m = wheel.radius_m
    width_m = wheel.width_m
    n = soil.n
    phi_rad = math.radians(soil.friction_angle_deg)
    cohesion_pa = soil.cohesion_kpa * 1000.0
    shear_modulus_m = soil.shear_modulus_k_m
    k_eff = _effective_modulus_pa_per_m_n(soil, width_m)

    theta_m = (_C1_THETA_M + _C2_THETA_M * abs(slip)) * theta_1

    # Angular grid from θ₂ = 0 to θ₁ (Wong assumes θ₂ = 0 for rigid wheels).
    theta: NDArray[np.float64] = np.linspace(0.0, theta_1, _N_QUAD)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_theta_1 = math.cos(theta_1)

    # Radial normal stress σ(θ). Piecewise per Wong:
    #   front (θ_m ≤ θ ≤ θ₁): σ = k_eff R^n (cos θ − cos θ₁)^n
    #   rear  (0   ≤ θ < θ_m): σ evaluated at a linearly-mapped θ★ such
    #     that θ★ = θ₁ at θ = 0 and θ★ = θ_m at θ = θ_m.
    arg_front = np.maximum(cos_theta - cos_theta_1, 0.0)
    sigma_front = k_eff * radius_m**n * arg_front**n

    if theta_m > 0.0:
        ratio = theta / theta_m
        theta_mapped = theta_1 - ratio * (theta_1 - theta_m)
        arg_rear = np.maximum(np.cos(theta_mapped) - cos_theta_1, 0.0)
        sigma_rear = k_eff * radius_m**n * arg_rear**n
    else:
        sigma_rear = np.zeros_like(theta)

    sigma = np.where(theta >= theta_m, sigma_front, sigma_rear)

    # Shear deformation j(θ) under rolling-with-slip kinematics (Wong).
    # Positive for driving (slip > 0); the tan-function shape of
    # Janosi-Hanamoto uses the magnitude of j, and the sign of the
    # resulting shear stress follows the sign of j.
    j = radius_m * ((theta_1 - theta) - (1.0 - slip) * (math.sin(theta_1) - sin_theta))

    # Janosi-Hanamoto shear stress τ(θ) = τ_max (1 − e^(−|j|/K)) sgn(j).
    tau_max = cohesion_pa + sigma * math.tan(phi_rad)
    tau = tau_max * (1.0 - np.exp(-np.abs(j) / shear_modulus_m)) * np.sign(j)

    # Force integrals (Wong eq. 4.24–4.26).
    integrand_w = sigma * cos_theta + tau * sin_theta
    integrand_dp = tau * cos_theta - sigma * sin_theta
    integrand_t = tau

    scale = width_m * radius_m
    vertical_load = scale * float(np.trapezoid(integrand_w, theta))
    drawbar_pull = scale * float(np.trapezoid(integrand_dp, theta))
    torque = scale * radius_m * float(np.trapezoid(integrand_t, theta))

    return vertical_load, drawbar_pull, torque


def _compaction_resistance(sinkage_m: float, wheel: WheelGeometry, soil: SoilParameters) -> float:
    """Bekker plate compaction resistance, reported as a diagnostic.

    .. math::

        R_c = b \\int_0^{z_0} p(z) \\, dz
            = \\frac{b\\,(k_c/b + k_\\phi)}{n+1}\\, z_0^{n+1}

    At slip = 0 the integrated drawbar pull should be approximately
    ``-R_c`` (pure compaction drag), which we assert in the tests.
    """
    if sinkage_m <= 0.0:
        return 0.0
    k_eff = _effective_modulus_pa_per_m_n(soil, wheel.width_m)
    return wheel.width_m * k_eff * sinkage_m ** (soil.n + 1.0) / (soil.n + 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def single_wheel_forces(
    wheel: WheelGeometry,
    soil: SoilParameters,
    vertical_load_n: float,
    slip: float,
) -> WheelForces:
    """Compute steady-state drawbar pull, torque, and sinkage for one wheel.

    The entry angle θ₁ is found by root-finding on the vertical-force
    residual (integrated W equals the applied load). All outputs are in
    SI units regardless of the soil-parameter unit conventions in the
    CSV catalogue.

    Parameters
    ----------
    wheel
        Wheel geometry.
    soil
        Soil parameters.
    vertical_load_n
        Normal load on the wheel, in newtons (lunar gravity already
        applied by the caller).
    slip
        Longitudinal slip ratio, in [-1, 1]. Positive for driving
        (``slip = 1 − V/(Rω)``).

    Returns
    -------
    WheelForces
        Per-wheel forces and kinematic quantities, plus the solved
        entry angle.

    Raises
    ------
    ValueError
        If no entry angle in ``(0, π/2)`` satisfies the force balance —
        typically because the wheel is fully buried (soil is too soft
        or load too high for the geometry).
    """
    if vertical_load_n <= 0.0:
        raise ValueError("vertical_load_n must be positive")
    if not -1.0 <= slip <= 1.0:
        raise ValueError("slip must lie in [-1, 1]")

    def residual(theta_1: float) -> float:
        w, _, _ = _integrate_forces(theta_1, wheel, soil, slip)
        return w - vertical_load_n

    theta_low = 1e-5
    theta_high = math.pi / 2.0 - 1e-4
    try:
        theta_1 = brentq(residual, theta_low, theta_high, xtol=1e-6, rtol=1e-6)
    except ValueError as exc:
        r_low = residual(theta_low)
        r_high = residual(theta_high)
        raise ValueError(
            "could not find entry angle satisfying vertical force balance "
            f"(load={vertical_load_n:.1f} N, R={wheel.radius_m:.3f} m, "
            f"b={wheel.width_m:.3f} m). At θ₁=ε residual={r_low:.1f} N, "
            f"at θ₁=π/2−ε residual={r_high:.1f} N. Likely wheel is fully "
            "buried (soil too soft or load too high for this geometry)."
        ) from exc

    _, drawbar_pull, torque = _integrate_forces(theta_1, wheel, soil, slip)
    sinkage = wheel.radius_m * (1.0 - math.cos(theta_1))
    rolling_resistance = _compaction_resistance(sinkage, wheel, soil)

    return WheelForces(
        drawbar_pull_n=drawbar_pull,
        driving_torque_nm=torque,
        sinkage_m=sinkage,
        rolling_resistance_n=rolling_resistance,
        slip=slip,
        entry_angle_rad=theta_1,
    )
