"""Lumped-parameter thermal survival check.

Binary pass/fail constraint: does a single-node thermal model of the
avionics enclosure stay within survivability limits (a) during peak sun
and (b) during lunar night? Inputs: avionics heat load, optional RHU
power, surface absorptivity / emissivity, total radiating area,
effective radiative-sink temperatures, and an assumed solar
projected-area fraction.

Thermal architecture is deliberately *not* a continuous design variable
in v1 -- adding it is a week-12 stretch goal (project_plan.md §3.1). For
Week 4 we only care whether a given design survives; the actual
temperatures come out too, for notebook diagnostics.

Model
-----
Single-node steady-state balance:

    Q_in(T)  = alpha * S_eff * A_sunlit + P_internal
    Q_out(T) = eps * sigma * A_rad * (T^4 - T_sink^4)
    Q_in     = Q_out

Solve analytically:

    T_eq = (T_sink^4 + Q_in / (eps * sigma * A_rad))**0.25

No iteration required, which keeps the survival check in the O(1) inner
loop of the mission evaluator.

- **Hot case** (peak sun, full operating power): sun at maximum local
  elevation for the scenario latitude; all avionics + RHU dissipate
  internally.
- **Cold case** (lunar night, hibernation): no solar input; rover
  draws ``hibernation_power_w`` plus the RHU.

References
----------
Gilmore, D. G. (ed.) *Spacecraft Thermal Control Handbook*, Vol. 1
(Aerospace Press / AIAA, 2002). Chapters 1-2 for radiative balance
formulation; Chapter 5 for electronics thermal design.

Heiken et al., *Lunar Sourcebook*, 1991, Ch. 3 & 9 for lunar regolith
surface-temperature extremes (~390 K subsolar, ~100 K at night).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from roverdevkit.power.solar import SOLAR_CONSTANT_AU_1_W_PER_M2

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

STEFAN_BOLTZMANN_W_PER_M2_K4: float = 5.670374419e-8


# ---------------------------------------------------------------------------
# Architecture container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThermalArchitecture:
    """Lumped-parameter thermal model for the avionics enclosure."""

    surface_area_m2: float
    """Total radiating area of the enclosure, m^2."""

    absorptivity: float = 0.3
    """Solar absorptivity alpha in [0, 1]. Default = white paint / OSR."""

    emissivity: float = 0.85
    """IR emissivity eps in [0, 1]. Default = moderate-emissivity coating."""

    solar_projected_area_fraction: float = 0.25
    """Fraction of ``surface_area_m2`` facing the sun at peak illumination.

    0.25 is a reasonable value for a cube-ish enclosure at mid-latitude
    (one of ~four exposed faces presents toward the sun). Override for
    a flat body-mounted slab (0.5) or a vertical panel (closer to 0.33)."""

    insulation_ua_w_per_k: float = 0.5
    """External conductance (W/K). Retained from the v0 stub for API
    compatibility; currently unused by the single-node model. Will be
    used when we split the enclosure into skin + interior in v2."""

    rhu_power_w: float = 0.0
    """Radioisotope heater unit dissipation, W. Zero = no RHUs."""

    hibernation_power_w: float = 2.0
    """Internal dissipation during lunar night in hibernation mode, W.

    Standard-mode avionics power (from the design vector) is used only
    for the hot case; during the cold case we assume the rover is
    hibernating with reduced draw. 2 W is a reasonable micro-rover
    survival-mode load (RTC + thermistor monitoring + heater control)."""

    sink_temp_peak_sun_k: float = 250.0
    """Effective radiative-sink temperature during peak sun, K.

    Mix of hot regolith (~390 K subsolar) visible below the rover and
    cold deep space visible above. 250 K is a defensible weighted
    value; ranges ~230-270 K in real missions."""

    sink_temp_lunar_night_k: float = 100.0
    """Effective radiative-sink temperature during lunar night, K.

    Regolith surface drops to ~100 K; with no solar input and most of
    the hemisphere looking at cold regolith plus deep space, 100 K is
    a slightly optimistic but common tradespace value."""

    min_operating_temp_c: float = -30.0
    """Enclosure interior must stay above this during the cold case."""

    max_operating_temp_c: float = 50.0
    """Enclosure interior must stay below this during the hot case."""

    def __post_init__(self) -> None:
        if self.surface_area_m2 <= 0.0:
            raise ValueError("surface_area_m2 must be positive.")
        if not 0.0 <= self.absorptivity <= 1.0:
            raise ValueError("absorptivity must lie in [0, 1].")
        if not 0.0 <= self.emissivity <= 1.0:
            raise ValueError("emissivity must lie in [0, 1].")
        if not 0.0 < self.solar_projected_area_fraction <= 1.0:
            raise ValueError("solar_projected_area_fraction must lie in (0, 1].")
        if self.insulation_ua_w_per_k <= 0.0:
            raise ValueError("insulation_ua_w_per_k must be positive.")
        if self.rhu_power_w < 0.0:
            raise ValueError("rhu_power_w must be non-negative.")
        if self.hibernation_power_w < 0.0:
            raise ValueError("hibernation_power_w must be non-negative.")
        if self.sink_temp_peak_sun_k <= 0.0 or self.sink_temp_lunar_night_k <= 0.0:
            raise ValueError("sink temperatures must be positive (Kelvin).")
        if self.min_operating_temp_c >= self.max_operating_temp_c:
            raise ValueError("min_operating_temp_c must be < max_operating_temp_c.")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThermalResult:
    """Hot- and cold-case equilibrium temperatures plus the survival flag."""

    peak_sun_temp_c: float
    lunar_night_temp_c: float
    survives: bool


# ---------------------------------------------------------------------------
# Core physics
# ---------------------------------------------------------------------------


def _equilibrium_temperature_k(
    q_in_w: float,
    t_sink_k: float,
    area_rad_m2: float,
    emissivity: float,
) -> float:
    """Single-node steady-state temperature from an absorbed heat load."""
    if q_in_w < 0.0:
        # No physical scenario in this module, but guard anyway.
        q_in_w = 0.0
    q_ratio = q_in_w / (emissivity * STEFAN_BOLTZMANN_W_PER_M2_K4 * area_rad_m2)
    return (t_sink_k**4 + q_ratio) ** 0.25


def _peak_sun_absorbed_w(
    architecture: ThermalArchitecture,
    latitude_deg: float,
    solar_constant_w_per_m2: float,
) -> float:
    """Absorbed solar at the scenario's peak sun elevation.

    Peak elevation on a diurnal lunar day at latitude ``phi`` with zero
    declination is ``90 - |phi|``, so the horizontal-plane insolation
    factor is ``cos(phi)``. Projected area is a fraction of the total
    surface area (configurable, default 0.25).
    """
    elevation_factor = math.cos(math.radians(abs(latitude_deg)))
    sunlit_area_m2 = architecture.surface_area_m2 * architecture.solar_projected_area_fraction
    return architecture.absorptivity * solar_constant_w_per_m2 * elevation_factor * sunlit_area_m2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_thermal(
    architecture: ThermalArchitecture,
    avionics_power_w: float,
    latitude_deg: float,
    *,
    solar_constant_w_per_m2: float = SOLAR_CONSTANT_AU_1_W_PER_M2,
) -> ThermalResult:
    """Steady-state hot- and cold-case temperatures plus the pass/fail flag.

    Parameters
    ----------
    architecture
        Lumped-parameter thermal model for the avionics enclosure.
    avionics_power_w
        Nominal (operating-mode) avionics dissipation, W. Comes from
        the design vector. Applied in the hot case; the cold case uses
        ``architecture.hibernation_power_w`` instead.
    latitude_deg
        Scenario latitude, deg, in [-90, 90].
    solar_constant_w_per_m2
        Top-of-atmosphere solar irradiance, W/m^2. Default 1 AU value.

    Returns
    -------
    ThermalResult
        Peak-sun and lunar-night temperatures (deg C) and a boolean
        ``survives`` = ``min_operating_temp_c <= lunar_night_temp_c
        and peak_sun_temp_c <= max_operating_temp_c``.
    """
    if avionics_power_w < 0.0:
        raise ValueError("avionics_power_w must be non-negative.")
    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError("latitude_deg must lie in [-90, 90].")

    a_rad = architecture.surface_area_m2
    eps = architecture.emissivity

    # Hot case: peak sun, operating power.
    q_solar = _peak_sun_absorbed_w(architecture, latitude_deg, solar_constant_w_per_m2)
    q_hot = q_solar + avionics_power_w + architecture.rhu_power_w
    t_hot_k = _equilibrium_temperature_k(q_hot, architecture.sink_temp_peak_sun_k, a_rad, eps)

    # Cold case: no sun, hibernation power.
    q_cold = architecture.hibernation_power_w + architecture.rhu_power_w
    t_cold_k = _equilibrium_temperature_k(q_cold, architecture.sink_temp_lunar_night_k, a_rad, eps)

    peak_c = t_hot_k - 273.15
    cold_c = t_cold_k - 273.15
    survives = (
        architecture.min_operating_temp_c <= cold_c and peak_c <= architecture.max_operating_temp_c
    )
    return ThermalResult(
        peak_sun_temp_c=peak_c,
        lunar_night_temp_c=cold_c,
        survives=survives,
    )


def survives_mission(
    architecture: ThermalArchitecture,
    avionics_power_w: float,
    latitude_deg: float,
    *,
    solar_constant_w_per_m2: float = SOLAR_CONSTANT_AU_1_W_PER_M2,
) -> bool:
    """Boolean wrapper around :func:`evaluate_thermal`.

    Kept as a separate entry point so the mission evaluator can call a
    name that matches what the schema describes as a pass/fail flag.
    """
    return evaluate_thermal(
        architecture,
        avionics_power_w,
        latitude_deg,
        solar_constant_w_per_m2=solar_constant_w_per_m2,
    ).survives


def default_architecture_for_design(
    surface_area_m2: float,
    *,
    rhu_power_w: float = 0.0,
    hibernation_power_w: float = 2.0,
) -> ThermalArchitecture:
    """Convenience factory producing a nominal enclosure from bare dimensions.

    Used by the mission evaluator when the design vector does not carry
    explicit thermal parameters (v1 of the project; see §3.1). Changing
    the defaults here is a controlled way to run a thermal-architecture
    sweep without plumbing new fields into ``DesignVector``.
    """
    base = ThermalArchitecture(surface_area_m2=surface_area_m2)
    return replace(base, rhu_power_w=rhu_power_w, hibernation_power_w=hibernation_power_w)
