"""Published-rover design vectors and mission scenarios for Week 5.

This module codifies the three rovers we compare the evaluator against
(project_plan.md §6 W5) as :class:`DesignVector` + :class:`MissionScenario`
pairs, plus the published truth numbers in
``data/published_traverse_data.csv``.

Scope and choices
-----------------
- **Included:** Pragyan (Chandrayaan-3, 2023), Yutu-2 (Chang'e-4, 2019),
  Sojourner (Mars Pathfinder, 1997). Rashid is *not* in the traverse
  comparison because its mission ended on the Hakuto-R lander failure
  before deployment; it was already validated for mass in Week 3 and
  re-appears in Week 12 as the rediscovery-test target.
- **Sojourner included despite being a Mars rover** to (a) give us a
  non-lunar comparison point with actual published traverse distance,
  and (b) exercise the evaluator's ``gravity_m_per_s2`` keyword as a
  scaling sanity check. Its row in the registry carries
  ``gravity_m_per_s2 = 3.71`` to signal the override.
- **Not a tradespace input.** These scenarios live next to the canonical
  four in :data:`SCENARIO_DIR` but are excluded from
  :func:`list_scenarios` so Phase-3 sweeps never pick them up.

Every design-vector field that is not directly published has an entry
in :attr:`RoverRegistryEntry.imputation_notes`; these notes mirror the
pattern from the Week-3 mass validation set for consistency.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from roverdevkit.mission.scenarios import load_scenario
from roverdevkit.power.thermal import ThermalArchitecture
from roverdevkit.schema import DesignVector, MissionScenario

GRAVITY_MOON_M_PER_S2: float = 1.625
GRAVITY_MARS_M_PER_S2: float = 3.71

DEFAULT_TRUTH_CSV: Path = (
    Path(__file__).resolve().parents[2] / "data" / "published_traverse_data.csv"
)


# ---------------------------------------------------------------------------
# Registry entry + truth data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoverRegistryEntry:
    """One rover bundled with the scenario it actually flew.

    Attributes
    ----------
    rover_name
        Short key used to look up published truth in
        :func:`load_truth_table`.
    design
        12-D design vector reconstructed from public specs + documented
        imputations.
    scenario
        Mission context matching the real rover's operating environment.
    gravity_m_per_s2
        Passed through to the evaluator as ``gravity_m_per_s2``. Lunar
        by default; Sojourner overrides to Mars.
    imputation_notes
        Per-field notes on which design-vector entries were imputed and
        how.
    """

    rover_name: str
    design: DesignVector
    scenario: MissionScenario
    gravity_m_per_s2: float
    thermal_architecture: ThermalArchitecture
    panel_efficiency: float
    """DC-level conversion efficiency at the rover's operating point.

    Distinct from the tradespace-default 0.28 (GaAs triple-junction
    beginning-of-life) because real rovers use different cell techs
    and see end-of-life degradation that the default doesn't model."""

    panel_dust_factor: float
    """Mission-integrated dust-transmission factor in (0, 1].

    Rover-specific because real dust accumulation is highly
    mission-dependent; lunar day-1 values differ from steady-state."""

    imputation_notes: str


@dataclass(frozen=True)
class PublishedTruth:
    """Published truth values for one rover-scenario pair."""

    rover_name: str
    scenario_name: str
    traverse_m_published: float
    traverse_m_low: float
    traverse_m_high: float
    peak_solar_power_w_published: float
    peak_solar_power_w_low: float
    peak_solar_power_w_high: float
    thermal_survival_published: bool
    mission_duration_published_days: float
    citation: str
    notes: str


# ---------------------------------------------------------------------------
# Registry builders
# ---------------------------------------------------------------------------


def _pragyan_entry() -> RoverRegistryEntry:
    # Published specs: 26 kg total (ISRO press kit), 6 wheels at r=85 mm,
    # ~50 W avionics during active ops, ~60 Wh battery, ~0.5 m^2
    # deployable solar array.
    # Imputations (mirrors mass_validation_set.csv row for consistency):
    # - wheel_width_m = 0.07 (scaled from n_wheels geometry);
    # - chassis_mass_kg = 10 (~38 % of total per class ROT);
    # - wheelbase_m = 0.5 (from published images);
    # - nominal_speed_mps = 0.01 (10 mm/s, published ops limit);
    # - drive_duty_cycle = 0.15 (~1.5 h drive per 10 h active window,
    #   consistent with ~100 m over 10 Earth days);
    # - grouser_height_m, grouser_count from class heritage (Rashid/
    #   Yutu-style 8 mm x 12 grousers).
    design = DesignVector(
        wheel_radius_m=0.085,
        wheel_width_m=0.07,
        grouser_height_m=0.008,
        grouser_count=12,
        n_wheels=6,
        chassis_mass_kg=10.0,
        wheelbase_m=0.5,
        solar_area_m2=0.5,
        battery_capacity_wh=60.0,
        avionics_power_w=20.0,
        nominal_speed_mps=0.01,
        drive_duty_cycle=0.15,
    )
    # Thermal: Pragyan did NOT carry RHUs and died in lunar night.
    # Default architecture (rhu_power_w=0) correctly predicts failure.
    thermal = ThermalArchitecture(
        surface_area_m2=0.25,
        rhu_power_w=0.0,
        hibernation_power_w=2.0,
    )
    return RoverRegistryEntry(
        rover_name="Pragyan",
        design=design,
        scenario=load_scenario("chandrayaan3_pragyan"),
        gravity_m_per_s2=GRAVITY_MOON_M_PER_S2,
        thermal_architecture=thermal,
        panel_efficiency=0.22,  # ISRO space-grade triple-junction, BOL
        panel_dust_factor=0.85,  # Lunar Day 1 only; limited dust build-up
        imputation_notes=(
            "wheel_width, wheelbase, grouser_height/count, chassis_mass, "
            "nominal_speed_mps, drive_duty_cycle imputed from class "
            "heritage and published ops. avionics_power set to 20 W "
            "(design-space floor for a 26 kg rover)."
        ),
    )


def _yutu2_entry() -> RoverRegistryEntry:
    # Published specs (Di et al. 2020; Ding et al. 2022): 135 kg total,
    # 6 wheels at r=150 mm, wheel width ~150 mm, two-wing deployable
    # solar array ~1.3 m^2, ~130 Wh Li-ion pack, continuous drive speed
    # 40 mm/s with a drive duty cycle concentrated in a few Earth-day
    # ops window per lunar day.
    # Imputations:
    # - chassis_mass_kg = 30 (the 70 kg validation-set value bakes in
    #   the 25 kg science payload; 30 kg is the "chassis+bus" minus
    #   payload for pure-mobility modelling);
    # - wheelbase_m = 1.0 (published photos);
    # - grouser specs: h=0.012 m x 18 (Yutu-class wheels are grousered);
    # - avionics_power_w = 20 (steady-state CPU+comms+sensors; 40 W only
    #   during peak drive+heater operation, which is a different case);
    # - drive_duty_cycle = 0.15 (~7 h drive per day during a 5-day active
    #   window; matches per-lunar-day ~25 m drive distance).
    # Note: Yutu-2 is out of the 5-50 kg design-space class; chassis_mass
    # is held at the schema's 35 kg ceiling, and the prediction will
    # under-call its capability proportionally. Documented limitation.
    design = DesignVector(
        wheel_radius_m=0.15,
        wheel_width_m=0.15,
        grouser_height_m=0.012,
        grouser_count=18,
        n_wheels=6,
        chassis_mass_kg=35.0,  # schema ceiling; true ~30-40 kg ex-payload
        wheelbase_m=1.0,
        solar_area_m2=1.3,
        battery_capacity_wh=130.0,
        avionics_power_w=20.0,
        nominal_speed_mps=0.04,
        drive_duty_cycle=0.15,
    )
    # Thermal: Yutu-class carries Pu-238 RHUs on a thermally-controlled
    # avionics box wrapped in MLI with low-alpha/high-eps surface
    # finish (silverised OSR, alpha~0.15). The lumped-parameter thermal
    # model assumes the full surface_area_m2 radiates to the cold sink,
    # which is pessimistic for a real MLI-insulated box; we use an
    # "effective radiating area" of 0.10 m^2 to represent the MLI
    # reduction. Combined with 15 W RHU + 5 W hibernation, this gives
    # cold-case equilibrium ~-18 C and hot-case ~+40 C, both in-spec.
    thermal = ThermalArchitecture(
        surface_area_m2=0.10,
        absorptivity=0.15,
        rhu_power_w=15.0,
        hibernation_power_w=5.0,
        max_operating_temp_c=60.0,  # industrial-temp-range Chinese avionics
    )
    return RoverRegistryEntry(
        rover_name="Yutu-2",
        design=design,
        scenario=load_scenario("change4_yutu2_per_lunar_day"),
        gravity_m_per_s2=GRAVITY_MOON_M_PER_S2,
        thermal_architecture=thermal,
        panel_efficiency=0.20,  # Chinese triple-junction EOL after many
        panel_dust_factor=0.55,  # lunar days (major dust accumulation)
        imputation_notes=(
            "Yutu-2 is out-of-class (135 kg vs 5-50 kg design space); "
            "chassis_mass held at 35 kg ceiling. wheelbase, grouser specs, "
            "drive_duty_cycle imputed from published images and the per-"
            "lunar-day ~25 m drive distance target."
        ),
    )


def _sojourner_entry() -> RoverRegistryEntry:
    # Published specs (Wilcox & Nguyen 1998): 10.6 kg total, 6 wheels at
    # r=65 mm x 80 mm width, ~0.22 m^2 solar array (GaAs), 40 Wh
    # primary battery, ~16 W peak power, cumulative ~100 m over 83 sols.
    # Imputations:
    # - chassis_mass_kg = 3.5 (~33 % of total per Wilcox & Nguyen);
    # - wheelbase_m = 0.3 (published mechanical drawings);
    # - grouser specs h=0.010 x 12 (Sojourner had stainless-steel
    #   cleats; 10 mm x 12 cleats per wheel);
    # - avionics_power_w = 5 (design-space floor; Sojourner's CPU was
    #   a 2 MHz 80C85 drawing very little);
    # - nominal_speed_mps = 0.01 (10 mm/s, Wilcox & Nguyen);
    # - drive_duty_cycle = 0.1 (Sojourner drove only a small fraction
    #   of each sol; ~1 m per sol x 100 sols / 83 sols ~ minutes of
    #   drive per day).
    design = DesignVector(
        wheel_radius_m=0.065,
        wheel_width_m=0.08,
        grouser_height_m=0.010,
        grouser_count=12,
        n_wheels=6,
        chassis_mass_kg=3.5,
        wheelbase_m=0.3,
        solar_area_m2=0.22,
        battery_capacity_wh=40.0,
        avionics_power_w=5.0,
        nominal_speed_mps=0.01,
        drive_duty_cycle=0.1,
    )
    # Thermal: Sojourner carried 3x Pu-238 RHUs (~1 W each) and used a
    # Warm Electronics Box with silica aerogel insulation plus phase-
    # change materials (Pathfinder spec). The lunar-tuned model is a
    # poor match; we approximate with a much warmer sink (Mars-night
    # is ~180 K, and the WEB's skin only sees ~200-220 K), small
    # effective radiating area (0.05 m^2 after aerogel), and the
    # nominal 3 W RHU power. This is a known limitation; the Sojourner
    # thermal prediction is more about "sanity-check the wrapper" than
    # physical fidelity.
    thermal = ThermalArchitecture(
        surface_area_m2=0.05,
        rhu_power_w=3.0,
        hibernation_power_w=2.0,
        sink_temp_lunar_night_k=210.0,  # Mars-night WEB skin proxy
    )
    return RoverRegistryEntry(
        rover_name="Sojourner",
        design=design,
        scenario=load_scenario("mpf_sojourner_ares_vallis"),
        gravity_m_per_s2=GRAVITY_MARS_M_PER_S2,
        thermal_architecture=thermal,
        panel_efficiency=0.17,  # GaAs/Ge single-junction of 1997 vintage
        panel_dust_factor=0.80,  # Mars dust accumulation over 83 sols
        imputation_notes=(
            "Mars-gravity case. chassis_mass_kg, wheelbase, grouser specs, "
            "drive_duty_cycle imputed per Wilcox & Nguyen 1998. Soil "
            "model uses GRC-1 (lunar simulant) as a rough proxy for "
            "Ares Vallis regolith; known limitation."
        ),
    )


def registry() -> tuple[RoverRegistryEntry, ...]:
    """Return the frozen tuple of (rover, scenario, gravity) triples."""
    return (_pragyan_entry(), _yutu2_entry(), _sojourner_entry())


def registry_by_name(name: str) -> RoverRegistryEntry:
    """Look up a single registry entry by rover name."""
    for entry in registry():
        if entry.rover_name == name:
            return entry
    raise KeyError(f"unknown rover {name!r}; registry has {[e.rover_name for e in registry()]}.")


# ---------------------------------------------------------------------------
# Published truth loader
# ---------------------------------------------------------------------------


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    raise ValueError(f"unparseable boolean: {value!r}")


def load_truth_table(csv_path: Path | str | None = None) -> list[PublishedTruth]:
    """Read ``data/published_traverse_data.csv``."""
    path = Path(csv_path) if csv_path else DEFAULT_TRUTH_CSV
    rows: list[PublishedTruth] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                PublishedTruth(
                    rover_name=row["rover_name"],
                    scenario_name=row["scenario_name"],
                    traverse_m_published=float(row["traverse_m_published"]),
                    traverse_m_low=float(row["traverse_m_low"]),
                    traverse_m_high=float(row["traverse_m_high"]),
                    peak_solar_power_w_published=float(row["peak_solar_power_w_published"]),
                    peak_solar_power_w_low=float(row["peak_solar_power_w_low"]),
                    peak_solar_power_w_high=float(row["peak_solar_power_w_high"]),
                    thermal_survival_published=_parse_bool(row["thermal_survival_published"]),
                    mission_duration_published_days=float(row["mission_duration_published_days"]),
                    citation=row["citation"],
                    notes=row["notes"],
                )
            )
    return rows


def truth_by_rover(rover_name: str, csv_path: Path | str | None = None) -> PublishedTruth:
    """Fetch the published-truth row for one rover."""
    for row in load_truth_table(csv_path):
        if row.rover_name == rover_name:
            return row
    raise KeyError(f"no published-truth row for rover {rover_name!r}.")
