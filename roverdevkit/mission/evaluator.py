"""Top-level mission evaluator.

This is the **primary artifact** of the project (project_plan.md §2). After
the W7.7 traverse-loop lift-out it runs in ~40 ms / mission with the
wheel-level SCM correction enabled (~30 ms BW-only); the
:mod:`roverdevkit.surrogate` layer is an *optional* acceleration and
uncertainty layer used for NSGA-II inner loops, batch sensitivity studies,
and prediction-interval calibration. Most Phase-3 workflows can run against
this evaluator directly. Every ML claim in the paper is grounded in what
this function computes.

Capability envelope vs operational utilisation
----------------------------------------------
The outputs describe what the *hardware can deliver* if ops commands it
to drive at the design vector's ``drive_duty_cycle`` for the whole
mission window. Real missions typically command **below** the designed
duty (Pragyan ~0.02, Yutu-2 ~0.015, Sojourner ~0.01) for reasons the
evaluator does not model: uplink / command cycles, science campaigns,
thermal hot-soak pauses, fault response. Utilisation-adjusted range at
a lower commanded duty ``u`` is a post-hoc rescaling --
``range_u = range_km * u / drive_duty_cycle`` for ``u <= drive_duty_cycle``
-- not an evaluator input. This is the same engineering-vs-operations
distinction that JPL Team X and ESA CDF studies use.

Pipeline
--------
1. Mass model  -> total vehicle mass + per-subsystem breakdown
   (:mod:`roverdevkit.mass`).
2. Thermal     -> binary survive-the-mission flag
   (:mod:`roverdevkit.power.thermal`).
3. Soil lookup -> Bekker-Wong parameters for the scenario's simulant
   (:mod:`roverdevkit.terramechanics.soils`).
4. Capability  -> max climbable slope on this soil
   (:mod:`roverdevkit.mission.capability`).
5. Traverse    -> time-stepped run-to-completion log
   (:mod:`roverdevkit.mission.traverse_sim`).
6. Aggregate   -> MissionMetrics (schema).

Public API::

    from roverdevkit.mission.evaluator import evaluate
    from roverdevkit.mission.scenarios import load_scenario
    from roverdevkit.schema import DesignVector

    metrics = evaluate(design, load_scenario("equatorial_mare_traverse"))

Design notes
------------
- The evaluator **always returns** a :class:`MissionMetrics` object; it
  does not short-circuit on design failures. Constraint flags
  (``thermal_survival``, ``motor_torque_ok``) and continuous metrics
  (``energy_margin_pct``, ``range_km``) encode the failure modes instead.
  This is critical for training the Phase-2 surrogate over the full
  design space including infeasible regions.
- ``motor_torque_ok`` is judged against the same peak-torque envelope
  the mass model used to size the motor subsystem
  (:func:`roverdevkit.mass.parametric_mers._motors_mass`). Keeping the
  two definitions tied together prevents silent drift.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from roverdevkit.mass.parametric_mers import (
    MassBreakdown,
    MassModelParams,
    estimate_mass_from_design,
)
from roverdevkit.mission.capability import max_climbable_slope_deg
from roverdevkit.mission.traverse_sim import TraverseLog, run_traverse
from roverdevkit.power.thermal import (
    ThermalArchitecture,
    default_architecture_for_design,
    survives_mission,
)
from roverdevkit.schema import DesignVector, MissionMetrics, MissionScenario
from roverdevkit.terramechanics.bekker_wong import SoilParameters, WheelGeometry
from roverdevkit.terramechanics.correction_model import (
    DEFAULT_CORRECTION_PATH,
    WheelLevelCorrection,
    load_correction_or_none,
)
from roverdevkit.terramechanics.soils import get_soil_parameters


@dataclass(frozen=True)
class DetailedEvaluation:
    """Full evaluator output: headline metrics plus supporting artefacts.

    Returned by :func:`evaluate_verbose`. Phase-2 dataset generation needs
    the :class:`TraverseLog` so it can compute aggregate sub-model stats
    (peak/mean/p95 of drawbar pull, sinkage, motor torque, solar power,
    battery SOC) that the single-scalar :class:`MissionMetrics` does not
    expose. The :class:`MassBreakdown` is kept alongside so per-subsystem
    mass is recoverable without re-running the mass model.
    """

    metrics: MissionMetrics
    log: TraverseLog
    mass: MassBreakdown


def _sizing_peak_torque_nm(
    total_mass_kg: float,
    wheel_radius_m: float,
    n_wheels: int,
    params: MassModelParams,
) -> float:
    """Peak per-wheel torque the motor subsystem is sized to deliver.

    Mirrors the calculation inside
    :func:`roverdevkit.mass.parametric_mers._motors_mass`. The safety
    factor is baked in -- this is the *available* torque ceiling, so
    ``motor_torque_ok`` is True iff the traverse-observed peak torque
    stays under this number.
    """
    weight_per_wheel_n = total_mass_kg * params.gravity_moon_m_per_s2 / n_wheels
    return (
        params.motor_sizing_safety_factor
        * params.motor_peak_friction_coef
        * weight_per_wheel_n
        * wheel_radius_m
    )


def _energy_margin_pct(log: TraverseLog, min_soc: float) -> float:
    """Discretionary-energy margin at end of mission, percent.

    0 % = battery sitting on the DoD floor; 100 % = full charge above
    the floor. Defined as ``(SOC_end - min_SOC) / (1 - min_SOC) * 100``
    with a clamp at 0 so unsurvivable missions return 0 rather than a
    negative number.

    This is the **reporting** metric (clipped, monotonically interpretable).
    For the surrogate-training signal that does not saturate at 0/100, see
    :func:`_energy_margin_raw_pct`.
    """
    if log.state_of_charge.size == 0:
        return 0.0
    soc_end = float(log.state_of_charge[-1])
    span = max(1e-9, 1.0 - min_soc)
    return max(0.0, (soc_end - min_soc) / span * 100.0)


def _energy_margin_raw_pct(log: TraverseLog) -> float:
    """Mission-integrated energy balance as a percentage of consumption.

    Defined as ``(E_generated - E_consumed) / E_consumed * 100``,
    unbounded on both sides. Negative ⇒ net energy deficit; >0 ⇒ surplus
    generation. Used by the Phase-2 surrogate because it does not
    saturate when SOC sits at 1.0 (benign scenarios) or at the DoD floor
    (polar night), unlike :func:`_energy_margin_pct`.

    Computed via trapezoidal integration of the traverse log's
    ``power_in_w`` (solar input) and ``power_out_w`` (avionics +
    mobility). Time is assumed monotonic and in seconds.
    """
    if log.t_s.size < 2:
        return 0.0
    t = log.t_s
    e_in_wh = float(np.trapezoid(log.power_in_w, t)) / 3600.0
    e_out_wh = float(np.trapezoid(log.power_out_w, t)) / 3600.0
    if e_out_wh <= 1e-9:
        return 0.0
    return (e_in_wh - e_out_wh) / e_out_wh * 100.0


def evaluate_verbose(
    design: DesignVector,
    scenario: MissionScenario,
    *,
    mass_params: MassModelParams | None = None,
    thermal_architecture: ThermalArchitecture | None = None,
    gravity_m_per_s2: float | None = None,
    soil_override: SoilParameters | None = None,
    use_scm_correction: bool = False,
    correction: WheelLevelCorrection | None = None,
    force_backend: str = "bw",
) -> DetailedEvaluation:
    """Full evaluator: headline metrics plus traverse log and mass breakdown.

    Same physics pipeline as :func:`evaluate`, but returns the supporting
    artefacts needed by the Phase-2 dataset builder (aggregate sub-model
    statistics from the :class:`TraverseLog`) and per-subsystem mass
    introspection for validation.

    Parameters
    ----------
    design
        12-D design vector.
    scenario
        Mission context (latitude, terrain, distance, sun geometry).
    mass_params
        Optional :class:`MassModelParams` override.
    thermal_architecture
        Optional :class:`ThermalArchitecture` override. If ``None``, a
        default enclosure is built from a fraction of the chassis using
        :func:`default_architecture_for_design`.
    gravity_m_per_s2
        Surface gravity override (e.g. for off-Moon test scenarios).
        All current registry rovers run at lunar gravity since the
        Mars-gravity Sojourner sentinel was removed (2026-04-25).
    soil_override
        Optional :class:`SoilParameters` to use instead of the
        catalogue lookup on ``scenario.soil_simulant``. The Phase-2
        LHS sweep uses this to inject per-sample jittered Bekker
        parameters so the surrogate learns a continuous soil → metric
        mapping instead of a four-category one
        (``project_plan.md`` §6).
    use_scm_correction
        When ``True`` and ``correction`` is ``None``, loads the
        production wheel-level correction artifact from
        :data:`roverdevkit.terramechanics.correction_model.DEFAULT_CORRECTION_PATH`
        and falls back to the BW-only path with a one-time ``UserWarning``
        if the file is missing (e.g. during the dataset rebuild that
        produces the artifact in the first place). Ignored when
        ``correction`` is not ``None`` so callers can pre-load once and
        opt in per-call without flipping a separate flag.
    correction
        Pre-loaded
        :class:`roverdevkit.terramechanics.correction_model.WheelLevelCorrection`.
        Allows the LHS dataset builder to load the artifact once and
        share it across worker processes / repeated calls instead of
        joblib-loading on every evaluate.
    force_backend
        Wheel-level force backend (``"bw"`` default or ``"scm"`` for
        the Week-7.7 bake-off — runs PyChrono SCM directly inside the
        slip solve). ``"scm"`` ignores ``correction`` /
        ``use_scm_correction`` since SCM-direct is the ground truth
        the correction tries to approximate.
    """
    if correction is None and use_scm_correction and force_backend != "scm":
        correction = load_correction_or_none(DEFAULT_CORRECTION_PATH, on_missing="warn")

    mass_params = mass_params or MassModelParams()
    if gravity_m_per_s2 is not None and not math.isclose(
        gravity_m_per_s2, mass_params.gravity_moon_m_per_s2
    ):
        mass_params = dataclasses.replace(mass_params, gravity_moon_m_per_s2=gravity_m_per_s2)
    active_g = mass_params.gravity_moon_m_per_s2

    breakdown: MassBreakdown = estimate_mass_from_design(design, params=mass_params)
    total_mass_kg = breakdown.total_kg

    if thermal_architecture is None:
        # Rough enclosure surface-area proxy: scales with chassis mass
        # via a cube-root law (box side ~ mass^(1/3) * density^(-1/3)).
        # 0.02 m^2/kg^(2/3) is a coarse calibration that gives ~0.07 m^2
        # for a 6 kg chassis and ~0.24 m^2 for a 30 kg chassis.
        surface_area_m2 = 0.02 * (design.chassis_mass_kg ** (2.0 / 3.0)) + 0.05
        thermal_architecture = default_architecture_for_design(surface_area_m2=surface_area_m2)
    thermal_ok = survives_mission(
        thermal_architecture,
        design.avionics_power_w,
        scenario.latitude_deg,
    )

    soil = (
        soil_override if soil_override is not None else get_soil_parameters(scenario.soil_simulant)
    )

    wheel = WheelGeometry(
        radius_m=design.wheel_radius_m,
        width_m=design.wheel_width_m,
        grouser_height_m=design.grouser_height_m,
        grouser_count=design.grouser_count,
    )
    slope_capability = max_climbable_slope_deg(
        wheel,
        soil,
        total_mass_kg=total_mass_kg,
        n_wheels=design.n_wheels,
        gravity_m_per_s2=active_g,
    )

    log = run_traverse(
        design,
        scenario,
        soil,
        total_mass_kg=total_mass_kg,
        gravity_m_per_s2=active_g,
        correction=correction,
        force_backend=force_backend,
    )

    range_km = float(log.position_m[-1]) / 1000.0
    energy_margin_pct = _energy_margin_pct(log, min_soc=0.15)
    energy_margin_raw_pct = _energy_margin_raw_pct(log)
    peak_torque_nm = float(np.max(np.abs(log.wheel_torque_nm))) if log.wheel_torque_nm.size else 0.0
    sinkage_max_m = float(np.max(log.sinkage_m)) if log.sinkage_m.size else 0.0

    torque_ceiling = _sizing_peak_torque_nm(
        total_mass_kg, design.wheel_radius_m, design.n_wheels, mass_params
    )
    _ = active_g  # documents that gravity flows through mass_params above
    motor_torque_ok = bool(peak_torque_nm <= torque_ceiling) and not log.rover_stalled

    # Guard against NaN/inf creeping out of any sub-model; cap to safe
    # defaults so downstream pydantic validation always succeeds.
    if not math.isfinite(range_km):
        range_km = 0.0
    if not math.isfinite(energy_margin_pct):
        energy_margin_pct = 0.0
    if not math.isfinite(energy_margin_raw_pct):
        energy_margin_raw_pct = 0.0
    if not math.isfinite(peak_torque_nm):
        peak_torque_nm = 0.0
    if not math.isfinite(sinkage_max_m):
        sinkage_max_m = 0.0

    metrics = MissionMetrics(
        range_km=range_km,
        energy_margin_pct=energy_margin_pct,
        slope_capability_deg=slope_capability,
        energy_margin_raw_pct=energy_margin_raw_pct,
        total_mass_kg=total_mass_kg,
        peak_motor_torque_nm=peak_torque_nm,
        sinkage_max_m=sinkage_max_m,
        thermal_survival=thermal_ok,
        motor_torque_ok=motor_torque_ok,
    )
    return DetailedEvaluation(metrics=metrics, log=log, mass=breakdown)


def evaluate(
    design: DesignVector,
    scenario: MissionScenario,
    *,
    mass_params: MassModelParams | None = None,
    thermal_architecture: ThermalArchitecture | None = None,
    gravity_m_per_s2: float | None = None,
    soil_override: SoilParameters | None = None,
    use_scm_correction: bool = False,
    correction: WheelLevelCorrection | None = None,
    force_backend: str = "bw",
) -> MissionMetrics:
    """Run the full mission evaluator on one design in one scenario.

    Thin wrapper around :func:`evaluate_verbose` that discards the
    :class:`TraverseLog` and :class:`MassBreakdown`. This is the
    canonical public entry point; callers that need the supporting
    artefacts (e.g. the Phase-2 dataset builder) should call
    ``evaluate_verbose`` directly.
    """
    return evaluate_verbose(
        design,
        scenario,
        mass_params=mass_params,
        thermal_architecture=thermal_architecture,
        gravity_m_per_s2=gravity_m_per_s2,
        soil_override=soil_override,
        use_scm_correction=use_scm_correction,
        correction=correction,
        force_backend=force_backend,
    ).metrics


def range_at_utilisation(
    metrics: MissionMetrics,
    design: DesignVector,
    operational_duty_cycle: float,
) -> float:
    """Rescale capability-envelope range to an operational duty cycle.

    ``MissionMetrics.range_km`` is the capability envelope at the design
    vector's ``drive_duty_cycle``. Real missions typically command a
    lower duty than the hardware can sustain (Pragyan ~0.02, Yutu-2
    ~0.015, Sojourner ~0.01). This helper answers the ops-layer query
    "how far does this rover go if operators only command it to drive
    fraction ``u`` of the time?" via linear rescaling of forward progress.

    The rescaling is exact when the rover's power balance at the lower
    duty remains in the same regime (battery never floors, thermal never
    flips). Outside that regime (much lower duty moves the rover into a
    steady-state battery-positive mode it was not in at design duty) the
    rescaling is a strict upper bound, because less driving means more
    solar charging time and a richer energy balance -- i.e. range can
    only go *up* relative to the linear projection, never down.

    Parameters
    ----------
    metrics
        Output of :func:`evaluate`.
    design
        The design vector that produced ``metrics``.
    operational_duty_cycle
        Desired operational utilisation ``u``. Must be in
        ``[0, design.drive_duty_cycle]``; passing a value above the
        designed duty is a coding error (the hardware was not sized for
        it) and raises ``ValueError``.

    Returns
    -------
    float
        Predicted range in km at the operational duty cycle.
    """
    if operational_duty_cycle < 0.0:
        raise ValueError(f"operational_duty_cycle must be >= 0 (got {operational_duty_cycle})")
    if operational_duty_cycle > design.drive_duty_cycle + 1e-9:
        raise ValueError(
            f"operational_duty_cycle {operational_duty_cycle} exceeds "
            f"designed duty {design.drive_duty_cycle}; the hardware "
            "was not sized to sustain that utilisation."
        )
    if design.drive_duty_cycle <= 1e-9:
        return 0.0
    return metrics.range_km * (operational_duty_cycle / design.drive_duty_cycle)
