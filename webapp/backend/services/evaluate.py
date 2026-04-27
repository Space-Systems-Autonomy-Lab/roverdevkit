"""Corrected mission evaluator dispatch for ``POST /evaluate``.

This service is a thin wrapper around
:func:`roverdevkit.mission.evaluator.evaluate_verbose`. The single-design
panel calls it for the deterministic median of each performance metric so
the chart's diamond marker is the ground-truth physics output rather
than the surrogate's regression of it; the surrogate's quantile heads
still supply the prediction interval around that median.

We use ``evaluate_verbose`` (rather than the lighter ``evaluate``) so we
can surface the *why* behind the constraint flags: the peak / cold
enclosure temperatures from the lumped-parameter thermal model and the
peak motor torque vs the sizing ceiling. The cost of the verbose path
is identical -- the underlying physics call is the same -- and the
extra fields are dropped on the floor for callers that only want
``MissionMetrics``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from roverdevkit.mass.parametric_mers import MassModelParams
from roverdevkit.mission.evaluator import evaluate_verbose
from roverdevkit.power.thermal import ThermalResult
from roverdevkit.schema import DesignVector, MissionMetrics, MissionScenario
from roverdevkit.surrogate.features import PRIMARY_REGRESSION_TARGETS
from roverdevkit.terramechanics.correction_model import WheelLevelCorrection


@dataclass(frozen=True)
class MotorTorqueDiagnostic:
    """Peak motor torque vs the sizing-time ceiling.

    The sizing-time ceiling is a closed-form per-wheel torque bound
    derived inside the mass model
    (:func:`roverdevkit.mass.parametric_mers._motors_mass`); we
    re-derive it here from the same closed form so the API stays
    independent of that private helper, and so a future schema bump in
    the mass model surfaces here as a deliberate test failure.
    """

    peak_torque_nm: float
    """Largest absolute per-wheel torque observed during the traverse."""

    ceiling_nm: float
    """Sizing-time per-wheel torque ceiling.

    ``ceiling = sf * mu * (m * g / N) * R`` with ``sf`` the motor
    sizing safety factor, ``mu`` the peak friction coefficient,
    ``m * g / N`` the per-wheel weight, and ``R`` the wheel radius.
    """

    rover_stalled: bool
    """Did the traverse loop hit a stall (zero forward progress)?"""

    @property
    def torque_ok(self) -> bool:
        """Did peak torque stay under the sizing ceiling?"""
        return self.peak_torque_nm <= self.ceiling_nm

    @property
    def survives(self) -> bool:
        """End-to-end mobility flag (matches ``MissionMetrics.motor_torque_ok``)."""
        return self.torque_ok and not self.rover_stalled


@dataclass(frozen=True)
class EvaluatorOutput:
    """Container the evaluate route translates into the HTTP response.

    Splitting this off from the Pydantic ``EvaluateResponse`` keeps the
    service layer dependency-free (it only knows core types) and makes
    the route a one-liner.
    """

    metrics: MissionMetrics
    thermal: ThermalResult
    motor_torque: MotorTorqueDiagnostic
    elapsed_ms: float
    used_scm_correction: bool


def _sizing_peak_torque_nm(
    total_mass_kg: float,
    wheel_radius_m: float,
    n_wheels: int,
    params: MassModelParams,
) -> float:
    """Re-derive the per-wheel torque ceiling the mass model sizes against.

    Mirrors :func:`roverdevkit.mission.evaluator._sizing_peak_torque_nm`
    -- duplicated here so the service module does not depend on a
    private helper in the core. Both formulas read ``sf * mu *
    (m*g/N) * R``; if the mass model ever rewrites the ceiling, both
    copies must change together (a unit test in
    ``roverdevkit/tests/`` already pins the mass-model copy).
    """
    weight_per_wheel_n = total_mass_kg * params.gravity_moon_m_per_s2 / n_wheels
    return (
        params.motor_sizing_safety_factor
        * params.motor_peak_friction_coef
        * weight_per_wheel_n
        * wheel_radius_m
    )


def evaluate_design(
    design: DesignVector,
    scenario: MissionScenario,
    *,
    correction: WheelLevelCorrection | None,
) -> EvaluatorOutput:
    """Run the corrected mission evaluator on one design × one scenario.

    Parameters
    ----------
    design
        Validated 12-D design vector (Pydantic has already enforced the
        bounds at the HTTP boundary).
    scenario
        One of the canonical scenarios resolved server-side.
    correction
        The shared wheel-level SCM correction artifact (loaded once per
        process by :func:`webapp.backend.loaders.get_correction`). Pass
        ``None`` to fall back to the BW-only evaluator; the route
        decides whether that fallback is acceptable.

    Returns
    -------
    EvaluatorOutput
        The full :class:`MissionMetrics` plus a wall-clock measurement,
        a flag indicating whether the SCM correction was actually
        applied, the :class:`ThermalResult` (peak / cold temperatures
        and limits), and a :class:`MotorTorqueDiagnostic` with peak
        torque, the sizing ceiling, and the rover-stalled flag.
    """
    t0 = time.perf_counter()
    detailed = evaluate_verbose(
        design,
        scenario,
        use_scm_correction=correction is not None,
        correction=correction,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    motor_torque = MotorTorqueDiagnostic(
        peak_torque_nm=float(detailed.metrics.peak_motor_torque_nm),
        ceiling_nm=_sizing_peak_torque_nm(
            detailed.metrics.total_mass_kg,
            design.wheel_radius_m,
            design.n_wheels,
            MassModelParams(),
        ),
        rover_stalled=bool(detailed.log.rover_stalled),
    )

    return EvaluatorOutput(
        metrics=detailed.metrics,
        thermal=detailed.thermal,
        motor_torque=motor_torque,
        elapsed_ms=elapsed_ms,
        used_scm_correction=correction is not None,
    )


def metrics_as_primary_dict(metrics: MissionMetrics) -> dict[str, float]:
    """Project ``MissionMetrics`` onto the four primary regression targets.

    The primary subset is what the surrogate predicts and what the
    chart renders, so the projection lives next to the dispatch to
    keep the column ordering aligned with
    :data:`roverdevkit.surrogate.features.PRIMARY_REGRESSION_TARGETS`.
    """
    src = {
        "range_km": metrics.range_km,
        "energy_margin_raw_pct": metrics.energy_margin_raw_pct,
        "slope_capability_deg": metrics.slope_capability_deg,
        "total_mass_kg": metrics.total_mass_kg,
    }
    return {target: float(src[target]) for target in PRIMARY_REGRESSION_TARGETS}
