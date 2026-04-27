"""Pydantic v2 schemas exposed at the HTTP boundary.

Design goal: be a *thin* mirror of :mod:`roverdevkit.schema` so the
frontend can talk to the backend in terms of the same `DesignVector` /
`MissionScenario` objects the Python core uses. Where it makes sense,
we re-export the core models verbatim (frozen + extra-forbid is fine
over JSON); where the API value-add is non-trivial — `PredictRequest`,
`PredictResponse`, `RegistryEntrySummary`, etc. — we define a dedicated
boundary type so a future schema bump on the core does not silently
break the OpenAPI surface.

All response models have ``model_config = ConfigDict(frozen=True)`` so
they are safe to share across requests and so callers cannot mutate
cached registry / scenario payloads.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from roverdevkit.schema import DesignVector, MissionScenario

# Re-export the core types unchanged. Pydantic v2 serialises both
# transparently to JSON; importing here keeps the OpenAPI schema names
# consistent with the Python core.
__all__ = [
    "DesignVector",
    "EvaluateMetric",
    "EvaluateRequest",
    "EvaluateResponse",
    "FeatureRow",
    "HealthResponse",
    "MissionScenario",
    "MotorTorqueDiagnosticOut",
    "PredictRequest",
    "PredictResponse",
    "PredictTarget",
    "RegistryEntrySummary",
    "RegistryListResponse",
    "ScenarioListResponse",
    "ScenarioWithSoil",
    "SoilParametersOut",
    "ThermalDiagnosticOut",
    "VersionResponse",
]


# ---------------------------------------------------------------------------
# Health / version
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Liveness + artifact-presence probe."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "degraded"] = "ok"
    surrogate_loaded: bool
    surrogate_targets: list[str]
    quantile_bundles_path: str


class VersionResponse(BaseModel):
    """Static version metadata for the about box."""

    model_config = ConfigDict(frozen=True)

    api_version: str
    package_version: str
    dataset_version: str
    quantile_bundles_path: str


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


class SoilParametersOut(BaseModel):
    """Bekker-Wong soil parameter snapshot, JSON-friendly.

    Mirrors :class:`roverdevkit.terramechanics.bekker_wong.SoilParameters`
    but as a plain Pydantic model so it serialises cleanly without
    dataclass field-ordering quirks.
    """

    model_config = ConfigDict(frozen=True)

    simulant: str
    n: float
    k_c: float
    k_phi: float
    cohesion_kpa: float
    friction_angle_deg: float
    shear_modulus_k_m: float


class ScenarioWithSoil(BaseModel):
    """Canonical mission scenario plus the nominal soil parameters.

    The soil block is included so the frontend can show the user what
    Bekker-Wong parameters were used as the surrogate's nominal soil
    values without an extra round-trip.
    """

    model_config = ConfigDict(frozen=True)

    scenario: MissionScenario
    soil: SoilParametersOut


class ScenarioListResponse(BaseModel):
    """List of canonical tradespace scenarios with brief metadata."""

    model_config = ConfigDict(frozen=True)

    scenarios: list[ScenarioWithSoil]


# ---------------------------------------------------------------------------
# Registry (real-rover validation set)
# ---------------------------------------------------------------------------


class RegistryEntrySummary(BaseModel):
    """A real-rover registry entry exposed to the frontend.

    Mirrors :class:`roverdevkit.validation.rover_registry.RoverRegistryEntry`
    excluding its non-JSON-friendly internals (the ``ThermalArchitecture``
    object). The thermal architecture is collapsed to a small dict so
    the frontend can show the user how the rover differs from the
    tradespace defaults without depending on the dataclass shape.
    """

    model_config = ConfigDict(frozen=True)

    rover_name: str
    is_flown: bool
    design: DesignVector
    scenario: MissionScenario
    gravity_m_per_s2: float
    thermal_architecture: dict[str, Any]
    panel_efficiency: float
    panel_dust_factor: float
    imputation_notes: str


class RegistryListResponse(BaseModel):
    """All real-rover registry entries (flown and design-target tiers)."""

    model_config = ConfigDict(frozen=True)

    rovers: list[RegistryEntrySummary]


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


PrimaryTarget = Literal[
    "range_km",
    "energy_margin_raw_pct",
    "slope_capability_deg",
    "total_mass_kg",
]


class FeatureRow(BaseModel):
    """The 25-D feature vector actually fed to the surrogate.

    Echoed back so the frontend can show the nominal soil / categorical
    values that were used; useful for "did I really pick the soil I
    thought I picked?" sanity checks and as the basis for OOD warnings
    in later steps.
    """

    model_config = ConfigDict(frozen=True)

    columns: list[str]
    values: list[Any]


class PredictRequest(BaseModel):
    """Input payload for :http:post:`/predict`.

    The user always submits a full :class:`DesignVector` (the schema's
    own bounds validation will reject anything outside the design
    space) plus a canonical scenario name. The scenario's nominal soil
    parameters are looked up server-side from the soil catalogue.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    design: DesignVector
    scenario_name: str = Field(
        description="Canonical scenario key (one of the four returned by /scenarios)."
    )
    repair_crossings: bool = Field(
        default=True,
        description=(
            "Row-wise sort the (q05, q50, q95) triple before returning. "
            "Cheap, never worsens empirical coverage, and avoids "
            "non-monotone reports to the frontend. Set False to inspect "
            "raw model output."
        ),
    )


class PredictTarget(BaseModel):
    """Per-target prediction triple."""

    model_config = ConfigDict(frozen=True)

    target: PrimaryTarget
    q05: float
    q50: float
    q95: float


class PredictResponse(BaseModel):
    """Median + 90 % PI for each primary regression target.

    See ``reports/week8_intervals_v4/SUMMARY.md`` for empirical coverage
    on the test split (target ≈ 90 %, achieved 86–92 % per scenario).
    """

    model_config = ConfigDict(frozen=True)

    scenario_name: str
    quantiles: tuple[float, float, float] = (0.05, 0.50, 0.95)
    predictions: list[PredictTarget]
    feature_row: FeatureRow


# ---------------------------------------------------------------------------
# Evaluate (deterministic mission evaluator with SCM correction)
# ---------------------------------------------------------------------------


class EvaluateRequest(BaseModel):
    """Input payload for :http:post:`/evaluate`.

    Drives the corrected mission evaluator
    (:func:`roverdevkit.mission.evaluator.evaluate`, BW + wheel-level
    SCM correction) on a single ``DesignVector`` and a canonical
    scenario. Used by the single-design panel as the source of truth
    for the median value of each performance metric; the surrogate's
    quantile heads supply the prediction-interval band around it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    design: DesignVector
    scenario_name: str = Field(
        description="Canonical scenario key (one of the four returned by /scenarios)."
    )


class EvaluateMetric(BaseModel):
    """Per-target deterministic value from the corrected evaluator."""

    model_config = ConfigDict(frozen=True)

    target: PrimaryTarget
    value: float


class ThermalDiagnosticOut(BaseModel):
    """Per-design output of the lumped-parameter thermal model.

    The single-design panel surfaces both temperatures so users can
    see *why* a thermal-survival flag fired (it's almost always the
    cold case for micro-rovers without RHUs). ``rhu_power_w`` is
    included because it's the most common knob users would reach for
    if they were sizing a real rover; in our design vector it is
    fixed at 0 W by convention -- thermal is a diagnostic, not a
    design lever, since W6.
    """

    model_config = ConfigDict(frozen=True)

    survives: bool
    """End-to-end pass / fail (= ``hot_case_ok and cold_case_ok``)."""

    peak_sun_temp_c: float
    lunar_night_temp_c: float
    min_operating_temp_c: float
    max_operating_temp_c: float
    rhu_power_w: float
    hibernation_power_w: float
    surface_area_m2: float
    hot_case_ok: bool
    cold_case_ok: bool


class MotorTorqueDiagnosticOut(BaseModel):
    """Peak motor torque vs the sizing-time per-wheel ceiling.

    See :class:`webapp.backend.services.evaluate.MotorTorqueDiagnostic`
    for the closed form.
    """

    model_config = ConfigDict(frozen=True)

    survives: bool
    """End-to-end pass / fail (matches ``MissionMetrics.motor_torque_ok``)."""

    peak_torque_nm: float
    ceiling_nm: float
    rover_stalled: bool
    torque_ok: bool


class EvaluateResponse(BaseModel):
    """Deterministic evaluator output for the four primary regression targets.

    Values match :class:`roverdevkit.schema.MissionMetrics` 1:1 for the
    primary subset; the response also surfaces structured constraint
    diagnostics (``thermal``, ``motor_torque``) so the frontend can
    explain *why* a flag fired without a second round-trip.
    """

    model_config = ConfigDict(frozen=True)

    scenario_name: str
    metrics: list[EvaluateMetric]
    thermal: ThermalDiagnosticOut
    motor_torque: MotorTorqueDiagnosticOut
    used_scm_correction: bool
    elapsed_ms: float
