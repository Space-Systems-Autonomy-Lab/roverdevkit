"""``POST /predict`` — surrogate point prediction with 90 % PI."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from roverdevkit.surrogate.features import PRIMARY_REGRESSION_TARGETS
from webapp.backend.loaders import (
    get_canonical_scenarios,
    get_quantile_bundles,
    get_soil_for_simulant,
)
from webapp.backend.schemas import (
    FeatureRow,
    PredictRequest,
    PredictResponse,
    PredictTarget,
)
from webapp.backend.services.predict import build_feature_row, predict_quantiles

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Return median + 90 % prediction intervals for the four primary targets.

    Pipeline
    --------
    1. Resolve the scenario from the canonical four (404 if unknown).
    2. Look up nominal Bekker-Wong soil parameters for the scenario's
       simulant.
    3. Assemble the 25-D feature row in the surrogate's training-time
       column order.
    4. Dispatch to every primary target's ``QuantileHeads`` head and
       collect ``(q05, q50, q95)`` triples.

    The surrogate is the W8 step-4 ``quantile_bundles.joblib``;
    ``q50`` is within R² 0.005 of the W8 step-3 tuned median (see
    ``reports/week8_intervals_v4/SUMMARY.md`` for the median sanity
    guardrail), so this single artifact powers both point estimates
    and PI envelopes.
    """
    scenarios = get_canonical_scenarios()
    if req.scenario_name not in scenarios:
        raise HTTPException(
            status_code=404,
            detail=(
                f"unknown scenario {req.scenario_name!r}. Pick one of {sorted(scenarios.keys())}."
            ),
        )
    scenario = scenarios[req.scenario_name]
    soil = get_soil_for_simulant(scenario.soil_simulant)

    try:
        bundles = get_quantile_bundles()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=("surrogate artifact not loaded; run scripts/calibrate_intervals.py first."),
        ) from exc

    X = build_feature_row(req.design, scenario, soil)
    preds = predict_quantiles(bundles, X, repair_crossings=req.repair_crossings)

    targets: list[PredictTarget] = []
    for target in PRIMARY_REGRESSION_TARGETS:
        cell = preds[target]
        targets.append(
            PredictTarget(
                target=target,  # type: ignore[arg-type]
                q05=cell["q05"],
                q50=cell["q50"],
                q95=cell["q95"],
            )
        )

    feature_row = FeatureRow(
        columns=list(X.columns),
        values=[v.item() if hasattr(v, "item") else v for v in X.iloc[0].tolist()],
    )

    return PredictResponse(
        scenario_name=req.scenario_name,
        predictions=targets,
        feature_row=feature_row,
    )
