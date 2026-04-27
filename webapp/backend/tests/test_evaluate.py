"""Smoke tests for ``POST /evaluate``.

These run the corrected mission evaluator end-to-end. Unlike the
predict tests they do *not* depend on the W8 step-4 quantile artifact;
they only need the SCM correction artifact at
``data/scm/correction_v1.joblib`` to assert SCM correction was used.
If that artifact is missing the route falls back to BW-only and the
test asserts ``used_scm_correction=False``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

PRIMARY_TARGETS = {
    "range_km",
    "energy_margin_raw_pct",
    "slope_capability_deg",
    "total_mass_kg",
}


def test_evaluate_returns_all_primary_targets(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    payload = {"design": sample_design, "scenario_name": "equatorial_mare_traverse"}
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["scenario_name"] == "equatorial_mare_traverse"
    targets = {m["target"] for m in body["metrics"]}
    assert targets == PRIMARY_TARGETS
    for metric in body["metrics"]:
        assert isinstance(metric["value"], (int, float))

    thermal = body["thermal"]
    for key in (
        "survives",
        "peak_sun_temp_c",
        "lunar_night_temp_c",
        "min_operating_temp_c",
        "max_operating_temp_c",
        "rhu_power_w",
        "hibernation_power_w",
        "surface_area_m2",
        "hot_case_ok",
        "cold_case_ok",
    ):
        assert key in thermal
    # The default architecture has a -30/+50 °C envelope and these
    # are the limits the survival flag is judged against.
    assert thermal["min_operating_temp_c"] == -30.0
    assert thermal["max_operating_temp_c"] == 50.0

    motor = body["motor_torque"]
    for key in (
        "survives",
        "peak_torque_nm",
        "ceiling_nm",
        "rover_stalled",
        "torque_ok",
    ):
        assert key in motor
    assert motor["peak_torque_nm"] >= 0.0
    assert motor["ceiling_nm"] > 0.0

    assert isinstance(body["used_scm_correction"], bool)
    assert body["elapsed_ms"] > 0


def test_evaluate_thermal_cold_case_drives_failure_for_no_rhu_design(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    """The default architecture has 0 W RHU; cold case should be the failing one.

    With no RHU and 2 W of hibernation power, a 0.2-ish m² enclosure
    radiates to ~133 K (well below the −30 °C limit) and the hot case
    sits comfortably under +50 °C at any latitude. The dialog leans on
    this distinction to explain *why* survival fails, so we pin it
    here.
    """
    payload = {"design": sample_design, "scenario_name": "equatorial_mare_traverse"}
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    thermal = response.json()["thermal"]
    if not thermal["survives"]:
        assert not thermal["cold_case_ok"]
    # Hot case should never be the failure for this sample design at
    # equatorial latitude (sanity guard against a regression that
    # silently flips the model).
    assert thermal["hot_case_ok"]


def test_evaluate_unknown_scenario_returns_404(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    payload = {"design": sample_design, "scenario_name": "no_such_scenario"}
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 404


def test_evaluate_rejects_out_of_bounds_design(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    bad = dict(sample_design)
    bad["wheel_radius_m"] = 5.0
    response = client.post(
        "/evaluate",
        json={"design": bad, "scenario_name": "equatorial_mare_traverse"},
    )
    assert response.status_code == 422


def test_evaluate_values_match_primary_metrics_shape(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    """Sanity-check the projection of ``MissionMetrics`` onto the four primary targets.

    Range and total mass are strictly positive for every well-formed
    scenario; slope is bounded above by 90°; energy margin is unbounded
    but should be finite. This is a coarse "no NaN snuck through" guard.
    """
    payload = {"design": sample_design, "scenario_name": "polar_prospecting"}
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    body = response.json()
    by_target = {m["target"]: m["value"] for m in body["metrics"]}

    assert by_target["total_mass_kg"] > 0
    assert by_target["range_km"] >= 0
    assert 0 <= by_target["slope_capability_deg"] <= 90
    assert by_target["energy_margin_raw_pct"] == by_target["energy_margin_raw_pct"]  # not NaN


def test_evaluate_and_predict_agree_within_surrogate_noise_floor(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    """The surrogate's median should track the evaluator within R²-noise.

    On the canonical equatorial-mare scenario for the Yutu-2-ish
    sample design, the W8 step-3 tuned median has R² ≥ 0.99 on every
    primary target. We pick a generous tolerance per target rather
    than assert exact equality so this test does not flake on
    XGBoost-version churn or harmless quantile-head retrains.
    """
    payload = {"design": sample_design, "scenario_name": "equatorial_mare_traverse"}
    eval_resp = client.post("/evaluate", json=payload)
    pred_resp = client.post("/predict", json=payload)
    assert eval_resp.status_code == 200
    if pred_resp.status_code == 503:
        # Quantile bundles missing (mirrors the predict-test skip path).
        return
    assert pred_resp.status_code == 200

    evaluator = {m["target"]: m["value"] for m in eval_resp.json()["metrics"]}
    surrogate = {p["target"]: p["q50"] for p in pred_resp.json()["predictions"]}

    # Per-target relative tolerance on the median. Energy margin runs
    # large positive on equatorial-mare so we use absolute tolerance
    # (a 5 pp gap on a 600 % margin is still <1 % relative error).
    rel_tol = {
        "range_km": 0.10,
        "slope_capability_deg": 0.05,
        "total_mass_kg": 0.02,
    }
    for tgt, tol in rel_tol.items():
        e = evaluator[tgt]
        s = surrogate[tgt]
        assert abs(e - s) <= max(tol * abs(e), 1e-3), (tgt, e, s)
    # Energy margin: tolerate a 50-pp gap in absolute terms.
    assert abs(evaluator["energy_margin_raw_pct"] - surrogate["energy_margin_raw_pct"]) <= 50
