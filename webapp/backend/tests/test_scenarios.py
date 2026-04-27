"""Smoke tests for ``/scenarios``."""

from __future__ import annotations

from fastapi.testclient import TestClient

CANONICAL = {
    "equatorial_mare_traverse",
    "polar_prospecting",
    "highland_slope_capability",
    "crater_rim_survey",
}


def test_list_scenarios_returns_canonical_four(client: TestClient) -> None:
    response = client.get("/scenarios")
    assert response.status_code == 200
    body = response.json()
    names = {entry["scenario"]["name"] for entry in body["scenarios"]}
    assert names == CANONICAL


def test_get_scenario_includes_soil_block(client: TestClient) -> None:
    response = client.get("/scenarios/equatorial_mare_traverse")
    assert response.status_code == 200
    body = response.json()
    assert body["scenario"]["name"] == "equatorial_mare_traverse"
    soil = body["soil"]
    assert soil["simulant"]
    assert soil["n"] > 0
    assert soil["k_phi"] > 0
    assert soil["friction_angle_deg"] > 0


def test_get_scenario_unknown_returns_404(client: TestClient) -> None:
    response = client.get("/scenarios/no_such_scenario")
    assert response.status_code == 404
