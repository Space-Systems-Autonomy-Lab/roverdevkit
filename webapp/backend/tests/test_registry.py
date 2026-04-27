"""Smoke tests for ``/registry``."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_list_registry_includes_known_rovers(client: TestClient) -> None:
    response = client.get("/registry")
    assert response.status_code == 200
    body = response.json()
    names = {entry["rover_name"] for entry in body["rovers"]}
    assert names >= {"Pragyan", "Yutu-2", "MoonRanger", "Rashid-1"}


def test_get_registry_entry_shape(client: TestClient) -> None:
    response = client.get("/registry/Pragyan")
    assert response.status_code == 200
    body = response.json()
    assert body["rover_name"] == "Pragyan"
    assert body["is_flown"] is True
    # Design vector must round-trip through the real DesignVector schema.
    assert 0.05 <= body["design"]["wheel_radius_m"] <= 0.20
    # Thermal architecture is collapsed to a dict but must include the
    # fields the frontend expects.
    therm = body["thermal_architecture"]
    assert "rhu_power_w" in therm
    assert "surface_area_m2" in therm


def test_get_registry_unknown_returns_404(client: TestClient) -> None:
    response = client.get("/registry/Curiosity")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "Available" in detail
