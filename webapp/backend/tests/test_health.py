"""Smoke tests for ``/healthz`` and ``/version``."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_healthz_returns_ok_when_artifact_present(
    client: TestClient, artifacts_present: bool
) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["surrogate_loaded"] is artifacts_present
    if artifacts_present:
        assert body["status"] == "ok"
        assert set(body["surrogate_targets"]) >= {
            "range_km",
            "energy_margin_raw_pct",
            "slope_capability_deg",
            "total_mass_kg",
        }
    else:
        assert body["status"] == "degraded"


def test_version_returns_metadata(client: TestClient) -> None:
    response = client.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "api_version",
        "package_version",
        "dataset_version",
        "quantile_bundles_path",
    }
    assert body["api_version"] == "0.1.0"
    assert body["dataset_version"] == "v5"
