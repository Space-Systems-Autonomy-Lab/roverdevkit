"""Canonical-scenario endpoints (``/scenarios``)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from webapp.backend.loaders import get_canonical_scenarios, get_soil_for_simulant
from webapp.backend.schemas import (
    ScenarioListResponse,
    ScenarioWithSoil,
    SoilParametersOut,
)

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


def _to_payload(name: str) -> ScenarioWithSoil:
    scenarios = get_canonical_scenarios()
    if name not in scenarios:
        raise HTTPException(status_code=404, detail=f"unknown scenario {name!r}")
    scen = scenarios[name]
    soil = get_soil_for_simulant(scen.soil_simulant)
    return ScenarioWithSoil(
        scenario=scen,
        soil=SoilParametersOut(
            simulant=scen.soil_simulant,
            n=soil.n,
            k_c=soil.k_c,
            k_phi=soil.k_phi,
            cohesion_kpa=soil.cohesion_kpa,
            friction_angle_deg=soil.friction_angle_deg,
            shear_modulus_k_m=soil.shear_modulus_k_m,
        ),
    )


@router.get("", response_model=ScenarioListResponse)
def list_canonical_scenarios() -> ScenarioListResponse:
    """List the four canonical tradespace scenarios with nominal soil params."""
    names = sorted(get_canonical_scenarios().keys())
    return ScenarioListResponse(scenarios=[_to_payload(n) for n in names])


@router.get("/{name}", response_model=ScenarioWithSoil)
def get_scenario(name: str) -> ScenarioWithSoil:
    """Return a single scenario plus its nominal soil parameters."""
    return _to_payload(name)
