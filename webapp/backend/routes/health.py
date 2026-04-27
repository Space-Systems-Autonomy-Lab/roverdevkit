"""``/healthz`` and ``/version`` endpoints."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from fastapi import APIRouter

from roverdevkit.surrogate.features import PRIMARY_REGRESSION_TARGETS
from webapp.backend.config import get_settings
from webapp.backend.loaders import get_quantile_bundles
from webapp.backend.schemas import HealthResponse, VersionResponse

router = APIRouter(tags=["meta"])


_API_VERSION = "0.1.0"


def _package_version() -> str:
    """Best-effort lookup of the installed ``roverdevkit`` package version."""
    try:
        return version("roverdevkit")
    except PackageNotFoundError:
        return "0.0.0+unknown"


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Liveness + artifact-presence probe.

    Returns ``status="degraded"`` when the surrogate artifact is
    missing so the frontend can show a "running without surrogate"
    banner instead of crashing on the first ``/predict`` call.
    """
    settings = get_settings()
    targets: list[str] = []
    surrogate_loaded = False
    try:
        bundles: dict[str, Any] = get_quantile_bundles()
        targets = sorted(bundles.keys())
        surrogate_loaded = all(t in bundles for t in PRIMARY_REGRESSION_TARGETS)
    except FileNotFoundError:
        surrogate_loaded = False

    return HealthResponse(
        status="ok" if surrogate_loaded else "degraded",
        surrogate_loaded=surrogate_loaded,
        surrogate_targets=targets,
        quantile_bundles_path=str(settings.quantile_bundles_path),
    )


@router.get("/version", response_model=VersionResponse)
def about() -> VersionResponse:
    """Static version metadata for the about box."""
    settings = get_settings()
    return VersionResponse(
        api_version=_API_VERSION,
        package_version=_package_version(),
        dataset_version=settings.dataset_version,
        quantile_bundles_path=str(settings.quantile_bundles_path),
    )
