"""FastAPI backend for the Phase-3 tradespace exploration tool.

The package is intentionally thin: every route delegates to the
existing :mod:`roverdevkit` core (mission evaluator, surrogate,
validation registry) so the web app cannot drift from the methodology
paper's reported numbers. See ``project_plan.md`` §6 / Phase 3 and
``project_log.md`` 2026-04-26 entry for the architectural rationale.
"""

from __future__ import annotations

__all__ = ["create_app"]


def create_app():  # type: ignore[no-untyped-def]
    """Re-export of :func:`webapp.backend.app.create_app` for convenience."""
    from webapp.backend.app import create_app as _factory

    return _factory()
