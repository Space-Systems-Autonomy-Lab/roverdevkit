"""Uvicorn entry point: ``uvicorn webapp.backend.main:app --reload``.

A module-level ``app`` is required by the standard ``uvicorn module:app``
discovery convention. The actual construction lives in
:func:`webapp.backend.app.create_app` so tests can build isolated
apps without going through this entry point.
"""

from __future__ import annotations

import logging

from webapp.backend.app import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)

app = create_app()
