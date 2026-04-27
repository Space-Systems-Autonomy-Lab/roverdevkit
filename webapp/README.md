# roverdevkit web app — interactive tradespace tool

Phase-3 deliverable from `project_plan.md` §6 / Phase 3.

```
webapp/
├── backend/        FastAPI app, in-process import of roverdevkit.*
└── frontend/       React 19 + Vite + TS + shadcn/ui (Week-10 step-2+)
```

## Backend (Week-10 step-1, MVP)

The backend is a thin FastAPI layer over the corrected mission evaluator
and the W8 step-4 quantile-XGBoost surrogate. It does **no** physics or
ML of its own — every endpoint composes the same Python objects the
notebooks and CLI scripts use, so behaviour cannot drift from the
methodology paper's reported numbers.

Routes shipped in step-1:

| Method | Path                       | Purpose                                                |
|--------|----------------------------|--------------------------------------------------------|
| GET    | `/healthz`                 | Liveness + artifact-presence probe.                    |
| GET    | `/version`                 | Dataset / surrogate / git versions for the about box.  |
| GET    | `/scenarios`               | List the 4 canonical tradespace scenarios.             |
| GET    | `/scenarios/{name}`        | Full `MissionScenario` plus nominal soil parameters.   |
| GET    | `/registry`                | Real-rover registry summary (Pragyan, Yutu-2, etc.).   |
| GET    | `/registry/{name}`         | Single rover entry with its design vector + scenario.  |
| POST   | `/predict`                 | Median + 90 % PI for a `(design, scenario)` pair.      |

Sweeps (`/sweep`), feasibility (`/feasibility`), and NSGA-II
(`/optimize`) land in subsequent steps.

## Run locally

From the repo root, with the `roverdevkit` conda env activated:

```bash
pip install -e ".[webapp]"
uvicorn webapp.backend.main:app --reload --port 8000
```

OpenAPI docs at <http://localhost:8000/docs>.

## Tests

```bash
pytest webapp/backend/tests -q
```

The test suite uses FastAPI's `TestClient` (httpx-backed) and shares
artifact loaders with the running app, so a green test run also
validates the joblib bundle on disk is loadable.
