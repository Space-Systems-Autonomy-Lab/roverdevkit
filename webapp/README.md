# roverdevkit web app — interactive tradespace tool

Phase-3 deliverable from `project_plan.md` §6 / Phase 3.

```
webapp/
├── backend/        FastAPI app, in-process import of roverdevkit.*
└── frontend/       React 19 + Vite + TS + shadcn/ui single-page app
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

## Frontend (Week-10 step-2, single-design panel)

The frontend is a Vite + React 19 + TypeScript single-page app. It
talks to the backend through a typed fetch client (`src/lib/api.ts`)
whose response types mirror the FastAPI Pydantic schemas
1:1; the dev server proxies `/healthz`, `/scenarios`, `/registry`,
`/version`, and `/predict` to `http://localhost:8000` so calls stay
relative regardless of deployment shape.

What ships in step-2:

- **Design Explorer** page — the only route in the MVP.
  - `ScenarioPicker` — drop-down sourced from `GET /scenarios`.
  - `DesignForm` — twelve numeric inputs covering the
    `DesignVector` schema, with bounds, units, and short
    descriptions pulled from `src/types/api.ts::DESIGN_BOUNDS`.
  - `PredictionPanel` — Plotly chart (median ♦ + 90 % PI line per
    target) plus a numeric q05 / q50 / q95 table. Driven by
    `POST /predict` via TanStack Query mutation.
- App shell — header with surrogate-status badge fed by `/healthz`
  and a footer with the project tagline.
- TanStack Query for server state, Zustand for the local
  design/scenario draft.

Stack:

| Concern             | Library                                       |
|---------------------|-----------------------------------------------|
| Build / dev server  | Vite 8                                        |
| UI framework        | React 19 + TypeScript 6                       |
| Styling             | Tailwind CSS v4 (CSS-first config)            |
| Component primitives| shadcn/ui (Button, Card, Input, Label, Select) |
| Server state        | @tanstack/react-query 5                       |
| Local UI state      | Zustand 5                                     |
| Charts              | plotly.js-dist-min + react-plotly.js          |
| Lint / format       | ESLint 10 + Prettier 3                        |

```bash
cd webapp/frontend
npm install
npm run dev      # hot-reload against localhost:8000 via Vite proxy
npm run build    # type-check + production bundle to dist/
npm run lint     # ESLint
```

Running both servers together (one terminal each):

```bash
# terminal 1
uvicorn webapp.backend.main:app --reload --port 8000

# terminal 2
cd webapp/frontend && npm run dev
# open http://localhost:5173
```

## Tests

```bash
pytest webapp/backend/tests -q
```

The test suite uses FastAPI's `TestClient` (httpx-backed) and shares
artifact loaders with the running app, so a green test run also
validates the joblib bundle on disk is loadable. A frontend test
harness (Vitest + React Testing Library) lands alongside the sweep
view in step-4 once the rendering surface is more than one page.
