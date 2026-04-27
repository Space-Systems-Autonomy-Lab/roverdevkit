# roverdevkit web app — interactive tradespace tool

Phase-3 deliverable from `project_plan.md` §6 / Phase 3.

```
webapp/
├── backend/        FastAPI app, in-process import of roverdevkit.*
└── frontend/       React 19 + Vite + TS + shadcn/ui single-page app
```

## Backend

The backend is a thin FastAPI layer over the corrected mission evaluator
and the W8 step-4 quantile-XGBoost surrogate. It does **no** physics or
ML of its own — every endpoint composes the same Python objects the
notebooks and CLI scripts use, so behaviour cannot drift from the
methodology paper's reported numbers.

Routes shipped through Week 10:

| Method | Path                       | Purpose                                                          |
|--------|----------------------------|------------------------------------------------------------------|
| GET    | `/healthz`                 | Liveness + artifact-presence probe.                              |
| GET    | `/version`                 | Dataset / surrogate / git versions for the about box.            |
| GET    | `/scenarios`               | List the 4 canonical tradespace scenarios.                       |
| GET    | `/scenarios/{name}`        | Full `MissionScenario` plus nominal soil parameters.             |
| GET    | `/registry`                | Real-rover registry summary (Pragyan, Yutu-2, etc.).             |
| GET    | `/registry/{name}`         | Single rover entry with its design vector + scenario.            |
| POST   | `/predict`                 | Surrogate median + 90 % PI for a `(design, scenario)` pair.      |
| POST   | `/evaluate`                | SCM-corrected mission evaluator on one `(design, scenario)`.     |

Sweeps (`/sweep`), feasibility (`/feasibility`), and NSGA-II
(`/optimize`) land in Week 11.

## Run locally

From the repo root, with the `roverdevkit` conda env activated:

```bash
conda activate roverdevkit
which uvicorn   # sanity check — should resolve inside the conda env
pip install -e ".[webapp]"   # only needed once
uvicorn webapp.backend.main:app --reload --port 8000
```

If `which uvicorn` points outside the conda env (e.g. /usr/bin/uvicorn,
or a brew-installed Python), your shell didn't pick up the conda
activation. Falling through to the bare `uvicorn` will hit a
`ModuleNotFoundError: No module named 'xgboost'` (or similar) because
the surrogate's runtime deps live in the conda env, not the system
interpreter. Use the absolute path as a fallback:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/roverdevkit/bin/uvicorn \
  webapp.backend.main:app --reload --port 8000
```

OpenAPI docs at <http://localhost:8000/docs>.

## Frontend

The frontend is a Vite + React 19 + TypeScript single-page app. It
talks to the backend through a typed fetch client (`src/lib/api.ts`)
whose response types mirror the FastAPI Pydantic schemas 1:1; the dev
server proxies `/healthz`, `/scenarios`, `/registry`, `/version`,
`/predict`, and `/evaluate` to `http://localhost:8000` so calls stay
relative regardless of deployment shape.

What ships through Week 10:

- **Design Explorer** page — the only route in the MVP.
  - `ScenarioPicker` — drop-down sourced from `GET /scenarios`.
  - `DesignForm` — slider + editable-number inputs for 11 continuous
    fields, segmented control for `n_wheels`. Coloured tick marks on
    each slider show the corresponding values for every selected
    real-rover overlay.
  - `RegistryOverlayPicker` — toggle Pragyan, Yutu-2, MoonRanger,
    Rashid-1 on/off; selections drive both the chart overlays and the
    slider tick marks.
  - `PredictionPanel` — Plotly chart with three layers per target:
    deterministic median ♦ from the corrected evaluator, surrogate's
    calibrated 90 % PI as a royal-blue band, and one coloured circle
    per selected real rover (also from the evaluator). Numeric
    q05 / median / q95 table below the chart. Footer chips for the
    constraint flags (thermal survival, motor torque); each chip
    opens a click-for-details Radix dialog with peak / cold
    temperatures, sizing ceiling, stall flag, and tailored failure
    explanation. Driven by parallel `POST /predict` + `POST /evaluate`
    via TanStack Query mutations.
- **About-this-model** dialog — researcher-facing explanation of the
  prediction stack (corrected evaluator + wheel-level SCM correction +
  quantile XGBoost) with performance numbers.
- App shell — header with surrogate-status badge fed by `/healthz`.
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

Or via the top-level Makefile (one terminal):

```bash
make webapp-dev      # boots backend + frontend, Ctrl+C kills both
make webapp-test     # backend pytest + frontend lint + frontend build
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
