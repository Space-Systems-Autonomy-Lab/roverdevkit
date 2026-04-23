# RoverDevKit

**ML-Accelerated Co-Design of Mobility and Power Subsystems for Lunar Micro-Rovers**

RoverDevKit is an open-source, mission-level rover evaluator that chains
terramechanics, solar power, and traverse simulation models, trains an ML
surrogate over the coupled wheel–chassis–power design space, and uses it to
explore mission-relevant Pareto fronts validated against the published design
points of real lunar micro-rovers (Rashid, Pragyan, Yutu-class).

This repository is a semester research project at MARSlab, Duke University.

## Project status

Pre-alpha — project scaffolding only. See [`project_plan.md`](project_plan.md)
for the full 15-week plan and [`project_brief.md`](project_brief.md) for the
research framing.

## Core contributions (target)

1. An open-source, fully-documented mission evaluator for lunar micro-rovers
   that takes a design vector and a mission profile and returns mission-level
   performance metrics (traverse range, energy margin, slope capability, mass).
2. A multi-fidelity ML surrogate over the coupled mobility–power design space
   that enables interactive tradespace exploration at millisecond latency.
3. Validation that the optimizer rediscovers the design points of real lunar
   micro-rovers within stated tolerances when given matching mission
   constraints, plus SHAP-based interpretable design rules.

## Architecture

```
Design Vector → Mission Evaluator (physics) → Mission Metrics
                         │
                         ▼
                 ML Surrogate (fast)
                         │
                         ▼
          Tradespace: sweeps · NSGA-II · SHAP
```

See [`project_plan.md` §2](project_plan.md) for the full system diagram and
[`project_plan.md` §10](project_plan.md) for the software architecture.

## Repository layout

```
roverdevkit/
├── data/              # Published rover specs, soil params, validation data
├── roverdevkit/       # Python package
│   ├── terramechanics/   # Bekker-Wong, PyChrono SCM wrapper, correction model
│   ├── power/            # Solar, battery, thermal survival
│   ├── mass/             # Parametric mass-estimating relationships
│   ├── mission/          # Evaluator, scenarios, traverse simulator
│   ├── surrogate/        # Training, models, features, uncertainty
│   ├── tradespace/       # Sweeps, NSGA-II, SHAP, visualization
│   └── validation/       # Rediscovery test, experimental comparison, error budget
├── notebooks/         # Interactive exploration & paper-reproduction notebooks
├── pretrained/        # Packaged surrogate models
└── tests/             # pytest test suite
```

## Installation

The primary supported setup is **miniforge + conda**, because PyChrono
(required for the multi-fidelity SCM correction layer in Path 2) is only
distributed via conda.

### With PyChrono SCM — primary path (Linux / macOS, Apple Silicon supported)

```bash
# macOS
brew install --cask miniforge

# Then, from the repo root:
mamba env create -f environment.yml
conda activate roverdevkit
pip install -e ".[dev]"          # test, lint, and notebook tooling
pytest -q                        # sanity check
```

The env uses **Python 3.12** (conda-forge's PyChrono builds skip 3.11).
See [`project_log.md`](project_log.md) for the decision log.

### Analytical track only (Path 1 fallback — any platform, any Python ≥ 3.11)

If PyChrono install fails on your platform, or you want a lightweight
analysis-only install:

```bash
pip install -e ".[dev]"
```

The project is designed so that Path 1 alone is sufficient to produce a
publishable result ([`project_plan.md` §5](project_plan.md)).

## Development

```bash
pytest                 # run tests
ruff check .           # lint
ruff format .          # format
mypy roverdevkit       # type-check
```

## Citing

A paper is in preparation. Citation information will be added on submission.

## License

MIT — see [`LICENSE`](LICENSE).
