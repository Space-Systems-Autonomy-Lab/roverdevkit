# Project log

Chronological record of non-trivial project decisions. The plan asks for
one in §6 W2 ("document the decision in the project log") for the
PyChrono go/no-go; this file is the permanent home for that and every
similar decision from here on.

Format: dated entry, with the decision, the context, and what changed as
a result. Keep entries terse.

---

## 2026-04-23 — Environment bootstrapped, PyChrono working (early win)

**Decision.** Use conda (miniforge) with Python **3.12** as the primary
environment, with PyChrono **active**. Path 2 of the three-path data
generation strategy is on the table from day one.

**Context.** The plan (§12) specifies conda for PyChrono. Installed
miniforge via `brew install --cask miniforge` on macOS arm64. Initial
`mamba env create -f environment.yml` with `python=3.11` failed because
the conda-forge PyChrono builds skip 3.11 — they target 3.10 and 3.12+.
Bumped to 3.12 and the solve succeeded; PyChrono 10.0.0 installs and
imports cleanly on Apple Silicon.

**Consequences.**

- `environment.yml` pinned to `python=3.12`.
- `pyproject.toml` kept at `requires-python = ">=3.11"` — the analytical
  path (Path 1) is fine on 3.11+; only the conda env forces 3.12 to
  unlock PyChrono.
- The week-2 PyChrono go/no-go gate (§6 W2) is provisionally **go**.
  Formal gate stays at end of week 2 so we confirm an actual SCM
  single-wheel run works end-to-end before committing.

## 2026-04-23 — Dropped pyDOE2, using scipy's LHS

**Decision.** Replace `pyDOE2` with `scipy.stats.qmc.LatinHypercube` for
the Week 6 LHS dataset generation.

**Context.** `pyDOE2` 1.3.0 is broken on Python 3.12 — it imports the
`imp` module, which was removed from the stdlib in 3.12. The package is
effectively unmaintained. `scipy.stats.qmc.LatinHypercube` has been in
scipy since 1.7 and is the maintained reference implementation; we
already depend on scipy.

**Consequences.**

- Removed `pyDOE2` from `pyproject.toml` and `environment.yml`.
- `project_plan.md` §12 still lists `pyDOE2` — flag to update on the
  next plan revision (low priority — not a blocker).
- No code changes elsewhere yet; the surrogate training code that will
  use LHS is not written until Week 6.

## 2026-04-23 — Verified install and test suite

Ran `pytest -q` under the fresh env: **8 passed, 4 xfailed**. The four
xfails are intentional — they mark tests against modules implemented in
Weeks 1–4 (Bekker-Wong, solar, evaluator). Schema validation tests all
pass, confirming the pydantic `DesignVector` / `MissionScenario` /
`MissionMetrics` enforce every bound in §3.1.

## 2026-04-23 — Bekker-Wong + Janosi-Hanamoto analytical model (Week 1)

**Decision.** `roverdevkit.terramechanics.bekker_wong.single_wheel_forces`
is implemented. Rigid wheel, θ₂ = 0, piecewise radial stress with
θ_m = (0.4 + 0.2·|s|)·θ₁ (Wong 2008, ch. 4). Entry angle solved by
`scipy.optimize.brentq` on the vertical-force residual; integrals
evaluated by `np.trapezoid` on a 100-point uniform θ grid.

**Context.** Pure-NumPy / SciPy path, no autodiff, no JIT. Benchmark:
**~0.46 ms per call** on M-series Mac — under the < 1 ms budget (§4).
Pytest has 13 physics-first-principles tests + 1 xfail placeholder for a
Wong worked example (populate in Week 9 with validation digitization).

**A subtlety worth remembering.** The pure Bekker-Wong integral for
drawbar pull at s = 0 can be slightly positive for high-friction soils
(Apollo regolith, φ ≈ 46°) because the kinematic Janosi-Hanamoto shear
term stays nonzero at zero slip. At realistic lunar per-wheel loads
(~4 N for a Rashid-class rover) DP(s = 0) is correctly small-negative
and matches the plate compaction resistance in magnitude; the pathology
only shows up in overloaded edge cases where sinkage exceeds ~4 cm. This
is a known ±15–30 % weakness of the analytical model and is exactly
what the Path-2 SCM correction layer is designed to absorb. Tests
therefore assert ``|DP(0)| ≪ |DP(0.3)|`` rather than ``DP(0) < 0``.

**Consequences.**

- Terramechanics is ahead of schedule for Week 1.
- `_integrate_forces` is exposed privately (leading underscore) but
  imported by tests for force-balance self-consistency. If Week 5's
  SCM calibration or Week 9's validation script needs it, promote to
  public API at that point.
- `ruff --fix` cleaned up four pre-existing lints in the Week-0 stubs
  (`UP037` quoted self-type annotations, `F401` unused `numpy`
  imports). Baseline is now lint-clean.
- Ready to move on to Week 2: PyChrono SCM harness + go/no-go gate.

## 2026-04-23 — PyChrono SCM single-wheel driver, Week-2 go/no-go: **GO**

**Decision.** `roverdevkit.terramechanics.pychrono_scm.single_wheel_forces_scm`
is implemented and validated. Path 2 of the three-path data generation
strategy (project_plan.md §5) is **confirmed feasible**; no need to
retreat to a pure-analytical + empirical-correction plan.

**Context.** The plan (§6 W2) gates Path 2 on wall-clock ≤ 10 s per
simulated wheel-second. Measured on the dev M-series laptop across
four fidelity configurations at Apollo-regolith nominal soil, 30 N
vertical load, slip = 0.2, R = 0.1 m, b = 0.06 m:

| config | mesh δ (mm) | drive (s) | wall / sim (×real) |
| --- | --- | --- | --- |
| fast    | 20 | 0.8 | 0.06 |
| default | 15 | 1.5 | 0.08 |
| high-fi | 10 | 2.0 | 0.17 |
| ultra   |  6 | 2.0 | 0.46 |

Even the **ultra** config runs at **0.46 s per wheel-second — a 22× margin**
under the 10 s budget. DP varies by <5 % across all four configs, so the
default δ = 15 mm is good enough for calibration data.

**Cross-check against the analytical model** at the same operating
point: analytical DP = 2.9 N, SCM DP ≈ 8.9 N; analytical T = 1.19 N·m,
SCM T = 1.43 N·m; analytical sinkage = 19.7 mm, SCM sinkage = 11.5 mm.
Both models agree on sign and order of magnitude; the factor-of-three
DP gap is exactly the kind of systematic offset the Path-2 correction
layer will learn in Week 5. All six `test_pychrono_scm.py` tests pass
in ~1.5 s under `pytest -m 'chrono and slow'`.

**Engineering nits we spent time on.**

- The osx-arm64 conda-forge PyChrono build has undefined Intel-OMP
  symbols (`__kmpc_dispatch_init_4`, etc.) in `libChrono_core.dylib`
  that aren't resolved via a `NEEDED` entry. We preload
  `libiomp5.dylib` via `ctypes.CDLL(..., RTLD_GLOBAL)` at module
  import time. Self-contained fix; no env-var gymnastics.
- The wheel needs a **collision shape** — pass `create_collision=True`
  and a `ChContactMaterialSMC` to `ChBodyEasyCylinder`, plus
  `EnableCollision(True)`. Without it SCM silently can't find the
  wheel and the whole sim runs at 0 contact force.
- All three of `ChLinkMotorLinearSpeed`, `ChLinkLockPrismatic`, and
  `ChLinkMotorRotationSpeed` use the **joint frame's local Z-axis**
  as their primary axis. Doc claims that linear motors use X are
  **wrong** in at least the 10.0.0 conda-forge build; verified
  empirically. Consequence: the X-motion motor frame needs
  `QuatFromAngleY(π/2)` to redirect local Z → world X.
- Sign of the rotation motor: with the Y-axis frame built from
  `QuatFromAngleX(−π/2)`, positive motor input produces positive
  world ω_y, which is forward rolling for a wheel translating in
  +X. Wrong sign saturates DP at the Mohr-Coulomb friction limit
  (μ·W) — that's the diagnostic.

**Consequences.**

- Single-wheel SCM is ready to feed Week 5's correction-layer
  calibration dataset.
- Path-2 data budget for calibration: at ~0.5 s per operating point
  (default config, 1.5 s drive), even 10k samples is a couple of
  hours sequential / 20 min on 4 cores. Plan's 2,000-sample budget
  (§5) is comfortable.
- New file `roverdevkit/terramechanics/pychrono_scm.py`; the
  pre-existing stub `scm_wrapper.py` is retained as the planned
  home for the Week-7 batch-orchestration layer (parallel runs,
  CSV I/O, resumable work queue).
- Tests: `tests/test_pychrono_scm.py` (6 tests, all marked
  `chrono`+`slow`, skipped in the default fast loop).
- Fast-loop status: `pytest -q` → **26 passed, 4 xfailed**.
  Slow loop: `pytest -m 'chrono and slow'` → **6 passed**.
  Ruff + mypy both clean.

