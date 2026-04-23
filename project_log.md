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
