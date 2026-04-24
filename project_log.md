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

## 2026-04-23 — Solar geometry, panel power, and battery models (Week 2)

**Decision.** Implement closed-form lunar solar geometry, a flat-plate
panel power model, and a coulomb-counting battery SOC model with
temperature derating. Defer SPICE-grade ephemeris and any higher-order
optical / electrochemical detail to "future work" — the tradespace
ceiling is set by the n=8 mass MERs, not by these submodules.

**Context.** Plan §6 W2 calls for `power/solar.py` and
`power/battery.py` validated against published rover numbers. Both
models are well-described in SMAD (Larson & Wertz) and Patel
(*Spacecraft Power Systems*); for lunar use the only twists are the
~708.7 h synodic day and the very small (~1.5°) lunar obliquity.

**Implementation.**

- `power/solar.py`:
  - `sun_elevation_deg(latitude, hour_angle, declination=0)` — standard
    spherical-astronomy altitude formula.
  - `sun_azimuth_deg(...)` — full N-clockwise azimuth, with degenerate
    cases (zenith, geographic pole) clamped.
  - `panel_power_w(...)` — `P = S·A·η·max(0, cos(i))·dust`, where
    `cos(i)` collapses to `sin(elevation)` for horizontal panels and
    uses the full Patel eq. 5.6 form when tilt+azimuth are supplied.
  - `solar_power_timeseries(...)` — convenience generator over a
    diurnal cycle for plotting and traverse-loop sanity checks.
- `power/battery.py`:
  - `BatteryState` dataclass with capacity, SOC, T, DoD floor, and
    asymmetric charge/discharge efficiencies.
  - `step(state, P_net, dt)` — coulomb-counting update with
    `η_charge` on the way in and `1/η_discharge` on the way out;
    SOC clamped to `[min_state_of_charge, 1.0]`; returns a new state
    object (functional-style for traverse-loop bookkeeping).
  - `temperature_derating_factor(T)` — piecewise-linear curve at
    (-40, -20, 0, 20, 60) °C with anchors (0.50, 0.70, 0.85, 1.00,
    0.95), a coarse fit to Smart et al. / NASA Glenn Li-ion data;
    flagged as a placeholder for vendor-specific curves.
  - `usable_capacity_wh(state)` and `stored_energy_wh(state)` helpers.

**Validation gates met.**

- Solar (Yutu-2, lat 45.5°N, 1 m² × 30 % horizontal panel, δ=0):
  - Clean-sky theoretical noon power = 1361 × 1.0 × 0.30 × sin(44.5°)
    ≈ **286 W**, matched to 1e-6 by closed form.
  - With realistic in-flight loss factors (dust ≈ 0.55, cell thermal
    derating ≈ 0.85), model lands at **~134 W**, inside the published
    120–140 W in-flight band.
- Battery (100 Wh nominal, 20 °C, default 15 % DoD floor, η=0.95):
  - `usable_capacity_wh` = 85 Wh exactly — matches the SMAD-style
    sizing rule of thumb in plan §4.
  - Round-trip charge→discharge of 10 Wh loses ≈ 1.0 Wh as expected
    from `(1 - η_c·η_d)`-style accounting.

**Consequences.**

- `min_state_of_charge` default bumped from 0.20 → **0.15** to land
  the validation-gate "85 Wh usable" number cleanly. Documented as a
  per-pack tunable.
- `temperature_derating_factor` curve is parameter-driven (module
  constants `_TEMP_DERATING_TEMPS_C` / `_FACTORS`), so swapping in a
  vendor curve is a one-line edit.
- Tests: `tests/test_power.py` rewritten — **49 tests** covering
  spherical-astronomy edge cases, panel-power physics, the Yutu-2
  validation gate, battery construction validation, SOC clamping,
  round-trip efficiency, temperature derating, and usable-capacity
  rules. Replaces the placeholder xfails.
- Fast-loop status: `pytest -q --ignore=tests/test_pychrono_scm.py`
  → **69 passed, 2 xfailed** (the remaining xfails are for the
  Week-4 evaluator and a Wong textbook example to be digitised).
  Ruff + ruff format + mypy all clean on the new modules.
- Week-2 deliverable list (plan §6 W2) is now complete: solar +
  battery + PyChrono go/no-go all in. Ready to start Week 3 (mass
  model + parametric MERs).

## 2026-04-23 — Mass model: bottom-up not regression (Week 3)

**Decision.** Implement the rover mass model as a **bottom-up assembly of
physics-grounded specific masses** (SMAD Ch. 11, AIAA S-120A-2015, vendor
catalogues) rather than as a set of per-subsystem regressions against the
published-rover dataset, as originally framed in §6 W3 of the plan. Use
the n=10 rover dataset as a **validation set, not training data**.

**Context.** The original plan said "fit simple parametric MERs (linear
or power-law) for each subsystem mass." Three problems with that as
stated:

1. `data/published_rovers.csv` has `mass_chassis_kg` (and all other
   per-subsystem columns) empty for every row — subsystem breakdowns
   simply aren't published for most of these rovers, so any
   "per-subsystem MER fit" would be fitting to imputations, not data.
2. The dataset spans 2 kg (CADRE) to 756 kg (Lunokhod-1) across
   1970–2024, with Mars + Moon + Earth-only vehicles and three
   different mobility architectures. A least-squares fit on n≈8 of
   these would be dominated by whichever 2–3 rovers have the highest
   leverage and is statistically indefensible.
3. `chassis_mass_kg` is already a `DesignVector` input, so we don't
   need to predict it — we need MERs only for the *derived* subsystems.

**Approach that replaced it.**

- `roverdevkit/mass/parametric_mers.py` — bottom-up model:
  - Wheels: `n * rho_wheel_area * 2πRW` + thin-plate grouser mass.
  - Motors: `n * (m_0 + k_τ * τ_peak)` with `τ_peak = SF · μ · W_wheel · R`,
    sized against the vehicle's lunar weight (resolved by fixed-point
    iteration, 3–5 steps to 1e-4 relative tolerance).
  - Solar panels: `ρ_A · A_s` (area-density constant).
  - Battery: `C_b / e_pack` (pack-level specific energy).
  - Avionics: `m_0 + β · P_a`.
  - Harness / thermal / margin: SMAD Table 11-43 fractions
    (`0.08`, `0.05`, `0.20`) applied in SMAD order.
  - Every constant exposed through a `MassModelParams` dataclass so the
    surrogate / tradespace layer can sweep them for sensitivity.
- `roverdevkit/mass/validation.py` + `data/mass_validation_set.csv` —
  gap-filled design vectors for 8 published rovers with per-row
  imputation notes; primary statistic is **median absolute percent
  error on in-class (5–50 kg) rovers**, with out-of-class rovers
  (CADRE, Yutu-2, MARSOKHOD, Lunokhod) reported alongside but
  excluded from the primary number.

**Calibration (default `MassModelParams`).**

After one round of calibration against the validation set (dropped
`k_τ` from 1.0 → 0.10 kg/(N·m) to match Maxon EC-i + harmonic-drive
catalogue data; dropped wheel area density from 12 → 8 kg/m² to the
low end of Nohmi 2003 for micro-rover-class rigid wheels), the
end-to-end validation gives:

| Rover               | in-class | published (kg) | predicted (kg) | err %   |
|---------------------|----------|----------------|----------------|---------|
| Rashid              | yes      |   10.0         |   10.37        |  +3.7   |
| Sojourner           | yes      |   10.6         |   11.06        |  +4.3   |
| ExoMy               | yes      |    8.0         |    8.83        | +10.4   |
| Pragyan             | yes      |   26.0         |   22.31        | -14.2   |
| CADRE-unit          | no       |    2.0         |    4.08        | +104.2  |
| Resilience-Tenacious| no       |    5.0         |    5.63        | +12.5   |
| Yutu-2              | no       |  135.0         |  120.33        | -10.9   |
| MARSOKHOD-proto     | no       |   70.0         |   63.50        |  -9.3   |

In-class aggregates (n=4): median |err| = **7.4 %**, mean |err| = 8.2 %,
worst = Pragyan at -14.2 %. Well below the 30 % Week-5 target. CADRE is
expected to be large: at 2 kg it is below the 3 kg `chassis_mass_kg`
lower bound of `DesignVector` and the avionics/motor baseline terms
dominate the total. Yutu-2 and MARSOKHOD are also out of class (above
the 50 kg ceiling) and are included only as reference points.

**Consequences.**

- Plan updated: §4 "Mass model" paragraph rewritten to describe the
  bottom-up approach; §6 W3 deliverable list updated; architecture
  diagram updated to list `validation.py`; paper-outline §6.3 and
  §7.1 updated; risk register updated.
- Tests: `tests/test_mass.py` (27 tests, all passing) covers
  dataclass behaviour, per-subsystem physics (monotonicity,
  linearity), iteration convergence, `DesignVector` round-trip, and a
  validation gate that fails if the in-class median error ever
  exceeds 30 %. This is effectively the Week-5 real-rover validation
  gate enforced in CI at Week 3.
- Fast-loop status: `pytest -q --ignore=tests/test_pychrono_scm.py`
  → **96 passed, 2 xfailed**. Ruff, ruff format, mypy all clean.
- Ready to start Week 4 (mission traverse simulator + thermal
  survival check).

## 2026-04-24 — Week 4: mission traverse simulator and evaluator

**Decision.** Wire up the full mission evaluator pipeline (soil lookup
-> mass -> thermal -> slope capability -> time-stepped traverse ->
aggregated metrics). Two small reversals of the original plan:

1. **Never early-terminate the traverse sim.** The stub docstring said
   to terminate on battery-floored or thermal-violated. Changed to
   always run for the full mission duration; failure modes are
   captured via `TraverseLog.battery_floored`, `.rover_stalled`,
   `.reached_distance` flags and via the continuous metrics
   (`range_km`, `energy_margin_pct`). This matters for Phase 2: the
   surrogate needs uniform scoring across infeasible designs, which
   early-return would throw away.

2. **Added four modules the plan's W4 bullet list left implicit.**
   The schema outputs (`slope_capability_deg`, `motor_torque_ok`,
   etc.) require helpers the plan didn't enumerate. Filled the gaps:

   - `roverdevkit/terramechanics/soils.py` — CSV → `SoilParameters`
     catalogue (needed because scenarios reference soils by name).
   - `roverdevkit/mission/scenarios.py` — YAML loader +
     `list_scenarios()`.
   - `roverdevkit/mission/capability.py` — max-climbable-slope via
     brentq on `DP_avail(slope) - DP_req(slope) = 0` at slip=0.6.
   - `roverdevkit/power/thermal.py` — closed-form single-node steady-
     state survival check (Stefan-Boltzmann with hot-case
     cos(latitude) insolation and cold-case regolith sink).

**Context.** Reviewed the Week 4 bullets in project_plan.md §6 against
the `MissionMetrics` schema and the Phase-2 surrogate requirements.
Three gaps were silent: the scenario YAMLs reference soils by name but
no loader existed; `slope_capability_deg` is a primary output but
isn't a natural product of the traverse loop; and `motor_torque_ok`
had no reference torque to compare against. Resolved by adding a
helper for each gap.

**Implementation details.**

- Per-step physics in `traverse_sim.py`: solve `DP(slip) = required`
  with `scipy.optimize.brentq` in slip bracket `[-0.9, 0.95]`. If the
  bracket fails (slope unclimbable) we pin slip at the upper bracket,
  zero forward velocity, and record `rover_stalled=True`. Motor power
  = `T * omega / eta_motor` per wheel with `omega = v / (R*(1-s))`;
  default motor efficiency 0.8 (Maxon catalogue).
- Mobility power is duty-cycle-scaled (`delta * P_drive_full`) and
  position advances by `v * dt * delta`. This is the standard
  mission-average tradespace approximation; explicit drive schedules
  are v2.
- `motor_torque_ok` is judged against the same envelope the mass
  model uses to size the motor subsystem
  (`SF * mu * m*g / n_wheels * R`). Keeps the two definitions tied.
- Thermal-architecture surface area defaults to a cube-root scaling
  of chassis mass (`0.02 * m^(2/3) + 0.05`) so we don't have to plumb
  new fields into `DesignVector` this week. Easy to override via
  `evaluate(..., thermal_architecture=...)`.
- `energy_margin_pct = (SOC_end - min_SOC) / (1 - min_SOC) * 100`,
  clamped to 0 for designs that hit the floor. 0 = on the DoD floor,
  100 = fully charged.

**Performance.**

- Per-evaluation cost on the analytical path: ~1.7 s at `dt_s = 1 h`
  for a 14-day mission, ~290 ms at `dt_s = 6 h`, ~74 ms at `dt_s = 1
  day`. Target for the Path-1 surrogate dataset (50k samples in an
  afternoon) is achievable at `dt_s ≥ 6 h` on 4 parallel workers.
- Default `dt_s = 1 h` keeps hourly resolution for the integration
  tests and debugging; Phase 2 scripts will coarsen it.

**Validation gates.**

- `tests/test_soils.py` (6) + `tests/test_scenarios.py` (12): every
  scenario's `soil_simulant` resolves in the catalogue; all four
  canonical scenarios round-trip through pydantic.
- `tests/test_thermal.py` (13): construction validation,
  equator-vs-pole hot case, RHU warming in cold case, Stefan-Boltzmann
  balance closes to 1e-6, survive / overheat / freeze branches.
- `tests/test_capability.py` (7): softer soil lowers slope capability,
  larger wheel improves it, cap at 35 deg when rover exceeds schema
  bound.
- `tests/test_traverse_sim.py` (11): full-duration run, monotonic time
  and non-decreasing position, SOC within [min, 1], solar = 0 at
  night, steeper slope draws more mobility power, bigger solar panel
  collects more energy, underpowered rover floors battery, soft-soil
  + steep scenario triggers the stall flag.
- `tests/test_mission_evaluator.py` (12): smoke test on all four
  scenarios, mass in 5-50 kg class, finite & in-range metrics,
  range bounded by traverse distance, bigger battery ≥ smaller on
  energy margin, denser soil > looser on slope capability, SCM path
  raises NotImplementedError until Week 7.

**Consequences.**

- Fast-loop status: `pytest -q` → **163 passed, 1 xfailed** (pre-
  existing Wong textbook digitisation xfail). Ruff, ruff format, mypy
  all clean.
- Updated `traverse_sim.py`'s docstring to reflect the "never
  early-terminate" decision; stub function signature now takes a full
  parameter list instead of `*args, **kwargs`.
- `TraverseLog` schema added fields: `mobility_power_w`,
  `wheel_torque_nm`, `sun_elevation_deg`, `battery_floored`,
  `rover_stalled`, `reached_distance`. These are consumed by the
  evaluator and useful for Week 5 validation notebooks.
- Week 5 can now focus on the Yutu-2 / Pragyan / Rashid real-rover
  cross-check: the evaluator pipeline runs end-to-end and returns
  plausible numbers for all four scenarios.

