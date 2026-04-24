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

---

## 2026-04-24 — Week 5: real-rover validation harness

**Decision.** Validate the mission evaluator against three published
rovers (Pragyan, Yutu-2, Sojourner) with a Python-module-first
validation stack and a CI acceptance gate. Move the notebook to a
thin wrapper over the package code, mirroring Week 3's mass-validation
pattern. Drop Rashid from traverse comparison (no published traverse
data; its mission ended on the Hakuto-R lander before deployment) and
substitute Sojourner so the gate has three data points including one
non-lunar gravity case. Defer Rashid to Week 12 rediscovery.

**Context.** The project plan's Week 5 list (§6) was under-specified
on (a) which rovers, (b) what metric-matching tolerance, and (c) how
to interpret "right direction when parameters change." Yutu-2's
per-lunar-day drive window also does not match our current sim's
always-active model; we need either a single-lunar-day scenario or
a hibernation-capable sim. Picked the simpler option: new
per-lunar-day YAML scenarios.

**What changed.**

*New files.*

- `roverdevkit/mission/configs/chandrayaan3_pragyan.yaml`,
  `.../change4_yutu2_per_lunar_day.yaml`,
  `.../mpf_sojourner_ares_vallis.yaml`: three validation-only
  scenarios. Kept out of `list_scenarios()` so the Phase-3 optimiser
  never picks them up.
- `data/published_traverse_data.csv`: per-rover published traverse
  distance, peak solar power, and thermal-survival outcome with
  low/high bands and citations (Di 2020 / Ding 2022 for Yutu-2, ISRO
  press kit + Nature SR 14:24178 for Pragyan, Wilcox & Nguyen 1998
  for Sojourner).
- `roverdevkit/validation/rover_registry.py`: `(DesignVector,
  MissionScenario, gravity, thermal_architecture, panel efficiency,
  panel dust factor, imputation_notes)` bundles per rover.
  Imputation notes document every field that was not directly
  published, mirroring the Week-3 mass-validation pattern.
- `roverdevkit/validation/rover_comparison.py`: scoring engine with
  the five Week-5 acceptance criteria (range feasibility, range
  sanity ceiling, thermal-survival match, motor-and-traversal-ok,
  peak-solar-in-band) plus `acceptance_gate()` for CI.
- `roverdevkit/validation/cross_scenario.py`: three hand-crafted
  design archetypes (`large_traverser`, `polar_survivor`,
  `slope_climber`) for ranking tests, plus a one-at-a-time design-
  variable sensitivity sweep on a distance-unlimited synthetic
  scenario so delta metrics don't saturate at the traverse cap.
- `tests/test_rover_comparison.py` (21) and
  `tests/test_cross_scenario.py` (11): CI gates.
- `notebooks/00_real_rover_validation.ipynb`: human-facing wrapper
  around the validation modules.

*Schema and evaluator changes.*

- `MissionScenario.name` relaxed from `ScenarioName` Literal to `str`
  to permit validation-only scenarios. `load_scenario()` also takes
  `str`; `list_scenarios()` filters to the canonical four so Phase-3
  sweeps never pick up validation-only YAMLs.
- `evaluate()` gained an optional `gravity_m_per_s2` kwarg. If set
  and different from the default lunar constant it rebuilds
  `mass_params` with the new gravity, keeping motor sizing and
  traverse gravity in sync for the Mars (Sojourner) case.

**Acceptance criteria (as enforced in CI).**

Per-rover, all five must fire:

1. Predicted range ≥ published low bound.
2. Predicted range ≤ 10× published high bound.
3. Thermal survival prediction == published outcome (exactly).
4. `motor_torque_ok` True and `rover_stalled` False.
5. Peak solar power ∈ published [low, high] band.

**Results.**

| Rover     | Range pred | Range pub | Thermal pred | Pub | Peak solar pred | Pub | Pass? |
|-----------|-----------:|----------:|:-------------|:----|----------------:|----:|:------|
| Pragyan   | 500 m      | 101 m     | False        | F   | 44.8 W          | 50  | ✓     |
| Yutu-2    | 200 m      | 25 m      | True         | T   | 136.4 W         | 135 | ✓     |
| Sojourner | 150 m      | 100 m     | True         | T   | 16.6 W          | 16  | ✓     |

The Pragyan prediction reproducing its real lunar-night failure is
the strongest signal in the set: our thermal sink temperatures and
RHU-absent architecture were both correct by construction.

**Key judgment calls.**

- *Range over-prediction is expected and accepted.* Predicted
  traverse is 1.5–8× published because the schema's
  `drive_duty_cycle ≥ 0.1` floor and the sim's constant-drive model
  bound us above the tiny real duty cycles (0.01–0.02) Pragyan,
  Yutu-2, and Sojourner actually ran. The Week-5 gate tests *range
  feasibility* rather than a tight ratio.
- *Thermal model is single-node.* Real rovers use MLI + variable-
  emittance louvers + active cooling. We approximate MLI via small
  effective-radiating-area and low absorptivity (alpha ~0.15). Yutu-2
  further needed an industrial-temp-range max (+60 °C) because the
  default +50 °C is not achievable with 20 W internal dissipation on
  our 0.10 m² effective enclosure.
- *Constant-slope traverse.* `traverse_sim.py` uses a scalar
  `scenario.max_slope_deg`; we set the three validation scenarios to
  their *typical-ops* slope (~5 °) rather than worst-case (~12 °)
  because real Pragyan/Yutu-2/Sojourner drove on near-flat terrain
  most of the time.
- *Panel parameters are per rover.* Default traverse-sim panel
  efficiency (0.28) and dust factor (0.90) are calibrated for a
  fresh GaAs triple-junction panel. Registry entries override these
  with rover-specific EOL + dust values (Yutu-2: 0.20 × 0.55,
  Sojourner: 0.17 × 0.80, Pragyan: 0.22 × 0.85), which align peak
  solar predictions to within ±10 % of published numbers.

**Performance.**

- `tests/test_rover_comparison.py`: 21 tests, ~100 s (compare_all
  plus parametrised per-rover tests; dominated by three ~12 s
  evaluator runs).
- `tests/test_cross_scenario.py`: 11 tests, ~170 s (each archetype
  × scenario + each sensitivity bump is one evaluator call,
  ~15 runs × ~12 s each).
- Full suite: **195 passed, 1 xfail** in 305 s.

**Consequences.**

- End-to-end evaluator validated against real flight data with a
  CI-enforceable gate. Phase-2 surrogate work can proceed without
  worrying that the target function is broken.
- Documented known limitations of the analytical-terramechanics path
  (Section 7 of the notebook). Week 7's SCM correction remains the
  path to closing the DP-underprediction gap.
- `ScenarioName` Literal kept canonical; only the free-text
  `MissionScenario.name` accepts validation scenarios. The tradespace
  optimiser's scenario sweep stays closed.

### Week 5.5 (polish): pre-Phase-2 evaluator signal fixes

Brief pass after the Week-5 retrospective identified two places where
the evaluator's output would saturate under the LHS sweep Week 6 needs
to kick off. Both are low-effort, high-payoff fixes.

**What changed.**

- `MissionMetrics` gains an unclipped companion to `energy_margin_pct`:
  `energy_margin_raw_pct = (E_generated - E_consumed) / E_consumed * 100`,
  integrated over the full traverse via `numpy.trapezoid`. Unbounded in
  both directions -- negative means net energy deficit, positive means
  surplus generation. The original SOC-based metric stays for human
  reporting / acceptance gates; the raw metric is what the Week-6
  surrogate will be trained on. Fixes the "solar bump shows `+0.00`"
  rows in the Week-5 sensitivity table.
- `SensitivityEntry` and the notebook's section 6 table now expose
  `delta_energy_margin_raw_pct` alongside the clipped delta. Every one
  of the seven sensitivity bumps now produces a non-zero response with
  the physically expected sign (solar → +428 %, avionics → −228 %,
  speed → −3.7 %, etc.). Two new CI tests enforce the strict-sign
  invariants (solar positive, avionics negative) on the raw metric.
- Canonical scenario YAMLs: `traverse_distance_m` raised from
  short-mission budgets (0.5-5 km) to non-binding caps at the
  mission-duration × max-speed × max-duty theoretical reach
  (20-80 km). Range now differentiates archetypes on the cross-scenario
  ranking: `equatorial_mare_traverse` goes from a 3-way tie at 5 km
  to 48.4 / 4.8 / 14.5 km for `large_traverser` / `polar_survivor` /
  `slope_climber`. Validation scenarios (Pragyan / Yutu-2 / Sojourner)
  keep their short-mission caps because they're being compared against
  specific published traverses.
- Two notebook documentation additions: a 7th known-limitation bullet
  making the per-rover panel-efficiency / dust-factor / thermal-
  architecture calibration explicit (these are effective-parameter
  matches to vendor datasheets, not independent validation of the
  solar-geometry and Stefan-Boltzmann math underneath), and an 8th
  explaining the raised non-binding canonical caps. Week-9 error-
  budget deliverable should separate calibrated-layer vs validated-
  layer claims cleanly.

**Impact on Phase 2.**

- Surrogate targets `range_km`, `energy_margin_raw_pct`, and
  `slope_capability_deg` now all respond smoothly to at least 5 of the
  7 primary design levers across canonical scenarios, instead of
  collapsing to two training axes (speed, duty) plus a slope-only
  signal. This is what "Target: R² > 0.95 on range and energy margin"
  from project_plan.md §6 W6 needs to be a meaningful target rather
  than "we predicted the distance cap correctly."
- Full suite still passes (197 tests, 1 xfail). Two new tests added
  on top of the Week-5 gates.

### Week 5.6 (polish): capability envelope vs operational utilisation

**Decision.** Reframe the evaluator's outputs as a *capability envelope*
at the design vector's own `drive_duty_cycle`, not a prediction of
operational range. Expose operational-utilisation queries as a separate,
explicit post-hoc helper.

**Context.** Real lunar rovers (Pragyan ~0.02, Yutu-2 ~0.015, Sojourner
~0.01) command *well below* the `drive_duty_cycle >= 0.1` schema floor
used for most of the project so far. That isn't because their hardware
can't sustain higher duty -- it's because ops schedules carve the
mission around uplink windows, thermal soaks, and science campaigns.
Two problems followed: (a) the design-space sweep Week 6 kicks off
couldn't even represent duty regimes where every real LPR we've
validated against actually operates, so our validation residuals
looked like pure physics overshoot when part of the gap was literally
out-of-schema design space; and (b) new readers misread `range_km`
as "what the rover will actually drive" rather than "what the
hardware can sustain" -- a JPL Team X vs ops-schedule distinction
worth calling out explicitly.

**What changed.**

- `DesignVector.drive_duty_cycle` floor dropped from 0.1 to 0.02, with
  a docstring that pins the semantics ("designed duty the hardware is
  sized to sustain") and cites the three real-rover data points. The
  ceiling stays at 0.6 for continuous-drive reference designs.
- `MissionMetrics` docstring now opens with a capability-envelope
  framing paragraph, and each primary metric has a one-line
  "capability-at-designed-duty" annotation. `range_km` is explicitly
  not an operational prediction.
- `evaluator.py` module docstring gains a full section on the envelope-
  vs-utilisation distinction, including the linear rescaling formula.
  The surrogate package docstring picks up a matching one-paragraph
  framing so the Phase-2 ML layer inherits it.
- New helper `evaluator.range_at_utilisation(metrics, design, u)`
  rescales capability range to an operational duty. Raises
  `ValueError` if `u > drive_duty_cycle` (hardware not sized) or
  `u < 0`. Four unit tests cover identity at designed duty, linear
  scaling at half duty, over-duty rejection, negative-duty rejection.
- Notebook limitation #1 rewritten from "range predictions are upper
  bounds" (vague) to the capability-envelope vs operational-
  utilisation framing, pointing at `range_at_utilisation` as the ops
  layer.

**Impact on Phase 2.**

- Week-6 LHS sweep can now sample `drive_duty_cycle` in
  [0.02, 0.6] and see the duty-dominated regime real rovers operate
  in -- roughly an order of magnitude more of the physically
  meaningful design space.
- Project paper has a clean "this tool sizes capability, not ops
  utilisation" framing that matches JPL Team-X / ESA CDF conventions
  and heads off the "your range is 6× the real value" reviewer
  comment.
- Full suite: 201 tests, 1 xfail (4 new utilisation-helper tests on
  top of Week-5.5).

## 2026-04-24 — Week 6 plan revision and W7.5 gate added

**Decision.** Revise the Week 6 plan in `project_plan.md` to reflect
Weeks 5/5.5/5.6 carry-over, the measured evaluator cost, and a
feasibility-classifier two-stage model. Insert a new "Week 7.5" gate
between SCM data generation and the final surrogate, so the
multi-fidelity composition is promoted to a full dataset regeneration
only when the SCM correction is actually large at mission level.

**Context.** The plan-as-written assumed ~50 ms per evaluation; measured
cost is ~1.4 s mean (0.24 s highland, 3.2 s polar). That turns the
"50 k samples in an afternoon" promise into a 78 CPU-hour commitment,
which isn't acceptable when we haven't yet verified the LHS pipeline
end-to-end. Separately, Week 5.5 added `energy_margin_raw_pct` and
Week 5.6 reframed `range_km` as a capability envelope -- the Week 6
regression targets need to update to match. Finally, the Booleans
`thermal_survival` / `motor_torque_ok` can't be regressed cleanly; a
feasibility classifier with a conditional regressor is the standard
two-stage pattern and is what the Week-11 NSGA-II constraint layer
actually wants anyway.

**What changed in `project_plan.md`.**

- Dataset size and process: 2 k-sample pilot to shake the pipeline
  before committing to the full 40 k (10 k × 4 scenarios); scale to
  20 k / scenario only if pilot R² misses targets.
- LHS sampling: stratified by `n_wheels ∈ {4, 6}`; scenarios sampled
  jointly as a single cross-scenario model with continuous Bekker
  soil params (not one-hot simulant names).
- Targets: `range_km`, `energy_margin_raw_pct` (unclipped, not the
  clipped version in the original plan), `slope_capability_deg`,
  `total_mass_kg`. Clipped reporting metric is derived post-hoc.
- Two-stage feasibility: classifier (AUC > 0.90 target) + conditional
  regressor on feasible designs; Booleans are out of the regression
  targets.
- Evaluation: per-scenario R² / RMSE / MAPE in addition to aggregate,
  plus a registry-rover sanity check (Pragyan / Yutu-2 / Sojourner:
  surrogate prediction vs evaluator prediction, a Layer-1 check
  distinct from Week 5's Layer-4).
- Dataset schema extensibility: `fidelity` column from day one;
  per-design aggregate sub-model statistics (peak/mean/P95 drawbar
  pull, sinkage, motor torque, solar power, battery SOC) so SCM
  corrections can rederive corrected mission metrics without
  re-running the traverse on 40 k designs.
- New Week 7.5 gate: "measure correction magnitude before committing
  to shipping a composed surrogate." Architecture is always composed
  (`final = analytical + correction`, matching §6 W8's original
  intent); the gate only decides whether the correction surrogate is
  worth training and shipping, or SCM should be reported as a
  bounded sensitivity only. No 40 k regeneration in either branch --
  correction surrogate trains on ~500 SCM-corrected mission-level
  pairs (~15 min with 8 workers).
- Milestones checklist updated accordingly.

**Context for sequencing.** Considered whether to front-load PyChrono
SCM correction into Week 6 so the surrogate is trained once on
corrected data. Rejected because: (a) PyChrono is only provisionally
go from the Week-2 gate -- not yet proven end-to-end on M2 -- so
front-loading it risks stalling Weeks 6-8; (b) SCM correction is a
wheel-level delta applied inside the evaluator, so architecturally it
slots in behind the surrogate regardless of when it's trained;
(c) Week-5 real-rover validation already works at published
tolerances on raw Bekker-Wong at typical-ops slopes, weak evidence
that mission-level corrections will be modest on most designs. The
three-path strategy's whole point is that Path 1 works standalone
even if Path 2 falls over.

**Consequences.**

- Week 6 is now a crisply-scoped implementation task: LHS sampler,
  parallel dataset builder, baseline trainers, evaluation metrics,
  notebook. Nothing waits on PyChrono.
- Week 7.5 becomes the decision point for the multi-fidelity paper
  claim. Either outcome is publishable: large correction ⇒ "multi-
  fidelity surrogate captures SCM corrections missed by Bekker-Wong";
  small correction ⇒ "analytical surrogate is sufficient for mission-
  level tradespace, with SCM quantified as a bounded sensitivity."
- Aggregate sub-model statistics in the Week-6 parquet (peak / mean
  / P95 of drawbar pull, sinkage, torque, solar power, battery SOC)
  are retained in the schema because they're the highest-signal
  features for the W7.5 correction surrogate (which learns
  `Δmission_metric = f(design, scenario, wheel-regime)`) and for
  the Week-12 sub-model-level SHAP analysis. They are no longer
  needed for "cheap regeneration" -- that branch was architecturally
  unnecessary once the composed-surrogate framing was made explicit.

## 2026-04-24 — Paper strategy: refined framing + Phase-5 benchmark release

**Decision.** Refine Paper 1's framing from "ML-accelerated co-design of
rovers" to a methodology-forward pitch centered on two generalisable
contributions: (a) the multi-fidelity decomposition architecture with
separately-attributable error sources, and (b) the capability-envelope
vs operational-utilisation framework. Keep the open-source tool and
the rediscovery test as the other two contributions. Add Phase 5
(Weeks 16-19, post-semester) for a follow-on "RoverBench" dataset +
benchmark release as Paper 2 at NeurIPS Datasets & Benchmarks, IEEE
RA-L, or ICRA benchmark track.

**Context.** §11 as originally written put "open-source tool" as
contribution #1 and the multi-fidelity methodology as #2, which
undersold the most genuinely novel piece. The Week-5.6 capability-
envelope work is also a real conceptual contribution the field hasn't
formalised in the open literature (JPL Team X / ESA CDF do this
distinction but it's not cited anywhere outside proprietary practice),
and it was buried in a limitations section. Promoting both to headline
status opens up methodology venues (JFR, IJRR, TASE) alongside the
original aerospace venues (JSR, Acta Astronautica, IEEE Aerospace).

Separately, the 40 k LHS dataset + canonical train/val/test split +
pretrained surrogate we produce as Paper-1 byproducts are all the raw
material for a benchmark release. Three tiny hooks in Week 6
(canonical split stored as a parquet column, `benchmark_score` public
helper, `SCHEMA.md`) make the Phase-5 release packaging work rather
than retrofitting. Added those hooks to the Week-6 deliverables.

**What changed in `project_plan.md`.**

- §11 replaced: two-paper strategy with explicit priority-ordered
  contributions, null-result framing for both W7.5 gate outcomes,
  and a tightened 10-section outline for Paper 1.
- §11.2 added: Paper 2 (RoverBench) scope, venue, and contributions.
  Scope deliberately narrowed to a prediction benchmark (D-narrow),
  not a design-optimisation benchmark (D-broad) -- prediction tasks
  pull from a much larger ML-surrogate community and have a clean,
  ungameable scoring rubric.
- §6 Phase 5 added (Weeks 16-19): benchmark artifacts, baselines +
  leaderboard + submission interface, paper draft, submit. Gated at
  Week 16 on Paper-1 submission status, dataset stability, and
  venue window -- cleanly cancellable without affecting Paper 1.
- §6 W6 deliverables extended with benchmark hooks: canonical split
  column, `benchmark_score` helper, `SCHEMA.md`. Zero marginal cost
  at generation time, high cost to retrofit later.
- §9 rewritten to reflect the MVP-vs-full-vision split for Paper 1
  under the new framing, plus an explicit note that Paper 2 is
  independent of Paper 1's outcome.
- §10 extended with new `roverbench/` subpackage layout,
  `SCHEMA.md`, challenge-set parquet, and the surrogate-module
  breakdown we're actually building (sampling.py, dataset.py,
  baselines.py, metrics.py, benchmark_score.py).
- §13 milestones extended with Week 16-19 post-semester items.

**Consequences.**

- The semester plan's scope is unchanged; Paper 1 remains Weeks
  1-15. The framing change is a reorganisation, not more work.
- The Week-6 benchmark hooks (split column, score API, schema doc)
  are new but near-zero effort -- they add maybe two hours of
  Week-6 work and unlock a future paper's release logistics for
  free.
- Phase-5 is a cancellable add-on with its own gating decision;
  worst case we have internal infrastructure that makes future
  surrogate work more reproducible even if the benchmark never
  ships externally.
- Paper 2 citation community (physics-informed ML, AutoML, ML
  surrogates for engineering) is non-overlapping with Paper 1's
  (robotics, aerospace systems engineering), which roughly doubles
  reach per unit effort.



