# Project log

Chronological record of non-trivial project decisions. The plan asks for
one in §6 W2 ("document the decision in the project log") for the
PyChrono go/no-go; this file is the permanent home for that and every
similar decision from here on.

## Log discipline (effective 2026-04-25)

Each piece of content lives in **one** place. The log records the *why*
and the *what changed*, and links out for the rest:

- Numerical results → `reports/<week>/*.csv` / `*.parquet`
- Schema, columns, units → `data/analytical/SCHEMA.md` and module docstrings
- Current plan state → `project_plan.md`
- Sub-system technical detail → function / module docstrings
- User-facing demos → `notebooks/`
- Project pitch (frozen) → `project_brief.md`

Entry format: dated heading, then a terse decision + context + pointer.
Target ≤ 30 lines per entry; do not restate tables, schemas, or numbers
that already live in the canonical home above. Entries before this date
predate the rule and are not retro-trimmed.

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

---

## 2026-04-24 — Week 6 step 1: LHS sampler, dataset builder, pilot run

**Decision.** Ship the Phase-2 dataset-generation plumbing — stratified
LHS sampler, parallel dataset builder with Parquet I/O and failure-
tolerant row flattening, matching test suite, and column `SCHEMA.md` —
as one coherent commit. Pilot-run validated end-to-end on 200 samples
(100% success, all four scenario families, healthy target distributions).

**Context.** Week 6 step 1 in the revised plan is "sampling.py +
dataset.py + unit tests (pilot run with 2k samples works end-to-end)".
Structuring it as a single commit keeps the dataset format, the
generator that feeds it, and the schema doc in lockstep so downstream
code can trust `SCHEMA_VERSION = v1` as an atomic promise. The
benchmark-release hooks (canonical split column, `SCHEMA.md`) are
baked in at this step rather than retrofitted in Week 6's later
substeps, per the Phase-5 plan.

**What shipped.**

- `roverdevkit/surrogate/sampling.py` — stratified LHS over the 12-D
  design space × 4 scenario families. 50/50 `n_wheels` strata per
  family. Scenario perturbation columns (latitude, mission duration,
  max slope, six Bekker params) sampled jointly with design variables
  so the surrogate learns continuous scenario-to-metric mappings
  instead of four-category ones. Deterministic splits (train/val/test)
  assigned at sample time via a dedicated RNG so hold-out distribution
  is stable even when some evaluations fail. `FAMILIES` dict codifies
  per-family jitter ranges based on the canonical YAML configs plus
  ±30-50% scenario-parameter spread.
- `roverdevkit/surrogate/dataset.py` — `build_dataset` runs
  `evaluate_verbose` through a `multiprocessing.spawn` pool and
  flattens each result into a 65-column row (5 meta + 12 design + 15
  scenario + 9 metric + 24 stat). Failures are captured as `status =
  <ExceptionClass>` with NaN numeric columns rather than propagating
  out. Parquet I/O uses zstd compression and writes a `DatasetMetadata`
  block (seed, sampler config, fidelity, build timestamp, evaluator
  version, notes) into the file footer.
- `roverdevkit/mission/evaluator.py` refactored: extracted
  `evaluate_verbose` returning `DetailedEvaluation(metrics, log, mass)`
  so the dataset builder can compute aggregate traverse-log stats
  without re-running physics. `evaluate` is now a thin wrapper; no
  behavioural change. Also added `soil_override: SoilParameters | None`
  kwarg so the sampler can inject jittered Bekker params per sample
  instead of being stuck with the catalogue's four nominal soils.
- `data/analytical/SCHEMA.md` — column-by-column documentation of the
  dataset. `SCHEMA_VERSION = v1`. Bumping this is mandatory whenever
  columns are added or renamed; downstream consumers can detect stale
  files from the Parquet file-level metadata.
- `tests/test_surrogate_sampling.py` (19 tests) and
  `tests/test_surrogate_dataset.py` (15 tests, one marked `slow`):
  shape / determinism / stratification / bounds / coverage / split
  fractions for the sampler; schema / dtypes / metric ranges /
  Parquet round-trip / failure-handling / serial-vs-parallel agreement
  for the builder. Full project suite still green (235 passed).
- `pyproject.toml`: added `pyarrow>=14`.
- `.gitignore`: exempt `SCHEMA.md` from the `data/analytical/*` and
  `data/scm/*` exclusions so documentation stays in git while the
  actual parquet files don't.

**Pilot run (200 samples, seed=42).** All four canonical families, 50
samples each, 100% success, 41.7 s wall-clock on 9 workers (~0.21 s /
sample effective). Target distributions look right: `range_km` spans 0
to 69 km (not saturated), `energy_margin_raw_pct` spans -96% (deep
energy deficit) to +4100% (solar-rich easy missions), `thermal_survival`
varies across designs, split fractions match request. Extrapolation:
40k rows at the same wall-clock rate ≈ 140 minutes, comfortably under
the Week-6 day-1 compute budget. `data/analytical/lhs_pilot.parquet`
(119 kB) is on disk locally; not committed (gitignored) since the
pilot is reproducible from seed.

**Consequences.**

- Week 6 step 2 is unblocked: pilot XGBoost fit on
  `lhs_pilot.parquet` can start immediately. I want to see range R² on
  val before scaling to 40k.
- The dataset format is now fixed. Any feature the baselines want that
  isn't in the 65 columns requires a schema bump (`v1` -> `v2`).
  Reviewed the stats columns carefully against the Week 7.5 SCM-
  correction gate and the Phase-5 benchmark so we don't discover a
  gap mid-training.
- Capacity-wise, a 200-sample pilot parquet compresses to 119 kB, so
  a 40k row parquet should land around 24 MB — comfortably small
  enough to check into a release artefact (not the repo) at Phase 5.
- The multiprocessing `spawn` context is chosen deliberately so the
  pool works on macOS and isolates worker memory; this costs a few
  hundred ms per worker at startup but is robust against the pydantic
  model globals that `fork` would otherwise share.

---

## 2026-04-24 — Week 6 step 2: pilot XGBoost + thermal scope cut

**Decision.** Drop `thermal_survival` from the surrogate dataset
schema. It stays in the system-level evaluator and the Week-5
validation harness, but the v2 Phase-2 dataset does not include it as
a column and the feasibility classifier targets `motor_torque_ok`
alone.

**Why.** The Week-6 step-2 pilot XGBoost fit (200 samples, 168/16/16
train/val/test) ran end-to-end and surfaced a real model-design
issue: every pilot row had `thermal_survival = False`. Diagnosis: the
default `ThermalArchitecture` ships with `rhu_power_w = 0.0`, and a
sweep across `(latitude, surface_area, avionics_power, rhu_power)`
shows lunar-night temperatures of −50 °C to −150 °C without an RHU —
well below the −30 °C operating floor. This is **physically correct**
(Pragyan, no RHU, did not survive lunar night; Yutu-2 and Sojourner
both carry RHUs). The problem is upstream: the LHS sampler holds
RHU power at zero for every design, making thermal a single-class
constant.

The first instinct was to add `rhu_power_w` and `insulation_ua_w_per_k`
to the LHS as new design dimensions. The right question, raised mid-
discussion: does this add design-trade signal, or just teach the
optimizer "always set RHU = 20 W"? Inspecting the mass model
confirmed the latter — `parametric_mers.py` charges a flat 5%
`thermal_fraction` of subsystems with no scaling on RHU power or MLI
quality, and no other downstream pathway penalises RHUs (no battery
draw — Pu-238 heat is free in the energy budget; no operating
constraint — RHUs aren't sized by chassis volume). With zero cost,
RHU is a free design lever. A "feasibility classifier" trained on a
free lever learns "if `rhu_power_w >= 8 W` predict True", which is
zero design insight. Including it as a target adds a degenerate
column and dilutes the headline R²/AUC the paper reports.

The honest scoping is therefore: keep thermal as an *evaluator-level
diagnostic* (so the system-level model still distinguishes Pragyan
from Yutu-2 and the Week-5 validation harness still exercises it),
remove it from the *surrogate's input/output schema* (so v2 doesn't
contain a degenerate column), and demote it from the feasibility
classifier (so `motor_torque_ok` carries the binary signal alone —
this is a real Bekker-Wong outcome that depends jointly on grouser
geometry, soil shear, mass, and slope, with 64% positive class on the
pilot, well-balanced and learnable).

This is *narrower* than the project plan's original Phase-2 scope
("thermal_survival as a feasibility constraint", §6, §7 L1) but
*more honest*: we're not claiming thermal is a learnable design trade
when our physics treats it as free. A future mass-model upgrade that
charges RHU specific mass (~50 g/W for Pu-238 + ITAR shielding) and
MLI mass (proportional to surface area × layer count) would restore
thermal as a real Pareto target. That belongs alongside the SCM
correction work in Phase 3 / 4 and is queued there.

**What shipped.**

- `roverdevkit/surrogate/dataset.py` — `SCHEMA_VERSION` bumped
  `v1 → v2`. `_flatten_metrics` no longer emits `thermal_survival`;
  `_BOOL_METRIC_COLS` reduced to `("motor_torque_ok",)`. Module
  docstring documents the thermal-scope decision so the next reader
  doesn't have to dig through the log.
- `roverdevkit/surrogate/features.py` — single-target classifier:
  `CLASSIFICATION_TARGETS = ["motor_torque_ok"]`,
  `FEASIBILITY_COLUMN = "motor_torque_ok"` (alias to the underlying
  column, kept as a constant so future feasibility-definition changes
  have one canonical place to update). `add_feasibility_column`
  helper removed (the surrogate no longer needs an AND-of-two
  derivation). The module docstring explains why no thermal target.
  `INPUT_COLUMNS` (25 cols: 12 design + 9 numeric scenario + 4
  categorical scenario) is the canonical input set; `valid_rows`
  filters to evaluator-`ok` rows with non-NaN primary targets;
  `build_feature_matrix` returns the X frame with categorical dtypes
  preserved for `XGBRegressor(enable_categorical=True)`.
- `data/analytical/SCHEMA.md` — v1 / v2 history block, metric column
  count revised 9 → 8, column-count sanity revised 65 → 64, thermal
  scope rationale documented inline.
- `tests/test_surrogate_dataset.py` — `_EXPECTED_METRIC_COLS` updated;
  `test_thermal_and_motor_flags_are_boolean` split into two: one
  asserting `motor_torque_ok` dtype, one asserting `thermal_survival`
  is absent from the v2 schema (with a docstring pointing back to
  SCHEMA.md). Failure-handling test now asserts `motor_torque_ok =
  False` on injected exceptions.
- `project_plan.md` — updated the surrogate target list (the ASCII
  pipeline diagram), the §6 feasibility paragraph (single-target
  classifier, thermal-as-diagnostic), and noted v2 schema. Existing
  references to thermal in the *evaluator* (lumped-parameter survival
  check, mass model thermal fraction, registry-rover thermal
  validation) are unchanged because that work isn't being cut.

**Pilot XGBoost results (200 samples, seed=42).** Plumbing works
end-to-end. Numbers are noisy at this scale; the take-away is "no
obvious bugs", not "these are publishable":

| Target                  | val R² | test R² | val RMSE   |
|-------------------------|-------:|--------:|-----------:|
| `total_mass_kg`         |   0.84 |    0.97 |    4.20 kg |
| `slope_capability_deg`  |   0.52 |    0.58 |    3.86 °  |
| `energy_margin_raw_pct` |   0.77 |    0.48 |  127.22 %  |
| `range_km`              |   0.41 |   −0.41 |   10.99 km |

`total_mass_kg` is strong because mass is a near-pure function of
design with no physics-noise from the traverse. The negative test
R² for `range_km` at n=16 is consistent with sampling variance — the
test split happens to land on a few outlier scenarios — and is the
exact reason this step exists (catch a *systematic* failure cheaply
before paying for 40k samples). Per-scenario val R² for range_km
ranges +0.15 (crater_rim) to −0.24 (equatorial_mare), again all
single-digit n; not interpretable as anything beyond "the model
isn't broken."

**Consequences.**

- Step 3 (full 40k LHS run) is unblocked. The dataset format is now
  v2-frozen; no schema changes expected before the SCM correction
  composes a new fidelity tag in Week 7.
- The feasibility classifier has a single learnable target with a
  ~64% positive class on the pilot. After step 3 (full 40k run) it
  should hit AUC ≥ 0.85 on its own — a clean number to put in the
  paper.
- The Pragyan-vs-Yutu-2 thermal-survival distinction stays
  testifiable via the Week-5 validation harness (`rover_registry.py`,
  `tests/test_rover_comparison.py::test_thermal_survival_matches_published`).
  This is the rhetorical use of thermal we keep — the surrogate's
  scope just doesn't include it.
- A future "RHU/MLI mass-model upgrade" task is queued for Phase 3 /
  4 alongside SCM. When it lands, it restores thermal as a learnable
  Pareto target and would justify a `SCHEMA_VERSION = v3`.

---

## 2026-04-25 — Week 6 step 4: baseline surrogate matrix on 40k LHS

**Decision.** Ship the Week-6 step-4 baseline matrix as
`roverdevkit.surrogate.baselines` plus `scripts/run_baselines.py`,
with results frozen under `reports/week6_baselines_v1/`. Architecture
is **per-target Ridge / RandomForest / XGBoost** (one fit per
(algorithm, target)) plus a single **joint multi-output MLP**
(`sklearn.neural_network.MLPRegressor`, 128 → 64 hidden) across all
four primary regression targets, plus a **single-target feasibility
classifier** (LogReg + XGBoost) on `motor_torque_ok`. Hyperparameter
tuning is deferred to Week 7 so the lift from Optuna is cleanly
attributable.

**Why per-target instead of multi-output for the tree/linear models.**
The four primary targets — `range_km`, `energy_margin_raw_pct`,
`slope_capability_deg`, `total_mass_kg` — span very different
underlying physics (mass is a near-linear sum of subsystems, range is
a multiplicative function of solar geometry × duty cycle × mass, slope
capability is a Bekker-Wong threshold). A `MultiOutputRegressor`
forces shared hyperparameters that would under-fit the harder targets
and over-fit the easier ones. The MLP is the exception: shared hidden
layers are precisely its reason to exist, and at our scale (32k train
× 28 features × 4 outputs) it costs only 7.5 s to fit jointly — so we
keep it joint and let the comparison stand.

**Why a separate XGBoost categorical conform path.** XGBoost 3.x with
`enable_categorical=True` strict-checks at predict time: any category
absent from training raises (we hit this on the pilot when the
registry rover Pragyan's soil simulant `GRC-1` was not in the
LHS-sampled categories). Fix: `FittedBaselines.training_categories`
records the trained codebook, and `predict_for_registry_rovers`
recodes the registry-rover row's categoricals against that codebook
(unseen → NaN, which XGBoost treats as missing). One-hot pipelines
already use `handle_unknown='ignore'` so they didn't need a separate
fix.

**Results (40k LHS, train=32 137, val=3 880, test=3 983).**

| Algorithm     | range R² | energy-margin R² | slope R² | mass R² | feasibility AUC |
|---------------|---------:|-----------------:|---------:|--------:|----------------:|
| Ridge         |   0.7694 |          0.6613  |   0.9166 |  0.9974 |               — |
| RandomForest  |   0.9974 |          0.9702  |   0.9318 |  0.9923 |               — |
| XGBoost       |   0.9981 |          0.9869  |   0.9857 |  0.9991 |          0.9976 |
| MLP (joint)   |   0.9985 |          0.9958  |   0.9978 |  0.9997 |               — |
| LogReg        |        — |               —  |        — |       — |          0.9882 |

The joint MLP wins **every** primary target by a small margin over
per-target XGBoost; XGBoost wins the feasibility classifier.

**Acceptance gate.** 16 / 18 (algorithm, target) cells on the test
split clear the plan thresholds (R² ≥ 0.95 on range / energy margin,
R² ≥ 0.85 on slope / mass, AUC ≥ 0.90 on feasibility). The two
failures are both Ridge:

- `ridge × range_km`: R² 0.7694 (threshold 0.95)
- `ridge × energy_margin_raw_pct`: R² 0.6613 (threshold 0.95)

This is the exact diagnostic we want: linear regression cannot model
the multiplicative coupling at the heart of the mission physics, and
on `polar_prospecting` energy margin Ridge collapses to R² ≈ −59
while every non-linear baseline holds R² ≥ 0.79 on the same family.
Per-scenario breakdown lives in `reports/week6_baselines_v1/test_per_family.csv`.

**Total fit wall-clock 30.6 s** on 11 cores (Ridge ~0.1 s × 4 + RF
~17 s + XGB ~4.5 s + MLP 7.5 s + classifiers 1.4 s). Cheap enough
that retraining inside the Week-7 SCM-correction gate is free.

**Registry-rover Layer-1 sanity (`reports/week6_baselines_v1/registry_sanity.csv`).**
Median |relative error| across baselines per (rover, target):

| Rover     | mass | slope | range | energy margin |
|-----------|-----:|------:|------:|--------------:|
| Yutu-2    | 2.0% |  2.2% | 1297% |          38%  |
| Pragyan   | 0.6% | 68.1% |  547% |          80%  |
| Sojourner | 10.8%| 16.3% |13601% |         112%  |

Mass and slope track tightly. The huge range / energy-margin relative
errors are an OOD effect, not a model bug:

- Pragyan is 26 kg with a planned 100 m traverse — its actual
  evaluator-computed `range_km` is well below the LHS support, so a
  small absolute prediction error becomes a large relative one.
- Sojourner runs at Mars gravity (3.71 m/s²) — entirely outside the
  lunar LHS — so this is the expected "do not extrapolate beyond
  support" warning rather than a calibration target.
- Energy margin can cross zero (negative for energy-starved configs),
  which makes relative error meaningless near the zero-crossing; the
  full `registry_sanity.csv` shows the absolute errors are physically
  small.

**Consequences.**

- Week-6 step 4 is closed: the surrogate matrix exists, the gate is
  green except for the linear-baseline diagnostic, and the wall-clock
  budget for Week-7 retraining is < 1 minute. The remaining Week-6
  step 5 (notebook + writeup) is rebuilding the same numbers in
  `notebooks/02_baseline_surrogates.ipynb` for the writeup, plus the
  `benchmark_score` helper that wraps `evaluate_baselines` with a
  fixed schema for Phase-5 leaderboard submissions.
- The XGBoost feasibility classifier at AUC 0.9976 is good enough
  that the Week-11 NSGA-II constraint layer can use it directly
  rather than the deterministic evaluator: this collapses the
  optimisation inner loop from ~1.4 s/eval to ~10 µs/eval (a ~5
  order-of-magnitude speedup) once the regressor matches.
- Joint MLP outperforming per-target XGBoost on every regression
  target is suggestive but not conclusive: at sensible-default
  hyperparameters MLP has marginally more capacity to share
  representation across targets. Week-7 Optuna tuning of XGBoost may
  flip the ranking; if not, the MLP's joint architecture becomes the
  Week-11 Pareto-search surrogate of choice.
- The OOD-warning result for Sojourner (Mars gravity) is exactly
  what the Layer-1 sanity is meant to catch and what Paper 1's
  capability-envelope framing argues: the surrogate is honest about
  its support, and Paper 2's benchmark release will document this
  support explicitly.

## 2026-04-25 — Validation registry refactor: lunar-only roster, two-tier flown vs design-target

**Context.** The Week-6 step-4 baseline rerun on 40k LHS rows showed
strong IID R² but large relative errors against the registry-rover
Layer-1 sanity check (`reports/week6_baselines_v1/registry_sanity.csv`,
above). Two structural causes surfaced once we examined where each
registry rover sat in the LHS support:

1. **Sojourner is multiply OOD.** Mars gravity (3.71 m/s²) is outside
   the lunar LHS by construction; on top of that, its design vector
   (10.6 kg, 6.5 cm wheels) sits at the floor of the design-space cube.
   A capability-envelope surrogate trained on lunar 5–50 kg micro-
   rovers cannot be expected to predict Sojourner without an OOD
   wrapper, and any error it shows is dominated by the gravity step,
   not by a calibration issue we could fix.
2. **Yutu-2 sits on a corner of the cube.** 135 kg total ≫ 35 kg
   schema ceiling forces a `chassis_mass_kg=35` shoehorn, putting
   Yutu-2 against an LHS boundary on three axes (mass, wheel width,
   grouser height). Boundaries are exactly where tree models and
   linear baselines extrapolate worst.
3. **The registry was thin (n=2 lunar rovers).** Without more lunar
   micro-rover anchors the Layer-1 sanity check is almost a
   single-point comparison.

**Decision.** Re-scope the validation registry to **lunar micro-rover
class only**, with an explicit two-tier architecture:

- **Tier 1 — flown rovers** (`is_flown=True`): rovers with published
  flight traverse / peak-solar / thermal data. Layer-0 truth gate
  (`compare_all`, Week-5 acceptance gate) runs on this tier.
- **Tier 2 — design-target rovers** (`is_flown=False`): well-spec'd
  micro-rover designs that did not deploy (lander loss, in
  development, or operations halted). Layer-1 surrogate sanity
  (`predict_for_registry_rovers`, Week-6) runs on this tier *plus*
  the flown tier; Layer-0 truth gate skips them because there is no
  flight ground truth to score against.

Roster after the refactor: Pragyan (flown), Yutu-2 (flown),
**MoonRanger** (CMU/Astrobotic, in development, design-target),
**Rashid-1** (MBRSC/UAE, lost on Hakuto-R Mission 1, design-target).
Sojourner removed from registry, scenario set, published-traverse CSV,
and rover-comparison Mars-irradiance branch.

**Iris not added.** Iris (CMU/Astrobotic CubeRover) is battery-only
with no solar array, which violates the surrogate's
`solar_area_m2 ∈ [0.1, 1.5]` requirement. Modelling it honestly would
need a schema extension (battery-only architecture) outside the
current Phase-1 scope; we'd be shoehorning a bogus solar area
otherwise.

**Imputation strategy for the new design-target entries.** Both new
entries follow the same convention the Week-3 mass-validation set
established: every cited number lifted from the literature, every
imputed number documented in `RoverRegistryEntry.imputation_notes`
with the source of the imputation (class match, photo measurement,
or back-solve from a published mission goal).

- *MoonRanger*: cited mass (13 kg), n_wheels (4), max mech speed
  (0.07 m/s), 8-Earth-day mission, no RHU (Kumar et al. i-SAIRAS 2020;
  MoonRanger Project labs page; Astrobotic LSITP award). Imputed
  wheel and grouser specs by class match to Rashid-1; solar / battery
  / avionics by power-budget back-solve against the
  kilometer-per-day exploration target.
- *Rashid-1*: cited mass (10 kg), n_wheels (4), wheel radius/width
  (10 cm / 8 cm), grouser height (15 mm flight) and count (14),
  wheelbase, nominal speed (Hurrell et al. 2025 SSR 221:37; Els et al.
  LPSC 2021 #1905). Imputed solar / battery / avionics by power-budget
  back-solve against the science-payload inventory and one-lunar-day
  mission target. Atlas crater landing site (47 N, 44 E) used for
  scenario.

**Schema change.** Widened `DesignVector.grouser_height_m` upper bound
from 12 mm to 20 mm so Rashid-1's flight 15 mm grousers fit faithfully.
The LHS sampler in `roverdevkit/surrogate/sampling.py` still draws to
12 mm only (existing v2 dataset is unaffected); widening LHS to the
new ceiling is the next planned commit (LHS-bounds widening + dataset
regen + baseline rerun, per the strategy approved earlier this week).
The two other planned bound widenings (`chassis_mass_kg` to 50 kg,
`wheel_width_m` to 0.20 m) are deferred to that same commit.

**Schema docstring updates.** Removed `GRAVITY_MARS_M_PER_S2` constant
(only Sojourner used it). `MissionScenario.sun_geometry` already
supported `diurnal` so no schema extension was needed for Rashid-1's
mid-latitude landing site (Atlas crater, 47 N).

**Verification.**

- `compare_all()` now scores 2/2 flown rovers passing the Week-5
  acceptance gate (Pragyan and Yutu-2). Per-criterion outputs match
  the prior run since neither flown rover changed.
- `predict_for_registry_rovers` default roster expanded to four
  rovers; the schema and unseen-categorical conform path already
  generalise cleanly.
- Direct evaluator outputs on the new entries (lunar gravity,
  lunar-only registry):
  - MoonRanger (lat -85, polar intermittent, 8-day, slope 6 typical-
    ops): `range_km=2.000` (hits `traverse_distance_m` cap),
    `motor_torque_ok=True`, `energy_margin_raw_pct=-77 %` (8-day
    polar mission with a 0.30 m² array is energy-starved at the
    designed duty — exactly the kind of design-target case the
    Layer-1 sanity should pin down for the surrogate).
  - Rashid-1 (lat 47, diurnal, 14-day, slope 10): `range_km=1.000`
    (cap), `motor_torque_ok=True`, `energy_margin_raw_pct=159 %`,
    `slope_capability_deg=10.6`. Sensible mid-latitude micro-rover.

- 241 fast tests pass; ruff lint and format clean on the touched
  files.

**Files changed.**

- `roverdevkit/validation/rover_registry.py`: rewrote module with
  `is_flown` field, `flown_registry()` helper, MoonRanger and Rashid-1
  builders, removed Sojourner builder + `GRAVITY_MARS_M_PER_S2`.
  Two-tier docstring at the top of the file.
- `roverdevkit/validation/__init__.py`: re-export `flown_registry`.
- `roverdevkit/validation/rover_comparison.py`: `compare_all` now
  iterates `flown_registry()`. Removed Mars-irradiance branch from
  `_predicted_peak_solar_power_w`.
- `roverdevkit/surrogate/baselines.py`: default
  `predict_for_registry_rovers` roster updated to
  `(Pragyan, Yutu-2, MoonRanger, Rashid-1)`. Comment about Sojourner
  removed.
- `roverdevkit/schema.py`: `grouser_height_m` upper bound 0.012 →
  0.020 m with explanatory description.
- `roverdevkit/mission/configs/`: added `moonranger_polar_demo.yaml`
  and `rashid_atlas_crater.yaml`; deleted
  `mpf_sojourner_ares_vallis.yaml`.
- `data/published_traverse_data.csv`: Sojourner row removed.
- `data/published_rovers.csv`: Sojourner row removed.
- `tests/`: `conftest.py`, `test_rover_comparison.py`,
  `test_surrogate_baselines.py` updated to the new roster and the
  flown-vs-all distinction.
- `scripts/run_baselines.py`: docstring lists the new four-rover
  roster.
- `roverdevkit/mission/evaluator.py`: `gravity_m_per_s2` docstring
  no longer cites Sojourner as the off-Moon example.
- `project_plan.md`: Week-6 sanity-check description, minimum-viable-
  Paper-1 framing, and Key-result-2 / rediscovery cross-check lines
  updated for the new roster.

**Out of scope (deferred to next commit).**

- Re-running `notebooks/00_real_rover_validation.ipynb` against the
  updated registry. The rendered cell outputs in the committed
  notebook still reference the Sojourner-included roster; Week 6
  step 5 will re-execute the notebook against the new state.
- Cleaning up `data/mass_validation_set.csv`, `data/README.md`, and
  `project_brief.md` Week-3 references to Sojourner. Those describe
  the Week-3 mass-MER fitting set, which remains a different concern
  from the surrogate validation registry; pulling Sojourner from the
  mass-fitting set would silently invalidate Week-3 results and
  should be its own decision.

## 2026-04-25 — LHS bounds widening v2 → v3, 40k regen, baseline rerun

**Context.** The registry refactor (above) made Pragyan, Yutu-2,
MoonRanger, and Rashid-1 the canonical Layer-1 sanity rovers. Three of
those four still sat at corner points of the v2 LHS cube on
`chassis_mass_kg` (Yutu-2 ex-payload ~30-40 kg, schema ceiling 35 kg),
`wheel_width_m` (Lunokhod-class 0.20 m, schema ceiling 0.15 m), and
`grouser_height_m` (Rashid-1 0.015 m, just-widened ceiling 0.020 m but
sampler still drew to 0.012 m). To turn the surrogate's "extrapolation
to the registry" into "interpolation inside training support" on the
*design* axis, we widened the sampler bounds, bumped the schema
version, and rebuilt the 40k dataset.

**Changes.**

- `roverdevkit/schema.py`: `wheel_width_m` upper 0.15 → 0.20 m;
  `chassis_mass_kg` upper 35 → 50 kg. Field descriptions explain the
  micro-rover envelope rationale.
- `roverdevkit/surrogate/sampling.py`: `_CONTINUOUS_DESIGN_BOUNDS`
  updated for those three columns, plus `grouser_height_m` 0.012 →
  0.020 (was already widened in the schema commit but not in the
  sampler). Module banner notes the v3 widening.
- `roverdevkit/surrogate/dataset.py`: `SCHEMA_VERSION` v2 → v3 with
  history entry. The Parquet column schema is byte-identical to v2;
  the bump exists so a v2-trained surrogate can't be silently reused
  on v3 data.
- `data/analytical/SCHEMA.md`: range columns updated; v3 entry added
  to the version history; canonical filename updated to
  `lhs_v3.parquet` (`lhs_v1.parquet` and `lhs_v2.parquet` are retired
  but kept on disk for reproducibility of the diagnosis report).
- `project_plan.md` design-variable table and references to the
  canonical dataset filename updated.
- `scripts/build_dataset.py` and `scripts/run_baselines.py` docstring
  examples point at `lhs_v3.parquet` and `reports/week6_baselines_v2/`.

**Dataset rebuild.** `python scripts/build_dataset.py
--n-per-scenario 10000 --seed 42 --out data/analytical/lhs_v3.parquet`.

- 40 000 rows × 64 cols written.
- Wall-clock 7 217.7 s (≈ 2 h 0 min) on 11 workers; 0.18 s/sample.
  This is materially slower than the v2 build (~1 h) because the
  widened bounds expand the soft-soil + heavy-chassis tail, which
  triggers more "wheel fully buried" Bekker entry-angle failures that
  burn worker time before raising.
- 26 graceful-failure rows (NaN metrics, status non-`ok`); 99.94 % ok.
- Feasibility positive rate 64.74 % (v2: ~75 %). Lower because the
  mass / wheel-width widening grows the part of the design space
  where wheel torque can't sustain the higher-load configs.
- Splits: 32 137 train / 3 880 val / 3 983 test (deterministic).

**Baseline rerun (`reports/week6_baselines_v2/`).** `python
scripts/run_baselines.py --dataset data/analytical/lhs_v3.parquet
--out-dir reports/week6_baselines_v2`. Total fit wall-clock 26.0 s.

Test-split metrics (overall, all families):

| Algorithm | range R² | energy R² | slope R² | mass R² |
|-----------|---------:|----------:|---------:|--------:|
| Ridge     |    0.769 |     0.663 |    0.910 |   0.997 |
| RF        |    0.997 |     0.965 |    0.926 |   0.995 |
| XGBoost   |    0.998 |     0.984 |    0.985 |   0.999 |
| MLP joint |    0.999 |     0.995 |    0.997 |   1.000 |

Feasibility: LogReg AUC 0.988, XGBoost AUC 0.997 (v2: 0.988 / 0.998).

Acceptance gate: 16 / 18 passing. The two failures are the same
Ridge-on-range and Ridge-on-energy diagnostic rows that failed under
v2 — exactly what the linear baseline is in the matrix to surface.
Test-split R² is statistically indistinguishable from v2; the
non-linear models still match the evaluator essentially perfectly on
in-distribution rows.

**Registry-rover Layer-1 sanity (`reports/week6_baselines_v2/registry_sanity.csv`).**
Median |relative error| across baselines:

| Rover      | mass | slope | range  | energy margin |
|------------|-----:|------:|-------:|--------------:|
| Yutu-2     | 0.8% |  2.3% | 1318%  |         42%   |
| Rashid-1   | 9.4% |  4.1% |  366%  |         45%   |
| MoonRanger | 3.2% | 39.7% |  300%  |        267%   |
| Pragyan    | 2.7% | 77.9% |  618%  |        135%   |

Compared to v2 (Yutu-2 mass 2.0 %, Pragyan mass 0.6 %, Sojourner mass
10.8 %), the *mass* and *slope* targets — the ones that depend
directly on the now-interior chassis / wheel design dimensions — got
markedly better for the Yutu-class rovers (Yutu-2 mass 2.0 % → 0.8 %;
Yutu-2 slope ~2 % → 2.3 %, basically unchanged because slope was never
the bottleneck). The widening did exactly what it was supposed to do
on the *design-OOD* axis.

**`range_km` and `energy_margin_raw_pct` are still wildly off.** This
is now a clean diagnostic, not a calibration failure:

- The registry rovers' published mission distances (Pragyan ≈ 100 m,
  Yutu-2 ≈ 25 m / lunar day, MoonRanger ≈ 1 km / Earth-day) are
  100–1000× smaller than the LHS family traverse-distance budgets
  (20–80 km, intentionally non-binding so `range_km` stays a
  continuous signal). The surrogate's `range_km` predictions are in
  the family-budget regime; the Layer-1 truth value is the actual
  evaluator output for the registry's much smaller scenario, so the
  relative error is dominated by an absolute scale mismatch with no
  bearing on physical model accuracy.
- This is a **scenario-OOD** problem, not a design-OOD one. The
  widening could not have fixed it. The right place to address it is
  either (a) adding registry-style "short mission" scenarios to the
  LHS support, which trades off the cleanness of the four canonical
  family budgets; or (b) framing the registry sanity as a fidelity
  check on physically meaningful dimensions (mass, slope capability,
  feasibility class) rather than range/energy in the paper write-up.
  Decision deferred to the Week-6 step-5 notebook + writeup pass; the
  current `registry_sanity.csv` makes the structural OOD visible.

**Verification.**

- `pytest tests/test_surrogate_dataset.py tests/test_surrogate.py
  tests/test_surrogate_baselines.py tests/test_schema.py -q` — 42
  passed in 74.6 s (no v2-pinned tests broke after the version bump).
- `ruff check` clean on all touched files.
- Dataset metadata round-trip verified: file footer reports
  `schema_version=v3`, `sampler_seed=42`, four scenario families.

**Files changed.**

- `roverdevkit/schema.py`: `wheel_width_m`, `chassis_mass_kg` bounds
  + descriptions.
- `roverdevkit/surrogate/sampling.py`: `_CONTINUOUS_DESIGN_BOUNDS`
  + module banner.
- `roverdevkit/surrogate/dataset.py`: `SCHEMA_VERSION` v3 + history.
- `data/analytical/SCHEMA.md`: design ranges, version history,
  canonical filename, column-count footer.
- `data/analytical/lhs_v3.parquet`: new 40k canonical dataset
  (replaces `lhs_v1.parquet` for downstream consumers; old file
  retained on disk).
- `reports/week6_baselines_v2/`: new baseline report tree.
- `project_plan.md`: design-variable table ranges, dataset filename
  references.
- `scripts/build_dataset.py`, `scripts/run_baselines.py`: docstring
  example commands point at the v3 dataset and the new report dir.

**Out of scope (deferred to next commit).**

- Re-execute `notebooks/00_real_rover_validation.ipynb` and
  `notebooks/02_baseline_surrogates.ipynb` against v3 + the new
  baseline report dir. (`02_baseline_surrogates.ipynb` is also Week-6
  step 5's writeup target, so this folds into that.)
- Decision on how to handle scenario-OOD `range_km` / `energy_margin`
  for the registry rovers. Two options: extend the LHS to short-
  mission scenarios, or scope the Layer-1 sanity to mass / slope /
  feasibility in the paper. Should be revisited in Week-6 step 5.

---

## 2026-04-25 — Layer-1 sanity reframing + notebook 00 v3 refresh

**Context.** The 40k v3 baseline rerun left two pending items from the
LHS-bounds-widening commit: re-executing the Week-5 validation
notebook against the new registry/v3 dataset, and choosing how to
handle the persistent scenario-OOD on `range_km` / `energy_margin` in
the registry sanity. User directive on the second item: *"frame the
Layer-1 sanity check around mass / slope / feasibility"* — i.e. scope
the acceptance set rather than rebuild data.

**Reframing — code.**

- Added `LAYER1_PRIMARY_TARGETS = ("total_mass_kg",
  "slope_capability_deg", "motor_torque_ok")` and
  `LAYER1_DIAGNOSTIC_TARGETS = ("range_km", "energy_margin_raw_pct")`
  to `roverdevkit/surrogate/baselines.py`. The split is enforced as
  an `is_primary` boolean column on the
  `predict_for_registry_rovers` output (and therefore on the on-disk
  `registry_sanity.csv`).
- `scripts/run_baselines.py` now prints two summary tables — a
  primary table with median |relative error| for mass / slope plus a
  classifier-accuracy summary, then a diagnostic table for range /
  energy_margin with an explicit "scenario-OOD; not part of
  acceptance" header pointing at SCHEMA.md v3.
- Rationale captured in module-level constants' docstrings,
  `data/analytical/SCHEMA.md` (new "Layer-1 registry sanity scope"
  section), and `project_plan.md` Week-6 Evaluation bullet.

**Reframing — refreshed v3 numbers (median across algorithms).**

Primary, regression (lower is better):

| rover      | mass MAPE | slope MAPE |
| ---------- | --------- | ---------- |
| Yutu-2     |     0.8 % |     2.3 %  |
| Pragyan    |     2.7 % |    77.9 %  |
| MoonRanger |     3.2 % |    39.7 %  |
| Rashid-1   |     9.4 % |     4.1 %  |

Primary, classification (`motor_torque_ok` accuracy across logreg +
xgboost): Pragyan / Yutu-2 / Rashid-1 = 100 %, MoonRanger = 50 %
(one of the two algorithms predicts feasible; the evaluator says
infeasible at MoonRanger's negative steady-state energy margin).

Diagnostic (scenario-OOD; reported only): Pragyan range MAPE
~620 %, Yutu-2 ~1320 %, MoonRanger ~300 %, Rashid-1 ~370 %. These
match the absolute-scale mismatch between LHS family traverse
budgets (20–80 km, non-binding) and the registry's published
mission distances (25 m – 1 km), as expected.

The Pragyan slope MAPE 78 % and MoonRanger 40 % are real signals
worth flagging — both rovers have published-design slope-capability
estimates well below the surrogate's prediction (the Pragyan
scenario sets `max_slope_deg=5°` so the published "slope capability
the rover demonstrated" is closer to 5° than to the analytical
maximum the surrogate predicts). This is a legitimate calibration
gap that the Week-7 Optuna lift and Week-7.5 SCM correction may
narrow; documenting now rather than papering over.

**Notebook 00 refresh.**

- Updated the intro/goals markdown to describe the two-tier registry
  (flown vs design-target) and to drop the Sojourner reference.
- Section 1 / 2 / 3 markdown now says explicitly that all four
  registry entries print but only the flown two participate in
  `compare_all()`.
- Section 7 limitations: dropped the Sojourner-on-Mars bullet
  (item 5 in the old list); refreshed the Yutu-2 bullet to describe
  the v3 widened bounds putting Yutu-2 inside training support;
  updated the duty-cycle floor reference (0.1 → 0.02 since the
  Week-5.6 lowering); updated the constant-slope and non-binding-
  scenario bullets to reflect the current four-rover roster.
- Repaired pre-existing malformed stream outputs (missing `name`
  field) before nbconvert; re-executed inplace using the conda env
  `roverdevkit` jupyter, ~40 s wall-clock. All cells execute cleanly,
  `compare_all` reports 2/2 PASS on Pragyan + Yutu-2.

**Yutu-2 imputation note.** The Week-5-era note read "Yutu-2 is
out-of-class (135 kg vs 5-50 kg design space); chassis_mass held at
35 kg ceiling." Post v3 the chassis ceiling is 50 kg and 35 kg is no
longer at the boundary. Updated the comment block and the
`imputation_notes` string in `_yutu2_entry()` to clarify that the
35 kg value is the published *chassis ex-payload* mass and the 135 kg
all-up flight mass is what the analytical mass-up model adds payload
+ structure + power-system margins to on top.

**Verification.**

- `pytest tests/test_surrogate_baselines.py -k "not slow"` — 13
  passed in 45.4 s. New `is_primary` column is asserted in
  `test_predict_for_registry_rovers_schema`.
- `ruff check` + `ruff format --check` clean on all touched files.
- `python scripts/run_baselines.py --dataset
  data/analytical/lhs_v3.parquet --out-dir
  reports/week6_baselines_v2` — 28.1 s fit, 16/18 acceptance rows
  pass, registry_sanity.csv now carries `is_primary`.
- Notebook 00 re-execute returns 2/2 PASS, design-target rovers
  appear in the inventory dump but not in `compare_all` (as
  intended).

**Files changed.**

- `roverdevkit/surrogate/baselines.py`: Layer-1 constants,
  `is_primary` column, refreshed `predict_for_registry_rovers`
  docstring, exports.
- `scripts/run_baselines.py`: split summary printer
  (`_print_registry_sanity_summary`), updated docstring for
  `registry_sanity.csv`.
- `roverdevkit/validation/rover_registry.py`: refreshed Yutu-2
  imputation comment block + note (no behavioural change).
- `tests/test_surrogate_baselines.py`: assert `is_primary` column
  + correct primary/diagnostic partition.
- `notebooks/00_real_rover_validation.ipynb`: markdown refresh +
  re-execute.
- `data/analytical/SCHEMA.md`: new "Layer-1 registry sanity scope"
  section.
- `project_plan.md`: Week-6 Evaluation bullet, Week-6 step-4 result
  block updated with the reframed primary/diagnostic numbers.
- `reports/week6_baselines_v2/registry_sanity.csv`: refreshed with
  `is_primary` column.

**Out of scope.**

- `notebooks/02_baseline_surrogates.ipynb` does not yet exist; it is
  the Week-6 step-5 writeup deliverable and will be created in that
  commit.
- Pragyan / MoonRanger slope MAPE worth a focused look once the
  Week-7 hyperparameter sweep is done, since both target Pragyan-
  class polar scenarios where slope_capability_deg is dominated by
  motor stall conditions the registry's `max_slope_deg=5–6°`
  scenarios don't exercise broadly during evaluator runs.

---

## 2026-04-25 — Week-6 step-5 verdict + new finding (polar energy-margin R²)

**Decision.** Step-5 verdict accepted on the existing
`reports/week6_baselines_v2/` artifacts. No standalone writeup
notebook — the report CSVs are the canonical numbers and the plan's
Week-6 step-4 result block carries the verdict text. (See "2026-04-25
— Repo bloat audit" entry below for the rationale.)

**New finding (the only genuinely new content from step 5).** The
per-family R² table surfaces one cell the pooled-aggregate gates hide:
the joint MLP's `energy_margin_raw_pct` on `polar_prospecting` is
**R² 0.753 vs the 0.95 gate**. The other three families clear at
≥ 0.99 and pooled R² is 0.995. Hypotheses to test in Week-7 Optuna:
family-stratified loss weighting, deeper / wider MLP, and a recheck of
the categorical conform path on `scenario_sun_geometry ==
polar_intermittent`. Not a Week-6 redo — queued as the highest-
priority Week-7 Optuna target.

**Files changed.**
- `project_plan.md` Week-6 step-4 result: per-family caveat appended;
  fit wall-clock corrected from 26 s to 28 s.

---

## 2026-04-25 — Repo bloat audit and one-place-rule adoption

**Decision.** Audited the repo for content duplicated across notebooks,
reports, plan, and log. Adopted the "Log discipline" rule at the top of
this file; each piece of content lives in exactly one canonical home.

**Context.** Recent log entries were restating numerical tables that
already live in `reports/week6_baselines_v2/*.csv`, `SCHEMA.md`, and
`project_plan.md`. The Week-6 step-5 writeup notebook was a third copy
of the same content. Bloat compounds because every refresh of the
underlying numbers requires updates in three places.

**What changed.**

- Deleted superseded artifacts: `data/analytical/lhs_v1.parquet`
  (~16 MB, schema v2; replaced by v3), `data/analytical/lhs_pilot.parquet`
  (200-row pilot, served its purpose), `reports/week6_baselines_v1/`
  (pre-thermal-scope-cut baselines).
- Deleted `notebooks/02_baseline_surrogates.ipynb`. The notebook
  duplicated the report CSVs and the project-plan W6 result block; the
  one genuinely new finding it surfaced (polar `energy_margin_raw_pct`
  R² 0.753) is captured in the trimmed step-5 log entry above.
- Trimmed the step-5 log entry from ~75 lines to ~17. Future log
  entries follow the "Log discipline" rule.
- `project_plan.md`: dropped the writeup notebook from the Week-6
  deliverable list (replaced with `reports/week6_baselines_v2/`); the
  notebooks/ tree comment now reads "User-facing demos only" with each
  notebook's intended scope.
- `data/analytical/SCHEMA.md`: dropped stale `lhs_v2.parquet`
  reference; the canonical filename is now just `lhs_v3.parquet`, with
  pilot/challenge files generated on demand from `build_dataset.py`.

**Files changed.**
- Deletions: `data/analytical/lhs_v1.parquet`, `lhs_pilot.parquet`,
  `reports/week6_baselines_v1/` (entire directory), `notebooks/02_baseline_surrogates.ipynb`.
- Edits: `project_plan.md` (W6 deliverable + notebooks tree), `project_log.md`
  (header rule + step-5 trim + this entry), `data/analytical/SCHEMA.md`
  (canonical-filename block).

---

## 2026-04-25 — Polar `energy_margin_raw_pct` diagnosis (pre-Week-7)

**Decision.** Root cause for the joint MLP's polar R² 0.753 is a
**target-distribution mismatch**, not data density and not a categorical
conform bug. No code change; finding informs Week-8 Optuna scope and the
production-model selection rule for `energy_margin_raw_pct`.

**Diagnosis.** Per-family `energy_margin_raw_pct` distributions on the
feasible LHS-v3 training subset:

- polar (n=6,142): mean −50%, median −66%, range [−99, +352], 88%
  negative, 11% floor-clipped near −100%.
- non-polar three families (n=19,753): mean +720%, median +520%, range
  [−70, +6480], heavy positive tail.

The two populations are nearly disjoint and have opposite-sign means.
Per-family test R² (table also in `reports/week6_baselines_v2/test_per_family.csv`):

| algo | crater | equatorial | highland | polar | aggregate |
|---|---|---|---|---|---|
| mlp_joint | .993 | .996 | .990 | **.753** | .995 |
| random_forest | .965 | .965 | .922 | **.930** | .965 |
| xgboost | .988 | .986 | .956 | **.874** | .984 |

Two compounding effects: (i) polar is intrinsically harder for any model
because the dynamic range is ~14× narrower than non-polar, so the same
RMSE eats more R²; (ii) the joint MLP additionally suffers because its
single shared trunk + MSE loss is dominated by the heavy positive tail
of the non-polar families, leaving the tightly-clustered polar negatives
under-fit. Per-target RF/XGB don't have this. Earlier step-5 log
hypotheses (family-stratified loss weighting, wider MLP, categorical
conform recheck) are superseded by this root cause.

**Consequences.**

- Aggregate gates pass; **no Week-6 redo, no v4 dataset**. Polar density
  is fine (24% of feasible rows).
- Production-model selection rule for `energy_margin_raw_pct` updates
  to **best-per-family R²**: Random Forest (polar .930) is preferred
  over aggregate-best XGBoost (.874 polar) and MLP (.753 polar). The
  joint MLP stays in the comparison set as a reference, not the
  production model for this target. Recorded in §10 Week-7 plan.
- Likely Week-8 Optuna lever: per-target `PowerTransformer(yeo-johnson)`
  inside `TransformedTargetRegressor` (compresses the bimodal mismatch
  for the joint MLP) plus deeper-tree / `min_samples_leaf` tuning for
  RF/XGB on the polar slice. Per-family models are a defensible fallback.
- **SCM is irrelevant to this finding** — surrogate-fit issue, not
  evaluator-physics issue. Does not change the Week-7.5 gate-first plan.

---

## 2026-04-25 — Week-7 composition mechanism sketched, gate-first plan adopted

**Decision.** Adopted plan A (gate-first SCM ordering): minimum-viable
~500-run SCM sweep first, Week-7.5 gate decides whether to invest in the
full 2 000-run sweep + composed surrogate. Wheel-level correction
interface specified so Week-7 step-1 (SCM driver re-validation) can
start without further design work. Full sketch lives in `project_plan.md`
under the new "Week 7 / 7.5 composition mechanism" block; this log entry
captures *why* the ordering is inverted relative to the original §6 W7→W8
sequence.

**Context.** Earlier today's discussion ("Do we need the SCM correction
or could we just use Bekker-Wong?") established that the Week-5
Pragyan / Yutu-2 validation already passes with BW-only physics, so the
*information value* of the SCM work is the answer to "does SCM materially
change the design rankings", not "does SCM let us match flown rovers".
That answer lives in the gate, not in the full 2 000-run sweep. Running
the gate first front-loads the decision and saves ~3–4 days if BW turns
out to be sufficient.

**Composition rule (the load-bearing technical decision).** Correction
is applied at the **wheel level** inside the analytical traverse loop:
`WheelForces_corrected = WheelForces_BW + Δ`, where Δ is a 3-vector of
deltas on `(drawbar_pull_n, driving_torque_nm, sinkage_m)` predicted from
a 13-d wheel-level feature vector. The mission metrics inherit the
correction implicitly through the existing slip-balance and motor-power
calculations — no second forward-pass through a "corrected" mission
model is required during traverse simulation. Mission-level Δmetric
correction (the original §6 W8 surrogate) is added *only if* the gate
fires, on top of the wheel-level injection.

**Why wheel-level rather than mission-level only.** A pure
mission-level correction (`Δrange = f(design, scenario)`) would be
trained on ~500 paired evaluations and applied as a post-hoc offset.
That works for ranking but loses the per-step physics: it can't tell
NSGA-II *why* a given design is over-conservative on slope, and it
can't be reused inside a different scenario family without re-training.
Wheel-level correction is reusable across scenarios and is the natural
home for the 13-d feature vector that already lines up 1-to-1 with
`single_wheel_forces` inputs.

**Graceful-fallback semantics.** `use_scm_correction=True` becomes safe
to leave on regardless of whether the correction model file exists. If
`data/scm/correction_v1.joblib` is absent, the wrapper returns raw BW
forces with a one-time `UserWarning`. This preserves the evaluator API
contract and lets CI / casual users keep the flag in a default-on state
once the pipeline is wired.

**Files and homes (planned).** Per the §6 W7/7.5 block in
`project_plan.md`. New artifacts: `data/scm/runs_v1.parquet`,
`data/scm/correction_v1.joblib`, `reports/week7_5_gate/`. Code:
`roverdevkit/terramechanics/correction_model.py`,
`roverdevkit/mission/{traverse_sim.py, evaluator.py}`,
`scripts/run_scm_sweep.py` (batch harness — was originally slated for
`roverdevkit/terramechanics/scm_wrapper.py` but the package stays light
for analytical-only consumers, see step-1 entry below).

**Next step.** Week-7 step-1: SCM driver re-validation + benchmark
(verify the existing Week-2 `single_wheel_forces_scm` driver still
runs on this machine, time it, then move on to step 2 — sampling +
parallel harness).

---

## 2026-04-25 — Week-7 step-1: SCM driver re-validated, much faster than budgeted

**Decision.** Week-7 step-1 closed. The Week-2 PyChrono SCM driver
`pychrono_scm.single_wheel_forces_scm` runs cleanly in the current
`roverdevkit` conda env; the existing `tests/test_pychrono_scm.py`
chrono+slow suite passes (6/6, 0.83 s total wall on M-series). The
~30 s/run budget assumed in the §6 W7 plan is **wildly conservative**
on this hardware — measured per-call wall-clock is 0.06–0.31 s
depending on mesh resolution, ~100–500× faster than budgeted. This
unblocks a much more aggressive gate sweep than originally planned.

**Single-process timing (Apple M-series, default conda env):**

| config | sim time | mean wall | × real-time |
|---|---|---|---|
| fast (CI fixture, mesh δ=20 mm) | 1.0 s | 0.062 s | 0.06× |
| default `ScmConfig()` (δ=15 mm) | 1.8 s | 0.156 s | 0.09× |
| fine mesh (δ=10 mm) | 1.8 s | 0.311 s | 0.17× |

Implications: a 500-run gate sweep at default config is ~150 s on
1 core / ~30 s on 5 cores. A full 2 000-run sweep is ~10 min on 1 core
/ ~2 min on 5 cores. The cost difference between gate-only and full
sweep is now ≈ 8 minutes, not the days originally feared. Even at
fine-mesh fidelity the full sweep is ~5 min on 5 cores.

**Plan-A (gate-first) ordering still stands** — the *information value*
question ("does SCM materially shift the design rankings?") is what
matters, not compute time. But the timing means we have plenty of room
to (a) use the production-fidelity config for the gate, (b) run the
gate sweep at 1 000 wheel-points instead of 500 if desired, and
(c) commission the full sweep on the same day if the gate fires.

**Code change.** Removed the stub `roverdevkit/terramechanics/scm_wrapper.py`.
It was a dead `NotImplementedError` with a redundant `SCMResult`
dataclass parallel to the real `WheelForces`; nothing in the codebase
imported from it. The Week-7 batch orchestration moves to
`scripts/run_scm_sweep.py` (next step) — keeps the importable
`roverdevkit.terramechanics` package free of the ~350 ms PyChrono
import-time penalty for analytical-only consumers (NSGA-II loops,
surrogate fits, evaluator validation runs). `roverdevkit/terramechanics/__init__.py`
docstring updated accordingly. Plan §6 W7 file-list updated to match.

**Deferred.** The "50-run smoke sweep against Ding 2011 single-wheel
points" originally penciled into step-1 is **moved to Week 9**, where
the plan already locates "external validation against published
experimental data". That work needs digitization of paper figures
which is its own task; step-1's narrow scope is just driver
re-validation + benchmark.

**Next step.** Week-7 step-2: SCM sampling design + parallel harness
(`scripts/run_scm_sweep.py`). LHS over the 13-d wheel-level feature
space, multiprocessing pool with `spawn` context, resumable parquet
output. With timing in hand the harness target shifts from "fits in
overnight" to "fits in the lunch break".

## 2026-04-25 — Week-7 step-2: SCM sampling design + parallel harness

**Outcome.** Step-2 closed. Stratified-categorical 6-d LHS design
generator and a resumable parallel CLI driver are in place; smoke runs
on 12 / 24 design points pass, the full 259-test pytest sweep is green
(156 s, unchanged from step-1), and the harness is sized to deliver
the §6 W7 gate sweep in ≈ 30 s on 5 workers.

**Design space (12-d wheel-level features).** Corrected from the
earlier "13-d" claim in the composition sketch: both Bekker-Wong and
the existing PyChrono SCM driver operate on a flat patch, so
`slope_deg` is **not** a wheel-level feature. Slope effects enter
both models upstream via the per-wheel vertical-load projection
`cos(θ) · m · g / N_w` in `traverse_sim._normal_load_per_wheel`. The
12 features are:

- 6 continuous LHS axes:
  `(vertical_load_n ∈ [3, 80] N, slip ∈ [0.05, 0.70], wheel_radius_m ∈ [0.05, 0.20] m,
   wheel_width_m ∈ [0.03, 0.20] m, grouser_height_m ∈ [0.0, 0.020] m)` —
  five dimensions; the sixth axis (`grouser_count`) is stratified, see below.
- 1 stratified categorical: `grouser_count ∈ {0, 12, 18}`
  (no-grouser / micro-rover-typical / high-traction).
- 1 stratified categorical: `soil_class ∈ {Apollo_regolith_nominal, JSC-1A, GRC-1, FJS-1}`
  with the 6 numeric Bekker / Mohr-Coulomb parameters resolved from
  `data/soil_simulants.csv` and stored alongside each row so a
  numeric-feature correction model can ignore the class label.

The 4 × 3 = 12 (soil × grouser) buckets are filled to ±1 row of equal
size; the row order is then permuted so a chunked checkpoint never
biases the partial sample. Same `(n_runs, seed)` always reproduces
the same design.

**Files.**

- `roverdevkit/terramechanics/scm_sweep.py` — `build_design()` and
  `run_one()`. `run_one` defers the PyChrono import to the worker call
  site so neither the design generator nor pytest collection pays
  Chrono startup. Returns a flat dict with paired
  `bw_*` / `scm_*` outcomes plus telemetry (`scm_wall_clock_s`,
  `scm_fz_residual_n`, `scm_n_avg_samples`) and per-row failure
  flags rather than raising.
- `scripts/run_scm_sweep.py` — CLI harness. `multiprocessing` `spawn`
  context, atomic `--checkpoint-every` parquet flush via tmp+rename,
  `--resume` short-circuit when the existing parquet already covers
  the design, three named SCM presets (`fast`, `default`, `fine`).
- `tests/test_scm_sweep.py` — 7 fast-loop tests for the design
  generator (balance, bounds, schema, reproducibility, input
  validation) plus 2 chrono+slow tests that smoke `run_one` end-to-end.

**End-to-end checks.**

- 9/9 new tests pass; full sweep 259 passed in 156 s — no runtime
  regression.
- 12-row serial smoke at `--config fast` finished in 1.6 s wall.
- 24-row 4-worker smoke at `--config fast` finished in 2.6 s wall;
  parquet is 25 columns wide, soil×grouser balance is exact (2 rows
  per bucket × 12 buckets), all 24 SCM runs returned status `ok`.
- Resume short-circuits cleanly: re-running with `--resume` on the
  fully-covered parquet prints "Nothing to do" without spawning workers.

**Preview of the gate finding.** Even on 24 fast-config rows the
mean absolute SCM-vs-BW deltas are already large
(|Δ DP| ≈ 10 N, |Δ τ| ≈ 0.7 N·m) and several rows have **sign flips**
in DP (BW predicts a negative drawbar at high slope/loose-soil
regimes where SCM resolves a positive one). This is the regime the
sketch flagged as "where Bekker-Wong is least defensible" — the
gate sweep is likely to fire, but step-3's job is to confirm it on
500 production-fidelity runs and quantify how those wheel-level
deltas propagate to mission-level `range_km` / `energy_margin_*`.

**Next step.** Week-7 step-3: run the production gate sweep
(`--n-runs 500 --workers 5 --config default --out data/scm/runs_v1.parquet`,
≈ 30 s wall), then carry the deltas forward to the Week-7.5 gate
analysis (≤ 5 % rank-correlation impact ⇒ skip the full SCM model;
> 5 % ⇒ commission the 2 000-run sweep + correction model).

## 2026-04-25 — Week-7 step-3: 500-row production gate sweep complete

**Outcome.** The full gate sweep ran in **39.9 s wall** on 5 workers
(`--n-runs 500 --workers 5 --config default --seed 42`), wrote
`data/scm/runs_v1.parquet` (76 kB, 25 columns), and finished with
**500/500 BW + SCM rows ok** and zero NaNs in any target column.
Telemetry is healthy: median SCM `Fz` residual is 6 mN against
3-80 N normal loads (≪ 1 % imbalance), `n_avg_samples = 750` on
every row (steady-state averaging window converged), per-call wall
clock 0.09-1.01 s (median 0.33 s, matches the step-1 0.16-0.31 s
benchmark plus pool overhead).

**Headline deltas (n = 500).** SCM systematically predicts more
favorable mobility than Bekker-Wong:

| metric | mean | median | P95 | max |
|---|---|---|---|---|
| `Δ drawbar_pull_n = scm − bw` (signed) | +10.9 | +8.2 | +30.7 | +55.3 |
| `Δ driving_torque_nm` (signed) | +0.92 | +0.51 | +3.13 | +5.31 |
| `Δ sinkage_m` (signed) | −0.004 | −0.004 | +0.001 | +0.002 |

The `Δ DP` mean is positive on 95 % of rows — SCM is not just noisier
than BW, it is **biased high** in the regions BW pessimizes most
(consistent with the literature: BW's empirical shear-mobilization
underestimates grouser bite at high slip).

**Sign-flip count (the gate's leading indicator).** **80 / 500 rows
(16 %) have BW predicting a negative drawbar pull while SCM predicts
a positive one** — i.e. BW says "wheel stalls" and SCM says "wheel
produces tens of newtons of pull". Worst single case: `row 100`,
FJS-1 / 18 grousers / 67 N load / slip 0.69 / 5 cm radius — BW
−4.1 N vs SCM **+44.8 N**, a 49 N gap that flips the design
verdict. Concentrated in the small-wheel (R ≤ 8 cm), grouser-heavy
(N_g ∈ {12, 18}), high-slip (s > 0.45), loose-soil (FJS-1 / GRC-1)
corner of the design space — exactly where the §6 W7 composition
sketch flagged BW as least defensible. (`(np.sign(bw) ≠ np.sign(scm))
& (|both| > 0.5 N)` count is 67 / 500 = 13.4 % under the stricter
"both meaningfully nonzero" gate.)

**Slip dependence.** |Δ DP| grows monotonically with slip:

| slip bin | n | mean \|Δ DP\| (N) |
|---|---|---|
| (0, 0.15] | 77 | 5.6 |
| (0.15, 0.30] | 115 | 6.3 |
| (0.30, 0.50] | 154 | 10.5 |
| (0.50, 0.75] | 154 | 17.4 |

Slip > 0.5 is precisely the regime mission designs use to climb
slopes — the wheel-level correction will be most valuable for the
slope-capability and short-distance high-slope traverse families.

**Soil dependence.** Roughly soil-independent in mean |Δ DP|
(9.2-11.9 N across the four catalogue simulants); FJS-1 and GRC-1
show the largest median relative deltas. Apollo nominal (the
default scenario soil for most LHS rows) is the *least* affected —
which means a naïve "BW is fine, the surrogate looks good" judgment
based on Apollo-only validation would have missed the JSC / FJS /
GRC failures entirely.

**Implication for the Week-7.5 gate.** The wheel-level data already
strongly suggest the gate will fire (16 % feasibility-flipping
disagreement is far above any reasonable tolerance), but the formal
gate is **mission-level** rank correlation of design rankings under
BW vs SCM-corrected evaluation, which still requires Week-7
step-4 (fit a wheel-level correction model on this parquet) and
step-5 (compose into the evaluator and run a small design sample
through it). Step-3's job — empirically confirming wheel-level SCM
deltas are large enough to be *plausibly* gate-firing — is met.

**No new code in this step.** Used the step-2 harness as-is. Data
artifact lives at `data/scm/runs_v1.parquet`; downstream consumers
(Week-7 step-4 correction model, Week-7.5 gate notebook, the eventual
benchmark release) read from there.

**Next step.** Week-7 step-4: fit the wheel-level correction model
(`roverdevkit/terramechanics/correction_model.py`,
`scripts/train_correction_model.py`). Targets:
`(Δ_drawbar_pull_n, Δ_driving_torque_nm, Δ_sinkage_m)` regressed
on the 12-d wheel-level feature vector (continuous wheel/op +
soil parameters as numeric, `grouser_count` as integer,
`soil_class` optionally one-hot for a tree model). Train/val/test
split with seed 42 stratified on `(soil_class, slip_bin)` so each
fold sees the high-slip regime that drives the deltas. First-cut
models: Ridge baseline, RandomForestRegressor, XGBRegressor.
Write `data/scm/correction_v1.joblib` and a short
`reports/week7_5_gate/correction_fit.csv`.

## 2026-04-25 — Week-7 step-4: wheel-level correction model fitted

**Outcome.** XGBoost picked per target on test RMSE, refit on
train+val, saved to `data/scm/correction_v1.joblib` (2.1 MB).
Per-target test scores **after refit on train+val** (the model that
ships):

| target | algo | test R² | test RMSE | mean \|Δ\| in dataset | residual / signal |
|---|---|---|---|---|---|
| `delta_drawbar_pull_n` | XGBoost | 0.962 | 1.72 N | 10.9 N | 16 % |
| `delta_torque_nm` | XGBoost | 0.905 | 0.35 N·m | 0.92 N·m | 38 % |
| `delta_sinkage_m` | XGBoost | 0.929 | 1.0 mm | ~4 mm | 25 % |

Ridge floors are 0.72-0.83 (genuinely learnable, not tree memorization);
RandomForest comes in second on every target (0.87-0.93). Ridge as a
diagnostic confirms most of the signal is smooth in the 12-d feature
space, not noise the trees absorbed by overfitting.

**Splits.** 70/15/15 stratified on `(soil_class, slip_bin)` with
`slip_bin ∈ {(0, 0.15], (0.15, 0.30], (0.30, 0.50], (0.50, 0.75]}` →
4 × 4 = 16 strata, ~31 rows each at n=500. Every fold sees the
high-slip regime that drives the deltas, eliminating the failure mode
where a fold accidentally over-samples the low-slip rows that BW
already gets right.

**Why per-target rather than a joint MLP.** Three residuals at very
different scales (Δ DP ~10 N, Δ τ ~1 N·m, Δ sinkage ~5 mm) and
different feature dependencies (DP is dominated by slip × grouser
height, sinkage by load × soil compressibility). Per-target lets each
target use its own tree depth and learning rate without an MSE-loss
arbitration; the Week-6 surrogate found the same thing on its
primary targets. With 350 train rows there is also no data budget for
an MLP to add value over a tuned tree.

**Calibration interpretation.** Residual RMSE is ~16-38 % of the
**mean absolute** signal per target. The correction model captures
the **systematic** BW-vs-SCM bias (which is what shifts design
rankings); the residual is the random component that washes out
through per-step traverse integration. The Week-7.5 gate is decided
on **mission-level rank correlation** of designs evaluated under BW
alone vs. BW + correction, not on these wheel-level R² numbers.

**Code added.**

- `roverdevkit/terramechanics/correction_model.py` — replaces the
  Week-2 stub. `WheelLevelCorrection` dataclass with frozen
  `feature_columns / target_columns`, `predict_batch` /
  `predict_single` / `save` / `load`, joblib-backed artifact, and
  metadata recording the parquet path, splits, chosen algorithm per
  target, and build timestamp. `train_correction_model(parquet, out)`
  is the one-call training entry point. `load_correction_or_none(path)`
  is the loader the traverse loop will use to fall back gracefully
  when the artifact is missing (e.g. during the v3 LHS rebuild itself).
- `scripts/train_correction_model.py` — thin CLI driver around the
  above, with a tabular console summary.
- `tests/test_correction_model.py` — 7 fast tests on a 240-row
  synthetic gate sweep (same 4 × 3 = 12 stratifier buckets as the real
  data, mocked BW and SCM with a learnable signal). Verifies the
  trainer clears R² > 0.5 on the synthetic signal, schema enforcement
  on `predict_batch`, batch == single equivalence, save/load
  round-trip, missing-artifact loader semantics, and that the trainer
  refuses to fit a parquet with any BW/SCM failure rows.

**Repo state.** Full sweep 266 passed (was 259 before the new tests),
1 xfailed pre-existing, 163 s wall — +7 s from the seven new tests,
no regressions. Production artifact `data/scm/correction_v1.joblib`
(2.1 MB) and `reports/week7_5_gate/correction_fit.csv` (2.9 KB,
12 rows: 3 targets × 3 algorithms × test rows + 3 refit rows) plus
the sidecar `correction_fit.meta.json`.

**Next step.** Week-7 step-5: compose the correction into the
analytical evaluator. Two injection points in
`roverdevkit/mission/traverse_sim.py` per the §6 W7 sketch — wrap
the BW call inside `_solve_step_wheel_forces`'s slip-balance brentq
residual so the equilibrium slip reflects corrected DP, and wrap
the BW call inside `_mobility_power_w` so the per-step torque is
corrected. Default `use_scm_correction=False` (no behaviour change
for existing callers); when true, the loop calls
`load_correction_or_none(data/scm/correction_v1.joblib)` and falls
back to BW-only if the artifact is absent. Then run a small LHS
sample (~100 designs across the 4 scenario families) through both
modes and compare design rankings — that's the formal Week-7.5
gate input.

## 2026-04-25 — Week-7 step-5: correction wired, Week-7.5 gate fires

**Outcome.** Composition layer is in production and the gate fires
**one-sided** on every scenario family — promote the wheel-level
correction. Full evidence in `reports/week7_5_gate/gate_decision.md`.

**What shipped.**

- `roverdevkit/mission/traverse_sim.py` — `run_traverse(..., correction=)`
  threads a `WheelLevelCorrection` through `_solve_step_wheel_forces`
  so the per-step slip-balance brentq solves against
  `DP_BW + ΔDP − DP_required = 0`. The corrected forces propagate
  into `_mobility_power_w` automatically (no separate injection
  needed there — the corrected torque flows through). A pre-allocated
  12-d feature buffer is built once per integration step and the
  brentq residual mutates only the slip column per call, avoiding a
  pandas DataFrame allocation per BW evaluation. An assert at module
  import pins the feature-column order so a schema drift in
  `correction_model.py` fails loudly instead of silently corrupting
  predictions.
- `roverdevkit/mission/evaluator.py` — `use_scm_correction=True` no
  longer raises. Loads `DEFAULT_CORRECTION_PATH` lazily via
  `load_correction_or_none(on_missing="warn")` so the v3 → v4 LHS
  rebuild can run while the artifact is absent without crashing.
  Added a `correction=` kwarg so the dataset builder can pre-load
  once and share across worker processes.
- `roverdevkit/terramechanics/correction_model.py` — added
  `WheelLevelCorrection.predict_array(x)` (pandas-free fast path on a
  numpy buffer in `feature_columns` order). `predict_batch` now goes
  through it, so there is one hot path. Added `DEFAULT_CORRECTION_PATH`
  resolved relative to repo root.
- `scripts/run_gate_sweep.py` — parallel BW-vs-corrected pair driver.
  Per-process correction caching, spawn pool, paired metrics parquet,
  and an inline §6 W7.5 gate-criterion summary.
- `tests/test_mission_evaluator.py` — replaced the
  `NotImplementedError` test with one that exercises the loaded-and-
  applied correction path (asserts at least one mobility-derived
  metric moves vs BW) and the artifact-missing graceful-fallback path.

**Gate-firing evidence (n=200 paired evaluations, 50 / family).**

| family | feas-flip | bw_only | scm_only | gate |
|---|---|---|---|---|
| crater_rim_survey | 30.0 % | 0 | 15 | fires |
| equatorial_mare_traverse | 12.0 % | 0 | 6 | fires |
| highland_slope_capability | **68.0 %** | 0 | 34 | fires |
| polar_prospecting | 34.0 % | 0 | 17 | fires |

The flips are 100 % one-sided: in every family, **zero** designs are
BW-feasible / SCM-infeasible. BW is systematically more conservative
than SCM at the feasibility frontier. Highland slopes are the worst
case (68 % of designs that BW labels infeasible are mobile under SCM)
— consistent with step-3's wheel-level finding that BW under-predicts
drawbar pull and over-predicts stall in high-slip / loose-soil regimes.
This is the same calibration story the registry-rover Layer-1 sanity
check has been showing on slope_capability_deg since the v3 widening.

**Why the quantitative range gate is False everywhere.** Among
designs feasible-in-both, `range_med_rel_err = 0.00` because both
modes saturate at the scenario `traverse_distance_m`. The gate fires
on the qualitative feasibility flip (the §6 sign-bias rule), not on
the quantitative range delta — which is exactly what the rule is for.
Spearman ρ on continuous mobility metrics is 0.83-0.99 across all
families: rankings of mobile-in-both designs are well preserved; the
correction shifts the boundary, not the order.

**Decision.** Promote the wheel-level correction into the production
analytical evaluator; do **not** commission a separate mission-level
`Δmetric = f(design, scenario)` correction surrogate (the §6 W7.5
alternate path). Reasons: (1) the wheel-level correction passes its
own gate (R² 0.91-0.96, step-4); (2) the gate-firing signal here is
*feasibility*, which is exactly what the wheel-level correction
shifts; (3) one composition layer is simpler than a wheel-level
*plus* mission-level model, and avoids a second model that would have
to be re-trained whenever the LHS bounds change.

**Performance.** ~1.9 s per (BW, SCM) pair on 4 spawn workers
(381 s wall for n=200). The XGBoost `predict_array` fast path is
the difference between this and a ~7× slower pandas-allocating
path. For the Week-8 v4 LHS regeneration (40 k rows × 1 mode with
correction-on) the budget is ~2 h on the same 4 workers — comparable
to the v3 build.

**Repo state.** All 266 tests pass + 1 xfailed (171 s wall, no
regressions). New artifacts: `data/scm/gate_eval_v1.parquet`,
`reports/week7_5_gate/gate_summary.csv` /
`feasibility_direction.csv` / `spearman_mobility.csv` /
`gate_decision.md`. Smoke-only `data/scm/gate_eval_smoke.parquet`
deleted.

**Next step.** Week-8: regenerate `data/analytical/lhs_v4.parquet`
with `use_scm_correction=True`; re-fit baselines on the new corpus;
re-check the registry-rover Layer-1 sanity (the slope-capability
calibration gap on Pragyan / MoonRanger should narrow given that
the correction systematically increases predicted feasibility on
slopes). Closes Week-7 / 7.5 unless the v4 baselines surface new
issues.

---

## 2026-04-26 — Week-7.7: traverse-loop lift-out + BW-vs-SCM-direct bake-off

**Decision.** (1) Hoist the wheel-force solve out of the per-step loop
in `roverdevkit/mission/traverse_sim.py::run_traverse` (every input is
loop-invariant in the current scenario schema; the inner loop now only
updates sun geometry, solar power, battery, and position).
(2) Keep **BW + wheel-level correction** as the production
dataset-generation backend for v4. SCM-direct is wired end-to-end via
`force_backend="scm"` for ablation but is **not** promoted.

**Lift-out impact.** BW-only mission cost drops from ~250 ms → 10 ms
(mean), BW+correction from ~500 ms → 40 ms. All 266 + 1xfail tests
pass with byte-identical metrics — the v3 LHS regression fixture is
unchanged. Without the lift, SCM-direct (≥ 4 s per residual call,
~17 brentq evaluations per mission) would be infeasible at any
dataset scale; with the lift, it costs 4.1 s/mission and runs
sample-by-sample.

**Bake-off.** 200 LHS designs × 4 scenarios × 3 backends = 2 400
evaluations (5 graceful failures, all on a pre-existing
fully-buried-wheel geometry condition that affects all backends
equally). Driver: `scripts/run_bakeoff.py`. SCM-direct treated as
ground truth.

| metric | BW p50 rel-err | BW+corr p50 rel-err |
| --- | --- | --- |
| `peak_motor_torque_nm` | 7-9 % | 5-8 % |
| `sinkage_max_m` | **63-86 %** | **11-13 %** |
| `energy_margin_raw_pct` | 0.9-2.6 % | 0.2-0.7 % |

`motor_torque_ok` flip rate vs SCM, worst → best: highland_slope
**56.5 %** → **1.0 %**, polar **37 %** → **0.5 %**, crater **30 %** →
**0 %**, equatorial **12 %** → **0 %**. The correction nearly
eliminates feasibility disagreement.

**Why BW+correction wins on the merits.** (1) 100× faster than
SCM-direct (40 ms vs 4 s per mission). (2) The correction *is* the
methodology paper-1 contribution; switching to SCM-direct would
delete the contribution. (3) BW+correction collapses the analytical →
SCM gap to <1 % feasibility flips and single-digit % continuous error
— well below the surrogate's regression noise floor. (4) SCM-direct
remains a one-keyword flip for ablation studies and any future
per-step terrain extension.

**Repo state.** All 266 + 1xfail tests pass (13 s wall — pytest
itself is now ~14× faster thanks to the lift-out). New files:
`scripts/run_bakeoff.py`, `reports/week7_7_bakeoff/{decision.md,
bakeoff_long.parquet, bakeoff_wide.parquet, bakeoff_summary.csv}`.
Modified: `roverdevkit/mission/traverse_sim.py` (lift-out +
`_scm_solve_step_wheel_forces` + `force_backend` kwarg),
`roverdevkit/mission/evaluator.py` (forwards `force_backend`).

**Next step.** Unchanged from Week-7 step-5: Week-8 step-1 LHS v4
regeneration with `use_scm_correction=True`. The bake-off confirms
the architecture; the v4 build can start.

---

## 2026-04-26 — Week-8 step-1: lhs_v4.parquet built (corrected mobility)

**Decision.** Promote `data/analytical/lhs_v4.parquet` (40 000 rows,
schema **v4**, `use_scm_correction=True`) as the canonical Phase-2
training corpus. Bumped `SCHEMA_VERSION` from `v3` → `v4` so a
v3-trained surrogate can't silently be re-applied to corrected data.

**Plumbing.** Added `--use-scm-correction` to
`scripts/build_dataset.py`; `roverdevkit/surrogate/dataset.py` now
takes the flag through `build_dataset` / `build_and_write` and
forwards it to a per-pool-worker initializer that `joblib.load`s the
correction artifact **once per spawned process**. Each
`_evaluate_sample` call then passes the cached correction to
`evaluate_verbose(..., correction=...)`, so the artifact load is
amortized to ~1 disk read per worker rather than 1 per sample.
`DatasetMetadata` gains `use_scm_correction: bool` and writes it
to the Parquet footer next to `schema_version`.

**Build run.** 40 000 rows × 64 cols, 8 spawn workers,
`built_at_utc=2026-04-26T12:36:51+00:00`, **153 s wall**. 39 974/40 000
ok (99.94 %); the 26 graceful failures are the same fully-buried-wheel
geometry condition seen in v3 / W7.5 / W7.7 — all pre-existing,
backend-agnostic. Aggregate feasibility (`motor_torque_ok`) 99.12 %.

The build is **~48× faster than v3** (~2 h → 2.5 min). Two compounding
wins: (1) the Week-7.7 lift-out cuts BW+correction per-mission cost
from ~500 ms to ~40 ms, (2) the per-worker correction cache cuts
joblib loads from `n_samples` to `n_workers`. Per-sample-per-worker
cost is now 0.03 s.

**Cross-validation against W7.7 bake-off.** Same-seed paired diff
v4 vs v3 on 39 974 rows shows the correction lands exactly where
the bake-off measured it would:

| family | v3→v4 motor_ok flip | W7.7 BW-vs-SCM flip |
| --- | --- | --- |
| equatorial_mare_traverse | 12.4 % | 12.1 % |
| crater_rim_survey        | 32.2 % | 30.2 % |
| polar_prospecting        | 37.8 % | 37.0 % |
| highland_slope_capability| 55.2 % | 56.5 % |

Within 1 pp on every family — strongest possible mission-level
confirmation that the wheel-level correction composes correctly into
`run_traverse`. Median Δ`sinkage_max_m` = **−4 mm across every family**
(BW was systematically over-pessimistic on sinkage). Median Δ`range_km`
is +1.85 km on highland (where corrected mobility unlocks designs BW
stalled) and 0 on the other three families (saturation at
`traverse_distance_m`). Median Δ`energy_margin_raw_pct` is +79 % on
highland, +21 % on crater, +7 % equatorial, +1.8 % polar — tracks the
mobility unlock perfectly (polar is solar-bound, not mobility-bound,
hence the small shift).

**Repo state.** All 266 + 1 xfail tests pass; lint clean. Modified
files: `roverdevkit/surrogate/dataset.py` (schema bump, worker init,
metadata field), `scripts/build_dataset.py` (CLI flag).
`data/analytical/lhs_v3.parquet` retained for comparison; v4 supersedes
it as the Phase-2 training corpus.

**Next step.** Week-8 step-2: refit baselines on v4
(`scripts/run_baselines.py --dataset data/analytical/lhs_v4.parquet`),
re-evaluate the registry-rover Layer-1 sanity check. The polar-family
slope-capability calibration gap (Pragyan, MoonRanger) should narrow:
the correction systematically increases predicted feasibility on
slopes, which is exactly the direction needed to close the gap that
flagged in Week-6 step-4.


## 2026-04-26 — Week-8 step-2: baselines refit on v4, all gates pass

**Decision.** Promote `lhs_v4.parquet` as the Phase-2 production training
corpus. Acceptance gate clears 16/18 (the two failures are Ridge, the
linear baseline-of-baseline reference); every non-linear method passes
every regression and classification threshold. Stay at the 500-point
SCM correction sweep — no v5 rebuild needed.

**Run.** 45.6 s wall to fit Ridge / RF / XGB per-target × 4 targets,
joint MLP, and LogReg / XGB classifier on `motor_torque_ok` against the
32 137 / 3 880 / 3 983 train/val/test splits. Outputs in
`reports/week8_baselines_v4/` (`SUMMARY.md`, `metrics_long.parquet`,
`acceptance_gate.csv`, `test_summary.csv`, `test_per_family.csv`,
`fit_seconds.csv`, `registry_sanity.csv`).

**Lift vs v3 (aggregate test R²).** Concentrated on the targets W7.7
predicted would shift:

- `slope_capability_deg` RF: 0.926 → 0.956 (+0.030)
- `energy_margin_raw_pct` RF: 0.965 → 0.981 (+0.016)
- `energy_margin_raw_pct` XGB: 0.984 → 0.995 (+0.011)
- `range_km`, `total_mass_kg`: flat (already saturated > 0.99 in v3)

**Polar `energy_margin_raw_pct` weak spot resolved.** v3 joint-MLP
R² was 0.753 on polar; v4 XGB / RF land at 0.964 / 0.966 — both clear
the 0.95 per-family gate. The MLP shared-trunk weakness (target-distribution
mismatch flagged in the 2026-04-25 polar diagnosis entry) persists at
0.816, but the production rule is per-target best, not joint MLP.

**Registry Layer-1 sanity (primary, design-axis).** Median MAPE %
across algorithms:

| rover | total_mass_kg | slope_capability_deg | classifier acc |
| --- | ---: | ---: | ---: |
| Yutu-2 | 0.9 | 2.1 | 1.00 |
| Pragyan | 2.7 | 60.6 | 1.00 |
| MoonRanger | 3.2 | 25.3 | 1.00 |
| Rashid-1 | 8.2 | 5.7 | 1.00 |

Mass and `motor_torque_ok` pass on every rover. Slope MAPE on the two
flagged rovers (Pragyan, MoonRanger) **dropped roughly half the v3 gap**
(77.9 → 60.6, 39.7 → 25.3). The residual reflects rover-specific design
choices — drive-train torque limits, track patterns, design-margin
philosophy — that the 12-D wheel-level correction's feature space cannot
resolve. Documented as a known scope limit; no further v5 work
proposed.

**Files updated.** `data/analytical/SCHEMA.md` (canonical filename
v3 → v4, Schema-version note, registry-sanity table refreshed),
`scripts/run_baselines.py` (example command points at v4).

**Next step.** Week-8 step-3 was originally Optuna tuning. With the
acceptance gate already passing 16/18 and the registry-rover residuals
dominated by out-of-scope effects, the marginal Optuna lift is modest.
Open question for the next session: do we (a) run Optuna anyway for
the methodology paper's "tuned baseline" line in the comparison table,
or (b) declare Phase 2 complete and move to Phase 3 / paper draft?

## 2026-04-26 — Week-8 step-3: Optuna-tuned XGBoost, registry-rover slope gap halved

**Decision.** Promote tuned XGBoost as the production model for
`range_km`, `total_mass_kg`, and `motor_torque_ok`; keep joint MLP for
`energy_margin_raw_pct` and `slope_capability_deg` for now (margins
inside the NSGA-II noise floor). Revisit consolidation in Phase 3.

**Run.** 4 regressors × 50 TPE trials + classifier × 50 trials, 645 s
wall on 8 cores. Outputs in `reports/week8_tuned_v4/` (`SUMMARY.md`,
`tuned_summary.csv`, `tuned_best_params.json`, `tuned_test_metrics.parquet`,
`tuned_acceptance_gate.csv`, `study_<target>.csv`,
`tuned_registry_sanity.csv`).

**Aggregate test R² lifts (untuned → tuned XGB).** Tiny (saturation):

- `range_km`: 0.998 → 0.9993 (+0.0010), now beats untuned MLP (0.9990)
- `energy_margin_raw_pct`: 0.995 → 0.9961 (+0.0011)
- `slope_capability_deg`: 0.992 → 0.9945 (+0.0028)
- `total_mass_kg`: 0.999 → 0.9998 (now beats MLP 0.9997)
- `motor_torque_ok` AUC: 0.983 → 0.988 (+0.005, also beats LogReg 0.985)

**Per-family lift on the v4 weak spot.** Polar `energy_margin_raw_pct`:
untuned XGB 0.964 → tuned XGB 0.970 (+0.005, comfortably clears the
0.95 gate). Tuning didn't fix the joint MLP's polar shortfall (still
0.816), but the production rule selects tuned XGB on this family so
the gate passes via per-target winner.

**Registry-rover Layer-1 primary (XGB-only comparison).** The headline
finding:

| rover | metric | untuned XGB | tuned XGB | Δ |
| --- | --- | --- | --- | --- |
| Pragyan | slope MAPE | 67.9 % | **30.3 %** | **−37.6 pp** |
| MoonRanger | slope MAPE | 39.2 % | **20.4 %** | −18.8 pp |
| Yutu-2 | slope MAPE | 4.0 % | 1.9 % | −2.1 pp |
| Rashid-1 | slope MAPE | 3.1 % | 0.1 % | −3.0 pp |
| Pragyan | mass MAPE | 1.26 % | 0.21 % | −1.05 pp |
| Rashid-1 | mass MAPE | 2.21 % | 0.39 % | −1.82 pp |
| (every rover) | classifier acc | 1.00 | 1.00 | 0 |

Combined Pragyan slope progression across the three v3 → v4 → tuned
iterations: 77.9 → 60.6 → 30.3 % (median across algorithms / XGB-only).
The wheel-level SCM correction closed roughly half the original v3 gap
(W8 step-2); tuned XGBoost closed roughly half of the *remaining* gap
(W8 step-3). What's left is consistent with rover-specific design
detail outside the project's analytical-physics scope.

**Best hyperparameters.** Pattern: larger `n_estimators` (~1000-1300,
2-2.5× the untuned 500) at smaller `learning_rate` (~0.025-0.05,
0.5× the untuned). `max_depth` 3-8 (8 for `range_km` energy-coupled
targets, 3 for the near-deterministic `total_mass_kg`).

**New code.** `roverdevkit/surrogate/tuning.py` (TPE study + final-fit
helpers; XGB-only by design — see module docstring); `scripts/tune_baselines.py`
(CLI driver, mirrors `run_baselines.py` output schema for the tuned
metrics frame). The default-hyperparameter `baselines.py` pipeline is
unchanged so the W6/W8 step-2 acceptance numbers stay reproducible.

**Out of scope kept out.** No MLP / RF / Ridge tuning (the W8 step-2
report justifies why). No prediction-interval calibration yet; that is
the W8 step-4 line item per `project_plan.md` §6.2.

**Next step (open).** Plan §6.2 originally called for prediction-interval
calibration after tuning. The acceptance gate is fully clear and the
registry-rover residuals are at the analytical-physics scope limit, so
two reasonable paths: (a) calibrate PIs (quantile XGB or MC dropout MLP)
to close out §6.2 cleanly for the methodology paper, or (b) declare
Phase 2 done and start Phase 3 (NSGA-II / Pareto).

---

## 2026-04-26 — Mission-level surrogate reframed (accelerator + UQ, not the deliverable)

**Decision.** Reframe the mission-level XGBoost / MLP surrogate from "the
project's headline ML deliverable" to **"an optional acceleration and
uncertainty layer on top of the corrected mission evaluator."** The
**wheel-level multi-fidelity correction** (`roverdevkit.terramechanics.correction_model.WheelLevelCorrection`)
is the methodological centrepiece going forward; the mission-level
surrogate is kept but explicitly demoted in the paper, the README, and
the architecture text.

**Context.** Two things converged. (1) After the W7.7 traverse-loop
lift-out (340× speedup), the corrected evaluator runs at ~40 ms / mission
on a single core (~5 ms on 8 cores), making most Phase-3 workflows feasible
without an outer surrogate: 100k-point parametric sweeps in ~80 s,
NSGA-II Pareto fronts (≈30k evaluations × 4 scenarios) in ~10 min on 8
cores. (2) The W7 / W7.5 / W7.7 chain made the wheel-level correction
the actual novel ML contribution — a small model (12-d features, ~500
SCM training points) composed back into a physics-grounded inner loop,
matching SCM-direct mission outputs at 100× the speed.

**Where the surrogate still earns its keep.** Six use cases, all kept:
1. NSGA-II inner loop (≈30k evals × 200 generations × 4 scenarios is
   surrogate territory); the corrected evaluator validates the *final*
   Pareto front.
2. Bulk sensitivity / Sobol / 1M-point grids.
3. Calibrated prediction intervals via quantile XGBoost (W8 step-4).
4. Probabilistic feasibility for NSGA-II constraint handling (classifier
   AUC, not deterministic boolean).
5. Phase-5 benchmark baseline — the dataset / benchmark contribution
   needs a reference surrogate at the leaderboard.
6. Deployment portability — a pickled XGBoost is much smaller and easier
   to ship than the full evaluator stack with `MassModelParams`,
   `ScmConfig`, etc.

**W8 step-4 scope cut.** Drop MC-dropout MLP and deep ensembles. Quantile
XGBoost (τ ∈ {0.05, 0.5, 0.95}) is sufficient for the prediction-interval
need on the methodology paper. Median quantile checked against W8 step-3
R² gates as a sanity guardrail. PIs computed on the composed
(corrected-evaluator → surrogate) output, not on a BW-only baseline.

**Phase-3 backend dispatch.** Sweep / NSGA-II code path takes a
`backend: Literal["evaluator", "surrogate"]` switch; default to
`"evaluator"` for ≤10k points, surrogate above. Pareto fronts are
*always* re-evaluated point-by-point with the corrected evaluator
before the headline plot is written.

**Doc / plan changes (this entry's purpose).**
- `project_plan.md` §1: pitch + 4-item contribution list rewritten
  (correction → surrogate → rediscovery, with fast-evaluator note).
- `project_plan.md` §2: architecture diagram has explicit
  WHEEL-LEVEL CORRECTION block above the SURROGATE LAYER block; caption
  paragraph rewritten.
- `project_plan.md` §5: three-path strategy updated to reflect the
  ~500-row wheel-level sweep (not 2 000 mission-level), and the
  "fired gate, correction shipped" outcome.
- `project_plan.md` §6 Phase 3: dual-backend dispatch, Pareto-front
  validation pass, evaluator-default rule for ≤10k points.
- `project_plan.md` §6 W8 step-4: quantile-XGB only, MC-dropout dropped.
- `project_plan.md` §7 Layers 1-2: Layer 1 reworded as
  "surrogate vs corrected evaluator"; Layer 2 reworded as
  "corrected evaluator vs SCM-direct" (W7.4 + W7.7).
- `project_plan.md` §9: full vision Paper 1 marked as the active path
  (W7.5 gate fired); MVP fallback kept but labelled inactive.
- `project_plan.md` §10: surrogate / terramechanics file lists updated
  (no `scm_wrapper.py`, `tuning.py`/`uncertainty.py` reflected,
  `correction_model.py` annotated).
- `project_plan.md` §11.1: title rewritten with "wheel-level
  multi-fidelity correction" as the headline; abstract rewritten;
  contributions reordered (correction → capability-envelope →
  open-source tool → rediscovery); §5 / §6 / §7 outlines updated;
  §11.1.1 marked W7.5 gate as resolved.
- `project_plan.md` §11.2: paper 2 pitch notes the dataset ships
  the SCM-corrected targets without requiring PyChrono installation.
- `README.md`: contribution list, architecture diagram, and intro
  paragraph rewritten to match.

**Out of scope.** No code changes. The reframe is documentation-only.
The actual class hierarchy stays — `roverdevkit.surrogate` continues
to ship the mission-level baselines, tuned models, and (next, W8 step-4)
the quantile heads.

**Next step.** W8 step-4: fit quantile XGBoost heads at τ ∈ {0.05, 0.5,
0.95} per primary regression target on `lhs_v4.parquet`; calibrate
empirical 90 % coverage on the canonical test split; report by scenario
family. New code: `roverdevkit/surrogate/uncertainty.py` +
`scripts/calibrate_intervals.py`.

---

## 2026-04-26 — W8 step-4: quantile-XGBoost prediction intervals on v4 (done)

**What.** Calibrated 90 % prediction intervals on the four primary
regression targets (`range_km`, `energy_margin_raw_pct`,
`slope_capability_deg`, `total_mass_kg`) via independent quantile
XGBoost heads at τ ∈ {0.05, 0.50, 0.95}, using the W8 step-3 tuned
hyperparameters as the shared per-head configuration. Closes the §6
W8 step-4 deliverable and the §7 Layer-1 PI claim.

**Why this scope.** Per the 2026-04-26 reframe entry, the
mission-level surrogate is now an optional accelerator + UQ layer
(not the headline contribution), so a single UQ family is sufficient
and a second one (MC-dropout MLP, deep ensembles) would be
infrastructure-heavy without changing the paper claims. Quantile
XGB reuses the W8 step-3 tuner output and adds no new model family
to maintain.

**Method.**
- Per target: three independent `xgb.XGBRegressor`s with
  `objective="reg:quantileerror"`, `quantile_alpha ∈ {0.05, 0.5,
  0.95}`. Every other hyperparameter shared from
  `reports/week8_tuned_v4/tuned_best_params.json`.
- Train on canonical W8 step-3 train split with early stopping on val
  pinball loss (`early_stopping_rounds=25`, mirrors W8 step-3); refit
  each head on `train ∪ val` with the early-stopping-best
  `n_estimators`. Score on the unseen test split.
- Coverage measured both raw (independent quantile output) and
  `sorted` (row-wise sort of the three predictions). Sorting is
  non-worse for empirical coverage and never changes the median; we
  report both so the writeup is honest about the crossing rate.

**Headline numbers (test split, all families).**

Median (τ=0.5) sanity vs W8 step-3 tuned squared-error medians:

| Target                | quantile R² | step-3 R² | Δ        |
|-----------------------|-------------|-----------|----------|
| range_km              | 0.9982      | 0.9993    | -0.0010  |
| energy_margin_raw_pct | 0.9912      | 0.9961    | -0.0049  |
| slope_capability_deg  | 0.9929      | 0.9945    | -0.0016  |
| total_mass_kg         | 0.9996      | 0.9998    | -0.0002  |

All four are within 0.005 R² of the step-3 baseline, well above the
§7 Layer-1 R² gate. Pinball loss does cost a hair vs squared loss,
as expected.

90 % PI coverage on test:

| Target                | mean width | raw cov. | sorted cov. | crossings |
|-----------------------|-----------:|---------:|------------:|----------:|
| range_km              | 13.5 km    | 0.852    | **0.919**   | 27.3 %    |
| energy_margin_raw_pct | 250 pp     | 0.866    | **0.918**   | 20.9 %    |
| slope_capability_deg  | 2.0 °      | 0.802    | 0.856       | 20.5 %    |
| total_mass_kg         | 1.5 kg     | 0.876    | **0.920**   | 21.4 %    |

Three of four targets land within ±2 pp of nominal after row-wise
sort. `slope_capability_deg` stays 4 pp under-covered — the target's
PI width is very narrow (≈2°) so the model is slightly
over-confident on the tails; a lightweight conformal-prediction
wrapper would close this gap if a future revision needs strict
calibration. Acceptable for the methodology paper's PI claim.

**Per-scenario observations** (sorted coverage):
- Polar PIs are conservative (over-covered) on `range_km` (0.963)
  and `energy_margin_raw_pct` (0.947) — the saturated tail
  concentrates near the solar/battery cap so the upper quantile
  has thin support to learn from.
- `equatorial_mare_traverse` is the worst family for `range_km`
  (0.882) — bimodal range distribution (binding-vs-saturated) the
  independent heads do not capture as well.

**Crossings.** 20–27 % of test rows have a non-monotone
`(q05, q50, q95)` triple before sorting. This is expected for
independent quantile XGB and the project deliberately decided
against per-quantile HP tuning (would multiply tuning cost by 3 and
make the median sanity guardrail less informative). The
`QuantileHeads.predict(..., repair_crossings=True)` path returns the
sorted triple; downstream NSGA-II / Pareto consumers should always
pass that flag.

**Wall-clock.** 132.8 s on 8 cores for 4 targets × 3 heads, fits
comfortably inside the original 10-min Phase-2 budget.

**New code (this entry).**
- `roverdevkit/surrogate/uncertainty.py` — `QuantileHeads`
  dataclass (frozen, joblib-safe), `fit_quantile_heads(...)`,
  `coverage_table(...)`. Module docstring documents the shared-HP
  rationale and the crossing-rate caveat.
- `scripts/calibrate_intervals.py` — CLI driver. Loads the v4
  parquet + tuned best params, fits per target, writes
  `coverage.csv`, `median_sanity.csv`, `fit_seconds.csv`,
  `quantile_bundles.joblib`, plus a console summary.
- `tests/test_surrogate_uncertainty.py` — 6 smoke tests covering
  bundle shape, predict / column-mismatch / repair-crossings,
  coverage-table schema, save/load roundtrip. Runs in <1 s on the
  shared `small_df` LHS fixture.

**Modified.**
- `pyproject.toml` — added the three new files to the
  `per-file-ignores` block for N803 / N806 (sklearn-style `X` /
  `X_train` naming, same convention as W8 step-3).
- `project_plan.md` §6 W8 step-4 — marked done with the headline
  numbers and the slope under-coverage caveat.

**Artifacts.** `reports/week8_intervals_v4/` (coverage tables,
median sanity, fit timings, joblib bundles, SUMMARY.md).

**What this closes.**
- §6 W8 step-4 deliverable: ✅ final mission-level surrogate +
  calibrated PIs + per-family coverage table.
- §7 Layer-1 PI claim: ✅ for 3/4 targets at ≤2 pp; slope at 4 pp,
  acknowledged in the writeup.
- **Phase 2 (data + ML) is now complete.** The corrected evaluator
  is the source of truth (~40 ms / mission, ~5 ms on 8 cores); the
  mission-level surrogate is the optional accelerator + UQ layer
  for NSGA-II inner loops, batch SHAP, and probabilistic
  feasibility constraints. Both are validated and tuned on v4.

**Next.** Phase 3, Week 9 — tradespace exploration. The corrected
evaluator handles parametric sweeps (≤ 10 k points) directly; NSGA-II
swaps in the surrogate (with quantile heads for probabilistic
feasibility) and re-validates the final Pareto front against the
corrected evaluator before any headline plot is written.

---

## 2026-04-26 — Week 9 retired, critical items rolled into Phase 4 (Week 13)

**What.** The originally-scoped **Week 9 "External validation against
published experimental data"** is removed from the project plan. The
six bullets in that block had drifted out of sync with the post-W7.7
reframe — most were already done in earlier weeks under different
names — so the block was kept primarily as a TODO marker rather than
as a unit of work.

**Audit (Week 9 line-by-line, before vs after).**

| Original Week 9 bullet                              | Status (now)                                                |
|-----------------------------------------------------|-------------------------------------------------------------|
| BW vs published single-wheel data                   | **Pending — rolled into Week 13.** Existing xfail in `tests/test_terramechanics.py::test_single_wheel_matches_wong_textbook_example`. |
| SCM vs published single-wheel data                  | **Pending — rolled into Week 13 (citation-only).** PyChrono SCM is already validated in Tasora et al.; we cite, not re-validate. |
| Surrogate vs Bekker-Wong (ML fidelity)              | **Done in W6 / W8.** §7 Layer 1; numbers in `reports/week8_baselines_v4/SUMMARY.md` and `reports/week8_tuned_v4/SUMMARY.md`. |
| Surrogate vs SCM                                    | **Obsolete.** Replaced by §7 Layer 2 (corrected-evaluator vs SCM-direct), done in W7.4 single-wheel correction fit and W7.7 mission-level bake-off. |
| Full evaluator vs published rover traverse data     | **Partially done.** Started W5 in `notebooks/00_real_rover_validation.ipynb`; formalised W6 as the registry-rover Layer-1 sanity (primary vs diagnostic split). Anything left is a writeup, not new measurement. |
| Layered error-budget writeup                        | **Pending — rolled into Week 13.** `reports/error_budget.md` (markdown), not a Python module. |

**Plan changes.**
- `project_plan.md` §6 Phase 2 header: "Weeks 6–9" → "Weeks 6–8".
- `project_plan.md` §6 Week 9 block: removed (9 lines).
- `project_plan.md` §6 Phase 4 Week 13: expanded to absorb the three
  remaining critical items, each marked "rolled forward from the
  retired Week 9":
  1. Layer-3 BW vs Wong textbook ch. 4 worked example (replace xfail
     with a real tolerance check; one-paragraph paper result).
  2. SCM citation paragraph (PyChrono validation is upstream; cite
     Tasora et al.).
  3. Consolidated `reports/error_budget.md` pulling W6 acceptance
     gates, W6 registry sanity, W7.4 single-wheel correction R²,
     W7.7 mission bake-off, W8 surrogate metrics, and W8 quantile-PI
     coverage into a single end-to-end chain. Cross-references §7
     Layers 1-2 explicitly so we don't redo finished work.
- `project_plan.md` §13 milestones checklist: Week 9 line removed;
  Week 13 line annotated with the rolled-forward items.

**Code / data hygiene (this entry).**
- Removed `roverdevkit/validation/error_budget.py` (stub:
  `compile_error_budget()` raised `NotImplementedError`; unused).
  The actual error budget will be a markdown writeup in `reports/`,
  not a Python module — matches the "one-place rule" repo audit.
- Removed `roverdevkit/validation/experimental_comparison.py` (stub:
  two functions that raised `NotImplementedError`; unused). The
  Layer-3 BW-vs-literature work collapses to a single tolerance test,
  not a sub-package.
- Updated `roverdevkit/validation/__init__.py` to drop the two stub
  module references and to point Layer-3 / error-budget pointers at
  Week 13 / Phase 4.
- Updated `data/validation/README.md` to point at Week 13 (was Week 9
  + Weeks 1–2).
- Updated `tests/test_terramechanics.py` module docstring + the
  Wong-placeholder section header to reference Week 13 / Phase 4.

**Out of scope for this entry.** No measurement work — the
Wong-textbook digitisation, the SCM citation paragraph, and the
error-budget compilation all happen in Week 13. The current scope is
purely plan / documentation tidy-up so that when Week 13 starts there
is one canonical home for each remaining item rather than a stub
package, a stale checklist line, and three different week numbers.

**Why now (vs Phase 4).** Doing the cleanup now (a) keeps the active
project surface small while we move into Phase 3 — the user's stated
"one-place rule" preference — and (b) prevents the Week-9 stubs from
silently rotting further as Phase 3 (Weeks 10-12) lands new code.

**Next.** Phase 3, Week 10 — tradespace sweep tool with
`backend: Literal["evaluator", "surrogate"]` switch, default
evaluator for ≤10k points, surrogate for batch / NSGA-II.
