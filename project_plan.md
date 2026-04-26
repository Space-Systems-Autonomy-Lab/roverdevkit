# RoverDevKit: ML-Accelerated Co-Design of Mobility and Power Subsystems for Lunar Micro-Rovers
## Semester Project Plan

---

## 1. Project Summary

**Title:** RoverDevKit — ML-Accelerated Co-Design of Mobility and Power Subsystems for Lunar Micro-Rovers

**One-sentence pitch:** Build an open-source mission-level rover evaluator that chains terramechanics, solar power, and traverse simulation models, lift it to higher fidelity with a wheel-level Bekker-Wong-to-PyChrono-SCM correction model, and use it to explore mission-relevant Pareto fronts validated against the published design points of real lunar micro-rovers.

**Core contributions:**

1. An open-source, fully-documented mission evaluator for lunar micro-rovers that takes a design vector and a mission profile and returns mission-level performance metrics (traverse range, energy margin, slope capability, mass) with every sub-model traceable to a cited source. After the Week-7.7 traverse-loop optimisation the corrected evaluator runs in ~40 ms / mission on a single core (~5 ms on 8 cores), making most Phase-3 workflows feasible without an outer surrogate.
2. A **wheel-level multi-fidelity correction model** (`roverdevkit.terramechanics.correction_model.WheelLevelCorrection`) that learns the Δ between Bekker-Wong analytical wheel forces and PyChrono SCM's higher-fidelity contact model from a small (~500-row) SCM single-wheel sweep, then composes back into the analytical traverse loop at every wheel-force step. This is the methodological centrepiece: it is what makes the evaluator multi-fidelity at millisecond cost.
3. A mission-level XGBoost / MLP surrogate over the corrected evaluator (Phase 2) that serves as (a) an inner-loop accelerator for the Phase-3 NSGA-II constraint search, (b) the home for calibrated prediction intervals (Week-8 step-4), and (c) the reference baseline for the Phase-5 benchmark release. The surrogate is *not* the headline contribution — the corrected evaluator is fast enough on its own for most workflows; the surrogate adds probabilistic feasibility, uncertainty quantification, and bulk inference at large batch sizes.
4. Validation that the optimizer rediscovers the design points of real lunar micro-rovers (Rashid, Pragyan, Yutu-class) within stated tolerances when given matching mission constraints, plus interpretable design rules extracted via SHAP that generalize across mission profiles.

**Why this is novel:** Existing rover surrogate work is almost entirely component-level (single wheel, single subsystem). System-level rover trade studies exist but are either proprietary (JPL Team X, ESA CDF) or use static spreadsheet models without ML. The novel piece here is the **wheel-level correction architecture** — a small ML model trained on cheap SCM single-wheel runs that composes into an analytical mission-level loop, lifting the whole stack to multi-fidelity without ever running SCM in the inner loop. No open-source tool combines a physics-based mission evaluator with that correction architecture and validation against real flown rovers. The "rediscover Rashid/Pragyan" validation is a concrete, falsifiable claim that's rare in this literature.

**Target rover class:** Micro-rovers in the 5–50 kg range (Rashid was 10 kg, Pragyan was 26 kg, CADRE units are ~2 kg). This is deliberate — micro-rovers have simpler power and thermal architectures than VIPER-class machines, and there are several recent flown examples to validate against. Scaling to larger rovers is future work.

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    DESIGN VECTOR (inputs)                       │
│                                                                 │
│  Mobility:    R, W, h_g, N_g, N_wheels                         │
│  Chassis:     m_chassis, wheelbase, ground_clearance           │
│  Power:       A_solar, C_battery, P_avionics                   │
│  Operations:  v_nominal, duty_cycle_drive                      │
│                                                                 │
│  Mission constraints (fixed per scenario):                      │
│    latitude, traverse_distance, terrain_class, sun_geometry    │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                  MISSION EVALUATOR (physics)                    │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐     │
│  │ Terramechanics│  │ Solar / Power│  │ Traverse Sim     │     │
│  │              │  │              │  │                  │     │
│  │ Bekker-Wong  │  │ Solar geom + │  │ Step through     │     │
│  │ + PyChrono   │  │ panel model +│  │ traverse, track  │     │
│  │ SCM corrections│ battery SOC   │  │ energy & slip    │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────────┘     │
│         │                 │                 │                  │
│         └─────────────────┴─────────────────┘                  │
│                           │                                    │
│                           ▼                                    │
│                    Mass model (parametric)                     │
│                    Thermal survival check                      │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                  MISSION METRICS (outputs)                      │
│                                                                 │
│  Primary:   range_km, energy_margin_pct, slope_capability_deg  │
│  Secondary: total_mass_kg, peak_motor_torque, sinkage_max      │
│  Constraint: motor_torque_ok (binary)                          │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│        WHEEL-LEVEL CORRECTION (multi-fidelity)                  │
│                                                                 │
│  Trained once from a ~500-row PyChrono SCM single-wheel sweep:  │
│    Δ = (Δdrawbar_pull, Δtorque, Δsinkage)                       │
│           = SCM(features_12d) − BW(features_12d)                │
│  Composed back into the BW traverse loop at every wheel step    │
│  → corrected evaluator runs at ~40 ms / mission (BW + Δ)        │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                  ML SURROGATE LAYER (optional)                  │
│                                                                 │
│  Trained once over the corrected evaluator's outputs (40k LHS): │
│    XGBoost / MLP per target + classifier on motor_torque_ok     │
│    Quantile XGBoost heads for 90% prediction intervals          │
│  Used for: NSGA-II inner loop, bulk SHAP/Sobol, UQ, benchmark   │
│  baseline. The corrected evaluator is the source of truth.      │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│              TRADESPACE EXPLORATION LAYER                       │
│                                                                 │
│  - Parametric sweeps (corrected evaluator on 8 cores; surrogate │
│    only when batch > 50k or PIs are required)                   │
│  - NSGA-II inner loop on surrogate, Pareto-front validation     │
│    against the corrected evaluator                              │
│  - SHAP-based design rule extraction                            │
│  - Mission scenario library (polar, equatorial, mare, highland) │
└────────────────────────────────────────────────────────────────┘
```

The key architectural choice is that the **corrected mission evaluator is the primary artifact**: Bekker-Wong + the wheel-level SCM correction, runnable on its own at ~40 ms / mission. The mission-level surrogate is an *optional acceleration and uncertainty layer* on top — a 500-1000× speedup at batch sizes where that matters (1M-point grids, sensitivity analyses), and the natural home for prediction intervals and probabilistic feasibility — but most Phase-3 workflows (NSGA-II Pareto-front search at ~20k evaluations, parametric sweeps, real-rover replays) are well-served by the evaluator on 8 cores. This matters because (a) every ML claim is grounded in a specific physics model the reviewer can inspect, (b) the evaluator can stand alone as a useful open-source tool even if the ML doesn't pan out, and (c) the wheel-level correction model is a small, individually-validatable artifact that does the heavy methodological lifting on its own.

---

## 3. Design Space and Mission Scenarios

### 3.1 Design Variables (12 dimensions)

| Category | Variable | Symbol | Range | Units |
|----------|----------|--------|-------|-------|
| Mobility | Wheel radius | R | 0.05–0.20 | m |
| Mobility | Wheel width | W | 0.03–0.20 | m (v3 widened from 0.15) |
| Mobility | Grouser height | h_g | 0–0.020 | m (v3 widened from 0.012) |
| Mobility | Grouser count | N_g | 0–24 | int |
| Mobility | Number of wheels | N_w | {4, 6} | int |
| Chassis | Chassis dry mass | m_c | 3–50 | kg (v3 widened from 35) |
| Chassis | Wheelbase | L_wb | 0.3–1.2 | m |
| Power | Solar array area | A_s | 0.1–1.5 | m² |
| Power | Battery capacity | C_b | 20–500 | Wh |
| Power | Avionics power draw | P_a | 5–40 | W |
| Operations | Nominal drive speed | v | 0.01–0.10 | m/s |
| Operations | Drive duty cycle | δ | 0.1–0.6 | — |

Thermal architecture is explicitly *not* included as a design variable in the first pass — it's a binary constraint check (does the rover survive lunar night? does the avionics box stay in spec during peak sun?) using a simple lumped-parameter model. Adding thermal as a full design dimension is a stretch goal for week 12 if everything else is on track.

### 3.2 Mission Scenarios

Four mission scenarios that the optimizer targets:

1. **Equatorial mare traverse** — Apollo-17-like terrain, 14-day mission, find km traversed before battery degradation.
2. **Polar prospecting** — high latitude, long shadows, intermittent sun, find max number of waypoints visited.
3. **Highland slope capability** — up to 25° slopes, find minimum mass design that can climb the worst slope at the worst soil parameters.
4. **Crater rim survey** — short traverse, lots of slope changes, find energy-optimal design.

Each scenario fixes the mission constraints (latitude, terrain class, sun geometry, traverse distance) and lets the optimizer search over the design vector. The Pareto fronts are generated *per-scenario*. This is what makes the SHAP analysis interesting later — you can ask "how does the optimal wheel size shift between equatorial and polar missions?" and get a real answer.

---

## 4. Sub-Model Specifications

Each sub-model needs to be (a) implementable from a published source, (b) testable against published data independently, and (c) cited explicitly in the paper.

**Terramechanics (mobility forces).** Primary: Bekker-Wong pressure-sinkage with Janosi-Hanamoto shear, implemented in pure Python from Wong's *Theory of Ground Vehicles* (4th ed., chapters 2–4). This gives drawbar pull, sinkage, driving torque as a function of wheel geometry, slip, load, and soil parameters in microseconds. Validation: published single-wheel testbed data (Ding 2011, Iizuka & Kubota 2011, Wong's own datasets). Correction layer: PyChrono SCM runs for the points where Bekker-Wong is known to be weakest (high slip, grousered wheels, sloped terrain). The surrogate learns the correction.

**Solar power.** Inputs: latitude, time of day, sun elevation/azimuth, panel area, panel tilt, panel efficiency, dust degradation factor. Output: instantaneous power generation. This is straightforward solar geometry — pull the model from any spacecraft thermal/power textbook (Larson & Wertz *SMAD*, or Patel *Spacecraft Power Systems*). For lunar sun geometry, NASA's published lunar ephemeris data is sufficient; SPICE is not needed for a tradespace tool. Validation: cross-check against published power profiles for Yutu-2 or VIPER concept studies.

**Battery.** Simple SOC tracking with charge/discharge efficiency, depth-of-discharge limits, and a temperature-derating factor. Lithium-ion at lunar temperatures has published curves (NASA Glenn reports). Validation: confirm a 100 Wh battery delivers ~85 Wh usable, etc.

**Mass model.** Bottom-up parametric mass model: each subsystem (wheels, motors+drives, solar panels, battery, avionics, harness, thermal, margin) is computed from a physics-grounded specific mass or a standard spacecraft-sizing fraction taken from SMAD, AIAA S-120A-2015, and vendor catalogues (Maxon/Faulhaber for motors, Spectrolab/AzurSpace for solar, NASA Glenn for Li-ion pack specific energy). Chassis mass enters as a design variable. The small set of published lunar micro-rovers (Rashid, Pragyan, Yutu-1/2, CADRE, Sojourner, MARSOKHOD prototype, ExoMy, Lunokhod) is used as a **validation set, not training data** — target is median absolute error ≤30% on in-class (5–50 kg) rovers. We deliberately avoid regressing per-subsystem masses on n≈8 heterogeneous rovers because (a) per-subsystem mass breakdowns aren't published for most of them and (b) the sample spans 2 kg to 756 kg across 50 years of technology, which makes any regression dominated by a handful of leverage points. The bottom-up approach keeps every coefficient individually citable (which also enables sensitivity sweeps of those coefficients in the surrogate layer).

**Traverse simulator.** A simple time-stepped loop: at each step, compute power generated (solar geom), power consumed (rolling resistance from terramechanics + avionics + thermal heaters), update battery SOC, advance position by `v × dt × δ`, check survival constraints, log everything. Run for the mission duration. This is ~300 lines of Python. Validation: run with Yutu-2 parameters and check that the predicted daily traverse distance is in the right ballpark compared to published actuals.

**Thermal survival check.** Binary: does a lumped-parameter thermal model of the avionics box stay within survivability limits during peak sun and lunar night? Inputs: avionics power, surface absorptivity/emissivity, RHU power if any, insulation. Pass/fail, not a continuous design variable in v1.

The total LOC for the evaluator including all sub-models should land around 2,000–3,000 lines of Python.

---

## 5. Three-Path Data Generation Strategy

This is specific to running on a MacBook and protects against PyChrono problems.

**Path 1 — Analytical (the safety net):** Run the full mission evaluator using only Bekker-Wong terramechanics. Cost: ~40 ms per mission post-W7.7 lift-out. Generate 40k LHS samples in ~2-3 minutes on 8 workers. This dataset alone is sufficient to train a usable surrogate and write the MVP paper.

**Path 2 — Wheel-level SCM correction (the upgrade, shipped):** Run PyChrono SCM single-wheel simulations at ~500 strategically-chosen points in the **wheel-feature** space (12-d: vertical load, slip, wheel geometry, soil parameters), **not** the full mission design space. Train a wheel-level correction regressor on `Δ = SCM − BW` and compose it into every Bekker-Wong wheel-force evaluation in the analytical traverse loop (`roverdevkit.terramechanics.correction_model.WheelLevelCorrection` → `roverdevkit.mission.traverse_sim`). Cost: ~0.16 s per SCM run × 500 runs / 5 parallel processes ≈ 16 s wall-clock for the gate sweep; the production correction model in `data/scm/correction_v1.joblib` was fit on the same 500-row sweep (W7.4). Cross-validated against direct SCM-in-the-loop on a shared sample at W7.7.

**Path 3 — Experimental validation (the credibility):** Curated single-wheel testbed data from published papers. Not used for training. Used only to validate that Path 1 + Path 2 produces realistic numbers.

This three-path structure makes the project robust on a laptop. The Week-7 SCM driver re-validation (W7-step-1) and the W7.5 correction-magnitude gate together resolve the SCM go/no-go decision: gate fired, correction is shipped, all of Phase 2 onward runs against the corrected evaluator.

---

## 6. 15-Week Schedule

### Phase 1: Evaluator Foundation (Weeks 1–5)

**Week 1: Environment and analytical terramechanics.**
- Set up project repo, conda environment, CI skeleton.
- Start PyChrono installation in parallel (treat as a background task, not blocking).
- Implement Bekker-Wong + Janosi-Hanamoto in pure Python (~300 lines). Unit tests against worked examples from Wong's textbook.
- **Deliverable:** `terramechanics/bekker_wong.py` with passing tests; PyChrono either installed or flagged as a risk to resolve in week 2.

**Week 2: Solar/power and battery models. PyChrono go/no-go.**
- Implement solar geometry and panel power model with latitude/time inputs.
- Implement battery SOC model with derating.
- Validate solar model: cross-check noon power for Yutu-2 latitude against published number.
- **Hard gate:** if PyChrono isn't running by Friday, commit to Path 1 + Path 3 only and document the decision in the project log. Don't keep fighting it.
- **Deliverable:** `power/solar.py`, `power/battery.py`, PyChrono decision recorded.

**Week 3: Bottom-up mass model and published-rover validation set.**
- Build a small database of published lunar micro-rover specs (Rashid, Pragyan, Yutu-1/2, CADRE, Sojourner, Lunokhod, plus 1–2 academic concepts) in `data/published_rovers.csv`.
- Implement a bottom-up mass model (`mass/parametric_mers.py`): per-subsystem mass from specific-mass constants cited to SMAD / AIAA S-120A / vendor catalogues, assembled via SMAD-style harness / thermal / margin fractions, with the motor subsystem sized to peak wheel torque by fixed-point iteration. Expose every coefficient through a `MassModelParams` dataclass so it can be swept in sensitivity studies.
- Curate a gap-filled design-vector validation set (`data/mass_validation_set.csv`) with imputation notes for every non-published field; run the bottom-up model against it and require **median absolute error ≤30% on in-class (5–50 kg) rovers**.
- **Deliverable:** `mass/parametric_mers.py`, `mass/validation.py`, `data/published_rovers.csv`, `data/mass_validation_set.csv`, all with citations and a test gate enforcing the 30% validation target.

**Week 4: Mission traverse simulator.**
- Time-stepped traverse loop integrating terramechanics + power + battery + mass.
- Implement the four mission scenarios as configuration files.
- Implement the thermal survival check (lumped-parameter, binary pass/fail).
- **Deliverable:** `mission_evaluator.py` — single function that takes a design vector + scenario and returns mission metrics.

**Week 5: Evaluator validation against real rovers.**
- Configure the evaluator with Yutu-2 parameters and run an Apollo-17-mare-like scenario. Compare predicted daily traverse distance and power profile against published Yutu-2 numbers.
- Configure with Pragyan parameters, run for the actual Chandrayaan-3 mission profile, compare.
- Configure with Rashid parameters (mission was short due to lander failure but specs are published).
- Document the discrepancies. The goal isn't ±5% accuracy — it's "predictions are in the right order of magnitude and the right *direction* when parameters change."
- **Deliverable:** `validation/real_rover_comparison.ipynb` with quantitative results and a discussion of where the model is and isn't trustworthy.

This is the most important week of the project. If the evaluator can't reproduce real rover behavior to within reasonable tolerances, fix it before doing anything ML-related. Don't move on if this validation is shaky.

### Phase 2: Data Generation and Surrogate (Weeks 6–8)

**Week 6: Analytical dataset (Path 1) and baseline surrogates.**

The original plan assumed ~50 ms per evaluation giving "50,000 LHS samples in an afternoon." Measured cost is ~1.4 s per evaluation (1.6 s on long equatorial missions, 3.2 s on polar), roughly 30× slower. The schedule, dataset size, and surrogate architecture below are adjusted for that reality and for Week-5/5.5/5.6 carry-over (unclipped energy margin, capability-envelope framing, lowered duty-cycle floor).

- **LHS sampling.** Stratified by `n_wheels ∈ {4, 6}` (only 2 levels, naive LHS undersamples). Continuous variables sampled via `scipy.stats.qmc.LatinHypercube`; integer variables (`grouser_count`) rounded from continuous samples; `n_wheels` split 50/50 across strata. Scenario parameters (latitude, Bekker soil params c / φ / kc / kφ / n, mission duration, max slope) sampled jointly so the model learns a single cross-scenario function instead of four per-scenario models. Record seed and sampler config in dataset metadata.
- **Dataset size.** Start with a **pilot run of 2k samples** (~5 min with 8 workers) to verify the pipeline and catch schema / NaN / multiprocessing bugs cheaply. Full run targets **10k per scenario × 4 scenarios = 40k rows** (~2 h with 8 workers). Scale to 20k/scenario only if the pilot surrogate misses the R² targets on range or energy margin.
- **Dataset schema (extensibility for W7/W8).** Each row stores: design vector (12 fields), scenario parameters as continuous features (not one-hot of simulant names), raw `MissionMetrics`, and a `fidelity: "analytical"` tag so SCM-corrected runs can be appended later if needed. Also store aggregate sub-model statistics (peak / mean / P95 of wheel drawbar pull, sinkage, motor torque, solar power, battery SOC). These stats serve two purposes downstream: (a) they're the most informative features for the Week-7.5 correction surrogate, which learns `Δmission_metric = f(design, scenario, wheel-regime)`; (b) they enable sub-model-level SHAP and failure-mode diagnostics in the Week-12 design-rule analysis. Written as Parquet to `data/analytical/lhs_v3.parquet` (current canonical; v1/v2 retired, see `data/analytical/SCHEMA.md`).
- **Regression targets.** `range_km` (capability at designed duty, per Week 5.6), `energy_margin_raw_pct` (unclipped, per Week 5.5 — the plan originally specified the clipped `energy_margin_pct` but that saturates across much of the design space), `slope_capability_deg`, `total_mass_kg`. The clipped reporting metric is derived post-hoc from the raw prediction.
- **Feasibility: single-target classifier + regressor (Week-6 step-2 scope cut).** The boolean `motor_torque_ok` is trained as the feasibility classifier (XGBoost / logistic), with the regressor trained on the feasible subset. This gives honest feasibility probabilities for the Week-11 NSGA-II constraint layer and keeps the regressor from wasting capacity on the `range_km = 0` failure mode. **Thermal scope:** `thermal_survival` is no longer a surrogate target. The current mass model treats RHU power and MLI quality as free design choices, so `thermal_survival` reduces to a near-trivial gate ("did you add an RHU?") with no design trade-off. The system-level evaluator still computes it as a diagnostic (preserved for Pragyan/Yutu-2 distinction in the Week-5 validation harness), and a future mass-model upgrade that charges RHU/MLI mass will let thermal re-enter the surrogate as a real Pareto target. See `data/analytical/SCHEMA.md` v1→v2 notes.
- **Baselines (Week-6 step-4 architecture).** Three families fit **per target** (one model per (algorithm, target) cell): Ridge linear, random forest, XGBoost. Plus one **joint multi-output MLP** (`sklearn.neural_network.MLPRegressor`, hidden layers 128→64) trained simultaneously over all four primary targets — the joint model is the only baseline that can share a representation across `range_km`, `energy_margin_raw_pct`, `slope_capability_deg`, and `total_mass_kg`. Per-target rather than joint multi-output for the tree/linear models because the four targets respond best to different hyperparameters; a shared `MultiOutputRegressor` would under-fit the harder ones (range, energy margin) and over-fit the easier ones (mass) at one set of shared hyperparams. 80/10/10 train/val/test split stratified by scenario; categoricals consumed natively by XGBoost (`enable_categorical=True`) and one-hot-encoded for Ridge/RF/MLP via a shared `ColumnTransformer`. Regressors trained on the feasible subset (`motor_torque_ok=True`); feasibility classifier (LogReg + XGBoost) trained on all rows so it sees both classes. Hyperparameter tuning is **deferred to Week 7** so the Week-6 numbers report sensible-default performance and the Week-7 Optuna lift is cleanly attributable.
- **Evaluation.** Report R² / RMSE / MAPE per target, broken out both aggregate and per-scenario (catches cases where the model is great on equatorial and terrible on polar). AUC / F1 / accuracy for the feasibility classifier. Plus a **registry-rover sanity check**: predict metrics for Pragyan, Yutu-2, MoonRanger, and Rashid-1 against their registry design vectors and compare surrogate predictions to the evaluator's own predictions (Layer-1, not Layer-4). Pragyan and Yutu-2 are flown rovers; MoonRanger and Rashid-1 are well-spec'd design-target lunar micro-rovers (never deployed) added for Layer-1 OOD coverage of the surrogate's input space. Guards against the surrogate doing well on IID LHS but being wrong exactly where we validate. Categorical conform path (training-codebook recoding with NaN for unseen levels) ensures registry rovers with simulants outside the LHS support do not crash XGBoost's strict categorical recode. The Mars-gravity Sojourner sentinel was removed when the project narrowed to lunar micro-rovers (project_log.md 2026-04-25). **Layer-1 acceptance scope.** The sanity check is graded on the **design-axis primary set** — `total_mass_kg`, `slope_capability_deg`, `motor_torque_ok` — where the v3 widened bounds put every registry rover inside training support. `range_km` and `energy_margin_raw_pct` are emitted alongside the primary table as scenario-OOD diagnostics only: the registry rovers' published mission distances are 100–1000× smaller than the LHS family budgets, so the relative error there reflects an absolute-scale mismatch and is not part of the acceptance set. The split is enforced in code via `LAYER1_PRIMARY_TARGETS` / `LAYER1_DIAGNOSTIC_TARGETS` in `roverdevkit/surrogate/baselines.py` and surfaced as an `is_primary` boolean column on `registry_sanity.csv`.
- **Target accuracy (unchanged from original plan).** R² > 0.95 for range and raw energy margin; R² > 0.85 for slope capability and total mass; AUC > 0.90 for feasibility.
- **Week-6 step-4 result (`reports/week6_baselines_v2/`, 40k LHS v3, 32k train / 3.9k val / 4.0k test).** 16/18 acceptance rows pass on the test split. The two failures are both **Ridge** (R² 0.77 on range, 0.66 on energy margin) — exactly the diagnostic we want: the linear baseline cannot model the multiplicative coupling of solar geometry × duty cycle × mass. The **joint MLP wins every primary target** by a small margin over per-target XGBoost: range R² 0.9986 (XGB 0.9980), energy-margin raw R² 0.9950 (XGB 0.9837), slope R² 0.9973 (XGB 0.9847), mass R² 0.9996 (XGB 0.9993). XGBoost wins feasibility classification (AUC 0.997 vs LogReg 0.988). Total fit wall-clock 28 s on 11 cores. **Per-family caveat (surfaced by step-5 writeup).** The pooled-aggregate gates hide one weak cell: the joint MLP's `energy_margin_raw_pct` on `polar_prospecting` lands at R² 0.753 (vs 0.99+ on the other three families). It is the only per-family violation in the report and is queued as the highest-priority Week-7 Optuna target — exactly what the per-family table exists to catch. The **Layer-1 registry sanity** is graded on the design-axis primary set after the 2026-04-25 reframing: median across algorithms gives Yutu-2 mass MAPE 0.8 % and slope MAPE 2.3 %; Rashid-1 9.4 % and 4.1 %; Pragyan 2.7 % and 78 %; MoonRanger 3.2 % and 40 %. Feasibility classifier accuracy is 100 % on Pragyan / Yutu-2 / Rashid-1 and 50 % on MoonRanger (one of the two algorithms disagrees with the evaluator's negative-margin call). The diagnostic set (`range_km`, `energy_margin`) is reported separately and remains badly miscalibrated against the registry's published short-mission distances; this is a *scenario*-OOD effect that v3 design-bound widening could not address and is intentionally excluded from Layer-1 acceptance — see `project_log.md` 2026-04-25 entries for full numbers and discussion.
- **Benchmark-release hooks (near-zero marginal cost; unlock Paper 2 cheaply).** Three design decisions to lock in at dataset-generation time so the Phase-5 benchmark release isn't a retrofit:
  1. **Canonical train/val/test split stored as a column in the parquet** (not re-split at training time). Seeded; deterministic; the same 10% is "the test set" for every downstream paper.
  2. **Evaluation script as a public library function**: `roverdevkit.surrogate.benchmark_score(predictions_df: pd.DataFrame) -> BenchmarkReport`. Takes predictions, returns R² / RMSE / MAPE / AUC per target and per scenario.
  3. **Versioned schema documentation** in `data/analytical/SCHEMA.md`: columns, types, units, citations. Required for any future dataset-release paper; trivial to write at generation time, painful to retrofit.
- **Deliverable:** `roverdevkit/surrogate/{sampling,dataset,baselines,metrics}.py` + `benchmark_score` helper, `data/analytical/lhs_v3.parquet` with version metadata and canonical split column, `data/analytical/SCHEMA.md`, `reports/week6_baselines_v2/` (acceptance gate, per-(algorithm × target × scenario family) metrics, fit-time table, Layer-1 registry sanity CSV with `is_primary` split), CI gates on sampling reproducibility, split stability, and a pilot-scale fit smoke test. (Step-5 results live in `reports/`; the project log captures the decision narrative. No standalone writeup notebook — those are reserved for user-facing demos of the final tool.)

**Week 7: PyChrono SCM data generation (Path 2) — if active.**
- If PyChrono is working: write the SCM single-wheel simulation wrapper, generate 2,000 strategically-sampled SCM runs (focus on grousered wheels, high slip, sloped terrain). Use `multiprocessing` with 4–5 parallel workers, run overnight and on weekends.
- If PyChrono is not working: skip ahead to physics-informed feature engineering instead. Engineer features like dimensionless sinkage `z/R`, effective contact area `R*W`, grouser volume fraction, etc. These boost accuracy of the analytical surrogate measurably.
- **Deliverable:** Either 2,000 SCM runs in `data/scm/` or a feature-engineered analytical surrogate with improved accuracy.

**Week 7.5: SCM correction-magnitude gate (new).**
The architecture is always **composed**: `final = analytical_surrogate + correction_surrogate`. This is §6 W8's original intent and avoids any regeneration of the Week-6 dataset. The gate only decides whether it's worth *training and shipping* the correction surrogate, or reporting SCM as a bounded sensitivity only.
- Generate ~500 SCM-corrected mission evaluations (analytical traverse loop with SCM corrections applied at the wheel level, sampled from the Week-6 LHS parquet). Cost: ~500 × (analytical + correction-lookup) ≈ 15 min with 8 workers.
- Measure the correction distribution: `Δrange_km`, `Δenergy_margin_raw_pct`, `Δslope_capability_deg` vs the analytical baseline.
- Decide:
  - If median `|Δrange_km| / range_km < 10%` *and* no systematic sign bias: **report SCM as a bounded sensitivity** in the Week-9 error budget. Ship the analytical-only surrogate.
  - Otherwise: **train the correction surrogate** on those 500 pairs with `Δmetric = f(design, scenario)` targets, and ship the composed surrogate. No regeneration of the 40 k LHS; the correction surrogate is a small second model layered on top.
The decision and its evidence go in `project_log.md`.

**Week 7 / 7.5 composition mechanism (sketch, 2026-04-25; gate-first ordering per `project_log.md` "Polar `energy_margin_raw_pct` diagnosis (pre-Week-7)" and the day-of plan-A decision).** Specifies the wheel-level correction interface so Week-7 step-1 can start without further design work. The ordering is **inverted**: a minimum-viable SCM sweep (≈500–1000 wheel runs) feeds the Week-7.5 gate first; the full 2000-run sweep is only commissioned if the gate triggers it.

- **Composition rule (per traverse step).** `WheelForces_corrected = WheelForces_BW + Δ`, where `Δ = (Δdrawbar_pull_n, Δdriving_torque_nm, Δsinkage_m)` is the output of a per-wheel correction regressor. DP and torque are the load-bearing corrections; the sinkage delta is carried for diagnostics/SHAP only. The mission-level metrics inherit the correction implicitly through the existing slip-balance and motor-power calculations (see "apply path" below).
- **Wheel-level feature vector (correction-model input, 12-d).** `vertical_load_n, slip, wheel_radius_m, wheel_width_m, grouser_height_m, grouser_count, soil.n, soil.k_c, soil.k_phi, soil.cohesion_kpa, soil.friction_angle_deg, soil.shear_modulus_k_m`. This is exactly the `single_wheel_forces` input set. No `slope_deg`: both BW and the existing SCM driver (`pychrono_scm.single_wheel_forces_scm`) operate on a flat patch, and slope effects enter both via the per-wheel vertical-load projection `cos(θ)·m·g/N_w` upstream in `traverse_sim._normal_load_per_wheel`. If a future iteration adds gravity-tilt to the SCM rig, slope can be added as a 13th feature without disturbing the existing harness. `grouser_count` is an integer (0…24); the other 11 features are continuous.
- **Apply path inside the analytical pipeline (`roverdevkit/mission/traverse_sim.py`).** Two injection points; both gated by `use_scm_correction=True` and a successful model load:
  1. `_solve_step_wheel_forces` — wrap the `single_wheel_forces` call inside the `brentq` residual so the slip-balance equation reflects the **corrected** DP. This is what determines equilibrium slip per step, which in turn drives sinkage and torque downstream.
  2. `_mobility_power_w` — use the **corrected** `driving_torque_nm` for `T·ω` mechanical power, then divide by motor efficiency for electrical draw.
- **Runtime budget.** Correction model must be vectorisable and cheap relative to the Bekker-Wong residual (~10–50 ms per call). RF, XGBoost, or sklearn MLP all clear ~1 ms per inference comfortably; not a bottleneck.
- **Loading and graceful fallback.** `single_wheel_forces_corrected(...)` is the new public helper. It checks for `data/scm/correction_v1.joblib` at first call:
  - File present → load once (process-local cache), apply correction, return `WheelForces` with deltas added.
  - File absent → return raw BW forces with a one-time `UserWarning`. This makes `use_scm_correction=True` safe to leave on whether or not the gate triggered shipping.
  Evaluator API (`use_scm_correction: bool`, currently a `NotImplementedError`) is unchanged; only the model-loading layer knows whether the correction is real or zero.
- **SCM single-wheel sampling design (~500 runs for the gate, expandable to 2 000 if shipped).** Stratified-categorical LHS over the 12-feature wheel-level space: 6-d continuous LHS over `(vertical_load_n, slip, wheel_radius_m, wheel_width_m, grouser_height_m, grouser_count)` jointly with the 6 numeric soil parameters drawn from the catalogue per simulant; balanced categorical assignment of `soil_class ∈ {Apollo_regolith_nominal, JSC-1A, GRC-1, FJS-1}` and `grouser_count_class ∈ {0, 12, 18}` so each of the 12 (soil × grouser) buckets gets ~42 rows at `n=500`. Slip range `[0.05, 0.70]` (skip s=0 where BW agrees by construction; skip s=1 skid). Wheel and load bounds match the v3 design schema (`wheel_radius_m ∈ [0.05, 0.20]`, `wheel_width_m ∈ [0.03, 0.20]`, `grouser_height_m ∈ [0, 0.020]`, `vertical_load_n ∈ [3, 80]` covering 3 kg / 4-wheel through 50 kg / 4-wheel at lunar gravity with slope and dynamic margin). At default `ScmConfig` and the measured 0.16 s/run on M-series, ≈ 500 × 0.16 / 5 cores ≈ 16 s wall-clock for the gate sweep — well within "fits in the lunch break".
- **Wheel-level correction model (Week-7.5 step).** For each SCM run, run BW at the same input; rows are `(features_13d, Δ_DP, Δ_T, Δ_z)` with `Δ = scm − bw`. Fit one regressor per delta target (start with RF; XGBoost as backup). Implements the `CorrectionModel` protocol already in `roverdevkit/terramechanics/correction_model.py`. Persisted as a `joblib` blob to `data/scm/correction_v1.joblib`.
- **Gate decision (already in the Week-7.5 block above).** After the wheel-level correction is in, run ~500 mission-level evaluations sampled from `data/analytical/lhs_v3.parquet` with `use_scm_correction=True`; compare against the analytical baseline; apply the median-`|Δrange|/range` < 10% + no-sign-bias rule. The gate output decides whether to ship the analytical-only surrogate (and report SCM as a §9 bounded sensitivity) or to fit a small mission-level correction surrogate `Δmetric = f(design, scenario)` on those 500 pairs. The wheel-level correction stays as the runtime composition layer either way.
- **Files and homes.** New: `scripts/run_scm_sweep.py` (parallel SCM single-wheel batch driver, resumable queue, parquet I/O — was previously slated for `roverdevkit/terramechanics/scm_wrapper.py`; moved to `scripts/` so the importable package stays light for analytical-only consumers). New artifacts: `data/scm/runs_v1.parquet` (raw SCM single-wheel results), `data/scm/correction_v1.joblib` (fitted wheel-level model), `reports/week7_5_gate/` (gate evidence + decision record). Modified: `roverdevkit/terramechanics/correction_model.py` (concrete `WheelLevelCorrectionModel`), `roverdevkit/mission/traverse_sim.py` (wrap two BW call sites with the corrected-forces helper), `roverdevkit/mission/evaluator.py` (drop the `NotImplementedError` on `use_scm_correction=True` once the wrapper is in). The low-level driver `roverdevkit/terramechanics/pychrono_scm.single_wheel_forces_scm` already exists from Week 2 and is unchanged.
- **Out of scope for this sketch.** Optuna tuning (Week 8); uncertainty quantification (Week 8); soil-bulldozing or grouser-stress sub-models (future, via the same `CorrectionModel` protocol); Mars gravity (excluded post-Sojourner removal).

**Week 8: Final surrogate with uncertainty quantification.**
- Step 1 (done): regenerate `data/analytical/lhs_v4.parquet` with the wheel-level SCM correction composed in row-wise (40 000 rows, 153 s wall on 8 workers; see `project_log.md` 2026-04-26 W8 step-1 entry).
- Step 2 (done): refit baselines on v4 — acceptance gate clears 16/18 (the two failures are Ridge, the linear baseline-of-baseline reference). v4 lift over v3: polar `energy_margin_raw_pct` XGB +0.090 R², slope_capability RF +0.030 R², registry-rover slope MAPE halved on Pragyan / MoonRanger. Outputs in `reports/week8_baselines_v4/`. See `project_log.md` 2026-04-26 W8 step-2 entry.
- Step 3 (done): Optuna TPE tuning of XGBoost (4 regressors + classifier, 50 trials each, 645 s wall on 8 cores). Tuned XGB now beats untuned MLP on `range_km` (0.9993 vs 0.9990) and `total_mass_kg` (0.9998 vs 0.9997) while remaining ~7× faster to fit; tuned XGB classifier AUC 0.988 also beats LogReg 0.985. Registry Pragyan slope MAPE 67.9 → 30.3 % (XGB-only comparison); MoonRanger 39.2 → 20.4 %. Outputs in `reports/week8_tuned_v4/`. New code: `roverdevkit/surrogate/tuning.py` + `scripts/tune_baselines.py`. The default `baselines.py` pipeline is unchanged so W6/W8 step-2 numbers stay reproducible. See `project_log.md` 2026-04-26 W8 step-3 entry.
- Step 4 (done): quantile-XGBoost prediction intervals on the v4 corpus. Three independent heads per primary target at τ ∈ {0.05, 0.50, 0.95}, sharing the W8 step-3 tuned hyperparameters per target (only `objective="reg:quantileerror"` / `quantile_alpha` differ). Median (τ=0.5) sanity guardrail clears: ΔR² vs step-3 tuned medians ∈ [−0.0049, −0.0002] across the four targets, all comfortably above the §7 Layer-1 gate. Empirical 90 % coverage on the canonical test split, after row-wise sorting of the three predictions: `range_km` 0.919, `energy_margin_raw_pct` 0.918, `total_mass_kg` 0.920, `slope_capability_deg` 0.856 (4 pp under-covered — narrow 2° PI, residual conformal correction is a §9 follow-up). Raw (un-sorted) crossings 21–27 % across targets; the bundle exposes both raw and `repair_crossings=True` APIs so downstream NSGA-II / Pareto consumers can choose. New code: `roverdevkit/surrogate/uncertainty.py` + `scripts/calibrate_intervals.py` + `tests/test_surrogate_uncertainty.py`. Outputs in `reports/week8_intervals_v4/`. Wall-clock 132.8 s on 8 cores. See `project_log.md` 2026-04-26 W8 step-4 entry.
- **Deliverable:** Final mission-level surrogate (corrected-evaluator outputs → tuned XGBoost medians + quantile heads) with calibrated 90 % prediction intervals and a coverage table by scenario family. Decision record on the surrogate's reframed role (accelerator + UQ, not primary deliverable) lives in `project_log.md`.

### Phase 3: Tradespace Tool (Weeks 10–12)

The post-W7.7 reframe (corrected evaluator at ~40 ms / mission, ~5 ms on 8 cores) means Phase 3 uses **two backends side-by-side** rather than treating the surrogate as the only fast path:

- **Corrected evaluator** is the source of truth and the default for any workload that fits in its budget — interactive sweeps up to ~10k points, NSGA-II Pareto-front *validation*, real-rover replays, sensitivity studies on `MassModelParams`. Roughly 80 s for a 100k-point grid on 8 cores.
- **Mission-level surrogate** is invoked when (a) batch size > 50k (1M-point sensitivity grids, large-scale Sobol), (b) NSGA-II inner-loop fitness function (≈20-30k evaluations × 200 generations across 4 scenarios), (c) probabilistic feasibility is required (NSGA-II constraint layer with calibrated AUC), or (d) prediction intervals are part of the answer (Pareto-front uncertainty bands).

The same code path can dispatch to either backend so notebooks and CLIs don't fork.

**Week 10: Parametric sweeps and constraint handling.**
- Build the sweep engine with a `backend: Literal["evaluator", "surrogate"]` switch (default: `"evaluator"` for ≤10k points, `"surrogate"` above). Both produce identical column schemas so downstream visualisations don't care which backend ran.
- Implement constraint checking: motor torque limits, mass budget, volume envelope, slope capability minimums. Constraint evaluation uses the feasibility classifier on the surrogate path (probabilistic) and `motor_torque_ok` on the evaluator path (deterministic).
- Build interactive Jupyter notebook with widgets for live tradespace exploration; widget-driven point queries hit the evaluator (sub-second latency is fine), bulk sweeps for the heatmap views hit the surrogate.
- **Deliverable:** `tradespace/sweeps.py` (with both backends) + `notebooks/01_interactive_exploration.ipynb`.

**Week 11: NSGA-II optimization.**
- Wrap pymoo NSGA-II around the **surrogate** as the inner-loop fitness function (median XGBoost predictions for objectives, classifier for the feasibility constraint). Three-objective Pareto: maximize range, minimize mass, maximize slope capability. Constraints from week 10.
- Run for all four mission scenarios.
- **Pareto-front validation pass.** After NSGA-II converges, re-evaluate every point on the final Pareto front (typically 100-500 points per scenario) with the **corrected evaluator** and report the surrogate-vs-evaluator delta on each objective; any point whose evaluator-true value falls behind a non-Pareto neighbour is flagged. This is the cheap way to inherit the surrogate's speed without inheriting its calibration error on the actual headline result.
- Generate Pareto front visualizations with surrogate prediction intervals (W8 step-4) as confidence bands.
- **Deliverable:** Pareto fronts for all four scenarios with the constraint-feasible region highlighted, evaluator-validated against the surrogate on the final fronts, with Pareto-uncertainty bands from the quantile heads.

**Week 12: Design rules and the rediscovery validation.**
- SHAP analysis on the trained surrogate. Generate global feature importance and partial dependence plots for the key design variables.
- Extract interpretable design rules: "below 15 kg total mass, 4-wheel configurations dominate; above, 6-wheel becomes Pareto-optimal because [reason]" — this kind of statement.
- **The rediscovery test:** Set up the optimizer with constraints matching Rashid's mission (mass budget ≤10 kg, equatorial-ish, short traverse). Does the optimizer produce a design in the neighborhood of actual Rashid? Do the same for Pragyan with its constraints. Plot both real rovers on the Pareto fronts.
- This is the headline validation result for the paper. If the optimizer's "best" Rashid-class design has wheel diameter within ~30% of actual Rashid, mass within ~25%, and lands on or near the Pareto front, there's a strong story. If it's wildly different, either explain why convincingly (the optimizer found a better design — defend that claim) or debug the evaluator.
- **Deliverable:** Design rule summary, rediscovery validation plots, packaged tool with README.

### Phase 4: Paper (Weeks 13–15)

**Week 13: Figures, results compilation, and remaining validation gaps.**
- All publication figures and results compilation.
- **Layer-3 validation gap (rolled forward from the retired Week 9):** digitise Wong textbook ch. 4 worked example for single-wheel Bekker-Wong validation; replace `tests/test_terramechanics.py::test_single_wheel_matches_wong_textbook_example` xfail with a real tolerance check; one-paragraph result for the paper.
- **SCM citation paragraph (rolled forward from the retired Week 9):** PyChrono SCM is already validated upstream (Tasora et al., MBSE benchmarks); cite rather than re-validate.
- **Consolidated error-budget writeup (rolled forward from the retired Week 9):** `reports/error_budget.md` pulling the existing W6 acceptance gates, W6 registry sanity, W7.4 single-wheel correction R² / RMSE, W7.7 mission-level bake-off, W8 surrogate-vs-evaluator metrics, and W8 quantile-PI coverage into a single chain "BW agrees with literature at ±X%, corrected evaluator agrees with SCM-direct at ±Y%, surrogate agrees with corrected evaluator at ±Z% → end-to-end ±W% on real rovers." Cross-reference §7 Layers 1-2 explicitly so the layers done during Phase 2 (W6/W8 surrogate gates) and Week 7 (multi-fidelity validation) are not redone.

**Week 14:** Full paper draft.

**Week 15:** Revision, internal review, submission package.

### Phase 5: Benchmark release — RoverBench (Weeks 16-19, post-semester)

Phase 5 packages the Phase-2 dataset and the Phase-2/3 surrogate into a reusable benchmark with baselines, documentation, and a submission interface. The benchmark-release hooks installed in Week 6 (canonical split, `benchmark_score` helper, `SCHEMA.md`) make this phase packaging work rather than new science. Nothing in Phase 5 is on the Paper-1 critical path; the phase runs entirely post-semester and is cleanly cancellable without affecting Paper 1.

**Phase-5 gating decision (start of Week 16).** Decide whether to run Phase 5 at all based on three checks:
1. Paper 1 submitted (or at final-review stage) so benchmark release doesn't distract from it.
2. The Week-6 dataset held up under Phase-2/3/4 scrutiny (no schema changes, no leakage discovered, no calibration errors that would force a regeneration).
3. A plausible venue window exists (NeurIPS D&B, IEEE RA-L, ICRA benchmark track). If not, defer to the next cycle.

If any check fails, park Phase 5 and revisit in the next submission cycle.

**Week 16: Benchmark artifacts.**
- Finalise the public dataset release: `data/analytical/lhs_v3.parquet` (or later if scaled/regenerated), with canonical split, schema doc, and per-row citation/provenance metadata.
- Freeze the `benchmark_score` API and extend with leaderboard-quality metric computation (per-scenario, per-target, feasibility AUC, calibration score if PIs are part of the submission).
- Add a second validation set beyond the LHS holdout: a small (~200 design) "challenge set" deliberately sampled from corners of the design space (low-mass polar, high-slope highland, extreme duty-cycle) that a good surrogate should handle but a poorly-generalising one will fail. Used as a supplementary leaderboard column.
- **Deliverable:** `data/analytical/` with release-ready artifacts, `roverdevkit.surrogate.benchmark_score` at a frozen v1 API, challenge set generated and cached.

**Week 17: Baselines, leaderboard, submission interface.**
- Package the Paper-1 composed surrogate (analytical baseline + correction surrogate, if W7.5 chose to ship it) as `pretrained/roverbench_v1_composed.pkl` with a loader. If W7.5 went with analytical-only, that's the Paper-1 baseline.
- Add minimal baselines for the leaderboard: linear regression, random forest, KNN (for a strong-memorization baseline), and the XGBoost baseline from Week 6. Scored against canonical split + challenge set.
- Submission interface: a `roverbench_submit` CLI that takes a pickled model or a CSV of predictions, runs `benchmark_score`, and produces a submission-formatted JSON + leaderboard entry.
- GitHub repo layout for the benchmark, Hugging Face Dataset card, optional Hugging Face Space for interactive evaluation.
- **Deliverable:** `roverbench/` subpackage or separate repo with dataset, baselines, submission tooling, and starter leaderboard populated with the ~5 baselines.

**Week 18: Benchmark paper draft.**
- Dataset description (per §11.2 outline), justification of the split and challenge set, baseline results table, proposed evaluation protocol.
- Figures: design-space coverage plot (t-SNE/PCA of the 40k LHS), per-scenario distribution plots, feasibility rate per scenario, baseline accuracy bar charts, challenge-set difficulty heatmap.
- Ethics / limitations statement: specific-mass calibration range, soil-parameter domain, capability-vs-utilisation framing, absence of thermal as a design variable.
- **Deliverable:** Complete Paper-2 draft at ~8 pages main + unlimited appendix (NeurIPS D&B format).

**Week 19: Review, polish, submit.**
- Internal review + advisor feedback.
- Final polish: citations, figure typography, reproducibility appendix, code/data archive DOI (Zenodo).
- Submission package for target venue + a preprint on arXiv (cs.LG + eess.SY cross-listing).
- Announcement post with the leaderboard link.
- **Deliverable:** Paper 2 submitted; leaderboard live; dataset archived with DOI.

**If Phase 5 slips.** The gating decision at Week 16 is the right moment to defer. Phase-5 work is additive — nothing downstream of Paper 1 depends on it, and the benchmark hooks in the Week-6 dataset remain valuable internal infrastructure regardless of whether the benchmark gets released externally this cycle or later.

---

## 7. Validation Strategy

The paper's credibility depends on a layered validation approach. Each layer addresses a different question.

### Layer 1: Surrogate vs Corrected Mission Evaluator (ML Fidelity)
**Question:** Does the mission-level surrogate accurately reproduce the corrected evaluator's predictions?
**Method:** Standard ML holdout evaluation on the canonical 10% test split of `data/analytical/lhs_v4.parquet` (40k rows, BW + wheel-level correction). Plus the registry-rover sanity check on the design-axis primary set (`total_mass_kg`, `slope_capability_deg`, `motor_torque_ok`).
**Acceptance criteria:** R² > 0.95 for range and energy margin; R² > 0.85 for slope capability; AUC > 0.90 for feasibility. *Note:* this is the surrogate-vs-evaluator delta only; the surrogate is not the source of truth — the corrected evaluator is. The corrected evaluator's accuracy against SCM-direct is Layer 2.

### Layer 2: Corrected Evaluator vs SCM-Direct (Multi-Fidelity Validation)
**Question:** Does the wheel-level correction recover SCM-grade physics in the mission-level inner loop?
**Method:** Two complementary checks.
- **Wheel-level (W7.4):** Hold-out test set on the SCM single-wheel sweep; per-target R² and RMSE on `Δdrawbar_pull_n`, `Δdriving_torque_nm`, `Δsinkage_m` for the chosen XGBoost regressor.
- **Mission-level (W7.7 bake-off):** Run BW-only, BW + correction, and SCM-direct (PyChrono in the traverse loop) on a shared LHS sample; compare against SCM-direct as ground truth. Report continuous-metric MAPE and feasibility-flip rates.
**Expected result (and observed at W7.7):** BW + correction ≥ 99 % feasibility-flip agreement with SCM-direct, ~10× lower continuous-metric error than BW alone, at 100× the wall-clock speed of SCM-direct.

### Layer 3: Sub-Models vs Published Experimental Data
**Question:** Do the individual sub-models match real measurements?
**Method:** Compare Bekker-Wong against single-wheel testbed data; compare solar model against published rover power profiles; compare battery model against datasheet curves.
**Expected result:** Within-15-30% agreement on terramechanics; closer agreement on solar and battery (these are well-understood physics).

### Layer 4: Mission Evaluator vs Real Rover Traverse Data
**Question:** Does the integrated evaluator reproduce real rover behavior?
**Method:** Configure with Yutu-2 / Pragyan / Rashid parameters and check predicted daily traverse, power profile, and survival against published mission data.
**Expected result:** Order-of-magnitude correct, trends correct when parameters change.

### Layer 5: The Rediscovery Test (Headline Validation)
**Question:** When the optimizer is given constraints matching a real mission, does it produce designs near the real rover?
**Method:** Run NSGA-II with mass budget, mission profile, and terrain matching Rashid; compare resulting Pareto-optimal designs to actual Rashid specs. Repeat for Pragyan.
**Acceptance criteria:** Real rover designs land within ~25-30% of optimizer suggestions on key dimensions, and on or near the Pareto front.

### Layer 6: Sensitivity and Robustness Analysis
**Question:** Are the tradespace conclusions robust to model uncertainty?
**Method:** Perturb soil parameters and `MassModelParams` specific-mass coefficients by ±20%, re-run optimization, check if qualitative conclusions persist.
**Expected result:** Qualitative conclusions stable, exact optimal dimensions shift.

---

## 8. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PyChrono fails to install or SCM doesn't work on M2 | Medium | Medium | Path 2 becomes optional; paper still works with Path 1 + Path 3. Gate at end of week 2. |
| Mission evaluator can't reproduce real rover behavior in Week 5 | Medium | High | Most likely cause is the mass-model calibration of specific-mass constants outside the 5–50 kg class. Mitigation: report parametric sensitivity to `MassModelParams` coefficients; loosen tolerances on the rediscovery validation. If still bad, narrow scope to mobility-power tradespace only and drop the "rediscover real rovers" framing. |
| Published-rover validation set is too sparse to cover every subsystem | Low | Medium | Augment with academic concept rovers, JPL/ESA published study reports, and the CADRE/VIPER design papers; impute the remaining fields with clearly marked rules-of-thumb in `mass_validation_set.csv`. |
| Surrogate accuracy too low on mission-level metrics | Low | Medium | Mission metrics are smoother than wheel-level metrics, so this is less of a risk than in a wheel-only project. If it happens, narrow the design space or add more training data. |
| SHAP design rules turn out to be obvious / boring | Medium | Low | Frame the contribution as "first quantitative validation of engineering intuition" rather than "novel design insights." Still publishable. |
| Reviewers say "this is just systems engineering with a fancier search algorithm" | Medium | Medium | Lean hard on (a) the open-source tool contribution, (b) the rediscovery validation, (c) the multi-fidelity training methodology. Don't oversell the ML novelty — the contribution is integration and validation, not new ML methods. |
| MacBook thermal throttling during overnight SCM runs | Medium | Low | Use `caffeinate -i`, run with laptop elevated for airflow, accept 50–70% sustained throughput in compute estimates. |

---

## 9. Minimum Viable Paper vs Full Vision

**Current state (post-W7.5 gate, 2026-04-25):** the gate fired and we shipped the wheel-level correction (`data/scm/correction_v1.joblib`, R² 0.91-0.96 on the three Δ targets). The Week-8 v4 LHS dataset and tuned XGBoost / MLP baselines are built on top of the corrected evaluator. The full-vision Paper 1 is therefore the active plan.

**Full vision Paper 1** (current path; wheel-level correction shipped, mission-level surrogate as accelerator + UQ): corrected mission evaluator with the wheel-level correction model, mission-level XGBoost / MLP surrogate with quantile-XGB prediction intervals, Layer-1-Layer-6 layered validation including the W7.7 BW-vs-SCM-direct bake-off as cross-validation of the correction architecture, capability-envelope framing, rediscovery validation against two flown rovers + two design-target rovers (Pragyan, Yutu-2, MoonRanger, Rashid-1), cross-scenario SHAP design rules, packaged open-source tool with pretrained correction model and surrogate. Publishable at *Journal of Field Robotics*, *International Journal of Robotics Research*, or *Journal of Spacecraft and Rockets*.

**MVP fallback Paper 1** (only invoked if a downstream regression forces a scope cut, e.g. surrogate fails calibration in Week-8 step-4 or rediscovery test fails in Week 12): same evaluator + correction stack, but reported as "uncorrected analytical evaluator with capability-envelope framing" plus a §6 sensitivity bound on the correction magnitude. Publishable at IEEE Aerospace, AIAA SciTech, or *Acta Astronautica*. Not the active plan; documented for completeness.

**Paper 2 release (Phase 5)** is independent of Paper 1's outcome and is gated at Week 16 on (a) Paper 1 submission status, (b) dataset stability, (c) venue window. Either MVP or full-vision Paper 1 can anchor Paper 2 as the reference baseline on the leaderboard; the benchmark contribution is dataset + task + baselines + leaderboard, not any single surrogate's accuracy.

---

## 10. Software Architecture

```
roverdevkit/
├── README.md, LICENSE (MIT), pyproject.toml, environment.yml
│
├── data/
│   ├── published_rovers.csv          # Specs + citations for ~10 real rovers
│   ├── soil_simulants.csv            # Bekker params for FJS-1, JSC-1A, GRC-1
│   ├── validation/                   # Single-wheel testbed data from literature
│   ├── analytical/                   # Generated LHS samples (Phase 2)
│   │   ├── lhs_v3.parquet            # Canonical 40k-row dataset + canonical split column (v1/v2 retired)
│   │   ├── SCHEMA.md                 # Versioned column spec, units, citations
│   │   └── challenge_v1.parquet      # ~200-design corner-case set (Phase 5)
│   └── scm/                          # PyChrono SCM runs (if Path 2 active)
│
├── roverdevkit/
│   ├── terramechanics/
│   │   ├── bekker_wong.py            # Analytical wheel-force model (BW + Janosi-Hanamoto)
│   │   ├── pychrono_scm.py           # PyChrono SCM single-wheel runner (lazy import; conda-only dep)
│   │   ├── scm_sweep.py              # 12-d wheel-feature LHS design + paired BW/SCM worker
│   │   └── correction_model.py       # WheelLevelCorrection (XGBoost on Δ_DP, Δ_torque, Δ_sinkage)
│   │
│   ├── power/
│   │   ├── solar.py                  # Solar geometry + panel model
│   │   ├── battery.py                # SOC + derating
│   │   └── thermal.py                # Lumped-parameter survival check
│   │
│   ├── mass/
│   │   ├── parametric_mers.py        # Bottom-up physics + specific-mass model
│   │   └── validation.py             # Cross-check vs data/mass_validation_set.csv
│   │
│   ├── mission/
│   │   ├── evaluator.py              # Top-level mission evaluator
│   │   ├── scenarios.py              # Four mission scenarios as configs
│   │   └── traverse_sim.py           # Time-stepped traverse loop
│   │
│   ├── surrogate/                    # Mission-level acceleration + UQ layer (Phase 2)
│   │   ├── sampling.py               # LHS sampler (stratified on n_wheels)
│   │   ├── dataset.py                # Parallel dataset builder over corrected evaluator, Parquet I/O
│   │   ├── features.py
│   │   ├── baselines.py              # Ridge, RF, XGBoost, MLP per target + feasibility classifier
│   │   ├── tuning.py                 # Optuna TPE on XGBoost (Week-8 step-3)
│   │   ├── uncertainty.py            # Quantile XGBoost heads for 90% PIs (Week-8 step-4)
│   │   ├── metrics.py                # R²/RMSE/MAPE, AUC/F1, per-scenario breakdowns
│   │   └── benchmark_score.py        # Public leaderboard metric API (Phase 5)
│   │
│   ├── tradespace/
│   │   ├── sweeps.py
│   │   ├── optimizer.py              # NSGA-II via pymoo
│   │   ├── design_rules.py           # SHAP + PDP
│   │   └── visualize.py
│   │
│   └── validation/
│       ├── rover_rediscovery.py      # The headline validation
│       ├── experimental_comparison.py
│       └── error_budget.py
│
├── notebooks/                              # User-facing demos only
│   ├── 00_real_rover_validation.ipynb      # Evaluator validation against published rovers
│   ├── 01_interactive_exploration.ipynb    # Week 10 — interactive surrogate sweeps
│   ├── 02_pareto_fronts.ipynb              # Week 11 — NSGA-II Pareto fronts
│   ├── 03_rediscover_real_rovers.ipynb     # Week 12 — rediscovery test (paper headline)
│   └── 04_reproduce_paper.ipynb            # Week 13 — regenerate every paper figure
│
├── pretrained/
│   ├── default_surrogate.pkl               # Ships with the package
│   └── roverbench_v1_composed.pkl          # Phase-5 reference baseline
│
├── roverbench/                             # Phase-5 benchmark release
│   ├── README.md                           # Task spec, submission format, leaderboard
│   ├── submission_schema.json              # Expected prediction format
│   ├── submit.py                           # roverbench_submit CLI
│   └── leaderboard.csv                     # Versioned baseline results
│
└── tests/
    ├── test_terramechanics.py
    ├── test_power.py
    ├── test_mission_evaluator.py
    ├── test_surrogate.py
    └── test_benchmark_score.py             # Phase-5 public metric API
```

The pretrained surrogate ships with the package so users can do tradespace exploration without installing PyChrono. The `roverbench/` subpackage is the public-facing benchmark interface for Paper 2.

---

## 11. Paper Strategy

Two papers come out of this codebase: one during the semester (methodology-forward), one as a post-semester benchmark release. The two are complementary, target non-overlapping citation communities, and share the dataset and surrogate as core artifacts.

### 11.1 Paper 1 — Methodology paper (semester, Weeks 13-15)

**Target venues:** *Journal of Field Robotics* (primary), *International Journal of Robotics Research*, or *Journal of Spacecraft and Rockets*. IEEE Aerospace or AIAA SciTech as MVP venues if the full claim has to be scoped down.

### Title
A Wheel-Level Multi-Fidelity Correction Architecture for Mission-Level Lunar Micro-Rover Co-Design, with Capability-Envelope Framing and Layered Validation

### Abstract (~250 words)
- **Problem.** Coupled wheel-power-mass trades for lunar micro-rovers are high-dimensional and currently done with proprietary tools (JPL Team X, ESA CDF) or static spreadsheet models. Existing open-source work is either component-level (single wheel) or conflates hardware capability with operational utilisation, producing range predictions that disagree with flown rovers by 5-10×. High-fidelity terramechanics (PyChrono SCM, DEM) is too slow to run in a mission-level inner loop.
- **Approach.** We introduce a **wheel-level multi-fidelity correction architecture**: a small ML model trained on a ~500-row SCM single-wheel sweep learns the residual between Bekker-Wong analytical wheel forces and SCM, then composes back into the Bekker-Wong traverse loop at every wheel-force step. The corrected analytical evaluator inherits SCM-grade accuracy while running at ~40 ms / mission — fast enough to use in NSGA-II inner loops, parametric sweeps, and large-batch sensitivity studies without an outer mission-level surrogate. We also formalise a **capability-envelope vs operational-utilisation** distinction that separates what the hardware can sustain from what ops schedules actually command.
- **Key result 1 (methodology).** Wheel-level correction validated against direct SCM-in-the-loop on a shared sample shows ≥99 % feasibility-flip agreement and ~10× lower continuous-metric error than uncorrected Bekker-Wong, at 100× the wall-clock speed of SCM-direct.
- **Key result 2 (framing).** Capability-envelope metric reproduces Pragyan and Yutu-2 hardware bounds at published ranges; operational utilisation is exposed as a post-hoc rescaling (`range_at_utilisation`) rather than a design variable.
- **Key result 3 (rediscovery).** Optimizer rediscovers Rashid and Pragyan design points within stated tolerances when given matching mission constraints.
- **Key result 4 (acceleration + UQ).** A mission-level XGBoost surrogate with calibrated quantile-regression prediction intervals serves as the NSGA-II inner-loop fitness function and the Phase-5 benchmark baseline; Pareto fronts are validated by re-evaluating with the corrected evaluator.
- **Deliverable.** Open-source tool (`roverdevkit`) with the corrected mission evaluator, the wheel-level correction model, an optional pretrained mission-level surrogate, and the four-scenario library. Data and baselines released.

### Contributions (in priority order)
1. **Wheel-level multi-fidelity correction architecture.** A small ML model (XGBoost on a 12-dimensional wheel-feature vector) trained on a ~500-row PyChrono SCM single-wheel sweep, composed back into the analytical Bekker-Wong traverse loop at every wheel-force step, lifts the entire mission evaluator to multi-fidelity at ~40 ms / mission cost. We empirically validate against direct SCM-in-the-loop on a shared sample (W7.7 bake-off) and show that BW + correction matches SCM-direct mission outputs while running 100× faster. The architecture is the methodological centrepiece: it is the only piece of new ML methodology in the paper, and it is what the methodology-paper contribution rests on.
2. **Capability-envelope framing** that explicitly separates hardware-sustainable performance from ops-commanded utilisation, with a post-hoc rescaling (`range_at_utilisation`) for ops queries — making mission-level metrics a function of *what the rover can sustain* rather than *what an ops team chose to schedule*.
3. **Open-source corrected mission evaluator** (not just a paper) validated against flown rover data, with the wheel-level correction model packaged as `roverdevkit.terramechanics.correction_model.WheelLevelCorrection` and an optional mission-level surrogate (XGBoost medians + quantile heads) for batch / UQ workflows.
4. **Rediscovery test** as a falsifiable end-to-end validation of the combined stack against real flown rovers (Pragyan as the headline target; Yutu-2 and the design-target rovers MoonRanger / Rashid-1 as cross-checks).

### 1. Introduction
- Coupled design optimisation for lunar micro-rovers: why it matters, why existing tools are proprietary or spreadsheet-based
- The capability-vs-utilisation confusion in the existing open literature
- The multi-fidelity gap: mission-level surrogates typically learn from a single fidelity
- Contribution statement

### 2. Background
- 2.1 Lunar micro-rover design heritage and trends
- 2.2 Terramechanics model fidelity hierarchy (Bekker-Wong, SCM, CRM, DEM)
- 2.3 Multi-fidelity surrogates (co-Kriging, residual / delta modeling, multi-fidelity neural networks)
- 2.4 ML surrogates in spacecraft/rover design — prior work and positioning
- 2.5 Capability-envelope engineering practice (JPL Team X, ESA CDF) and the gap in open literature

### 3. Capability-Envelope Framework
- 3.1 Design variables (12-dim) and why `drive_duty_cycle` is "designed duty," not operational utilisation
- 3.2 Mission metrics as capability-at-designed-duty (`range_km`, `energy_margin_raw_pct`, …)
- 3.3 Post-hoc `range_at_utilisation` rescaling and its domain of validity
- 3.4 Mission scenario library (equatorial mare, polar prospecting, highland slope, crater rim)

### 4. Mission Evaluator
- 4.1 Sub-model descriptions (terramechanics, solar/power, battery, mass, traverse, thermal)
- 4.2 Sub-model validation against published experimental data
- 4.3 Bottom-up mass model calibration against a flown-rover validation set (target MedAE ≤ 30% in-class)

### 5. Wheel-Level Multi-Fidelity Correction Architecture
- 5.1 Why correct at the wheel level rather than the mission level: data efficiency (12-d feature space, ~500 SCM runs) vs mission-level residual surrogate (>100-d feature space, would need 10× the SCM data); reusable across scenarios; physics-grounded; preserves the analytical traverse loop's semantics
- 5.2 SCM single-wheel sampling design: stratified-categorical LHS over the 12-d wheel-feature space, soil and grouser-class balanced, slip ∈ [0.05, 0.70]
- 5.3 Composition rule: `WheelForces_corrected = WheelForces_BW + Δ`, applied at the two `traverse_sim` injection points (slip-balance residual and `_mobility_power_w`); graceful fallback when the artifact is absent
- 5.4 W7.5 gate evidence: Bekker-Wong-only vs corrected mission outputs across 500 LHS samples; feasibility-flip rates and sign-flip rates per scenario family; gate decision threshold and outcome
- 5.5 W7.7 cross-validation: corrected-evaluator vs SCM-direct (PyChrono inside the traverse loop) on a shared sample; correction architecture matches SCM-direct mission outputs at 100× the speed
- 5.6 Per-layer error budget: wheel-level correction R² + RMSE per Δ target, mission-level propagation, baseline-physics residual against published wheel testbed data, and (in §8) end-to-end real-rover residual

### 6. Mission-Level Surrogate as an Acceleration and Uncertainty Layer
- 6.1 Why a mission-level surrogate at all (post-W7.7 reframe): inner-loop accelerator for NSGA-II, batch-mode sensitivity studies, calibrated prediction intervals, probabilistic feasibility — *not* a primary speed contribution since the corrected evaluator already runs at ~40 ms / mission
- 6.2 Baseline models (ridge, random forest, XGBoost, MLP) and two-stage feasibility classifier + regressor as canonical baselines for the Phase-5 benchmark
- 6.3 Hyperparameter tuning (Optuna TPE on XGBoost) and calibrated 90 % prediction intervals via quantile XGBoost
- 6.4 Accuracy results: aggregate and per-scenario, plus surrogate-vs-evaluator delta on the Phase-3 Pareto fronts
- 6.5 Registry-rover sanity check (Layer-1 at real operating points), with the design-axis primary set vs scenario-OOD diagnostic split

### 7. Tradespace Exploration
- 7.1 NSGA-II setup with the surrogate as the inner-loop fitness function (median XGBoost predictions for objectives, feasibility classifier as a probabilistic constraint), three-objective Pareto, Pareto-uncertainty bands from the quantile heads
- 7.2 Pareto fronts for the four mission scenarios, with surrogate-vs-corrected-evaluator deltas reported on each final front (the corrected evaluator is the source of truth; the surrogate is the search-loop accelerator)
- 7.3 SHAP-based design rules on the surrogate (mission-level), with the wheel-level correction contribution exposed as a sub-model trace through the corrected evaluator's per-step diagnostics
- 7.4 Cross-scenario insights: how optimal wheel size, drive duty, solar area, and battery sizing shift between equatorial mare, polar prospecting, highland slope, and crater rim missions

### 8. Validation Against Real Rovers
- 8.1 Layered error budget (Layers 1-4 from §7 of this plan)
- 8.2 Rediscovery test: optimizer vs Rashid, Pragyan
- 8.3 Sensitivity to `MassModelParams` and soil parameters

### 9. Discussion
- 9.1 Limitations: specific-mass calibration range (5-50 kg), SCM fidelity, gravity scaling, thermal simplification
- 9.2 When to trust the surrogate vs when to run the full evaluator
- 9.3 When SCM correction matters and when Bekker-Wong is sufficient (W7.5 gate outcome)
- 9.4 Comparison with existing trade study tools (Team X, CDF, FastFEMP, commercial MBSE)
- 9.5 Generalisation of the decomposition architecture to other terramechanics-constrained robots (agricultural, off-road, Mars rovers)

### 10. Conclusion
- Summary of contributions
- Open-source tool availability, pretrained surrogate release
- Future work: thermal as a design dimension, larger rover classes, higher-fidelity terramechanics, multi-rover mission design

### 11.1.1 W7.5 gate outcome (resolved 2026-04-25)

The W7.5 gate fired: the wheel-level correction produces 12-68 % feasibility-flip rates against BW-only across scenario families, comfortably above the 10 % action threshold (`reports/week7_5_gate/gate_decision.md`). The paper therefore takes the **"large correction"** framing: the wheel-level correction architecture recovers SCM-grade physics that Bekker-Wong misses in high-slip / soft-soil / grousered-wheel regimes, and the corrected mission evaluator becomes the source of truth for the rest of the methodology. The W7.7 bake-off against direct SCM-in-the-loop confirms the composed BW + correction pipeline matches SCM-direct mission outputs at 100× the speed.

(The original "null-result" framing — "Bekker-Wong is sufficient, here is where it isn't" — is preserved as a fallback in §11.2 paper 2 framing should the gate result change under future Bekker-Wong refinements; it does not currently apply.)

---

### 11.2 Paper 2 — Benchmark release (post-semester, Weeks 16-19)

**Target venues:** *NeurIPS Datasets & Benchmarks Track* (primary, June deadline if timing works; otherwise NeurIPS 2027), *IEEE Robotics and Automation Letters* benchmark track, or *ICRA* benchmark track.

### Title
RoverBench: An Open Benchmark for Mission-Level Design Surrogates of Lunar Micro-Rovers

### Pitch
Release the 40k LHS v4 dataset (corrected evaluator outputs — i.e. BW + wheel-level SCM correction), held-out test split, evaluation script, the wheel-level correction model itself as a callable artifact, and the Paper-1 mission-level surrogate as a baseline. Others train their own surrogates against the corrected-evaluator targets and submit predictions; we maintain a leaderboard on surrogate-vs-evaluator accuracy (R² / RMSE / MAPE on `range_km`, `energy_margin_raw_pct`, `slope_capability_deg`, feasibility AUC). The benchmark makes the multi-fidelity correction architecture of Paper 1 reproducible and competitive — and uniquely, it ships the high-fidelity (SCM-corrected) targets without requiring users to install PyChrono.

### Scope (deliberately narrow)
- **Prediction task only**, not a design-optimisation task. Given `(design, scenario)`, predict `MissionMetrics`. Scoring is held-out RMSE / R² / AUC, nothing more.
- **Dataset**: Week-6 LHS parquet + Week-7 correction subset, with a canonical train/val/test split frozen at dataset generation time.
- **Baselines**: linear, random forest, XGBoost, MLP, Paper-1 composed surrogate. Leaderboard scored on held-out test split.
- **Infrastructure**: GitHub repo with submission-format spec + eval script; a Hugging Face Dataset card + Space for optional interactive evaluation; manual leaderboard updates (no automated CI).

### Why D-narrow (not D-broad design-optimisation benchmark)
- Prediction tasks pull from a broad ML-surrogate / physics-informed-ML community (thousands of active researchers); design-optimisation benchmarks pull from a ~50-person community.
- Prediction benchmarks have a crisp scoring rubric that's not gameable; design-optimisation benchmarks require fairness adjudication.
- Prediction benchmarks require a parquet + eval script; design-optimisation benchmarks require a submission pipeline with input validation.
- The dataset exists as a byproduct of Paper 1; the benchmark is packaging work, not new science.

### Contributions
1. **The first open mission-level rover-design benchmark** with validated baselines and a held-out test split.
2. **Reference implementation** of the Paper-1 multi-fidelity surrogate as a leaderboard entry.
3. **Dataset documentation and citations** traceable per-column to primary sources (SMAD, AIAA, vendor catalogues, NASA Glenn, cited flown rovers).

See §6 "Phase 5: Benchmark Release" for the concrete week-by-week work.

---

## 12. Key Dependencies and Tools

| Tool | Purpose | Installation |
|------|---------|-------------|
| PyChrono | SCM simulation (optional, Path 2) | `conda install projectchrono::pychrono -c conda-forge` |
| scikit-learn | ML baselines | `pip install scikit-learn` |
| XGBoost | Gradient-boosted trees | `pip install xgboost` |
| PyTorch | Neural network surrogate | `pip install torch` |
| pymoo | Multi-objective optimization | `pip install pymoo` |
| SHAP | Explainability | `pip install shap` |
| Optuna | Hyperparameter tuning | `pip install optuna` |
| pyDOE2 | Latin Hypercube Sampling | `pip install pyDOE2` |
| matplotlib / plotly | Visualization | `pip install matplotlib plotly` |
| ipywidgets | Interactive notebooks | `pip install ipywidgets` |

---

## 13. Weekly Milestones Checklist

- [ ] Week 1: Bekker-Wong implemented and tested; PyChrono install in progress
- [ ] Week 2: Solar/power/battery models complete; PyChrono go/no-go decision made
- [ ] Week 3: Mass model and published rover database complete
- [ ] Week 4: Mission evaluator end-to-end functional
- [ ] Week 5: Real rover validation complete (critical gate)
- [ ] Week 6: Analytical dataset (40k rows, pilot-gated) generated with canonical split + SCHEMA.md + benchmark_score helper; baseline surrogates trained; feasibility classifier trained; registry-rover sanity check passed
- [ ] Week 7: SCM data generated (if active) or feature engineering complete
- [ ] Week 7.5: SCM correction-magnitude gate decided; ship composed surrogate vs bounded sensitivity recorded in project log
- [ ] Week 8: Final surrogate with uncertainty quantification
- [ ] Week 10: Tradespace sweep tool functional
- [ ] Week 11: NSGA-II Pareto fronts generated for all scenarios
- [ ] Week 12: SHAP analysis, rediscovery validation, tool packaged
- [ ] Week 13: All figures generated; Layer-3 BW-vs-Wong literature check replaces xfail; consolidated `reports/error_budget.md` complete
- [ ] Week 14: Paper 1 draft complete
- [ ] Week 15: Paper 1 revised and ready for submission
- [ ] Week 16 *(post-semester, gated)*: RoverBench dataset + challenge set + `benchmark_score` v1 API frozen
- [ ] Week 17 *(post-semester)*: Baselines packaged, submission interface and leaderboard live
- [ ] Week 18 *(post-semester)*: Paper 2 (benchmark release) draft complete
- [ ] Week 19 *(post-semester)*: Paper 2 submitted; Zenodo archive with DOI