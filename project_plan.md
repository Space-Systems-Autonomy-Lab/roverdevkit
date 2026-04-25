# RoverDevKit: ML-Accelerated Co-Design of Mobility and Power Subsystems for Lunar Micro-Rovers
## Semester Project Plan

---

## 1. Project Summary

**Title:** RoverDevKit — ML-Accelerated Co-Design of Mobility and Power Subsystems for Lunar Micro-Rovers

**One-sentence pitch:** Build an open-source mission-level rover evaluator that chains terramechanics, solar power, and traverse simulation models, train an ML surrogate over the coupled wheel-chassis-power design space, and use it to explore mission-relevant Pareto fronts validated against the published design points of real lunar micro-rovers.

**Core contributions:**

1. An open-source, fully-documented mission evaluator for lunar micro-rovers that takes a design vector and a mission profile and returns mission-level performance metrics (traverse range, energy margin, slope capability, mass) with every sub-model traceable to a cited source.
2. A multi-fidelity ML surrogate over the coupled mobility-power design space that enables interactive tradespace exploration and multi-objective optimization at millisecond latency.
3. Validation that the optimizer rediscovers the design points of real lunar micro-rovers (Rashid, Pragyan, Yutu-class) within stated tolerances when given matching mission constraints, plus interpretable design rules extracted via SHAP that generalize across mission profiles.

**Why this is novel:** Existing rover surrogate work is almost entirely component-level (single wheel, single subsystem). System-level rover trade studies exist but are either proprietary (JPL Team X, ESA CDF) or use static spreadsheet models without ML. No open-source tool combines a physics-based mission evaluator, an ML surrogate over the coupled design space, and validation against real flown rovers. The "rediscover Rashid/Pragyan" validation is a concrete, falsifiable claim that's rare in this literature.

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
│                  ML SURROGATE LAYER                             │
│                                                                 │
│  Multi-fidelity training:                                       │
│    - 50,000 cheap analytical evaluations (Bekker-Wong path)    │
│    - 2,000 PyChrono SCM corrections (high-fidelity path)       │
│  Models: gradient-boosted trees (primary), NN ensemble (UQ)    │
│  Output: predicted mission metrics + uncertainty               │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│              TRADESPACE EXPLORATION LAYER                       │
│                                                                 │
│  - Parametric sweeps (interactive Jupyter widgets)              │
│  - NSGA-II Pareto front generation with surrogate-in-the-loop  │
│  - SHAP-based design rule extraction                           │
│  - Mission scenario library (polar, equatorial, mare, highland)│
└────────────────────────────────────────────────────────────────┘
```

The key architectural choice is that the **mission evaluator is the primary artifact**, and the surrogate is a fast approximation of it. This matters because it means the evaluator can stand alone as a useful open-source tool even if the ML doesn't pan out, and because every ML claim is grounded in a specific physics model the reviewer can inspect.

---

## 3. Design Space and Mission Scenarios

### 3.1 Design Variables (12 dimensions)

| Category | Variable | Symbol | Range | Units |
|----------|----------|--------|-------|-------|
| Mobility | Wheel radius | R | 0.05–0.20 | m |
| Mobility | Wheel width | W | 0.03–0.15 | m |
| Mobility | Grouser height | h_g | 0–0.012 | m |
| Mobility | Grouser count | N_g | 0–24 | int |
| Mobility | Number of wheels | N_w | {4, 6} | int |
| Chassis | Chassis dry mass | m_c | 3–35 | kg |
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

**Path 1 — Analytical (the safety net):** Run the full mission evaluator using only Bekker-Wong terramechanics. Cost: ~10–50 ms per evaluation. Generate 50,000 LHS samples in an afternoon. This dataset alone is sufficient to train a usable surrogate and write a publishable paper.

**Path 2 — SCM correction layer (the upgrade):** Run PyChrono SCM single-wheel simulations at ~2,000 strategically-chosen points in the design space. Don't try to cover the whole space with SCM — focus on regions where Bekker-Wong is known to disagree with experiments (high slip, grousered wheels, soft soil, sloped terrain). Train a *correction model* that predicts the delta between SCM and Bekker-Wong, and apply that correction inside the mission evaluator. Cost: ~30 seconds per SCM run × 2,000 runs / 5 parallel processes ≈ 3.5 hours of compute, but plan for ~2 weeks of intermittent batch runs to account for debugging, reruns, and not pinning the laptop at 100% continuously.

**Path 3 — Experimental validation (the credibility):** Curated single-wheel testbed data from published papers. Not used for training. Used only to validate that Path 1 + Path 2 produces realistic numbers.

This three-path structure makes the project robust on a laptop. If PyChrono never works, publish with Path 1 + Path 3 and frame the paper as "tradespace exploration methodology validated against experiments." If PyChrono works, add Path 2 and frame as "multi-fidelity tradespace exploration with simulator-grounded surrogate." Both are publishable. The fallback isn't a degraded version of the project — it's a different paper of similar quality.

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

### Phase 2: Data Generation and Surrogate (Weeks 6–9)

**Week 6: Analytical dataset (Path 1) and baseline surrogates.**

The original plan assumed ~50 ms per evaluation giving "50,000 LHS samples in an afternoon." Measured cost is ~1.4 s per evaluation (1.6 s on long equatorial missions, 3.2 s on polar), roughly 30× slower. The schedule, dataset size, and surrogate architecture below are adjusted for that reality and for Week-5/5.5/5.6 carry-over (unclipped energy margin, capability-envelope framing, lowered duty-cycle floor).

- **LHS sampling.** Stratified by `n_wheels ∈ {4, 6}` (only 2 levels, naive LHS undersamples). Continuous variables sampled via `scipy.stats.qmc.LatinHypercube`; integer variables (`grouser_count`) rounded from continuous samples; `n_wheels` split 50/50 across strata. Scenario parameters (latitude, Bekker soil params c / φ / kc / kφ / n, mission duration, max slope) sampled jointly so the model learns a single cross-scenario function instead of four per-scenario models. Record seed and sampler config in dataset metadata.
- **Dataset size.** Start with a **pilot run of 2k samples** (~5 min with 8 workers) to verify the pipeline and catch schema / NaN / multiprocessing bugs cheaply. Full run targets **10k per scenario × 4 scenarios = 40k rows** (~2 h with 8 workers). Scale to 20k/scenario only if the pilot surrogate misses the R² targets on range or energy margin.
- **Dataset schema (extensibility for W7/W8).** Each row stores: design vector (12 fields), scenario parameters as continuous features (not one-hot of simulant names), raw `MissionMetrics`, and a `fidelity: "analytical"` tag so SCM-corrected runs can be appended later if needed. Also store aggregate sub-model statistics (peak / mean / P95 of wheel drawbar pull, sinkage, motor torque, solar power, battery SOC). These stats serve two purposes downstream: (a) they're the most informative features for the Week-7.5 correction surrogate, which learns `Δmission_metric = f(design, scenario, wheel-regime)`; (b) they enable sub-model-level SHAP and failure-mode diagnostics in the Week-12 design-rule analysis. Written as Parquet to `data/analytical/lhs_v1.parquet`.
- **Regression targets.** `range_km` (capability at designed duty, per Week 5.6), `energy_margin_raw_pct` (unclipped, per Week 5.5 — the plan originally specified the clipped `energy_margin_pct` but that saturates across much of the design space), `slope_capability_deg`, `total_mass_kg`. The clipped reporting metric is derived post-hoc from the raw prediction.
- **Feasibility: single-target classifier + regressor (Week-6 step-2 scope cut).** The boolean `motor_torque_ok` is trained as the feasibility classifier (XGBoost / logistic), with the regressor trained on the feasible subset. This gives honest feasibility probabilities for the Week-11 NSGA-II constraint layer and keeps the regressor from wasting capacity on the `range_km = 0` failure mode. **Thermal scope:** `thermal_survival` is no longer a surrogate target. The current mass model treats RHU power and MLI quality as free design choices, so `thermal_survival` reduces to a near-trivial gate ("did you add an RHU?") with no design trade-off. The system-level evaluator still computes it as a diagnostic (preserved for Pragyan/Yutu-2 distinction in the Week-5 validation harness), and a future mass-model upgrade that charges RHU/MLI mass will let thermal re-enter the surrogate as a real Pareto target. See `data/analytical/SCHEMA.md` v1→v2 notes.
- **Baselines.** Ridge linear, random forest, XGBoost, small MLP. Multi-output across the four regression targets. 80/10/10 train/val/test split stratified by scenario.
- **Evaluation.** Report R² / RMSE / MAPE per target, broken out both aggregate and per-scenario (catches cases where the model is great on equatorial and terrible on polar). AUC / F1 for the feasibility classifier. Plus a **registry-rover sanity check**: predict metrics for Pragyan / Yutu-2 / Sojourner against their registry design vectors and compare surrogate predictions to the evaluator's own predictions (Layer-1, not Layer-4). Guards against the surrogate doing well on IID LHS but being wrong exactly where we validate.
- **Target accuracy (unchanged from original plan).** R² > 0.95 for range and raw energy margin; R² > 0.85 for slope capability; AUC > 0.90 for feasibility.
- **Benchmark-release hooks (near-zero marginal cost; unlock Paper 2 cheaply).** Three design decisions to lock in at dataset-generation time so the Phase-5 benchmark release isn't a retrofit:
  1. **Canonical train/val/test split stored as a column in the parquet** (not re-split at training time). Seeded; deterministic; the same 10% is "the test set" for every downstream paper.
  2. **Evaluation script as a public library function**: `roverdevkit.surrogate.benchmark_score(predictions_df: pd.DataFrame) -> BenchmarkReport`. Takes predictions, returns R² / RMSE / MAPE / AUC per target and per scenario.
  3. **Versioned schema documentation** in `data/analytical/SCHEMA.md`: columns, types, units, citations. Required for any future dataset-release paper; trivial to write at generation time, painful to retrofit.
- **Deliverable:** `roverdevkit/surrogate/{sampling,dataset,baselines,metrics}.py` + `benchmark_score` helper, `data/analytical/lhs_v1.parquet` with version metadata and canonical split column, `data/analytical/SCHEMA.md`, `notebooks/02_baseline_surrogates.ipynb` with the aggregate + per-scenario accuracy table and registry sanity check, CI gates on sampling reproducibility, split stability, and a pilot-scale fit smoke test.

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

**Week 8: Final surrogate with uncertainty quantification.**
- Whichever architecture the Week-7.5 gate chose: hyperparameter tuning with Optuna, add uncertainty quantification (quantile regression for XGBoost, MC dropout or small deep ensemble for the MLP), calibrate prediction intervals, check that 90% PIs cover ~90% of held-out test points.
- If the composed surrogate is in play, calibrate PIs on the composed output, not on the baseline surrogate alone.
- **Deliverable:** Final surrogate (analytical-only or composed) with calibrated uncertainty and a clear decision record on the multi-fidelity question.

**Week 9: External validation against published experimental data.**
- Layer the validations explicitly:
  - Bekker-Wong vs published single-wheel data (does the analytical model match physical experiments?).
  - SCM vs published single-wheel data, if applicable.
  - Surrogate vs Bekker-Wong (ML fidelity to its training data).
  - Surrogate vs SCM, if applicable.
  - Full mission evaluator vs published rover traverse data (this was already started in week 5, formalize it here).
- Write up the error budget honestly: "the surrogate predicts mission range with ±X% error relative to the analytical evaluator, which itself agrees with published wheel testbed data to within ±Y%, giving end-to-end error of ±Z% on real rover comparison."
- **Deliverable:** `validation/` directory with all comparison plots and a layered error budget.

### Phase 3: Tradespace Tool (Weeks 10–12)

**Week 10: Parametric sweeps and constraint handling.**
- Build the sweep engine: user fixes some variables, sweeps others, evaluator runs in surrogate mode for speed.
- Implement constraint checking: motor torque limits, mass budget, volume envelope, slope capability minimums.
- Build interactive Jupyter notebook with widgets for live tradespace exploration.
- **Deliverable:** `tradespace/sweeps.py` + `notebooks/01_interactive_exploration.ipynb`.

**Week 11: NSGA-II optimization.**
- Wrap pymoo NSGA-II around the surrogate. Three-objective Pareto: maximize range, minimize mass, maximize slope capability. Constraints from week 10.
- Run for all four mission scenarios.
- Generate Pareto front visualizations.
- **Deliverable:** Pareto fronts for all four scenarios with the constraint-feasible region highlighted.

**Week 12: Design rules and the rediscovery validation.**
- SHAP analysis on the trained surrogate. Generate global feature importance and partial dependence plots for the key design variables.
- Extract interpretable design rules: "below 15 kg total mass, 4-wheel configurations dominate; above, 6-wheel becomes Pareto-optimal because [reason]" — this kind of statement.
- **The rediscovery test:** Set up the optimizer with constraints matching Rashid's mission (mass budget ≤10 kg, equatorial-ish, short traverse). Does the optimizer produce a design in the neighborhood of actual Rashid? Do the same for Pragyan with its constraints. Plot both real rovers on the Pareto fronts.
- This is the headline validation result for the paper. If the optimizer's "best" Rashid-class design has wheel diameter within ~30% of actual Rashid, mass within ~25%, and lands on or near the Pareto front, there's a strong story. If it's wildly different, either explain why convincingly (the optimizer found a better design — defend that claim) or debug the evaluator.
- **Deliverable:** Design rule summary, rediscovery validation plots, packaged tool with README.

### Phase 4: Paper (Weeks 13–15)

**Week 13:** All publication figures, results compilation, fill any remaining gaps.

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
- Finalise the public dataset release: `data/analytical/lhs_v1.parquet` (or v2 if scaled/regenerated), with canonical split, schema doc, and per-row citation/provenance metadata.
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

### Layer 1: Surrogate vs Mission Evaluator (ML Fidelity)
**Question:** Does the surrogate accurately reproduce the mission evaluator's predictions?
**Method:** Standard ML holdout evaluation on 10% test set.
**Acceptance criteria:** R² > 0.95 for range and energy margin; R² > 0.85 for slope capability.

### Layer 2: Bekker-Wong vs SCM (Analytical vs Higher-Fidelity)
**Question:** Where does the analytical terramechanics model agree with PyChrono SCM, and where does it disagree?
**Method:** Compare both models across the design space, identify systematic differences.
**Expected result:** General agreement in trends, with known discrepancies at high slip and on grousered wheels (where SCM's 3D contact handling matters).

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

**Minimum viable Paper 1** (SCM fails, W7.5 says "report as bounded sensitivity," Phase-5 deferred): analytical-only mission evaluator with capability-envelope framing, single-fidelity surrogate with two-stage feasibility model, validation against published wheel data and the Week-5 flown-rover registry (Rashid / Pragyan / Yutu-2 / Sojourner), parametric sweeps, NSGA-II Pareto fronts, SHAP-based design rules. Frame around contributions #2 (capability-envelope framework) and #3 (open-source tool) with #1 (decomposition architecture) written as "we present the decomposition architecture and quantify that correction magnitude is small enough in this regime to report as a sensitivity." Publishable at IEEE Aerospace, AIAA SciTech, or *Acta Astronautica*.

**Full vision Paper 1** (SCM works, W7.5 ships composed surrogate): multi-fidelity evaluator with PyChrono SCM corrections, composed surrogate `final = analytical + correction` with uncertainty-aware prediction intervals, per-layer error budget, rediscovery validation against two flown rovers, cross-scenario design rules, packaged open-source tool with pretrained models. Publishable at *Journal of Field Robotics*, *International Journal of Robotics Research*, or *Journal of Spacecraft and Rockets*.

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
│   │   ├── lhs_v1.parquet            # Canonical 40k-row dataset + canonical split column
│   │   ├── SCHEMA.md                 # Versioned column spec, units, citations
│   │   └── challenge_v1.parquet      # ~200-design corner-case set (Phase 5)
│   └── scm/                          # PyChrono SCM runs (if Path 2 active)
│
├── roverdevkit/
│   ├── terramechanics/
│   │   ├── bekker_wong.py            # Analytical model
│   │   ├── scm_wrapper.py            # PyChrono SCM single-wheel runner
│   │   └── correction_model.py       # ML correction layer
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
│   ├── surrogate/
│   │   ├── sampling.py               # LHS sampler (stratified on n_wheels)
│   │   ├── dataset.py                # Parallel dataset builder, split helpers, Parquet I/O
│   │   ├── features.py
│   │   ├── baselines.py              # Linear, RF, XGBoost, MLP; feasibility classifier
│   │   ├── models.py                 # Composed multi-fidelity surrogate (Phase 2 final)
│   │   ├── metrics.py                # R²/RMSE/MAPE, AUC/F1, per-scenario breakdowns
│   │   ├── benchmark_score.py        # Public leaderboard metric API (Phase 5)
│   │   ├── train.py
│   │   └── uncertainty.py
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
├── notebooks/
│   ├── 00_real_rover_validation.ipynb      # Week 5
│   ├── 01_interactive_exploration.ipynb
│   ├── 02_baseline_surrogates.ipynb        # Week 6 results
│   ├── 03_pareto_fronts.ipynb
│   ├── 04_rediscover_real_rovers.ipynb
│   └── 05_reproduce_paper.ipynb
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
Capability-Envelope Tradespace Exploration for Lunar Micro-Rovers using a Multi-Fidelity Surrogate with Layered Validation

### Abstract (~250 words)
- **Problem.** Coupled wheel-power-mass trades for lunar micro-rovers are high-dimensional and currently done with proprietary tools (JPL Team X, ESA CDF) or static spreadsheet models. Existing open-source work is either component-level (single wheel) or conflates hardware capability with operational utilisation, producing range predictions that disagree with flown rovers by 5-10×.
- **Approach.** We introduce a **decomposition architecture** for multi-fidelity mission-level surrogates in which the baseline analytical model, a high-fidelity SCM correction, and the surrogate ML fidelity contribute **separately attributable** error sources. We also formalise a **capability-envelope vs operational-utilisation** distinction that separates what the hardware can sustain from what ops schedules actually command.
- **Key result 1 (methodology).** Decomposition architecture achieves R² > 0.95 for mission range and energy margin, with each error layer independently validated against wheel testbed data, published rover traverses, and ML holdout.
- **Key result 2 (framing).** The capability-envelope metric reproduces Pragyan/Yutu-2/Sojourner hardware bounds at published ranges; operational utilisation is exposed as a post-hoc rescaling (`range_at_utilisation`) rather than a design variable.
- **Key result 3 (rediscovery).** Optimizer rediscovers Rashid and Pragyan design points within stated tolerances when given matching mission constraints.
- **Key result 4 (SCM verdict).** Quantified when high-fidelity SCM corrections materially move mission-level answers vs when Bekker-Wong is sufficient, resolving an open question for practitioners.
- **Deliverable.** Open-source tool (`roverdevkit`) with pretrained surrogates, the mission evaluator, and the four-scenario library. Data and baselines released.

### Contributions (in priority order)
1. **Multi-fidelity decomposition architecture** with separately-attributable error sources (analytical baseline + SCM correction surrogate + mission-level surrogate, each with its own validation layer and error budget).
2. **Capability-envelope framing** that explicitly separates hardware-sustainable performance from ops-commanded utilisation, with a post-hoc rescaling (`range_at_utilisation`) for ops queries.
3. **Open-source mission evaluator and pretrained surrogates** (not just a paper) validated against flown rover data.
4. **Rediscovery test** as a falsifiable end-to-end validation of the combined stack against real flown rovers (Rashid, Pragyan; Yutu-2 / Sojourner as cross-checks).

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

### 5. Multi-Fidelity Decomposition Architecture
- 5.1 Analytical baseline: LHS (40k+) over 12-dim design space crossed with 4 scenarios
- 5.2 SCM wheel-level correction model: when and why Bekker-Wong is weak
- 5.3 Mission-level correction surrogate: residual modeling of `Δmetric = f(design, scenario)`
- 5.4 Composed architecture: `final = analytical_surrogate + correction_surrogate`
- 5.5 Per-layer error budget: surrogate fidelity, correction magnitude, baseline physics, real-rover residual

### 6. Surrogate Training and Uncertainty
- 6.1 Baseline models (ridge, random forest, XGBoost, MLP), two-stage feasibility classifier + regressor
- 6.2 Hyperparameter tuning (Optuna), calibrated prediction intervals
- 6.3 Accuracy results: aggregate and per-scenario
- 6.4 Registry-rover sanity check (Layer-1 at real operating points)

### 7. Tradespace Exploration
- 7.1 NSGA-II setup, three-objective Pareto, constraints including the feasibility classifier
- 7.2 Pareto fronts for the four mission scenarios
- 7.3 SHAP-based design rules; decomposed SHAP attribution into baseline + correction contributions
- 7.4 Cross-scenario insights

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

### 11.1.1 Null-result framing (W7.5 gate)

Both outcomes of the W7.5 correction-magnitude gate are publishable with equivalent framing:
- **Large correction (> 10% median |Δrange|):** "Multi-fidelity decomposition recovers an SCM correction that Bekker-Wong misses; we quantify the correction per scenario."
- **Small correction (< 10%):** "Bekker-Wong at mission level is sufficient within X% for the micro-rover regime at operational slopes. We quantify *where* high-fidelity physics matters and where it does not, saving practitioners unnecessary SCM runs."

The paper's §5, §8, and §9 sections are written so either outcome slots in without structural changes. The decomposition architecture is the contribution; the correction magnitude is evidence about its utility in one domain.

---

### 11.2 Paper 2 — Benchmark release (post-semester, Weeks 16-19)

**Target venues:** *NeurIPS Datasets & Benchmarks Track* (primary, June deadline if timing works; otherwise NeurIPS 2027), *IEEE Robotics and Automation Letters* benchmark track, or *ICRA* benchmark track.

### Title
RoverBench: An Open Benchmark for Mission-Level Design Surrogates of Lunar Micro-Rovers

### Pitch
Release the 40k (50k if scaled) LHS dataset, held-out test split, evaluation script, and Paper-1 surrogate as a baseline. Others train their own surrogates and submit predictions; we maintain a leaderboard on surrogate-vs-evaluator accuracy (R² / RMSE / MAPE on `range_km`, `energy_margin_raw_pct`, `slope_capability_deg`, feasibility AUC). The benchmark makes the multi-fidelity methodology claim of Paper 1 reproducible and competitive.

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
- [ ] Week 9: Layered validation complete with error budget
- [ ] Week 10: Tradespace sweep tool functional
- [ ] Week 11: NSGA-II Pareto fronts generated for all scenarios
- [ ] Week 12: SHAP analysis, rediscovery validation, tool packaged
- [ ] Week 13: All figures generated
- [ ] Week 14: Paper 1 draft complete
- [ ] Week 15: Paper 1 revised and ready for submission
- [ ] Week 16 *(post-semester, gated)*: RoverBench dataset + challenge set + `benchmark_score` v1 API frozen
- [ ] Week 17 *(post-semester)*: Baselines packaged, submission interface and leaderboard live
- [ ] Week 18 *(post-semester)*: Paper 2 (benchmark release) draft complete
- [ ] Week 19 *(post-semester)*: Paper 2 submitted; Zenodo archive with DOI