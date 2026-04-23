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
│  Constraint: thermal_survival (binary), motor_torque_ok        │
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

**Mass model.** Parametric mass estimating relationships for chassis, wheels, motors, solar panels, battery, harness, structure. Most MERs come from proprietary databases; the publishable version builds MERs from the small set of published lunar micro-rovers (Rashid, Pragyan, Yutu-1/2, CADRE, Sojourner, MARSOKHOD prototypes, Lunokhod). n=8 is small. Document this explicitly as a limitation. The point of the surrogate isn't to predict mass to ±1% — it's to capture the *trends* (bigger wheels weigh more in a known way).

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

**Week 3: Mass model and parametric MERs.**
- Build a small database of published lunar micro-rover specs (Rashid, Pragyan, Yutu-1/2, CADRE, Sojourner, Lunokhod, plus 1–2 academic concepts). Spreadsheet with mass breakdown by subsystem.
- Fit simple parametric MERs (linear or power-law) for each subsystem mass as a function of design variables.
- Document the small-n caveat explicitly in `mass_model.py` docstring.
- **Deliverable:** `mass_model.py` + `data/published_rovers.csv` with citations.

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
- LHS sample 50,000 design vectors across all four mission scenarios. Run the analytical evaluator. This should complete in a few hours.
- Train baselines: linear regression, random forest, XGBoost, small NN. Multi-output models predicting all mission metrics simultaneously.
- 80/10/10 train/val/test split, evaluate R² and RMSE per output.
- **Deliverable:** Baseline surrogate accuracy table. Target: R² > 0.95 on range and energy margin (these are smooth functions of inputs); R² > 0.85 on slope capability (more nonlinear).

**Week 7: PyChrono SCM data generation (Path 2) — if active.**
- If PyChrono is working: write the SCM single-wheel simulation wrapper, generate 2,000 strategically-sampled SCM runs (focus on grousered wheels, high slip, sloped terrain). Use `multiprocessing` with 4–5 parallel workers, run overnight and on weekends.
- If PyChrono is not working: skip ahead to physics-informed feature engineering instead. Engineer features like dimensionless sinkage `z/R`, effective contact area `R*W`, grouser volume fraction, etc. These boost accuracy of the analytical surrogate measurably.
- **Deliverable:** Either 2,000 SCM runs in `data/scm/` or a feature-engineered analytical surrogate with improved accuracy.

**Week 8: Multi-fidelity surrogate (or refined single-fidelity).**
- If SCM data is available: train a correction model that predicts (SCM output − Bekker-Wong output) as a function of inputs. Compose: `final_prediction = analytical + correction`. This is a much easier learning problem than learning SCM from scratch.
- If not: do hyperparameter tuning with Optuna on the analytical surrogate, add uncertainty quantification (quantile regression for trees, MC dropout or deep ensembles for NNs).
- Either way: implement and calibrate prediction intervals. Check that 90% PIs cover ~90% of test points.
- **Deliverable:** Final surrogate with calibrated uncertainty.

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
**Method:** Perturb soil parameters and MER coefficients by ±20%, re-run optimization, check if qualitative conclusions persist.
**Expected result:** Qualitative conclusions stable, exact optimal dimensions shift.

---

## 8. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PyChrono fails to install or SCM doesn't work on M2 | Medium | Medium | Path 2 becomes optional; paper still works with Path 1 + Path 3. Gate at end of week 2. |
| Mission evaluator can't reproduce real rover behavior in Week 5 | Medium | High | Most likely cause is mass model — the n=8 MER fits will be loose. Mitigation: report parametric sensitivity to MER assumptions; loosen tolerances on the rediscovery validation. If still bad, narrow scope to mobility-power tradespace only and drop the "rediscover real rovers" framing. |
| Published rover specs are too sparse to fit MERs | Low | Medium | Augment with academic concept rovers and JPL/ESA published study reports. CADRE and VIPER design papers are particularly useful. |
| Surrogate accuracy too low on mission-level metrics | Low | Medium | Mission metrics are smoother than wheel-level metrics, so this is less of a risk than in a wheel-only project. If it happens, narrow the design space or add more training data. |
| SHAP design rules turn out to be obvious / boring | Medium | Low | Frame the contribution as "first quantitative validation of engineering intuition" rather than "novel design insights." Still publishable. |
| Reviewers say "this is just systems engineering with a fancier search algorithm" | Medium | Medium | Lean hard on (a) the open-source tool contribution, (b) the rediscovery validation, (c) the multi-fidelity training methodology. Don't oversell the ML novelty — the contribution is integration and validation, not new ML methods. |
| MacBook thermal throttling during overnight SCM runs | Medium | Low | Use `caffeinate -i`, run with laptop elevated for airflow, accept 50–70% sustained throughput in compute estimates. |

---

## 9. Minimum Viable Paper vs Full Vision

**Minimum viable** (everything that's at risk goes wrong): Bekker-Wong-only mission evaluator, single-fidelity surrogate, validation against published wheel data and one or two real rovers, parametric sweeps with simple constraint handling, basic SHAP analysis. Frame as "Open-source tradespace exploration framework for lunar micro-rover co-design." Publishable at IEEE Aerospace, AIAA SciTech, or *Acta Astronautica*.

**Full vision** (everything works): Multi-fidelity evaluator with PyChrono SCM corrections, uncertainty-aware surrogate, NSGA-II Pareto fronts for four mission scenarios, rediscovery validation against three real rovers, design rules with cross-scenario comparisons, packaged open-source tool with pretrained models and tutorial notebooks. Publishable at *Journal of Field Robotics*, *Journal of Spacecraft and Rockets*, or *Acta Astronautica* with stronger framing.

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
│   ├── analytical/                   # Generated 50k LHS samples
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
│   │   └── parametric_mers.py        # MERs from published rovers
│   │
│   ├── mission/
│   │   ├── evaluator.py              # Top-level mission evaluator
│   │   ├── scenarios.py              # Four mission scenarios as configs
│   │   └── traverse_sim.py           # Time-stepped traverse loop
│   │
│   ├── surrogate/
│   │   ├── train.py
│   │   ├── models.py                 # GBT, NN, multi-fidelity composition
│   │   ├── features.py
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
│   ├── 01_interactive_exploration.ipynb
│   ├── 02_pareto_fronts.ipynb
│   ├── 03_rediscover_real_rovers.ipynb
│   └── 04_reproduce_paper.ipynb
│
├── pretrained/
│   └── default_surrogate.pkl         # Ships with the package
│
└── tests/
    ├── test_terramechanics.py
    ├── test_power.py
    ├── test_mission_evaluator.py
    └── test_surrogate.py
```

The pretrained surrogate ships with the package so users can do tradespace exploration without installing PyChrono.

---

## 11. Paper Outline

**Target venue:** *Journal of Field Robotics*, *Journal of Spacecraft and Rockets*, or *Acta Astronautica* (full vision); IEEE Aerospace or AIAA SciTech (minimum viable).

### Title
RoverDevKit: ML-Accelerated Co-Design of Mobility and Power for Lunar Micro-Rovers

### Abstract (~250 words)
- Problem: Coupled wheel-power-mass trades for lunar micro-rovers are high-dimensional and currently done with proprietary tools or static spreadsheets
- Approach: Open-source mission evaluator + multi-fidelity ML surrogate + NSGA-II optimization
- Key result 1: Surrogate achieves R² > 0.95 at 10,000× speedup over the evaluator
- Key result 2: Optimizer rediscovers Rashid and Pragyan design points within stated tolerances
- Key result 3: SHAP analysis reveals [specific cross-scenario design insight]
- Key result 4: Tool released as open-source software with pretrained models

### 1. Introduction
- Importance of coupled design optimization for lunar micro-rovers
- Current state: proprietary tools (Team X, CDF) or manual spreadsheet trades
- Gap: no open-source, ML-accelerated tradespace tool with verifiable mission-level evaluator
- Contribution statement

### 2. Background
- 2.1 Lunar micro-rover design heritage and trends
- 2.2 Terramechanics models and their tradeoffs (Bekker-Wong, SCM, CRM, DEM)
- 2.3 ML surrogates in spacecraft/rover design — prior work and positioning
- 2.4 Multi-objective optimization for engineering design

### 3. Mission Evaluator
- 3.1 Design variable parameterization
- 3.2 Sub-model descriptions (terramechanics, power, mass, traverse)
- 3.3 Mission scenario definitions
- 3.4 Validation of individual sub-models

### 4. Multi-Fidelity Surrogate
- 4.1 Data generation strategy (LHS, analytical and SCM paths)
- 4.2 Surrogate architecture and training
- 4.3 Uncertainty quantification
- 4.4 Surrogate accuracy results

### 5. Tradespace Exploration
- 5.1 NSGA-II setup and constraints
- 5.2 Pareto fronts for the four mission scenarios
- 5.3 SHAP-based design rules
- 5.4 Cross-scenario insights

### 6. Validation Against Real Rovers
- 6.1 Mission evaluator vs published rover traverse data
- 6.2 Rediscovery test: optimizer vs Rashid, Pragyan
- 6.3 Sensitivity to MER and soil parameter uncertainty

### 7. Discussion
- 7.1 Limitations: MER small-n, SCM fidelity, gravity scaling, thermal simplification
- 7.2 When to trust the surrogate vs when to run full evaluator
- 7.3 Implications for real mission design
- 7.4 Comparison with existing trade study tools

### 8. Conclusion
- Summary of contributions
- Open-source tool availability
- Future work: thermal as a design dimension, larger rover classes, higher-fidelity terramechanics, multi-rover mission design

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
- [ ] Week 6: Analytical dataset generated, baseline surrogates trained
- [ ] Week 7: SCM data generated (if active) or feature engineering complete
- [ ] Week 8: Final surrogate with uncertainty quantification
- [ ] Week 9: Layered validation complete with error budget
- [ ] Week 10: Tradespace sweep tool functional
- [ ] Week 11: NSGA-II Pareto fronts generated for all scenarios
- [ ] Week 12: SHAP analysis, rediscovery validation, tool packaged
- [ ] Week 13: All figures generated
- [ ] Week 14: Paper draft complete
- [ ] Week 15: Paper revised and ready for submission