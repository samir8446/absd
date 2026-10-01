# Self-updating digital twin for Li-ion battery diagnostics (v5.2)

Streamlit platform and Python engine for the internship study on NASA Ames 18650 LiCoO₂/graphite ageing
data. A physical cell and its virtual copy are synchronised cycle by cycle to diagnose ageing, forecast
remaining life and optimise operation and maintenance.

## Study objectives (Missions)

| Mission | Questions |
|---|---|
| M1 · Health | Which parameter best represents health? One or several? How do operating conditions and degradation mechanisms (LLI, LAM, CL) shape ageing? |
| M2 · Self-updating twin | Can future degradation be predicted accurately? Which variables are most informative? How often should the model update? |
| M3 · Operation & maintenance | Maximise Profit = Revenue − EnergyCost − MaintenanceCost (J_op = Revenue − EnergyCost; J_maint = MaintenanceCost + PerformanceLoss) by choosing the operating policy and the replacement time jointly. |

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py                     # "Load synthetic demo cohort" works without data
pytest                                   # 65 tests (pip install -r requirements-dev.txt)
python study.py --master data/battery_master_data.parquet --imp data/impedance.parquet --out results/study
python study.py --synthetic --out results/study_demo
pip install -r requirements-service.txt && uvicorn service:app --port 8000   # streaming REST service
```

Deploying on Streamlit Community Cloud: commit **`app.py` and `twin_engine.py` together** (the app checks
`REQUIRED_ENGINE`), Python 3.12, `requirements.txt` without comments, then *Reboot*. Cached results are keyed
on the engine version, so engine updates always recompute.

## Files

| File | Role |
|---|---|
| `twin_engine.py` | All computation, no UI (~6,000 lines, numbered sections) |
| `app.py` | Streamlit front end (5 views) |
| `study.py` | Offline cohort study → CSVs, `summary.md`, manifest |
| `service.py` | Streaming twin: framework-free `TwinRegistry` + optional FastAPI endpoints |
| `benchmark.py` | Offline forecast-origin sweep (ML, Twin, PINN, HB) |
| `tests/test_twin_engine.py` | 65 tests: gradient checks, synthetic-truth recovery, every model and study function |

## Views

1. **Diagnostics.** Scope switch: single battery, selected batteries (up to 8) or whole fleet. Contents:
   - health status: gauges, alerts, risk matrix, triage, event log, HTML report;
   - fade comparison and raw telemetry (5 cycle-selection modes);
   - cohort by ambient temperature;
   - health-indicator ranking and PCA (M1);
   - ICA/DVA and LLI/LAM/CL proxies;
   - half-cell OCV fitting (quantitative LLI, LAM_PE, LAM_NE);
   - stress and safety exposure.
2. **Live twin.** A battery is streamed one discharge at a time (play, pause, step) through:
   - the ECM twin (dual EKF);
   - the mechanistic particle filter (SEI · plating · LAM, driven by measured temperature and current);
   - the particle filter on the physics power law;
   - the adaptive trend Kalman filter;
   - hierarchical Bayes;
   - a live ensemble weighted by each model's recent 5-cycle-ahead error.

   The view also shows live accuracy per model, mechanism shares, the equations, the measurement ablation
   and the update-frequency study (M2).
3. **Models & forecasting: a learning ladder.** A training scheme at the top applies to every level:
   *within a battery* (train on its first part, test on the rest; optionally also learn from the other batteries)
   or *across batteries* (train on chosen batteries, forecast one or more test batteries after seeing the start
   of each). The leaderboard averages over the test batteries:
   - Level 1 · Baselines: persistence, linear trend, decision tree, Bayesian ridge.
   - Level 2 · Classical ML: random forest, extra trees, Gaussian process.
   - Level 3 · Boosting: histogram GB; XGBoost and LightGBM when installed (`requirements-ml.txt`).
   - Level 4 · Deep learning: GRU and single-head Transformer (numpy autodiff, no extra dependency).
   - Level 5 · Hybrid & physics-informed: mechanistic PINN, hierarchical Bayes.
   - Level 6 · First principles: single-particle electrochemical model with SEI growth (`spm_discharge`,
     `spm_forecast`).
   Earlier ML features kept below the ladder:
   - Four curated models: Gaussian Process, Extra Trees, Hist. Gradient Boosting and Bayesian Ridge.
   - Two tasks: forecasting (fade-rate model, conformal bands) and estimating SOH from indicators (random, chronological or by-battery splits).
   - Leakage-free auto-tuning.
   - Early-life ΔQ(V) lifetime model (Severson et al. 2019).
   - Cross-cell benchmark.
4. **Operations & control (M3).**
   - Plant calibrated on the selected battery (`calibrate_plant`).
   - Twin-aware policy, with energy cost included.
   - Integrated operation + replacement optimisation (renewal-reward with sudden-failure hazard).
   - Dynamic-programming policy over (SOH, season), solved with Dinkelbach's method.
   - What-if scenario planner.
5. **Study results.** Runs the live twin on every usable battery and reports per condition group:
   - forecast error, accuracy and band coverage;
   - paired Wilcoxon tests (ensemble vs twin);
   - mechanism-physics checks;
   - the update interval;
   - the plant calibration.

   It states the answer to each Mission question and exports HTML and CSV.

## Key engine API

| Area | Functions / classes |
|---|---|
| Data | `ParquetStore`, `build_cycle_table`, `robust_bol_capacity`, `condition_groups` |
| Diagnostics | `rank_health_indicators`, `hi_pca`, `stress_factor_regression`, `ica_evolution`, `dva_evolution`, `degradation_modes`, `half_cell_trajectory`, `detect_knee`, `fleet_status`, `fleet_events` |
| Twin | `run_dual_twin` (`DualTwinConfig`: `capacity_every`, `voltage_n_eff`, `adaptive`, `adaptive_q`), `twin_config_for` (pulse-aware), `twin_forecast`, `twin_ablation`, `update_frequency_study` |
| Live / Bayesian | `live_multi_model`, `MechanisticStream`, `hierarchical_bayes_forecast`, `live_skill_table`, `live_forecast_accuracy`, `StreamingTwin` |
| ML | `MODEL_SPECS`, `train_ml_forecast`, `train_soh_estimator`, `tune_ml_forecast`, `tune_soh_estimator`, `early_life_lifetime` |
| PINN (offline) | `train_pinn`, `MechanisticPINN` (optional `mode_targets` from half-cell LLI/LAM) |
| Study | `cohort_validation`, `cohort_summary`, `paired_model_test`, `mechanism_checks`, `calibrate_plant` |
| Operations | `CellPhysics`, `Economics`, `MaintenanceModel`, `integrated_om_study`, `solve_replacement_dp`, `evaluate_dp_policy`, `scenario_projection` |
| Testing | `make_synthetic_master(..., knee=...)` with known ground truth |

## Data

- **Master parquet** (one row per sample): `Cell_ID, Cycle_Index, Cycle_Type (charge/discharge/impedance), Time_s, Voltage_V, Current_A, Temp_C, Capacity_Ah[, Ambient_C]`.
- **Impedance parquet:** `Re_ohm, Rct_ohm` per EIS test.
- **End of life:** one definition, the control-bar EOL capacity (default 1.4 Ah, the NASA convention). The Operations replacement SOH is a separate, optimised decision.
- **Known dataset issues, handled:**
  - B0049–B0056: crashed logging, repaired baseline, separate group;
  - B0025–B0028: square-wave load, pulse-aware twin;
  - erratic 4 °C runs: outlier flags.

## v5.1 fixes (from the first real-data study)
Condition groups (explicit crashed-logging ids, new "Mixed conditions" group with load/ambient levels),
no duplicate forecasts, best model chosen on common cells with coverage flags, per-cell paired tests,
power labels for small groups, EOL = k consecutive cycles below threshold with "EOL not reachable in
data" status, twin NIS consistency guard. Root cause of twin divergence on B0033-B0044: rate-dependent
capacity under mixed loads (open: rate-normalised SOH). LiCoO2 potential restricted to its valid domain
(y >= 0.48), which also fixes a latent half-cell-fit bound.

## Known limits

- **Synthetic evidence.** Most performance numbers so far come from synthetic cohorts. The study results on the real NASA data are the evidence to report.
- **Mechanism attribution.** It is model-based; validate it against half-cell LLI/LAM.
- **Hot knees.** Hot cells with a sudden knee remain hard for every model; the live ensemble helps most there.
- **Mission 3 simplifications.** Mission 3 uses a lumped plant and an assumed sudden-failure hazard, so read the location of the optimum rather than its absolute value.

## References

- Plett (EKF twins)
- Dubarry et al. 2012; Birkl et al. 2017 (degradation modes, half-cell fitting)
- Ramadass et al. 2004; Doyle et al. 1996 (electrode potentials)
- Severson et al. 2019 (early-life ΔQ)
- Saha & Goebel 2009 (particle filter on NASA cells)
- Saxena et al. 2010 (prognostic metrics)
- Coble & Hines 2009 (health-indicator criteria)
- Rufino Júnior et al. 2024; Menye et al. 2025 (degradation reviews)
