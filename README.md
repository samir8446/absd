# Battery digital twin & operando diagnostics (v4.1)

```bash
pip install -r requirements.txt
streamlit run app.py                      # "Load synthetic demo cohort" works with no data
pip install -r requirements-dev.txt && pytest
python benchmark.py --master data/master.parquet --imp data/impedance.parquet --out results/bench.parquet
python benchmark.py --synthetic --fracs 0.3 0.5 --paradigms ML Twin SemiEmp PF   # ground-truth sanity run
docker build -t battery-twin . && docker run -p 8501:8501 -v twin-data:/data battery-twin
```

Files: `twin_engine.py` (all computation, no UI), `app.py` (Streamlit views), `benchmark.py`
(offline sweep → results file the app can load), `tests/` (35 tests incl. gradient checks and
synthetic-truth recovery). Set `TWIN_CACHE_DIR` to persist downloads and uploads; `GIT_COMMIT`
is recorded in run manifests.

## What answers which project question

| Mission question | Where in the app | Engine function |
|---|---|---|
| M1 · Which parameter best represents health? | Data › health indicators | `rank_health_indicators` |
| M1 · One parameter or several? | Data › PCA card | `hi_pca` |
| M1 · How do operating conditions influence degradation? | Data › cohort expander, stress & safety | `stress_factor_regression`, `stress_exposure` |
| M1 · Degradation mechanisms (LLI / LAM / CL) | Data › ICA, DVA, modes | `ica_evolution`, `dva_evolution`, `degradation_modes`, `detect_knee` |
| M2 · Can future degradation be predicted accurately? | Models › comparison, cross-cell benchmark | `compare_paradigms`, `run_benchmark` |
| M2 · Which variables are most informative? | Models › ablation (information gain) | `twin_ablation` |
| M2 · How often should the model update? | Models › update frequency | `update_frequency_study` |
| M3 · Integrated operation + maintenance | Operations › integrated optimisation | `integrated_om_study`, `renewal_evaluation` |

Forecasting paradigms: ML fade-rate surrogates (RF, GBR, GPR, SVR, MLP, Ridge) with conformal
bands; dual-EKF ECM twin; semi-empirical power law in Ah; particle filter (double exponential);
hybrid PINN (Butler–Volmer, SEI-type fade law).

Known limits: activation energy is not identifiable per cell (fixed or pooled by default);
split-conformal bands under-cover when the target cell is unlike the calibration cells; the PINN
ensemble band is epistemic only; LLI/LAM values are proxies from ~1C full-cell data; the
sudden-failure hazard in the O&M study is an assumed model, so read the optimum's location rather
than its absolute value; the one-step policy optimises per-cycle margin, not lifetime rate (DP is
future work).
