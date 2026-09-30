# Battery digital twin & operando diagnostics (v4)

```bash
pip install -r requirements.txt
streamlit run app.py                      # "Load synthetic demo cohort" works with no data
pip install -r requirements-dev.txt && pytest
python benchmark.py --master data/master.parquet --imp data/impedance.parquet --out results/bench.parquet
python benchmark.py --synthetic --fracs 0.3 0.5 --paradigms ML Twin   # ground-truth sanity run
docker build -t battery-twin . && docker run -p 8501:8501 -v twin-data:/data battery-twin
```

Files: `twin_engine.py` (all computation, no UI), `app.py` (Streamlit views), `benchmark.py`
(offline sweep → results file the app can load), `tests/` (22 tests incl. gradient checks and
synthetic-truth recovery). Set `TWIN_CACHE_DIR` to persist downloads and uploads; `GIT_COMMIT`
is recorded in run manifests.

Known limits: activation energy is not identifiable per cell (fixed or pooled by default);
split-conformal bands under-cover when the target cell is unlike the calibration cells;
the PINN ensemble band is epistemic only; the one-step policy optimises per-cycle profit,
not lifetime profit rate (needs DP over SOH).
