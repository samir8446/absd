"""Tests for twin_engine: correctness (autodiff, metrics), synthetic-truth recovery
(dual twin, pooled Arrhenius), regression guards (ML extrapolation, bands, censoring),
reliability (hashing, validation, configs) and an end-to-end benchmark smoke test.

Run with ``pytest -q`` or, without pytest, ``python tests/test_twin_engine.py``."""
from __future__ import annotations

import functools
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import twin_engine as te  # noqa: E402

K_TRUE = (4e-4, 5e-4, 3.5e-4, 6e-4, 4.5e-4, 5e-4)
AMBIENTS = (24.0, 34.0, 43.0, 24.0, 34.0, 43.0)


@functools.lru_cache(maxsize=1)
def synthetic():
    master, imp, truth = te.make_synthetic_master(n_cells=6, n_cycles=90, ambients=AMBIENTS,
                                                  k_true=K_TRUE, seed=1)
    store = te.ParquetStore.from_dataframe(master)
    ct = te.build_cycle_table(store)
    return store, ct, imp, truth


# ------------------------------------------------------------------ autodiff --
def test_autodiff_matches_finite_differences():
    rng = np.random.default_rng(0)
    W = te.Tensor(rng.normal(size=(3, 4)))
    b = te.Tensor(rng.normal(size=(1, 4)))
    c = te.Tensor(rng.normal(size=(4, 1)))
    x = rng.normal(size=(5, 3))

    def loss():
        h = (te.Tensor(x) @ W + b).tanh()
        y = (h @ c).softplus() + (h @ c).sigmoid() * 0.5 + (h @ c).asinh()
        z = (y / (1.0 + y.square())).exp() - (y + 2.0).log()
        return z.square().mean()

    assert te.gradient_check(loss, [W, b, c]) < 1e-5


def test_pinn_states_derivative_matches_finite_difference():
    net = te.HybridPINN(te.PINNConfig(seed=3), 2.0, 0.045, 0.07, 150.0)
    n = np.array([10.0, 60.0, 120.0])
    h = 1e-4
    s = net.states(n)
    up, dn = net.states(n + h), net.states(n - h)
    for key in ("SOH", "R_int", "R_ct"):
        fd = (up[key].data - dn[key].data).ravel() / (2 * h)
        assert np.allclose(s["d" + key].data.ravel(), fd, rtol=1e-5, atol=1e-10), key


# ------------------------------------------------------------------- features --
def test_cycle_table_from_synthetic():
    _, ct, _, truth = synthetic()
    assert set(ct["Cell_ID"]) == set(truth)
    assert ct.groupby("Cell_ID")["n"].max().eq(90).all()
    first = ct.sort_values("n").groupby("Cell_ID")["SOH"].first()
    assert np.allclose(first, 1.0, atol=0.01)            # robust (smoothed) beginning-of-life baseline
    assert (ct.groupby("Cell_ID")["cum_Ah"].diff().dropna() > 0).all()
    assert {"outlier", "regen", "R_dc_ohm", "C_bol_Ah"} <= set(ct.columns)


def test_build_cycle_table_reports_bad_cells():
    store, _, _, _ = synthetic()
    master = store._frame.copy()
    bad = master[master["Cell_ID"] == "S001"].copy()
    bad["Cell_ID"] = "BAD"
    bad["Cycle_Type"] = "charge"                       # no discharge cycles at all
    errors = {}
    ct = te.build_cycle_table(te.ParquetStore.from_dataframe(pd.concat([master, bad])), errors=errors)
    assert "BAD" in errors and "BAD" not in set(ct["Cell_ID"])


def test_validate_master_flags_problems():
    store, _, _, _ = synthetic()
    df = store.cell_frame("S001")
    assert te.validate_master(df) == []
    broken = df.copy()
    broken.loc[broken.index[:500], "Voltage_V"] = 12.0
    assert any("Voltage_V" in m for m in te.validate_master(broken))


# ----------------------------------------------------------------- observers --
def test_dual_twin_recovers_degradation_rate_and_tracks_soh():
    store, ct, imp, truth = synthetic()
    cell = "S004"
    meta = te.cell_meta(ct)
    res = te.run_dual_twin(store.cell_frame(cell), ct, imp, cell, te.TwinParameters(),
                           te.DualTwinConfig(), te.similar_cells(meta, cell))
    k_est, k_true = res.per_cycle["k_ah"].iloc[-1], truth[cell]["k_ah"]
    assert abs(k_est / k_true - 1) < 0.2
    merged = res.per_cycle.merge(truth[cell]["trajectory"], on="n")
    rmse = float(np.sqrt(np.mean((merged["SOH"] - merged["SOH_true"]) ** 2)))
    assert rmse < 0.01
    assert res.state_cov.shape == (len(res.per_cycle), 4, 4)


def test_voltage_feedback_beats_open_loop():
    store, ct, imp, _ = synthetic()
    cell = "S004"
    meta = te.cell_meta(ct)
    tab, _ = te.twin_ablation(store.cell_frame(cell), ct, imp, cell, te.TwinParameters(), None,
                              te.similar_cells(meta, cell), 36, 0.7)
    open_loop = tab.loc["Open loop (population prior)", "Tracking RMSE"]
    voltage = tab.loc["Voltage (partial window)", "Tracking RMSE"]
    assert voltage < 0.5 * open_loop


def test_twin_forecast_band_and_rul_samples():
    store, ct, imp, _ = synthetic()
    cell = "S006"
    meta = te.cell_meta(ct)
    res = te.run_dual_twin(store.cell_frame(cell), ct, imp, cell, te.TwinParameters(), None,
                           te.similar_cells(meta, cell))
    fc = te.twin_forecast(res, 30, 135, 0.8, level=0.9)
    fut = fc.n_grid > 30
    assert np.all(fc.lo[fut] <= fc.soh[fut] + 1e-12) and np.all(fc.soh[fut] <= fc.hi[fut] + 1e-12)
    assert np.all(np.diff(fc.hi[fut] - fc.lo[fut]) >= -1e-3)          # band widens with horizon
    assert np.isfinite(fc.rul_samples).mean() > 0.5


# ------------------------------------------------------------------------ ML --
def test_increment_ml_extrapolates_beyond_training_horizon():
    _, ct, _, _ = synthetic()
    r = te.train_ml_forecast(ct, "S004", 36, "Random Forest", strategy="increment")
    n_train_max = 90
    assert r.soh_pred[-1] < r.soh_pred[n_train_max - 1] - 0.01       # keeps fading past n = 90
    assert np.all(np.diff(r.soh_pred[36:]) <= 1e-12)                  # monotone forecast


def test_conformal_band_brackets_point_forecast():
    _, ct, _, _ = synthetic()
    r = te.train_ml_forecast(ct, "S002", 30, "Gradient Boosting", conformal_cells=3)
    assert r.soh_lo is not None and len(r.calibration_cells) == 3
    assert np.all(r.soh_lo <= r.soh_pred + 1e-12) and np.all(r.soh_pred <= r.soh_hi + 1e-12)
    assert r.metrics.coverage is not None and 0 <= r.metrics.coverage <= 1


# ------------------------------------------------------------------- metrics --
def test_forecast_metrics_censoring_and_alpha_lambda():
    n = np.arange(1, 101)
    y = 1 - 0.002 * n                                   # crosses 0.8 at n = 100 -> not before end
    m = te.forecast_metrics(n, y, n, y, 50, soh_eol=0.7)
    assert m.censored and m.rul_true is None and m.rul_true_lb == 50
    y2 = 1 - 0.004 * n                                  # crosses 0.8 after n = 50
    pred = 1 - 0.0042 * n
    m2 = te.forecast_metrics(n, y2, n, pred, 20, soh_eol=0.8, alpha=0.2)
    assert not m2.censored and m2.rul_true is not None
    assert m2.alpha_lambda_ok is True and 0.8 <= m2.rel_accuracy <= 1.0
    lo, hi = pred - 0.01, pred + 0.01
    m3 = te.forecast_metrics(n, y2, n, pred, 20, 0.8, lo, hi)
    assert m3.coverage is not None and abs(m3.band_width - 0.02) < 1e-9


def test_prognostic_horizon():
    bench = pd.DataFrame({"cell": "A", "paradigm": "X", "n0": [10, 20, 30, 40],
                          "rul_pred": [200, 45, 70, 58], "eol_true": [100, 100, 100, 100]})
    ph = te.prognostic_horizon(bench, alpha=0.2)
    assert float(ph["PH_cycles"].iloc[0]) == 100 - 30               # enters and stays from n0 = 30


# ------------------------------------------------------------------------ PINN --
def test_pooled_arrhenius_brackets_true_activation_energy():
    _, ct, _, _ = synthetic()
    est = te.estimate_pooled_arrhenius(ct, n_boot=200)
    assert est["identifiable"] and est["n_cells"] == 6
    assert est["ci_lo"] < 30e3 < est["ci_hi"]


def test_pinn_fixed_ea_is_not_optimised():
    _, ct, imp, _ = synthetic()
    r = te.train_pinn(ct, imp, "S001", 30, te.PINNConfig(epochs=50, ea_mode="fixed", ea_fixed_J_mol=42e3))
    assert abs(r.physics["Ea_kJ_mol"] - 42.0) < 1e-9


def test_pinn_ensemble_returns_band_and_status_table():
    _, ct, imp, _ = synthetic()
    r = te.train_pinn_ensemble(ct, imp, "S001", 30, te.PINNConfig(epochs=60), seeds=(0, 1, 2))
    assert r.n_members == 3 and r.soh_lo is not None
    assert r.physics_table.loc["Ea_kJ_mol", "status"] == "fixed"


# ---------------------------------------------------------------- operations --
def test_perturb_physics_and_plant_mismatch():
    p = te.CellPhysics(dt_s=60.0)
    rng = np.random.default_rng(0)
    assert te.perturb_physics(p, 0.0, rng) is p
    q = te.perturb_physics(p, 0.3, rng)
    assert q.k_ah != p.k_ah and q.ocv_offset_V != 0.0
    e = te.Economics()
    life = te.simulate_life(te.make_policy("Fixed 2 A"), p, e, max_cycles=30, plant=q)
    assert len(life) == 30 and life["SOH"].is_monotonic_decreasing


def test_mismatch_study_measures_against_compliant_baselines():
    df = te.mismatch_study(te.Economics(), te.CellPhysics(), levels=(0.0,), n_draws=1,
                           policies=("Twin-Aware", "Fixed 2 A"))
    assert {"advantage_pct", "best_compliant", "twin_violations"} <= set(df.columns)


# --------------------------------------------------------------- reliability --
def test_persist_upload_hashes_full_content(tmp_path=None):
    import tempfile
    d = tmp_path or tempfile.mkdtemp()
    body = b"x" * (2 << 20)
    a = b"PAR1" + body + b"A" + b"PAR1"
    b = b"PAR1" + body + b"B" + b"PAR1"             # identical first MB and length
    pa, pb = te.persist_upload(a, "a.parquet", d), te.persist_upload(b, "b.parquet", d)
    assert pa != pb


def test_config_validation():
    for bad in (lambda: te.TwinParameters(sigma_v=-1), lambda: te.PINNConfig(ea_mode="guess"),
                lambda: te.Economics(currents_A=(1.0, 2.0), price_per_Ah=(1.0,)),
                lambda: te.DualTwinConfig(voltage_window_frac=1.5),
                lambda: te.BenchmarkConfig(paradigms=("Oracle",)), lambda: te.CellPhysics(C_bol_Ah=0)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("invalid configuration accepted")


def test_manifest_is_json_serialisable():
    man = te.run_manifest({"cfg": te.BenchmarkConfig(), "arr": np.arange(3), "nan": float("nan")}, "key")
    txt = json.dumps(man)
    assert "engine_version" in man and man["config"]["nan"] is None and "numpy" in txt


# ---------------------------------------------------------------- end-to-end --
def test_small_benchmark_runs():
    store, ct, imp, _ = synthetic()
    cfg = te.BenchmarkConfig(fracs=(0.4,), paradigms=("ML", "Twin"), conformal_cells=2, eol_ah=1.6)
    bench, resid = te.run_benchmark(store, ct, imp, cfg, cells=["S002", "S006"])
    assert len(bench) == 4 and bench["error"].isna().all()
    summ = te.benchmark_summary(bench)
    assert "α-λ hit rate" in summ.columns and len(te.coverage_by_horizon(resid)) > 0


def test_compare_paradigms_end_to_end():
    store, ct, imp, _ = synthetic()
    res = te.compare_paradigms(store.cell_frame("S003"), ct, imp, "S003", 0.4, te.TwinParameters(),
                               te.PINNConfig(epochs=80), "Ridge", conformal_cells=2, pinn_seeds=(0, 1))
    assert not res.errors, res.errors
    assert set(res.bands) == set(res.predictions)
    assert te.TWIN_NAMES["dual"] in res.rul_samples


# ------------------------------------------------ v4.1: Mission 1 diagnostics --
def test_charge_side_features_track_ageing():
    _, ct, _, _ = synthetic()
    d = ct[ct["Cell_ID"] == "S004"].sort_values("n")
    for col in ("t_cc_s", "t_cv_s", "E_ch_Wh", "E_dis_Wh", "eff_energy"):
        assert d[col].notna().mean() > 0.9, col
    assert d["t_cc_s"].iloc[-5:].mean() < d["t_cc_s"].iloc[:5].mean()      # less charge accepted in CC
    assert d["t_cv_s"].iloc[-5:].mean() > d["t_cv_s"].iloc[:5].mean()      # longer CV tail (resistance)
    assert (d["eff_energy"].between(0.5, 1.0)).all()


def test_health_indicator_ranking_and_pca():
    _, ct, imp, _ = synthetic()
    tab = te.rank_health_indicators(ct, imp)
    assert {"Capacity_Ah", "R_dc_ohm", "Rct_ohm"} <= set(tab.index)
    for c in ("|ρ| with SOH", "Monotonicity", "Trendability", "Prognosability", "Fitness"):
        assert tab[c].between(0, 1).all(), c
    assert tab.loc["Capacity_Ah", "LOCO RMSE (SOH)"] < 0.01
    pca = te.hi_pca(ct, imp)
    assert pca["available"] and abs(pca["explained"].sum() - 1) < 1e-9
    assert pca["explained"][0] > 0.5                                        # one dominant ageing direction


def test_knee_detection():
    n = np.arange(1, 161, dtype=float)
    kinked = np.where(n < 100, 1 - 1e-3 * n, 0.9 - 4e-3 * (n - 100))
    rng = np.random.default_rng(0)
    k = te.detect_knee(n, kinked + 0.002 * rng.standard_normal(len(n)))
    assert k["found"] and abs(k["knee_n"] - 100) <= 8 and k["ratio"] > 2
    lin = te.detect_knee(n, 1 - 1.5e-3 * n + 0.002 * rng.standard_normal(len(n)))
    assert not lin["found"]


def test_dva_and_degradation_modes():
    store, ct, imp, _ = synthetic()
    prep = te.prepare_cell(store.cell_frame("S004"))
    ctc = ct[ct["Cell_ID"] == "S004"]
    dva = te.dva_evolution(prep, ctc, 4)
    assert len(dva) >= 3 and all(np.all(c.dvdq >= 0) for c in dva)
    assert dva[-1].capacity_Ah < dva[0].capacity_Ah
    modes = te.degradation_modes(te.ica_evolution(prep, ctc, 5), ctc, te.valid_eis(imp, "S004"))
    assert {"LLI (proxy)", "LAM (proxy)", "CL: R_dc growth", "CL: EIS Rₑ+R_ct growth"} <= set(modes.columns)
    last = modes.iloc[-1]
    assert abs(last["LLI (proxy)"] + last["LAM (proxy)"] - last["Capacity loss"]) < 1e-6 or last["LLI (proxy)"] == 0
    assert last["CL: EIS Rₑ+R_ct growth"] > 0


def test_stress_exposure_and_condition_regression():
    _, ct, _, _ = synthetic()
    cold = ct.copy()
    cold.loc[cold["Cell_ID"] == "S001", "T_ch_C"] = 2.0
    sx = te.stress_exposure(cold)
    assert sx.loc[sx["Cell_ID"] == "S001", "plating"].all()
    assert (sx.loc[sx["Cell_ID"] == "S001", "PRI"] > 0).all()
    assert not sx.loc[sx["Cell_ID"] == "S002", "plating"].any()
    summ = te.stress_summary(ct)
    assert len(summ) == 6 and summ.max().max() <= 100 + 1e-9
    reg = te.stress_factor_regression(ct)
    assert reg["available"] and reg["coefficients"].loc["Ea (kJ/mol)", "Estimate"] > 0


# ------------------------------------------------- v4.1: prognostic paradigms --
def test_semi_empirical_recovers_linear_throughput_law():
    _, ct, _, _ = synthetic()
    f = te.semi_empirical_forecast(ct, "S004", 36, eol_ah=1.6)
    assert 0.7 < f.params["z"] < 1.3                                        # synthetic fade is linear in Ah
    fut = f.n_grid > 36
    assert np.all(f.lo[fut] <= f.soh[fut] + 1e-12) and np.all(f.soh[fut] <= f.hi[fut] + 1e-12)
    assert f.metrics.rmse < 0.02


def test_particle_filter_forecast():
    _, ct, _, _ = synthetic()
    f = te.particle_filter_forecast(ct, "S004", 36, eol_ah=1.6, n_particles=1500)
    assert f.params["prior_cells"] >= 3 and f.metrics.rmse < 0.04
    assert f.rul_samples is not None and len(f.rul_samples) > 100
    fut = f.n_grid > 36
    assert np.all(f.lo[fut] <= f.hi[fut])


def test_gaussian_process_surrogate():
    _, ct, _, _ = synthetic()
    r = te.train_ml_forecast(ct, "S004", 36, "Gaussian Process", eol_ah=1.6, conformal_cells=0)
    assert np.isfinite(r.metrics.rmse) and r.metrics.rmse < 0.05


def test_compare_paradigms_includes_new_paradigms_without_pinn():
    store, ct, imp, _ = synthetic()
    res = te.compare_paradigms(store.cell_frame("S003"), ct, imp, "S003", 0.4, te.TwinParameters(),
                               te.PINNConfig(epochs=20), "Ridge", conformal_cells=0, run_pinn=False)
    assert te.SEMI_NAME in res.predictions and te.PF_NAME in res.predictions
    assert te.PINN_NAME not in res.predictions and te.PINN_NAME not in res.errors
    assert {te.SEMI_NAME, te.PF_NAME} <= set(res.rul_samples)


# --------------------------------------------------- v4.1: Mission 2 studies --
def test_update_frequency_study_and_information_gain():
    store, ct, imp, _ = synthetic()
    soh_eol = te.soh_eol_for(float(ct[ct["Cell_ID"] == "S004"]["C_bol_Ah"].iloc[0]), 1.6)
    tab, res = te.update_frequency_study(store.cell_frame("S004"), ct, imp, "S004", te.TwinParameters(), None,
                                         36, soh_eol, intervals=(1, 5, 30))
    assert tab.loc[1, "Updates per 100 cycles"] > tab.loc[5, "Updates per 100 cycles"] > tab.loc[30, "Updates per 100 cycles"]
    assert tab.loc[30, "Tracking RMSE"] > tab.loc[1, "Tracking RMSE"]
    assert tab.loc[30, "Info gain per update (nats)"] > tab.loc[1, "Info gain per update (nats)"] > 0
    assert tab.attrs["recommended"] in (1, 5, 30)
    try:
        te.DualTwinConfig(update_every=0)
        raise AssertionError("update_every=0 accepted")
    except ValueError:
        pass


# --------------------------------------------------- v4.1: Mission 3 studies --
def test_energy_accounting_in_operations():
    p, e = te.CellPhysics(dt_s=60.0), te.Economics(energy_price_per_Wh=0.02)
    pred = te.predict_cycle(np.array([1.0, p.R_int0, p.R_ct0]), [1.0, 4.0], 20.0, p)
    assert np.all(pred["e_in"] > pred["e_out"]) and np.all(pred["e_out"] > 0)
    assert pred["e_out"][0] / pred["e_in"][0] > pred["e_out"][1] / pred["e_in"][1]   # higher current, lower eff.
    life = te.simulate_life(te.make_policy("Fixed 2 A"), p, e, max_cycles=40)
    s = te.summarise_life(life)
    assert abs(s["J_op"] - (s["revenue"] - s["energy_cost"])) < 1e-9 and 0.5 < s["energy_eff"] < 1


def test_maintenance_hazard_and_integrated_study():
    mm = te.MaintenanceModel()
    h = mm.hazard(np.array([0.95, 0.75, 0.68, 0.6]))
    assert np.all(np.diff(h) > 0) and h[0] < 1e-3
    e, p = te.Economics(), te.CellPhysics()
    study = te.integrated_om_study(e, p, mm, weights=(0.0, 1.0), fixed_currents=(2.0,),
                                   thresholds=(0.62, 0.70, 0.80))
    assert set(study["policy"]) == {"Twin-aware · w = 0", "Twin-aware · w = 1", "Fixed 2 A"}
    for _, d in study.groupby("policy"):
        d = d.sort_values("threshold")
        assert d["p_failure"].is_monotonic_decreasing                          # replacing earlier = less risk
        assert d["cycles"].is_monotonic_decreasing
    opt = te.integrated_optimum(study)
    assert opt["best"]["violations"] == 0
    r = study.iloc[0]
    assert abs(r["profit"] - (r["J_op"] - r["maintenance_cost"])) < 1e-6


def test_benchmark_with_new_paradigms():
    store, ct, imp, _ = synthetic()
    cfg = te.BenchmarkConfig(fracs=(0.4,), paradigms=("SemiEmp", "PF"), eol_ah=1.6)
    bench, _ = te.run_benchmark(store, ct, imp, cfg, cells=["S002"])
    assert set(bench["paradigm"]) == {te.SEMI_NAME, te.PF_NAME} and bench["error"].isna().all()


def test_cycle_index_dtype_from_foreign_parquet():
    """Colab/pyarrow files may store Cycle_Index as int32 or float64 (EIS too): ingestion and
    every EIS-aligned analysis must work regardless."""
    m, imp, _ = te.make_synthetic_master(n_cells=3, n_cycles=30, seed=3)
    for dt in ("int32", "float64"):
        mm, ii = m.copy(), imp.copy()
        mm["Cycle_Index"] = mm["Cycle_Index"].astype(dt)
        ii["Cycle_Index"] = ii["Cycle_Index"].astype(dt)
        store = te.ParquetStore.from_dataframe(mm)
        ct = te.build_cycle_table(store)
        assert ct["Cell_ID"].nunique() == 3 and ct["t_cc_s"].notna().mean() > 0.9
        assert te.attach_eis(ct, ii)["Rct_ohm"].notna().any()
        prep = te.prepare_cell(store.cell_frame("S001"))
        ctc = ct[ct["Cell_ID"] == "S001"]
        modes = te.degradation_modes(te.ica_evolution(prep, ctc, 4), ctc, te.valid_eis(ii, "S001"))
        assert "CL: EIS Rₑ+R_ct growth" in modes.columns


def test_ingestion_error_names_the_cause():
    m, _, _ = te.make_synthetic_master(n_cells=2, n_cycles=10, seed=1)
    m = m.assign(Capacity_Ah=np.nan)
    try:
        te.build_cycle_table(te.ParquetStore.from_dataframe(m))
        raise AssertionError("no error raised")
    except te.DataError as exc:
        assert "Per-cell reasons" in str(exc)


def test_robust_baseline_repairs_crashed_logging_segment():
    """B0049-B0056 style: an invalid low start followed by an upward level shift."""
    m, _, _ = te.make_synthetic_master(n_cells=3, n_cycles=40, seed=1)
    dis = m[(m["Cell_ID"] == "S002") & (m["Cycle_Type"] == "discharge")]
    first12 = sorted(dis["Cycle_Index"].unique())[:12]
    m.loc[(m["Cell_ID"] == "S002") & m["Cycle_Index"].isin(first12), "Capacity_Ah"] *= 0.3
    ct = te.build_cycle_table(te.ParquetStore.from_dataframe(m))
    good = ct[~ct["outlier"]]
    assert good["SOH"].max() < 1.02
    assert ct.loc[ct["Cell_ID"] == "S002", "outlier"].sum() == 12
    assert ct.loc[ct["Cell_ID"] == "S002", "baseline_suspect"].all()
    assert not ct.loc[ct["Cell_ID"] == "S001", "baseline_suspect"].any()


def test_model_registry_all_models_and_param_validation():
    X = np.random.default_rng(0).normal(size=(60, 3))
    y = X[:, 0] - 0.5 * X[:, 1] ** 2
    assert len(te.ML_MODELS) >= 10
    for name in te.ML_MODELS:
        m = te.make_model(name, 0, te.default_params(name)).fit(X, y)
        assert np.all(np.isfinite(m.predict(X))), name
    for bad in ({"n_estimators": 5}, {"nope": 1}):
        try:
            te.validate_params("Random Forest", bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    assert te.validate_params("MLP", {"hidden": "32, 16"})["hidden"] == "32,16"


def test_ml_forecast_hyperparams_and_cross_battery():
    _, ct, _, _ = synthetic()
    f = te.train_ml_forecast(ct, "S004", 36, "Extra Trees", eol_ah=1.6, model_params={"n_estimators": 60})
    assert np.isfinite(f.metrics.accuracy) and f.metrics.accuracy > 95 and f.metrics.fade_skill > 0.5
    x = te.train_ml_forecast(ct, "S004", 10, "Ridge", eol_ah=1.6, train_cells=["S005"])
    assert np.isfinite(x.metrics.rmse)
    try:
        te.train_ml_forecast(ct, "S004", 10, "Ridge", train_cells=["S004"])
        raise AssertionError("target-only training list accepted")
    except ValueError:
        pass


def test_soh_estimator_splits_and_leakage_guard():
    _, ct, imp, _ = synthetic()
    for split, kw in (("random", {}), ("chronological", {}),
                      ("by_cell", {"train_cells": ["S001", "S002", "S003"], "test_cells": ["S004"]})):
        r = te.train_soh_estimator(ct, imp, "Random Forest", split=split, test_frac=0.3,
                                   params={"n_estimators": 60}, **kw)
        assert set(r.metrics.index) == {"train", "test"} and r.metrics.loc["test", "Cycles"] >= 3
        assert r.metrics.loc["test", "R²"] > 0 and len(r.importance) == len(r.features)
    r = te.train_soh_estimator(ct, imp, "Ridge", split="by_cell", train_cells=["S001", "S002"], test_cells=["S006"])
    assert set(r.predictions.loc[r.predictions["set"] == "test", "Cell_ID"]) == {"S006"}
    try:
        te.train_soh_estimator(ct, imp, "Ridge", features=["Capacity_Ah", "R_dc_ohm"])
        raise AssertionError("capacity leakage accepted")
    except ValueError:
        pass


def test_mechanistic_pinn_gradients_and_mechanisms():
    _, ct, imp, _ = synthetic()
    cfg = te.PINNConfig(epochs=1, physics="mechanistic", hidden=6, n_colloc=12)
    ctc = ct[ct["Cell_ID"] == "S004"]
    data = te.pinn_training_data(ctc, te.valid_eis(imp, "S004"), 36, 120, cfg)
    net = te.MechanisticPINN(cfg, 2.0, 0.045, 0.07, 120.0)
    assert te.gradient_check(lambda: net.total(net.losses(data)), net.parameters) < 1e-5
    r = te.train_pinn(ct, imp, "S004", 36, te.PINNConfig(epochs=300, physics="mechanistic"), eol_ah=1.6)
    mech = r.mechanisms
    assert {"Q_SEI", "Q_plating", "Q_LAM"} <= set(mech.columns) and (mech[["Q_SEI", "Q_plating", "Q_LAM"]] >= 0).all().all()
    assert np.allclose(1 - mech[["Q_SEI", "Q_plating", "Q_LAM"]].sum(axis=1), r.soh, atol=1e-9)
    no_pl = te.train_pinn(ct, imp, "S004", 36, te.PINNConfig(epochs=50, physics="mechanistic", use_plating=False))
    assert np.allclose(no_pl.mechanisms["Q_plating"], 0)
    try:
        te.PINNConfig(physics="mechanistic", use_sei=False, use_plating=False, use_lam=False)
        raise AssertionError("no-mechanism config accepted")
    except ValueError:
        pass


if __name__ == "__main__":                            # minimal runner when pytest is absent
    failures = 0
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:                      # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
