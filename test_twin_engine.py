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
from pathlib import Path
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
    r = te.train_ml_forecast(ct, "S004", 36, "Extra Trees", strategy="increment")
    n_train_max = 90
    assert r.soh_pred[-1] < r.soh_pred[n_train_max - 1] - 0.01       # keeps fading past n = 90
    assert np.all(np.diff(r.soh_pred[36:]) <= 1e-12)                  # monotone forecast


def test_conformal_band_brackets_point_forecast():
    _, ct, _, _ = synthetic()
    r = te.train_ml_forecast(ct, "S002", 30, "Hist. Gradient Boosting", conformal_cells=3)
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
                               te.PINNConfig(epochs=80), "Bayesian Ridge", conformal_cells=2, pinn_seeds=(0, 1))
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

def test_gaussian_process_surrogate():
    _, ct, _, _ = synthetic()
    r = te.train_ml_forecast(ct, "S004", 36, "Gaussian Process", eol_ah=1.6, conformal_cells=0)
    assert np.isfinite(r.metrics.rmse) and r.metrics.rmse < 0.05



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
    assert {"Decision Tree", "Random Forest", "Extra Trees", "Hist. Gradient Boosting", "Gaussian Process",
            "Bayesian Ridge"} <= set(te.ML_MODELS)
    for name in te.ML_MODELS:
        m = te.make_model(name, 0, te.default_params(name)).fit(X, y)
        assert np.all(np.isfinite(m.predict(X))), name
    for bad in ({"n_estimators": 5}, {"nope": 1}):
        try:
            te.validate_params("Extra Trees", bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass


def test_ml_forecast_hyperparams_and_cross_battery():
    _, ct, _, _ = synthetic()
    f = te.train_ml_forecast(ct, "S004", 36, "Extra Trees", eol_ah=1.6, model_params={"n_estimators": 60})
    assert np.isfinite(f.metrics.accuracy) and f.metrics.accuracy > 95 and f.metrics.fade_skill > 0.5
    x = te.train_ml_forecast(ct, "S004", 10, "Bayesian Ridge", eol_ah=1.6, train_cells=["S005"])
    assert np.isfinite(x.metrics.rmse)
    try:
        te.train_ml_forecast(ct, "S004", 10, "Bayesian Ridge", train_cells=["S004"])
        raise AssertionError("target-only training list accepted")
    except ValueError:
        pass


def test_soh_estimator_splits_and_leakage_guard():
    _, ct, imp, _ = synthetic()
    for split, kw in (("random", {}), ("chronological", {}),
                      ("by_cell", {"train_cells": ["S001", "S002", "S003"], "test_cells": ["S004"]})):
        r = te.train_soh_estimator(ct, imp, "Extra Trees", split=split, test_frac=0.3,
                                   params={"n_estimators": 60}, **kw)
        assert set(r.metrics.index) == {"train", "test"} and r.metrics.loc["test", "Cycles"] >= 3
        assert r.metrics.loc["test", "R²"] > 0 and len(r.importance) == len(r.features)
    r = te.train_soh_estimator(ct, imp, "Bayesian Ridge", split="by_cell", train_cells=["S001", "S002"], test_cells=["S006"])
    assert set(r.predictions.loc[r.predictions["set"] == "test", "Cell_ID"]) == {"S006"}
    try:
        te.train_soh_estimator(ct, imp, "Bayesian Ridge", features=["Capacity_Ah", "R_dc_ohm"])
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


def test_fleet_status_events_and_scenarios():
    _, ct, _, _ = synthetic()
    fs = te.fleet_status(ct, eol_ah=1.6)
    assert len(fs) == 6 and set(fs["Risk"]) <= set(te.RISK_LEVELS)
    assert fs["risk_level"].is_monotonic_decreasing                       # triage order: worst first
    past = fs[fs["SOH"] <= fs["SOH_EOL"]]
    assert (past["Risk"] == "Critical").all() and (past["Quick RUL"] == 0).all()
    assert not fs["Alerts"].str.contains("past end of life, .*near end of life").any()
    ev = te.fleet_events(ct)
    assert {"Cell_ID", "n", "Severity", "Event"} <= set(ev.columns) and len(ev) > 0
    sf = te.stress_factor_regression(ct)
    sp = te.scenario_projection(sf, [{"T_C": 24, "I_A": 2.0}, {"T_C": 43, "I_A": 2.0}], 300, soh_eol=0.8)
    life = sp.groupby("scenario")["life_to_EOL"].first()
    assert life.iloc[1] < life.iloc[0]                                     # hotter -> shorter life
    assert (sp["lo"] <= sp["hi"] + 1e-12).all()


def test_half_cell_fit_recovers_degradation_modes():
    rng = np.random.default_rng(0)
    Cp, Cn, y0, x0 = 4.0, 2.7, 0.50, 0.85                    # y_end = 0.975: inside the physical window
    q = np.linspace(0, 1.9, 150)
    f0 = te.fit_half_cell(q, te.full_cell_ocv(q, Cp, Cn, y0, x0, 0.04) + 0.002 * rng.standard_normal(len(q)))
    assert f0.rmse_mV < 3 and abs(f0.params["Cp_Ah"] - Cp) < 0.1 and abs(f0.params["x0"] - x0) < 0.02
    N0 = x0 * Cn + y0 * Cp
    Cp2, Cn2 = Cp * 0.94, Cn * 0.97                      # truth: LAM_PE 6 %, LAM_NE 3 %, LLI 8 %
    x02 = (N0 * 0.92 - y0 * Cp2) / Cn2
    q2 = np.linspace(0, 1.65, 150)
    f1 = te.fit_half_cell(q2, te.full_cell_ocv(q2, Cp2, Cn2, y0, x02, 0.05) + 0.002 * rng.standard_normal(len(q2)),
                          np.array([f0.params[k] for k in te.HALF_CELL_KEYS]), global_search=False)
    lli = 100 * (1 - f1.li_inventory_Ah / f0.li_inventory_Ah)
    lam_pe = 100 * (1 - f1.params["Cp_Ah"] / f0.params["Cp_Ah"])
    assert abs(lli - 8) < 2.5 and abs(lam_pe - 6) < 2.5
    # electrode potentials are physical
    assert 3.8 < float(te.ocp_lco(0.9)) < 4.4 and 0.0 < float(te.ocp_graphite(0.5)) < 0.3


def test_half_cell_trajectory_on_cycle_data():
    store, ct, _, _ = synthetic()
    tab, fits = te.half_cell_trajectory(te.prepare_cell(store.cell_frame("S004")), ct[ct["Cell_ID"] == "S004"], 4)
    assert len(fits) >= 3 and {"LLI (%)", "LAM_PE (%)", "LAM_NE (%)", "Fit RMSE (mV)"} <= set(tab.columns)
    assert tab["LLI (%)"].iloc[0] == 0 and np.isfinite(tab["Fit RMSE (mV)"]).all()


def test_replacement_dp_is_safe_and_competitive():
    e, p = te.Economics(), te.CellPhysics()
    dp = te.solve_replacement_dp(e, p, n_soh=50, n_phase=8)
    assert dp.action.shape == (50, 8) and (dp.action < 0).any() and (dp.action >= 0).any()
    assert np.isfinite(dp.rho) and dp.rho > 0
    assert np.all(np.nan_to_num(dp.replace_boundary, nan=1.0) < 0.95)            # never replace a new cell
    res, life = te.evaluate_dp_policy(dp, e, p)
    assert res["violations"] == 0 and np.isfinite(res["rate"]) and res["rate"] > 0
    # DPPolicy honours the measured-ambient cold guard
    pol = te.DPPolicy(dp)
    assert pol(np.array([0.95, p.R_int0, p.R_ct0]), -5.0, p, e, cycle=0) <= e.cold_max_current_A


def test_delta_q_features_and_early_life_lifetime():
    _, ct, _, _ = synthetic()
    assert set(te.QV_COLS) <= set(ct.columns) and ct[te.QV_COLS[5]].notna().mean() > 0.9
    d = ct[ct["Cell_ID"] == "S006"]
    dq = te.delta_q_curve(d, 2, 40)
    assert dq is not None and np.nanmean(dq) < 0                     # aged cell delivers less charge at each V
    el = te.early_life_lifetime(ct, eol_ah=1.6, n_b=20)
    if el["available"]:
        assert el["corr_logvar_loglife"] < 0                          # Severson: larger variance -> shorter life
        assert el["mape_pct"] < el["baseline_mape_pct"]
    desc = te.early_life_descriptors(ct[~ct["outlier"]], 30)
    assert "dq_logvar" in desc.columns and desc["dq_logvar"].notna().all()


def test_hierarchical_bayes_shrinks_with_data_and_is_calibrated():
    _, ct, _, _ = synthetic()
    early = te.hierarchical_bayes_forecast(ct, "S004", 12, eol_ah=1.6)
    late = te.hierarchical_bayes_forecast(ct, "S004", 45, eol_ah=1.6)
    assert late.params["prior_weight"] < early.params["prior_weight"]     # self-updating: data take over
    assert late.metrics.rmse < 0.02 and late.metrics.coverage >= 0.5
    pop = te.hierarchical_population(ct, exclude="S004")
    assert pop["available"] and pop["Sigma"].shape == (2, 2)



def test_pinn_half_cell_mode_coupling():
    _, ct, imp, _ = synthetic()
    g = ct[(ct["Cell_ID"] == "S004") & ~ct["outlier"]]
    loss = 100 * (1 - g["SOH"].to_numpy())
    mt = pd.DataFrame({"n": g["n"].to_numpy()[::6], "LLI (%)": 0.7 * loss[::6], "LAM_PE (%)": 0.2 * loss[::6],
                       "LAM_NE (%)": 0.1 * loss[::6]})
    r = te.train_pinn(ct, imp, "S004", 36, te.PINNConfig(epochs=600, physics="mechanistic"), eol_ah=1.6,
                      mode_targets=mt)
    mech = r.mechanisms.set_index("n").loc[36]
    share_lli = (mech["Q_SEI"] + mech["Q_plating"]) / mech.sum()
    assert 0.5 < share_lli < 0.85 and "modes" in r.history.columns


def test_hyperparameter_tuning_is_leakage_free_and_never_worse_on_validation():
    _, ct, imp, _ = synthetic()
    r = te.tune_ml_forecast(ct, "S004", 36, "Bayesian Ridge", n_iter=4, n_val=2)
    assert r.best_score <= r.default_score + 1e-12 and len(r.trials) == 4
    assert "S004" not in r.validation                                      # the target is never a validation cell
    te.validate_params("Bayesian Ridge", r.best_params)
    e = te.tune_soh_estimator(ct, imp, "Bayesian Ridge", n_iter=4, split="by_cell", train_cells=["S001", "S002", "S003", "S005"],
                              test_cells=["S004"])
    assert e.best_score <= e.default_score + 1e-12 and "grouped" in e.validation
    rng = np.random.default_rng(0)
    for name in te.ML_MODELS:
        te.validate_params(name, te.sample_params(name, rng))              # search space stays valid


def test_twin_capacity_checks_and_correlated_voltage_fix_bias():
    m, imp, _ = te.make_synthetic_master(n_cells=6, n_cycles=120, seed=0, noise_v=0.01)
    store = te.ParquetStore.from_dataframe(m)
    ct = te.build_cycle_table(store)
    g = ct[(ct["Cell_ID"] == "S005") & ~ct["outlier"]]

    def track(cfg):
        r = te.run_dual_twin(store.cell_frame("S005"), ct, imp, "S005", te.TwinParameters(), cfg)
        e = g[["n", "SOH"]].merge(r.per_cycle[["n", "SOH", "SOH_std"]], on="n", suffixes=("", "_e"))
        err = e["SOH_e"] - e["SOH"]
        return abs(err.mean()), float(np.mean(np.abs(err) <= 2 * e["SOH_std"]))

    b0, c0 = track(te.DualTwinConfig(voltage_n_eff=25, adaptive=False))
    b1, c1 = track(te.DualTwinConfig(capacity_every=10))
    assert b1 < 0.5 * b0 and c1 > c0 + 0.3
    try:
        te.DualTwinConfig(capacity_every=-1)
        raise AssertionError("negative capacity_every accepted")
    except ValueError:
        pass


def _knee_cohort():
    m, imp, _ = te.make_synthetic_master(n_cells=8, n_cycles=130, ambients=(24, 34, 43, 4, 24, 34, 43, 24), seed=0,
                                         noise_v=0.01, knee={1: (50, 3.0)})
    store = te.ParquetStore.from_dataframe(m)
    return store, te.build_cycle_table(store), imp


def test_live_multi_model_beats_twin_on_a_knee_and_reweights():
    store, ct, imp = _knee_cohort()
    cell = "S002"
    g = ct[(ct["Cell_ID"] == cell) & ~ct["outlier"]]
    soh_eol = te.soh_eol_for(float(g["C_bol_Ah"].iloc[0]), 1.6)
    ekf = te.run_dual_twin(store.cell_frame(cell), ct, imp, cell, te.TwinParameters(),
                           te.DualTwinConfig(capacity_every=10))
    frames, track = te.live_multi_model(ct, cell, ekf, soh_eol)
    assert {"twin", "mech", "pf", "trend", "hb", "ens"} <= set(frames[max(frames)].forecasts)
    err = {}
    for n0, fr in frames.items():
        if 60 <= n0 <= g["n"].max() - 20 and n0 % 10 == 0:
            w = g[(g["n"] > n0) & (g["n"] <= n0 + 20)]
            for mdl, (med, _, _) in fr.forecasts.items():
                err.setdefault(mdl, []).append(np.sqrt(np.mean((np.interp(w["n"], fr.n_grid, med) - w["SOH"]) ** 2)))
    assert np.mean(err["ens"]) < 0.7 * np.mean(err["twin"])               # ensemble clearly better after the knee
    wts = frames[max(frames)].weights
    assert abs(sum(wts.values()) - 1) < 1e-9 and wts["twin"] < 0.5         # weight moved away from the twin
    sk = te.live_skill_table(track)
    assert len(sk) >= 4 and (sk["RMSE (5-step ahead)"] > 0).all()


def test_adaptive_process_noise_helps_after_a_knee_without_hurting_normal_cells():
    store, ct, imp = _knee_cohort()

    def fc_err(cell, cfg):
        g = ct[(ct["Cell_ID"] == cell) & ~ct["outlier"]]
        soh_eol = te.soh_eol_for(float(g["C_bol_Ah"].iloc[0]), 1.6)
        r = te.run_dual_twin(store.cell_frame(cell), ct, imp, cell, te.TwinParameters(), cfg)
        e = []
        for n0 in range(60, int(g["n"].max()) - 20, 10):
            f = te.twin_forecast(r, n0, int(g["n"].max()) + 10, soh_eol, 0.9)
            w = g[(g["n"] > n0) & (g["n"] <= n0 + 20)]
            e.append(np.sqrt(np.mean((np.interp(w["n"], f.n_grid, f.soh) - w["SOH"]) ** 2)))
        return float(np.mean(e))

    base = te.DualTwinConfig(capacity_every=10, adaptive_q=False)
    adapt = te.DualTwinConfig(capacity_every=10)
    assert fc_err("S002", adapt) <= fc_err("S002", base) + 1e-9
    assert fc_err("S001", adapt) <= 1.2 * fc_err("S001", base) + 1e-4


def test_live_accuracy_tables():
    store, ct, imp = _knee_cohort()
    cell = "S001"
    g = ct[(ct["Cell_ID"] == cell) & ~ct["outlier"]]
    soh_eol = te.soh_eol_for(float(g["C_bol_Ah"].iloc[0]), 1.6)
    ekf = te.run_dual_twin(store.cell_frame(cell), ct, imp, cell, te.TwinParameters(), te.DualTwinConfig())
    frames, track = te.live_multi_model(ct, cell, ekf, soh_eol, models=("twin", "trend"))
    sk = te.live_skill_table(track)
    assert "Accuracy (%)" in sk and sk["Accuracy (%)"].between(80, 100).all()
    eol = te.first_crossing(g["n"].to_numpy(), g["SOH"].to_numpy(), soh_eol, smooth=5)
    early = te.live_forecast_accuracy(frames[min(frames)], g, soh_eol, eol)
    assert len(early) >= 2 and early["Accuracy (%)"].between(0, 100).all()
    late = te.live_forecast_accuracy(frames[max(k for k in frames if k < g["n"].max())], g, soh_eol,
                                     int(g["n"].min()))                     # EOL already passed
    assert late["EOL error (cycles)"].isna().all()


def test_mechanistic_particle_filter_follows_operating_conditions():
    m, imp, _ = te.make_synthetic_master(n_cells=6, n_cycles=110, ambients=(24, 4, 43, 4, 24, 43),
                                         currents=(2.0, 2.0, 2.0, 2.0, 4.0, 4.0), seed=0, noise_v=0.01)
    store = te.ParquetStore.from_dataframe(m)
    ct = te.build_cycle_table(store)
    shares = {}
    for cell in ("S001", "S002", "S006"):
        g = ct[(ct["Cell_ID"] == cell) & ~ct["outlier"]]
        soh_eol = te.soh_eol_for(float(g["C_bol_Ah"].iloc[0]), 1.6)
        frames, track = te.live_multi_model(ct, cell, None, soh_eol, models=("mech", "trend"))
        last = frames[max(frames)]
        assert "mech" in last.forecasts and last.mech_shares is not None
        assert abs(sum(last.mech_shares.values()) - 1) < 1e-9
        assert {"share_SEI", "share_plating", "share_LAM"} <= set(track.columns)
        shares[cell] = last.mech_shares
        errs = [np.sqrt(np.mean((np.interp(g[(g["n"] > n0) & (g["n"] <= n0 + 20)]["n"], fr.n_grid, fr.forecasts["mech"][0])
                                 - g[(g["n"] > n0) & (g["n"] <= n0 + 20)]["SOH"]) ** 2))
                for n0, fr in frames.items() if 40 <= n0 <= g["n"].max() - 20 and n0 % 10 == 0]
        assert np.mean(errs) < 0.03
    assert shares["S002"]["plating"] > shares["S001"]["plating"] + 0.2          # cold cell: plating dominates
    assert shares["S002"]["plating"] > 0.4
    s = te.MechanisticStream(2.0, 1.0, n_particles=200)
    s.step(4.0, 2.0, 0.998)
    paths, mech = s.forecast(10, 4.0, 2.0, n_samples=50)
    assert paths.shape == (50, 10) and mech.shape == (10, 3) and (np.diff(mech, axis=0) >= -1e-12).all()


def test_condition_groups_and_pulse_aware_twin():
    m, imp, _ = te.make_synthetic_master(n_cells=4, n_cycles=40, ambients=(24, 4, 43, 24), currents=(2, 2, 2, 4), seed=0)
    ct = te.build_cycle_table(te.ParquetStore.from_dataframe(m))
    g = te.condition_groups(ct)
    assert list(g.loc[["S001", "S002", "S003", "S004"], "Group"]) == ["Reference", "Cold", "Hot", "High current"]
    assert (g["rest_frac"] < 0.2).all()
    pulsed = ct.copy()
    pulsed.loc[pulsed["Cell_ID"] == "S001", "rest_frac"] = 0.5
    assert te.condition_groups(pulsed).loc["S001", "Group"] == "Pulsed load"
    cfg = te.twin_config_for(pulsed, "S001")
    assert not cfg.use_voltage and cfg.capacity_every > 0
    assert te.twin_config_for(pulsed, "S002").use_voltage


def test_cohort_validation_summary_tests_and_mechanism_checks():
    m, imp, _ = te.make_synthetic_master(n_cells=6, n_cycles=90, ambients=(24, 4, 24, 4, 24, 43),
                                         currents=(2, 2, 2, 2, 2, 2), seed=0, noise_v=0.01)
    store = te.ParquetStore.from_dataframe(m)
    ct = te.build_cycle_table(store)
    val, mech = te.cohort_validation(store, ct, imp, 1.6, models=("mech", "pf", "trend"))
    assert {"Cell_ID", "Group", "model", "rmse", "coverage", "accuracy"} <= set(val.columns)
    assert set(val["model"]) >= {"mech", "pf", "trend", "ens"}
    summ = te.cohort_summary(val)
    assert "All batteries" in summ.index.get_level_values(0)
    pt = te.paired_model_test(val, "ens", "trend")
    assert "All batteries" in pt.index and 0 <= pt.loc["All batteries", "Wilcoxon p"] <= 1
    chk = te.mechanism_checks(mech)
    assert "Lithium plating dominates in the cold" in chk.index
    cold = mech[mech["Group"] == "Cold"]["share_plating"].median()
    ref = mech[mech["Group"] == "Reference"]["share_plating"].median()
    assert cold > ref


def test_plant_calibration_reproduces_the_cells_fade():
    _, ct, imp, _ = synthetic()
    q, tab = te.calibrate_plant(ct, imp, "S004")
    assert {"C_bol_Ah", "R_int0", "R_ct0", "k_ah"} <= set(tab.index)
    meta = te.cell_meta(ct)
    g = ct[(ct["Cell_ID"] == "S004") & ~ct["outlier"]]
    w = g[g["n"] <= 0.6 * g["n"].max()]
    rate = np.polyfit(w["cum_Ah"] - w["cum_Ah"].iloc[0], 1 - w["SOH"], 1)[0]
    model_rate = q.k_ah * float(te.degradation_stress(meta.loc["S004", "T_mean_C"], meta.loc["S004", "I_dis_A"], q))
    assert abs(model_rate / rate - 1) < 1e-6


def test_streaming_twin_and_service_registry():
    import importlib
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    service = importlib.import_module("service")
    _, ct, _, _ = synthetic()
    reg = service.TwinRegistry()
    g = ct[(ct["Cell_ID"] == "S006") & ~ct["outlier"]]
    reg.register("S006", service.BatteryConfig(c_bol_Ah=float(g["C_bol_Ah"].iloc[0]), eol_soh=0.8))
    for _, r in g.iterrows():
        s = reg.ingest("S006", r["Capacity_Ah"], r["T_mean_C"], r["I_dis_A"])
    assert s["status"] == "tracking" and s["battery_id"] == "S006"
    assert abs(sum(s["weights"].values()) - 1) < 1e-9 and s["dominant_mechanism"] in te.MECH_NAMES
    assert reg.fleet()[0]["battery_id"] == "S006"
    try:
        reg.ingest("S006", -1.0, 25.0, 2.0)
        raise AssertionError("negative capacity accepted")
    except ValueError:
        pass


# ---------------------------------------------------------------- v5.1 Part A: study correctness --
def test_A1_groups_mixed_conditions_and_known_corrupted_ids():
    m, imp, _ = te.make_synthetic_master(n_cells=4, n_cycles=40, ambients=(24, 43, 24, 24), seed=0)
    m["Cell_ID"] = m["Cell_ID"].replace({"S004": "B0050"})
    cyc = sorted(m.loc[m["Cell_ID"] == "S001", "Cycle_Index"].unique())
    half = m["Cell_ID"].eq("S001") & m["Cycle_Index"].isin(cyc[len(cyc) // 2:])
    m.loc[half, "Ambient_C"] = 44.0                                        # two ambient levels
    ct = te.build_cycle_table(te.ParquetStore.from_dataframe(m))
    g = te.condition_groups(ct)
    assert g.loc["S001", "Group"] == "Mixed conditions" and "24" in g.loc["S001", "Ambient levels (°C)"]
    assert g.loc["B0050", "Group"] == "Corrupted logging"
    assert g.loc["S002", "Group"] == "Hot" and "Load levels (A)" in g.columns
    ex = te.condition_groups(ct, mixed="exclude")
    assert ex.loc["S001", "excluded"] and not ex.loc["S002", "excluded"]


def test_A2_no_duplicate_forecasts_for_close_origins():
    m, imp, _ = te.make_synthetic_master(n_cells=3, n_cycles=30, seed=0)
    store = te.ParquetStore.from_dataframe(m)
    ct = te.build_cycle_table(store)
    g = ct[(ct["Cell_ID"] == "S001") & ~ct["outlier"]]
    frames, _ = te.live_multi_model(ct, "S001", None, 0.8, models=("trend",))
    rows = te._eval_frames(frames, g, (0.3, 0.32, 0.5), 20, 0.8)
    keys = [(r["n0"], r["model"]) for r in rows]
    assert len(keys) == len(set(keys)) and len({r["n0"] for r in rows}) == 2


def test_A3_best_model_only_on_common_cells():
    rows = []
    for c in ("C1", "C2", "C3"):
        rows.append({"Cell_ID": c, "Group": "Cold", "origin": 0.3, "n0": 10, "model": "a", "rmse": 0.02,
                     "accuracy": 98.0, "coverage": 0.9, "rul_error": np.nan})
    rows += [{"Cell_ID": "C1", "Group": "Cold", "origin": 0.3, "n0": 10, "model": "b", "rmse": 0.005,
              "accuracy": 99.5, "coverage": 0.9, "rul_error": np.nan}]                  # b only on the easy cell
    rows += [{"Cell_ID": c, "Group": "Cold", "origin": 0.3, "n0": 10, "model": "b", "rmse": 0.03,
              "accuracy": 97.0, "coverage": 0.9, "rul_error": np.nan} for c in ()]
    val = pd.DataFrame(rows)
    s = te.cohort_summary(val)
    cold = s.loc["Cold"]
    assert cold.loc["b", "partial"] and cold.loc["b", "cell coverage"] == "1/3 cells"
    bm = te.best_models(s)
    assert bm.loc["Cold", "common cells"] == 1
    # on the common cell b is better, but it must never be declared best without the flag being visible
    assert "partial" in s.columns and s.loc[("Cold", "b"), "partial"]


def test_A4_per_cell_pairing_and_underpowered_label():
    rows = []
    for c in range(6):
        for o in (0.3, 0.5):
            rows.append({"Cell_ID": f"C{c}", "Group": "Reference", "origin": o, "model": "ens", "rmse": 0.01 + 0.001 * c})
            rows.append({"Cell_ID": f"C{c}", "Group": "Reference", "origin": o, "model": "twin", "rmse": 0.02 + 0.001 * c})
    pt = te.paired_model_test(pd.DataFrame(rows), "ens", "twin")
    assert pt.loc["All batteries", "cells"] == 6                            # not 12 (two origins per cell)
    assert abs(te.min_achievable_p(2, 8) - 1 / 45) < 1e-12
    mech = pd.DataFrame({"Cell_ID": list("abcd"), "Group": ["Cold", "Cold", "Reference", "Reference"],
                         "share_plating": [0.8, 0.7, 0.1, 0.2], "share_SEI": [0.1, 0.2, 0.6, 0.5],
                         "share_LAM": [0.1, 0.1, 0.3, 0.3], "mech_rmse": [0.01] * 4})
    chk = te.mechanism_checks(mech)
    assert chk.loc["Lithium plating dominates in the cold", "Verdict"].startswith("underpowered")


def test_A5_single_dip_is_not_end_of_life():
    n = np.arange(1, 41)
    soh = np.linspace(1.0, 0.8, 40)
    soh[10] = 0.65                                                            # one low-capacity outlier run
    eol, status = te.eol_crossing(n, soh, 0.7)
    assert eol is None and status == "EOL not reachable in data"
    soh2 = np.concatenate([np.linspace(1.0, 0.72, 30), np.full(10, 0.68)])
    eol2, status2 = te.eol_crossing(n, soh2, 0.7)
    assert eol2 == 31 and status2 == "reached"


# ------------------------------------------------------------------ learning ladder (v5.2) --
def test_ladder_registry_levels_and_optional_boosting():
    assert {"Decision Tree", "Random Forest"} <= set(te.ML_MODELS)
    assert te.MODEL_SPECS["Decision Tree"].level == 1 and te.MODEL_SPECS["Hist. Gradient Boosting"].level == 3
    for name, pkg in te.OPTIONAL_ML.items():
        assert (name in te.ML_MODELS) == te._has(pkg)                 # optional libraries appear only if installed


def test_baselines_are_valid_references():
    _, ct, _, _ = synthetic()
    p = te.baseline_forecast(ct, "S004", 36, "persistence", eol_ah=1.6)
    t = te.baseline_forecast(ct, "S004", 36, "trend", eol_ah=1.6)
    fut = p.n_grid > 36
    assert np.allclose(np.diff(p.soh[fut]), 0)                         # persistence is flat
    assert np.all(np.diff(t.soh[fut]) <= 1e-12)                        # trend never increases
    assert t.metrics.rmse < p.metrics.rmse                             # a fading cell beats persistence


def test_deep_sequence_models_train_and_forecast():
    _, ct, _, _ = synthetic()
    for kind in te.SEQ_MODELS:
        f = te.seq_forecast(ct, "S004", 36, kind, eol_ah=1.6, epochs=40, n_members=1)
        fut = f.n_grid > 36
        assert np.all(np.isfinite(f.soh)) and np.all(np.diff(f.soh[fut]) <= 1e-12)
        assert f.metrics.rmse < 0.1 and f.params["training windows"] > 100


def test_spm_first_principles_physics_and_forecast():
    p = te.SPMParams()
    base = te.spm_discharge(p, 2.0, 25.0)
    assert 3.6 < base["V"][0] < 4.3 and base["V"][-1] <= 2.7 + 0.05
    assert np.all(np.diff(base["V"]) <= 1e-6)                          # discharge voltage decreases
    lli = te.spm_discharge(p, 2.0, 25.0, lli=0.1)["capacity_Ah"]
    assert lli < base["capacity_Ah"] - 0.1                             # lost lithium lowers capacity
    assert te.spm_discharge(p, 4.0, 25.0)["capacity_Ah"] < base["capacity_Ah"]      # rate capability
    assert te.spm_discharge(p, 2.0, 4.0)["capacity_Ah"] < base["capacity_Ah"]       # cold
    assert abs(float(te.ocp_lco(0.42)) - float(te.ocp_lco(0.48))) < 1e-9            # pole clipped
    _, ct, _, _ = synthetic()
    f = te.spm_forecast(ct, "S004", 36, eol_ah=1.6)
    assert f.metrics.rmse < 0.02 and f.params["SEI reaction term a (1/Ah)"] > 0


def test_training_scheme_options_for_population_and_sequence_models():
    _, ct, _, _ = synthetic()
    full = te.hierarchical_population(ct, exclude="S004")
    sub = te.hierarchical_population(ct, exclude="S004", cells=["S001", "S002", "S003"])
    assert full["available"] and full["n_cells"] == 5
    assert sub["available"] and set(sub["cells"]) == {"S001", "S002", "S003"}
    assert not te.hierarchical_population(ct, exclude="S004", cells=["S001"])["available"]
    f = te.hierarchical_bayes_forecast(ct, "S004", 30, 1.6, train_cells=["S001", "S002", "S003"])
    assert np.isfinite(f.metrics.rmse)
    a = te.seq_forecast(ct, "S004", 36, te.SEQ_MODELS[0], 1.6, epochs=20, n_members=1, train_cells=["S001"])
    b = te.seq_forecast(ct, "S004", 36, te.SEQ_MODELS[0], 1.6, epochs=20, n_members=1)
    assert a.params["training windows"] < b.params["training windows"]      # fewer training batteries -> fewer windows


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
