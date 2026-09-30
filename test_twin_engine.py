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
    assert np.allclose(first, 1.0)
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
