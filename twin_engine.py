"""
twin_engine.py - computational backend of the Battery Digital Twin (v3)
=======================================================================

Pure computation: no Streamlit, no plotting. Every public function is typed and
deterministic (seeded) so results can be cached by the UI layer.

Modules
  1. Data access        ParquetStore (lazy per-cell reads, in-memory fallback),
                        download_to_cache (streamed URL download, e.g. GitHub Releases)
  2. Cycle features     prepare_cell, build_cycle_table, cell_meta, outlier flags
  3. Diagnostics        Incremental Capacity Analysis (dQ/dV), peak tracking,
                        indicative LLI / LAM / resistive-shift classification
  4. ML surrogates      RF, GBR, SVR, MLP, Ridge forecasting of SOH and RUL
  5. EKF digital twin   1-RC ECM + Extended Kalman Filter (voltage feedback)
  6. Autodiff           minimal reverse-mode AD on numpy (no torch dependency)
  7. PINN               hybrid physics-data network: Arrhenius / SEI-type fade law,
                        coupled resistance-growth kinetics, Butler-Volmer overpotential
  8. Comparison         ML vs ECM+EKF vs PINN on a common forecast-origin protocol
  9. Operations         twin-aware operational optimisation and economics

Conventions
  current I > 0 = charge, I < 0 = discharge (NASA 'Current_measured')
  n = 1-based discharge-cycle counter; SOH = capacity / first valid capacity of the cell
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import shutil
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

R_GAS = 8.314462618          # J mol^-1 K^-1
FARADAY = 96485.33212        # C mol^-1
DEFAULT_EOL_AH = 1.4

MASTER_COLUMNS = ["Cell_ID", "Cycle_Index", "Cycle_Type", "Time_s", "Voltage_V",
                  "Current_A", "Temp_C", "Capacity_Ah"]
OPTIONAL_COLUMNS = ["Ambient_C"]
IMPEDANCE_COLUMNS = ["Cell_ID", "Cycle_Index", "Re_ohm", "Rct_ohm"]

ProgressFn = Optional[Callable[[float, str], None]]


def _report(progress: ProgressFn, frac: float, msg: str) -> None:
    if progress is not None:
        progress(float(min(max(frac, 0.0), 1.0)), msg)


# =============================================================================
# 1. DATA ACCESS
# =============================================================================
class DataError(RuntimeError):
    """Raised for unreadable, malformed or incomplete input data."""


def is_parquet_file(path: Union[str, Path]) -> bool:
    """Cheap validity check: Parquet files start and end with the magic bytes PAR1."""
    try:
        p = Path(path)
        if p.stat().st_size < 12:
            return False
        with open(p, "rb") as fh:
            head = fh.read(4)
            fh.seek(-4, os.SEEK_END)
            tail = fh.read(4)
        return head == b"PAR1" and tail == b"PAR1"
    except OSError:
        return False


def download_to_cache(url: str, cache_dir: Union[str, Path, None] = None,
                      progress: ProgressFn = None, timeout: float = 60.0,
                      chunk_bytes: int = 1 << 20, max_retries: int = 3) -> Path:
    """Stream a (large) Parquet file from a URL, e.g. a GitHub Release asset, to a
    local cache file. Downloads to a temp file first and renames only after the
    Parquet magic bytes validate, so an interrupted download never poisons the cache."""
    if not url.lower().startswith(("http://", "https://")):
        raise DataError(f"Not an http(s) URL: {url}")
    cache = Path(cache_dir or Path(tempfile.gettempdir()) / "battery_twin_cache")
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".parquet")
    if target.exists() and is_parquet_file(target):
        _report(progress, 1.0, "cached")
        return target

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        tmp = target.with_suffix(f".part{attempt}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "battery-digital-twin/3.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    buf = resp.read(chunk_bytes)
                    if not buf:
                        break
                    fh.write(buf)
                    done += len(buf)
                    if total:
                        _report(progress, done / total, f"{done / 1e6:.0f} / {total / 1e6:.0f} MB")
                    else:
                        _report(progress, 0.0, f"{done / 1e6:.0f} MB")
            if not is_parquet_file(tmp):
                raise DataError("Downloaded file is not a valid Parquet file "
                                "(check that the URL points at the raw asset).")
            shutil.move(str(tmp), str(target))
            _report(progress, 1.0, "download complete")
            return target
        except Exception as exc:          # network errors, HTTP errors, validation
            last_err = exc
            tmp.unlink(missing_ok=True)
            time.sleep(min(2 ** attempt, 8))
    raise DataError(f"Download failed after {max_retries} attempts: {last_err}")


def persist_upload(raw: bytes, name: str, cache_dir: Union[str, Path, None] = None) -> Path:
    """Write uploaded bytes to disk so large files can be read lazily per cell."""
    cache = Path(cache_dir or Path(tempfile.gettempdir()) / "battery_twin_cache")
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"upload_{hashlib.sha1(raw[:1 << 20] + str(len(raw)).encode()).hexdigest()[:16]}.parquet"
    if not target.exists():
        target.write_bytes(raw)
    if not is_parquet_file(target):
        target.unlink(missing_ok=True)
        raise DataError(f"'{name}' is not a valid Parquet file.")
    return target


class ParquetStore:
    """Lazy access to the master telemetry. Reads one cell at a time (predicate
    push-down via pyarrow) so the multi-million-row file never has to sit in memory.
    Can also wrap an in-memory DataFrame (tests, small data)."""

    def __init__(self, path: Union[str, Path, None] = None, frame: Optional[pd.DataFrame] = None):
        if (path is None) == (frame is None):
            raise ValueError("Provide exactly one of path or frame")
        self.path = Path(path) if path is not None else None
        self._frame = normalise_master(frame) if frame is not None else None
        self._cells: Optional[List[str]] = None
        self._columns = self._read_columns()
        missing = [c for c in MASTER_COLUMNS if c not in self._columns]
        if missing:
            raise DataError(f"Master file is missing columns: {missing}")

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "ParquetStore":
        return cls(frame=df)

    @property
    def key(self) -> str:
        """Stable identity for caching."""
        if self.path is not None:
            st_ = self.path.stat()
            return f"{self.path}:{st_.st_size}:{st_.st_mtime}"
        return f"frame:{len(self._frame)}:{pd.util.hash_pandas_object(self._frame.head(1000)).sum()}"

    def _read_columns(self) -> List[str]:
        if self._frame is not None:
            return list(self._frame.columns)
        try:
            import pyarrow.parquet as pq
            return list(pq.read_schema(self.path).names)
        except Exception as exc:
            raise DataError(f"Cannot read Parquet schema: {exc}") from exc

    @property
    def has_ambient(self) -> bool:
        return "Ambient_C" in self._columns

    def cells(self) -> List[str]:
        if self._cells is None:
            if self._frame is not None:
                vals = self._frame["Cell_ID"].unique()
            else:
                vals = pd.read_parquet(self.path, columns=["Cell_ID"])["Cell_ID"].astype(str).unique()
            self._cells = sorted(str(v) for v in vals)
        return self._cells

    def cell_frame(self, cell_id: str) -> pd.DataFrame:
        cols = MASTER_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in self._columns]
        if self._frame is not None:
            return self._frame.loc[self._frame["Cell_ID"] == cell_id, cols].copy()
        df = None
        try:                                              # predicate push-down
            import pyarrow as pa
            import pyarrow.dataset as ds
            dset = ds.dataset(str(self.path), format="parquet")
            tbl = dset.to_table(columns=cols,
                                filter=ds.field("Cell_ID").cast(pa.string()) == cell_id)
            df = tbl.to_pandas()
        except Exception:
            try:
                df = pd.read_parquet(self.path, columns=cols, filters=[("Cell_ID", "==", cell_id)])
            except Exception:
                full = pd.read_parquet(self.path, columns=cols)
                df = full[full["Cell_ID"].astype(str) == cell_id]
        return normalise_master(df)


    def iter_cells(self, batch_rows: int = 500_000):
        """Yield (cell_id, raw frame) for every cell in ONE streaming pass over the file.
        Row batches are split at Cell_ID changes and each cell is emitted as soon as its
        block ends, so peak memory is one cell plus one batch. Cells that are not stored
        contiguously are re-read with a filtered read at the end (last yield wins)."""
        if self._frame is not None:
            for cid in self.cells():
                yield cid, self.cell_frame(cid)
            return
        import pyarrow.parquet as pq
        cols = MASTER_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in self._columns]
        pf = pq.ParquetFile(str(self.path))
        current: Optional[str] = None
        buf: List[pd.DataFrame] = []
        done: set = set()
        late: set = set()
        for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
            df = normalise_master(batch.to_pandas())
            if df.empty:
                continue
            ids = df["Cell_ID"].to_numpy()
            cuts = np.flatnonzero(ids[1:] != ids[:-1]) + 1
            starts, ends = np.r_[0, cuts], np.r_[cuts, len(df)]
            for a, b in zip(starts, ends):
                cid = str(ids[a])
                if cid == current:
                    buf.append(df.iloc[a:b])
                    continue
                if current is not None:
                    yield current, pd.concat(buf, ignore_index=True)
                    done.add(current)
                if cid in done:
                    late.add(cid)
                current, buf = cid, [df.iloc[a:b]]
        if current is not None:
            yield current, pd.concat(buf, ignore_index=True)
        for cid in sorted(late):
            yield cid, self.cell_frame(cid)


def normalise_master(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("Cell_ID", "Cycle_Type"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    for col in ("Time_s", "Voltage_V", "Current_A", "Temp_C", "Capacity_Ah", "Ambient_C"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def load_impedance(source: Union[str, Path, bytes, pd.DataFrame]) -> pd.DataFrame:
    """Read and validate the EIS ground-truth table."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, (bytes, bytearray)):
        df = pd.read_parquet(io.BytesIO(source))
    else:
        df = pd.read_parquet(source)
    missing = [c for c in IMPEDANCE_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"Impedance file is missing columns: {missing}")
    df["Cell_ID"] = df["Cell_ID"].astype(str)
    for c in ("Re_ohm", "Rct_ohm"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def valid_eis(imp: Optional[pd.DataFrame], cell_id: str, r_max: float = 0.5) -> pd.DataFrame:
    """EIS fits of one cell inside a physically plausible window (drops unconverged fits)."""
    if imp is None or imp.empty:
        return pd.DataFrame(columns=IMPEDANCE_COLUMNS)
    e = imp[imp["Cell_ID"] == cell_id]
    e = e[e["Re_ohm"].between(1e-4, r_max) & e["Rct_ohm"].between(1e-4, r_max)]
    return e.sort_values("Cycle_Index").reset_index(drop=True)


# =============================================================================
# 2. CYCLE FEATURES
# =============================================================================
def prepare_cell(cell_df: pd.DataFrame) -> pd.DataFrame:
    """Chronological rows of one cell with per-row dt (s). Time_s restarts each cycle."""
    df = cell_df.dropna(subset=["Current_A", "Voltage_V", "Time_s"]).copy()
    df["Temp_C"] = df["Temp_C"].fillna(df["Temp_C"].median() if df["Temp_C"].notna().any() else 24.0)
    df = df[df["Cycle_Type"].isin(["charge", "discharge"])]
    df = df.sort_values(["Cycle_Index", "Time_s"], kind="mergesort").reset_index(drop=True)
    df["dt"] = df.groupby("Cycle_Index")["Time_s"].diff().fillna(0.0).clip(lower=0.0, upper=600.0)
    return df


def _cell_cycle_features(df: pd.DataFrame, cell_id: str) -> pd.DataFrame:
    """Per-discharge-cycle features of one prepared cell."""
    ah_row = np.abs(df["Current_A"].to_numpy()) * df["dt"].to_numpy() / 3600.0
    per_cycle_ah = pd.Series(ah_row).groupby(df["Cycle_Index"].to_numpy()).sum()
    cum_ah = per_cycle_ah.cumsum()

    dis = df[df["Cycle_Type"] == "discharge"]
    if dis.empty:
        return pd.DataFrame()
    g = dis.groupby("Cycle_Index", sort=True)
    load = dis[dis["Current_A"] < -0.1]
    gl = load.groupby("Cycle_Index", sort=True)
    first = g.first()
    first_load = gl.first()

    out = pd.DataFrame({
        "Capacity_Ah": g["Capacity_Ah"].first(),
        "T_mean_C": gl["Temp_C"].mean(),
        "T_max_C": gl["Temp_C"].max(),
        "I_dis_A": gl["Current_A"].median().abs(),
        "V_min_V": gl["Voltage_V"].min(),
        "t_dis_s": gl["Time_s"].max() - gl["Time_s"].min(),
    })
    if "Ambient_C" in dis.columns:
        out["Ambient_C"] = g["Ambient_C"].first()
    # Load-step voltage drop: rest voltage (first sample, I ~ 0) minus first loaded sample
    rest_ok = first["Current_A"].abs() < 0.1
    dv = (first["Voltage_V"] - first_load["Voltage_V"]).where(rest_ok)
    out["dV_step_V"] = dv
    out["R_dc_ohm"] = dv / out["I_dis_A"]
    out["cum_Ah"] = cum_ah.reindex(out.index).to_numpy()

    out = out.reset_index().rename(columns={"index": "Cycle_Index"})
    out.insert(0, "Cell_ID", cell_id)
    out = out.dropna(subset=["Capacity_Ah"]).reset_index(drop=True)
    out["n"] = np.arange(1, len(out) + 1)
    return out


def flag_outliers(ct: pd.DataFrame, rel_tol: float = 0.12, min_ah: float = 0.3) -> pd.Series:
    """Hampel-style flag per cell: capacity far from its rolling median, or implausibly low
    (the NASA 4 degC sets contain unexplained near-zero capacity runs)."""
    flags = pd.Series(False, index=ct.index)
    for _, d in ct.groupby("Cell_ID"):
        cap = d["Capacity_Ah"]
        med = cap.rolling(9, center=True, min_periods=3).median()
        flags.loc[d.index] = ((cap - med).abs() > rel_tol * med) | (cap < min_ah)
    return flags


def build_cycle_table(store: ParquetStore, cells: Optional[Sequence[str]] = None,
                      progress: ProgressFn = None) -> pd.DataFrame:
    """One row per (cell, discharge cycle) with capacity, SOH, operating conditions,
    load-step resistance and cumulative throughput. Streams one cell at a time."""
    wanted = set(cells or store.cells())
    total = max(len(wanted), 1)
    feats: Dict[str, pd.DataFrame] = {}
    for cell, raw in store.iter_cells():
        if cell not in wanted:
            continue
        _report(progress, len(feats) / total, f"features {cell}")
        try:
            f = _cell_cycle_features(prepare_cell(raw), cell)
            if not f.empty:
                feats[cell] = f               # last yield wins (non-contiguous cells)
        except Exception:
            continue          # one malformed cell must not break the dashboard
    _report(progress, 1.0, "features done")
    frames = [feats[c] for c in sorted(feats)]
    if not frames:
        raise DataError("No discharge cycles with capacity found in the master file.")
    ct = pd.concat(frames, ignore_index=True)
    ct["outlier"] = flag_outliers(ct)
    c_bol = ct[~ct["outlier"]].groupby("Cell_ID")["Capacity_Ah"].first()
    ct["C_bol_Ah"] = ct["Cell_ID"].map(c_bol)
    ct["SOH"] = ct["Capacity_Ah"] / ct["C_bol_Ah"]
    return ct


def cell_meta(ct: pd.DataFrame) -> pd.DataFrame:
    """Cell-level operating conditions (used as ML features and for display)."""
    good = ct[~ct["outlier"]]
    agg = good.groupby("Cell_ID").agg(
        cycles=("n", "max"), C_bol_Ah=("C_bol_Ah", "first"),
        cap_last_Ah=("Capacity_Ah", "last"), I_dis_A=("I_dis_A", "median"),
        V_cut_V=("V_min_V", "median"), T_mean_C=("T_mean_C", "median"))
    if "Ambient_C" in ct.columns:
        agg["Ambient_C"] = good.groupby("Cell_ID")["Ambient_C"].median()
    else:
        agg["Ambient_C"] = agg["T_mean_C"]
    agg["fade_pct"] = 100 * (1 - agg["cap_last_Ah"] / agg["C_bol_Ah"])
    return agg


def soh_eol_for(c_bol_ah: float, eol_ah: float = DEFAULT_EOL_AH) -> float:
    return float(eol_ah / c_bol_ah)


# =============================================================================
# 3. INCREMENTAL CAPACITY ANALYSIS
# =============================================================================
@dataclass
class ICACurve:
    cycle_index: int
    n: int
    voltage: np.ndarray
    dqdv: np.ndarray
    capacity_Ah: float
    peak_V: float
    peak_height: float
    peak_area: float


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral compatible with numpy 1.x (trapz) and 2.x (trapezoid)."""
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y, x))


def ica_curve(cell_df: pd.DataFrame, cycle_index: int, n: int = 0, dv: float = 0.01,
              smooth_window: int = 9, ir_ohm: Optional[float] = None,
              edge_exclusion_V: float = 0.05) -> Optional[ICACurve]:
    """dQ/dV of one constant-current discharge by voltage binning: the charge passed in
    each sample (|I| dt) is histogrammed on a uniform voltage grid and divided by the bin
    width, then Savitzky-Golay smoothed. Binning is robust to the non-monotonic voltage
    recovery caused by self-heating early in discharge (which breaks Q(V) inversion).
    With ir_ohm given, V is IR-compensated (V + |I| R) to remove the ohmic shift."""
    from scipy.signal import find_peaks, savgol_filter

    d = cell_df[(cell_df["Cycle_Index"] == cycle_index) & (cell_df["Current_A"] < -0.1)]
    if len(d) < 30:
        return None
    I = np.abs(d["Current_A"].to_numpy())
    dq = I * d["dt"].to_numpy() / 3600.0
    v = d["Voltage_V"].to_numpy() + (I * ir_ohm if ir_ohm else 0.0)
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    edges = np.arange(lo, hi + dv, dv)
    if len(edges) < smooth_window + 3:
        return None
    hist, _ = np.histogram(v, bins=edges, weights=dq)
    grid = 0.5 * (edges[1:] + edges[:-1])
    win = smooth_window if smooth_window % 2 else smooth_window + 1
    dqdv = np.clip(savgol_filter(hist / dv, win, 2), 0, None)
    inner = (grid > lo + edge_exclusion_V) & (grid < hi - edge_exclusion_V)
    if not inner.any():
        return None
    sub = np.where(inner, dqdv, 0.0)
    peaks, _ = find_peaks(sub, prominence=0.05 * sub.max() if sub.max() > 0 else None)
    k = int(peaks[np.argmax(sub[peaks])]) if len(peaks) else int(np.argmax(sub))
    half = int(round(0.1 / dv))
    a, b = max(0, k - half), min(len(grid), k + half + 1)
    return ICACurve(int(cycle_index), int(n), grid, dqdv, float(dq.sum()),
                    float(grid[k]), float(dqdv[k]), _trapz(dqdv[a:b], grid[a:b]))


def ica_evolution(cell_df: pd.DataFrame, ct_cell: pd.DataFrame, n_curves: int = 8,
                  ir_compensate: bool = True, **kw) -> List[ICACurve]:
    """ICA curves for evenly spaced non-outlier discharge cycles across the cell's life."""
    good = ct_cell[~ct_cell["outlier"]].reset_index(drop=True)
    if good.empty:
        return []
    idx = np.unique(np.linspace(0, len(good) - 1, min(n_curves, len(good))).astype(int))
    out = []
    for i in idx:
        row = good.iloc[i]
        r = float(row["R_dc_ohm"]) if ir_compensate and np.isfinite(row["R_dc_ohm"]) else None
        c = ica_curve(cell_df, int(row["Cycle_Index"]), int(row["n"]), ir_ohm=r, **kw)
        if c is not None:
            out.append(c)
    return out


def diagnose_degradation(curves: Sequence[ICACurve]) -> Dict[str, Any]:
    """Indicative degradation-mode reading from the first vs last ICA curve.

    Heuristics after Dubarry et al. (J. Power Sources 2012), qualitative only: at ~1C the
    curves are not near-equilibrium, so peak shapes are broadened by polarisation.
      - main peak shifting to lower voltage (discharge)   -> resistive / kinetic loss
      - peak height falling faster than total capacity     -> loss of active material (LAM)
      - capacity lost while the main peak is preserved     -> loss of lithium inventory (LLI)
    """
    if len(curves) < 2:
        return {"available": False, "reason": "Need at least two valid ICA curves."}
    a, b = curves[0], curves[-1]
    cap_ratio = b.capacity_Ah / a.capacity_Ah
    height_ratio = b.peak_height / a.peak_height if a.peak_height > 0 else float("nan")
    area_ratio = b.peak_area / a.peak_area if a.peak_area > 0 else float("nan")
    shift_mV = 1000 * (a.peak_V - b.peak_V)
    flags = {
        "resistive_shift": shift_mV > 30,
        "LAM_signature": height_ratio < cap_ratio - 0.05,
        "LLI_signature": (cap_ratio < 0.95) and (height_ratio >= cap_ratio - 0.05),
    }
    lines = []
    if flags["LLI_signature"]:
        lines.append("Capacity loss with the main peak largely preserved -> consistent with "
                     "loss of lithium inventory (e.g. SEI growth).")
    if flags["LAM_signature"]:
        lines.append("Main peak shrinks faster than capacity -> consistent with loss of "
                     "active material.")
    if flags["resistive_shift"]:
        lines.append(f"Main peak shifted {shift_mV:.0f} mV down -> residual polarisation "
                     "growth (resistive / kinetic).")
    if not lines:
        lines.append("No dominant signature above the heuristic thresholds.")
    return {"available": True, "n_ref": a.n, "n_cur": b.n, "cap_ratio": cap_ratio,
            "height_ratio": height_ratio, "area_ratio": area_ratio, "shift_mV": shift_mV,
            "flags": flags, "interpretation": lines}


# =============================================================================
# 4. ML SURROGATES
# =============================================================================
ML_MODELS = ("Random Forest", "Gradient Boosting", "SVR", "MLP", "Ridge")
ML_FEATURES = ["n", "Ambient_C", "I_dis_A", "V_cut_V"]   # operating plan only -> no leakage


def make_model(name: str, seed: int = 0):
    """Factory for the five surrogate regressors (scaled pipelines where needed)."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.svm import SVR

    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                                         subsample=0.8, random_state=seed)
    if name == "SVR":
        return make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.005, gamma="scale"))
    if name == "MLP":
        return make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=3000,
                                          learning_rate_init=3e-3, random_state=seed))
    if name == "Ridge":
        return make_pipeline(StandardScaler(), PolynomialFeatures(3), Ridge(alpha=1.0))
    raise ValueError(f"Unknown model: {name}")


def _feature_frame(ct: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Per-cycle features from the operating plan: cycle count + cell-level conditions."""
    f = ct[["Cell_ID", "n"]].copy()
    for col in ("Ambient_C", "I_dis_A", "V_cut_V"):
        f[col] = f["Cell_ID"].map(meta[col])
    return f[ML_FEATURES]


@dataclass
class ForecastMetrics:
    rmse: float
    mae: float
    r2: float
    rul_true: Optional[int]
    rul_pred: Optional[int]
    n_eval: int

    @property
    def rul_error(self) -> Optional[int]:
        if self.rul_true is None or self.rul_pred is None:
            return None
        return self.rul_pred - self.rul_true


def first_crossing(n: np.ndarray, y: np.ndarray, threshold: float, after: float = 0,
                   smooth: int = 1) -> Optional[int]:
    """First cycle (> after) where y falls below threshold (optional rolling median)."""
    s = pd.Series(y).rolling(smooth, center=True, min_periods=1).median().to_numpy() if smooth > 1 else y
    m = (n > after) & (s < threshold)
    return int(n[m][0]) if m.any() else None


def forecast_metrics(n_obs: np.ndarray, y_obs: np.ndarray, n_grid: np.ndarray,
                     y_grid: np.ndarray, n0: int, soh_eol: float) -> ForecastMetrics:
    """Error on the held-out horizon (n > n0) and RUL counted from the forecast origin."""
    mask = n_obs > n0
    y_hat = np.interp(n_obs[mask], n_grid, y_grid)
    y_true = y_obs[mask]
    if mask.sum() == 0:
        return ForecastMetrics(float("nan"), float("nan"), float("nan"), None, None, 0)
    err = y_hat - y_true
    ss = np.sum((y_true - y_true.mean()) ** 2)
    eol_true = first_crossing(n_obs, y_obs, soh_eol, after=n0, smooth=5)
    eol_pred = first_crossing(n_grid, y_grid, soh_eol, after=n0)
    return ForecastMetrics(
        rmse=float(np.sqrt(np.mean(err ** 2))), mae=float(np.mean(np.abs(err))),
        r2=float(1 - np.sum(err ** 2) / ss) if ss > 0 else float("nan"),
        rul_true=None if eol_true is None else eol_true - n0,
        rul_pred=None if eol_pred is None else eol_pred - n0, n_eval=int(mask.sum()))


@dataclass
class MLForecast:
    model: str
    cell_id: str
    n0: int
    n_grid: np.ndarray
    soh_pred: np.ndarray
    metrics: ForecastMetrics
    train_rows: int
    fit_seconds: float


def train_ml_forecast(ct: pd.DataFrame, cell_id: str, n0: int, model_name: str,
                      use_population: bool = True, target_weight: float = 5.0,
                      eol_ah: float = DEFAULT_EOL_AH, horizon_factor: float = 1.5,
                      seed: int = 0) -> MLForecast:
    """Fit on (other cells, all cycles) + (target cell, n <= n0); forecast target n > n0.
    Features are operating-plan quantities only, so the forecast never peeks at the
    future capacity. The target cell's early data are up-weighted to personalise."""
    good = ct[~ct["outlier"]]
    meta = cell_meta(ct)
    X_all = _feature_frame(good, meta)
    y_all = good["SOH"].to_numpy()
    is_target = (good["Cell_ID"] == cell_id).to_numpy()
    train = is_target & (good["n"].to_numpy() <= n0)
    if use_population:
        train |= ~is_target
    if train.sum() < 5:
        raise ValueError("Not enough training data for the ML surrogate.")
    w = np.where(is_target[train], target_weight, 1.0)

    model = make_model(model_name, seed)
    t0 = time.time()
    try:
        model.fit(X_all[train].to_numpy(), y_all[train], **_sample_weight_kw(model, w))
    except TypeError:
        model.fit(X_all[train].to_numpy(), y_all[train])
    fit_s = time.time() - t0

    tgt = good[good["Cell_ID"] == cell_id]
    n_max = int(max(tgt["n"].max(), n0) * horizon_factor)
    n_grid = np.arange(1, n_max + 1)
    Xg = pd.DataFrame({"n": n_grid})
    for col in ("Ambient_C", "I_dis_A", "V_cut_V"):
        Xg[col] = meta.loc[cell_id, col]
    pred = model.predict(Xg[ML_FEATURES].to_numpy())
    soh_eol = soh_eol_for(meta.loc[cell_id, "C_bol_Ah"], eol_ah)
    metrics = forecast_metrics(tgt["n"].to_numpy(), tgt["SOH"].to_numpy(), n_grid, pred, n0, soh_eol)
    return MLForecast(model_name, cell_id, n0, n_grid, pred, metrics, int(train.sum()), fit_s)


def _sample_weight_kw(model, w: np.ndarray) -> dict:
    """Route sample weights to the final estimator of a pipeline when supported."""
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        last = model.steps[-1][0]
        if last == "mlpregressor":           # MLPRegressor has no sample_weight in older sklearn
            return {}
        return {f"{last}__sample_weight": w}
    return {"sample_weight": w}


# =============================================================================
# 5. EKF DIGITAL TWIN (1-RC ECM)
# =============================================================================
@dataclass
class TwinParameters:
    C_bol_Ah: float = 2.0
    k_ah: float = 5e-4
    Ea_J_mol: float = 30e3
    T_ref_C: float = 24.0
    T_cold_C: float = 15.0
    k_cold: float = 0.2
    beta_int: float = 1.0
    beta_ct: float = 3.0
    k_cold_R: float = 0.02
    tau_rc_s: float = 60.0
    sigma_v: float = 0.08
    q_soc_per_ah: float = 2e-3
    q_vrc: float = 1e-3
    q_soh_per_ah: float = 5e-4
    q_r_frac_per_ah: float = 2e-3
    p0_soh: float = 0.01
    p0_r_frac: float = 0.1
    soc_anchor_sigma: float = 0.01
    gate_sigma: float = 4.0
    soh_min: float = 0.5
    soh_max: float = 1.1


@dataclass
class OperatingCondition:
    current_A: float
    temp_C: float
    voltage_V: float

    @classmethod
    def from_any(cls, u: Any) -> "OperatingCondition":
        if isinstance(u, OperatingCondition):
            return u
        i, t, v = u
        return cls(float(i), float(t), float(v))


class OCVModel:
    """OCV(SOC) lookup with linear extrapolation beyond the grid (keeps EKF gradients alive)."""

    def __init__(self, soc_grid: np.ndarray, ocv_grid: np.ndarray):
        self.soc = np.asarray(soc_grid, dtype=float)
        self.ocv = np.asarray(ocv_grid, dtype=float)
        self.docv = np.gradient(self.ocv, self.soc)
        self.slope_lo, self.slope_hi = float(self.docv[0]), float(self.docv[-1])

    def __call__(self, s: float) -> float:
        if s < self.soc[0]:
            return float(self.ocv[0] + self.slope_lo * (s - self.soc[0]))
        if s > self.soc[-1]:
            return float(self.ocv[-1] + self.slope_hi * (s - self.soc[-1]))
        return float(np.interp(s, self.soc, self.ocv))

    def slope(self, s: float) -> float:
        if s < self.soc[0]:
            return self.slope_lo
        if s > self.soc[-1]:
            return self.slope_hi
        return float(np.interp(s, self.soc, self.docv))

    @classmethod
    def from_discharge(cls, cell_df: pd.DataFrame, n_cycles: int, r_int: float, r_ct: float,
                       tau_s: float, n_bins: int = 40) -> "OCVModel":
        """Pseudo-OCV from the first discharge(s): OCV = V - I R_int0 - V_rc (1-RC simulated)."""
        dis = cell_df[cell_df["Cycle_Type"] == "discharge"]
        first = dis["Cycle_Index"].drop_duplicates().sort_values().head(n_cycles)
        soc_pts, ocv_pts = [], []
        for c in first:
            d = dis[dis["Cycle_Index"] == c]
            I, dt = d["Current_A"].to_numpy(), d["dt"].to_numpy()
            ah = np.cumsum(np.abs(I) * dt) / 3600
            vrc = np.zeros(len(I))
            for j in range(1, len(I)):
                a = math.exp(-dt[j] / tau_s)
                vrc[j] = a * vrc[j - 1] + (1 - a) * r_ct * I[j]
            load = I < -0.1
            if load.sum() < 10:
                continue
            soc_pts.append(1 - ah[load] / ah[load][-1])
            ocv_pts.append(d["Voltage_V"].to_numpy()[load] - I[load] * r_int - vrc[load])
        if not soc_pts:
            raise DataError("No usable discharge cycle to build the OCV curve.")
        soc, ocv = np.concatenate(soc_pts), np.concatenate(ocv_pts)
        edges = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(soc, edges) - 1, 0, n_bins - 1)
        grid = pd.DataFrame({"b": idx, "soc": soc, "ocv": ocv}).groupby("b").median()
        s, v = grid["soc"].to_numpy(), grid["ocv"].to_numpy()
        v = np.maximum.accumulate(v) + np.arange(len(v)) * 1e-5
        return cls(s, v)


class BatteryDigitalTwin:
    """Closed-loop EKF on x = [SOC, V_rc, SOH, R_int, R_ct] with V = OCV + I R_int + V_rc."""
    STATE_NAMES = ("SOC", "V_rc", "SOH", "R_int", "R_ct")

    def __init__(self, ocv: OCVModel, params: Optional[TwinParameters] = None,
                 soh0: float = 1.0, r_int0: float = 0.045, r_ct0: float = 0.07,
                 soc0: float = 1.0, closed_loop: bool = True):
        self.ocv = ocv
        self.params = params or TwinParameters()
        self.closed_loop = closed_loop
        self.r0 = np.array([r_int0, r_ct0])
        p = self.params
        self.x0 = np.array([soc0, 0.0, soh0, r_int0, r_ct0], dtype=float)
        self.P0 = np.diag([0.02 ** 2, 0.01 ** 2, p.p0_soh ** 2,
                           (p.p0_r_frac * r_int0) ** 2, (p.p0_r_frac * r_ct0) ** 2])
        self.reset()

    def reset(self) -> None:
        self.x, self.P = self.x0.copy(), self.P0.copy()
        self.n_updates = self.n_rejected = 0
        self.last_vpred = self.last_residual = float("nan")

    @property
    def soh_std(self) -> float:
        return math.sqrt(max(self.P[2, 2], 0.0))

    def stress_factor(self, temp_C: float) -> float:
        p = self.params
        arr = math.exp(p.Ea_J_mol / R_GAS * (1 / (p.T_ref_C + 273.15) - 1 / (temp_C + 273.15)))
        return arr * (1.0 + p.k_cold * max(0.0, p.T_cold_C - temp_C))

    def f(self, x: np.ndarray, u: OperatingCondition, dt: float) -> np.ndarray:
        p = self.params
        soc, vrc, soh, r_int, r_ct = x
        I = u.current_A
        a = math.exp(-dt / p.tau_rc_s)
        dsoh = p.k_ah * self.stress_factor(u.temp_C) * abs(I) * dt / 3600
        cold_R = 1.0 + p.k_cold_R * max(0.0, p.T_cold_C - u.temp_C)
        return np.array([
            soc + I * dt / (3600 * p.C_bol_Ah * max(soh, 1e-3)),
            a * vrc + (1 - a) * r_ct * I,
            soh - dsoh,
            r_int + p.beta_int * self.r0[0] * dsoh * cold_R,
            r_ct + p.beta_ct * self.r0[1] * dsoh * cold_R,
        ])

    def F_jac(self, x: np.ndarray, u: OperatingCondition, dt: float) -> np.ndarray:
        p = self.params
        I, soh = u.current_A, max(x[2], 1e-3)
        a = math.exp(-dt / p.tau_rc_s)
        F = np.eye(5)
        F[0, 2] = -I * dt / (3600 * p.C_bol_Ah * soh ** 2)
        F[1, 1] = a
        F[1, 4] = (1 - a) * I
        return F

    def Q_proc(self, u: OperatingCondition, dt: float) -> np.ndarray:
        p = self.params
        ah = abs(u.current_A) * dt / 3600
        return np.diag([p.q_soc_per_ah ** 2 * ah, p.q_vrc ** 2 * (dt > 0),
                        p.q_soh_per_ah ** 2 * ah,
                        (p.q_r_frac_per_ah * self.r0[0]) ** 2 * ah,
                        (p.q_r_frac_per_ah * self.r0[1]) ** 2 * ah])

    def h(self, x: np.ndarray, u: OperatingCondition) -> float:
        return self.ocv(x[0]) + u.current_A * x[3] + x[1]

    def H_jac(self, x: np.ndarray, u: OperatingCondition) -> np.ndarray:
        return np.array([self.ocv.slope(x[0]), 1.0, 0.0, u.current_A, 0.0])

    def predict(self, u: OperatingCondition, dt: float) -> None:
        F = self.F_jac(self.x, u, dt)
        self.x = self.f(self.x, u, dt)
        self.P = F @ self.P @ F.T + self.Q_proc(u, dt)

    def _kalman_update(self, residual: float, H: np.ndarray, r_var: float, gate: bool = True) -> bool:
        PH = self.P @ H
        S = float(H @ PH + r_var)
        if gate and abs(residual) > self.params.gate_sigma * math.sqrt(S):
            self.n_rejected += 1
            return False
        K = PH / S
        self.x = self.x + K * residual
        IKH = np.eye(5) - np.outer(K, H)
        self.P = IKH @ self.P @ IKH.T + r_var * np.outer(K, K)
        self._clamp()
        return True

    def _clamp(self) -> None:
        p = self.params
        self.x[0] = min(max(self.x[0], -0.3), 1.2)
        self.x[2] = min(max(self.x[2], p.soh_min), p.soh_max)
        self.x[3] = max(self.x[3], 1e-4)
        self.x[4] = max(self.x[4], 1e-4)

    def update_state(self, u_k: Any, dt: float, voltage_feedback: bool = True) -> np.ndarray:
        u = OperatingCondition.from_any(u_k)
        self.predict(u, dt)
        self.last_vpred = self.h(self.x, u)
        self.last_residual = u.voltage_V - self.last_vpred
        if self.closed_loop and voltage_feedback and math.isfinite(u.voltage_V):
            if self._kalman_update(self.last_residual, self.H_jac(self.x, u), self.params.sigma_v ** 2):
                self.n_updates += 1
        return self.x[2:5].copy()

    def start_cycle(self, cycle_type: str) -> None:
        """Protocol knowledge applied as resets (not updates) so it cannot leak into SOH
        through the SOC-SOH cross-covariance: RC relaxed at rest; discharge starts full."""
        self._reset_state(1, 0.0, 0.005)
        if cycle_type == "discharge":
            self._reset_state(0, 1.0, self.params.soc_anchor_sigma)

    def _reset_state(self, i: int, value: float, sigma: float) -> None:
        self.x[i] = value
        self.P[i, :] = 0.0
        self.P[:, i] = 0.0
        self.P[i, i] = sigma ** 2


def replay(twin: BatteryDigitalTwin, cell_df: pd.DataFrame, feedback_types: Tuple[str, ...] = ("discharge",),
           progress: ProgressFn = None, every: int = 20000) -> pd.DataFrame:
    tele = cell_df[["Current_A", "Temp_C", "Voltage_V", "dt"]].to_numpy(dtype=float)
    cyc = cell_df["Cycle_Index"].to_numpy()
    ctype = cell_df["Cycle_Type"].astype(str).to_numpy()
    n = len(tele)
    out = np.empty((n, 8))
    prev = None
    for i in range(n):
        if cyc[i] != prev:
            twin.start_cycle(ctype[i])
            prev = cyc[i]
        I, T, V, dt = tele[i]
        twin.update_state((I, T, V), dt, voltage_feedback=ctype[i] in feedback_types)
        out[i, :5] = twin.x
        out[i, 5] = twin.soh_std
        out[i, 6] = twin.last_vpred
        out[i, 7] = twin.last_residual
        if progress is not None and i % every == 0:
            _report(progress, i / n, "EKF replay")
    hist = pd.DataFrame(out, columns=list(BatteryDigitalTwin.STATE_NAMES) + ["SOH_std", "V_pred", "Residual_V"])
    hist["Cycle_Index"] = cyc.astype(int)
    hist["Cycle_Type"] = ctype
    return hist


def calibrate_k_ah_from_table(ct: pd.DataFrame, cells: Sequence[str], p: TwinParameters) -> float:
    """Least-squares k_ah (SOH loss per stress-weighted Ah) from per-cycle throughput and
    temperature of calibration cells - no raw-telemetry pass needed."""
    xs, ys = [], []
    for cell in cells:
        d = ct[(ct["Cell_ID"] == cell) & ~ct["outlier"]].sort_values("n")
        if len(d) < 3:
            continue
        T = d["T_mean_C"].fillna(p.T_ref_C).to_numpy()
        stress = np.exp(p.Ea_J_mol / R_GAS * (1 / (p.T_ref_C + 273.15) - 1 / (T + 273.15))) * \
            (1 + p.k_cold * np.maximum(0, p.T_cold_C - T))
        dah = np.diff(np.r_[0.0, d["cum_Ah"].to_numpy()])
        x = np.cumsum(stress * dah)
        xs.append(x - x[0])
        ys.append(1 - d["SOH"].to_numpy())
    if not xs:
        raise ValueError("No usable calibration cells.")
    x, y = np.concatenate(xs), np.concatenate(ys)
    return float(np.clip(np.sum(x * y) / np.sum(x * x), 1e-6, 1e-2))


def similar_cells(meta: pd.DataFrame, cell_id: str, max_cells: int = 4) -> List[str]:
    """Calibration partners: same ambient (+-5 degC), closest discharge current."""
    m = meta.drop(index=cell_id, errors="ignore")
    if cell_id not in meta.index or m.empty:
        return list(m.index[:max_cells])
    ref = meta.loc[cell_id]
    cand = m[(m["Ambient_C"] - ref["Ambient_C"]).abs() <= 5]
    if cand.empty:
        cand = m
    order = (cand["I_dis_A"] - ref["I_dis_A"]).abs().sort_values()
    return list(order.index[:max_cells])


@dataclass
class EKFResult:
    cell_id: str
    params: Dict[str, float]
    r_int0: float
    r_ct0: float
    per_cycle: pd.DataFrame           # n, Cycle_Index, SOH, SOH_std, R_int, R_ct, innov_mean_mV, innov_std_mV
    trace: pd.DataFrame               # downsampled per-row trace for plotting
    n_updates: int
    n_rejected: int
    runtime_s: float


def run_ekf(cell_df: pd.DataFrame, ct: pd.DataFrame, imp: Optional[pd.DataFrame], cell_id: str,
            params: TwinParameters, calib_cells: Sequence[str], closed_loop: bool = True,
            progress: ProgressFn = None, trace_points: int = 6000) -> EKFResult:
    """Calibrate the prior on other cells, initialise from the first capacity and EIS,
    replay the full telemetry, and summarise per discharge cycle (causal estimates)."""
    t0 = time.time()
    p = TwinParameters(**asdict(params))
    try:
        p.k_ah = calibrate_k_ah_from_table(ct, [c for c in calib_cells if c != cell_id], p)
    except ValueError:
        pass
    ct_cell = ct[ct["Cell_ID"] == cell_id].sort_values("n")
    if ct_cell.empty:
        raise DataError(f"{cell_id}: no discharge capacity data.")
    p.C_bol_Ah = float(ct_cell["C_bol_Ah"].iloc[0])
    eis = valid_eis(imp, cell_id)
    if len(eis):
        first = eis.head(3)
        r_int0, r_ct0 = float(first["Re_ohm"].median()), float(first["Rct_ohm"].median())
    else:
        r_int0, r_ct0 = 0.045, 0.07
    df = prepare_cell(cell_df)
    ocv = OCVModel.from_discharge(df, 1, r_int0, r_ct0, p.tau_rc_s)
    twin = BatteryDigitalTwin(ocv, p, r_int0=r_int0, r_ct0=r_ct0, closed_loop=closed_loop)
    hist = replay(twin, df, progress=progress)

    eoc = hist.groupby("Cycle_Index")[["SOH", "SOH_std", "R_int", "R_ct"]].last()
    dres = hist[hist["Cycle_Type"] == "discharge"].groupby("Cycle_Index")["Residual_V"]
    eoc["innov_mean_mV"] = 1000 * dres.mean()
    eoc["innov_std_mV"] = 1000 * dres.std()
    per = ct_cell[["n", "Cycle_Index"]].merge(eoc.reset_index(), on="Cycle_Index", how="left")
    step = max(1, len(hist) // trace_points)
    trace = hist.iloc[::step][["Cycle_Index", "Cycle_Type", "SOH", "R_int", "R_ct", "Residual_V"]]
    trace = trace.reset_index(drop=True)
    _report(progress, 1.0, "EKF done")
    return EKFResult(cell_id, asdict(p), r_int0, r_ct0, per, trace,
                     twin.n_updates, twin.n_rejected, time.time() - t0)


def ekf_forecast(ekf: EKFResult, n0: int, n_max: int, fit_frac: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Forecast from the EKF estimate at n0: linear rate fitted on the EKF's own SOH
    trajectory over the last fit_frac of the observed window (self-updated rate)."""
    per = ekf.per_cycle.dropna(subset=["SOH"])
    obs = per[per["n"] <= n0]
    if len(obs) < 3:
        raise ValueError("Forecast origin too early for the EKF forecast.")
    win = obs[obs["n"] >= n0 * (1 - fit_frac)]
    slope, intercept = np.polyfit(win["n"], win["SOH"], 1)
    soh_n0 = float(obs["SOH"].iloc[-1])
    n_grid = np.arange(1, n_max + 1)
    pred = np.where(n_grid <= n0, np.interp(n_grid, obs["n"], obs["SOH"]),
                    soh_n0 + min(slope, 0.0) * (n_grid - n0))
    return n_grid, pred


# =============================================================================
# 6. MINIMAL REVERSE-MODE AUTODIFF (numpy)
# =============================================================================
class Tensor:
    """Tiny reverse-mode autodiff tensor. Enough for PINN losses that contain first
    derivatives of the network w.r.t. its input (propagated as ordinary tensor ops)."""
    __slots__ = ("data", "grad", "_prev", "_backward")
    __array_priority__ = 1000          # make numpy defer to Tensor's reflected operators

    def __init__(self, data: Any, _prev: Tuple["Tensor", ...] = ()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad: Optional[np.ndarray] = None
        self._prev = _prev
        self._backward: Optional[Callable[[], None]] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def _acc(self, g: np.ndarray) -> None:
        g = _unbroadcast(g, self.data.shape)
        self.grad = g if self.grad is None else self.grad + g

    def backward(self) -> None:
        topo, seen, stack = [], set(), [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._prev:
                if id(p) not in seen:
                    stack.append((p, False))
        for node in topo:
            node.grad = None
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if node._backward is not None and node.grad is not None:
                node._backward()

    # ---- arithmetic ----
    def __add__(self, o: Any) -> "Tensor":
        o = _t(o)
        out = Tensor(self.data + o.data, (self, o))

        def bw():
            self._acc(out.grad)
            o._acc(out.grad)
        out._backward = bw
        return out

    __radd__ = __add__

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, (self,))
        out._backward = lambda: self._acc(-out.grad)
        return out

    def __sub__(self, o: Any) -> "Tensor":
        return self + (-_t(o))

    def __rsub__(self, o: Any) -> "Tensor":
        return _t(o) + (-self)

    def __mul__(self, o: Any) -> "Tensor":
        o = _t(o)
        out = Tensor(self.data * o.data, (self, o))

        def bw():
            self._acc(out.grad * o.data)
            o._acc(out.grad * self.data)
        out._backward = bw
        return out

    __rmul__ = __mul__

    def __truediv__(self, o: Any) -> "Tensor":
        o = _t(o)
        out = Tensor(self.data / o.data, (self, o))

        def bw():
            self._acc(out.grad / o.data)
            o._acc(-out.grad * self.data / o.data ** 2)
        out._backward = bw
        return out

    def __rtruediv__(self, o: Any) -> "Tensor":
        return _t(o) / self

    def __matmul__(self, o: "Tensor") -> "Tensor":
        out = Tensor(self.data @ o.data, (self, o))

        def bw():
            self._acc(out.grad @ o.data.T)
            o._acc(self.data.T @ out.grad)
        out._backward = bw
        return out

    # ---- elementwise functions ----
    def tanh(self) -> "Tensor":
        y = np.tanh(self.data)
        out = Tensor(y, (self,))
        out._backward = lambda: self._acc(out.grad * (1 - y ** 2))
        return out

    def exp(self) -> "Tensor":
        y = np.exp(np.clip(self.data, -700, 700))
        out = Tensor(y, (self,))
        out._backward = lambda: self._acc(out.grad * y)
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), (self,))
        out._backward = lambda: self._acc(out.grad / self.data)
        return out

    def softplus(self) -> "Tensor":
        out = Tensor(np.logaddexp(0.0, self.data), (self,))
        out._backward = lambda: self._acc(out.grad * _sigmoid(self.data))
        return out

    def sigmoid(self) -> "Tensor":
        y = _sigmoid(self.data)
        out = Tensor(y, (self,))
        out._backward = lambda: self._acc(out.grad * y * (1 - y))
        return out

    def asinh(self) -> "Tensor":
        out = Tensor(np.arcsinh(self.data), (self,))
        out._backward = lambda: self._acc(out.grad / np.sqrt(1 + self.data ** 2))
        return out

    def square(self) -> "Tensor":
        out = Tensor(self.data ** 2, (self,))
        out._backward = lambda: self._acc(out.grad * 2 * self.data)
        return out

    def sum(self) -> "Tensor":
        out = Tensor(self.data.sum(), (self,))
        out._backward = lambda: self._acc(np.full_like(self.data, out.grad))
        return out

    def mean(self) -> "Tensor":
        m = self.data.size
        out = Tensor(self.data.mean(), (self,))
        out._backward = lambda: self._acc(np.full_like(self.data, out.grad / m))
        return out


def _t(x: Any) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1 / (1 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1 + np.exp(-np.abs(x))))


def _unbroadcast(g: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class Adam:
    def __init__(self, params: List[Tensor], lr: float = 5e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, clip: float = 5.0):
        self.params, self.lr, self.b1, self.b2, self.eps, self.clip = params, lr, betas[0], betas[1], eps, clip
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self, lr: Optional[float] = None) -> float:
        lr = self.lr if lr is None else lr
        self.t += 1
        grads = [p.grad if p.grad is not None else np.zeros_like(p.data) for p in self.params]
        gnorm = math.sqrt(sum(float(np.sum(g * g)) for g in grads))
        scale = min(1.0, self.clip / (gnorm + 1e-12))
        for i, (p, g) in enumerate(zip(self.params, grads)):
            g = g * scale
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= lr * mh / (np.sqrt(vh) + self.eps)
        return gnorm


# =============================================================================
# 7. PHYSICS-INFORMED NEURAL NETWORK
# =============================================================================
@dataclass
class PINNConfig:
    hidden: int = 24
    epochs: int = 2500
    lr: float = 5e-3
    seed: int = 0
    lambda_data: float = 1.0          # capacity (SOH) data
    lambda_phys: float = 1.0          # fade law + resistance-growth kinetics residuals
    lambda_bv: float = 0.3            # Butler-Volmer load-step overpotential consistency
    lambda_eis: float = 0.5           # EIS R_e / R_ct anchors
    n_colloc: int = 200
    horizon_factor: float = 1.5
    soh_scale: float = 0.01
    rate_scale: float = 1e-3
    dv_scale: float = 0.02
    loss_eps: float = 0.02            # regulariser of the self-limiting (SEI-type) term
    loss_ref: float = 0.10
    T_ref_C: float = 24.0
    log_every: int = 25


@dataclass
class PINNResult:
    cell_id: str
    n0: int
    n_grid: np.ndarray
    soh: np.ndarray
    r_int: np.ndarray
    r_ct: np.ndarray
    physics: Dict[str, float]
    history: pd.DataFrame
    metrics: Optional[ForecastMetrics]
    train_seconds: float


class HybridPINN:
    """SOH(n), R_int(n), R_ct(n) from a small tanh network of the normalised discharge-cycle
    count t = n / n_scale, with initial conditions built in:
        SOH   = 1 - t * softplus(o_s)
        R_int = R_int0 * (1 + t * softplus(o_i)),   R_ct = R_ct0 * (1 + t * softplus(o_c))

    Governing equations enforced as soft residuals on collocation points spanning the
    whole horizon (they are what makes the extrapolation physical):
      Fade law (Arrhenius, SEI-type self-limiting, throughput-driven):
        dSOH/dn = -k * exp[Ea/R (1/T_ref - 1/T)] * (2 C_bol SOH) * ((1 - SOH + eps)/L_ref)^(-m)
        m in [-1, 2]:  m > 0 self-limiting (m = 1: parabolic, diffusion-limited SEI growth),
                       m = 0 linear in throughput, m < 0 self-accelerating (knee-like:
                       LAM / plating feedback). m is learned, so the data choose the regime.
      Resistance-growth kinetics coupled to fade:
        dR_int/dn = gamma_int R_int0 (-dSOH/dn),   dR_ct/dn = gamma_ct R_ct0 (-dSOH/dn)
      Butler-Volmer (symmetric, alpha = 0.5) at the discharge load step, with the
      exchange current linked to the small-signal R_ct (i0 = R T / (F R_ct)):
        dV_step = I (R_int + R_x) + (2 R T / F) asinh( I F R_ct / (2 R T) )
      R_x >= 0 lumps fast polarisation inside one sampling interval that EIS assigns elsewhere.
    Learnable physical parameters: k, Ea, m, gamma_int, gamma_ct, R_x."""

    def __init__(self, cfg: PINNConfig, c_bol: float, r_int0: float, r_ct0: float, n_scale: float):
        self.cfg, self.c_bol, self.r_int0, self.r_ct0, self.n_scale = cfg, c_bol, r_int0, r_ct0, n_scale
        rng = np.random.default_rng(cfg.seed)
        H = cfg.hidden

        def xavier(a: int, b: int) -> Tensor:
            return Tensor(rng.normal(0, math.sqrt(2.0 / (a + b)), size=(a, b)))

        self.W1, self.b1 = xavier(1, H), Tensor(rng.normal(0, 0.5, size=(1, H)))
        self.W2, self.b2 = xavier(H, H), Tensor(np.zeros((1, H)))
        self.Ws, self.bs = xavier(H, 1), Tensor([[_inv_softplus(0.3)]])
        self.Wi, self.bi = xavier(H, 1), Tensor([[_inv_softplus(0.3)]])
        self.Wc, self.bc = xavier(H, 1), Tensor([[_inv_softplus(0.6)]])
        # physical parameters (unconstrained)
        self.log_k = Tensor(math.log(5e-4))
        self.ea_raw = Tensor(_logit((30e3 - 10e3) / 70e3))
        self.m_raw = Tensor(0.0)                      # m = 0.5 at start
        self.gi_raw = Tensor(_inv_softplus(1.0))
        self.gc_raw = Tensor(_inv_softplus(3.0))
        self.rx_raw = Tensor(_inv_softplus(0.5))

    @property
    def parameters(self) -> List[Tensor]:
        return [self.W1, self.b1, self.W2, self.b2, self.Ws, self.bs, self.Wi, self.bi,
                self.Wc, self.bc, self.log_k, self.ea_raw, self.m_raw, self.gi_raw,
                self.gc_raw, self.rx_raw]

    # constrained physical parameters
    def k(self) -> Tensor: return self.log_k.exp()
    def Ea(self) -> Tensor: return 10e3 + 70e3 * self.ea_raw.sigmoid()
    def m(self) -> Tensor: return -1.0 + 3.0 * self.m_raw.sigmoid()
    def gamma_int(self) -> Tensor: return self.gi_raw.softplus()
    def gamma_ct(self) -> Tensor: return self.gc_raw.softplus()
    def R_x(self) -> Tensor: return self.r_int0 * self.rx_raw.softplus()

    def states(self, n: np.ndarray) -> Dict[str, Tensor]:
        """States and their derivatives d/dn (forward-mode tangent through the network)."""
        t = Tensor(np.asarray(n, dtype=float).reshape(-1, 1) / self.n_scale)
        h1 = (t @ self.W1 + self.b1).tanh()
        g1 = (1 - h1.square()) * self.W1                      # dh1/dt
        h2 = (h1 @ self.W2 + self.b2).tanh()
        g2 = (1 - h2.square()) * (g1 @ self.W2)               # dh2/dt
        out = {}
        for key, W, b, r0 in (("SOH", self.Ws, self.bs, None), ("R_int", self.Wi, self.bi, self.r_int0),
                              ("R_ct", self.Wc, self.bc, self.r_ct0)):
            o, go = h2 @ W + b, g2 @ W
            sp, sg = o.softplus(), o.sigmoid()
            d_dt = sp + t * sg * go                           # d(t * softplus(o))/dt
            if r0 is None:
                out["SOH"] = 1 - t * sp
                out["dSOH"] = -d_dt / self.n_scale
            else:
                out[key] = r0 * (1 + t * sp)
                out["d" + key] = r0 * d_dt / self.n_scale
        return out

    def losses(self, data: Dict[str, np.ndarray]) -> Dict[str, Tensor]:
        cfg = self.cfg
        L: Dict[str, Tensor] = {}
        s = self.states(data["n_soh"])
        L["data"] = ((s["SOH"] - data["soh"]) / cfg.soh_scale).square().mean()

        c = self.states(data["n_col"])
        T = data["T_col"] + 273.15
        arr = (self.Ea() * ((1.0 / (cfg.T_ref_C + 273.15) - 1.0 / T) / R_GAS)).exp()
        lost = 1 - c["SOH"]
        shape = (self.m() * (-1.0) * ((lost + cfg.loss_eps) / cfg.loss_ref).log()).exp()
        rate = self.k() * arr * (2 * self.c_bol) * c["SOH"] * shape
        fade = -c["dSOH"]
        r_soh = (c["dSOH"] + rate) / cfg.rate_scale
        r_int = (c["dR_int"] - self.gamma_int() * self.r_int0 * fade) / (self.r_int0 * cfg.rate_scale)
        r_ct = (c["dR_ct"] - self.gamma_ct() * self.r_ct0 * fade) / (self.r_ct0 * cfg.rate_scale)
        L["phys"] = r_soh.square().mean() + 0.5 * (r_int.square().mean() + r_ct.square().mean())

        if len(data["n_eis"]):
            e = self.states(data["n_eis"])
            L["eis"] = ((e["R_ct"] - data["rct"]) / self.r_ct0).square().mean() + \
                ((e["R_int"] - data["re"]) / self.r_int0).square().mean()
        if len(data["n_bv"]):
            b = self.states(data["n_bv"])
            I, Tk = data["I_bv"], data["T_bv"] + 273.15
            vt = 2 * R_GAS * Tk / FARADAY
            eta = vt * (b["R_ct"] * (I / vt)).asinh()        # asinh(I F R_ct / (2RT))
            dv = I * (b["R_int"] + self.R_x()) + eta
            L["bv"] = ((dv - data["dv"]) / cfg.dv_scale).square().mean()
        return L

    def total(self, L: Dict[str, Tensor]) -> Tensor:
        cfg = self.cfg
        tot = cfg.lambda_data * L["data"] + cfg.lambda_phys * L["phys"]
        if "eis" in L:
            tot = tot + cfg.lambda_eis * L["eis"]
        if "bv" in L:
            tot = tot + cfg.lambda_bv * L["bv"]
        return tot

    def physics_summary(self, n0: int) -> Dict[str, float]:
        s = self.states(np.array([1.0, float(n0)]))
        rct0, rctn = float(s["R_ct"].data[0, 0]), float(s["R_ct"].data[1, 0])
        Tk = self.cfg.T_ref_C + 273.15
        i0 = lambda r: R_GAS * Tk / (FARADAY * r)
        vt = 2 * R_GAS * Tk / FARADAY
        return {"k_per_Ah": float(self.k().data), "Ea_kJ_mol": float(self.Ea().data) / 1e3,
                "m_SEI_exponent": float(self.m().data), "gamma_int": float(self.gamma_int().data),
                "gamma_ct": float(self.gamma_ct().data), "R_x_mOhm": 1e3 * float(self.R_x().data),
                "i0_start_A": i0(rct0), "i0_at_n0_A": i0(rctn),
                "eta_ct_2A_start_mV": 1e3 * vt * math.asinh(2.0 * rct0 / vt),
                "eta_ct_2A_n0_mV": 1e3 * vt * math.asinh(2.0 * rctn / vt)}


def _inv_softplus(y: float) -> float:
    return float(math.log(math.expm1(y)))


def _logit(p: float) -> float:
    return float(math.log(p / (1 - p)))


def pinn_training_data(ct_cell: pd.DataFrame, eis: pd.DataFrame, n0: int, n_horizon: int,
                       cfg: PINNConfig) -> Dict[str, np.ndarray]:
    """Everything the PINN may see: capacity, EIS and load-step data up to n0 only.
    Collocation temperature beyond n0 is the planned (training-window mean) temperature."""
    good = ct_cell[~ct_cell["outlier"]].sort_values("n")
    obs = good[good["n"] <= n0]
    T_obs = obs["T_mean_C"].fillna(cfg.T_ref_C)
    T_plan = float(T_obs.mean()) if len(T_obs) else cfg.T_ref_C
    n_col = np.linspace(1, n_horizon, cfg.n_colloc)
    T_col = np.where(n_col <= n0, np.interp(n_col, obs["n"], T_obs), T_plan) if len(obs) > 1 \
        else np.full_like(n_col, T_plan)

    e = eis.copy()
    if len(e):
        # map each EIS sweep to the discharge-cycle count at or before it
        ci = ct_cell.sort_values("Cycle_Index")[["Cycle_Index", "n"]]
        e = pd.merge_asof(e.sort_values("Cycle_Index"), ci, on="Cycle_Index", direction="backward")
        e["n"] = e["n"].fillna(1)
        e = e[e["n"] <= n0]
    bv = obs.dropna(subset=["dV_step_V", "I_dis_A"])
    bv = bv[(bv["dV_step_V"] > 0) & (bv["dV_step_V"] < 1.5)]
    col = lambda d, c: d[c].to_numpy(dtype=float).reshape(-1, 1)
    return {"n_soh": obs["n"].to_numpy(dtype=float), "soh": col(obs, "SOH"),
            "n_col": n_col, "T_col": T_col.reshape(-1, 1),
            "n_eis": e["n"].to_numpy(dtype=float) if len(e) else np.array([]),
            "rct": col(e, "Rct_ohm") if len(e) else np.zeros((0, 1)),
            "re": col(e, "Re_ohm") if len(e) else np.zeros((0, 1)),
            "n_bv": bv["n"].to_numpy(dtype=float), "I_bv": col(bv, "I_dis_A"),
            "T_bv": col(bv, "T_mean_C"), "dv": col(bv, "dV_step_V")}


def train_pinn(ct: pd.DataFrame, imp: Optional[pd.DataFrame], cell_id: str, n0: int,
               cfg: Optional[PINNConfig] = None, eol_ah: float = DEFAULT_EOL_AH,
               progress: ProgressFn = None) -> PINNResult:
    cfg = cfg or PINNConfig()
    ct_cell = ct[ct["Cell_ID"] == cell_id].sort_values("n")
    if ct_cell.empty:
        raise DataError(f"{cell_id}: no discharge data.")
    c_bol = float(ct_cell["C_bol_Ah"].iloc[0])
    eis = valid_eis(imp, cell_id)
    r_int0 = float(eis.head(3)["Re_ohm"].median()) if len(eis) else 0.045
    r_ct0 = float(eis.head(3)["Rct_ohm"].median()) if len(eis) else 0.07
    n_horizon = int(max(ct_cell["n"].max(), n0) * cfg.horizon_factor)
    data = pinn_training_data(ct_cell, eis, n0, n_horizon, cfg)
    if len(data["n_soh"]) < 5:
        raise ValueError("Forecast origin too early: need at least 5 capacity points.")

    net = HybridPINN(cfg, c_bol, r_int0, r_ct0, float(n_horizon))
    opt = Adam(net.parameters, lr=cfg.lr)
    hist, t0 = [], time.time()
    for ep in range(1, cfg.epochs + 1):
        L = net.losses(data)
        tot = net.total(L)
        tot.backward()
        lr = cfg.lr * (0.1 ** (ep / cfg.epochs))            # exponential decay to lr/10
        opt.step(lr)
        if ep % cfg.log_every == 0 or ep == 1:
            hist.append({"epoch": ep, "total": float(tot.data),
                         **{k: float(v.data) for k, v in L.items()},
                         "k_per_Ah": float(net.k().data), "Ea_kJ_mol": float(net.Ea().data) / 1e3,
                         "m": float(net.m().data)})
            _report(progress, ep / cfg.epochs, f"PINN epoch {ep}")

    n_grid = np.arange(1, n_horizon + 1)
    s = net.states(n_grid)
    soh = s["SOH"].data.ravel()
    soh_eol = soh_eol_for(c_bol, eol_ah)
    metrics = forecast_metrics(ct_cell.loc[~ct_cell["outlier"], "n"].to_numpy(),
                               ct_cell.loc[~ct_cell["outlier"], "SOH"].to_numpy(), n_grid, soh, n0, soh_eol)
    return PINNResult(cell_id, n0, n_grid, soh, s["R_int"].data.ravel(), s["R_ct"].data.ravel(),
                      net.physics_summary(n0), pd.DataFrame(hist), metrics, time.time() - t0)


# =============================================================================
# 8. PARADIGM COMPARISON
# =============================================================================
@dataclass
class ComparisonResult:
    cell_id: str
    n0: int
    soh_eol: float
    measured: pd.DataFrame
    eis: pd.DataFrame
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]]
    metrics: Dict[str, ForecastMetrics]
    ekf: Optional[EKFResult]
    pinn: Optional[PINNResult]
    ml: Optional[MLForecast]
    errors: Dict[str, str]


def compare_paradigms(cell_df: pd.DataFrame, ct: pd.DataFrame, imp: Optional[pd.DataFrame],
                      cell_id: str, n0_frac: float, twin_params: TwinParameters,
                      pinn_cfg: PINNConfig, ml_model: str = "Gradient Boosting",
                      eol_ah: float = DEFAULT_EOL_AH, ekf: Optional[EKFResult] = None,
                      progress: ProgressFn = None) -> ComparisonResult:
    """Common protocol: everything up to discharge cycle n0 may be used, cycles > n0 are
    held out. (1) ML: population + early target data, operating-plan features.
    (2) ECM + EKF: causal voltage-feedback estimate up to n0, then self-updated rate.
    (3) Hybrid PINN: early target data + physics residuals over the full horizon.
    A failure in one paradigm is reported, not raised."""
    ct_cell = ct[ct["Cell_ID"] == cell_id].sort_values("n")
    good = ct_cell[~ct_cell["outlier"]]
    if len(good) < 10:
        raise DataError(f"{cell_id}: too few valid discharge cycles for a comparison.")
    n0 = int(max(5, round(n0_frac * good["n"].max())))
    c_bol = float(ct_cell["C_bol_Ah"].iloc[0])
    soh_eol = soh_eol_for(c_bol, eol_ah)
    n_max = int(good["n"].max() * 1.5)
    preds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    mets: Dict[str, ForecastMetrics] = {}
    errors: Dict[str, str] = {}
    meta = cell_meta(ct)
    n_obs, y_obs = good["n"].to_numpy(), good["SOH"].to_numpy()

    ml_res = None
    try:
        _report(progress, 0.02, f"ML surrogate ({ml_model})")
        ml_res = train_ml_forecast(ct, cell_id, n0, ml_model, eol_ah=eol_ah)
        preds[f"ML · {ml_model}"] = (ml_res.n_grid, ml_res.soh_pred)
        mets[f"ML · {ml_model}"] = ml_res.metrics
    except Exception as exc:
        errors["ML"] = str(exc)

    try:
        if ekf is None:
            sub = lambda f, m: _report(progress, 0.05 + 0.6 * f, m)
            ekf = run_ekf(cell_df, ct, imp, cell_id, twin_params, similar_cells(meta, cell_id), progress=sub)
        n_g, p_g = ekf_forecast(ekf, n0, n_max)
        preds["ECM + EKF"] = (n_g, p_g)
        mets["ECM + EKF"] = forecast_metrics(n_obs, y_obs, n_g, p_g, n0, soh_eol)
    except Exception as exc:
        errors["ECM + EKF"] = str(exc)

    pinn_res = None
    try:
        cfg = PINNConfig(**{**asdict(pinn_cfg), "horizon_factor": 1.5})
        sub = lambda f, m: _report(progress, 0.65 + 0.35 * f, m)
        pinn_res = train_pinn(ct, imp, cell_id, n0, cfg, eol_ah, progress=sub)
        preds["Hybrid PINN"] = (pinn_res.n_grid, pinn_res.soh)
        mets["Hybrid PINN"] = pinn_res.metrics
    except Exception as exc:
        errors["Hybrid PINN"] = str(exc)

    _report(progress, 1.0, "comparison done")
    return ComparisonResult(cell_id, n0, soh_eol, good[["n", "Cycle_Index", "SOH", "Capacity_Ah"]],
                            valid_eis(imp, cell_id), preds, mets, ekf, pinn_res, ml_res, errors)


def metrics_table(metrics: Dict[str, ForecastMetrics]) -> pd.DataFrame:
    rows = []
    for name, m in metrics.items():
        rows.append({"Paradigm": name, "RMSE": m.rmse, "MAE": m.mae, "R²": m.r2,
                     "RUL true": m.rul_true, "RUL pred": m.rul_pred, "RUL error": m.rul_error,
                     "held-out cycles": m.n_eval})
    return pd.DataFrame(rows).set_index("Paradigm")


# =============================================================================
# 9. TWIN-AWARE OPERATIONAL OPTIMISATION
# =============================================================================
@dataclass
class Economics:
    currents_A: Tuple[float, ...] = (1.0, 2.0, 4.0)
    price_per_Ah: Tuple[float, ...] = (0.80, 1.00, 1.25)
    replacement_cost: float = 150.0
    soh_eol: float = 0.70
    degradation_weight: float = 1.0
    T_max_C: float = 55.0
    min_ah_frac: float = 0.25
    cold_derate_below_C: Optional[float] = 10.0
    cold_max_current_A: float = 2.0

    @property
    def cost_per_soh(self) -> float:
        return self.replacement_cost / (1.0 - self.soh_eol)

    @property
    def decision_cost_per_soh(self) -> float:
        return self.degradation_weight * self.cost_per_soh

    def price(self, current_A: float) -> float:
        return self.price_per_Ah[list(self.currents_A).index(current_A)]


@dataclass
class CellPhysics:
    C_bol_Ah: float = 2.0
    R_int0: float = 0.05
    R_ct0: float = 0.07
    tau_rc_s: float = 60.0
    V_cut: float = 2.7
    I_charge_A: float = 1.5
    rest_h: float = 1.0
    k_ah: float = 5e-4
    Ea_J_mol: float = 30e3
    T_ref_C: float = 24.0
    T_cold_C: float = 15.0
    k_cold: float = 0.2
    I_ref_A: float = 2.0
    alpha_c: float = 0.5
    beta_int: float = 1.0
    beta_ct: float = 3.0
    k_cold_R: float = 0.02
    Ea_Rint_J_mol: float = 15e3
    Ea_Rct_J_mol: float = 40e3
    C_th_J_K: float = 45.0
    R_th_K_W: float = 8.0
    T_abort_C: float = 60.0
    dt_s: float = 20.0


def generic_ocv(soc: np.ndarray) -> np.ndarray:
    soc = np.asarray(soc, dtype=float)
    return 3.0 + 0.45 * (1 - np.exp(-18 * soc)) + 0.55 * soc + 0.2 * soc ** 4


def _arrhenius(Ea: float, T_C: Any, T_ref_C: float) -> np.ndarray:
    return np.exp(Ea / R_GAS * (1 / (np.asarray(T_C) + 273.15) - 1 / (T_ref_C + 273.15)))


def degradation_stress(T_C: Any, I_A: Any, p: CellPhysics) -> np.ndarray:
    thermal = 1 / _arrhenius(p.Ea_J_mol, T_C, p.T_ref_C)
    cold = 1 + p.k_cold * np.maximum(0, p.T_cold_C - np.asarray(T_C)) * (np.asarray(I_A) / p.I_ref_A)
    crate = (np.asarray(I_A) / p.I_ref_A) ** p.alpha_c
    return thermal * cold * crate


def simulate_discharge(theta: np.ndarray, currents: Sequence[float], T_amb: float, p: CellPhysics):
    soh, r_int, r_ct = theta
    I = np.atleast_1d(np.asarray(currents, dtype=float))
    n = len(I)
    Q = p.C_bol_Ah * soh
    a = math.exp(-p.dt_s / p.tau_rc_s)
    soc, vrc = np.ones(n), np.zeros(n)
    T, T_peak = np.full(n, float(T_amb)), np.full(n, float(T_amb))
    ah, dsoh, t = np.zeros(n), np.zeros(n), np.zeros(n)
    active = np.ones(n, dtype=bool)
    for _ in range(int(20 * 3600 / p.dt_s)):
        ri = r_int * _arrhenius(p.Ea_Rint_J_mol, T, p.T_ref_C)
        rc = r_ct * _arrhenius(p.Ea_Rct_J_mol, T, p.T_ref_C)
        vrc_new = a * vrc + (1 - a) * rc * I
        v = generic_ocv(soc) - I * ri - vrc_new
        active &= (v > p.V_cut) & (T < p.T_abort_C)
        if not active.any():
            break
        step_ah = np.where(active, I * p.dt_s / 3600, 0.0)
        dsoh += p.k_ah * degradation_stress(T, I, p) * step_ah
        ah += step_ah
        soc -= step_ah / Q
        t += np.where(active, p.dt_s, 0.0)
        T = np.where(active, T + p.dt_s / p.C_th_J_K * (I ** 2 * (ri + rc) - (T - T_amb) / p.R_th_K_W), T)
        T_peak = np.maximum(T_peak, T)
        vrc = np.where(active, vrc_new, vrc)
    return ah, dsoh, t / 3600, T_peak


def predict_cycle(theta: np.ndarray, currents: Sequence[float], T_amb: float, p: CellPhysics) -> Dict[str, np.ndarray]:
    ah, dsoh_dis, t_dis, T_peak = simulate_discharge(theta, currents, T_amb, p)
    _, r_int, r_ct = theta
    r_tot = r_int * _arrhenius(p.Ea_Rint_J_mol, T_amb, p.T_ref_C) + r_ct * _arrhenius(p.Ea_Rct_J_mol, T_amb, p.T_ref_C)
    T_ch = T_amb + p.I_charge_A ** 2 * r_tot * p.R_th_K_W
    dsoh_ch = p.k_ah * degradation_stress(T_ch, p.I_charge_A, p) * ah
    return {"ah": ah, "dsoh": dsoh_dis + dsoh_ch, "hours": t_dis + ah / p.I_charge_A + p.rest_h, "T_peak": T_peak}


def age_theta(theta: np.ndarray, dsoh: float, T_amb: float, p: CellPhysics) -> np.ndarray:
    soh, r_int, r_ct = theta
    cold_R = 1 + p.k_cold_R * max(0.0, p.T_cold_C - T_amb)
    return np.array([soh - dsoh, r_int + p.beta_int * p.R_int0 * dsoh * cold_R,
                     r_ct + p.beta_ct * p.R_ct0 * dsoh * cold_R])


def optimal_policy(theta_hat: np.ndarray, T_amb: float, p: CellPhysics, e: Economics) -> float:
    """One-step model-predictive choice: max price*Ah - weight*cost*dSOH subject to safety."""
    I = np.array(e.currents_A)
    pred = predict_cycle(theta_hat, I, T_amb, p)
    profit = np.array(e.price_per_Ah) * pred["ah"] - e.decision_cost_per_soh * pred["dsoh"]
    feasible = (pred["T_peak"] <= e.T_max_C) & (pred["ah"] >= e.min_ah_frac * p.C_bol_Ah * theta_hat[0])
    if e.cold_derate_below_C is not None and T_amb < e.cold_derate_below_C:
        feasible &= I <= e.cold_max_current_A
    if not feasible.any():
        return float(I.min())
    return float(I[np.argmax(np.where(feasible, profit, -np.inf))])


def make_policy(name: str) -> Callable[[np.ndarray, float, CellPhysics, Economics], float]:
    if name == "Twin-Aware":
        return optimal_policy
    current = float(name.split()[1])
    return lambda theta_hat, T_amb, p, e: current


def ambient_profile(k: int, mean_C: float = 20.0, amp_C: float = 16.0, period_cycles: float = 80.0) -> float:
    return mean_C + amp_C * math.sin(2 * math.pi * k / period_cycles)


def simulate_life(policy: Callable, p: CellPhysics, e: Economics, max_cycles: int = 2000,
                  sigma_soh: float = 0.005, sigma_r_frac: float = 0.03, seed: int = 0,
                  ambient_mean_C: float = 20.0, ambient_amp_C: float = 16.0) -> pd.DataFrame:
    """Plant = twin physics; the policy sees theta + estimation noise. Runs to EOL."""
    rng = np.random.default_rng(seed)
    theta = np.array([1.0, p.R_int0, p.R_ct0])
    rows = []
    for k in range(max_cycles):
        if theta[0] < e.soh_eol:
            break
        T_amb = ambient_profile(k, ambient_mean_C, ambient_amp_C)
        theta_hat = theta * np.array([1.0, 1 + sigma_r_frac * rng.standard_normal(),
                                      1 + sigma_r_frac * rng.standard_normal()])
        theta_hat[0] += sigma_soh * rng.standard_normal()
        I_sel = policy(theta_hat, T_amb, p, e)
        out = {key: float(v[0]) for key, v in predict_cycle(theta, [I_sel], T_amb, p).items()}
        cold_breach = (e.cold_derate_below_C is not None and T_amb < e.cold_derate_below_C
                       and I_sel > e.cold_max_current_A)
        rows.append(dict(cycle=k, T_amb=T_amb, I=I_sel, SOH=theta[0], SOH_hat=theta_hat[0],
                         R_ct_hat=theta_hat[2], ah=out["ah"], hours=out["hours"], T_peak=out["T_peak"],
                         revenue=e.price(I_sel) * out["ah"],
                         profit=e.price(I_sel) * out["ah"] - e.cost_per_soh * out["dsoh"],
                         violation=bool(out["T_peak"] > e.T_max_C or cold_breach)))
        theta = age_theta(theta, out["dsoh"], T_amb, p)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["cum_profit"] = df["profit"].cumsum()
        df["cum_hours"] = df["hours"].cumsum()
    return df


def summarise_life(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"cycles": 0, "Ah": 0.0, "profit": 0.0, "hours": 0.0, "profit_per_h": float("nan"),
                "violations": 0, "mean_I": float("nan")}
    profit, hours = float(df["profit"].sum()), float(df["hours"].sum())
    return {"cycles": len(df), "Ah": float(df["ah"].sum()), "profit": profit, "hours": hours,
            "profit_per_h": profit / hours if hours else float("nan"),
            "violations": int(df["violation"].sum()), "mean_I": float(df["I"].mean())}
