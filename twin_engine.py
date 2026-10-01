"""
twin_engine.py - computational backend of the Battery Digital Twin (v4)
=======================================================================

Pure computation: no Streamlit, no plotting. Every public function is typed and
deterministic (seeded) so results can be cached by the UI layer and reproduced offline.

Modules
  1. Data access        ParquetStore (lazy per-cell reads), download with optional SHA-256
                        pinning, content-hashed uploads, schema / range validation
  2. Cycle features     prepare_cell, build_cycle_table (per-cell error report),
                        outlier and capacity-regeneration flags
  3. Diagnostics        Incremental Capacity Analysis (dQ/dV), indicative LLI / LAM reading
  4. ML surrogates      "increment" strategy: autonomous fade-rate model dSOH/dn = g(SOH, x)
                        integrated forward (extrapolates beyond the training horizon);
                        "direct" legacy strategy kept for comparison; cross-cell,
                        horizon-dependent conformal bands
  5. ECM observers      (a) joint per-sample EKF (legacy, with NIS consistency statistics)
                        (b) dual time-scale twin: per-cycle EKF on [SOH, R_int, R_ct, log k]
                            with partial-window voltage, load-step resistance and (optional)
                            capacity measurements; Monte-Carlo forecast -> RUL distribution;
                            measurement ablation (does voltage feedback add information?)
  6. Autodiff           minimal reverse-mode AD on numpy + finite-difference gradient check
  7. PINN               hybrid physics-data network; activation energy fixed, pooled across
                        the cohort (Arrhenius regression with bootstrap CI) or learned;
                        multi-seed ensemble for uncertainty and identifiability
  8. Metrics            forecast error, censored RUL, relative accuracy, alpha-lambda,
                        prognostic horizon, band coverage (Saxena et al., 2010)
  9. Comparison         ML vs ECM twin vs PINN on a common forecast-origin protocol
 10. Benchmark          forecast-origin sweep x cells x paradigms (+ per-horizon residuals)
 11. Operations         twin-aware control with plant/model mismatch and a mismatch study
 12. Synthetic data     ground-truth generator (tests, demos) and run manifests

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
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

ENGINE_VERSION = "4.2.0"
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


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


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


def file_sha256(path: Union[str, Path], chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(chunk_bytes), b""):
            h.update(buf)
    return h.hexdigest()


def default_cache_dir() -> Path:
    """Content-addressed cache root; override with $TWIN_CACHE_DIR (e.g. a Docker volume)."""
    return Path(os.environ.get("TWIN_CACHE_DIR") or Path(tempfile.gettempdir()) / "battery_twin_cache")


def download_to_cache(url: str, cache_dir: Union[str, Path, None] = None,
                      progress: ProgressFn = None, timeout: float = 60.0,
                      chunk_bytes: int = 1 << 20, max_retries: int = 3,
                      sha256: Optional[str] = None) -> Path:
    """Stream a (large) Parquet file from a URL to a local cache file.

    Downloads to a temp file and renames only after validation, so an interrupted download
    never poisons the cache. With ``sha256`` given, the full-file digest must match the pin
    (also re-checked for an existing cached copy)."""
    if not url.lower().startswith(("http://", "https://")):
        raise DataError(f"Not an http(s) URL: {url}")
    pin = sha256.lower().strip() if sha256 else None
    cache = Path(cache_dir or default_cache_dir())
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (hashlib.sha256(url.encode()).hexdigest()[:20] + ".parquet")
    if target.exists() and is_parquet_file(target) and (pin is None or file_sha256(target) == pin):
        _report(progress, 1.0, "cached")
        return target

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        tmp = target.with_suffix(f".part{attempt}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"battery-digital-twin/{ENGINE_VERSION}"})
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
            if pin is not None and file_sha256(tmp) != pin:
                raise DataError("SHA-256 mismatch: the downloaded file differs from the pinned digest.")
            shutil.move(str(tmp), str(target))
            _report(progress, 1.0, "download complete")
            return target
        except Exception as exc:          # network errors, HTTP errors, validation
            last_err = exc
            tmp.unlink(missing_ok=True)
            if isinstance(exc, DataError) and "SHA-256" in str(exc):
                break                     # a wrong digest will not fix itself on retry
            time.sleep(min(2 ** attempt, 8))
    raise DataError(f"Download failed after {max_retries} attempts: {last_err}")


def persist_upload(raw: bytes, name: str, cache_dir: Union[str, Path, None] = None) -> Path:
    """Write uploaded bytes to disk (content-addressed by full SHA-256) so large files can
    be read lazily per cell. Two different files can never share a cache path."""
    cache = Path(cache_dir or default_cache_dir())
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"upload_{hashlib.sha256(raw).hexdigest()[:24]}.parquet"
    if not target.exists():
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(raw)
        shutil.move(str(tmp), str(target))
    if not is_parquet_file(target):
        target.unlink(missing_ok=True)
        raise DataError(f"'{name}' is not a valid Parquet file.")
    return target


class ParquetStore:
    """Lazy access to the master telemetry. Reads one cell at a time (predicate
    push-down via pyarrow) so the multi-million-row file never has to sit in memory.
    Can also wrap an in-memory DataFrame (tests, synthetic data)."""

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
        return f"frame:{len(self._frame)}:{int(pd.util.hash_pandas_object(self._frame.head(2000)).sum())}"

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


def validate_master(df: pd.DataFrame) -> List[str]:
    """Schema and physical-range checks on raw telemetry (one cell or the whole file).
    Returns human-readable issues; an empty list means the data passed."""
    issues: List[str] = []
    missing = [c for c in MASTER_COLUMNS if c not in df.columns]
    if missing:
        return [f"Missing columns: {missing}"]
    if df.empty:
        return ["No rows."]
    for c in ("Voltage_V", "Current_A", "Time_s"):
        frac = float(df[c].isna().mean())
        if frac > 0.01:
            issues.append(f"{c}: {100 * frac:.1f}% missing values")
    v = df["Voltage_V"].dropna()
    bad_v = float(((v < 1.5) | (v > 5.0)).mean()) if len(v) else 0.0
    if bad_v > 1e-3:
        issues.append(f"Voltage_V: {100 * bad_v:.2f}% of samples outside 1.5–5.0 V")
    i = df["Current_A"].abs().dropna()
    if len(i) and float((i > 20).mean()) > 1e-3:
        issues.append("Current_A: samples above 20 A (unit or sign error?)")
    t = df["Temp_C"].dropna()
    if len(t) and float(((t < -40) | (t > 90)).mean()) > 1e-3:
        issues.append("Temp_C: samples outside −40–90 °C")
    unknown = sorted(set(df["Cycle_Type"].astype(str).unique()) - {"charge", "discharge", "impedance"})
    if unknown:
        issues.append(f"Unknown Cycle_Type values: {unknown}")
    back = df.groupby(["Cell_ID", "Cycle_Index"], sort=False)["Time_s"].diff()
    if float((back < 0).mean()) > 1e-3:
        issues.append("Time_s decreases within cycles (unsorted or concatenated records)")
    return issues


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


def _int_key(df: pd.DataFrame, col: str = "Cycle_Index") -> pd.DataFrame:
    """merge_asof needs identical key dtypes; parquet files written elsewhere may hold the
    cycle index as int32 or float64, so normalise to int64 before every as-of merge."""
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce").round().astype("int64")
    return out


CHARGE_COLUMNS = ["t_cc_s", "t_cv_s", "Q_ch_Ah", "E_ch_Wh", "T_ch_C", "I_ch_A", "V_ch_max_V"]


def _charge_features(df: pd.DataFrame, cv_tol_V: float = 0.01) -> pd.DataFrame:
    """Per charge cycle (CC-CV): CC duration (until the voltage first reaches the CV level),
    CV duration, charged Ah and Wh, mean temperature, CC current and maximum voltage.
    CC time shrinks with capacity loss; CV time grows with polarisation (conductivity loss),
    so both are operando health indicators available on every charge."""
    ch = df[(df["Cycle_Type"] == "charge") & (df["Current_A"] > 0.01)]
    if ch.empty:
        return pd.DataFrame(columns=["Cycle_Index"] + CHARGE_COLUMNS)
    rows = []
    for ci, d in ch.groupby("Cycle_Index", sort=True):
        t, v, i = d["Time_s"].to_numpy(), d["Voltage_V"].to_numpy(), d["Current_A"].to_numpy()
        dt = d["dt"].to_numpy()
        if len(t) < 3:
            continue
        vmax = float(np.nanmax(v))
        k = int(np.argmax(v >= vmax - cv_tol_V))
        t_cc = float(t[k] - t[0])
        rows.append({"Cycle_Index": int(ci), "t_cc_s": t_cc, "t_cv_s": float(t[-1] - t[k]),
                     "Q_ch_Ah": float(np.sum(i * dt) / 3600.0), "E_ch_Wh": float(np.sum(v * i * dt) / 3600.0),
                     "T_ch_C": float(np.nanmean(d["Temp_C"])),
                     "I_ch_A": float(np.nanpercentile(i[: max(k, 1)], 90)), "V_ch_max_V": vmax})
    return pd.DataFrame(rows, columns=["Cycle_Index"] + CHARGE_COLUMNS)


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

    load_w = load.assign(_e=load["Voltage_V"] * load["Current_A"].abs() * load["dt"] / 3600.0)
    out = pd.DataFrame({
        "Capacity_Ah": g["Capacity_Ah"].first(),
        "T_mean_C": gl["Temp_C"].mean(),
        "T_max_C": gl["Temp_C"].max(),
        "T_start_C": g["Temp_C"].first(),
        "I_dis_A": gl["Current_A"].median().abs(),
        "V_min_V": gl["Voltage_V"].min(),
        "t_dis_s": gl["Time_s"].max() - gl["Time_s"].min(),
        "E_dis_Wh": load_w.groupby("Cycle_Index", sort=True)["_e"].sum(),
    })
    out["V_mean_V"] = out["E_dis_Wh"] / out["Capacity_Ah"].where(out["Capacity_Ah"] > 0)
    out["dT_C"] = out["T_max_C"] - out["T_start_C"]
    if "Ambient_C" in dis.columns:
        out["Ambient_C"] = g["Ambient_C"].first()
    # Load-step voltage drop: rest voltage (first sample, I ~ 0) minus first loaded sample
    rest_ok = first["Current_A"].abs() < 0.1
    dv = (first["Voltage_V"] - first_load["Voltage_V"]).where(rest_ok)
    out["dV_step_V"] = dv
    out["R_dc_ohm"] = dv / out["I_dis_A"]
    out["cum_Ah"] = cum_ah.reindex(out.index).to_numpy()

    out = out.reset_index().rename(columns={"index": "Cycle_Index"})
    ch = _charge_features(df)
    out["Cycle_Index"] = out["Cycle_Index"].astype("int64")      # parquet may store int32 / float
    if not ch.empty:
        ch["Cycle_Index"] = ch["Cycle_Index"].astype("int64")
        out = pd.merge_asof(out.sort_values("Cycle_Index"), ch.sort_values("Cycle_Index"),
                            on="Cycle_Index", direction="backward")     # the charge preceding each discharge
        out["eff_energy"] = out["E_dis_Wh"] / out["E_ch_Wh"].where(out["E_ch_Wh"] > 0)
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


def flag_regeneration(ct: pd.DataFrame, jump_frac: float = 0.01) -> pd.Series:
    """Capacity-regeneration events: an upward capacity jump larger than ``jump_frac`` of the
    initial capacity relative to the median of the three preceding valid cycles. In the NASA
    data these follow rest periods (relaxation of concentration gradients / lithium
    redistribution). They are real electrochemistry, not outliers, so they are kept in the
    data and flagged for display and interpretation."""
    flags = pd.Series(False, index=ct.index)
    ok = ct[~ct["outlier"]] if "outlier" in ct.columns else ct
    for _, d in ok.groupby("Cell_ID"):
        cap = d["Capacity_Ah"]
        prev = cap.shift(1).rolling(3, min_periods=1).median()
        flags.loc[d.index] = ((cap - prev) > jump_frac * float(cap.iloc[0])).fillna(False).to_numpy()
    return flags


def robust_bol_capacity(ct: pd.DataFrame, early_frac: float = 0.25, min_cycles: int = 5,
                        jump_ratio: float = 1.3) -> Tuple[pd.Series, pd.Series, pd.Index]:
    """Beginning-of-life capacity that survives bad first cycles.

    The first recorded capacity is unreliable in parts of the NASA set: B0049-B0056 were
    logged with crashed software (an invalid low segment followed by an upward level shift),
    and formation / warm-up lifts early capacities at 43 degC. Procedure per cell:
      1. find an upward level shift in the first half of life (5-cycle median after the step
         >= jump_ratio x median before it); cycles before it are an invalid segment;
      2. baseline = max of the 5-cycle rolling median over the first 25% (>= 5) valid cycles.
    Returns (baseline per cell, suspect flag per cell, row index of invalid-segment cycles)."""
    base, suspect, invalid = {}, {}, []
    for cid, d in ct[~ct["outlier"]].sort_values("n").groupby("Cell_ID"):
        cap = d["Capacity_Ah"].astype(float).to_numpy()
        cut = 0
        half = len(cap) // 2
        best = jump_ratio
        for j in range(3, max(half, 3)):
            before, after = np.median(cap[max(0, j - 5):j]), np.median(cap[j:j + 5])
            if before > 0 and after / before >= best:
                best, cut = after / before, j
        if cut:
            invalid.extend(d.index[:cut])
        rest = pd.Series(cap[cut:])
        k = max(min_cycles, int(math.ceil(early_frac * len(rest))))
        early = rest.head(k).rolling(5, center=True, min_periods=1).median()
        base[cid] = float(early.max()) if len(early) else float("nan")
        suspect[cid] = bool(cut)
    return pd.Series(base, dtype=float), pd.Series(suspect, dtype=bool), pd.Index(invalid)


def build_cycle_table(store: ParquetStore, cells: Optional[Sequence[str]] = None,
                      progress: ProgressFn = None,
                      errors: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """One row per (cell, discharge cycle) with capacity, SOH, operating conditions,
    load-step resistance and cumulative throughput. Streams one cell at a time.

    A malformed cell never breaks the table, but it is no longer skipped silently: pass a
    dict as ``errors`` to receive {cell_id: reason} for every cell that was dropped."""
    wanted = set(cells or store.cells())
    total = max(len(wanted), 1)
    feats: Dict[str, pd.DataFrame] = {}
    errs: Dict[str, str] = {}
    for cell, raw in store.iter_cells():
        if cell not in wanted:
            continue
        _report(progress, len(feats) / total, f"features {cell}")
        try:
            f = _cell_cycle_features(prepare_cell(raw), cell)
            if f.empty:
                errs[cell] = "no discharge cycles with a recorded capacity"
            else:
                feats[cell] = f               # last yield wins (non-contiguous cells)
                errs.pop(cell, None)
        except Exception as exc:
            errs[cell] = f"{type(exc).__name__}: {exc}"
    if errors is not None:
        errors.update(errs)
    _report(progress, 1.0, "features done")
    frames = [feats[c] for c in sorted(feats)]
    if not frames:
        reasons = pd.Series(errs).value_counts()
        detail = "; ".join(f"{r} ({k} cell{'s' if k > 1 else ''})" for r, k in reasons.head(3).items())
        raise DataError("No discharge cycles with capacity found in the master file."
                        + (f" Per-cell reasons: {detail}" if detail else ""))
    ct = pd.concat(frames, ignore_index=True)
    ct["outlier"] = flag_outliers(ct)
    c_bol, suspect, invalid = robust_bol_capacity(ct)
    ct.loc[invalid, "outlier"] = True
    ct["C_bol_Ah"] = ct["Cell_ID"].map(c_bol)
    ct["baseline_suspect"] = ct["Cell_ID"].map(suspect).fillna(False).astype(bool)
    ct["SOH"] = ct["Capacity_Ah"] / ct["C_bol_Ah"]
    # capacities far above the robust baseline (logging artefacts) are outliers, not regeneration
    ct.loc[ct["SOH"] > 1.15, "outlier"] = True
    ct["regen"] = flag_regeneration(ct)
    return ct


def build_cycle_table_with_report(store: ParquetStore, cells: Optional[Sequence[str]] = None,
                                  progress: ProgressFn = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    errors: Dict[str, str] = {}
    ct = build_cycle_table(store, cells, progress, errors)
    return ct, errors


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
    agg["Ambient_C"] = agg["Ambient_C"].fillna(agg["T_mean_C"])
    agg["fade_pct"] = 100 * (1 - agg["cap_last_Ah"] / agg["C_bol_Ah"])
    if "regen" in ct.columns:
        agg["regen_events"] = good.groupby("Cell_ID")["regen"].sum().astype(int)
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
# 8. METRICS (defined early: used by every forecaster)
# =============================================================================
@dataclass
class ForecastMetrics:
    """Held-out forecast quality. RUL is counted from the forecast origin n0.

    ``censored``: the measured SOH never crossed the EOL threshold within the recorded life,
    so the true RUL is only known to exceed ``rul_true_lb`` cycles (right-censored)."""
    rmse: float
    mae: float
    r2: float
    rul_true: Optional[int]
    rul_pred: Optional[int]
    n_eval: int
    censored: bool = False
    rul_true_lb: Optional[int] = None
    coverage: Optional[float] = None
    band_width: Optional[float] = None
    alpha: float = 0.2
    mape: float = float("nan")         # mean absolute percentage error on held-out SOH
    fade_skill: float = float("nan")   # 1 - SSE / SSE(no-further-fade forecast)

    @property
    def accuracy(self) -> float:
        """100 x (1 - MAPE): the share of the held-out SOH the forecast gets right."""
        return 100.0 * (1.0 - self.mape) if np.isfinite(self.mape) else float("nan")

    @property
    def rul_error(self) -> Optional[int]:
        if self.rul_true is None or self.rul_pred is None:
            return None
        return self.rul_pred - self.rul_true

    @property
    def rel_accuracy(self) -> Optional[float]:
        """Relative accuracy RA = 1 - |RUL_true - RUL_pred| / RUL_true (Saxena et al., 2010)."""
        if self.rul_error is None or not self.rul_true:
            return None
        return 1.0 - abs(self.rul_error) / self.rul_true

    @property
    def alpha_lambda_ok(self) -> Optional[bool]:
        """Alpha-lambda test at this forecast origin: |RUL error| <= alpha * RUL_true."""
        if self.rul_error is None or self.rul_true is None:
            return None
        return abs(self.rul_error) <= self.alpha * self.rul_true

    @property
    def censor_consistent(self) -> Optional[bool]:
        """For censored cells: is the prediction compatible with 'RUL > lower bound'?"""
        if not self.censored or self.rul_true_lb is None:
            return None
        return self.rul_pred is None or self.rul_pred >= self.rul_true_lb


def first_crossing(n: np.ndarray, y: np.ndarray, threshold: float, after: float = 0,
                   smooth: int = 1) -> Optional[int]:
    """First cycle (> after) where y falls below threshold (optional rolling median)."""
    n = np.asarray(n)
    y = np.asarray(y, dtype=float)
    s = pd.Series(y).rolling(smooth, center=True, min_periods=1).median().to_numpy() if smooth > 1 else y
    m = (n > after) & (s < threshold)
    return int(n[m][0]) if m.any() else None


def forecast_metrics(n_obs: np.ndarray, y_obs: np.ndarray, n_grid: np.ndarray,
                     y_grid: np.ndarray, n0: int, soh_eol: float,
                     lo: Optional[np.ndarray] = None, hi: Optional[np.ndarray] = None,
                     alpha: float = 0.2) -> ForecastMetrics:
    """Error on the held-out horizon (n > n0), RUL counted from the forecast origin,
    right-censoring, and (when a band is given) empirical coverage and mean band width."""
    n_obs = np.asarray(n_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    mask = n_obs > n0
    if mask.sum() == 0:
        return ForecastMetrics(float("nan"), float("nan"), float("nan"), None, None, 0, alpha=alpha)
    y_hat = np.interp(n_obs[mask], n_grid, y_grid)
    y_true = y_obs[mask]
    err = y_hat - y_true
    ss = np.sum((y_true - y_true.mean()) ** 2)
    eol_true = first_crossing(n_obs, y_obs, soh_eol, after=n0, smooth=5)
    eol_pred = first_crossing(np.asarray(n_grid), np.asarray(y_grid), soh_eol, after=n0)
    censored = eol_true is None
    coverage = width = None
    if lo is not None and hi is not None:
        lo_i, hi_i = np.interp(n_obs[mask], n_grid, lo), np.interp(n_obs[mask], n_grid, hi)
        coverage = float(np.mean((y_true >= lo_i) & (y_true <= hi_i)))
        width = float(np.mean(hi_i - lo_i))
    # persistence reference: SOH stays at its (smoothed) level at the origin
    before = n_obs <= n0
    y_ref = float(np.median(y_obs[before][-5:])) if before.any() else float(y_true[0])
    ss_ref = float(np.sum((y_true - y_ref) ** 2))
    return ForecastMetrics(
        rmse=float(np.sqrt(np.mean(err ** 2))), mae=float(np.mean(np.abs(err))),
        r2=float(1 - np.sum(err ** 2) / ss) if ss > 0 else float("nan"),
        mape=float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-9))),
        fade_skill=float(1 - np.sum(err ** 2) / ss_ref) if ss_ref > 0 else float("nan"),
        rul_true=None if eol_true is None else int(eol_true - n0),
        rul_pred=None if eol_pred is None else int(eol_pred - n0), n_eval=int(mask.sum()),
        censored=censored, rul_true_lb=int(n_obs.max() - n0) if censored else None,
        coverage=coverage, band_width=width, alpha=alpha)


def horizon_residuals(n_obs: np.ndarray, y_obs: np.ndarray, n_grid: np.ndarray, y_grid: np.ndarray,
                      n0: int, lo: Optional[np.ndarray] = None, hi: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Per held-out cycle: horizon h = n - n0, signed error and whether the band covered it."""
    n_obs = np.asarray(n_obs, dtype=float)
    mask = n_obs > n0
    n_e = n_obs[mask]
    y_t = np.asarray(y_obs, dtype=float)[mask]
    err = np.interp(n_e, n_grid, y_grid) - y_t
    out = pd.DataFrame({"h": (n_e - n0).astype(int), "err": err})
    if lo is not None and hi is not None:
        out["in_band"] = (y_t >= np.interp(n_e, n_grid, lo)) & (y_t <= np.interp(n_e, n_grid, hi))
    else:
        out["in_band"] = np.nan
    return out


def prognostic_horizon(bench: pd.DataFrame, alpha: float = 0.2) -> pd.DataFrame:
    """Prognostic horizon per (cell, paradigm) from a forecast-origin sweep (Saxena 2010):
    PH = EOL_true - n_i, where n_i is the first origin after which every predicted EOL stays
    within +/- alpha * EOL_true. 0 if never; NaN for censored cells."""
    rows = []
    for (cell, par), d in bench.groupby(["cell", "paradigm"]):
        d = d.sort_values("n0")
        eol = d["eol_true"].dropna()
        if eol.empty:
            rows.append({"cell": cell, "paradigm": par, "PH_cycles": np.nan, "EOL_true": np.nan})
            continue
        eol_true = float(eol.iloc[0])
        pred = d["n0"] + d["rul_pred"]
        ok = ((pred - eol_true).abs() <= alpha * eol_true).fillna(False).to_numpy()
        ph = 0.0
        for i in range(len(ok)):
            if ok[i:].all():
                ph = eol_true - float(d["n0"].iloc[i])
                break
        rows.append({"cell": cell, "paradigm": par, "PH_cycles": ph, "EOL_true": eol_true})
    return pd.DataFrame(rows)


def metrics_table(metrics: Dict[str, ForecastMetrics]) -> pd.DataFrame:
    rows = []
    for name, m in metrics.items():
        rows.append({"Paradigm": name, "Accuracy (%)": m.accuracy, "Fade skill": m.fade_skill,
                     "RMSE": m.rmse, "MAE": m.mae, "R²": m.r2,
                     "RUL true": m.rul_true, "RUL pred": m.rul_pred, "RUL error": m.rul_error,
                     "RA": m.rel_accuracy, "α-λ ok": m.alpha_lambda_ok,
                     "Censored": m.censored, "RUL lower bound": m.rul_true_lb,
                     "Coverage": m.coverage, "Band width": m.band_width,
                     "held-out cycles": m.n_eval})
    return pd.DataFrame(rows).set_index("Paradigm")


# =============================================================================
# 4. ML SURROGATES
# =============================================================================
GPR_MAX_ROWS = 700                     # exact GP is O(n^3): subsample the training rows
ML_STRATEGIES = ("increment", "direct")
ML_FEATURES = ["n", "Ambient_C", "I_dis_A", "V_cut_V"]            # direct (legacy) strategy
RATE_FEATURES = ["SOH_state", "Ambient_C", "I_dis_A", "V_cut_V", "early_slope", "early_rdc_growth"]


@dataclass(frozen=True)
class HyperParam:
    key: str
    label: str
    kind: str                         # int | float | log | choice | layers
    default: Any
    low: Any = None
    high: Any = None
    options: Tuple[Any, ...] = ()
    help: str = ""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    blurb: str
    params: Tuple[HyperParam, ...]
    scaled: bool = True               # needs feature standardisation
    poly: bool = False                # polynomial feature expansion (degree is a hyperparameter)


_P = HyperParam
MODEL_SPECS: Dict[str, ModelSpec] = {m.name: m for m in (
    ModelSpec("Random Forest", "Tree ensemble (bagging)", "Averages many decorrelated trees; robust default.",
              (_P("n_estimators", "Trees", "int", 300, 50, 1000), _P("max_depth", "Max depth (0 = none)", "int", 0, 0, 30),
               _P("min_samples_leaf", "Min samples per leaf", "int", 3, 1, 30),
               _P("max_features", "Features per split", "choice", 1.0, options=(1.0, "sqrt", "log2", 0.5))), scaled=False),
    ModelSpec("Extra Trees", "Tree ensemble (bagging)", "Randomised split thresholds: smoother, less variance than RF.",
              (_P("n_estimators", "Trees", "int", 400, 50, 1000), _P("max_depth", "Max depth (0 = none)", "int", 0, 0, 30),
               _P("min_samples_leaf", "Min samples per leaf", "int", 2, 1, 30)), scaled=False),
    ModelSpec("Gradient Boosting", "Tree ensemble (boosting)", "Sequential trees fitting residuals; strong on tabular data.",
              (_P("n_estimators", "Stages", "int", 400, 50, 2000), _P("learning_rate", "Learning rate", "log", 0.05, 0.005, 0.5),
               _P("max_depth", "Tree depth", "int", 3, 1, 8), _P("subsample", "Row subsample", "float", 0.8, 0.3, 1.0)),
              scaled=False),
    ModelSpec("Hist. Gradient Boosting", "Tree ensemble (boosting)", "LightGBM-style histogram boosting; fast, handles NaN.",
              (_P("max_iter", "Iterations", "int", 300, 50, 2000), _P("learning_rate", "Learning rate", "log", 0.05, 0.005, 0.5),
               _P("max_leaf_nodes", "Leaves per tree", "int", 31, 4, 128), _P("l2_regularization", "L2 regularisation", "log", 1e-3, 1e-6, 10.0)),
              scaled=False),
    ModelSpec("AdaBoost", "Tree ensemble (boosting)", "Re-weights hard samples; boosted shallow trees.",
              (_P("n_estimators", "Stages", "int", 300, 20, 1000), _P("learning_rate", "Learning rate", "log", 0.1, 0.005, 2.0),
               _P("max_depth", "Base-tree depth", "int", 4, 1, 10)), scaled=False),
    ModelSpec("Decision Tree", "Single tree", "Interpretable piecewise-constant baseline.",
              (_P("max_depth", "Max depth", "int", 8, 1, 30), _P("min_samples_leaf", "Min samples per leaf", "int", 5, 1, 50)),
              scaled=False),
    ModelSpec("k-Nearest Neighbours", "Instance-based", "Averages the k most similar training states.",
              (_P("n_neighbors", "Neighbours k", "int", 10, 1, 60), _P("weights", "Weighting", "choice", "distance", options=("distance", "uniform")),
               _P("p", "Minkowski power (1 = Manhattan, 2 = Euclid)", "int", 2, 1, 3))),
    ModelSpec("Gaussian Process", "Kernel / Bayesian", "Smooth non-parametric fit with native uncertainty (O(n³)).",
              (_P("length_scale", "Initial RBF length scale", "log", 1.0, 0.05, 20.0), _P("noise", "Initial noise level", "log", 0.1, 1e-4, 1.0),
               _P("restarts", "Optimiser restarts", "int", 1, 0, 5))),
    ModelSpec("SVR", "Kernel", "Epsilon-insensitive RBF support-vector regression.",
              (_P("C", "Regularisation C", "log", 10.0, 0.01, 1000.0), _P("epsilon", "Epsilon tube", "log", 0.05, 1e-3, 1.0),
               _P("gamma", "Kernel gamma", "choice", "scale", options=("scale", "auto", 0.01, 0.1, 1.0)))),
    ModelSpec("Kernel Ridge", "Kernel", "Closed-form RBF kernel ridge regression.",
              (_P("alpha", "Regularisation alpha", "log", 0.1, 1e-4, 100.0), _P("gamma", "RBF gamma", "log", 0.1, 1e-3, 10.0))),
    ModelSpec("MLP", "Neural network", "Feed-forward network (ReLU), early stopping optional.",
              (_P("hidden", "Hidden layers", "layers", "64,64", help="comma-separated widths, e.g. 128,64,32"),
               _P("alpha", "L2 penalty", "log", 1e-3, 1e-6, 1.0), _P("learning_rate_init", "Learning rate", "log", 3e-3, 1e-4, 0.1),
               _P("max_iter", "Max epochs", "int", 3000, 200, 10000),
               _P("activation", "Activation", "choice", "relu", options=("relu", "tanh", "logistic")))),
    ModelSpec("Ridge", "Linear (polynomial)", "Polynomial features with L2 shrinkage.",
              (_P("degree", "Polynomial degree", "int", 3, 1, 5), _P("alpha", "Regularisation alpha", "log", 1.0, 1e-4, 1000.0)),
              poly=True),
    ModelSpec("ElasticNet", "Linear (polynomial)", "Polynomial features with L1 + L2 (sparse) shrinkage.",
              (_P("degree", "Polynomial degree", "int", 2, 1, 5), _P("alpha", "Regularisation alpha", "log", 1e-3, 1e-6, 10.0),
               _P("l1_ratio", "L1 ratio", "float", 0.5, 0.0, 1.0)), poly=True),
    ModelSpec("Bayesian Ridge", "Linear (polynomial)", "Polynomial features, evidence-maximised shrinkage.",
              (_P("degree", "Polynomial degree", "int", 3, 1, 5),), poly=True),
)}
ML_MODELS = tuple(MODEL_SPECS)


def default_params(name: str) -> Dict[str, Any]:
    return {h.key: h.default for h in MODEL_SPECS[name].params}


def validate_params(name: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge user hyperparameters over the defaults and range-check them."""
    if name not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {name}")
    out = default_params(name)
    for k, v in (params or {}).items():
        spec = next((h for h in MODEL_SPECS[name].params if h.key == k), None)
        if spec is None:
            raise ValueError(f"{name}: unknown hyperparameter {k!r}")
        if spec.kind in ("int", "float", "log"):
            v = int(v) if spec.kind == "int" else float(v)
            if not (spec.low <= v <= spec.high):
                raise ValueError(f"{name}: {spec.label} must be in [{spec.low}, {spec.high}]")
        elif spec.kind == "choice" and v not in spec.options:
            raise ValueError(f"{name}: {spec.label} must be one of {spec.options}")
        elif spec.kind == "layers":
            widths = [int(x) for x in str(v).replace(" ", "").split(",") if x]
            if not widths or min(widths) < 1 or max(widths) > 1024 or len(widths) > 6:
                raise ValueError(f"{name}: hidden layers must be 1-6 widths in 1..1024")
            v = ",".join(map(str, widths))
        out[k] = v
    return out


def make_model(name: str, seed: int = 0, params: Optional[Dict[str, Any]] = None):
    """Factory for the regressors in MODEL_SPECS, with validated hyperparameters. Every model
    is wrapped with median imputation (health indicators can have gaps) and standardisation
    or polynomial expansion where the family needs it."""
    from sklearn import ensemble as E, linear_model as LM, neighbors, tree
    from sklearn.impute import SimpleImputer
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.svm import SVR

    p = validate_params(name, params)
    spec = MODEL_SPECS[name]
    depth = (lambda d: None if int(d) == 0 else int(d))
    if name == "Random Forest":
        est = E.RandomForestRegressor(n_estimators=p["n_estimators"], max_depth=depth(p["max_depth"]),
                                      min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                                      n_jobs=-1, random_state=seed)
    elif name == "Extra Trees":
        est = E.ExtraTreesRegressor(n_estimators=p["n_estimators"], max_depth=depth(p["max_depth"]),
                                    min_samples_leaf=p["min_samples_leaf"], n_jobs=-1, random_state=seed)
    elif name == "Gradient Boosting":
        est = E.GradientBoostingRegressor(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                                          max_depth=p["max_depth"], subsample=p["subsample"], random_state=seed)
    elif name == "Hist. Gradient Boosting":
        est = E.HistGradientBoostingRegressor(max_iter=p["max_iter"], learning_rate=p["learning_rate"],
                                              max_leaf_nodes=p["max_leaf_nodes"],
                                              l2_regularization=p["l2_regularization"], random_state=seed)
    elif name == "AdaBoost":
        est = E.AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=p["max_depth"]), n_estimators=p["n_estimators"],
                                  learning_rate=p["learning_rate"], random_state=seed)
    elif name == "Decision Tree":
        est = tree.DecisionTreeRegressor(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
                                         random_state=seed)
    elif name == "k-Nearest Neighbours":
        est = neighbors.KNeighborsRegressor(n_neighbors=p["n_neighbors"], weights=p["weights"], p=p["p"])
    elif name == "Gaussian Process":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
        kern = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=p["length_scale"], length_scale_bounds=(1e-2, 1e2)) \
            + WhiteKernel(p["noise"], (1e-6, 1.0))
        est = GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=p["restarts"], random_state=seed)
    elif name == "SVR":
        est = SVR(C=p["C"], epsilon=p["epsilon"], gamma=p["gamma"])
    elif name == "Kernel Ridge":
        est = KernelRidge(alpha=p["alpha"], kernel="rbf", gamma=p["gamma"])
    elif name == "MLP":
        est = MLPRegressor(hidden_layer_sizes=tuple(int(x) for x in p["hidden"].split(",")), alpha=p["alpha"],
                           learning_rate_init=p["learning_rate_init"], max_iter=p["max_iter"],
                           activation=p["activation"], random_state=seed)
    elif name == "Ridge":
        est = LM.Ridge(alpha=p["alpha"])
    elif name == "ElasticNet":
        est = LM.ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"], max_iter=20000)
    elif name == "Bayesian Ridge":
        est = LM.BayesianRidge()
    steps = [SimpleImputer(strategy="median")]
    if spec.scaled:
        steps.append(StandardScaler())
    if spec.poly:
        steps += [PolynomialFeatures(int(p["degree"]), include_bias=False), StandardScaler()]
    return make_pipeline(*steps, est)


class _StandardisedTarget:
    """Fits the regressor on a z-scored target. Fade rates are ~1e-3 per cycle, far below
    SVR's epsilon-tube and the MLP's natural output scale; standardising fixes both."""

    def __init__(self, model):
        self.model = model
        self.mu, self.sd = 0.0, 1.0

    def fit(self, X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> "_StandardisedTarget":
        self.mu = float(np.mean(y))
        self.sd = float(np.std(y)) or 1.0
        yz = (y - self.mu) / self.sd
        kw = _sample_weight_kw(self.model, w) if w is not None else {}
        try:
            self.model.fit(X, yz, **kw)
        except TypeError:
            self.model.fit(X, yz)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X) * self.sd + self.mu


def _sample_weight_kw(model, w: np.ndarray) -> dict:
    """Route sample weights to the final estimator of a pipeline when supported."""
    import inspect
    from sklearn.pipeline import Pipeline
    est = model.steps[-1][1] if isinstance(model, Pipeline) else model
    try:
        ok = "sample_weight" in inspect.signature(est.fit).parameters
    except (TypeError, ValueError):
        ok = False
    if not ok:
        return {}
    return {f"{model.steps[-1][0]}__sample_weight": w} if isinstance(model, Pipeline) else {"sample_weight": w}


def early_life_descriptors(good: pd.DataFrame, n_window: int) -> pd.DataFrame:
    """Per-cell early-life descriptors computed only from cycles n <= n_window (no leakage):
    early_slope       - linear SOH slope (per 100 cycles)
    early_rdc_growth  - relative load-step resistance growth (per 100 cycles)
    Missing values are filled with the cohort median."""
    rows = {}
    for cid, d in good.groupby("Cell_ID"):
        w = d[d["n"] <= max(n_window, 3)].sort_values("n")
        slope = np.nan
        if len(w) >= 3:
            slope = float(np.polyfit(w["n"], w["SOH"], 1)[0]) * 100
        rg = np.nan
        r = w[np.isfinite(w["R_dc_ohm"]) & (w["R_dc_ohm"] > 0)]
        if len(r) >= 4:
            r0 = float(r["R_dc_ohm"].head(3).median())
            if r0 > 0:
                rg = float(np.polyfit(r["n"], r["R_dc_ohm"] / r0, 1)[0]) * 100
        rows[cid] = {"early_slope": slope, "early_rdc_growth": rg}
    out = pd.DataFrame.from_dict(rows, orient="index")
    for c in out.columns:
        med = out[c].median()
        out[c] = out[c].fillna(0.0 if not np.isfinite(med) else med)
    return out


def _rate_samples(d: pd.DataFrame, n_limit: float, h: int = 5, smooth: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """(SOH state, forward fade rate per cycle) pairs from one cell's cycles n <= n_limit."""
    d = d[d["n"] <= n_limit].sort_values("n")
    if len(d) < h + 2:
        return np.empty(0), np.empty(0)
    n = d["n"].to_numpy(dtype=float)
    s = d["SOH"].rolling(smooth, center=True, min_periods=1).median().to_numpy()
    i = np.arange(len(d) - h)
    rate = (s[i + h] - s[i]) / (n[i + h] - n[i])
    return s[i], rate


def _gp_subsample(model_name: str, X: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact GP regression cannot take sample weights and scales as O(n^3): keep at most
    GPR_MAX_ROWS rows, sampling with probability proportional to the weight so the target
    cell's early data stay over-represented as for the other models."""
    if model_name != "Gaussian Process" or len(y) <= GPR_MAX_ROWS:
        return X, y, w
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), GPR_MAX_ROWS, replace=False, p=w / w.sum())
    return X[idx], y[idx], w[idx]


def _ml_curve(good: pd.DataFrame, meta: pd.DataFrame, target: str, n0: int, model_name: str,
              strategy: str, use_population: bool, target_weight: float, seed: int, n_max: int,
              limits: Dict[str, int], params: Optional[Dict[str, Any]] = None,
              train_cells: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """One point forecast. ``limits`` caps the cycles usable per cell (the target and, for
    conformal calibration runs, the calibration cell); other cells are used in full when
    ``use_population``."""
    n_grid = np.arange(1, n_max + 1)
    obs = good[(good["Cell_ID"] == target) & (good["n"] <= n0)].sort_values("n")
    if len(obs) < 5:
        raise ValueError("Not enough target observations before the forecast origin.")

    if train_cells is not None:                      # explicit source cells (+ the limited target/cal cells)
        cells = [c for c in meta.index if c in set(train_cells) or c in limits]
    else:
        cells = [c for c in meta.index if (use_population or c in limits)]
    t0 = time.time()
    if strategy == "direct":
        allowed = np.zeros(len(good), dtype=bool)
        for c in cells:
            m = (good["Cell_ID"] == c).to_numpy()
            if c in limits:
                m = m & (good["n"] <= limits[c]).to_numpy()
            allowed = allowed | m
        if allowed.sum() < 5:
            raise ValueError("Not enough training data for the ML surrogate.")
        f = good[["Cell_ID", "n"]].copy()
        for col in ("Ambient_C", "I_dis_A", "V_cut_V"):
            f[col] = f["Cell_ID"].map(meta[col])
        X = f[ML_FEATURES].to_numpy(dtype=float)[allowed]
        y = good["SOH"].to_numpy()[allowed]
        w = np.where((good["Cell_ID"] == target).to_numpy()[allowed], target_weight, 1.0)
        X, y, w = _gp_subsample(model_name, X, y, w, seed)
        model = _StandardisedTarget(make_model(model_name, seed, params)).fit(X, y, w)
        Xg = np.column_stack([n_grid] + [np.full(len(n_grid), float(meta.loc[target, c]))
                                         for c in ("Ambient_C", "I_dis_A", "V_cut_V")])
        pred = model.predict(Xg)
        return n_grid, pred, int(allowed.sum()), time.time() - t0

    # ---- increment strategy: autonomous fade-rate model, integrated forward ----
    desc = early_life_descriptors(good, n0)
    Xs, ys, ws = [], [], []
    for c in cells:
        d = good[good["Cell_ID"] == c]
        s, r = _rate_samples(d, limits.get(c, np.inf))
        if not len(s):
            continue
        cond = [float(meta.loc[c, k]) for k in ("Ambient_C", "I_dis_A", "V_cut_V")]
        dd = [float(desc.loc[c, k]) if c in desc.index else 0.0 for k in ("early_slope", "early_rdc_growth")]
        Xs.append(np.column_stack([s] + [np.full(len(s), v) for v in cond + dd]))
        ys.append(r)
        ws.append(np.full(len(s), target_weight if c == target else 1.0))
    if not Xs or sum(len(v) for v in ys) < 10:
        raise ValueError("Not enough training data for the ML fade-rate model.")
    X, y, w = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
    X, y, w = _gp_subsample(model_name, X, y, w, seed)
    model = _StandardisedTarget(make_model(model_name, seed, params)).fit(X, y, w)

    # The rate depends on SOH only for a fixed cell -> evaluate once on an SOH grid.
    soh_grid = np.linspace(0.3, 1.1, 401)
    cond = [float(meta.loc[target, k]) for k in ("Ambient_C", "I_dis_A", "V_cut_V")]
    dd = [float(desc.loc[target, k]) if target in desc.index else 0.0 for k in ("early_slope", "early_rdc_growth")]
    Xg = np.column_stack([soh_grid] + [np.full(len(soh_grid), v) for v in cond + dd])
    rate_grid = np.minimum(model.predict(Xg), 0.0)      # regeneration is transient: no net recovery
    s_obs = obs["SOH"].rolling(5, center=True, min_periods=1).median()
    pred = np.interp(n_grid, obs["n"], s_obs).astype(float)
    s = float(obs["SOH"].tail(5).median())
    for i in range(n0, n_max):                          # n_grid[i] = i + 1 > n0
        s = s + float(np.interp(s, soh_grid, rate_grid))
        pred[i] = s
    return n_grid, pred, int(len(y)), time.time() - t0


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
    strategy: str = "increment"
    soh_lo: Optional[np.ndarray] = None
    soh_hi: Optional[np.ndarray] = None
    band_level: Optional[float] = None
    calibration_cells: Tuple[str, ...] = ()


def _conformal_width(good: pd.DataFrame, meta: pd.DataFrame, target: str, n0: int, frac: float,
                     model_name: str, strategy: str, use_population: bool, target_weight: float,
                     seed: int, level: float, n_cal: int, horizon_bin: int = 10,
                     params: Optional[Dict[str, Any]] = None, train_cells: Optional[Sequence[str]] = None
                     ) -> Tuple[Optional[Callable[[np.ndarray], np.ndarray]], Tuple[str, ...]]:
    """Cross-cell, horizon-dependent split-conformal half-width q(h).

    Each calibration cell is forecast with the *same* protocol (origin at the same fraction
    of its life, target's future withheld); absolute errors are pooled per horizon bin and
    the finite-sample-corrected ``level`` quantile is taken, then made non-decreasing in h.
    Validity assumes cells are exchangeable, which is approximately true within a cohort
    tested under similar conditions (calibration partners are chosen that way)."""
    cal = [c for c in calibration_partners(meta, target, n_cal) if int(meta.loc[c, "cycles"]) >= 20]
    if not cal:
        return None, ()
    hs, es = [], []
    for c in cal:
        cyc = int(meta.loc[c, "cycles"])
        n0_c = int(max(5, round(frac * cyc)))
        try:
            n_g, p_g, _, _ = _ml_curve(good, meta, c, n0_c, model_name, strategy, use_population,
                                       target_weight, seed, int(cyc * 1.2) + 1, {target: n0, c: n0_c},
                                       params, train_cells)
        except ValueError:
            continue
        d = good[(good["Cell_ID"] == c) & (good["n"] > n0_c)]
        if d.empty:
            continue
        hs.append(d["n"].to_numpy() - n0_c)
        es.append(np.abs(np.interp(d["n"], n_g, p_g) - d["SOH"].to_numpy()))
    if not hs:
        return None, tuple(cal)
    h, e = np.concatenate(hs), np.concatenate(es)
    edges = np.arange(0, h.max() + horizon_bin, horizon_bin)
    centers, qs = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (h > a) & (h <= b)
        k = int(m.sum())
        if k < 5:
            continue
        lev = min(1.0, level * (1 + 1 / k))
        centers.append(0.5 * (a + b))
        qs.append(float(np.quantile(e[m], lev)))
    if not qs:
        return None, tuple(cal)
    centers, qs = np.array(centers), np.maximum.accumulate(np.array(qs))
    slope = max((qs[-1] - qs[0]) / (centers[-1] - centers[0]), 0.0) if len(qs) > 1 else 0.0

    def width(hh: np.ndarray) -> np.ndarray:
        hh = np.asarray(hh, dtype=float)
        w = np.interp(hh, centers, qs)
        return np.where(hh > centers[-1], qs[-1] + slope * (hh - centers[-1]), w)

    return width, tuple(cal)


def calibration_partners(meta: pd.DataFrame, cell_id: str, n: int) -> List[str]:
    """Nearest cells in operating conditions (ambient in 10 degC units + discharge current
    in A), always returning up to n partners - unlike similar_cells, which only returns
    same-ambient cells when any exist."""
    m = meta.drop(index=cell_id, errors="ignore")
    if m.empty or cell_id not in meta.index:
        return list(m.index[:n])
    ref = meta.loc[cell_id]
    dist = (m["Ambient_C"] - ref["Ambient_C"]).abs() / 10.0 + (m["I_dis_A"] - ref["I_dis_A"]).abs()
    return list(dist.sort_values(kind="mergesort").index[:n])


def train_ml_forecast(ct: pd.DataFrame, cell_id: str, n0: int, model_name: str,
                      use_population: bool = True, target_weight: float = 5.0,
                      eol_ah: float = DEFAULT_EOL_AH, horizon_factor: float = 1.5,
                      seed: int = 0, strategy: str = "increment", conformal_cells: int = 0,
                      band_level: float = 0.9, alpha: float = 0.2,
                      model_params: Optional[Dict[str, Any]] = None,
                      train_cells: Optional[Sequence[str]] = None) -> MLForecast:
    """Fit on (other cells) + (target cell, n <= n0); forecast target n > n0.

    strategy="increment" (default) learns an autonomous fade-rate law dSOH/dn = g(SOH, x),
    with x = operating plan + early-life descriptors, and integrates it from the observed
    SOH at n0. Because the input is the *state* (SOH) rather than the cycle count, tree
    ensembles no longer flat-line beyond the largest cycle count seen in training.
    strategy="direct" is the legacy SOH(n, x) regression, kept for comparison.
    conformal_cells > 0 adds a cross-cell conformal band at ``band_level``.
    model_params: hyperparameters (see MODEL_SPECS). train_cells: explicit source cells
    (e.g. "train on B0005, predict B0006"); the target always contributes only n <= n0."""
    if strategy not in ML_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose one of {ML_STRATEGIES}")
    good = ct[~ct["outlier"]]
    meta = cell_meta(ct)
    tgt = good[good["Cell_ID"] == cell_id]
    if tgt.empty:
        raise DataError(f"{cell_id}: no valid cycles.")
    n_max = int(max(tgt["n"].max(), n0) * horizon_factor)
    model_params = validate_params(model_name, model_params)
    if train_cells is not None:
        train_cells = [c for c in train_cells if c != cell_id and c in meta.index]
        if not train_cells:
            raise ValueError("Choose at least one training cell different from the target.")
    n_grid, pred, rows, fit_s = _ml_curve(good, meta, cell_id, n0, model_name, strategy, use_population,
                                          target_weight, seed, n_max, {cell_id: n0}, model_params, train_cells)
    lo = hi = None
    cal: Tuple[str, ...] = ()
    if conformal_cells > 0:
        frac = n0 / float(meta.loc[cell_id, "cycles"])
        wfn, cal = _conformal_width(good, meta, cell_id, n0, frac, model_name, strategy, use_population,
                                    target_weight, seed, band_level, conformal_cells,
                                    params=model_params, train_cells=train_cells)
        if wfn is not None:
            w = np.where(n_grid > n0, wfn(np.maximum(n_grid - n0, 0)), 0.0)
            lo, hi = pred - w, pred + w
    soh_eol = soh_eol_for(meta.loc[cell_id, "C_bol_Ah"], eol_ah)
    metrics = forecast_metrics(tgt["n"].to_numpy(), tgt["SOH"].to_numpy(), n_grid, pred, n0, soh_eol,
                               lo, hi, alpha)
    return MLForecast(model_name, cell_id, n0, n_grid, pred, metrics, rows, fit_s, strategy,
                      lo, hi, band_level if lo is not None else None, cal)


# =============================================================================
# 5. ECM OBSERVERS
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

    def __post_init__(self) -> None:
        pos = ("C_bol_Ah", "k_ah", "Ea_J_mol", "tau_rc_s", "sigma_v", "q_soc_per_ah", "q_soh_per_ah",
               "q_r_frac_per_ah", "p0_soh", "p0_r_frac", "soc_anchor_sigma", "gate_sigma")
        bad = [k for k in pos if not (_finite(getattr(self, k)) and getattr(self, k) > 0)]
        if bad:
            raise ValueError(f"TwinParameters must be positive: {bad}")
        if not 0 < self.soh_min < self.soh_max:
            raise ValueError("TwinParameters: require 0 < soh_min < soh_max")


def twin_stress(T_C: Any, p: TwinParameters) -> np.ndarray:
    """Degradation stress factor: Arrhenius on cell temperature times a linear cold penalty."""
    T = np.asarray(T_C, dtype=float)
    arr = np.exp(p.Ea_J_mol / R_GAS * (1 / (p.T_ref_C + 273.15) - 1 / (T + 273.15)))
    return arr * (1.0 + p.k_cold * np.maximum(0.0, p.T_cold_C - T))


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

    def values(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        v = np.interp(s, self.soc, self.ocv)
        v = np.where(s < self.soc[0], self.ocv[0] + self.slope_lo * (s - self.soc[0]), v)
        return np.where(s > self.soc[-1], self.ocv[-1] + self.slope_hi * (s - self.soc[-1]), v)

    def slopes(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        d = np.interp(s, self.soc, self.docv)
        d = np.where(s < self.soc[0], self.slope_lo, d)
        return np.where(s > self.soc[-1], self.slope_hi, d)

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


# ---------------------------------------------------------------- joint EKF --
class BatteryDigitalTwin:
    """Closed-loop joint EKF on x = [SOC, V_rc, SOH, R_int, R_ct] with V = OCV + I R_int + V_rc,
    updated at every telemetry sample (legacy observer; see run_dual_twin for the per-cycle
    dual time-scale estimator)."""
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
        self.last_vpred = self.last_residual = self.last_nis = float("nan")

    @property
    def soh_std(self) -> float:
        return math.sqrt(max(self.P[2, 2], 0.0))

    def stress_factor(self, temp_C: float) -> float:
        return float(twin_stress(temp_C, self.params))

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
        self.last_nis = residual * residual / S       # normalised innovation squared, E = 1
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
        self.last_nis = float("nan")
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
    out = np.empty((n, 9))
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
        out[i, 8] = twin.last_nis
        if progress is not None and i % every == 0:
            _report(progress, i / n, "EKF replay")
    hist = pd.DataFrame(out, columns=list(BatteryDigitalTwin.STATE_NAMES) +
                        ["SOH_std", "V_pred", "Residual_V", "NIS"])
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
        stress = twin_stress(d["T_mean_C"].fillna(p.T_ref_C).to_numpy(), p)
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
    per_cycle: pd.DataFrame           # n, Cycle_Index, SOH, SOH_std, R_int, R_ct, innov_mean_mV, innov_std_mV, NIS_norm
    trace: pd.DataFrame               # downsampled per-row trace (joint) or per-cycle table (dual)
    n_updates: int
    n_rejected: int
    runtime_s: float
    kind: str = "joint"               # "joint" (per-sample) or "dual" (per-cycle)
    state_cov: Optional[np.ndarray] = None   # dual: (n_cycles, 4, 4) posterior covariances
    config: Dict[str, Any] = field(default_factory=dict)


def _initial_resistances(imp: Optional[pd.DataFrame], cell_id: str) -> Tuple[float, float]:
    eis = valid_eis(imp, cell_id)
    if len(eis):
        first = eis.head(3)
        return float(first["Re_ohm"].median()), float(first["Rct_ohm"].median())
    return 0.045, 0.07


def run_ekf(cell_df: pd.DataFrame, ct: pd.DataFrame, imp: Optional[pd.DataFrame], cell_id: str,
            params: TwinParameters, calib_cells: Sequence[str], closed_loop: bool = True,
            progress: ProgressFn = None, trace_points: int = 6000) -> EKFResult:
    """Joint per-sample EKF: calibrate the prior on other cells, initialise from the first
    capacity and EIS, replay the full telemetry, summarise per discharge cycle (causal)."""
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
    r_int0, r_ct0 = _initial_resistances(imp, cell_id)
    df = prepare_cell(cell_df)
    ocv = OCVModel.from_discharge(df, 1, r_int0, r_ct0, p.tau_rc_s)
    twin = BatteryDigitalTwin(ocv, p, r_int0=r_int0, r_ct0=r_ct0, closed_loop=closed_loop)
    hist = replay(twin, df, progress=progress)

    eoc = hist.groupby("Cycle_Index")[["SOH", "SOH_std", "R_int", "R_ct"]].last()
    dis = hist[hist["Cycle_Type"] == "discharge"].groupby("Cycle_Index")
    eoc["innov_mean_mV"] = 1000 * dis["Residual_V"].mean()
    eoc["innov_std_mV"] = 1000 * dis["Residual_V"].std()
    eoc["NIS_norm"] = dis["NIS"].mean()
    per = ct_cell[["n", "Cycle_Index"]].merge(eoc.reset_index(), on="Cycle_Index", how="left")
    step = max(1, len(hist) // trace_points)
    trace = hist.iloc[::step][["Cycle_Index", "Cycle_Type", "SOH", "R_int", "R_ct", "Residual_V"]]
    trace = trace.reset_index(drop=True)
    _report(progress, 1.0, "EKF done")
    return EKFResult(cell_id, asdict(p), r_int0, r_ct0, per, trace,
                     twin.n_updates, twin.n_rejected, time.time() - t0, kind="joint",
                     config={"closed_loop": closed_loop})


def ekf_forecast(ekf: EKFResult, n0: int, n_max: int, fit_frac: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy forecast for the joint EKF: linear rate fitted on its own SOH trajectory."""
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


# ------------------------------------------------------ dual time-scale twin --
@dataclass
class DualTwinConfig:
    """Per-cycle (slow time-scale) EKF on theta = [SOH, R_int, R_ct, log k].

    Process (between consecutive discharges, throughput A and stress s(T)):
        SOH' = SOH - k s A,   R' = R + beta R0 k s A,   log k' = log k + w
    Measurements on each discharge (any subset):
      voltage   - subsampled terminal voltage over a *partial* window (first
                  ``voltage_window_frac`` of the beginning-of-life capacity), so the
                  cut-off time - which would reveal capacity directly - is never seen;
      load step - R_dc = (V_rest - V_1) / |I_1|  ~  R_int + R_ct (1 - e^(-dt/tau));
      capacity  - full discharged capacity (off by default: operando cells rarely see
                  a full reference discharge)."""
    use_voltage: bool = True
    use_rdc: bool = True
    use_capacity: bool = False
    voltage_window_frac: float = 0.6
    n_voltage_pts: int = 25
    sigma_v: float = 0.015
    sigma_rdc_frac: float = 0.15
    sigma_q_Ah: float = 0.01
    q_soh: float = 1e-3
    q_logk: float = 0.03
    q_r_frac: float = 0.01
    p0_soh: float = 0.01
    p0_logk: float = 0.7
    p0_r_frac: float = 0.15
    robust_nis_quantile: float = 0.999
    update_every: int = 1             # assimilate measurements every m-th discharge (Mission 2)

    def __post_init__(self) -> None:
        if int(self.update_every) < 1:
            raise ValueError("update_every must be >= 1")
        if not 0 < self.voltage_window_frac <= 1:
            raise ValueError("voltage_window_frac must be in (0, 1]")
        if self.n_voltage_pts < 3:
            raise ValueError("n_voltage_pts must be >= 3")
        pos = ("sigma_v", "sigma_rdc_frac", "sigma_q_Ah", "q_soh", "q_logk", "q_r_frac",
               "p0_soh", "p0_logk", "p0_r_frac")
        bad = [k for k in pos if not getattr(self, k) > 0]
        if bad:
            raise ValueError(f"DualTwinConfig must be positive: {bad}")
        if not 0.5 < self.robust_nis_quantile < 1:
            raise ValueError("robust_nis_quantile must be in (0.5, 1)")


TWIN_ABLATIONS: Dict[str, Dict[str, bool]] = {
    "Open loop (population prior)": dict(use_voltage=False, use_rdc=False, use_capacity=False),
    "Voltage (partial window)": dict(use_voltage=True, use_rdc=False, use_capacity=False),
    "Voltage + load-step R": dict(use_voltage=True, use_rdc=True, use_capacity=False),
    "Voltage + R + capacity": dict(use_voltage=True, use_rdc=True, use_capacity=True),
}


@dataclass
class _DualPrep:
    table: pd.DataFrame
    arrays: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, float]]
    ocv: OCVModel
    c_bol: float
    r_int0: float
    r_ct0: float
    k_prior: float
    params: TwinParameters


def _discharge_arrays(df: pd.DataFrame) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """cycle -> (I, dt, V of loaded samples, rest voltage before the load step)."""
    out = {}
    dis = df[df["Cycle_Type"] == "discharge"]
    for ci, d in dis.groupby("Cycle_Index", sort=True):
        I = d["Current_A"].to_numpy(dtype=float)
        V = d["Voltage_V"].to_numpy(dtype=float)
        dt = d["dt"].to_numpy(dtype=float)
        v_rest = float(V[0]) if abs(I[0]) < 0.1 else float("nan")
        load = I < -0.1
        if load.sum() < 10:
            continue
        out[int(ci)] = (I[load], dt[load], V[load], v_rest)
    return out


def _dual_prepare(cell_df: pd.DataFrame, ct: pd.DataFrame, imp: Optional[pd.DataFrame], cell_id: str,
                  params: TwinParameters, calib_cells: Sequence[str]) -> _DualPrep:
    p = TwinParameters(**asdict(params))
    ct_cell = ct[ct["Cell_ID"] == cell_id].sort_values("n").reset_index(drop=True)
    if ct_cell.empty:
        raise DataError(f"{cell_id}: no discharge capacity data.")
    try:
        k_prior = calibrate_k_ah_from_table(ct, [c for c in calib_cells if c != cell_id], p)
    except ValueError:
        k_prior = p.k_ah
    c_bol = float(ct_cell["C_bol_Ah"].iloc[0])
    p.C_bol_Ah = c_bol
    r_int0, r_ct0 = _initial_resistances(imp, cell_id)
    df = prepare_cell(cell_df)
    ocv = OCVModel.from_discharge(df, 1, r_int0, r_ct0, p.tau_rc_s)
    return _DualPrep(ct_cell, _discharge_arrays(df), ocv, c_bol, r_int0, r_ct0, k_prior, p)


def _dual_measurement(x: np.ndarray, arr: Tuple[np.ndarray, np.ndarray, np.ndarray, float],
                      prep: _DualPrep, cfg: DualTwinConfig, capacity: float):
    """Stacked measurement (z, h(x), H, R diag, voltage residuals) for one discharge."""
    I, dt, V, v_rest = arr
    p, ocv, C = prep.params, prep.ocv, prep.c_bol
    soh = max(float(x[0]), 0.3)
    cum = np.cumsum(np.abs(I) * dt) / 3600.0
    soc = 1.0 - cum / (C * soh)
    a = np.exp(-dt / p.tau_rc_s)
    g = np.empty(len(I))                    # V_rc = R_ct * g  (linear in R_ct)
    acc = 0.0
    for j in range(len(I)):
        acc = a[j] * acc + (1.0 - a[j]) * I[j]
        g[j] = acc
    z, h, H, R = [], [], [], []
    vres = np.empty(0)
    if cfg.use_voltage:
        idx = np.flatnonzero(cum <= cfg.voltage_window_frac * C)
        if len(idx) >= 5:
            sel = np.unique(idx[np.linspace(0, len(idx) - 1, cfg.n_voltage_pts).astype(int)])
            vp = ocv.values(soc[sel]) + I[sel] * x[1] + x[2] * g[sel]
            dv_dsoh = ocv.slopes(soc[sel]) * cum[sel] / (C * soh ** 2)
            z.append(V[sel])
            h.append(vp)
            H.append(np.column_stack([dv_dsoh, I[sel], g[sel], np.zeros(len(sel))]))
            R.append(np.full(len(sel), cfg.sigma_v ** 2))
            vres = V[sel] - vp
    if cfg.use_rdc and np.isfinite(v_rest) and abs(I[0]) > 0.1:
        i0 = abs(I[0])
        zr = (v_rest - V[0]) / i0
        hr = x[1] + x[2] * (1 - a[0]) + (ocv(1.0) - ocv(float(soc[0]))) / i0
        dh_dsoh = -ocv.slope(float(soc[0])) * cum[0] / (C * soh ** 2) / i0
        if 0 < zr < 1.0:
            z.append(np.array([zr]))
            h.append(np.array([hr]))
            H.append(np.array([[dh_dsoh, 1.0, 1 - a[0], 0.0]]))
            R.append(np.array([(cfg.sigma_rdc_frac * max(hr, 1e-3)) ** 2]))
    if cfg.use_capacity and _finite(capacity):
        z.append(np.array([float(capacity)]))
        h.append(np.array([C * soh]))
        H.append(np.array([[C, 0.0, 0.0, 0.0]]))
        R.append(np.array([cfg.sigma_q_Ah ** 2]))
    if not z:
        return None
    return np.concatenate(z), np.concatenate(h), np.vstack(H), np.concatenate(R), vres


def _dual_filter(prep: _DualPrep, cfg: DualTwinConfig, progress: ProgressFn = None
                 ) -> Tuple[pd.DataFrame, np.ndarray, int, int]:
    from scipy.stats import chi2

    p = prep.params
    x = np.array([1.0, prep.r_int0, prep.r_ct0, math.log(prep.k_prior)])
    P = np.diag([cfg.p0_soh ** 2, (cfg.p0_r_frac * prep.r_int0) ** 2,
                 (cfg.p0_r_frac * prep.r_ct0) ** 2, cfg.p0_logk ** 2])
    Qd = np.diag([cfg.q_soh ** 2, (cfg.q_r_frac * prep.r_int0) ** 2,
                  (cfg.q_r_frac * prep.r_ct0) ** 2, cfg.q_logk ** 2])
    lk_lo, lk_hi = math.log(1e-7), math.log(5e-2)
    rows, covs = [], []
    prev_cum: Optional[float] = None
    n_upd = n_infl = 0
    N = len(prep.table)
    any_meas = cfg.use_voltage or cfg.use_rdc or cfg.use_capacity
    for i, r in enumerate(prep.table.itertuples(index=False)):
        T = float(r.T_mean_C) if _finite(r.T_mean_C) else p.T_ref_C
        s = float(twin_stress(T, p))
        A = 0.0
        if _finite(r.cum_Ah):
            if prev_cum is not None:
                A = max(float(r.cum_Ah) - prev_cum, 0.0)
            prev_cum = float(r.cum_Ah)
        if i > 0:
            d = math.exp(x[3]) * s * A
            cold_R = 1.0 + p.k_cold_R * max(0.0, p.T_cold_C - T)
            gi, gc = p.beta_int * prep.r_int0 * cold_R, p.beta_ct * prep.r_ct0 * cold_R
            F = np.eye(4)
            F[0, 3], F[1, 3], F[2, 3] = -d, gi * d, gc * d
            x = x + np.array([-d, gi * d, gc * d, 0.0])
            P = F @ P @ F.T + Qd
        innov_mean = innov_std = nis_norm = float("nan")
        ig_soh = ig_logk = 0.0
        m = 0
        arr = prep.arrays.get(int(r.Cycle_Index))
        due = (i % int(cfg.update_every)) == 0
        if any_meas and due and not bool(r.outlier) and arr is not None:
            p_soh0, p_lk0 = float(P[0, 0]), float(P[3, 3])
            meas = _dual_measurement(x, arr, prep, cfg, r.Capacity_Ah)
            if meas is not None:
                z, hx, H, Rv, vres = meas
                m = len(z)
                res = z - hx
                S = H @ P @ H.T + np.diag(Rv)
                nis = float(res @ np.linalg.solve(S, res))
                thr = float(chi2.ppf(cfg.robust_nis_quantile, m))
                if nis > thr:                       # robust: inflate R instead of rejecting
                    Rv = Rv * (nis / thr)
                    S = H @ P @ H.T + np.diag(Rv)
                    n_infl += 1
                K = np.linalg.solve(S, H @ P).T
                x = x + K @ res
                IKH = np.eye(4) - K @ H
                P = IKH @ P @ IKH.T + K @ np.diag(Rv) @ K.T
                P = 0.5 * (P + P.T)
                n_upd += 1
                nis_norm = nis / m
                ig_soh = 0.5 * math.log(max(p_soh0, 1e-30) / max(float(P[0, 0]), 1e-30))
                ig_logk = 0.5 * math.log(max(p_lk0, 1e-30) / max(float(P[3, 3]), 1e-30))
                if len(vres):
                    innov_mean, innov_std = 1000 * float(vres.mean()), 1000 * float(vres.std())
        x[0] = min(max(x[0], 0.3), 1.2)
        x[1] = max(x[1], 1e-4)
        x[2] = max(x[2], 1e-4)
        x[3] = min(max(x[3], lk_lo), lk_hi)
        rows.append({"n": int(r.n), "Cycle_Index": int(r.Cycle_Index), "SOH": x[0],
                     "SOH_std": math.sqrt(max(P[0, 0], 0.0)), "R_int": x[1], "R_ct": x[2],
                     "k_ah": math.exp(x[3]), "logk_std": math.sqrt(max(P[3, 3], 0.0)),
                     "innov_mean_mV": innov_mean, "innov_std_mV": innov_std,
                     "NIS_norm": nis_norm, "n_meas": m, "stress": s, "dAh": A,
                     "ig_soh": ig_soh, "ig_logk": ig_logk})
        covs.append(P.copy())
        if progress is not None and i % 10 == 0:
            _report(progress, i / max(N, 1), "dual twin")
    return pd.DataFrame(rows), np.array(covs), n_upd, n_infl


def run_dual_twin(cell_df: pd.DataFrame, ct: pd.DataFrame, imp: Optional[pd.DataFrame], cell_id: str,
                  params: TwinParameters, cfg: Optional[DualTwinConfig] = None,
                  calib_cells: Sequence[str] = (), progress: ProgressFn = None,
                  _prep: Optional[_DualPrep] = None) -> EKFResult:
    """Dual time-scale ECM twin (per-cycle EKF, see DualTwinConfig). Estimates are causal:
    the state at cycle n uses only data up to n. The degradation-rate prior log k comes
    from cohort calibration partners, and the filter personalises it with the cell's own
    measurements - this is what makes the twin 'self-updating'."""
    t0 = time.time()
    cfg = cfg or DualTwinConfig()
    prep = _prep or _dual_prepare(cell_df, ct, imp, cell_id, params, calib_cells)
    per, covs, n_upd, n_infl = _dual_filter(prep, cfg, progress)
    pr = asdict(prep.params)
    pr.update({"k_ah": float(per["k_ah"].iloc[-1]), "k_prior": prep.k_prior})
    _report(progress, 1.0, "dual twin done")
    return EKFResult(cell_id, pr, prep.r_int0, prep.r_ct0, per, per.copy(), n_upd, n_infl,
                     time.time() - t0, kind="dual", state_cov=covs, config=asdict(cfg))


@dataclass
class TwinForecast:
    n_grid: np.ndarray
    soh: np.ndarray
    lo: Optional[np.ndarray]
    hi: Optional[np.ndarray]
    rul_samples: Optional[np.ndarray]       # cycles after n0; NaN = no EOL within horizon
    level: float


def twin_forecast(res: EKFResult, n0: int, n_max: int, soh_eol: float, level: float = 0.9,
                  n_samples: int = 500, seed: int = 0) -> TwinForecast:
    """Forecast from the observer state at n0.

    dual: Monte-Carlo propagation of the calibrated fade law. (SOH, log k) are sampled from
    the posterior at n0 (joint Gaussian), log k continues its random walk, and the planned
    operating point (stress x throughput per cycle) is the mean of the last 20 observed
    cycles. Returns the median path, a central ``level`` band and the RUL distribution.
    joint: legacy linear extrapolation without a band."""
    if res.kind != "dual" or res.state_cov is None:
        n_g, p_g = ekf_forecast(res, n0, n_max)
        return TwinForecast(n_g, p_g, None, None, None, level)
    per = res.per_cycle
    obs = per[(per["n"] <= n0) & per["SOH"].notna()]
    if len(obs) < 3:
        raise ValueError("Forecast origin too early for the twin forecast.")
    i = int(obs.index[-1])
    mu = np.array([per.at[i, "SOH"], math.log(per.at[i, "k_ah"])])
    C = res.state_cov[i][np.ix_([0, 3], [0, 3])]
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    C = (V * np.maximum(w, 1e-12)) @ V.T
    recent = obs.tail(20)
    load = recent.loc[recent["dAh"] > 0, "stress"] * recent.loc[recent["dAh"] > 0, "dAh"]
    sA = float(load.mean()) if len(load) else float((recent["stress"] * recent["dAh"]).mean())
    q_logk = float(res.config.get("q_logk", 0.03))
    rng = np.random.default_rng(seed)
    smp = rng.multivariate_normal(mu, C, size=n_samples)
    H = max(n_max - n0, 1)
    logk = smp[:, [1]] + np.cumsum(q_logk * rng.standard_normal((n_samples, H)), axis=1)
    paths = smp[:, [0]] - np.cumsum(np.exp(logk) * sA, axis=1)
    n_grid = np.arange(1, n_max + 1)
    med = np.interp(n_grid, obs["n"], obs["SOH"]).astype(float)
    lo, hi = med.copy(), med.copy()
    qa, qb = (1 - level) / 2, 1 - (1 - level) / 2
    fut = n_grid > n0
    k = int(fut.sum())
    med[fut] = np.median(paths[:, :k], axis=0)
    lo[fut] = np.quantile(paths[:, :k], qa, axis=0)
    hi[fut] = np.quantile(paths[:, :k], qb, axis=0)
    below = paths < soh_eol
    hit = below.any(axis=1)
    rul = np.where(hit, below.argmax(axis=1) + 1, np.nan).astype(float)
    return TwinForecast(n_grid, med, lo, hi, rul, level)


def twin_ablation(cell_df: pd.DataFrame, ct: pd.DataFrame, imp: Optional[pd.DataFrame], cell_id: str,
                  params: TwinParameters, base_cfg: Optional[DualTwinConfig], calib_cells: Sequence[str],
                  n0: int, soh_eol: float, level: float = 0.9, progress: ProgressFn = None
                  ) -> Tuple[pd.DataFrame, Dict[str, EKFResult]]:
    """Does voltage feedback add information? Runs the dual twin with measurement subsets
    (open loop = population prior only) and reports causal tracking error over the whole
    life, filter consistency (NIS) and the forecast from n0."""
    base = base_cfg or DualTwinConfig()
    prep = _dual_prepare(cell_df, ct, imp, cell_id, params, calib_cells)
    good = ct[(ct["Cell_ID"] == cell_id) & ~ct["outlier"]].sort_values("n")
    n_max = int(good["n"].max() * 1.5)
    rows, results = [], {}
    for j, (name, flags) in enumerate(TWIN_ABLATIONS.items()):
        _report(progress, j / len(TWIN_ABLATIONS), name)
        cfg = replace(base, **flags)
        r = run_dual_twin(cell_df, ct, imp, cell_id, params, cfg, calib_cells, _prep=prep)
        results[name] = r
        est = good[["n", "SOH"]].merge(r.per_cycle[["n", "SOH", "SOH_std"]], on="n", suffixes=("", "_est"))
        err = est["SOH_est"] - est["SOH"]
        cover = float(np.mean(np.abs(err) <= 1.645 * est["SOH_std"]))
        fc = twin_forecast(r, n0, n_max, soh_eol, level)
        fm = forecast_metrics(good["n"].to_numpy(), good["SOH"].to_numpy(), fc.n_grid, fc.soh, n0,
                              soh_eol, fc.lo, fc.hi)
        rows.append({"Configuration": name, "Tracking RMSE": float(np.sqrt(np.mean(err ** 2))),
                     "Tracking 90% coverage": cover, "Mean NIS / dof": float(r.per_cycle["NIS_norm"].mean()),
                     "k personalised / prior": float(r.per_cycle["k_ah"].iloc[-1] / prep.k_prior),
                     "Info gain SOH (nats/100 cyc)": 100 * float(r.per_cycle["ig_soh"].mean()),
                     "Info gain log k (nats/100 cyc)": 100 * float(r.per_cycle["ig_logk"].mean()),
                     "Forecast RMSE": fm.rmse, "RUL error": fm.rul_error, "Forecast coverage": fm.coverage})
    _report(progress, 1.0, "ablation done")
    return pd.DataFrame(rows).set_index("Configuration"), results


# =============================================================================
# 6. MINIMAL REVERSE-MODE AUTODIFF (numpy)
# =============================================================================
class Tensor:
    """Tiny reverse-mode autodiff tensor. Enough for PINN losses that contain first
    derivatives of the network w.r.t. its input (propagated as ordinary tensor ops).
    Verified against central finite differences by ``gradient_check`` (see tests)."""
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


def gradient_check(loss_fn: Callable[[], Tensor], params: Sequence[Tensor], eps: float = 1e-6,
                   max_entries: int = 40, seed: int = 0) -> float:
    """Max relative error between reverse-mode gradients and central finite differences
    over up to ``max_entries`` randomly chosen parameter entries."""
    rng = np.random.default_rng(seed)
    loss = loss_fn()
    loss.backward()
    analytic = [p.grad.copy() if p.grad is not None else np.zeros_like(p.data) for p in params]
    worst = 0.0
    entries = [(i, j) for i, p in enumerate(params) for j in range(p.data.size)]
    pick = rng.choice(len(entries), size=min(max_entries, len(entries)), replace=False)
    for e in pick:
        i, j = entries[e]
        flat = params[i].data.reshape(-1)
        old = flat[j]
        flat[j] = old + eps
        up = float(loss_fn().data)
        flat[j] = old - eps
        dn = float(loss_fn().data)
        flat[j] = old
        fd = (up - dn) / (2 * eps)
        an = float(analytic[i].reshape(-1)[j])
        worst = max(worst, abs(fd - an) / max(1e-7, abs(fd) + abs(an)))
    return worst


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
EA_MODES = ("fixed", "pooled", "learned")


PINN_PHYSICS = ("lumped", "mechanistic")


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
    ea_mode: str = "fixed"            # fixed | pooled | learned  (see HybridPINN)
    ea_fixed_J_mol: float = 30e3
    init_jitter: float = 0.0          # prior-randomised physical initialisation (ensembles)
    # ---- mechanism-resolved physics (MechanisticPINN) ----
    physics: str = "lumped"           # lumped (single fade law) | mechanistic (SEI + plating + LAM)
    use_sei: bool = True
    use_plating: bool = True
    use_lam: bool = True
    lambda_volt: float = 0.3          # mean-discharge-voltage (electrochemical) consistency
    Ea_sei_J_mol: float = 30e3        # SEI growth (solvent reduction / diffusion)
    Ea_plating_J_mol: float = 50e3    # apparent, *negative* temperature dependence of plating
    Ea_lam_J_mol: float = 20e3        # particle cracking / dissolution
    Ea_ct_J_mol: float = 40e3         # charge-transfer kinetics (exchange current)
    beta_lam: float = 1.0             # C-rate exponent of LAM
    I_charge_A: float = 1.5           # NASA protocol CC charge current
    T_plating_onset_C: float = 10.0   # cold-plating onset (smooth gate width 3 K)
    lambda_prior: float = 0.02        # weak log-normal priors on the rate constants (identifiability)

    def __post_init__(self) -> None:
        if self.physics not in PINN_PHYSICS:
            raise ValueError(f"physics must be one of {PINN_PHYSICS}")
        if self.physics == "mechanistic" and not (self.use_sei or self.use_plating or self.use_lam):
            raise ValueError("mechanistic PINN needs at least one degradation mechanism")
        if self.ea_mode not in EA_MODES:
            raise ValueError(f"ea_mode must be one of {EA_MODES}")
        if self.hidden < 2 or self.epochs < 1 or self.n_colloc < 5:
            raise ValueError("PINNConfig: hidden >= 2, epochs >= 1, n_colloc >= 5")
        lams = (self.lambda_data, self.lambda_phys, self.lambda_bv, self.lambda_eis)
        if min(lams) < 0:
            raise ValueError("PINNConfig: loss weights must be non-negative")
        if self.init_jitter < 0:
            raise ValueError("PINNConfig: init_jitter must be non-negative")
        if not (self.lr > 0 and 5e3 <= self.ea_fixed_J_mol <= 150e3):
            raise ValueError("PINNConfig: lr > 0 and 5 <= Ea_fixed <= 150 kJ/mol")


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
    soh_lo: Optional[np.ndarray] = None
    soh_hi: Optional[np.ndarray] = None
    physics_table: Optional[pd.DataFrame] = None     # ensemble mean / std / cv / status per parameter
    n_members: int = 1
    ea_mode: str = "fixed"
    ea_J_mol: Optional[float] = None
    mechanisms: Optional[pd.DataFrame] = None        # n, Q_SEI, Q_plating, Q_LAM (mechanistic physics)
    physics_kind: str = "lumped"


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
                       m = 0 linear in throughput, m < 0 self-accelerating (knee-like).
      Resistance-growth kinetics coupled to fade:
        dR_int/dn = gamma_int R_int0 (-dSOH/dn),   dR_ct/dn = gamma_ct R_ct0 (-dSOH/dn)
      Butler-Volmer (symmetric, alpha = 0.5) at the discharge load step, with the
      exchange current linked to the small-signal R_ct (i0 = R T / (F R_ct)):
        dV_step = I (R_int + R_x) + (2 R T / F) asinh( I F R_ct / (2 R T) )

    Identifiability of Ea: a single cell runs at one ambient temperature, so the
    collocation temperatures barely vary and Ea cannot be learned from that cell alone.
    ea_mode = "fixed" (literature value) or "pooled" (cohort Arrhenius regression, see
    estimate_pooled_arrhenius) removes Ea from the optimiser; "learned" keeps the legacy
    behaviour and should be read together with the ensemble spread."""

    def __init__(self, cfg: PINNConfig, c_bol: float, r_int0: float, r_ct0: float, n_scale: float,
                 ea_value: Optional[float] = None):
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
        # physical parameters (unconstrained). With init_jitter > 0 each seed starts from a
        # different draw around the nominal values, so an ensemble reveals parameters the
        # data do not constrain (they stay spread out) instead of hiding them.
        j = cfg.init_jitter
        z = (lambda: j * float(rng.standard_normal())) if j > 0 else (lambda: 0.0)
        self.log_k = Tensor(math.log(5e-4) + z())
        self.ea_raw = Tensor(_logit((30e3 - 10e3) / 70e3) + 2 * z())
        self.m_raw = Tensor(0.0 + 2 * z())            # m = 0.5 at start (no jitter)
        self.gi_raw = Tensor(_inv_softplus(1.0) + z())
        self.gc_raw = Tensor(_inv_softplus(3.0) + z())
        self.rx_raw = Tensor(_inv_softplus(0.5) + z())
        self.ea_const: Optional[float] = None
        if cfg.ea_mode != "learned":
            self.ea_const = float(ea_value if ea_value is not None else cfg.ea_fixed_J_mol)

    @property
    def parameters(self) -> List[Tensor]:
        ps = [self.W1, self.b1, self.W2, self.b2, self.Ws, self.bs, self.Wi, self.bi,
              self.Wc, self.bc, self.log_k, self.m_raw, self.gi_raw, self.gc_raw, self.rx_raw]
        if self.ea_const is None:
            ps.append(self.ea_raw)
        return ps

    # constrained physical parameters
    def k(self) -> Tensor: return self.log_k.exp()

    def Ea(self) -> Tensor:
        if self.ea_const is not None:
            return Tensor(self.ea_const)
        return 10e3 + 70e3 * self.ea_raw.sigmoid()

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

    def monitor(self) -> Dict[str, float]:
        return {"k_per_Ah": float(self.k().data), "Ea_kJ_mol": float(self.Ea().data) / 1e3,
                "m": float(self.m().data)}

    def mechanism_table(self, n: np.ndarray) -> Optional[pd.DataFrame]:
        return None

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


class MechanisticPINN(HybridPINN):
    """Mechanism-resolved hybrid PINN. The network carries three latent capacity-loss states
    (fractions of C_bol) with built-in initial conditions, Q_x = t * softplus(o_x), and
        SOH = 1 - Q_SEI - Q_pl - Q_LAM
    Degradation kinetics per discharge cycle n (throughput per cycle A_n = 2 C_bol SOH,
    Arrhenius factor A(T; Ea) = exp[Ea/R (1/T_ref - 1/T)]):

    SEI growth, mixed reaction/diffusion control (Ploehn 2004; Pinson & Bazant 2013):
        dQ_SEI/dn = k_SEI A(T; Ea_SEI) (A_n / C_bol) / (1 + Q_SEI / delta)
        Q_SEI << delta: reaction-limited, linear; Q_SEI >> delta: diffusion-limited, sqrt(n).
    Lithium plating, favoured by cold, charge current and pore clogging (Waldmann 2014;
    Yang et al. 2017 - the plating/porosity feedback that produces the knee):
        dQ_pl/dn = k_pl exp[Ea_pl/R (1/T - 1/T_ref)] (I_ch / C_bol) (g_cold(T) + kappa Q_LAM / 0.05)
        g_cold = 1 / (1 + exp((T - T_onset) / 3 K)): cold plating; the kappa term is LAM-triggered
        plating, which also occurs at room temperature once pores clog.
    Weak log-normal priors on k_SEI, k_pl, k_LAM keep the split identifiable on one cell.
    Loss of active material by particle cracking / dissolution, rate-driven and
    self-accelerating as the remaining material carries more current (Laresgoiti 2015):
        dQ_LAM/dn = k_LAM A(T; Ea_LAM) (I_dis / C_bol)^beta (A_n / C_bol) (1 + Q_LAM / eps)
    Resistance, from the mechanisms:
        dR_int/dn = R_int0 rho_SEI dQ_SEI/dn / 0.1          SEI film resistance ~ thickness
        dR_ct/dn  = R_ct0 (g_LAM dQ_LAM/dn + g_SEI dQ_SEI/dn) / 0.1   active-area loss, blocking film
    Electrochemistry (terminal voltage), with Arrhenius charge-transfer kinetics
    R_ct(T) = R_ct exp[Ea_ct/R (1/T - 1/T_ref)]:
        Load step:   dV = I (R_int + R_x) + (2RT/F) asinh(I F R_ct(T) / (2RT))      (Butler-Volmer)
        Discharge:   V_mean = U_bar - I (R_int + R_x) - (2RT/F) asinh(I F R_ct(T) / (2RT))
    Disabled mechanisms are removed from the network and the balance."""

    MECHS = ("SEI", "plating", "LAM")

    def __init__(self, cfg: PINNConfig, c_bol: float, r_int0: float, r_ct0: float, n_scale: float,
                 ea_value: Optional[float] = None):
        super().__init__(cfg, c_bol, r_int0, r_ct0, n_scale, ea_value)
        rng = np.random.default_rng(cfg.seed + 7)
        H = cfg.hidden
        j = cfg.init_jitter
        z = (lambda: j * float(rng.standard_normal())) if j > 0 else (lambda: 0.0)
        self.on = {"SEI": cfg.use_sei, "plating": cfg.use_plating, "LAM": cfg.use_lam}
        self.Wq = Tensor(rng.normal(0, math.sqrt(2.0 / (H + 3)), size=(H, 3)))
        self.bq = Tensor([[_inv_softplus(0.15), _inv_softplus(0.01), _inv_softplus(0.05)]])
        self.log_ksei = Tensor(math.log(4e-4) + z())
        self.delta_raw = Tensor(_inv_softplus(0.05) + z())
        self.log_kpl = Tensor(math.log(2e-5) + z())
        self.kappa_raw = Tensor(_inv_softplus(1.0) + z())
        self.log_klam = Tensor(math.log(1e-4) + z())
        self.eps_raw = Tensor(_inv_softplus(0.05) + z())
        self.rho_raw = Tensor(_inv_softplus(1.0) + z())
        self.gl_raw = Tensor(_inv_softplus(2.0) + z())
        self.gs_raw = Tensor(_inv_softplus(1.0) + z())
        self.u_raw = Tensor(_inv_softplus(3.7 - 3.0))     # U_bar = 3.0 + softplus(.)

    @property
    def parameters(self) -> List[Tensor]:
        ps = [self.W1, self.b1, self.W2, self.b2, self.Wq, self.bq, self.Wi, self.bi, self.Wc, self.bc,
              self.rx_raw, self.u_raw, self.rho_raw, self.gs_raw]
        if self.on["SEI"]:
            ps += [self.log_ksei, self.delta_raw]
        if self.on["plating"]:
            ps += [self.log_kpl, self.kappa_raw]
        if self.on["LAM"]:
            ps += [self.log_klam, self.eps_raw, self.gl_raw]
        return ps

    # constrained parameters
    def k_sei(self) -> Tensor: return self.log_ksei.exp()
    def delta(self) -> Tensor: return self.delta_raw.softplus() + 1e-3
    def k_pl(self) -> Tensor: return self.log_kpl.exp()
    def kappa(self) -> Tensor: return self.kappa_raw.softplus()
    def k_lam(self) -> Tensor: return self.log_klam.exp()
    def eps(self) -> Tensor: return self.eps_raw.softplus() + 1e-3
    def rho(self) -> Tensor: return self.rho_raw.softplus()
    def g_lam(self) -> Tensor: return self.gl_raw.softplus()
    def g_sei(self) -> Tensor: return self.gs_raw.softplus()
    def U_bar(self) -> Tensor: return 3.0 + self.u_raw.softplus()
    def k(self) -> Tensor: return self.k_sei()                   # reporting compatibility

    def states(self, n: np.ndarray) -> Dict[str, Tensor]:
        t = Tensor(np.asarray(n, dtype=float).reshape(-1, 1) / self.n_scale)
        h1 = (t @ self.W1 + self.b1).tanh()
        g1 = (1 - h1.square()) * self.W1
        h2 = (h1 @ self.W2 + self.b2).tanh()
        g2 = (1 - h2.square()) * (g1 @ self.W2)
        oq, gq = h2 @ self.Wq + self.bq, g2 @ self.Wq
        spq, sgq = oq.softplus(), oq.sigmoid()
        Q = t * spq                                             # (N, 3) latent losses
        dQ = (spq + t * sgq * gq) / self.n_scale                # d/dn
        mask = np.array([[float(self.on[m]) for m in self.MECHS]])
        Q, dQ = Q * mask, dQ * mask
        ones = np.ones((3, 1))
        out: Dict[str, Tensor] = {"Q": Q, "dQ": dQ, "SOH": 1 - Q @ Tensor(ones), "dSOH": -(dQ @ Tensor(ones))}
        for key, W, b, r0 in (("R_int", self.Wi, self.bi, self.r_int0), ("R_ct", self.Wc, self.bc, self.r_ct0)):
            o, go = h2 @ W + b, g2 @ W
            sp, sg = o.softplus(), o.sigmoid()
            out[key] = r0 * (1 + t * sp)
            out["d" + key] = r0 * (sp + t * sg * go) / self.n_scale
        return out

    @staticmethod
    def _col(M: Tensor, i: int) -> Tensor:
        sel = np.zeros((M.data.shape[1], 1))
        sel[i, 0] = 1.0
        return M @ Tensor(sel)

    def _arr(self, Ea: float, T: np.ndarray) -> np.ndarray:
        return np.exp(Ea / R_GAS * (1.0 / (self.cfg.T_ref_C + 273.15) - 1.0 / T))

    def _eta_ct(self, I: np.ndarray, Tk: np.ndarray, r_ct: Tensor) -> Tensor:
        vt = 2 * R_GAS * Tk / FARADAY
        r_T = r_ct * np.exp(self.cfg.Ea_ct_J_mol / R_GAS * (1.0 / Tk - 1.0 / (self.cfg.T_ref_C + 273.15)))
        return vt * (r_T * (I / vt)).asinh()

    def losses(self, data: Dict[str, np.ndarray]) -> Dict[str, Tensor]:
        cfg = self.cfg
        L: Dict[str, Tensor] = {}
        s = self.states(data["n_soh"])
        L["data"] = ((s["SOH"] - data["soh"]) / cfg.soh_scale).square().mean()

        c = self.states(data["n_col"])
        Tk = data["T_col"] + 273.15
        c_rate = data["I_col"] / self.c_bol
        thr = 2 * c["SOH"]                                     # A_n / C_bol
        Qs, Qp, Ql = (self._col(c["Q"], i) for i in range(3))
        dQs, dQp, dQl = (self._col(c["dQ"], i) for i in range(3))
        res = []
        if self.on["SEI"]:
            rate = self.k_sei() * self._arr(cfg.Ea_sei_J_mol, Tk) * thr / (1 + Qs / self.delta())
            res.append((dQs - rate) / cfg.rate_scale)
        if self.on["plating"]:
            cold = np.exp(cfg.Ea_plating_J_mol / R_GAS * (1.0 / Tk - 1.0 / (cfg.T_ref_C + 273.15)))
            gate = 1.0 / (1.0 + np.exp((Tk - 273.15 - cfg.T_plating_onset_C) / 3.0))
            rate = self.k_pl() * cold * (cfg.I_charge_A / self.c_bol) * (gate + self.kappa() * Ql / 0.05)
            res.append((dQp - rate) / cfg.rate_scale)
        if self.on["LAM"]:
            rate = self.k_lam() * self._arr(cfg.Ea_lam_J_mol, Tk) * (c_rate ** cfg.beta_lam) * thr \
                * (1 + Ql / self.eps())
            res.append((dQl - rate) / cfg.rate_scale)
        phys = res[0].square().mean()
        for r in res[1:]:
            phys = phys + r.square().mean()
        prior = Tensor(0.0)
        for lk, nominal, on in ((self.log_ksei, 4e-4, self.on["SEI"]), (self.log_kpl, 2e-5, self.on["plating"]),
                                (self.log_klam, 1e-4, self.on["LAM"])):
            if on:
                prior = prior + ((lk - math.log(nominal)) / 2.0).square()   # sigma = 2 in log space (x7)
        L["prior"] = prior
        r_int = (c["dR_int"] - self.r_int0 * self.rho() * dQs / 0.1) / (self.r_int0 * cfg.rate_scale)
        r_ct = (c["dR_ct"] - self.r_ct0 * (self.g_lam() * dQl + self.g_sei() * dQs) / 0.1) \
            / (self.r_ct0 * cfg.rate_scale)
        L["phys"] = phys + 0.5 * (r_int.square().mean() + r_ct.square().mean())

        if len(data["n_eis"]):
            e = self.states(data["n_eis"])
            L["eis"] = ((e["R_ct"] - data["rct"]) / self.r_ct0).square().mean() + \
                ((e["R_int"] - data["re"]) / self.r_int0).square().mean()
        if len(data["n_bv"]):
            b = self.states(data["n_bv"])
            I, Tb = data["I_bv"], data["T_bv"] + 273.15
            dv = I * (b["R_int"] + self.R_x()) + self._eta_ct(I, Tb, b["R_ct"])
            L["bv"] = ((dv - data["dv"]) / cfg.dv_scale).square().mean()
        if len(data.get("n_v", [])) and cfg.lambda_volt > 0:
            v = self.states(data["n_v"])
            I, Tv = data["I_v"], data["T_v"] + 273.15
            vm = self.U_bar() - I * (v["R_int"] + self.R_x()) - self._eta_ct(I, Tv, v["R_ct"])
            L["volt"] = ((vm - data["v_mean"]) / cfg.dv_scale).square().mean()
        return L

    def total(self, L: Dict[str, Tensor]) -> Tensor:
        tot = super().total(L) + self.cfg.lambda_prior * L["prior"]
        if "volt" in L:
            tot = tot + self.cfg.lambda_volt * L["volt"]
        return tot

    def monitor(self) -> Dict[str, float]:
        return {"k_SEI": float(self.k_sei().data), "k_plating": float(self.k_pl().data),
                "k_LAM": float(self.k_lam().data)}

    def mechanism_table(self, n: np.ndarray) -> pd.DataFrame:
        s = self.states(n)
        Q = s["Q"].data
        return pd.DataFrame({"n": np.asarray(n), "Q_SEI": Q[:, 0], "Q_plating": Q[:, 1], "Q_LAM": Q[:, 2]})

    def physics_summary(self, n0: int) -> Dict[str, float]:
        s = self.states(np.array([1.0, float(n0)]))
        rct0, rctn = float(s["R_ct"].data[0, 0]), float(s["R_ct"].data[1, 0])
        Tk = self.cfg.T_ref_C + 273.15
        vt = 2 * R_GAS * Tk / FARADAY
        Q = s["Q"].data[1]
        tot = max(float(Q.sum()), 1e-12)
        out = {"U_bar_V": float(self.U_bar().data), "R_x_mOhm": 1e3 * float(self.R_x().data),
               "rho_SEI": float(self.rho().data), "gamma_ct_SEI": float(self.g_sei().data),
               "i0_start_A": R_GAS * Tk / (FARADAY * rct0), "i0_at_n0_A": R_GAS * Tk / (FARADAY * rctn),
               "eta_ct_2A_n0_mV": 1e3 * vt * math.asinh(2.0 * rctn / vt)}
        if self.on["SEI"]:
            out.update({"k_SEI_per_cycle": float(self.k_sei().data), "delta_SEI": float(self.delta().data),
                        "share_SEI_at_n0": float(Q[0]) / tot})
        if self.on["plating"]:
            out.update({"k_plating_per_cycle": float(self.k_pl().data), "kappa_LAM_to_plating": float(self.kappa().data),
                        "share_plating_at_n0": float(Q[1]) / tot})
        if self.on["LAM"]:
            out.update({"k_LAM_per_cycle": float(self.k_lam().data), "eps_LAM_acceleration": float(self.eps().data),
                        "gamma_ct_LAM": float(self.g_lam().data), "share_LAM_at_n0": float(Q[2]) / tot})
        return out


def _inv_softplus(y: float) -> float:
    return float(math.log(math.expm1(y)))


def _logit(p: float) -> float:
    return float(math.log(p / (1 - p)))


def estimate_pooled_arrhenius(ct: pd.DataFrame, min_ambient_C: float = 15.0, window_frac: float = 0.5,
                              T_ref_C: float = 24.0, I_ref_A: float = 2.0, n_boot: int = 300,
                              seed: int = 0, min_temp_span_C: float = 10.0) -> Dict[str, Any]:
    """Cohort-level activation energy from cells at different temperatures.

    Per cell: early fade rate per Ah = slope of (1 - SOH) vs cumulative throughput over the
    first ``window_frac`` of life. Then
        ln(rate) = a - (Ea/R) (1/T - 1/T_ref) + alpha ln(I / I_ref)
    by least squares across cells (alpha dropped if the current does not vary), with a
    cell-level bootstrap CI. Cells below ``min_ambient_C`` are excluded because cold ageing
    (plating) follows a different mechanism than the Arrhenius SEI term.
    ``identifiable`` is False unless the cell temperatures span at least min_temp_span_C."""
    good = ct[~ct["outlier"]]
    meta = cell_meta(ct)
    recs = []
    for cid, d in good.groupby("Cell_ID"):
        if cid not in meta.index or meta.loc[cid, "Ambient_C"] < min_ambient_C:
            continue
        d = d.sort_values("n")
        d = d[d["n"] <= max(5, window_frac * d["n"].max())]
        d = d[np.isfinite(d["cum_Ah"])]
        if len(d) < 5:
            continue
        x = d["cum_Ah"].to_numpy() - d["cum_Ah"].iloc[0]
        y = 1 - d["SOH"].to_numpy()
        if x.max() <= 0:
            continue
        rate = float(np.polyfit(x, y, 1)[0])
        if rate <= 0:
            continue
        recs.append({"Cell_ID": cid, "rate_per_Ah": rate, "T_C": float(meta.loc[cid, "T_mean_C"]),
                     "I_A": float(meta.loc[cid, "I_dis_A"])})
    tab = pd.DataFrame(recs)
    out: Dict[str, Any] = {"table": tab, "Ea_J_mol": None, "ci_lo": None, "ci_hi": None,
                           "alpha_I": None, "n_cells": len(tab), "identifiable": False,
                           "temp_span_C": 0.0}
    if len(tab) < 3:
        return out
    span = float(tab["T_C"].max() - tab["T_C"].min())
    out["temp_span_C"] = span
    use_I = tab["I_A"].std() > 0.1 and len(tab) >= 4

    def fit(t: pd.DataFrame) -> Tuple[float, Optional[float]]:
        invT = -(1 / (t["T_C"] + 273.15) - 1 / (T_ref_C + 273.15)) / R_GAS
        cols = [np.ones(len(t)), invT]
        if use_I:
            cols.append(np.log(t["I_A"] / I_ref_A))
        coef, *_ = np.linalg.lstsq(np.column_stack(cols), np.log(t["rate_per_Ah"]), rcond=None)
        return float(coef[1]), (float(coef[2]) if use_I else None)

    ea, al = fit(tab)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = tab.iloc[rng.integers(0, len(tab), len(tab))]
        if s["T_C"].max() - s["T_C"].min() < 1.0:
            continue
        try:
            boots.append(fit(s)[0])
        except np.linalg.LinAlgError:
            continue
    lo, hi = (np.quantile(boots, [0.05, 0.95]) if len(boots) >= 20 else (np.nan, np.nan))
    out.update({"Ea_J_mol": ea, "ci_lo": float(lo), "ci_hi": float(hi), "alpha_I": al,
                "identifiable": bool(span >= min_temp_span_C and 5e3 < ea < 150e3)})
    return out


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
        ci = _int_key(ct_cell.sort_values("Cycle_Index")[["Cycle_Index", "n"]])
        e = pd.merge_asof(_int_key(e).sort_values("Cycle_Index"), ci, on="Cycle_Index", direction="backward")
        e["n"] = e["n"].fillna(1)
        e = e[e["n"] <= n0]
    bv = obs.dropna(subset=["dV_step_V", "I_dis_A"])
    bv = bv[(bv["dV_step_V"] > 0) & (bv["dV_step_V"] < 1.5)]
    col = lambda d, c: d[c].to_numpy(dtype=float).reshape(-1, 1)
    I_plan = float(obs["I_dis_A"].median()) if len(obs) else 2.0
    I_col = np.where(n_col <= n0, np.interp(n_col, obs["n"], obs["I_dis_A"].fillna(I_plan)), I_plan) \
        if len(obs) > 1 else np.full_like(n_col, I_plan)
    vm = obs.dropna(subset=["V_mean_V", "I_dis_A"]) if "V_mean_V" in obs.columns else obs.iloc[0:0]
    vm = vm[(vm["V_mean_V"] > 2.5) & (vm["V_mean_V"] < 4.3)]
    return {"I_col": I_col.reshape(-1, 1),
            "n_v": vm["n"].to_numpy(dtype=float), "v_mean": col(vm, "V_mean_V") if len(vm) else np.zeros((0, 1)),
            "I_v": col(vm, "I_dis_A") if len(vm) else np.zeros((0, 1)),
            "T_v": col(vm, "T_mean_C") if len(vm) else np.zeros((0, 1)),
            "n_soh": obs["n"].to_numpy(dtype=float), "soh": col(obs, "SOH"),
            "n_col": n_col, "T_col": T_col.reshape(-1, 1),
            "n_eis": e["n"].to_numpy(dtype=float) if len(e) else np.array([]),
            "rct": col(e, "Rct_ohm") if len(e) else np.zeros((0, 1)),
            "re": col(e, "Re_ohm") if len(e) else np.zeros((0, 1)),
            "n_bv": bv["n"].to_numpy(dtype=float), "I_bv": col(bv, "I_dis_A"),
            "T_bv": col(bv, "T_mean_C"), "dv": col(bv, "dV_step_V")}


def train_pinn(ct: pd.DataFrame, imp: Optional[pd.DataFrame], cell_id: str, n0: int,
               cfg: Optional[PINNConfig] = None, eol_ah: float = DEFAULT_EOL_AH,
               progress: ProgressFn = None, ea_value: Optional[float] = None,
               alpha: float = 0.2) -> PINNResult:
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

    cls = MechanisticPINN if cfg.physics == "mechanistic" else HybridPINN
    net = cls(cfg, c_bol, r_int0, r_ct0, float(n_horizon), ea_value=ea_value)
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
                         **{k: float(v.data) for k, v in L.items()}, **net.monitor()})
            _report(progress, ep / cfg.epochs, f"PINN epoch {ep}")

    n_grid = np.arange(1, n_horizon + 1)
    s = net.states(n_grid)
    soh = s["SOH"].data.ravel()
    soh_eol = soh_eol_for(c_bol, eol_ah)
    good = ct_cell[~ct_cell["outlier"]]
    metrics = forecast_metrics(good["n"].to_numpy(), good["SOH"].to_numpy(), n_grid, soh, n0, soh_eol,
                               alpha=alpha)
    return PINNResult(cell_id, n0, n_grid, soh, s["R_int"].data.ravel(), s["R_ct"].data.ravel(),
                      net.physics_summary(n0), pd.DataFrame(hist), metrics, time.time() - t0,
                      ea_mode=cfg.ea_mode, ea_J_mol=float(net.Ea().data),
                      mechanisms=net.mechanism_table(n_grid), physics_kind=cfg.physics)


def train_pinn_ensemble(ct: pd.DataFrame, imp: Optional[pd.DataFrame], cell_id: str, n0: int,
                        cfg: Optional[PINNConfig] = None, seeds: Sequence[int] = (0, 1, 2),
                        eol_ah: float = DEFAULT_EOL_AH, progress: ProgressFn = None,
                        ea_value: Optional[float] = None, level: float = 0.9,
                        alpha: float = 0.2) -> PINNResult:
    """Deep ensemble over initialisation seeds. Returns the median trajectory, a band from
    the member spread (an *epistemic* spread only - with few members it is an envelope,
    not a calibrated interval) and a per-parameter spread table: a coefficient of
    variation above ~0.25 means the data do not pin that parameter down."""
    cfg = cfg or PINNConfig()
    if cfg.init_jitter <= 0:
        cfg = replace(cfg, init_jitter=0.5)
    members: List[PINNResult] = []
    for j, sd in enumerate(seeds):
        sub = (lambda f, m, j=j: _report(progress, (j + f) / len(seeds), f"member {j + 1}/{len(seeds)}: {m}")) \
            if progress else None
        members.append(train_pinn(ct, imp, cell_id, n0, replace(cfg, seed=int(sd)), eol_ah, sub, ea_value, alpha))
    if len(members) == 1:
        return members[0]
    S = np.vstack([m.soh for m in members])
    qa, qb = (1 - level) / 2, 1 - (1 - level) / 2
    soh = np.median(S, axis=0)
    lo, hi = np.quantile(S, qa, axis=0), np.quantile(S, qb, axis=0)
    ref = members[0]
    lo[ref.n_grid <= n0] = soh[ref.n_grid <= n0]
    hi[ref.n_grid <= n0] = soh[ref.n_grid <= n0]
    phys = pd.DataFrame([m.physics for m in members])
    table = pd.DataFrame({"mean": phys.mean(), "std": phys.std(ddof=1)})
    table["cv"] = (table["std"] / table["mean"].abs()).replace([np.inf, -np.inf], np.nan)
    table["status"] = np.where(table["cv"] < 0.25, "constrained", "not constrained")
    if cfg.ea_mode != "learned" and "Ea_kJ_mol" in table.index:
        table.loc["Ea_kJ_mol", "status"] = "fixed" if cfg.ea_mode == "fixed" else "pooled (cohort)"
        table.loc["Ea_kJ_mol", "cv"] = np.nan
    ct_cell = ct[(ct["Cell_ID"] == cell_id) & ~ct["outlier"]]
    soh_eol = soh_eol_for(float(ct_cell["C_bol_Ah"].iloc[0]), eol_ah)
    metrics = forecast_metrics(ct_cell["n"].to_numpy(), ct_cell["SOH"].to_numpy(), ref.n_grid, soh, n0,
                               soh_eol, lo, hi, alpha)
    return PINNResult(cell_id, n0, ref.n_grid, soh, np.median([m.r_int for m in members], axis=0),
                      np.median([m.r_ct for m in members], axis=0), phys.median().to_dict(), ref.history,
                      metrics, float(sum(m.train_seconds for m in members)), lo, hi, table,
                      len(members), cfg.ea_mode, ref.ea_J_mol,
                      mechanisms=(pd.concat([m.mechanisms for m in members]).groupby("n", as_index=False).median()
                                  if ref.mechanisms is not None else None),
                      physics_kind=cfg.physics)


# =============================================================================
# 9. PARADIGM COMPARISON
# =============================================================================
TWIN_NAMES = {"dual": "ECM twin · dual EKF", "joint": "ECM twin · joint EKF"}
PINN_NAME = "Hybrid PINN"


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
    bands: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)
    rul_samples: Dict[str, np.ndarray] = field(default_factory=dict)
    observer: str = "dual"
    band_level: float = 0.9
    ea: Optional[Dict[str, Any]] = None
    prognostics: Dict[str, Any] = field(default_factory=dict)     # semi-empirical / particle filter


class _Skip(Exception):
    """Internal: a paradigm the caller switched off."""


def resolve_ea(ct: pd.DataFrame, cfg: PINNConfig) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """Activation energy for the PINN according to cfg.ea_mode (pooled falls back to the
    fixed literature value when the cohort does not identify Ea)."""
    if cfg.ea_mode == "fixed":
        return cfg.ea_fixed_J_mol, None
    if cfg.ea_mode == "pooled":
        est = estimate_pooled_arrhenius(ct, T_ref_C=cfg.T_ref_C)
        if est["identifiable"] and est["Ea_J_mol"] is not None:
            return float(est["Ea_J_mol"]), est
        return cfg.ea_fixed_J_mol, est
    return None, None


def compare_paradigms(cell_df: pd.DataFrame, ct: pd.DataFrame, imp: Optional[pd.DataFrame],
                      cell_id: str, n0_frac: float, twin_params: TwinParameters,
                      pinn_cfg: PINNConfig, ml_model: str = "Gradient Boosting",
                      eol_ah: float = DEFAULT_EOL_AH, ekf: Optional[EKFResult] = None,
                      progress: ProgressFn = None, *, observer: str = "dual",
                      dual_cfg: Optional[DualTwinConfig] = None, ml_strategy: str = "increment",
                      conformal_cells: int = 4, band_level: float = 0.9,
                      pinn_seeds: Sequence[int] = (0,), alpha: float = 0.2,
                      extra: Sequence[str] = ("semi", "pf"), run_pinn: bool = True) -> ComparisonResult:
    """Common protocol: everything up to discharge cycle n0 may be used, cycles > n0 are
    held out. (1) ML: population + early target data, conformal band.
    (2) ECM twin: causal observer estimate at n0, then physics forecast (MC band for dual).
    (3) Hybrid PINN: early target data + physics residuals over the full horizon; ensemble
    band when several seeds are given. A failure in one paradigm is reported, not raised."""
    if observer not in TWIN_NAMES:
        raise ValueError(f"observer must be one of {tuple(TWIN_NAMES)}")
    ct_cell = ct[ct["Cell_ID"] == cell_id].sort_values("n")
    good = ct_cell[~ct_cell["outlier"]]
    if len(good) < 10:
        raise DataError(f"{cell_id}: too few valid discharge cycles for a comparison.")
    n0 = int(max(5, round(n0_frac * good["n"].max())))
    c_bol = float(ct_cell["C_bol_Ah"].iloc[0])
    soh_eol = soh_eol_for(c_bol, eol_ah)
    n_max = int(good["n"].max() * 1.5)
    preds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    bands: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    mets: Dict[str, ForecastMetrics] = {}
    ruls: Dict[str, np.ndarray] = {}
    errors: Dict[str, str] = {}
    meta = cell_meta(ct)
    n_obs, y_obs = good["n"].to_numpy(), good["SOH"].to_numpy()

    ml_res = None
    try:
        _report(progress, 0.02, f"ML surrogate ({ml_model})")
        ml_res = train_ml_forecast(ct, cell_id, n0, ml_model, eol_ah=eol_ah, strategy=ml_strategy,
                                   conformal_cells=conformal_cells, band_level=band_level, alpha=alpha)
        name = f"ML · {ml_model}"
        preds[name] = (ml_res.n_grid, ml_res.soh_pred)
        mets[name] = ml_res.metrics
        if ml_res.soh_lo is not None:
            bands[name] = (ml_res.n_grid, ml_res.soh_lo, ml_res.soh_hi)
    except Exception as exc:
        errors["ML"] = str(exc)

    tname = TWIN_NAMES[observer]
    try:
        if ekf is None or ekf.kind != observer:
            sub = lambda f, m: _report(progress, 0.15 + 0.4 * f, m)
            calib = similar_cells(meta, cell_id)
            if observer == "dual":
                ekf = run_dual_twin(cell_df, ct, imp, cell_id, twin_params, dual_cfg, calib, progress=sub)
            else:
                ekf = run_ekf(cell_df, ct, imp, cell_id, twin_params, calib, progress=sub)
        fc = twin_forecast(ekf, n0, n_max, soh_eol, band_level)
        preds[tname] = (fc.n_grid, fc.soh)
        mets[tname] = forecast_metrics(n_obs, y_obs, fc.n_grid, fc.soh, n0, soh_eol, fc.lo, fc.hi, alpha)
        if fc.lo is not None:
            bands[tname] = (fc.n_grid, fc.lo, fc.hi)
        if fc.rul_samples is not None:
            ruls[tname] = fc.rul_samples
    except Exception as exc:
        errors[tname] = str(exc)

    progs: Dict[str, Any] = {}
    for key, fn in (("semi", semi_empirical_forecast), ("pf", particle_filter_forecast)):
        if key not in extra:
            continue
        try:
            _report(progress, 0.55, "semi-empirical law" if key == "semi" else "particle filter")
            f = fn(ct, cell_id, n0, eol_ah, band_level, alpha=alpha)
            preds[f.name] = (f.n_grid, f.soh)
            mets[f.name] = f.metrics
            bands[f.name] = (f.n_grid, f.lo, f.hi)
            ruls[f.name] = f.rul_samples
            progs[f.name] = f
        except Exception as exc:
            errors[SEMI_NAME if key == "semi" else PF_NAME] = str(exc)

    pinn_res = None
    ea_info = None
    try:
        if not run_pinn:
            raise _Skip()
        cfg = replace(pinn_cfg, horizon_factor=1.5)
        ea_value, ea_info = resolve_ea(ct, cfg)
        sub = lambda f, m: _report(progress, 0.55 + 0.45 * f, m)
        if len(pinn_seeds) > 1:
            pinn_res = train_pinn_ensemble(ct, imp, cell_id, n0, cfg, pinn_seeds, eol_ah, sub, ea_value,
                                           band_level, alpha)
        else:
            pinn_res = train_pinn(ct, imp, cell_id, n0, replace(cfg, seed=int(pinn_seeds[0])), eol_ah, sub,
                                  ea_value, alpha)
        preds[PINN_NAME] = (pinn_res.n_grid, pinn_res.soh)
        mets[PINN_NAME] = pinn_res.metrics
        if pinn_res.soh_lo is not None:
            bands[PINN_NAME] = (pinn_res.n_grid, pinn_res.soh_lo, pinn_res.soh_hi)
    except _Skip:
        pass
    except Exception as exc:
        errors[PINN_NAME] = str(exc)

    _report(progress, 1.0, "comparison done")
    return ComparisonResult(cell_id, n0, soh_eol, good[["n", "Cycle_Index", "SOH", "Capacity_Ah"]],
                            valid_eis(imp, cell_id), preds, mets, ekf, pinn_res, ml_res, errors,
                            bands, ruls, observer, band_level, ea_info, progs)


# =============================================================================
# 10. CROSS-CELL BENCHMARK
# =============================================================================
BENCH_PARADIGMS = ("ML", "Twin", "PINN", "SemiEmp", "PF")


@dataclass
class BenchmarkConfig:
    fracs: Tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6)
    paradigms: Tuple[str, ...] = BENCH_PARADIGMS
    ml_model: str = "Gradient Boosting"
    ml_strategy: str = "increment"
    conformal_cells: int = 4
    band_level: float = 0.9
    alpha: float = 0.2
    eol_ah: float = DEFAULT_EOL_AH
    pinn_epochs: int = 1000
    pinn_seeds: Tuple[int, ...] = (0,)
    ea_mode: str = "fixed"
    use_capacity: bool = False
    min_cycles: int = 30

    def __post_init__(self) -> None:
        if not self.fracs or not all(0.05 <= f <= 0.95 for f in self.fracs):
            raise ValueError("fracs must lie in [0.05, 0.95]")
        unknown = set(self.paradigms) - set(BENCH_PARADIGMS)
        if unknown:
            raise ValueError(f"Unknown paradigms: {unknown}")
        if self.ea_mode not in EA_MODES:
            raise ValueError(f"ea_mode must be one of {EA_MODES}")


def run_benchmark(store: ParquetStore, ct: pd.DataFrame, imp: Optional[pd.DataFrame],
                  cfg: Optional[BenchmarkConfig] = None, cells: Optional[Sequence[str]] = None,
                  twin_params: Optional[TwinParameters] = None, dual_cfg: Optional[DualTwinConfig] = None,
                  progress: ProgressFn = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast-origin sweep x cells x paradigms. The ML surrogate is always trained with the
    target cell's future withheld (leave-one-cell-out on the held-out horizon); the dual twin
    is replayed once per cell (it is causal) and forecast from every origin.

    Returns (forecasts, residuals): one row per (cell, origin, paradigm) with RMSE, RUL,
    censoring, RA, alpha-lambda and band coverage; and one row per held-out cycle with the
    horizon, signed error and band hit (for coverage-vs-horizon calibration)."""
    cfg = cfg or BenchmarkConfig()
    tp = twin_params or TwinParameters()
    dc = replace(dual_cfg or DualTwinConfig(), use_capacity=cfg.use_capacity)
    meta = cell_meta(ct)
    pool = [c for c in meta.index if int(meta.loc[c, "cycles"]) >= cfg.min_cycles]
    cells = [c for c in (cells or pool) if c in pool]
    pcfg = PINNConfig(epochs=cfg.pinn_epochs, ea_mode=cfg.ea_mode)
    ea_value, _ = resolve_ea(ct, pcfg) if "PINN" in cfg.paradigms else (None, None)
    rows, resid = [], []
    total = max(len(cells) * len(cfg.fracs), 1)
    step = 0
    for cell in cells:
        good = ct[(ct["Cell_ID"] == cell) & ~ct["outlier"]].sort_values("n")
        n_obs, y_obs = good["n"].to_numpy(), good["SOH"].to_numpy()
        soh_eol = soh_eol_for(float(good["C_bol_Ah"].iloc[0]), cfg.eol_ah)
        n_max = int(good["n"].max() * 1.5)
        twin = None
        if "Twin" in cfg.paradigms:
            try:
                twin = run_dual_twin(store.cell_frame(cell), ct, imp, cell, tp, dc, similar_cells(meta, cell))
            except Exception:
                twin = None
        for frac in cfg.fracs:
            n0 = int(max(5, round(frac * good["n"].max())))
            _report(progress, step / total, f"{cell} · n0 = {n0}")
            step += 1
            cands: List[Tuple[str, Callable[[], Tuple[np.ndarray, np.ndarray, Any, Any]]]] = []
            if "ML" in cfg.paradigms:
                def _ml():
                    r = train_ml_forecast(ct, cell, n0, cfg.ml_model, eol_ah=cfg.eol_ah, strategy=cfg.ml_strategy,
                                          conformal_cells=cfg.conformal_cells, band_level=cfg.band_level)
                    return r.n_grid, r.soh_pred, r.soh_lo, r.soh_hi
                cands.append((f"ML · {cfg.ml_model}", _ml))
            if twin is not None:
                def _tw():
                    f = twin_forecast(twin, n0, n_max, soh_eol, cfg.band_level)
                    return f.n_grid, f.soh, f.lo, f.hi
                cands.append((TWIN_NAMES["dual"], _tw))
            if "PINN" in cfg.paradigms:
                def _pn():
                    if len(cfg.pinn_seeds) > 1:
                        r = train_pinn_ensemble(ct, imp, cell, n0, pcfg, cfg.pinn_seeds, cfg.eol_ah,
                                                ea_value=ea_value, level=cfg.band_level)
                    else:
                        r = train_pinn(ct, imp, cell, n0, replace(pcfg, seed=cfg.pinn_seeds[0]), cfg.eol_ah,
                                       ea_value=ea_value)
                    return r.n_grid, r.soh, r.soh_lo, r.soh_hi
                cands.append((PINN_NAME, _pn))
            if "SemiEmp" in cfg.paradigms:
                def _se():
                    f = semi_empirical_forecast(ct, cell, n0, cfg.eol_ah, cfg.band_level)
                    return f.n_grid, f.soh, f.lo, f.hi
                cands.append((SEMI_NAME, _se))
            if "PF" in cfg.paradigms:
                def _pf():
                    f = particle_filter_forecast(ct, cell, n0, cfg.eol_ah, cfg.band_level, n_particles=2000)
                    return f.n_grid, f.soh, f.lo, f.hi
                cands.append((PF_NAME, _pf))
            for name, fn in cands:
                t0 = time.time()
                try:
                    n_g, p_g, lo, hi = fn()
                except Exception as exc:
                    rows.append({"cell": cell, "frac": frac, "n0": n0, "paradigm": name, "error": str(exc)})
                    continue
                m = forecast_metrics(n_obs, y_obs, n_g, p_g, n0, soh_eol, lo, hi, cfg.alpha)
                rows.append({"cell": cell, "frac": frac, "n0": n0, "paradigm": name,
                             "rmse": m.rmse, "mae": m.mae, "r2": m.r2, "rul_true": m.rul_true,
                             "rul_pred": m.rul_pred, "rul_error": m.rul_error, "censored": m.censored,
                             "rul_true_lb": m.rul_true_lb, "rel_accuracy": m.rel_accuracy,
                             "alpha_lambda_ok": m.alpha_lambda_ok, "coverage": m.coverage,
                             "band_width": m.band_width,
                             "eol_true": None if m.rul_true is None else n0 + m.rul_true,
                             "ambient_C": float(meta.loc[cell, "Ambient_C"]),
                             "runtime_s": time.time() - t0, "error": None})
                hr = horizon_residuals(n_obs, y_obs, n_g, p_g, n0, lo, hi)
                hr.insert(0, "paradigm", name)
                hr.insert(0, "frac", frac)
                hr.insert(0, "cell", cell)
                resid.append(hr)
    _report(progress, 1.0, "benchmark done")
    bench = pd.DataFrame(rows)
    residuals = pd.concat(resid, ignore_index=True) if resid else pd.DataFrame(
        columns=["cell", "frac", "paradigm", "h", "err", "in_band"])
    return bench, residuals


def benchmark_summary(bench: pd.DataFrame, alpha: float = 0.2) -> pd.DataFrame:
    """Per-paradigm aggregate across cells and forecast origins."""
    ok = bench[bench["error"].isna()] if "error" in bench.columns else bench
    if ok.empty:
        return pd.DataFrame()
    ph = prognostic_horizon(ok, alpha)
    rows = []
    for par, d in ok.groupby("paradigm"):
        unc = d[~d["censored"].astype(bool)]
        al = unc["alpha_lambda_ok"].dropna().astype(bool)
        rows.append({"Paradigm": par, "Cells": d["cell"].nunique(), "Forecasts": len(d),
                     "Median RMSE": float(d["rmse"].median()), "Mean RMSE": float(d["rmse"].mean()),
                     "Mean RA": float(unc["rel_accuracy"].dropna().mean()) if len(unc) else np.nan,
                     "α-λ hit rate": float(al.mean()) if len(al) else np.nan,
                     "Mean coverage": float(d["coverage"].dropna().mean()) if d["coverage"].notna().any() else np.nan,
                     "Mean band width": float(d["band_width"].dropna().mean()) if d["band_width"].notna().any() else np.nan,
                     "Censored forecasts": int(d["censored"].astype(bool).sum()),
                     "Mean PH (cycles)": float(ph.loc[ph["paradigm"] == par, "PH_cycles"].mean())})
    return pd.DataFrame(rows).set_index("Paradigm")


def coverage_by_horizon(residuals: pd.DataFrame, bin_cycles: int = 20) -> pd.DataFrame:
    """Empirical band coverage and RMSE per horizon bin and paradigm (calibration check)."""
    r = residuals.dropna(subset=["err"]).copy()
    if r.empty:
        return pd.DataFrame(columns=["paradigm", "h_bin", "coverage", "rmse", "count"])
    r["h_bin"] = (np.ceil(r["h"] / bin_cycles) * bin_cycles).astype(int)
    r["in_band_f"] = pd.to_numeric(r["in_band"], errors="coerce")
    g = r.groupby(["paradigm", "h_bin"])
    out = pd.DataFrame({"coverage": g["in_band_f"].mean(),
                        "rmse": g["err"].apply(lambda e: float(np.sqrt(np.mean(np.square(e))))),
                        "count": g.size()}).reset_index()
    return out


# =============================================================================
# 11. TWIN-AWARE OPERATIONAL OPTIMISATION
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
    # "cycle" (default): per-cycle margin price*Ah - w*cost*dSOH. "rate": Dinkelbach form
    # margin - rho*hours (see tune_rate_reference). Finding (v4 tests): no one-step objective
    # optimises the lifetime profit rate, because a greedy policy cannot see that faster
    # cycling shrinks future capacity and revenue; "cycle" scored best. Optimising the
    # lifetime KPI properly requires dynamic programming over SOH.
    objective: str = "cycle"
    rate_ref: Optional[float] = None  # Dinkelbach parameter rho (CU/h), see tune_rate_reference
    energy_price_per_Wh: float = 0.02  # charging energy cost (CU/Wh): J_op = revenue - energy cost

    def __post_init__(self) -> None:
        if self.objective not in ("rate", "cycle"):
            raise ValueError("Economics.objective must be 'rate' or 'cycle'")
        self.currents_A = tuple(float(c) for c in self.currents_A)
        self.price_per_Ah = tuple(float(c) for c in self.price_per_Ah)
        if len(self.currents_A) != len(self.price_per_Ah) or not self.currents_A:
            raise ValueError("Economics: one price per current is required")
        if min(self.currents_A) <= 0 or min(self.price_per_Ah) < 0:
            raise ValueError("Economics: currents must be positive and prices non-negative")
        if not 0 < self.soh_eol < 1:
            raise ValueError("Economics: soh_eol must be in (0, 1)")
        if self.replacement_cost < 0 or self.degradation_weight < 0 or self.energy_price_per_Wh < 0:
            raise ValueError("Economics: costs, prices and weights must be non-negative")

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
    ocv_offset_V: float = 0.0

    def __post_init__(self) -> None:
        pos = ("C_bol_Ah", "R_int0", "R_ct0", "tau_rc_s", "I_charge_A", "k_ah", "Ea_J_mol", "I_ref_A",
               "C_th_J_K", "R_th_K_W", "dt_s")
        bad = [k for k in pos if not (_finite(getattr(self, k)) and getattr(self, k) > 0)]
        if bad:
            raise ValueError(f"CellPhysics must be positive: {bad}")


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
    ah, dsoh, t, e_out = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    active = np.ones(n, dtype=bool)
    for _ in range(int(20 * 3600 / p.dt_s)):
        ri = r_int * _arrhenius(p.Ea_Rint_J_mol, T, p.T_ref_C)
        rc = r_ct * _arrhenius(p.Ea_Rct_J_mol, T, p.T_ref_C)
        vrc_new = a * vrc + (1 - a) * rc * I
        v = generic_ocv(soc) + p.ocv_offset_V - I * ri - vrc_new
        active &= (v > p.V_cut) & (T < p.T_abort_C)
        if not active.any():
            break
        step_ah = np.where(active, I * p.dt_s / 3600, 0.0)
        e_out += step_ah * v
        dsoh += p.k_ah * degradation_stress(T, I, p) * step_ah
        ah += step_ah
        soc -= step_ah / Q
        t += np.where(active, p.dt_s, 0.0)
        T = np.where(active, T + p.dt_s / p.C_th_J_K * (I ** 2 * (ri + rc) - (T - T_amb) / p.R_th_K_W), T)
        T_peak = np.maximum(T_peak, T)
        vrc = np.where(active, vrc_new, vrc)
    return ah, dsoh, t / 3600, T_peak, e_out


def predict_cycle(theta: np.ndarray, currents: Sequence[float], T_amb: float, p: CellPhysics) -> Dict[str, np.ndarray]:
    ah, dsoh_dis, t_dis, T_peak, e_out = simulate_discharge(theta, currents, T_amb, p)
    soh, r_int, r_ct = theta
    r_tot = r_int * _arrhenius(p.Ea_Rint_J_mol, T_amb, p.T_ref_C) + r_ct * _arrhenius(p.Ea_Rct_J_mol, T_amb, p.T_ref_C)
    T_ch = T_amb + p.I_charge_A ** 2 * r_tot * p.R_th_K_W
    dsoh_ch = p.k_ah * degradation_stress(T_ch, p.I_charge_A, p) * ah
    # charging energy: OCV averaged over the SOC window replaced, plus the resistive overpotential
    Q = p.C_bol_Ah * max(float(soh), 1e-3)
    s_lo = np.clip(1.0 - ah / Q, 0.0, 1.0)
    grid = np.linspace(0.0, 1.0, 21)[None, :]
    ocv_mean = np.mean(generic_ocv(s_lo[:, None] + (1.0 - s_lo[:, None]) * grid), axis=1) + p.ocv_offset_V
    e_in = ah * (ocv_mean + p.I_charge_A * r_tot)
    return {"ah": ah, "dsoh": dsoh_dis + dsoh_ch, "hours": t_dis + ah / p.I_charge_A + p.rest_h, "T_peak": T_peak,
            "e_out": e_out, "e_in": e_in}


def age_theta(theta: np.ndarray, dsoh: float, T_amb: float, p: CellPhysics) -> np.ndarray:
    soh, r_int, r_ct = theta
    cold_R = 1 + p.k_cold_R * max(0.0, p.T_cold_C - T_amb)
    return np.array([soh - dsoh, r_int + p.beta_int * p.R_int0 * dsoh * cold_R,
                     r_ct + p.beta_ct * p.R_ct0 * dsoh * cold_R])


def optimal_policy(theta_hat: np.ndarray, T_amb: float, p: CellPhysics, e: Economics) -> float:
    """One-step model-predictive choice subject to safety. objective="rate" maximises
    (price*Ah - weight*cost*dSOH) / cycle hours - the same profit rate the policy is scored
    on; "cycle" maximises the per-cycle margin (legacy, favours slow low-stress cycling).
    ``p`` is the *model* the policy believes in, which may differ from the plant."""
    I = np.array(e.currents_A)
    pred = predict_cycle(theta_hat, I, T_amb, p)
    profit = np.array(e.price_per_Ah) * pred["ah"] - e.energy_price_per_Wh * pred["e_in"] \
        - e.decision_cost_per_soh * pred["dsoh"]
    if e.objective == "rate":
        # Long-run rate sum(margin)/sum(hours) is maximised by maximising margin - rho*hours
        # with rho the optimal rate (Dinkelbach); greedily maximising each cycle's own ratio is
        # not optimal for a ratio of sums. Without a tuned rho, fall back to the per-cycle ratio.
        if e.rate_ref is not None:
            profit = profit - e.rate_ref * pred["hours"]
        else:
            profit = profit / np.maximum(pred["hours"], 1e-6)
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


def tune_rate_reference(p: CellPhysics, e: Economics, iterations: int = 4, tol: float = 1e-3,
                        ambient_mean_C: float = 20.0, ambient_amp_C: float = 16.0,
                        seed: int = 0) -> Tuple[Economics, List[float]]:
    """Dinkelbach fixed-point iteration for the long-run profit rate rho*: simulate with
    objective margin - rho*hours, set rho to the achieved profit per hour, repeat. Tuned on
    the policy's *model* (never the plant). Returns the tuned Economics and the rho path."""
    if e.objective != "rate":
        return e, []
    rho = 0.0
    path = [rho]
    for _ in range(iterations):
        e_i = replace(e, rate_ref=rho)
        s = summarise_life(simulate_life(optimal_policy, p, e_i, seed=seed,
                                         ambient_mean_C=ambient_mean_C, ambient_amp_C=ambient_amp_C))
        new = float(s["profit_per_h"])
        path.append(new)
        if not math.isfinite(new) or abs(new - rho) < tol:
            rho = new if math.isfinite(new) else rho
            break
        rho = new
    return replace(e, rate_ref=rho), path


def ambient_profile(k: int, mean_C: float = 20.0, amp_C: float = 16.0, period_cycles: float = 80.0) -> float:
    return mean_C + amp_C * math.sin(2 * math.pi * k / period_cycles)


def perturb_physics(p: CellPhysics, level: float, rng: np.random.Generator) -> CellPhysics:
    """A plant that differs from the model: log-normal multiplicative perturbations of
    relative size ``level`` on ageing, resistance and thermal parameters, a relative
    perturbation of Ea, and an OCV offset of ~50 mV x level. level = 0 returns p."""
    if level <= 0:
        return p
    q = asdict(p)
    for k in ("k_ah", "beta_int", "beta_ct", "R_int0", "R_ct0", "C_th_J_K", "R_th_K_W", "alpha_c", "k_cold"):
        q[k] = q[k] * float(np.exp(level * rng.standard_normal()))
    q["Ea_J_mol"] = q["Ea_J_mol"] * float(np.clip(1 + level * rng.standard_normal(), 0.3, 3.0))
    q["ocv_offset_V"] = q["ocv_offset_V"] + 0.05 * level * float(rng.standard_normal())
    return CellPhysics(**q)


def simulate_life(policy: Callable, p: CellPhysics, e: Economics, max_cycles: int = 2000,
                  sigma_soh: float = 0.005, sigma_r_frac: float = 0.03, seed: int = 0,
                  ambient_mean_C: float = 20.0, ambient_amp_C: float = 16.0,
                  plant: Optional[CellPhysics] = None) -> pd.DataFrame:
    """Closed-loop lifecycle. The policy plans with the model ``p``; the true cell evolves
    with ``plant`` (defaults to ``p``, i.e. a perfect model). The policy observes the plant
    state through estimation noise (the observer)."""
    plant = plant or p
    rng = np.random.default_rng(seed)
    theta = np.array([1.0, plant.R_int0, plant.R_ct0])
    rows = []
    for k in range(max_cycles):
        if theta[0] < e.soh_eol:
            break
        T_amb = ambient_profile(k, ambient_mean_C, ambient_amp_C)
        theta_hat = theta * np.array([1.0, 1 + sigma_r_frac * rng.standard_normal(),
                                      1 + sigma_r_frac * rng.standard_normal()])
        theta_hat[0] += sigma_soh * rng.standard_normal()
        I_sel = policy(theta_hat, T_amb, p, e)
        out = {key: float(v[0]) for key, v in predict_cycle(theta, [I_sel], T_amb, plant).items()}
        cold_breach = (e.cold_derate_below_C is not None and T_amb < e.cold_derate_below_C
                       and I_sel > e.cold_max_current_A)
        rows.append(dict(cycle=k, T_amb=T_amb, I=I_sel, SOH=theta[0], SOH_hat=theta_hat[0],
                         R_ct_hat=theta_hat[2], ah=out["ah"], hours=out["hours"], T_peak=out["T_peak"],
                         revenue=e.price(I_sel) * out["ah"], e_in=out["e_in"], e_out=out["e_out"],
                         energy_cost=e.energy_price_per_Wh * out["e_in"], dsoh=out["dsoh"],
                         degr_cost=e.cost_per_soh * out["dsoh"],
                         profit=e.price(I_sel) * out["ah"] - e.energy_price_per_Wh * out["e_in"]
                         - e.cost_per_soh * out["dsoh"],
                         violation=bool(out["T_peak"] > e.T_max_C or cold_breach)))
        theta = age_theta(theta, out["dsoh"], T_amb, plant)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["cum_profit"] = df["profit"].cumsum()
        df["cum_hours"] = df["hours"].cumsum()
    return df


def summarise_life(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"cycles": 0, "Ah": 0.0, "profit": 0.0, "hours": 0.0, "profit_per_h": float("nan"),
                "violations": 0, "mean_I": float("nan"), "revenue": 0.0, "energy_cost": 0.0, "J_op": 0.0,
                "degradation_cost": 0.0, "energy_eff": float("nan")}
    profit, hours = float(df["profit"].sum()), float(df["hours"].sum())
    rev = float(df["revenue"].sum())
    ecost = float(df["energy_cost"].sum()) if "energy_cost" in df.columns else 0.0
    return {"cycles": len(df), "Ah": float(df["ah"].sum()), "profit": profit, "hours": hours,
            "profit_per_h": profit / hours if hours else float("nan"),
            "violations": int(df["violation"].sum()), "mean_I": float(df["I"].mean()),
            "revenue": rev, "energy_cost": ecost, "J_op": rev - ecost,
            "degradation_cost": float(df["degr_cost"].sum()) if "degr_cost" in df.columns else float("nan"),
            "energy_eff": float(df["e_out"].sum() / df["e_in"].sum()) if "e_in" in df.columns and df["e_in"].sum() > 0
            else float("nan")}


def mismatch_study(e: Economics, p_model: CellPhysics, levels: Sequence[float] = (0.0, 0.1, 0.2, 0.35, 0.5),
                   n_draws: int = 4, seed: int = 0, ambient_mean_C: float = 20.0, ambient_amp_C: float = 16.0,
                   policies: Sequence[str] = ("Twin-Aware", "Fixed 1 A", "Fixed 2 A", "Fixed 4 A"),
                   progress: ProgressFn = None, fast_dt_s: Optional[float] = 60.0) -> pd.DataFrame:
    """How much of the twin-aware advantage survives model error? For each mismatch level
    and random draw a plant is sampled with perturb_physics; every policy runs against the
    same plant while the twin-aware policy keeps planning with the nominal model.
    ``fast_dt_s`` coarsens the discharge integration step for this study only."""
    if fast_dt_s:
        p_model = replace(p_model, dt_s=float(fast_dt_s))
    rows = []
    total = max(len(levels) * n_draws, 1)
    j = 0
    for lev in levels:
        for d in range(n_draws if lev > 0 else 1):
            rng = np.random.default_rng(seed + 1000 * d + int(1e4 * lev))
            plant = perturb_physics(p_model, float(lev), rng)
            _report(progress, j / total, f"mismatch {lev:.2f} · draw {d + 1}")
            j += 1 if lev > 0 else n_draws
            for pol in policies:
                s = summarise_life(simulate_life(make_policy(pol), p_model, e, seed=seed + d,
                                                 ambient_mean_C=ambient_mean_C, ambient_amp_C=ambient_amp_C,
                                                 plant=plant))
                rows.append({"level": float(lev), "draw": d, "policy": pol, **s})
    _report(progress, 1.0, "mismatch study done")
    df = pd.DataFrame(rows)
    # Baselines that breach the thermal / cold-derating limits are not admissible competitors:
    # the advantage is measured against the best *compliant* fixed policy (and, for reference,
    # against the best fixed policy regardless of violations).
    fx = df[df["policy"] != "Twin-Aware"]
    best_ok = fx[fx["violations"] == 0].groupby(["level", "draw"])["profit_per_h"].max().rename("best_compliant")
    best_any = fx.groupby(["level", "draw"])["profit_per_h"].max().rename("best_any")
    twin = df[df["policy"] == "Twin-Aware"].set_index(["level", "draw"])[["profit_per_h", "violations"]]
    twin = twin.rename(columns={"profit_per_h": "twin", "violations": "twin_violations"})
    adv = pd.concat([twin, best_ok, best_any], axis=1).reset_index()
    adv["advantage_pct"] = 100 * (adv["twin"] - adv["best_compliant"]) / adv["best_compliant"].abs()
    adv["advantage_vs_any_pct"] = 100 * (adv["twin"] - adv["best_any"]) / adv["best_any"].abs()
    return df.merge(adv.drop(columns=["twin"]), on=["level", "draw"], how="left")


# =============================================================================
# 12. SYNTHETIC GROUND TRUTH & RUN MANIFESTS
# =============================================================================
def make_synthetic_master(n_cells: int = 4, n_cycles: int = 80,
                          ambients: Sequence[float] = (24.0, 34.0, 43.0, 24.0),
                          currents: Sequence[float] = (2.0, 2.0, 2.0, 2.0),
                          k_true: Optional[Sequence[float]] = None, seed: int = 0,
                          dt_s: float = 20.0, noise_v: float = 0.003, c_bol: float = 2.0,
                          eis_every: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Synthetic telemetry in the master schema from the twin's own physics: generic OCV,
    1-RC ECM, throughput-driven Arrhenius fade with coupled resistance growth. Returns
    (master, impedance, truth) where truth holds each cell's k_ah and SOH trajectory.
    Used by the tests (recovery of known parameters) and as an offline demo dataset."""
    p = TwinParameters()
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    imp_rows: List[Dict[str, Any]] = []
    truth: Dict[str, Dict[str, Any]] = {}
    for c in range(n_cells):
        cid = f"S{c + 1:03d}"
        T_amb = float(ambients[c % len(ambients)])
        I_dis = float(currents[c % len(currents)])
        k = float(k_true[c]) if k_true is not None else 4e-4 * (1.0 + 0.25 * (c % 3))
        r_int0 = 0.045 * (1 + 0.05 * rng.standard_normal())
        r_ct0 = 0.070 * (1 + 0.05 * rng.standard_normal())
        soh, r_int, r_ct = 1.0, r_int0, r_ct0
        ci = 0
        traj = []
        for n in range(n_cycles):
            Q = c_bol * soh
            # ---- charge: CC 1.5 A to 4.2 V, then CV until the current decays (coarse sampling) ----
            ci += 1
            r_tot = r_int + r_ct
            t, s_ch = 0.0, 0.0
            while True:
                v_ch = float(generic_ocv(s_ch) + 1.5 * r_tot)
                rows.append({"Cell_ID": cid, "Cycle_Index": ci, "Cycle_Type": "charge", "Time_s": t,
                             "Voltage_V": min(v_ch, 4.2), "Current_A": 1.5, "Temp_C": T_amb + 1.0,
                             "Capacity_Ah": np.nan, "Ambient_C": T_amb})
                if v_ch >= 4.2 or s_ch >= 1.0:
                    break
                t += 120.0
                s_ch = min(1.0, s_ch + 1.5 * 120.0 / 3600.0 / Q)
            tau_cv = 900.0 * r_tot / 0.115                       # CV tail lengthens with resistance
            for tc in np.arange(120.0, 5 * tau_cv, 120.0):
                rows.append({"Cell_ID": cid, "Cycle_Index": ci, "Cycle_Type": "charge", "Time_s": t + tc,
                             "Voltage_V": 4.2, "Current_A": float(1.5 * math.exp(-tc / tau_cv)),
                             "Temp_C": T_amb + 0.5, "Capacity_Ah": np.nan, "Ambient_C": T_amb})
            # ---- discharge: rest sample then CC load to 2.7 V ----
            ci += 1
            dis: List[Tuple[float, float, float, float]] = [(0.0, float(generic_ocv(1.0)), 0.0, T_amb)]
            soc, vrc, t, ah = 1.0, 0.0, 0.0, 0.0
            a = math.exp(-dt_s / p.tau_rc_s)
            temps = []
            while True:
                t += dt_s
                ah += I_dis * dt_s / 3600
                soc = 1 - ah / Q
                vrc = a * vrc + (1 - a) * r_ct * (-I_dis)
                T = T_amb + 0.5 + 3.0 * (1 - soc)
                v = float(generic_ocv(soc)) - I_dis * r_int + vrc + noise_v * rng.standard_normal()
                dis.append((t, v, -I_dis, T))
                temps.append(T)
                if v < 2.7 or t > 6 * 3600:
                    break
            cap = ah
            for (tt, vv, ii, TT) in dis:
                rows.append({"Cell_ID": cid, "Cycle_Index": ci, "Cycle_Type": "discharge", "Time_s": tt,
                             "Voltage_V": vv, "Current_A": ii, "Temp_C": TT, "Capacity_Ah": cap,
                             "Ambient_C": T_amb})
            traj.append({"n": n + 1, "SOH_true": soh, "capacity": cap, "R_int": r_int, "R_ct": r_ct})
            if eis_every and n % eis_every == 0:
                ci += 1
                imp_rows.append({"Cell_ID": cid, "Cycle_Index": ci,
                                 "Re_ohm": r_int * (1 + 0.02 * rng.standard_normal()),
                                 "Rct_ohm": r_ct * (1 + 0.02 * rng.standard_normal())})
            # ---- ageing over this charge + discharge ----
            A = (Q + cap)
            dsoh = k * float(twin_stress(np.mean(temps), p)) * A
            soh -= dsoh
            r_int += p.beta_int * r_int0 * dsoh
            r_ct += p.beta_ct * r_ct0 * dsoh
        truth[cid] = {"k_ah": k, "T_amb": T_amb, "I_dis": I_dis, "r_int0": r_int0, "r_ct0": r_ct0,
                      "trajectory": pd.DataFrame(traj)}
    master = normalise_master(pd.DataFrame(rows))
    imp = pd.DataFrame(imp_rows, columns=IMPEDANCE_COLUMNS)
    return master, imp, truth


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def run_manifest(config: Dict[str, Any], data_key: str = "", extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Provenance record for a result: engine and library versions, platform, UTC time,
    data identity, the full configuration and seeds. Serialise with json.dumps."""
    import datetime
    import importlib
    import platform
    import sys

    versions = {}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "pyarrow", "plotly", "streamlit"):
        try:
            versions[mod] = getattr(importlib.import_module(mod), "__version__", "?")
        except Exception:
            versions[mod] = None
    return {"engine_version": ENGINE_VERSION,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "packages": versions, "git_commit": os.environ.get("GIT_COMMIT", ""),
            "data_key": data_key, "config": _jsonable(config), "extra": _jsonable(extra or {})}


# =============================================================================
# 13. HEALTH INDICATORS (Mission 1: which parameter best represents health?)
# =============================================================================
@dataclass(frozen=True)
class HealthIndicator:
    key: str
    label: str
    unit: str
    mode: str            # degradation mode it mainly reflects (LLI / LAM / CL / thermal)
    source: str          # discharge / charge / EIS


HI_CATALOG: Dict[str, HealthIndicator] = {h.key: h for h in (
    HealthIndicator("Capacity_Ah", "Discharge capacity", "Ah", "LLI + LAM", "discharge"),
    HealthIndicator("E_dis_Wh", "Discharge energy", "Wh", "LLI + LAM + CL", "discharge"),
    HealthIndicator("V_mean_V", "Mean discharge voltage", "V", "CL (polarisation)", "discharge"),
    HealthIndicator("R_dc_ohm", "Load-step DC resistance", "Ω", "CL", "discharge"),
    HealthIndicator("t_dis_s", "Discharge duration", "s", "LLI + LAM", "discharge"),
    HealthIndicator("dT_C", "Temperature rise on discharge", "°C", "CL (Joule heat)", "discharge"),
    HealthIndicator("t_cc_s", "CC-charge duration", "s", "LLI + LAM", "charge"),
    HealthIndicator("t_cv_s", "CV-charge duration", "s", "CL", "charge"),
    HealthIndicator("Q_ch_Ah", "Charged capacity", "Ah", "LLI + LAM", "charge"),
    HealthIndicator("eff_energy", "Round-trip energy efficiency", "–", "CL", "charge + discharge"),
    HealthIndicator("Re_ohm", "EIS electrolyte resistance Rₑ", "Ω", "CL (electrolyte / contact)", "EIS"),
    HealthIndicator("Rct_ohm", "EIS charge-transfer resistance R_ct", "Ω", "CL (interphase / SEI)", "EIS"),
)}


def attach_eis(ct: pd.DataFrame, imp: Optional[pd.DataFrame], max_gap_cycles: int = 30) -> pd.DataFrame:
    """Add Re_ohm / Rct_ohm to the cycle table by per-cell interpolation over the discharge count
    between EIS tests (no extrapolation beyond ``max_gap_cycles`` of the nearest test)."""
    out = ct.copy()
    out["Re_ohm"] = np.nan
    out["Rct_ohm"] = np.nan
    if imp is None or imp.empty:
        return out
    for cid, d in out.groupby("Cell_ID"):
        e = valid_eis(imp, cid)
        if e.empty:
            continue
        ci = _int_key(d.sort_values("Cycle_Index")[["Cycle_Index", "n"]])
        e = pd.merge_asof(_int_key(e).sort_values("Cycle_Index"), ci, on="Cycle_Index", direction="backward")
        e["n"] = e["n"].fillna(0.5)
        e = e.groupby("n", as_index=False)[["Re_ohm", "Rct_ohm"]].median()
        if len(e) < 1:
            continue
        n = d["n"].to_numpy(dtype=float)
        near = np.min(np.abs(n[:, None] - e["n"].to_numpy()[None, :]), axis=1) <= max_gap_cycles
        for col in ("Re_ohm", "Rct_ohm"):
            vals = np.interp(n, e["n"], e[col]) if len(e) > 1 else np.full(len(n), float(e[col].iloc[0]))
            out.loc[d.index, col] = np.where(near, vals, np.nan)
    return out


def _monotonicity(x: np.ndarray) -> float:
    dx = np.diff(x)
    dx = dx[np.isfinite(dx)]
    pos, neg = int((dx > 0).sum()), int((dx < 0).sum())
    return float(abs(pos - neg) / (pos + neg)) if pos + neg else float("nan")


def rank_health_indicators(ct: pd.DataFrame, imp: Optional[pd.DataFrame] = None, min_cycles: int = 20,
                           smooth: int = 5, min_coverage: float = 0.6) -> pd.DataFrame:
    """Rank candidate health indicators with the prognostic-parameter criteria of Coble & Hines
    (2009) plus a cross-cell predictive test.

    Per cell each HI is smoothed (rolling median) and normalised to its beginning-of-life value
    (x / x_BOL), which removes cell-to-cell offsets. Then:
      |rho|           median |Spearman(HI, SOH)| within cells   (association with capacity health)
      monotonicity    median |#increases - #decreases| / (n - 1)
      trendability    |mean sign(rho(HI, n))| x median |rho(HI, n)|  (same direction in every cell)
      prognosability  exp(-std(end values) / mean |end - start|) across cells
      LOCO RMSE       SOH predicted from the HI alone by a quadratic map fitted on the other
                      cells (leave-one-cell-out): can this single parameter stand in for SOH?
    The capacity row is the reference (it *is* SOH up to a constant)."""
    from scipy.stats import spearmanr

    data = attach_eis(ct, imp)
    good = data[~data["outlier"]]
    cells = [c for c, d in good.groupby("Cell_ID") if len(d) >= min_cycles]
    rows = []
    for key, hi in HI_CATALOG.items():
        if key not in good.columns:
            continue
        per: Dict[str, pd.DataFrame] = {}
        rho_s, rho_n, mono, starts, ends = [], [], [], [], []
        for c in cells:
            d = good[good["Cell_ID"] == c].sort_values("n")
            x = d[key].astype(float)
            if x.notna().mean() < min_coverage or x.notna().sum() < 8:
                continue
            xs = x.interpolate(limit_direction="both").rolling(smooth, center=True, min_periods=1).median()
            base = float(xs.head(3).median())
            if not np.isfinite(base) or abs(base) < 1e-12:
                continue
            rel = (xs / base).to_numpy()
            per[c] = pd.DataFrame({"rel": rel, "SOH": d["SOH"].to_numpy()})
            rho_s.append(spearmanr(rel, d["SOH"]).statistic)
            rn = spearmanr(rel, d["n"]).statistic
            rho_n.append(rn)
            mono.append(_monotonicity(rel))
            starts.append(rel[0])
            ends.append(rel[-1])
        if len(per) < 2:
            continue
        rho_s, rho_n = np.array(rho_s, dtype=float), np.array(rho_n, dtype=float)
        trend = float(abs(np.nanmean(np.sign(rho_n))) * np.nanmedian(np.abs(rho_n)))
        spread = float(np.mean(np.abs(np.array(ends) - np.array(starts))))
        prog = float(np.exp(-np.std(ends) / spread)) if spread > 1e-9 else 0.0
        # leave-one-cell-out quadratic map rel-HI -> SOH
        errs = []
        for c in per:
            tr = pd.concat([per[o] for o in per if o != c])
            if len(tr) < 10 or tr["rel"].std() < 1e-9:
                continue
            coef = np.polyfit(tr["rel"], tr["SOH"], 2)
            te_ = per[c]
            errs.append(np.sqrt(np.mean((np.polyval(coef, te_["rel"]) - te_["SOH"]) ** 2)))
        rows.append({"key": key, "Indicator": hi.label, "Unit": hi.unit, "Mode": hi.mode, "Source": hi.source,
                     "Cells": len(per), "|ρ| with SOH": float(np.nanmedian(np.abs(rho_s))),
                     "Monotonicity": float(np.nanmedian(mono)), "Trendability": trend,
                     "Prognosability": prog, "LOCO RMSE (SOH)": float(np.median(errs)) if errs else np.nan,
                     "Direction": "rises" if np.nanmedian(rho_n) > 0 else "falls"})
    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab
    tab["Fitness"] = tab[["|ρ| with SOH", "Monotonicity", "Trendability", "Prognosability"]].mean(axis=1)
    return tab.sort_values(["Fitness"], ascending=False).set_index("key")


def hi_trajectories(ct: pd.DataFrame, imp: Optional[pd.DataFrame], cell_id: str,
                    keys: Sequence[str], smooth: int = 5) -> pd.DataFrame:
    """Beginning-of-life-normalised HI trajectories of one cell (long format)."""
    d = attach_eis(ct, imp)
    d = d[(d["Cell_ID"] == cell_id) & ~d["outlier"]].sort_values("n")
    out = []
    for k in keys:
        if k not in d.columns or d[k].notna().sum() < 3:
            continue
        x = d[k].astype(float).interpolate(limit_direction="both").rolling(smooth, center=True, min_periods=1).median()
        base = float(x.head(3).median())
        if np.isfinite(base) and abs(base) > 1e-12:
            out.append(pd.DataFrame({"n": d["n"].to_numpy(), "key": k, "rel": (x / base).to_numpy()}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["n", "key", "rel"])


def hi_pca(ct: pd.DataFrame, imp: Optional[pd.DataFrame] = None, keys: Optional[Sequence[str]] = None,
           min_coverage: float = 0.8) -> Dict[str, Any]:
    """Can degradation be represented by one parameter? PCA of the standardised, BOL-normalised
    indicators (capacity excluded) pooled over cells. If PC1 carries most of the variance and
    tracks SOH, a single health parameter suffices; material variance in PC2 means capacity
    fade and power fade (resistance) evolve partly independently and a multi-parameter state
    (as in theta = [SOH, R_int, R_ct]) is warranted."""
    data = attach_eis(ct, imp)
    good = data[~data["outlier"]].copy()
    keys = [k for k in (keys or HI_CATALOG) if k != "Capacity_Ah" and k in good.columns]
    rel = []
    for c, d in good.groupby("Cell_ID"):
        d = d.sort_values("n")
        cols = {}
        for k in keys:
            x = d[k].astype(float)
            if x.notna().mean() < min_coverage:
                continue
            x = x.interpolate(limit_direction="both").rolling(5, center=True, min_periods=1).median()
            b = float(x.head(3).median())
            if np.isfinite(b) and abs(b) > 1e-12:
                cols[k] = (x / b).to_numpy()
        if cols:
            f = pd.DataFrame(cols)
            f["SOH"] = d["SOH"].to_numpy()
            f["Cell_ID"] = c
            rel.append(f)
    if not rel:
        return {"available": False}
    X = pd.concat(rel, ignore_index=True)
    use = [k for k in keys if k in X.columns and X[k].notna().mean() >= min_coverage]
    X = X.dropna(subset=use)
    if len(use) < 2 or len(X) < 10:
        return {"available": False}
    Z = (X[use] - X[use].mean()) / X[use].std(ddof=0).replace(0, 1.0)
    U, S, Vt = np.linalg.svd(Z.to_numpy(), full_matrices=False)
    var = S ** 2 / np.sum(S ** 2)
    pcs = Z.to_numpy() @ Vt.T
    corr = [float(abs(np.corrcoef(pcs[:, i], X["SOH"])[0, 1])) for i in range(min(3, len(var)))]
    load = pd.DataFrame(Vt[: min(3, len(use))].T, index=use, columns=[f"PC{i + 1}" for i in range(min(3, len(use)))])
    return {"available": True, "explained": var, "loadings": load, "pc_soh_corr": corr, "n_rows": len(X),
            "keys": use, "single_parameter": bool(var[0] >= 0.8 and corr[0] >= 0.8)}


# =============================================================================
# 14. DEGRADATION MODES (LLI / LAM / CL), DVA AND KNEE DETECTION
# =============================================================================
@dataclass
class DVACurve:
    cycle_index: int
    n: int
    q: np.ndarray             # discharged capacity (Ah)
    dvdq: np.ndarray          # |dV/dQ| (V/Ah)
    capacity_Ah: float
    peak_q: np.ndarray        # positions of interior |dV/dQ| peaks (Ah)


def dva_curve(cell_df: pd.DataFrame, cycle_index: int, n: int = 0, dq: float = 0.02,
              smooth_window: int = 9, edge_frac: float = 0.08) -> Optional[DVACurve]:
    """Differential voltage analysis of one constant-current discharge: V is averaged on a
    uniform capacity grid, differentiated and Savitzky-Golay smoothed. DVA peaks mark
    electrode phase transitions; the capacity between peaks tracks the electrode that owns
    them (graphite staging on the negative side), so shrinking peak spacing indicates loss of
    active material on that electrode, while a rigid shift of all features indicates loss of
    lithium inventory (Bloom et al. 2005; Keil & Jossen 2017; Rufino Junior et al. 2024)."""
    from scipy.signal import find_peaks, savgol_filter

    d = cell_df[(cell_df["Cycle_Index"] == cycle_index) & (cell_df["Current_A"] < -0.1)]
    if len(d) < 30:
        return None
    q = np.cumsum(np.abs(d["Current_A"].to_numpy()) * d["dt"].to_numpy()) / 3600.0
    v = d["Voltage_V"].to_numpy()
    edges = np.arange(0.0, q[-1] + dq, dq)
    if len(edges) < smooth_window + 4:
        return None
    idx = np.clip(np.digitize(q, edges) - 1, 0, len(edges) - 2)
    vb = pd.Series(v).groupby(idx).mean()
    qc = 0.5 * (edges[vb.index.to_numpy()] + edges[vb.index.to_numpy() + 1])
    vv = vb.to_numpy()
    if len(vv) < smooth_window + 2:
        return None
    win = smooth_window if smooth_window % 2 else smooth_window + 1
    dvdq = np.abs(savgol_filter(np.gradient(vv, qc), win, 2))
    lo, hi = edge_frac * q[-1], (1 - edge_frac) * q[-1]
    inner = np.where((qc > lo) & (qc < hi), dvdq, 0.0)
    pk, _ = find_peaks(inner, prominence=0.15 * inner.max() if inner.max() > 0 else None)
    return DVACurve(int(cycle_index), int(n), qc, dvdq, float(q[-1]), qc[pk])


def dva_evolution(cell_df: pd.DataFrame, ct_cell: pd.DataFrame, n_curves: int = 6, **kw) -> List[DVACurve]:
    good = ct_cell[~ct_cell["outlier"]].reset_index(drop=True)
    if good.empty:
        return []
    idx = np.unique(np.linspace(0, len(good) - 1, min(n_curves, len(good))).astype(int))
    out = []
    for i in idx:
        c = dva_curve(cell_df, int(good.loc[i, "Cycle_Index"]), int(good.loc[i, "n"]), **kw)
        if c is not None:
            out.append(c)
    return out


def degradation_modes(curves: Sequence[ICACurve], ct_cell: pd.DataFrame,
                      eis: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Indicative LLI / LAM / CL trajectories (percent of beginning-of-life) from features the
    NASA data offer: ICA main-peak area (active material behind the dominant phase transition),
    total capacity, and resistance (load-step R_dc; EIS R_e + R_ct where available).

        LAM_proxy = (1 - A_peak / A_peak,0) * (A_peak,0 / Q_0)     capacity lost *inside* the main peak
        LLI_proxy = max(Q_loss - LAM_proxy, 0)                        capacity lost elsewhere
        CL        = R / R_0 - 1                                       conductivity loss (power fade)

    At ~1C without half-cell references these are proxies, not a quantitative mode analysis;
    the decomposition follows the LLI / LAM / CL taxonomy of Birkl et al. (2017) and Menye
    et al. (2025)."""
    if len(curves) < 2:
        return pd.DataFrame()
    a0, q0 = curves[0].peak_area, curves[0].capacity_Ah
    good = ct_cell[~ct_cell["outlier"]].sort_values("n")
    r = good.set_index("n")["R_dc_ohm"].astype(float)
    r = r.interpolate(limit_direction="both").rolling(5, center=True, min_periods=1).median()
    r0 = float(r.head(3).median()) if len(r) else float("nan")
    rows = []
    for c in curves:
        q_loss = max(0.0, 1 - c.capacity_Ah / q0)
        lam = max(0.0, (1 - c.peak_area / a0) * (a0 / q0)) if a0 > 0 else float("nan")
        lli = max(q_loss - (lam if np.isfinite(lam) else 0.0), 0.0)
        cl = float(r.get(c.n, np.nan) / r0 - 1) if np.isfinite(r0) and r0 > 0 else float("nan")
        rows.append({"n": c.n, "Capacity loss": 100 * q_loss, "LAM (proxy)": 100 * lam, "LLI (proxy)": 100 * lli,
                     "CL: R_dc growth": 100 * cl, "Peak shift (mV)": 1000 * (curves[0].peak_V - c.peak_V)})
    out = pd.DataFrame(rows)
    if eis is not None and len(eis) >= 2 and {"Re_ohm", "Rct_ohm"} <= set(eis.columns):
        ci = _int_key(ct_cell.sort_values("Cycle_Index")[["Cycle_Index", "n"]])
        e = pd.merge_asof(_int_key(eis).sort_values("Cycle_Index"), ci, on="Cycle_Index", direction="backward")
        e["n"] = e["n"].fillna(1)
        e = e.groupby("n")[["Re_ohm", "Rct_ohm"]].median()
        tot = e["Re_ohm"] + e["Rct_ohm"]
        out["CL: EIS Rₑ+R_ct growth"] = 100 * (np.interp(out["n"], tot.index, tot) / float(tot.iloc[0]) - 1)
    return out


def detect_knee(n: np.ndarray, soh: np.ndarray, min_seg: int = 8, min_ratio: float = 1.5,
                alpha: float = 0.01) -> Dict[str, Any]:
    """Knee point of a capacity-fade curve: continuous two-segment piecewise-linear fit
    (Bacon-Watts style) with the breakpoint chosen by least squares, accepted when the late
    slope is at least ``min_ratio`` times steeper and an F-test rejects the one-line model.
    The knee marks the transition to the non-linear ageing stage (lithium plating / LAM
    acceleration) after which 'sudden death' becomes likely; it is a maintenance trigger."""
    from scipy.stats import f as f_dist

    n = np.asarray(n, dtype=float)
    y = pd.Series(np.asarray(soh, dtype=float)).rolling(5, center=True, min_periods=1).median().to_numpy()
    out: Dict[str, Any] = {"found": False, "knee_n": None, "soh_at_knee": None}
    N = len(n)
    if N < 2 * min_seg + 2:
        return out
    X1 = np.column_stack([np.ones(N), n])
    sse1 = float(np.sum((y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]) ** 2))
    best = None
    for k in range(min_seg, N - min_seg):
        X = np.column_stack([np.ones(N), n, np.maximum(n - n[k], 0.0)])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        sse = float(np.sum((y - X @ beta) ** 2))
        if best is None or sse < best[0]:
            best = (sse, k, beta)
    sse2, k, beta = best
    s1, s2 = float(beta[1]), float(beta[1] + beta[2])
    F = ((sse1 - sse2) / 2) / (sse2 / max(N - 4, 1)) if sse2 > 0 else np.inf
    p = float(1 - f_dist.cdf(F, 2, max(N - 4, 1))) if np.isfinite(F) else 0.0
    ratio = s2 / s1 if s1 < 0 else (np.inf if s2 < 0 else 0.0)
    found = bool(s2 < 0 and ratio >= min_ratio and p < alpha)
    out.update({"found": found, "knee_n": int(n[k]), "soh_at_knee": float(np.interp(n[k], n, y)),
                "slope_before": s1, "slope_after": s2, "ratio": float(ratio), "p_value": p})
    return out


# =============================================================================
# 15. OPERATING STRESS, SAFETY EXPOSURE AND OPERATING-CONDITION EFFECTS
# =============================================================================
@dataclass
class SafetyLimits:
    """Screening thresholds for an 18650 LiCoO2/graphite cell (NASA spec: 4.2 V CC-CV at 1.5 A,
    2 A nominal discharge, 2.7 V nominal cut-off) and literature stress regimes
    (Rufino Junior et al. 2024; Menye et al. 2025): SEI growth accelerates above ~45 °C and
    the SEI starts to decompose approaching 60 °C (thermal-runaway precursor); charging in
    the cold favours lithium plating; deep discharge below ~2.5 V risks copper dissolution
    and active-material damage; high C-rates add heat and particle cracking."""
    T_warn_C: float = 45.0
    T_crit_C: float = 60.0
    T_plating_C: float = 10.0
    V_deep_V: float = 2.5
    V_over_V: float = 4.22
    crate_high: float = 1.5


STRESS_MECHANISMS = {
    "hot": ("Elevated temperature", "SEI growth → LLI; electrolyte oxidation; CEI growth"),
    "critical_T": ("Near SEI-decomposition onset", "SEI breakdown, gas evolution; thermal-runaway precursor"),
    "plating": ("Cold charging", "Lithium plating → LLI, dendrites, internal-short risk"),
    "deep": ("Deep discharge", "Cu current-collector dissolution, LAM, cathode stress"),
    "high_rate": ("High discharge rate", "Joule heat, particle cracking → LAM, CL"),
    "over_v": ("Over-voltage", "Cathode (LiCoO₂) instability, electrolyte oxidation"),
}


def stress_exposure(ct: pd.DataFrame, limits: Optional[SafetyLimits] = None) -> pd.DataFrame:
    """Per-cycle stressor flags and a plating-risk index
    PRI = (I_charge / C_bol) * max(0, T_plating - T_charge) / 10."""
    L = limits or SafetyLimits()
    d = ct.copy()
    T = d["T_max_C"].astype(float)
    d["hot"] = T >= L.T_warn_C
    d["critical_T"] = T >= L.T_crit_C
    tch = d["T_ch_C"].astype(float) if "T_ch_C" in d.columns else d["T_mean_C"].astype(float)
    ich = d["I_ch_A"].astype(float) if "I_ch_A" in d.columns else pd.Series(1.5, index=d.index)
    d["PRI"] = (ich.fillna(1.5) / d["C_bol_Ah"]) * np.maximum(0.0, L.T_plating_C - tch.fillna(25.0)) / 10.0
    d["plating"] = d["PRI"] > 0
    d["deep"] = d["V_min_V"].astype(float) < L.V_deep_V
    d["high_rate"] = d["I_dis_A"].astype(float) / d["C_bol_Ah"] >= L.crate_high
    vmax = d["V_ch_max_V"].astype(float) if "V_ch_max_V" in d.columns else pd.Series(np.nan, index=d.index)
    d["over_v"] = vmax > L.V_over_V
    return d


def stress_summary(ct: pd.DataFrame, limits: Optional[SafetyLimits] = None) -> pd.DataFrame:
    """Share of each cell's cycles exposed to each stressor (percent), plus peak temperature."""
    d = stress_exposure(ct, limits)
    d = d[~d["outlier"]]
    g = d.groupby("Cell_ID")
    out = pd.DataFrame({STRESS_MECHANISMS[k][0]: 100 * g[k].mean() for k in STRESS_MECHANISMS})
    out["Peak T (°C)"] = g["T_max_C"].max()
    out["Max PRI"] = g["PRI"].max()
    return out


def stress_factor_regression(ct: pd.DataFrame, window_frac: float = 0.5, n_boot: int = 300,
                             seed: int = 0) -> Dict[str, Any]:
    """How do operating conditions influence degradation? Cohort regression of the early fade
    rate per Ah on the stress factors available in the NASA design:
        ln(rate) = b0 - (Ea/R)(1/T - 1/T_ref) + a ln(I/I_ref) + c (V_cut - 2.7) + d [T < 15 °C]
    with a cell-level bootstrap CI. Terms without variation in the cohort are dropped."""
    good = ct[~ct["outlier"]]
    meta = cell_meta(ct)
    recs = []
    for cid, d in good.groupby("Cell_ID"):
        d = d.sort_values("n")
        d = d[(d["n"] <= max(5, window_frac * d["n"].max())) & np.isfinite(d["cum_Ah"])]
        if len(d) < 5:
            continue
        x = d["cum_Ah"].to_numpy() - d["cum_Ah"].iloc[0]
        if x.max() <= 0:
            continue
        rate = float(np.polyfit(x, 1 - d["SOH"].to_numpy(), 1)[0])
        if rate <= 0:
            continue
        recs.append({"Cell_ID": cid, "rate": rate, "T_K": float(meta.loc[cid, "T_mean_C"]) + 273.15,
                     "I": float(meta.loc[cid, "I_dis_A"]), "Vc": float(meta.loc[cid, "V_cut_V"]),
                     "cold": float(meta.loc[cid, "Ambient_C"] < 15)})
    tab = pd.DataFrame(recs)
    out: Dict[str, Any] = {"available": False, "table": tab}
    if len(tab) < 4:
        return out
    Tref = 24 + 273.15
    cols = {"intercept": np.ones(len(tab)),
            "Ea (kJ/mol)": -(1 / tab["T_K"] - 1 / Tref) / R_GAS / 1e-3,
            "current exponent": np.log(tab["I"].clip(lower=0.1) / 2.0),
            "cut-off voltage (per V)": tab["Vc"] - 2.7,
            "cold regime (×)": tab["cold"]}
    X = pd.DataFrame(cols)
    min_std = {"Ea (kJ/mol)": 0.005, "current exponent": 0.1,           # ~4 K temperature spread
               "cut-off voltage (per V)": 0.05, "cold regime (×)": 0.1}
    X = X[[c for c in X.columns if c == "intercept" or X[c].std() > min_std.get(c, 1e-9)]]
    y = np.log(tab["rate"].to_numpy())
    if len(tab) <= X.shape[1]:
        X = X[["intercept"] + [c for c in X.columns if c in ("Ea (kJ/mol)", "current exponent")]]
    beta = np.linalg.lstsq(X.to_numpy(), y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(tab), len(tab))
        try:
            boots.append(np.linalg.lstsq(X.to_numpy()[i], y[i], rcond=None)[0])
        except np.linalg.LinAlgError:
            continue
    B = np.array(boots)
    res = []
    for j, c in enumerate(X.columns):
        if c == "intercept":
            continue
        est, lo, hi = beta[j], np.nanpercentile(B[:, j], 5), np.nanpercentile(B[:, j], 95)
        if c == "cold regime (×)":
            est, lo, hi = np.exp(est), np.exp(lo), np.exp(hi)
        res.append({"Factor": c, "Estimate": float(est), "90% CI low": float(lo), "90% CI high": float(hi)})
    pred = X.to_numpy() @ beta
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2) if len(y) > 2 else float("nan")
    out.update({"available": True, "coefficients": pd.DataFrame(res).set_index("Factor"), "r2": float(r2),
                "n_cells": len(tab)})
    return out


# =============================================================================
# 16. SEMI-EMPIRICAL AND PARTICLE-FILTER PROGNOSTICS
# =============================================================================
SEMI_NAME = "Semi-empirical · power law"
PF_NAME = "Particle filter · double exp."


@dataclass
class ProgForecast:
    name: str
    n_grid: np.ndarray
    soh: np.ndarray
    lo: Optional[np.ndarray]
    hi: Optional[np.ndarray]
    rul_samples: Optional[np.ndarray]
    params: Dict[str, Any]
    metrics: Optional[ForecastMetrics] = None


def _paths_to_forecast(name: str, n_grid: np.ndarray, obs_n: np.ndarray, obs_soh: np.ndarray, n0: int,
                       paths: np.ndarray, soh_eol: float, level: float, params: Dict[str, Any]) -> ProgForecast:
    """Median / central band / RUL samples from Monte-Carlo SOH paths over n > n0."""
    med = np.interp(n_grid, obs_n, pd.Series(obs_soh).rolling(5, center=True, min_periods=1).median()).astype(float)
    lo, hi = med.copy(), med.copy()
    fut = n_grid > n0
    k = int(fut.sum())
    qa, qb = (1 - level) / 2, 1 - (1 - level) / 2
    med[fut] = np.median(paths[:, :k], axis=0)
    lo[fut] = np.quantile(paths[:, :k], qa, axis=0)
    hi[fut] = np.quantile(paths[:, :k], qb, axis=0)
    below = paths[:, :k] < soh_eol
    hit = below.any(axis=1)
    rul = np.where(hit, below.argmax(axis=1) + 1, np.nan).astype(float)
    return ProgForecast(name, n_grid, med, lo, hi, rul, params)


def _mean_ah_per_cycle(good: pd.DataFrame, n0: int, last: int = 20) -> float:
    d = good[good["n"] <= n0].sort_values("n")
    da = d["cum_Ah"].diff().dropna()
    da = da[(da > 0) & np.isfinite(da)].tail(last)
    if len(da):
        return float(da.median())
    return float(2 * d["Capacity_Ah"].tail(last).median())


def semi_empirical_forecast(ct: pd.DataFrame, cell_id: str, n0: int, eol_ah: float = DEFAULT_EOL_AH,
                            level: float = 0.9, n_samples: int = 400, horizon_factor: float = 1.5,
                            z_bounds: Tuple[float, float] = (0.4, 2.5), seed: int = 0,
                            alpha: float = 0.2) -> ProgForecast:
    """Semi-empirical cycle-ageing law in throughput (Wang et al., J. Power Sources 196 (2011)
    3942; the 'semi-empirical' class of Rufino Junior et al. 2024):
        Q_loss(Ah) = B(T, I) * Ah^z,      SOH = 1 - Q_loss
    B absorbs the Arrhenius and current stress at this cell's operating point; z is the
    kinetics exponent (z = 0.5 diffusion-limited SEI growth, z = 1 linear, z > 1 accelerating).
    Fitted in log space on the target's cycles <= n0 with a cohort prior on z (ridge towards
    the median exponent of the other cells). Uncertainty: the parameter covariance of the
    log-linear fit plus residual scatter, propagated by Monte Carlo. Future throughput per
    cycle = median of the last 20 observed cycles."""
    good_all = ct[~ct["outlier"]]
    good = good_all[good_all["Cell_ID"] == cell_id].sort_values("n")
    obs = good[(good["n"] <= n0) & np.isfinite(good["cum_Ah"])]
    if len(obs) < 8:
        raise ValueError("Semi-empirical model: too few observations before n0.")
    ah0 = float(good["cum_Ah"].iloc[0])

    def fit(d: pd.DataFrame, z_prior: Optional[float] = None, lam: float = 0.0):
        """Nonlinear least squares in SOH space: SOH = s0 (1 - B Ah^z), theta = (s0, log B, z).
        Fitting the level s0 jointly avoids the log-space fragility of small, offset losses."""
        from scipy.optimize import least_squares

        a = np.maximum(d["cum_Ah"].to_numpy() - float(d["cum_Ah"].iloc[0]), 1e-6)
        y = d["SOH"].to_numpy(dtype=float)
        if len(y) < 6 or a.max() < 1.0:
            return None
        amax = float(a.max())

        def f(th, aa):
            return th[0] * (1 - np.exp(th[1]) * (aa / amax) ** th[2])

        def resid(th):
            r = f(th, a) - y
            if z_prior is not None and lam > 0:
                r = np.append(r, math.sqrt(lam) * 0.01 * (th[2] - z_prior))
            return r

        loss0 = max(float(y[:3].mean() - y[-3:].mean()), 1e-3)
        x0 = np.array([float(np.median(y[:3])), math.log(loss0), z_prior or 1.0])
        sol = least_squares(resid, x0, bounds=([0.8, -15.0, z_bounds[0]], [1.2, 0.0, z_bounds[1]]))
        J = sol.jac
        dof = max(len(y) - 3, 1)
        s2 = float(np.sum((f(sol.x, a) - y) ** 2) / dof)
        cov = s2 * np.linalg.pinv(J.T @ J)
        return sol.x, cov, math.sqrt(s2), amax

    zs = []
    for c, d in good_all.groupby("Cell_ID"):
        if c == cell_id or len(d) < 15:
            continue
        r = fit(d.sort_values("n"))
        if r is not None:
            zs.append(float(r[0][2]))
    z_prior = float(np.clip(np.median(zs), *z_bounds)) if zs else 1.0
    res = fit(obs, z_prior, lam=float(len(obs)) * 0.05)
    if res is None:
        raise ValueError("Semi-empirical model: capacity loss still within noise at n0.")
    beta, cov, s, amax = res
    rng = np.random.default_rng(seed)
    th = rng.multivariate_normal(beta, cov + 1e-12 * np.eye(3), size=n_samples)
    th[:, 2] = np.clip(th[:, 2], *z_bounds)
    n_max = int(good["n"].max() * horizon_factor)
    n_grid = np.arange(1, n_max + 1)
    dah = _mean_ah_per_cycle(good, n0)
    a_now = float(obs["cum_Ah"].iloc[-1] - ah0)
    n_last = int(obs["n"].iloc[-1])
    fut_n = n_grid[n_grid > n0]
    ah_f = np.maximum(a_now + (fut_n - n_last) * dah, 1e-6)
    paths = th[:, [0]] * (1 - np.exp(th[:, [1]]) * (ah_f[None, :] / amax) ** th[:, [2]])
    paths = paths + s * 0.5 * rng.standard_normal((n_samples, 1))          # persistent level uncertainty
    s0 = float(beta[0])
    soh_eol = soh_eol_for(float(good["C_bol_Ah"].iloc[0]), eol_ah)
    fc = _paths_to_forecast(SEMI_NAME, n_grid, good["n"].to_numpy(), good["SOH"].to_numpy(), n0, paths, soh_eol,
                            level, {"B": float(math.exp(beta[1]) / amax ** beta[2]), "z": float(beta[2]), "z_prior": z_prior,
                                    "SOH_start": s0,
                                    "Ah_per_cycle": dah, "cohort_cells": len(zs)})
    fc.metrics = forecast_metrics(good["n"].to_numpy(), good["SOH"].to_numpy(), n_grid, fc.soh, n0, soh_eol,
                                  fc.lo, fc.hi, alpha)
    return fc


def _dexp(theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    """SOH(t) = a e^(b t) + c e^(d t); theta rows = particles, t = n / 100."""
    a, b, c, d = (theta[:, i:i + 1] for i in range(4))
    return a * np.exp(b * t[None, :]) + c * np.exp(d * t[None, :])


_PF_BOUNDS = np.array([[0.7, 1.3], [-3.0, 0.3], [-0.6, 0.3], [0.0, 3.0]])


def _fit_dexp(n: np.ndarray, soh: np.ndarray) -> Optional[np.ndarray]:
    from scipy.optimize import curve_fit

    t = np.asarray(n, dtype=float) / 100.0
    f = lambda tt, a, b, c, d: a * np.exp(b * tt) + c * np.exp(d * tt)
    try:
        p, _ = curve_fit(f, t, soh, p0=[1.0, -0.1, -0.01, 1.0], bounds=(_PF_BOUNDS[:, 0], _PF_BOUNDS[:, 1]),
                         maxfev=20000)
        return p
    except Exception:
        return None


def particle_filter_forecast(ct: pd.DataFrame, cell_id: str, n0: int, eol_ah: float = DEFAULT_EOL_AH,
                             level: float = 0.9, n_particles: int = 3000, sigma_obs: float = 0.012,
                             jitter: float = 0.02, horizon_factor: float = 1.5, seed: int = 0,
                             alpha: float = 0.2) -> ProgForecast:
    """Bayesian particle-filter prognostic with the double-exponential capacity model used on
    the NASA Ames cells by Saha & Goebel (2009): SOH(n) = a e^(b n) + c e^(d n).

    Prior: parameters fitted to every *other* cell (population knowledge), a Gaussian in
    parameter space with inflated covariance. The target's cycles <= n0 are assimilated one at
    a time (Student-t likelihood, robust to regeneration spikes), with systematic resampling
    when the effective sample size drops below N/2 and small roughening jitter against
    particle impoverishment. Every surviving particle is then extrapolated, giving the
    median path, a central band and the RUL distribution."""
    good_all = ct[~ct["outlier"]]
    good = good_all[good_all["Cell_ID"] == cell_id].sort_values("n")
    obs = good[good["n"] <= n0]
    if len(obs) < 5:
        raise ValueError("Particle filter: too few observations before n0.")
    fits = []
    for c, d in good_all.groupby("Cell_ID"):
        if c == cell_id or len(d) < 15:
            continue
        p = _fit_dexp(d["n"].to_numpy(), d["SOH"].to_numpy())
        if p is not None:
            fits.append(p)
    rng = np.random.default_rng(seed)
    if len(fits) >= 3:
        F = np.array(fits)
        mu, C = F.mean(axis=0), np.cov(F.T) * 2.0 + np.diag([1e-4, 1e-3, 1e-4, 1e-2])
    else:
        mu, C = np.array([1.0, -0.1, -0.01, 1.0]), np.diag([0.02, 0.1, 0.01, 0.5]) ** 2
    parts = rng.multivariate_normal(mu, C, size=n_particles)
    parts = np.clip(parts, _PF_BOUNDS[:, 0], _PF_BOUNDS[:, 1])
    logw = np.zeros(n_particles)
    nu = 4.0
    scale = np.abs(mu) * jitter + 1e-4
    n_resample = 0
    for n_k, y_k in zip(obs["n"].to_numpy(dtype=float), obs["SOH"].to_numpy(dtype=float)):
        pred = _dexp(parts, np.array([n_k / 100.0]))[:, 0]
        r = (y_k - pred) / sigma_obs
        logw += -0.5 * (nu + 1) * np.log1p(r ** 2 / nu)
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
        if 1.0 / np.sum(w ** 2) < n_particles / 2:
            pos = (rng.random() + np.arange(n_particles)) / n_particles
            idx = np.minimum(np.searchsorted(np.cumsum(w), pos), n_particles - 1)
            parts = parts[idx] + rng.standard_normal(parts.shape) * scale
            parts = np.clip(parts, _PF_BOUNDS[:, 0], _PF_BOUNDS[:, 1])
            logw = np.zeros(n_particles)
            n_resample += 1
    w = np.exp(logw - logw.max())
    w /= w.sum()
    idx = rng.choice(n_particles, size=min(800, n_particles), p=w)
    post = parts[idx]
    n_max = int(good["n"].max() * horizon_factor)
    n_grid = np.arange(1, n_max + 1)
    fut = n_grid[n_grid > n0]
    paths = _dexp(post, fut / 100.0)
    paths = paths + sigma_obs * 0.3 * rng.standard_normal((len(post), 1))
    soh_eol = soh_eol_for(float(good["C_bol_Ah"].iloc[0]), eol_ah)
    fc = _paths_to_forecast(PF_NAME, n_grid, good["n"].to_numpy(), good["SOH"].to_numpy(), n0, paths, soh_eol,
                            level, {"posterior_mean": post.mean(axis=0).tolist(), "prior_cells": len(fits),
                                    "resampling_steps": n_resample, "n_particles": n_particles})
    fc.metrics = forecast_metrics(good["n"].to_numpy(), good["SOH"].to_numpy(), n_grid, fc.soh, n0, soh_eol,
                                  fc.lo, fc.hi, alpha)
    return fc


# =============================================================================
# 17. MISSION 2 STUDIES: UPDATE FREQUENCY
# =============================================================================
def update_frequency_study(cell_df: pd.DataFrame, ct: pd.DataFrame, imp: Optional[pd.DataFrame], cell_id: str,
                           params: TwinParameters, base_cfg: Optional[DualTwinConfig], n0: int, soh_eol: float,
                           intervals: Sequence[int] = (1, 2, 5, 10, 20, 50), level: float = 0.9,
                           progress: ProgressFn = None, rel_tol: float = 1.25, abs_tol: float = 0.005
                           ) -> Tuple[pd.DataFrame, Dict[int, EKFResult]]:
    """How often should the twin update? The dual twin assimilates measurements only every m-th
    discharge (between updates it runs open loop on the fade law). Reported per m: causal
    tracking RMSE and the worst error, 90% tracking coverage, measurements used per 100
    cycles, forecast RMSE / RUL error from n0, and the information gain per update. The knee
    of tracking error versus measurement cost is the recommended update interval."""
    base = base_cfg or DualTwinConfig()
    meta = cell_meta(ct)
    prep = _dual_prepare(cell_df, ct, imp, cell_id, params, similar_cells(meta, cell_id))
    good = ct[(ct["Cell_ID"] == cell_id) & ~ct["outlier"]].sort_values("n")
    n_max = int(good["n"].max() * 1.5)
    rows, results = [], {}
    for j, m in enumerate(intervals):
        _report(progress, j / len(intervals), f"update every {m} cycle(s)")
        r = run_dual_twin(cell_df, ct, imp, cell_id, params, replace(base, update_every=int(m)), _prep=prep)
        results[int(m)] = r
        est = good[["n", "SOH"]].merge(r.per_cycle[["n", "SOH", "SOH_std"]], on="n", suffixes=("", "_est"))
        err = est["SOH_est"] - est["SOH"]
        fc = twin_forecast(r, n0, n_max, soh_eol, level)
        fm = forecast_metrics(good["n"].to_numpy(), good["SOH"].to_numpy(), fc.n_grid, fc.soh, n0, soh_eol,
                              fc.lo, fc.hi)
        upd = r.per_cycle[r.per_cycle["n_meas"] > 0]
        rows.append({"Update every (cycles)": int(m), "Updates per 100 cycles": 100 * len(upd) / max(len(r.per_cycle), 1),
                     "Tracking RMSE": float(np.sqrt(np.mean(err ** 2))), "Worst |error|": float(np.abs(err).max()),
                     "Tracking 90% coverage": float(np.mean(np.abs(err) <= 1.645 * est["SOH_std"])),
                     "Info gain per update (nats)": float((upd["ig_soh"] + upd["ig_logk"]).mean()) if len(upd) else 0.0,
                     "Forecast RMSE": fm.rmse, "RUL error": fm.rul_error, "Forecast coverage": fm.coverage})
    _report(progress, 1.0, "update study done")
    tab = pd.DataFrame(rows).set_index("Update every (cycles)")
    base_rmse = float(tab["Tracking RMSE"].iloc[0])
    # sparsest schedule whose tracking error stays within rel_tol x the every-cycle error OR below
    # abs_tol (0.5% SOH ~ capacity-measurement repeatability): extra updates below that buy nothing
    ok = tab[(tab["Tracking RMSE"] <= rel_tol * base_rmse) | (tab["Tracking RMSE"] <= abs_tol)]
    tab.attrs["recommended"] = int(ok.index.max()) if len(ok) else int(tab.index.min())
    return tab, results


# =============================================================================
# 18. MISSION 3: INTEGRATED OPERATION AND MAINTENANCE OPTIMISATION
# =============================================================================
@dataclass
class MaintenanceModel:
    """Maintenance economics and end-of-life risk.

    Planned replacement at the chosen SOH threshold costs ``replacement_cost`` plus
    ``planned_downtime_h`` of lost operation. Ageing cells can also fail suddenly (the
    'sudden death' after the knee reported in both reviews: lithium plating, LAM
    acceleration, internal shorts), modelled as a per-cycle hazard that rises smoothly once
    SOH falls below ``hazard_soh``:
        h(SOH) = h_base + h_max / (1 + exp((SOH - hazard_soh) / hazard_width))
    An unplanned failure costs ``unplanned_factor`` x the replacement cost plus longer
    downtime. PerformanceLoss (J_maint) is the revenue lost because an aged cell delivers
    less charge per cycle than a fresh one."""
    replacement_cost: float = 150.0
    planned_downtime_h: float = 24.0
    unplanned_factor: float = 3.0
    unplanned_downtime_h: float = 120.0
    hazard_soh: float = 0.68
    hazard_width: float = 0.02
    h_base: float = 1e-5
    h_max: float = 0.03

    def __post_init__(self) -> None:
        if min(self.replacement_cost, self.planned_downtime_h, self.unplanned_downtime_h, self.h_base,
               self.h_max) < 0 or self.unplanned_factor < 1 or self.hazard_width <= 0:
            raise ValueError("MaintenanceModel: invalid (negative) cost, downtime or hazard parameter")

    def hazard(self, soh: np.ndarray) -> np.ndarray:
        soh = np.asarray(soh, dtype=float)
        return np.clip(self.h_base + self.h_max / (1 + np.exp((soh - self.hazard_soh) / self.hazard_width)), 0, 1)


def renewal_evaluation(life: pd.DataFrame, threshold: float, e: Economics, mm: MaintenanceModel) -> Dict[str, float]:
    """Expected long-run economics of one renewal cycle (install -> replace) truncated at the
    replacement threshold, with sudden-failure risk (renewal-reward theorem: the long-run
    profit rate equals E[profit per renewal] / E[duration per renewal]).
        Profit = Revenue - EnergyCost - MaintenanceCost     (project objective)
        J_op   = Revenue - EnergyCost
        J_maint = MaintenanceCost + PerformanceLoss"""
    d = life[life["SOH"] >= threshold]
    if d.empty:
        return {"threshold": threshold, "cycles": 0, "rate": float("nan")}
    h = mm.hazard(d["SOH"].to_numpy())
    surv_start = np.concatenate([[1.0], np.cumprod(1 - h)[:-1]])      # alive at the start of cycle k
    p_fail = 1 - float(np.prod(1 - h))
    rev = float(np.sum(surv_start * d["revenue"]))
    ecost = float(np.sum(surv_start * d["energy_cost"]))
    hours = float(np.sum(surv_start * d["hours"]))
    fresh_rev = float(d["revenue"].iloc[0] / max(d["SOH"].iloc[0], 1e-6))
    perf_loss = float(np.sum(surv_start * (fresh_rev * d["SOH"] - d["revenue"]).clip(lower=0)))
    maint = (1 - p_fail) * mm.replacement_cost + p_fail * mm.unplanned_factor * mm.replacement_cost
    down = (1 - p_fail) * mm.planned_downtime_h + p_fail * mm.unplanned_downtime_h
    profit = rev - ecost - maint
    total_h = hours + down
    exp_cycles = float(np.sum(surv_start))
    return {"threshold": threshold, "cycles": exp_cycles, "hours": total_h, "revenue": rev, "energy_cost": ecost,
            "maintenance_cost": maint, "performance_loss": perf_loss, "J_op": rev - ecost,
            "J_maint": maint + perf_loss, "profit": profit, "rate": profit / total_h if total_h > 0 else float("nan"),
            "p_failure": p_fail, "violations": int(d["violation"].sum()), "mean_I": float(d["I"].mean()),
            "availability": hours / total_h if total_h > 0 else float("nan")}


def integrated_om_study(e: Economics, p: CellPhysics, mm: Optional[MaintenanceModel] = None,
                        weights: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
                        fixed_currents: Sequence[float] = (1.0, 2.0, 4.0),
                        thresholds: Sequence[float] = (0.60, 0.64, 0.68, 0.72, 0.76, 0.80, 0.84),
                        ambient_mean_C: float = 20.0, ambient_amp_C: float = 16.0,
                        plant: Optional[CellPhysics] = None, fast_dt_s: Optional[float] = 60.0,
                        seed: int = 0, progress: ProgressFn = None) -> pd.DataFrame:
    """Joint optimisation of the operating policy and the maintenance (replacement) decision.

    Operating knob: the twin-aware policy's shadow price of SOH, w x replacement cost /
    (1 - 0.70) per unit SOH (w = 0 ignores ageing and maximises the immediate margin; large w
    protects the cell), plus fixed-current baselines. Maintenance knob: the replacement SOH
    threshold. Each policy is simulated once to the lowest threshold (the trajectory does not
    depend on the threshold) and every threshold is evaluated by truncation with
    renewal-reward economics and the sudden-failure hazard. Returns one row per
    (policy, threshold); the maximum 'rate' among compliant rows is the integrated optimum."""
    mm = mm or MaintenanceModel(replacement_cost=e.replacement_cost)
    floor = float(min(thresholds)) - 0.01
    if fast_dt_s:
        p = replace(p, dt_s=float(fast_dt_s))
        plant = replace(plant, dt_s=float(fast_dt_s)) if plant is not None else None
    scale = (1 - floor) / (1 - 0.70)                    # keeps the decision price independent of the floor
    runs: List[Tuple[str, float, Callable]] = [(f"Twin-aware · w = {w:g}", float(w), optimal_policy) for w in weights]
    runs += [(f"Fixed {c:g} A", float("nan"), make_policy(f"Fixed {c:g} A")) for c in fixed_currents]
    rows = []
    for j, (name, w, pol) in enumerate(runs):
        _report(progress, j / len(runs), name)
        e_run = replace(e, soh_eol=floor, degradation_weight=(w if np.isfinite(w) else 1.0) * scale,
                        replacement_cost=mm.replacement_cost)
        life = simulate_life(pol, p, e_run, seed=seed, ambient_mean_C=ambient_mean_C,
                             ambient_amp_C=ambient_amp_C, plant=plant)
        if life.empty:
            continue
        for th in thresholds:
            r = renewal_evaluation(life, float(th), e, mm)
            r.update({"policy": name, "w": w, "kind": "twin" if np.isfinite(w) else "fixed"})
            rows.append(r)
    _report(progress, 1.0, "integrated study done")
    return pd.DataFrame(rows)


def integrated_optimum(study: pd.DataFrame) -> Dict[str, Any]:
    """Best compliant (policy, threshold) and the best of each baseline family."""
    ok = study[(study["violations"] == 0) & study["rate"].notna()]
    if ok.empty:
        return {}
    best = ok.loc[ok["rate"].idxmax()]
    out = {"best": best.to_dict()}
    for fam in ("twin", "fixed"):
        f = ok[ok["kind"] == fam]
        if len(f):
            out[f"best_{fam}"] = f.loc[f["rate"].idxmax()].to_dict()
    rtf = ok[ok["threshold"] == ok["threshold"].min()]
    if len(rtf):
        out["run_to_floor"] = rtf.loc[rtf["rate"].idxmax()].to_dict()
    return out


# =============================================================================
# 19. SOH ESTIMATION FROM HEALTH INDICATORS (diagnosis, train/test splits)
# =============================================================================
ESTIMATION_SPLITS = ("random", "chronological", "by_cell")
# indicators that *are* capacity measurements would make SOH estimation trivial (leakage)
CAPACITY_LEAKS = ("Capacity_Ah", "E_dis_Wh", "t_dis_s", "Q_ch_Ah")
DEFAULT_EST_FEATURES = ("R_dc_ohm", "V_mean_V", "dT_C", "t_cc_s", "t_cv_s", "eff_energy", "T_mean_C", "I_dis_A")


@dataclass
class EstimationResult:
    model: str
    params: Dict[str, Any]
    split: str
    features: List[str]
    predictions: pd.DataFrame        # Cell_ID, n, SOH, SOH_pred, set (train / test)
    metrics: pd.DataFrame            # rows train / test: R², RMSE, MAE, accuracy
    importance: pd.DataFrame         # permutation importance on the test set
    fit_seconds: float
    train_cells: List[str]
    test_cells: List[str]


def estimation_frame(ct: pd.DataFrame, imp: Optional[pd.DataFrame], features: Sequence[str],
                     normalise: bool = True) -> pd.DataFrame:
    """Per-cycle design matrix for SOH estimation. With ``normalise`` each indicator is divided
    by its beginning-of-life value (median of the first 3 valid cycles), which removes
    cell-to-cell offsets so a model can transfer between cells."""
    d = attach_eis(ct, imp)
    d = d[~d["outlier"]].sort_values(["Cell_ID", "n"]).copy()
    feats = [f for f in features if f in d.columns]
    if not feats:
        raise ValueError("None of the selected indicators exist in the cycle table.")
    if normalise:
        for f in feats:
            if f in ("T_mean_C", "I_dis_A", "Ambient_C", "n"):
                continue                                         # operating conditions stay absolute
            base = d.groupby("Cell_ID")[f].transform(lambda x: x.dropna().head(3).median())
            d[f] = d[f] / base.where(base.abs() > 1e-12)
    return d[["Cell_ID", "n", "SOH"] + feats]


def train_soh_estimator(ct: pd.DataFrame, imp: Optional[pd.DataFrame], model_name: str,
                        features: Sequence[str] = DEFAULT_EST_FEATURES, params: Optional[Dict[str, Any]] = None,
                        split: str = "chronological", test_frac: float = 0.3,
                        train_cells: Optional[Sequence[str]] = None, test_cells: Optional[Sequence[str]] = None,
                        normalise: bool = True, seed: int = 0, n_repeats: int = 5) -> EstimationResult:
    """Diagnosis task: estimate the *current* SOH from operando health indicators measured on
    the same cycle (no capacity test needed). Splits:
      random         random test_frac of all cycles (interpolation; optimistic, cycles of the
                     same cell are correlated)
      chronological  per cell, the last test_frac of its life is the test set (extrapolation
                     in time on the same cells)
      by_cell        train on train_cells, test on test_cells (transfer to unseen batteries)
    Capacity-derived indicators are refused because they would leak the target."""
    from sklearn.metrics import mean_absolute_error, r2_score

    if split not in ESTIMATION_SPLITS:
        raise ValueError(f"split must be one of {ESTIMATION_SPLITS}")
    leaks = [f for f in features if f in CAPACITY_LEAKS]
    if leaks:
        raise ValueError(f"These indicators measure capacity directly and would leak SOH: {leaks}")
    if not 0.05 <= test_frac <= 0.9:
        raise ValueError("test fraction must be between 0.05 and 0.9")
    params = validate_params(model_name, params)
    d = estimation_frame(ct, imp, features, normalise)
    feats = [c for c in d.columns if c not in ("Cell_ID", "n", "SOH")]
    d = d.dropna(subset=["SOH"])
    d = d[d[feats].notna().mean(axis=1) >= 0.5]                 # need at least half the indicators
    rng = np.random.default_rng(seed)
    if split == "random":
        is_test = rng.random(len(d)) < test_frac
    elif split == "chronological":
        rank = d.groupby("Cell_ID")["n"].rank(pct=True)
        is_test = (rank > 1 - test_frac).to_numpy()
    else:
        cells = sorted(d["Cell_ID"].unique())
        test_cells = [c for c in (test_cells or []) if c in cells]
        train_cells = [c for c in (train_cells or [c for c in cells if c not in test_cells]) if c in cells]
        if not test_cells or not train_cells or set(test_cells) & set(train_cells):
            raise ValueError("by_cell split needs disjoint, non-empty train and test cell lists")
        d = d[d["Cell_ID"].isin(train_cells + test_cells)]
        is_test = d["Cell_ID"].isin(test_cells).to_numpy()
    tr, te_ = d[~is_test], d[is_test]
    if len(tr) < 10 or len(te_) < 3:
        raise ValueError("Split leaves too few training (< 10) or test (< 3) cycles.")
    X_tr, y_tr = tr[feats].to_numpy(float), tr["SOH"].to_numpy(float)
    X_te, y_te = te_[feats].to_numpy(float), te_["SOH"].to_numpy(float)
    X_fit, y_fit, _ = _gp_subsample(model_name, X_tr, y_tr, np.ones(len(y_tr)), seed)
    t0 = time.time()
    model = _StandardisedTarget(make_model(model_name, seed, params)).fit(X_fit, y_fit)
    fit_s = time.time() - t0
    p_tr, p_te = model.predict(X_tr), model.predict(X_te)

    def row(y, p):
        return {"R²": float(r2_score(y, p)) if len(y) > 1 and np.var(y) > 0 else float("nan"),
                "RMSE": float(np.sqrt(np.mean((p - y) ** 2))), "MAE": float(mean_absolute_error(y, p)),
                "Accuracy (%)": float(100 * (1 - np.mean(np.abs(p - y) / np.maximum(np.abs(y), 1e-9)))),
                "Cycles": int(len(y))}

    metrics = pd.DataFrame({"train": row(y_tr, p_tr), "test": row(y_te, p_te)}).T

    # permutation importance on the test set: RMSE increase when one indicator is shuffled
    base_rmse = float(np.sqrt(np.mean((p_te - y_te) ** 2)))
    imp_mean, imp_std = [], []
    for j in range(X_te.shape[1]):
        deltas = []
        for _ in range(n_repeats):
            Xp = X_te.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            deltas.append(float(np.sqrt(np.mean((model.predict(Xp) - y_te) ** 2))) - base_rmse)
        imp_mean.append(float(np.mean(deltas)))
        imp_std.append(float(np.std(deltas)))
    importance = pd.DataFrame({"Indicator": [HI_CATALOG[f].label if f in HI_CATALOG else f for f in feats],
                               "key": feats, "Importance (ΔRMSE)": imp_mean,
                               "std": imp_std}).sort_values("Importance (ΔRMSE)", ascending=False)
    preds = pd.concat([tr.assign(SOH_pred=p_tr, set="train"), te_.assign(SOH_pred=p_te, set="test")])
    return EstimationResult(model_name, params, split, feats, preds[["Cell_ID", "n", "SOH", "SOH_pred", "set"]],
                            metrics, importance.reset_index(drop=True), fit_s,
                            sorted(tr["Cell_ID"].unique()), sorted(te_["Cell_ID"].unique()))
