"""
app.py - Battery Digital Twin · Operando Diagnostics Dashboard (Streamlit)
Run locally:  streamlit run app.py
"""

from __future__ import annotations

import os
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

import twin_engine as te

st.set_page_config(page_title="Battery Digital Twin · Operando Diagnostics", page_icon="🔋",
                   layout="wide", initial_sidebar_state="expanded")

# =============================================================================
# Constants & styling
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MASTER_NAME, IMP_NAME = "battery_master_data.parquet", "impedance_ground_truth.parquet"

C = {  # colour system
    "bg": "#0b1120", "panel": "#0f172a", "grid": "#334155", "text": "#f1f5f9", "muted": "#cbd5e1",
    "accent": "#22d3ee", "measured": "#f8fafc", "ekf": "#38bdf8", "pinn": "#f472b6", "ml": "#a3e635",
    "eis": "#fbbf24", "eol": "#f87171", "r_int": "#fb923c", "r_ct": "#34d399",
    1.0: "#60a5fa", 2.0: "#34d399", 4.0: "#f87171",
}
PARADIGM_COLORS = {"ECM + EKF": C["ekf"], "Hybrid PINN": C["pinn"]}
POLICIES = ["Twin-Aware", "Fixed 1 A", "Fixed 2 A", "Fixed 4 A"]

st.markdown(f"""
<style>
  .block-container {{padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px;}}
  html, body, [class*="css"] {{font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; color: {C['text']};}}
  .hero {{background: linear-gradient(120deg, #0f172a 0%, #0b2a3a 55%, #134e4a 100%);
         border: 1px solid {C['grid']}; border-radius: 16px; padding: 22px 28px; margin-bottom: 1rem;}}
  .hero h1 {{font-size: 1.65rem; margin: 0; color: #ffffff; font-weight: 700; letter-spacing: -0.01em;}}
  .hero p {{margin: 6px 0 0 0; color: {C['muted']}; font-size: 0.95rem;}}
  .pill {{display: inline-block; padding: 2px 10px; margin: 8px 6px 0 0; border-radius: 999px;
         background: rgba(34,211,238,0.12); color: {C['accent']}; font-size: 0.75rem;
         border: 1px solid rgba(34,211,238,0.35); font-weight: 600;}}
  div[data-testid="stMetric"] {{background: {C['panel']}; border: 1px solid {C['grid']};
         border-radius: 12px; padding: 12px 16px;}}
  div[data-testid="stMetricLabel"] p {{color: {C['muted']}; font-size: 0.8rem; text-transform: uppercase;
         letter-spacing: 0.04em; font-weight: 600;}}
  div[data-testid="stMetricValue"] {{color: #ffffff; font-weight: 700;}}
  .card {{background: {C['panel']}; border: 1px solid {C['grid']}; border-radius: 12px;
         padding: 14px 18px; margin: 4px 0 12px 0;}}
  .card h4 {{margin: 0 0 6px 0; font-size: 0.9rem; color: {C['accent']}; text-transform: uppercase;
             letter-spacing: 0.05em; font-weight: 700;}}
  .card p, .card li {{color: {C['text']}; font-size: 0.9rem; margin: 2px 0;}}
  .section {{font-size: 1.1rem; font-weight: 700; color: #ffffff; margin: 1.2rem 0 0.4rem 0;
             border-left: 4px solid {C['accent']}; padding-left: 10px;}}
  button[data-baseweb="tab"] {{font-size: 0.95rem; font-weight: 600;}}
  section[data-testid="stSidebar"] {{border-right: 1px solid {C['grid']};}}
</style>
""", unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)


def card(title: str, lines: List[str]) -> None:
    body = "".join(f"<li>{ln}</li>" for ln in lines)
    st.markdown(f'<div class="card"><h4>{title}</h4><ul style="margin:0;padding-left:18px">{body}</ul></div>',
                unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 420, title: Optional[str] = None) -> go.Figure:
    """Applies crisp, high-contrast, publication-grade styling to all figures."""
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C["panel"],
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=C["text"]),
        margin=dict(l=65, r=24, t=50 if title else 20, b=50),
        title=dict(text=title, x=0.01, font=dict(size=15, color="#ffffff", family="Inter, sans-serif")) if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, 
                    bgcolor="rgba(15,23,42,0.8)", bordercolor=C["grid"], borderwidth=1,
                    font=dict(color=C["text"], size=11)),
        hovermode="x unified"
    )
    fig.update_xaxes(gridcolor=C["grid"], zerolinecolor=C["grid"], title_font=dict(size=13, color="#ffffff"), tickfont=dict(color=C["muted"]))
    fig.update_yaxes(gridcolor=C["grid"], zerolinecolor=C["grid"], title_font=dict(size=13, color="#ffffff"), tickfont=dict(color=C["muted"]))
    return fig


def _st_version() -> Tuple[int, int]:
    try:
        major, minor = st.__version__.split(".")[:2]
        return int(major), int(minor)
    except Exception:
        return (1, 0)


def show(fig: go.Figure) -> None:
    if _st_version() >= (1, 50):
        st.plotly_chart(fig, theme=None, width="stretch")
    else:
        st.plotly_chart(fig, theme=None, use_container_width=True)


def fail(where: str, exc: BaseException) -> None:
    st.error(f"{where}: {exc}")
    with st.expander("Technical details"):
        st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:])


def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "")) or os.environ.get(name, "")
    except Exception:
        return os.environ.get(name, "")


# =============================================================================
# Cached data layer
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_store(path: str, mtime: float) -> te.ParquetStore:
    return te.ParquetStore(path)


@st.cache_resource(show_spinner=False)
def persist_upload(name: str, size: int, _raw: bytes) -> str:
    return str(te.persist_upload(_raw, name))


@st.cache_data(show_spinner=False)
def load_impedance_path(path: str, mtime: float) -> pd.DataFrame:
    return te.load_impedance(path)


@st.cache_data(show_spinner=False, max_entries=16)
def cycle_table(_store: te.ParquetStore, store_key: str) -> pd.DataFrame:
    return te.build_cycle_table(_store)


@st.cache_resource(show_spinner=False, max_entries=3)
def cell_frame(_store: te.ParquetStore, store_key: str, cell: str) -> pd.DataFrame:
    return _store.cell_frame(cell)


@st.cache_resource(show_spinner=False, max_entries=3)
def prepared_cell(_store: te.ParquetStore, store_key: str, cell: str) -> pd.DataFrame:
    return te.prepare_cell(cell_frame(_store, store_key, cell))


@st.cache_data(show_spinner=False, max_entries=32)
def ica_cached(_prep: pd.DataFrame, _ct_cell: pd.DataFrame, key: str, cell: str,
               n_curves: int, ir: bool, dv: float, window: int) -> List[te.ICACurve]:
    return te.ica_evolution(_prep, _ct_cell, n_curves, ir_compensate=ir, dv=dv, smooth_window=window)


@st.cache_data(show_spinner=False, max_entries=64)
def ml_cached(_ct: pd.DataFrame, key: str, cell: str, n0: int, model: str,
              use_pop: bool, eol_ah: float) -> te.MLForecast:
    return te.train_ml_forecast(_ct, cell, n0, model, use_population=use_pop, eol_ah=eol_ah)


@st.cache_data(show_spinner=False, max_entries=64)
def run_policy(policy: str, econ: Dict[str, Any], phys: Dict[str, Any], amb_mean: float, amb_amp: float) -> pd.DataFrame:
    return te.simulate_life(te.make_policy(policy), te.CellPhysics(**phys), te.Economics(**econ),
                            ambient_mean_C=amb_mean, ambient_amp_C=amb_amp)


def resolve_sources(up_master, up_imp, master_url: str, imp_url: str
                    ) -> Tuple[Optional[te.ParquetStore], Optional[pd.DataFrame], List[str]]:
    notes: List[str] = []
    store, imp = None, None

    master_path: Optional[Path] = None
    if up_master is not None:
        master_path = Path(persist_upload(up_master.name, up_master.size, up_master.getvalue()))
        notes.append(f"Telemetry: uploaded `{up_master.name}`")
    else:
        for cand in (DATA_DIR / MASTER_NAME, APP_DIR / MASTER_NAME):
            if cand.exists():
                master_path = cand
                notes.append(f"Telemetry: local `{cand.relative_to(APP_DIR)}`")
                break
    if master_path is None and master_url:
        bar = st.sidebar.progress(0.0, text="Downloading telemetry…")
        master_path = te.download_to_cache(master_url, progress=lambda f, m: bar.progress(f, text=f"Telemetry {m}"))
        bar.empty()
        notes.append("Telemetry: remote URL (cached on disk)")
    if master_path is not None:
        store = get_store(str(master_path), master_path.stat().st_mtime)

    if up_imp is not None:
        imp = te.load_impedance(up_imp.getvalue())
        notes.append(f"EIS: uploaded `{up_imp.name}`")
    else:
        imp_path = next((p for p in (DATA_DIR / IMP_NAME, APP_DIR / IMP_NAME) if p.exists()), None)
        if imp_path is None and imp_url:
            imp_path = te.download_to_cache(imp_url)
        if imp_path is not None:
            imp = load_impedance_path(str(imp_path), imp_path.stat().st_mtime)
            notes.append("EIS: loaded")
    return store, imp, notes


# =============================================================================
# Sidebar
# =============================================================================
st.markdown("""
<div class="hero">
  <h1>🔋 Self-Updating Battery Digital Twin</h1>
  <p>NASA Ames 18650 (LiCoO₂ | graphite) aging data · incremental capacity diagnostics ·
     ECM + Extended Kalman Filter · physics-informed neural network · twin-aware operation</p>
  <span class="pill">Butler–Volmer kinetics</span><span class="pill">Arrhenius / SEI fade law</span>
  <span class="pill">dQ/dV · LLI / LAM</span><span class="pill">RUL forecasting</span>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Data sources")
    with st.expander("Upload / remote files", expanded=False):
        up_master = st.file_uploader("Master telemetry (.parquet)", type=["parquet"], key="up_master")
        up_imp = st.file_uploader("EIS ground truth (.parquet)", type=["parquet"], key="up_imp")
        master_url = st.text_input("Telemetry URL (GitHub Release asset)", value=_secret("MASTER_PARQUET_URL"))
        imp_url = st.text_input("EIS URL", value=_secret("IMPEDANCE_PARQUET_URL"))

store: Optional[te.ParquetStore] = None
imp: Optional[pd.DataFrame] = None
ct: Optional[pd.DataFrame] = None
try:
    store, imp, source_notes = resolve_sources(up_master, up_imp, master_url.strip(), imp_url.strip())
    if store is not None:
        with st.spinner("Extracting per-cycle features (one-off, cached)…"):
            ct = cycle_table(store, store.key)
except Exception as exc:
    source_notes = []
    fail("Data loading failed", exc)

if store is None or ct is None:
    st.info(f"No telemetry loaded. Place `{MASTER_NAME}` (and `{IMP_NAME}`) in `./data/`, upload them "
            "in the sidebar, or provide a remote URL (`MASTER_PARQUET_URL` secret / environment variable).")
    st.stop()

meta = te.cell_meta(ct)
cells = list(meta.index)
with st.sidebar:
    st.caption(" · ".join(source_notes))
    st.markdown("### Cell under test")
    cell = st.selectbox("Cell ID", cells, index=cells.index("B0005") if "B0005" in cells else 0)
    eol_ah = st.number_input("End-of-life threshold (Ah)", 0.8, 2.0, te.DEFAULT_EOL_AH, 0.05,
                             help="NASA: 1.4 Ah (30 % fade); sets B33–B40 stopped at 1.6 Ah.")

    st.markdown("### EKF observer")
    sigma_v = st.slider("σᵥ — voltage model error (V)", 0.005, 0.200, 0.080, 0.005, format="%.3f")
    Q_SOH = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    Q_R = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
    q_soh = st.select_slider("q_SOH — SOH random walk (/√Ah)", Q_SOH, value=5e-4, format_func=lambda v: f"{v:.0e}")
    q_r = st.select_slider("q_R — resistance random walk (R₀/√Ah)", Q_R, value=2e-3, format_func=lambda v: f"{v:.0e}")
    tau = st.slider("τ — RC time constant (s)", 10, 300, 60, 10)

    st.markdown("### Hybrid PINN")
    epochs = st.select_slider("Training epochs", [500, 1000, 1500, 2500, 4000], value=2500)
    lam_phys = st.select_slider("λ physics (fade + resistance kinetics)", [0.0, 0.1, 0.3, 1.0, 3.0, 10.0], value=1.0)
    lam_bv = st.select_slider("λ Butler–Volmer", [0.0, 0.1, 0.3, 1.0, 3.0], value=0.3)
    lam_eis = st.select_slider("λ EIS anchors", [0.0, 0.1, 0.5, 1.0, 3.0], value=0.5)
    debug = st.toggle("Show technical errors", value=False)

twin_params = te.TwinParameters(sigma_v=sigma_v, q_soh_per_ah=q_soh, q_r_frac_per_ah=q_r, tau_rc_s=float(tau))
pinn_cfg = te.PINNConfig(epochs=int(epochs), lambda_phys=float(lam_phys), lambda_bv=float(lam_bv),
                         lambda_eis=float(lam_eis))
ct_cell = ct[ct["Cell_ID"] == cell].sort_values("n")
c_bol = float(meta.loc[cell, "C_bol_Ah"])
soh_eol = te.soh_eol_for(c_bol, eol_ah)
eis_cell = te.valid_eis(imp, cell)

tab_eda, tab_twin, tab_ops = st.tabs(["①  EDA · Diagnostics · ML Forecasting",
                                     "②  EKF Observer vs Hybrid PINN",
                                     "③  Twin-Aware Operation & Economics"])


# =============================================================================
# Figures
# =============================================================================
def fig_fade(ct_all: pd.DataFrame, cell_id: str, y: str, eol_line: Optional[float]) -> go.Figure:
    fig = go.Figure()
    for cid, d in ct_all[~ct_all["outlier"]].groupby("Cell_ID"):
        if cid == cell_id:
            continue
        fig.add_trace(go.Scatter(x=d["n"], y=d[y], mode="lines", line=dict(color="#475569", width=1.2),
                                 name="Other cells", legendgroup="others", showlegend=False,
                                 hovertemplate=f"{cid}<br>n=%{{x}}<br>{y}=%{{y:.3f}}<extra></extra>"))
    d = ct_all[ct_all["Cell_ID"] == cell_id]
    good, bad = d[~d["outlier"]], d[d["outlier"]]
    fig.add_trace(go.Scatter(x=good["n"], y=good[y], mode="lines+markers", name=cell_id,
                             line=dict(color=C["accent"], width=3), marker=dict(size=5, color=C["accent"])))
    if len(bad):
        fig.add_trace(go.Scatter(x=bad["n"], y=bad[y], mode="markers", name="flagged outliers",
                                 marker=dict(symbol="x", size=9, color=C["eol"], line=dict(width=2))))
    if eol_line is not None:
        fig.add_hline(y=eol_line, line_dash="dash", line_color=C["eol"], annotation_text="EOL Threshold",
                      annotation_position="bottom right", annotation_font=dict(color=C["eol"], size=12))
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="Capacity (Ah)" if y == "Capacity_Ah" else "SOH (–)")
    return style_fig(fig, 430, "Capacity-Fade Trajectory (All Cohort Cells in Grey)")


def fig_ica(curves: List[te.ICACurve]) -> go.Figure:
    fig = go.Figure()
    cols = sample_colorscale("Viridis", list(np.linspace(0.1, 0.95, max(len(curves), 2))))
    for c, col in zip(curves, cols):
        fig.add_trace(go.Scatter(x=c.voltage, y=c.dqdv, mode="lines", name=f"n = {c.n}",
                                 line=dict(color=col, width=2.5)))
        fig.add_trace(go.Scatter(x=[c.peak_V], y=[c.peak_height], mode="markers", showlegend=False,
                                 marker=dict(color=col, size=10, line=dict(color="white", width=1.5)),
                                 hovertemplate="peak %{x:.3f} V<extra></extra>"))
    fig.update_xaxes(title_text="Voltage (V)", autorange="reversed")
    fig.update_yaxes(title_text="dQ/dV (Ah V⁻¹)")
    fig.update_layout(hovermode="closest")
    return style_fig(fig, 430, "Incremental Capacity Evolution (dQ/dV)")


def fig_peaks(curves: List[te.ICACurve]) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    n = [c.n for c in curves]
    fig.add_trace(go.Scatter(x=n, y=[c.peak_V for c in curves], mode="lines+markers", name="Peak voltage",
                             line=dict(color=C["accent"], width=2.5), marker=dict(size=6)), secondary_y=False)
    fig.add_trace(go.Scatter(x=n, y=[c.peak_height for c in curves], mode="lines+markers", name="Peak height",
                             line=dict(color=C["pinn"], width=2.5), marker=dict(size=6)), secondary_y=True)
    fig.update_yaxes(title_text="Peak V (V)", secondary_y=False, title_font=dict(color=C["accent"]))
    fig.update_yaxes(title_text="Peak dQ/dV", secondary_y=True, showgrid=False, title_font=dict(color=C["pinn"]))
    fig.update_xaxes(title_text="Discharge cycle n")
    return style_fig(fig, 300, "Main-Peak Shift & Amplitude")


def _forecast_frame(fig: go.Figure, n0: int, n_max: float, soh_eol_: float, row: Optional[int] = None) -> None:
    kw = dict(row=row, col=1) if row else {}
    fig.add_vrect(x0=n0, x1=n_max, fillcolor="rgba(148,163,184,0.08)", line_width=0, layer="below", **kw)
    fig.add_vline(x=n0, line_dash="dot", line_color=C["muted"], **kw)
    fig.add_hline(y=soh_eol_, line_dash="dash", line_color=C["eol"], **kw)


def fig_ml(results: List[te.MLForecast], ct_cell_: pd.DataFrame, n0: int, soh_eol_: float) -> go.Figure:
    fig = go.Figure()
    good = ct_cell_[~ct_cell_["outlier"]]
    fig.add_trace(go.Scatter(x=good["n"], y=good["SOH"], mode="markers", name="Measured SOH",
                             marker=dict(color=C["measured"], size=6, opacity=0.8)))
    palette = sample_colorscale("Turbo", list(np.linspace(0.15, 0.85, max(len(results), 2))))
    for r, col in zip(results, palette):
        fig.add_trace(go.Scatter(x=r.n_grid, y=r.soh_pred, mode="lines", name=r.model, line=dict(color=col, width=2.5)))
    n_max = max(float(r.n_grid.max()) for r in results)
    _forecast_frame(fig, n0, n_max, soh_eol_)
    fig.add_annotation(x=n0, y=1.0, text="Forecast Origin (n₀)", showarrow=False, xanchor="left",
                       font=dict(color=C["accent"], size=12))
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="State of Health (SOH)")
    return style_fig(fig, 460, "Machine Learning Surrogate Forecasts (Held-out Future Shaded)")


def fig_compare(res: te.ComparisonResult) -> go.Figure:
    fig = go.Figure()
    m = res.measured
    fig.add_trace(go.Scatter(x=m["n"], y=m["SOH"], mode="markers", name="Measured SOH",
                             marker=dict(color=C["measured"], size=6)))
    if res.ekf is not None:
        e = res.ekf.per_cycle.dropna(subset=["SOH"])
        fig.add_trace(go.Scatter(x=np.r_[e["n"], e["n"][::-1]],
                                 y=np.r_[e["SOH"] + 2 * e["SOH_std"], (e["SOH"] - 2 * e["SOH_std"])[::-1]],
                                 fill="toself", fillcolor="rgba(56,189,248,0.15)", line=dict(width=0),
                                 name="EKF ±2σ Uncertainty", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=e["n"], y=e["SOH"], mode="lines", name="EKF Estimate (Causal)",
                                 line=dict(color=C["ekf"], width=2.5)))
    for name, (n_g, p_g) in res.predictions.items():
        col = PARADIGM_COLORS.get(name, C["ml"])
        mask = n_g >= res.n0
        fig.add_trace(go.Scatter(x=n_g[mask], y=p_g[mask], mode="lines", name=f"{name} Forecast",
                                 line=dict(color=col, width=3, dash="dash" if name == "ECM + EKF" else "solid")))
        if name == "Hybrid PINN":
            fig.add_trace(go.Scatter(x=n_g[~mask], y=p_g[~mask], mode="lines", name="PINN Fitted Training Zone",
                                     line=dict(color=col, width=2, dash="dot")))
    n_max = max(float(n.max()) for n, _ in res.predictions.values()) if res.predictions else float(m["n"].max())
    _forecast_frame(fig, res.n0, n_max, res.soh_eol)
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="State of Health (SOH)")
    return style_fig(fig, 480, f"Cell {res.cell_id}: Multi-Paradigm SOH Tracking & Future Horizon Prediction (n₀ = {res.n0})")


def fig_params(res: te.ComparisonResult) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Ohmic Resistance (R_int)", "Charge-Transfer Resistance (R_ct)"))
    for row, key, eis_col, col in ((1, "R_int", "Re_ohm", C["r_int"]), (2, "R_ct", "Rct_ohm", C["r_ct"])):
        if res.ekf is not None:
            e = res.ekf.per_cycle.dropna(subset=[key])
            fig.add_trace(go.Scatter(x=e["n"], y=1e3 * e[key], mode="lines", name=f"EKF {key}",
                                     line=dict(color=C["ekf"], width=2.5)), row=row, col=1)
        if res.pinn is not None:
            fig.add_trace(go.Scatter(x=res.pinn.n_grid, y=1e3 * getattr(res.pinn, key.lower()), mode="lines",
                                     name=f"PINN {key}", line=dict(color=C["pinn"], width=2.5)), row=row, col=1)
        if len(res.eis):
            ci = res.measured.sort_values("Cycle_Index")[["Cycle_Index", "n"]]
            ee = pd.merge_asof(res.eis.sort_values("Cycle_Index"), ci, on="Cycle_Index", direction="backward")
            fig.add_trace(go.Scatter(x=ee["n"].fillna(1), y=1e3 * ee[eis_col], mode="markers",
                                     name=f"EIS Ground Truth ({eis_col.split('_')[0]})",
                                     marker=dict(color=C["eis"], symbol="x", size=10, line=dict(width=1.5))), row=row, col=1)
        fig.add_vline(x=res.n0, line_dash="dot", line_color=C["muted"], row=row, col=1)
        fig.update_yaxes(title_text="Resistance (mΩ)", row=row, col=1)
    fig.update_xaxes(title_text="Discharge cycle n", row=2, col=1)
    return style_fig(fig, 520, "Internal Parameter Convergence vs. Electrochemical EIS Ground Truth")


def fig_innovation(ekf: te.EKFResult, sigma_v_: float) -> go.Figure:
    e = ekf.per_cycle.dropna(subset=["innov_mean_mV"])
    fig = go.Figure()
    fig.add_hrect(y0=-2e3 * sigma_v_, y1=2e3 * sigma_v_, fillcolor="rgba(52,211,153,0.1)", line_width=0)
    fig.add_trace(go.Scatter(x=np.r_[e["n"], e["n"][::-1]],
                             y=np.r_[e["innov_mean_mV"] + e["innov_std_mV"], (e["innov_mean_mV"] - e["innov_std_mV"])[::-1]],
                             fill="toself", fillcolor="rgba(56,189,248,0.2)", line=dict(width=0),
                             name="±1 s.d. within cycle", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=e["n"], y=e["innov_mean_mV"], mode="lines", name="Mean Voltage Innovation",
                             line=dict(color=C["ekf"], width=2.5)))
    fig.add_hline(y=0, line_color=C["muted"], line_dash="dash")
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="Voltage Residual (V_meas − V_pred) [mV]")
    return style_fig(fig, 340, "EKF Voltage Innovation & Consistency Validation (±2σᵥ Band)")


def fig_pinn_loss(p: te.PINNResult) -> go.Figure:
    fig = go.Figure()
    for key, col in (("total", C["text"]), ("data", C["measured"]), ("phys", C["pinn"]),
                     ("bv", C["accent"]), ("eis", C["eis"])):
        if key in p.history.columns:
            fig.add_trace(go.Scatter(x=p.history["epoch"], y=p.history[key], mode="lines", name=f"Loss: {key}",
                                     line=dict(color=col, width=2.5 if key == "total" else 1.5)))
    fig.update_yaxes(type="log", title_text="Loss Value (Normalized Log Scale)")
    fig.update_xaxes(title_text="Training Epoch")
    return style_fig(fig, 340, "PINN Multi-Component Optimization Loss Curves")


def fig_policy(df: pd.DataFrame, policy: str, baselines: Dict[str, pd.DataFrame], soh_eol_ops: float) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.28, 0.32, 0.4],
                        specs=[[{"secondary_y": True}], [{}], [{}]],
                        subplot_titles=("Applied Current & Ambient Thermal Profile", "Battery State of Health (SOH)", "Cumulative Economic Profit"))
    for I in (1.0, 2.0, 4.0):
        mm = df["I"] == I
        if mm.any():
            fig.add_trace(go.Scatter(x=df.loc[mm, "cycle"], y=df.loc[mm, "I"], mode="markers", name=f"{I:g} A Rate",
                                     marker=dict(color=C[I], size=7)), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df["cycle"], y=df["T_amb"], mode="lines", name="Ambient Temperature",
                             line=dict(color=C["muted"], width=1.5, dash="dot")), row=1, col=1, secondary_y=True)
    for name, d in {policy: df, **baselines}.items():
        is_main = name == policy
        col = "#ffffff" if is_main else C.get(float(name.split()[1]), C["muted"])
        style = dict(color=col, width=3 if is_main else 1.5, dash=None if is_main else "dash")
        fig.add_trace(go.Scatter(x=d["cycle"], y=d["SOH"], mode="lines", name=name, line=style,
                                 legendgroup=name), row=2, col=1)
        fig.add_trace(go.Scatter(x=d["cycle"], y=d["cum_profit"], mode="lines", name=name, line=style,
                                 legendgroup=name, showlegend=False), row=3, col=1)
    fig.add_hline(y=soh_eol_ops, line_dash="dash", line_color=C["eol"], row=2, col=1)
    fig.update_yaxes(title_text="Current (A)", tickvals=[1, 2, 4], row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Temp (°C)", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="SOH", row=2, col=1)
    fig.update_yaxes(title_text="Profit (CU)", row=3, col=1)
    fig.update_xaxes(title_text="Operational Cycle Index", row=3, col=1)
    return style_fig(fig, 740, "Twin-Aware Lifecycle Operation & Economic Yield Optimization")


# =============================================================================
# Tab 1 · EDA, diagnostics, ML forecasting
# =============================================================================
with tab_eda:
    try:
        k = st.columns(6)
        k[0].metric("Cell ID", cell)
        k[1].metric("C_bol Capacity", f"{c_bol:.3f} Ah")
        k[2].metric("Total Cycles", int(meta.loc[cell, "cycles"]))
        k[3].metric("Capacity Fade", f"{meta.loc[cell, 'fade_pct']:.1f} %")
        k[4].metric("Ambient · I_dis", f"{meta.loc[cell, 'Ambient_C']:.0f} °C · {meta.loc[cell, 'I_dis_A']:.1f} A")
        k[5].metric("Valid EIS Sweeps", len(eis_cell))

        section("Capacity Degradation Analysis")
        yvar = st.radio("Y-axis Metric", ["SOH", "Capacity_Ah"], horizontal=True, label_visibility="collapsed")
        show(fig_fade(ct, cell, yvar, soh_eol if yvar == "SOH" else eol_ah))

        section("Incremental Capacity Analysis (dQ/dV Diagnostics)")
        c1, c2, c3, c4 = st.columns(4)
        n_curves = c1.slider("Curves across life", 2, 12, 6)
        ir = c2.toggle("IR-compensate voltage", value=True,
                       help="V + |I|·R_dc with load-step resistance: removes ohmic shift so remaining "
                            "peak shifts isolate non-ohmic polarization and thermodynamic changes.")
        dv = c3.select_slider("Voltage bin (mV)", [5, 10, 15, 20], value=10) / 1000
        win = c4.select_slider("Smoothing window (bins)", [5, 7, 9, 11, 15, 21], value=9)
        with st.spinner("Computing differential capacity dQ/dV curves…"):
            prep = prepared_cell(store, store.key, cell)
            curves = ica_cached(prep, ct_cell, store.key, cell, n_curves, ir, dv, win)
        if not curves:
            st.warning("No discharge cycle long enough for robust differential capacity processing.")
        else:
            a, b = st.columns([3, 2])
            with a:
                show(fig_ica(curves))
            with b:
                show(fig_peaks(curves))
                diag = te.diagnose_degradation(curves)
                if diag["available"]:
                    card(f"Indicative Diagnostic · n={diag['n_ref']} → n={diag['n_cur']}",
                         [f"Capacity ratio {diag['cap_ratio']:.2f} · peak-height ratio {diag['height_ratio']:.2f} "
                          f"· peak shift {diag['shift_mV']:.0f} mV"] + diag["interpretation"])
            st.caption("NASA life-cycling runs at ~1C, causing peak broadening from overpotential. "
                       "Precise quantitative mode separation (LLI vs LAM) requires slow C/20 testing protocols.")

        section("Machine-Learning Surrogates · SOH & RUL Forecasting")
        c1, c2, c3 = st.columns([3, 2, 2])
        models = c1.multiselect("Select ML Paradigms", list(te.ML_MODELS), default=list(te.ML_MODELS))
        frac = c2.slider("Forecast origin (fraction of life observed)", 0.2, 0.8, 0.4, 0.05, key="ml_frac")
        use_pop = c3.toggle("Train on cohort population", value=True,
                            help="Leverage multi-cell aging trends combined with target cell early history.")
        n0_ml = int(max(5, round(frac * meta.loc[cell, "cycles"])))
        if st.button("▶  Train & Evaluate ML Models", type="primary", key="ml_go"):
            prog = st.progress(0.0)
            out: List[te.MLForecast] = []
            for i, mname in enumerate(models):
                prog.progress(i / max(len(models), 1), text=f"Training {mname}…")
                try:
                    out.append(ml_cached(ct, store.key, cell, n0_ml, mname, use_pop, float(eol_ah)))
                except Exception as exc:
                    st.warning(f"{mname}: {exc}")
            prog.empty()
            st.session_state["ml"] = {"cfg": (cell, n0_ml, tuple(models), use_pop, eol_ah), "res": out}
        saved = st.session_state.get("ml")
        if saved and saved["res"]:
            if saved["cfg"][0] != cell:
                st.info("Displayed results correspond to a different cell; click Train & Evaluate to update.")
            else:
                show(fig_ml(saved["res"], ct_cell, saved["cfg"][1], soh_eol))
                tbl = pd.DataFrame([{"Model": r.model, "RMSE": r.metrics.rmse, "MAE": r.metrics.mae,
                                     "R²": r.metrics.r2, "RUL true": r.metrics.rul_true,
                                     "RUL pred": r.metrics.rul_pred, "RUL error": r.metrics.rul_error,
                                     "Train rows": r.train_rows, "Fit time (s)": r.fit_seconds}
                                    for r in saved["res"]]).set_index("Model")
                st.dataframe(tbl.style.format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.3f}", "Fit time (s)": "{:.2f}"},
                                            na_rep="—").highlight_min(subset=["RMSE"], color="#134e4a"))
                st.caption("Inputs are strictly causal operating parameters (cycle count, thermal profile, discharge current, cut-off limit) "
                           "ensuring no data leakage across the prediction horizon.")
    except Exception as exc:
        fail("Tab ① error", exc) if debug else st.error(f"Tab ① error: {exc}")


# =============================================================================
# Tab 2 · EKF vs PINN (and ML) comparison
# =============================================================================
with tab_twin:
    try:
        st.markdown("Benchmarking parallel forecasting paradigms from identical information up to **n₀**: "
                    "**Pure Machine Learning**, **ECM + Extended Kalman Filter** (causal observer feedback), and the "
                    "**Hybrid Physics-Informed Neural Network (PINN)** fusing data-driven updates with governing electrochemical laws.")
        c1, c2, c3 = st.columns([2, 2, 2])
        frac2 = c1.slider("Forecast origin n₀ (fraction of life)", 0.2, 0.8, 0.4, 0.05, key="cmp_frac")
        ml_pick = c2.selectbox("Benchmark ML Model", list(te.ML_MODELS), index=1)
        reuse = c3.toggle("Reuse cached EKF replay", value=True,
                          help="The EKF filter is strictly causal; state estimations at n₀ do not depend on future data.")
        ekf_key = (store.key, cell, tuple(sorted(asdict(twin_params).items())))
        cmp_cfg = (ekf_key, frac2, ml_pick, tuple(sorted(asdict(pinn_cfg).items())), eol_ah)

        if st.button("▶  Run Digital Twin Comparative Analysis", type="primary", key="cmp_go"):
            bar = st.progress(0.0, text="Initializing simulation…")
            try:
                ekf_cached = st.session_state.get("ekf")
                ekf_res = ekf_cached["res"] if (reuse and ekf_cached and ekf_cached["key"] == ekf_key) else None
                raw = cell_frame(store, store.key, cell)
                res = te.compare_paradigms(raw, ct, imp, cell, frac2, twin_params, pinn_cfg, ml_pick, eol_ah,
                                           ekf=ekf_res, progress=lambda f, m: bar.progress(f, text=m))
                if res.ekf is not None:
                    st.session_state["ekf"] = {"key": ekf_key, "res": res.ekf}
                st.session_state["cmp"] = {"cfg": cmp_cfg, "res": res}
            except Exception as exc:
                fail("Comparison failed", exc)
            bar.empty()

        saved = st.session_state.get("cmp")
        if saved:
            res: te.ComparisonResult = saved["res"]
            if saved["cfg"] != cmp_cfg:
                st.warning("Parameters modified since last execution; displaying previous session results.")
            for name, msg in res.errors.items():
                st.warning(f"{name} encountered an issue: {msg}")
            cols = st.columns(max(len(res.metrics), 1))
            for col, (name, m) in zip(cols, res.metrics.items()):
                rul = "—" if m.rul_pred is None else f"{m.rul_pred}"
                delta = None if m.rul_error is None else f"RUL error {m.rul_error:+d} cycles"
                col.metric(f"{name} · RMSE", f"{m.rmse:.4f}", delta, delta_color="off")
            show(fig_compare(res))

            a, b = st.columns(2)
            with a:
                show(fig_params(res))
            with b:
                if res.ekf is not None:
                    show(fig_innovation(res.ekf, sigma_v))
                if res.pinn is not None:
                    show(fig_pinn_loss(res.pinn))

            section("Performance Scorecard")
            st.dataframe(te.metrics_table(res.metrics).style.format(
                {"RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.3f}"}, na_rep="—"))

            if res.pinn is not None:
                section("Identified Electrochemical Parameters (Hybrid PINN)")
                ph = res.pinn.physics
                p1, p2 = st.columns(2)
                with p1:
                    card("Degradation Kinetics", [
                        f"Rate constant k = {ph['k_per_Ah']:.2e} per Ah",
                        f"Apparent activation energy Eₐ = {ph['Ea_kJ_mol']:.1f} kJ mol⁻¹",
                        f"Fade exponent m = {ph['m_SEI_exponent']:.2f} "
                        + ("(self-limiting SEI growth)" if ph['m_SEI_exponent'] > 0.2 else
                           "(accelerating knee-onset)" if ph['m_SEI_exponent'] < -0.2 else "(linear throughput dependency)"),
                        f"Resistance coupling γ_int = {ph['gamma_int']:.2f}, γ_ct = {ph['gamma_ct']:.2f}"])
                with p2:
                    card("Charge-Transfer Dynamics (Butler–Volmer)", [
                        f"Exchange current i₀: {ph['i0_start_A']:.3f} A → {ph['i0_at_n0_A']:.3f} A at n₀",
                        f"Overpotential η_ct at 2A: {ph['eta_ct_2A_start_mV']:.0f} mV → {ph['eta_ct_2A_n0_mV']:.0f} mV",
                        f"Extra load polarization R_x = {ph['R_x_mOhm']:.1f} mΩ",
                        f"PINN optimization time = {res.pinn.train_seconds:.1f} s"])
                st.caption("Exchange current i₀ represents a lumped cell-level metric derived from linearised Butler–Volmer relations.")
            with st.expander("Methodology · Governing Equations"):
                st.latex(r"V = U_{OCV}(SOC) + I\,R_{int} + V_{RC},\quad "
                         r"V_{RC}^{k+1} = e^{-\Delta t/\tau}V_{RC}^{k} + (1-e^{-\Delta t/\tau})R_{ct}I")
                st.latex(r"\frac{dSOH}{dn} = -k\,e^{\frac{E_a}{R}\left(\frac{1}{T_{ref}}-\frac{1}{T}\right)}"
                         r"\,(2C_{bol}\,SOH)\left(\frac{1-SOH+\epsilon}{L_{ref}}\right)^{-m}")
                st.latex(r"\frac{dR_{j}}{dn} = \gamma_j R_{j,0}\left(-\frac{dSOH}{dn}\right),\qquad "
                         r"\Delta V_{step} = I(R_{int}+R_x) + \frac{2RT}{F}\sinh^{-1}\!\left(\frac{I F R_{ct}}{2RT}\right)")
    except Exception as exc:
        fail("Tab ② error", exc) if debug else st.error(f"Tab ② error: {exc}")


# =============================================================================
# Tab 3 · Twin-aware operation & economics
# =============================================================================
with tab_ops:
    try:
        st.markdown("The **twin-aware** policy simulates available discharge options at every cycle using the ECM + "
                    "degradation model to maximize expected net revenue: "
                    "**Price(I)·Ah − w·(Replacement Cost / Usable SOH)·ΔSOH** under strict thermal boundaries.")
        c1, c2, c3 = st.columns(3)
        policy = c1.selectbox("Decision-Making Policy", POLICIES)
        weight = c2.slider("Degradation penalty weight w", 0.25, 3.0, 1.0, 0.25, disabled=policy != "Twin-Aware",
                           help="Lower = aggressive utilization (sweat the asset); higher = conservative longevity strategy.")
        replacement = c3.number_input("Battery replacement cost (Currency Units)", 10.0, 2000.0, 150.0, 10.0)
        with st.expander("Economic & Environmental Configuration"):
            p1, p2, p3 = st.columns(3)
            price = (p1.number_input("Revenue per Ah @ 1A", 0.0, 10.0, 0.80, 0.05),
                     p2.number_input("Revenue per Ah @ 2A", 0.0, 10.0, 1.00, 0.05),
                     p3.number_input("Revenue per Ah @ 4A", 0.0, 10.0, 1.25, 0.05))
            s1, s2, s3, s4 = st.columns(4)
            t_max = s1.slider("Max cell temperature limit (°C)", 40, 60, 55)
            cold_rule = s2.toggle("Cold-temperature derating rule", value=True)
            cold_thr = s3.slider("Cold threshold (°C)", 0, 20, 10, disabled=not cold_rule)
            soh_eol_ops = s4.slider("Replacement threshold SOH", 0.6, 0.85, 0.70, 0.01)
            e1, e2 = st.columns(2)
            amb_mean = e1.slider("Mean ambient temperature (°C)", 0, 35, 20)
            amb_amp = e2.slider("Seasonal ambient amplitude (°C)", 0, 20, 16)
            ekf_saved = st.session_state.get("ekf")
            use_twin = st.toggle("Initialize physics from latest EKF estimation run",
                                 value=False, disabled=ekf_saved is None)
            show_base = st.toggle("Overlay fixed-current baselines", value=True)

        econ = te.Economics(price_per_Ah=tuple(price), replacement_cost=float(replacement), soh_eol=float(soh_eol_ops),
                            degradation_weight=float(weight), T_max_C=float(t_max),
                            cold_derate_below_C=float(cold_thr) if cold_rule else None)
        phys = te.CellPhysics()
        if use_twin and ekf_saved:
            r = ekf_saved["res"]
            phys = te.CellPhysics(k_ah=float(r.params["k_ah"]), R_int0=float(r.r_int0), R_ct0=float(r.r_ct0))
        ops_cfg = (policy, tuple(sorted(asdict(econ).items())), tuple(sorted(asdict(phys).items())),
                   amb_mean, amb_amp, show_base)

        if st.button("▶  Run Full Lifecycle Simulation", type="primary", key="ops_go"):
            with st.spinner("Simulating operational trajectory to end-of-life…"):
                df = run_policy(policy, asdict(econ), asdict(phys), float(amb_mean), float(amb_amp))
                base = {b: run_policy(b, asdict(econ), asdict(phys), float(amb_mean), float(amb_amp))
                        for b in POLICIES[1:] if show_base and b != policy}
                st.session_state["ops"] = {"cfg": ops_cfg, "df": df, "base": base, "policy": policy,
                                           "soh_eol": float(soh_eol_ops)}

        saved = st.session_state.get("ops")
        if saved:
            if saved["cfg"] != ops_cfg:
                st.warning("Configuration updated since previous run; displaying cached results.")
            s = te.summarise_life(saved["df"])
            k = st.columns(5)
            k[0].metric("Cycles to Replacement", s["cycles"])
            k[1].metric("Net Lifetime Profit", f"{s['profit']:.1f}")
            k[2].metric("Long-Run Profit Rate / h", f"{s['profit_per_h']:.3f}")
            k[3].metric("Mean Operating Current", f"{s['mean_I']:.2f} A")
            k[4].metric("Safety Violations", s["violations"])
            show(fig_policy(saved["df"], saved["policy"], saved["base"], saved["soh_eol"]))
            rows = [{"Policy": saved["policy"], **s}] + \
                [{"Policy": n, **te.summarise_life(d)} for n, d in saved["base"].items()]
            tbl = pd.DataFrame(rows).set_index("Policy")[["cycles", "Ah", "profit", "profit_per_h", "mean_I", "violations"]]
            st.dataframe(tbl.style.format({"Ah": "{:.0f}", "profit": "{:.1f}", "profit_per_h": "{:.3f}",
                                           "mean_I": "{:.2f}"}))
            st.caption("Net lifetime profit incorporates cumulative operational revenue minus terminal replacement costs.")
    except Exception as exc:
        fail("Tab ③ error", exc) if debug else st.error(f"Tab ③ error: {exc}")
