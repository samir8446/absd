"""
app.py - Battery Digital Twin & Operando Diagnostics Platform (Streamlit front-end)
==================================================================================

UI layer only; all computation lives in ``twin_engine.py``.

Design rules enforced in this file
  * Theme resiliency - custom CSS never hard-codes a text colour. Body text inherits the
    active Streamlit theme; surfaces and borders are translucent neutrals that read on both
    light and dark backgrounds. Plotly figures receive an explicit light or dark palette
    resolved from the active theme (``st.context.theme``, Streamlit >= 1.46) with a manual
    override in the sidebar.
  * Figures - one ``style_fig`` for every chart: bold title in a reserved top band, legend
    horizontally centred in a reserved bottom band below the x-axis title (bordered,
    theme-matched), unified cross-hair hover with a legible tooltip.
  * Layout - global cell selector and dataset status at the top of the page; three tabs:
    data & diagnostics, models & forecasting, operations & control.
"""

from __future__ import annotations

import html
import math
import os
import re
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

import twin_engine as te

st.set_page_config(
    page_title="Battery Digital Twin · Operando Diagnostics",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Constants & version gates
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MASTER_NAME, IMP_NAME = "battery_master_data.parquet", "impedance_ground_truth.parquet"
POLICIES = ["Twin-Aware", "Fixed 1 A", "Fixed 2 A", "Fixed 4 A"]
FONT = "Inter, 'Source Sans Pro', 'Helvetica Neue', Arial, sans-serif"


def _version_tuple(v: str) -> Tuple[int, int]:
    nums = [int(x) for x in re.findall(r"\d+", v)[:2]]
    nums += [0] * (2 - len(nums))
    return nums[0], nums[1]


ST_VERSION = _version_tuple(st.__version__)
PLOTLY_VERSION = _version_tuple(plotly.__version__)
# legend.yref="container" (plotly >= 5.15) pins the legend to the figure's bottom edge in
# pixels, independent of plot-area height and subplot layout.
LEGEND_CONTAINER_REF = PLOTLY_VERSION >= (5, 15)


# =============================================================================
# Theme-aware palettes
# =============================================================================
@dataclass(frozen=True)
class Palette:
    mode: str
    template: str
    text: str
    muted: str
    grid: str
    axis: str
    plot_bg: str
    hover_bg: str
    legend_bg: str
    legend_border: str
    accent: str
    measured: str
    cohort: str
    ekf: str
    pinn: str
    eis: str
    eol: str
    r_int: str
    r_ct: str
    currents: Tuple[Tuple[float, str], ...]
    series: Tuple[str, ...]
    ica_range: Tuple[float, float]

    def current_color(self, amps: float) -> str:
        return dict(self.currents).get(float(amps), self.muted)


DARK = Palette(
    mode="dark", template="plotly_dark",
    text="#f1f5f9", muted="#cbd5e1", grid="rgba(148,163,184,0.16)", axis="#64748b",
    plot_bg="rgba(148,163,184,0.05)", hover_bg="#0b1220",
    legend_bg="rgba(11,18,32,0.90)", legend_border="#64748b",
    accent="#38bdf8", measured="#f8fafc", cohort="rgba(148,163,184,0.38)",
    ekf="#38bdf8", pinn="#f472b6", eis="#fbbf24", eol="#f87171",
    r_int="#fb923c", r_ct="#34d399",
    currents=((1.0, "#60a5fa"), (2.0, "#34d399"), (4.0, "#f87171")),
    series=("#a3e635", "#facc15", "#c084fc", "#fb7185", "#2dd4bf"),
    ica_range=(0.15, 0.95),
)

LIGHT = Palette(
    mode="light", template="plotly_white",
    text="#0f172a", muted="#334155", grid="rgba(15,23,42,0.09)", axis="#94a3b8",
    plot_bg="rgba(15,23,42,0.02)", hover_bg="#ffffff",
    legend_bg="rgba(255,255,255,0.95)", legend_border="#94a3b8",
    accent="#0369a1", measured="#0f172a", cohort="rgba(100,116,139,0.35)",
    ekf="#0369a1", pinn="#be185d", eis="#b45309", eol="#dc2626",
    r_int="#c2410c", r_ct="#047857",
    currents=((1.0, "#1d4ed8"), (2.0, "#047857"), (4.0, "#b91c1c")),
    series=("#4d7c0f", "#b45309", "#7e22ce", "#be123c", "#0f766e"),
    # Viridis' yellow end has too little contrast on white, so stop before it.
    ica_range=(0.0, 0.82),
)


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def detect_base_theme() -> str:
    """Active Streamlit theme: st.context.theme (>= 1.46), then config, then light."""
    try:
        t = getattr(getattr(st.context, "theme", None), "type", None)
        if t in ("light", "dark"):
            return t
    except Exception:
        pass
    try:
        base = st.get_option("theme.base")
        if base in ("light", "dark"):
            return base
    except Exception:
        pass
    return "light"


def resolve_palette(choice: str) -> Palette:
    mode = choice.lower() if choice in ("Light", "Dark") else detect_base_theme()
    return DARK if mode == "dark" else LIGHT


def inject_css(P: Palette) -> None:
    """Text colours are inherited from the Streamlit theme; only accents use the palette.
    Surfaces are translucent neutral greys, which read on both light and dark bases."""
    css = f"""<style>
:root {{
  --bt-accent: {P.accent};
  --bt-accent-soft: {rgba(P.accent, 0.12)};
  --bt-accent-line: {rgba(P.accent, 0.45)};
  --bt-surface: rgba(128,128,128,0.07);
  --bt-border: rgba(128,128,128,0.30);
}}
.block-container {{padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1600px;}}
.bt-hero {{
  border: 1px solid var(--bt-border); border-left: 6px solid var(--bt-accent);
  border-radius: 10px; padding: 20px 28px; margin-bottom: 1rem; background: var(--bt-surface);
}}
.bt-hero .bt-title {{font-size: 1.65rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.25; color: inherit;}}
.bt-hero .bt-sub {{margin-top: 6px; font-size: 0.98rem; line-height: 1.5; opacity: 0.82; color: inherit; max-width: 80ch;}}
.bt-pill {{
  display: inline-block; padding: 3px 11px; margin: 10px 8px 0 0; border-radius: 6px;
  font-size: 0.8rem; font-weight: 650; background: var(--bt-accent-soft);
  color: var(--bt-accent); border: 1px solid var(--bt-accent-line);
}}
.bt-chip {{
  display: inline-block; padding: 2px 10px; margin: 4px 6px 0 0; border-radius: 999px;
  font-size: 0.82rem; background: var(--bt-surface); border: 1px solid var(--bt-border); color: inherit;
}}
.bt-status {{font-size: 0.95rem; line-height: 1.6; color: inherit;}}
.bt-section {{
  font-size: 1.18rem; font-weight: 800; margin: 1.8rem 0 0.7rem 0; padding-left: 12px;
  border-left: 4px solid var(--bt-accent); color: inherit; letter-spacing: -0.01em;
}}
.bt-card {{
  background: var(--bt-surface); border: 1px solid var(--bt-border); border-radius: 10px;
  padding: 16px 20px; margin: 8px 0 14px 0;
}}
.bt-card h4 {{margin: 0 0 8px 0; font-size: 0.98rem; font-weight: 800; color: var(--bt-accent);}}
.bt-card li {{font-size: 0.93rem; line-height: 1.5; margin: 3px 0; color: inherit;}}
div[data-testid="stMetric"] {{
  background: var(--bt-surface); border: 1px solid var(--bt-border); border-radius: 10px; padding: 14px 18px;
}}
div[data-testid="stMetricLabel"] p {{font-size: 0.82rem; font-weight: 650; opacity: 0.78;}}
div[data-testid="stMetricValue"] {{font-weight: 750;}}
button[data-baseweb="tab"] p {{font-size: 1.02rem; font-weight: 700;}}
</style>"""
    st.markdown(css, unsafe_allow_html=True)


# =============================================================================
# Small UI helpers
# =============================================================================
def section(title: str) -> None:
    st.markdown(f'<div class="bt-section">{html.escape(title)}</div>', unsafe_allow_html=True)


def card(title: str, lines: Sequence[str]) -> None:
    body = "".join(f"<li>{html.escape(ln)}</li>" for ln in lines)
    st.markdown(f'<div class="bt-card"><h4>{html.escape(title)}</h4>'
                f'<ul style="margin:0;padding-left:20px">{body}</ul></div>', unsafe_allow_html=True)


def columns(spec: Any, **kw: Any):
    """st.columns with graceful fallback for kwargs unsupported by older Streamlit."""
    try:
        return st.columns(spec, **kw)
    except TypeError:
        return st.columns(spec)


def fmt(value: Any, spec: str = ".3f", unit: str = "") -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return f"{v:{spec}}{(' ' + unit) if unit else ''}"


def report_error(where: str, exc: BaseException, verbose: bool) -> None:
    st.error(f"{where}: {exc}")
    if verbose:
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:])


def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "")) or os.environ.get(name, "")
    except Exception:
        return os.environ.get(name, "")


PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "scale": 3},
}


def show(fig: go.Figure, key: str) -> None:
    """Render with Plotly styling intact (theme=None) across Streamlit versions."""
    base = dict(theme=None, config=PLOT_CONFIG)
    attempts: List[Dict[str, Any]] = []
    if ST_VERSION >= (1, 50):
        attempts.append(dict(width="stretch", key=key))
    attempts += [dict(use_container_width=True, key=key), dict(use_container_width=True)]
    for extra in attempts:
        try:
            st.plotly_chart(fig, **base, **extra)
            return
        except TypeError:
            continue


def show_table(data: Any) -> None:
    if ST_VERSION >= (1, 50):
        try:
            st.dataframe(data, width="stretch")
            return
        except TypeError:
            pass
    st.dataframe(data, use_container_width=True)


# =============================================================================
# Figure styling
# =============================================================================
AXIS_BLOCK_PX = 62      # tick labels + x-axis title below the plot area
LEGEND_ROW_PX = 24
TITLE_BAND_PX = 92


def _legend_entries(fig: go.Figure) -> List[str]:
    names, groups = [], set()
    for tr in fig.data:
        if tr.showlegend is False or not tr.name:
            continue
        if tr.legendgroup:
            if tr.legendgroup in groups:
                continue
            groups.add(tr.legendgroup)
        names.append(str(tr.name))
    return names


def _legend_rows(names: Sequence[str], width_px: int) -> int:
    needed = sum(7.2 * len(n) + 48 for n in names)
    return max(1, math.ceil(needed / max(width_px - 80, 200)))


def _style_subplot_titles(fig: go.Figure, P: Palette) -> None:
    for ann in fig.layout.annotations:
        ann.font = dict(size=13, color=P.text, family=FONT)


def style_fig(fig: go.Figure, P: Palette, height: int = 460, title: Optional[str] = None,
              width_hint: int = 1100, hovermode: str = "x unified") -> go.Figure:
    """Publication layout: reserved title band on top, reserved legend band at the bottom
    (below the x-axis title), unified cross-hair hover. ``width_hint`` is the approximate
    rendered width and only sizes the legend band when entries wrap onto several rows."""
    names = _legend_entries(fig)
    rows = _legend_rows(names, width_hint) if names else 0
    legend_px = rows * LEGEND_ROW_PX + 14 if rows else 0
    top = TITLE_BAND_PX if title else 40
    bottom = AXIS_BLOCK_PX + (legend_px + 18 if rows else 0)
    H = int(height + max(rows - 1, 0) * LEGEND_ROW_PX)

    if LEGEND_CONTAINER_REF:
        legend_pos = dict(yref="container", y=8 / H, yanchor="bottom")
    else:
        plot_h = max(H - top - bottom, 60)
        legend_pos = dict(y=-(AXIS_BLOCK_PX + 8) / plot_h, yanchor="top")

    fig.update_layout(
        template=P.template, height=H, showlegend=bool(names),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=P.plot_bg,
        font=dict(family=FONT, size=13, color=P.text),
        margin=dict(l=76, r=30, t=top, b=bottom, pad=4),
        legend=dict(orientation="h", xanchor="center", x=0.5,
                    bgcolor=P.legend_bg, bordercolor=P.legend_border, borderwidth=1,
                    font=dict(family=FONT, size=12, color=P.text),
                    itemsizing="constant", **legend_pos),
        hovermode=hovermode,
        hoverlabel=dict(bgcolor=P.hover_bg, bordercolor=P.legend_border, namelength=-1,
                        font=dict(family=FONT, size=12, color=P.text)),
    )
    if title:
        fig.update_layout(title=dict(text=f"<b>{html.escape(title)}</b>", x=0.012, xanchor="left",
                                     y=1 - 16 / H, yanchor="top",
                                     font=dict(family=FONT, size=16, color=P.text)))
    axis = dict(gridcolor=P.grid, zerolinecolor=P.grid, linecolor=P.axis, showline=True, mirror=True,
                ticks="outside", tickcolor=P.axis, title_standoff=10,
                title_font=dict(family=FONT, size=13, color=P.text),
                tickfont=dict(family=FONT, size=12, color=P.muted))
    fig.update_xaxes(**axis, showspikes=True, spikemode="across", spikesnap="cursor",
                     spikethickness=1, spikedash="dot", spikecolor=P.muted)
    fig.update_yaxes(**axis)
    return fig


def _forecast_frame(fig: go.Figure, n0: int, n_max: float, soh_eol: float, P: Palette,
                    row: Optional[int] = None) -> None:
    kw = dict(row=row, col=1) if row else {}
    fig.add_vrect(x0=n0, x1=n_max, fillcolor=rgba(P.accent, 0.07), line_width=0, layer="below", **kw)
    fig.add_vline(x=n0, line_dash="dot", line_color=P.accent, line_width=1.5, **kw)
    fig.add_hline(y=soh_eol, line_dash="dash", line_color=P.eol, line_width=1.5,
                  annotation_text="EOL", annotation_position="bottom left",
                  annotation_font=dict(color=P.eol, size=12), **kw)


def _origin_label(fig: go.Figure, n0: int, P: Palette) -> None:
    fig.add_annotation(x=n0, xref="x", y=1.0, yref="paper", xanchor="left", yanchor="bottom",
                       text=f"<b> forecast origin n₀ = {n0}</b>", showarrow=False,
                       font=dict(color=P.accent, size=12, family=FONT))


# =============================================================================
# Figures
# =============================================================================
def fig_fade(ct_all: pd.DataFrame, cell_id: str, y: str, eol_line: Optional[float], P: Palette) -> go.Figure:
    fig = go.Figure()
    first = True
    for cid, d in ct_all[~ct_all["outlier"]].groupby("Cell_ID"):
        if cid == cell_id:
            continue
        # Cohort traces are context only: excluded from the unified tooltip to keep it readable.
        fig.add_trace(go.Scatter(x=d["n"], y=d[y], mode="lines", line=dict(color=P.cohort, width=1.3),
                                 name="Cohort cells", legendgroup="cohort", showlegend=first,
                                 hoverinfo="skip"))
        first = False
    d = ct_all[ct_all["Cell_ID"] == cell_id]
    good, bad = d[~d["outlier"]], d[d["outlier"]]
    fig.add_trace(go.Scatter(x=good["n"], y=good[y], mode="lines+markers", name=f"Target cell {cell_id}",
                             line=dict(color=P.accent, width=3), marker=dict(size=5, color=P.accent),
                             hovertemplate="%{y:.4f}<extra></extra>"))
    if len(bad):
        fig.add_trace(go.Scatter(x=bad["n"], y=bad[y], mode="markers", name="Flagged outliers",
                                 marker=dict(symbol="x", size=9, color=P.eol, line=dict(width=2)),
                                 hovertemplate="%{y:.4f} (outlier)<extra></extra>"))
    if eol_line is not None:
        fig.add_hline(y=eol_line, line_dash="dash", line_color=P.eol, line_width=1.5,
                      annotation_text="End of life", annotation_position="bottom right",
                      annotation_font=dict(color=P.eol, size=12))
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="Discharge capacity (Ah)" if y == "Capacity_Ah" else "State of health SOH (–)")
    return style_fig(fig, P, 470, f"Capacity fade trajectory, {cell_id} against cohort")


def fig_ica(curves: List[te.ICACurve], P: Palette) -> go.Figure:
    fig = go.Figure()
    lo, hi = P.ica_range
    cols = sample_colorscale("Viridis", list(np.linspace(lo, hi, max(len(curves), 2))))
    for c, col in zip(curves, cols):
        fig.add_trace(go.Scatter(x=c.voltage, y=c.dqdv, mode="lines", name=f"n = {c.n}",
                                 line=dict(color=col, width=2.4),
                                 hovertemplate=f"n = {c.n}<br>%{{x:.3f}} V · %{{y:.2f}} Ah/V<extra></extra>"))
        fig.add_trace(go.Scatter(x=[c.peak_V], y=[c.peak_height], mode="markers", showlegend=False,
                                 marker=dict(color=col, size=10, line=dict(color=P.text, width=1.2)),
                                 hovertemplate=f"peak, n = {c.n}<br>%{{x:.3f}} V<extra></extra>"))
    fig.update_xaxes(title_text="Cell voltage (V)", autorange="reversed")
    fig.update_yaxes(title_text="dQ/dV (Ah V⁻¹)")
    return style_fig(fig, P, 470, "Incremental capacity (dQ/dV) across life", width_hint=900,
                     hovermode="closest")


def fig_peaks(curves: List[te.ICACurve], P: Palette) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    n = [c.n for c in curves]
    fig.add_trace(go.Scatter(x=n, y=[c.peak_V for c in curves], mode="lines+markers", name="Peak voltage",
                             line=dict(color=P.accent, width=2.5), marker=dict(size=7),
                             hovertemplate="%{y:.3f} V<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=n, y=[c.peak_height for c in curves], mode="lines+markers", name="Peak height",
                             line=dict(color=P.pinn, width=2.5), marker=dict(size=7),
                             hovertemplate="%{y:.2f} Ah/V<extra></extra>"), secondary_y=True)
    fig.update_xaxes(title_text="Discharge cycle n")
    style_fig(fig, P, 340, "Main peak evolution", width_hint=620)
    # Colour-coded axis titles are applied after the shared style so they are not overwritten.
    fig.update_yaxes(title_text="Peak voltage (V)", title_font_color=P.accent, secondary_y=False)
    fig.update_yaxes(title_text="Peak height (Ah V⁻¹)", title_font_color=P.pinn, showgrid=False,
                     mirror=False, secondary_y=True)
    return fig


def fig_ml(results: List[te.MLForecast], ct_cell: pd.DataFrame, n0: int, soh_eol: float,
           P: Palette) -> go.Figure:
    fig = go.Figure()
    good = ct_cell[~ct_cell["outlier"]]
    fig.add_trace(go.Scatter(x=good["n"], y=good["SOH"], mode="markers", name="Measured SOH",
                             marker=dict(color=P.measured, size=6, opacity=0.85),
                             hovertemplate="%{y:.4f}<extra></extra>"))
    for i, r in enumerate(results):
        col = P.series[i % len(P.series)]
        fig.add_trace(go.Scatter(x=r.n_grid, y=r.soh_pred, mode="lines", name=r.model,
                                 line=dict(color=col, width=2.6), hovertemplate="%{y:.4f}<extra></extra>"))
    n_max = max(float(r.n_grid.max()) for r in results)
    _forecast_frame(fig, n0, n_max, soh_eol, P)
    _origin_label(fig, n0, P)
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="State of health SOH (–)")
    return style_fig(fig, P, 500, "ML surrogate SOH forecasts")


def fig_compare(res: te.ComparisonResult, P: Palette) -> go.Figure:
    colors = {"ECM + EKF": P.ekf, "Hybrid PINN": P.pinn}
    fig = go.Figure()
    m = res.measured
    fig.add_trace(go.Scatter(x=m["n"], y=m["SOH"], mode="markers", name="Measured SOH",
                             marker=dict(color=P.measured, size=6, opacity=0.85),
                             hovertemplate="%{y:.4f}<extra></extra>"))
    if res.ekf is not None:
        e = res.ekf.per_cycle.dropna(subset=["SOH"])
        n_arr = e["n"].to_numpy()
        up = (e["SOH"] + 2 * e["SOH_std"]).to_numpy()
        lo = (e["SOH"] - 2 * e["SOH_std"]).to_numpy()
        fig.add_trace(go.Scatter(x=np.r_[n_arr, n_arr[::-1]], y=np.r_[up, lo[::-1]], fill="toself",
                                 fillcolor=rgba(P.ekf, 0.18), line=dict(width=0),
                                 name="EKF ±2σ band", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=n_arr, y=e["SOH"], mode="lines", name="EKF causal estimate",
                                 line=dict(color=P.ekf, width=2.6), hovertemplate="%{y:.4f}<extra></extra>"))
    for i, (name, (n_g, p_g)) in enumerate(res.predictions.items()):
        col = colors.get(name, P.series[i % len(P.series)])
        mask = n_g >= res.n0
        fig.add_trace(go.Scatter(x=n_g[mask], y=p_g[mask], mode="lines", name=f"{name} forecast",
                                 line=dict(color=col, width=3, dash="dash" if name == "ECM + EKF" else "solid"),
                                 hovertemplate="%{y:.4f}<extra></extra>"))
        if name == "Hybrid PINN":
            fig.add_trace(go.Scatter(x=n_g[~mask], y=p_g[~mask], mode="lines", name="PINN fit (n ≤ n₀)",
                                     line=dict(color=col, width=2, dash="dot"),
                                     hovertemplate="%{y:.4f}<extra></extra>"))
    n_max = max(float(n.max()) for n, _ in res.predictions.values()) if res.predictions else float(m["n"].max())
    _forecast_frame(fig, res.n0, n_max, res.soh_eol, P)
    _origin_label(fig, res.n0, P)
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="State of health SOH (–)")
    return style_fig(fig, P, 540, f"Forecast benchmark on {res.cell_id}: ML, ECM + EKF and hybrid PINN")


def fig_params(res: te.ComparisonResult, P: Palette) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.13,
                        subplot_titles=("Ohmic resistance R_int", "Charge-transfer resistance R_ct"))
    _style_subplot_titles(fig, P)
    for row, key, eis_col, label in ((1, "R_int", "Re_ohm", "Re"), (2, "R_ct", "Rct_ohm", "Rct")):
        if res.ekf is not None:
            e = res.ekf.per_cycle.dropna(subset=[key])
            fig.add_trace(go.Scatter(x=e["n"], y=1e3 * e[key], mode="lines", name=f"EKF {key}",
                                     line=dict(color=P.ekf, width=2.4),
                                     hovertemplate="%{y:.1f} mΩ<extra></extra>"), row=row, col=1)
        if res.pinn is not None:
            fig.add_trace(go.Scatter(x=res.pinn.n_grid, y=1e3 * getattr(res.pinn, key.lower()), mode="lines",
                                     name=f"PINN {key}", line=dict(color=P.pinn, width=2.4),
                                     hovertemplate="%{y:.1f} mΩ<extra></extra>"), row=row, col=1)
        if len(res.eis):
            ci = res.measured.sort_values("Cycle_Index")[["Cycle_Index", "n"]]
            ee = pd.merge_asof(res.eis.sort_values("Cycle_Index"), ci, on="Cycle_Index", direction="backward")
            fig.add_trace(go.Scatter(x=ee["n"].fillna(1), y=1e3 * ee[eis_col], mode="markers",
                                     name=f"EIS {label}",
                                     marker=dict(color=P.eis, symbol="x", size=9, line=dict(width=1.5)),
                                     hovertemplate="%{y:.1f} mΩ (EIS)<extra></extra>"), row=row, col=1)
        fig.add_vline(x=res.n0, line_dash="dot", line_color=P.accent, row=row, col=1)
        fig.update_yaxes(title_text="Resistance (mΩ)", row=row, col=1)
    fig.update_xaxes(title_text="Discharge cycle n", row=2, col=1)
    return style_fig(fig, P, 580, "Identified resistances against EIS", width_hint=720)


def fig_innovation(ekf: te.EKFResult, sigma_v: float, P: Palette) -> go.Figure:
    e = ekf.per_cycle.dropna(subset=["innov_mean_mV"])
    n_arr = e["n"].to_numpy()
    fig = go.Figure()
    fig.add_hrect(y0=-2e3 * sigma_v, y1=2e3 * sigma_v, fillcolor=rgba(P.r_ct, 0.10), line_width=0,
                  annotation_text="±2σᵥ", annotation_position="top left",
                  annotation_font=dict(color=P.r_ct, size=12))
    up = (e["innov_mean_mV"] + e["innov_std_mV"]).to_numpy()
    lo = (e["innov_mean_mV"] - e["innov_std_mV"]).to_numpy()
    fig.add_trace(go.Scatter(x=np.r_[n_arr, n_arr[::-1]], y=np.r_[up, lo[::-1]], fill="toself",
                             fillcolor=rgba(P.ekf, 0.18), line=dict(width=0), name="±1 std within cycle",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=n_arr, y=e["innov_mean_mV"], mode="lines", name="Mean innovation",
                             line=dict(color=P.ekf, width=2.4), hovertemplate="%{y:.1f} mV<extra></extra>"))
    fig.add_hline(y=0, line_color=P.muted, line_dash="dash", line_width=1)
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="V_meas − V_pred (mV)")
    return style_fig(fig, P, 400, "EKF voltage innovation", width_hint=720)


def fig_pinn_loss(p: te.PINNResult, P: Palette) -> go.Figure:
    fig = go.Figure()
    spec = (("total", P.text, "Total"), ("data", P.r_ct, "Data"), ("phys", P.pinn, "Physics"),
            ("bv", P.accent, "Butler–Volmer"), ("eis", P.eis, "EIS anchor"))
    for key, col, label in spec:
        if key in p.history.columns:
            fig.add_trace(go.Scatter(x=p.history["epoch"], y=p.history[key], mode="lines", name=label,
                                     line=dict(color=col, width=3 if key == "total" else 1.6),
                                     hovertemplate="%{y:.3e}<extra></extra>"))
    fig.update_yaxes(type="log", title_text="Loss (normalised, log)")
    fig.update_xaxes(title_text="Epoch")
    return style_fig(fig, P, 400, "Hybrid PINN training losses", width_hint=720)


def fig_policy(df: pd.DataFrame, policy: str, baselines: Dict[str, pd.DataFrame], soh_eol: float,
               P: Palette) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.28, 0.32, 0.40],
                        specs=[[{"secondary_y": True}], [{}], [{}]],
                        subplot_titles=("Selected current and ambient temperature",
                                        "State of health", "Cumulative net profit"))
    _style_subplot_titles(fig, P)
    for amps in (1.0, 2.0, 4.0):
        mm = df["I"] == amps
        if mm.any():
            fig.add_trace(go.Scatter(x=df.loc[mm, "cycle"], y=df.loc[mm, "I"], mode="markers",
                                     name=f"{amps:g} A selected", marker=dict(color=P.current_color(amps), size=7),
                                     hovertemplate="%{y:g} A<extra></extra>"),
                          row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df["cycle"], y=df["T_amb"], mode="lines", name="Ambient temperature",
                             line=dict(color=P.muted, width=1.5, dash="dot"),
                             hovertemplate="%{y:.1f} °C<extra></extra>"), row=1, col=1, secondary_y=True)
    for name, d in {policy: df, **baselines}.items():
        is_main = name == policy
        if is_main:
            style = dict(color=P.text, width=3)
        else:
            style = dict(color=P.current_color(float(name.split()[1])), width=1.6, dash="dash")
        fig.add_trace(go.Scatter(x=d["cycle"], y=d["SOH"], mode="lines", name=name, line=style,
                                 legendgroup=name, hovertemplate="%{y:.4f}<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(x=d["cycle"], y=d["cum_profit"], mode="lines", name=name, line=style,
                                 legendgroup=name, showlegend=False,
                                 hovertemplate="%{y:.1f}<extra></extra>"), row=3, col=1)
    fig.add_hline(y=soh_eol, line_dash="dash", line_color=P.eol, line_width=1.5, row=2, col=1)
    fig.update_xaxes(title_text="Operating cycle", row=3, col=1)
    style_fig(fig, P, 820, f"Lifecycle simulation: {policy} policy")
    fig.update_yaxes(title_text="Current (A)", tickvals=[1, 2, 4], row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Ambient (°C)", row=1, col=1, secondary_y=True, showgrid=False, mirror=False)
    fig.update_yaxes(title_text="SOH (–)", row=2, col=1)
    fig.update_yaxes(title_text="Profit (CU)", row=3, col=1)
    return fig


# =============================================================================
# Cached data access
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
def run_policy(policy: str, econ: Dict[str, Any], phys: Dict[str, Any],
               amb_mean: float, amb_amp: float) -> pd.DataFrame:
    return te.simulate_life(te.make_policy(policy), te.CellPhysics(**phys), te.Economics(**econ),
                            ambient_mean_C=amb_mean, ambient_amp_C=amb_amp)


def resolve_sources(up_master: Any, up_imp: Any, master_url: str, imp_url: str
                    ) -> Tuple[Optional[te.ParquetStore], Optional[pd.DataFrame], List[str]]:
    notes: List[str] = []
    store: Optional[te.ParquetStore] = None
    imp: Optional[pd.DataFrame] = None

    master_path: Optional[Path] = None
    if up_master is not None:
        master_path = Path(persist_upload(up_master.name, up_master.size, up_master.getvalue()))
        notes.append(f"Telemetry: uploaded {up_master.name}")
    else:
        for cand in (DATA_DIR / MASTER_NAME, APP_DIR / MASTER_NAME):
            if cand.exists():
                master_path = cand
                notes.append("Telemetry: local file")
                break
    if master_path is None and master_url:
        bar = st.sidebar.progress(0.0, text="Downloading telemetry…")
        master_path = te.download_to_cache(master_url, progress=lambda f, m: bar.progress(f, text=f"Telemetry {m}"))
        bar.empty()
        notes.append("Telemetry: remote URL (cached)")
    if master_path is not None:
        store = get_store(str(master_path), master_path.stat().st_mtime)

    if up_imp is not None:
        imp = te.load_impedance(up_imp.getvalue())
        notes.append(f"EIS: uploaded {up_imp.name}")
    else:
        imp_path = next((p for p in (DATA_DIR / IMP_NAME, APP_DIR / IMP_NAME) if p.exists()), None)
        if imp_path is None and imp_url:
            imp_path = te.download_to_cache(imp_url)
        if imp_path is not None:
            imp = load_impedance_path(str(imp_path), imp_path.stat().st_mtime)
            notes.append("EIS: loaded")
    if imp is None:
        notes.append("EIS: not available")
    return store, imp, notes


# =============================================================================
# Sidebar: data sources, appearance, diagnostics
# =============================================================================
with st.sidebar:
    st.markdown("### Data sources")
    up_master = st.file_uploader("Master telemetry (.parquet)", type=["parquet"], key="up_master")
    up_imp = st.file_uploader("EIS ground truth (.parquet)", type=["parquet"], key="up_imp")
    with st.expander("Remote URLs", expanded=False):
        master_url = st.text_input("Telemetry URL", value=_secret("MASTER_PARQUET_URL"))
        imp_url = st.text_input("EIS URL", value=_secret("IMPEDANCE_PARQUET_URL"))

    st.markdown("### Appearance")
    theme_choice = st.radio(
        "Figure palette", ["Auto", "Light", "Dark"], horizontal=True,
        help="Auto follows the active Streamlit theme (Settings → Theme). "
             "Choose Light or Dark if your Streamlit version cannot report the theme.")
    debug = st.toggle("Show tracebacks on errors", value=False)

P = resolve_palette(theme_choice)
inject_css(P)

# =============================================================================
# Hero
# =============================================================================
pills = "".join(f'<span class="bt-pill">{t}</span>' for t in
                ("Butler–Volmer kinetics", "Arrhenius / SEI fade laws", "ECM + EKF observer",
                 "Hybrid PINN", "Twin-aware control"))
st.markdown(
    '<div class="bt-hero">'
    '<div class="bt-title">🔋 Battery Digital Twin &amp; Operando Diagnostics</div>'
    '<div class="bt-sub">NASA Ames 18650 LiCoO₂ / graphite ageing telemetry: incremental capacity analysis, '
    'ML surrogates, an extended Kalman filter observer and a physics-informed neural network, '
    'benchmarked on a common forecast-origin protocol.</div>'
    f'<div>{pills}</div></div>',
    unsafe_allow_html=True,
)

# =============================================================================
# Data ingestion
# =============================================================================
store: Optional[te.ParquetStore] = None
imp: Optional[pd.DataFrame] = None
ct: Optional[pd.DataFrame] = None
source_notes: List[str] = []
try:
    store, imp, source_notes = resolve_sources(up_master, up_imp, master_url.strip(), imp_url.strip())
    if store is not None:
        with st.spinner("Extracting per-cycle features…"):
            ct = cycle_table(store, store.key)
except Exception as exc:
    report_error("Data ingestion failed", exc, verbose=True)

if store is None or ct is None:
    st.info("No telemetry loaded. Place `battery_master_data.parquet` in `./data/`, "
            "upload it in the sidebar, or set a remote URL.")
    st.stop()

meta = te.cell_meta(ct)
cells = list(meta.index)


def _cell_label(cid: str) -> str:
    row = meta.loc[cid]
    return (f"{cid}  ({fmt(row['Ambient_C'], '.0f', '°C')}, {fmt(row['I_dis_A'], '.1f', 'A')}, "
            f"{int(row['cycles'])} cycles)")


# =============================================================================
# Global control bar: target cell, EOL threshold, dataset status
# =============================================================================
with st.container(border=True):
    g1, g2, g3 = columns([1.5, 1.0, 2.3], gap="large", vertical_alignment="center")
    with g1:
        cell = st.selectbox("Target cell", cells, key="cell", format_func=_cell_label,
                            index=cells.index("B0005") if "B0005" in cells else 0)
    with g2:
        eol_ah = st.number_input("End-of-life capacity (Ah)", 0.8, 2.0, float(te.DEFAULT_EOL_AH), 0.05)

    c_bol = float(meta.loc[cell, "C_bol_Ah"])
    soh_eol = te.soh_eol_for(c_bol, eol_ah)
    ct_cell = ct[ct["Cell_ID"] == cell].sort_values("n")
    eis_cell = te.valid_eis(imp, cell)

    with g3:
        chips = "".join(f'<span class="bt-chip">{html.escape(n)}</span>' for n in source_notes)
        st.markdown(
            f'<div class="bt-status">Analysing <b>{html.escape(cell)}</b>: '
            f'SOH<sub>EOL</sub> = {soh_eol:.3f} ({eol_ah:.2f} Ah of {c_bol:.3f} Ah initial)</div>'
            f'<div>{chips}</div>',
            unsafe_allow_html=True,
        )

tab_data, tab_models, tab_ops = st.tabs([
    "📊 Data & diagnostics",
    "🧠 Models & forecasting",
    "⚙️ Operations & control",
])

# =============================================================================
# Tab 1: data presentation & analysis
# =============================================================================
with tab_data:
    try:
        k = st.columns(6)
        k[0].metric("Target cell", cell)
        k[1].metric("Initial capacity", fmt(c_bol, ".3f", "Ah"))
        k[2].metric("Valid discharge cycles", int(meta.loc[cell, "cycles"]))
        k[3].metric("Capacity fade", fmt(meta.loc[cell, "fade_pct"], ".1f", "%"))
        k[4].metric("Ambient / discharge current",
                    f"{fmt(meta.loc[cell, 'Ambient_C'], '.0f', '°C')} / {fmt(meta.loc[cell, 'I_dis_A'], '.1f', 'A')}")
        k[5].metric("Valid EIS sweeps", len(eis_cell))

        section("Capacity fade against the cohort")
        yvar = st.radio("Metric", ["SOH", "Capacity_Ah"], horizontal=True,
                        format_func=lambda v: "State of health" if v == "SOH" else "Capacity (Ah)")
        show(fig_fade(ct, cell, yvar, soh_eol if yvar == "SOH" else eol_ah, P), key="fade")

        section("Incremental capacity analysis (dQ/dV)")
        c1, c2, c3, c4 = st.columns(4)
        n_curves = c1.slider("Curves across life", 2, 12, 6)
        ir = c2.toggle("IR-compensate voltage", value=True,
                       help="Adds |I|·R_dc to the terminal voltage to remove the ohmic shift.")
        dv = c3.select_slider("Voltage bin (mV)", [5, 10, 15, 20], value=10) / 1000
        win = c4.select_slider("Savitzky–Golay window", [5, 7, 9, 11, 15, 21], value=9)
        with st.spinner("Computing dQ/dV curves…"):
            prep = prepared_cell(store, store.key, cell)
            curves = ica_cached(prep, ct_cell, store.key, cell, n_curves, ir, dv, win)
        if not curves:
            st.warning("Not enough voltage resolution in this cell's discharges to compute dQ/dV.")
        else:
            a, b = st.columns([3, 2], gap="medium")
            with a:
                show(fig_ica(curves, P), key="ica")
            with b:
                show(fig_peaks(curves, P), key="ica_peaks")
                diag = te.diagnose_degradation(curves)
                if diag["available"]:
                    card(f"Degradation reading, n = {diag['n_ref']} to n = {diag['n_cur']}",
                         [f"Capacity ratio {diag['cap_ratio']:.2f}, peak height ratio "
                          f"{diag['height_ratio']:.2f}, peak shift {diag['shift_mV']:.0f} mV"]
                         + list(diag["interpretation"]))
            st.caption("NASA discharges run near 1C, so curves are not near-equilibrium: peak broadening "
                       "partly reflects kinetic overpotential and the mode reading is indicative only.")
    except Exception as exc:
        report_error("Data analysis failed", exc, debug)


# =============================================================================
# Tab 2: models & predictive forecasting
# =============================================================================
with tab_models:
    try:
        # ---------------- ML surrogates ----------------
        section("Machine-learning surrogates")
        c1, c2, c3 = st.columns([3, 2, 2])
        models = c1.multiselect("Models", list(te.ML_MODELS), default=list(te.ML_MODELS))
        frac = c2.slider("Forecast origin (fraction of life observed)", 0.2, 0.8, 0.4, 0.05, key="ml_frac")
        use_pop = c3.toggle("Train on the multi-cell cohort", value=True,
                            help="Combine population ageing with the target cell's early cycles.")
        n0_ml = int(max(5, round(frac * meta.loc[cell, "cycles"])))
        ml_cfg = (cell, n0_ml, tuple(models), use_pop, float(eol_ah))

        if st.button("Train ML surrogates", type="primary", key="ml_go", disabled=not models):
            prog = st.progress(0.0)
            out: List[te.MLForecast] = []
            for i, mname in enumerate(models):
                prog.progress(i / max(len(models), 1), text=f"Training {mname}…")
                try:
                    out.append(ml_cached(ct, store.key, cell, n0_ml, mname, use_pop, float(eol_ah)))
                except Exception as exc:
                    st.warning(f"{mname}: {exc}")
            prog.empty()
            st.session_state["ml"] = {"cfg": ml_cfg, "res": out}

        saved_ml = st.session_state.get("ml")
        if saved_ml and saved_ml["res"]:
            if saved_ml["cfg"][0] != cell:
                st.info("The stored ML results belong to another cell. Train the surrogates to update them.")
            else:
                if saved_ml["cfg"] != ml_cfg:
                    st.warning("Settings changed since the last run; the results below use the previous settings.")
                show(fig_ml(saved_ml["res"], ct_cell, saved_ml["cfg"][1], soh_eol, P), key="ml_fig")
                tbl = pd.DataFrame([{"Model": r.model, "RMSE": r.metrics.rmse, "MAE": r.metrics.mae,
                                     "R²": r.metrics.r2, "RUL true": r.metrics.rul_true,
                                     "RUL pred": r.metrics.rul_pred, "RUL error": r.metrics.rul_error,
                                     "Train rows": r.train_rows, "Fit time (s)": r.fit_seconds}
                                    for r in saved_ml["res"]]).set_index("Model")
                show_table(tbl.style.format(
                    {"RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.3f}", "RUL true": "{:.0f}",
                     "RUL pred": "{:.0f}", "RUL error": "{:+.0f}", "Fit time (s)": "{:.2f}"}, na_rep="—")
                    .highlight_min(subset=["RMSE"], props="background-color: rgba(52,211,153,0.28); font-weight: 700;"))

        # ---------------- EKF vs PINN ----------------
        section("Physics-informed benchmark: ECM + EKF against hybrid PINN")
        with st.expander("Observer and PINN hyperparameters", expanded=False):
            h1, h2 = st.columns(2, gap="large")
            with h1:
                st.markdown("**ECM + EKF observer**")
                sigma_v = st.slider("σᵥ voltage noise (V)", 0.005, 0.200, 0.080, 0.005, format="%.3f")
                q_soh = st.select_slider("q_SOH random walk (per Ah)",
                                         [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
                                         value=5e-4, format_func=lambda v: f"{v:.0e}")
                q_r = st.select_slider("q_R random walk (fraction per Ah)",
                                       [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2],
                                       value=2e-3, format_func=lambda v: f"{v:.0e}")
                tau = st.slider("τ RC time constant (s)", 10, 300, 60, 10)
            with h2:
                st.markdown("**Hybrid PINN**")
                epochs = st.select_slider("Training epochs", [500, 1000, 1500, 2500, 4000], value=2500)
                lam_phys = st.select_slider("λ physics", [0.0, 0.1, 0.3, 1.0, 3.0, 10.0], value=1.0)
                lam_bv = st.select_slider("λ Butler–Volmer", [0.0, 0.1, 0.3, 1.0, 3.0], value=0.3)
                lam_eis = st.select_slider("λ EIS anchoring", [0.0, 0.1, 0.5, 1.0, 3.0], value=0.5)

        twin_params = te.TwinParameters(sigma_v=sigma_v, q_soh_per_ah=q_soh, q_r_frac_per_ah=q_r, tau_rc_s=float(tau))
        pinn_cfg = te.PINNConfig(epochs=int(epochs), lambda_phys=float(lam_phys),
                                 lambda_bv=float(lam_bv), lambda_eis=float(lam_eis))

        c1, c2, c3 = st.columns(3)
        frac2 = c1.slider("Forecast origin n₀ (fraction of life)", 0.2, 0.8, 0.4, 0.05, key="cmp_frac")
        ml_pick = c2.selectbox("ML reference model", list(te.ML_MODELS), index=1)
        reuse = c3.toggle("Reuse cached EKF replay", value=True,
                          help="Skips the full-telemetry replay when cell and observer settings are unchanged.")
        ekf_key = (store.key, cell, tuple(sorted(asdict(twin_params).items())))
        cmp_cfg = (ekf_key, frac2, ml_pick, tuple(sorted(asdict(pinn_cfg).items())), float(eol_ah))

        if st.button("Run benchmark", type="primary", key="cmp_go"):
            bar = st.progress(0.0, text="Initialising observer…")
            try:
                ekf_cached = st.session_state.get("ekf")
                ekf_res = ekf_cached["res"] if (reuse and ekf_cached and ekf_cached["key"] == ekf_key) else None
                raw = cell_frame(store, store.key, cell)
                res_new = te.compare_paradigms(raw, ct, imp, cell, frac2, twin_params, pinn_cfg, ml_pick, eol_ah,
                                               ekf=ekf_res, progress=lambda f, m: bar.progress(f, text=m))
                if res_new.ekf is not None:
                    st.session_state["ekf"] = {"key": ekf_key, "res": res_new.ekf}
                st.session_state["cmp"] = {"cfg": cmp_cfg, "res": res_new}
            except Exception as exc:
                report_error("Benchmark failed", exc, debug)
            finally:
                bar.empty()

        saved_cmp = st.session_state.get("cmp")
        if saved_cmp:
            res: te.ComparisonResult = saved_cmp["res"]
            if res.cell_id != cell:
                st.info(f"The stored benchmark belongs to {res.cell_id}. Run the benchmark to update it.")
            else:
                if saved_cmp["cfg"] != cmp_cfg:
                    st.warning("Settings changed since the last run; the results below use the previous settings.")
                for name, msg in res.errors.items():
                    st.warning(f"{name}: {msg}")

                mcols = st.columns(max(len(res.metrics), 1))
                for col, (name, m) in zip(mcols, res.metrics.items()):
                    delta = None if m.rul_error is None else f"RUL error {m.rul_error:+d} cycles"
                    col.metric(f"{name} RMSE", fmt(m.rmse, ".4f"), delta, delta_color="off")

                show(fig_compare(res, P), key="cmp_fig")
                with st.expander("Held-out metrics table"):
                    show_table(te.metrics_table(res.metrics).style.format(
                        {"RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.3f}", "RUL true": "{:.0f}",
                         "RUL pred": "{:.0f}", "RUL error": "{:+.0f}"}, na_rep="—"))

                a, b = st.columns(2, gap="medium")
                with a:
                    show(fig_params(res, P), key="cmp_params")
                with b:
                    if res.ekf is not None:
                        show(fig_innovation(res.ekf, float(res.ekf.params.get("sigma_v", sigma_v)), P),
                             key="cmp_innov")
                    if res.pinn is not None:
                        show(fig_pinn_loss(res.pinn, P), key="cmp_loss")

                if res.pinn is not None:
                    section("Identified electrochemical parameters (hybrid PINN)")
                    ph = res.pinn.physics
                    m_exp = ph["m_SEI_exponent"]
                    regime = ("self-limiting, SEI-like growth" if m_exp > 0.2 else
                              "self-accelerating, knee-like" if m_exp < -0.2 else "near-linear in throughput")
                    p1, p2 = st.columns(2, gap="medium")
                    with p1:
                        card("Degradation kinetics", [
                            f"Rate constant k = {ph['k_per_Ah']:.2e} per Ah",
                            f"Activation energy Eₐ = {ph['Ea_kJ_mol']:.1f} kJ mol⁻¹",
                            f"Fade exponent m = {m_exp:.2f} ({regime})",
                            f"Resistance coupling γ_int = {ph['gamma_int']:.2f}, γ_ct = {ph['gamma_ct']:.2f}"])
                    with p2:
                        card("Charge transfer (Butler–Volmer)", [
                            f"Exchange current i₀: {ph['i0_start_A']:.3f} A at start, {ph['i0_at_n0_A']:.3f} A at n₀",
                            f"Overpotential η_ct at 2 A: {ph['eta_ct_2A_start_mV']:.0f} mV to {ph['eta_ct_2A_n0_mV']:.0f} mV",
                            f"Lumped fast polarisation R_x = {ph['R_x_mOhm']:.1f} mΩ",
                            f"Training time {res.pinn.train_seconds:.1f} s"])
    except Exception as exc:
        report_error("Models tab failed", exc, debug)


# =============================================================================
# Tab 3: operations & optimal control
# =============================================================================
with tab_ops:
    try:
        section("Twin-aware operating strategy")
        st.markdown("The twin-aware policy predicts each available discharge current with the ECM and "
                    "degradation model, then picks the one with the highest expected revenue minus "
                    "weighted ageing cost, within the thermal and cold-weather limits.")
        c1, c2, c3 = st.columns(3)
        policy = c1.selectbox("Policy", POLICIES)
        weight = c2.slider("Degradation penalty weight w", 0.25, 3.0, 1.0, 0.25, disabled=policy != "Twin-Aware")
        replacement = c3.number_input("Replacement cost (CU)", 10.0, 2000.0, 150.0, 10.0)

        with st.expander("Economic and environmental constraints"):
            p1, p2, p3 = st.columns(3)
            price = (p1.number_input("Revenue per Ah at 1 A", 0.0, 10.0, 0.80, 0.05),
                     p2.number_input("Revenue per Ah at 2 A", 0.0, 10.0, 1.00, 0.05),
                     p3.number_input("Revenue per Ah at 4 A", 0.0, 10.0, 1.25, 0.05))
            s1, s2, s3, s4 = st.columns(4)
            t_max = s1.slider("Max cell temperature (°C)", 40, 60, 55)
            cold_rule = s2.toggle("Cold-temperature derating", value=True)
            cold_thr = s3.slider("Cold threshold (°C)", 0, 20, 10, disabled=not cold_rule)
            soh_eol_ops = s4.slider("Replacement SOH", 0.60, 0.85, 0.70, 0.01)
            e1, e2 = st.columns(2)
            amb_mean = e1.slider("Mean ambient (°C)", 0, 35, 20)
            amb_amp = e2.slider("Seasonal ambient amplitude (°C)", 0, 20, 16)
            ekf_saved = st.session_state.get("ekf")
            use_twin = st.toggle("Initialise physics from the latest EKF run", value=False,
                                 disabled=ekf_saved is None,
                                 help="Available after a benchmark run in the Models tab.")
            show_base = st.toggle("Overlay fixed-current baselines", value=True)

        econ = te.Economics(price_per_Ah=tuple(price), replacement_cost=float(replacement),
                            soh_eol=float(soh_eol_ops), degradation_weight=float(weight),
                            T_max_C=float(t_max), cold_derate_below_C=float(cold_thr) if cold_rule else None)
        phys = te.CellPhysics()
        if use_twin and ekf_saved:
            r = ekf_saved["res"]
            phys = te.CellPhysics(k_ah=float(r.params["k_ah"]), R_int0=float(r.r_int0), R_ct0=float(r.r_ct0))
        ops_cfg = (policy, tuple(sorted(asdict(econ).items())), tuple(sorted(asdict(phys).items())),
                   amb_mean, amb_amp, show_base)

        if st.button("Run lifecycle simulation", type="primary", key="ops_go"):
            with st.spinner("Simulating to end of life…"):
                df = run_policy(policy, asdict(econ), asdict(phys), float(amb_mean), float(amb_amp))
                base = {bname: run_policy(bname, asdict(econ), asdict(phys), float(amb_mean), float(amb_amp))
                        for bname in POLICIES[1:] if show_base and bname != policy}
                st.session_state["ops"] = {"cfg": ops_cfg, "df": df, "base": base, "policy": policy,
                                           "soh_eol": float(soh_eol_ops)}

        saved_ops = st.session_state.get("ops")
        if saved_ops:
            if saved_ops["cfg"] != ops_cfg:
                st.warning("Settings changed since the last run; the results below use the previous settings.")
            if saved_ops["df"].empty:
                st.warning("The simulation produced no cycles: the cell starts below the replacement SOH.")
            else:
                s = te.summarise_life(saved_ops["df"])
                k = st.columns(5)
                k[0].metric("Cycles to replacement", s["cycles"])
                k[1].metric("Net lifetime profit", fmt(s["profit"], ".1f", "CU"))
                k[2].metric("Profit rate", fmt(s["profit_per_h"], ".3f", "CU/h"))
                k[3].metric("Mean current", fmt(s["mean_I"], ".2f", "A"))
                k[4].metric("Constraint violations", s["violations"])

                show(fig_policy(saved_ops["df"], saved_ops["policy"], saved_ops["base"], saved_ops["soh_eol"], P),
                     key="ops_fig")

                rows = [{"Policy": saved_ops["policy"], **s}] + \
                       [{"Policy": n, **te.summarise_life(d)} for n, d in saved_ops["base"].items()]
                tbl = pd.DataFrame(rows).set_index("Policy")[["cycles", "Ah", "profit", "profit_per_h",
                                                              "mean_I", "violations"]]
                tbl.columns = ["Cycles", "Throughput (Ah)", "Profit (CU)", "Profit rate (CU/h)",
                               "Mean current (A)", "Violations"]
                show_table(tbl.style.format({"Throughput (Ah)": "{:.0f}", "Profit (CU)": "{:.1f}",
                                             "Profit rate (CU/h)": "{:.3f}", "Mean current (A)": "{:.2f}"},
                                            na_rep="—")
                           .highlight_max(subset=["Profit rate (CU/h)"],
                                          props="background-color: rgba(52,211,153,0.28); font-weight: 700;"))
    except Exception as exc:
        report_error("Operations tab failed", exc, debug)
