"""
app.py - Battery Digital Twin & Operando Diagnostics Platform (Streamlit front-end, v4)
=====================================================================================

UI layer only; all computation lives in ``twin_engine.py``.

Design rules enforced in this file
  * Theme resiliency - custom CSS never hard-codes a text colour. Body text inherits the
    active Streamlit theme; surfaces and borders are translucent neutrals. Plotly figures
    get an explicit light, dark or publication palette resolved from the active theme
    (``st.context.theme``, Streamlit >= 1.46) with a manual override in the sidebar.
  * Colour - Okabe-Ito colour-blind-safe palette; every paradigm is double-encoded with a
    line dash / marker symbol so figures survive greyscale printing.
  * Figures - one ``style_fig``: title in a reserved top band, legend centred in a reserved
    bottom band below the x-axis title (bordered, theme-matched), unified cross-hair hover.
    Forecasts always show their uncertainty band. Every key figure has HTML / SVG / CSV export.
  * Layout - global cell selector at the top; three views (data & diagnostics, models &
    forecasting, operations & control). Only the active view is executed (lazy navigation),
    and interactive sub-sections run as fragments so their widgets do not rerun the page.
  * Provenance - every result can be downloaded with a JSON run manifest.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
RESULTS_DIR = APP_DIR / "results"
MASTER_NAME, IMP_NAME = "battery_master_data.parquet", "impedance_ground_truth.parquet"
POLICIES = ["Twin-Aware", "Fixed 1 A", "Fixed 2 A", "Fixed 4 A"]
VIEWS = ["📊 Data & diagnostics", "🧠 Models & forecasting", "⚙️ Operations & control"]
SANS = "Inter, 'Source Sans Pro', 'Helvetica Neue', Arial, sans-serif"
SERIF = "'STIX Two Text', 'Times New Roman', Times, serif"


def _version_tuple(v: str) -> Tuple[int, int]:
    nums = [int(x) for x in re.findall(r"\d+", v)[:2]]
    nums += [0] * (2 - len(nums))
    return nums[0], nums[1]


ST_VERSION = _version_tuple(st.__version__)
PLOTLY_VERSION = _version_tuple(plotly.__version__)
LEGEND_CONTAINER_REF = PLOTLY_VERSION >= (5, 15)
try:
    import kaleido  # noqa: F401
    HAS_KALEIDO = True
except Exception:
    HAS_KALEIDO = False

# st.fragment (>= 1.37): widgets inside a fragment rerun only that fragment.
fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None) or (lambda f: f)


# =============================================================================
# Theme-aware, colour-blind-safe palettes (Okabe & Ito, 2008)
# =============================================================================
@dataclass(frozen=True)
class Palette:
    mode: str
    template: str
    font: str
    text: str
    muted: str
    grid: str
    axis: str
    paper_bg: str
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
    semi: str = "#009E73"
    pf: str = "#F0E442"
    mode_colors: Tuple[str, ...] = ("#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#D55E00")

    def current_color(self, amps: float) -> str:
        return dict(self.currents).get(float(amps), self.muted)


# Double encoding: each paradigm / series also gets its own dash and marker.
DASHES = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")
SYMBOLS = ("circle", "square", "diamond", "triangle-up", "cross", "star")
CURRENT_SYMBOL = {1.0: "circle", 2.0: "square", 4.0: "diamond"}
PARADIGM_DASH = {"twin": "dash", "pinn": "solid", "ml": "dashdot"}

DARK = Palette(
    mode="dark", template="plotly_dark", font=SANS,
    text="#f1f5f9", muted="#cbd5e1", grid="rgba(148,163,184,0.16)", axis="#64748b",
    paper_bg="rgba(0,0,0,0)", plot_bg="rgba(148,163,184,0.05)", hover_bg="#0b1220",
    legend_bg="rgba(11,18,32,0.90)", legend_border="#64748b",
    accent="#56B4E9", measured="#f8fafc", cohort="rgba(148,163,184,0.38)",
    ekf="#56B4E9", pinn="#CC79A7", eis="#E69F00", eol="#D55E00",
    r_int="#E69F00", r_ct="#009E73",
    currents=((1.0, "#56B4E9"), (2.0, "#009E73"), (4.0, "#D55E00")),
    series=("#E69F00", "#009E73", "#F0E442", "#D55E00", "#CC79A7"),
    ica_range=(0.15, 0.95),
)

LIGHT = Palette(
    mode="light", template="plotly_white", font=SANS,
    text="#0f172a", muted="#334155", grid="rgba(15,23,42,0.09)", axis="#94a3b8",
    paper_bg="rgba(0,0,0,0)", plot_bg="rgba(15,23,42,0.02)", hover_bg="#ffffff",
    legend_bg="rgba(255,255,255,0.95)", legend_border="#94a3b8",
    accent="#0072B2", measured="#111827", cohort="rgba(100,116,139,0.35)",
    ekf="#0072B2", pinn="#CC79A7", eis="#E69F00", eol="#D55E00",
    r_int="#E69F00", r_ct="#009E73",
    currents=((1.0, "#0072B2"), (2.0, "#009E73"), (4.0, "#D55E00")),
    # yellow (#F0E442) has too little contrast on white; use black-ish grey instead.
    series=("#E69F00", "#009E73", "#D55E00", "#56B4E9", "#555555"),
    ica_range=(0.0, 0.82),
    pf="#555555", mode_colors=("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"),
)

# Publication style: opaque white background, serif type, black axes (journal figures).
PUBLICATION = replace(
    LIGHT, mode="publication", font=SERIF, text="#000000", muted="#222222",
    grid="rgba(0,0,0,0.08)", axis="#000000", paper_bg="#ffffff", plot_bg="#ffffff",
    legend_bg="#ffffff", legend_border="#000000", cohort="rgba(0,0,0,0.25)")


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


def resolve_palette(choice: str, publication: bool) -> Palette:
    if publication:
        return PUBLICATION
    mode = choice.lower() if choice in ("Light", "Dark") else detect_base_theme()
    return DARK if mode == "dark" else LIGHT


def inject_css(P: Palette) -> None:
    """Text colours are inherited from the Streamlit theme; only accents use the palette."""
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
.bt-hero .bt-sub {{margin-top: 6px; font-size: 0.98rem; line-height: 1.5; opacity: 0.82; color: inherit; max-width: 84ch;}}
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


def nav(options: Sequence[str], key: str) -> str:
    """Lazy view switcher: only the returned view is executed on each run."""
    choice, rendered = None, False
    seg = getattr(st, "segmented_control", None)
    if seg is not None:
        try:
            choice = seg("View", list(options), default=options[0], key=key, label_visibility="collapsed")
            rendered = True
        except TypeError:
            rendered = False
    if not rendered:
        choice = st.radio("View", list(options), horizontal=True, key=key, label_visibility="collapsed")
    if choice is None:                     # segmented control deselected: keep the last view
        choice = st.session_state.get(f"{key}_last", options[0])
    st.session_state[f"{key}_last"] = choice
    return choice


def download(label: str, data: Any, file_name: str, mime: str, key: str) -> None:
    """Download button that does not rerun the app when supported (on_click='ignore')."""
    try:
        st.download_button(label, data, file_name=file_name, mime=mime, key=key, on_click="ignore")
    except TypeError:
        st.download_button(label, data, file_name=file_name, mime=mime, key=key)


def manifest_button(config: Dict[str, Any], key: str, data_key: str, extra: Optional[Dict[str, Any]] = None) -> None:
    man = te.run_manifest(config, data_key, extra)
    download("Run manifest (JSON)", json.dumps(man, indent=2), f"{key}_manifest.json", "application/json",
             key=f"man_{key}")


PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "svg", "scale": 1},
}


def show(fig: go.Figure, key: str, data: Optional[pd.DataFrame] = None, export: bool = True) -> None:
    """Render with Plotly styling intact (theme=None) across Streamlit versions, plus an
    export row: interactive HTML, SVG / PDF (when kaleido is installed) and the plotted data."""
    base = dict(theme=None, config=PLOT_CONFIG)
    attempts: List[Dict[str, Any]] = []
    if ST_VERSION >= (1, 50):
        attempts.append(dict(width="stretch", key=key))
    attempts += [dict(use_container_width=True, key=key), dict(use_container_width=True)]
    for extra in attempts:
        try:
            st.plotly_chart(fig, **base, **extra)
            break
        except TypeError:
            continue
    if export:
        export_row(fig, key, data)


def show_selectable(fig: go.Figure, key: str) -> Optional[Any]:
    """Chart with point selection (Streamlit >= 1.35). Returns the selection event or None."""
    cfg = dict(PLOT_CONFIG, modeBarButtonsToRemove=["lasso2d"])
    for extra in ([dict(width="stretch")] if ST_VERSION >= (1, 50) else []) + [dict(use_container_width=True)]:
        try:
            return st.plotly_chart(fig, theme=None, config=cfg, key=key, on_select="rerun",
                                   selection_mode="points", **extra)
        except TypeError:
            continue
    show(fig, key, export=False)
    return None


def export_row(fig: go.Figure, key: str, data: Optional[pd.DataFrame]) -> None:
    holder = getattr(st, "popover", None)
    ctx = holder("Export", use_container_width=False) if holder else st.expander("Export")
    with ctx:
        download("Interactive HTML", fig.to_html(include_plotlyjs="cdn", full_html=True), f"{key}.html",
                 "text/html", key=f"dl_html_{key}")
        if HAS_KALEIDO:
            try:
                download("SVG (vector)", fig.to_image(format="svg"), f"{key}.svg", "image/svg+xml",
                         key=f"dl_svg_{key}")
                download("PDF (vector)", fig.to_image(format="pdf"), f"{key}.pdf", "application/pdf",
                         key=f"dl_pdf_{key}")
            except Exception as exc:                     # kaleido needs a browser runtime
                st.caption(f"Static export unavailable: {exc}")
        else:
            st.caption("Install `kaleido` for SVG / PDF export (the modebar camera saves SVG too).")
        if data is not None and len(data):
            download("Plotted data (CSV)", data.to_csv(index=False), f"{key}.csv", "text/csv",
                     key=f"dl_csv_{key}")


def show_table(data: Any) -> None:
    if ST_VERSION >= (1, 50):
        try:
            st.dataframe(data, width="stretch")
            return
        except TypeError:
            pass
    st.dataframe(data, use_container_width=True)


def paradigm_style(name: str, P: Palette, i: int = 0) -> Tuple[str, str, str]:
    """(colour, dash, marker) for a forecast series - colour AND dash encode the paradigm."""
    if name.startswith("ECM twin"):
        return P.ekf, PARADIGM_DASH["twin"], "square"
    if name == te.PINN_NAME:
        return P.pinn, PARADIGM_DASH["pinn"], "diamond"
    if name == te.SEMI_NAME:
        return P.semi, "dot", "triangle-up"
    if name == te.PF_NAME:
        return P.pf, "longdash", "star"
    return P.series[i % len(P.series)], DASHES[(i + 3) % len(DASHES)], SYMBOLS[i % len(SYMBOLS)]


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
        ann.font = dict(size=13, color=P.text, family=P.font)


def style_fig(fig: go.Figure, P: Palette, height: int = 460, title: Optional[str] = None,
              width_hint: int = 1100, hovermode: str = "x unified") -> go.Figure:
    """Publication layout: reserved title band on top, reserved legend band at the bottom
    (below the x-axis title), unified cross-hair hover."""
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
        paper_bgcolor=P.paper_bg, plot_bgcolor=P.plot_bg,
        font=dict(family=P.font, size=13, color=P.text),
        margin=dict(l=76, r=30, t=top, b=bottom, pad=4),
        legend=dict(orientation="h", xanchor="center", x=0.5,
                    bgcolor=P.legend_bg, bordercolor=P.legend_border, borderwidth=1,
                    font=dict(family=P.font, size=12, color=P.text),
                    itemsizing="constant", **legend_pos),
        hovermode=hovermode,
        hoverlabel=dict(bgcolor=P.hover_bg, bordercolor=P.legend_border, namelength=-1,
                        font=dict(family=P.font, size=12, color=P.text)),
    )
    if title:
        fig.update_layout(title=dict(text=f"<b>{html.escape(title)}</b>", x=0.012, xanchor="left",
                                     y=1 - 16 / H, yanchor="top",
                                     font=dict(family=P.font, size=16, color=P.text)))
    axis = dict(gridcolor=P.grid, zerolinecolor=P.grid, linecolor=P.axis, showline=True, mirror=True,
                ticks="outside", tickcolor=P.axis, title_standoff=10,
                title_font=dict(family=P.font, size=13, color=P.text),
                tickfont=dict(family=P.font, size=12, color=P.muted))
    fig.update_xaxes(**axis, showspikes=True, spikemode="across", spikesnap="cursor",
                     spikethickness=1, spikedash="dot", spikecolor=P.muted)
    fig.update_yaxes(**axis)
    return fig


def _forecast_frame(fig: go.Figure, n0: int, n_max: float, soh_eol: float, P: Palette,
                    row: Optional[int] = None) -> None:
    kw = dict(row=row, col=1) if row else {}
    fig.add_vrect(x0=n0, x1=n_max, fillcolor=rgba(P.accent, 0.06), line_width=0, layer="below", **kw)
    fig.add_vline(x=n0, line_dash="dot", line_color=P.accent, line_width=1.5, **kw)
    fig.add_hline(y=soh_eol, line_dash="dash", line_color=P.eol, line_width=1.5,
                  annotation_text="EOL", annotation_position="bottom left",
                  annotation_font=dict(color=P.eol, size=12), **kw)


def _origin_label(fig: go.Figure, n0: int, P: Palette) -> None:
    fig.add_annotation(x=n0, xref="x", y=1.0, yref="paper", xanchor="left", yanchor="bottom",
                       text=f"<b> forecast origin n₀ = {n0}</b>", showarrow=False,
                       font=dict(color=P.accent, size=12, family=P.font))


def add_band(fig: go.Figure, n: np.ndarray, lo: np.ndarray, hi: np.ndarray, color: str, name: str,
             n_from: Optional[float] = None, group: Optional[str] = None, showlegend: bool = True,
             row: Optional[int] = None, col: Optional[int] = None, alpha: float = 0.16) -> None:
    n, lo, hi = np.asarray(n, float), np.asarray(lo, float), np.asarray(hi, float)
    m = np.isfinite(lo) & np.isfinite(hi)
    if n_from is not None:
        m &= n >= n_from
    if not m.any():
        return
    x = np.r_[n[m], n[m][::-1]]
    y = np.r_[hi[m], lo[m][::-1]]
    fig.add_trace(go.Scatter(x=x, y=y, fill="toself", fillcolor=rgba(color, alpha), line=dict(width=0),
                             name=name, legendgroup=group, showlegend=showlegend, hoverinfo="skip"),
                  row=row, col=col)


# =============================================================================
# Figures: data & diagnostics
# =============================================================================
def fig_fade(ct_all: pd.DataFrame, cell_id: str, y: str, eol_line: Optional[float], P: Palette,
             knee: Optional[Dict[str, Any]] = None) -> go.Figure:
    fig = go.Figure()
    first = True
    for cid, d in ct_all[~ct_all["outlier"]].groupby("Cell_ID"):
        if cid == cell_id:
            continue
        fig.add_trace(go.Scatter(x=d["n"], y=d[y], mode="lines", line=dict(color=P.cohort, width=1.3),
                                 name="Cohort cells", legendgroup="cohort", showlegend=first,
                                 hoverinfo="skip"))
        first = False
    d = ct_all[ct_all["Cell_ID"] == cell_id]
    good, bad = d[~d["outlier"]], d[d["outlier"]]
    fig.add_trace(go.Scatter(x=good["n"], y=good[y], mode="lines+markers", name=f"Target cell {cell_id}",
                             line=dict(color=P.accent, width=3), marker=dict(size=6, color=P.accent),
                             customdata=good[["Cycle_Index"]].to_numpy(),
                             hovertemplate="%{y:.4f} · cycle %{customdata[0]}<extra></extra>"))
    if "regen" in d.columns:
        rg = good[good["regen"].astype(bool)]
        if len(rg):
            fig.add_trace(go.Scatter(x=rg["n"], y=rg[y], mode="markers", name="Capacity regeneration",
                                     marker=dict(symbol="triangle-up", size=12, color=P.r_ct,
                                                 line=dict(color=P.text, width=1)),
                                     hovertemplate="%{y:.4f} (regeneration)<extra></extra>"))
    if len(bad):
        fig.add_trace(go.Scatter(x=bad["n"], y=bad[y], mode="markers", name="Flagged outliers",
                                 marker=dict(symbol="x", size=9, color=P.eol, line=dict(width=2)),
                                 hovertemplate="%{y:.4f} (outlier)<extra></extra>"))
    if eol_line is not None:
        fig.add_hline(y=eol_line, line_dash="dash", line_color=P.eol, line_width=1.5,
                      annotation_text="End of life", annotation_position="bottom right",
                      annotation_font=dict(color=P.eol, size=12))
    if knee and knee.get("found"):
        kn = int(knee["knee_n"])
        ky = float(good.loc[(good["n"] - kn).abs().idxmin(), y]) if len(good) else None
        fig.add_vline(x=kn, line_dash="dashdot", line_color=P.pinn, line_width=1.5)
        fig.add_trace(go.Scatter(x=[kn], y=[ky], mode="markers", name="Knee point (non-linear ageing onset)",
                                 marker=dict(symbol="star", size=16, color=P.pinn, line=dict(color=P.text, width=1)),
                                 hovertemplate=f"knee at n = {kn}<extra></extra>"))
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="Discharge capacity (Ah)" if y == "Capacity_Ah" else "State of health SOH (–)")
    fig.update_layout(clickmode="event+select")
    return style_fig(fig, P, 470, f"Capacity fade trajectory, {cell_id} against cohort")


def fig_cohort_grid(ct_all: pd.DataFrame, meta: pd.DataFrame, cell_id: str, P: Palette) -> go.Figure:
    """Small multiples: one panel per ambient temperature, colour + marker = discharge current."""
    amb = meta["Ambient_C"].round(0)
    groups = sorted(amb.dropna().unique())
    ncol = min(4, max(len(groups), 1))
    nrow = math.ceil(len(groups) / ncol) if groups else 1
    fig = make_subplots(rows=nrow, cols=ncol, shared_yaxes=True, shared_xaxes=False,
                        horizontal_spacing=0.04, vertical_spacing=0.16,
                        subplot_titles=[f"Ambient {g:.0f} °C" for g in groups])
    _style_subplot_titles(fig, P)
    seen = set()
    good = ct_all[~ct_all["outlier"]]
    for gi, g in enumerate(groups):
        r, c = gi // ncol + 1, gi % ncol + 1
        for cid in amb[amb == g].index:
            d = good[good["Cell_ID"] == cid]
            I = float(round(meta.loc[cid, "I_dis_A"]))
            is_t = cid == cell_id
            label = f"{I:g} A discharge"
            col = P.accent if is_t else P.current_color(I)
            fig.add_trace(go.Scatter(
                x=d["n"], y=d["SOH"], mode="lines+markers",
                line=dict(color=col, width=3.2 if is_t else 1.4),
                marker=dict(symbol=CURRENT_SYMBOL.get(I, "circle"), size=4, maxdisplayed=12),
                name=f"Target {cid}" if is_t else label, legendgroup="target" if is_t else label,
                showlegend=(is_t or label not in seen), hovertemplate=f"{cid}: %{{y:.3f}}<extra></extra>"),
                row=r, col=c)
            if not is_t:
                seen.add(label)
        fig.update_xaxes(title_text="Cycle n" if r == nrow else None, row=r, col=c)
    fig.update_yaxes(title_text="SOH (–)", col=1)
    return style_fig(fig, P, 300 + 230 * nrow, "Cohort fade by ambient temperature", hovermode="closest")


def fig_cycle_trace(prep: pd.DataFrame, cycle_index: int, n: int, P: Palette) -> go.Figure:
    d = prep[prep["Cycle_Index"] == cycle_index]
    t = d["Time_s"] / 60.0
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("Terminal voltage", "Current", "Cell temperature"))
    _style_subplot_titles(fig, P)
    for row, col, key, unit, color in ((1, 1, "Voltage_V", "V", P.accent), (2, 1, "Current_A", "A", P.r_ct),
                                       (3, 1, "Temp_C", "°C", P.eis)):
        fig.add_trace(go.Scatter(x=t, y=d[key], mode="lines", line=dict(color=color, width=2.2),
                                 name=key.split("_")[0], showlegend=False,
                                 hovertemplate=f"%{{y:.3f}} {unit}<extra></extra>"), row=row, col=col)
    fig.update_yaxes(title_text="V", row=1, col=1)
    fig.update_yaxes(title_text="A", row=2, col=1)
    fig.update_yaxes(title_text="°C", row=3, col=1)
    fig.update_xaxes(title_text="Time in cycle (min)", row=3, col=1)
    return style_fig(fig, P, 560, f"Raw telemetry of discharge n = {n} (cycle index {cycle_index})",
                     width_hint=720)


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
                             line=dict(color=P.accent, width=2.5), marker=dict(size=7, symbol="circle"),
                             hovertemplate="%{y:.3f} V<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=n, y=[c.peak_height for c in curves], mode="lines+markers", name="Peak height",
                             line=dict(color=P.pinn, width=2.5, dash="dash"), marker=dict(size=7, symbol="diamond"),
                             hovertemplate="%{y:.2f} Ah/V<extra></extra>"), secondary_y=True)
    fig.update_xaxes(title_text="Discharge cycle n")
    style_fig(fig, P, 340, "Main peak evolution", width_hint=620)
    fig.update_yaxes(title_text="Peak voltage (V)", title_font_color=P.accent, secondary_y=False)
    fig.update_yaxes(title_text="Peak height (Ah V⁻¹)", title_font_color=P.pinn, showgrid=False,
                     mirror=False, secondary_y=True)
    return fig


# =============================================================================
# Figures: models & forecasting
# =============================================================================
def fig_ml(results: List[te.MLForecast], ct_cell: pd.DataFrame, n0: int, soh_eol: float,
           P: Palette) -> go.Figure:
    fig = go.Figure()
    good = ct_cell[~ct_cell["outlier"]]
    for i, r in enumerate(results):
        if r.soh_lo is not None:
            col, _, _ = paradigm_style(f"ML · {r.model}", P, i)
            add_band(fig, r.n_grid, r.soh_lo, r.soh_hi, col, f"{r.model} {int(100 * (r.band_level or 0.9))}% band",
                     n_from=n0, group=r.model, alpha=0.12)
    fig.add_trace(go.Scatter(x=good["n"], y=good["SOH"], mode="markers", name="Measured SOH",
                             marker=dict(color=P.measured, size=6, opacity=0.85),
                             hovertemplate="%{y:.4f}<extra></extra>"))
    for i, r in enumerate(results):
        col, dash, sym = paradigm_style(f"ML · {r.model}", P, i)
        fig.add_trace(go.Scatter(x=r.n_grid, y=r.soh_pred, mode="lines+markers", name=r.model, legendgroup=r.model,
                                 line=dict(color=col, width=2.6, dash=dash),
                                 marker=dict(symbol=sym, size=7, maxdisplayed=10),
                                 hovertemplate="%{y:.4f}<extra></extra>"))
    n_max = max(float(r.n_grid.max()) for r in results)
    _forecast_frame(fig, n0, n_max, soh_eol, P)
    _origin_label(fig, n0, P)
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="State of health SOH (–)")
    return style_fig(fig, P, 520, "ML surrogate SOH forecasts with cross-cell conformal bands")


def fig_compare(res: te.ComparisonResult, P: Palette) -> go.Figure:
    fig = go.Figure()
    for i, (name, (n_g, lo, hi)) in enumerate(res.bands.items()):
        col, _, _ = paradigm_style(name, P, i)
        add_band(fig, n_g, lo, hi, col, f"{name} {int(100 * res.band_level)}% band", n_from=res.n0,
                 group=name, alpha=0.13)
    m = res.measured
    fig.add_trace(go.Scatter(x=m["n"], y=m["SOH"], mode="markers", name="Measured SOH",
                             marker=dict(color=P.measured, size=6, opacity=0.85),
                             hovertemplate="%{y:.4f}<extra></extra>"))
    if res.ekf is not None:
        e = res.ekf.per_cycle.dropna(subset=["SOH"])
        add_band(fig, e["n"], e["SOH"] - 2 * e["SOH_std"], e["SOH"] + 2 * e["SOH_std"], P.ekf,
                 "Observer ±2σ", group="obs", alpha=0.10)
        fig.add_trace(go.Scatter(x=e["n"], y=e["SOH"], mode="lines", name="Observer causal estimate",
                                 legendgroup="obs", line=dict(color=P.ekf, width=1.8, dash="dot"),
                                 hovertemplate="%{y:.4f}<extra></extra>"))
    for i, (name, (n_g, p_g)) in enumerate(res.predictions.items()):
        col, dash, sym = paradigm_style(name, P, i)
        mask = n_g >= res.n0
        fig.add_trace(go.Scatter(x=n_g[mask], y=p_g[mask], mode="lines+markers", name=f"{name} forecast",
                                 legendgroup=name, line=dict(color=col, width=3, dash=dash),
                                 marker=dict(symbol=sym, size=8, maxdisplayed=8),
                                 hovertemplate="%{y:.4f}<extra></extra>"))
    n_max = max(float(n.max()) for n, _ in res.predictions.values()) if res.predictions else float(m["n"].max())
    _forecast_frame(fig, res.n0, n_max, res.soh_eol, P)
    _origin_label(fig, res.n0, P)
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="State of health SOH (–)")
    return style_fig(fig, P, 560, f"Forecast benchmark on {res.cell_id} with uncertainty bands")


def fig_rul_hist(samples: Dict[str, np.ndarray], rul_true: Optional[int], lb: Optional[int], P: Palette) -> go.Figure:
    """Overlaid RUL distributions of every probabilistic paradigm (twin MC, semi-empirical MC,
    particle filter), with the true RUL or its censoring bound."""
    fig = go.Figure()
    s_max = 0.0
    notes = []
    for i, (name, smp) in enumerate(samples.items()):
        smp = np.asarray(smp, dtype=float)
        s = smp[np.isfinite(smp)]
        col, _, _ = paradigm_style(name, P, i)
        miss = 1 - len(s) / max(len(smp), 1)
        if miss > 0.005:
            notes.append(f"{name.split(' · ')[0]} {100 * miss:.0f}% beyond horizon")
        if not len(s):
            continue
        s_max = max(s_max, float(s.max()))
        fig.add_trace(go.Histogram(x=s, nbinsx=40, histnorm="probability density", name=name,
                                   marker=dict(color=rgba(col, 0.45), line=dict(color=col, width=1)),
                                   hovertemplate="%{x} cycles<extra>" + html.escape(name) + "</extra>"))
        med = float(np.median(s))
        fig.add_vline(x=med, line_color=col, line_width=2.2, line_dash="dot")
    if rul_true is not None:
        fig.add_vline(x=rul_true, line_color=P.eol, line_dash="dash", line_width=2.5,
                      annotation_text=f"true {rul_true}", annotation_position="top left",
                      annotation_font=dict(color=P.eol, size=12))
    elif lb is not None:
        fig.add_vrect(x0=lb, x1=max(s_max, lb) + 1, fillcolor=rgba(P.eol, 0.08), line_width=0,
                      annotation_text=f"censored: true RUL > {lb}", annotation_position="top left",
                      annotation_font=dict(color=P.eol, size=12))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="Remaining useful life after n₀ (cycles)")
    fig.update_yaxes(title_text="Probability density")
    title = "RUL distributions (dotted = median)" + (f" · {'; '.join(notes)}" if notes else "")
    return style_fig(fig, P, 400, title, width_hint=720, hovermode="closest")


def fig_params(res: te.ComparisonResult, P: Palette) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.13,
                        subplot_titles=("Ohmic resistance R_int", "Charge-transfer resistance R_ct"))
    _style_subplot_titles(fig, P)
    for row, key, eis_col, label in ((1, "R_int", "Re_ohm", "Re"), (2, "R_ct", "Rct_ohm", "Rct")):
        show_leg = row == 1
        if res.ekf is not None:
            e = res.ekf.per_cycle.dropna(subset=[key])
            fig.add_trace(go.Scatter(x=e["n"], y=1e3 * e[key], mode="lines", name="Observer", legendgroup="obs",
                                     showlegend=show_leg, line=dict(color=P.ekf, width=2.4, dash="dash"),
                                     hovertemplate="%{y:.1f} mΩ<extra></extra>"), row=row, col=1)
        if res.pinn is not None:
            fig.add_trace(go.Scatter(x=res.pinn.n_grid, y=1e3 * getattr(res.pinn, key.lower()), mode="lines",
                                     name="PINN", legendgroup="pinn", showlegend=show_leg,
                                     line=dict(color=P.pinn, width=2.4),
                                     hovertemplate="%{y:.1f} mΩ<extra></extra>"), row=row, col=1)
        if len(res.eis):
            ci = te._int_key(res.measured.sort_values("Cycle_Index")[["Cycle_Index", "n"]])
            ee = pd.merge_asof(te._int_key(res.eis).sort_values("Cycle_Index"), ci, on="Cycle_Index",
                               direction="backward")
            fig.add_trace(go.Scatter(x=ee["n"].fillna(1), y=1e3 * ee[eis_col], mode="markers", name="EIS",
                                     legendgroup="eis", showlegend=show_leg,
                                     marker=dict(color=P.eis, symbol="x", size=9, line=dict(width=1.5)),
                                     hovertemplate=f"%{{y:.1f}} mΩ (EIS {label})<extra></extra>"), row=row, col=1)
        fig.add_vline(x=res.n0, line_dash="dot", line_color=P.accent, row=row, col=1)
        fig.update_yaxes(title_text="Resistance (mΩ)", row=row, col=1)
    fig.update_xaxes(title_text="Discharge cycle n", row=2, col=1)
    return style_fig(fig, P, 580, "Identified resistances against EIS", width_hint=720)


def fig_innovation(ekf: te.EKFResult, sigma_v: float, P: Palette) -> go.Figure:
    """Voltage innovation (top) and normalised innovation squared NIS / dof (bottom).
    A consistent filter has NIS / dof ~ 1: well below 1 means over-conservative noise
    settings, well above 1 means over-confidence or model error."""
    e = ekf.per_cycle.dropna(subset=["innov_mean_mV"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, row_heights=[0.55, 0.45],
                        subplot_titles=("Voltage innovation V_meas − V_pred", "Filter consistency NIS / dof"))
    _style_subplot_titles(fig, P)
    fig.add_hrect(y0=-2e3 * sigma_v, y1=2e3 * sigma_v, fillcolor=rgba(P.r_ct, 0.10), line_width=0, row=1, col=1)
    add_band(fig, e["n"], e["innov_mean_mV"] - e["innov_std_mV"], e["innov_mean_mV"] + e["innov_std_mV"],
             P.ekf, "±1 std within cycle", row=1, col=1, alpha=0.18)
    fig.add_trace(go.Scatter(x=e["n"], y=e["innov_mean_mV"], mode="lines", name="Mean innovation",
                             line=dict(color=P.ekf, width=2.4), hovertemplate="%{y:.1f} mV<extra></extra>"),
                  row=1, col=1)
    if "NIS_norm" in ekf.per_cycle.columns:
        nis = ekf.per_cycle.dropna(subset=["NIS_norm"])
        fig.add_trace(go.Scatter(x=nis["n"], y=nis["NIS_norm"], mode="markers", name="NIS / dof",
                                 marker=dict(color=P.pinn, size=5, symbol="diamond"),
                                 hovertemplate="%{y:.2f}<extra></extra>"), row=2, col=1)
        fig.add_hline(y=1.0, line_dash="dash", line_color=P.muted, row=2, col=1,
                      annotation_text="consistent = 1", annotation_font=dict(color=P.muted, size=11))
        fig.update_yaxes(type="log", title_text="NIS / dof", row=2, col=1)
    fig.update_yaxes(title_text="mV", row=1, col=1)
    fig.update_xaxes(title_text="Discharge cycle n", row=2, col=1)
    return style_fig(fig, P, 520, "Observer innovations and consistency", width_hint=720)


def fig_pinn_loss(p: te.PINNResult, P: Palette) -> go.Figure:
    fig = go.Figure()
    spec = (("total", P.text, "Total", "solid"), ("data", P.r_ct, "Data", "dash"),
            ("phys", P.pinn, "Physics", "dot"), ("bv", P.accent, "Butler–Volmer", "dashdot"),
            ("eis", P.eis, "EIS anchor", "longdash"))
    for key, col, label, dash in spec:
        if key in p.history.columns:
            fig.add_trace(go.Scatter(x=p.history["epoch"], y=p.history[key], mode="lines", name=label,
                                     line=dict(color=col, width=3 if key == "total" else 1.8, dash=dash),
                                     hovertemplate="%{y:.3e}<extra></extra>"))
    fig.update_yaxes(type="log", title_text="Loss (normalised, log)")
    fig.update_xaxes(title_text="Epoch")
    return style_fig(fig, P, 400, "Hybrid PINN training losses", width_hint=720)


def fig_ablation(results: Dict[str, te.EKFResult], ct_cell: pd.DataFrame, P: Palette) -> go.Figure:
    good = ct_cell[~ct_cell["outlier"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=good["n"], y=good["SOH"], mode="markers", name="Measured SOH",
                             marker=dict(color=P.measured, size=5, opacity=0.8),
                             hovertemplate="%{y:.4f}<extra></extra>"))
    cols = (P.eol, P.eis, P.r_ct, P.ekf)
    for i, (name, r) in enumerate(results.items()):
        e = r.per_cycle
        fig.add_trace(go.Scatter(x=e["n"], y=e["SOH"], mode="lines", name=name,
                                 line=dict(color=cols[i % len(cols)], width=2.4, dash=DASHES[i % len(DASHES)]),
                                 hovertemplate="%{y:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="Causal SOH estimate (–)")
    return style_fig(fig, P, 460, "Measurement ablation: what each signal adds to the observer")


# ---------------------------------------------------------------- benchmark --
def _bench_ok(bench: pd.DataFrame) -> pd.DataFrame:
    return bench[bench["error"].isna()] if "error" in bench.columns else bench


def fig_bench_parity(bench: pd.DataFrame, alpha: float, P: Palette) -> go.Figure:
    b = _bench_ok(bench).dropna(subset=["rul_true", "rul_pred"])
    fig = go.Figure()
    top = float(max(b["rul_true"].max(), b["rul_pred"].max(), 10)) if len(b) else 100.0
    x = np.array([0, top])
    add_band(fig, x, x * (1 - alpha), x * (1 + alpha), P.muted, f"±{int(alpha * 100)}% accuracy cone",
             alpha=0.12)
    fig.add_trace(go.Scatter(x=x, y=x, mode="lines", line=dict(color=P.muted, dash="dash", width=1.2),
                             name="Perfect prediction", hoverinfo="skip"))
    for i, (par, d) in enumerate(b.groupby("paradigm")):
        col, _, sym = paradigm_style(par, P, i)
        fig.add_trace(go.Scatter(x=d["rul_true"], y=d["rul_pred"], mode="markers", name=par,
                                 marker=dict(color=col, symbol=sym, size=9, line=dict(color=P.text, width=0.6)),
                                 customdata=d[["cell", "n0"]].to_numpy(),
                                 hovertemplate="%{customdata[0]} · n₀ %{customdata[1]}<br>true %{x}, pred %{y}"
                                               "<extra></extra>"))
    fig.update_xaxes(title_text="True RUL after n₀ (cycles)")
    fig.update_yaxes(title_text="Predicted RUL (cycles)")
    return style_fig(fig, P, 480, "RUL parity (uncensored forecasts)", width_hint=720, hovermode="closest")


def fig_alpha_lambda(bench: pd.DataFrame, alpha: float, P: Palette) -> go.Figure:
    """Normalised alpha-lambda plot: RUL_pred / RUL_true against the fraction of life elapsed
    lambda = n0 / EOL_true; the alpha cone is 1 +/- alpha."""
    b = _bench_ok(bench).dropna(subset=["rul_true", "rul_pred", "eol_true"])
    b = b[b["rul_true"] > 0]
    fig = go.Figure()
    add_band(fig, np.array([0.0, 1.0]), np.array([1 - alpha] * 2), np.array([1 + alpha] * 2), P.muted,
             f"α = {alpha:.2f} cone", alpha=0.12)
    fig.add_hline(y=1.0, line_dash="dash", line_color=P.muted, line_width=1)
    for i, (par, d) in enumerate(b.groupby("paradigm")):
        col, dash, sym = paradigm_style(par, P, i)
        lam = d["n0"] / d["eol_true"]
        ratio = d["rul_pred"] / d["rul_true"]
        fig.add_trace(go.Scatter(x=lam, y=ratio, mode="markers", name=par,
                                 marker=dict(color=col, symbol=sym, size=9, line=dict(color=P.text, width=0.6)),
                                 customdata=d[["cell"]].to_numpy(),
                                 hovertemplate="%{customdata[0]}: λ %{x:.2f}, ratio %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Fraction of life elapsed λ = n₀ / EOL", range=[0, 1])
    fig.update_yaxes(title_text="RUL_pred / RUL_true", type="log")
    return style_fig(fig, P, 480, "α-λ accuracy across cells", width_hint=720, hovermode="closest")


def fig_horizon(cov: pd.DataFrame, level: float, P: Palette) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Band coverage vs horizon", "RMSE vs horizon"))
    _style_subplot_titles(fig, P)
    for i, (par, d) in enumerate(cov.groupby("paradigm")):
        col, dash, sym = paradigm_style(par, P, i)
        d = d.sort_values("h_bin")
        if d["coverage"].notna().any():
            fig.add_trace(go.Scatter(x=d["h_bin"], y=d["coverage"], mode="lines+markers", name=par, legendgroup=par,
                                     line=dict(color=col, dash=dash, width=2.4), marker=dict(symbol=sym, size=8),
                                     hovertemplate="%{y:.2f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=d["h_bin"], y=d["rmse"], mode="lines+markers", name=par, legendgroup=par,
                                 showlegend=not d["coverage"].notna().any(),
                                 line=dict(color=col, dash=dash, width=2.4), marker=dict(symbol=sym, size=8),
                                 hovertemplate="%{y:.4f}<extra></extra>"), row=1, col=2)
    fig.add_hline(y=level, line_dash="dash", line_color=P.eol, row=1, col=1,
                  annotation_text=f"nominal {int(level * 100)}%", annotation_font=dict(color=P.eol, size=11))
    fig.update_yaxes(title_text="Empirical coverage", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="SOH RMSE", row=1, col=2)
    fig.update_xaxes(title_text="Forecast horizon h (cycles)")
    return style_fig(fig, P, 440, "Calibration and error growth over the forecast horizon")


# =============================================================================
# Figures: operations
# =============================================================================
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
                                     name=f"{amps:g} A selected",
                                     marker=dict(color=P.current_color(amps), size=7,
                                                 symbol=CURRENT_SYMBOL.get(amps, "circle")),
                                     hovertemplate="%{y:g} A<extra></extra>"),
                          row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df["cycle"], y=df["T_amb"], mode="lines", name="Ambient temperature",
                             line=dict(color=P.muted, width=1.5, dash="dot"),
                             hovertemplate="%{y:.1f} °C<extra></extra>"), row=1, col=1, secondary_y=True)
    for j, (name, d) in enumerate({policy: df, **baselines}.items()):
        if name == policy:
            style = dict(color=P.text, width=3)
        else:
            style = dict(color=P.current_color(float(name.split()[1])), width=1.8, dash=DASHES[1 + j % 4])
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


def fig_mismatch(ms: pd.DataFrame, P: Palette) -> go.Figure:
    tw = ms[ms["policy"] == "Twin-Aware"]
    fig = go.Figure()
    fig.add_hline(y=0, line_color=P.muted, line_dash="dash", line_width=1)
    for key, name, col, sym in (("advantage_pct", "vs best compliant fixed policy", P.ekf, "circle"),
                                ("advantage_vs_any_pct", "vs best fixed policy incl. violators", P.eis, "square")):
        fig.add_trace(go.Box(x=tw["level"], y=tw[key], name=name, marker=dict(color=col, symbol=sym, size=8),
                             line=dict(color=col), boxpoints="all", jitter=0.3, pointpos=0,
                             hovertemplate="%{y:+.1f}%<extra></extra>"))
    fig.update_layout(boxmode="group")
    fig.update_xaxes(title_text="Plant / model mismatch level (relative std of perturbed physics)")
    fig.update_yaxes(title_text="Twin-aware profit-rate advantage (%)")
    return style_fig(fig, P, 460, "Robustness of the twin-aware policy to model error", hovermode="closest")


# =============================================================================
# Figures: health indicators, degradation modes, stress (Mission 1)
# =============================================================================
HI_CRITERIA = ("|ρ| with SOH", "Monotonicity", "Trendability", "Prognosability")


def fig_hi_rank(tab: pd.DataFrame, P: Palette) -> go.Figure:
    """Grouped horizontal bars: the four prognostic-parameter criteria per indicator."""
    t = tab.sort_values("Fitness")
    fig = go.Figure()
    for i, crit in enumerate(HI_CRITERIA):
        fig.add_trace(go.Bar(y=t["Indicator"], x=t[crit], orientation="h", name=crit,
                             marker=dict(color=P.mode_colors[i % len(P.mode_colors)],
                                         pattern=dict(shape=("", "/", ".", "x")[i], fgcolor=P.text, size=5,
                                                      solidity=0.15)),
                             hovertemplate="%{x:.2f}<extra>" + crit + "</extra>"))
    fig.add_trace(go.Scatter(y=t["Indicator"], x=t["Fitness"], mode="markers", name="Fitness (mean)",
                             marker=dict(symbol="diamond", size=12, color=P.text, line=dict(color=P.plot_bg, width=1)),
                             hovertemplate="fitness %{x:.2f}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_xaxes(title_text="Criterion score (0 – 1, higher is better)", range=[0, 1.05])
    fig.update_yaxes(title_text=None, automargin=True)
    return style_fig(fig, P, 120 + 42 * len(t), "Which parameter best represents health? Prognostic-parameter criteria",
                     hovermode="closest")


def fig_hi_traj(traj: pd.DataFrame, P: Palette, cell_id: str) -> go.Figure:
    fig = go.Figure()
    for i, (k, d) in enumerate(traj.groupby("key", sort=False)):
        hi = te.HI_CATALOG[k]
        fig.add_trace(go.Scatter(x=d["n"], y=d["rel"], mode="lines", name=f"{hi.label} ({hi.mode})",
                                 line=dict(color=P.series[i % len(P.series)], width=2.4, dash=DASHES[i % len(DASHES)]),
                                 hovertemplate="%{y:.3f}<extra>" + html.escape(hi.label) + "</extra>"))
    fig.add_hline(y=1.0, line_color=P.muted, line_dash="dot", line_width=1)
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="Value / beginning-of-life value")
    return style_fig(fig, P, 440, f"Top health indicators on {cell_id}, normalised to beginning of life")


def fig_pca(pca: Dict[str, Any], P: Palette) -> go.Figure:
    ev = pca["explained"][: min(6, len(pca["explained"]))]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58], horizontal_spacing=0.14,
                        subplot_titles=("Variance explained by each component", "Loadings of PC1 and PC2"))
    _style_subplot_titles(fig, P)
    fig.add_trace(go.Bar(x=[f"PC{i + 1}" for i in range(len(ev))], y=100 * ev, name="Explained variance",
                         marker=dict(color=P.accent), hovertemplate="%{y:.1f}%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[f"PC{i + 1}" for i in range(len(ev))], y=100 * np.cumsum(ev), mode="lines+markers",
                             name="Cumulative", line=dict(color=P.eol, dash="dash"), marker=dict(symbol="square"),
                             hovertemplate="%{y:.1f}%<extra></extra>"), row=1, col=1)
    L = pca["loadings"]
    labels = [te.HI_CATALOG[k].label for k in L.index]
    for j, pc in enumerate([c for c in L.columns[:2]]):
        fig.add_trace(go.Bar(y=labels, x=L[pc], orientation="h", name=f"{pc} loading",
                             marker=dict(color=P.mode_colors[(j + 1) % len(P.mode_colors)],
                                         pattern=dict(shape=("", "/")[j], fgcolor=P.text, solidity=0.15)),
                             hovertemplate="%{x:.2f}<extra>" + pc + "</extra>"), row=1, col=2)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="%", range=[0, 105], row=1, col=1)
    fig.update_xaxes(title_text="Loading", row=1, col=2)
    fig.update_yaxes(automargin=True, row=1, col=2)
    return style_fig(fig, P, 460, "Can degradation be represented by one parameter? PCA of the health indicators",
                     hovermode="closest")


def fig_dva(curves: List[te.DVACurve], P: Palette) -> go.Figure:
    fig = go.Figure()
    ns = [c.n for c in curves]
    cols = sample_colorscale("Viridis", list(np.linspace(*P.ica_range, len(curves))))
    for c, col in zip(curves, cols):
        fig.add_trace(go.Scatter(x=c.q, y=c.dvdq, mode="lines", name=f"n = {c.n}", line=dict(color=col, width=2.2),
                                 hovertemplate="%{x:.2f} Ah: %{y:.2f} V/Ah<extra>n = " + str(c.n) + "</extra>"))
    fig.update_xaxes(title_text="Discharged capacity Q (Ah)")
    fig.update_yaxes(title_text="|dV/dQ| (V/Ah)", type="log")
    return style_fig(fig, P, 440, f"Differential voltage analysis across life (n = {min(ns)} to {max(ns)})")


def fig_modes(modes: pd.DataFrame, P: Palette) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Capacity-loss modes (% of initial capacity)", "Conductivity loss (% growth)"))
    _style_subplot_titles(fig, P)
    for i, col in enumerate(["Capacity loss", "LLI (proxy)", "LAM (proxy)"]):
        if col in modes:
            fig.add_trace(go.Scatter(x=modes["n"], y=modes[col], mode="lines+markers", name=col,
                                     line=dict(color=P.mode_colors[i], width=2.6, dash=DASHES[i]),
                                     marker=dict(symbol=SYMBOLS[i], size=8),
                                     hovertemplate="%{y:.1f}%<extra>" + col + "</extra>"), row=1, col=1)
    for j, col in enumerate([c for c in modes.columns if c.startswith("CL")]):
        fig.add_trace(go.Scatter(x=modes["n"], y=modes[col], mode="lines+markers", name=col,
                                 line=dict(color=P.mode_colors[3 + j % 2], width=2.6, dash=DASHES[3 + j % 2]),
                                 marker=dict(symbol=SYMBOLS[3 + j % 2], size=8),
                                 hovertemplate="%{y:.1f}%<extra>" + col + "</extra>"), row=1, col=2)
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="%", row=1, col=1)
    fig.update_yaxes(title_text="%", row=1, col=2)
    return style_fig(fig, P, 440, "Degradation-mode trajectories: LLI · LAM · conductivity loss (CL)")


def fig_stress(sx: pd.DataFrame, limits: te.SafetyLimits, P: Palette, cell_id: str) -> go.Figure:
    d = sx[(sx["Cell_ID"] == cell_id) & ~sx["outlier"]].sort_values("n")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("Peak cell temperature per discharge", "Minimum discharge voltage",
                                        "Plating-risk index during charge"))
    _style_subplot_titles(fig, P)
    fig.add_trace(go.Scatter(x=d["n"], y=d["T_max_C"], mode="lines+markers", name="T_max",
                             line=dict(color=P.eol, width=2), marker=dict(size=5),
                             hovertemplate="%{y:.1f} °C<extra></extra>"), row=1, col=1)
    fig.add_hline(y=limits.T_warn_C, line_dash="dot", line_color=P.eis, row=1, col=1,
                  annotation_text=f"accelerated SEI growth ≥ {limits.T_warn_C:.0f} °C",
                  annotation_font=dict(color=P.eis, size=11))
    fig.add_hline(y=limits.T_crit_C, line_dash="dash", line_color=P.eol, row=1, col=1,
                  annotation_text=f"SEI decomposition onset ≈ {limits.T_crit_C:.0f} °C",
                  annotation_font=dict(color=P.eol, size=11))
    fig.add_trace(go.Scatter(x=d["n"], y=d["V_min_V"], mode="lines+markers", name="V_min",
                             line=dict(color=P.accent, width=2), marker=dict(size=5, symbol="square"),
                             hovertemplate="%{y:.3f} V<extra></extra>"), row=2, col=1)
    fig.add_hline(y=limits.V_deep_V, line_dash="dash", line_color=P.eol, row=2, col=1,
                  annotation_text=f"deep discharge < {limits.V_deep_V:.1f} V",
                  annotation_font=dict(color=P.eol, size=11))
    fig.add_trace(go.Bar(x=d["n"], y=d["PRI"], name="Plating-risk index", marker=dict(color=P.r_ct),
                         hovertemplate="%{y:.2f}<extra></extra>"), row=3, col=1)
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="V", row=2, col=1)
    fig.update_yaxes(title_text="PRI", row=3, col=1)
    fig.update_xaxes(title_text="Discharge cycle n", row=3, col=1)
    return style_fig(fig, P, 720, f"Operating-stress and safety exposure, {cell_id}", hovermode="x unified")


# =============================================================================
# Figures: Mission 2 and Mission 3 studies
# =============================================================================
def fig_update_freq(tab: pd.DataFrame, results: Dict[int, te.EKFResult], ct_cell: pd.DataFrame, P: Palette) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42], horizontal_spacing=0.12,
                        subplot_titles=("Causal SOH estimate by update interval", "Accuracy versus measurement cost"))
    _style_subplot_titles(fig, P)
    good = ct_cell[~ct_cell["outlier"]]
    fig.add_trace(go.Scatter(x=good["n"], y=good["SOH"], mode="markers", name="Measured SOH",
                             marker=dict(color=P.measured, size=5, opacity=0.8),
                             hovertemplate="%{y:.4f}<extra></extra>"), row=1, col=1)
    for i, (m, r) in enumerate(results.items()):
        e = r.per_cycle
        fig.add_trace(go.Scatter(x=e["n"], y=e["SOH"], mode="lines", name=f"every {m} cycle(s)",
                                 line=dict(color=P.series[i % len(P.series)], width=2, dash=DASHES[i % len(DASHES)]),
                                 hovertemplate="%{y:.4f}<extra>every " + str(m) + "</extra>"), row=1, col=1)
    x = tab["Updates per 100 cycles"]
    fig.add_trace(go.Scatter(x=x, y=tab["Tracking RMSE"], mode="lines+markers+text", name="Tracking RMSE",
                             text=[f"m={m}" for m in tab.index], textposition="top center",
                             textfont=dict(color=P.muted, size=11),
                             line=dict(color=P.ekf, width=2.4), marker=dict(size=9),
                             hovertemplate="%{y:.4f}<extra>tracking RMSE</extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=tab["Forecast RMSE"], mode="lines+markers", name="Forecast RMSE from n₀",
                             line=dict(color=P.pinn, width=2.4, dash="dash"), marker=dict(symbol="diamond", size=9),
                             hovertemplate="%{y:.4f}<extra>forecast RMSE</extra>"), row=1, col=2)
    fig.update_xaxes(title_text="Discharge cycle n", row=1, col=1)
    fig.update_xaxes(title_text="Measurement updates per 100 cycles", type="log", row=1, col=2)
    fig.update_yaxes(title_text="SOH (–)", row=1, col=1)
    fig.update_yaxes(title_text="SOH RMSE", row=1, col=2)
    return style_fig(fig, P, 480, "How often should the twin update?")


def fig_om_heatmap(study: pd.DataFrame, opt: Dict[str, Any], P: Palette) -> go.Figure:
    piv = study.pivot(index="policy", columns="threshold", values="rate")
    viol = study.pivot(index="policy", columns="threshold", values="violations")
    pf = study.pivot(index="policy", columns="threshold", values="p_failure")
    order = sorted(piv.index, key=lambda s: (s.startswith("Fixed"), s))
    piv, viol, pf = piv.loc[order], viol.loc[order], pf.loc[order]
    txt = [[("✕ " if viol.loc[r, c] > 0 else "") + f"{piv.loc[r, c]:.3f}" for c in piv.columns] for r in piv.index]
    custom = np.dstack([pf.to_numpy(), viol.to_numpy()])
    fig = go.Figure(go.Heatmap(
        z=piv.to_numpy(), x=[f"{c:.2f}" for c in piv.columns], y=piv.index, text=txt, texttemplate="%{text}",
        textfont=dict(size=11), customdata=custom, colorscale="RdBu", zmid=0,
        colorbar=dict(title=dict(text="CU/h", font=dict(color=P.text)), tickfont=dict(color=P.muted)),
        hovertemplate="%{y} · replace at SOH %{x}<br>profit rate %{z:.4f} CU/h<br>"
                      "P(sudden failure) %{customdata[0]:.2f} · violations %{customdata[1]}<extra></extra>"))
    b = opt.get("best")
    if b:
        fig.add_trace(go.Scatter(x=[f"{b['threshold']:.2f}"], y=[b["policy"]], mode="markers", name="Integrated optimum",
                                 marker=dict(symbol="star", size=22, color=P.pf if P.mode == "dark" else "#F0E442",
                                             line=dict(color=P.text, width=1.5)),
                                 hoverinfo="skip"))
    fig.update_xaxes(title_text="Replacement threshold (SOH)", showspikes=False)
    fig.update_yaxes(title_text=None, automargin=True, showspikes=False)
    return style_fig(fig, P, 160 + 38 * len(order),
                     "Integrated optimisation: long-run profit rate by operating policy × maintenance threshold",
                     hovermode="closest")


def fig_om_tradeoff(study: pd.DataFrame, P: Palette) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Profit rate versus replacement threshold", "Sudden-failure probability per life"))
    _style_subplot_titles(fig, P)
    for i, (pol, d) in enumerate(study.groupby("policy", sort=False)):
        d = d.sort_values("threshold")
        is_fixed = pol.startswith("Fixed")
        col = P.current_color(float(pol.split()[1])) if is_fixed else P.series[i % len(P.series)]
        style = dict(color=col, width=1.6 if is_fixed else 2.6, dash="dot" if is_fixed else DASHES[i % len(DASHES)])
        fig.add_trace(go.Scatter(x=d["threshold"], y=d["rate"], mode="lines+markers", name=pol, legendgroup=pol,
                                 line=style, marker=dict(symbol=SYMBOLS[i % len(SYMBOLS)], size=7),
                                 hovertemplate="%{y:.4f} CU/h<extra>" + html.escape(pol) + "</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=d["threshold"], y=d["p_failure"], mode="lines+markers", name=pol, legendgroup=pol,
                                 showlegend=False, line=style, marker=dict(symbol=SYMBOLS[i % len(SYMBOLS)], size=7),
                                 hovertemplate="%{y:.2f}<extra>" + html.escape(pol) + "</extra>"), row=1, col=2)
    fig.update_xaxes(title_text="Replacement threshold (SOH)")
    fig.update_yaxes(title_text="CU/h", row=1, col=1)
    fig.update_yaxes(title_text="P(failure before replacement)", range=[0, 1.02], row=1, col=2)
    return style_fig(fig, P, 470, "Maintenance trade-off: replace early (cost) or late (risk and performance loss)")


# =============================================================================
# Cached data access & computation
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_store(path: str, mtime: float) -> te.ParquetStore:
    return te.ParquetStore(path)


@st.cache_resource(show_spinner=False)
def demo_data() -> Tuple[te.ParquetStore, pd.DataFrame]:
    """Synthetic cohort with known ground truth (offline demo / reviewer mode)."""
    master, imp, _ = te.make_synthetic_master(
        n_cells=8, n_cycles=120, ambients=(24, 34, 43, 4, 24, 34, 43, 24),
        currents=(2, 2, 2, 2, 1, 4, 4, 2), k_true=(4e-4, 5e-4, 3.5e-4, 5e-4, 6e-4, 4.5e-4, 5e-4, 5.5e-4), seed=7)
    return te.ParquetStore.from_dataframe(master), imp


@st.cache_resource(show_spinner=False)
def persist_upload(name: str, size: int, _raw: bytes) -> str:
    return str(te.persist_upload(_raw, name))


@st.cache_data(show_spinner=False)
def load_impedance_path(path: str, mtime: float) -> pd.DataFrame:
    return te.load_impedance(path)


@st.cache_data(show_spinner=False, max_entries=16)
def cycle_table(_store: te.ParquetStore, store_key: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    return te.build_cycle_table_with_report(_store)


@st.cache_resource(show_spinner=False, max_entries=3)
def cell_frame(_store: te.ParquetStore, store_key: str, cell: str) -> pd.DataFrame:
    return _store.cell_frame(cell)


@st.cache_resource(show_spinner=False, max_entries=3)
def prepared_cell(_store: te.ParquetStore, store_key: str, cell: str) -> pd.DataFrame:
    return te.prepare_cell(cell_frame(_store, store_key, cell))


@st.cache_data(show_spinner=False, max_entries=8)
def validation_cached(_store: te.ParquetStore, store_key: str, cell: str) -> List[str]:
    return te.validate_master(cell_frame(_store, store_key, cell))


@st.cache_data(show_spinner=False, max_entries=32)
def ica_cached(_prep: pd.DataFrame, _ct_cell: pd.DataFrame, key: str, cell: str,
               n_curves: int, ir: bool, dv: float, window: int) -> List[te.ICACurve]:
    return te.ica_evolution(_prep, _ct_cell, n_curves, ir_compensate=ir, dv=dv, smooth_window=window)


@st.cache_data(show_spinner=False, max_entries=64)
def ml_cached(_ct: pd.DataFrame, key: str, cell: str, n0: int, model: str, use_pop: bool, eol_ah: float,
              strategy: str, conformal_cells: int, level: float) -> te.MLForecast:
    return te.train_ml_forecast(_ct, cell, n0, model, use_population=use_pop, eol_ah=eol_ah, strategy=strategy,
                                conformal_cells=conformal_cells, band_level=level)


@st.cache_data(show_spinner=False, max_entries=8)
def pooled_ea_cached(_ct: pd.DataFrame, key: str, min_ambient: float) -> Dict[str, Any]:
    return te.estimate_pooled_arrhenius(_ct, min_ambient_C=min_ambient)


@st.cache_data(show_spinner=False, max_entries=4)
def hi_rank_cached(_ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str) -> pd.DataFrame:
    return te.rank_health_indicators(_ct, _imp)


@st.cache_data(show_spinner=False, max_entries=4)
def hi_pca_cached(_ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str) -> Dict[str, Any]:
    return te.hi_pca(_ct, _imp)


@st.cache_data(show_spinner=False, max_entries=4)
def stress_cached(_ct: pd.DataFrame, key: str, limits: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    L = te.SafetyLimits(**limits)
    return te.stress_exposure(_ct, L), te.stress_summary(_ct, L)


@st.cache_data(show_spinner=False, max_entries=4)
def stress_factors_cached(_ct: pd.DataFrame, key: str) -> Dict[str, Any]:
    return te.stress_factor_regression(_ct)


@st.cache_data(show_spinner=False, max_entries=32)
def knee_cached(_ct_cell: pd.DataFrame, key: str, cell: str) -> Dict[str, Any]:
    g = _ct_cell[~_ct_cell["outlier"]]
    return te.detect_knee(g["n"].to_numpy(), g["SOH"].to_numpy())


@st.cache_data(show_spinner=False, max_entries=16)
def modes_cached(_prep: pd.DataFrame, _ct_cell: pd.DataFrame, _eis: pd.DataFrame, key: str, cell: str,
                 n_curves: int) -> Tuple[List[te.DVACurve], pd.DataFrame]:
    dva = te.dva_evolution(_prep, _ct_cell, n_curves)
    ica = te.ica_evolution(_prep, _ct_cell, n_curves)
    return dva, te.degradation_modes(ica, _ct_cell, _eis)


@st.cache_data(show_spinner=False, max_entries=8)
def om_study_cached(econ: Dict[str, Any], phys: Dict[str, Any], maint: Dict[str, Any], amb_mean: float,
                    amb_amp: float, plant: Optional[Dict[str, Any]], weights: Tuple[float, ...],
                    thresholds: Tuple[float, ...]) -> pd.DataFrame:
    return te.integrated_om_study(te.Economics(**econ), te.CellPhysics(**phys), te.MaintenanceModel(**maint),
                                  weights=weights, thresholds=thresholds, ambient_mean_C=amb_mean,
                                  ambient_amp_C=amb_amp, plant=te.CellPhysics(**plant) if plant else None)


@st.cache_data(show_spinner=False, max_entries=64)
def run_policy(policy: str, econ: Dict[str, Any], phys: Dict[str, Any], amb_mean: float, amb_amp: float,
               plant: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    return te.simulate_life(te.make_policy(policy), te.CellPhysics(**phys), te.Economics(**econ),
                            ambient_mean_C=amb_mean, ambient_amp_C=amb_amp,
                            plant=te.CellPhysics(**plant) if plant else None)


@st.cache_data(show_spinner=False, max_entries=16)
def tune_rho_cached(econ: Dict[str, Any], phys: Dict[str, Any], amb_mean: float, amb_amp: float
                    ) -> Tuple[float, List[float]]:
    e, path = te.tune_rate_reference(te.CellPhysics(**phys), te.Economics(**econ),
                                     ambient_mean_C=amb_mean, ambient_amp_C=amb_amp)
    return float(e.rate_ref or 0.0), path


@st.cache_data(show_spinner=False)
def load_bench_file(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def read_uploaded_table(up: Any) -> pd.DataFrame:
    raw = up.getvalue()
    import io
    return pd.read_parquet(io.BytesIO(raw)) if up.name.endswith(".parquet") else pd.read_csv(io.BytesIO(raw))


def resolve_sources(up_master: Any, up_imp: Any, master_url: str, imp_url: str, master_sha: str
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
        master_path = te.download_to_cache(master_url, progress=lambda f, m: bar.progress(f, text=f"Telemetry {m}"),
                                           sha256=master_sha or None)
        bar.empty()
        notes.append("Telemetry: remote URL (cached" + (", SHA-256 verified)" if master_sha else ")"))
    if master_path is not None:
        store = get_store(str(master_path), master_path.stat().st_mtime)
    elif st.session_state.get("demo"):
        store, imp = demo_data()
        notes.append("Telemetry: synthetic demo cohort (known ground truth)")

    if up_imp is not None:
        imp = te.load_impedance(up_imp.getvalue())
        notes.append(f"EIS: uploaded {up_imp.name}")
    elif imp is None:
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
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("### Data sources")
    up_master = st.file_uploader("Master telemetry (.parquet)", type=["parquet"], key="up_master")
    up_imp = st.file_uploader("EIS ground truth (.parquet)", type=["parquet"], key="up_imp")
    with st.expander("Remote URLs", expanded=False):
        master_url = st.text_input("Telemetry URL", value=_secret("MASTER_PARQUET_URL"))
        master_sha = st.text_input("Telemetry SHA-256 pin (optional)", value=_secret("MASTER_PARQUET_SHA256"),
                                   help="Full-file digest; the download is rejected if it differs.")
        imp_url = st.text_input("EIS URL", value=_secret("IMPEDANCE_PARQUET_URL"))
    if st.session_state.get("demo"):
        if st.button("Leave synthetic demo"):
            st.session_state["demo"] = False
            st.rerun()

    st.markdown("### Appearance")
    theme_choice = st.radio("Figure palette", ["Auto", "Light", "Dark"], horizontal=True,
                            help="Auto follows the active Streamlit theme (Settings → Theme).")
    publication = st.toggle("Publication style", value=False,
                            help="White background, serif type and black axes, for export into papers.")
    debug = st.toggle("Show tracebacks on errors", value=False)

P = resolve_palette(theme_choice, publication)
inject_css(P)

# =============================================================================
# Hero
# =============================================================================
pills = "".join(f'<span class="bt-pill">{t}</span>' for t in
                ("Health-indicator ranking", "LLI · LAM · CL modes", "Dual time-scale EKF twin",
                 "Particle filter", "Semi-empirical law", "Hybrid PINN ensemble", "Conformal ML bands",
                 "Integrated O&M optimisation"))
st.markdown(
    '<div class="bt-hero">'
    '<div class="bt-title">🔋 Battery Digital Twin &amp; Operando Diagnostics</div>'
    '<div class="bt-sub">NASA Ames 18650 LiCoO₂ / graphite ageing telemetry: incremental capacity analysis, '
    'ML surrogates, a self-updating ECM twin and a physics-informed neural network, benchmarked across cells '
    'and forecast origins with calibrated uncertainty.</div>'
    f'<div>{pills}</div></div>',
    unsafe_allow_html=True,
)

# =============================================================================
# Data ingestion
# =============================================================================
store: Optional[te.ParquetStore] = None
imp: Optional[pd.DataFrame] = None
ct: Optional[pd.DataFrame] = None
cell_errors: Dict[str, str] = {}
source_notes: List[str] = []
try:
    store, imp, source_notes = resolve_sources(up_master, up_imp, master_url.strip(), imp_url.strip(),
                                               master_sha.strip())
    if store is not None:
        with st.spinner("Extracting per-cycle features…"):
            ct, cell_errors = cycle_table(store, store.key)
except Exception as exc:
    report_error("Data ingestion failed", exc, verbose=True)

if store is None or ct is None:
    st.info("No telemetry loaded. Place `battery_master_data.parquet` in `./data/`, upload it in the sidebar, "
            "set a remote URL, or explore the platform with a synthetic cohort whose true parameters are known.")
    if st.button("Load synthetic demo cohort", type="primary"):
        st.session_state["demo"] = True
        st.rerun()
    st.stop()

meta = te.cell_meta(ct)
cells = list(meta.index)
DATA_KEY = store.key


def _cell_label(cid: str) -> str:
    row = meta.loc[cid]
    return (f"{cid}  ({fmt(row['Ambient_C'], '.0f', '°C')}, {fmt(row['I_dis_A'], '.1f', 'A')}, "
            f"{int(row['cycles'])} cycles)")


# =============================================================================
# Global control bar
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
        if cell_errors:
            chips += f'<span class="bt-chip">⚠ {len(cell_errors)} cell(s) skipped</span>'
        st.markdown(
            f'<div class="bt-status">Analysing <b>{html.escape(cell)}</b>: '
            f'SOH<sub>EOL</sub> = {soh_eol:.3f} ({eol_ah:.2f} Ah of {c_bol:.3f} Ah initial)</div>'
            f'<div>{chips}</div>',
            unsafe_allow_html=True,
        )

view = nav(VIEWS, key="view")


# =============================================================================
# View 1: data & diagnostics
# =============================================================================
@fragment
def fade_explorer() -> None:
    """Fade chart linked to the raw telemetry: select a point to open that discharge."""
    yvar = st.radio("Metric", ["SOH", "Capacity_Ah"], horizontal=True, key="fade_metric",
                    format_func=lambda v: "State of health" if v == "SOH" else "Capacity (Ah)")
    fig = fig_fade(ct, cell, yvar, soh_eol if yvar == "SOH" else eol_ah, P, knee_cached(ct_cell, DATA_KEY, cell))
    event = show_selectable(fig, key=f"fade_{cell}")
    good = ct_cell[~ct_cell["outlier"]]
    export_row(fig, "fade", ct[["Cell_ID", "n", "Cycle_Index", "SOH", "Capacity_Ah", "outlier", "regen"]])
    picked_n: Optional[int] = None
    try:
        pts = event.selection.points if event is not None else []
        if pts:
            picked_n = int(round(float(pts[0]["x"])))
    except Exception:
        picked_n = None
    if picked_n is None:
        st.caption("Click a point on the target-cell trace to inspect that discharge's raw telemetry.")
        picked_n = int(st.select_slider("…or pick a discharge cycle", options=good["n"].tolist(),
                                        value=int(good["n"].iloc[0]), key=f"trace_n_{cell}"))
    row = ct_cell[ct_cell["n"] == picked_n]
    if not row.empty:
        ci = int(row["Cycle_Index"].iloc[0])
        prep = prepared_cell(store, DATA_KEY, cell)
        show(fig_cycle_trace(prep, ci, picked_n, P), key=f"trace_{cell}",
             data=prep[prep["Cycle_Index"] == ci][["Time_s", "Voltage_V", "Current_A", "Temp_C"]])


@fragment
def ica_section() -> None:
    c1, c2, c3, c4 = st.columns(4)
    n_curves = c1.slider("Curves across life", 2, 12, 6)
    ir = c2.toggle("IR-compensate voltage", value=True,
                   help="Adds |I|·R_dc to the terminal voltage to remove the ohmic shift.")
    dv = c3.select_slider("Voltage bin (mV)", [5, 10, 15, 20], value=10) / 1000
    win = c4.select_slider("Savitzky–Golay window", [5, 7, 9, 11, 15, 21], value=9)
    with st.spinner("Computing dQ/dV curves…"):
        prep = prepared_cell(store, DATA_KEY, cell)
        curves = ica_cached(prep, ct_cell, DATA_KEY, cell, n_curves, ir, dv, win)
    if not curves:
        st.warning("Not enough voltage resolution in this cell's discharges to compute dQ/dV.")
        return
    a, b = st.columns([3, 2], gap="medium")
    with a:
        data = pd.concat([pd.DataFrame({"n": c.n, "V": c.voltage, "dQdV": c.dqdv}) for c in curves])
        show(fig_ica(curves, P), key="ica", data=data)
    with b:
        show(fig_peaks(curves, P), key="ica_peaks", export=False)
        diag = te.diagnose_degradation(curves)
        if diag["available"]:
            card(f"Degradation reading, n = {diag['n_ref']} to n = {diag['n_cur']}",
                 [f"Capacity ratio {diag['cap_ratio']:.2f}, peak height ratio "
                  f"{diag['height_ratio']:.2f}, peak shift {diag['shift_mV']:.0f} mV"]
                 + list(diag["interpretation"]))
    st.caption("NASA discharges run near 1C, so curves are not near-equilibrium: peak broadening partly "
               "reflects kinetic overpotential and the mode reading is indicative only.")


@fragment
def health_indicator_section() -> None:
    st.markdown("Every candidate health indicator is normalised to its beginning-of-life value and scored with the "
                "prognostic-parameter criteria of Coble & Hines: association with SOH, monotonicity, trendability "
                "(same direction in every cell) and prognosability (cells end at similar values). *LOCO RMSE* asks "
                "whether the indicator alone can stand in for SOH on a cell it has never seen.")
    with st.spinner("Scoring health indicators across the cohort…"):
        tab = hi_rank_cached(ct, imp, DATA_KEY)
        pca = hi_pca_cached(ct, imp, DATA_KEY)
    if tab.empty:
        st.info("Not enough cells with 20+ valid cycles to rank health indicators.")
        return
    show(fig_hi_rank(tab, P), key="hi_rank", data=tab.reset_index())
    show_table(tab.drop(columns=["Unit"]).style.format(
        {"|ρ| with SOH": "{:.2f}", "Monotonicity": "{:.2f}", "Trendability": "{:.2f}", "Prognosability": "{:.2f}",
         "LOCO RMSE (SOH)": "{:.4f}", "Fitness": "{:.2f}"}, na_rep="—")
        .highlight_max(subset=["Fitness"], props="background-color: rgba(0,158,115,0.25); font-weight: 700;"))
    non_cap = tab.drop(index=[k for k in ("Capacity_Ah", "t_dis_s", "Q_ch_Ah", "t_cc_s", "E_dis_Wh")
                              if k in tab.index], errors="ignore")
    best_power = non_cap.index[0] if len(non_cap) else None
    a, b = st.columns([1, 1], gap="medium")
    with a:
        top = [k for k in tab.index[:3]] + ([best_power] if best_power and best_power not in tab.index[:3] else [])
        traj = te.hi_trajectories(ct, imp, cell, top)
        if len(traj):
            show(fig_hi_traj(traj, P, cell), key="hi_traj", data=traj)
    with b:
        if pca.get("available"):
            show(fig_pca(pca, P), key="hi_pca", export=False)
    if pca.get("available"):
        ev = pca["explained"]
        verdict = ("a single health parameter captures the ageing of this cohort" if pca["single_parameter"] else
                   "ageing is multi-dimensional: capacity fade and power fade (resistance) evolve partly "
                   "independently, which supports a multi-parameter state θ = [SOH, R_int, R_ct]")
        card("Mission 1 answer", [
            f"Best-ranked indicator: {tab.iloc[0]['Indicator']} (fitness {tab.iloc[0]['Fitness']:.2f}); "
            f"best power-fade indicator: {te.HI_CATALOG[best_power].label if best_power else '—'}",
            f"PC1 explains {100 * ev[0]:.0f}% of indicator variance (|corr| with SOH {pca['pc_soh_corr'][0]:.2f}); "
            f"PC2 explains {100 * ev[1]:.0f}% → {verdict}."])


@fragment
def modes_section() -> None:
    st.markdown("The white-box view of ageing (Birkl et al. 2017; Menye et al. 2025) splits capacity and power fade "
                "into **loss of lithium inventory** (SEI/CEI growth, plating), **loss of active material** (particle "
                "cracking, transition-metal dissolution, delamination) and **conductivity loss** (interphase and "
                "contact resistance). DVA complements ICA: features that move together indicate LLI, while shrinking "
                "distances between DVA peaks indicate LAM on the electrode that owns them.")
    n_curves = st.slider("Curves across life", 3, 10, 6, key="modes_n")
    with st.spinner("Computing DVA and mode trajectories…"):
        prep = prepared_cell(store, DATA_KEY, cell)
        dva, modes = modes_cached(prep, ct_cell, eis_cell, DATA_KEY, cell, n_curves)
    a, b = st.columns([1, 1], gap="medium")
    with a:
        if dva:
            data = pd.concat([pd.DataFrame({"n": c.n, "Q_Ah": c.q, "dVdQ": c.dvdq}) for c in dva])
            show(fig_dva(dva, P), key="dva", data=data)
        else:
            st.info("Not enough voltage resolution for DVA on this cell.")
    with b:
        if len(modes):
            show(fig_modes(modes, P), key="modes", data=modes)
    if len(modes):
        last = modes.iloc[-1]
        dom = "LLI" if last["LLI (proxy)"] >= last["LAM (proxy)"] else "LAM"
        card(f"Mode reading at n = {int(last['n'])}", [
            f"Capacity loss {last['Capacity loss']:.1f}% = LLI ≈ {last['LLI (proxy)']:.1f}% + LAM ≈ "
            f"{last['LAM (proxy)']:.1f}% → dominant capacity-loss mode: {dom}",
            f"Conductivity loss: load-step resistance {last['CL: R_dc growth']:+.0f}%"
            + (f", EIS Rₑ+R_ct {last['CL: EIS Rₑ+R_ct growth']:+.0f}%" if "CL: EIS Rₑ+R_ct growth" in modes else "")
            + f"; ICA peak shift {last['Peak shift (mV)']:.0f} mV"])
    st.caption("Proxies from ~1C full-cell data without half-cell references: use them for trends and for comparing "
               "cells, not as absolute mode quantities.")


@fragment
def stress_section() -> None:
    with st.expander("Screening thresholds", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        lim = dict(T_warn_C=float(c1.number_input("Accelerated-ageing T (°C)", 30.0, 60.0, 45.0, 1.0)),
                   T_crit_C=float(c2.number_input("SEI-decomposition onset (°C)", 45.0, 90.0, 60.0, 1.0)),
                   T_plating_C=float(c3.number_input("Plating-risk charge T (°C)", 0.0, 25.0, 10.0, 1.0)),
                   V_deep_V=float(c4.number_input("Deep-discharge limit (V)", 2.0, 3.0, 2.5, 0.05)))
    sx, summ = stress_cached(ct, DATA_KEY, lim)
    L = te.SafetyLimits(**lim)
    show(fig_stress(sx, L, P, cell), key="stress", data=sx[sx["Cell_ID"] == cell][
        ["n", "T_max_C", "V_min_V", "PRI", "hot", "critical_T", "plating", "deep", "high_rate"]])
    mech = pd.DataFrame([{"Stressor": v[0], "Mechanism (literature)": v[1]} for v in te.STRESS_MECHANISMS.values()])
    a, b = st.columns([3, 2], gap="medium")
    with a:
        st.markdown("**Share of cycles exposed (%) per cell**")
        show_table(summ.style.format("{:.0f}", subset=[c for c in summ.columns if c not in ("Max PRI",)])
                   .format("{:.2f}", subset=["Max PRI"]))
    with b:
        st.markdown("**Stressor → degradation mechanism**")
        show_table(mech.set_index("Stressor"))


def condition_effects_section() -> None:
    sf = stress_factors_cached(ct, DATA_KEY)
    if not sf.get("available"):
        st.info("Operating-condition regression needs at least four cells with a measurable early fade rate.")
        return
    st.markdown(f"Cohort regression of the early fade rate per Ah on the NASA design factors "
                f"({sf['n_cells']} cells, R² = {sf['r2']:.2f}). Factors without variation are dropped automatically.")
    show_table(sf["coefficients"].style.format("{:.3g}"))
    st.caption("Eₐ > 0: hotter cells fade faster per Ah (Arrhenius SEI growth). Current exponent > 0: rate-driven "
               "damage. Cold regime (×) > 1: the 4 °C cells age faster than Arrhenius predicts (lithium plating). "
               "Wide intervals mean the cohort cannot separate the factor from cell-to-cell variation.")


def view_data() -> None:
    k = st.columns(6)
    k[0].metric("Target cell", cell)
    k[1].metric("Initial capacity", fmt(c_bol, ".3f", "Ah"))
    k[2].metric("Valid discharge cycles", int(meta.loc[cell, "cycles"]))
    k[3].metric("Capacity fade", fmt(meta.loc[cell, "fade_pct"], ".1f", "%"))
    k[4].metric("Ambient / discharge current",
                f"{fmt(meta.loc[cell, 'Ambient_C'], '.0f', '°C')} / {fmt(meta.loc[cell, 'I_dis_A'], '.1f', 'A')}")
    k[5].metric("Regeneration events", int(meta.loc[cell].get("regen_events", 0) or 0),
                help="Upward capacity jumps (> 1 % of initial) after rest periods; kept in the data and "
                     "marked ▲ on the fade chart.")

    kn = knee_cached(ct_cell, DATA_KEY, cell)
    if kn.get("found"):
        st.caption(f"Knee point detected at n = {kn['knee_n']} (SOH {kn['soh_at_knee']:.3f}): fade accelerates "
                   f"{kn['ratio']:.1f}× (F-test p = {kn['p_value']:.1g}). Marked ★ on the fade chart; beyond it the "
                   "risk of sudden capacity loss rises, which the maintenance optimiser accounts for.")
    issues = validation_cached(store, DATA_KEY, cell)
    n_out = int(ct_cell["outlier"].sum())
    with st.expander(f"Data quality: {len(issues)} telemetry issue(s), {n_out} outlier cycle(s), "
                     f"{len(cell_errors)} skipped cell(s)", expanded=bool(issues)):
        if issues:
            for msg in issues:
                st.warning(msg)
        else:
            st.success("Schema, physical ranges and time ordering passed for this cell.")
        if cell_errors:
            show_table(pd.DataFrame({"Cell": list(cell_errors), "Reason": list(cell_errors.values())}).set_index("Cell"))

    section("Capacity fade against the cohort")
    fade_explorer()

    section("Cohort by ambient temperature")
    show(fig_cohort_grid(ct, meta, cell, P), key="cohort_grid",
         data=ct.loc[~ct["outlier"], ["Cell_ID", "n", "SOH"]].merge(
             meta[["Ambient_C", "I_dis_A"]].reset_index(), on="Cell_ID"))
    with st.expander("Mission 1 · How do operating conditions influence degradation?", expanded=False):
        condition_effects_section()

    section("Mission 1 · Which parameter best represents health?")
    health_indicator_section()

    section("Incremental capacity analysis (dQ/dV)")
    ica_section()

    section("Degradation modes: LLI · LAM · conductivity loss")
    modes_section()

    section("Operating stress and safety exposure")
    stress_section()


# =============================================================================
# View 2: models & forecasting
# =============================================================================
def methods_panel() -> None:
    with st.expander("Methods and references", expanded=False):
        st.markdown("**ML surrogate (increment strategy).** An autonomous fade-rate law learnt across the cohort "
                    "and integrated from the observed state at the forecast origin; x holds the operating plan "
                    "and early-life descriptors (early fade slope, early resistance growth).")
        st.latex(r"\frac{d\,\mathrm{SOH}}{dn} = g_\theta(\mathrm{SOH}, \mathbf{x}) \le 0,\qquad "
                 r"\widehat{\mathrm{SOH}}_{n+1} = \widehat{\mathrm{SOH}}_n + g_\theta(\widehat{\mathrm{SOH}}_n, \mathbf{x})")
        st.markdown("Bands: cross-cell split conformal, horizon-dependent half-width q(h) from calibration "
                    "cells forecast with the same protocol (valid under cell exchangeability).")
        st.markdown("**Dual time-scale ECM twin.** Per-cycle EKF on θ = [SOH, R_int, R_ct, log k]:")
        st.latex(r"\mathrm{SOH}_{k+1} = \mathrm{SOH}_k - e^{\log k}\, s(T_k)\, A_k,\quad "
                 r"R_{k+1} = R_k + \beta R_0\, e^{\log k} s(T_k) A_k,\quad \log k_{k+1} = \log k_k + w_k")
        st.latex(r"V_j = \mathrm{OCV}\!\left(1 - \tfrac{Q_j}{C_0\,\mathrm{SOH}}\right) + I_j R_\mathrm{int} + R_\mathrm{ct}\, g_j,"
                 r"\qquad s(T) = e^{\frac{E_a}{R}\left(\frac{1}{T_\mathrm{ref}} - \frac{1}{T}\right)}\,[1 + k_c (T_c - T)^+]")
        st.markdown("Voltage is used only over the first part of each discharge (a fixed fraction of the "
                    "beginning-of-life capacity), so the cut-off time never leaks capacity; the load-step "
                    "resistance and, optionally, the full capacity are extra measurements. Forecasts are Monte-Carlo "
                    "propagations of the posterior (SOH, log k).")
        st.markdown("**Hybrid PINN.** Network states with built-in initial conditions and soft physics residuals:")
        st.latex(r"\frac{d\,\mathrm{SOH}}{dn} = -k\, e^{\frac{E_a}{R}\left(\frac{1}{T_\mathrm{ref}}-\frac{1}{T}\right)}"
                 r"\, 2C_0\,\mathrm{SOH}\left(\frac{1-\mathrm{SOH}+\varepsilon}{L_\mathrm{ref}}\right)^{-m},\qquad "
                 r"\Delta V_\mathrm{step} = I(R_\mathrm{int}+R_x) + \frac{2RT}{F}\,\sinh^{-1}\!\frac{I F R_\mathrm{ct}}{2RT}")
        st.markdown("Eₐ is fixed or pooled across cells (ln rate vs 1/T regression with bootstrap CI): a single "
                    "cell at one temperature cannot identify it. Seed ensembles with randomised physical "
                    "initialisation report which parameters the data constrain.")
        st.markdown("**Semi-empirical law** (Wang et al. 2011), in throughput, fitted in log space with a cohort "
                    "ridge prior on the exponent:")
        st.latex(r"\mathrm{SOH} = 1 - B(T, I)\,\mathrm{Ah}^{z},\qquad z \approx 0.5\ \text{(SEI, diffusion-limited)},"
                 r"\quad z \approx 1\ \text{(linear)},\quad z > 1\ \text{(accelerating)}")
        st.markdown("**Particle filter** (Saha & Goebel 2009) on the double-exponential capacity model, prior from "
                    "cohort fits, Student-t likelihood, systematic resampling:")
        st.latex(r"\mathrm{SOH}(n) = a\,e^{b n} + c\,e^{d n},\qquad w_k^{(i)} \propto w_{k-1}^{(i)}\,"
                 r"p\!\left(y_k \mid \theta^{(i)}\right)")
        st.markdown("**Health-indicator ranking** (Coble & Hines 2009): monotonicity, trendability, prognosability "
                    "and |Spearman ρ| with SOH on beginning-of-life-normalised indicators, plus a leave-one-cell-out "
                    "test; PCA decides whether one parameter suffices. **Degradation modes**: LLI / LAM / CL proxies "
                    "from ICA peak area, capacity and resistance (Birkl et al. 2017); DVA peak spacing. **Knee**: "
                    "two-segment piecewise-linear fit with an F-test.")
        st.markdown("**Integrated O&M** (Mission 3): renewal-reward long-run profit rate with sudden-failure hazard "
                    "h(SOH) after the knee,")
        st.latex(r"\max_{\pi,\,S_\mathrm{rep}}\ \frac{\mathbb{E}\left[\sum_k S_{k-1}\,(R_k - E_k)\right] - "
                 r"\mathbb{E}[C_\mathrm{maint}]}{\mathbb{E}\left[\sum_k S_{k-1}\,t_k\right] + \mathbb{E}[t_\mathrm{down}]},"
                 r"\qquad S_k = \prod_{j \le k}\left(1 - h(\mathrm{SOH}_j)\right)")
        st.markdown(
            "| Model class (Rufino Júnior et al. 2024, Fig. 1) | Implemented as |\n|---|---|\n"
            "| Empirical | Direct ML regression SOH(n) (baseline) |\n"
            "| Semi-empirical | Power law in Ah; Arrhenius / current stress regression |\n"
            "| Equivalent-circuit | 1-RC ECM inside the dual EKF twin and the operations plant |\n"
            "| Electrochemical / physics-based | Butler–Volmer and SEI-type fade law in the hybrid PINN |\n"
            "| Data-driven | RF, GBR, GPR, SVR, MLP, Ridge fade-rate models with conformal bands |\n"
            "| Bayesian filtering | Dual EKF (states + rate), particle filter |")
        st.markdown("**Metrics.** RMSE on held-out cycles; RUL from the origin with right-censoring; relative "
                    "accuracy, α-λ and prognostic horizon after Saxena et al.; empirical band coverage.")
        st.markdown(
            "References: G. L. Plett, *J. Power Sources* 134 (2004) 252–292 (EKF for BMS, parts 1–3) · "
            "M. Dubarry et al., *J. Power Sources* 219 (2012) 204–216 (ICA degradation modes) · "
            "A. Saxena et al., *Int. J. PHM* 1 (2010) (prognostic metrics) · "
            "K. A. Severson et al., *Nature Energy* 4 (2019) 383–391 (early-life prediction) · "
            "M. Raissi et al., *J. Comput. Phys.* 378 (2019) 686–707 (PINNs) · "
            "A. N. Angelopoulos & S. Bates, arXiv:2107.07511 (conformal prediction) · "
            "B. Saha & K. Goebel, NASA Ames Prognostics Data Repository (battery data set) and *Annual Conf. PHM "
            "Society* (2009) (particle-filter prognostics) · "
            "C. A. Rufino Júnior et al., *Energies* 17 (2024) 3372 (degradation mechanisms, model taxonomy) · "
            "J. S. Menye, M.-B. Camara & B. Dakyo, *Energies* 18 (2025) 342 (degradation and failure mechanisms, "
            "LLI/LAM/CL, stress factors) · "
            "J. Wang et al., *J. Power Sources* 196 (2011) 3942–3948 (semi-empirical cycle-life model) · "
            "C. R. Birkl et al., *J. Power Sources* 341 (2017) 373–386 (degradation diagnostics) · "
            "J. Coble & J. W. Hines, *Annual Conf. PHM Society* (2009) (prognostic-parameter criteria).")


def ml_section() -> None:
    section("Machine-learning surrogates")
    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    models = c1.multiselect("Models", list(te.ML_MODELS), default=["Random Forest", "Gradient Boosting", "MLP"])
    frac = c2.slider("Forecast origin (fraction of life observed)", 0.2, 0.8, 0.4, 0.05, key="ml_frac")
    strategy = c3.selectbox("Strategy", list(te.ML_STRATEGIES),
                            format_func=lambda s: "Fade-rate model (increment)" if s == "increment"
                            else "Direct SOH(n) regression (legacy)",
                            help="The increment strategy learns dSOH/dn as a function of SOH, so tree models "
                                 "keep extrapolating beyond the longest training life.")
    use_pop = c4.toggle("Train on the cohort", value=True)
    b1, b2 = st.columns(2)
    conf = b1.slider("Conformal calibration cells (0 = no band)", 0, 8, 4)
    level = b2.select_slider("Band level", [0.8, 0.9, 0.95], value=0.9, key="ml_level")
    n0_ml = int(max(5, round(frac * meta.loc[cell, "cycles"])))
    ml_cfg = dict(cell=cell, n0=n0_ml, models=tuple(models), use_population=use_pop, eol_ah=float(eol_ah),
                  strategy=strategy, conformal_cells=conf, band_level=level)

    if st.button("Train ML surrogates", type="primary", key="ml_go", disabled=not models):
        prog = st.progress(0.0)
        out: List[te.MLForecast] = []
        for i, mname in enumerate(models):
            prog.progress(i / max(len(models), 1), text=f"Training {mname}…")
            try:
                out.append(ml_cached(ct, DATA_KEY, cell, n0_ml, mname, use_pop, float(eol_ah), strategy, conf, level))
            except Exception as exc:
                st.warning(f"{mname}: {exc}")
        prog.empty()
        st.session_state["ml"] = {"cfg": ml_cfg, "res": out}

    saved = st.session_state.get("ml")
    if not (saved and saved["res"]):
        return
    if saved["cfg"]["cell"] != cell:
        st.info("The stored ML results belong to another cell. Train the surrogates to update them.")
        return
    if saved["cfg"] != ml_cfg:
        st.warning("Settings changed since the last run; the results below use the previous settings.")
    res = saved["res"]
    data = pd.concat([pd.DataFrame({"model": r.model, "n": r.n_grid, "soh": r.soh_pred,
                                    "lo": r.soh_lo if r.soh_lo is not None else np.nan,
                                    "hi": r.soh_hi if r.soh_hi is not None else np.nan}) for r in res])
    show(fig_ml(res, ct_cell, saved["cfg"]["n0"], soh_eol, P), key="ml_fig", data=data)
    tbl = pd.DataFrame([{"Model": r.model, "RMSE": r.metrics.rmse, "MAE": r.metrics.mae, "R²": r.metrics.r2,
                         "Coverage": r.metrics.coverage, "Band width": r.metrics.band_width,
                         "RUL true": r.metrics.rul_true, "RUL pred": r.metrics.rul_pred,
                         "RUL error": r.metrics.rul_error, "Censored": r.metrics.censored,
                         "Fit time (s)": r.fit_seconds} for r in res]).set_index("Model")
    show_table(tbl.style.format(
        {"RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.3f}", "Coverage": "{:.2f}", "Band width": "{:.4f}",
         "RUL true": "{:.0f}", "RUL pred": "{:.0f}", "RUL error": "{:+.0f}", "Fit time (s)": "{:.2f}"}, na_rep="—")
        .highlight_min(subset=["RMSE"], props="background-color: rgba(0,158,115,0.25); font-weight: 700;"))
    cal = res[0].calibration_cells
    if cal:
        st.caption(f"Conformal calibration cells: {', '.join(cal)}. Coverage below the nominal level signals "
                   "that this cell ages differently from its calibration partners (exchangeability violated).")
    manifest_button(ml_cfg, "ml", DATA_KEY)


def physics_section() -> None:
    section("Physics-informed benchmark: ML, ECM twin and hybrid PINN")
    with st.expander("Observer and PINN settings", expanded=False):
        h1, h2 = st.columns(2, gap="large")
        with h1:
            st.markdown("**ECM twin observer**")
            observer = st.radio("Observer", ["dual", "joint"], horizontal=True,
                                format_func=lambda o: "Dual time-scale (per cycle, fast)" if o == "dual"
                                else "Joint EKF (per sample, slow)")
            if observer == "dual":
                use_cap = st.toggle("Use full-discharge capacity measurements", value=False,
                                    help="Off = operando setting: only partial-window voltage and load-step "
                                         "resistance, as in cells that never see a reference discharge.")
                win_frac = st.slider("Voltage window (fraction of BOL capacity)", 0.3, 0.9, 0.6, 0.05)
                sig_v = st.slider("σᵥ voltage model error (mV)", 5, 60, 15, 1)
                q_logk = st.select_slider("q_log k random walk per cycle", [0.005, 0.01, 0.02, 0.03, 0.05, 0.1],
                                          value=0.03)
                dual_cfg = te.DualTwinConfig(use_capacity=use_cap, voltage_window_frac=float(win_frac),
                                             sigma_v=sig_v / 1000.0, q_logk=float(q_logk))
                twin_params = te.TwinParameters()
            else:
                dual_cfg = te.DualTwinConfig()
                sigma_v = st.slider("σᵥ voltage noise (V)", 0.005, 0.200, 0.080, 0.005, format="%.3f")
                q_soh = st.select_slider("q_SOH random walk (per Ah)", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
                                         value=5e-4, format_func=lambda v: f"{v:.0e}")
                tau = st.slider("τ RC time constant (s)", 10, 300, 60, 10)
                twin_params = te.TwinParameters(sigma_v=sigma_v, q_soh_per_ah=q_soh, tau_rc_s=float(tau))
        with h2:
            st.markdown("**Hybrid PINN**")
            epochs = st.select_slider("Training epochs", [300, 500, 1000, 1500, 2500, 4000], value=1500)
            members = st.slider("Ensemble members (seeds)", 1, 5, 3,
                                help="More than one member adds an epistemic band and an identifiability table.")
            ea_mode = st.selectbox("Activation energy Eₐ", list(te.EA_MODES), index=0,
                                   format_func={"fixed": "Fixed (literature value)",
                                                "pooled": "Pooled across the cohort",
                                                "learned": "Learned per cell (not identifiable)"}.get)
            ea_fixed = st.number_input("Fixed Eₐ (kJ/mol)", 10.0, 120.0, 30.0, 1.0, disabled=ea_mode != "fixed")
            lam_phys = st.select_slider("λ physics", [0.0, 0.1, 0.3, 1.0, 3.0, 10.0], value=1.0)
            lam_bv = st.select_slider("λ Butler–Volmer", [0.0, 0.1, 0.3, 1.0, 3.0], value=0.3)
            lam_eis = st.select_slider("λ EIS anchoring", [0.0, 0.1, 0.5, 1.0, 3.0], value=0.5)
        pinn_cfg = te.PINNConfig(epochs=int(epochs), lambda_phys=float(lam_phys), lambda_bv=float(lam_bv),
                                 lambda_eis=float(lam_eis), ea_mode=ea_mode, ea_fixed_J_mol=float(ea_fixed) * 1e3)

    if ea_mode == "pooled":
        est = pooled_ea_cached(ct, DATA_KEY, 15.0)
        if est["Ea_J_mol"] is None:
            st.info("Pooled Eₐ needs at least three cells above 15 °C; the fixed value will be used.")
        else:
            verdict = ("identified" if est["identifiable"] else
                       f"not identified (temperature span {est['temp_span_C']:.0f} °C) — fixed value used")
            card("Cohort Arrhenius regression", [
                f"Eₐ = {est['Ea_J_mol'] / 1e3:.1f} kJ/mol, 90% bootstrap CI "
                f"{fmt(est['ci_lo'] / 1e3 if est['ci_lo'] else None, '.1f')}–"
                f"{fmt(est['ci_hi'] / 1e3 if est['ci_hi'] else None, '.1f')} kJ/mol, {verdict}",
                f"{est['n_cells']} cells, temperature span {est['temp_span_C']:.0f} °C"
                + (f", current exponent α = {est['alpha_I']:.2f}" if est.get("alpha_I") is not None else "")])

    c1, c2, c3, c4 = st.columns(4)
    frac2 = c1.slider("Forecast origin n₀ (fraction of life)", 0.2, 0.8, 0.4, 0.05, key="cmp_frac")
    ml_pick = c2.selectbox("ML reference model", list(te.ML_MODELS), index=1)
    level = c3.select_slider("Band level", [0.8, 0.9, 0.95], value=0.9, key="cmp_level")
    reuse = c4.toggle("Reuse cached observer run", value=True)
    d1, d2, d3 = st.columns(3)
    use_semi = d1.toggle("Semi-empirical power law", value=True,
                         help="Q_loss = B·Ah^z (Wang et al. 2011) with a cohort prior on z; Monte-Carlo band.")
    use_pf = d2.toggle("Particle filter (double exponential)", value=True,
                       help="Saha & Goebel (2009) capacity model; population prior, sequential Bayesian update.")
    run_pinn = d3.toggle("Hybrid PINN", value=True, help="Switch off for a fast comparison (the PINN takes seconds).")
    extra = tuple(k for k, on in (("semi", use_semi), ("pf", use_pf)) if on)
    obs_key = (DATA_KEY, cell, observer, tuple(sorted(asdict(twin_params).items())),
               tuple(sorted(asdict(dual_cfg).items())))
    cmp_cfg = dict(cell=cell, observer=observer, frac=frac2, ml_model=ml_pick, band_level=level,
                   eol_ah=float(eol_ah), twin=asdict(twin_params), dual=asdict(dual_cfg), pinn=asdict(pinn_cfg),
                   pinn_seeds=list(range(members)), extra=list(extra), run_pinn=run_pinn)

    if st.button("Run benchmark on this cell", type="primary", key="cmp_go"):
        bar = st.progress(0.0, text="Initialising observer…")
        try:
            cached = st.session_state.get("ekf")
            ekf_res = cached["res"] if (reuse and cached and cached["key"] == obs_key) else None
            raw = cell_frame(store, DATA_KEY, cell)
            res_new = te.compare_paradigms(raw, ct, imp, cell, frac2, twin_params, pinn_cfg, ml_pick, eol_ah,
                                           ekf=ekf_res, progress=lambda f, m: bar.progress(f, text=m),
                                           observer=observer, dual_cfg=dual_cfg, band_level=level,
                                           pinn_seeds=tuple(range(members)), extra=extra, run_pinn=run_pinn)
            if res_new.ekf is not None:
                st.session_state["ekf"] = {"key": obs_key, "res": res_new.ekf}
            st.session_state["cmp"] = {"cfg": cmp_cfg, "res": res_new}
        except Exception as exc:
            report_error("Benchmark failed", exc, debug)
        finally:
            bar.empty()

    saved = st.session_state.get("cmp")
    if not saved:
        return
    res: te.ComparisonResult = saved["res"]
    if res.cell_id != cell:
        st.info(f"The stored benchmark belongs to {res.cell_id}. Run the benchmark to update it.")
        return
    if saved["cfg"] != cmp_cfg:
        st.warning("Settings changed since the last run; the results below use the previous settings.")
    for name, msg in res.errors.items():
        st.warning(f"{name}: {msg}")

    mcols = st.columns(max(len(res.metrics), 1))
    for col, (name, m) in zip(mcols, res.metrics.items()):
        if m.censored:
            delta = f"censored: true RUL > {m.rul_true_lb}"
        elif m.rul_error is not None:
            delta = f"RUL error {m.rul_error:+d} cycles"
        else:
            delta = None
        cov = "" if m.coverage is None else f" · coverage {m.coverage:.0%}"
        col.metric(f"{name} RMSE{cov}", fmt(m.rmse, ".4f"), delta, delta_color="off")

    data = pd.concat([pd.DataFrame({"paradigm": k, "n": n, "soh": p}) for k, (n, p) in res.predictions.items()]) \
        if res.predictions else None
    show(fig_compare(res, P), key="cmp_fig", data=data)
    with st.expander("Held-out metrics table"):
        show_table(te.metrics_table(res.metrics).style.format(
            {"RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.3f}", "RUL true": "{:.0f}", "RUL pred": "{:.0f}",
             "RUL error": "{:+.0f}", "RA": "{:.2f}", "Coverage": "{:.2f}", "Band width": "{:.4f}",
             "RUL lower bound": "{:.0f}"}, na_rep="—"))
    manifest_button(cmp_cfg, "comparison", DATA_KEY, {"errors": res.errors, "ea": (res.ea or {}).get("Ea_J_mol")})

    a, b = st.columns(2, gap="medium")
    with a:
        show(fig_params(res, P), key="cmp_params", export=False)
        if res.rul_samples:
            m = next(iter(res.metrics.values()), None)
            show(fig_rul_hist(res.rul_samples, m.rul_true if m else None, m.rul_true_lb if m else None, P),
                 key="rul_hist", data=pd.concat([pd.DataFrame({"paradigm": k, "rul_samples": v})
                                                 for k, v in res.rul_samples.items()]))
        pr = getattr(res, "prognostics", {}) or {}
        lines = []
        if te.SEMI_NAME in pr:
            q = pr[te.SEMI_NAME].params
            regime = ("diffusion-limited SEI (≈ √Ah)" if q["z"] < 0.75 else "near-linear in throughput"
                      if q["z"] < 1.25 else "accelerating")
            lines.append(f"Semi-empirical: z = {q['z']:.2f} ({regime}; cohort prior {q['z_prior']:.2f}), "
                         f"B = {q['B']:.2e}, {q['Ah_per_cycle']:.2f} Ah per cycle ahead")
        if te.PF_NAME in pr:
            q = pr[te.PF_NAME].params
            lines.append(f"Particle filter: prior from {q['prior_cells']} cohort fits, {q['n_particles']} particles, "
                         f"{q['resampling_steps']} resampling steps")
        if lines:
            card("Prognostic model parameters", lines)
    with b:
        if res.ekf is not None:
            sig = float(res.ekf.config.get("sigma_v", res.ekf.params.get("sigma_v", 0.02)))
            show(fig_innovation(res.ekf, sig, P), key="cmp_innov", data=res.ekf.per_cycle)
            if res.ekf.kind == "dual":
                pc = res.ekf.per_cycle
                st.caption(f"Degradation rate k: prior {res.ekf.params.get('k_prior', float('nan')):.2e} → "
                           f"personalised {pc['k_ah'].iloc[-1]:.2e} per Ah · mean NIS/dof "
                           f"{pc['NIS_norm'].mean():.2f} · runtime {res.ekf.runtime_s:.2f} s")
        if res.pinn is not None:
            show(fig_pinn_loss(res.pinn, P), key="cmp_loss", export=False)

    if res.pinn is not None:
        section("Identified electrochemical parameters (hybrid PINN)")
        ph = res.pinn.physics
        m_exp = ph["m_SEI_exponent"]
        regime = ("self-limiting, SEI-like growth" if m_exp > 0.2 else
                  "self-accelerating, knee-like" if m_exp < -0.2 else "near-linear in throughput")
        ea_note = {"fixed": "fixed", "pooled": "pooled across cohort", "learned": "learned"}.get(res.pinn.ea_mode, "")
        p1, p2 = st.columns(2, gap="medium")
        with p1:
            card("Degradation kinetics", [
                f"Rate constant k = {ph['k_per_Ah']:.2e} per Ah",
                f"Activation energy Eₐ = {ph['Ea_kJ_mol']:.1f} kJ mol⁻¹ ({ea_note})",
                f"Fade exponent m = {m_exp:.2f} ({regime})",
                f"Resistance coupling γ_int = {ph['gamma_int']:.2f}, γ_ct = {ph['gamma_ct']:.2f}"])
        with p2:
            card("Charge transfer (Butler–Volmer)", [
                f"Exchange current i₀: {ph['i0_start_A']:.3f} A at start, {ph['i0_at_n0_A']:.3f} A at n₀",
                f"Overpotential η_ct at 2 A: {ph['eta_ct_2A_start_mV']:.0f} mV to {ph['eta_ct_2A_n0_mV']:.0f} mV",
                f"Lumped fast polarisation R_x = {ph['R_x_mOhm']:.1f} mΩ",
                f"Training time {res.pinn.train_seconds:.1f} s ({res.pinn.n_members} member(s))"])
        if res.pinn.physics_table is not None:
            st.markdown("**Identifiability across ensemble members** (randomised physical initialisation; "
                        "coefficient of variation above 0.25 = not constrained by this cell's data)")
            show_table(res.pinn.physics_table.style.format({"mean": "{:.4g}", "std": "{:.3g}", "cv": "{:.2f}"},
                                                           na_rep="—"))
            st.caption("The ensemble band reflects optimisation / initialisation spread only (epistemic), "
                       "not measurement noise, and is usually too narrow to be a calibrated interval.")


def ablation_section() -> None:
    section("Does voltage feedback add information? Measurement ablation")
    st.markdown("The dual twin is re-run with measurement subsets. *Open loop* uses only the cohort prior; "
                "the gap between it and the voltage-only observer is the information carried by the voltage signal.")
    frac = st.slider("Forecast origin for the ablation forecasts", 0.2, 0.8, 0.4, 0.05, key="abl_frac")
    n0 = int(max(5, round(frac * meta.loc[cell, "cycles"])))
    cfg = dict(cell=cell, n0=n0, eol_ah=float(eol_ah))
    if st.button("Run ablation", key="abl_go"):
        with st.spinner("Running four observer configurations…"):
            try:
                tab, results = te.twin_ablation(cell_frame(store, DATA_KEY, cell), ct, imp, cell, te.TwinParameters(),
                                                None, te.similar_cells(meta, cell), n0, soh_eol)
                st.session_state["abl"] = {"cfg": cfg, "tab": tab, "res": results}
            except Exception as exc:
                report_error("Ablation failed", exc, debug)
    saved = st.session_state.get("abl")
    if saved and saved["cfg"]["cell"] == cell:
        if saved["cfg"] != cfg:
            st.warning("Settings changed since the last run; the results below use the previous settings.")
        show(fig_ablation(saved["res"], ct_cell, P), key="abl_fig", data=saved["tab"].reset_index())
        show_table(saved["tab"].style.format({"Tracking RMSE": "{:.4f}", "Tracking 90% coverage": "{:.2f}",
                                              "Mean NIS / dof": "{:.2f}", "k personalised / prior": "{:.2f}",
                                              "Forecast RMSE": "{:.4f}", "RUL error": "{:+.0f}",
                                              "Forecast coverage": "{:.2f}", "Info gain SOH (nats/100 cyc)": "{:.2f}",
                                              "Info gain log k (nats/100 cyc)": "{:.2f}"}, na_rep="—"))
        st.caption("Information gain = entropy reduction of the SOH and degradation-rate posteriors per 100 cycles: "
                   "the measurement set with the largest gain is the most informative (Mission 2). "
                   "Tracking RMSE compares the causal estimate with measured SOH over the whole life. When "
                   "capacity is among the measurements this is partly circular; the operando rows are the fair test.")


def cross_cell_section() -> None:
    section("Cross-cell benchmark: forecast-origin sweep")
    st.markdown("A single cell at a single origin is an anecdote. This section evaluates every paradigm on every "
                "cell at several origins. Run the full sweep offline with `python benchmark.py` "
                "(writes `results/benchmark.parquet`) or a quick subset here.")
    src = st.radio("Source", ["Run a quick sweep here", "Load results file"], horizontal=True, key="bench_src")
    if src == "Load results file":
        default = RESULTS_DIR / "benchmark.parquet"
        up_b = st.file_uploader("Benchmark results (.parquet / .csv)", type=["parquet", "csv"], key="up_bench")
        up_r = st.file_uploader("Residuals (optional, *_residuals)", type=["parquet", "csv"], key="up_resid")
        try:
            if up_b is not None:
                bench = read_uploaded_table(up_b)
                resid = read_uploaded_table(up_r) if up_r is not None else None
                st.session_state["bench"] = {"bench": bench, "resid": resid, "cfg": {"source": up_b.name}}
            elif default.exists() and "bench" not in st.session_state:
                bench = load_bench_file(str(default), default.stat().st_mtime)
                rp = RESULTS_DIR / "benchmark_residuals.parquet"
                resid = load_bench_file(str(rp), rp.stat().st_mtime) if rp.exists() else None
                st.session_state["bench"] = {"bench": bench, "resid": resid, "cfg": {"source": str(default)}}
        except Exception as exc:
            report_error("Could not read the benchmark file", exc, debug)
    else:
        c1, c2, c3 = st.columns(3)
        pool = [c for c in cells if int(meta.loc[c, "cycles"]) >= 30]
        sel = c1.multiselect("Cells", pool, default=pool[: min(6, len(pool))], key="bench_cells")
        fracs = c2.multiselect("Origins (fraction of life)", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], default=[0.3, 0.5],
                               key="bench_fracs")
        pars = c3.multiselect("Paradigms", list(te.BENCH_PARADIGMS), default=["ML", "Twin", "SemiEmp", "PF"],
                              key="bench_pars", format_func={"ML": "ML surrogate", "Twin": "ECM twin (dual EKF)",
                                                             "PINN": "Hybrid PINN", "SemiEmp": "Semi-empirical",
                                                             "PF": "Particle filter"}.get,
                              help="The PINN adds ~seconds per cell and origin.")
        cfg = te.BenchmarkConfig(fracs=tuple(sorted(fracs)) or (0.4,), paradigms=tuple(pars) or ("Twin",),
                                 eol_ah=float(eol_ah), pinn_epochs=500, conformal_cells=3)
        est = len(sel) * len(cfg.fracs) * (0.5 + 1.5 * ("ML" in pars) + 3 * ("PINN" in pars) + 0.5 * ("PF" in pars))
        if st.button(f"Run sweep (≈ {est:.0f} s)", key="bench_go", disabled=not sel):
            bar = st.progress(0.0, text="Starting…")
            try:
                bench, resid = te.run_benchmark(store, ct, imp, cfg, cells=sel,
                                                progress=lambda f, m: bar.progress(f, text=m))
                st.session_state["bench"] = {"bench": bench, "resid": resid, "cfg": asdict(cfg) | {"cells": sel}}
            except Exception as exc:
                report_error("Benchmark sweep failed", exc, debug)
            finally:
                bar.empty()

    saved = st.session_state.get("bench")
    if not saved:
        return
    bench, resid = saved["bench"], saved["resid"]
    alpha = float(saved["cfg"].get("alpha", 0.2)) if isinstance(saved["cfg"], dict) else 0.2
    level = float(saved["cfg"].get("band_level", 0.9)) if isinstance(saved["cfg"], dict) else 0.9
    if "error" in bench.columns and bench["error"].notna().any():
        with st.expander(f"{int(bench['error'].notna().sum())} failed forecast(s)"):
            show_table(bench[bench["error"].notna()][["cell", "frac", "paradigm", "error"]])
    summ = te.benchmark_summary(bench, alpha)
    if summ.empty:
        st.warning("No successful forecasts in these results.")
        return
    show_table(summ.style.format({"Median RMSE": "{:.4f}", "Mean RMSE": "{:.4f}", "Mean RA": "{:.2f}",
                                  "α-λ hit rate": "{:.2f}", "Mean coverage": "{:.2f}", "Mean band width": "{:.4f}",
                                  "Mean PH (cycles)": "{:.0f}"}, na_rep="—"))
    a, b = st.columns(2, gap="medium")
    with a:
        show(fig_bench_parity(bench, alpha, P), key="bench_parity", data=_bench_ok(bench))
    with b:
        show(fig_alpha_lambda(bench, alpha, P), key="bench_al", data=_bench_ok(bench))
    if resid is not None and len(resid):
        cov = te.coverage_by_horizon(resid)
        show(fig_horizon(cov, level, P), key="bench_horizon", data=cov)
    with st.expander("Prognostic horizon per cell"):
        show_table(te.prognostic_horizon(_bench_ok(bench), alpha).pivot(index="cell", columns="paradigm",
                                                                         values="PH_cycles"))
    download("Benchmark rows (CSV)", bench.to_csv(index=False), "benchmark.csv", "text/csv", key="dl_bench")
    manifest_button(saved["cfg"] if isinstance(saved["cfg"], dict) else {}, "benchmark", DATA_KEY)


def update_frequency_section() -> None:
    section("Mission 2 · How often should the twin update?")
    st.markdown("The dual twin assimilates measurements only every *m*-th discharge and runs open loop on its "
                "calibrated fade law in between. The recommended interval is the sparsest schedule whose tracking "
                "error stays within 25% of updating every cycle or below 0.5% SOH (about the repeatability of a "
                "capacity measurement): it sets the diagnostic cost of the twin in operation.")
    c1, c2 = st.columns(2)
    ivals = c1.multiselect("Update intervals (cycles)", [1, 2, 3, 5, 10, 20, 50], default=[1, 2, 5, 10, 20, 50],
                           key="uf_ivals")
    frac = c2.slider("Forecast origin", 0.2, 0.8, 0.4, 0.05, key="uf_frac")
    n0 = int(max(5, round(frac * meta.loc[cell, "cycles"])))
    cfg = dict(cell=cell, n0=n0, intervals=sorted(ivals), eol_ah=float(eol_ah))
    if st.button("Run update-frequency study", key="uf_go", disabled=not ivals):
        with st.spinner("Replaying the twin with sparse updates…"):
            try:
                tab, results = te.update_frequency_study(cell_frame(store, DATA_KEY, cell), ct, imp, cell,
                                                         te.TwinParameters(), None, n0, soh_eol,
                                                         intervals=tuple(sorted(ivals)))
                st.session_state["uf"] = {"cfg": cfg, "tab": tab, "res": results}
            except Exception as exc:
                report_error("Update-frequency study failed", exc, debug)
    saved = st.session_state.get("uf")
    if saved and saved["cfg"]["cell"] == cell:
        if saved["cfg"] != cfg:
            st.warning("Settings changed since the last run; the results below use the previous settings.")
        tab = saved["tab"]
        show(fig_update_freq(tab, saved["res"], ct_cell, P), key="uf_fig", data=tab.reset_index())
        show_table(tab.style.format({"Updates per 100 cycles": "{:.0f}", "Tracking RMSE": "{:.4f}",
                                     "Worst |error|": "{:.4f}", "Tracking 90% coverage": "{:.2f}",
                                     "Info gain per update (nats)": "{:.3f}", "Forecast RMSE": "{:.4f}",
                                     "RUL error": "{:+.0f}", "Forecast coverage": "{:.2f}"}, na_rep="—"))
        rec = tab.attrs.get("recommended")
        if rec:
            card("Mission 2 answer", [f"Recommended update interval for {cell}: every {rec} discharge cycle(s) "
                                      f"({tab.loc[rec, 'Updates per 100 cycles']:.0f} updates per 100 cycles, tracking "
                                      f"RMSE {tab.loc[rec, 'Tracking RMSE']:.4f} vs "
                                      f"{tab['Tracking RMSE'].iloc[0]:.4f} when updating every cycle)",
                                      "Information gain per update rises as updates get sparser (each measurement "
                                      "corrects more drift), but total information per 100 cycles falls."])
        manifest_button(cfg, "update_frequency", DATA_KEY)


def view_models() -> None:
    methods_panel()
    ml_section()
    physics_section()
    ablation_section()
    update_frequency_section()
    cross_cell_section()


# =============================================================================
# View 3: operations & optimal control
# =============================================================================
def view_ops() -> None:
    section("Twin-aware operating strategy")
    st.markdown("The twin-aware policy predicts each available discharge current with the ECM and degradation "
                "model, then picks the best objective value within the thermal and cold-weather limits. "
                "The *plant* can differ from the policy's *model* to test robustness.")
    c1, c2, c3, c4 = st.columns(4)
    policy = c1.selectbox("Policy", POLICIES)
    objective = c2.selectbox("Objective", ["cycle", "rate"], disabled=policy != "Twin-Aware",
                             format_func=lambda o: "Per-cycle margin" if o == "cycle"
                             else "Long-run profit rate (Dinkelbach)",
                             help="Finding: no one-step objective optimises the lifetime profit rate; the "
                                  "per-cycle margin scored best in tests. A DP policy is the principled fix.")
    weight = c3.slider("Degradation penalty weight w", 0.25, 3.0, 1.0, 0.25, disabled=policy != "Twin-Aware")
    replacement = c4.number_input("Replacement cost (CU)", 10.0, 2000.0, 150.0, 10.0)

    with st.expander("Economic, environmental and model-error settings"):
        p1, p2, p3, p4 = st.columns(4)
        price = (p1.number_input("Revenue per Ah at 1 A", 0.0, 10.0, 0.80, 0.05),
                 p2.number_input("Revenue per Ah at 2 A", 0.0, 10.0, 1.00, 0.05),
                 p3.number_input("Revenue per Ah at 4 A", 0.0, 10.0, 1.25, 0.05))
        e_price = p4.number_input("Charging energy price (CU/Wh)", 0.0, 0.5, 0.02, 0.005, format="%.3f",
                                  help="EnergyCost in J_op = Revenue − EnergyCost; resistive losses make aged "
                                       "cells more expensive to charge per delivered Ah.")
        s1, s2, s3, s4 = st.columns(4)
        t_max = s1.slider("Max cell temperature (°C)", 40, 60, 55)
        cold_rule = s2.toggle("Cold-temperature derating", value=True)
        cold_thr = s3.slider("Cold threshold (°C)", 0, 20, 10, disabled=not cold_rule)
        soh_eol_ops = s4.slider("Replacement SOH", 0.60, 0.85, 0.70, 0.01)
        e1, e2, e3, e4 = st.columns(4)
        amb_mean = e1.slider("Mean ambient (°C)", 0, 35, 20)
        amb_amp = e2.slider("Seasonal ambient amplitude (°C)", 0, 20, 16)
        mismatch = e3.slider("Plant / model mismatch", 0.0, 0.6, 0.0, 0.05,
                             help="Relative std of the perturbation applied to the plant's ageing, resistance, "
                                  "thermal and OCV parameters; the policy keeps the nominal model.")
        seed = e4.number_input("Mismatch draw (seed)", 0, 999, 0, 1, disabled=mismatch == 0)
        ekf_saved = st.session_state.get("ekf")
        use_twin = st.toggle("Initialise the model from the latest observer run", value=False,
                             disabled=ekf_saved is None, help="Available after a benchmark run in the Models view.")
        show_base = st.toggle("Overlay fixed-current baselines", value=True)

    econ = te.Economics(price_per_Ah=tuple(price), replacement_cost=float(replacement), soh_eol=float(soh_eol_ops),
                        degradation_weight=float(weight), T_max_C=float(t_max),
                        cold_derate_below_C=float(cold_thr) if cold_rule else None, objective=objective,
                        energy_price_per_Wh=float(e_price))
    phys = te.CellPhysics()
    if use_twin and ekf_saved:
        r = ekf_saved["res"]
        phys = te.CellPhysics(k_ah=float(r.params["k_ah"]), R_int0=float(r.r_int0), R_ct0=float(r.r_ct0))
    plant = te.perturb_physics(phys, float(mismatch), np.random.default_rng(int(seed))) if mismatch > 0 else None
    ops_cfg = dict(policy=policy, econ=asdict(econ), phys=asdict(phys), plant=asdict(plant) if plant else None,
                   amb_mean=amb_mean, amb_amp=amb_amp, show_base=show_base)

    if st.button("Run lifecycle simulation", type="primary", key="ops_go"):
        with st.spinner("Simulating to end of life…"):
            econ_run = econ
            rho_path: List[float] = []
            if policy == "Twin-Aware" and objective == "rate":
                rho, rho_path = tune_rho_cached(asdict(econ), asdict(phys), float(amb_mean), float(amb_amp))
                econ_run = replace(econ, rate_ref=rho)
            pl = asdict(plant) if plant else None
            df = run_policy(policy, asdict(econ_run), asdict(phys), float(amb_mean), float(amb_amp), pl)
            base = {b: run_policy(b, asdict(econ), asdict(phys), float(amb_mean), float(amb_amp), pl)
                    for b in POLICIES[1:] if show_base and b != policy}
            st.session_state["ops"] = {"cfg": ops_cfg, "df": df, "base": base, "policy": policy,
                                       "soh_eol": float(soh_eol_ops), "rho": rho_path}

    saved = st.session_state.get("ops")
    if saved:
        if saved["cfg"] != ops_cfg:
            st.warning("Settings changed since the last run; the results below use the previous settings.")
        if saved["df"].empty:
            st.warning("The simulation produced no cycles: the cell starts below the replacement SOH.")
        else:
            s = te.summarise_life(saved["df"])
            k = st.columns(7)
            k[0].metric("Cycles to replacement", s["cycles"])
            k[1].metric("J_op = revenue − energy", fmt(s["J_op"], ".1f", "CU"))
            k[2].metric("Energy cost", fmt(s["energy_cost"], ".1f", "CU"))
            k[3].metric("Net lifetime profit", fmt(s["profit"], ".1f", "CU"),
                        help="J_op minus the amortised replacement cost of the SOH consumed.")
            k[4].metric("Profit rate", fmt(s["profit_per_h"], ".3f", "CU/h"))
            k[5].metric("Round-trip energy efficiency", fmt(100 * s["energy_eff"], ".1f", "%"))
            k[6].metric("Constraint violations", s["violations"])
            if saved.get("rho"):
                st.caption("Dinkelbach iterations ρ: " + " → ".join(f"{v:.4f}" for v in saved["rho"]))
            show(fig_policy(saved["df"], saved["policy"], saved["base"], saved["soh_eol"], P), key="ops_fig",
                 data=saved["df"])
            rows = [{"Policy": saved["policy"], **s}] + \
                   [{"Policy": n, **te.summarise_life(d)} for n, d in saved["base"].items()]
            tbl = pd.DataFrame(rows).set_index("Policy")[["cycles", "Ah", "J_op", "energy_cost", "profit",
                                                          "profit_per_h", "energy_eff", "mean_I", "violations"]]
            tbl.columns = ["Cycles", "Throughput (Ah)", "J_op (CU)", "Energy cost (CU)", "Profit (CU)",
                           "Profit rate (CU/h)", "Energy efficiency", "Mean current (A)", "Violations"]
            show_table(tbl.style.format({"Throughput (Ah)": "{:.0f}", "J_op (CU)": "{:.1f}", "Energy cost (CU)": "{:.1f}",
                                         "Profit (CU)": "{:.1f}", "Profit rate (CU/h)": "{:.3f}",
                                         "Energy efficiency": "{:.1%}", "Mean current (A)": "{:.2f}"}, na_rep="—")
                       .highlight_max(subset=["Profit rate (CU/h)"],
                                      props="background-color: rgba(0,158,115,0.25); font-weight: 700;"))
            st.caption("Compare policies with zero violations: a fixed policy that breaches the thermal or "
                       "cold-derating limits can post a higher profit rate but is not an admissible competitor.")
            manifest_button(saved["cfg"], "operations", DATA_KEY)

    section("Robustness study: twin advantage under plant / model mismatch")
    m1, m2 = st.columns(2)
    levels = m1.multiselect("Mismatch levels", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], default=[0.0, 0.2, 0.4])
    draws = m2.slider("Random plants per level", 1, 8, 3)
    n_runs = len(POLICIES) * (1 + (len([x for x in levels if x > 0]) * draws))
    if st.button(f"Run mismatch study (≈ {1.5 * n_runs:.0f} s)", key="mm_go", disabled=not levels):
        bar = st.progress(0.0, text="Sampling plants…")
        try:
            ms = te.mismatch_study(econ, phys, levels=tuple(sorted(levels)), n_draws=int(draws),
                                   ambient_mean_C=float(amb_mean), ambient_amp_C=float(amb_amp),
                                   progress=lambda f, m: bar.progress(f, text=m))
            st.session_state["mm"] = {"df": ms, "cfg": dict(ops_cfg, levels=levels, draws=draws)}
        except Exception as exc:
            report_error("Mismatch study failed", exc, debug)
        finally:
            bar.empty()
    mm = st.session_state.get("mm")
    if mm:
        ms = mm["df"]
        show(fig_mismatch(ms, P), key="mm_fig", data=ms)
        tw = ms[ms["policy"] == "Twin-Aware"]
        agg = tw.groupby("level").agg(draws=("draw", "nunique"), mean_adv=("advantage_pct", "mean"),
                                      min_adv=("advantage_pct", "min"), max_adv=("advantage_pct", "max"),
                                      twin_violations=("twin_violations", "sum"))
        agg.columns = ["Plants", "Mean advantage (%)", "Worst (%)", "Best (%)", "Twin violations"]
        show_table(agg.style.format({"Mean advantage (%)": "{:+.1f}", "Worst (%)": "{:+.1f}", "Best (%)": "{:+.1f}"}))
        st.caption("Advantage = twin-aware profit rate relative to the best fixed-current policy that stays "
                   "within limits on the same plant. Simulation evidence only: the plant family is synthetic.")
        manifest_button(mm["cfg"], "mismatch", DATA_KEY)

    integrated_section(econ, phys, plant, float(amb_mean), float(amb_amp), ops_cfg)


def integrated_section(econ: te.Economics, phys: te.CellPhysics, plant: Optional[te.CellPhysics],
                       amb_mean: float, amb_amp: float, ops_cfg: Dict[str, Any]) -> None:
    section("Mission 3 · Integrated operation and maintenance optimisation")
    st.markdown("Operation and maintenance are optimised **jointly** with the same twin model. The operating knob is "
                "the policy's shadow price of SOH, *w* (0 = run for immediate margin, large = protect the cell); the "
                "maintenance knob is the replacement threshold. Each combination is scored by the long-run profit "
                "rate of the renewal cycle (install → replace), with **Profit = Revenue − EnergyCost − "
                "MaintenanceCost**, planned downtime, and a sudden-failure hazard after the knee (unplanned "
                "replacement is more expensive and takes longer).")
    with st.expander("Maintenance and risk model", expanded=False):
        m1, m2, m3, m4 = st.columns(4)
        maint = dict(
            replacement_cost=float(econ.replacement_cost),
            planned_downtime_h=float(m1.number_input("Planned downtime (h)", 0.0, 500.0, 24.0, 4.0)),
            unplanned_factor=float(m2.number_input("Unplanned cost multiplier", 1.0, 10.0, 3.0, 0.5)),
            unplanned_downtime_h=float(m3.number_input("Unplanned downtime (h)", 0.0, 1000.0, 120.0, 10.0)),
            hazard_soh=float(m4.slider("Sudden-death onset SOH", 0.55, 0.85, 0.68, 0.01,
                                       help="Hazard midpoint; set from the knee statistics of the cohort.")),
            h_max=float(st.select_slider("Peak failure hazard per cycle", [0.005, 0.01, 0.02, 0.03, 0.05, 0.1],
                                         value=0.03)))
        c1, c2 = st.columns(2)
        weights = tuple(c1.multiselect("Operating aggressiveness w (twin-aware)", [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
                                       default=[0.0, 0.5, 1.0, 2.0, 4.0]))
        thresholds = tuple(sorted(c2.multiselect("Replacement thresholds (SOH)",
                                                 [0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80,
                                                  0.82, 0.84, 0.86],
                                                 default=[0.60, 0.64, 0.68, 0.70, 0.72, 0.74, 0.76, 0.80, 0.84])))
    cfg = dict(ops_cfg, maint=maint, weights=list(weights), thresholds=list(thresholds))
    n_runs = len(weights) + 3
    if st.button(f"Run integrated optimisation (≈ {1.0 * n_runs:.0f} s)", key="om_go",
                 disabled=not (weights and thresholds)):
        with st.spinner("Simulating lives and evaluating replacement thresholds…"):
            try:
                study = om_study_cached(asdict(econ), asdict(phys), maint, amb_mean, amb_amp,
                                        asdict(plant) if plant else None, weights, thresholds)
                st.session_state["om"] = {"cfg": cfg, "study": study}
            except Exception as exc:
                report_error("Integrated optimisation failed", exc, debug)
    saved = st.session_state.get("om")
    if not saved:
        return
    if saved["cfg"] != cfg:
        st.warning("Settings changed since the last run; the results below use the previous settings.")
    study = saved["study"]
    opt = te.integrated_optimum(study)
    show(fig_om_heatmap(study, opt, P), key="om_heat", data=study)
    st.caption("✕ marks combinations that breach the thermal or cold-derating limits; they are excluded from the "
               "optimum. Hover for the sudden-failure probability.")
    show(fig_om_tradeoff(study, P), key="om_trade", export=False)
    if opt:
        b = opt["best"]
        k = st.columns(5)
        k[0].metric("Optimal policy", b["policy"])
        k[1].metric("Optimal replacement SOH", f"{b['threshold']:.2f}")
        k[2].metric("Long-run profit rate", fmt(b["rate"], ".4f", "CU/h"))
        k[3].metric("P(sudden failure) per life", fmt(100 * b["p_failure"], ".1f", "%"))
        k[4].metric("Availability", fmt(100 * b["availability"], ".1f", "%"))
        lines = []
        bf = opt.get("best_fixed")
        if bf and bf["rate"]:
            lines.append(f"Versus the best compliant fixed-current policy ({bf['policy']}, replace at "
                         f"{bf['threshold']:.2f}): {100 * (b['rate'] - bf['rate']) / abs(bf['rate']):+.1f}% profit rate")
        same = study[(study["policy"] == b["policy"])].set_index("threshold")
        lo_th, hi_th = same.index.min(), same.index.max()
        lines.append(f"Same policy, replacing at {lo_th:.2f} (run towards failure): rate {same.loc[lo_th, 'rate']:.4f}, "
                     f"P(failure) {same.loc[lo_th, 'p_failure']:.0%}; replacing at {hi_th:.2f} (early): rate "
                     f"{same.loc[hi_th, 'rate']:.4f}")
        lines.append(f"J_op {b['J_op']:.1f} CU and J_maint {b['J_maint']:.1f} CU per renewal "
                     f"(maintenance {b['maintenance_cost']:.1f} + performance loss {b['performance_loss']:.1f})")
        card("Mission 3 answer", lines)
    tbl = study[["policy", "threshold", "cycles", "J_op", "J_maint", "profit", "rate", "p_failure", "availability",
                 "violations"]].rename(columns={"policy": "Policy", "threshold": "Replace at SOH",
                                                "cycles": "Expected cycles", "profit": "Profit per renewal",
                                                "rate": "Profit rate (CU/h)", "p_failure": "P(failure)",
                                                "availability": "Availability", "violations": "Violations"})
    with st.expander("All combinations"):
        show_table(tbl.style.format({"Replace at SOH": "{:.2f}", "Expected cycles": "{:.0f}", "J_op": "{:.1f}",
                                     "J_maint": "{:.1f}", "Profit per renewal": "{:.1f}",
                                     "Profit rate (CU/h)": "{:.4f}", "P(failure)": "{:.1%}",
                                     "Availability": "{:.1%}"}, na_rep="—"))
    st.caption("Simulation evidence on the synthetic plant family (CellPhysics); the hazard model is an assumption "
               "calibrated to the knee literature, so treat the optimum's location, not its absolute value, as the "
               "result. With plant/model mismatch set above, the policy plans with the nominal model.")
    manifest_button(cfg, "integrated_om", DATA_KEY)


# =============================================================================
# Dispatch: only the active view runs
# =============================================================================
_VIEW_FN: Dict[str, Callable[[], None]] = {VIEWS[0]: view_data, VIEWS[1]: view_models, VIEWS[2]: view_ops}
try:
    _VIEW_FN.get(view, view_data)()
except Exception as exc:
    report_error(f"{view} failed", exc, debug)
