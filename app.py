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
import time
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
VIEWS = [":material/dashboard: Operations centre", ":material/monitoring: Data & diagnostics",
         ":material/psychology: Models & forecasting", ":material/tune: Operations & control"]
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
# 24 distinguishable, colour-blind-considerate hues (Okabe-Ito + Paul Tol bright/muted/vibrant)
CELL_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#882255", "#117733",
               "#DDCC77", "#332288", "#AA4499", "#44AA99", "#999933", "#EE7733", "#0077BB", "#CC3311",
               "#33BBEE", "#EE3377", "#228833", "#4477AA", "#AA3377", "#66CCEE", "#BBBB00", "#7F3C8D")
CELL_SYMBOLS = ("circle", "square", "diamond", "triangle-up", "triangle-down", "star", "hexagon", "pentagon",
                "cross", "x", "star-triangle-up", "hourglass")


def cell_style(cell_id: str, all_cells: Sequence[str]) -> Tuple[str, str]:
    """Stable colour + marker per battery across every chart (sorted cohort order)."""
    order = sorted(all_cells)
    i = order.index(cell_id) if cell_id in order else hash(cell_id)
    return CELL_COLORS[i % len(CELL_COLORS)], CELL_SYMBOLS[(i // len(CELL_COLORS) + i) % len(CELL_SYMBOLS)]


def cell_label(cell_id: str, meta: pd.DataFrame) -> str:
    if cell_id not in meta.index:
        return cell_id
    m = meta.loc[cell_id]
    return f"{cell_id} · {m['Ambient_C']:.0f} °C · {m['I_dis_A']:.1f} A · {m['V_cut_V']:.1f} V"
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
  --bt-border: rgba(128,128,128,0.28);
  --bt-grad: linear-gradient(120deg, #0B3D91 0%, #0072B2 38%, #009E73 72%, #56B4E9 100%);
  --bt-grad-warm: linear-gradient(90deg, #0072B2, #009E73, #E69F00);
  --bt-shadow: 0 1px 2px rgba(0,0,0,0.06), 0 6px 18px rgba(0,0,0,0.06);
}}
.block-container {{padding-top: 1.0rem; padding-bottom: 3rem; max-width: 1500px;}}
/* ---------- hero: fixed dark gradient, so white text is safe in both themes ---------- */
.bt-hero {{
  position: relative; overflow: hidden; border-radius: 18px; padding: 28px 34px 22px 34px; margin-bottom: 1.1rem;
  background: var(--bt-grad); box-shadow: 0 10px 30px rgba(0,60,120,0.25);
}}
.bt-hero::before {{
  content: ""; position: absolute; inset: 0; opacity: 0.18; pointer-events: none;
  background-image: linear-gradient(rgba(255,255,255,0.35) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,0.35) 1px, transparent 1px);
  background-size: 28px 28px; mask-image: linear-gradient(90deg, transparent 0%, black 60%);
  -webkit-mask-image: linear-gradient(90deg, transparent 0%, black 60%);
}}
.bt-hero::after {{
  content: ""; position: absolute; right: -60px; top: -80px; width: 320px; height: 320px; border-radius: 50%;
  background: radial-gradient(circle, rgba(255,255,255,0.28), rgba(255,255,255,0) 70%); pointer-events: none;
}}
.bt-hero .bt-title {{position: relative; font-size: 1.9rem; font-weight: 850; letter-spacing: -0.025em;
  line-height: 1.2; color: #FFFFFF;}}
.bt-hero .bt-sub {{position: relative; margin-top: 8px; font-size: 1.0rem; line-height: 1.55; color: rgba(255,255,255,0.92);
  max-width: 92ch;}}
.bt-hero .bt-kicker {{position: relative; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: rgba(255,255,255,0.80); margin-bottom: 6px;}}
.bt-pill {{
  position: relative; display: inline-block; padding: 4px 12px; margin: 12px 8px 0 0; border-radius: 999px;
  font-size: 0.8rem; font-weight: 650; color: #FFFFFF; background: rgba(255,255,255,0.14);
  border: 1px solid rgba(255,255,255,0.35); backdrop-filter: blur(6px);
}}
.bt-chip {{
  display: inline-block; padding: 3px 11px; margin: 4px 6px 0 0; border-radius: 999px;
  font-size: 0.82rem; background: var(--bt-surface); border: 1px solid var(--bt-border); color: inherit;
}}
.bt-status {{font-size: 0.95rem; line-height: 1.6; color: inherit;}}
/* ---------- numbered section headers ---------- */
.bt-section {{
  display: flex; align-items: center; gap: 12px; font-size: 1.22rem; font-weight: 800;
  margin: 2.2rem 0 0.8rem 0; color: inherit; letter-spacing: -0.01em;
  padding-bottom: 8px; border-bottom: 1px solid var(--bt-border);
}}
.bt-section .bt-num {{
  flex: none; display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px;
  border-radius: 9px; background: var(--bt-grad); color: #FFFFFF; font-size: 0.9rem; font-weight: 800;
  box-shadow: 0 3px 10px rgba(0,114,178,0.35);
}}
/* ---------- cards & KPI tiles ---------- */
.bt-card {{
  background: var(--bt-surface); border: 1px solid var(--bt-border); border-left: 4px solid var(--bt-accent);
  border-radius: 12px; padding: 16px 20px; margin: 10px 0 16px 0; box-shadow: var(--bt-shadow);
}}
.bt-card h4 {{margin: 0 0 8px 0; font-size: 0.98rem; font-weight: 800; color: var(--bt-accent);}}
.bt-card li {{font-size: 0.93rem; line-height: 1.55; margin: 3px 0; color: inherit;}}
div[data-testid="stMetric"] {{
  position: relative; overflow: hidden; background: var(--bt-surface); border: 1px solid var(--bt-border);
  border-radius: 14px; padding: 16px 18px 14px 18px; box-shadow: var(--bt-shadow);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
div[data-testid="stMetric"]::before {{
  content: ""; position: absolute; left: 0; top: 0; right: 0; height: 4px; background: var(--bt-grad-warm);
}}
div[data-testid="stMetric"]:hover {{transform: translateY(-2px); box-shadow: 0 10px 24px rgba(0,0,0,0.10);}}
div[data-testid="stMetricLabel"] p {{font-size: 0.8rem; font-weight: 700; opacity: 0.75; letter-spacing: 0.01em;}}
div[data-testid="stMetricValue"] {{font-weight: 800; letter-spacing: -0.01em;}}
/* ---------- plots, expanders, tabs, buttons ---------- */
div[data-testid="stPlotlyChart"] {{
  border: 1px solid var(--bt-border); border-radius: 14px; padding: 6px 4px 2px 4px; background: var(--bt-surface);
  box-shadow: var(--bt-shadow); margin-bottom: 0.4rem;
}}
div[data-testid="stExpander"] details {{border-radius: 12px; border: 1px solid var(--bt-border);}}
div[data-testid="stExpander"] summary {{font-weight: 650;}}
button[data-baseweb="tab"] {{font-weight: 700; font-size: 0.98rem;}}
div[data-baseweb="tab-highlight"] {{background: var(--bt-grad-warm) !important; height: 3px !important;}}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
  background: var(--bt-grad); border: 0; color: #FFFFFF; font-weight: 700; border-radius: 10px;
  box-shadow: 0 4px 14px rgba(0,114,178,0.35);
}}
.stButton > button[kind="primary"]:hover {{filter: brightness(1.08); box-shadow: 0 6px 18px rgba(0,114,178,0.45);}}
div[data-testid="stDataFrame"], div[data-testid="stTable"] {{border-radius: 12px; overflow: hidden;}}
section[data-testid="stSidebar"] {{border-right: 1px solid var(--bt-border);}}
/* ---------- animated hero gradient ---------- */
.bt-hero {{background-size: 220% 220%; animation: bt-flow 18s ease-in-out infinite;}}
@keyframes bt-flow {{0% {{background-position: 0% 50%;}} 50% {{background-position: 100% 50%;}} 100% {{background-position: 0% 50%;}}}}
/* ---------- industrial status bar ---------- */
.bt-statusbar {{
  display: flex; flex-wrap: wrap; align-items: center; gap: 18px; margin: 4px 0 14px 0; padding: 9px 16px;
  border-radius: 12px; font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace; font-size: 0.78rem;
  letter-spacing: 0.06em; background: #0E1726; color: #9FB3C8; border: 1px solid #1F2E45;
  box-shadow: inset 0 0 0 1px rgba(86,180,233,0.08);
}}
.bt-statusbar b {{color: #E6EEF7; font-weight: 700;}}
.bt-statusbar .bt-clock {{margin-left: auto; color: #6B819A;}}
.bt-live {{display: inline-flex; align-items: center; gap: 8px; color: #3DDC97; font-weight: 800;}}
.bt-dot {{width: 9px; height: 9px; border-radius: 50%; background: #3DDC97; box-shadow: 0 0 0 0 rgba(61,220,151,0.7);
  animation: bt-pulse 1.8s infinite;}}
@keyframes bt-pulse {{0% {{box-shadow: 0 0 0 0 rgba(61,220,151,0.65);}} 70% {{box-shadow: 0 0 0 9px rgba(61,220,151,0);}}
  100% {{box-shadow: 0 0 0 0 rgba(61,220,151,0);}}}}
.bt-sev {{padding: 2px 9px; border-radius: 6px; font-weight: 800;}}
.bt-sev-crit {{background: rgba(213,94,0,0.18); color: #FF8A4C; border: 1px solid rgba(213,94,0,0.45);}}
.bt-sev-warn {{background: rgba(230,159,0,0.16); color: #F5C04A; border: 1px solid rgba(230,159,0,0.40);}}
/* ---------- alert banner ---------- */
.bt-alert {{display: flex; gap: 12px; align-items: flex-start; padding: 12px 16px; border-radius: 12px; margin: 6px 0 12px 0;
  border: 1px solid var(--bt-border); background: var(--bt-surface); box-shadow: var(--bt-shadow);}}
.bt-alert .bt-badge {{flex: none; padding: 3px 10px; border-radius: 8px; color: #fff; font-weight: 800; font-size: 0.8rem;}}
.bt-alert .bt-msg {{font-size: 0.93rem; line-height: 1.5; color: inherit;}}
</style>"""
    st.markdown(css, unsafe_allow_html=True)


# =============================================================================
# Small UI helpers
# =============================================================================
def section(title: str, icon: Optional[str] = None) -> None:
    """Numbered section header (numbering restarts in every view)."""
    n = st.session_state.get("_sec_n", 0) + 1
    st.session_state["_sec_n"] = n
    st.markdown(f'<div class="bt-section"><span class="bt-num">{n}</span><span>{html.escape(title)}</span></div>',
                unsafe_allow_html=True)


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


def fig_cohort_grid(ct_all: pd.DataFrame, meta: pd.DataFrame, cell_id: str, P: Palette,
                    normalise_x: bool = False) -> go.Figure:
    """One full-width panel per ambient temperature, stacked; every battery has its own colour and
    marker (the same as in every other chart) and its own legend entry."""
    amb = meta["Ambient_C"].round(0)
    groups = sorted(amb.dropna().unique())
    nrow = max(len(groups), 1)
    fig = make_subplots(rows=nrow, cols=1, shared_xaxes=False, vertical_spacing=0.28 / nrow + 0.04,
                        subplot_titles=[f"Ambient {g:.0f} °C · {int((amb == g).sum())} cells" for g in groups])
    _style_subplot_titles(fig, P)
    good = ct_all[~ct_all["outlier"]]
    cells = list(meta.index)
    for gi, g in enumerate(groups):
        for cid in sorted(amb[amb == g].index):
            d = good[good["Cell_ID"] == cid].sort_values("n")
            if d.empty:
                continue
            col, sym = cell_style(cid, cells)
            is_t = cid == cell_id
            x = d["n"] / d["n"].max() if normalise_x else d["n"]
            fig.add_trace(go.Scatter(
                x=x, y=d["SOH"], mode="lines+markers", name=cell_label(cid, meta) + ("  ★ target" if is_t else ""),
                legendgroup=cid, line=dict(color=col, width=3.4 if is_t else 1.8),
                marker=dict(symbol=sym, size=7 if is_t else 5, maxdisplayed=14, line=dict(color=P.plot_bg, width=0.5)),
                hovertemplate=f"<b>{cid}</b> n=%{{x}}: SOH %{{y:.3f}}<extra></extra>"), row=gi + 1, col=1)
        fig.update_yaxes(title_text="SOH (–)", row=gi + 1, col=1)
        fig.update_xaxes(title_text="Fraction of recorded life" if normalise_x else "Discharge cycle n",
                         row=gi + 1, col=1)
    fig.update_layout(legend=dict(groupclick="togglegroup"))
    return style_fig(fig, P, 200 + 290 * nrow, "Cohort fade by ambient temperature (click a legend entry to hide a cell)",
                     hovermode="closest")


TRACE_MODES = ("Single cycle", "Choose cycles", "Every k-th cycle", "Range of cycles", "All cycles")


def fig_cycle_traces(prep: pd.DataFrame, picks: List[Tuple[int, int]], P: Palette, x_axis: str = "time",
                     max_pts: int = 500) -> go.Figure:
    """Overlay raw telemetry of several discharges, coloured along life (Viridis, early = dark).
    x_axis = "time" (minutes into the cycle) or "capacity" (discharged Ah: aligns curves of
    different length and shows capacity fade and polarisation directly)."""
    picks = sorted(picks, key=lambda p: p[1])
    many = len(picks) > 10
    cols = sample_colorscale("Viridis", list(np.linspace(*P.ica_range, max(len(picks), 2))))[: len(picks)]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Terminal voltage", "Current", "Cell temperature"))
    _style_subplot_titles(fig, P)
    ns = [n for _, n in picks]
    for (ci, n), col in zip(picks, cols):
        d = prep[prep["Cycle_Index"] == ci]
        if d.empty:
            continue
        if len(d) > max_pts:
            d = d.iloc[np.linspace(0, len(d) - 1, max_pts).astype(int)]
        if x_axis == "capacity":
            x = np.cumsum(np.clip(-d["Current_A"].to_numpy(), 0, None) * d["dt"].to_numpy()) / 3600.0
        else:
            x = d["Time_s"].to_numpy() / 60.0
        for row, key, unit in ((1, "Voltage_V", "V"), (2, "Current_A", "A"), (3, "Temp_C", "°C")):
            fig.add_trace(go.Scatter(
                x=x, y=d[key], mode="lines", name=f"n = {n}", legendgroup=f"n{n}",
                showlegend=(row == 1 and not many), line=dict(color=col, width=1.4 if many else 2.2),
                hovertemplate=f"n = {n}: %{{y:.3f}} {unit}<extra></extra>"), row=row, col=1)
    if many:                                      # colour bar instead of a long legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(colorscale=[[0, cols[0]], [1, cols[-1]]], cmin=min(ns), cmax=max(ns),
                                             color=[min(ns)], showscale=True,
                                             colorbar=dict(title=dict(text="cycle n", font=dict(color=P.text)),
                                                           tickfont=dict(color=P.muted), thickness=12, len=0.9))),
                      row=1, col=1)
    fig.update_yaxes(title_text="V", row=1, col=1)
    fig.update_yaxes(title_text="A", row=2, col=1)
    fig.update_yaxes(title_text="°C", row=3, col=1)
    fig.update_xaxes(title_text="Discharged capacity (Ah)" if x_axis == "capacity" else "Time in cycle (min)",
                     row=3, col=1)
    title = (f"Raw telemetry, discharge n = {ns[0]}" if len(ns) == 1 else
             f"Raw telemetry of {len(ns)} discharges (n = {min(ns)} to {max(ns)})")
    return style_fig(fig, P, 720, title)


def fig_fade_compare(ct_all: pd.DataFrame, meta: pd.DataFrame, cells: Sequence[str], y: str,
                     eol_line: Optional[float], P: Palette, knees: Dict[str, Dict[str, Any]],
                     show_cohort: bool = True, x_mode: str = "n") -> go.Figure:
    """Fade trajectories of 1 - 8 selected batteries (own colour + marker each) over the muted cohort;
    regeneration (▲), outliers (×) and knee points (★) per selected cell."""
    fig = go.Figure()
    allc = list(meta.index)
    xcol = {"n": "n", "Ah": "cum_Ah", "frac": "_frac"}[x_mode]
    d_all = ct_all.copy()
    d_all["_frac"] = d_all["n"] / d_all.groupby("Cell_ID")["n"].transform("max")
    if show_cohort:
        bg = d_all[~d_all["outlier"] & ~d_all["Cell_ID"].isin(cells)].sort_values(["Cell_ID", xcol])
        xs, ys = [], []
        for _, d in bg.groupby("Cell_ID"):
            xs += d[xcol].tolist() + [None]
            ys += d[y].tolist() + [None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Other cells", line=dict(color=P.cohort, width=1),
                                 hoverinfo="skip", opacity=0.7))
    for cid in cells:
        d = d_all[d_all["Cell_ID"] == cid].sort_values("n")
        good = d[~d["outlier"]]
        col, sym = cell_style(cid, allc)
        fig.add_trace(go.Scatter(x=good[xcol], y=good[y], mode="lines+markers", name=cell_label(cid, meta),
                                 legendgroup=cid, customdata=np.column_stack([good["n"], [cid] * len(good)]),
                                 line=dict(color=col, width=2.8), marker=dict(symbol=sym, size=6, color=col),
                                 hovertemplate=f"<b>{cid}</b> n=%{{customdata[0]}}: %{{y:.4f}}<extra></extra>"))
        rg = good[good["regen"]]
        if len(rg):
            fig.add_trace(go.Scatter(x=rg[xcol], y=rg[y], mode="markers", name=f"{cid} regeneration",
                                     legendgroup=cid, showlegend=False,
                                     marker=dict(symbol="triangle-up", size=11, color=col,
                                                 line=dict(color=P.text, width=1)),
                                     hovertemplate=f"{cid} regeneration n=%{{x}}<extra></extra>"))
        ol = d[d["outlier"]]
        if len(ol):
            fig.add_trace(go.Scatter(x=ol[xcol], y=ol[y].clip(upper=1.2 if y == "SOH" else None), mode="markers",
                                     name=f"{cid} excluded", legendgroup=cid, showlegend=False,
                                     marker=dict(symbol="x", size=8, color=col, opacity=0.6),
                                     hovertemplate=f"{cid} excluded point<extra></extra>"))
        k = knees.get(cid, {})
        if k.get("found"):
            kr = good.iloc[(good["n"] - k["knee_n"]).abs().argsort()[:1]]
            fig.add_trace(go.Scatter(x=kr[xcol], y=kr[y], mode="markers", name=f"{cid} knee", legendgroup=cid,
                                     showlegend=False, marker=dict(symbol="star", size=18, color=col,
                                                                   line=dict(color=P.text, width=1.5)),
                                     hovertemplate=f"{cid} knee at n = {k['knee_n']}<extra></extra>"))
    if eol_line is not None:
        fig.add_hline(y=eol_line, line_dash="dash", line_color=P.eol, line_width=1.5,
                      annotation_text="End of life", annotation_position="bottom right",
                      annotation_font=dict(color=P.eol, size=12))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="▲ regeneration · × excluded · ★ knee",
                             marker=dict(color="rgba(0,0,0,0)"), hoverinfo="skip"))
    fig.update_xaxes(title_text={"n": "Discharge cycle n", "Ah": "Cumulative throughput (Ah)",
                                 "frac": "Fraction of recorded life"}[x_mode])
    fig.update_yaxes(title_text="State of health SOH (–)" if y == "SOH" else "Capacity (Ah)")
    ttl = (f"Capacity fade, {cells[0]}" if len(cells) == 1 else f"Capacity fade: {len(cells)} batteries compared")
    return style_fig(fig, P, 560, ttl, hovermode="closest")


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
            col, _ = model_style(r.model)
            add_band(fig, r.n_grid, r.soh_lo, r.soh_hi, col, f"{r.model} {int(100 * (r.band_level or 0.9))}% band",
                     n_from=n0, group=r.model, alpha=0.12)
    fig.add_trace(go.Scatter(x=good["n"], y=good["SOH"], mode="markers", name="Measured SOH",
                             marker=dict(color=P.measured, size=6, opacity=0.85),
                             hovertemplate="%{y:.4f}<extra></extra>"))
    for i, r in enumerate(results):
        col, sym = model_style(r.model)
        dash = DASHES[i % len(DASHES)]
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


def model_style(name: str) -> Tuple[str, str]:
    models = list(te.ML_MODELS)
    i = models.index(name) if name in models else 0
    return CELL_COLORS[(i * 5) % len(CELL_COLORS)], CELL_SYMBOLS[i % len(CELL_SYMBOLS)]


def fig_leaderboard(tbl: pd.DataFrame, metric: str, P: Palette, title: str, higher_better: bool = True) -> go.Figure:
    t = tbl.dropna(subset=[metric]).sort_values(metric, ascending=not higher_better)
    fig = go.Figure(go.Bar(
        y=t.index, x=t[metric], orientation="h", text=[f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}" for v in t[metric]],
        textposition="outside", textfont=dict(color=P.text),
        marker=dict(color=[model_style(m)[0] for m in t.index], line=dict(color=P.text, width=0.5)),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>", showlegend=False))
    fig.update_xaxes(title_text=metric)
    fig.update_yaxes(autorange="reversed", automargin=True)
    return style_fig(fig, P, 140 + 34 * len(t), title, hovermode="closest")


def fig_parity(df: pd.DataFrame, color_by: str, P: Palette, title: str, all_cells: Sequence[str] = ()) -> go.Figure:
    """Measured vs predicted SOH (test points), one colour per model or per battery."""
    fig = go.Figure()
    lo = float(min(df["SOH"].min(), df["SOH_pred"].min())) - 0.01
    hi = float(max(df["SOH"].max(), df["SOH_pred"].max())) + 0.01
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                             line=dict(color=P.muted, dash="dash", width=1.5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[lo, hi, hi, lo], y=[lo - 0.02, hi - 0.02, hi + 0.02, lo + 0.02], fill="toself",
                             fillcolor=rgba(P.muted, 0.10), line=dict(width=0), name="±0.02 SOH", hoverinfo="skip"))
    for key, d in df.groupby(color_by, sort=False):
        col, sym = model_style(key) if color_by == "model" else cell_style(key, all_cells or df[color_by].unique())
        fig.add_trace(go.Scatter(x=d["SOH"], y=d["SOH_pred"], mode="markers", name=str(key),
                                 marker=dict(color=col, symbol=sym, size=7, opacity=0.8,
                                             line=dict(color=P.plot_bg, width=0.5)),
                                 hovertemplate=f"{key}<br>measured %{{x:.4f}}<br>predicted %{{y:.4f}}<extra></extra>"))
    fig.update_xaxes(title_text="Measured SOH", range=[lo, hi])
    fig.update_yaxes(title_text="Predicted SOH", range=[lo, hi], scaleanchor="x", scaleratio=1)
    return style_fig(fig, P, 620, title, hovermode="closest")


def fig_importance(imp: pd.DataFrame, P: Palette, model: str) -> go.Figure:
    t = imp.sort_values("Importance (ΔRMSE)")
    fig = go.Figure(go.Bar(y=t["Indicator"], x=t["Importance (ΔRMSE)"], orientation="h",
                           error_x=dict(type="data", array=t["std"], color=P.muted),
                           marker=dict(color=P.accent), showlegend=False,
                           hovertemplate="%{y}: +%{x:.4f} RMSE when shuffled<extra></extra>"))
    fig.update_xaxes(title_text="RMSE increase when the indicator is shuffled (test set)")
    fig.update_yaxes(automargin=True)
    return style_fig(fig, P, 160 + 36 * len(t), f"Which indicators does {model} rely on? Permutation importance",
                     hovermode="closest")


def fig_est_traj(pred: pd.DataFrame, P: Palette, model: str, all_cells: Sequence[str]) -> go.Figure:
    fig = go.Figure()
    for cid, d in pred.groupby("Cell_ID"):
        col, sym = cell_style(cid, all_cells)
        d = d.sort_values("n")
        fig.add_trace(go.Scatter(x=d["n"], y=d["SOH"], mode="markers", name=f"{cid} measured", legendgroup=cid,
                                 marker=dict(color=col, symbol=sym, size=6, opacity=0.55),
                                 hovertemplate=f"{cid} measured %{{y:.4f}}<extra></extra>"))
        te_ = d[d["set"] == "test"]
        fig.add_trace(go.Scatter(x=te_["n"], y=te_["SOH_pred"], mode="lines", name=f"{cid} estimated (test)",
                                 legendgroup=cid, line=dict(color=col, width=2.6),
                                 hovertemplate=f"{cid} estimated %{{y:.4f}}<extra></extra>"))
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="SOH (–)")
    return style_fig(fig, P, 520, f"{model}: SOH estimated from operando indicators on the test cycles")


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
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.16,
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
                                 hovertemplate="%{y:.4f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=level, line_dash="dash", line_color=P.eol, row=1, col=1,
                  annotation_text=f"nominal {int(level * 100)}%", annotation_font=dict(color=P.eol, size=11))
    fig.update_yaxes(title_text="Empirical coverage", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="SOH RMSE", row=2, col=1)
    fig.update_xaxes(title_text="Forecast horizon h (cycles)")
    return style_fig(fig, P, 770, "Calibration and error growth over the forecast horizon")


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
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.16,
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
                             hovertemplate="%{x:.2f}<extra>" + pc + "</extra>"), row=2, col=1)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="%", range=[0, 105], row=1, col=1)
    fig.update_xaxes(title_text="Loading", row=2, col=1)
    fig.update_yaxes(automargin=True, row=2, col=1)
    return style_fig(fig, P, 805, "Can degradation be represented by one parameter? PCA of the health indicators",
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
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.16,
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
                                 hovertemplate="%{y:.1f}%<extra>" + col + "</extra>"), row=2, col=1)
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="%", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    return style_fig(fig, P, 770, "Degradation-mode trajectories: LLI · LAM · conductivity loss (CL)")


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
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.16,
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
                             hovertemplate="%{y:.4f}<extra>tracking RMSE</extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=tab["Forecast RMSE"], mode="lines+markers", name="Forecast RMSE from n₀",
                             line=dict(color=P.pinn, width=2.4, dash="dash"), marker=dict(symbol="diamond", size=9),
                             hovertemplate="%{y:.4f}<extra>forecast RMSE</extra>"), row=2, col=1)
    fig.update_xaxes(title_text="Discharge cycle n", row=1, col=1)
    fig.update_xaxes(title_text="Measurement updates per 100 cycles", type="log", row=2, col=1)
    fig.update_yaxes(title_text="SOH (–)", row=1, col=1)
    fig.update_yaxes(title_text="SOH RMSE", row=2, col=1)
    return style_fig(fig, P, 840, "How often should the twin update?")


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
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.16,
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
                                 hovertemplate="%{y:.2f}<extra>" + html.escape(pol) + "</extra>"), row=2, col=1)
    fig.update_xaxes(title_text="Replacement threshold (SOH)")
    fig.update_yaxes(title_text="CU/h", row=1, col=1)
    fig.update_yaxes(title_text="P(failure before replacement)", range=[0, 1.02], row=2, col=1)
    return style_fig(fig, P, 822, "Maintenance trade-off: replace early (cost) or late (risk and performance loss)")


# =============================================================================
# Figures: operations centre (fleet monitoring)
# =============================================================================
RISK_COLORS = {"Healthy": "#009E73", "Watch": "#E6B800", "Warning": "#E69F00", "Critical": "#D55E00"}
RISK_ICON = {"Healthy": "🟢", "Watch": "🟡", "Warning": "🟠", "Critical": "🔴"}


def fig_gauges(row: pd.Series, P: Palette, limits: te.SafetyLimits) -> go.Figure:
    """Instrument cluster for one battery: SOH, quick RUL, resistance growth, peak temperature."""
    fig = go.Figure()
    eol = float(row["SOH_EOL"])
    common = dict(bgcolor=rgba(P.muted, 0.08), borderwidth=0)
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=100 * float(row["SOH"]), number=dict(suffix=" %", font=dict(size=34, color=P.text)),
        title=dict(text="State of health", font=dict(size=14, color=P.muted)), domain=dict(x=[0.0, 0.22], y=[0, 1]),
        gauge=dict(axis=dict(range=[50, 105], tickcolor=P.muted, tickfont=dict(color=P.muted)),
                   bar=dict(color=P.accent, thickness=0.28),
                   steps=[dict(range=[50, 100 * eol], color=rgba("#D55E00", 0.35)),
                          dict(range=[100 * eol, 100 * eol + 10], color=rgba("#E69F00", 0.30)),
                          dict(range=[100 * eol + 10, 105], color=rgba("#009E73", 0.25))],
                   threshold=dict(line=dict(color=P.eol, width=4), thickness=0.85, value=100 * eol), **common)))
    rul = float(row["Quick RUL"])
    fig.add_trace(go.Indicator(
        mode="number", value=rul if np.isfinite(rul) else 999,
        number=dict(suffix=" cyc" if np.isfinite(rul) else "+ cyc", font=dict(size=40, color=RISK_COLORS[row["Risk"]]),
                    valueformat=".0f"),
        title=dict(text=f"Quick RUL<br><span style='font-size:12px'>{RISK_ICON[row['Risk']]} {row['Risk']}</span>",
                   font=dict(size=14, color=P.muted)), domain=dict(x=[0.27, 0.47], y=[0.1, 0.9])))
    rg = float(row["R growth (%)"]) if np.isfinite(row["R growth (%)"]) else 0.0
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=rg, number=dict(suffix=" %", font=dict(size=30, color=P.text), valueformat="+.0f"),
        title=dict(text="Resistance growth", font=dict(size=14, color=P.muted)), domain=dict(x=[0.52, 0.74], y=[0, 1]),
        gauge=dict(axis=dict(range=[min(-20, rg), max(150, rg)], tickcolor=P.muted, tickfont=dict(color=P.muted)),
                   bar=dict(color=P.r_ct, thickness=0.28),
                   steps=[dict(range=[min(-20, rg), 50], color=rgba("#009E73", 0.22)),
                          dict(range=[50, 100], color=rgba("#E69F00", 0.28)),
                          dict(range=[100, max(150, rg)], color=rgba("#D55E00", 0.30))], **common)))
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=float(row["Peak T (°C)"]),
        number=dict(suffix=" °C", font=dict(size=30, color=P.text), valueformat=".1f"),
        title=dict(text="Peak cell temperature", font=dict(size=14, color=P.muted)), domain=dict(x=[0.79, 1.0], y=[0, 1]),
        gauge=dict(axis=dict(range=[0, 80], tickcolor=P.muted, tickfont=dict(color=P.muted)),
                   bar=dict(color=P.eis, thickness=0.28),
                   steps=[dict(range=[0, limits.T_warn_C], color=rgba("#009E73", 0.22)),
                          dict(range=[limits.T_warn_C, limits.T_crit_C], color=rgba("#E69F00", 0.30)),
                          dict(range=[limits.T_crit_C, 80], color=rgba("#D55E00", 0.32))],
                   threshold=dict(line=dict(color=P.eol, width=3), thickness=0.8, value=limits.T_crit_C), **common)))
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=P.text))
    return fig


def fig_fleet_map(fs: pd.DataFrame, P: Palette) -> go.Figure:
    """Treemap: fleet → ambient group → battery. Tile area = cycles run, colour = health margin
    (1 = new, 0 = at end of life, < 0 = past it)."""
    d = fs.reset_index()
    d["group"] = d["Ambient_C"].round(0).map(lambda t: f"{t:.0f} °C")
    ids, labels, parents, values, colors, text = ["fleet"], ["Fleet"], [""], [0], [float(d["Health margin"].mean())], [""]
    for g, gg in d.groupby("group"):
        ids.append(f"g/{g}"); labels.append(f"Ambient {g}"); parents.append("fleet"); values.append(0)
        colors.append(float(gg["Health margin"].mean())); text.append(f"{len(gg)} cells")
        for _, r in gg.iterrows():
            ids.append(f"c/{r['Cell_ID']}"); labels.append(r["Cell_ID"]); parents.append(f"g/{g}")
            values.append(max(int(r["Cycles"]), 1)); colors.append(float(r["Health margin"]))
            text.append(f"SOH {100 * r['SOH']:.1f}%<br>{RISK_ICON[r['Risk']]} {r['Risk']}")
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values, text=text, branchvalues="remainder",
        texttemplate="<b>%{label}</b><br>%{text}", textfont=dict(size=14),
        marker=dict(colors=colors, colorscale=[[0, "#D55E00"], [0.35, "#E69F00"], [0.6, "#F0E442"], [1, "#009E73"]],
                    cmin=0, cmax=1, line=dict(color=P.plot_bg, width=2),
                    colorbar=dict(title=dict(text="Health margin", font=dict(color=P.text)), tickformat=".0%",
                                  tickfont=dict(color=P.muted), thickness=12)),
        hovertemplate="<b>%{label}</b><br>%{text}<br>health margin %{color:.0%}<extra></extra>",
        pathbar=dict(visible=True)))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=P.text),
                      title=dict(text="Fleet health map (area = cycles run, colour = remaining health margin)",
                                 font=dict(size=16, color=P.text), x=0.01))
    return fig


def fig_risk_matrix(fs: pd.DataFrame, P: Palette, cell_id: str) -> go.Figure:
    """Risk matrix: remaining life (x) against degradation speed (y)."""
    d = fs.reset_index()
    cap = float(np.nanmax(d["Quick RUL"].replace(np.inf, np.nan))) if np.isfinite(d["Quick RUL"].replace(np.inf, np.nan)).any() else 300
    cap = max(cap * 1.2, 50)
    x = d["Quick RUL"].replace(np.inf, cap).clip(lower=1)
    fig = go.Figure()
    fig.add_vrect(x0=1, x1=15, fillcolor=rgba("#D55E00", 0.10), line_width=0)
    fig.add_vrect(x0=15, x1=50, fillcolor=rgba("#E69F00", 0.08), line_width=0)
    for risk in te.RISK_LEVELS:
        m = d["Risk"] == risk
        if not m.any():
            continue
        fig.add_trace(go.Scatter(
            x=x[m], y=d.loc[m, "Fade per 100 cycles (%)"], mode="markers+text", name=f"{RISK_ICON[risk]} {risk}",
            text=d.loc[m, "Cell_ID"], textposition="top center", textfont=dict(size=11, color=P.text),
            marker=dict(size=10 + 16 * np.sqrt(d.loc[m, "Cycles"] / d["Cycles"].max()), color=RISK_COLORS[risk],
                        opacity=0.85, line=dict(width=[3 if c == cell_id else 1 for c in d.loc[m, "Cell_ID"]],
                                                color=P.text)),
            customdata=np.column_stack([d.loc[m, "SOH"], d.loc[m, "Alerts"]]),
            hovertemplate="<b>%{text}</b><br>quick RUL %{x:.0f} cycles<br>fade %{y:.2f} %/100 cycles<br>"
                          "SOH %{customdata[0]:.3f}<br>%{customdata[1]}<extra></extra>"))
    fig.update_xaxes(title_text="Quick RUL (cycles, log scale; right edge = not declining)", type="log")
    fig.update_yaxes(title_text="Recent fade (SOH % per 100 cycles)")
    return style_fig(fig, P, 520, "Risk matrix: remaining life versus degradation speed (size = cycles run)",
                     hovermode="closest")


def fig_scenarios(sp: pd.DataFrame, soh_eol: float, P: Palette) -> go.Figure:
    fig = go.Figure()
    for i, (name, d) in enumerate(sp.groupby("scenario", sort=False)):
        col = CELL_COLORS[i % len(CELL_COLORS)]
        add_band(fig, d["n"].to_numpy(), d["lo"].to_numpy(), d["hi"].to_numpy(), col, f"{name} band", group=name,
                 alpha=0.13)
        life = d["life_to_EOL"].iloc[0]
        fig.add_trace(go.Scatter(x=d["n"], y=d["SOH"], mode="lines", name=f"{name} · EOL at {life if life else '>'} cycles",
                                 legendgroup=name, line=dict(color=col, width=3, dash=DASHES[i % len(DASHES)]),
                                 hovertemplate="%{y:.3f}<extra>" + html.escape(name) + "</extra>"))
    fig.add_hline(y=soh_eol, line_dash="dash", line_color=P.eol, annotation_text="End of life",
                  annotation_font=dict(color=P.eol))
    fig.update_xaxes(title_text="Cycle n")
    fig.update_yaxes(title_text="Projected SOH (–)")
    return style_fig(fig, P, 520, "What-if: projected fade for each operating scenario (cohort stress-factor law)")


def build_report_html(title: str, sections: List[Tuple[str, Any]]) -> str:
    """Self-contained HTML report: each section is (heading, plotly figure | DataFrame | text)."""
    parts, first = [], True
    for head, obj in sections:
        parts.append(f"<h2>{html.escape(head)}</h2>")
        if isinstance(obj, go.Figure):
            parts.append(obj.to_html(full_html=False, include_plotlyjs="cdn" if first else False))
            first = False
        elif isinstance(obj, pd.DataFrame):
            parts.append(obj.to_html(classes="tbl", float_format=lambda v: f"{v:.4g}", border=0, na_rep="—"))
        else:
            parts.append(f"<p>{html.escape(str(obj))}</p>")
    css = ("body{font-family:Inter,Segoe UI,Arial,sans-serif;max-width:1180px;margin:32px auto;color:#1f2933;}"
           "header{background:linear-gradient(120deg,#0B3D91,#0072B2 40%,#009E73);color:#fff;padding:26px 32px;"
           "border-radius:16px}h1{margin:0;font-size:26px}h2{margin-top:34px;border-bottom:2px solid #0072B2;"
           "padding-bottom:6px;font-size:19px}table.tbl{border-collapse:collapse;font-size:13px;width:100%}"
           ".tbl th{background:#eef4fa;text-align:left}.tbl td,.tbl th{padding:6px 10px;border-bottom:1px solid #dde3ea}")
    stamp = time.strftime("%Y-%m-%d %H:%M")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{css}</style></head><body><header><h1>{html.escape(title)}</h1>"
            f"<div>Battery digital twin · engine {te.ENGINE_VERSION} · generated {stamp}</div></header>"
            + "".join(parts) + "</body></html>")


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
              strategy: str, conformal_cells: int, level: float, params_json: str = "{}",
              train_cells: Optional[Tuple[str, ...]] = None) -> te.MLForecast:
    return te.train_ml_forecast(_ct, cell, n0, model, use_population=use_pop, eol_ah=eol_ah, strategy=strategy,
                                conformal_cells=conformal_cells, band_level=level,
                                model_params=json.loads(params_json),
                                train_cells=list(train_cells) if train_cells is not None else None)


@st.cache_data(show_spinner=False, max_entries=64)
def est_cached(_ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str, model: str, params_json: str,
               features: Tuple[str, ...], split: str, test_frac: float, train_cells: Tuple[str, ...],
               test_cells: Tuple[str, ...], normalise: bool) -> te.EstimationResult:
    return te.train_soh_estimator(_ct, _imp, model, features, json.loads(params_json), split, test_frac,
                                  list(train_cells) or None, list(test_cells) or None, normalise)


@st.cache_data(show_spinner=False, max_entries=8)
def pooled_ea_cached(_ct: pd.DataFrame, key: str, min_ambient: float) -> Dict[str, Any]:
    return te.estimate_pooled_arrhenius(_ct, min_ambient_C=min_ambient)


@st.cache_data(show_spinner=False, max_entries=8)
def fleet_cached(_ct: pd.DataFrame, key: str, eol_ah: float, limits: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    L = te.SafetyLimits(**limits)
    return te.fleet_status(_ct, eol_ah, L), te.fleet_events(_ct, L)


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
                 "Mechanistic PINN: SEI · plating · LAM", "Particle filter", "Semi-empirical law",
                 "14-model ML workbench", "Conformal bands", "Integrated O&M optimisation"))
st.markdown(
    '<div class="bt-hero">'
    '<div class="bt-kicker">Self-updating digital twin · NASA Ames Li-ion ageing data</div>'
    '<div class="bt-title">🔋 Battery Digital Twin &amp; Operando Diagnostics</div>'
    '<div class="bt-sub">NASA Ames 18650 LiCoO₂ / graphite ageing telemetry: incremental capacity analysis, '
    'a 14-model ML workbench, a self-updating ECM twin and a mechanism-resolved physics-informed neural '
    'network (SEI growth, lithium plating, loss of active material, Butler–Volmer kinetics), benchmarked '
    'across batteries and forecast origins with calibrated uncertainty.</div>'
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

_fs_bar, _ = fleet_cached(ct, DATA_KEY, float(eol_ah), asdict(te.SafetyLimits()))
_n_crit = int((_fs_bar["Risk"] == "Critical").sum()) if len(_fs_bar) else 0
_n_warn = int((_fs_bar["Risk"] == "Warning").sum()) if len(_fs_bar) else 0
st.markdown(
    '<div class="bt-statusbar">'
    '<span class="bt-live"><span class="bt-dot"></span>TWIN ONLINE</span>'
    f'<span>ENGINE <b>v{te.ENGINE_VERSION}</b></span>'
    f'<span>SOURCE <b>{"SYNTHETIC DEMO" if st.session_state.get("demo") else "TELEMETRY"}</b></span>'
    f'<span>FLEET <b>{len(meta)}</b> CELLS · <b>{int(ct["n"].count())}</b> CYCLES</span>'
    f'<span class="bt-sev bt-sev-crit">{_n_crit} CRITICAL</span>'
    f'<span class="bt-sev bt-sev-warn">{_n_warn} WARNING</span>'
    f'<span class="bt-clock">{time.strftime("%Y-%m-%d %H:%M")} UTC</span>'
    '</div>', unsafe_allow_html=True)
view = nav(VIEWS, key="view")
st.session_state["_sec_n"] = 0


# =============================================================================
# View 1: data & diagnostics
# =============================================================================
@fragment
def fade_explorer() -> None:
    """Multi-battery fade chart linked to multi-cycle raw telemetry."""
    all_cells = list(meta.index)
    c1, c2, c3, c4 = st.columns([3, 1.3, 1.6, 1.1])
    cells = c1.multiselect("Batteries to compare", all_cells, default=[cell], max_selections=8, key="fade_cells",
                           format_func=lambda c: cell_label(c, meta),
                           help="Up to 8 batteries; each keeps the same colour in every chart.") or [cell]
    yvar = c2.radio("Metric", ["SOH", "Capacity_Ah"], key="fade_metric",
                    format_func=lambda v: "State of health" if v == "SOH" else "Capacity (Ah)")
    x_mode = c3.radio("x-axis", ["n", "Ah", "frac"], key="fade_x",
                      format_func={"n": "Cycle number", "Ah": "Ah throughput", "frac": "Fraction of life"}.get)
    show_cohort = c4.toggle("Show cohort", value=len(cells) == 1, key="fade_cohort")
    knees = {c: knee_cached(ct[ct["Cell_ID"] == c], DATA_KEY, c) for c in cells}
    eol_line = (soh_eol if yvar == "SOH" else eol_ah) if len(cells) == 1 else (None if yvar == "SOH" else eol_ah)
    fig = fig_fade_compare(ct, meta, cells, yvar, eol_line, P, knees, show_cohort, x_mode)
    event = show_selectable(fig, key="fade_multi")
    export_row(fig, "fade", ct[ct["Cell_ID"].isin(cells)][["Cell_ID", "n", "Cycle_Index", "cum_Ah", "SOH",
                                                           "Capacity_Ah", "outlier", "regen"]])
    if len(cells) > 1:
        rows = []
        for c in cells:
            g = ct[(ct["Cell_ID"] == c) & ~ct["outlier"]]
            k = knees[c]
            rows.append({"Battery": cell_label(c, meta), "Cycles": int(g["n"].max()),
                         "Final SOH": float(g["SOH"].tail(3).median()),
                         "Fade per 100 cycles (%)": 100 * (1 - float(g["SOH"].tail(3).median()))
                         / max(float(g["n"].max()), 1) * 100,
                         "Knee at n": k["knee_n"] if k.get("found") else None,
                         "Regeneration events": int(g["regen"].sum())})
        show_table(pd.DataFrame(rows).set_index("Battery").style.format(
            {"Final SOH": "{:.3f}", "Fade per 100 cycles (%)": "{:.2f}", "Knee at n": "{:.0f}"}, na_rep="—"))

    # ---- raw telemetry of one or many discharges ----
    st.markdown("##### Raw telemetry")
    picked_cell, picked_n = cells[0], None
    try:
        pts = event.selection.points if event is not None else []
        if pts and pts[0].get("customdata") is not None:
            picked_n, picked_cell = int(float(pts[0]["customdata"][0])), str(pts[0]["customdata"][1])
    except Exception:
        picked_n = None
    t1, t2, t3 = st.columns([2, 2, 1.4])
    tcell = t1.selectbox("Battery", cells, index=cells.index(picked_cell) if picked_cell in cells else 0,
                         key="trace_cell", format_func=lambda c: cell_label(c, meta))
    mode = t2.selectbox("Cycles to plot", TRACE_MODES, index=0, key="trace_mode")
    x_axis = t3.radio("x-axis", ["time", "capacity"], key="trace_x",
                      format_func={"time": "Time", "capacity": "Discharged Ah"}.get)
    tc = ct[(ct["Cell_ID"] == tcell) & ~ct["outlier"]].sort_values("n")
    ns = tc["n"].astype(int).tolist()
    if not ns:
        st.info("No valid discharges for this battery.")
        return
    if mode == "Single cycle":
        default = picked_n if (picked_n in ns and picked_cell == tcell) else ns[0]
        chosen = [int(st.select_slider("Discharge cycle", options=ns, value=default, key=f"trace_one_{tcell}"))]
        if picked_n is None:
            st.caption("Tip: click any point on the fade chart to open that discharge.")
    elif mode == "Choose cycles":
        step = max(1, len(ns) // 4)
        chosen = st.multiselect("Discharge cycles", ns, default=ns[::step][:5], key=f"trace_many_{tcell}")
    elif mode == "Every k-th cycle":
        k = st.number_input("k", 1, max(1, len(ns)), max(1, len(ns) // 10), key=f"trace_k_{tcell}")
        chosen = ns[:: int(k)]
    elif mode == "Range of cycles":
        a, b = st.select_slider("Range", options=ns, value=(ns[0], ns[min(len(ns) - 1, 20)]), key=f"trace_rng_{tcell}")
        chosen = [n for n in ns if a <= n <= b]
    else:
        chosen = ns
    if not chosen:
        st.info("Select at least one discharge.")
        return
    if len(chosen) > 80:
        st.caption(f"{len(chosen)} discharges: each is downsampled for speed; the colour bar maps colour to cycle.")
    picks = [(int(tc.loc[tc["n"] == n, "Cycle_Index"].iloc[0]), int(n)) for n in chosen]
    prep = prepared_cell(store, DATA_KEY, tcell)
    cis = [c for c, _ in picks]
    show(fig_cycle_traces(prep, picks, P, x_axis, max_pts=200 if len(picks) > 40 else 500), key="trace_multi",
         data=prep[prep["Cycle_Index"].isin(cis)][["Cycle_Index", "Time_s", "Voltage_V", "Current_A", "Temp_C"]])


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
    a = b = st.container()  # stacked full width
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
    a = b = st.container()  # stacked full width
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
    a = b = st.container()  # stacked full width
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
    a = b = st.container()  # stacked full width
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


def risk_banner(row: pd.Series, cell_id: str) -> None:
    col = RISK_COLORS[row["Risk"]]
    st.markdown(f'<div class="bt-alert"><span class="bt-badge" style="background:{col}">{html.escape(row["Risk"].upper())}'
                f'</span><span class="bt-msg"><b>{html.escape(cell_id)}</b>: {html.escape(row["Alerts"])}</span></div>',
                unsafe_allow_html=True)


def view_overview() -> None:
    lim = asdict(te.SafetyLimits())
    fs, ev = fleet_cached(ct, DATA_KEY, float(eol_ah), lim)
    if fs.empty:
        st.info("Not enough valid cycles to build the fleet overview.")
        return
    section("Fleet at a glance")
    k = st.columns(6)
    k[0].metric("Batteries", len(fs))
    k[1].metric("Mean SOH", fmt(100 * fs["SOH"].mean(), ".1f", "%"))
    k[2].metric("Past end of life", int((fs["SOH"] <= fs["SOH_EOL"]).sum()))
    k[3].metric("Critical / warning", f"{int((fs['Risk'] == 'Critical').sum())} / {int((fs['Risk'] == 'Warning').sum())}")
    k[4].metric("Knees detected", int(fs["Knee"].sum()))
    k[5].metric("Fleet cycles logged", f"{int(fs['Cycles'].sum()):,}")

    section(f"Instrument cluster · {cell}")
    if cell in fs.index:
        row = fs.loc[cell]
        risk_banner(row, cell)
        show(fig_gauges(row, P, te.SafetyLimits(**lim)), key="gauges", export=False)
        st.caption("Quick RUL is a robust linear trend of the last 20 cycles, for triage. The Models view gives the "
                   "full probabilistic RUL from the twin, PINN and particle filter.")

    section("Fleet health map")
    show(fig_fleet_map(fs, P), key="fleet_map", data=fs.reset_index())

    section("Risk triage")
    show(fig_risk_matrix(fs, P, cell), key="risk_matrix", data=fs.reset_index())
    tbl = fs.reset_index()[["Cell_ID", "Risk", "SOH", "Health margin", "Quick RUL", "Fade per 100 cycles (%)",
                            "R growth (%)", "Peak T (°C)", "Cycles", "Ambient_C", "I_dis_A", "Alerts"]].copy()
    tbl["Risk"] = tbl["Risk"].map(lambda r: f"{RISK_ICON[r]} {r}")
    tbl["Quick RUL"] = tbl["Quick RUL"].replace(np.inf, np.nan)
    try:
        st.dataframe(tbl, hide_index=True, use_container_width=True, height=min(640, 38 + 35 * len(tbl)),
                     column_config={
                         "Cell_ID": st.column_config.TextColumn("Battery", width="small"),
                         "SOH": st.column_config.ProgressColumn("SOH", min_value=0.0, max_value=1.05, format="%.3f"),
                         "Health margin": st.column_config.ProgressColumn("Health margin", min_value=0.0, max_value=1.0,
                                                                          format="%.2f"),
                         "Quick RUL": st.column_config.NumberColumn("Quick RUL", format="%.0f cyc"),
                         "Fade per 100 cycles (%)": st.column_config.NumberColumn("Fade /100 cyc", format="%.2f %%"),
                         "R growth (%)": st.column_config.NumberColumn("R growth", format="%+.0f %%"),
                         "Peak T (°C)": st.column_config.NumberColumn("Peak T", format="%.1f °C"),
                         "Ambient_C": st.column_config.NumberColumn("Ambient", format="%.0f °C"),
                         "I_dis_A": st.column_config.NumberColumn("Load", format="%.1f A"),
                         "Alerts": st.column_config.TextColumn("Alerts", width="large")})
    except Exception:
        show_table(tbl)

    section("Event log")
    sev = st.multiselect("Severity", ["Critical", "Warning", "Watch", "Info"], default=["Critical", "Warning", "Watch"],
                         key="ev_sev")
    only = st.toggle("Only the selected battery", value=False, key="ev_only")
    evf = ev[ev["Severity"].isin(sev)]
    if only:
        evf = evf[evf["Cell_ID"] == cell]
    evf = evf.assign(Severity=evf["Severity"].map(lambda s_: {"Critical": "🔴", "Warning": "🟠", "Watch": "🟡",
                                                              "Info": "🔵"}[s_] + " " + s_))
    try:
        st.dataframe(evf, hide_index=True, use_container_width=True, height=min(420, 38 + 35 * max(len(evf), 1)),
                     column_config={"Cell_ID": st.column_config.TextColumn("Battery"),
                                    "n": st.column_config.NumberColumn("Cycle", format="%d")})
    except Exception:
        show_table(evf)

    section("Reports")
    st.markdown("One-click, self-contained HTML report (interactive charts, opens in any browser; print to PDF "
                "from the browser for a static copy).")
    if st.button(f"Build report for {cell}", key="rep_go", icon=":material/description:", type="primary"):
        with st.spinner("Assembling report…"):
            knees = {cell: knee_cached(ct_cell, DATA_KEY, cell)}
            secs: List[Tuple[str, Any]] = [
                ("Summary", f"{cell_label(cell, meta)} · SOH {100 * fs.loc[cell, 'SOH']:.1f}% · risk "
                            f"{fs.loc[cell, 'Risk']} · alerts: {fs.loc[cell, 'Alerts']}" if cell in fs.index else cell),
                ("Instrument cluster", fig_gauges(fs.loc[cell], P, te.SafetyLimits(**lim))) if cell in fs.index
                else ("Instrument cluster", "n/a"),
                ("Capacity fade", fig_fade_compare(ct, meta, [cell], "SOH", soh_eol, P, knees, True, "n")),
                ("Fleet risk matrix", fig_risk_matrix(fs, P, cell)),
                ("Fleet status", fs.drop(columns=["risk_level"])),
                ("Event log (this battery)", ev[ev["Cell_ID"] == cell]),
            ]
            for key_, title in (("ml", "ML forecasts"), ("cmp", "Physics-informed comparison")):
                saved = st.session_state.get(key_)
                if saved and key_ == "ml" and saved.get("res") and saved["cfg"]["cell"] == cell:
                    secs.append((title, fig_ml(saved["res"], ct_cell, saved["cfg"]["n0"], soh_eol, P)))
                if saved and key_ == "cmp" and saved.get("res") is not None:
                    try:
                        secs.append((title, fig_compare(saved["res"], P)))
                        secs.append(("Forecast metrics", te.metrics_table(saved["res"].metrics)))
                    except Exception:
                        pass
            st.session_state["report"] = (cell, build_report_html(f"Battery health report · {cell}", secs))
    rep = st.session_state.get("report")
    if rep and rep[0] == cell:
        download("Download report (HTML)", rep[1], f"battery_report_{cell}.html", "text/html", key="rep_dl")


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

    section("Capacity fade: compare batteries")
    fade_explorer()

    section("Cohort by ambient temperature")
    norm_x = st.toggle("Normalise x to fraction of recorded life", value=False, key="grid_norm",
                       help="Aligns cells with very different cycle counts.")
    show(fig_cohort_grid(ct, meta, cell, P, norm_x), key="cohort_grid",
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


def hyperparam_editor(models: Sequence[str], prefix: str) -> Dict[str, Dict[str, Any]]:
    """One expander per selected model, widgets generated from te.MODEL_SPECS."""
    out: Dict[str, Dict[str, Any]] = {}
    for name in models:
        spec = te.MODEL_SPECS[name]
        with st.expander(f"{name} · {spec.family}", expanded=False, icon=":material/tune:"):
            st.caption(spec.blurb)
            cols = st.columns(min(4, max(len(spec.params), 1)))
            vals: Dict[str, Any] = {}
            for i, h in enumerate(spec.params):
                k = f"{prefix}_{name}_{h.key}"
                c = cols[i % len(cols)]
                if h.kind == "int":
                    vals[h.key] = int(c.number_input(h.label, int(h.low), int(h.high), int(h.default), key=k, help=h.help or None))
                elif h.kind == "float":
                    vals[h.key] = float(c.slider(h.label, float(h.low), float(h.high), float(h.default), key=k))
                elif h.kind == "log":
                    grid = sorted(set([float(f"{v:.3g}") for v in np.geomspace(h.low, h.high, 25)] + [float(h.default)]))
                    vals[h.key] = float(c.select_slider(h.label, grid, value=float(h.default), key=k,
                                                        format_func=lambda v: f"{v:.3g}"))
                elif h.kind == "choice":
                    opts = list(h.options)
                    vals[h.key] = c.selectbox(h.label, opts, index=opts.index(h.default), key=k)
                else:
                    vals[h.key] = c.text_input(h.label, str(h.default), key=k, help=h.help or None)
            try:
                out[name] = te.validate_params(name, vals)
            except ValueError as exc:
                st.error(str(exc))
                out[name] = te.default_params(name)
    return out


R2_NOTE = ("**Reading the scores.** *Accuracy* = 100 × (1 − mean absolute percentage error). *Fade skill* compares "
           "the forecast with the naive assumption that SOH stops falling at n₀: 1 is perfect, 0 is no better than "
           "doing nothing, negative is worse. *R²* compares with the mean of the held-out SOH; over a short forecast "
           "window SOH hardly varies, so a small bias already makes R² negative. It is shown honestly rather than "
           "clipped, but fade skill and accuracy are the easier scores to read. In the estimation task the test set "
           "spans the whole ageing range and R² behaves as usual.")


def ml_section() -> None:
    section("Machine-learning workbench", icon=":material/model_training:")
    st.markdown("Two tasks, 14 models, every hyperparameter adjustable. **Forecast**: predict future SOH from the past "
                "(prognosis). **Estimate**: infer the present SOH from operando indicators measured on the same cycle "
                "(diagnosis, no capacity test needed).")
    models = st.multiselect("Models", list(te.ML_MODELS), default=["Random Forest", "Extra Trees", "Gradient Boosting",
                                                                    "Hist. Gradient Boosting", "Gaussian Process",
                                                                    "Ridge"],
                            key="ml_models", help="Select any number; the leaderboard ranks them.")
    params = hyperparam_editor(models, "hp")
    t_fc, t_est = st.tabs([":material/trending_down: Forecast future SOH", ":material/biotech: Estimate SOH from indicators"])
    with t_fc:
        ml_forecast_tab(models, params)
    with t_est:
        ml_estimation_tab(models, params)
    with st.expander("How to read R², accuracy and fade skill", icon=":material/help:"):
        st.markdown(R2_NOTE)


def ml_forecast_tab(models: Sequence[str], params: Dict[str, Dict[str, Any]]) -> None:
    all_cells = list(meta.index)
    c1, c2 = st.columns([2, 2])
    source = c1.radio("Training data", ["own", "cohort", "cross"], key="ml_source",
                      format_func={"own": "This battery only (its early life)",
                                   "cohort": "This battery's early life + all other batteries",
                                   "cross": "Chosen batteries → predict this battery"}.get)
    frac = c2.slider("Train fraction of this battery's life (rest = test)", 0.05, 0.9, 0.4, 0.05, key="ml_frac",
                     help="Cycles up to this fraction are seen in training; all later cycles are the test set.")
    n0_ml = int(max(5, round(frac * meta.loc[cell, "cycles"])))
    n_test = int((ct_cell[~ct_cell["outlier"]]["n"] > n0_ml).sum())
    c2.caption(f"Train: cycles 1–{n0_ml} · Test: {n_test} later cycles")
    train_cells: Optional[Tuple[str, ...]] = None
    if source == "cross":
        others = [c for c in all_cells if c != cell]
        near = te.calibration_partners(meta, cell, 3) if hasattr(te, "calibration_partners") else others[:3]
        train_cells = tuple(st.multiselect("Train on these batteries", others, default=[c for c in near if c in others],
                                           key="ml_train_cells", format_func=lambda c: cell_label(c, meta)))
    d1, d2, d3 = st.columns(3)
    strategy = d1.selectbox("Strategy", list(te.ML_STRATEGIES), key="ml_strategy",
                            format_func=lambda s: "Fade-rate model (increment)" if s == "increment"
                            else "Direct SOH(n) regression (legacy)",
                            help="The increment strategy learns dSOH/dn as a function of SOH, so tree models "
                                 "keep extrapolating beyond the longest training life.")
    conf = d2.slider("Conformal calibration cells (0 = no band)", 0, 8, 3, key="ml_conf")
    level = d3.select_slider("Band level", [0.8, 0.9, 0.95], value=0.9, key="ml_level")
    use_pop = source == "cohort"
    ml_cfg = dict(cell=cell, n0=n0_ml, models=tuple(models), source=source, train_cells=train_cells,
                  eol_ah=float(eol_ah), strategy=strategy, conformal_cells=conf, band_level=level,
                  params={m: params.get(m) for m in models})
    ready = bool(models) and (source != "cross" or bool(train_cells))
    if st.button("Train and forecast", type="primary", key="ml_go", disabled=not ready, icon=":material/play_arrow:"):
        prog = st.progress(0.0)
        out: List[te.MLForecast] = []
        for i, mname in enumerate(models):
            prog.progress(i / max(len(models), 1), text=f"Training {mname}…")
            try:
                out.append(ml_cached(ct, DATA_KEY, cell, n0_ml, mname, use_pop, float(eol_ah), strategy, conf, level,
                                     json.dumps(params.get(mname, {}), sort_keys=True), train_cells))
            except Exception as exc:
                st.warning(f"{mname}: {exc}")
        prog.empty()
        st.session_state["ml"] = {"cfg": ml_cfg, "res": out}
    saved = st.session_state.get("ml")
    if not (saved and saved["res"]):
        return
    if saved["cfg"]["cell"] != cell:
        st.info("The stored forecasts belong to another battery. Train again to update them.")
        return
    if saved["cfg"] != ml_cfg:
        st.warning("Settings changed since the last run; the results below use the previous settings.")
    res = saved["res"]
    n0s = saved["cfg"]["n0"]
    data = pd.concat([pd.DataFrame({"model": r.model, "n": r.n_grid, "soh": r.soh_pred,
                                    "lo": r.soh_lo if r.soh_lo is not None else np.nan,
                                    "hi": r.soh_hi if r.soh_hi is not None else np.nan}) for r in res])
    show(fig_ml(res, ct_cell, n0s, soh_eol, P), key="ml_fig", data=data)
    tbl = pd.DataFrame([{"Model": r.model, "Accuracy (%)": r.metrics.accuracy, "Fade skill": r.metrics.fade_skill,
                         "R²": r.metrics.r2, "RMSE": r.metrics.rmse, "MAE": r.metrics.mae,
                         "Coverage": r.metrics.coverage, "RUL true": r.metrics.rul_true, "RUL pred": r.metrics.rul_pred,
                         "RUL error": r.metrics.rul_error, "Fit time (s)": r.fit_seconds} for r in res]).set_index("Model")
    best = tbl["RMSE"].idxmin()
    k = st.columns(4)
    k[0].metric("Best model", best)
    k[1].metric("Accuracy", fmt(tbl.loc[best, "Accuracy (%)"], ".2f", "%"))
    k[2].metric("Fade skill", fmt(tbl.loc[best, "Fade skill"], ".3f"))
    k[3].metric("RMSE", fmt(tbl.loc[best, "RMSE"], ".4f"))
    show(fig_leaderboard(tbl, "Fade skill", P, "Leaderboard: fade skill on the held-out cycles (higher is better)"),
         key="ml_board", data=tbl.reset_index())
    show_table(tbl.style.format(
        {"Accuracy (%)": "{:.2f}", "Fade skill": "{:.3f}", "R²": "{:.3f}", "RMSE": "{:.4f}", "MAE": "{:.4f}",
         "Coverage": "{:.2f}", "RUL true": "{:.0f}", "RUL pred": "{:.0f}", "RUL error": "{:+.0f}",
         "Fit time (s)": "{:.2f}"}, na_rep="—")
        .highlight_min(subset=["RMSE"], props="background-color: rgba(0,158,115,0.25); font-weight: 700;"))
    good = ct_cell[(~ct_cell["outlier"]) & (ct_cell["n"] > n0s)]
    par = pd.concat([pd.DataFrame({"model": r.model, "SOH": good["SOH"].to_numpy(),
                                   "SOH_pred": np.interp(good["n"], r.n_grid, r.soh_pred)}) for r in res])
    if len(par):
        show(fig_parity(par, "model", P, "Parity on the test cycles: forecast vs measured SOH"), key="ml_parity",
             data=par)
    cal = res[0].calibration_cells
    if cal:
        st.caption(f"Conformal calibration cells: {', '.join(cal)}. Coverage below the nominal level signals that "
                   "this battery ages differently from its calibration partners.")
    manifest_button(ml_cfg, "ml", DATA_KEY)


def ml_estimation_tab(models: Sequence[str], params: Dict[str, Dict[str, Any]]) -> None:
    all_cells = list(meta.index)
    avail = [k for k in te.HI_CATALOG if k not in te.CAPACITY_LEAKS and k in ct.columns or k in ("Re_ohm", "Rct_ohm")]
    avail += [k for k in ("T_mean_C", "I_dis_A", "n") if k in ct.columns]
    labels = {**{k: te.HI_CATALOG[k].label for k in te.HI_CATALOG}, "T_mean_C": "Mean cell temperature",
              "I_dis_A": "Discharge current", "n": "Cycle number"}
    c1, c2 = st.columns([3, 1])
    feats = c1.multiselect("Input indicators (capacity-derived ones are excluded: they would leak SOH)", avail,
                           default=[k for k in te.DEFAULT_EST_FEATURES if k in avail], key="est_feats",
                           format_func=lambda k: labels.get(k, k))
    normalise = c2.toggle("Normalise to beginning of life", value=True, key="est_norm",
                          help="Divides each indicator by its early-life value, so models transfer between batteries.")
    d1, d2 = st.columns(2)
    split = d1.radio("Train / test split", list(te.ESTIMATION_SPLITS), key="est_split",
                     format_func={"random": "Random cycles (interpolation, optimistic)",
                                  "chronological": "Late life of every battery (extrapolation in time)",
                                  "by_cell": "Train on some batteries, test on others (transfer)"}.get)
    test_frac = d2.slider("Test fraction", 0.1, 0.8, 0.3, 0.05, key="est_frac", disabled=split == "by_cell")
    tr_cells: Tuple[str, ...] = ()
    te_cells: Tuple[str, ...] = ()
    if split == "by_cell":
        e1, e2 = st.columns(2)
        te_cells = tuple(e2.multiselect("Test batteries", all_cells, default=[cell], key="est_test_cells",
                                        format_func=lambda c: cell_label(c, meta)))
        rest = [c for c in all_cells if c not in te_cells]
        tr_cells = tuple(e1.multiselect("Train batteries", rest, default=rest, key="est_train_cells",
                                        format_func=lambda c: cell_label(c, meta)))
    cfg = dict(models=tuple(models), feats=tuple(feats), split=split, test_frac=test_frac, train=tr_cells,
               test=te_cells, normalise=normalise, params={m: params.get(m) for m in models})
    ready = bool(models) and bool(feats) and (split != "by_cell" or (tr_cells and te_cells))
    if st.button("Train estimators", type="primary", key="est_go", disabled=not ready, icon=":material/play_arrow:"):
        prog = st.progress(0.0)
        out: Dict[str, te.EstimationResult] = {}
        for i, mname in enumerate(models):
            prog.progress(i / max(len(models), 1), text=f"Training {mname}…")
            try:
                out[mname] = est_cached(ct, imp, DATA_KEY, mname, json.dumps(params.get(mname, {}), sort_keys=True),
                                        tuple(feats), split, float(test_frac), tr_cells, te_cells, normalise)
            except Exception as exc:
                st.warning(f"{mname}: {exc}")
        prog.empty()
        st.session_state["est"] = {"cfg": cfg, "res": out}
    saved = st.session_state.get("est")
    if not (saved and saved["res"]):
        return
    if saved["cfg"] != cfg:
        st.warning("Settings changed since the last run; the results below use the previous settings.")
    res: Dict[str, te.EstimationResult] = saved["res"]
    tbl = pd.DataFrame([{"Model": m, "Test R²": r.metrics.loc["test", "R²"], "Test RMSE": r.metrics.loc["test", "RMSE"],
                         "Test accuracy (%)": r.metrics.loc["test", "Accuracy (%)"],
                         "Train R²": r.metrics.loc["train", "R²"], "Train RMSE": r.metrics.loc["train", "RMSE"],
                         "Overfit gap (RMSE)": r.metrics.loc["test", "RMSE"] - r.metrics.loc["train", "RMSE"],
                         "Fit time (s)": r.fit_seconds} for m, r in res.items()]).set_index("Model")
    best = tbl["Test RMSE"].idxmin()
    k = st.columns(4)
    k[0].metric("Best model", best)
    k[1].metric("Test R²", fmt(tbl.loc[best, "Test R²"], ".3f"))
    k[2].metric("Test accuracy", fmt(tbl.loc[best, "Test accuracy (%)"], ".2f", "%"))
    k[3].metric("Test RMSE", fmt(tbl.loc[best, "Test RMSE"], ".4f"))
    show(fig_leaderboard(tbl, "Test R²", P, "Leaderboard: R² on the test set (higher is better)"), key="est_board",
         data=tbl.reset_index())
    show_table(tbl.style.format({"Test R²": "{:.3f}", "Test RMSE": "{:.4f}", "Test accuracy (%)": "{:.2f}",
                                 "Train R²": "{:.3f}", "Train RMSE": "{:.4f}", "Overfit gap (RMSE)": "{:+.4f}",
                                 "Fit time (s)": "{:.2f}"}, na_rep="—")
               .highlight_min(subset=["Test RMSE"], props="background-color: rgba(0,158,115,0.25); font-weight: 700;"))
    pick = st.selectbox("Inspect model", list(res), index=list(res).index(best), key="est_pick")
    r = res[pick]
    test = r.predictions[r.predictions["set"] == "test"]
    show(fig_parity(test.assign(model=pick), "Cell_ID", P, f"{pick}: estimated vs measured SOH on the test set",
                    all_cells), key="est_parity", data=test)
    show(fig_est_traj(r.predictions[r.predictions["Cell_ID"].isin(test["Cell_ID"].unique())], P, pick, all_cells),
         key="est_traj", data=r.predictions)
    show(fig_importance(r.importance, P, pick), key="est_imp", data=r.importance)
    st.caption(f"Train batteries: {', '.join(r.train_cells)} · Test batteries: {', '.join(r.test_cells)}. "
               "A large overfit gap (test RMSE ≫ train RMSE) means the model memorises; raise regularisation or "
               "reduce depth in its hyperparameter panel.")
    manifest_button(cfg, "estimation", DATA_KEY)


def pinn_equations_panel() -> None:
    with st.expander("Governing equations of the mechanism-resolved PINN", icon=":material/functions:"):
        st.markdown("The network $\\mathcal{N}(n)$ outputs three latent capacity losses (fractions of the "
                    "beginning-of-life capacity) and two resistances, each with its initial condition built in. "
                    "Collocation points span 1.5× the horizon, so the forecast obeys the kinetics beyond the data. "
                    "$A(T;E) = \\exp\\left[\\tfrac{E}{R}\\left(\\tfrac{1}{T_\\mathrm{ref}} - \\tfrac{1}{T}\\right)\\right]$, "
                    "$A_n = 2\\,C_0\\,\\mathrm{SOH}$ is the charge throughput of cycle $n$.")
        st.markdown("**1 · Capacity balance** (loss of lithium inventory = SEI + plating; loss of active material)")
        st.latex(r"\mathrm{SOH}(n) = 1 - Q_\mathrm{SEI}(n) - Q_\mathrm{pl}(n) - Q_\mathrm{LAM}(n)")
        st.markdown("**2 · SEI growth**: solvent reduction at the graphite surface, reaction-limited while the film "
                    "is thin and diffusion-limited as it thickens (Ploehn 2004; Pinson & Bazant 2013)")
        st.latex(r"\frac{dQ_\mathrm{SEI}}{dn} = k_\mathrm{SEI}\,A(T;E_\mathrm{SEI})\,\frac{A_n}{C_0}\,"
                 r"\frac{1}{1 + Q_\mathrm{SEI}/\delta}\;\;\Rightarrow\;\; Q_\mathrm{SEI} \propto n\ (\text{thin}),"
                 r"\quad \propto \sqrt{n}\ (\text{thick})")
        st.markdown("**3 · Lithium plating**: favoured by cold and by charge current, and triggered at any "
                    "temperature once LAM clogs the pores (Waldmann 2014; Yang et al. 2017), which produces the knee")
        st.latex(r"\frac{dQ_\mathrm{pl}}{dn} = k_\mathrm{pl}\,e^{\frac{E_\mathrm{pl}}{R}\left(\frac1T - "
                 r"\frac1{T_\mathrm{ref}}\right)}\,\frac{I_\mathrm{ch}}{C_0}\left[g_\mathrm{cold}(T) + "
                 r"\kappa\,\frac{Q_\mathrm{LAM}}{0.05}\right],\qquad g_\mathrm{cold} = "
                 r"\frac{1}{1 + e^{(T - T_\mathrm{onset})/3\,\mathrm{K}}}")
        st.markdown("**4 · Loss of active material**: particle cracking and dissolution, driven by C-rate and "
                    "self-accelerating as the remaining material carries more current (Laresgoiti 2015)")
        st.latex(r"\frac{dQ_\mathrm{LAM}}{dn} = k_\mathrm{LAM}\,A(T;E_\mathrm{LAM})\left(\frac{I}{C_0}\right)^{\beta}"
                 r"\frac{A_n}{C_0}\left(1 + \frac{Q_\mathrm{LAM}}{\varepsilon}\right)")
        st.markdown("**5 · Resistance from the mechanisms**: the SEI film adds ohmic resistance, while active-area "
                    "loss and surface films raise the charge-transfer resistance")
        st.latex(r"\frac{dR_\mathrm{int}}{dn} = R_\mathrm{int,0}\,\rho_\mathrm{SEI}\,\frac{1}{0.1}\frac{dQ_\mathrm{SEI}}{dn},"
                 r"\qquad \frac{dR_\mathrm{ct}}{dn} = R_\mathrm{ct,0}\,\frac{1}{0.1}\left(\gamma_\mathrm{LAM}"
                 r"\frac{dQ_\mathrm{LAM}}{dn} + \gamma_\mathrm{SEI}\frac{dQ_\mathrm{SEI}}{dn}\right)")
        st.markdown("**6 · Electrochemistry**: Butler–Volmer charge transfer (α = 0.5) with Arrhenius kinetics, "
                    "constraining the load-step voltage drop and the mean discharge voltage")
        st.latex(r"\eta_\mathrm{ct} = \frac{2RT}{F}\,\sinh^{-1}\!\left(\frac{I\,F\,R_\mathrm{ct}(T)}{2RT}\right),\qquad "
                 r"R_\mathrm{ct}(T) = R_\mathrm{ct}\,e^{\frac{E_\mathrm{ct}}{R}\left(\frac1T - \frac1{T_\mathrm{ref}}\right)},"
                 r"\qquad i_0 = \frac{RT}{F\,R_\mathrm{ct}}")
        st.latex(r"\Delta V_\mathrm{step} = I\,(R_\mathrm{int} + R_x) + \eta_\mathrm{ct},\qquad "
                 r"\bar V_\mathrm{dis} = \bar U - I\,(R_\mathrm{int} + R_x) - \eta_\mathrm{ct}")
        st.markdown("**7 · Loss**")
        st.latex(r"\mathcal{L} = \lambda_\mathrm{d}\,\mathcal{L}_\mathrm{SOH} + \lambda_\mathrm{p}\sum_{m}\|r_m\|^2 "
                 r"+ \lambda_\mathrm{BV}\,\mathcal{L}_{\Delta V} + \lambda_V\,\mathcal{L}_{\bar V} + "
                 r"\lambda_\mathrm{EIS}\,\mathcal{L}_\mathrm{EIS} + \lambda_\pi \sum_m \left(\frac{\ln k_m - "
                 r"\ln k_m^0}{2}\right)^2")
        st.caption("Weak log-normal priors on the rate constants keep the mechanism split identifiable on one "
                   "cell; with several ensemble members the identifiability table shows which parameters the data "
                   "actually pin down. From capacity alone, SEI and LAM are only partly separable. EIS and "
                   "voltage terms help, and plating is only resolved on cold or knee-bearing cells.")


def fig_mechanisms(mech: pd.DataFrame, n0: int, ct_cell: pd.DataFrame, P: Palette) -> go.Figure:
    fig = go.Figure()
    names = {"Q_SEI": "SEI growth (LLI)", "Q_plating": "Lithium plating (LLI)", "Q_LAM": "Loss of active material"}
    cols = {"Q_SEI": P.mode_colors[0], "Q_plating": P.mode_colors[3], "Q_LAM": P.mode_colors[1]}
    for k in ("Q_SEI", "Q_plating", "Q_LAM"):
        if k in mech and mech[k].abs().max() > 0:
            fig.add_trace(go.Scatter(x=mech["n"], y=100 * mech[k], mode="lines", stackgroup="loss", name=names[k],
                                     line=dict(color=cols[k], width=1.5), fillcolor=rgba(cols[k], 0.55),
                                     hovertemplate="%{y:.2f}%<extra>" + names[k] + "</extra>"))
    good = ct_cell[~ct_cell["outlier"]]
    fig.add_trace(go.Scatter(x=good["n"], y=100 * (1 - good["SOH"]), mode="markers", name="Measured capacity loss",
                             marker=dict(color=P.measured, size=5, opacity=0.8),
                             hovertemplate="%{y:.2f}%<extra>measured</extra>"))
    fig.add_vline(x=n0, line_dash="dot", line_color=P.muted, annotation_text="n₀",
                  annotation_font=dict(color=P.muted))
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="Capacity loss (% of C₀)")
    return style_fig(fig, P, 500, "PINN mechanism decomposition: which process consumes the capacity?")


def physics_section() -> None:
    section("Physics-informed benchmark: ML, ECM twin, PINN, semi-empirical and particle filter")
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
            physics = st.radio("Physics", list(te.PINN_PHYSICS), index=1, key="pinn_physics", horizontal=True,
                               format_func={"lumped": "Lumped fade law", "mechanistic": "Mechanism-resolved"}.get,
                               help="Mechanism-resolved: separate SEI, lithium-plating and LAM kinetics with "
                                    "resistance coupling and an electrochemical voltage equation.")
            mech_on = physics == "mechanistic"
            m1, m2, m3 = st.columns(3)
            use_sei = m1.toggle("SEI", value=True, disabled=not mech_on, key="pinn_sei")
            use_pl = m2.toggle("Plating", value=True, disabled=not mech_on, key="pinn_pl")
            use_lam = m3.toggle("LAM", value=True, disabled=not mech_on, key="pinn_lam")
            lam_volt = st.select_slider("λ voltage (mean discharge voltage)", [0.0, 0.1, 0.3, 1.0, 3.0], value=0.3,
                                        disabled=not mech_on)
            t_on = st.slider("Cold-plating onset (°C)", 0.0, 25.0, 10.0, 1.0, disabled=not mech_on)
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
        if mech_on and not (use_sei or use_pl or use_lam):
            st.warning("Enable at least one degradation mechanism; using SEI.")
            use_sei = True
        pinn_cfg = te.PINNConfig(epochs=int(epochs), lambda_phys=float(lam_phys), lambda_bv=float(lam_bv),
                                 lambda_eis=float(lam_eis), ea_mode=ea_mode, ea_fixed_J_mol=float(ea_fixed) * 1e3,
                                 physics=physics, use_sei=use_sei, use_plating=use_pl, use_lam=use_lam,
                                 lambda_volt=float(lam_volt), T_plating_onset_C=float(t_on))
    pinn_equations_panel()

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

    a = b = st.container()  # stacked full width
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

    if res.pinn is not None and getattr(res.pinn, "physics_kind", "lumped") == "mechanistic":
        section("Degradation mechanisms identified by the PINN")
        ph = res.pinn.physics
        if res.pinn.mechanisms is not None:
            show(fig_mechanisms(res.pinn.mechanisms, res.n0, ct_cell, P), key="pinn_mech",
                 data=res.pinn.mechanisms)
        shares = {k: ph.get(f"share_{k}_at_n0") for k in ("SEI", "plating", "LAM")}
        dom = max((k for k in shares if shares[k] is not None), key=lambda k: shares[k], default=None)
        kin = []
        if "k_SEI_per_cycle" in ph:
            kin.append(f"SEI: k = {ph['k_SEI_per_cycle']:.2e} per cycle, δ = {ph['delta_SEI']:.3f} "
                       f"({'diffusion-limited, √n' if ph['delta_SEI'] < 0.02 else 'reaction-limited, near-linear'}); "
                       f"{100 * ph['share_SEI_at_n0']:.0f}% of the loss at n₀")
        if "k_plating_per_cycle" in ph:
            kin.append(f"Plating: k = {ph['k_plating_per_cycle']:.2e}, LAM coupling κ = {ph['kappa_LAM_to_plating']:.2f}; "
                       f"{100 * ph['share_plating_at_n0']:.0f}% of the loss")
        if "k_LAM_per_cycle" in ph:
            kin.append(f"LAM: k = {ph['k_LAM_per_cycle']:.2e}, acceleration ε = {ph['eps_LAM_acceleration']:.3f}; "
                       f"{100 * ph['share_LAM_at_n0']:.0f}% of the loss")
        card(f"Degradation kinetics (dominant at n₀: {dom or '—'})", kin)
        card("Electrochemistry (Butler–Volmer, Arrhenius kinetics)", [
            f"Exchange current i₀: {ph['i0_start_A']:.3f} A at start, {ph['i0_at_n0_A']:.3f} A at n₀; "
            f"η_ct at 2 A = {ph['eta_ct_2A_n0_mV']:.0f} mV at n₀",
            f"Mean open-circuit voltage Ū = {ph['U_bar_V']:.3f} V; fast polarisation R_x = {ph['R_x_mOhm']:.1f} mΩ",
            f"SEI film resistance coupling ρ = {ph['rho_SEI']:.2f}"
            + (f"; active-area loss coupling γ_LAM = {ph['gamma_ct_LAM']:.2f}" if "gamma_ct_LAM" in ph else ""),
            f"Training time {res.pinn.train_seconds:.1f} s ({res.pinn.n_members} member(s))"])
        if res.pinn.physics_table is not None:
            st.markdown("**Identifiability across ensemble members** (coefficient of variation above 0.25 = not "
                        "constrained by this cell's data)")
            show_table(res.pinn.physics_table.style.format({"mean": "{:.4g}", "std": "{:.3g}", "cv": "{:.2f}"},
                                                           na_rep="—"))
    elif res.pinn is not None:
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
    a = b = st.container()  # stacked full width
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
    scenario_section()


def scenario_section() -> None:
    section("What-if scenario planner")
    sf = stress_factors_cached(ct, DATA_KEY)
    if not sf.get("available"):
        st.info("The planner needs the cohort stress-factor regression (at least four cells with measurable fade).")
        return
    st.markdown("Edit the table to compare operating scenarios. Projections use the stress-factor law fitted on this "
                f"cohort ({sf['n_cells']} cells, R² = {sf['r2']:.2f}; factors: "
                f"{', '.join(c for c in sf['columns'] if c != 'intercept')}), with a bootstrap band. Factors that do "
                "not vary in the data are held at their reference value.")
    default = pd.DataFrame([{"Scenario": "Reference (24 °C, 2 A)", "Ambient (°C)": 24.0, "Discharge current (A)": 2.0,
                             "Cut-off (V)": 2.7},
                            {"Scenario": "Hot site (43 °C)", "Ambient (°C)": 43.0, "Discharge current (A)": 2.0,
                             "Cut-off (V)": 2.7},
                            {"Scenario": "Hot + high load", "Ambient (°C)": 43.0, "Discharge current (A)": 4.0,
                             "Cut-off (V)": 2.7},
                            {"Scenario": "Cold site (4 °C)", "Ambient (°C)": 4.0, "Discharge current (A)": 2.0,
                             "Cut-off (V)": 2.7}])
    try:
        ed = st.data_editor(default, num_rows="dynamic", hide_index=True, use_container_width=True, key="scen_tbl",
                            column_config={"Ambient (°C)": st.column_config.NumberColumn(min_value=-20.0, max_value=60.0),
                                           "Discharge current (A)": st.column_config.NumberColumn(min_value=0.1,
                                                                                                  max_value=10.0),
                                           "Cut-off (V)": st.column_config.NumberColumn(min_value=2.0, max_value=3.2)})
    except Exception:
        ed = default
    if not isinstance(ed, pd.DataFrame) or ed.empty:
        ed = default
    horizon = st.slider("Projection horizon (cycles)", 100, 1500, 500, 50, key="scen_h")
    scen = [{"name": str(r["Scenario"]), "T_C": float(r["Ambient (°C)"]), "I_A": float(r["Discharge current (A)"]),
             "V_cut": float(r["Cut-off (V)"])} for _, r in ed.dropna().iterrows()][:8]
    try:
        sp = te.scenario_projection(sf, scen, horizon, c_bol=float(meta["C_bol_Ah"].median()),
                                    soh_eol=float(soh_eol))
    except Exception as exc:
        report_error("Scenario projection failed", exc, debug)
        return
    show(fig_scenarios(sp, soh_eol, P), key="scen_fig", data=sp)
    life = sp.groupby("scenario", sort=False).agg(life=("life_to_EOL", "first"), rate=("rate_per_Ah", "first"))
    ref = life["life"].iloc[0]
    k = st.columns(min(len(life), 4))
    for i, (name, r) in enumerate(life.head(4).iterrows()):
        delta = (f"{100 * (r['life'] - ref) / ref:+.0f}% vs first" if (ref and r["life"] and i) else None)
        k[i].metric(name, f"{int(r['life'])} cycles" if r["life"] else f"> {horizon} cycles", delta=delta)


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
_VIEW_FN: Dict[str, Callable[[], None]] = {VIEWS[0]: view_overview, VIEWS[1]: view_data, VIEWS[2]: view_models,
                                           VIEWS[3]: view_ops}
try:
    _VIEW_FN.get(view, view_data)()
except Exception as exc:
    report_error(f"{view} failed", exc, debug)
