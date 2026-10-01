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

# ---- engine / app version handshake -------------------------------------------------------------
# Streamlit can keep an old copy of twin_engine in memory after a redeploy (it reruns app.py but does
# not always re-import changed modules), and app.py and twin_engine.py must come from the same release.
REQUIRED_ENGINE = "5.2"
if not str(getattr(te, "ENGINE_VERSION", "0")).startswith(REQUIRED_ENGINE):
    import importlib
    te = importlib.reload(te)

st.set_page_config(
    page_title="Battery Digital Twin · Operando Diagnostics",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not str(getattr(te, "ENGINE_VERSION", "0")).startswith(REQUIRED_ENGINE):
    st.error(f"**Version mismatch.** This app.py needs twin_engine.py version {REQUIRED_ENGINE}.x, but the file "
             f"deployed next to it is version {getattr(te, 'ENGINE_VERSION', 'unknown')}. Upload **both** files from "
             "the same release to the repository (same folder), then reboot the app (Manage app → ⋮ → Reboot).")
    st.stop()

# =============================================================================
# Constants & version gates
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
RESULTS_DIR = APP_DIR / "results"
MASTER_NAME, IMP_NAME = "battery_master_data.parquet", "impedance_ground_truth.parquet"
POLICIES = ["Twin-Aware", "Fixed 1 A", "Fixed 2 A", "Fixed 4 A"]
VIEWS = [":material/monitoring: Diagnostics", ":material/play_circle: Live twin",
         ":material/psychology: Models & forecasting", ":material/tune: Operations & control",
         ":material/fact_check: Study results"]
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
/* ---------- explanations, recommendations, ladder, guide ---------- */
.bt-explain {{margin: -2px 0 14px 0; padding: 9px 14px; border-radius: 10px; font-size: 0.88rem; line-height: 1.5;
  background: var(--bt-accent-soft); border: 1px solid var(--bt-accent-line); color: inherit;}}
.bt-explain-tag {{font-weight: 800; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--bt-accent); margin-right: 6px;}}
.bt-reco {{margin: 4px 0 14px 0; padding: 12px 16px; border-radius: 14px; border: 1px solid var(--bt-border);
  background: linear-gradient(135deg, rgba(0,114,178,0.10), rgba(0,158,115,0.08)); box-shadow: var(--bt-shadow);
  animation: bt-rise .5s ease both;}}
.bt-reco-h {{font-weight: 800; font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; opacity: 0.8;
  margin-bottom: 6px;}}
.bt-reco-item {{display: flex; gap: 10px; align-items: baseline; font-size: 0.93rem; line-height: 1.5; margin: 3px 0;}}
.bt-reco-dot {{flex: none; width: 8px; height: 8px; border-radius: 50%; background: var(--bt-grad-warm);
  box-shadow: 0 0 0 3px rgba(0,158,115,0.15); transform: translateY(-1px);}}
.bt-ladder {{display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; margin: 6px 0 4px 0;}}
.bt-step {{position: relative; padding: 10px 10px 9px 10px; border-radius: 12px; border: 1px solid var(--bt-border);
  background: var(--bt-surface); transition: transform .15s ease, box-shadow .15s ease;}}
.bt-step:hover {{transform: translateY(-2px); box-shadow: var(--bt-shadow);}}
.bt-step-n {{display: inline-flex; width: 24px; height: 24px; border-radius: 7px; align-items: center; justify-content: center;
  font-weight: 800; font-size: 0.8rem; color: #fff; background: #8C8C8C;}}
.bt-step-done .bt-step-n {{background: var(--bt-grad);}}
.bt-step-done::after {{content: "✓"; position: absolute; right: 10px; top: 8px; color: #009E73; font-weight: 900;}}
.bt-step-t {{margin-top: 6px; font-size: 0.8rem; font-weight: 700; line-height: 1.25;}}
.bt-guide {{padding: 10px 12px; border-radius: 12px; border: 1px solid var(--bt-border); background: var(--bt-surface);
  margin-bottom: 10px;}}
.bt-guide-h {{font-weight: 800; font-size: 0.82rem; margin-bottom: 6px;}}
.bt-guide-bar {{height: 6px; border-radius: 99px; background: rgba(128,128,128,0.2); overflow: hidden; margin-bottom: 8px;}}
.bt-guide-bar > div {{height: 100%; background: var(--bt-grad-warm); transition: width .6s ease;}}
.bt-guide-row {{display: flex; gap: 8px; font-size: 0.82rem; opacity: 0.6; margin: 2px 0;}}
.bt-guide-row.ok {{opacity: 1; font-weight: 650;}}
.bt-guide-row.ok span {{color: #009E73;}}
@keyframes bt-rise {{from {{opacity: 0; transform: translateY(6px);}} to {{opacity: 1; transform: none;}}}}
div[data-testid="stPlotlyChart"], div[data-testid="stMetric"], .bt-card {{animation: bt-rise .45s ease both;}}
.bt-section {{animation: bt-rise .4s ease both;}}
@media (max-width: 900px) {{.bt-ladder {{grid-template-columns: repeat(3, 1fr);}}}}
@media (prefers-reduced-motion: reduce) {{.bt-reco, .bt-section, div[data-testid="stPlotlyChart"], div[data-testid="stMetric"] {{animation: none;}}}}
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


EXPLAIN: Dict[str, str] = {
    # diagnostics
    "gauges": "Instrument cluster: the battery's state of health, quick remaining-life estimate, resistance growth and peak temperature, coloured by risk. Use it to judge at a glance whether this battery needs attention.",
    "risk_matrix": "Each bubble is a battery: remaining life (x) against how fast it is degrading (y). Top-left means little life left and fading fast: act on those first.",
    "fleet_map": "Treemap of the fleet grouped by ambient temperature; tile size = cycles run, colour = remaining health margin. It shows whether problems cluster under one operating condition.",
    "fade_multi": "Capacity (or SOH) against cycle number for the selected batteries. It shows how fast each battery ages, where fade accelerates (★ knee) and where capacity recovers after rest (▲).",
    "trace_multi": "Raw voltage, current and temperature inside the chosen discharges. Late-life cycles reach the cut-off voltage sooner and heat more: the physical signature of ageing.",
    "cohort_": "SOH of the chosen batteries that share one ambient temperature. Comparing panels shows how temperature changes the ageing speed.",
    "hi_rank": "Candidate health indicators scored on four criteria (link to SOH, monotonicity, same trend in every cell, similar end values). The top one is the best single measure of health (Mission 1).",
    "hi_traj": "The best indicators of this battery normalised to their starting value, showing how each drifts as the battery ages.",
    "hi_pca": "Principal component analysis of all indicators. If one component explains most of the variance, one parameter is enough to describe health; otherwise ageing is multi-dimensional.",
    "ica": "Incremental capacity curves (dQ/dV). Peaks are electrode phase transitions; their shrinking and shifting reveal which ageing mechanism is active.",
    "ica_peaks": "Position and height of the main ICA peak over life: height loss points to loss of active material, a downward shift to growing resistance.",
    "dva": "Differential voltage curves (dV/dQ): the distance between peaks tracks the electrode that owns them, so shrinking spacing indicates active-material loss.",
    "modes": "Indicative split of the capacity loss into lithium-inventory loss (LLI), active-material loss (LAM) and conductivity loss (CL) over life.",
    "hc_modes": "Quantitative degradation modes from fitting electrode potentials to the discharge curve: lithium inventory lost and active material lost on each electrode.",
    "hc_fits": "Measured pseudo-OCV curves against the fitted half-cell model, and the electrode potentials behind them: how well the physics explains the data.",
    "stress": "Peak temperature, minimum voltage and cold-charging risk per cycle, with the literature thresholds. Exposure to these stressors explains faster ageing.",
    # live twin
    "rp_main": "The live twin sees the battery one discharge at a time. Each model forecasts the rest of life from what it has seen so far; grey circles (if revealed) are the future it has not seen.",
    "rp_track": "Top: each model's predicted end of life converging as data arrive. Middle: how much the live ensemble trusts each model. Bottom: which degradation mechanism is consuming the capacity.",
    "abl_fig": "The twin re-run with different measurement sets: the gap between open loop and voltage-only is the information the sensors add (Mission 2).",
    "uf_fig": "Tracking accuracy when the twin is updated every m cycles: it shows how rarely the twin can be updated without losing accuracy (Mission 2).",
    # models
    "lad_base": "Level 1 references: 'nothing changes' (persistence) and 'the recent straight line continues' (linear trend). A model is only useful if it beats these.",
    "lad_deep": "Level 4 deep sequence models (GRU, Transformer) read a window of past SOH values and predict the next cycle, repeatedly, to build the forecast.",
    "lad_hybrid": "Level 5 hybrid models combine physics equations with learning: the mechanistic PINN obeys the degradation kinetics; hierarchical Bayes combines a physics law with fleet knowledge.",
    "lad_spm": "Level 6 first-principles model: capacity follows from simulated discharges of a single-particle electrochemical model while SEI growth removes lithium.",
    "lad_spm_curves": "Simulated discharge curves of the single-particle model: the effect of lost lithium, higher current and cold on voltage and delivered capacity.",
    "lad_board": "All models run so far on this battery and forecast origin, from the simplest to the most advanced, scored on the same future cycles. The best is highlighted.",
    "ml_fig": "Forecasts of the selected ML models from the forecast origin (dotted line), with their uncertainty bands, against the measured SOH.",
    "ml_board": "Leaderboard of the ML models on the held-out cycles (fade skill: 1 = perfect, 0 = no better than assuming no further fade).",
    "ml_parity": "Predicted against measured SOH on the test cycles: points on the diagonal are perfect predictions.",
    "est_board": "SOH estimation from operando indicators: R² on the test set per model.",
    "est_parity": "Estimated against measured SOH for the test cycles, coloured by battery.",
    "est_traj": "Measured SOH (points) and the estimated SOH (line) of the test batteries over life.",
    "est_imp": "How much each indicator matters: the error increase when its values are shuffled.",
    "el_dq": "Change of the discharge curve between an early and a later cycle (ΔQ(V)), coloured by eventual life: large changes early predict a short life.",
    "el_parity": "Cycle life predicted from the first cycles only against the actual life (leave-one-battery-out).",
    "bench_": "Cross-battery benchmark: forecasts from several origins on several batteries, summarised with prognostic metrics.",
    # operations
    "ops_fig": "Simulated life under the chosen operating policy: currents chosen each cycle, SOH and cumulative profit.",
    "mm_fig": "Robustness: the twin-aware policy's advantage when the real battery differs from the model.",
    "om_heat": "Long-run profit rate for each operating policy and replacement threshold; the star is the best compliant combination (Mission 3).",
    "om_trade": "Replacing early costs money; replacing late raises the risk of sudden failure. The optimum balances the two.",
    "dp_fig": "Dynamic-programming policy: the best action (current or replace) for every state of health and season.",
    "scen_fig": "What-if planner: projected fade under each operating scenario, from the stress law fitted on this cohort.",
    # study
    "st_hi": "Health-indicator ranking across the whole cohort (Mission 1).",
    "st_mech": "Mechanism shares per condition group, one point per battery: plating should dominate in the cold, SEI when hot, LAM under high current.",
    "st_models": "Median forecast error per condition group and model on real data, with interquartile bars (Mission 2).",
}


def note(text: str) -> None:
    st.markdown(f'<div class="bt-explain"><span class="bt-explain-tag">What this shows</span> {html.escape(text)}</div>',
                unsafe_allow_html=True)


def explain(key: str) -> None:
    """Small 'what this shows' paragraph under a chart or table (catalogue lookup, exact key or prefix)."""
    text = EXPLAIN.get(key) or next((v for k, v in EXPLAIN.items() if k.endswith("_") and key.startswith(k)), None)
    if text:
        st.markdown(f'<div class="bt-explain"><span class="bt-explain-tag">What this shows</span> '
                    f'{html.escape(text)}</div>', unsafe_allow_html=True)


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
    explain(key)
    if export:
        export_row(fig, key, data)


def show_selectable(fig: go.Figure, key: str) -> Optional[Any]:
    """Chart with point selection (Streamlit >= 1.35). Returns the selection event or None."""
    cfg = dict(PLOT_CONFIG, modeBarButtonsToRemove=["lasso2d"])
    for extra in ([dict(width="stretch")] if ST_VERSION >= (1, 50) else []) + [dict(use_container_width=True)]:
        try:
            ev = st.plotly_chart(fig, theme=None, config=cfg, key=key, on_select="rerun",
                                 selection_mode="points", **extra)
            explain(key)
            return ev
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


def show_table(data: Any, note: Optional[str] = None) -> None:
    _show_table(data)
    if note:
        st.markdown(f'<div class="bt-explain"><span class="bt-explain-tag">What this shows</span> '
                    f'{html.escape(note)}</div>', unsafe_allow_html=True)


def _show_table(data: Any) -> None:
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
    if name == te.HB_NAME:
        return "#882255", "dashdot", "hexagon"
    return P.series[i % len(P.series)], DASHES[(i + 3) % len(DASHES)], SYMBOLS[i % len(SYMBOLS)]


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


AXIS_BLOCK_PX = 62      # tick labels + x-axis title below the plot area
LEGEND_ROW_PX = 24
TITLE_BAND_PX = 92


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


def _break_gaps(x: np.ndarray, y: np.ndarray, factor: float = 5.0, min_gap: float = 8.0) -> Tuple[List, List]:
    """Insert None where consecutive cycles are far apart, so missing data are shown as gaps
    instead of long straight lines."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return list(x), list(y)
    dx = np.diff(x)
    step = float(np.median(dx)) if len(dx) else 1.0
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(x)):
        if dx[i - 1] > max(factor * step, min_gap):
            xs.append(None)
            ys.append(None)
        xs.append(x[i])
        ys.append(y[i])
    return xs, ys


def fig_cohort_group(ct_all: pd.DataFrame, meta: pd.DataFrame, group_cells: Sequence[str], highlight: Sequence[str],
                     P: Palette, title: str, normalise_x: bool = False) -> go.Figure:
    """One ambient-temperature group with its own legend (one entry per battery)."""
    good = ct_all[~ct_all["outlier"]]
    allc = list(meta.index)
    fig = go.Figure()
    for cid in sorted(group_cells):
        d = good[good["Cell_ID"] == cid].sort_values("n")
        if d.empty:
            continue
        col, sym = cell_style(cid, allc)
        is_t = cid in highlight
        x = (d["n"] / d["n"].max()).to_numpy() if normalise_x else d["n"].to_numpy()
        xs, ys = _break_gaps(d["n"].to_numpy(), d["SOH"].to_numpy())
        if normalise_x:
            nmax = float(d["n"].max())
            xs = [v / nmax if v is not None else None for v in xs]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=cell_label(cid, meta) + ("  ★" if is_t else ""),
            line=dict(color=col, width=3.6 if is_t else 1.8), connectgaps=False,
            marker=dict(symbol=sym, size=7 if is_t else 5, maxdisplayed=16, line=dict(color=P.plot_bg, width=0.5)),
            opacity=1.0 if (is_t or not highlight) else 0.75,
            hovertemplate=f"<b>{cid}</b> n=%{{x}}: SOH %{{y:.3f}}<extra></extra>"))
    fig.update_xaxes(title_text="Fraction of recorded life" if normalise_x else "Discharge cycle n")
    fig.update_yaxes(title_text="SOH (–)")
    return style_fig(fig, P, 470, title, hovermode="closest")


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


def fig_risk_matrix(fs: pd.DataFrame, P: Palette, cell_id: str, highlight: Optional[Sequence[str]] = None) -> go.Figure:
    """Risk matrix: remaining life (x) against degradation speed (y). With ``highlight`` the selected
    batteries are drawn in colour and the rest of the fleet as grey context."""
    d_all = fs.reset_index()
    d = d_all[d_all["Cell_ID"].isin(highlight)] if highlight else d_all
    finite = d_all["Quick RUL"].replace(np.inf, np.nan)
    cap = float(np.nanmax(finite)) if np.isfinite(finite).any() else 300
    cap = max(cap * 1.2, 50)
    x = d["Quick RUL"].replace(np.inf, cap).clip(lower=1)
    fig = go.Figure()
    if highlight:
        ctx = d_all[~d_all["Cell_ID"].isin(highlight)]
        if len(ctx):
            fig.add_trace(go.Scatter(
                x=ctx["Quick RUL"].replace(np.inf, cap).clip(lower=1), y=ctx["Fade per 100 cycles (%)"],
                mode="markers", name="Rest of fleet", marker=dict(size=8, color=P.cohort, opacity=0.55),
                text=ctx["Cell_ID"], hovertemplate="%{text}<extra>fleet</extra>"))
    fig.add_vrect(x0=1, x1=15, fillcolor=rgba("#D55E00", 0.10), line_width=0)
    fig.add_vrect(x0=15, x1=50, fillcolor=rgba("#E69F00", 0.08), line_width=0)
    for risk in te.RISK_LEVELS:
        m = d["Risk"] == risk
        if not m.any():
            continue
        fig.add_trace(go.Scatter(
            x=x[m], y=d.loc[m, "Fade per 100 cycles (%)"], mode="markers+text", name=f"{RISK_ICON[risk]} {risk}",
            text=d.loc[m, "Cell_ID"], textposition="top center", textfont=dict(size=11, color=P.text),
            marker=dict(size=10 + 16 * np.sqrt(d.loc[m, "Cycles"] / d_all["Cycles"].max()), color=RISK_COLORS[risk],
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
# Figures: live replay, DP policy, half-cell fitting
# =============================================================================


LIVE_COLORS = {"twin": "#0072B2", "mech": "#D55E00", "pf": "#E69F00", "trend": "#009E73",
               "hb": "#CC79A7", "ens": None}
LIVE_DASH = {"twin": "dash", "mech": "solid", "pf": "dot", "trend": "dashdot", "hb": "longdash",
             "ens": "solid"}
MECH_COLORS = {"SEI": "#0072B2", "plating": "#56B4E9", "LAM": "#E69F00"}


def _live_color(m: str, P: Palette) -> str:
    return LIVE_COLORS.get(m) or P.text


def fig_live_main(ct_cell: pd.DataFrame, pc: pd.DataFrame, fr: Any, n: int, soh_eol: float, P: Palette,
                  reveal: bool, n_max: int, show_models: Sequence[str],
                  acc: Optional[Dict[str, float]] = None) -> go.Figure:
    good = ct_cell[~ct_cell["outlier"]]
    seen, future = good[good["n"] <= n], good[good["n"] > n]
    est = pc[pc["n"] <= n]
    fig = go.Figure()
    add_band(fig, est["n"].to_numpy(), (est["SOH"] - 2 * est["SOH_std"]).to_numpy(),
             (est["SOH"] + 2 * est["SOH_std"]).to_numpy(), P.ekf, "Twin estimate ±2σ", group="est", alpha=0.15)
    if fr is not None and "ens" in fr.forecasts and "ens" in show_models:
        med, lo, hi = fr.forecasts["ens"]
        add_band(fig, fr.n_grid, lo, hi, P.accent, "Live ensemble 90% band", group="ens", alpha=0.14)
    if reveal and len(future):
        fig.add_trace(go.Scatter(x=future["n"], y=future["SOH"], mode="markers", name="Future (hidden from models)",
                                 marker=dict(color=P.muted, size=5, opacity=0.35, symbol="circle-open"),
                                 hovertemplate="%{y:.4f}<extra>future</extra>"))
    fig.add_trace(go.Scatter(x=seen["n"], y=seen["SOH"], mode="markers", name="Measured so far",
                             marker=dict(color=P.measured, size=6), hovertemplate="%{y:.4f}<extra>measured</extra>"))
    fig.add_trace(go.Scatter(x=est["n"], y=est["SOH"], mode="lines", name="Twin estimate (causal)", legendgroup="est",
                             line=dict(color=P.ekf, width=2.6), hovertemplate="%{y:.4f}<extra>twin</extra>"))
    if fr is not None:
        for m in show_models:
            if m not in fr.forecasts:
                continue
            med = fr.forecasts[m][0]
            w = fr.weights.get(m)
            a = (acc or {}).get(m)
            label = te.LIVE_MODELS[m] + (f" · acc {a:.1f}%" if a is not None and np.isfinite(a) else "") \
                + (f" · w = {w:.2f}" if w is not None and m != "ens" else "")
            fig.add_trace(go.Scatter(x=fr.n_grid, y=med, mode="lines", name=label, legendgroup=m,
                                     line=dict(color=_live_color(m, P), width=4 if m == "ens" else 2.2, dash=LIVE_DASH[m]),
                                     hovertemplate="%{y:.4f}<extra>" + html.escape(te.LIVE_MODELS[m]) + "</extra>"))
    fig.add_vline(x=n, line_color=P.ekf, line_width=1.5, line_dash="dot",
                  annotation_text=f"now: n = {n}", annotation_font=dict(color=P.ekf, size=12))
    fig.add_hline(y=soh_eol, line_dash="dash", line_color=P.eol, annotation_text="End of life",
                  annotation_font=dict(color=P.eol))
    lo_y = min(float(good["SOH"].min()), soh_eol) - 0.04
    fig.update_xaxes(title_text="Discharge cycle n", range=[0, n_max])
    fig.update_yaxes(title_text="State of health SOH (–)", range=[lo_y, 1.04])
    return style_fig(fig, P, 560, "Live multi-model twin: every model assimilates one discharge at a time",
                     hovermode="x unified")


def fig_live_track(track: pd.DataFrame, n: int, eol_true: Optional[int], P: Palette, n_max: int,
                   show_models: Sequence[str]) -> go.Figure:
    has_mech = any(c.startswith("share_") for c in track.columns)
    nrow = 3 if has_mech else 2
    fig = make_subplots(rows=nrow, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        row_heights=[0.42, 0.29, 0.29] if has_mech else [0.55, 0.45],
                        subplot_titles=("Predicted end-of-life cycle converging as data arrive",
                                        "Live ensemble weights (recent 5-cycle-ahead skill)")
                        + (("Which mechanism consumes the capacity? (mechanistic PF, share of loss so far)",)
                           if has_mech else ()))
    _style_subplot_titles(fig, P)
    t = track[track["n"] <= n]
    for m in show_models:
        c = f"eol_{m}"
        if c in t:
            fig.add_trace(go.Scatter(x=t["n"], y=t[c], mode="lines", name=te.LIVE_MODELS[m], legendgroup=m,
                                     line=dict(color=_live_color(m, P), width=3.4 if m == "ens" else 1.8,
                                               dash=LIVE_DASH[m]), connectgaps=False,
                                     hovertemplate="%{y:.0f}<extra>" + html.escape(te.LIVE_MODELS[m]) + "</extra>"),
                          row=1, col=1)
    if eol_true is not None:
        fig.add_hline(y=eol_true, line_dash="dash", line_color=P.eol, row=1, col=1,
                      annotation_text=f"actual EOL n = {eol_true}", annotation_font=dict(color=P.eol))
    if has_mech:
        for k_ in te.MECH_NAMES:
            c = f"share_{k_}"
            if c in t:
                fig.add_trace(go.Scatter(x=t["n"], y=t[c], mode="lines", stackgroup="mech", name=f"{k_} share",
                                         line=dict(color=MECH_COLORS[k_], width=0.8),
                                         fillcolor=rgba(MECH_COLORS[k_], 0.65),
                                         hovertemplate="%{y:.0%}<extra>" + k_ + "</extra>"), row=3, col=1)
        fig.update_yaxes(title_text="share", range=[0, 1], tickformat=".0%", row=3, col=1)
    for m in [m for m in ("twin", "mech", "pf", "trend", "hb") if f"w_{m}" in t]:
        fig.add_trace(go.Scatter(x=t["n"], y=t[f"w_{m}"], mode="lines", stackgroup="w", name=f"weight · {te.LIVE_MODELS[m]}",
                                 legendgroup=m, showlegend=False, line=dict(color=_live_color(m, P), width=0.8),
                                 fillcolor=rgba(_live_color(m, P), 0.6),
                                 hovertemplate="%{y:.2f}<extra>" + html.escape(te.LIVE_MODELS[m]) + "</extra>"),
                      row=2, col=1)
    fig.update_xaxes(range=[0, n_max])
    fig.update_xaxes(title_text="Discharge cycle n (data assimilated so far)", row=nrow, col=1)
    fig.update_yaxes(title_text="EOL cycle", row=1, col=1)
    fig.update_yaxes(title_text="weight", range=[0, 1], row=2, col=1)
    return style_fig(fig, P, 880 if has_mech else 680, None)


def fig_dp_policy(dp: te.DPResult, P: Palette) -> go.Figure:
    z = np.where(dp.action < 0, -1, np.array(dp.currents)[np.clip(dp.action, 0, None)])
    labels = ["Replace"] + [f"{c:g} A" for c in dp.currents]
    vals = [-1] + list(dp.currents)
    palette = ["#D55E00", "#56B4E9", "#009E73", "#E69F00", "#CC79A7"][: len(vals)]
    idx = np.vectorize(lambda v: vals.index(v))(z)
    cs = []
    for i, c in enumerate(palette):
        cs += [[i / len(palette), c], [(i + 1) / len(palette), c]]
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.14, row_heights=[0.72, 0.28],
                        subplot_titles=("Optimal action by state of health and season", "Ambient temperature"))
    _style_subplot_titles(fig, P)
    fig.add_trace(go.Heatmap(x=dp.phases, y=dp.soh_grid, z=idx, colorscale=cs, zmin=-0.5, zmax=len(palette) - 0.5,
                             colorbar=dict(tickvals=list(range(len(labels))), ticktext=labels, thickness=14, len=0.7,
                                           y=0.64, tickfont=dict(color=P.text)),
                             customdata=np.array(labels)[idx],
                             hovertemplate="season cycle %{x:.0f} · SOH %{y:.3f}<br>%{customdata}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=dp.phases, y=dp.replace_boundary, mode="lines+markers", name="Replacement boundary",
                             line=dict(color=P.text, width=2.5, dash="dash"), marker=dict(size=6),
                             hovertemplate="replace below SOH %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dp.phases, y=dp.T_amb, mode="lines+markers", name="Ambient", showlegend=False,
                             line=dict(color=P.eis, width=2.5), hovertemplate="%{y:.1f} °C<extra></extra>"),
                  row=2, col=1)
    fig.update_yaxes(title_text="SOH (–)", row=1, col=1)
    fig.update_yaxes(title_text="°C", row=2, col=1)
    fig.update_xaxes(title_text="Cycle within the seasonal period", row=2, col=1)
    return style_fig(fig, P, 700, f"Dynamic-programming policy · optimal long-run profit rate ρ* = {dp.rho:.4f} CU/h",
                     hovermode="closest")


def fig_half_cell_fits(fits: List[te.HalfCellFit], P: Palette) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.14,
                        subplot_titles=("Pseudo-OCV (IR-compensated) and half-cell model fit",
                                        "Electrode potentials at beginning of life (vs Li/Li⁺)"))
    _style_subplot_titles(fig, P)
    cols = sample_colorscale("Viridis", list(np.linspace(*P.ica_range, max(len(fits), 2))))
    for f, c in zip(fits, cols):
        fig.add_trace(go.Scatter(x=f.q, y=f.v, mode="markers", name=f"n = {f.n} data", legendgroup=f"n{f.n}",
                                 marker=dict(color=c, size=4, opacity=0.55),
                                 hovertemplate="%{y:.3f} V<extra>data</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=f.q, y=f.v_fit, mode="lines", name=f"n = {f.n} fit ({f.rmse_mV:.1f} mV)",
                                 legendgroup=f"n{f.n}", line=dict(color=c, width=2.4),
                                 hovertemplate="%{y:.3f} V<extra>fit</extra>"), row=1, col=1)
    f0 = fits[0]
    q = f0.q
    y = f0.params["y0"] + q / f0.params["Cp_Ah"]
    x = f0.params["x0"] - q / f0.params["Cn_Ah"]
    fig.add_trace(go.Scatter(x=q, y=te.ocp_lco(y), mode="lines", name="Positive: LiCoO₂",
                             line=dict(color=P.accent, width=2.6), hovertemplate="%{y:.3f} V<extra>LCO</extra>"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=q, y=te.ocp_graphite(x), mode="lines", name="Negative: graphite",
                             line=dict(color=P.r_ct, width=2.6), hovertemplate="%{y:.3f} V<extra>graphite</extra>"),
                  row=2, col=1)
    fig.update_xaxes(title_text="Discharged capacity (Ah)", row=2, col=1)
    fig.update_yaxes(title_text="Cell voltage (V)", row=1, col=1)
    fig.update_yaxes(title_text="Potential vs Li/Li⁺ (V)", row=2, col=1)
    return style_fig(fig, P, 760, None, hovermode="closest")


def fig_half_cell_modes(tab: pd.DataFrame, P: Palette) -> go.Figure:
    fig = go.Figure()
    for i, (col, name) in enumerate((("Capacity loss (%)", "Capacity loss"), ("LLI (%)", "Loss of lithium inventory"),
                                     ("LAM_PE (%)", "LAM positive (LiCoO₂)"), ("LAM_NE (%)", "LAM negative (graphite)"))):
        fig.add_trace(go.Scatter(x=tab["n"], y=tab[col], mode="lines+markers", name=name,
                                 line=dict(color=P.mode_colors[i], width=3 if i else 2, dash=DASHES[i]),
                                 marker=dict(symbol=SYMBOLS[i], size=9),
                                 hovertemplate="%{y:.2f}%<extra>" + name + "</extra>"))
    fig.add_hline(y=0, line_color=P.muted, line_width=1)
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="% of beginning-of-life value")
    return style_fig(fig, P, 500, "Quantitative degradation modes from half-cell fitting")


def fig_dq_curves(ct_all: pd.DataFrame, el: Dict[str, Any], P: Palette) -> go.Figure:
    """Delta-Q(V) = Q_nb(V) - Q_na(V) per cell, coloured by log10 cycle life (Severson et al. 2019, Fig. 2)."""
    tab = el["table"].set_index("Cell_ID")
    lives = tab["life"].dropna()
    lo, hi = (np.log10(lives.min()), np.log10(lives.max())) if len(lives) else (1, 3)
    fig = go.Figure()
    for cid, d in ct_all[~ct_all["outlier"]].groupby("Cell_ID"):
        dq = te.delta_q_curve(d, el["n_a"], el["n_b"])
        if dq is None:
            continue
        life = tab.loc[cid, "life"] if cid in tab.index else np.nan
        frac = (np.log10(life) - lo) / max(hi - lo, 1e-9) if np.isfinite(life) else None
        col = sample_colorscale("Viridis", [float(np.clip(frac, 0, 1))])[0] if frac is not None else P.cohort
        fig.add_trace(go.Scatter(x=1000 * dq, y=te.QV_GRID, mode="lines", name=cid, showlegend=False,
                                 line=dict(color=col, width=2 if frac is not None else 1, dash="solid" if frac is not None else "dot"),
                                 hovertemplate=f"<b>{cid}</b> life {life if np.isfinite(life) else 'censored'}"
                                               "<br>ΔQ %{x:.1f} mAh at %{y:.2f} V<extra></extra>"))
    if len(lives):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(colorscale="Viridis", cmin=lo, cmax=hi, color=[lo], showscale=True,
                                             colorbar=dict(title=dict(text="log₁₀ life", font=dict(color=P.text)),
                                                           tickfont=dict(color=P.muted), thickness=12))))
    fig.add_vline(x=0, line_color=P.muted, line_width=1)
    fig.update_xaxes(title_text=f"ΔQ(V) = Q(n={el['n_b']}) − Q(n={el['n_a']})  (mAh)")
    fig.update_yaxes(title_text="Voltage (V)")
    return style_fig(fig, P, 520, "Early-life ΔQ(V) curves, coloured by eventual cycle life (dotted = censored)",
                     hovermode="closest")


def fig_lifetime_parity(el: Dict[str, Any], P: Palette, all_cells: Sequence[str]) -> go.Figure:
    d = el["loco"]
    fig = go.Figure()
    lo, hi = float(min(d["life"].min(), d["life_pred"].min())) * 0.85, float(max(d["life"].max(), d["life_pred"].max())) * 1.15
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect", line=dict(color=P.muted, dash="dash")))
    fig.add_trace(go.Scatter(x=[lo, hi, hi, lo], y=[lo * 0.8, hi * 0.8, hi * 1.2, lo * 1.2], fill="toself",
                             fillcolor=rgba(P.muted, 0.1), line=dict(width=0), name="±20 %", hoverinfo="skip"))
    for _, r in d.iterrows():
        col, sym = cell_style(r["Cell_ID"], all_cells)
        fig.add_trace(go.Scatter(x=[r["life"]], y=[r["life_pred"]], mode="markers+text", text=[r["Cell_ID"]],
                                 textposition="top center", textfont=dict(size=10, color=P.muted), showlegend=False,
                                 marker=dict(color=col, symbol=sym, size=11, line=dict(color=P.text, width=0.8)),
                                 hovertemplate=f"<b>{r['Cell_ID']}</b><br>actual %{{x:.0f}} · predicted %{{y:.0f}} cycles"
                                               "<extra></extra>"))
    fig.update_xaxes(title_text="Actual cycles to end of life", type="log")
    fig.update_yaxes(title_text="Predicted from early cycles (leave-one-cell-out)", type="log")
    return style_fig(fig, P, 520, f"Early-life lifetime prediction · MAPE {el['mape_pct']:.1f}% "
                                  f"(cohort-mean baseline {el['baseline_mape_pct']:.1f}%)", hovermode="closest")


# =============================================================================
# Figures: study results (cohort-wide evidence)
# =============================================================================
def fig_cohort_models(summary: pd.DataFrame, P: Palette) -> go.Figure:
    """Median 20-cycle forecast RMSE per condition group and live model, with interquartile bars."""
    d = summary.reset_index()
    groups = [g for g in list(te.GROUP_ORDER) + ["All batteries"] if g in set(d["Group"])]
    fig = go.Figure()
    for m in [m for m in ("twin", "mech", "pf", "trend", "hb", "ens") if m in set(d["model"])]:
        dm = d[d["model"] == m].set_index("Group").reindex(groups)
        col = LIVE_COLORS.get(m) or P.text
        fig.add_trace(go.Bar(
            x=groups, y=dm["median RMSE"], name=te.LIVE_MODELS[m], marker=dict(color=col, line=dict(color=P.text, width=0.4)),
            error_y=dict(type="data", symmetric=False, array=(dm["IQR high"] - dm["median RMSE"]).clip(lower=0),
                         arrayminus=(dm["median RMSE"] - dm["IQR low"]).clip(lower=0), color=P.muted, thickness=1),
            customdata=np.column_stack([dm["cells"].fillna(0), dm["coverage"]]),
            hovertemplate="%{x}<br>median RMSE %{y:.4f}<br>%{customdata[0]:.0f} cells · coverage %{customdata[1]:.0%}"
                          "<extra>" + html.escape(te.LIVE_MODELS[m]) + "</extra>"))
    fig.update_layout(barmode="group", bargap=0.2)
    fig.update_yaxes(title_text="Median 20-cycle forecast RMSE (SOH)")
    fig.update_xaxes(title_text=None)
    return style_fig(fig, P, 520, "Forecast error by operating condition and model (bars: interquartile range)",
                     hovermode="closest")


def fig_mech_groups(mech: pd.DataFrame, P: Palette) -> go.Figure:
    """Mechanism shares attributed by the mechanistic particle filter, per condition group."""
    fig = go.Figure()
    groups = [g for g in te.GROUP_ORDER if g in set(mech["Group"])]
    for k in te.MECH_NAMES:
        fig.add_trace(go.Box(x=mech["Group"], y=mech[f"share_{k}"], name=k, marker_color=MECH_COLORS[k],
                             boxpoints="all", jitter=0.4, pointpos=0, line=dict(width=1.5),
                             hovertemplate="%{x}: %{y:.0%}<extra>" + k + "</extra>"))
    fig.update_layout(boxmode="group")
    fig.update_xaxes(categoryorder="array", categoryarray=groups)
    fig.update_yaxes(title_text="Share of capacity lost", tickformat=".0%", range=[0, 1])
    return style_fig(fig, P, 480, "Which mechanism dominates under which conditions? (mechanistic PF, one point per battery)",
                     hovermode="closest")


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
def replay_cached(_cell_df: pd.DataFrame, _ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str, cell: str,
                  soh_eol: float, level: float, cap_every: int = 10) -> Tuple[te.EKFResult, pd.DataFrame]:
    r = te.run_dual_twin(_cell_df, _ct, _imp, cell, te.TwinParameters(),
                         te.twin_config_for(_ct, cell, te.DualTwinConfig(capacity_every=cap_every)))
    n_max = int(r.per_cycle["n"].max() * 1.6)
    rows = []
    for n in r.per_cycle["n"].astype(int):
        if n < 5:
            continue
        f = te.twin_forecast(r, int(n), n_max, soh_eol, level)
        rs = f.rul_samples[np.isfinite(f.rul_samples)] if f.rul_samples is not None else np.array([])
        if len(rs):
            rows.append({"n": int(n), "eol_med": n + float(np.median(rs)), "eol_lo": n + float(np.quantile(rs, 0.05)),
                         "eol_hi": n + float(np.quantile(rs, 0.95)), "frac_censored": 1 - len(rs) / len(f.rul_samples)})
    return r, pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=8)
def live_cached(_ekf: te.EKFResult, _ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str, cell: str,
                soh_eol: float, level: float, cap_every: int, models: Tuple[str, ...]) -> Tuple[Dict[int, Any], pd.DataFrame]:
    return te.live_multi_model(_ct, cell, _ekf, soh_eol, level, models, imp=_imp)


@st.cache_data(show_spinner=False, max_entries=4)
def dp_cached(econ: Dict[str, Any], phys: Dict[str, Any], maint: Dict[str, Any], amb_mean: float, amb_amp: float,
              n_soh: int, n_phase: int) -> te.DPResult:
    return te.solve_replacement_dp(te.Economics(**econ), te.CellPhysics(**phys), te.MaintenanceModel(**maint),
                                   n_soh=n_soh, n_phase=n_phase, ambient_mean_C=amb_mean, ambient_amp_C=amb_amp)


@st.cache_data(show_spinner=False, max_entries=16)
def half_cell_cached(_prep: pd.DataFrame, _ct_cell: pd.DataFrame, key: str, cell: str, n_curves: int,
                     ir: bool) -> Tuple[pd.DataFrame, List[te.HalfCellFit]]:
    return te.half_cell_trajectory(_prep, _ct_cell, n_curves, ir)


@st.cache_data(show_spinner=False, max_entries=8)
def early_life_cached(_ct: pd.DataFrame, key: str, eol_ah: float, n_a: int, n_b: int) -> Dict[str, Any]:
    return te.early_life_lifetime(_ct, eol_ah, n_a, n_b)


@st.cache_data(show_spinner=False, max_entries=4)
def update_cohort_cached(_store: Any, _ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str, eol_ah: float,
                         cells: Tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for c in cells:
        g = _ct[(_ct["Cell_ID"] == c) & ~_ct["outlier"]]
        soh_eol_c = te.soh_eol_for(float(g["C_bol_Ah"].iloc[0]), eol_ah)
        n0 = int(max(5, round(0.4 * g["n"].max())))
        try:
            tab, _ = te.update_frequency_study(_store.cell_frame(c), _ct, _imp, c, te.TwinParameters(),
                                               te.twin_config_for(_ct, c, te.DualTwinConfig()), n0, soh_eol_c,
                                               intervals=(1, 5, 20))
            rows.append({"Cell_ID": c, "recommended": tab.attrs.get("recommended"),
                         **{f"RMSE every {m}": float(tab.loc[m, "Tracking RMSE"]) for m in tab.index}})
        except Exception:
            continue
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=16)
def calibrated_plant_cached(_ct: pd.DataFrame, _imp: Optional[pd.DataFrame], key: str, cell: str
                            ) -> Tuple[te.CellPhysics, pd.DataFrame]:
    return te.calibrate_plant(_ct, _imp, cell)


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
HERO_CSS = """<style>
.bt-hero {display: flex; align-items: center; gap: 28px; padding: 22px 30px 20px 30px;}
.bt-hero-text {position: relative; flex: 1 1 auto; min-width: 0;}
.bt-hero .bt-title {font-size: 1.85rem;}
.bt-hero .bt-sub {max-width: 60ch; font-size: 1.02rem;}
.bt-art {position: relative; flex: 0 0 460px; width: 460px; height: 210px;}
.bt-art svg {position: absolute; inset: 0; overflow: visible;}
.bt-pkt {position: absolute; left: 0; top: 0; width: 9px; height: 9px; border-radius: 50%;
  offset-anchor: 50% 50%; offset-rotate: 0deg; animation: bt-move 2.8s linear infinite; opacity: 0;}
.bt-pkt-up {offset-path: path('M 118 72 C 190 14, 270 14, 342 72'); background: #7FE3FF;
  box-shadow: 0 0 10px 2px rgba(127,227,255,0.9);}
.bt-pkt-down {offset-path: path('M 342 150 C 270 208, 190 208, 118 150'); background: #FFD166;
  box-shadow: 0 0 10px 2px rgba(255,209,102,0.9);}
@keyframes bt-move {0% {offset-distance: 0%; opacity: 0;} 12% {opacity: 1;} 88% {opacity: 1;}
  100% {offset-distance: 100%; opacity: 0;}}
.bt-level {transform-box: fill-box; transform-origin: 50% 100%; animation: bt-charge 7s ease-in-out infinite;}
@keyframes bt-charge {0%, 100% {transform: scaleY(1);} 50% {transform: scaleY(0.3);}}
.bt-draw {stroke-dasharray: 240; stroke-dashoffset: 240; animation: bt-draw 7s ease-in-out infinite;}
@keyframes bt-draw {0% {stroke-dashoffset: 240;} 55%, 90% {stroke-dashoffset: 0;} 100% {stroke-dashoffset: 0; opacity: 0;}}
.bt-ring {transform-box: fill-box; transform-origin: center; animation: bt-ring 2.2s ease-out infinite;}
@keyframes bt-ring {0% {transform: scale(1); opacity: 0.9;} 100% {transform: scale(2.0); opacity: 0;}}
.bt-scan {animation: bt-scan 3.5s ease-in-out infinite;}
@keyframes bt-scan {0%, 100% {transform: translateY(0); opacity: 0.0;} 10% {opacity: 0.8;} 50% {transform: translateY(118px); opacity: 0.8;} 60% {opacity: 0;}}
.bt-glow {animation: bt-glow 3s ease-in-out infinite;}
@keyframes bt-glow {0%, 100% {opacity: 0.35;} 50% {opacity: 0.9;}}
@media (max-width: 1150px) {.bt-art {display: none;}}
@media (prefers-reduced-motion: reduce) {.bt-pkt, .bt-level, .bt-draw, .bt-ring, .bt-scan, .bt-glow {animation: none;}
  .bt-pkt {opacity: 0;} .bt-draw {stroke-dashoffset: 0;}}
</style>"""

HERO_ART = """<div class="bt-art" aria-hidden="true">
<svg width="460" height="210" viewBox="0 0 460 210" xmlns="http://www.w3.org/2000/svg">
 <defs>
  <linearGradient id="btMetal" x1="0" x2="1" y1="0" y2="0">
   <stop offset="0" stop-color="#8FA6BE"/><stop offset="0.45" stop-color="#EEF5FB"/><stop offset="1" stop-color="#6F869F"/>
  </linearGradient>
  <linearGradient id="btCharge" x1="0" x2="0" y1="0" y2="1">
   <stop offset="0" stop-color="#9BF6C9"/><stop offset="1" stop-color="#1FB57A"/>
  </linearGradient>
 </defs>
 <!-- data links -->
 <path d="M 118 72 C 190 14, 270 14, 342 72" fill="none" stroke="rgba(127,227,255,0.55)" stroke-width="1.6" stroke-dasharray="4 5"/>
 <path d="M 342 150 C 270 208, 190 208, 118 150" fill="none" stroke="rgba(255,209,102,0.55)" stroke-width="1.6" stroke-dasharray="4 5"/>
 <text x="230" y="22" text-anchor="middle" font-size="10.5" letter-spacing="1.5" fill="rgba(255,255,255,0.92)" font-weight="700">V · I · T TELEMETRY</text>
 <text x="230" y="206" text-anchor="middle" font-size="10.5" letter-spacing="1.5" fill="rgba(255,255,255,0.92)" font-weight="700">SOH · RUL · CONTROL</text>
 <!-- update engine -->
 <circle class="bt-ring" cx="230" cy="111" r="24" fill="none" stroke="#7FE3FF" stroke-width="2"/>
 <circle cx="230" cy="111" r="24" fill="rgba(14,23,38,0.55)" stroke="rgba(255,255,255,0.75)" stroke-width="1.4"/>
 <text x="230" y="108" text-anchor="middle" font-size="11" font-weight="800" fill="#FFFFFF">EKF</text>
 <text x="230" y="121" text-anchor="middle" font-size="8" fill="rgba(255,255,255,0.8)">update</text>
 <line x1="206" y1="111" x2="150" y2="111" stroke="rgba(255,255,255,0.25)" stroke-width="1"/>
 <line x1="254" y1="111" x2="316" y2="111" stroke="rgba(255,255,255,0.25)" stroke-width="1"/>
 <!-- physical 18650 cell -->
 <ellipse class="bt-glow" cx="70" cy="112" rx="58" ry="82" fill="rgba(255,209,102,0.10)"/>
 <rect x="58" y="30" width="24" height="12" rx="3" fill="#DDE7F1" stroke="rgba(255,255,255,0.8)"/>
 <rect x="30" y="40" width="80" height="145" rx="14" fill="url(#btMetal)" stroke="rgba(255,255,255,0.85)" stroke-width="1.5"/>
 <rect x="42" y="56" width="56" height="113" rx="7" fill="rgba(11,61,145,0.45)"/>
 <rect class="bt-level" x="42" y="56" width="56" height="113" rx="7" fill="url(#btCharge)"/>
 <text x="70" y="118" text-anchor="middle" font-size="15" font-weight="900" fill="rgba(11,40,80,0.75)">+</text>
 <text x="70" y="200" text-anchor="middle" font-size="9.5" letter-spacing="1.4" font-weight="800" fill="#FFFFFF">PHYSICAL CELL</text>
 <!-- digital twin -->
 <rect x="362" y="30" width="24" height="12" rx="3" fill="none" stroke="#7FE3FF" stroke-width="1.4" stroke-dasharray="3 3"/>
 <rect x="350" y="40" width="80" height="145" rx="14" fill="rgba(127,227,255,0.10)" stroke="#7FE3FF" stroke-width="1.6" stroke-dasharray="6 4"/>
 <g stroke="rgba(127,227,255,0.22)" stroke-width="1">
  <line x1="360" y1="70" x2="420" y2="70"/><line x1="360" y1="100" x2="420" y2="100"/>
  <line x1="360" y1="130" x2="420" y2="130"/><line x1="360" y1="160" x2="420" y2="160"/>
  <line x1="375" y1="55" x2="375" y2="172"/><line x1="395" y1="55" x2="395" y2="172"/><line x1="415" y1="55" x2="415" y2="172"/>
 </g>
 <polyline class="bt-draw" points="360,64 372,67 384,72 396,80 405,90 412,104 417,122 421,146" fill="none" stroke="#FFFFFF" stroke-width="2.4" stroke-linecap="round"/>
 <line x1="360" y1="140" x2="422" y2="140" stroke="#FF8A4C" stroke-width="1.4" stroke-dasharray="3 3"/>
 <rect class="bt-scan" x="352" y="50" width="76" height="3" rx="1.5" fill="rgba(127,227,255,0.9)"/>
 <text x="390" y="200" text-anchor="middle" font-size="9.5" letter-spacing="1.4" font-weight="800" fill="#FFFFFF">DIGITAL TWIN</text>
</svg>
<span class="bt-pkt bt-pkt-up" style="animation-delay:0s"></span>
<span class="bt-pkt bt-pkt-up" style="animation-delay:0.7s"></span>
<span class="bt-pkt bt-pkt-up" style="animation-delay:1.4s"></span>
<span class="bt-pkt bt-pkt-up" style="animation-delay:2.1s"></span>
<span class="bt-pkt bt-pkt-down" style="animation-delay:0.35s"></span>
<span class="bt-pkt bt-pkt-down" style="animation-delay:1.75s"></span>
</div>"""

pills = "".join(f'<span class="bt-pill">{t}</span>' for t in
                ("Self-updating EKF twin", "Mechanistic PINN", "Bayesian prognostics", "O&amp;M optimisation"))
HERO_ART = re.sub(r"<!--.*?-->", "", HERO_ART)
HERO_ART = " ".join(line.strip() for line in HERO_ART.splitlines())      # one line: no markdown code blocks
HERO_CSS = " ".join(line.strip() for line in HERO_CSS.splitlines())
st.markdown(
    HERO_CSS +
    '<div class="bt-hero"><div class="bt-hero-text">'
    '<div class="bt-kicker">Self-updating digital twin · NASA Ames Li-ion data</div>'
    '<div class="bt-title">🔋 Battery Digital Twin</div>'
    '<div class="bt-sub">A physical cell and its virtual copy, synchronised cycle by cycle: diagnose ageing, '
    'forecast remaining life and decide when to act.</div>'
    f'<div>{pills}</div></div>' + HERO_ART + '</div>',
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
DATA_KEY = f"{store.key}|engine-{te.ENGINE_VERSION}"   # engine upgrades invalidate every cached result


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
        eol_ah = st.number_input("End-of-life capacity (Ah)", 0.8, 2.0, float(te.DEFAULT_EOL_AH), 0.05,
                                 help="The single end-of-life definition used by diagnostics, forecasts, RUL and the "
                                      "study (NASA convention: 1.4 Ah = 30 % fade of 2 Ah). Each battery's SOH "
                                      "threshold is this capacity divided by its own beginning-of-life capacity. "
                                      "The replacement SOH in Operations is a separate, optimised decision.")

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
    cells = list(st.session_state.get("_diag_sel") or [cell])
    c1.markdown(f"**{len(cells)} batter{'y' if len(cells) == 1 else 'ies'}** from the selection above"
                + ("" if len(cells) <= 8 else " (whole fleet: click legend entries to hide cells)"))
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
        sel_ = st.session_state.get("_diag_sel")
        if sel_:
            summ = summ[summ.index.isin(sel_)]
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


DIAG_SCOPES = ("single", "selected", "fleet")
DIAG_SCOPE_LABEL = {"single": ":material/battery_full: Single battery",
                    "selected": ":material/stacks: Selected batteries",
                    "fleet": ":material/grid_view: Whole fleet"}


def _pick_critical(fs: pd.DataFrame, k: int = 6) -> None:
    st.session_state["diag_cells"] = list(fs.index[:k])
    st.session_state["diag_scope"] = DIAG_SCOPE_LABEL["selected"]


def battery_status_card(cid: str, fs: pd.DataFrame, lim: Dict[str, float], key: str) -> None:
    """Alert banner, gauge cluster and key facts for one battery."""
    if cid not in fs.index:
        st.info(f"{cid}: not enough valid cycles for a status card.")
        return
    row = fs.loc[cid]
    risk_banner(row, cell_label(cid, meta))
    show(fig_gauges(row, P, te.SafetyLimits(**lim)), key=f"gauges_{key}", export=False)
    k = st.columns(5)
    k[0].metric("Valid cycles", int(meta.loc[cid, "cycles"]))
    k[1].metric("Initial capacity", fmt(meta.loc[cid, "C_bol_Ah"], ".3f", "Ah"))
    k[2].metric("Capacity fade", fmt(meta.loc[cid, "fade_pct"], ".1f", "%"))
    k[3].metric("Fade per 100 cycles", fmt(row["Fade per 100 cycles (%)"], ".2f", "%"))
    k[4].metric("Regeneration events", int(meta.loc[cid].get("regen_events", 0) or 0))


def triage_table(fs: pd.DataFrame, key: str) -> None:
    tbl = fs.reset_index()[["Cell_ID", "Risk", "SOH", "Health margin", "Quick RUL", "Fade per 100 cycles (%)",
                            "R growth (%)", "Peak T (°C)", "Cycles", "Ambient_C", "I_dis_A", "Alerts"]].copy()
    tbl["Risk"] = tbl["Risk"].map(lambda r: f"{RISK_ICON[r]} {r}")
    tbl["Quick RUL"] = tbl["Quick RUL"].replace(np.inf, np.nan)
    try:
        st.dataframe(tbl, hide_index=True, use_container_width=True, height=min(640, 38 + 35 * len(tbl)), key=key,
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


def event_log(ev: pd.DataFrame, cells: Sequence[str], key: str) -> None:
    c1, c2 = st.columns([3, 1])
    sev = c1.multiselect("Severity", ["Critical", "Warning", "Watch", "Info"], default=["Critical", "Warning", "Watch"],
                         key=f"{key}_sev")
    evf = ev[ev["Severity"].isin(sev) & ev["Cell_ID"].isin(cells)]
    c2.metric("Events shown", len(evf))
    evf = evf.assign(Severity=evf["Severity"].map(lambda s_: {"Critical": "🔴", "Warning": "🟠", "Watch": "🟡",
                                                              "Info": "🔵"}[s_] + " " + s_))
    if evf.empty:
        st.success("No events of the chosen severity for this selection.")
        return
    try:
        st.dataframe(evf, hide_index=True, use_container_width=True, height=min(420, 38 + 35 * len(evf)), key=key,
                     column_config={"Cell_ID": st.column_config.TextColumn("Battery"),
                                    "n": st.column_config.NumberColumn("Cycle", format="%d")})
    except Exception:
        show_table(evf)


def selection_report(sel: Sequence[str], fs: pd.DataFrame, ev: pd.DataFrame, lim: Dict[str, float]) -> None:
    label = sel[0] if len(sel) == 1 else f"{len(sel)} batteries"
    if st.button(f"Build report · {label}", key="rep_go", icon=":material/description:", type="primary"):
        with st.spinner("Assembling report…"):
            knees = {c: knee_cached(ct[ct["Cell_ID"] == c], DATA_KEY, c) for c in sel[:12]}
            fs_sel = fs[fs.index.isin(sel)]
            secs: List[Tuple[str, Any]] = [
                ("Selection", ", ".join(cell_label(c, meta) for c in sel)),
                ("Status", fs_sel.drop(columns=["risk_level"])),
                ("Capacity fade", fig_fade_compare(ct, meta, list(sel), "SOH", soh_eol if len(sel) == 1 else None, P,
                                                   knees, len(sel) == 1, "n")),
                ("Risk matrix (selection against the fleet)", fig_risk_matrix(fs, P, sel[0],
                                                                              list(sel) if len(sel) < len(fs) else None)),
            ]
            for c in sel[:8]:
                if c in fs.index:
                    secs.append((f"Instrument cluster · {c}", fig_gauges(fs.loc[c], P, te.SafetyLimits(**lim))))
                    secs.append((f"Events · {c}", ev[ev["Cell_ID"] == c]))
            saved = st.session_state.get("cmp")
            if saved and saved.get("res") is not None and saved["res"].cell_id in sel:
                try:
                    secs.append((f"Forecasts · {saved['res'].cell_id}", fig_compare(saved["res"], P)))
                    secs.append(("Forecast metrics", te.metrics_table(saved["res"].metrics)))
                except Exception:
                    pass
            st.session_state["report"] = (tuple(sel), build_report_html(f"Battery health report · {label}", secs))
    rep = st.session_state.get("report")
    if rep and tuple(rep[0]) == tuple(sel):
        download("Download report (HTML)", rep[1], f"battery_report_{label.replace(' ', '_')}.html", "text/html",
                 key="rep_dl")


def status_section(scope: str, sel: List[str], fs: pd.DataFrame, ev: pd.DataFrame, lim: Dict[str, float]) -> None:
    """Health status for the current scope: instrument cards (single / selected) or triage (fleet)."""
    if fs.empty:
        st.info("Not enough valid cycles to assess health.")
        return
    if scope == "fleet":
        k = st.columns(6)
        k[0].metric("Batteries", len(fs))
        k[1].metric("Mean SOH", fmt(100 * fs["SOH"].mean(), ".1f", "%"))
        k[2].metric("Past end of life", int((fs["SOH"] <= fs["SOH_EOL"]).sum()))
        k[3].metric("Critical / warning", f"{int((fs['Risk'] == 'Critical').sum())} / "
                                          f"{int((fs['Risk'] == 'Warning').sum())}")
        k[4].metric("Knees detected", int(fs["Knee"].sum()))
        k[5].metric("Fleet cycles logged", f"{int(fs['Cycles'].sum()):,}")
        show(fig_fleet_map(fs, P), key="fleet_map", data=fs.reset_index())
        show(fig_risk_matrix(fs, P, cell), key="risk_matrix", data=fs.reset_index())
        triage_table(fs, "triage_all")
        note("Every battery ranked worst-first: health margin bars, quick remaining life, fade speed and the alerts behind the risk level.")
    else:
        for c in sel:
            battery_status_card(c, fs, lim, c)
        if len(sel) > 1:
            triage_table(fs[fs.index.isin(sel)], "triage_sel")
            note("Side-by-side status of the selected batteries: which one is closest to end of life and why.")
        show(fig_risk_matrix(fs, P, sel[0], sel), key="risk_matrix", data=fs.reset_index())
        st.caption("Selected batteries in colour, the rest of the fleet in grey. Quick RUL is a robust trend of the "
                   "last 20 cycles for triage; the Models view gives the full probabilistic RUL.")
    with st.expander("Event log", expanded=scope != "fleet", icon=":material/list_alt:"):
        event_log(ev, sel, "ev")
        note("Chronological log of notable events (end of life, knees, over-temperature, cold charging, regeneration): when each battery's trouble started.")
    with st.expander("Report", icon=":material/description:"):
        st.caption("Self-contained HTML with interactive charts for the current selection (print to PDF from the "
                   "browser).")
        selection_report(sel if scope != "fleet" else list(fs.index[:8]), fs, ev, lim)


def _best_key(tab: pd.DataFrame, col: str, lowest: bool = False) -> Optional[str]:
    """Index of the largest (or lowest) finite value in ``col``; None when empty or all NaN."""
    if tab is None or not len(tab) or col not in tab:
        return None
    v = pd.to_numeric(tab[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if not len(v):
        return None
    return v.idxmin() if lowest else v.idxmax()


def live_accuracy_cards(sk: pd.DataFrame, fr: Any, show_models: Sequence[str], reveal: bool,
                        eol_true: Optional[int]) -> None:
    models = [m for m in show_models if fr is not None and m in fr.forecasts]
    if not models:
        return
    st.markdown("**Live accuracy** · 5-cycle-ahead predictions scored so far (causal)")
    by = sk.set_index("key") if len(sk) else pd.DataFrame()
    best = _best_key(by, "Accuracy (%)")
    cols = st.columns(len(models))
    for c, m in zip(cols, models):
        name = te.LIVE_MODELS[m].split(" · ")[0]
        if m in by.index and np.isfinite(by.loc[m, "Accuracy (%)"]):
            c.metric(("★ " if m == best else "") + name, f"{by.loc[m, 'Accuracy (%)']:.2f}%",
                     delta=f"RMSE {by.loc[m, 'RMSE (5-step ahead)']:.4f}", delta_color="off",
                     help=f"{int(by.loc[m, 'Predictions scored'])} predictions scored; bias "
                          f"{by.loc[m, 'Bias']:+.4f} (positive = optimistic).")
        else:
            c.metric(name, "—", help="Scored once the first 5-cycle-ahead prediction can be checked.")
    if reveal:
        good = ct_cell[~ct_cell["outlier"]]
        hind = te.live_forecast_accuracy(fr, good, soh_eol, eol_true)
        if len(hind):
            hb = hind.set_index("key")
            best_h = _best_key(hb, "Accuracy (%)")
            st.markdown(f"**Forecast accuracy against the actual future** · forecasts made at n = {fr.n}, scored on "
                        f"the {int(hb['Cycles scored'].max())} cycles that followed (hindsight)")
            cols = st.columns(len(models))
            for c, m in zip(cols, models):
                if m in hb.index and np.isfinite(hb.loc[m, "Accuracy (%)"]):
                    r = hb.loc[m]
                    eol_txt = (f"EOL error {r['EOL error (cycles)']:+.0f} cyc" if np.isfinite(r["EOL error (cycles)"])
                               else f"coverage {100 * r['Band coverage']:.0f}%")
                    c.metric(("★ " if m == best_h else "") + te.LIVE_MODELS[m].split(" · ")[0],
                             f"{r['Accuracy (%)']:.2f}%", delta=eol_txt, delta_color="off",
                             help=f"RMSE {r['RMSE']:.4f}; band coverage {100 * r['Band coverage']:.0f}%.")


def _study_answers(res: Dict[str, Any]) -> Dict[str, List[str]]:
    """Plain-language answers to the Mission questions, generated from the evidence."""
    A: Dict[str, List[str]] = {"M1": [], "M2": [], "M3": []}
    hi = res.get("hi")
    if hi is not None and len(hi):
        top = hi.iloc[0]
        A["M1"].append(f"Best health indicator: {top['Indicator']} (fitness {top['Fitness']:.2f}); best operando "
                       f"power-fade indicator: {hi[~hi.index.isin(['Capacity_Ah', 'E_dis_Wh', 't_dis_s', 'Q_ch_Ah', 't_cc_s'])].iloc[0]['Indicator'] if len(hi) > 1 else '—'}.")
    pca = res.get("pca") or {}
    if pca.get("available"):
        ev = pca["explained"]
        A["M1"].append(f"One parameter or several? PC1 explains {100 * ev[0]:.0f}% of indicator variance and PC2 "
                       f"{100 * ev[1]:.0f}%: " + ("a single health parameter suffices." if pca["single_parameter"] else
                                                  "ageing is multi-dimensional (capacity and power fade diverge), "
                                                  "so the twin tracks SOH and two resistances."))
    sf = res.get("sf") or {}
    if sf.get("available"):
        co = sf["coefficients"]
        parts = [f"{f}: {r['Estimate']:.3g} [{r['90% CI low']:.3g}, {r['90% CI high']:.3g}]" for f, r in co.iterrows()]
        A["M1"].append(f"Operating conditions ({sf['n_cells']} cells, R² {sf['r2']:.2f}): " + "; ".join(parts) + ".")
    mc = res.get("mech_checks")
    if mc is not None and len(mc):
        A["M1"].append("Mechanisms: " + "; ".join(f"{k.lower()} → {r['Verdict']}" for k, r in mc.iterrows()) + ".")
    summ = res.get("summary")
    if summ is not None and len(summ):
        allb = summ.loc["All batteries"] if "All batteries" in summ.index.get_level_values(0) else None
        if allb is not None:
            best = allb["median RMSE"].idxmin()
            A["M2"].append(f"Across all batteries the most accurate 20-cycle forecaster is {te.LIVE_MODELS[best]} "
                           f"(median RMSE {allb.loc[best, 'median RMSE']:.4f}, accuracy "
                           f"{allb.loc[best, 'median accuracy (%)']:.2f}%, band coverage {allb.loc[best, 'coverage']:.0%}).")
        for grp in [g for g in te.GROUP_ORDER if g in summ.index.get_level_values(0)]:
            gs = summ.loc[grp]
            b = gs["median RMSE"].idxmin()
            A["M2"].append(f"{grp}: best {te.LIVE_MODELS[b]} ({gs.loc[b, 'median RMSE']:.4f}; twin "
                           f"{gs['median RMSE'].get('twin', float('nan')):.4f}).")
    pt = res.get("paired")
    if pt is not None and len(pt) and "All batteries" in pt.index:
        r = pt.loc["All batteries"]
        A["M2"].append(f"Live ensemble vs twin (paired): better in {r['ens better in']} forecasts, median difference "
                       f"{r['median difference']:+.4f}, Wilcoxon p = {r['Wilcoxon p']:.3g}.")
    up = res.get("update")
    if up is not None and len(up):
        rec = up["recommended"].dropna()
        if len(rec):
            A["M2"].append(f"Update interval: the sparsest schedule within tolerance is every {int(rec.median())} "
                           f"cycle(s) (median over {len(rec)} batteries; range {int(rec.min())}–{int(rec.max())}).")
    om = st.session_state.get("om")
    dp = st.session_state.get("dp")
    if dp:
        A["M3"].append(f"Dynamic programming on the calibrated plant: long-run profit rate {dp['eval']['rate']:.4f} CU/h, "
                       f"replacement at SOH {dp['eval']['threshold']:.3f}, P(sudden failure) "
                       f"{100 * dp['eval']['p_failure']:.1f}%.")
    if om:
        b = te.integrated_optimum(om["study"]).get("best")
        if b:
            A["M3"].append(f"Best grid policy: {b['policy']}, replace at SOH {b['threshold']:.2f} → {b['rate']:.4f} CU/h.")
    if not A["M3"]:
        A["M3"].append("Run the integrated optimisation and the DP in Operations & control (with the calibrated "
                       "plant) to fill in this answer.")
    return A


def view_study() -> None:
    recommendations("study")
    section("Study results: answers to the Mission questions")
    st.markdown("Runs the live multi-model twin on **every usable battery** and scores each model's forecasts "
                "(made at 30% and 50% of recorded life, over the next 20 cycles) per operating-condition group, "
                "with paired statistical tests, mechanism-physics checks and a cohort update-interval study. "
                "Results use the NASA data loaded in this session; nothing here is synthetic unless the demo "
                "cohort is loaded.")
    groups = te.condition_groups(ct)
    with st.expander(f"Evaluation groups · {groups['Group'].nunique()} groups, {len(groups)} batteries",
                     icon=":material/category:"):
        show_table(groups[["Group", "Ambient_C", "I_dis_A", "rest_frac", "cycles"]].sort_values("Group")
                   .style.format({"Ambient_C": "{:.0f}", "I_dis_A": "{:.1f}", "rest_frac": "{:.2f}"}))
        st.caption("Pulsed-load cells (square-wave discharge) use a pulse-aware twin (no partial-window voltage "
                   "model). Corrupted-logging cells are reported separately so they do not distort the others.")
    c1, c2 = st.columns(2)
    n_est = int((groups["cycles"] >= 25).sum())
    run = c1.button(f"Run cohort study (≈ {3 * n_est + 20} s)", key="study_go", type="primary",
                    icon=":material/science:")
    with_update = c2.toggle("Include cohort update-interval study", value=True, key="study_upd")
    if run:
        prog = st.progress(0.0, text="Starting…")
        cb = lambda f, m: prog.progress(min(max(float(f), 0.0), 1.0), text=m)
        try:
            val, mech = te.cohort_validation(store, ct, imp, float(eol_ah), progress=cb)
            res = {"val": val, "mech": mech, "summary": te.cohort_summary(val),
                   "paired": te.paired_model_test(val, "ens", "twin") if len(val) else pd.DataFrame(),
                   "mech_checks": te.mechanism_checks(mech), "hi": hi_rank_cached(ct, imp, DATA_KEY),
                   "pca": hi_pca_cached(ct, imp, DATA_KEY), "sf": stress_factors_cached(ct, DATA_KEY)}
            if with_update:
                prog.progress(0.95, text="Update-interval study on representative batteries…")
                reps = tuple(groups.reset_index().sort_values("cycles", ascending=False).groupby("Group").head(2)["Cell_ID"])
                res["update"] = update_cohort_cached(store, ct, imp, DATA_KEY, float(eol_ah), reps)
            st.session_state["study"] = {"key": DATA_KEY, "res": res}
        except Exception as exc:
            report_error("Cohort study failed", exc, debug)
        prog.empty()
    saved = st.session_state.get("study")
    if not saved or saved["key"] != DATA_KEY:
        st.info("Press **Run cohort study** to generate the evidence. It runs once per dataset and is kept for the "
                "session.")
        return
    res = saved["res"]
    A = _study_answers(res)
    section("Mission 1 · Health, operating conditions and mechanisms")
    card("Answers", A["M1"] or ["—"])
    if res.get("hi") is not None and len(res["hi"]):
        show(fig_hi_rank(res["hi"], P), key="st_hi", data=res["hi"].reset_index())
    if len(res["mech"]):
        show(fig_mech_groups(res["mech"], P), key="st_mech", data=res["mech"])
    if res.get("mech_checks") is not None and len(res["mech_checks"]):
        show_table(res["mech_checks"].style.format({"median share (group)": "{:.0%}", "median share (reference)": "{:.0%}",
                                                    "Mann-Whitney p": "{:.3g}"}, na_rep="—"))
        st.caption("The attribution is model-based. Where the half-cell fit is available, compare plating + SEI with "
                   "the fitted LLI and the LAM share with the fitted LAM in Diagnostics.")
    section("Mission 2 · Prediction accuracy, informative variables and update frequency")
    card("Answers", A["M2"] or ["—"])
    if len(res["summary"]):
        show(fig_cohort_models(res["summary"], P), key="st_models", data=res["summary"].reset_index())
        show_table(res["summary"].style.format({"median RMSE": "{:.4f}", "IQR low": "{:.4f}", "IQR high": "{:.4f}",
                                                "median accuracy (%)": "{:.2f}", "coverage": "{:.0%}",
                                                "median |RUL error|": "{:.0f}"}, na_rep="—"))
    if res.get("paired") is not None and len(res["paired"]):
        st.markdown("**Is the live ensemble reliably better than the twin?** (paired Wilcoxon test on forecast RMSE)")
        show_table(res["paired"].style.format({c: "{:.4f}" for c in res["paired"].columns if "RMSE" in c or "difference" in c}
                                              | {"Wilcoxon p": "{:.3g}"}, na_rep="—"))
    if res.get("update") is not None and len(res["update"]):
        st.markdown("**How often should the twin update?** (tracking RMSE when updating every 1, 5 or 20 cycles)")
        show_table(res["update"].set_index("Cell_ID").style.format("{:.4f}", subset=[c for c in res["update"].columns
                                                                                   if c.startswith("RMSE")]))
    section("Mission 3 · Integrated operation and maintenance")
    card("Answers", A["M3"])
    try:
        _, tab = calibrated_plant_cached(ct, imp, DATA_KEY, cell)
        st.markdown(f"**Plant calibrated on {cell}** (used by Operations & control when calibration is on)")
        show_table(tab.style.format({"Value": "{:.4g}"}))
    except Exception:
        pass
    section("Export")
    val_csv = res["val"].to_csv(index=False).encode()
    download("Download all forecast scores (CSV)", val_csv, "cohort_validation.csv", "text/csv", key="st_csv")
    if st.button("Build study report (HTML)", key="st_rep", icon=":material/description:"):
        secs: List[Tuple[str, Any]] = [("Mission 1 answers", " | ".join(A["M1"]))]
        if res.get("hi") is not None and len(res["hi"]):
            secs.append(("Health-indicator ranking", fig_hi_rank(res["hi"], P)))
        if len(res["mech"]):
            secs += [("Mechanism shares by condition", fig_mech_groups(res["mech"], P)),
                     ("Mechanism physics checks", res["mech_checks"])]
        secs += [("Mission 2 answers", " | ".join(A["M2"]))]
        if len(res["summary"]):
            secs += [("Forecast error by condition and model", fig_cohort_models(res["summary"], P)),
                     ("Cohort summary", res["summary"].reset_index())]
        if res.get("paired") is not None and len(res["paired"]):
            secs.append(("Live ensemble vs twin (paired test)", res["paired"].reset_index()))
        if res.get("update") is not None and len(res["update"]):
            secs.append(("Update-interval study", res["update"]))
        secs += [("Mission 3 answers", " | ".join(A["M3"])), ("Evaluation groups", groups.reset_index())]
        st.session_state["study_report"] = build_report_html("Self-updating digital twin · study results", secs)
    if st.session_state.get("study_report"):
        download("Download study report (HTML)", st.session_state["study_report"], "study_results.html", "text/html",
                 key="st_rep_dl")


def view_replay() -> None:
    recommendations("live")
    section(f"Live twin replay · {cell}")
    st.markdown("The recorded life of the battery is streamed one discharge at a time. At every step the dual "
                "time-scale EKF assimilates only the partial-window voltage and load-step resistance of the new "
                "cycle, updates SOH, resistances and the personal degradation rate *k*, and re-forecasts the "
                "remaining life. Nothing after the cursor is visible to the twin.")
    level = 0.9
    cap_every = st.select_slider("Reference capacity check every N cycles (0 = operando only)", [0, 5, 10, 20, 50],
                                 value=10, key="rp_cap",
                                 help="Between checks the twin sees only partial-window voltage and load-step "
                                      "resistance. Set 0 to watch pure operando tracking, including its drift.")
    show_models = st.multiselect(
        "Live models", list(te.LIVE_MODELS),
        default=[m for m in ("twin", "mech", "pf", "trend", "ens") if m in te.LIVE_MODELS], key="rp_models",
        format_func=te.LIVE_MODELS.get,
        help="ECM twin: physics + operando voltage. Particle filter: power law with the fleet prior, robust to knees. "
             "Adaptive trend KF: level-slope-curvature filter that bends quickly when fade accelerates. Hierarchical "
             "Bayes: fleet-informed power law. Mechanistic PF: SEI, lithium-plating and LAM kinetics driven by each "
             "cycle's measured temperature and current, so the operating conditions decide which mechanism grows. "
             "Live ensemble: weights every model by its recent 5-cycle-ahead error.")
    with st.spinner("Streaming the battery through all models (runs once, then animates)…"):
        res, _track_old = replay_cached(cell_frame(store, DATA_KEY, cell), ct, imp, DATA_KEY, cell, float(soh_eol),
                                        level, int(cap_every))
        st.session_state["ekf"] = {"key": (DATA_KEY, cell, "live"), "res": res}     # feeds Operations initialisation
        run_models = ("twin", "mech", "pf", "trend", "hb")
        frames, track = live_cached(res, ct, imp, DATA_KEY, cell, float(soh_eol), level, int(cap_every), run_models)
    pc = res.per_cycle
    ns = pc["n"].astype(int).tolist()
    if len(ns) < 6:
        st.info("This battery has too few cycles for a replay.")
        return
    key_n, key_p = f"rp_n_{cell}", "rp_play"
    st.session_state.setdefault(key_n, ns[min(5, len(ns) - 1)])
    st.session_state.setdefault(key_p, False)
    c = st.columns([1, 1, 1, 1, 1.4, 1.4, 1.4])
    playing = bool(st.session_state[key_p])
    if c[0].button("Pause" if playing else "Play", key="rp_toggle", type="primary",
                   icon=":material/pause:" if playing else ":material/play_arrow:"):
        st.session_state[key_p] = not playing
        if not playing and st.session_state[key_n] >= ns[-1]:
            st.session_state[key_n] = ns[5]
        st.rerun()
    if c[1].button("Step", key="rp_step", icon=":material/skip_next:"):
        st.session_state[key_n] = min(ns[-1], st.session_state[key_n] + 1)
    if c[2].button("Back", key="rp_back", icon=":material/skip_previous:"):
        st.session_state[key_n] = max(ns[0], st.session_state[key_n] - 1)
    if c[3].button("Reset", key="rp_reset", icon=":material/replay:"):
        st.session_state[key_n] = ns[min(5, len(ns) - 1)]
        st.session_state[key_p] = False
    speed = c[4].select_slider("Cycles per frame", [1, 2, 3, 5, 10], value=2, key="rp_speed")
    interval = c[5].select_slider("Frame interval (s)", [0.4, 0.7, 1.0, 1.5], value=0.7, key="rp_int")
    reveal = c[6].toggle("Reveal future data", value=False, key="rp_reveal",
                         help="Show the cycles the twin has not seen yet, to judge the forecast.")
    good = ct_cell[~ct_cell["outlier"]]
    eol_true = te.first_crossing(good["n"].to_numpy(), good["SOH"].to_numpy(), soh_eol, smooth=5)
    n_max = int(max(pc["n"].max() * 1.35, (eol_true or 0) * 1.1))
    frag = getattr(st, "fragment", None)

    def frame() -> None:
        if st.session_state.get(key_p):
            nxt = st.session_state[key_n] + int(speed)
            if nxt >= ns[-1]:
                nxt, st.session_state[key_p] = ns[-1], False
            st.session_state[key_n] = nxt
        n = int(st.session_state[key_n])
        n = min(ns, key=lambda v: abs(v - n))
        row = pc[pc["n"] == n].iloc[0]
        fr = frames.get(n) or (frames[max(k_ for k_ in frames if k_ <= n)] if any(k_ <= n for k_ in frames) else None)
        meas = good[good["n"] <= n]["SOH"].tail(1)
        ens_rul = fr.rul.get("ens") if fr is not None else None
        lead = max(fr.weights, key=fr.weights.get) if fr is not None and fr.weights else None
        k = st.columns(6)
        k[0].metric("Cycle", f"{n} / {ns[-1]}")
        k[1].metric("Twin SOH", f"{row['SOH']:.3f}", delta=f"±{2 * row['SOH_std']:.3f} (2σ)", delta_color="off")
        k[2].metric("Measured SOH", f"{float(meas.iloc[0]):.3f}" if len(meas) else "—")
        k[3].metric("Ensemble RUL", f"{ens_rul[0]:.0f} cyc" if ens_rul and np.isfinite(ens_rul[0]) else "beyond horizon",
                    delta=(f"90%: {ens_rul[1]:.0f}–{ens_rul[2]:.0f}" if ens_rul and np.isfinite(ens_rul[1])
                           and np.isfinite(ens_rul[2]) else None), delta_color="off")
        k[4].metric("Most trusted now", te.LIVE_MODELS[lead].split(" · ")[0] if lead else "—",
                    delta=f"weight {fr.weights[lead]:.2f}" if lead else None, delta_color="off")
        sh = fr.mech_shares if fr is not None else None
        if sh:
            dom = max(sh, key=sh.get)
            k[5].metric("Dominant mechanism", dom, delta=f"{100 * sh[dom]:.0f}% of the loss so far", delta_color="off",
                        help="Mechanistic PF: share of the capacity lost so far owed to SEI growth, lithium plating "
                             "and loss of active material, driven by the measured temperature and current.")
        else:
            k[5].metric("Personal k / prior", f"{row['k_ah'] / res.params.get('k_prior', row['k_ah']):.2f}×",
                        help="The twin's own degradation rate relative to the fleet prior (Sage–Husa adaptive).")
        st.progress(min(1.0, (n - ns[0]) / max(ns[-1] - ns[0], 1)),
                    text=f"{'▶ streaming' if st.session_state.get(key_p) else '⏸ paused'} · "
                         f"{len(good[good['n'] <= n])} discharges assimilated")
        sk = te.live_skill_table(track[track["n"] <= n])
        acc_map = dict(zip(sk["key"], sk["Accuracy (%)"])) if len(sk) else {}
        show(fig_live_main(ct_cell, pc, fr, n, soh_eol, P, reveal, n_max, show_models, acc_map), key="rp_main",
             export=False)
        live_accuracy_cards(sk, fr, show_models, reveal, eol_true)
        show(fig_live_track(track, n, eol_true, P, n_max, show_models), key="rp_track", export=False)
        if len(sk):
            with st.expander("Live model scoreboard", icon=":material/leaderboard:"):
                show_table(sk.drop(columns=["key"]).set_index("Model").style.format(
                    {"Accuracy (%)": "{:.2f}", "RMSE (5-step ahead)": "{:.4f}", "MAE": "{:.4f}", "Bias": "{:+.4f}",
                     "Final weight": "{:.2f}"}, na_rep="—")
                    .highlight_max(subset=["Accuracy (%)"], props="background-color: rgba(0,158,115,0.25); font-weight: 700;"))
                st.caption("Scored causally: every prediction was made 5 cycles before the measurement it is "
                           "compared with, using only data available at that time. Accuracy = 100 × (1 − mean "
                           "absolute percentage error).")

    if frag is not None:
        try:
            frag(run_every=f"{interval}s" if st.session_state.get(key_p) else None)(frame)()
        except TypeError:
            frame()
    else:
        frame()
    with st.expander("Mechanistic model: governing equations", icon=":material/functions:"):
        st.markdown("The mechanistic particle filter integrates the degradation kinetics cycle by cycle with the "
                    "**measured** cell temperature T and current I, so the operating conditions decide which mechanism "
                    "grows: Arrhenius acceleration of SEI growth when hot, the cold gate of lithium plating below "
                    "~10 °C, the C-rate power of LAM under high current. Rate constants and latent losses are "
                    "estimated jointly by a particle filter from the measured capacity.")
        st.latex(r"\mathrm{SOH} = s_0 - Q_\mathrm{SEI} - Q_\mathrm{pl} - Q_\mathrm{LAM}")
        st.latex(r"\Delta Q_\mathrm{SEI} = k_\mathrm{SEI}\,e^{\frac{E_\mathrm{SEI}}{R}\left(\frac{1}{T_\mathrm{ref}}"
                 r"-\frac{1}{T}\right)}\,\frac{2\,\mathrm{SOH}}{1 + Q_\mathrm{SEI}/\delta}")
        st.latex(r"\Delta Q_\mathrm{pl} = k_\mathrm{pl}\,e^{\frac{E_\mathrm{pl}}{R}\left(\frac{1}{T}-\frac{1}"
                 r"{T_\mathrm{ref}}\right)}\,\frac{I_\mathrm{ch}}{C_0}\left[\frac{1}{1+e^{(T-T_\mathrm{onset})/3}}"
                 r" + \kappa\,\frac{Q_\mathrm{LAM}}{0.05}\right]")
        st.latex(r"\Delta Q_\mathrm{LAM} = k_\mathrm{LAM}\,e^{\frac{E_\mathrm{LAM}}{R}\left(\frac{1}{T_\mathrm{ref}}"
                 r"-\frac{1}{T}\right)}\left(\frac{I}{C_0}\right)^{\beta} 2\,\mathrm{SOH}\left(1 + "
                 r"\frac{Q_\mathrm{LAM}}{\varepsilon}\right)")
        st.caption("Priors: E_SEI = 30, E_pl = 50, E_LAM = 20 kJ/mol; rate constants log-normal around literature-scale "
                   "values. The attributed shares are model-based: validate them against the half-cell LLI / LAM in "
                   "Diagnostics (plating and SEI are LLI, LAM is LAM).")
    pinn_equations_panel()
    methods_panel()
    ablation_section()
    update_frequency_section()
    st.caption("Harsh operation (cold plating, knees, high temperature) breaks any single physics law. The live "
               "ensemble moves its weight, cycle by cycle, to the models that are currently predicting well: watch "
               "the weight panel shift when the fade accelerates. "
               "The EOL band narrowing and the personal rate k settling are the self-updating behaviour: the twin "
               "starts from the population prior and converges on this battery's own ageing law as evidence "
               "accumulates. Toggle 'Reveal future data' to judge each forecast against what actually happened.")


@fragment
def half_cell_section() -> None:
    st.markdown("The ICA/DVA proxies above become quantitative here. The pseudo-OCV (discharge voltage + |I|·R_dc) is "
                "fitted with literature half-cell potentials. Four parameters describe the electrode balance: positive "
                "and negative capacities C_p and C_n, and their stoichiometries y₀ (LiᵧCoO₂) and x₀ (LiₓC₆) at the top of "
                "charge. A fifth, η, absorbs the residual polarisation. Their evolution separates loss of lithium "
                "inventory from loss of active material on each electrode.")
    with st.expander("Model equations", icon=":material/functions:"):
        st.latex(r"V(Q) = U_\mathrm{p}\!\left(y_0 + \tfrac{Q}{C_\mathrm{p}}\right) - "
                 r"U_\mathrm{n}\!\left(x_0 - \tfrac{Q}{C_\mathrm{n}}\right) - \eta")
        st.latex(r"\mathrm{LAM_{PE}} = 1 - \frac{C_\mathrm{p}}{C_\mathrm{p,0}},\quad "
                 r"\mathrm{LAM_{NE}} = 1 - \frac{C_\mathrm{n}}{C_\mathrm{n,0}},\quad "
                 r"\mathrm{LLI} = 1 - \frac{x_0 C_\mathrm{n} + y_0 C_\mathrm{p}}{(x_0 C_\mathrm{n} + y_0 C_\mathrm{p})_0}")
        st.caption("U_p: LiCoO₂ (Ramadass et al., J. Electrochem. Soc. 151, 2004). U_n: graphite (Doyle et al., "
                   "J. Electrochem. Soc. 143, 1996). Global search (differential evolution) for the first curve, then "
                   "warm-started local fits (Dubarry et al. 2012; Birkl et al. 2017). On synthetic truth the method "
                   "recovers LLI and LAM_PE within about 1 percentage point.")
    c1, c2 = st.columns(2)
    n_curves = c1.slider("Curves across life", 3, 12, 6, key="hc_n")
    ir = c2.toggle("IR-compensate", value=True, key="hc_ir")
    if st.button("Fit half-cell model", key="hc_go", icon=":material/science:", type="primary"):
        with st.spinner("Fitting electrode balance (global search on the first curve)…"):
            try:
                st.session_state["hc"] = (cell, half_cell_cached(prepared_cell(store, DATA_KEY, cell), ct_cell, DATA_KEY,
                                                                 cell, n_curves, ir))
            except Exception as exc:
                report_error("Half-cell fit failed", exc, debug)
    saved = st.session_state.get("hc")
    if not saved or saved[0] != cell:
        return
    tab, fits = saved[1]
    if not fits:
        st.warning("No curve could be fitted for this battery.")
        return
    last = tab.iloc[-1]
    k = st.columns(4)
    k[0].metric("Loss of lithium inventory", fmt(last["LLI (%)"], ".1f", "%"))
    k[1].metric("LAM positive (LiCoO₂)", fmt(last["LAM_PE (%)"], ".1f", "%"))
    k[2].metric("LAM negative (graphite)", fmt(last["LAM_NE (%)"], ".1f", "%"))
    k[3].metric("Median fit RMSE", fmt(tab["Fit RMSE (mV)"].median(), ".1f", "mV"))
    show(fig_half_cell_modes(tab, P), key="hc_modes", data=tab)
    show(fig_half_cell_fits([fits[0], fits[len(fits) // 2], fits[-1]] if len(fits) > 2 else fits, P), key="hc_fits",
         export=False)
    show_table(tab.set_index("n").style.format("{:.3f}"))
    dom = max(("LLI (%)", "LAM_PE (%)", "LAM_NE (%)"), key=lambda c_: last[c_])
    card("Reading", [f"Dominant mode at n = {int(last['n'])}: {dom.split(' ')[0]}. LLI points to SEI growth or "
                     "plating; LAM_PE to cathode cracking or cobalt dissolution; LAM_NE to graphite exfoliation or "
                     "particle isolation.",
                     "At ~1C the pseudo-OCV still carries kinetic and diffusion overpotentials. Residuals above "
                     "~15 mV or η drifting strongly mean the absolute split is uncertain: compare trends between cells "
                     "rather than trusting single values."])


def _reco_items(view: str) -> List[Tuple[str, str]]:
    ss = st.session_state
    items: List[Tuple[str, str]] = []
    try:
        fs, _ = fleet_cached(ct, DATA_KEY, float(eol_ah), asdict(te.SafetyLimits()))
        n_crit = int((fs["Risk"] == "Critical").sum()) if len(fs) else 0
    except Exception:
        n_crit = 0
    grp = te.condition_groups(ct)["Group"].get(cell, "")
    if view == "diag":
        if n_crit and ss.get("diag_scope", "").endswith("Single battery"):
            items.append(("priority_high", f"{n_crit} batteries are critical: switch the scope to *Selected batteries* "
                                           "and press **Select most critical**."))
        if not ss.get("hc") or ss["hc"][0] != cell:
            items.append(("science", "Quantify the degradation modes of this battery: run **half-cell OCV fitting** below."))
        if grp in ("Reference", ""):
            items.append(("thermostat", "Compare this battery with a **cold (4 °C)** one in the cohort plot: plating shows as fast early fade."))
        items.append(("play_circle", "Open the **Live twin** to watch the models learn this battery cycle by cycle."))
    elif view == "live":
        items.append(("visibility", "Switch on **Reveal future data** and pause at 30–50% of life to judge each forecast."))
        if grp in ("Cold", "Hot", "High current", "Mixed conditions"):
            items.append(("warning", f"This is a **{grp.lower()}** battery: watch the ensemble weights move away from the ECM twin."))
        else:
            items.append(("swap_horiz", "Pick a cold or hot battery in the control bar to see the mechanistic model take over."))
    elif view == "models":
        done = {e["level"] for e in ladder_entries()}
        nxt = next((lv for lv in LEVEL_INFO if lv not in done and lv not in (2, 3)), None)
        if not done:
            items.append(("flag", "Start with **Level 1 · Baselines**: every other model must beat the linear trend."))
        elif nxt:
            items.append(("trending_up", f"Next: **Level {nxt} · {LEVEL_INFO[nxt][0]}**, then compare in the leaderboard."))
        if not ss.get("ml"):
            items.append(("model_training", "In the ML workbench, press **Auto-tune** before judging the tree and boosting models."))
        items.append(("fact_check", "One battery is an anecdote: confirm the ranking in the **cross-cell benchmark** or the Study results."))
    elif view == "ops":
        if not ss.get("ops_calib", True):
            items.append(("tune", "Turn on **Calibrate the plant** so the optimisation plans for this real battery."))
        if not ss.get("om"):
            items.append(("insights", "Run the **integrated optimisation**, then **Solve DP** to see how close the simple policy is to optimal."))
        items.append(("science", "Use the **what-if planner** to test a hotter or colder site before deploying."))
    elif view == "study":
        if not ss.get("study"):
            items.append(("science", "Press **Run cohort study**: it produces the evidence for your results chapter."))
        else:
            items.append(("description", "Export the **study report** and check groups with few cells (labelled underpowered)."))
    return items[:3]


def recommendations(view: str) -> None:
    items = _reco_items(view)
    if not items:
        return
    cards = "".join(f'<div class="bt-reco-item"><span class="bt-reco-dot"></span>{_md_bold(t)}</div>' for _, t in items)
    st.markdown(f'<div class="bt-reco"><div class="bt-reco-h">Recommended next steps</div>{cards}</div>',
                unsafe_allow_html=True)


def _md_bold(text: str) -> str:
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"\*(.+?)\*", r"<i>\1</i>", t)


def sidebar_guide() -> None:
    ss = st.session_state
    steps = [("Data loaded", True), ("Half-cell modes fitted", bool(ss.get("hc"))),
             ("Live twin opened", bool(ss.get("ekf"))), ("Ladder models compared", bool(ss.get("ladder"))),
             ("M3 optimisation run", bool(ss.get("om") or ss.get("dp"))), ("Cohort study run", bool(ss.get("study")))]
    done = sum(ok for _, ok in steps)
    rows = "".join(f'<div class="bt-guide-row{" ok" if ok else ""}"><span>{"✓" if ok else "○"}</span>{html.escape(t)}</div>'
                   for t, ok in steps)
    with st.sidebar:
        st.markdown(f'<div class="bt-guide"><div class="bt-guide-h">Study progress · {done}/{len(steps)}</div>'
                    f'<div class="bt-guide-bar"><div style="width:{100 * done / len(steps):.0f}%"></div></div>{rows}</div>',
                    unsafe_allow_html=True)


def view_data() -> None:
    global cell, ct_cell, eis_cell, c_bol, soh_eol
    recommendations("diag")
    lim = asdict(te.SafetyLimits())
    fs, ev = fleet_cached(ct, DATA_KEY, float(eol_ah), lim)
    all_cells = list(meta.index)
    labels = [DIAG_SCOPE_LABEL[k] for k in DIAG_SCOPES]
    st.session_state.setdefault("diag_scope", labels[0])
    c1, c2 = st.columns([3, 1.2], vertical_alignment="bottom")
    with c1:
        try:
            choice = st.segmented_control("Scope", labels, key="diag_scope", label_visibility="collapsed")
        except Exception:
            choice = st.radio("Scope", labels, key="diag_scope", horizontal=True, label_visibility="collapsed")
    choice = choice or labels[0]
    scope = DIAG_SCOPES[labels.index(choice)]
    c2.button("Select most critical", key="pick_crit", icon=":material/priority_high:", on_click=_pick_critical,
              args=(fs,), help="Switches to 'Selected batteries' with the six highest-risk cells.")
    if scope == "single":
        sel = [cell]
        st.caption(f"Analysing the target battery from the control bar: **{cell_label(cell, meta)}**.")
    elif scope == "selected":
        st.session_state.setdefault("diag_cells", [cell])
        sel = st.multiselect("Batteries", all_cells, key="diag_cells", max_selections=8,
                             format_func=lambda c: cell_label(c, meta),
                             help="Up to 8 batteries; every section below follows this selection.") or [cell]
    else:
        sel = all_cells
    st.session_state["_diag_sel"] = sel

    section("Health status")
    status_section(scope, sel, fs, ev, lim)

    bad = {c: e for c, e in cell_errors.items()}
    with st.expander(f"Data quality · {len(bad)} skipped cell(s)", icon=":material/fact_check:"):
        for c in sel[:8]:
            issues = validation_cached(store, DATA_KEY, c)
            n_out = int(ct[(ct["Cell_ID"] == c)]["outlier"].sum())
            (st.warning if issues else st.success)(
                f"{c}: {len(issues)} telemetry issue(s), {n_out} excluded cycle(s)"
                + ("" if not issues else " · " + "; ".join(issues[:3])))
        if len(sel) > 8:
            st.caption("Per-cell checks are listed for the first eight batteries.")
        if bad:
            show_table(pd.DataFrame({"Cell": list(bad), "Reason": list(bad.values())}).set_index("Cell"))

    section("Capacity fade: compare batteries")
    fade_explorer()

    section("Cohort by ambient temperature")
    norm_x = st.toggle("Normalise x to fraction of recorded life", value=False, key="grid_norm",
                       help="Aligns cells with very different cycle counts.")
    grid_sel = sel if scope != "fleet" else [cell]
    amb = meta["Ambient_C"].round(0)
    groups = sorted(amb.dropna().unique())
    only_sel = scope != "fleet" and st.toggle("Only groups containing the selection", value=False, key="grid_only")
    show_cells = st.multiselect("Batteries to show (empty = all)", list(meta.index), default=[], key="grid_cells",
                                format_func=lambda c: cell_label(c, meta),
                                help="Choose which batteries appear in the temperature panels; panels without any "
                                     "chosen battery are hidden.")
    for g in groups:
        members = list(amb[amb == g].index)
        if show_cells:
            members = [c for c in members if c in show_cells]
            if not members:
                continue
        if only_sel and not set(members) & set(grid_sel):
            continue
        show(fig_cohort_group(ct, meta, members, grid_sel, P, f"Ambient {g:.0f} °C · {len(members)} cells", norm_x),
             key=f"cohort_{int(g)}",
             data=ct.loc[~ct["outlier"] & ct["Cell_ID"].isin(members), ["Cell_ID", "n", "SOH"]])
    with st.expander("Mission 1 · How do operating conditions influence degradation?", expanded=False):
        condition_effects_section()

    if len(sel) > 1:
        st.markdown("---")
        focus = st.selectbox("Focus battery for the detailed analyses below", sel,
                             index=sel.index(cell) if cell in sel else 0, key="diag_focus",
                             format_func=lambda c: cell_label(c, meta),
                             help="Indicator trajectories, ICA / DVA, degradation modes, half-cell fitting and the "
                                  "stress timeline analyse one battery at a time.")
        cell = focus
        ct_cell = ct[ct["Cell_ID"] == cell].sort_values("n")
        eis_cell = te.valid_eis(imp, cell)
        c_bol = float(meta.loc[cell, "C_bol_Ah"])
        soh_eol = te.soh_eol_for(c_bol, eol_ah)

    section("Mission 1 · Which parameter best represents health?")
    health_indicator_section()

    section("Incremental capacity analysis (dQ/dV)")
    ica_section()

    section("Degradation modes: LLI · LAM · conductivity loss")
    modes_section()

    section("Quantitative mode analysis: half-cell OCV fitting")
    half_cell_section()

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
        st.markdown("**Hierarchical Bayes** (partial pooling): every cell has its own power-law parameters, drawn "
                    "from a fleet distribution whose mean depends on the operating conditions. The fleet is the prior "
                    "and the battery's own data update it (Laplace posterior):")
        st.latex(r"\mathrm{SOH} = s_0\left[1 - e^{a}\left(\tfrac{\mathrm{Ah}}{100}\right)^{z}\right],\qquad "
                 r"(a, z)_c \sim \mathcal{N}\!\left(G\,x_c,\ \Sigma\right),\quad x_c = \left[1,\ \tfrac{1}{T_\mathrm{ref}} - "
                 r"\tfrac{1}{T_c},\ \ln\tfrac{I_c}{2\,\mathrm{A}}\right]")
        st.markdown("**Particle filter** on the same physics law with the hierarchical prior (Student-t likelihood, "
                    "systematic resampling); **physics-mean GP**: hierarchical-Bayes trend plus a GP residual in "
                    "throughput; **stacked ensemble** weighted by backtest skill on the battery's own history:")
        st.latex(r"w_k^{(i)} \propto w_{k-1}^{(i)}\,p\!\left(y_k \mid \theta^{(i)}\right),\qquad "
                 r"\mathrm{SOH}(\mathrm{Ah}) = m_\theta(\mathrm{Ah}) + f(\mathrm{Ah}),\ f \sim \mathcal{GP}(0, k_\mathrm{RBF}),"
                 r"\qquad \hat y = \sum_i \frac{\mathrm{RMSE}_i^{-2}}{\sum_j \mathrm{RMSE}_j^{-2}}\,\hat y_i")
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
            "| Bayesian filtering / inference | Dual EKF (states + rate); particle filter and hierarchical "
            "Bayes on the physics power law; physics-mean GP |\n"
            "| Early-life data-driven | ΔQ(V) elastic net (Severson et al. 2019) |\n"
            "| Model combination | Skill-weighted stacked ensemble |")
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
    section("Levels 1–3 · Machine-learning workbench (trees, forests, boosting)", icon=":material/model_training:")
    st.markdown("Two tasks, 8 curated models, every hyperparameter adjustable. **Forecast**: predict future SOH from the past "
                "(prognosis). **Estimate**: infer the present SOH from operando indicators measured on the same cycle "
                "(diagnosis, no capacity test needed).")
    models = st.multiselect("Models (L1 simple → L3 boosting)", sorted(te.ML_MODELS, key=lambda m: te.MODEL_SPECS[m].level),
                            default=["Decision Tree", "Random Forest", "Extra Trees", "Hist. Gradient Boosting"],
                            key="ml_models", format_func=lambda m: f"L{te.MODEL_SPECS[m].level} · {m}",
                            help="Select any number; the leaderboard ranks them.")
    missing_opt = [n for n in te.OPTIONAL_ML if n not in te.ML_MODELS]
    if missing_opt:
        st.caption("Also available once installed on the server: " + ", ".join(missing_opt) +
                   " (add `xgboost` / `lightgbm` to requirements.txt).")
    params = hyperparam_editor(models, "hp")
    t_fc, t_est = st.tabs([":material/trending_down: Forecast future SOH", ":material/biotech: Estimate SOH from indicators"])
    with t_fc:
        ml_forecast_tab(models, params)
    with t_est:
        ml_estimation_tab(models, params)
    with st.expander("How to read R², accuracy and fade skill", icon=":material/help:"):
        st.markdown(R2_NOTE)


def tuning_block(task: str, models: Sequence[str], params: Dict[str, Dict[str, Any]], run: Callable[[str], Any]
                 ) -> Dict[str, Dict[str, Any]]:
    """Auto-tune button + results; returns the parameters to use (tuned where available and enabled)."""
    store_key = f"tuned_{task}"
    tuned: Dict[str, Any] = st.session_state.setdefault(store_key, {})
    c1, c2, c3 = st.columns([1.4, 1.2, 2])
    n_iter = c2.select_slider("Candidates per model", [8, 12, 16, 24, 32], value=12, key=f"{task}_niter")
    if c1.button("Auto-tune hyperparameters", key=f"{task}_tune", icon=":material/auto_fix_high:",
                 disabled=not models, help="Random search validated without touching the test data."):
        prog = st.progress(0.0)
        for i, m in enumerate(models):
            prog.progress(i / len(models), text=f"Tuning {m}…")
            try:
                tuned[m] = run(m, int(n_iter))
            except Exception as exc:
                st.warning(f"{m}: tuning failed ({exc})")
        prog.empty()
    use = c3.toggle("Use tuned parameters", value=bool(tuned), key=f"{task}_use_tuned",
                    disabled=not tuned, help="Off = the values in the hyperparameter panels above.")
    if tuned:
        rows = [{"Model": m, "Validation (default)": r.default_score, "Validation (tuned)": r.best_score,
                 "Improvement (%)": r.improvement_pct, "Tuned settings": ", ".join(f"{k}={v:.3g}" if isinstance(v, float)
                                                                                    else f"{k}={v}"
                                                                                    for k, v in r.best_params.items()),
                 "Time (s)": r.seconds} for m, r in tuned.items() if m in models]
        if rows:
            with st.expander(f"Tuning results · {next(iter(tuned.values())).validation}", expanded=False,
                             icon=":material/insights:"):
                show_table(pd.DataFrame(rows).set_index("Model").style.format(
                    {"Validation (default)": "{:.4f}", "Validation (tuned)": "{:.4f}", "Improvement (%)": "{:+.0f}",
                     "Time (s)": "{:.1f}"}))
                st.caption("Score = mean + 0.5 SD of the validation RMSE (consistency across batteries is "
                           "rewarded). Defaults are always a candidate, so tuned ≤ default on validation. The "
                           "test data are never used, so tuned settings can still be worse on one particular "
                           "battery: judge them on the leaderboard and the cross-cell benchmark.")
    out = dict(params)
    if use:
        out.update({m: r.best_params for m, r in tuned.items() if m in models})
    return out


def ml_forecast_tab(models: Sequence[str], params: Dict[str, Dict[str, Any]]) -> None:
    sc = scheme()
    show_t = sc["show"]
    n0_show = sc["n0"][show_t]
    if sc["mode"] == "within":
        source = "cohort" if sc["use_cohort"] else "own"
        st.markdown(f"**Training scheme:** within {show_t}: cycles 1–{n0_show} for training"
                    + (", plus the other batteries' full histories" if sc["use_cohort"] else " only")
                    + " (change it at the top of this view).")
    else:
        source = "cross"
        st.markdown(f"**Training scheme:** across batteries: learn from {len(sc['train'])} batteries, forecast "
                    f"{len(sc['targets'])} test battery(ies) (change it at the top of this view).")
    train_cells: Optional[Tuple[str, ...]] = tuple(sc["train"]) if source == "cross" else None
    d1, d2, d3 = st.columns(3)
    strategy = "increment"
    d1.markdown("**Fade-rate model**  \n*dSOH/dn = g(SOH, conditions, early-life features)*, integrated forward")
    conf = d2.slider("Conformal calibration cells (0 = no band)", 0, 8, 3, key="ml_conf")
    level = d3.select_slider("Band level", [0.8, 0.9, 0.95], value=0.9, key="ml_level")
    use_pop = source == "cohort"
    params = tuning_block(f"fc_{show_t}_{n0_show}_{source}", models, params,
                          lambda m, k: te.tune_ml_forecast(ct, show_t, n0_show, m, n_iter=k,
                                                           use_population=source != "own", train_cells=train_cells))
    ml_cfg = dict(cell=show_t, n0=n0_show, targets=tuple(sc["targets"]), models=tuple(models), source=source,
                  train_cells=train_cells, eol_ah=float(eol_ah), strategy=strategy, conformal_cells=conf,
                  band_level=level, params={m: params.get(m) for m in models})
    ready = bool(models) and (source != "cross" or bool(train_cells))
    n_runs = len(models) * len(sc["targets"])
    if st.button(f"Train and forecast ({n_runs} run{'s' if n_runs > 1 else ''})", type="primary", key="ml_go",
                 disabled=not ready, icon=":material/play_arrow:"):
        prog = st.progress(0.0)
        by_target: Dict[str, List[te.MLForecast]] = {}
        k_ = 0
        for t in sc["targets"]:
            for mname in models:
                prog.progress(k_ / max(n_runs, 1), text=f"Training {mname} for {t}…")
                k_ += 1
                try:
                    r = ml_cached(ct, DATA_KEY, t, sc["n0"][t], mname, use_pop, float(eol_ah), strategy, conf, level,
                                  json.dumps(params.get(mname, {}), sort_keys=True), train_cells)
                    by_target.setdefault(t, []).append(r)
                    ladder_add(f"ML · {mname}", te.MODEL_SPECS[mname].level if mname in te.MODEL_SPECS else 2,
                               sc["n0"][t], r.n_grid, r.soh_pred, r.soh_lo, r.soh_hi, r.metrics, t)
                except Exception as exc:
                    st.warning(f"{mname} on {t}: {exc}")
        prog.empty()
        st.session_state["ml"] = {"cfg": ml_cfg, "res": by_target.get(show_t, []), "by_target": by_target}
    saved = st.session_state.get("ml")
    if saved and saved.get("by_target") and show_t in saved["by_target"] and saved["cfg"]["cell"] != show_t:
        saved = dict(saved, res=saved["by_target"][show_t], cfg=dict(saved["cfg"], cell=show_t, n0=sc["n0"][show_t]))
    saved = st.session_state.get("ml")
    if not (saved and saved["res"]):
        return
    if saved["cfg"]["cell"] != show_t:
        st.info("The stored forecasts belong to another battery or scheme. Train again to update them.")
        return
    if {k: v for k, v in saved["cfg"].items() if k not in ("cell", "n0")} != \
            {k: v for k, v in ml_cfg.items() if k not in ("cell", "n0")}:
        st.warning("Settings changed since the last run; the results below use the previous settings.")
    res = saved["res"]
    n0s = saved["cfg"]["n0"]
    ct_show = ct[ct["Cell_ID"] == show_t].sort_values("n")
    eol_show = te.soh_eol_for(float(meta.loc[show_t, "C_bol_Ah"]), float(eol_ah))
    data = pd.concat([pd.DataFrame({"model": r.model, "n": r.n_grid, "soh": r.soh_pred,
                                    "lo": r.soh_lo if r.soh_lo is not None else np.nan,
                                    "hi": r.soh_hi if r.soh_hi is not None else np.nan}) for r in res])
    show(fig_ml(res, ct_show, n0s, eol_show, P), key="ml_fig", data=data)
    tbl = pd.DataFrame([{"Model": r.model, "Accuracy (%)": r.metrics.accuracy, "Fade skill": r.metrics.fade_skill,
                         "R²": r.metrics.r2, "RMSE": r.metrics.rmse, "MAE": r.metrics.mae,
                         "Coverage": r.metrics.coverage, "RUL true": r.metrics.rul_true, "RUL pred": r.metrics.rul_pred,
                         "RUL error": r.metrics.rul_error, "Fit time (s)": r.fit_seconds} for r in res]).set_index("Model")
    best = _best_key(tbl, "RMSE", lowest=True) or tbl.index[0]
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
    note("Forecast scores of each ML model on the cycles after the origin: accuracy, fade skill, R², errors, band coverage and remaining-life error.")
    good = ct_show[(~ct_show["outlier"]) & (ct_show["n"] > n0s)]
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
    params = tuning_block(f"est_{split}_{test_frac}_{hash((tr_cells, te_cells, tuple(feats))) % 10**6}", models, params,
                          lambda m, k: te.tune_soh_estimator(ct, imp, m, tuple(feats), split, float(test_frac),
                                                             list(tr_cells) or None, list(te_cells) or None, normalise,
                                                             n_iter=k))
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
    best = _best_key(tbl, "Test RMSE", lowest=True) or tbl.index[0]
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
    note("Estimation scores on the test cycles and on the training cycles: a large gap between them means the model memorises instead of learning.")
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
        pars = c3.multiselect("Paradigms", [p_ for p_ in te.BENCH_PARADIGMS if p_ != "PINN"],
                              default=[p_ for p_ in ("ML", "Twin", "HB") if p_ in te.BENCH_PARADIGMS],
                              key="bench_pars", format_func={"ML": "ML surrogate", "Twin": "ECM twin (dual EKF)",
                                                             "PINN": "Hybrid PINN", "SemiEmp": "Semi-empirical",
                                                             "PF": "Particle filter", "HB": "Hierarchical Bayes",
                                                             "GP": "Physics-mean GP"}.get,
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


def early_life_section() -> None:
    section("Early-life lifetime prediction from ΔQ(V)")
    st.markdown("Severson et al. (*Nature Energy* 4, 2019) showed that the variance of ΔQ(V), the change of the "
                "discharge curve between an early and a slightly later cycle, predicts cycle life long before capacity "
                "fades visibly. Here the same idea is applied to your cohort, with a leave-one-cell-out elastic net. "
                "The ΔQ(V) statistics also feed the ML fade-rate models as early-life descriptors.")
    c1, c2 = st.columns(2)
    n_a = c1.slider("Reference cycle", 1, 10, 2, key="el_a")
    n_b = c2.slider("Comparison cycle (early window end)", n_a + 5, 80, max(n_a + 5, 20), key="el_b")
    el = early_life_cached(ct, DATA_KEY, float(eol_ah), int(n_a), int(n_b))
    if el["table"].empty:
        st.info("Not enough cells with this many cycles.")
        return
    show(fig_dq_curves(ct, el, P), key="el_dq", data=el["table"])
    if not el["available"]:
        st.info("Fewer than five cells reached end of life with usable ΔQ(V): the lifetime regression is not "
                "available (lower the end-of-life capacity in the control bar to include more cells).")
        return
    k = st.columns(4)
    k[0].metric("Cells with known life", el["n_cells"])
    k[1].metric("corr(log var ΔQ, log life)", fmt(el["corr_logvar_loglife"], ".2f"))
    k[2].metric("LOCO lifetime error", fmt(el["mape_pct"], ".1f", "%"))
    k[3].metric("Cohort-mean baseline", fmt(el["baseline_mape_pct"], ".1f", "%"))
    show(fig_lifetime_parity(el, P, list(meta.index)), key="el_parity", data=el["loco"])
    st.caption("NASA cells live 40–200 cycles and the cohort mixes temperatures, currents and cut-offs. The early "
               "window is therefore much shorter than Severson's (cycle 10 → 100), and the elastic net also sees the "
               "operating conditions. A strongly negative correlation means ΔQ(V) carries lifetime information.")


def ml_methods_panel() -> None:
    with st.expander("Methods: machine-learning models", icon=":material/menu_book:"):
        st.markdown("**Fade-rate forecasting.** A regressor learns the degradation rate as a function of the current "
                    "state, the operating conditions and early-life descriptors, and is integrated forward from the "
                    "forecast origin, so tree models keep extrapolating beyond the longest training life:")
        st.latex(r"\frac{d\,\mathrm{SOH}}{dn} = g_\phi\!\left(\mathrm{SOH},\ T,\ I,\ V_\mathrm{cut},\ "
                 r"\text{early slope},\ \text{early } R \text{ growth},\ \log\operatorname{var}\Delta Q(V)\right) \le 0")
        st.markdown("**Uncertainty.** Split-conformal bands calibrated on the batteries closest in operating conditions, "
                    "widening with the forecast horizon. **Estimation.** Present SOH from operando indicators measured on "
                    "the same cycle, with random, chronological or by-battery splits. **Tuning.** Random search validated "
                    "by forecast backtests on other batteries (forecasting) or grouped cross-validation by battery "
                    "(estimation). **Early life.** Elastic net on ΔQ(V) statistics (Severson et al. 2019).")
        st.markdown("**Scores.** Accuracy = 100 × (1 − MAPE); fade skill = 1 − SSE / SSE of a 'no further fade' "
                    "forecast; R², RMSE and MAE on held-out cycles; RUL error and α-λ accuracy (Saxena et al. 2010). "
                    "Physics-based and mechanistic models (twin, particle filters, PINN) live in the Live twin view.")


LEVEL_INFO = {
    1: ("Baselines", "Persistence, linear trend, a single decision tree, Bayesian ridge: the references to beat."),
    2: ("Classical ML", "Random Forest, Extra Trees, Gaussian Process: robust learners for small data."),
    3: ("Boosting", "Histogram gradient boosting, XGBoost, LightGBM: strongest on tabular data."),
    4: ("Deep learning", "GRU recurrent network and Transformer: learn from sequences of past SOH."),
    5: ("Hybrid & physics-informed", "Mechanistic PINN and hierarchical Bayes: equations plus learning."),
    6: ("First principles", "Single-particle electrochemical model with SEI growth: physics only, calibrated."),
}
LEVEL_COLORS = {1: "#8C8C8C", 2: "#0072B2", 3: "#009E73", 4: "#AA4499", 5: "#E69F00", 6: "#D55E00"}


def scheme() -> Dict[str, Any]:
    return st.session_state.get("_scheme") or {"mode": "within", "targets": [cell], "train": None,
                                               "n0": {cell: int(max(5, round(0.4 * meta.loc[cell, "cycles"])))},
                                               "use_cohort": True, "show": cell}


def scheme_controls() -> Dict[str, Any]:
    """One training scheme for every level of the ladder."""
    all_cells = list(meta.index)
    mode = st.radio("Training scheme", ["within", "across"], horizontal=True, key="sch_mode",
                    format_func={"within": ":material/call_split: Within a battery: train on its first part, test on the rest",
                                 "across": ":material/swap_horiz: Across batteries: train on some batteries, predict others"}.get)
    if mode == "within":
        c1, c2 = st.columns([3, 2])
        frac = c1.slider("Training part of the battery's life (the rest is the test)", 0.1, 0.9, 0.4, 0.05, key="ml_frac")
        use_cohort = c2.toggle("Models may also learn from the other batteries' full histories", value=True,
                               key="sch_cohort", help="Off = strictly this battery only (fewer data, harder for ML).")
        n0 = int(max(5, round(frac * meta.loc[cell, "cycles"])))
        sc = {"mode": "within", "targets": [cell], "train": None if use_cohort else [], "use_cohort": use_cohort,
              "n0": {cell: n0}, "show": cell, "frac": frac}
        st.caption(f"Battery {cell}: training on cycles 1–{n0}, testing on cycles {n0 + 1}–{int(meta.loc[cell, 'cycles'])}.")
    else:
        c1, c2 = st.columns(2)
        near = te.calibration_partners(meta, cell, 4)
        train = c1.multiselect("Training batteries", [c for c in all_cells], default=[c for c in near if c != cell],
                               key="sch_train", format_func=lambda c: cell_label(c, meta))
        test = c2.multiselect("Test batteries (one or more)", [c for c in all_cells if c not in train],
                              default=[cell] if cell not in train else [], key="sch_test",
                              format_func=lambda c: cell_label(c, meta))
        frac = st.slider("Start of each test battery the models may see (the rest is forecast)", 0.05, 0.6, 0.15, 0.05,
                         key="sch_seen", help="Models need a few cycles of the new battery to know its current state.")
        n0 = {t: int(max(10, round(frac * meta.loc[t, "cycles"]))) for t in test}
        show_t = st.selectbox("Battery to display in the charts", test or [cell], key="sch_show",
                              format_func=lambda c: cell_label(c, meta)) if test else cell
        sc = {"mode": "across", "targets": test, "train": train, "use_cohort": False, "n0": n0, "show": show_t,
              "frac": frac}
        if not train or not test:
            st.warning("Choose at least one training and one test battery.")
        else:
            st.caption(f"Learning from {len(train)} batteries, forecasting {len(test)}: each test battery is seen for its "
                       f"first {100 * frac:.0f}% of life. Baselines, the PINN and the first-principles model only need "
                       "the test battery itself; the other levels learn from the training batteries.")
    st.session_state["_scheme"] = sc
    return sc


def ladder_add(name: str, level: int, n0: int, n_grid: np.ndarray, soh: np.ndarray, lo: Optional[np.ndarray],
               hi: Optional[np.ndarray], metrics: Any, target: Optional[str] = None) -> None:
    store_ = st.session_state.setdefault("ladder", {})
    store_[(target or cell, int(n0), name)] = {"name": name, "level": int(level), "n_grid": np.asarray(n_grid),
                                               "soh": np.asarray(soh), "lo": None if lo is None else np.asarray(lo),
                                               "hi": None if hi is None else np.asarray(hi), "m": metrics,
                                               "target": target or cell}


def ladder_entries(n0: Optional[int] = None, target: Optional[str] = None) -> List[Dict[str, Any]]:
    sc = scheme()
    t = target or sc["show"]
    n = n0 if n0 is not None else sc["n0"].get(t)
    return sorted([v for (c, nn, _), v in st.session_state.get("ladder", {}).items() if c == t and nn == n],
                  key=lambda v: (v["level"], v["name"]))


def all_target_entries() -> List[Dict[str, Any]]:
    sc = scheme()
    out = []
    for t in sc["targets"]:
        out += ladder_entries(sc["n0"][t], t)
    return out


def fig_ladder(entries: Sequence[Dict[str, Any]], n0: int, P: Palette, title: str, target: Optional[str] = None) -> go.Figure:
    t = target or scheme()["show"]
    ctt = ct[(ct["Cell_ID"] == t) & ~ct["outlier"]]
    eol_t = te.soh_eol_for(float(meta.loc[t, "C_bol_Ah"]), float(eol_ah))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ctt["n"], y=ctt["SOH"], mode="markers", name=f"Measured SOH · {t}",
                             marker=dict(color=P.measured, size=5, opacity=0.75), hovertemplate="%{y:.4f}<extra>measured</extra>"))
    for i, e in enumerate(entries):
        col = LEVEL_COLORS.get(e["level"], P.text)
        sel = e["n_grid"] >= n0
        if e["lo"] is not None and e["hi"] is not None and len(entries) <= 3:
            add_band(fig, e["n_grid"][sel], e["lo"][sel], e["hi"][sel], col, f"{e['name']} band", group=e["name"], alpha=0.12)
        fig.add_trace(go.Scatter(x=e["n_grid"][sel], y=e["soh"][sel], mode="lines", name=f"L{e['level']} · {e['name']}",
                                 legendgroup=e["name"], line=dict(color=col, width=2.6, dash=DASHES[i % len(DASHES)]),
                                 hovertemplate="%{y:.4f}<extra>" + html.escape(e["name"]) + "</extra>"))
    fig.add_vline(x=n0, line_dash="dot", line_color=P.muted, annotation_text="training | test",
                  annotation_font=dict(color=P.muted))
    fig.add_hline(y=eol_t, line_dash="dash", line_color=P.eol, annotation_text="End of life",
                  annotation_font=dict(color=P.eol))
    fig.update_xaxes(title_text="Discharge cycle n")
    fig.update_yaxes(title_text="SOH (–)")
    return style_fig(fig, P, 520, f"{title} · {t}")


def _ladder_table(entries: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Per model: metrics averaged over the test batteries (one row per model)."""
    rows = [{"Level": f"L{e['level']} · {LEVEL_INFO[e['level']][0]}", "Model": e["name"], "Battery": e["target"],
             "Accuracy (%)": e["m"].accuracy, "RMSE": e["m"].rmse, "Fade skill": e["m"].fade_skill,
             "Coverage": e["m"].coverage, "RUL error": e["m"].rul_error} for e in entries if e["m"] is not None]
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    agg = d.groupby(["Level", "Model"], as_index=False).agg(
        **{"Test batteries": ("Battery", "nunique"), "Accuracy (%)": ("Accuracy (%)", "mean"), "RMSE": ("RMSE", "mean"),
           "Fade skill": ("Fade skill", "mean"), "Coverage": ("Coverage", "mean"),
           "RUL error": ("RUL error", lambda x: float(np.nanmean(np.abs(x))) if x.notna().any() else np.nan)})
    return agg


TABLE_FMT = {"Accuracy (%)": "{:.2f}", "RMSE": "{:.4f}", "Fade skill": "{:.3f}", "Coverage": "{:.0%}", "RUL error": "{:.0f}"}


def ladder_level_block(level: int, key: str) -> None:
    sc = scheme()
    shown = [e for e in ladder_entries() if e["level"] == level]
    allt = [e for e in all_target_entries() if e["level"] == level]
    if shown:
        show(fig_ladder(shown, sc["n0"][sc["show"]], P, f"Level {level} · {LEVEL_INFO[level][0]}"), key=key, export=False)
    if allt:
        multi = len(sc["targets"]) > 1
        show_table(_ladder_table(allt).set_index("Model").style.format(TABLE_FMT, na_rep="—"),
                   note=("Scores averaged over the test batteries (RUL error = mean absolute error in cycles)." if multi else
                         "Scores on the test cycles of this battery."))


def _run_targets(fn: Callable[[str, int], Any], label: str) -> None:
    """Run a forecaster for every test battery of the current scheme, with progress and per-battery errors."""
    sc = scheme()
    prog = st.progress(0.0)
    for i, t in enumerate(sc["targets"]):
        prog.progress(i / max(len(sc["targets"]), 1), text=f"{label}: {t}")
        try:
            fn(t, sc["n0"][t])
        except Exception as exc:
            st.warning(f"{label} on {t}: {exc}")
    prog.empty()


def ladder_intro() -> None:
    done = {e["level"] for e in ladder_entries()}
    steps = "".join(
        f'<div class="bt-step{" bt-step-done" if lv in done else ""}"><div class="bt-step-n">{lv}</div>'
        f'<div class="bt-step-t">{html.escape(LEVEL_INFO[lv][0])}</div></div>' for lv in LEVEL_INFO)
    st.markdown(f'<div class="bt-ladder">{steps}</div>', unsafe_allow_html=True)
    st.caption("Work top to bottom: every level adds one idea. A level is ticked once one of its models has run on the "
               "displayed battery; the leaderboard at the end compares them all on the same test cycles.")


def ladder_baselines() -> None:
    section("Level 1 · Baselines: the references every model must beat")
    st.markdown("Two forecasts that need no learning at all. If an advanced model cannot beat the **linear trend**, "
                "its complexity is not paying off.")
    if st.button("Run baselines", key="lad_b_go", icon=":material/play_arrow:", type="primary"):
        def run(t, n0):
            for kind in ("persistence", "trend"):
                f = te.baseline_forecast(ct, t, n0, kind, float(eol_ah))
                ladder_add(f.name, 1, n0, f.n_grid, f.soh, f.lo, f.hi, f.metrics, t)
        _run_targets(run, "Baselines")
    ladder_level_block(1, "lad_base")


def ladder_deep() -> None:
    section("Level 4 · Deep learning: recurrent network and Transformer")
    st.markdown("Sequence models read a window of past SOH values (plus temperature and current) and predict the next "
                "cycle; feeding each prediction back builds the whole forecast. Three networks with different seeds "
                "form a small ensemble for the band. They are trained on all other batteries and this battery's past.")
    c = st.columns(4)
    kinds = c[0].multiselect("Networks", list(te.SEQ_MODELS), default=list(te.SEQ_MODELS), key="lad_d_kinds")
    L = c[1].select_slider("Window (cycles)", [5, 8, 10, 15, 20], value=10, key="lad_d_L")
    ep = c[2].select_slider("Epochs", [100, 200, 300, 500], value=200, key="lad_d_ep")
    mem = c[3].select_slider("Ensemble members", [1, 2, 3, 5], value=3, key="lad_d_mem")
    if st.button(f"Train deep models (≈ {int(len(kinds) * mem * ep / 60 * max(len(scheme()['targets']), 1)) + 2} s)", key="lad_d_go",
                 icon=":material/neurology:", type="primary", disabled=not kinds):
        sc = scheme()
        def run(t, n0):
            for k in kinds:
                f = te.seq_forecast(ct, t, n0, k, float(eol_ah), L=int(L), epochs=int(ep), n_members=int(mem),
                                    train_cells=sc["train"])
                ladder_add(k, 4, n0, f.n_grid, f.soh, f.lo, f.hi, f.metrics, t)
        _run_targets(run, "Deep models")
    ladder_level_block(4, "lad_deep")
    with st.expander("Why no large language model (LLM) forecaster?", icon=":material/help:"):
        st.markdown("LLMs and time-series foundation models are trained on text or on millions of generic series; "
                    "with ~30 batteries they add no physical knowledge and cannot be validated against it, and "
                    "running them needs large downloads or paid APIs. They are useful *around* the twin instead: "
                    "explaining results, drafting reports, or answering operator questions from the twin's outputs. "
                    "The deep models here (GRU, Transformer) are the same architectures at a size these data can "
                    "support.")


def ladder_hybrid() -> None:
    section("Level 5 · Hybrid & physics-informed models")
    st.markdown("**Mechanistic PINN**: a neural network trained to fit the data *and* obey the SEI, plating and "
                "loss-of-active-material equations. **Hierarchical Bayes**: a physics fade law whose parameters "
                "start from what the whole fleet taught us and are updated by this battery's data.")
    c = st.columns(3)
    run_p = c[0].toggle("Mechanistic PINN", value=True, key="lad_h_pinn")
    ep = c[1].select_slider("PINN epochs", [500, 1000, 1500, 2500], value=1000, key="lad_h_ep")
    run_hb = c[2].toggle("Hierarchical Bayes", value=True, key="lad_h_hb")
    if st.button("Run hybrid models", key="lad_h_go", icon=":material/hub:", type="primary"):
        sc = scheme()
        def run(t, n0):
            if run_p:
                r = te.train_pinn(ct, imp, t, n0, te.PINNConfig(epochs=int(ep), physics="mechanistic"), float(eol_ah))
                ladder_add("Mechanistic PINN", 5, n0, r.n_grid, r.soh, r.soh_lo, r.soh_hi, r.metrics, t)
            if run_hb:
                pop = sc["train"] if sc["mode"] == "across" else None
                f = te.hierarchical_bayes_forecast(ct, t, n0, float(eol_ah), train_cells=pop)
                ladder_add("Hierarchical Bayes", 5, n0, f.n_grid, f.soh, f.lo, f.hi, f.metrics, t)
        _run_targets(run, "Hybrid models")
    ladder_level_block(5, "lad_hybrid")


def fig_spm_curves(P: Palette) -> go.Figure:
    p = te.SPMParams()
    I = float(meta.loc[cell, "I_dis_A"])
    T = float(meta.loc[cell, "T_mean_C"])
    fig = go.Figure()
    for name, kw, col, dash in (("New cell", dict(), P.accent, "solid"), ("10% lithium lost", dict(lli=0.10), "#E69F00", "dash"),
                                ("20% lithium lost", dict(lli=0.20), "#D55E00", "dot"),
                                ("2× current", dict(I_A=2 * I), "#AA4499", "dashdot"), ("Cold (4 °C)", dict(T_C=4.0), "#56B4E9", "longdash")):
        args = dict(p=p, I_A=I, T_C=T)
        args.update(kw)
        r = te.spm_discharge(**args)
        fig.add_trace(go.Scatter(x=r["q_Ah"], y=r["V"], mode="lines", name=f"{name} · {r['capacity_Ah']:.2f} Ah",
                                 line=dict(color=col, width=2.6, dash=dash), hovertemplate="%{y:.3f} V<extra>" + name + "</extra>"))
    fig.update_xaxes(title_text="Discharged capacity (Ah)")
    fig.update_yaxes(title_text="Terminal voltage (V)")
    return style_fig(fig, P, 470, f"Single-particle model: discharge at {I:.1f} A, {T:.0f} °C")


def ladder_first_principles() -> None:
    section("Level 6 · First principles: single-particle electrochemical model")
    st.markdown("Each electrode is one spherical particle: lithium diffuses inside it (Fick's law), crosses the "
                "surface with Butler–Volmer kinetics, and the voltage is the difference of the LiCoO₂ and graphite "
                "potentials minus the losses. Ageing is SEI growth consuming lithium, "
                "LLI = a·Ah + b·√Ah (reaction- plus diffusion-limited), fitted to this battery's capacity history.")
    show(fig_spm_curves(P), key="lad_spm_curves", export=False)
    with st.expander("Governing equations", icon=":material/functions:"):
        st.latex(r"\frac{\partial \theta}{\partial t} = \frac{D}{r^2}\frac{\partial}{\partial r}\left(r^2 \frac{\partial \theta}{\partial r}\right),"
                 r"\qquad D\,\frac{\partial \theta}{\partial r}\Big|_{r=R} = \mp\frac{I}{3600\,C}\,\frac{R}{3}")
        st.latex(r"V = U_p(\theta_{p,s}) - U_n(\theta_{n,s}) - \frac{2RT}{F}\left[\sinh^{-1}\frac{I}{2 i_{0,p}} + "
                 r"\sinh^{-1}\frac{I}{2 i_{0,n}}\right] - I\,(R_\Omega + R_\mathrm{film}),\quad i_0 \propto \sqrt{\theta(1-\theta)}")
        st.latex(r"\mathrm{LLI}(\mathrm{Ah}) = a\,\mathrm{Ah} + b\,\sqrt{\mathrm{Ah}},\qquad x_0 \rightarrow x_0 - \mathrm{LLI},"
                 r"\qquad R_\mathrm{film} \propto \mathrm{LLI}")
    if st.button("Run first-principles forecast", key="lad_s_go", icon=":material/science:", type="primary"):
        def run(t, n0):
            f = te.spm_forecast(ct, t, n0, float(eol_ah))
            ladder_add("SPM + SEI", 6, n0, f.n_grid, f.soh, f.lo, f.hi, f.metrics, t)
            st.session_state["spm_params"] = f.params
        _run_targets(run, "First-principles model")
    ladder_level_block(6, "lad_spm")
    if st.session_state.get("spm_params"):
        show_table(pd.Series(st.session_state["spm_params"], name="Value").to_frame().style.format("{:.4g}"),
                   note="Fitted SEI kinetics: a large reaction term means near-linear fade, a large diffusion term "
                        "means fade that slows down as the SEI film thickens.")


def ladder_leaderboard() -> None:
    section("Leaderboard: every level on the same test cycles")
    sc = scheme()
    shown = ladder_entries()
    allt = all_target_entries()
    if not allt:
        st.info("Run at least one level above; results appear here side by side.")
        return
    if shown:
        show(fig_ladder(shown, sc["n0"][sc["show"]], P, "All models"), key="lad_board", export=False)
    tab = _ladder_table(allt).sort_values("RMSE")
    best = tab.iloc[0]
    k = st.columns(3)
    k[0].metric("Best model", best["Model"], delta=best["Level"], delta_color="off")
    k[1].metric("Mean accuracy", fmt(best["Accuracy (%)"], ".2f", "%"),
                help=f"Averaged over {int(best['Test batteries'])} test battery(ies).")
    base = tab[tab["Model"] == "Baseline · linear trend"]
    if len(base):
        k[2].metric("Gain over linear trend", fmt(100 * (1 - best["RMSE"] / base["RMSE"].iloc[0]), ".0f", "%"),
                    help="How much lower the best model's error is than the simplest credible baseline.")
    incomplete = tab[tab["Test batteries"] < len(sc["targets"])]
    show_table(tab.set_index("Model").style.format(TABLE_FMT, na_rep="—")
               .highlight_min(subset=["RMSE"], props="background-color: rgba(0,158,115,0.25); font-weight: 700;"),
               note="Sorted by error on the test cycles" + (" and averaged over the test batteries" if len(sc["targets"]) > 1
                                                             else "") + ". More complexity is only worth it when it "
                    "clearly beats the lower levels and keeps honest uncertainty (coverage near 90%).")
    if len(incomplete):
        st.warning("Not all models ran on every test battery: " + ", ".join(
            f"{m} ({int(n)}/{len(sc['targets'])})" for m, n in zip(incomplete["Model"], incomplete["Test batteries"])) +
                   ". Compare them with care.")
    if st.button("Clear leaderboard", key="lad_clear", icon=":material/delete:"):
        st.session_state["ladder"] = {}
        st.rerun()


def view_models() -> None:
    recommendations("models")
    section("Learning ladder: from simple references to first principles")
    sc = scheme_controls()
    if not sc["targets"]:
        return
    ladder_intro()
    ml_methods_panel()
    ladder_baselines()
    ml_section()
    ladder_deep()
    ladder_hybrid()
    ladder_first_principles()
    ladder_leaderboard()
    early_life_section()
    cross_cell_section()


# =============================================================================
# View 3: operations & optimal control
# =============================================================================
def view_ops() -> None:
    recommendations("ops")
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
        soh_eol_ops = s4.slider("Replacement SOH (decision)", 0.60, 0.85, 0.70, 0.01,
                                help="When the operator replaces the cell: a maintenance decision optimised in the "
                                     "integrated study and the DP, not the physical end-of-life definition.")
        e1, e2, e3, e4 = st.columns(4)
        amb_mean = e1.slider("Mean ambient (°C)", 0, 35, 20)
        amb_amp = e2.slider("Seasonal ambient amplitude (°C)", 0, 20, 16)
        mismatch = e3.slider("Plant / model mismatch", 0.0, 0.6, 0.0, 0.05,
                             help="Relative std of the perturbation applied to the plant's ageing, resistance, "
                                  "thermal and OCV parameters; the policy keeps the nominal model.")
        seed = e4.number_input("Mismatch draw (seed)", 0, 999, 0, 1, disabled=mismatch == 0)
        ekf_saved = st.session_state.get("ekf")
        use_twin = st.toggle("Initialise the model from the latest observer run", value=False,
                             disabled=ekf_saved is None, help="Available after opening the Live twin for a battery.")
        show_base = st.toggle("Overlay fixed-current baselines", value=True)

    econ = te.Economics(price_per_Ah=tuple(price), replacement_cost=float(replacement), soh_eol=float(soh_eol_ops),
                        degradation_weight=float(weight), T_max_C=float(t_max),
                        cold_derate_below_C=float(cold_thr) if cold_rule else None, objective=objective,
                        energy_price_per_Wh=float(e_price))
    calib = st.toggle(f"Calibrate the plant on the selected battery ({cell})", value=True, key="ops_calib",
                      help="Capacity, resistances and the battery's own fade rate from its data; activation energy, "
                           "current exponent and cold multiplier from the cohort stress regression. Off = a generic "
                           "18650 cell.")
    if calib:
        try:
            phys, calib_tab = calibrated_plant_cached(ct, imp, DATA_KEY, cell)
            with st.expander(f"Calibrated plant for {cell}", icon=":material/tune:"):
                show_table(calib_tab.style.format({"Value": "{:.4g}"}))
                st.caption("Mission 1–2 results feed Mission 3: the optimiser now plans for this battery's measured "
                           "behaviour instead of a generic cell.")
        except Exception as exc:
            report_error("Plant calibration failed; using the generic cell", exc, debug)
            phys = te.CellPhysics()
    else:
        phys = te.CellPhysics()
    if use_twin and ekf_saved:
        r = ekf_saved["res"]
        phys = replace(phys, k_ah=float(r.params["k_ah"]), R_int0=float(r.r_int0), R_ct0=float(r.r_ct0))
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
    dp_section(econ, phys, plant, float(amb_mean), float(amb_amp))
    scenario_section()


def dp_section(econ: te.Economics, phys: te.CellPhysics, plant: Optional[te.CellPhysics], amb_mean: float,
               amb_amp: float) -> None:
    section("Optimal operation and replacement by dynamic programming")
    st.markdown("The twin-aware policy above is one-step (greedy). Here the full sequential problem is solved as a "
                "semi-Markov decision process. State: (SOH, season phase). Actions: discharge current or replace. "
                "The objective is the true long-run profit rate, including the sudden-failure hazard, energy cost "
                "and downtime. Dinkelbach's method finds the rate ρ* at which a new cell is exactly worth its "
                "renewal. The DP policy is then run in closed loop, with observer noise and any plant mismatch set "
                "above, and scored exactly like the grid study.")
    with st.expander("Bellman equation", icon=":material/functions:"):
        st.latex(r"W_\rho(s, j) = \max\Big\{-C_\mathrm{plan} - \rho\,t_\mathrm{plan},\;\max_I\big[r_I - \rho\,t_I "
                 r"+ h(s)\,(-C_\mathrm{unpl} - \rho\,t_\mathrm{unpl}) + (1 - h(s))\,\mathbb{E}\,W_\rho(s - \Delta s_I, j')"
                 r"\big]\Big\},\qquad \rho^*:\ W_{\rho^*}(1, j_0) = 0")
    saved_om = st.session_state.get("om")
    maint = saved_om["cfg"]["maint"] if saved_om else asdict(te.MaintenanceModel(replacement_cost=econ.replacement_cost))
    c1, c2 = st.columns(2)
    n_soh = c1.select_slider("SOH grid points", [50, 70, 90, 120], value=90, key="dp_ns")
    n_ph = c2.select_slider("Season bins", [8, 12, 16, 24], value=16, key="dp_nph")
    if st.button(f"Solve DP (≈ {4 + n_soh * n_ph // 150} s)", key="dp_go", icon=":material/account_tree:", type="primary"):
        with st.spinner("Tabulating cycle outcomes and running value iteration…"):
            try:
                dp = dp_cached(asdict(econ), asdict(phys), maint, amb_mean, amb_amp, int(n_soh), int(n_ph))
                ev, life = te.evaluate_dp_policy(dp, econ, phys, te.MaintenanceModel(**maint), plant=plant,
                                                 ambient_mean_C=amb_mean, ambient_amp_C=amb_amp)
                st.session_state["dp"] = {"dp": dp, "eval": ev, "life": life}
            except Exception as exc:
                report_error("Dynamic programming failed", exc, debug)
    saved = st.session_state.get("dp")
    if not saved:
        return
    dp, ev = saved["dp"], saved["eval"]
    k = st.columns(5)
    k[0].metric("Model-optimal rate ρ*", fmt(dp.rho, ".4f", "CU/h"), help="Upper bound under the DP's model.")
    k[1].metric("Closed-loop rate", fmt(ev["rate"], ".4f", "CU/h"))
    k[2].metric("Replaced at SOH", fmt(ev["threshold"], ".3f"))
    k[3].metric("P(sudden failure)", fmt(100 * ev["p_failure"], ".1f", "%"))
    k[4].metric("Violations", int(ev["violations"]))
    show(fig_dp_policy(dp, P), key="dp_fig", data=pd.DataFrame(dp.action, index=dp.soh_grid, columns=dp.phases))
    if saved_om:
        opt = te.integrated_optimum(saved_om["study"])
        b = opt.get("best")
        if b:
            gap = 100 * (ev["rate"] - b["rate"]) / abs(b["rate"])
            card("DP versus the best grid policy", [
                f"Grid optimum: {b['policy']}, replace at {b['threshold']:.2f} → {b['rate']:.4f} CU/h; "
                f"DP closed loop {ev['rate']:.4f} CU/h ({gap:+.1f}%).",
                f"The simple policy reaches {100 * b['rate'] / dp.rho:.0f}% of the model optimum ρ*. When the gap is "
                "small, the one-step policy plus a replacement threshold is already near-optimal for this plant. "
                "The DP adds a certificate of that, plus a season-dependent replacement boundary."])
    else:
        st.caption("Run the integrated optimisation above to compare the DP with the best grid policy.")


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
sidebar_guide()
_VIEW_FN: Dict[str, Callable[[], None]] = {VIEWS[0]: view_data, VIEWS[1]: view_replay, VIEWS[2]: view_models,
                                           VIEWS[3]: view_ops, VIEWS[4]: view_study}
try:
    _VIEW_FN.get(view, view_data)()
except Exception as exc:
    report_error(f"{view} failed", exc, debug)
