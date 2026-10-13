# Author: Your Name
# CancerLens — educational data exploration, statistical analysis and patient Q&A application.
# This app does not provide diagnosis, risk assessment, or treatment advice.
#
# Analysis stack: inferential statistics (SciPy), multivariate methods and machine
# learning (scikit-learn), interactive graphics (Plotly), app framework (Streamlit).

from __future__ import annotations

import inspect
import itertools
import math
import warnings
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    normalized_mutual_info_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    learning_curve,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
try:  # sklearn convergence warnings are noisy for the MLP / logistic models
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:  # pragma: no cover
    pass


# ------------------------------------------------------------
# App setup and visual system
# ------------------------------------------------------------
st.set_page_config(
    page_title="CancerLens | Patient Data Atelier",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=IM+Fell+English:ital@0;1&family=Manrope:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --paper: #eee5d4;
        --paper-light: #faf5eb;
        --ink: #28231d;
        --walnut: #503725;
        --wood: #765337;
        --brass: #bd965b;
        --moss: #69765b;
        --clay: #9b6650;
        --line: rgba(75, 53, 34, .18);
    }

    @keyframes rise {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse-dot {
        0%, 100% { box-shadow: 0 0 0 0 rgba(189,150,91,.55); }
        50%      { box-shadow: 0 0 0 7px rgba(189,150,91,0); }
    }

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
        color: var(--ink);
    }

    .stApp {
        background:
            radial-gradient(ellipse at 8% 4%, rgba(180, 140, 79, .17), transparent 32%),
            radial-gradient(ellipse at 93% 88%, rgba(88, 102, 72, .12), transparent 32%),
            repeating-linear-gradient(0deg,
                rgba(79, 54, 34, .022) 0px,
                rgba(79, 54, 34, .022) 1px,
                transparent 1px,
                transparent 5px),
            var(--paper);
    }

    [data-testid="stSidebar"] {
        background:
            repeating-linear-gradient(90deg,
                rgba(255,255,255,.025) 0px,
                rgba(255,255,255,.025) 2px,
                transparent 2px,
                transparent 10px),
            linear-gradient(155deg, #34251c, #60432c 58%, #3b3025);
        border-right: 1px solid rgba(255,255,255,.12);
    }

    [data-testid="stSidebar"] * { color: #f6ead5; }
    [data-testid="stSidebar"] input { color: #2d261d !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #2d261d !important; }

    .block-container {
        max-width: 1480px;
        padding-top: 4.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Cormorant Garamond', Georgia, serif !important;
        color: var(--walnut);
        letter-spacing: -.45px;
    }

    .masthead {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 3px 2px 15px;
        border-bottom: 1px solid var(--line);
        margin-bottom: 18px;
    }

    .masthead-name {
        font-family: 'Cormorant Garamond', Georgia, serif;
        color: var(--walnut);
        font-size: 24px;
        font-weight: 700;
        letter-spacing: 2px;
    }

    .masthead-note {
        color: #806d54;
        font: 10px 'DM Mono', monospace;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .live-dot {
        display: inline-block;
        width: 8px; height: 8px;
        margin-right: 8px;
        border-radius: 50%;
        background: var(--brass);
        animation: pulse-dot 2.2s infinite;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 38px 42px;
        margin: 10px 0 20px;
        border: 1px solid rgba(255,255,255,.16);
        border-radius: 22px;
        color: #fff8ec;
        background:
            repeating-linear-gradient(2deg,
                rgba(255,255,255,.025) 0px,
                rgba(255,255,255,.025) 2px,
                transparent 2px,
                transparent 10px),
            linear-gradient(125deg, #30241c 0%, #68482f 57%, #354033 100%);
        box-shadow: 0 16px 38px rgba(53, 39, 25, .19);
        animation: rise .7s ease both;
    }

    .hero:after {
        content: "✦";
        position: absolute;
        right: 38px;
        top: 4px;
        font: 130px 'Cormorant Garamond', Georgia, serif;
        color: rgba(236, 211, 164, .14);
        transform: rotate(12deg);
    }

    .hero-kicker {
        color: #ddc495;
        font: 500 10px 'DM Mono', monospace;
        letter-spacing: 2.3px;
        text-transform: uppercase;
    }

    .hero-title {
        color: #fff8ec;
        font: 600 clamp(42px, 5vw, 64px)/.98 'Cormorant Garamond', Georgia, serif;
        letter-spacing: -1px;
        margin: 12px 0;
    }

    .hero-copy {
        color: #e7dac6;
        max-width: 760px;
        font-size: 14px;
        line-height: 1.8;
    }

    .hero-chips { margin-top: 18px; }

    .chip {
        display: inline-block;
        margin: 0 7px 7px 0;
        padding: 4px 11px;
        border: 1px solid rgba(236,211,164,.38);
        border-radius: 999px;
        color: #ecd3a4;
        font: 500 10px 'DM Mono', monospace;
        letter-spacing: 1px;
        text-transform: uppercase;
        background: rgba(255,255,255,.05);
    }

    .chip-dark {
        color: var(--walnut);
        border-color: rgba(80,55,37,.28);
        background: rgba(189,150,91,.14);
    }

    .metric-tile {
        position: relative;
        overflow: hidden;
        background: rgba(255, 251, 242, .83);
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 16px 18px;
        min-height: 104px;
        box-shadow: 0 5px 15px rgba(74, 54, 34, .06);
        transition: transform .18s ease, box-shadow .18s ease;
        animation: rise .6s ease both;
    }

    .metric-tile:before {
        content: "";
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
        background: var(--tile-accent, #bd965b);
    }

    .metric-tile:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px rgba(74, 54, 34, .12);
    }

    .metric-label {
        color: #766653;
        font: 10px 'DM Mono', monospace;
        letter-spacing: 1.1px;
        text-transform: uppercase;
    }

    .metric-value {
        color: var(--walnut);
        font: 700 31px 'Cormorant Garamond', Georgia, serif;
        margin-top: 5px;
    }

    .metric-sub {
        color: #8a7860;
        font: 11px 'DM Mono', monospace;
        margin-top: 1px;
    }

    .paper-panel {
        background: rgba(255, 251, 242, .75);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 7px 20px rgba(74, 54, 34, .045);
        animation: rise .6s ease both;
    }

    .insight-card {
        background: linear-gradient(145deg, rgba(255,251,242,.92), rgba(244,233,213,.8));
        border: 1px solid var(--line);
        border-top: 3px solid var(--card-accent, #bd965b);
        border-radius: 14px;
        padding: 15px 18px;
        min-height: 128px;
        box-shadow: 0 7px 18px rgba(74, 54, 34, .06);
        animation: rise .7s ease both;
    }

    .insight-card .kicker {
        color: #8a7860;
        font: 10px 'DM Mono', monospace;
        letter-spacing: 1.3px;
        text-transform: uppercase;
    }

    .insight-card .headline {
        color: var(--walnut);
        font: 700 22px/1.15 'Cormorant Garamond', Georgia, serif;
        margin: 5px 0 6px;
    }

    .insight-card .body {
        color: #5b4b3a;
        font-size: 12.5px;
        line-height: 1.55;
    }

    .section-head { margin: 4px 0 10px; }

    .section-kicker {
        color: #8a7860;
        font: 500 10px 'DM Mono', monospace;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .section-title {
        color: var(--walnut);
        font: 700 28px/1.1 'Cormorant Garamond', Georgia, serif;
    }

    .note {
        background: rgba(255, 250, 238, .84);
        border-left: 3px solid var(--brass);
        border-radius: 0 10px 10px 0;
        padding: 13px 16px;
        color: #514333;
        font-size: 13px;
    }

    .verdict {
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 13px;
        color: #3d3226;
        border: 1px solid var(--line);
        background: rgba(255,251,242,.8);
    }

    .verdict b { color: var(--walnut); }

    .stButton > button {
        border: 1px solid #65462f;
        border-radius: 10px;
        color: #fff8eb;
        background: linear-gradient(130deg, #795538, #493326);
        font-weight: 700;
    }

    .stButton > button:hover {
        color: #fff8eb;
        border-color: #bd965b;
        box-shadow: 0 5px 15px rgba(80, 54, 36, .17);
    }

    /* Keep text readable even when the viewer's browser/Streamlit theme is dark */
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
    [data-testid="stMain"] [data-testid="stCaptionContainer"],
    [data-testid="stMain"] [data-testid="stCaptionContainer"] p,
    [data-testid="stMain"] [data-testid="stExpander"] summary,
    [data-testid="stMain"] [data-testid="stExpander"] summary p,
    [data-testid="stMain"] label,
    [data-testid="stMain"] label p,
    section.main [data-testid="stMarkdownContainer"] p,
    section.main label p {
        color: #3a2f24 !important;
    }

    [data-testid="stMain"] .stButton > button [data-testid="stMarkdownContainer"] p,
    [data-testid="stMain"] .stButton > button p {
        color: #fff8eb !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        flex-wrap: wrap;
        border-bottom: none;
        padding: 4px 2px 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 38px;
        padding: 0 20px;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: rgba(255,251,242,.7);
        transition: all .18s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        border-color: var(--brass);
        box-shadow: 0 3px 10px rgba(74,54,34,.10);
    }

    .stTabs [data-baseweb="tab"] p {
        font-weight: 700;
        color: #6b5640;
        font-size: 13px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(130deg, #795538, #493326);
        border-color: #65462f;
        box-shadow: 0 5px 14px rgba(80,54,36,.22);
    }

    .stTabs [aria-selected="true"] p { color: #fff8eb !important; }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

    [data-testid="stMetric"] {
        background: rgba(255,251,242,.8);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 12px 16px;
        box-shadow: 0 5px 14px rgba(74,54,34,.05);
    }

    [data-testid="stMetricLabel"] p {
        font: 10px 'DM Mono', monospace !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #766653;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Cormorant Garamond', Georgia, serif;
        color: var(--walnut);
        font-weight: 700;
    }

    [data-testid="stChatMessage"] {
        background: rgba(255, 251, 242, .76);
        border: 1px solid rgba(77, 55, 35, .14);
        border-radius: 14px;
        margin-bottom: 10px;
    }

    [data-testid="stExpander"] {
        background: rgba(255, 251, 242, .55);
        border: 1px solid var(--line);
        border-radius: 12px;
    }

    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Plotly theme + display helpers
# ------------------------------------------------------------
PALETTE = [
    "#65452e", "#bd965b", "#69765b", "#9b6650",
    "#6f8796", "#c9b48a", "#3d5a50", "#a0522d",
]
LEVEL_RAMP = ["#4f6b5a", "#8a9a5b", "#bd965b", "#b5653a", "#7d2e2a"]
SCALE_WARM = [[0, "#faf5eb"], [0.5, "#d9b779"], [1, "#503725"]]
SCALE_DIV = [[0, "#3d5a50"], [0.5, "#faf5eb"], [1, "#8c3b2e"]]
SCALE_SEQ_GREEN = [[0, "#faf5eb"], [0.5, "#a9b78b"], [1, "#3d5a50"]]
SERIF = "Cormorant Garamond, Georgia, serif"
SANS = "Manrope, sans-serif"

_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters
_fig_counter = itertools.count()


def polish(fig, title=None, height=420, legend=True):
    """Apply the CancerLens plotting theme to any Plotly figure."""
    layout = dict(
        font=dict(family=SANS, size=12, color="#3a2f24"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fbf7ee",
        colorway=PALETTE,
        height=height,
        margin=dict(l=48, r=24, t=64 if title else 24, b=44),
        showlegend=legend,
        legend=dict(
            bgcolor="rgba(255,251,242,.75)",
            bordercolor="rgba(75,53,34,.18)",
            borderwidth=1,
        ),
        hoverlabel=dict(bgcolor="#fffaf0", font=dict(family=SANS, color="#2d261d")),
    )
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(family=SERIF, size=22, color="#503725"),
            x=0.01,
        )
    fig.update_layout(**layout)
    fig.update_xaxes(
        gridcolor="rgba(128,102,71,.15)",
        zeroline=False,
        linecolor="#cabb9f",
    )
    fig.update_yaxes(
        gridcolor="rgba(128,102,71,.15)",
        zeroline=False,
        linecolor="#cabb9f",
    )
    return fig


def show_fig(fig):
    key = f"cl_fig_{next(_fig_counter)}"
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def _new_width_api() -> bool:
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
        return (major, minor) >= (1, 50)
    except Exception:
        return False


NEW_WIDTH_API = _new_width_api()


def show_df(frame, **kwargs):
    if NEW_WIDTH_API:
        st.dataframe(frame, width="stretch", **kwargs)
    else:
        st.dataframe(frame, use_container_width=True, **kwargs)


def metric_tile(label: str, value: str, sub: str | None = None, accent: str | None = None):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    style = f' style="--tile-accent:{accent};"' if accent else ""
    st.markdown(
        f"""
        <div class="metric-tile"{style}>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(kicker: str, headline: str, body: str, accent: str = "#bd965b"):
    st.markdown(
        f"""
        <div class="insight-card" style="--card-accent:{accent};">
            <div class="kicker">{kicker}</div>
            <div class="headline">{headline}</div>
            <div class="body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_head(kicker: str, title: str, caption: str | None = None):
    st.markdown(
        f"""
        <div class="section-head">
            <div class="section-kicker">{kicker}</div>
            <div class="section-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


def bounded_slider(label: str, low: int, high: int, value: int, **kwargs):
    """st.slider that degrades gracefully when the allowed range collapses."""
    if high <= low:
        return low
    return st.slider(label, low, high, int(min(max(value, low), high)), **kwargs)


def verdict(html: str):
    st.markdown(f'<div class="verdict">{html}</div>', unsafe_allow_html=True)


def fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def stars(p) -> str:
    if p is None or np.isnan(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def neglog10(p):
    return -np.log10(np.clip(np.asarray(p, dtype=float), 1e-300, 1.0))


def effect_label(value: float, thresholds=(0.01, 0.06, 0.14)) -> str:
    small, medium, large = thresholds
    if value < small:
        return "negligible"
    if value < medium:
        return "small"
    if value < large:
        return "medium"
    return "large"


# ------------------------------------------------------------
# Data and app helpers
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_excel_bytes(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name)


@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def get_excel_sheets(file_bytes: bytes) -> list[str]:
    workbook = pd.ExcelFile(BytesIO(file_bytes))
    return workbook.sheet_names


def normalize_name(value: str) -> str:
    return "".join(
        character.lower()
        for character in str(value)
        if character.isalnum()
    )


def clean_column(name) -> str:
    text = str(name).strip().replace("_", " ")
    return text if text.isupper() else text.title()


def detect_target(frame: pd.DataFrame):
    accepted = {
        "level",
        "cancerlevel",
        "target",
        "diagnosis",
        "label",
        "class",
        "outcome",
    }
    for column in frame.columns:
        if normalize_name(column) in accepted:
            return column
    return None


def find_column(frame: pd.DataFrame, keywords: set[str]):
    for column in frame.columns:
        if normalize_name(column) in keywords:
            return column
    return None


ORDINAL_HINTS = [
    ["none", "low", "medium", "high"],
    ["low", "moderate", "high"],
    ["low", "medium", "high"],
    ["mild", "moderate", "severe"],
    ["stage i", "stage ii", "stage iii", "stage iv"],
    ["i", "ii", "iii", "iv"],
]


def ordered_levels(series: pd.Series):
    """Return (ordered labels, is_ordinal) for a categorical target."""
    labels = [str(v) for v in pd.unique(series.dropna())]
    lowered = {label.lower().strip(): label for label in labels}
    if len(lowered) >= 2:
        for hint in ORDINAL_HINTS:
            if set(lowered) <= set(hint):
                return [lowered[h] for h in hint if h in lowered], True
    try:
        return sorted(labels, key=float), True
    except ValueError:
        return sorted(labels), False


def class_color_map(order: list[str], ordinal: bool) -> dict:
    if ordinal and len(order) > 1:
        picks = np.linspace(0, len(LEVEL_RAMP) - 1, len(order)).round().astype(int)
        return {label: LEVEL_RAMP[i] for label, i in zip(order, picks)}
    return {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(order)}


def style_axis(ax):  # kept for backwards compatibility with matplotlib snippets
    ax.set_facecolor("#fbf7ee")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cabb9f")
    ax.spines["bottom"].set_color("#cabb9f")
    ax.tick_params(colors="#554635")
    ax.grid(axis="y", alpha=.17, color="#806647")


def local_dataset():
    candidates = [
        Path("data/cancer patient data sets.xlsx"),
        Path("data/cancer_patient_data.xlsx"),
        Path("data/cancer_patient_data_sets.xlsx"),
        Path("cancer patient data sets.xlsx"),
        Path("cancer_patient_data_sets.xlsx"),
        Path("data/cancer_patient_data.csv"),
    ]

    for path in candidates:
        if path.exists():
            try:
                if path.suffix.lower() == ".csv":
                    return pd.read_csv(path)
                return pd.read_excel(path)
            except Exception:
                continue

    return None


# ------------------------------------------------------------
# Statistical engine
# ------------------------------------------------------------
def adjust_pvalues(pvalues, method: str = "Holm") -> np.ndarray:
    """Holm / Bonferroni / Benjamini-Hochberg adjusted p-values."""
    p = np.nan_to_num(np.asarray(pvalues, dtype=float), nan=1.0)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty(n)
    if method.startswith("Bonf"):
        adjusted = np.minimum(p * n, 1.0)
    elif method.startswith("Benjamini"):
        ranked = p[order] * n / (np.arange(n) + 1)
        monotone = np.minimum.accumulate(ranked[::-1])[::-1]
        adjusted[order] = np.minimum(monotone, 1.0)
    else:  # Holm step-down
        stepped = (n - np.arange(n)) * p[order]
        adjusted[order] = np.minimum(np.maximum.accumulate(stepped), 1.0)
    return adjusted


def scaled_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    matrix = frame[list(features)].apply(pd.to_numeric, errors="coerce")
    imputed = SimpleImputer(strategy="median").fit_transform(matrix)
    return StandardScaler().fit_transform(imputed)


def eta_squared(groups) -> float:
    pooled = np.concatenate(groups)
    grand = pooled.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((pooled - grand) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def cramers_v(table: pd.DataFrame) -> float:
    """Bias-corrected Cramér's V (Bergsma, 2013)."""
    observed = table.to_numpy(dtype=float)
    n = observed.sum()
    if n <= 1 or min(observed.shape) < 2:
        return 0.0
    chi2 = stats.chi2_contingency(observed, correction=False)[0]
    r, k = observed.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def group_values(frame, feature, target, order):
    values = pd.to_numeric(frame[feature], errors="coerce")
    labels = frame[target].astype(str)
    return {
        label: values[(labels == label) & values.notna()].to_numpy(float)
        for label in order
    }


@st.cache_data(show_spinner=False)
def feature_evidence(frame, features, target, order):
    """One row per feature: omnibus tests, effect sizes, trend and association."""
    mapping = {label: i for i, label in enumerate(order)}
    group = frame[target].astype(str)
    code = group.map(mapping)
    rows = []

    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        keep = (values.notna() & code.notna()).to_numpy()
        v = values.to_numpy(float)[keep]
        c = code.to_numpy(float)[keep]
        g = group.to_numpy()[keep]

        if len(v) < 10 or np.ptp(v) == 0:
            continue

        groups = [v[g == label] for label in order]
        groups = [arr for arr in groups if len(arr) > 1]
        if len(groups) < 2:
            continue

        n = len(v)
        h_stat, p_kw = stats.kruskal(*groups)
        f_stat, p_f = stats.f_oneway(*groups)
        eta2 = eta_squared(groups)
        eps2 = float(np.clip(h_stat * (n + 1) / (n ** 2 - 1), 0, 1))
        cohen_f = float(np.sqrt(eta2 / (1 - eta2))) if eta2 < 0.999999 else 9.99
        tau, p_tau = stats.kendalltau(v, c)
        rho, p_rho = stats.spearmanr(v, c)

        binned = (
            pd.Series(v)
            if len(np.unique(v)) <= 12
            else pd.qcut(pd.Series(v), 5, duplicates="drop")
        )
        table = pd.crosstab(binned.astype(str).to_numpy(), g)
        if min(table.shape) >= 2:
            p_chi = stats.chi2_contingency(table)[1]
            v_cram = cramers_v(table)
        else:
            p_chi, v_cram = np.nan, 0.0

        discrete = bool(np.allclose(v, np.round(v)) and v.min() >= 0)
        mi = mutual_info_classif(
            v.reshape(-1, 1),
            c.astype(int),
            discrete_features=discrete,
            random_state=0,
        )[0]
        entropy = stats.entropy(np.bincount(c.astype(int)))
        mi_norm = float(mi / entropy) if entropy > 0 else 0.0

        rows.append(
            {
                "Feature": feature,
                "n": n,
                "KW_H": float(h_stat),
                "KW_p": float(p_kw),
                "Epsilon2": eps2,
                "ANOVA_F": float(f_stat),
                "ANOVA_p": float(p_f),
                "Eta2": eta2,
                "Cohen_f": cohen_f,
                "Tau": float(0 if np.isnan(tau) else tau),
                "Tau_p": float(1 if np.isnan(p_tau) else p_tau),
                "Rho": float(0 if np.isnan(rho) else rho),
                "Rho_p": float(1 if np.isnan(p_rho) else p_rho),
                "CramersV": v_cram,
                "Chi2_p": float(p_chi) if not np.isnan(p_chi) else np.nan,
                "MI_norm": mi_norm,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["AbsTau"] = result["Tau"].abs()
        result["Composite"] = (
            result[["Epsilon2", "CramersV", "MI_norm", "AbsTau"]]
            .rank(pct=True)
            .mean(axis=1)
        )
    return result


@st.cache_data(show_spinner=False)
def normality_tables(frame, features, target, order):
    """Per-group normality (Shapiro-Wilk, D'Agostino K²) and Brown-Forsythe test."""
    group_rows, variance_rows = [], []
    for feature in features:
        groups = group_values(frame, feature, target, order)
        for label, x in groups.items():
            if len(x) >= 8 and np.ptp(x) > 0:
                w_stat, p_sw = stats.shapiro(x)
                k2, p_k2 = stats.normaltest(x)
                group_rows.append(
                    {
                        "Feature": feature,
                        "Group": label,
                        "n": len(x),
                        "Skewness": float(stats.skew(x)),
                        "ExcessKurtosis": float(stats.kurtosis(x)),
                        "Shapiro_W": float(w_stat),
                        "Shapiro_p": float(p_sw),
                        "DAgostino_p": float(p_k2),
                    }
                )
        usable = [x for x in groups.values() if len(x) > 1 and np.ptp(x) > 0]
        if len(usable) >= 2:
            l_stat, p_lev = stats.levene(*usable, center="median")
            variance_rows.append(
                {"Feature": feature, "BrownForsythe_W": float(l_stat), "BrownForsythe_p": float(p_lev)}
            )
    return pd.DataFrame(group_rows), pd.DataFrame(variance_rows)


@st.cache_data(show_spinner=False)
def posthoc_pairs(frame, feature, target, order, n_boot, seed):
    """Pairwise Mann-Whitney / Welch tests with Cliff's delta and Hedges' g (bootstrap CI)."""
    rng = np.random.default_rng(seed)
    groups = group_values(frame, feature, target, order)
    rows = []
    for a, b in itertools.combinations(order, 2):
        x, y = groups[a], groups[b]
        if len(x) < 3 or len(y) < 3:
            continue
        nx, ny = len(x), len(y)
        u_stat, p_mw = stats.mannwhitneyu(x, y, alternative="two-sided")
        cliff = 1 - 2 * u_stat / (nx * ny)  # P(y > x) - P(y < x)
        p_welch = stats.ttest_ind(y, x, equal_var=False)[1]

        correction = 1 - 3 / (4 * (nx + ny) - 9)
        pooled_sd = math.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
        g = correction * (y.mean() - x.mean()) / pooled_sd if pooled_sd > 0 else 0.0

        bx = x[rng.integers(0, nx, (n_boot, nx))]
        by = y[rng.integers(0, ny, (n_boot, ny))]
        sd_b = np.sqrt(((nx - 1) * bx.var(axis=1, ddof=1) + (ny - 1) * by.var(axis=1, ddof=1)) / (nx + ny - 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            g_boot = correction * (by.mean(axis=1) - bx.mean(axis=1)) / sd_b
        lo, hi = np.nanpercentile(g_boot, [2.5, 97.5]) if np.isfinite(g_boot).any() else (np.nan, np.nan)

        rows.append(
            {
                "Comparison": f"{b} − {a}",
                "MannWhitney_p": float(p_mw),
                "Welch_p": float(p_welch),
                "CliffsDelta": float(cliff),
                "HedgesG": float(g),
                "G_lo": float(lo),
                "G_hi": float(hi),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def correlation_matrices(frame, features, method):
    """Pairwise correlation and p-value matrices (Pearson / Spearman / Kendall)."""
    X = frame[list(features)].apply(pd.to_numeric, errors="coerce")
    fn = {"Pearson": stats.pearsonr, "Spearman": stats.spearmanr, "Kendall": stats.kendalltau}[method]
    p = len(features)
    R, P = np.eye(p), np.zeros((p, p))
    for i, j in itertools.combinations(range(p), 2):
        pair = X.iloc[:, [i, j]].dropna()
        if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
            r, pv = np.nan, np.nan
        else:
            res = fn(pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy())
            r, pv = float(res[0]), float(res[1])
        R[i, j] = R[j, i] = r
        P[i, j] = P[j, i] = pv
    return (
        pd.DataFrame(R, index=features, columns=features),
        pd.DataFrame(P, index=features, columns=features),
    )


def linkage_order(R: pd.DataFrame, method: str = "average"):
    """Hierarchical clustering of variables on 1 - |r| distance."""
    dist = 1 - np.abs(np.nan_to_num(R.to_numpy(), nan=0.0))
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    Z = sch.linkage(squareform(dist, checks=False), method=method)
    return Z, sch.leaves_list(Z)


def partial_correlation(R: np.ndarray) -> np.ndarray:
    precision = np.linalg.pinv(np.nan_to_num(R, nan=0.0))
    d = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    partial = -precision / d
    np.fill_diagonal(partial, 1.0)
    return partial


def vif_from_corr(R: np.ndarray) -> np.ndarray:
    return np.diag(np.linalg.pinv(np.nan_to_num(R, nan=0.0)))


def factorability(X: np.ndarray):
    """Bartlett's sphericity test and Kaiser-Meyer-Olkin measure."""
    n, p = X.shape
    R = np.corrcoef(X, rowvar=False)
    _, logdet = np.linalg.slogdet(R)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * logdet
    dof = p * (p - 1) / 2
    p_val = float(stats.chi2.sf(chi2, dof))
    partial = partial_correlation(R)
    mask = ~np.eye(p, dtype=bool)
    r2 = (R ** 2) * mask
    q2 = (partial ** 2) * mask
    kmo_total = float(r2.sum() / (r2.sum() + q2.sum()))
    kmo_each = r2.sum(axis=0) / (r2.sum(axis=0) + q2.sum(axis=0))
    return {"chi2": float(chi2), "dof": int(dof), "p": p_val, "kmo": kmo_total, "kmo_each": kmo_each}


@st.cache_data(show_spinner=False)
def two_group_resampling(frame, feature, target, group_a, group_b, statistic, n_resamples, seed):
    """Bootstrap CI and permutation test for a difference in mean / median."""
    groups = group_values(frame, feature, target, [group_a, group_b])
    x, y = groups[group_a], groups[group_b]
    fn = np.mean if statistic == "Mean" else np.median
    rng = np.random.default_rng(seed)

    observed = float(fn(y) - fn(x))
    boot = fn(y[rng.integers(0, len(y), (n_resamples, len(y)))], axis=1) - fn(
        x[rng.integers(0, len(x), (n_resamples, len(x)))], axis=1
    )

    pooled = np.concatenate([x, y])
    shuffled = pooled[rng.random((n_resamples, len(pooled))).argsort(axis=1)]
    perm = fn(shuffled[:, len(x):], axis=1) - fn(shuffled[:, : len(x)], axis=1)
    p_perm = float((1 + np.sum(np.abs(perm) >= abs(observed) - 1e-12)) / (n_resamples + 1))

    return {
        "observed": observed,
        "boot": boot,
        "perm": perm,
        "ci": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
        "p_perm": p_perm,
        "nx": len(x),
        "ny": len(y),
    }


def t_power(effect_size: float, n_per_group, alpha: float):
    n = np.asarray(n_per_group, dtype=float)
    dof = 2 * n - 2
    noncentrality = effect_size * np.sqrt(n / 2)
    critical = stats.t.ppf(1 - alpha / 2, dof)
    return 1 - stats.nct.cdf(critical, dof, noncentrality) + stats.nct.cdf(-critical, dof, noncentrality)


@st.cache_data(show_spinner=False)
def pca_bundle(frame, features, n_perm, seed):
    X = scaled_matrix(frame, features)
    n, p = X.shape
    pca = PCA(random_state=seed).fit(X)
    scores = pca.transform(X)
    eigenvalues = pca.explained_variance_

    rng = np.random.default_rng(seed)
    simulated = np.empty((n_perm, p))
    for i in range(n_perm):
        permuted = rng.permuted(X, axis=0)
        simulated[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(permuted, rowvar=False)))[::-1]
    horn = np.percentile(simulated, 95, axis=0)

    retain = 0
    while retain < p and eigenvalues[retain] > horn[retain]:
        retain += 1

    return {
        "scores": scores,
        "evr": pca.explained_variance_ratio_,
        "eig": eigenvalues,
        "horn": horn,
        "retain_horn": retain,
        "retain_kaiser": int(np.sum(eigenvalues > 1)),
        "loadings": pca.components_.T * np.sqrt(eigenvalues),
        "fact": factorability(X),
    }


@st.cache_data(show_spinner=False)
def cluster_sweep(frame, features, k_max, seed):
    X = scaled_matrix(frame, features)
    rows = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        labels = km.labels_
        row = {
            "k": k,
            "Inertia": float(km.inertia_),
            "Silhouette": float(silhouette_score(X, labels)),
            "CalinskiHarabasz": float(calinski_harabasz_score(X, labels)),
            "DaviesBouldin": float(davies_bouldin_score(X, labels)),
        }
        try:
            gm = GaussianMixture(k, covariance_type="diag", n_init=2, random_state=seed).fit(X)
            row["BIC"], row["AIC"] = float(gm.bic(X)), float(gm.aic(X))
        except Exception:
            row["BIC"], row["AIC"] = np.nan, np.nan
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def kmeans_labels(frame, features, k, seed):
    X = scaled_matrix(frame, features)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


@st.cache_data(show_spinner=False)
def factor_bundle(frame, features, k, seed):
    X = scaled_matrix(frame, features)
    fa = FactorAnalysis(n_components=k, rotation="varimax", random_state=seed).fit(X)
    loadings = fa.components_.T
    return {
        "loadings": loadings,
        "noise": fa.noise_variance_,
        "ss": (loadings ** 2).sum(axis=0),
        "scores": fa.transform(X),
        "loglik": float(fa.score(X)),
    }


@st.cache_data(show_spinner=False)
def outlier_bundle(frame, features, contamination, seed):
    X = scaled_matrix(frame, features)
    centered = X - X.mean(axis=0)
    inverse = np.linalg.pinv(np.cov(centered, rowvar=False))
    d2 = np.einsum("ij,jk,ik->i", centered, inverse, centered)
    cutoff = float(stats.chi2.ppf(0.999, X.shape[1]))
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed).fit(X)
    return {
        "d2": d2,
        "cutoff": cutoff,
        "iso_score": -iso.score_samples(X),
        "iso_flag": iso.predict(X) == -1,
    }


@st.cache_data(show_spinner=False)
def tsne_embed(frame, features, perplexity, seed):
    X = scaled_matrix(frame, features)
    perplexity = float(min(perplexity, max(5, (len(X) - 1) / 3)))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(X)


@st.cache_data(show_spinner=False)
def lda_embed(frame, features, target, order, seed):
    keep = frame[target].astype(str).isin(order).to_numpy()
    X = scaled_matrix(frame, features)[keep]
    y = frame[target].astype(str)[keep].map({label: i for i, label in enumerate(order)}).to_numpy()
    n_comp = max(1, min(2, len(order) - 1, X.shape[1]))
    lda = LinearDiscriminantAnalysis(n_components=n_comp).fit(X, y)
    return lda.transform(X), keep, lda.explained_variance_ratio_


# ------------------------------------------------------------
# Machine-learning helpers
# ------------------------------------------------------------
def make_one_hot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric: list[str], categorical: list[str]):
    transforms = []
    if numeric:
        transforms.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            )
        )
    if categorical:
        transforms.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encode", make_one_hot()),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transforms, sparse_threshold=0)


def model_zoo(seed: int) -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
        "Gaussian Naive Bayes": GaussianNB(),
        "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
        "SVM (RBF kernel)": SVC(probability=True, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=seed
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=seed
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=800, early_stopping=True, random_state=seed
        ),
    }


def split_feature_types(frame: pd.DataFrame, chosen: list[str]):
    numeric = [c for c in chosen if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in chosen if c not in numeric]
    return numeric, categorical


def nadeau_bengio(diff: np.ndarray, folds: int) -> tuple[float, float]:
    """Corrected resampled t-test for paired cross-validation scores."""
    k = len(diff)
    if k < 3:
        return 0.0, 1.0
    mean, var = float(diff.mean()), float(diff.var(ddof=1))
    if var <= 1e-18:
        return (0.0, 1.0) if abs(mean) < 1e-12 else (np.inf, 0.0)
    t_stat = mean / math.sqrt((1 / k + 1 / max(folds - 1, 1)) * var)
    return t_stat, float(2 * stats.t.sf(abs(t_stat), k - 1))


def clean_feature_names(names) -> list[str]:
    return [str(n).split("__", 1)[-1] for n in names]


def run_experiment(frame, chosen, target, order, model_names, folds, repeats, test_size, seed, progress):
    """Cross-validate every selected model, then evaluate each on a held-out split."""
    data_ = frame[chosen + [target]].dropna(subset=[target])
    X = data_[chosen]
    mapping = {label: i for i, label in enumerate(order)}
    y = data_[target].astype(str).map(mapping).to_numpy()
    n_classes = len(order)
    numeric, categorical = split_feature_types(frame, chosen)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=seed, stratify=y
    )
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    auc_name = "roc_auc" if n_classes == 2 else "roc_auc_ovr"
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "roc_auc": auc_name,
        "log_loss": "neg_log_loss",
    }

    zoo = model_zoo(seed)
    cv_scores, holdout, importance, native = {}, {}, {}, {}
    total_steps = len(model_names) * 2

    for index, name in enumerate(model_names):
        pipeline = Pipeline([("prep", build_preprocessor(numeric, categorical)), ("clf", zoo[name])])

        progress.progress(index / total_steps, text=f"Cross-validating · {name}")
        res = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, error_score=np.nan)
        cv_scores[name] = {
            "accuracy": res["test_accuracy"],
            "balanced_accuracy": res["test_balanced_accuracy"],
            "f1_macro": res["test_f1_macro"],
            "roc_auc": res["test_roc_auc"],
            "log_loss": -res["test_log_loss"],
            "fit_time": res["fit_time"],
        }

        progress.progress((index + 0.5) / total_steps, text=f"Hold-out evaluation · {name}")
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)
        holdout[name] = {"proba": proba, "pred": proba.argmax(axis=1)}

        perm = permutation_importance(
            pipeline, X_test, y_test, scoring="neg_log_loss", n_repeats=8, random_state=seed
        )
        importance[name] = pd.DataFrame(
            {"Feature": chosen, "Mean": perm.importances_mean, "SD": perm.importances_std}
        ).sort_values("Mean", ascending=False)

        clf = pipeline.named_steps["clf"]
        feature_names = clean_feature_names(pipeline.named_steps["prep"].get_feature_names_out())
        if hasattr(clf, "feature_importances_"):
            native[name] = ("importance", feature_names, np.asarray(clf.feature_importances_))
        elif hasattr(clf, "coef_"):
            native[name] = ("coefficients", feature_names, np.atleast_2d(clf.coef_))

    progress.progress(1.0, text="Experiment complete")
    return {
        "chosen": chosen,
        "order": order,
        "folds": folds,
        "repeats": repeats,
        "test_size": test_size,
        "seed": seed,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "y_test": y_test,
        "cv": cv_scores,
        "holdout": holdout,
        "importance": importance,
        "native": native,
    }


def mean_ci(values: np.ndarray, confidence: float = 0.95):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float(values.mean()) if len(values) else np.nan, np.nan, np.nan
    mean = values.mean()
    margin = stats.t.ppf((1 + confidence) / 2, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values))
    return float(mean), float(mean - margin), float(mean + margin)


@st.cache_data(show_spinner=False)
def learning_curve_data(frame, chosen, target, order, model_name, seed):
    data_ = frame[chosen + [target]].dropna(subset=[target])
    X = data_[chosen]
    y = data_[target].astype(str).map({label: i for i, label in enumerate(order)}).to_numpy()
    numeric, categorical = split_feature_types(frame, chosen)
    pipeline = Pipeline(
        [("prep", build_preprocessor(numeric, categorical)), ("clf", model_zoo(seed)[model_name])]
    )
    sizes, train_scores, val_scores = learning_curve(
        pipeline,
        X,
        y,
        train_sizes=np.linspace(0.15, 1.0, 6),
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        scoring="accuracy",
    )
    return sizes, train_scores, val_scores


# ------------------------------------------------------------
# Model Studio: results renderer (defined here, called from the page below)
# ------------------------------------------------------------
def render_model_results(bundle: dict):
    order = bundle["order"]
    n_classes = len(order)
    cv = bundle["cv"]
    names = list(cv.keys())
    folds = bundle["folds"]
    y_test = bundle["y_test"]
    colors = class_color_map(order, target_ordinal)

    # ---- leaderboard data ----
    rows = []
    for name in names:
        s = cv[name]
        acc_mean, acc_lo, acc_hi = mean_ci(s["accuracy"])
        pred = bundle["holdout"][name]["pred"]
        rows.append(
            {
                "Model": name,
                "CV accuracy": acc_mean,
                "CI low": acc_lo,
                "CI high": acc_hi,
                "SD": float(np.nanstd(s["accuracy"], ddof=1)),
                "Balanced acc.": float(np.nanmean(s["balanced_accuracy"])),
                "Macro F1": float(np.nanmean(s["f1_macro"])),
                "ROC AUC": float(np.nanmean(s["roc_auc"])),
                "Log loss": float(np.nanmean(s["log_loss"])),
                "Hold-out acc.": float((pred == y_test).mean()),
                "Fit time (s)": float(np.nanmean(s["fit_time"])),
            }
        )
    board = pd.DataFrame(rows).sort_values("CV accuracy", ascending=False).reset_index(drop=True)
    champion = board.iloc[0]["Model"]

    ho = bundle["holdout"][champion]
    kappa_weights = "quadratic" if target_ordinal else None
    kappa = cohen_kappa_score(y_test, ho["pred"], weights=kappa_weights)
    mcc = matthews_corrcoef(y_test, ho["pred"])

    k1, k2, k3, k4, k5 = st.columns([1.3, 1.5, 1, 0.8, 0.9])
    best_row = board.iloc[0]
    with k1:
        metric_tile("CHAMPION", str(champion), f"{len(names)} models compared", "#503725")
    with k2:
        metric_tile(
            "CV ACCURACY",
            f"{best_row['CV accuracy']:.1%}",
            f"95% CI {best_row['CI low']:.1%} – {best_row['CI high']:.1%}",
            "#69765b",
        )
    with k3:
        metric_tile("HOLD-OUT ACCURACY", f"{best_row['Hold-out acc.']:.1%}", f"n = {bundle['n_test']}", "#bd965b")
    with k4:
        metric_tile("MCC", f"{mcc:.3f}", "Matthews corr.", "#9b6650")
    with k5:
        metric_tile("WEIGHTED κ" if target_ordinal else "COHEN’S κ", f"{kappa:.3f}", "quadratic weights" if target_ordinal else "chance-corrected", "#6f8796")

    if board.iloc[0]["CV accuracy"] >= 0.98:
        st.warning(
            "**Near-perfect separability.** Cross-validated accuracy above 98% is unusual for real clinical "
            "data. It suggests the label is almost a deterministic function of the recorded fields (a "
            "synthetic or derived label, or leakage). Treat these numbers as a property of this dataset, "
            "not as evidence of clinical performance."
        )

    (
        board_tab,
        compare_tab,
        champion_tab,
        explain_tab,
        curve_tab,
    ) = st.tabs(
        [
            "Leaderboard",
            "Statistical comparison",
            "Model diagnostics",
            "Explainability",
            "Learning curve",
        ]
    )

    # ---------------- Leaderboard ----------------
    with board_tab:
        left, right = st.columns(2)
        with left:
            bar = go.Figure(
                go.Bar(
                    x=board["Model"],
                    y=board["CV accuracy"],
                    marker=dict(color=board["CV accuracy"], colorscale=SCALE_WARM, line=dict(color="#503725", width=1)),
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=(board["CI high"] - board["CV accuracy"]).fillna(0),
                        arrayminus=(board["CV accuracy"] - board["CI low"]).fillna(0),
                        color="#503725",
                    ),
                    text=[f"{v:.1%}" for v in board["CV accuracy"]],
                    textposition="inside",
                )
            )
            lo = max(0.0, float(board["CI low"].min()) - 0.1)
            bar.update_yaxes(range=[lo, 1.02], tickformat=".0%", title="Accuracy")
            bar.update_xaxes(tickangle=-25)
            polish(bar, f"Repeated {folds}-fold CV accuracy with 95% CI", 460, legend=False)
            show_fig(bar)

        with right:
            box = go.Figure()
            for index, name in enumerate(board["Model"]):
                box.add_trace(
                    go.Box(
                        y=cv[name]["accuracy"],
                        name=name,
                        boxpoints="all",
                        jitter=0.45,
                        pointpos=0,
                        marker=dict(size=5, opacity=0.6),
                        line=dict(color=PALETTE[index % len(PALETTE)]),
                        showlegend=False,
                    )
                )
            box.update_yaxes(tickformat=".0%", title="Fold accuracy")
            box.update_xaxes(tickangle=-25)
            polish(box, "Fold-to-fold variability", 460, legend=False)
            show_fig(box)

        radar = go.Figure()
        metrics = ["CV accuracy", "Balanced acc.", "Macro F1", "ROC AUC", "Hold-out acc."]
        for index, row in board.iterrows():
            values = [row[m] for m in metrics]
            radar.add_trace(
                go.Scatterpolar(
                    r=values + values[:1],
                    theta=metrics + metrics[:1],
                    name=row["Model"],
                    fill="toself",
                    opacity=0.35,
                    line=dict(color=PALETTE[index % len(PALETTE)]),
                )
            )
        lower = max(0.0, float(board[metrics].to_numpy().min()) - 0.05)
        radar.update_layout(
            polar=dict(
                bgcolor="#fbf7ee",
                radialaxis=dict(range=[lower, 1], gridcolor="rgba(128,102,71,.25)", tickformat=".0%"),
            )
        )
        polish(radar, "Multi-metric profile", 460)
        show_fig(radar)

        try:
            config = {
                "CV accuracy": st.column_config.ProgressColumn("CV accuracy", min_value=0, max_value=1, format="%.3f"),
                "Hold-out acc.": st.column_config.ProgressColumn("Hold-out acc.", min_value=0, max_value=1, format="%.3f"),
                "ROC AUC": st.column_config.ProgressColumn("ROC AUC", min_value=0, max_value=1, format="%.3f"),
            }
            show_df(board.round(4), hide_index=True, column_config=config)
        except Exception:
            show_df(board.round(4), hide_index=True)
        st.download_button(
            "Download leaderboard (CSV)",
            board.to_csv(index=False).encode("utf-8-sig"),
            file_name="model_leaderboard.csv",
            mime="text/csv",
        )

    # ---------------- Statistical comparison ----------------
    with compare_tab:
        if len(names) < 2:
            st.info("Select at least two classifiers to compare them statistically.")
        else:
            acc = {n: np.asarray(cv[n]["accuracy"], dtype=float) for n in board["Model"]}
            ordered = list(board["Model"])
            m = len(ordered)
            delta = np.zeros((m, m))
            raw_p = np.ones((m, m))
            pair_index = list(itertools.combinations(range(m), 2))
            raw_list = []
            for i, j in pair_index:
                d = acc[ordered[i]] - acc[ordered[j]]
                delta[i, j], delta[j, i] = d.mean(), -d.mean()
                _, p = nadeau_bengio(d, folds)
                raw_list.append(p)
            adj_list = adjust_pvalues(raw_list, ADJ)
            adj_p = np.ones((m, m))
            for (i, j), p in zip(pair_index, adj_list):
                adj_p[i, j] = adj_p[j, i] = p

            text = np.empty((m, m), dtype=object)
            for i in range(m):
                for j in range(m):
                    text[i, j] = "—" if i == j else f"Δ={delta[i, j]:+.3f}<br>p={fmt_p(adj_p[i, j])}"

            heat = go.Figure(
                go.Heatmap(
                    z=delta,
                    x=ordered,
                    y=ordered,
                    text=text,
                    texttemplate="%{text}",
                    textfont=dict(size=10),
                    colorscale=SCALE_DIV,
                    zmid=0,
                    colorbar=dict(title="Δ accuracy<br>(row − col)"),
                )
            )
            heat.update_yaxes(autorange="reversed")
            polish(heat, f"Pairwise Nadeau–Bengio corrected t-tests ({ADJ.split(' ')[0]}-adjusted)", 520, legend=False)
            show_fig(heat)

            if m >= 3:
                try:
                    chi2, p_fr = stats.friedmanchisquare(*[acc[n] for n in ordered])
                    verdict(
                        f"<b>Friedman omnibus test</b> across {m} models and {len(acc[ordered[0]])} CV folds: "
                        f"χ²({m - 1}) = {chi2:.2f}, p {fmt_p(p_fr)}. "
                        + ("At least one model differs reliably." if p_fr < ALPHA else "No reliable difference among the models was detected.")
                    )
                except ValueError:
                    st.info("Friedman test is undefined when all scores are identical.")

            st.caption(
                "Resampled CV scores are correlated because training sets overlap; the Nadeau–Bengio "
                "correction inflates the variance accordingly, giving more honest p-values than a naïve paired t-test."
            )

    # ---------------- Champion diagnostics ----------------
    with champion_tab:
        selected = st.selectbox("Model to inspect", list(board["Model"]), key="diag_model")
        hold = bundle["holdout"][selected]
        proba, pred = hold["proba"], hold["pred"]
        onehot = np.eye(n_classes)[y_test]

        accuracy = float((pred == y_test).mean())
        rng = np.random.default_rng(SEED)
        correct = (pred == y_test).astype(float)
        boot = correct[rng.integers(0, len(correct), (int(N_RESAMPLES), len(correct)))].mean(axis=1)
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

        if n_classes == 2:
            auc_macro = roc_auc_score(y_test, proba[:, 1])
        else:
            auc_macro = roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=list(range(n_classes)))
        kappa_sel = cohen_kappa_score(y_test, pred, weights="quadratic" if target_ordinal else None)
        brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
        ll = log_loss(y_test, proba, labels=list(range(n_classes)))

        left, right = st.columns([1, 1.3])
        with left:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=accuracy * 100,
                    number=dict(suffix="%", valueformat=".1f", font=dict(family=SERIF, size=46, color="#503725")),
                    title=dict(text="Hold-out accuracy", font=dict(family=SERIF, size=20, color="#503725")),
                    gauge=dict(
                        axis=dict(range=[0, 100]),
                        bar=dict(color="#765337"),
                        bgcolor="#faf5eb",
                        steps=[
                            dict(range=[0, 60], color="#e6c9b8"),
                            dict(range=[60, 85], color="#ecdcb8"),
                            dict(range=[85, 100], color="#dfe5cf"),
                        ],
                        threshold=dict(line=dict(color="#8c3b2e", width=4), value=100 / n_classes),
                    ),
                )
            )
            polish(gauge, None, 300, legend=False)
            show_fig(gauge)
            st.caption(f"Red line = chance level (1/{n_classes}). Bootstrap 95% CI: [{ci_lo:.1%}, {ci_hi:.1%}].")

        with right:
            a, b, c, d = st.columns(4)
            a.metric("Balanced acc.", f"{balanced_accuracy_score(y_test, pred):.3f}")
            b.metric("Macro F1", f"{f1_score(y_test, pred, average='macro'):.3f}")
            c.metric("Macro AUC", f"{auc_macro:.3f}")
            d.metric("MCC", f"{matthews_corrcoef(y_test, pred):.3f}")
            a, b, c, d = st.columns(4)
            a.metric("Weighted κ" if target_ordinal else "Cohen’s κ", f"{kappa_sel:.3f}")
            b.metric("Log loss", f"{ll:.3f}")
            c.metric("Brier score", f"{brier:.3f}")
            d.metric("Test records", f"{len(y_test)}")

        left, right = st.columns(2)
        with left:
            cm = confusion_matrix(y_test, pred, labels=list(range(n_classes)))
            cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            cm_text = np.array([[f"{cm[i, j]}<br>{cm_norm[i, j]:.0%}" for j in range(n_classes)] for i in range(n_classes)])
            cm_fig = go.Figure(
                go.Heatmap(
                    z=cm_norm,
                    x=order,
                    y=order,
                    text=cm_text,
                    texttemplate="%{text}",
                    textfont=dict(size=13),
                    colorscale=SCALE_SEQ_GREEN,
                    zmin=0,
                    zmax=1,
                    showscale=False,
                )
            )
            cm_fig.update_yaxes(autorange="reversed", title="Actual")
            cm_fig.update_xaxes(title="Predicted")
            polish(cm_fig, "Confusion matrix (counts · row %)", 460, legend=False)
            show_fig(cm_fig)

        with right:
            roc_fig = go.Figure()
            for i, label in enumerate(order):
                positive = y_test == i
                if positive.sum() == 0 or positive.all():
                    continue
                fpr, tpr, _ = roc_curve(positive, proba[:, i])
                auc_i = roc_auc_score(positive, proba[:, i])
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{label} (AUC {auc_i:.3f})", line=dict(color=colors[label], width=3)))
            if n_classes > 2:
                fpr, tpr, _ = roc_curve(onehot.ravel(), proba.ravel())
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="micro-average", line=dict(color="#28231d", width=2, dash="dot")))
            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#a99a80", dash="dash"), showlegend=False))
            roc_fig.update_xaxes(title="False positive rate")
            roc_fig.update_yaxes(title="True positive rate")
            polish(roc_fig, "One-vs-rest ROC curves", 460)
            show_fig(roc_fig)

        left, right = st.columns(2)
        with left:
            pr_fig = go.Figure()
            for i, label in enumerate(order):
                positive = y_test == i
                if positive.sum() == 0:
                    continue
                precision, recall, _ = precision_recall_curve(positive, proba[:, i])
                ap = average_precision_score(positive, proba[:, i])
                pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"{label} (AP {ap:.3f})", line=dict(color=colors[label], width=3)))
            pr_fig.update_xaxes(title="Recall")
            pr_fig.update_yaxes(title="Precision", range=[0, 1.02])
            polish(pr_fig, "Precision–recall curves", 440)
            show_fig(pr_fig)

        with right:
            from sklearn.calibration import calibration_curve

            cal_fig = go.Figure()
            cal_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#a99a80", dash="dash"), name="perfect"))
            for i, label in enumerate(order):
                positive = (y_test == i).astype(int)
                if positive.sum() == 0:
                    continue
                try:
                    frac, mean_pred = calibration_curve(positive, proba[:, i], n_bins=6, strategy="quantile")
                except ValueError:
                    continue
                cal_fig.add_trace(go.Scatter(x=mean_pred, y=frac, mode="lines+markers", name=label, line=dict(color=colors[label], width=3)))
            cal_fig.update_xaxes(title="Mean predicted probability")
            cal_fig.update_yaxes(title="Observed frequency")
            polish(cal_fig, "Calibration (reliability) curves", 440)
            show_fig(cal_fig)

        report = classification_report(
            y_test,
            pred,
            labels=list(range(n_classes)),
            target_names=[str(x) for x in order],
            zero_division=0,
            output_dict=True,
        )
        report_frame = pd.DataFrame(report).T
        per_class = report_frame.loc[[str(x) for x in order], ["precision", "recall", "f1-score"]]
        rep_fig = px.imshow(per_class, text_auto=".3f", aspect="auto", color_continuous_scale=SCALE_SEQ_GREEN, zmin=0, zmax=1)
        polish(rep_fig, "Per-class precision · recall · F1", 300, legend=False)
        show_fig(rep_fig)
        with st.expander("Full classification report"):
            show_df(report_frame.round(3))

    # ---------------- Explainability ----------------
    with explain_tab:
        selected = st.selectbox("Model to explain", list(board["Model"]), key="explain_model")
        perm = bundle["importance"][selected].head(15).iloc[::-1]

        left, right = st.columns(2)
        with left:
            pfig = go.Figure(
                go.Bar(
                    x=perm["Mean"],
                    y=perm["Feature"],
                    orientation="h",
                    marker=dict(color=perm["Mean"], colorscale=SCALE_WARM, line=dict(color="#503725", width=1)),
                    error_x=dict(type="data", array=perm["SD"], color="#503725"),
                )
            )
            pfig.update_xaxes(title="Increase in log loss when the feature is shuffled (mean ± SD, 8 repeats)")
            polish(pfig, "Permutation importance (hold-out)", 520, legend=False)
            show_fig(pfig)

        with right:
            native = bundle["native"].get(selected)
            if native is None:
                st.info("This model family has no built-in importance or coefficients; permutation importance is shown instead.")
            else:
                kind, feature_names, values = native
                if kind == "importance":
                    series = pd.Series(values, index=feature_names).nlargest(15).iloc[::-1]
                    nfig = go.Figure(
                        go.Bar(
                            x=series.values,
                            y=series.index,
                            orientation="h",
                            marker=dict(color=series.values, colorscale=SCALE_SEQ_GREEN),
                        )
                    )
                    polish(nfig, "Built-in (impurity) importance", 520, legend=False)
                    show_fig(nfig)
                else:
                    rows_ = [f"{order[-1]} vs rest"] if values.shape[0] == 1 else list(order)
                    coef = pd.DataFrame(values, index=rows_, columns=feature_names)
                    top = coef.abs().max(axis=0).nlargest(15).index
                    limit = float(np.abs(coef[top].to_numpy()).max())
                    cfig = px.imshow(
                        coef[top].T,
                        text_auto=".2f",
                        aspect="auto",
                        color_continuous_scale=SCALE_DIV,
                        zmin=-limit,
                        zmax=limit,
                    )
                    polish(cfig, "Standardised coefficients by class", 520, legend=False)
                    show_fig(cfig)

        # consensus view across all models
        consensus = pd.DataFrame(
            {name: frame_.set_index("Feature")["Mean"] for name, frame_ in bundle["importance"].items()}
        )
        ranks = consensus.rank(ascending=False)
        consensus_order = ranks.mean(axis=1).sort_values().index
        norm = consensus.div(consensus.max().replace(0, 1), axis=1).loc[consensus_order]
        cons_fig = go.Figure(
            go.Heatmap(
                z=norm.T.to_numpy(),
                x=norm.index,
                y=norm.columns,
                colorscale=SCALE_WARM,
                text=ranks.loc[consensus_order].T.to_numpy().astype(int),
                texttemplate="%{text}",
                textfont=dict(size=9),
                showscale=False,
                hovertemplate="%{y} · %{x}<br>rank %{text}<extra></extra>",
            )
        )
        cons_fig.update_xaxes(tickangle=-45)
        polish(cons_fig, "Importance consensus across models (cell text = rank; colour = relative importance)", 120 + 60 * len(consensus.columns), legend=False)
        show_fig(cons_fig)

        if len(consensus.columns) >= 2:
            rho = consensus.corr(method="spearman")
            st.caption(
                "Mean Spearman rank agreement of permutation importances between models: "
                f"{rho.to_numpy()[np.triu_indices(len(rho), k=1)].mean():.2f}."
            )

    # ---------------- Learning curve ----------------
    with curve_tab:
        selected = st.selectbox("Model", list(board["Model"]), key="curve_model")
        if st.checkbox("Compute learning curve (5-fold, six training sizes)", key="curve_toggle"):
            with st.spinner("Fitting models on growing training sets…"):
                sizes, train_scores, val_scores = learning_curve_data(
                    data, list(bundle["chosen"]), target_column, list(order), selected, SEED
                )
            fig = go.Figure()
            for scores, name, color in [(train_scores, "Training", "#765337"), (val_scores, "Validation", "#69765b")]:
                mean, sd = scores.mean(axis=1), scores.std(axis=1)
                fig.add_trace(go.Scatter(x=np.r_[sizes, sizes[::-1]], y=np.r_[mean + sd, (mean - sd)[::-1]], fill="toself", fillcolor="rgba(118,83,55,.15)" if name == "Training" else "rgba(105,118,91,.18)", line=dict(width=0), hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=sizes, y=mean, mode="lines+markers", name=name, line=dict(color=color, width=3)))
            fig.update_xaxes(title="Training records")
            fig.update_yaxes(title="Accuracy", tickformat=".0%")
            polish(fig, f"Learning curve · {selected}", 460)
            show_fig(fig)
            gap = float(train_scores[-1].mean() - val_scores[-1].mean())
            verdict(
                f"<b>Bias–variance reading.</b> At full size the training–validation gap is {gap:+.3f}. "
                + (
                    "A visible gap points to variance (over-fitting); more data or regularisation may help."
                    if gap > 0.05
                    else "A small gap with a flat validation curve suggests the model has saturated the available signal."
                )
            )
        else:
            st.info("Tick the box to compute the curve (cached afterwards).")

    st.caption(
        "A single dataset cannot establish clinical validity. These models are for coursework, not clinical use."
    )


# ------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="font:700 30px 'Cormorant Garamond',serif;
                letter-spacing:2px;color:#f6ead5;">CANCERLENS</div>
    <div style="font:italic 14px 'IM Fell English',serif;
                color:#ddc69f;">An atlas of patterns and questions</div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.write("")

PAGES = [
    "Welcome",
    "Explore the Data",
    "Statistical Lab",
    "Multivariate Atlas",
    "Model Studio",
    "Patient Q&A",
    "Field Notes",
]
page = st.sidebar.radio("CHOOSE A CHAPTER", PAGES)

# Global analysis settings used by the statistical chapters.
if page in {"Welcome", "Explore the Data", "Statistical Lab", "Multivariate Atlas", "Model Studio"}:
    st.sidebar.divider()
    st.sidebar.markdown("### Analysis settings")
    ALPHA = st.sidebar.select_slider(
        "Significance level α",
        options=[0.10, 0.05, 0.01, 0.001],
        value=0.05,
    )
    ADJ = st.sidebar.selectbox(
        "Multiple-testing correction",
        ["Holm", "Bonferroni", "Benjamini–Hochberg (FDR)"],
    )
    SEED = int(
        st.sidebar.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
    )
    N_RESAMPLES = st.sidebar.select_slider(
        "Bootstrap / permutation draws",
        options=[500, 1000, 2000, 5000, 10000],
        value=2000,
    )
else:
    ALPHA, ADJ, SEED, N_RESAMPLES = 0.05, "Holm", 42, 2000

# API key field is shown only when the chat page is selected.
api_key = ""
api_model = "gpt-4.1-mini"

if page == "Patient Q&A":
    st.sidebar.divider()
    st.sidebar.markdown("### Conversation settings")
    api_key = st.sidebar.text_input(
        "OpenAI API key",
        type="password",
        placeholder="Paste your API key",
        help="The key is used for this session and is not written into app.py.",
    )
    api_model = st.sidebar.text_input(
        "API model",
        value="gpt-4.1-mini",
        help="Enter a model enabled for your OpenAI API project.",
    )
    st.sidebar.caption("API usage may incur charges.")

# ------------------------------------------------------------
# Prominent upload desk
# ------------------------------------------------------------
st.markdown(
    """
    <div class="masthead">
        <div class="masthead-name">CANCERLENS</div>
        <div class="masthead-note"><span class="live-dot"></span>Patient data · Statistical atlas · 02</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    upload_left, upload_right = st.columns([3, 1])

    with upload_left:
        st.markdown("### Dataset desk")
        st.caption(
            "Upload an Excel workbook or CSV. Excel workbooks with multiple "
            "sheets will show a sheet selector."
        )

    with upload_right:
        st.caption("Accepted files")
        st.markdown("`XLSX` · `XLS` · `CSV`")

    uploaded_file = st.file_uploader(
        "Choose your patient dataset",
        type=["xlsx", "xls", "csv"],
        key="patient_dataset_upload",
        help="After selection, the app will confirm the filename and load the data.",
        label_visibility="visible",
    )

data = None
data_source = "No file loaded"
upload_error = None
target_column = None

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".csv"):
            data = read_csv_bytes(file_bytes)
            data_source = uploaded_file.name
        else:
            sheets = get_excel_sheets(file_bytes)

            if not sheets:
                raise ValueError("This workbook contains no readable sheets.")

            selected_sheet = st.selectbox(
                "Choose a worksheet",
                options=sheets,
                key="excel_sheet_choice",
            )
            data = read_excel_bytes(file_bytes, selected_sheet)
            data_source = f"{uploaded_file.name} · {selected_sheet}"

        if data is not None and data.empty:
            raise ValueError("The selected file or worksheet contains no rows.")

    except Exception as error:
        data = None
        upload_error = str(error)

elif "dataset_override" in st.session_state:
    data = st.session_state.dataset_override
    data_source = st.session_state.get("dataset_source", "Loaded dataset")

else:
    data = local_dataset()
    if data is not None:
        data_source = "Bundled project file"

if upload_error:
    st.error(
        "The file could not be loaded. For Excel, confirm that the workbook "
        "is not password-protected and that `openpyxl` is installed. "
        f"Details: {upload_error}"
    )
elif data is not None:
    cleaned = [clean_column(column) for column in data.columns]
    if len(set(cleaned)) != len(cleaned):
        cleaned = [str(column).strip() for column in data.columns]
    data.columns = cleaned
    target_column = detect_target(data)

    st.success(
        f"Dataset ready: **{data_source}** · "
        f"**{len(data):,} rows** · **{len(data.columns)} fields**"
    )
    with st.expander("Preview loaded data", expanded=False):
        show_df(data.head(8), hide_index=True)
else:
    st.info(
        "No dataset is loaded yet. Choose an Excel or CSV file above. "
        "The page navigation remains available while you upload."
    )


# ------------------------------------------------------------
# Shared analysis context
# ------------------------------------------------------------
ID_NAMES = {"id", "patientid", "index", "unnamed0"}
id_fields: list[str] = []
numeric_features: list[str] = []
class_order: list[str] = []
target_ordinal = False
class_colors: dict = {}
target_ok = False

if data is not None:
    id_fields = [c for c in data.columns if normalize_name(c) in ID_NAMES]
    numeric_features = [
        column
        for column in data.select_dtypes(include=np.number).columns
        if column != target_column
        and column not in id_fields
        and data[column].nunique(dropna=True) > 1
    ]
    if target_column:
        class_order, target_ordinal = ordered_levels(data[target_column])
        class_colors = class_color_map(class_order, target_ordinal)
        target_ok = len(class_order) >= 2 and len(numeric_features) >= 2


def need_target():
    """Stop the page with a helpful message when the analysis cannot run."""
    if data is None:
        st.warning("Upload a dataset at the Dataset desk above.")
        st.stop()
    if not target_ok:
        st.warning(
            "This chapter needs a categorical target field (for example `Level`, "
            "`Diagnosis`, `Label`, `Class` or `Outcome`) with at least two groups "
            "and two numeric features."
        )
        st.stop()


def plot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy with a string target and an ordinal code column for plotting."""
    out = frame.copy()
    out[target_column] = out[target_column].astype(str)
    out["_code"] = out[target_column].map({label: i for i, label in enumerate(class_order)})
    return out


def evidence_table(frame: pd.DataFrame) -> pd.DataFrame:
    ev = feature_evidence(frame, list(numeric_features), target_column, list(class_order)).copy()
    if not ev.empty:
        ev["KW_p_adj"] = adjust_pvalues(ev["KW_p"].to_numpy(), ADJ)
        ev["Tau_p_adj"] = adjust_pvalues(ev["Tau_p"].to_numpy(), ADJ)
    return ev


# ------------------------------------------------------------
# Welcome dashboard
# ------------------------------------------------------------
if page == "Welcome":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">A DATA ATLAS · CANCER PATIENT DATA</div>
            <div class="hero-title">Patterns in the record.<br>Evidence for the questions.</div>
            <div class="hero-copy">
                A visual studio for cohort exploration, rigorous hypothesis testing,
                multivariate structure discovery, and cross-validated classification.
                Built for learning from data — never for individual diagnosis.
            </div>
            <div class="hero-chips">
                <span class="chip">Kruskal–Wallis</span>
                <span class="chip">Effect sizes</span>
                <span class="chip">Permutation tests</span>
                <span class="chip">PCA · Horn's parallel analysis</span>
                <span class="chip">Factor analysis</span>
                <span class="chip">t-SNE</span>
                <span class="chip">Repeated CV</span>
                <span class="chip">Nadeau–Bengio test</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if data is None:
        st.markdown(
            """
            <div class="note">
                Start at the Dataset desk above. Once the file loads, the
                overview, statistical lab, multivariate atlas, model studio
                and field inventory will use the uploaded data.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    missing_total = int(data.isna().sum().sum())
    completeness = 1 - missing_total / max(data.size, 1)
    counts = (
        data[target_column].fillna("Missing").astype(str).value_counts()
        if target_column
        else pd.Series(dtype=int)
    )
    evenness = (
        float(stats.entropy(counts.to_numpy()) / math.log(len(counts)))
        if len(counts) > 1
        else float("nan")
    )

    ev = evidence_table(data) if target_ok else pd.DataFrame()
    top_driver = ev.sort_values("Composite", ascending=False).iloc[0]["Feature"] if not ev.empty else "—"

    cols = st.columns(6)
    tiles = [
        ("RECORDS", f"{len(data):,}", f"{data.duplicated().sum()} duplicate rows", "#765337"),
        ("NUMERIC FEATURES", f"{len(numeric_features)}", f"{len(data.columns)} fields in total", "#bd965b"),
        ("COMPLETENESS", f"{completeness:.1%}", f"{missing_total:,} missing cells", "#69765b"),
        ("TARGET", str(target_column or "Not found"), f"{len(class_order)} label groups" if class_order else "—", "#9b6650"),
        ("CLASS EVENNESS", "—" if np.isnan(evenness) else f"{evenness:.2f}", "Pielou J′ (1 = balanced)", "#6f8796"),
        ("STRONGEST DRIVER", str(top_driver), "composite evidence rank", "#503725"),
    ]
    for col, (label, value, sub, accent) in zip(cols, tiles):
        with col:
            metric_tile(label, value, sub, accent)

    st.write("")

    if target_ok and not ev.empty:
        n_sig = int((ev["KW_p_adj"] < ALPHA).sum())
        best = ev.sort_values("Epsilon2", ascending=False).iloc[0]
        direction = "rise" if best["Tau"] > 0 else "fall"
        gof = stats.chisquare(counts.to_numpy()) if len(counts) > 1 else None

        c1, c2, c3 = st.columns(3)
        with c1:
            insight_card(
                "Largest group separation",
                f"{best['Feature']}",
                f"Kruskal–Wallis ε² = {best['Epsilon2']:.2f} ({effect_label(best['Epsilon2'], (0.01, 0.08, 0.26))} effect). "
                f"Scores tend to {direction} with the ordered label (Kendall τ<sub>b</sub> = {best['Tau']:+.2f}).",
                "#8c3b2e",
            )
        with c2:
            insight_card(
                f"Multiplicity-controlled ({ADJ.split(' ')[0]})",
                f"{n_sig} of {len(ev)} features",
                f"differ across label groups at α = {ALPHA:g} after correction. "
                "Statistical significance is not the same as practical importance — see effect sizes.",
                "#69765b",
            )
        with c3:
            if gof is not None:
                insight_card(
                    "Class balance (χ² goodness of fit)",
                    f"χ² = {gof.statistic:.1f}, p = {fmt_p(gof.pvalue)}",
                    "Tests whether the recorded label groups are equally frequent. "
                    "Class weights are used in the models to guard against imbalance.",
                    "#bd965b",
                )

    st.write("")

    overview_tab, quality_tab, preview_tab = st.tabs(
        ["Cohort overview", "Data quality audit", "Record preview"]
    )

    with overview_tab:
        if target_column:
            left, right = st.columns(2)

            with left:
                donut = go.Figure(
                    go.Pie(
                        labels=counts.index,
                        values=counts.values,
                        hole=0.58,
                        sort=False,
                        marker=dict(
                            colors=[class_colors.get(str(l), "#9a8c76") for l in counts.index],
                            line=dict(color="#faf5eb", width=3),
                        ),
                        textinfo="label+percent",
                        pull=[0.03] * len(counts),
                    )
                )
                donut.add_annotation(
                    text=f"<b>{len(data):,}</b><br>records",
                    showarrow=False,
                    font=dict(family=SERIF, size=22, color="#503725"),
                )
                polish(donut, "Recorded label composition", 390, legend=False)
                show_fig(donut)

            with right:
                rows = []
                for label, k in counts.items():
                    ci = stats.binomtest(int(k), int(counts.sum())).proportion_ci(
                        confidence_level=0.95, method="wilson"
                    )
                    rows.append((str(label), k / counts.sum(), ci.low, ci.high))
                prop = pd.DataFrame(rows, columns=["Label", "Share", "Lo", "Hi"])
                bar = go.Figure(
                    go.Bar(
                        x=prop["Label"],
                        y=prop["Share"],
                        marker_color=[class_colors.get(l, "#9a8c76") for l in prop["Label"]],
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=prop["Hi"] - prop["Share"],
                            arrayminus=prop["Share"] - prop["Lo"],
                            color="#503725",
                        ),
                        text=[f"{s:.1%}" for s in prop["Share"]],
                        textposition="outside",
                    )
                )
                bar.update_yaxes(tickformat=".0%", title="Share of records")
                polish(bar, "Label shares with 95% Wilson intervals", 390, legend=False)
                show_fig(bar)

            left, right = st.columns([1.15, 1])
            with left:
                age_col = find_column(data, {"age", "ageyears"})
                if age_col and pd.api.types.is_numeric_dtype(data[age_col]):
                    hist = px.histogram(
                        plot_frame(data),
                        x=age_col,
                        color=target_column,
                        nbins=24,
                        barmode="stack",
                        color_discrete_map=class_colors,
                        category_orders={target_column: class_order},
                    )
                    hist.update_traces(marker_line_color="#faf5eb", marker_line_width=1)
                    hist.update_yaxes(title="Records")
                    polish(hist, f"{age_col} structure by label", 400)
                    show_fig(hist)
                else:
                    st.info("No age field was found for the age-structure chart.")

            with right:
                sun = data[[target_column]].copy()
                sun[target_column] = sun[target_column].astype(str)
                gender_col = find_column(data, {"gender", "sex"})
                age_col = find_column(data, {"age", "ageyears"})
                path = [target_column]
                if gender_col:
                    sun[gender_col] = data[gender_col].astype(str).radd(f"{gender_col} ")
                    path.append(gender_col)
                if age_col and pd.api.types.is_numeric_dtype(data[age_col]):
                    sun["Age band"] = pd.cut(
                        data[age_col],
                        bins=[0, 20, 30, 40, 50, 60, 200],
                        labels=["≤20", "21–30", "31–40", "41–50", "51–60", "60+"],
                    ).astype(str)
                    path.append("Age band")
                if len(path) > 1:
                    sunburst = px.sunburst(
                        sun,
                        path=path,
                        color=target_column,
                        color_discrete_map=class_colors,
                    )
                    sunburst.update_traces(insidetextorientation="radial")
                    polish(sunburst, "Cohort hierarchy", 400, legend=False)
                    show_fig(sunburst)
        else:
            st.warning(
                "No common target field was detected. The app searches for "
                "Level, Target, Diagnosis, Label, Class, or Outcome."
            )

    with quality_tab:
        q_left, q_right = st.columns([1, 2])

        with q_left:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=completeness * 100,
                    number=dict(suffix="%", font=dict(family=SERIF, size=44, color="#503725")),
                    title=dict(text="Cell completeness", font=dict(family=SERIF, size=20, color="#503725")),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickcolor="#8a7860"),
                        bar=dict(color="#69765b"),
                        bgcolor="#faf5eb",
                        borderwidth=1,
                        bordercolor="#cabb9f",
                        steps=[
                            dict(range=[0, 80], color="#e6c9b8"),
                            dict(range=[80, 95], color="#ecdcb8"),
                            dict(range=[95, 100], color="#dfe5cf"),
                        ],
                    ),
                )
            )
            polish(gauge, None, 300, legend=False)
            show_fig(gauge)

            st.metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")
            constant = [c for c in data.columns if data[c].nunique(dropna=True) <= 1]
            st.metric("Constant fields", f"{len(constant)}")

        with q_right:
            audit_rows = []
            for column in numeric_features:
                s = pd.to_numeric(data[column], errors="coerce")
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() if iqr > 0 else 0.0
                audit_rows.append(
                    {
                        "Field": column,
                        "Missing %": float(s.isna().mean() * 100),
                        "Unique": int(s.nunique()),
                        "Mean": float(s.mean()),
                        "SD": float(s.std()),
                        "Skewness": float(s.skew()),
                        "IQR outliers %": float(outliers * 100),
                        "Distribution": np.histogram(s.dropna(), bins=12)[0].tolist(),
                    }
                )
            audit = pd.DataFrame(audit_rows)
            if audit.empty:
                st.info("No numeric fields to audit.")
            else:
                try:
                    config = {
                        "Missing %": st.column_config.ProgressColumn(
                            "Missing %", min_value=0, max_value=100, format="%.1f"
                        ),
                        "IQR outliers %": st.column_config.ProgressColumn(
                            "IQR outliers %", min_value=0, max_value=25, format="%.1f"
                        ),
                        "Mean": st.column_config.NumberColumn(format="%.2f"),
                        "SD": st.column_config.NumberColumn(format="%.2f"),
                        "Skewness": st.column_config.NumberColumn(format="%.2f"),
                        "Distribution": st.column_config.BarChartColumn("Distribution"),
                    }
                    show_df(audit, hide_index=True, column_config=config, height=420)
                except Exception:
                    show_df(audit.drop(columns=["Distribution"]), hide_index=True)

        miss = data.isna().mean().mul(100).sort_values(ascending=False)
        if miss.max() > 0:
            fig = px.bar(
                x=miss.index,
                y=miss.values,
                labels={"x": "Field", "y": "Missing (%)"},
                color=miss.values,
                color_continuous_scale=SCALE_WARM,
            )
            fig.update_coloraxes(showscale=False)
            polish(fig, "Missingness by field", 320, legend=False)
            show_fig(fig)

    with preview_tab:
        show_df(data.head(25), hide_index=True)
        st.download_button(
            "Download loaded data as CSV",
            data.to_csv(index=False).encode("utf-8-sig"),
            file_name="cancer_patient_data.csv",
            mime="text/csv",
        )

    st.markdown(
        """
        <div class="note">
            Patterns in an observational dataset do not establish causation.
            This app does not estimate individual risk, diagnose cancer, or
            recommend treatment.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# Interactive EDA
# ------------------------------------------------------------
elif page == "Explore the Data":
    need_target()

    section_head(
        "CHAPTER II · EXPLORATION",
        "Explore the collection",
        "Use the filters to shape a cohort, then compare its distributions with interactive graphics.",
    )

    with st.popover("Open the filter controls", **({"width": "stretch"} if NEW_WIDTH_API else {"use_container_width": True})):
        selected_labels = []
        filter_field = "None"
        numeric_range = None

        available = sorted(data[target_column].dropna().astype(str).unique())
        selected_labels = st.multiselect(
            f"Include {target_column} labels",
            available,
            default=available,
        )

        filter_field = st.selectbox("Numeric filter field", ["None"] + numeric_features)

        if filter_field != "None":
            low = float(data[filter_field].min())
            high = float(data[filter_field].max())
            if low < high:
                numeric_range = st.slider(
                    f"{filter_field} interval",
                    min_value=low,
                    max_value=high,
                    value=(low, high),
                )

    filtered = data.copy()

    if selected_labels:
        filtered = filtered[filtered[target_column].astype(str).isin(selected_labels)]

    if numeric_range is not None:
        filtered = filtered[filtered[filter_field].between(*numeric_range)]

    if len(filtered) < 20:
        st.warning("Fewer than 20 records match the filters; widen the cohort to continue.")
        st.stop()

    active_order = [label for label in class_order if label in set(filtered[target_column].astype(str))]
    pf = plot_frame(filtered)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Records in current cohort", f"{len(filtered):,}", f"{len(filtered) / len(data):.0%} of all", delta_color="off")
    m2.metric("Label groups", f"{len(active_order)}")
    m3.metric("Numeric features", f"{len(numeric_features)}")
    m4.metric("Missing cells", f"{int(filtered.isna().sum().sum()):,}")

    (
        dist_tab,
        compare_tab,
        profile_tab,
        parallel_tab,
        matrix_tab,
        quality_tab,
    ) = st.tabs(
        [
            "Distribution",
            "Compare groups",
            "Profile radar & heatmap",
            "Parallel coordinates",
            "Scatter matrix & 3-D",
            "Missingness",
        ]
    )

    with dist_tab:
        field = st.selectbox("Choose a numeric field", numeric_features, key="dist_field")
        left, right = st.columns(2)

        with left:
            hist = px.histogram(
                pf,
                x=field,
                color=target_column,
                marginal="box",
                barmode="overlay",
                opacity=0.72,
                nbins=30,
                color_discrete_map=class_colors,
                category_orders={target_column: active_order},
            )
            polish(hist, f"Distribution of {field}", 430)
            show_fig(hist)

        with right:
            ecdf = px.ecdf(
                pf,
                x=field,
                color=target_column,
                color_discrete_map=class_colors,
                category_orders={target_column: active_order},
            )
            polish(ecdf, f"Empirical CDF of {field}", 430)
            show_fig(ecdf)

        values = pd.to_numeric(filtered[field], errors="coerce").dropna()
        if len(values) >= 8 and values.nunique() > 1:
            w_stat, p_sw = stats.shapiro(values)
            k2, p_k2 = stats.normaltest(values)
            verdict(
                f"<b>Shape diagnostics for {field}.</b> Skewness {stats.skew(values):+.2f}, "
                f"excess kurtosis {stats.kurtosis(values):+.2f}. Shapiro–Wilk W = {w_stat:.3f} "
                f"(p {fmt_p(p_sw)}); D’Agostino K² p {fmt_p(p_k2)}. "
                + (
                    "Normality is rejected, so rank-based tests are the safer default."
                    if p_sw < ALPHA
                    else "No clear departure from normality was detected."
                )
            )

        summary = (
            filtered.groupby(filtered[target_column].astype(str))[field]
            .agg(["count", "mean", "std", "median", "min", "max"])
            .reindex(active_order)
            .round(3)
        )
        show_df(summary)

    with compare_tab:
        field = st.selectbox("Feature to compare", numeric_features, key="boxplot_field")
        show_points = st.toggle("Show every observation (jittered)", value=False)
        left, right = st.columns([1.5, 1])

        with left:
            violin = px.violin(
                pf,
                x=target_column,
                y=field,
                color=target_column,
                box=True,
                points="all" if show_points else "outliers",
                color_discrete_map=class_colors,
                category_orders={target_column: active_order},
            )
            violin.update_traces(meanline_visible=True, jitter=0.35, marker=dict(size=3, opacity=0.5))
            polish(violin, f"{field} by recorded label", 450, legend=False)
            show_fig(violin)

        with right:
            rows = []
            for label in active_order:
                x = pd.to_numeric(pf.loc[pf[target_column] == label, field], errors="coerce").dropna()
                if len(x) > 1:
                    mean, lo, hi = mean_ci(x.to_numpy())
                    rows.append((label, mean, lo, hi))
            ci_frame = pd.DataFrame(rows, columns=["Label", "Mean", "Lo", "Hi"])
            line = go.Figure(
                go.Scatter(
                    x=ci_frame["Label"],
                    y=ci_frame["Mean"],
                    mode="lines+markers",
                    line=dict(color="#765337", width=3),
                    marker=dict(size=12, color=[class_colors.get(l, "#765337") for l in ci_frame["Label"]]),
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=ci_frame["Hi"] - ci_frame["Mean"],
                        arrayminus=ci_frame["Mean"] - ci_frame["Lo"],
                        color="#503725",
                        thickness=2,
                    ),
                )
            )
            polish(line, "Mean ± 95% CI (dose–response view)", 450, legend=False)
            show_fig(line)

        groups = [g for g in group_values(filtered, field, target_column, active_order).values() if len(g) > 1]
        if len(groups) >= 2 and np.ptp(np.concatenate(groups)) > 0:
            h_stat, p_kw = stats.kruskal(*groups)
            n_total = sum(len(g) for g in groups)
            eps2 = h_stat * (n_total + 1) / (n_total ** 2 - 1)
            verdict(
                f"<b>Kruskal–Wallis:</b> H = {h_stat:.2f}, p {fmt_p(p_kw)}, ε² = {eps2:.3f} "
                f"({effect_label(eps2, (0.01, 0.08, 0.26))} effect). Open the <b>Statistical Lab</b> "
                "for post-hoc contrasts, assumption checks and resampling."
            )

    with profile_tab:
        top_n = bounded_slider(
            "Features on the radar (ranked by effect size)",
            min(5, len(numeric_features)),
            min(20, len(numeric_features)),
            12,
        )
        ev_all = feature_evidence(filtered, list(numeric_features), target_column, active_order)
        ranked = ev_all.sort_values("Epsilon2", ascending=False)["Feature"].tolist()[:top_n]

        z = filtered[ranked].apply(pd.to_numeric, errors="coerce")
        z = (z - z.mean()) / z.std(ddof=0)
        z[target_column] = filtered[target_column].astype(str)
        profile = z.groupby(target_column)[ranked].mean().reindex(active_order)

        left, right = st.columns([1, 1.1])
        with left:
            radar = go.Figure()
            for label in active_order:
                r = profile.loc[label].tolist()
                radar.add_trace(
                    go.Scatterpolar(
                        r=r + r[:1],
                        theta=ranked + ranked[:1],
                        name=label,
                        fill="toself",
                        opacity=0.55,
                        line=dict(color=class_colors.get(label), width=2),
                    )
                )
            radar.update_layout(
                polar=dict(
                    bgcolor="#fbf7ee",
                    radialaxis=dict(gridcolor="rgba(128,102,71,.25)", tickfont=dict(size=9)),
                    angularaxis=dict(tickfont=dict(size=10)),
                )
            )
            polish(radar, "Standardised group profiles", 520)
            show_fig(radar)

        with right:
            limit = float(np.nanmax(np.abs(profile.to_numpy())))
            heat = px.imshow(
                profile.T,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale=SCALE_DIV,
                zmin=-limit,
                zmax=limit,
            )
            heat.update_coloraxes(colorbar=dict(title="mean z"))
            polish(heat, "Group means (z-scores)", 520, legend=False)
            show_fig(heat)

    with parallel_tab:
        defaults = (
            feature_evidence(filtered, list(numeric_features), target_column, active_order)
            .sort_values("Epsilon2", ascending=False)["Feature"]
            .tolist()[:7]
        )
        chosen = st.multiselect(
            "Dimensions (2–10)",
            numeric_features,
            default=defaults,
            max_selections=10,
        )
        if len(chosen) >= 2:
            ramp = (
                [[i / (len(active_order) - 1), class_colors[l]] for i, l in enumerate(active_order)]
                if len(active_order) > 1
                else SCALE_WARM
            )
            pc = px.parallel_coordinates(
                pf[chosen + ["_code"]].dropna(),
                dimensions=chosen,
                color="_code",
                color_continuous_scale=ramp,
            )
            pc.update_coloraxes(showscale=False)
            polish(pc, "Every patient as a line — ordered label in colour", 520, legend=False)
            pc.update_layout(margin=dict(l=70, r=50, t=90, b=40))
            show_fig(pc)
            st.caption(
                "Drag along any axis to brush a sub-population. Colours run from the lowest "
                "to the highest ordered label."
            )
        else:
            st.info("Choose at least two dimensions.")

    with matrix_tab:
        left, right = st.columns([1.1, 1])
        with left:
            splom_fields = st.multiselect(
                "Scatter-matrix fields (2–6)",
                numeric_features,
                default=numeric_features[: min(5, len(numeric_features))],
                max_selections=6,
            )
            if len(splom_fields) >= 2:
                splom = px.scatter_matrix(
                    pf,
                    dimensions=splom_fields,
                    color=target_column,
                    color_discrete_map=class_colors,
                    category_orders={target_column: active_order},
                )
                splom.update_traces(diagonal_visible=False, marker=dict(size=3.5, opacity=0.55))
                polish(splom, "Scatter-plot matrix", 620)
                show_fig(splom)

        with right:
            c3 = st.columns(3)
            ax_x = c3[0].selectbox("X", numeric_features, index=0, key="x3d")
            ax_y = c3[1].selectbox("Y", numeric_features, index=min(1, len(numeric_features) - 1), key="y3d")
            ax_z = c3[2].selectbox("Z", numeric_features, index=min(2, len(numeric_features) - 1), key="z3d")
            jitter = st.slider("Jitter (for ordinal scores)", 0.0, 0.5, 0.15, 0.05)
            rng = np.random.default_rng(SEED)
            pf3 = pf.copy()
            for axis_name in {ax_x, ax_y, ax_z}:
                pf3[axis_name] = pd.to_numeric(pf3[axis_name], errors="coerce") + rng.normal(0, jitter, len(pf3))
            scatter3d = px.scatter_3d(
                pf3,
                x=ax_x,
                y=ax_y,
                z=ax_z,
                color=target_column,
                color_discrete_map=class_colors,
                category_orders={target_column: active_order},
                opacity=0.78,
            )
            scatter3d.update_traces(marker=dict(size=3))
            polish(scatter3d, "Rotatable 3-D cohort view", 540)
            show_fig(scatter3d)

    with quality_tab:
        miss = filtered.isna().mean().mul(100).sort_values(ascending=False)
        fig = px.bar(
            x=miss.index,
            y=miss.values,
            labels={"x": "Field", "y": "Missing (%)"},
            color=miss.values,
            color_continuous_scale=SCALE_WARM,
        )
        fig.update_coloraxes(showscale=False)
        polish(fig, "Missing values by field", 380, legend=False)
        show_fig(fig)

        with st.expander("Inspect filtered records"):
            show_df(filtered, hide_index=True)


# ------------------------------------------------------------
# Statistical Lab
# ------------------------------------------------------------
elif page == "Statistical Lab":
    need_target()

    section_head(
        "CHAPTER III · INFERENCE",
        "Statistical Lab",
        "Assumption checks, omnibus and post-hoc tests, effect sizes, correlation structure, "
        "association measures and resampling — with multiplicity control.",
    )

    features_list = list(numeric_features)
    order_list = list(class_order)
    ev = evidence_table(data)

    if ev.empty:
        st.warning("No feature had enough variation to be tested.")
        st.stop()

    n_sig = int((ev["KW_p_adj"] < ALPHA).sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Features tested", f"{len(ev)}")
    k2.metric(f"Significant (α={ALPHA:g}, {ADJ.split(' ')[0]})", f"{n_sig}")
    k3.metric("Median ε²", f"{ev['Epsilon2'].median():.3f}")
    k4.metric("Largest |τb|", f"{ev['AbsTau'].max():.2f}", help="Kendall tau-b against the ordered label")

    (
        evidence_tab,
        assume_tab,
        posthoc_tab,
        corr_tab,
        assoc_tab,
        resample_tab,
        power_tab,
    ) = st.tabs(
        [
            "Evidence board",
            "Assumption checks",
            "Post-hoc contrasts",
            "Correlation structure",
            "Association measures",
            "Bootstrap & permutation",
            "Power analysis",
        ]
    )

    # ---------------- Evidence board ----------------
    with evidence_tab:
        left, right = st.columns([1.15, 1])

        with left:
            volcano = go.Figure(
                go.Scatter(
                    x=ev["Epsilon2"],
                    y=neglog10(ev["KW_p_adj"]),
                    mode="markers+text",
                    text=ev["Feature"],
                    textposition="top center",
                    textfont=dict(size=9, color="#6b5640"),
                    marker=dict(
                        size=10 + 40 * ev["AbsTau"],
                        color=ev["Tau"],
                        colorscale=SCALE_DIV,
                        cmin=-1,
                        cmax=1,
                        line=dict(color="#faf5eb", width=1.5),
                        colorbar=dict(title="τ<sub>b</sub>"),
                        opacity=0.88,
                    ),
                    customdata=np.stack([ev["KW_H"], ev["Tau"], ev["CramersV"]], axis=-1),
                    hovertemplate=(
                        "<b>%{text}</b><br>ε² = %{x:.3f}<br>−log10 p(adj) = %{y:.1f}"
                        "<br>H = %{customdata[0]:.1f}<br>τb = %{customdata[1]:+.2f}"
                        "<br>Cramér V = %{customdata[2]:.2f}<extra></extra>"
                    ),
                )
            )
            volcano.add_hline(
                y=-math.log10(ALPHA),
                line=dict(color="#8c3b2e", dash="dash"),
                annotation_text=f"α = {ALPHA:g}",
                annotation_position="bottom right",
            )
            volcano.update_xaxes(title="Effect size  ε²  (Kruskal–Wallis)")
            volcano.update_yaxes(title=f"−log10 adjusted p  ({ADJ.split(' ')[0]})")
            polish(volcano, "Volcano plot · significance vs effect size", 560, legend=False)
            show_fig(volcano)

        with right:
            ranked = ev.sort_values("Composite", ascending=True)
            lollipop = go.Figure()
            lollipop.add_trace(
                go.Scatter(
                    x=ranked["Composite"],
                    y=ranked["Feature"],
                    mode="markers",
                    marker=dict(size=11, color=ranked["Composite"], colorscale=SCALE_WARM, line=dict(color="#503725", width=1)),
                    hovertemplate="%{y}: %{x:.2f}<extra></extra>",
                )
            )
            for _, row in ranked.iterrows():
                lollipop.add_shape(
                    type="line",
                    x0=0,
                    x1=row["Composite"],
                    y0=row["Feature"],
                    y1=row["Feature"],
                    line=dict(color="#cabb9f", width=2),
                )
            lollipop.update_xaxes(title="Composite evidence score (mean percentile rank)", range=[0, 1.02])
            polish(lollipop, "Feature ranking", 560, legend=False)
            show_fig(lollipop)

        st.markdown("#### Multi-metric evidence heatmap")
        metrics = ev.set_index("Feature")[["Epsilon2", "Eta2", "CramersV", "MI_norm", "AbsTau"]].copy()
        metrics.columns = ["ε² (KW)", "η² (ANOVA)", "Cramér V", "Normalised MI", "|τb| trend"]
        metrics = metrics.loc[ev.sort_values("Composite", ascending=False)["Feature"]]
        scaled = (metrics - metrics.min()) / (metrics.max() - metrics.min()).replace(0, 1)
        heat = go.Figure(
            go.Heatmap(
                z=scaled.T.to_numpy(),
                x=metrics.index,
                y=metrics.columns,
                text=metrics.T.round(2).to_numpy(),
                texttemplate="%{text}",
                textfont=dict(size=9),
                colorscale=SCALE_WARM,
                showscale=False,
                hovertemplate="%{x}<br>%{y}: %{text}<extra></extra>",
            )
        )
        heat.update_xaxes(tickangle=-45)
        polish(heat, "Five independent lenses on the same question (colour = column-wise min–max)", 400, legend=False)
        show_fig(heat)

        table = ev.sort_values("Composite", ascending=False)[
            [
                "Feature", "KW_H", "KW_p", "KW_p_adj", "Epsilon2",
                "ANOVA_F", "ANOVA_p", "Eta2", "Cohen_f", "Tau", "Rho",
                "CramersV", "MI_norm", "Composite",
            ]
        ]
        with st.expander("Full test table (all statistics)", expanded=False):
            try:
                show_df(
                    table,
                    hide_index=True,
                    column_config={
                        "KW_p": st.column_config.NumberColumn("KW p", format="%.2e"),
                        "KW_p_adj": st.column_config.NumberColumn("KW p (adj)", format="%.2e"),
                        "ANOVA_p": st.column_config.NumberColumn("ANOVA p", format="%.2e"),
                        "Composite": st.column_config.ProgressColumn("Composite", min_value=0, max_value=1, format="%.2f"),
                        "Epsilon2": st.column_config.ProgressColumn("ε²", min_value=0, max_value=1, format="%.3f"),
                    },
                )
            except Exception:
                show_df(table, hide_index=True)
            st.download_button(
                "Download test table (CSV)",
                table.to_csv(index=False).encode("utf-8-sig"),
                file_name="feature_evidence.csv",
                mime="text/csv",
            )

    # ---------------- Assumption checks ----------------
    with assume_tab:
        norm_table, var_table = normality_tables(data, features_list, target_column, order_list)

        left, right = st.columns([1.2, 1])
        with left:
            if not norm_table.empty:
                pivot = norm_table.pivot(index="Feature", columns="Group", values="Shapiro_p").reindex(columns=order_list)
                heat = go.Figure(
                    go.Heatmap(
                        z=neglog10(pivot.to_numpy()),
                        x=pivot.columns,
                        y=pivot.index,
                        text=np.vectorize(fmt_p)(pivot.to_numpy()),
                        texttemplate="%{text}",
                        textfont=dict(size=9),
                        colorscale=SCALE_WARM,
                        colorbar=dict(title="−log10 p"),
                    )
                )
                heat.update_yaxes(autorange="reversed")
                polish(heat, "Shapiro–Wilk p-values by group (darker = stronger non-normality)", 620, legend=False)
                show_fig(heat)

        with right:
            if not var_table.empty:
                var_table = var_table.copy()
                var_table["Equal variance?"] = np.where(var_table["BrownForsythe_p"] < ALPHA, "No", "Yes")
                bf = go.Figure(
                    go.Bar(
                        x=neglog10(var_table["BrownForsythe_p"]),
                        y=var_table["Feature"],
                        orientation="h",
                        marker_color=np.where(var_table["BrownForsythe_p"] < ALPHA, "#9b6650", "#69765b"),
                        hovertemplate="%{y}<br>−log10 p = %{x:.2f}<extra></extra>",
                    )
                )
                bf.add_vline(x=-math.log10(ALPHA), line=dict(color="#503725", dash="dash"))
                bf.update_yaxes(autorange="reversed")
                bf.update_xaxes(title="−log10 p")
                polish(bf, "Brown–Forsythe test of equal variances", 620, legend=False)
                show_fig(bf)

        if not norm_table.empty:
            share = float((norm_table["Shapiro_p"] < ALPHA).mean())
            verdict(
                f"<b>Reading the checks.</b> {share:.0%} of feature × group cells reject normality at α = {ALPHA:g}, "
                f"and {int((var_table['BrownForsythe_p'] < ALPHA).sum()) if not var_table.empty else 0} features show "
                "heterogeneous variances. Discrete 1–9 ordinal scores are rarely normal, so the lab leads with "
                "rank-based tests (Kruskal–Wallis, Mann–Whitney) and reports ANOVA only as a sensitivity check."
            )

        st.markdown("#### Q–Q plots against the normal distribution")
        qq_field = st.selectbox("Feature", features_list, key="qq_field")
        qq = go.Figure()
        for label, values in group_values(data, qq_field, target_column, order_list).items():
            if len(values) > 3 and np.ptp(values) > 0:
                (osm, osr), (slope, intercept, r_fit) = stats.probplot(values, dist="norm")
                color = class_colors.get(label)
                qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name=f"{label} (r={r_fit:.3f})", marker=dict(color=color, size=5, opacity=0.65)))
                qq.add_trace(go.Scatter(x=[osm.min(), osm.max()], y=[slope * osm.min() + intercept, slope * osm.max() + intercept], mode="lines", showlegend=False, line=dict(color=color, width=1.5, dash="dot")))
        qq.update_xaxes(title="Theoretical quantiles")
        qq.update_yaxes(title="Sample quantiles")
        polish(qq, f"Q–Q plot · {qq_field}", 440)
        show_fig(qq)

        if not norm_table.empty:
            with st.expander("Per-group normality table"):
                show_df(norm_table.round(4), hide_index=True)

    # ---------------- Post-hoc contrasts ----------------
    with posthoc_tab:
        ph_field = st.selectbox(
            "Feature for post-hoc analysis",
            features_list,
            index=features_list.index(ev.sort_values("Epsilon2", ascending=False).iloc[0]["Feature"]),
            key="ph_field",
        )
        pairs = posthoc_pairs(data, ph_field, target_column, order_list, int(N_RESAMPLES), SEED).copy()

        if pairs.empty:
            st.info("Not enough observations per group for pairwise contrasts.")
        else:
            pairs["MW_p_adj"] = adjust_pvalues(pairs["MannWhitney_p"].to_numpy(), ADJ)
            pairs["Welch_p_adj"] = adjust_pvalues(pairs["Welch_p"].to_numpy(), ADJ)

            left, right = st.columns(2)
            with left:
                forest = go.Figure()
                forest.add_trace(
                    go.Scatter(
                        x=pairs["HedgesG"],
                        y=pairs["Comparison"],
                        mode="markers",
                        marker=dict(size=13, color="#503725", symbol="diamond"),
                        error_x=dict(
                            type="data",
                            symmetric=False,
                            array=pairs["G_hi"] - pairs["HedgesG"],
                            arrayminus=pairs["HedgesG"] - pairs["G_lo"],
                            color="#bd965b",
                            thickness=3,
                            width=8,
                        ),
                        hovertemplate="%{y}<br>g = %{x:.2f}<extra></extra>",
                    )
                )
                forest.add_vline(x=0, line=dict(color="#8a7860", dash="dash"))
                for edge, name in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
                    forest.add_vline(x=edge, line=dict(color="rgba(105,118,91,.35)", width=1))
                    forest.add_vline(x=-edge, line=dict(color="rgba(105,118,91,.35)", width=1))
                forest.update_xaxes(title="Hedges’ g with bootstrap 95% CI (guides at ±0.2 / 0.5 / 0.8)")
                polish(forest, "Forest plot of pairwise effects", 400, legend=False)
                show_fig(forest)

            with right:
                cliff = go.Figure(
                    go.Bar(
                        x=pairs["Comparison"],
                        y=pairs["CliffsDelta"],
                        marker_color=np.where(pairs["MW_p_adj"] < ALPHA, "#69765b", "#c9b48a"),
                        text=[f"δ={d:+.2f}<br>p={fmt_p(p)}" for d, p in zip(pairs["CliffsDelta"], pairs["MW_p_adj"])],
                        textposition="outside",
                    )
                )
                cliff.update_yaxes(title="Cliff’s δ  (P(later>earlier) − P(later<earlier))", range=[-1.1, 1.1])
                polish(cliff, "Non-parametric dominance (green = significant after correction)", 400, legend=False)
                show_fig(cliff)

            table = pairs[
                ["Comparison", "MannWhitney_p", "MW_p_adj", "Welch_p", "Welch_p_adj", "CliffsDelta", "HedgesG", "G_lo", "G_hi"]
            ]
            show_df(
                table.round(5),
                hide_index=True,
            )
            verdict(
                "<b>Interpretation.</b> Cliff’s δ is the probability-of-superiority effect size that goes with "
                "Mann–Whitney; Hedges’ g is the small-sample-corrected standardised mean difference. "
                f"p-values are adjusted across the {len(pairs)} pairwise contrasts using <b>{ADJ}</b>."
            )

    # ---------------- Correlation structure ----------------
    with corr_tab:
        c_left, c_right, c_third = st.columns(3)
        method = c_left.selectbox("Correlation coefficient", ["Spearman", "Pearson", "Kendall"], key="corr_method")
        cluster_it = c_right.toggle("Cluster variables (hierarchical, average linkage)", value=True)
        view = c_third.radio("Matrix", ["Correlation", "Partial correlation"], horizontal=True)

        R, P = correlation_matrices(data, features_list, method)
        if view == "Partial correlation":
            matrix = pd.DataFrame(partial_correlation(R.to_numpy()), index=R.index, columns=R.columns)
        else:
            matrix = R

        Z, leaves = linkage_order(R)
        names = [features_list[i] for i in leaves] if cluster_it else features_list
        shown = matrix.loc[names, names]
        pshown = P.loc[names, names]
        annotation = np.where(
            pshown.to_numpy() < 0.001, "***",
            np.where(pshown.to_numpy() < 0.01, "**", np.where(pshown.to_numpy() < 0.05, "*", "")),
        )
        if view == "Partial correlation":
            annotation = np.full_like(annotation, "")

        left, right = st.columns([1.6, 1])
        with left:
            corr_heat = go.Figure(
                go.Heatmap(
                    z=shown.to_numpy(),
                    x=names,
                    y=names,
                    zmin=-1,
                    zmax=1,
                    colorscale=SCALE_DIV,
                    text=annotation,
                    texttemplate="%{text}",
                    textfont=dict(size=9, color="#28231d"),
                    colorbar=dict(title="r"),
                    hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
                )
            )
            corr_heat.update_yaxes(autorange="reversed")
            corr_heat.update_xaxes(tickangle=-45)
            polish(
                corr_heat,
                f"{method} {'partial ' if view == 'Partial correlation' else ''}correlation matrix"
                + ("  (* p<.05  ** p<.01  *** p<.001)" if view == "Correlation" else ""),
                700,
                legend=False,
            )
            show_fig(corr_heat)

        with right:
            labels = [features_list[i] for i in leaves]
            dendro = sch.dendrogram(Z, no_plot=True, labels=features_list)
            dfig = go.Figure()
            for xs, ys in zip(dendro["icoord"], dendro["dcoord"]):
                dfig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#765337", width=2), hoverinfo="skip", showlegend=False))
            dfig.update_xaxes(
                tickvals=[5 + 10 * i for i in range(len(dendro["ivl"]))],
                ticktext=dendro["ivl"],
                tickangle=-60,
                showgrid=False,
            )
            dfig.update_yaxes(title="1 − |r|")
            polish(dfig, "Variable dendrogram", 340, legend=False)
            show_fig(dfig)

            vif = pd.Series(vif_from_corr(np.corrcoef(scaled_matrix(data, features_list), rowvar=False)), index=features_list).sort_values()
            vif_fig = go.Figure(
                go.Bar(
                    x=vif.values,
                    y=vif.index,
                    orientation="h",
                    marker_color=np.where(vif.values > 5, "#9b6650", "#69765b"),
                )
            )
            vif_fig.add_vline(x=5, line=dict(color="#503725", dash="dash"), annotation_text="VIF = 5")
            polish(vif_fig, "Variance inflation factors", 340, legend=False)
            show_fig(vif_fig)

        pair_rows = []
        upper = np.triu_indices(len(features_list), k=1)
        for i, j in zip(*upper):
            pair_rows.append((features_list[i], features_list[j], R.iloc[i, j], P.iloc[i, j]))
        pair_frame = pd.DataFrame(pair_rows, columns=["Feature A", "Feature B", "r", "p"]).dropna()
        pair_frame["p (adj)"] = adjust_pvalues(pair_frame["p"].to_numpy(), ADJ)
        pair_frame["|r|"] = pair_frame["r"].abs()
        st.markdown("#### Strongest pairs")
        show_df(
            pair_frame.sort_values("|r|", ascending=False).head(15).drop(columns="|r|").round(4),
            hide_index=True,
        )

    # ---------------- Association measures ----------------
    with assoc_tab:
        st.markdown("#### Categorical association with the label")
        st.caption(
            "Each integer-scored field is cross-tabulated with the label: χ² test of independence, "
            "bias-corrected Cramér’s V and normalised mutual information."
        )
        chi_table = ev[["Feature", "CramersV", "Chi2_p", "MI_norm"]].copy()
        chi_table["Chi2_p_adj"] = adjust_pvalues(chi_table["Chi2_p"].to_numpy(), ADJ)
        chi_table = chi_table.sort_values("CramersV", ascending=False)

        left, right = st.columns([1.2, 1])
        with left:
            vbar = go.Figure(
                go.Bar(
                    x=chi_table["Feature"],
                    y=chi_table["CramersV"],
                    marker=dict(color=chi_table["CramersV"], colorscale=SCALE_WARM, line=dict(color="#503725", width=1)),
                    text=[stars(p) for p in chi_table["Chi2_p_adj"]],
                    textposition="outside",
                )
            )
            vbar.update_xaxes(tickangle=-45)
            vbar.update_yaxes(title="Cramér’s V (bias-corrected)", range=[0, 1.05])
            polish(vbar, "Association strength with the label", 470, legend=False)
            show_fig(vbar)

        with right:
            mi_scatter = go.Figure(
                go.Scatter(
                    x=chi_table["CramersV"],
                    y=chi_table["MI_norm"],
                    mode="markers+text",
                    text=chi_table["Feature"],
                    textposition="top center",
                    textfont=dict(size=9),
                    marker=dict(size=12, color="#765337", opacity=0.8, line=dict(color="#faf5eb", width=1.5)),
                )
            )
            mi_scatter.update_xaxes(title="Cramér’s V")
            mi_scatter.update_yaxes(title="Normalised mutual information")
            polish(mi_scatter, "Two association lenses agree?", 470, legend=False)
            show_fig(mi_scatter)

        st.markdown("#### Drill-down contingency table")
        d_left, d_right = st.columns([1, 2])
        drill = d_left.selectbox("Field", features_list, key="drill_field")
        sub = data[[drill, target_column]].dropna()
        sub[target_column] = sub[target_column].astype(str)
        if sub[drill].nunique() > 12:
            sub["bin"] = pd.qcut(sub[drill], 5, duplicates="drop").astype(str)
            row_var = "bin"
        else:
            row_var = drill
        table = pd.crosstab(sub[row_var], sub[target_column]).reindex(columns=order_list, fill_value=0)

        if min(table.shape) >= 2:
            chi2, p_chi, dof, expected = stats.chi2_contingency(table)
            g_stat, p_g, _, _ = stats.chi2_contingency(table, lambda_="log-likelihood")
            residuals = (table - expected) / np.sqrt(expected)
            lim = max(3.0, float(np.nanmax(np.abs(residuals.to_numpy()))))
            with d_right:
                res_heat = px.imshow(
                    residuals.round(2),
                    text_auto=".1f",
                    aspect="auto",
                    color_continuous_scale=SCALE_DIV,
                    zmin=-lim,
                    zmax=lim,
                    labels=dict(x=target_column, y=drill, color="Pearson residual"),
                )
                polish(res_heat, "Standardised (Pearson) residuals — |value|>2 marks cells driving the association", 420, legend=False)
                show_fig(res_heat)
            verdict(
                f"<b>χ²({dof}) = {chi2:.1f}</b>, p {fmt_p(p_chi)} · likelihood-ratio G = {g_stat:.1f}, p {fmt_p(p_g)} · "
                f"Cramér’s V = {cramers_v(table):.3f}. "
                f"Share of cells with expected count < 5: {(expected < 5).mean():.0%}."
            )

    # ---------------- Bootstrap & permutation ----------------
    with resample_tab:
        r1, r2, r3, r4 = st.columns(4)
        rs_field = r1.selectbox("Feature", features_list, key="rs_field")
        rs_stat = r2.selectbox("Statistic", ["Mean", "Median"], key="rs_stat")
        rs_a = r3.selectbox("Reference group", order_list, index=0, key="rs_a")
        rs_b = r4.selectbox("Comparison group", order_list, index=len(order_list) - 1, key="rs_b")

        if rs_a == rs_b:
            st.info("Choose two different groups.")
        else:
            res = two_group_resampling(data, rs_field, target_column, rs_a, rs_b, rs_stat, int(N_RESAMPLES), SEED)
            left, right = st.columns(2)

            with left:
                bfig = go.Figure(go.Histogram(x=res["boot"], nbinsx=50, marker_color="#bd965b", opacity=0.85))
                bfig.add_vline(x=res["observed"], line=dict(color="#503725", width=3), annotation_text="observed")
                bfig.add_vrect(x0=res["ci"][0], x1=res["ci"][1], fillcolor="#69765b", opacity=0.18, line_width=0)
                bfig.add_vline(x=0, line=dict(color="#8c3b2e", dash="dash"))
                bfig.update_xaxes(title=f"Δ {rs_stat.lower()}  ({rs_b} − {rs_a})")
                polish(bfig, f"Bootstrap distribution ({N_RESAMPLES:,} draws)", 400, legend=False)
                show_fig(bfig)

            with right:
                pfig = go.Figure(go.Histogram(x=res["perm"], nbinsx=50, marker_color="#9b9ea0", opacity=0.85))
                pfig.add_vline(x=res["observed"], line=dict(color="#8c3b2e", width=3), annotation_text="observed")
                pfig.add_vline(x=-res["observed"], line=dict(color="#8c3b2e", width=2, dash="dot"))
                pfig.update_xaxes(title="Difference under the null (labels shuffled)")
                polish(pfig, "Permutation null distribution", 400, legend=False)
                show_fig(pfig)

            verdict(
                f"<b>Observed Δ{rs_stat.lower()} = {res['observed']:+.3f}</b> "
                f"(n = {res['ny']} vs {res['nx']}). Percentile bootstrap 95% CI: "
                f"[{res['ci'][0]:+.3f}, {res['ci'][1]:+.3f}]. "
                f"Two-sided permutation p = {res['p_perm']:.4f} "
                f"(smallest attainable ≈ {1 / (N_RESAMPLES + 1):.4f})."
            )

    # ---------------- Power analysis ----------------
    with power_tab:
        st.caption(
            "Prospective power of a two-sample t-test (two-sided) computed from the non-central t distribution."
        )
        p1, p2 = st.columns(2)
        eff = p1.slider("Standardised effect size (Cohen’s d)", 0.1, 1.5, 0.5, 0.05)
        target_power = p2.slider("Target power", 0.7, 0.99, 0.8, 0.01)

        n_grid = np.arange(5, 401)
        curve = go.Figure()
        needed = None
        for d_value, color in [(eff * 0.5, "#c9b48a"), (eff, "#503725"), (min(eff * 1.5, 2.5), "#69765b")]:
            power = t_power(d_value, n_grid, ALPHA)
            curve.add_trace(go.Scatter(x=n_grid, y=power, mode="lines", name=f"d = {d_value:.2f}", line=dict(color=color, width=3)))
            if abs(d_value - eff) < 1e-9:
                reached = np.where(power >= target_power)[0]
                needed = int(n_grid[reached[0]]) if len(reached) else None
        curve.add_hline(y=target_power, line=dict(color="#8c3b2e", dash="dash"))
        if needed:
            curve.add_vline(x=needed, line=dict(color="#8c3b2e", dash="dot"), annotation_text=f"n = {needed}/group")
        curve.update_xaxes(title="Sample size per group")
        curve.update_yaxes(title="Power", range=[0, 1.02])
        polish(curve, f"Power curves at α = {ALPHA:g}", 440)
        show_fig(curve)

        group_sizes = pd.Series(
            data[target_column].astype(str).value_counts().reindex(order_list), name="n"
        )
        smallest = int(group_sizes.min())
        achieved = float(t_power(eff, np.array([smallest]), ALPHA)[0])
        verdict(
            f"<b>Required sample:</b> {needed if needed else '>400'} per group for d = {eff:.2f} at power {target_power:.0%}. "
            f"With the smallest recorded group (n = {smallest}) the achieved power is <b>{achieved:.1%}</b>; "
            f"the minimum detectable d at 80% power is "
            f"{next((d for d in np.arange(0.05, 3, 0.01) if t_power(d, np.array([smallest]), ALPHA)[0] >= 0.8), float('nan')):.2f}."
        )


# ------------------------------------------------------------
# Multivariate Atlas
# ------------------------------------------------------------
elif page == "Multivariate Atlas":
    need_target()

    section_head(
        "CHAPTER IV · STRUCTURE",
        "Multivariate Atlas",
        "Dimension reduction, latent factors, unsupervised clustering and outlier detection — "
        "the label is used only for colouring and for external validation.",
    )

    features_list = list(numeric_features)
    order_list = list(class_order)
    codes = data[target_column].astype(str).map({label: i for i, label in enumerate(order_list)})
    label_series = data[target_column].astype(str)

    pca_tab, manifold_tab, cluster_tab, factor_tab, outlier_tab = st.tabs(
        [
            "PCA & parallel analysis",
            "Manifolds (t-SNE · LDA)",
            "Clustering",
            "Factor analysis",
            "Outliers & anomalies",
        ]
    )

    # ---------------- PCA ----------------
    with pca_tab:
        with st.spinner("Running PCA with Horn’s parallel analysis…"):
            bundle = pca_bundle(data, features_list, 200, SEED)

        fact = bundle["fact"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Bartlett’s sphericity", f"χ²({fact['dof']}) = {fact['chi2']:.0f}", f"p {fmt_p(fact['p'])}", delta_color="off")
        k2.metric("KMO sampling adequacy", f"{fact['kmo']:.2f}", "≥0.6 is acceptable", delta_color="off")
        k3.metric("Components (Kaiser > 1)", f"{bundle['retain_kaiser']}")
        k4.metric("Components (Horn)", f"{bundle['retain_horn']}", "95th-percentile permutation", delta_color="off")

        n_pc = len(bundle["eig"])
        left, right = st.columns(2)
        with left:
            scree = make_subplots(specs=[[{"secondary_y": True}]])
            scree.add_trace(
                go.Bar(x=list(range(1, n_pc + 1)), y=bundle["evr"] * 100, name="Variance explained (%)", marker_color="#bd965b"),
                secondary_y=False,
            )
            scree.add_trace(
                go.Scatter(x=list(range(1, n_pc + 1)), y=np.cumsum(bundle["evr"]) * 100, name="Cumulative (%)", mode="lines+markers", line=dict(color="#503725", width=3)),
                secondary_y=True,
            )
            scree.update_xaxes(title="Component")
            scree.update_yaxes(title="Variance explained (%)", secondary_y=False)
            scree.update_yaxes(title="Cumulative (%)", range=[0, 102], secondary_y=True, showgrid=False)
            polish(scree, "Scree plot & cumulative variance", 420)
            show_fig(scree)

        with right:
            horn = go.Figure()
            horn.add_trace(go.Scatter(x=list(range(1, n_pc + 1)), y=bundle["eig"], name="Observed eigenvalue", mode="lines+markers", line=dict(color="#503725", width=3)))
            horn.add_trace(go.Scatter(x=list(range(1, n_pc + 1)), y=bundle["horn"], name="Permuted 95th percentile", mode="lines", line=dict(color="#9b6650", dash="dash", width=3)))
            horn.add_hline(y=1, line=dict(color="#69765b", dash="dot"), annotation_text="Kaiser = 1")
            horn.update_xaxes(title="Component")
            horn.update_yaxes(title="Eigenvalue")
            polish(horn, "Horn’s parallel analysis", 420)
            show_fig(horn)

        scores = pd.DataFrame(bundle["scores"][:, :3], columns=["PC1", "PC2", "PC3"], index=data.index)
        scores[target_column] = label_series

        left, right = st.columns([1.1, 1])
        with left:
            loadings = bundle["loadings"]
            biplot = px.scatter(
                scores,
                x="PC1",
                y="PC2",
                color=target_column,
                color_discrete_map=class_colors,
                category_orders={target_column: order_list},
                opacity=0.55,
                render_mode="webgl",
            )
            scale = np.abs(bundle["scores"][:, :2]).max() / max(np.abs(loadings[:, :2]).max(), 1e-9) * 0.85
            for i, name in enumerate(features_list):
                biplot.add_trace(
                    go.Scatter(
                        x=[0, loadings[i, 0] * scale],
                        y=[0, loadings[i, 1] * scale],
                        mode="lines+text",
                        text=["", name],
                        textposition="top center",
                        textfont=dict(size=9, color="#503725"),
                        line=dict(color="rgba(80,55,37,.55)", width=1.5),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            biplot.update_xaxes(title=f"PC1 ({bundle['evr'][0]:.1%})")
            biplot.update_yaxes(title=f"PC2 ({bundle['evr'][1]:.1%})")
            polish(biplot, "PCA biplot — scores and variable loadings", 560)
            show_fig(biplot)

        with right:
            pc3 = px.scatter_3d(
                scores,
                x="PC1",
                y="PC2",
                z="PC3",
                color=target_column,
                color_discrete_map=class_colors,
                category_orders={target_column: order_list},
                opacity=0.75,
            )
            pc3.update_traces(marker=dict(size=3))
            polish(pc3, "First three components", 560)
            show_fig(pc3)

        show_k = bounded_slider("Components shown in the loading heatmap", 2, min(10, n_pc), min(5, n_pc))
        load_frame = pd.DataFrame(
            bundle["loadings"][:, :show_k],
            index=features_list,
            columns=[f"PC{i + 1}" for i in range(show_k)],
        )
        load_heat = px.imshow(
            load_frame,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=SCALE_DIV,
            zmin=-1,
            zmax=1,
        )
        polish(load_heat, "Component loadings (correlation of variable with component)", 640, legend=False)
        show_fig(load_heat)

        # Does the label separate on the leading components?
        sep_rows = []
        for i in range(min(5, n_pc)):
            groups = [bundle["scores"][(label_series == l).to_numpy(), i] for l in order_list]
            h_stat, p_kw = stats.kruskal(*groups)
            n_all = len(label_series)
            sep_rows.append((f"PC{i + 1}", h_stat, p_kw, h_stat * (n_all + 1) / (n_all ** 2 - 1)))
        sep = pd.DataFrame(sep_rows, columns=["Component", "Kruskal–Wallis H", "p", "ε²"])
        with st.expander("Do the leading components separate the labels?"):
            show_df(sep.round(4), hide_index=True)

    # ---------------- Manifolds ----------------
    with manifold_tab:
        left, right = st.columns(2)

        with left:
            st.markdown("#### Linear discriminant projection (supervised)")
            lda_scores, lda_keep, lda_ratio = lda_embed(data, features_list, target_column, order_list, SEED)
            lda_labels = label_series[lda_keep].to_numpy()
            if lda_scores.shape[1] >= 2:
                lda_fig = px.scatter(
                    x=lda_scores[:, 0],
                    y=lda_scores[:, 1],
                    color=lda_labels,
                    color_discrete_map=class_colors,
                    category_orders={"color": order_list},
                    opacity=0.6,
                    labels={"x": f"LD1 ({lda_ratio[0]:.1%})", "y": f"LD2 ({lda_ratio[1]:.1%})", "color": target_column},
                )
            else:
                lda_fig = px.strip(
                    x=lda_scores[:, 0],
                    y=lda_labels,
                    color=lda_labels,
                    color_discrete_map=class_colors,
                    labels={"x": "LD1", "y": target_column},
                )
            polish(lda_fig, "LDA — directions that maximise class separation", 520)
            show_fig(lda_fig)

        with right:
            st.markdown("#### t-SNE embedding (non-linear)")
            perplexity = st.slider("Perplexity", 5, 60, 30, 5)
            if st.button("Compute t-SNE embedding", key="tsne_btn"):
                st.session_state["tsne_ready"] = True
            if st.session_state.get("tsne_ready"):
                with st.spinner("Optimising the embedding…"):
                    emb = tsne_embed(data, features_list, perplexity, SEED)
                tsne_fig = px.scatter(
                    x=emb[:, 0],
                    y=emb[:, 1],
                    color=label_series.to_numpy(),
                    color_discrete_map=class_colors,
                    category_orders={"color": order_list},
                    opacity=0.65,
                    labels={"x": "t-SNE 1", "y": "t-SNE 2", "color": target_column},
                )
                tsne_fig.update_traces(marker=dict(size=6))
                polish(tsne_fig, f"t-SNE · perplexity {perplexity}", 480)
                show_fig(tsne_fig)
                st.caption(
                    "Distances between far-apart clusters in t-SNE are not meaningful; "
                    "use the plot for local neighbourhood structure only."
                )
            else:
                st.info("Press the button to compute the embedding (cached afterwards).")

    # ---------------- Clustering ----------------
    with cluster_tab:
        k_max = st.slider("Largest k to evaluate", 4, 10, 8)
        with st.spinner("Scoring cluster solutions…"):
            sweep = cluster_sweep(data, features_list, k_max, SEED)

        panels = [
            ("Inertia", "Elbow (inertia)", None),
            ("Silhouette", "Silhouette ↑", "max"),
            ("CalinskiHarabasz", "Calinski–Harabasz ↑", "max"),
            ("DaviesBouldin", "Davies–Bouldin ↓", "min"),
            ("BIC", "GMM BIC ↓", "min"),
            ("AIC", "GMM AIC ↓", "min"),
        ]
        sweep_fig = make_subplots(rows=2, cols=3, subplot_titles=[p[1] for p in panels])
        for index, (column, _, best) in enumerate(panels):
            r, c = divmod(index, 3)
            sweep_fig.add_trace(
                go.Scatter(x=sweep["k"], y=sweep[column], mode="lines+markers", line=dict(color="#765337", width=3), marker=dict(size=8), showlegend=False),
                row=r + 1,
                col=c + 1,
            )
            if best:
                pick = sweep.loc[sweep[column].idxmax() if best == "max" else sweep[column].idxmin()]
                sweep_fig.add_trace(
                    go.Scatter(x=[pick["k"]], y=[pick[column]], mode="markers", marker=dict(size=16, color="#8c3b2e", symbol="star"), showlegend=False),
                    row=r + 1,
                    col=c + 1,
                )
        polish(sweep_fig, "Internal validity indices across k", 560, legend=False)
        show_fig(sweep_fig)

        suggested = int(sweep.loc[sweep["Silhouette"].idxmax(), "k"])
        k_choice = st.slider("Chosen number of clusters", 2, k_max, min(max(suggested, 2), k_max))
        labels_k = kmeans_labels(data, features_list, k_choice, SEED)
        cluster_names = np.array([f"Cluster {i + 1}" for i in labels_k])

        ari = adjusted_rand_score(label_series, labels_k)
        nmi = normalized_mutual_info_score(label_series, labels_k)
        cont = pd.crosstab(cluster_names, label_series).reindex(columns=order_list, fill_value=0)
        v_cluster = cramers_v(cont)
        sil = float(sweep.loc[sweep["k"] == k_choice, "Silhouette"].iloc[0])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Silhouette (internal)", f"{sil:.3f}")
        m2.metric("Adjusted Rand index", f"{ari:.3f}", help="External validation against the recorded label")
        m3.metric("Normalised MI", f"{nmi:.3f}")
        m4.metric("Cramér’s V (cluster × label)", f"{v_cluster:.3f}")

        left, right = st.columns(2)
        with left:
            z = pd.DataFrame(scaled_matrix(data, features_list), columns=features_list)
            z["cluster"] = cluster_names
            centroid = z.groupby("cluster")[features_list].mean()
            limit = float(np.abs(centroid.to_numpy()).max())
            prof = px.imshow(
                centroid.T,
                text_auto=".1f",
                aspect="auto",
                color_continuous_scale=SCALE_DIV,
                zmin=-limit,
                zmax=limit,
            )
            polish(prof, "Cluster centroids (z-scores)", 600, legend=False)
            show_fig(prof)

        with right:
            sank_nodes = list(cont.index) + [f"{target_column}: {l}" for l in order_list]
            source, target_idx, value = [], [], []
            for i, c_name in enumerate(cont.index):
                for j, l in enumerate(order_list):
                    if cont.iloc[i, j] > 0:
                        source.append(i)
                        target_idx.append(len(cont.index) + j)
                        value.append(int(cont.iloc[i, j]))
            sankey = go.Figure(
                go.Sankey(
                    node=dict(
                        label=sank_nodes,
                        color=["#bd965b"] * len(cont.index) + [class_colors.get(l, "#765337") for l in order_list],
                        pad=18,
                        thickness=18,
                        line=dict(color="#faf5eb", width=1),
                    ),
                    link=dict(source=source, target=target_idx, value=value, color="rgba(118,83,55,.28)"),
                )
            )
            polish(sankey, "Clusters ↔ recorded label (Sankey)", 600, legend=False)
            show_fig(sankey)

        verdict(
            "<b>External validation.</b> ARI and NMI compare an <i>unsupervised</i> partition with the recorded label: "
            f"ARI = {ari:.2f} means {'strong' if ari > 0.6 else 'moderate' if ari > 0.3 else 'weak'} agreement beyond chance. "
            "Internal indices favour compact, well-separated groups, which need not coincide with clinical categories."
        )

    # ---------------- Factor analysis ----------------
    with factor_tab:
        fact = pca_bundle(data, features_list, 200, SEED)
        suggested_k = max(2, min(fact["retain_horn"], 6))
        k_factors = bounded_slider("Number of factors (varimax rotation)", 2, min(8, len(features_list) - 1), suggested_k)
        fb = factor_bundle(data, features_list, k_factors, SEED)

        communality = 1 - fb["noise"] / np.maximum(np.var(scaled_matrix(data, features_list), axis=0), 1e-9)
        left, right = st.columns([1.4, 1])
        with left:
            floads = pd.DataFrame(
                fb["loadings"],
                index=features_list,
                columns=[f"Factor {i + 1}" for i in range(k_factors)],
            )
            strongest = floads.abs().idxmax(axis=1)
            order_idx = sorted(range(len(features_list)), key=lambda i: (strongest.iloc[i], -abs(floads.iloc[i][strongest.iloc[i]])))
            floads = floads.iloc[order_idx]
            fheat = px.imshow(
                floads,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale=SCALE_DIV,
                zmin=-1,
                zmax=1,
            )
            polish(fheat, "Rotated factor loadings", 640, legend=False)
            show_fig(fheat)

        with right:
            comm = pd.Series(communality, index=features_list).sort_values()
            comm_fig = go.Figure(
                go.Bar(
                    x=comm.values,
                    y=comm.index,
                    orientation="h",
                    marker=dict(color=comm.values, colorscale=SCALE_SEQ_GREEN),
                )
            )
            comm_fig.update_xaxes(title="Communality (variance shared with the factors)", range=[0, 1])
            polish(comm_fig, "Communalities", 640, legend=False)
            show_fig(comm_fig)

        share = fb["ss"] / len(features_list)
        verdict(
            f"<b>{k_factors} factors</b> explain about {share.sum():.0%} of total variance "
            f"(per-factor SS loadings: {', '.join(f'{v:.2f}' for v in fb['ss'])}). "
            f"Average log-likelihood = {fb['loglik']:.2f}. Features loading above |0.4| on a factor define its theme; "
            "loadings are exploratory and should be confirmed in an independent sample."
        )

        f_scores = pd.DataFrame(fb["scores"][:, :2], columns=["Factor 1", "Factor 2"])
        f_scores[target_column] = label_series.to_numpy()
        fs_fig = px.scatter(
            f_scores,
            x="Factor 1",
            y="Factor 2",
            color=target_column,
            color_discrete_map=class_colors,
            category_orders={target_column: order_list},
            opacity=0.55,
            marginal_x="violin",
            marginal_y="violin",
        )
        polish(fs_fig, "Factor-score space", 560)
        show_fig(fs_fig)

    # ---------------- Outliers ----------------
    with outlier_tab:
        contamination = st.slider("Isolation Forest contamination", 0.01, 0.15, 0.05, 0.01)
        ob = outlier_bundle(data, features_list, contamination, SEED)
        n_maha = int((ob["d2"] > ob["cutoff"]).sum())
        n_iso = int(ob["iso_flag"].sum())
        both = int(((ob["d2"] > ob["cutoff"]) & ob["iso_flag"]).sum())

        m1, m2, m3 = st.columns(3)
        m1.metric("Mahalanobis outliers (χ² p<0.001)", f"{n_maha}")
        m2.metric("Isolation Forest flags", f"{n_iso}")
        m3.metric("Flagged by both", f"{both}")

        left, right = st.columns(2)
        with left:
            order_d = np.sort(ob["d2"])
            theo = stats.chi2.ppf((np.arange(1, len(order_d) + 1) - 0.5) / len(order_d), len(features_list))
            qq_fig = go.Figure()
            qq_fig.add_trace(go.Scatter(x=theo, y=order_d, mode="markers", marker=dict(color="#765337", size=5, opacity=0.65), name="D²"))
            top = max(theo.max(), order_d.max())
            qq_fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color="#8c3b2e", dash="dash"), name="y = x"))
            qq_fig.add_hline(y=ob["cutoff"], line=dict(color="#69765b", dash="dot"), annotation_text="χ² 99.9% cutoff")
            qq_fig.update_xaxes(title="Theoretical χ² quantiles")
            qq_fig.update_yaxes(title="Squared Mahalanobis distance")
            polish(qq_fig, "Mahalanobis D² vs χ² quantiles", 480)
            show_fig(qq_fig)

        with right:
            pb = pca_bundle(data, features_list, 200, SEED)
            status = np.where(
                (ob["d2"] > ob["cutoff"]) & ob["iso_flag"],
                "Both methods",
                np.where(ob["iso_flag"], "Isolation Forest", np.where(ob["d2"] > ob["cutoff"], "Mahalanobis", "Typical")),
            )
            out_frame = pd.DataFrame(
                {"PC1": pb["scores"][:, 0], "PC2": pb["scores"][:, 1], "Status": status, "Anomaly score": ob["iso_score"]}
            )
            out_fig = px.scatter(
                out_frame,
                x="PC1",
                y="PC2",
                color="Status",
                color_discrete_map={
                    "Typical": "#c9b48a",
                    "Isolation Forest": "#6f8796",
                    "Mahalanobis": "#bd965b",
                    "Both methods": "#8c3b2e",
                },
                opacity=0.7,
                hover_data={"Anomaly score": ":.3f"},
            )
            polish(out_fig, "Anomalies on the PCA plane", 480)
            show_fig(out_fig)

        flagged = pd.DataFrame(
            {
                "Row": data.index,
                "D²": ob["d2"],
                "Isolation score": ob["iso_score"],
                "Status": status,
                target_column: label_series.to_numpy(),
            }
        )
        flagged = flagged[flagged["Status"] != "Typical"].sort_values("D²", ascending=False)
        with st.expander(f"Flagged records ({len(flagged)})"):
            show_df(flagged.round(3), hide_index=True)
        st.caption(
            "Anomaly flags identify unusual records for data-quality review; they are not "
            "clinical findings and should not be removed without justification."
        )


# ------------------------------------------------------------
# Machine-learning studio
# ------------------------------------------------------------
elif page == "Model Studio":
    need_target()

    section_head(
        "CHAPTER V · PREDICTION",
        "Model Studio",
        "Compare classifiers with repeated stratified cross-validation, test the differences "
        "statistically, and inspect the champion on a held-out split.",
    )

    model_features = [
        column for column in data.columns
        if column != target_column and column not in id_fields
    ]
    zoo_names = list(model_zoo(SEED).keys())

    with st.container(border=True):
        st.subheader("Experiment controls")
        chosen_features = st.multiselect(
            "Prediction fields",
            model_features,
            default=model_features,
        )
        chosen_models = st.multiselect(
            "Classifiers to compare",
            zoo_names,
            default=["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM (RBF kernel)"],
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
        with c2:
            cv_repeats = st.slider("Repeats", 1, 5, 2)
        with c3:
            test_size = st.slider("Hold-out percentage", 15, 40, 25, step=5)

        run = st.button(
            "Run the experiment",
            type="primary",
            disabled=not (chosen_features and chosen_models),
        )

    signature = (data_source, len(data), tuple(class_order), target_column)

    if run:
        progress = st.progress(0.0, text="Preparing the experiment…")
        try:
            result = run_experiment(
                data,
                list(chosen_features),
                target_column,
                list(class_order),
                list(chosen_models),
                cv_folds,
                cv_repeats,
                test_size,
                SEED,
                progress,
            )
            result["signature"] = signature
            st.session_state["ms_bundle"] = result
            st.toast("Experiment complete")
        except Exception as error:
            st.error(
                "The experiment could not run. Check the class counts, selected "
                f"fields and classifiers. Details: {type(error).__name__}: {error}"
            )
        progress.empty()

    stored = st.session_state.get("ms_bundle")
    if stored is not None and stored.get("signature") == signature:
        render_model_results(stored)
    else:
        st.info("Choose fields and classifiers, then press **Run the experiment**.")


# ------------------------------------------------------------
# Patient Q&A
# ------------------------------------------------------------
elif page == "Patient Q&A":
    st.markdown("## Patient Q&A")
    st.caption(
        "General health education and help understanding this app's charts."
    )

    st.markdown(
        """
        <div class="note">
            Do not enter names, medical record numbers, contact details,
            or identifiable health information. This assistant cannot diagnose,
            assess personal risk, or recommend treatment. For urgent symptoms,
            contact local emergency services or a qualified clinician.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Topics this assistant can discuss"):
        st.markdown(
            """
            - Explain general medical terms in plain English.
            - Suggest questions to discuss with a licensed clinician.
            - Explain charts, model metrics, and dataset fields.
            - Summarize the analysis workflow and its limitations.
            """
        )

    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_key = ""

    active_key = api_key or secret_key

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    chat_left, chat_right = st.columns([4, 1])

    with chat_left:
        st.markdown("### The reading room")

    with chat_right:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

    for message in st.session_state.chat_history:
        with st.chat_message(
            message["role"],
            avatar="✦" if message["role"] == "assistant" else "●",
        ):
            st.markdown(message["content"])

    if not active_key:
        st.info("Add your OpenAI API key in the sidebar to enable chat.")

    question = st.chat_input(
        "Ask a general question about cancer or this app..."
    )

    if question:
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user", avatar="●"):
            st.markdown(question)

        if not active_key:
            st.warning("Enter an API key in the sidebar, then ask again.")
        else:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=active_key)
                recent_messages = [
                    {
                        "role": message["role"],
                        "content": message["content"],
                    }
                    for message in st.session_state.chat_history[-12:]
                ]

                context = ""
                if data is not None:
                    context = (
                        f"\nDataset has {len(data)} rows and "
                        f"{len(data.columns)} fields. "
                        "Field names: "
                        + ", ".join(map(str, data.columns))
                    )
                    if target_column:
                        context += f"\nDetected target: {target_column}."

                instructions = (
                    "You are a careful, compassionate health-information "
                    "education assistant inside a student data-science app. "
                    "Always respond in clear English. You may explain general "
                    "health concepts and this app's dataset, charts, and model "
                    "metrics. Never diagnose, estimate an individual's cancer "
                    "risk, interpret personal symptoms as a diagnosis, or "
                    "recommend medication or treatment. Encourage users to "
                    "discuss personal concerns with a licensed clinician. For "
                    "possible emergencies or severe symptoms, advise contacting "
                    "local emergency services immediately. Do not request "
                    "identifying or sensitive personal health details. State "
                    "uncertainty when appropriate. The dataset is educational "
                    "and not clinically validated."
                    + context
                )

                with st.chat_message("assistant", avatar="✦"):
                    with st.spinner("Composing a careful response..."):
                        response = client.responses.create(
                            model=api_model.strip(),
                            instructions=instructions,
                            input=recent_messages,
                        )
                        answer = response.output_text.strip()

                    if not answer:
                        answer = "I could not prepare a text response. Please try again."

                    st.markdown(answer)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as error:
                st.error(
                    "The request failed. Check the API key, model name, and "
                    f"API project access. Details: {error}"
                )



# ------------------------------------------------------------
# Field notes
# ------------------------------------------------------------
elif page == "Field Notes":
    section_head("APPENDIX", "Field Notes")

    source_tab, methods_tab, formula_tab, inventory_tab = st.tabs(
        ["Source & questions", "Methods & limits", "Formula sheet", "Field inventory"]
    )

    with source_tab:
        st.markdown(
            """
            ### Dataset source

            This app uses the supplied workbook associated with the
            [Cancer Patients Dataset on Kaggle](https://www.kaggle.com/datasets/rishidamarla/cancer-patients-data).

            ### Questions

            1. How are the recorded labels distributed, and is the cohort balanced?
            2. Which features differ across the ordered label groups — and by how much?
            3. Is there a monotone (dose–response) trend between each feature and the label?
            4. What latent structure (components, factors, clusters) exists without using the label?
            5. How do competing classifiers perform on held-out records, and are their differences reliable?
            """
        )

    with methods_tab:
        st.markdown(
            """
            **Inference.** Group differences use the Kruskal–Wallis rank test (ANOVA as a sensitivity check),
            with ε² and η² effect sizes; post-hoc contrasts use Mann–Whitney with Cliff's δ and Hedges' g
            (bootstrap CIs). Ordered trends use Kendall's τ<sub>b</sub>. Associations use χ², bias-corrected
            Cramér's V and mutual information. All families of tests are adjusted with Holm, Bonferroni or
            Benjamini–Hochberg.

            **Resampling.** Percentile bootstrap intervals and Monte-Carlo permutation tests complement the
            asymptotic results; power curves use the non-central t distribution.

            **Multivariate.** PCA is accompanied by Horn's parallel analysis, Bartlett's test and the KMO
            measure. Varimax factor analysis, k-means / Gaussian-mixture sweeps (silhouette, Calinski–Harabasz,
            Davies–Bouldin, BIC/AIC), t-SNE, LDA, Mahalanobis distance and Isolation Forest are provided.

            **Prediction.** Pipelines impute, scale and encode inside each cross-validation fold to avoid
            leakage. Models are compared with repeated stratified k-fold CV, the Nadeau–Bengio corrected
            resampled t-test and the Friedman test; the champion is examined on a hold-out set with ROC/PR,
            calibration, permutation importance and learning curves.

            **Limits.** Dataset sampling, provenance, and label quality may limit generalisation. The recorded
            label is almost perfectly predictable from the features, which is unusual for real clinical data.
            Results are educational examples, not clinical validation or medical advice.
            """,
            unsafe_allow_html=True,
        )

    with formula_tab:
        left, right = st.columns(2)
        with left:
            st.markdown("**Kruskal–Wallis H and effect size**")
            st.latex(r"H=\frac{12}{N(N+1)}\sum_{i=1}^{k}\frac{R_i^{2}}{n_i}-3(N+1),\qquad \varepsilon^{2}=\frac{H\,(N+1)}{N^{2}-1}")
            st.markdown("**Eta-squared (one-way ANOVA)**")
            st.latex(r"\eta^{2}=\frac{SS_{between}}{SS_{total}}")
            st.markdown("**Cliff's δ (U of the later group) and Hedges' g**")
            st.latex(r"\delta=\frac{2U}{n_1 n_2}-1,\qquad g=\Big(1-\tfrac{3}{4(n_1+n_2)-9}\Big)\frac{\bar x_2-\bar x_1}{s_p}")
            st.markdown("**Bias-corrected Cramér's V**")
            st.latex(r"\tilde V=\sqrt{\frac{\tilde\varphi^{2}}{\min(\tilde k-1,\,\tilde r-1)}},\quad \tilde\varphi^{2}=\max\!\Big(0,\tfrac{\chi^{2}}{n}-\tfrac{(k-1)(r-1)}{n-1}\Big)")
            st.markdown("**Holm step-down adjustment**")
            st.latex(r"p^{adj}_{(i)}=\max_{j\le i}\min\{1,(m-j+1)\,p_{(j)}\}")
        with right:
            st.markdown("**Nadeau–Bengio corrected resampled t**")
            st.latex(r"t=\frac{\bar d}{\sqrt{\big(\tfrac{1}{K}+\tfrac{n_{test}}{n_{train}}\big)\hat\sigma_d^{2}}}")
            st.markdown("**Mahalanobis distance**")
            st.latex(r"D^{2}=(\mathbf x-\boldsymbol\mu)^{\top}\Sigma^{-1}(\mathbf x-\boldsymbol\mu)\ \sim\ \chi^{2}_{p}")
            st.markdown("**Bartlett's test of sphericity**")
            st.latex(r"\chi^{2}=-\Big(n-1-\tfrac{2p+5}{6}\Big)\ln|R|,\quad df=\tfrac{p(p-1)}{2}")
            st.markdown("**Kaiser–Meyer–Olkin**")
            st.latex(r"KMO=\frac{\sum_{i\ne j} r_{ij}^{2}}{\sum_{i\ne j} r_{ij}^{2}+\sum_{i\ne j} a_{ij}^{2}}")
            st.markdown("**Matthews correlation (multiclass)**")
            st.latex(r"MCC=\frac{c\,s-\sum_k p_k t_k}{\sqrt{(s^{2}-\sum_k p_k^{2})(s^{2}-\sum_k t_k^{2})}}")

    with inventory_tab:
        if data is None:
            st.info("Upload a dataset to see its field inventory.")
        else:
            inv_rows = []
            for column in data.columns:
                s = data[column]
                row = {
                    "Field": column,
                    "Type": str(s.dtype),
                    "Missing": int(s.isna().sum()),
                    "Unique": int(s.nunique(dropna=True)),
                }
                if pd.api.types.is_numeric_dtype(s) and s.notna().any():
                    row["Min"] = float(s.min())
                    row["Max"] = float(s.max())
                    row["Distribution"] = np.histogram(s.dropna(), bins=10)[0].tolist()
                else:
                    row["Min"], row["Max"], row["Distribution"] = None, None, []
                inv_rows.append(row)
            inventory = pd.DataFrame(inv_rows)
            try:
                show_df(
                    inventory,
                    hide_index=True,
                    column_config={
                        "Distribution": st.column_config.BarChartColumn("Distribution"),
                        "Min": st.column_config.NumberColumn(format="%.1f"),
                        "Max": st.column_config.NumberColumn(format="%.1f"),
                    },
                )
            except Exception:
                show_df(inventory.drop(columns=["Distribution"]), hide_index=True)

st.divider()
st.caption(
    "CANCERLENS · A student data-science project · "
    "Streamlit · pandas · NumPy · SciPy · scikit-learn · Plotly · OpenAI API"
)