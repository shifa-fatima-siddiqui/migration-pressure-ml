"""
Climate-Induced Migration Pressure — India District Dashboard
Run with:  streamlit run dashboard.py
Expects migration_dataset.csv and migration_rf_model.joblib in the same folder.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Migration Pressure · India",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = "data/processed/migration_dataset.csv"
MODEL_PATH = "models/random_forest_model.pkl"

FEATURE_COLS = [
    "rainfall_cv_imp",
    "drought_freq_imp",
    "flood_freq_imp",
    "yield_cv_imp",
    "production_std_imp",
    "mpi_imp",
    "headcount_ratio_imp",
    "marginal_worker_rate_imp",
    "worker_ratio_imp",
]

FEATURE_LABELS = {
    "rainfall_cv_imp":           "Rainfall variability (CV)",
    "drought_freq_imp":          "Drought frequency",
    "flood_freq_imp":            "Flood frequency",
    "yield_cv_imp":              "Crop yield variability (CV)",
    "production_std_imp":        "Production std dev",
    "mpi_imp":                   "Multidimensional Poverty Index",
    "headcount_ratio_imp":       "MPI headcount ratio",
    "marginal_worker_rate_imp":  "Marginal worker rate",
    "worker_ratio_imp":          "Worker ratio",
}

CATEGORY_COLORS = {
    "Low":    "#2ecc71",   # green
    "Medium": "#f39c12",   # amber
    "High":   "#e74c3c",   # red
}

CATEGORY_ORDER = ["Low", "Medium", "High"]

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Base typography ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0f1923;
        border-right: 1px solid #1e2d3d;
    }
    section[data-testid="stSidebar"] * {
        color: #c8d6e5 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label {
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #637d98 !important;
        font-weight: 500;
    }

    /* ── Main background ── */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* ── Score card ── */
    .score-card {
        background: #0f1923;
        border: 1px solid #1e2d3d;
        border-radius: 12px;
        padding: 28px 32px;
        text-align: center;
    }
    .score-label {
        font-size: 11px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #637d98;
        margin-bottom: 6px;
        font-weight: 500;
    }
    .score-value {
        font-family: 'DM Mono', monospace;
        font-size: 52px;
        font-weight: 500;
        color: #e8f0f8;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .score-sub {
        font-size: 12px;
        color: #637d98;
        margin-top: 8px;
    }

    /* ── Category badge ── */
    .badge {
        display: inline-block;
        padding: 6px 20px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-top: 4px;
    }

    /* ── Rank card ── */
    .rank-card {
        background: #0f1923;
        border: 1px solid #1e2d3d;
        border-radius: 12px;
        padding: 28px 32px;
        text-align: center;
    }

    /* ── Stat grid ── */
    .stat-row {
        display: flex;
        gap: 12px;
        margin-bottom: 12px;
    }
    .stat-item {
        flex: 1;
        background: #0f1923;
        border: 1px solid #1e2d3d;
        border-radius: 10px;
        padding: 16px 18px;
    }
    .stat-item-label {
        font-size: 10px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #637d98;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .stat-item-value {
        font-family: 'DM Mono', monospace;
        font-size: 20px;
        font-weight: 500;
        color: #e8f0f8;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 11px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #637d98;
        font-weight: 600;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e2d3d;
    }

    /* ── High-risk table ── */
    .risk-table th {
        background: #0d1520 !important;
        color: #637d98 !important;
        font-size: 10px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .risk-table td {
        color: #c8d6e5;
        font-size: 13px;
    }

  

    /* ── Plot backgrounds ── */
    .stPlotlyChart, .stPyplot {
        background: transparent;
    }

    /* ── Divider ── */
    hr {
        border-color: #1e2d3d;
    }

     /* =========================
   FIX HEADING (your issue)
========================= */
.page-title {
    color: #e6edf3 !important;
    font-weight: 800 !important;
    font-size: 32px !important;
    opacity: 1 !important;
}



/* =========================
   DROPDOWN IMPROVEMENT
========================= */
/* =========================
   DROPDOWN FIX (safe version)
========================= */
div[data-baseweb="select"] > div {
    background-color: #111c26 !important;
    border: 1px solid #2a3a4d !important;
    border-radius: 8px !important;
}

/* Selected text */
div[data-baseweb="select"] span {
    color: #e6edf3 !important;
}

/* Dropdown menu */
ul[role="listbox"] {
    background-color: #0f1923 !important;
}
/* =========================
   SIDEBAR LABELS
========================= */
label {
    color: #9fb3c8 !important;
    font-size: 11px !important;
    letter-spacing: 0.08em;
}

/* =========================
   ADD DEPTH (cards glow)
========================= */
.score-card, .rank-card, .stat-item {
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    transition: all 0.2s ease;
}

.score-card:hover, .rank-card:hover, .stat-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(79, 140, 255, 0.15);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data + model loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["rank"] = df["migration_pressure_score"].rank(
        ascending=False, method="min"
    ).astype(int)
    df["migration_pressure_category"] = pd.Categorical(
        df["migration_pressure_category"], categories=CATEGORY_ORDER, ordered=True
    )
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df   = load_data()
model = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — state / district selectors
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🗺️ Migration Pressure")
    st.markdown(
        "<span style='font-size:11px;color:#637d98;letter-spacing:.08em;"
        "text-transform:uppercase;font-weight:500;'>India · District Explorer</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    states = sorted(df["state"].dropna().unique())
    selected_state = st.selectbox("State", states, index=states.index("BIHAR") if "BIHAR" in states else 0)

    districts_in_state = sorted(df[df["state"] == selected_state]["district"].dropna().unique())
    selected_district  = st.selectbox("District", districts_in_state)

    st.markdown("---")
    st.markdown(
        "<span style='font-size:11px;color:#637d98;'>Model: Random Forest · 640 districts</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span style='font-size:11px;color:#637d98;'>Target: migration_pressure_score</span>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Selected district row
# ─────────────────────────────────────────────────────────────────────────────

row = df[(df["state"] == selected_state) & (df["district"] == selected_district)].iloc[0]

score    = row["migration_pressure_score"]
category = row["migration_pressure_category"]
rank     = int(row["rank"])
n_total  = len(df)
cat_color = CATEGORY_COLORS[category]

# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<h1 style="
    color:#d62828;
    font-weight:800;
    font-size:34px;
    margin-bottom:4px;
    opacity:1;
">
Climate-Induced Migration Pressure
</h1>

<p style="
    color:#4a5568;
    font-size:14px;
    margin-top:0;
">
India · District-level analysis · 2011 Census aligned
</p>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────
# Top KPI row: Score · Category · Rank
# ─────────────────────────────────────────────────────────────────────────────

col_score, col_cat, col_rank = st.columns([1, 1, 1])

with col_score:
    st.markdown(
        f'<div class="score-card">'
        f'  <div class="score-label">Migration pressure score</div>'
        f'  <div class="score-value">{score:.3f}</div>'
        f'  <div class="score-sub">{selected_district}, {selected_state}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with col_cat:
    st.markdown(
        f'<div class="score-card">'
        f'  <div class="score-label">Risk category</div>'
        f'  <div style="margin-top:12px;">'
        f'    <span class="badge" style="background:{cat_color}22;color:{cat_color};'
        f'border:1.5px solid {cat_color}66;font-size:22px;padding:10px 32px;">'
        f'      {category}'
        f'    </span>'
        f'  </div>'
        f'  <div class="score-sub" style="margin-top:14px;">Quantile-based classification</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with col_rank:
    percentile = round((1 - (rank - 1) / n_total) * 100, 1)
    st.markdown(
        f'<div class="rank-card">'
        f'  <div class="score-label">National rank</div>'
        f'  <div class="score-value" style="font-size:38px;">'
        f'    #{rank} <span style="font-size:18px;color:#637d98;">/ {n_total}</span>'
        f'  </div>'
        f'  <div class="score-sub">Top {percentile}% by pressure score</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Component score mini-stats
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Component scores — selected district</div>', unsafe_allow_html=True)

comp_cols  = ["score_climate", "score_agriculture", "score_poverty", "score_demographics"]
comp_names = ["Climate instability", "Agriculture stress", "Poverty index", "Demographic pressure"]
comp_weights = [0.35, 0.25, 0.25, 0.15]

cols = st.columns(4)
for col, comp_col, name, w in zip(cols, comp_cols, comp_names, comp_weights):
    val = row[comp_col]
    # Color intensity based on value
    r_hex = int(min(255, val * 2.5 * 255))
    g_hex = int(max(0, (1 - val * 2) * 180))
    bar_color = f"rgb({r_hex},{g_hex},80)"
    with col:
        st.markdown(
            f'<div class="stat-item">'
            f'  <div class="stat-item-label">{name}</div>'
            f'  <div class="stat-item-value">{val:.3f}</div>'
            f'  <div style="margin-top:8px;height:4px;background:#1e2d3d;border-radius:2px;">'
            f'    <div style="width:{val*100:.0f}%;height:100%;background:{bar_color};border-radius:2px;"></div>'
            f'  </div>'
            f'  <div style="font-size:10px;color:#637d98;margin-top:4px;">weight {w}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Visualizations — histogram + feature importance
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Score distribution &amp; model feature importance</div>', unsafe_allow_html=True)

col_hist, col_imp = st.columns([1.1, 0.9])

# ── Histogram ───────────────────────────────────────────────────────────────
with col_hist:
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0b1219")
    ax.set_facecolor("#0b1219")

    scores = df["migration_pressure_score"]

    # Shade category regions
    q33 = scores.quantile(1 / 3)
    q67 = scores.quantile(2 / 3)
    ax.axvspan(scores.min(), q33, alpha=0.08, color="#2ecc71")
    ax.axvspan(q33,           q67, alpha=0.08, color="#f39c12")
    ax.axvspan(q67, scores.max(), alpha=0.08, color="#e74c3c")

    ax.hist(scores, bins=32, color="#2a5f8f", edgecolor="#0b1219", linewidth=0.4, zorder=3)

    # Selected district marker
    ax.axvline(score, color="#e8f0f8", linewidth=1.5, linestyle="--", zorder=5, alpha=0.9)
    ax.text(score + 0.005, ax.get_ylim()[1] * 0.93,
            f"{selected_district[:14]}", color="#e8f0f8", fontsize=8, va="top")

    # Category labels
    for x, label, c in [(scores.min(), "Low", "#2ecc71"),
                         (q33,          "Medium", "#f39c12"),
                         (q67,          "High", "#e74c3c")]:
        ax.text(x + 0.003, ax.get_ylim()[1] * 0.05, label,
                color=c, fontsize=8, alpha=0.75, fontweight="600")

    ax.set_xlabel("Migration pressure score", color="#637d98", fontsize=10)
    ax.set_ylabel("Number of districts",      color="#637d98", fontsize=10)
    ax.set_title("Score distribution — all 640 districts", color="#c8d6e5", fontsize=11, pad=12)
    ax.tick_params(colors="#637d98", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d3d")
    ax.grid(axis="y", color="#1e2d3d", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── Feature importance bar chart ────────────────────────────────────────────
with col_imp:
    importances = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values()

    labels = [FEATURE_LABELS[f] for f in importances.index]
    values = importances.values

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0b1219")
    ax.set_facecolor("#0b1219")

    # Color bars by feature group
    group_colors = {
        "rainfall": "#3498db",
        "drought":  "#3498db",
        "flood":    "#3498db",
        "yield":    "#2ecc71",
        "production": "#2ecc71",
        "mpi":      "#e74c3c",
        "headcount": "#e74c3c",
        "marginal": "#9b59b6",
        "worker":   "#9b59b6",
    }
    bar_colors = [
        next((c for k, c in group_colors.items() if k in f), "#637d98")
        for f in importances.index
    ]

    bars = ax.barh(labels, values, color=bar_colors, edgecolor="#0b1219", linewidth=0.3)

    for bar, val in zip(bars, values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="#c8d6e5", fontsize=8)

    ax.set_xlabel("Importance", color="#637d98", fontsize=10)
    ax.set_title("Random Forest — feature importances", color="#c8d6e5", fontsize=11, pad=12)
    ax.tick_params(colors="#637d98", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d3d")
    ax.grid(axis="x", color="#1e2d3d", linewidth=0.5, alpha=0.7)

    # Legend
    legend_items = [
        mpatches.Patch(color="#3498db", label="Climate"),
        mpatches.Patch(color="#2ecc71", label="Agriculture"),
        mpatches.Patch(color="#e74c3c", label="Poverty"),
        mpatches.Patch(color="#9b59b6", label="Demographics"),
    ]
    ax.legend(handles=legend_items, loc="lower right",
              fontsize=8, framealpha=0, labelcolor="#c8d6e5")

    ax.set_xlim(0, values.max() * 1.18)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Top 10 high-risk districts table
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Top 10 highest-pressure districts</div>', unsafe_allow_html=True)

top10 = (
    df.nlargest(10, "migration_pressure_score")[
        ["rank", "state", "district",
         "migration_pressure_score", "migration_pressure_category",
         "score_climate", "score_agriculture", "score_poverty", "score_demographics"]
    ]
    .rename(columns={
        "rank":                       "Rank",
        "state":                      "State",
        "district":                   "District",
        "migration_pressure_score":   "Score",
        "migration_pressure_category": "Category",
        "score_climate":              "Climate",
        "score_agriculture":          "Agriculture",
        "score_poverty":              "Poverty",
        "score_demographics":         "Demographics",
    })
    .reset_index(drop=True)
)

top10["Score"]       = top10["Score"].round(4)
top10["Climate"]     = top10["Climate"].round(3)
top10["Agriculture"] = top10["Agriculture"].round(3)
top10["Poverty"]     = top10["Poverty"].round(3)
top10["Demographics"]= top10["Demographics"].round(3)

def _highlight_category(val):
    color = CATEGORY_COLORS.get(str(val), "")
    return f"color: {color}; font-weight: 600;" if color else ""

def _highlight_score(val):
    return "font-family: 'DM Mono', monospace; font-weight: 500;" if isinstance(val, float) else ""

styled = (
    top10.style
    .applymap(_highlight_category, subset=["Category"])
    .set_properties(**{
        "background-color": "#0b1219",
        "color": "#c8d6e5",
        "border": "1px solid #1e2d3d",
        "font-size": "13px",
    })
    .set_table_styles([
        {"selector": "th",
         "props": [("background-color", "#0d1520"),
                   ("color", "#637d98"),
                   ("font-size", "10px"),
                   ("letter-spacing", "0.08em"),
                   ("text-transform", "uppercase"),
                   ("border", "1px solid #1e2d3d")]},
        {"selector": "tr:hover td",
         "props": [("background-color", "#1a2535")]},
    ])
    .hide(axis="index")
    .format({"Score": "{:.4f}", "Climate": "{:.3f}",
             "Agriculture": "{:.3f}", "Poverty": "{:.3f}",
             "Demographics": "{:.3f}"})
)

st.dataframe(
    top10,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Rank":         st.column_config.NumberColumn("Rank",   width="small"),
        "Score":        st.column_config.NumberColumn("Score",  format="%.4f"),
        "Climate":      st.column_config.ProgressColumn("Climate",    min_value=0, max_value=1, format="%.3f"),
        "Agriculture":  st.column_config.ProgressColumn("Agriculture", min_value=0, max_value=1, format="%.3f"),
        "Poverty":      st.column_config.ProgressColumn("Poverty",    min_value=0, max_value=1, format="%.3f"),
        "Demographics": st.column_config.ProgressColumn("Demographics",min_value=0, max_value=1, format="%.3f"),
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# Top 10 lowest-pressure districts table
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Top 10 lowest-pressure districts</div>', unsafe_allow_html=True)

bottom10 = (
    df.nsmallest(10, "migration_pressure_score")[[
        "rank", "state", "district",
        "migration_pressure_score", "migration_pressure_category",
        "score_climate", "score_agriculture", "score_poverty", "score_demographics"
    ]]
    .rename(columns={
        "rank":                       "Rank",
        "state":                      "State",
        "district":                   "District",
        "migration_pressure_score":   "Score",
        "migration_pressure_category": "Category",
        "score_climate":              "Climate",
        "score_agriculture":          "Agriculture",
        "score_poverty":              "Poverty",
        "score_demographics":         "Demographics",
    })
    .reset_index(drop=True)
)

# Round values
bottom10["Score"]       = bottom10["Score"].round(4)
bottom10["Climate"]     = bottom10["Climate"].round(3)
bottom10["Agriculture"] = bottom10["Agriculture"].round(3)
bottom10["Poverty"]     = bottom10["Poverty"].round(3)
bottom10["Demographics"]= bottom10["Demographics"].round(3)

# Display (same style as top table)
st.dataframe(
    bottom10,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Rank":         st.column_config.NumberColumn("Rank",   width="small"),
        "Score":        st.column_config.NumberColumn("Score",  format="%.4f"),
        "Climate":      st.column_config.ProgressColumn("Climate",    min_value=0, max_value=1, format="%.3f"),
        "Agriculture":  st.column_config.ProgressColumn("Agriculture", min_value=0, max_value=1, format="%.3f"),
        "Poverty":      st.column_config.ProgressColumn("Poverty",    min_value=0, max_value=1, format="%.3f"),
        "Demographics": st.column_config.ProgressColumn("Demographics",min_value=0, max_value=1, format="%.3f"),
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<span style='font-size:11px;color:#3d5166;'>"
    "Data sources: Census PCA 2011 · IMD Rainfall 1901–2017 · "
    "Agriculture 2000–2011 · MPI 2015–16 · "
    "Score = weighted sum of MinMax-scaled climate, agriculture, poverty &amp; demographic features."
    "</span>",
    unsafe_allow_html=True,
)