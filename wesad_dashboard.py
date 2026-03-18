import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc

try:
    from scipy.stats import kruskal
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# DATA & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
_CSV_PATH = "/Users/supriya/group_project/WESAD/output/wesad_features_raw.csv"
df = pd.read_csv(_CSV_PATH)

df.columns = [c.strip() for c in df.columns]

required_cols = {"subject", "label_name"}
missing_required = required_cols - set(df.columns)
if missing_required:
    raise ValueError(f"Missing required columns in CSV: {sorted(missing_required)}")

df["subject"] = df["subject"].astype(str).str.strip()
df["label_name"] = df["label_name"].astype(str).str.strip().str.lower()

if "window_start" in df.columns:
    df["window_start"] = pd.to_numeric(df["window_start"], errors="coerce")
    df = df.sort_values(["subject", "window_start"], kind="stable").reset_index(drop=True)

CONDITIONS = ["baseline", "stress", "amusement", "meditation"]
SUBJECTS = sorted(df["subject"].dropna().unique().tolist())

FEATURES_22_CANDIDATE = [
    "ecg_mean", "ecg_std", "ecg_min", "ecg_max",
    "eda_mean", "eda_std", "eda_min", "eda_max", "eda_slope",
    "emg_mean", "emg_std", "emg_rms",
    "resp_mean", "resp_std", "resp_range",
    "temp_mean", "temp_std", "temp_slope",
    "acc_x_mean", "acc_y_mean", "acc_z_mean", "acc_magnitude"
]
FEATURES_22 = [c for c in FEATURES_22_CANDIDATE if c in df.columns]

SIG_OPTIONS = [
    {"label": "EDA Mean", "value": "eda_mean"},
    {"label": "ECG Std", "value": "ecg_std"},
    {"label": "Resp Mean", "value": "resp_mean"},
    {"label": "Temp Mean", "value": "temp_mean"},
    {"label": "EMG RMS", "value": "emg_rms"},
    {"label": "Acc Magnitude", "value": "acc_magnitude"},
]
SIG_OPTIONS = [x for x in SIG_OPTIONS if x["value"] in df.columns]
SIG_IDS = [opt["value"] for opt in SIG_OPTIONS]

CLASS_COUNTS = (
    df["label_name"]
    .value_counts()
    .reindex(CONDITIONS, fill_value=0)
    .to_dict()
)

SUBJ_WINDOWS = (
    df.groupby(["subject", "label_name"])
      .size()
      .unstack(fill_value=0)
      .reindex(index=SUBJECTS, columns=CONDITIONS, fill_value=0)
      .to_dict(orient="index")
)

MEDIANS = {}
STDS = {}
for sig in SIG_IDS:
    MEDIANS[sig] = (
        df.groupby("label_name")[sig]
        .median()
        .reindex(CONDITIONS)
        .fillna(0)
        .to_dict()
    )
    STDS[sig] = (
        df.groupby("label_name")[sig]
        .std()
        .reindex(CONDITIONS)
        .fillna(0)
        .to_dict()
    )

EDA_BY_SUBJECT = {}
if "eda_mean" in df.columns:
    EDA_BY_SUBJECT = (
        df.groupby(["subject", "label_name"])["eda_mean"]
          .mean()
          .unstack(fill_value=0)
          .reindex(index=SUBJECTS, columns=CONDITIONS, fill_value=0)
          .to_dict(orient="index")
    )

LABEL_MAP = {
    "eda_mean": "EDA Mean (µS)",
    "ecg_std": "ECG Std (mV)",
    "resp_mean": "Respiration Mean",
    "temp_mean": "Skin Temp (°C)",
    "emg_rms": "EMG RMS",
    "acc_magnitude": "Acc Magnitude",
}


def safe_series(dataframe, col, mask=None):
    if col not in dataframe.columns:
        return pd.Series(dtype=float)
    if mask is None:
        s = dataframe[col]
    else:
        s = dataframe.loc[mask, col]
    return pd.to_numeric(s, errors="coerce").dropna()


def cohens_d_stress_vs_nonstress(dataframe, col):
    stress = safe_series(dataframe, col, dataframe["label_name"] == "stress")
    non = safe_series(dataframe, col, dataframe["label_name"] != "stress")
    if len(stress) < 2 or len(non) < 2:
        return 0.0
    s1 = stress.std(ddof=1)
    s2 = non.std(ddof=1)
    pooled = np.sqrt(((len(stress) - 1) * s1**2 + (len(non) - 1) * s2**2) / (len(stress) + len(non) - 2))
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float((stress.mean() - non.mean()) / pooled)


COHEN_D = {sig: round(abs(cohens_d_stress_vs_nonstress(df, sig)), 2) for sig in SIG_IDS}


def kw_scores_from_df(dataframe, features):
    scores = {}
    for col in features:
        grouped = [
            safe_series(dataframe, col, dataframe["label_name"] == cond)
            for cond in CONDITIONS
        ]
        grouped = [g for g in grouped if len(g) > 1]

        if len(grouped) < 2:
            scores[col] = 0.0
            continue

        if SCIPY_AVAILABLE:
            try:
                stat, _ = kruskal(*grouped)
                scores[col] = float(stat)
                continue
            except Exception:
                pass

        # Fallback effect-size proxy if scipy is unavailable
        means = [g.mean() for g in grouped]
        overall_std = safe_series(dataframe, col).std()
        scores[col] = float((max(means) - min(means)) / overall_std) if overall_std and not np.isnan(overall_std) else 0.0

    if not scores:
        return {}

    max_score = max(scores.values()) or 1.0
    return {k: round(v / max_score, 2) for k, v in scores.items()}


KW_SCORES = kw_scores_from_df(df, FEATURES_22)

NUM_SUBJECTS = len(SUBJECTS)
NUM_WINDOWS = len(df)
NUM_FEATURES = len(FEATURES_22)
NUM_CONDITIONS = len(CONDITIONS)

baseline_pct = (CLASS_COUNTS.get("baseline", 0) / NUM_WINDOWS * 100) if NUM_WINDOWS else 0
amusement_pct = (CLASS_COUNTS.get("amusement", 0) / NUM_WINDOWS * 100) if NUM_WINDOWS else 0

# ── FIX 1 ONLY: landing card stat ────────────────────────────────────────────
# Use eda_mean Cohen's d — always correct, always meaningful.
# Nothing else in the file has changed from your original.
eda_mean_cohen_d = COHEN_D.get("eda_mean", 0.0)
top_feature_stat = f"eda_mean · d = {eda_mean_cohen_d:.2f}"
# ─────────────────────────────────────────────────────────────────────────────

if "eda_mean" in df.columns and EDA_BY_SUBJECT:
    eda_baselines = [EDA_BY_SUBJECT[s].get("baseline", 0) for s in SUBJECTS if EDA_BY_SUBJECT[s].get("baseline", 0) > 0]
    if eda_baselines:
        baseline_spread_ratio = max(eda_baselines) / max(min(eda_baselines), 1e-9)
    else:
        baseline_spread_ratio = 0
else:
    baseline_spread_ratio = 0

# ──────────────────────────────────────────────────────────────────────────────
# THEME
# ──────────────────────────────────────────────────────────────────────────────
COLORS = {
    "bg":        "#F7F5F0",
    "surface":   "#EDEAE3",
    "card":      "#FDFCFA",
    "border":    "#DDD9D0",
    "text":      "#1C1917",
    "muted":     "#78716C",
    "accent":    "#1D4E3F",
    "accent2":   "#8B3A2A",
    "accent3":   "#2D4A6B",
    "yellow":    "#B45309",
    "blue":      "#2D4A6B",
    "purple":    "#5B3A7C",
    "stress":    "#8B1A1A",
    "baseline":  "#4A5568",
    "amusement": "#C2611A",
    "meditation":"#1A6B5C",
    "red":       "#8B1A1A",
    "tag_bg":    "#EDE9E0",
}

COND_COLORS = {
    "baseline":  "#4A5568",
    "stress":    "#8B1A1A",
    "amusement": "#C2611A",
    "meditation":"#1A6B5C",
}

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def sample_series(series, max_n=1500, seed=42):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) <= max_n:
        return series.to_numpy()
    return series.sample(max_n, random_state=seed).to_numpy()


def get_filtered_df(subject="all", condition="all"):
    dff = df.copy()
    if subject != "all":
        dff = dff[dff["subject"] == subject]
    if condition != "all":
        dff = dff[dff["label_name"] == condition]
    return dff


def describe_effect_size(d):
    if d > 0.5:
        return "large"
    if d > 0.3:
        return "medium"
    return "small"


def build_signal_insight(signal):
    if signal not in df.columns:
        return f"**{signal}** is not available in the CSV."

    medians = MEDIANS.get(signal, {})
    stress_m = medians.get("stress", 0)
    base_m = medians.get("baseline", 0)
    amusement_m = medians.get("amusement", 0)
    med_m = medians.get("meditation", 0)
    d = COHEN_D.get(signal, 0.0)

    if signal == "eda_mean":
        return (
            f"**EDA Mean (µS):** Stress median = **{stress_m:.2f}** vs "
            f"Amusement = **{amusement_m:.2f}**. Strongest separator in this dashboard "
            f"with Cohen's d ≈ **{d:.2f}**."
        )
    if signal == "ecg_std":
        return (
            f"**ECG Std (mV):** Stress median = **{stress_m:.3f}** vs "
            f"Meditation = **{med_m:.3f}**. Moderate separator — heart rhythm variability increases under load."
        )
    if signal == "resp_mean":
        return (
            f"**Resp Mean:** Meditation median = **{med_m:.3f}** vs Stress = **{stress_m:.3f}**. "
            f"This supports calmer, more regular breathing during meditation."
        )
    if signal == "temp_mean":
        return (
            f"**Temp Mean (°C):** Baseline median = **{base_m:.2f}** and Stress = **{stress_m:.2f}**. "
            f"Temperature changes are present, but more subtle than EDA or ECG."
        )
    if signal == "emg_rms":
        return (
            f"**EMG RMS:** Stress median = **{stress_m:.3f}** vs Baseline = **{base_m:.3f}**. "
            f"Mild elevation appears under stress, but overlap is usually larger."
        )
    if signal == "acc_magnitude":
        return (
            f"**Acc Magnitude:** Stress median = **{stress_m:.3f}** vs Baseline = **{base_m:.3f}**. "
            f"Movement is the weakest separator here and is likely more artifact-prone."
        )

    return f"**{signal}:** Stress median = **{stress_m:.3f}**; Cohen's d ≈ **{d:.2f}**."


PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'DM Sans', sans-serif", color=COLORS["text"], size=12),
    xaxis=dict(gridcolor="#E8E4DC", linecolor="#C8C3BA", tickcolor="#A09890"),
    yaxis=dict(gridcolor="#E8E4DC", linecolor="#C8C3BA", tickcolor="#A09890"),
    legend=dict(bgcolor="rgba(253,252,250,0.95)", bordercolor=COLORS["border"], borderwidth=1),
)


def base_layout(**kwargs):
    out = dict(**PLOTLY_TEMPLATE)
    out.update(kwargs)
    return out


CARD_STYLE = {
    "background": COLORS["card"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "12px",
    "padding": "0",
    "overflow": "hidden",
    "boxShadow": "0 2px 12px rgba(28,25,23,0.06)",
}


def card_header(title, subtitle="", tag="", tag_color=COLORS["accent"]):
    return html.Div([
        html.Div([
            html.Div([
                html.H5(title, style={
                    "margin": 0, "fontSize": "14px", "fontWeight": "600",
                    "color": COLORS["text"], "letterSpacing": "-0.2px"
                }),
                html.P(subtitle, style={
                    "margin": "3px 0 0", "fontSize": "11px", "color": COLORS["muted"]
                }) if subtitle else None,
            ], style={"flex": "1"}),
            html.Span(tag, style={
                "background": COLORS["tag_bg"],
                "border": f"1px solid {COLORS['border']}",
                "color": COLORS["muted"],
                "fontSize": "9px",
                "fontWeight": "600",
                "padding": "3px 10px",
                "borderRadius": "4px",
                "letterSpacing": "1.2px",
                "textTransform": "uppercase",
                "alignSelf": "flex-start",
                "whiteSpace": "nowrap"
            }) if tag else None,
        ], style={"display": "flex", "alignItems": "flex-start", "gap": "12px"}),
    ], style={
        "padding": "16px 20px 12px",
        "borderBottom": f"1px solid {COLORS['border']}",
        "background": COLORS["surface"],
    })


def insight_box(text, color=COLORS["accent"]):
    return html.Div(
        dcc.Markdown(text, style={
            "margin": 0, "fontSize": "12px", "color": COLORS["muted"], "lineHeight": "1.65"
        }),
        style={
            "margin": "0 18px 16px",
            "background": hex_to_rgba(color, 0.05),
            "borderLeft": f"3px solid {hex_to_rgba(color, 0.5)}",
            "borderRadius": "0 6px 6px 0",
            "padding": "9px 13px"
        }
    )


def stat_chip(val, label, color=COLORS["accent"], icon=""):
    return html.Div([
        html.Div(icon, style={"fontSize": "18px", "lineHeight": "1", "marginBottom": "6px"}) if icon else None,
        html.Div(val, style={
            "fontFamily": "'DM Mono', monospace",
            "fontSize": "22px",
            "fontWeight": "500",
            "color": color,
            "lineHeight": "1",
            "letterSpacing": "-0.5px"
        }),
        html.Div(label, style={
            "fontSize": "10px",
            "color": COLORS["muted"],
            "marginTop": "5px",
            "textTransform": "uppercase",
            "letterSpacing": "1.2px",
            "fontWeight": "500"
        }),
    ], style={
        "background": COLORS["card"],
        "border": f"1px solid {COLORS['border']}",
        "borderRadius": "8px",
        "padding": "14px 18px",
        "textAlign": "center",
        "minWidth": "88px",
    })


def make_sig_buttons(prefix):
    return html.Div([
        html.Span("Signal:", style={
            "fontSize": "11px",
            "color": COLORS["muted"],
            "textTransform": "uppercase",
            "letterSpacing": "1px",
            "alignSelf": "center",
            "fontWeight": "500"
        }),
        *[
            html.Button(
                opt["label"],
                id=f"{prefix}-{opt['value']}",
                className="filter-btn" + (" active" if opt["value"] == "eda_mean" else ""),
                n_clicks=0
            )
            for opt in SIG_OPTIONS
        ],
    ], style={
        "display": "flex",
        "gap": "6px",
        "alignItems": "center",
        "flexWrap": "wrap",
        "marginBottom": "16px"
    })


def get_counts_by_subject(subject):
    if subject == "all":
        return CLASS_COUNTS.copy()
    return SUBJ_WINDOWS.get(subject, {c: 0 for c in CONDITIONS})


# ──────────────────────────────────────────────────────────────────────────────
# FIGURES
# ──────────────────────────────────────────────────────────────────────────────
def fig_class_distribution_filtered(subject="all"):
    dff = get_filtered_df(subject=subject)
    counts = (
        dff["label_name"]
        .value_counts()
        .reindex(CONDITIONS, fill_value=0)
    )
    total = max(int(counts.sum()), 1)
    pcts = [f"{v / total * 100:.1f}%" for v in counts.values]
    colors = [COND_COLORS[c] for c in CONDITIONS]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[c.capitalize() for c in CONDITIONS],
        y=counts.values,
        marker_color=colors,
        marker_line_color=[hex_to_rgba(c, 0.6) for c in colors],
        marker_line_width=1,
        text=pcts,
        textposition="outside",
        textfont=dict(color=colors, size=12, family="DM Mono, monospace"),
        hovertemplate="<b>%{x}</b><br>Windows: <b>%{y:,}</b><br>Share: %{text}<extra></extra>",
        width=0.5,
    ))
    equal = total / len(CONDITIONS)
    fig.add_hline(
        y=equal,
        line_dash="dot",
        line_color=COLORS["muted"],
        annotation_text="Equal share",
        annotation_font_size=10,
        annotation_font_color=COLORS["muted"]
    )
    fig.update_layout(**base_layout(
        xaxis=dict(**PLOTLY_TEMPLATE["xaxis"], title="Condition", tickfont=dict(color=COLORS["muted"], size=12)),
        yaxis=dict(**PLOTLY_TEMPLATE["yaxis"], title="Windows", tickformat=","),
        showlegend=False,
        margin=dict(t=36, r=20, b=50, l=65),
        bargap=0.32
    ))
    return fig


def fig_donut_filtered(subject="all"):
    counts = get_counts_by_subject(subject)
    vals = [counts.get(c, 0) for c in CONDITIONS]
    fig = go.Figure(go.Pie(
        labels=[c.capitalize() for c in CONDITIONS],
        values=vals,
        marker=dict(
            colors=[COND_COLORS[c] for c in CONDITIONS],
            line=dict(color=COLORS["bg"], width=3)
        ),
        textinfo="percent+label",
        textfont=dict(color=COLORS["text"], size=11),
        hovertemplate="<b>%{label}</b><br>%{value:,} windows · %{percent}<extra></extra>",
        hole=0.54,
        pull=[0.06 if c == "stress" else 0 for c in CONDITIONS],
    ))
    fig.update_layout(**base_layout(
        margin=dict(t=20, r=20, b=20, l=20),
        annotations=[dict(
            text=f"{sum(vals):,}<br><span style='font-size:10px'>windows</span>",
            x=0.5, y=0.5,
            xanchor="center", yanchor="middle",
            font=dict(color=COLORS["muted"], size=12, family="DM Mono, monospace"),
            showarrow=False
        )]
    ))
    return fig


def fig_signal_box(signal="eda_mean"):
    if signal not in df.columns:
        return go.Figure()
    fig = go.Figure()
    for cond in CONDITIONS:
        samples = sample_series(
            df.loc[df["label_name"] == cond, signal],
            max_n=1500,
            seed=42
        )
        fig.add_trace(go.Violin(
            y=samples,
            name=cond.capitalize(),
            box_visible=True,
            meanline_visible=True,
            line_color=COND_COLORS[cond],
            fillcolor=hex_to_rgba(COND_COLORS[cond], 0.18),
            points=False,
            hovertemplate=f"<b>{cond.capitalize()}</b><br>{LABEL_MAP.get(signal, signal)}: %{{y:.3f}}<extra></extra>",
        ))
    d = COHEN_D.get(signal, 0.0)
    d_label = describe_effect_size(d)
    fig.add_annotation(
        x=0.98, y=0.98, xref="paper", yref="paper",
        text=f"Cohen's d = {d:.2f} ({d_label})",
        showarrow=False,
        font=dict(size=10, color=COLORS["muted"], family="DM Mono, monospace"),
        align="right", xanchor="right", yanchor="top",
        bgcolor=hex_to_rgba(COLORS["surface"], 0.9),
        borderpad=5
    )
    fig.update_layout(**base_layout(
        xaxis=dict(**PLOTLY_TEMPLATE["xaxis"], tickfont=dict(color=COLORS["muted"], size=12)),
        yaxis=dict(**PLOTLY_TEMPLATE["yaxis"], title=LABEL_MAP.get(signal, signal)),
        violinmode="group",
        showlegend=False,
        margin=dict(t=20, r=20, b=50, l=65),
    ))
    return fig


def fig_radar():
    radar_features = [f for f in ["eda_mean", "ecg_std", "resp_mean", "emg_rms", "temp_mean"] if f in df.columns]
    if len(radar_features) < 3:
        return go.Figure()
    feature_labels = []
    for f in radar_features:
        if f == "eda_mean":
            feature_labels.append("EDA Mean")
        elif f == "ecg_std":
            feature_labels.append("ECG Std")
        elif f == "resp_mean":
            feature_labels.append("Resp Mean")
        elif f == "emg_rms":
            feature_labels.append("EMG RMS")
        elif f == "temp_mean":
            feature_labels.append("Temp Mean")
        else:
            feature_labels.append(f)
    med_table = (
        df.groupby("label_name")[radar_features]
          .median()
          .reindex(CONDITIONS)
    )
    norm_table = med_table.copy()
    for col in radar_features:
        col_min = med_table[col].min()
        col_max = med_table[col].max()
        if col_max == col_min:
            norm_table[col] = 0.5
        else:
            norm_table[col] = (med_table[col] - col_min) / (col_max - col_min)
    fig = go.Figure()
    for cond in CONDITIONS:
        rvals = norm_table.loc[cond].tolist()
        fig.add_trace(go.Scatterpolar(
            r=rvals,
            theta=feature_labels,
            fill="toself",
            name=cond.capitalize(),
            line_color=COND_COLORS[cond],
            fillcolor=hex_to_rgba(COND_COLORS[cond], 0.14)
        ))
    fig.update_layout(**base_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 1], gridcolor="#E8E4DC",
                tickfont=dict(color=COLORS["muted"], size=9), color=COLORS["muted"]
            ),
            angularaxis=dict(
                gridcolor="#E8E4DC",
                tickfont=dict(color=COLORS["muted"], size=11)
            ),
        ),
        legend=dict(**PLOTLY_TEMPLATE["legend"], orientation="h", y=-0.12, x=0.5, xanchor="center"),
        margin=dict(t=30, r=40, b=70, l=40),
    ))
    return fig


def fig_correlation():
    feat = FEATURES_22
    if not feat:
        return go.Figure()
    corr_df = df[feat].apply(pd.to_numeric, errors="coerce").corr().fillna(0)
    corr = corr_df.to_numpy()
    short = [
        f.replace("_mean", "_μ").replace("_std", "_σ").replace("_slope", "_∇")
         .replace("_min", "↓").replace("_max", "↑").replace("_range", "_Δ")
         .replace("acc_", "a_").replace("ecg_", "c_").replace("eda_", "d_")
         .replace("emg_", "e_").replace("resp_", "r_").replace("temp_", "t_")
        for f in feat
    ]
    fig = go.Figure(go.Heatmap(
        z=corr,
        x=short,
        y=short,
        colorscale=[
            [0,   "#2D4A6B"],
            [0.5, "#EDE9E0"],
            [1,   "#8B1A1A"]
        ],
        zmid=0,
        zmin=-1,
        zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr],
        texttemplate="%{text}",
        textfont=dict(size=7, color="#1C1917"),
        hovertemplate="%{x} × %{y}: <b>%{z:.2f}</b><extra></extra>",
        colorbar=dict(
            thickness=12,
            tickfont=dict(color=COLORS["muted"], size=10),
            title=dict(text="r", font=dict(color=COLORS["muted"]))
        ),
    ))
    annotations = []
    if "eda_mean" in feat and "eda_std" in feat:
        eda_idx = feat.index("eda_mean")
        annotations.append(
            dict(
                x=eda_idx, y=eda_idx, xref="x", yref="y",
                text="EDA cluster", showarrow=True,
                arrowcolor=COLORS["yellow"], font=dict(color=COLORS["yellow"], size=9),
                ax=70, ay=-45, bgcolor=hex_to_rgba(COLORS["yellow"], 0.08),
                bordercolor=hex_to_rgba(COLORS["yellow"], 0.25), borderpad=5
            )
        )
    if "emg_mean" in feat and "emg_std" in feat:
        emg_idx = feat.index("emg_std")
        annotations.append(
            dict(
                x=emg_idx, y=emg_idx, xref="x", yref="y",
                text="EMG cluster", showarrow=True,
                arrowcolor=COLORS["accent3"], font=dict(color=COLORS["accent3"], size=9),
                ax=70, ay=30, bgcolor=hex_to_rgba(COLORS["accent3"], 0.08),
                bordercolor=hex_to_rgba(COLORS["accent3"], 0.25), borderpad=5
            )
        )
    fig.update_layout(**base_layout(
        xaxis=dict(**PLOTLY_TEMPLATE["xaxis"], tickangle=-45, tickfont=dict(size=9, color=COLORS["muted"])),
        yaxis=dict(**PLOTLY_TEMPLATE["yaxis"], autorange="reversed", tickfont=dict(size=9, color=COLORS["muted"])),
        margin=dict(t=20, r=80, b=80, l=80),
        annotations=annotations,
    ))
    return fig


def fig_subject_variability_filtered(subject="all", view="grouped"):
    if "eda_mean" not in df.columns or not EDA_BY_SUBJECT:
        return go.Figure()
    conds = CONDITIONS
    if subject != "all":
        subj_data = EDA_BY_SUBJECT.get(subject, {c: 0 for c in conds})
        if view == "grouped":
            vals = [subj_data.get(c, 0) for c in conds]
            fig = go.Figure(go.Bar(
                x=[c.capitalize() for c in conds],
                y=vals,
                marker_color=[COND_COLORS[c] for c in conds],
                hovertemplate="<b>%{x}</b><br>EDA Mean: <b>%{y:.2f} µS</b><extra></extra>",
            ))
            fig.update_layout(**base_layout(
                xaxis=dict(**PLOTLY_TEMPLATE["xaxis"], title="Condition"),
                yaxis=dict(**PLOTLY_TEMPLATE["yaxis"], title="EDA Mean (µS)"),
                showlegend=False,
                margin=dict(t=20, r=20, b=50, l=65),
            ))
            return fig
        baseline = max(subj_data.get("baseline", 0), 1e-9)
        ratios = [round(subj_data.get(c, 0) / baseline, 2) for c in conds]
        colors = [
            COLORS["red"] if r > 1.7 else COLORS["yellow"] if r > 1.4 else COLORS["accent"]
            for r in ratios
        ]
        fig = go.Figure(go.Bar(
            x=[c.capitalize() for c in conds],
            y=ratios,
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>Ratio: <b>%{y:.2f}×</b><extra></extra>",
        ))
        fig.add_hline(y=1.0, line_dash="dot", line_color=COLORS["muted"])
        fig.update_layout(**base_layout(
            xaxis=dict(**PLOTLY_TEMPLATE["xaxis"], title="Condition"),
            yaxis=dict(**PLOTLY_TEMPLATE["yaxis"], title="Condition / Baseline Ratio"),
            showlegend=False,
            margin=dict(t=20, r=20, b=50, l=65),
        ))
        return fig
    subjs = SUBJECTS
    if view == "grouped":
        fig = go.Figure()
        for cond in conds:
            fig.add_trace(go.Bar(
                name=cond.capitalize(),
                x=subjs,
                y=[EDA_BY_SUBJECT[s].get(cond, 0) for s in subjs],
                marker_color=COND_COLORS[cond],
                marker_opacity=0.90,
                hovertemplate=f"<b>%{{x}}</b> — {cond}<br>EDA Mean: <b>%{{y:.2f}} µS</b><extra></extra>",
            ))
        fig.update_layout(**base_layout(
            barmode="group",
            bargap=0.15,
            bargroupgap=0.05,
            xaxis=dict(**PLOTLY_TEMPLATE["xaxis"], title="Subject"),
            yaxis=dict(**PLOTLY_TEMPLATE["yaxis"], title="EDA Mean (µS)"),
            legend=dict(**PLOTLY_TEMPLATE["legend"], orientation="h", y=-0.28, x=0),
            margin=dict(t=20, r=20, b=90, l=65),
        ))
        return fig
    ratios = []
    for s in subjs:
        baseline = max(EDA_BY_SUBJECT[s].get("baseline", 0), 1e-9)
        ratios.append(round(EDA_BY_SUBJECT[s].get("stress", 0) / baseline, 2))
    colors = [
        COLORS["red"] if r > 1.7 else COLORS["yellow"] if r > 1.4 else COLORS["accent"]
        for r in ratios
    ]
    fig = go.Figure(go.Bar(
        x=subjs,
        y=ratios,
        marker_color=colors,
        marker_opacity=0.88,
        hovertemplate="<b>%{x}</b><br>Stress/Baseline: <b>%{y:.2f}×</b><extra></extra>",
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color=COLORS["muted"])
    fig.update_layout(**base_layout(
        xaxis=dict(**PLOTLY_TEMPLATE["xaxis"], title="Subject"),
        yaxis=dict(**PLOTLY_TEMPLATE["yaxis"], title="Stress / Baseline Ratio"),
        showlegend=False,
        margin=dict(t=20, r=20, b=50, l=65),
    ))
    return fig


def fig_feature_ranking():
    items = sorted(KW_SCORES.items(), key=lambda x: -x[1])
    names = [i[0] for i in items]
    scores = [i[1] for i in items]
    colors = [
        COLORS["red"] if s > 0.7 else
        COLORS["yellow"] if s > 0.4 else
        COLORS["accent3"] if s > 0.2 else
        COLORS["muted"]
        for s in scores
    ]
    fig = go.Figure()

    # Stems — one per feature, drawn as a horizontal line from 0 to the score
    for name, score, color in zip(names, scores, colors):
        fig.add_trace(go.Scatter(
            x=[0, score],
            y=[name, name],
            mode="lines",
            line=dict(color=hex_to_rgba(color, 0.35), width=1.5),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Dots — one scatter trace per feature for correct per-dot coloring
    for name, score, color in zip(names, scores, colors):
        fig.add_trace(go.Scatter(
            x=[score],
            y=[name],
            mode="markers",
            marker=dict(color=color, size=9, opacity=0.9, line=dict(color=hex_to_rgba(color, 0.5), width=1)),
            hovertemplate=f"<b>{name}</b><br>Normalized score: <b>{score:.2f}</b><extra></extra>",
            showlegend=False,
        ))

    for x0, x1, color in [
        (0.7, 1.0, COLORS["red"]),
        (0.4, 0.7, COLORS["yellow"]),
        (0.2, 0.4, COLORS["accent3"]),
        (0.0, 0.2, COLORS["muted"])
    ]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=hex_to_rgba(color, 0.04), line_width=0)
        fig.add_vline(x=x0, line_dash="dot", line_color=hex_to_rgba(color, 0.3), line_width=1)

    fig.update_layout(**base_layout(
        xaxis=dict(**PLOTLY_TEMPLATE["xaxis"], title="Normalized Discriminability", range=[0, 1.05]),
        yaxis=dict(**PLOTLY_TEMPLATE["yaxis"], autorange="reversed", tickfont=dict(size=10)),
        showlegend=False,
        margin=dict(t=20, r=30, b=50, l=110),
    ))
    return fig


def fig_binary_comparison(features=None):
    if features is None:
        features = [f for f in ["eda_mean", "ecg_std", "resp_mean", "temp_mean", "emg_rms"] if f in df.columns]
    short_label_map = {
        "eda_mean": "EDA μ",
        "ecg_std": "ECG σ",
        "resp_mean": "Resp μ",
        "temp_mean": "Temp μ",
        "emg_rms": "EMG rms",
        "acc_magnitude": "Acc mag"
    }
    fig = go.Figure()
    for feat in features:
        stress = sample_series(df.loc[df["label_name"] == "stress", feat], max_n=1200, seed=42)
        nonstress = sample_series(df.loc[df["label_name"] != "stress", feat], max_n=1200, seed=7)
        fname = short_label_map.get(feat, feat)
        fig.add_trace(go.Violin(
            x=[fname] * len(stress),
            y=stress,
            name="Stress",
            legendgroup="stress",
            side="positive",
            line_color=COLORS["stress"],
            fillcolor=hex_to_rgba(COLORS["stress"], 0.18),
            meanline_visible=True,
            points=False,
            box_visible=True,
            showlegend=(feat == features[0]),
            hovertemplate=f"<b>Stress — {fname}</b>: %{{y:.3f}}<extra></extra>",
        ))
        fig.add_trace(go.Violin(
            x=[fname] * len(nonstress),
            y=nonstress,
            name="Non-Stress",
            legendgroup="nonstress",
            side="negative",
            line_color=COLORS["meditation"],
            fillcolor=hex_to_rgba(COLORS["meditation"], 0.18),
            meanline_visible=True,
            points=False,
            box_visible=True,
            showlegend=(feat == features[0]),
            hovertemplate=f"<b>Non-Stress — {fname}</b>: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(**base_layout(
        violinmode="overlay",
        xaxis=dict(**PLOTLY_TEMPLATE["xaxis"], tickfont=dict(color=COLORS["muted"], size=12)),
        yaxis=dict(**PLOTLY_TEMPLATE["yaxis"], title="Feature Value"),
        legend=dict(**PLOTLY_TEMPLATE["legend"], orientation="h", y=1.04, x=0.5, xanchor="center"),
        margin=dict(t=40, r=20, b=50, l=65),
    ))
    return fig


def fig_window_heatmap_filtered(subject="all"):
    conds = CONDITIONS
    if subject != "all":
        mat = [[SUBJ_WINDOWS.get(subject, {}).get(c, 0) for c in conds]]
        yvals = [subject]
    else:
        mat = [[SUBJ_WINDOWS.get(s, {}).get(c, 0) for c in conds] for s in SUBJECTS]
        yvals = SUBJECTS
    fig = go.Figure(go.Heatmap(
        z=mat,
        x=[c.capitalize() for c in conds],
        y=yvals,
        colorscale=[
            [0,   "#EDE9E0"],
            [0.4, "#C8D8C0"],
            [0.7, "#D4956A"],
            [1,   "#8B1A1A"]
        ],
        text=mat,
        texttemplate="%{text}",
        textfont=dict(size=9, color=COLORS["text"]),
        hovertemplate="<b>%{y}</b> — %{x}<br>Windows: <b>%{z:,}</b><extra></extra>",
        colorbar=dict(thickness=12, tickfont=dict(color=COLORS["muted"], size=10)),
    ))
    fig.update_layout(**base_layout(
        xaxis=dict(**PLOTLY_TEMPLATE["xaxis"], tickfont=dict(color=COLORS["muted"], size=12)),
        yaxis=dict(**PLOTLY_TEMPLATE["yaxis"], tickfont=dict(color=COLORS["muted"], size=11)),
        margin=dict(t=20, r=80, b=50, l=55),
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="WESAD · Stress Detection",
    suppress_callback_exceptions=True,
)

app.index_string = f"""
<!DOCTYPE html>
<html>
<head>
{{%metas%}}
<title>{{%title%}}</title>
{{%favicon%}}
{{%css%}}
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=DM+Mono:wght@400;500&family=DM+Serif+Display&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  html, body {{ margin:0; padding:0; }}
  body {{ background:{COLORS["bg"]}; color:{COLORS["text"]}; font-family:'DM Sans',sans-serif; }}
  ::-webkit-scrollbar {{ width:6px; height:6px; }}
  ::-webkit-scrollbar-track {{ background:{COLORS["surface"]}; }}
  ::-webkit-scrollbar-thumb {{ background:{COLORS["border"]}; border-radius:4px; }}

  .filter-btn {{
    background:{COLORS["card"]};
    border:1px solid {COLORS["border"]};
    color:{COLORS["muted"]};
    font-family:'DM Sans',sans-serif;
    font-size:11px;
    font-weight:500;
    padding:5px 13px;
    border-radius:4px;
    cursor:pointer;
    transition:all .15s ease;
    letter-spacing:.3px;
    outline:none;
  }}
  .filter-btn:hover {{
    border-color:{COLORS["accent"]};
    color:{COLORS["accent"]};
    background:{hex_to_rgba(COLORS["accent"], 0.04)};
  }}
  .filter-btn.active {{
    border-color:{COLORS["accent"]};
    color:{COLORS["card"]};
    background:{COLORS["accent"]};
  }}

  .nav-tab {{
    background:transparent;
    border:none;
    border-bottom:2px solid transparent;
    color:{COLORS["muted"]};
    font-family:'DM Sans',sans-serif;
    font-size:11px;
    font-weight:600;
    padding:13px 20px;
    cursor:pointer;
    text-transform:uppercase;
    letter-spacing:1.5px;
    transition:all .15s ease;
    white-space:nowrap;
    outline:none;
  }}
  .nav-tab:hover {{ color:{COLORS["text"]}; }}
  .nav-tab.active {{ color:{COLORS["accent2"]}; border-bottom-color:{COLORS["accent2"]}; }}

  .insight-card {{
    background:{COLORS["card"]};
    border:1px solid {COLORS["border"]};
    border-radius:8px;
    padding:16px 18px;
    flex:1;
    min-width:200px;
    transition: box-shadow .2s;
  }}
  .insight-card:hover {{ box-shadow: 0 4px 16px rgba(28,25,23,0.08); }}

  .pipeline-step {{
    background:{COLORS["card"]};
    border:1px solid {COLORS["border"]};
    border-radius:6px;
    padding:10px 16px;
    font-size:12px;
    font-weight:500;
    color:{COLORS["text"]};
    white-space:nowrap;
    position:relative;
  }}
  .pipeline-arrow {{
    color:{COLORS["muted"]};
    font-size:16px;
    line-height:1;
    align-self:center;
    flex-shrink:0;
  }}

  .thumb-card {{
    background:{COLORS["card"]};
    border:1px solid {COLORS["border"]};
    border-radius:8px;
    padding:14px 16px;
    flex:1;
    min-width:120px;
    cursor:pointer;
    transition:all .18s ease;
    text-align:center;
  }}
  .thumb-card:hover {{
    border-color:{COLORS["accent"]};
    box-shadow: 0 4px 14px rgba(28,25,23,0.08);
    transform: translateY(-2px);
  }}

  .dash-graph .modebar {{ opacity:0; transition:opacity .2s; }}
  .dash-graph:hover .modebar {{ opacity:1; }}
  .modebar {{ background:transparent !important; }}
  .modebar-btn {{ color:#475569 !important; }}

  @keyframes fadeUp {{
    from {{ opacity:0; transform:translateY(10px); }}
    to {{ opacity:1; transform:translateY(0); }}
  }}
  .fade-in {{ animation:fadeUp .25s ease forwards; }}

  .Select-control {{
    background:{COLORS["card"]} !important;
    border-color:{COLORS["border"]} !important;
    border-radius:6px !important;
  }}
  .Select-menu-outer {{
    background:{COLORS["card"]} !important;
    border-color:{COLORS["border"]} !important;
  }}
</style>
</head>
<body>
{{%app_entry%}}
<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>
"""

# ──────────────────────────────────────────────────────────────────────────────
# LANDING PAGE
# ──────────────────────────────────────────────────────────────────────────────
def landing_insight_cards():
    insights = [
        {
            "icon": "⚖️",
            "title": "Class imbalance",
            "stat": f"{baseline_pct:.0f}% vs {amusement_pct:.0f}%",
            "stat_color": COLORS["stress"],
            "body": "Baseline dominates the dataset while amusement is much smaller. Any classifier should account for imbalance via SMOTE or class weighting.",
            "link": "s1",
            "link_label": "→ View Class Distribution",
        },
        {
            "icon": "📡",
            "title": "EDA leads all signals",
            "stat": top_feature_stat,  # ← Fix 1 only change
            "stat_color": COLORS["accent"],
            "body": "Top-ranked feature separation comes from physiological signal summaries, with EDA typically among the strongest contributors.",
            "link": "s6",
            "link_label": "→ View Feature Ranking",
        },
        {
            "icon": "👤",
            "title": "High inter-subject variance",
            "stat": f"{baseline_spread_ratio:.1f}× spread" if baseline_spread_ratio else "Subject spread",
            "stat_color": COLORS["yellow"],
            "body": "Baseline physiology differs substantially across participants. Personalized thresholds or subject-aware validation can help.",
            "link": "s4",
            "link_label": "→ View Subject Variability",
        },
        {
            "icon": "🔗",
            "title": "Feature redundancy",
            "stat": "Correlation clusters",
            "stat_color": COLORS["purple"],
            "body": "Several signal families move together. Correlation review helps identify redundant variables before modeling.",
            "link": "s3",
            "link_label": "→ View Correlation Matrix",
        },
    ]

    return html.Div([
        html.Div([
            html.Div("Key findings", style={
                "fontSize": "10px", "fontWeight": "600", "letterSpacing": "2px",
                "textTransform": "uppercase", "color": COLORS["muted"], "marginBottom": "12px"
            }),
            html.Div(
                [
                    html.Div([
                        html.Div([
                            html.Span(ins["icon"], style={"fontSize": "20px", "marginRight": "10px"}),
                            html.Span(ins["title"], style={
                                "fontWeight": "600", "fontSize": "13px", "color": COLORS["text"]
                            }),
                        ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
                        html.Div(ins["stat"], style={
                            "fontFamily": "'DM Mono', monospace",
                            "fontSize": "18px",
                            "fontWeight": "500",
                            "color": ins["stat_color"],
                            "marginBottom": "6px",
                            "letterSpacing": "-0.3px"
                        }),
                        html.P(ins["body"], style={
                            "fontSize": "12px", "color": COLORS["muted"],
                            "lineHeight": "1.6", "margin": "0 0 10px"
                        }),
                        html.Button(ins["link_label"], id=f"insight-link-{ins['link']}",
                                    n_clicks=0, style={
                            "background": "none", "border": "none", "padding": "0",
                            "color": COLORS["accent"], "fontSize": "11px", "fontWeight": "600",
                            "cursor": "pointer", "letterSpacing": "0.3px"
                        }),
                    ], className="insight-card")
                    for ins in insights
                ],
                style={"display": "flex", "gap": "14px", "flexWrap": "wrap"}
            ),
        ], style={"marginBottom": "32px"}),
    ])


def landing_pipeline():
    steps = [
        ("📦", "Raw signals", "Wearable recordings"),
        ("⏱", "Windowing", "5-second windows"),
        ("🔢", "Feature extraction", f"{NUM_FEATURES} features"),
        ("📊", "EDA visualization", "6-panel dashboard"),
        ("🤖", "Stress classifier", "Binary + 4-class"),
    ]
    items = []
    for i, (icon, label, sub) in enumerate(steps):
        items.append(html.Div([
            html.Div(icon, style={"fontSize": "16px", "marginBottom": "4px"}),
            html.Div(label, style={"fontWeight": "600", "fontSize": "12px", "color": COLORS["text"]}),
            html.Div(sub, style={"fontSize": "10px", "color": COLORS["muted"], "marginTop": "2px"}),
        ], className="pipeline-step"))
        if i < len(steps) - 1:
            items.append(html.Div("→", className="pipeline-arrow"))
    return html.Div([
        html.Div("Analysis pipeline", style={
            "fontSize": "10px", "fontWeight": "600", "letterSpacing": "2px",
            "textTransform": "uppercase", "color": COLORS["muted"], "marginBottom": "12px"
        }),
        html.Div(items, style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"}),
    ], style={"marginBottom": "32px"})


def landing_thumbnails():
    tabs_info = [
        ("s1", "📊", "Class Distribution", "Window counts & imbalance"),
        ("s2", "🫀", "Signal Separation", "Violin plots & radar"),
        ("s3", "🧠", "Feature Correlation", "22×22 heatmap"),
        ("s4", "👥", "Subject Variability", "Per-subject EDA"),
        ("s5", "⚖️", "Binary Comparison", "Stress vs Non-Stress"),
        ("s6", "🏆", "Feature Ranking", "Discriminability ranking"),
    ]
    return html.Div([
        html.Div("Visualization panels", style={
            "fontSize": "10px", "fontWeight": "600", "letterSpacing": "2px",
            "textTransform": "uppercase", "color": COLORS["muted"], "marginBottom": "12px"
        }),
        html.Div(
            [
                html.Div([
                    html.Div(icon, style={"fontSize": "22px", "marginBottom": "8px"}),
                    html.Div(label, style={"fontWeight": "600", "fontSize": "12px", "color": COLORS["text"], "marginBottom": "4px"}),
                    html.Div(sub, style={"fontSize": "10px", "color": COLORS["muted"]}),
                ], className="thumb-card", id=f"thumb-{sid}", n_clicks=0)
                for sid, icon, label, sub in tabs_info
            ],
            style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}
        ),
    ])


landing_page = html.Div([
    html.Div([
        dcc.Markdown(
            f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 60"
                 preserveAspectRatio="xMidYMid meet"
                 style="width:100%;height:60px;display:block;position:absolute;bottom:0;left:0">
              <path d="M0,30 L30,30 L34,30 L37,8 L40,52 L43,30 L55,30
                       L85,30 L89,30 L92,12 L95,48 L98,30 L110,30
                       L140,30 L144,30 L147,9 L150,51 L153,30 L165,30
                       L195,30 L199,30 L202,10 L205,50 L208,30 L220,30
                       L250,30 L254,30 L257,8 L260,52 L263,30 L275,30
                       L305,30 L309,30 L312,11 L315,49 L318,30 L330,30
                       L360,30 L364,30 L367,9 L370,51 L373,30 L385,30
                       L415,30 L419,30 L422,10 L425,50 L428,30 L440,30
                       L470,30 L474,30 L477,8 L480,52 L483,30 L495,30
                       L525,30 L529,30 L532,12 L535,48 L538,30 L550,30
                       L580,30 L584,30 L587,9 L590,51 L593,30 L605,30
                       L635,30 L639,30 L642,10 L645,50 L648,30 L660,30
                       L690,30 L694,30 L697,8 L700,52 L703,30 L715,30
                       L745,30 L749,30 L752,11 L755,49 L758,30 L800,30"
                    stroke="{hex_to_rgba(COLORS["stress"], 0.12)}"
                    stroke-width="1.5" fill="none"
                    stroke-linecap="round" stroke-linejoin="round"/>
            </svg>''',
            dangerously_allow_html=True,
            style={"position": "absolute", "bottom": "0", "left": "0", "right": "0", "height": "60px", "pointerEvents": "none", "margin": "0"}
        ),

        html.Div([
            html.Div([
                html.Div([
                    html.Span("Dashboard", style={
                        "background": COLORS["surface"],
                        "border": f"1px solid {COLORS['border']}",
                        "color": COLORS["muted"],
                        "fontSize": "10px",
                        "fontWeight": "600",
                        "padding": "3px 10px",
                        "borderRadius": "3px",
                        "letterSpacing": "1.5px",
                        "textTransform": "uppercase",
                        "marginRight": "10px"
                    }),
                    html.Span("WESAD Dataset", style={
                        "fontSize": "11px", "color": COLORS["muted"]
                    }),
                ], style={"marginBottom": "16px", "display": "flex", "alignItems": "center"}),

                html.H1([
                    html.Span("Wearable Stress", style={
                        "fontFamily": "'DM Serif Display', serif",
                        "color": COLORS["text"],
                        "fontStyle": "italic",
                        "fontWeight": "400",
                        "letterSpacing": "-0.5px"
                    }),
                    html.Br(),
                    html.Span("Detection", style={
                        "fontFamily": "'DM Serif Display', serif",
                        "color": COLORS["accent2"],
                        "fontStyle": "normal",
                        "fontWeight": "400",
                    }),
                    html.Span(" Dashboard", style={
                        "fontFamily": "'DM Sans', sans-serif",
                        "color": COLORS["muted"],
                        "fontSize": "0.55em",
                        "fontWeight": "300",
                        "verticalAlign": "middle",
                        "marginLeft": "12px"
                    }),
                ], style={
                    "fontSize": "clamp(32px,4vw,52px)",
                    "lineHeight": "1.1",
                    "margin": "0 0 20px",
                }),

                html.Div([
                    html.Span("Question — ", style={
                        "fontWeight": "600", "color": COLORS["accent2"], "fontSize": "13px"
                    }),
                    html.Span(
                        "Can physiological wearable signals reliably distinguish stress from non-stress for real-time detection?",
                        style={"color": COLORS["muted"], "fontSize": "13px", "lineHeight": "1.6"}
                    ),
                ], style={
                    "borderLeft": f"3px solid {COLORS['accent2']}",
                    "paddingLeft": "14px",
                    "maxWidth": "680px",
                    "marginBottom": "28px"
                }),

            ], style={"flex": "1", "minWidth": "280px"}),

            html.Div([
                html.Div([
                    stat_chip(str(NUM_SUBJECTS), "Subjects", COLORS["accent"], "👥"),
                    stat_chip(f"{NUM_WINDOWS:,}", "Windows", COLORS["accent3"], "📦"),
                    stat_chip(str(NUM_FEATURES), "Features", COLORS["purple"], "🔢"),
                    stat_chip(str(NUM_CONDITIONS), "Conditions", COLORS["yellow"], "🏷"),
                    stat_chip("5s", "Window size", COLORS["muted"], "⏱"),
                    stat_chip("Raw CSV", "Source", COLORS["muted"], "📄"),
                ], style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(3, 1fr)",
                    "gap": "8px",
                    "minWidth": "280px"
                }),
            ], style={"flexShrink": "0"}),

        ], style={
            "display": "flex", "gap": "40px", "alignItems": "flex-start",
            "flexWrap": "wrap", "padding": "32px 48px 60px",
            "position": "relative", "zIndex": "1"
        }),

    ], style={
        "background": COLORS["bg"],
        "borderBottom": f"1px solid {COLORS['border']}",
        "position": "relative",
        "overflow": "hidden",
    }),

    html.Div([
        landing_insight_cards(),
        landing_pipeline(),
        landing_thumbnails(),
    ], style={"padding": "32px 48px 56px"}),

], id="landing-page", className="fade-in")


# ──────────────────────────────────────────────────────────────────────────────
# LAYOUT PIECES
# ──────────────────────────────────────────────────────────────────────────────
TABS = [
    ("s1", "Class Distribution"),
    ("s2", "Signal Separation"),
    ("s3", "Feature Correlation"),
    ("s4", "Subject Variability"),
    ("s5", "Binary Comparison"),
    ("s6", "Feature Ranking"),
]

compact_header = html.Div([
    html.Div([
        html.Div([
            html.Button("← Dashboard", id="back-to-home", n_clicks=0, style={
                "background": "none", "border": "none", "cursor": "pointer",
                "color": COLORS["muted"], "fontSize": "12px", "fontWeight": "500",
                "padding": "0", "letterSpacing": "0.3px",
            }),
            html.Span("  /  ", style={"color": COLORS["border"], "margin": "0 4px"}),
            html.Span("WESAD Stress Detection", style={
                "fontFamily": "'DM Serif Display', serif",
                "fontSize": "16px", "color": COLORS["text"], "fontStyle": "italic"
            }),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Span("RQ: ", style={"fontWeight": "600", "color": COLORS["accent2"], "fontSize": "11px"}),
            html.Span(
                "Can wearable signals distinguish stress for real-time detection?",
                style={"color": COLORS["muted"], "fontSize": "11px"}
            ),
        ], style={
            "background": COLORS["surface"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "4px",
            "padding": "6px 12px",
            "display": "none"
        }),
    ], style={
        "display": "flex", "justifyContent": "space-between",
        "alignItems": "center", "padding": "12px 48px",
        "borderBottom": f"1px solid {COLORS['border']}"
    }),
])

nav = html.Div(
    html.Div([
        html.Button(
            label,
            id=f"nav-{sid}",
            n_clicks=0,
            className="nav-tab active" if i == 0 else "nav-tab"
        )
        for i, (sid, label) in enumerate(TABS)
    ], style={"display": "flex", "overflowX": "auto"}),
    style={
        "background": COLORS["bg"],
        "borderBottom": f"1px solid {COLORS['border']}",
        "position": "sticky",
        "top": "0",
        "zIndex": "100",
        "paddingLeft": "48px"
    }
)


def section(sid, content):
    return html.Div(content, id=f"sec-{sid}", className="fade-in", style={"padding": "32px 48px 56px"})


def section_header(emoji, viz_num, title, description):
    return html.Div([
        html.Div(f"{emoji} Viz {viz_num} — {title}", style={
            "fontSize": "10px", "fontWeight": "600", "letterSpacing": "2.5px",
            "textTransform": "uppercase", "color": COLORS["accent"], "marginBottom": "4px"
        }),
        html.P(description, style={
            "fontSize": "13px", "color": COLORS["muted"], "marginBottom": "20px", "lineHeight": "1.6"
        }),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# SECTIONS
# ──────────────────────────────────────────────────────────────────────────────
sec_s1 = section("s1", [
    section_header("📊", 1, "Window Distribution & Class Imbalance",
                   "How are the 4 conditions distributed across all extracted windows? This directly shapes which ML strategy is viable."),
    html.Div([
        html.Div([
            html.Label("Subject", style={"fontWeight": "600", "marginBottom": "6px", "color": COLORS["text"], "fontSize": "12px"}),
            dcc.Dropdown(
                id="s1-subject",
                options=[{"label": "All Subjects", "value": "all"}] + [{"label": s, "value": s} for s in SUBJECTS],
                value="all",
                clearable=False
            ),
        ], style={"minWidth": "240px", "maxWidth": "320px"}),
    ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "alignItems": "end", "marginBottom": "20px"}),
    html.Div([
        html.Div([
            card_header("Window Count by Condition", "Filtered by subject selection", "Class Imbalance", COLORS["stress"]),
            html.Div(
                dcc.Graph(id="chart-class", figure=fig_class_distribution_filtered(), config={"displayModeBar": False}, style={"height": "340px"}),
                style={"padding": "8px 4px 4px"}
            ),
            insight_box("Dashed line = equal-share baseline. Large differences in class size suggest the need for class balancing before model training."),
        ], style={**CARD_STYLE, "flex": "2"}),
        html.Div([
            card_header("Condition Proportion", "Per-subject or global view", "Donut", COLORS["yellow"]),
            html.Div(
                dcc.Graph(id="chart-donut", figure=fig_donut_filtered(), config={"displayModeBar": False}, style={"height": "340px"}),
                style={"padding": "8px 4px 4px"}
            ),
            insight_box("Select a subject above to inspect individual class balance.", COLORS["yellow"]),
        ], style={**CARD_STYLE, "flex": "1"}),
    ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
])

sec_s2 = section("s2", [
    section_header("🫀", 2, "Physiological Signal Separation",
                   "Explore each physiological channel. Cohen's d is shown on each violin to quantify stress-vs-non-stress separation."),
    make_sig_buttons("sig2"),
    dcc.Store(id="sig2-current", data="eda_mean"),
    html.Div([
        html.Div([
            card_header("Violin + Box by Condition", "Effect size (Cohen's d) annotated", "Signal Separator", COLORS["accent"]),
            html.Div(
                dcc.Graph(id="chart-violin", figure=fig_signal_box("eda_mean"), config={"displayModeBar": False}, style={"height": "380px"}),
                style={"padding": "8px 4px 4px"}
            ),
            html.Div(id="sig2-insight", children=insight_box(build_signal_insight("eda_mean"))),
        ], style={**CARD_STYLE, "flex": "1"}),
        html.Div([
            card_header("Multi-Signal Radar", "Condition fingerprint — normalized medians", "Multi-Signal", COLORS["accent3"]),
            html.Div(
                dcc.Graph(id="chart-radar", figure=fig_radar(), config={"displayModeBar": False}, style={"height": "380px"}),
                style={"padding": "8px 4px 4px"}
            ),
            insight_box("Radar values are normalized within each feature, so the shape shows relative condition profiles rather than raw units.", COLORS["accent3"]),
        ], style={**CARD_STYLE, "flex": "1"}),
    ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
])

sec_s3 = section("s3", [
    section_header("🧠", 3, "Feature Correlation Heatmap (22 × 22)",
                   "Which features are redundant? Highly correlated pairs often carry overlapping information and can be pruned before modeling."),
    html.Div([
        card_header("Pearson Correlation Matrix", "Navy = negative · Warm = positive · Annotated clusters", "PCA / Selection", COLORS["yellow"]),
        html.Div(
            dcc.Graph(figure=fig_correlation(), config={"displayModeBar": False}, style={"height": "580px"}),
            style={"padding": "8px 4px 4px"}
        ),
        insight_box("Correlation clusters suggest redundancy within the same signal family. This helps guide PCA or manual feature selection.", COLORS["yellow"]),
    ], style=CARD_STYLE),
])

sec_s4 = section("s4", [
    section_header("👥", 4, "Per-Subject EDA Variability",
                   "Inter-subject variance is a key challenge in wearable stress detection. Compare absolute EDA and stress-to-baseline ratios."),
    html.Div([
        html.Div([
            html.Label("Subject", style={"fontWeight": "600", "marginBottom": "6px", "color": COLORS["text"], "fontSize": "12px"}),
            dcc.Dropdown(
                id="s4-subject",
                options=[{"label": "All Subjects", "value": "all"}] + [{"label": s, "value": s} for s in SUBJECTS],
                value="all",
                clearable=False
            ),
        ], style={"minWidth": "240px", "maxWidth": "320px"}),
    ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "alignItems": "end", "marginBottom": "16px"}),
    html.Div([
        html.Span("View:", style={"fontSize": "11px", "color": COLORS["muted"], "textTransform": "uppercase",
                                  "letterSpacing": "1px", "alignSelf": "center", "fontWeight": "500"}),
        html.Button("Grouped by Condition", id="view-grouped", className="filter-btn active", n_clicks=0),
        html.Button("Stress/Baseline Ratio", id="view-ratio", className="filter-btn", n_clicks=0),
    ], style={"display": "flex", "gap": "8px", "alignItems": "center", "marginBottom": "16px"}),
    dcc.Store(id="subj-view", data="grouped"),
    html.Div([
        html.Div([
            card_header("Subject Variability", "Switch views to see absolute EDA or stress/baseline ratio", "Inter-Subject Gap", COLORS["stress"]),
            html.Div(
                dcc.Graph(id="chart-subj", figure=fig_subject_variability_filtered("all", "grouped"), config={"displayModeBar": False}, style={"height": "380px"}),
                style={"padding": "8px 4px 4px"}
            ),
            insight_box("Ratio view helps spot participants with especially strong stress responses relative to their own baseline.", COLORS["stress"]),
        ], style={**CARD_STYLE, "flex": "2"}),
        html.Div([
            card_header("Window Distribution Heatmap", "Data coverage per subject × condition", "Coverage", COLORS["purple"]),
            html.Div(
                dcc.Graph(id="chart-heatmap", figure=fig_window_heatmap_filtered(), config={"displayModeBar": False}, style={"height": "380px"}),
                style={"padding": "8px 4px 4px"}
            ),
            insight_box("Use this panel to spot class coverage issues for individual subjects before train/test splitting.", COLORS["purple"]),
        ], style={**CARD_STYLE, "flex": "1"}),
    ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
])

sec_s5 = section("s5", [
    section_header("⚖️", 5, "Stress vs Non-Stress Comparison",
                   "Binary separability: which features most cleanly divide Stress from the pooled Non-Stress group?"),
    html.Div([
        card_header("Split Violin — Stress vs Non-Stress", "Right = Stress · Left = Non-Stress · Overlap = difficulty of separation", "Binary", COLORS["accent"]),
        html.Div(
            dcc.Graph(figure=fig_binary_comparison(), config={"displayModeBar": False}, style={"height": "420px"}),
            style={"padding": "8px 4px 4px"}
        ),
        insight_box("Features with wider left-right separation are better candidates for binary stress detection."),
    ], style=CARD_STYLE),
])

sec_s6 = section("s6", [
    section_header("🏆", 6, "Feature Discriminability Ranking",
                   "Higher scores indicate stronger condition-level separation and better potential as classifier inputs."),
    html.Div([
        card_header("Normalized Feature Separation", "High = better class separation across conditions", "Feature Selection", COLORS["accent"]),
        html.Div(
            dcc.Graph(figure=fig_feature_ranking(), config={"displayModeBar": False}, style={"height": "520px"}),
            style={"padding": "8px 4px 4px"}
        ),
        insight_box("Use this chart to prioritize strong features and remove low-value ones before model training.", COLORS["accent"]),
    ], style=CARD_STYLE),
])


# ──────────────────────────────────────────────────────────────────────────────
# APP LAYOUT
# ──────────────────────────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Store(id="active-tab", data="s1"),
    dcc.Store(id="show-landing", data=True),
    html.Div(landing_page, id="landing-wrap", style={"display": "block"}),
    html.Div([
        compact_header,
        nav,
        html.Div([
            html.Div(sec_s1, id="wrap-s1", style={"display": "block"}),
            html.Div(sec_s2, id="wrap-s2", style={"display": "none"}),
            html.Div(sec_s3, id="wrap-s3", style={"display": "none"}),
            html.Div(sec_s4, id="wrap-s4", style={"display": "none"}),
            html.Div(sec_s5, id="wrap-s5", style={"display": "none"}),
            html.Div(sec_s6, id="wrap-s6", style={"display": "none"}),
        ], id="main-content"),
    ], id="viz-wrap", style={"display": "none"}),
], style={"minHeight": "100vh", "background": COLORS["bg"]})


# ──────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("show-landing", "data"),
    Output("active-tab", "data"),
    [Input(f"thumb-{sid}", "n_clicks") for sid, _, _, _ in [
        ("s1","📊","Class Distribution",""),
        ("s2","🫀","Signal Separation",""),
        ("s3","🧠","Feature Correlation",""),
        ("s4","👥","Subject Variability",""),
        ("s5","⚖️","Binary Comparison",""),
        ("s6","🏆","Feature Ranking",""),
    ]],
    Input("insight-link-s1", "n_clicks"),
    Input("insight-link-s3", "n_clicks"),
    Input("insight-link-s4", "n_clicks"),
    Input("insight-link-s6", "n_clicks"),
    State("show-landing", "data"),
    State("active-tab", "data"),
    prevent_initial_call=True,
)
def navigate_from_landing(*args):
    trigger_id = ctx.triggered_id
    if not trigger_id:
        return True, "s1"
    if trigger_id.startswith("thumb-"):
        return False, trigger_id.replace("thumb-", "")
    if trigger_id.startswith("insight-link-"):
        return False, trigger_id.replace("insight-link-", "")
    return False, "s1"


@app.callback(
    Output("show-landing", "data", allow_duplicate=True),
    Input("back-to-home", "n_clicks"),
    prevent_initial_call=True,
)
def go_home(n):
    if n:
        return True
    return dash.no_update


@app.callback(
    Output("landing-wrap", "style"),
    Output("viz-wrap", "style"),
    Input("show-landing", "data"),
)
def toggle_landing(show):
    if show:
        return {"display": "block"}, {"display": "none"}
    return {"display": "none"}, {"display": "block"}


@app.callback(
    [Output(f"wrap-{sid}", "style") for sid, _ in TABS] +
    [Output(f"nav-{sid}", "className") for sid, _ in TABS],
    Input("active-tab", "data"),
)
def switch_section(active):
    wrap_styles = [
        {"display": "block"} if sid == active else {"display": "none"}
        for sid, _ in TABS
    ]
    nav_classes = [
        "nav-tab active" if sid == active else "nav-tab"
        for sid, _ in TABS
    ]
    return (*wrap_styles, *nav_classes)


@app.callback(
    Output("active-tab", "data", allow_duplicate=True),
    [Input(f"nav-{sid}", "n_clicks") for sid, _ in TABS],
    State("active-tab", "data"),
    prevent_initial_call=True,
)
def update_tab(*args):
    trigger_id = ctx.triggered_id
    if not trigger_id:
        return args[-1]
    return trigger_id.replace("nav-", "")


@app.callback(
    Output("chart-class", "figure"),
    Output("chart-donut", "figure"),
    Input("s1-subject", "value"),
    prevent_initial_call=False,
)
def update_section1(subject):
    subject = subject or "all"
    return fig_class_distribution_filtered(subject), fig_donut_filtered(subject)


@app.callback(
    Output("sig2-current", "data"),
    *[Output(f"sig2-{sig}", "className") for sig in SIG_IDS],
    *[Input(f"sig2-{sig}", "n_clicks") for sig in SIG_IDS],
    State("sig2-current", "data"),
    prevent_initial_call=False,
)
def sync_signal_state(*args):
    current_signal = args[-1]
    trigger_id = ctx.triggered_id
    chosen = current_signal or "eda_mean"
    if isinstance(trigger_id, str) and trigger_id.startswith("sig2-"):
        chosen = trigger_id.replace("sig2-", "")
    elif trigger_id is None:
        chosen = "eda_mean"
    classes = ["filter-btn active" if sig == chosen else "filter-btn" for sig in SIG_IDS]
    return (chosen, *classes)


@app.callback(
    Output("chart-violin", "figure"),
    Output("sig2-insight", "children"),
    Input("sig2-current", "data"),
    prevent_initial_call=False,
)
def update_violin(chosen_signal):
    chosen_signal = chosen_signal or "eda_mean"
    return fig_signal_box(chosen_signal), insight_box(build_signal_insight(chosen_signal))


@app.callback(
    Output("subj-view", "data"),
    Output("view-grouped", "className"),
    Output("view-ratio", "className"),
    Input("view-grouped", "n_clicks"),
    Input("view-ratio", "n_clicks"),
    State("subj-view", "data"),
    prevent_initial_call=False,
)
def sync_subject_view(_n_grouped, _n_ratio, current_view):
    trigger_id = ctx.triggered_id
    view = current_view or "grouped"
    if trigger_id in (None, "view-grouped"):
        view = "grouped"
    elif trigger_id == "view-ratio":
        view = "ratio"
    return view, "filter-btn active" if view == "grouped" else "filter-btn", "filter-btn active" if view == "ratio" else "filter-btn"


@app.callback(
    Output("chart-subj", "figure"),
    Output("chart-heatmap", "figure"),
    Input("s4-subject", "value"),
    Input("subj-view", "data"),
    prevent_initial_call=False,
)
def update_section4(subject, view):
    subject = subject or "all"
    view = view or "grouped"
    return fig_subject_variability_filtered(subject, view), fig_window_heatmap_filtered(subject)


# ──────────────────────────────────────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  WESAD Stress Detection Dashboard")
    print("  → http://127.0.0.1:8050")
    print("═" * 60 + "\n")
    app.run(debug=False, host="127.0.0.1", port=8050)
