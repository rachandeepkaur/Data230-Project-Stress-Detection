import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="WESAD ML Results Dashboard",
    layout="wide",
)

DARK_BROWN = "#6B3A2A"
LIGHT_CORAL = "#D4715E"
RED = "#D4534B"
BROWN = "#A0522D"
LIGHT_RED = "#E8CCBA"
LIGHT_BROWN = "#C9A882"
CREAM = "#FFF8F2"
TEXT = "#3B1F0B"

st.markdown("""
<style>
    .main { background-color: #FFFFFF; }
    .block-container { padding-top: 1.5rem; }

    .title-banner {
        background: linear-gradient(135deg, #A0522D 0%, #D4534B 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.8rem;
    }
    .title-banner h1 { color: #FFFFFF; font-size: 2.1rem; margin: 0; }
    .title-banner p { color: #FFF8F2; font-size: 1.05rem; margin: 0.5rem 0 0 0; opacity: 0.92; }

    .metric-card {
        background: #FFFFFF;
        border: 2px solid #C9A882;
        border-radius: 12px;
        padding: 1.3rem 1rem;
        text-align: center;
        box-shadow: 0 3px 12px rgba(160,82,45,0.08);
        transition: transform 0.15s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(160,82,45,0.13); }
    .metric-card .label {
        color: #A0522D; font-size: 0.82rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.6px;
    }
    .metric-card .value { font-size: 2.1rem; font-weight: 700; margin-top: 0.3rem; }
    .metric-card .value.red { color: #D4534B; }
    .metric-card .value.brown { color: #A0522D; }

    .insight-box {
        background: #FFF8F2; border-left: 4px solid #D4534B;
        padding: 0.9rem 1.1rem; border-radius: 0 10px 10px 0;
        color: #3B1F0B; font-size: 0.93rem; line-height: 1.55;
        margin-top: 0.5rem; margin-bottom: 1.2rem;
    }

    .key-findings {
        background: #FFF8F2; border: 2px solid #D4534B;
        border-radius: 12px; padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    .key-findings h3 { color: #D4534B; margin: 0 0 0.6rem 0; font-size: 1.15rem; }
    .key-findings ul { margin: 0; padding-left: 1.3rem; color: #3B1F0B; line-height: 1.8; }
    .key-findings li { margin-bottom: 0.2rem; }
    .key-findings b { color: #A0522D; }

    .divider { border: none; border-top: 1.5px solid #C9A882; margin: 1.8rem 0; }

    .sidebar-card {
        background: #FFF8F2; padding: 0.9rem; border-radius: 10px;
        font-size: 0.88rem; color: #3B1F0B; margin-bottom: 0.8rem;
        border: 1px solid #E8CCBA;
    }

    .summary-table {
        width: 100%; border-collapse: collapse; margin-top: 0.5rem;
        font-size: 0.95rem; color: #3B1F0B;
    }
    .summary-table th {
        background: #A0522D; color: #FFFFFF; padding: 0.7rem 1rem;
        text-align: left; font-weight: 600;
    }
    .summary-table td {
        padding: 0.6rem 1rem; border-bottom: 1px solid #C9A882;
        transition: background 0.15s;
    }
    .summary-table tr:nth-child(even) td { background: #FFF8F2; }
    .summary-table tr:hover td { background: #F0E0D4; }
    .summary-table .highlight { font-weight: 700; color: #D4534B; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_approach1():
    df = pd.read_csv("Approach1_ML/approach1_correct_loso_results.csv")
    return df.sort_values("subject").reset_index(drop=True)

@st.cache_data
def load_approach2():
    df = pd.read_csv("Approach2_ML/output_v4/cnn_final_comparison_ready.csv")
    return df.sort_values("subject").reset_index(drop=True)

@st.cache_data
def load_cnn_report():
    return pd.read_csv(
        "Approach2_ML/output_v4/cnn_report_locked_thresh_0.4420.csv", index_col=0,
    )


a1 = load_approach1()
a2 = load_approach2()
cnn_report = load_cnn_report()
subjects = a1["subject"].tolist()

rf_acc = a1["accuracy"].mean()
cnn_acc = a2["accuracy"].mean()
rf_f1 = a1["f1"].mean()
cnn_f1 = a2["f1"].mean()
rf_auc = a1["roc_auc"].mean()
cnn_auc = a2["auc"].mean()
rf_bal = a1["balanced_accuracy"].mean()
cnn_recall_ns = cnn_report.loc["non-stress", "recall"]
cnn_recall_s = cnn_report.loc["stress", "recall"]
cnn_bal = (cnn_recall_ns + cnn_recall_s) / 2
cnn_wins = int((a2["f1"] > a1["f1"]).sum())
rf_zeros = int((a1["f1"] == 0).sum())

f1_diff = a2["f1"] - a1["f1"]

LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(color=TEXT, size=13),
    margin=dict(t=35, b=50),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        bgcolor="rgba(255,248,242,0.9)", bordercolor=LIGHT_BROWN, borderwidth=1,
        font=dict(size=12),
    ),
)

st.markdown("""
<div class="title-banner">
    <h1>Wearable Stress Detection Using Multi-Modal Physiological Signals &mdash; ML Dashboard</h1>
    <p>Binary stress classification on the WESAD dataset comparing
       Random Forest and Dual-Stream 1D CNN under Leave-One-Subject-Out evaluation</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"<h2 style='color:{BROWN}; margin-bottom:0.5rem;'>Project Overview</h2>", unsafe_allow_html=True)
    st.markdown("""<div class="sidebar-card">
        <b>Course:</b> DATA 230<br>
        <b>Dataset:</b> WESAD (15 subjects)<br>
        <b>Task:</b> Binary stress detection<br>
        <b>Evaluation:</b> Leave-One-Subject-Out<br><br>
        15 subjects wore chest and wrist sensors while performing
        baseline (non-stress) and stress-inducing tasks. The WESAD dataset
        records signals like heart rate, skin conductance, and temperature,
        which the models use to tell stress apart from non-stress.
    </div>""", unsafe_allow_html=True)
    with st.expander("Random Forest"):
        st.markdown("""<div class="sidebar-card">
            <b>Model:</b> Random Forest (300 trees)<br>
            <b>Features:</b> 22 hand-crafted statistical<br>
            <b>Balancing:</b> SMOTE + Undersampling<br>
            <b>Threshold:</b> Per-fold tuned
        </div>""", unsafe_allow_html=True)
    with st.expander("1D CNN"):
        st.markdown("""<div class="sidebar-card">
            <b>Model:</b> Dual-Stream 1D CNN<br>
            <b>Input:</b> Raw windowed signals<br>
            <b>Balancing:</b> Focal Loss + Augmentation<br>
            <b>Threshold:</b> Locked at 0.442
        </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class="key-findings">
    <h3>Key Findings at a Glance</h3>
    <ul>
        <li>The <b>1D CNN</b> achieves a higher average F1 ({cnn_f1:.3f}) than <b>Random Forest</b> ({rf_f1:.3f}), making it the better overall model.</li>
        <li>Random Forest completely fails on <b>{rf_zeros} subjects</b> (F1 = 0), while the CNN keeps all subjects above zero.</li>
        <li>Both models reach similar ROC-AUC (~0.89), but the CNN converts that into more accurate final predictions.</li>
        <li>The CNN wins on <b>{cnn_wins} of 15</b> subjects, with the largest gaps on S2, S5, and S7.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, label, val, cls in [
    (c1, "RF Mean Accuracy", f"{rf_acc:.3f}", "brown"),
    (c2, "CNN Mean Accuracy", f"{cnn_acc:.3f}", "red"),
    (c3, "RF Mean F1", f"{rf_f1:.3f}", "brown"),
    (c4, "CNN Mean F1", f"{cnn_f1:.3f}", "red"),
]:
    with col:
        st.markdown(f"""<div class="metric-card">
            <div class="label">{label}</div>
            <div class="value {cls}">{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

f1_lift = cnn_f1 - rf_f1

def fmt(v):
    return f"{v:.3f}"

def winner(a, b):
    return ' class="highlight"' if a > b else ""

tab1, tab2, tab3, tab4 = st.tabs([
    "Overall Metrics", "Head-to-Head by Subject",
    "F1 Distribution", "Summary Table",
])

with tab1:
    metrics = ["Accuracy", "F1 Score", "ROC-AUC"]
    rf_vals = [rf_acc, rf_f1, rf_auc]
    cnn_vals = [cnn_acc, cnn_f1, cnn_auc]

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=metrics, y=rf_vals, name="Random Forest", marker_color=DARK_BROWN,
        text=[f"{v:.3f}" for v in rf_vals], textposition="outside",
        textfont=dict(size=13, color=DARK_BROWN),
        hovertemplate="%{x}: %{y:.3f}<extra>Random Forest</extra>",
    ))
    fig1.add_trace(go.Bar(
        x=metrics, y=cnn_vals, name="1D CNN", marker_color=LIGHT_CORAL,
        text=[f"{v:.3f}" for v in cnn_vals], textposition="outside",
        textfont=dict(size=13, color=LIGHT_CORAL),
        hovertemplate="%{x}: %{y:.3f}<extra>1D CNN</extra>",
    ))
    fig1.update_layout(**LAYOUT, barmode="group", yaxis_title="Score",
                       yaxis_range=[0, 1.15], height=430)
    fig1.update_xaxes(showgrid=False)
    fig1.update_yaxes(gridcolor="#EDE6DF")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown(
        f'<div class="insight-box"><b>Takeaway:</b> The CNN scores <b>+{f1_lift:.3f}</b> '
        f'higher on F1 than Random Forest. Both models have similar ROC-AUC (~0.89), but '
        f'the CNN is better at correctly identifying stressed vs. non-stressed windows '
        f'when it makes its final yes/no decision.</div>',
        unsafe_allow_html=True,
    )

with tab2:
    merged = pd.DataFrame({
        "subject": subjects,
        "rf_f1": a1["f1"].values,
        "cnn_f1": a2["f1"].values,
    })
    merged["gap"] = merged["cnn_f1"] - merged["rf_f1"]
    merged = merged.sort_values("gap", ascending=True).reset_index(drop=True)

    fig2 = go.Figure()

    for _, row in merged.iterrows():
        fig2.add_trace(go.Scatter(
            x=[row["rf_f1"], row["cnn_f1"]], y=[row["subject"], row["subject"]],
            mode="lines", line=dict(color=LIGHT_BROWN, width=2.5),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig2.add_trace(go.Scatter(
        x=merged["rf_f1"], y=merged["subject"], mode="markers",
        name="Random Forest",
        marker=dict(size=13, color=DARK_BROWN, line=dict(width=1.5, color="white")),
        hovertemplate="<b>%{y}</b><br>RF F1: %{x:.3f}<extra></extra>",
    ))
    fig2.add_trace(go.Scatter(
        x=merged["cnn_f1"], y=merged["subject"], mode="markers",
        name="1D CNN",
        marker=dict(size=13, color=RED, line=dict(width=1.5, color="white")),
        hovertemplate="<b>%{y}</b><br>CNN F1: %{x:.3f}<extra></extra>",
    ))

    fig2.update_layout(
        **LAYOUT, height=520, xaxis_title="F1 Score",
        yaxis_title="", xaxis_range=[-0.05, 1.1],
    )
    fig2.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
    )
    fig2.update_xaxes(gridcolor="#EDE6DF")
    fig2.update_yaxes(showgrid=False)
    st.plotly_chart(fig2, use_container_width=True)

    long_lines = int((merged["gap"].abs() > 0.3).sum())
    st.markdown(
        f'<div class="insight-box"><b>Takeaway:</b> Each row is one subject. The brown dot '
        f'is Random Forest and the red dot is the CNN. Long connecting lines mean the two '
        f'models disagree a lot on that person. <b>{long_lines} subjects</b> have gaps '
        f'larger than 0.30 -- in every case the CNN rescues subjects where RF scored near '
        f'zero. Short lines at the top show subjects both models handle equally well.</div>',
        unsafe_allow_html=True,
    )

with tab3:
    fig3 = go.Figure()
    fig3.add_trace(go.Box(
        y=a1["f1"], name="Random Forest",
        marker_color=DARK_BROWN, line_color=DARK_BROWN,
        boxmean=True, fillcolor="rgba(107,58,42,0.15)",
        hovertemplate="RF F1: %{y:.3f}<extra></extra>",
    ))
    fig3.add_trace(go.Box(
        y=a2["f1"], name="1D CNN",
        marker_color=RED, line_color=RED,
        boxmean=True, fillcolor="rgba(212,83,75,0.15)",
        hovertemplate="CNN F1: %{y:.3f}<extra></extra>",
    ))
    fig3.update_layout(
        **LAYOUT, height=450, yaxis_title="F1 Score", yaxis_range=[-0.05, 1.15],
        showlegend=False,
    )
    fig3.update_xaxes(showgrid=False)
    fig3.update_yaxes(gridcolor="#EDE6DF")
    st.plotly_chart(fig3, use_container_width=True)

    rf_median = a1["f1"].median()
    cnn_median = a2["f1"].median()
    rf_std = a1["f1"].std()
    cnn_std = a2["f1"].std()

    st.markdown(
        f'<div class="insight-box"><b>Takeaway:</b> The box plot shows how consistent '
        f'each model is across all 15 subjects. Random Forest has a median F1 of '
        f'<b>{rf_median:.3f}</b> with high spread (std = {rf_std:.3f}), meaning it works '
        f'great for some people but completely fails for others. The CNN has a median of '
        f'<b>{cnn_median:.3f}</b> with much tighter spread (std = {cnn_std:.3f}), meaning '
        f'it delivers reliable results regardless of who is wearing the sensor.</div>',
        unsafe_allow_html=True,
    )

with tab4:
    st.markdown(f"""
    <table class="summary-table">
        <tr>
            <th>Metric</th>
            <th>Random Forest</th>
            <th>1D CNN</th>
            <th>Difference</th>
        </tr>
        <tr>
            <td>Mean Accuracy</td>
            <td{winner(rf_acc, cnn_acc)}>{fmt(rf_acc)}</td>
            <td{winner(cnn_acc, rf_acc)}>{fmt(cnn_acc)}</td>
            <td>{'+' if cnn_acc-rf_acc>=0 else ''}{cnn_acc-rf_acc:.3f}</td>
        </tr>
        <tr>
            <td>Mean F1 Score</td>
            <td{winner(rf_f1, cnn_f1)}>{fmt(rf_f1)}</td>
            <td{winner(cnn_f1, rf_f1)}>{fmt(cnn_f1)}</td>
            <td>{'+' if cnn_f1-rf_f1>=0 else ''}{cnn_f1-rf_f1:.3f}</td>
        </tr>
        <tr>
            <td>Mean ROC-AUC</td>
            <td{winner(rf_auc, cnn_auc)}>{fmt(rf_auc)}</td>
            <td{winner(cnn_auc, rf_auc)}>{fmt(cnn_auc)}</td>
            <td>{'+' if cnn_auc-rf_auc>=0 else ''}{cnn_auc-rf_auc:.3f}</td>
        </tr>
        <tr>
            <td>Mean Balanced Accuracy</td>
            <td{winner(rf_bal, cnn_bal)}>{fmt(rf_bal)}</td>
            <td{winner(cnn_bal, rf_bal)}>{fmt(cnn_bal)}</td>
            <td>{'+' if cnn_bal-rf_bal>=0 else ''}{cnn_bal-rf_bal:.3f}</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="insight-box"><b>Takeaway:</b> The CNN wins on every overall metric. '
        'Its biggest advantage is that it works for almost all subjects, while Random '
        'Forest completely fails (F1 = 0) on several people. This makes the CNN the '
        'stronger choice if the model needs to work on new, unseen individuals.</div>',
        unsafe_allow_html=True,
    )
