import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="1D CNN Analysis", layout="wide")

DARK_RED = "#A0312A"
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
        background: linear-gradient(135deg, #D4534B 0%, #D4534B 100%);
        padding: 2rem 2.5rem; border-radius: 14px; margin-bottom: 1.8rem;
    }
    .title-banner h1 { color: #FFFFFF; font-size: 2rem; margin: 0; }
    .title-banner p { color: #FFF8F2; font-size: 1rem; margin: 0.4rem 0 0 0; opacity: 0.92; }
    .metric-card {
        background: #FFFFFF; border: 2px solid #E8CCBA; border-radius: 12px;
        padding: 1.2rem 1rem; text-align: center;
        box-shadow: 0 3px 12px rgba(212,83,75,0.08);
    }
    .metric-card .label { color: #D4534B; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; }
    .metric-card .value { font-size: 2rem; font-weight: 700; margin-top: 0.3rem; color: #D4534B; }
    .section-header { color: #D4534B; font-size: 1.35rem; font-weight: 700; border-left: 4px solid #D4534B; padding-left: 0.9rem; margin-top: 1.5rem; margin-bottom: 0.4rem; }
    .insight-box { background: #FFF8F2; border-left: 4px solid #D4534B; padding: 0.9rem 1.1rem; border-radius: 0 10px 10px 0; color: #3B1F0B; font-size: 0.93rem; line-height: 1.55; margin-top: 0.5rem; margin-bottom: 1.2rem; }
    .divider { border: none; border-top: 1.5px solid #E8CCBA; margin: 1.8rem 0; }
    .sidebar-card { background: #FFF8F2; padding: 0.9rem; border-radius: 10px; font-size: 0.88rem; color: #3B1F0B; margin-bottom: 0.8rem; border: 1px solid #E8CCBA; }
</style>
""", unsafe_allow_html=True)

LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(color=TEXT, size=13), margin=dict(t=35, b=50),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        bgcolor="rgba(255,248,242,0.9)", bordercolor=LIGHT_RED, borderwidth=1,
    ),
)


@st.cache_data
def load_approach2():
    df = pd.read_csv("Approach2_ML/output_v4/cnn_final_comparison_ready.csv")
    return df.sort_values("subject").reset_index(drop=True)

@st.cache_data
def load_cnn_report():
    return pd.read_csv(
        "Approach2_ML/output_v4/cnn_report_locked_thresh_0.4420.csv", index_col=0,
    )


a2 = load_approach2()
cnn_report = load_cnn_report()
subjects = a2["subject"].tolist()

st.markdown("""
<div class="title-banner">
    <h1>Dual-Stream 1D Convolutional Neural Network</h1>
    <p>Raw windowed chest and wrist signals processed through parallel convolutional
       streams with Focal Loss, online data augmentation, and a three-way LOSO split</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    with st.expander("Method Summary", expanded=True):
        st.markdown("""<div class="sidebar-card">
            <b>1.</b> Per-subject Z-score normalise raw signals<br>
            <b>2.</b> Create 5-sec sliding windows from chest + wrist<br>
            <b>3.</b> Hold out 1 subject, split remaining into Val-A, Val-B, Train<br>
            <b>4.</b> Train dual-stream CNN with Focal Loss + augmentation<br>
            <b>5.</b> Val-A for early stopping, Val-B for threshold tuning<br>
            <b>6.</b> Test on held-out subject (never seen)
        </div>""", unsafe_allow_html=True)
    with st.expander("Locked Threshold"):
        st.markdown("""<div class="sidebar-card">
            Phase 3 uses a single <b>locked threshold = 0.442</b> (mean of all fold-specific
            thresholds) to simulate real deployment where per-subject tuning is impossible.
        </div>""", unsafe_allow_html=True)
    with st.expander("Why No Feature Importance?"):
        st.markdown("""<div class="sidebar-card">
            Unlike Random Forest (which uses 22 hand-crafted features), the 1D CNN
            learns directly from <b>raw signal windows</b>. Each "feature" is a single
            time-step in a waveform, so traditional importance scores do not produce
            meaningful results. Techniques like Grad-CAM exist but require
            specialised interpretation that goes beyond standard feature ranking.
        </div>""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, label, val in [
    (c1, "Mean Accuracy", f"{a2['accuracy'].mean():.3f}"),
    (c2, "Mean F1", f"{a2['f1'].mean():.3f}"),
    (c3, "Mean ROC-AUC", f"{a2['auc'].mean():.3f}"),
    (c4, "Locked Threshold", "0.442"),
]:
    with col:
        st.markdown(f"""<div class="metric-card">
            <div class="label">{label}</div><div class="value">{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

best = a2.loc[a2["f1"].idxmax()]
worst = a2.loc[a2["f1"].idxmin()]
ns_support = int(cnn_report.loc["non-stress", "support"])
s_support = int(cnn_report.loc["stress", "support"])
ns_recall = cnn_report.loc["non-stress", "recall"]
s_recall = cnn_report.loc["stress", "recall"]
tp = int(round(s_recall * s_support))
fn = s_support - tp
tn = int(round(ns_recall * ns_support))
fp = ns_support - tn
total = ns_support + s_support

non_stress = [
    cnn_report.loc["non-stress", "precision"],
    cnn_report.loc["non-stress", "recall"],
    cnn_report.loc["non-stress", "f1-score"],
]
stress = [
    cnn_report.loc["stress", "precision"],
    cnn_report.loc["stress", "recall"],
    cnn_report.loc["stress", "f1-score"],
]

tab1, tab2, tab3 = st.tabs([
    "Per-Subject F1", "Confusion Matrix", "Classification Report",
])

with tab1:
    colors = [RED if v >= a2["f1"].mean() else LIGHT_RED for v in a2["f1"]]

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=subjects, y=a2["f1"], marker_color=colors,
        text=[f"{v:.3f}" for v in a2["f1"]], textposition="outside",
        textfont=dict(size=11, color=TEXT),
        hovertemplate="Subject: %{x}<br>F1 (locked): %{y:.3f}<br>F1 (fold-specific): %{customdata:.3f}<extra></extra>",
        customdata=a2["f1_phase1"],
    ))
    fig1.add_hline(y=a2["f1"].mean(), line_dash="dot", line_color=BROWN,
                   annotation_text=f"Mean: {a2['f1'].mean():.3f}",
                   annotation_position="top right", annotation_font_color=BROWN)
    fig1.update_layout(**LAYOUT, height=450, yaxis_title="F1 Score",
                       xaxis_title="Subject", yaxis_range=[0, 1.1])
    fig1.update_layout(legend=dict(visible=False))
    fig1.update_xaxes(showgrid=False)
    fig1.update_yaxes(gridcolor="#EDE6DF")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown(
        f'<div class="insight-box"><b>Takeaway:</b> <b>{best["subject"]}</b> gets the '
        f'best score with F1 = {best["f1"]:.3f}. The hardest subject is '
        f'<b>{worst["subject"]}</b> (F1 = {worst["f1"]:.3f}). Unlike Random Forest, the '
        f'CNN avoids complete failures because it learns patterns directly from the raw '
        f'body signals instead of relying on pre-built features.</div>',
        unsafe_allow_html=True,
    )

with tab2:
    cm_text = [[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]]

    fig_cm = go.Figure(data=go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Pred: Non-Stress", "Pred: Stress"],
        y=["True: Stress", "True: Non-Stress"],
        text=cm_text,
        texttemplate="%{text}",
        textfont=dict(size=18, color="white"),
        colorscale=[[0, LIGHT_CORAL], [1, DARK_RED]],
        showscale=False,
        hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{text}<extra></extra>",
    ))
    fig_cm.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color=TEXT, size=13), height=380,
        margin=dict(t=20, b=40),
        xaxis=dict(title="Predicted Label", side="bottom"),
        yaxis=dict(title="True Label", autorange="reversed"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown(
        f'<div class="insight-box"><b>Takeaway:</b> Out of <b>{total:,}</b> total windows '
        f'across all 15 subjects, the CNN correctly classified <b>{tn:,}</b> non-stress and '
        f'<b>{tp:,}</b> stress windows. It mislabeled <b>{fp:,}</b> non-stress windows as '
        f'stress (false alarms) and missed <b>{fn:,}</b> actual stress windows. The false '
        f'alarm rate is low, meaning the model rarely cries wolf.</div>',
        unsafe_allow_html=True,
    )

with tab3:
    class_metrics = ["Precision", "Recall", "F1 Score"]

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=class_metrics, y=non_stress, name="Non-Stress", marker_color=LIGHT_CORAL,
        text=[f"{v:.3f}" for v in non_stress], textposition="outside",
        textfont=dict(size=13, color=BROWN),
        hovertemplate="%{x}: %{y:.3f}<extra>Non-Stress</extra>",
    ))
    fig3.add_trace(go.Bar(
        x=class_metrics, y=stress, name="Stress", marker_color=DARK_RED,
        text=[f"{v:.3f}" for v in stress], textposition="outside",
        textfont=dict(size=13, color=DARK_RED),
        hovertemplate="%{x}: %{y:.3f}<extra>Stress</extra>",
    ))
    fig3.update_layout(**LAYOUT, barmode="group", height=430,
                       yaxis_title="Score", yaxis_range=[0, 1.1])
    fig3.update_xaxes(showgrid=False)
    fig3.update_yaxes(gridcolor="#EDE6DF")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        f'<div class="insight-box"><b>Takeaway:</b> The model is better at spotting '
        f'non-stress (F1 = {non_stress[2]:.3f}) than stress (F1 = {stress[2]:.3f}), which '
        f'is expected because non-stress windows make up about 78% of the data. Even so, '
        f'the CNN correctly flags roughly 3 out of every 4 stressed windows, which is a '
        f'strong result given the uneven data split.</div>',
        unsafe_allow_html=True,
    )
