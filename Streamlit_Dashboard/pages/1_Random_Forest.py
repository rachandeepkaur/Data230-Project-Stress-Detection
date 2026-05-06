import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Random Forest Analysis", layout="wide")

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
        background: linear-gradient(135deg, #A0522D 0%, #A0522D 100%);
        padding: 2rem 2.5rem; border-radius: 14px; margin-bottom: 1.8rem;
    }
    .title-banner h1 { color: #FFFFFF; font-size: 2rem; margin: 0; }
    .title-banner p { color: #FFF8F2; font-size: 1rem; margin: 0.4rem 0 0 0; opacity: 0.92; }
    .metric-card {
        background: #FFFFFF; border: 2px solid #C9A882; border-radius: 12px;
        padding: 1.2rem 1rem; text-align: center;
        box-shadow: 0 3px 12px rgba(160,82,45,0.08);
    }
    .metric-card .label { color: #A0522D; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; }
    .metric-card .value { font-size: 2rem; font-weight: 700; margin-top: 0.3rem; color: #A0522D; }
    .section-header { color: #A0522D; font-size: 1.35rem; font-weight: 700; border-left: 4px solid #A0522D; padding-left: 0.9rem; margin-top: 1.5rem; margin-bottom: 0.4rem; }
    .insight-box { background: #FFF8F2; border-left: 4px solid #A0522D; padding: 0.9rem 1.1rem; border-radius: 0 10px 10px 0; color: #3B1F0B; font-size: 0.93rem; line-height: 1.55; margin-top: 0.5rem; margin-bottom: 1.2rem; }
    .divider { border: none; border-top: 1.5px solid #C9A882; margin: 1.8rem 0; }
    .sidebar-card { background: #FFF8F2; padding: 0.9rem; border-radius: 10px; font-size: 0.88rem; color: #3B1F0B; margin-bottom: 0.8rem; border: 1px solid #C9A882; }
</style>
""", unsafe_allow_html=True)

LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(color=TEXT, size=13), margin=dict(t=35, b=50),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        bgcolor="rgba(255,248,242,0.9)", bordercolor=LIGHT_BROWN, borderwidth=1,
    ),
)


@st.cache_data
def load_approach1():
    df = pd.read_csv("Approach1_ML/approach1_correct_loso_results.csv")
    return df.sort_values("subject").reset_index(drop=True)

@st.cache_data
def load_shap_importance():
    return pd.read_csv("Approach1_ML/shap_feature_importance.csv")


a1 = load_approach1()
shap_df = load_shap_importance()
subjects = a1["subject"].tolist()

SIGNAL_COLORS = {
    "ecg": "#D4534B", "eda": "#A0522D", "emg": "#C9A882",
    "resp": "#D4715E", "temp": "#6B3A2A", "acc": "#E8CCBA",
}

def get_signal_color(feature_name):
    prefix = feature_name.split("_")[0]
    return SIGNAL_COLORS.get(prefix, LIGHT_BROWN)

st.markdown("""
<div class="title-banner">
    <h1>Random Forest Classifier</h1>
    <p>300-tree ensemble trained on 22 hand-crafted statistical features with
       SMOTE oversampling, random undersampling, and per-fold threshold optimisation</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    with st.expander("Method Summary", expanded=True):
        st.markdown("""<div class="sidebar-card">
            <b>1.</b> Extract 22 statistical features from 5-sec windows<br>
            <b>2.</b> For each LOSO fold, hold out one subject as test<br>
            <b>3.</b> Balance training data with SMOTE + undersampling<br>
            <b>4.</b> Tune threshold via inner 5-fold GroupKFold<br>
            <b>5.</b> Retrain on all 14 subjects, predict on held-out
        </div>""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, label, val in [
    (c1, "Mean Accuracy", f"{a1['accuracy'].mean():.3f}"),
    (c2, "Mean F1", f"{a1['f1'].mean():.3f}"),
    (c3, "Mean ROC-AUC", f"{a1['roc_auc'].mean():.3f}"),
    (c4, "Mean Balanced Acc", f"{a1['balanced_accuracy'].mean():.3f}"),
]:
    with col:
        st.markdown(f"""<div class="metric-card">
            <div class="label">{label}</div><div class="value">{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

best = a1.loc[a1["f1"].idxmax()]
zeros = a1[a1["f1"] == 0]["subject"].tolist()

tab1, tab2 = st.tabs([
    "Per-Subject F1",
    "Feature Importance (SHAP)",
])

with tab1:
    colors = [RED if v >= a1["f1"].mean() else LIGHT_BROWN for v in a1["f1"]]

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=subjects, y=a1["f1"], marker_color=colors,
        text=[f"{v:.3f}" for v in a1["f1"]], textposition="outside",
        textfont=dict(size=11, color=TEXT),
        hovertemplate="Subject: %{x}<br>F1: %{y:.3f}<br>Threshold: %{customdata:.2f}<extra></extra>",
        customdata=a1["threshold_used"],
    ))
    fig1.add_hline(y=a1["f1"].mean(), line_dash="dot", line_color=BROWN,
                   annotation_text=f"Mean: {a1['f1'].mean():.3f}",
                   annotation_position="top right", annotation_font_color=BROWN)
    fig1.update_layout(**LAYOUT, height=450, yaxis_title="F1 Score",
                       xaxis_title="Subject", yaxis_range=[0, 1.1])
    fig1.update_layout(legend=dict(visible=False))
    fig1.update_xaxes(showgrid=False)
    fig1.update_yaxes(gridcolor="#EDE6DF")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown(
        f'<div class="insight-box"><b>Takeaway:</b> Results differ a lot from person to '
        f'person. <b>{best["subject"]}</b> gets the highest F1 of {best["f1"]:.3f}, while '
        f'<b>{", ".join(zeros)}</b> score zero -- the model could not detect any stress at '
        f'all for those subjects. This shows how hard it is for one model to work well '
        f'across different people using only statistical features.</div>',
        unsafe_allow_html=True,
    )

with tab2:
    shap_sorted = shap_df.sort_values("mean_shap_value", ascending=True)
    bar_colors_shap = [get_signal_color(f) for f in shap_sorted["feature"]]

    fig_shap = go.Figure()
    fig_shap.add_trace(go.Bar(
        y=shap_sorted["feature"], x=shap_sorted["mean_shap_value"],
        orientation="h", marker_color=bar_colors_shap,
        text=[f"{v:.4f}" for v in shap_sorted["mean_shap_value"]],
        textposition="outside", textfont=dict(size=11, color=TEXT),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    ))
    fig_shap.update_layout(
        **LAYOUT, height=560, xaxis_title="Mean |SHAP| Value",
        yaxis_title="",
    )
    fig_shap.update_layout(margin=dict(l=120, t=35, b=50), legend=dict(visible=False))
    fig_shap.update_yaxes(showgrid=False)
    fig_shap.update_xaxes(gridcolor="#EDE6DF")
    st.plotly_chart(fig_shap, use_container_width=True)

    shap_top3 = shap_df.head(3)["feature"].tolist()
    st.markdown(
        f'<div class="insight-box"><b>Takeaway:</b> SHAP (SHapley Additive exPlanations) '
        f'measures each feature\'s actual impact on every single prediction, making the '
        f'model\'s decisions transparent and interpretable. '
        f'The top three are <b>{shap_top3[0]}</b>, <b>{shap_top3[1]}</b>, and '
        f'<b>{shap_top3[2]}</b>. Heart rate variability (ecg_std) and skin conductance '
        f'fluctuations (eda_std) are the strongest stress indicators -- when these signals '
        f'spike, the model is most likely to flag a window as stressed.</div>',
        unsafe_allow_html=True,
    )
