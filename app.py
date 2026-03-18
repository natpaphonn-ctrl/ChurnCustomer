import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

COLORS = {
    "Critical": "#C0392B",
    "High": "#E67E22",
    "Medium": "#F39C12",
    "Low": "#27AE60",
}
RISK_ORDER = ["Critical", "High", "Medium", "Low"]
CSV_PATH = Path(__file__).parent / "Churn_RiskScore_Explained_2026_0316.csv"
CHARTS_DIR = Path(__file__).parent / "charts"


@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df["risk_level"] = pd.Categorical(df["risk_level"], categories=RISK_ORDER, ordered=True)
    return df


df_raw = load_data()

# ── Sidebar Filters ─────────────────────────────────────────────────────────
st.sidebar.header("Filters")
selected_risks = st.sidebar.multiselect(
    "Risk Level",
    options=RISK_ORDER,
    default=RISK_ORDER,
)
score_min, score_max = st.sidebar.slider(
    "Churn Score Range",
    min_value=0.0,
    max_value=100.0,
    value=(0.0, 100.0),
    step=0.1,
)

df = df_raw[
    (df_raw["risk_level"].isin(selected_risks))
    & (df_raw["churn_score"] >= score_min)
    & (df_raw["churn_score"] <= score_max)
]

# ── Header ──────────────────────────────────────────────────────────────────
st.title("Churn Prediction Dashboard")
st.caption("งวด 0316 — ข้อมูลรันในเครื่อง local เท่านั้น")

# ── Row 1: KPI Cards ────────────────────────────────────────────────────────
total = len(df)
critical = (df["risk_level"] == "Critical").sum()
high = (df["risk_level"] == "High").sum()
avg_score = df["churn_score"].mean() if total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", f"{total:,}")
c2.metric("Critical Risk", f"{critical:,}", delta=f"{critical / total * 100:.1f}%" if total else "0%")
c3.metric("High Risk", f"{high:,}", delta=f"{high / total * 100:.1f}%" if total else "0%")
c4.metric("Avg Churn Score", f"{avg_score:.1f}")

st.divider()

# ── Row 2: Risk Distribution ────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("สัดส่วนลูกค้าตาม Risk Level")
    risk_counts = (
        df.groupby("risk_level", observed=True)
        .size()
        .reindex(RISK_ORDER)
        .dropna()
        .reset_index(name="count")
    )
    fig_donut = px.pie(
        risk_counts,
        names="risk_level",
        values="count",
        hole=0.45,
        color="risk_level",
        color_discrete_map=COLORS,
    )
    fig_donut.update_traces(textinfo="label+percent+value", textposition="outside")
    fig_donut.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_donut, use_container_width=True)

with right:
    st.subheader("Churn Score Distribution")
    fig_hist = px.histogram(
        df,
        x="churn_score",
        nbins=50,
        color="risk_level",
        color_discrete_map=COLORS,
        category_orders={"risk_level": RISK_ORDER},
    )
    fig_hist.update_layout(
        xaxis_title="Churn Score",
        yaxis_title="จำนวนลูกค้า",
        bargap=0.05,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ── Row 3: SHAP Feature Importance ──────────────────────────────────────────
st.subheader("Top 10 Features by Mean |SHAP| Value")

reason_cols = [f"reason_{i}" for i in range(1, 6)]
shap_cols = [f"reason_{i}_shap" for i in range(1, 6)]

# Melt reason names and SHAP values side-by-side
records = []
for i in range(1, 6):
    sub = df[["userNo", f"reason_{i}", f"reason_{i}_shap"]].rename(
        columns={f"reason_{i}": "feature", f"reason_{i}_shap": "shap"}
    )
    records.append(sub)

melted = pd.concat(records, ignore_index=True).dropna(subset=["feature", "shap"])
melted["abs_shap"] = melted["shap"].abs()

top10 = (
    melted.groupby("feature")["abs_shap"]
    .mean()
    .nlargest(10)
    .sort_values()
    .reset_index()
)

fig_shap = px.bar(
    top10,
    x="abs_shap",
    y="feature",
    orientation="h",
    color_discrete_sequence=["#2980B9"],
)
fig_shap.update_layout(
    xaxis_title="Mean |SHAP value|",
    yaxis_title="",
    margin=dict(t=20, b=20),
    height=400,
)
st.plotly_chart(fig_shap, use_container_width=True)

st.divider()

# ── Row 4: Risk Level Breakdown ─────────────────────────────────────────────
st.subheader("SHAP Values by Risk Level")

shap_img = CHARTS_DIR / "shap_by_risk_level.png"
if shap_img.exists():
    st.image(str(shap_img), use_container_width=True)
else:
    # Fallback: compute grouped bar from data
    melted_risk = []
    for i in range(1, 6):
        sub = df[["risk_level", f"reason_{i}", f"reason_{i}_shap"]].rename(
            columns={f"reason_{i}": "feature", f"reason_{i}_shap": "shap"}
        )
        melted_risk.append(sub)
    mr = pd.concat(melted_risk, ignore_index=True).dropna(subset=["feature", "shap"])
    mr["abs_shap"] = mr["shap"].abs()

    top_features = mr.groupby("feature")["abs_shap"].mean().nlargest(8).index.tolist()
    mr_top = mr[mr["feature"].isin(top_features)]
    grouped = mr_top.groupby(["risk_level", "feature"], observed=True)["abs_shap"].mean().reset_index()

    fig_risk = px.bar(
        grouped,
        x="feature",
        y="abs_shap",
        color="risk_level",
        barmode="group",
        color_discrete_map=COLORS,
        category_orders={"risk_level": RISK_ORDER},
    )
    fig_risk.update_layout(
        xaxis_title="Feature",
        yaxis_title="Mean |SHAP value|",
        margin=dict(t=20, b=20),
        height=450,
    )
    st.plotly_chart(fig_risk, use_container_width=True)
