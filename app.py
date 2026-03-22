import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
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
PRED_CSV_PATH = Path(__file__).parent / "Churn_Pred_0401.csv"
CHARTS_DIR = Path(__file__).parent / "charts"
MODELS_DIR = Path(__file__).parent / "models"
ENSEMBLE_CONFIG_PATH = MODELS_DIR / "ensemble_config.json"


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

st.divider()

# ── Row 5: Ensemble Scoring ────────────────────────────────────────────────
st.header("Ensemble Scoring — RF v3 + LSTM")
st.caption("เปรียบเทียบ score จาก Random Forest v3, LSTM และ Ensemble (ค่าเฉลี่ยถ่วงน้ำหนัก)")


@st.cache_data
def load_ensemble_config():
    """Load ensemble configuration from JSON."""
    if ENSEMBLE_CONFIG_PATH.exists():
        with open(ENSEMBLE_CONFIG_PATH) as f:
            return json.load(f)
    return None


@st.cache_resource
def load_rf_model():
    """Load the RF v3 model."""
    import joblib
    rf_path = MODELS_DIR / "churn_v3_extended.joblib"
    if rf_path.exists():
        return joblib.load(rf_path)
    return None


@st.cache_resource
def load_lstm_model():
    """Load the LSTM churn model."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    class ChurnLSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout, bidirectional=True,
            )
            self.fc = nn.Linear(hidden_size * 2, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return torch.sigmoid(out).squeeze(-1)

    lstm_path = MODELS_DIR / "lstm_churn.pt"
    if not lstm_path.exists():
        return None
    model = ChurnLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.3)
    model.load_state_dict(torch.load(lstm_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


@st.cache_data
def load_pred_data():
    """Load prediction input CSV for ensemble scoring."""
    if PRED_CSV_PATH.exists():
        return pd.read_csv(PRED_CSV_PATH)
    return None


def get_item_columns(pred_df):
    """Extract item/round columns from prediction DataFrame."""
    return [c for c in pred_df.columns if c.startswith("item")]


def run_rf_scoring(rf_model, pred_df, item_cols):
    """Score customers using RF v3 with feature engineering v2."""
    import sys, importlib.util
    fe_path = str(MODELS_DIR / "feature_engineering.py")
    spec = importlib.util.spec_from_file_location("feature_engineering", fe_path)
    fe_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fe_mod)

    feats = fe_mod.engineer_features_v2(pred_df, pred_df, item_cols)
    expected = rf_model.feature_names_in_
    for col in expected:
        if col not in feats.columns:
            feats[col] = 0
    feats = feats[expected]
    feats = feats.fillna(0)
    proba = rf_model.predict_proba(feats)[:, 1]
    return proba * 100  # scale to 0-100


def run_lstm_scoring(lstm_model, pred_df, item_cols):
    """Score customers using LSTM on raw purchase sequences."""
    import torch
    mat = pred_df[item_cols].values.astype(float)
    mat = np.where(np.isnan(mat), 0.0, mat)
    mat = np.log1p(mat)
    # Shape: (n_customers, n_rounds, 1)
    tensor = torch.tensor(mat, dtype=torch.float32).unsqueeze(-1)
    with torch.no_grad():
        proba = lstm_model(tensor).numpy()
    return proba * 100  # scale to 0-100


ensemble_cfg = load_ensemble_config()

if ensemble_cfg is None:
    st.warning("ไม่พบไฟล์ models/ensemble_config.json")
else:
    # ── Show ensemble configuration ────────────────────────────────────
    st.subheader("Ensemble Configuration")
    cfg_left, cfg_right = st.columns(2)
    with cfg_left:
        st.markdown("**Model Weights**")
        w_rf = ensemble_cfg["w_rf"]
        w_lstm = ensemble_cfg["w_lstm"]
        fig_weights = go.Figure(go.Bar(
            x=[w_rf, w_lstm],
            y=["RF v3", "LSTM"],
            orientation="h",
            marker_color=["#2980B9", "#8E44AD"],
            text=[f"{w_rf:.0%}", f"{w_lstm:.0%}"],
            textposition="auto",
        ))
        fig_weights.update_layout(
            xaxis_title="Weight",
            xaxis=dict(range=[0, 1]),
            margin=dict(t=20, b=20, l=60, r=20),
            height=200,
        )
        st.plotly_chart(fig_weights, use_container_width=True)
    with cfg_right:
        st.markdown("**Ensemble Performance (Validation)**")
        perf_df = pd.DataFrame({
            "Metric": ["AUC-ROC", "AUC-PR", "F1 Score"],
            "Value": [
                f"{ensemble_cfg.get('auc_roc', 'N/A'):.4f}" if isinstance(ensemble_cfg.get('auc_roc'), (int, float)) else "N/A",
                f"{ensemble_cfg.get('auc_pr', 'N/A'):.4f}" if isinstance(ensemble_cfg.get('auc_pr'), (int, float)) else "N/A",
                f"{ensemble_cfg.get('f1', 'N/A'):.4f}" if isinstance(ensemble_cfg.get('f1'), (int, float)) else "N/A",
            ],
        })
        st.dataframe(perf_df, hide_index=True, use_container_width=True)
        st.markdown(
            f"**Formula**: `ensemble_score = {w_rf} * RF_score + {w_lstm} * LSTM_score`"
        )

    st.divider()

    # ── Run ensemble scoring ───────────────────────────────────────────
    st.subheader("Run Ensemble Scoring")
    pred_df = load_pred_data()

    if pred_df is None:
        st.info("ไม่พบไฟล์ Churn_Pred สำหรับ scoring — วางไฟล์ Churn_Pred_MMDD.csv ในโฟลเดอร์โปรเจกต์")
    else:
        item_cols = get_item_columns(pred_df)
        st.write(f"Loaded **{len(pred_df):,}** customers, **{len(item_cols)}** round columns")

        if st.button("Score with Ensemble", type="primary"):
            rf_model = load_rf_model()
            lstm_model = load_lstm_model()

            if rf_model is None:
                st.error("ไม่สามารถโหลด RF v3 model (models/churn_v3_extended.joblib)")
            elif lstm_model is None:
                st.error(
                    "ไม่สามารถโหลด LSTM model (models/lstm_churn.pt) — "
                    "ตรวจสอบว่า torch ติดตั้งแล้วและไฟล์ .pt มีอยู่"
                )
            else:
                with st.spinner("Scoring with RF v3..."):
                    rf_scores = run_rf_scoring(rf_model, pred_df, item_cols)
                with st.spinner("Scoring with LSTM..."):
                    lstm_scores = run_lstm_scoring(lstm_model, pred_df, item_cols)

                ensemble_scores = w_rf * rf_scores + w_lstm * lstm_scores
                ensemble_scores = np.clip(ensemble_scores, 0, 100)

                results = pd.DataFrame({
                    "userNo": pred_df["userNo"],
                    "rf_score": np.round(rf_scores, 1),
                    "lstm_score": np.round(lstm_scores, 1),
                    "ensemble_score": np.round(ensemble_scores, 1),
                })
                results["risk_level"] = pd.cut(
                    results["ensemble_score"],
                    bins=[-1, 25, 50, 75, 100],
                    labels=["Low", "Medium", "High", "Critical"],
                )

                # Store in session state so charts persist
                st.session_state["ensemble_results"] = results

        # ── Display results if available ───────────────────────────────
        if "ensemble_results" in st.session_state:
            results = st.session_state["ensemble_results"]
            st.success(f"Scored **{len(results):,}** customers with ensemble model")

            # KPI row
            ens_c1, ens_c2, ens_c3 = st.columns(3)
            ens_c1.metric("Avg RF Score", f"{results['rf_score'].mean():.1f}")
            ens_c2.metric("Avg LSTM Score", f"{results['lstm_score'].mean():.1f}")
            ens_c3.metric("Avg Ensemble Score", f"{results['ensemble_score'].mean():.1f}")

            st.divider()

            # ── Comparison Chart: RF vs LSTM vs Ensemble ───────────────
            st.subheader("RF vs LSTM vs Ensemble — Score Distribution")

            chart_left, chart_right = st.columns(2)

            with chart_left:
                # Overlaid histograms
                fig_comp = go.Figure()
                for col, name, color in [
                    ("rf_score", "RF v3", "#2980B9"),
                    ("lstm_score", "LSTM", "#8E44AD"),
                    ("ensemble_score", "Ensemble", "#E74C3C"),
                ]:
                    fig_comp.add_trace(go.Histogram(
                        x=results[col],
                        name=name,
                        opacity=0.6,
                        marker_color=color,
                        nbinsx=50,
                    ))
                fig_comp.update_layout(
                    barmode="overlay",
                    xaxis_title="Churn Score",
                    yaxis_title="จำนวนลูกค้า",
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                    margin=dict(t=20, b=20),
                    height=400,
                )
                st.plotly_chart(fig_comp, use_container_width=True)

            with chart_right:
                # Risk level distribution comparison
                def risk_counts_for(scores, name):
                    levels = pd.cut(
                        scores, bins=[-1, 25, 50, 75, 100],
                        labels=["Low", "Medium", "High", "Critical"],
                    )
                    counts = levels.value_counts().reindex(RISK_ORDER).fillna(0)
                    return pd.DataFrame({"risk_level": counts.index, "count": counts.values, "Model": name})

                risk_comp = pd.concat([
                    risk_counts_for(results["rf_score"], "RF v3"),
                    risk_counts_for(results["lstm_score"], "LSTM"),
                    risk_counts_for(results["ensemble_score"], "Ensemble"),
                ])
                fig_risk_comp = px.bar(
                    risk_comp,
                    x="risk_level",
                    y="count",
                    color="Model",
                    barmode="group",
                    color_discrete_map={"RF v3": "#2980B9", "LSTM": "#8E44AD", "Ensemble": "#E74C3C"},
                    category_orders={"risk_level": RISK_ORDER},
                )
                fig_risk_comp.update_layout(
                    xaxis_title="Risk Level",
                    yaxis_title="จำนวนลูกค้า",
                    margin=dict(t=20, b=20),
                    height=400,
                )
                st.plotly_chart(fig_risk_comp, use_container_width=True)

            # ── Scatter: RF vs LSTM ────────────────────────────────────
            st.subheader("RF vs LSTM Score Scatter")
            sample_n = min(5000, len(results))
            sample = results.sample(n=sample_n, random_state=42)
            fig_scatter = px.scatter(
                sample,
                x="rf_score",
                y="lstm_score",
                color="risk_level",
                color_discrete_map=COLORS,
                category_orders={"risk_level": RISK_ORDER},
                opacity=0.4,
                hover_data=["userNo", "ensemble_score"],
            )
            fig_scatter.add_shape(
                type="line", x0=0, y0=0, x1=100, y1=100,
                line=dict(dash="dash", color="gray"),
            )
            fig_scatter.update_layout(
                xaxis_title="RF v3 Score",
                yaxis_title="LSTM Score",
                margin=dict(t=20, b=20),
                height=500,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption(
                f"แสดง {sample_n:,} จาก {len(results):,} ลูกค้า — "
                "จุดเหนือเส้นทแยง = LSTM ให้ score สูงกว่า RF"
            )

            st.divider()

            # ── Individual customer lookup ─────────────────────────────
            st.subheader("ค้นหาลูกค้า — เปรียบเทียบ Score")
            search_user = st.text_input("userNo", placeholder="เช่น P12235603")
            if search_user:
                match = results[results["userNo"] == search_user]
                if match.empty:
                    st.warning(f"ไม่พบ userNo: {search_user}")
                else:
                    row = match.iloc[0]
                    lu1, lu2, lu3, lu4 = st.columns(4)
                    lu1.metric("RF v3 Score", f"{row['rf_score']:.1f}")
                    lu2.metric("LSTM Score", f"{row['lstm_score']:.1f}")
                    lu3.metric("Ensemble Score", f"{row['ensemble_score']:.1f}")
                    lu4.metric("Risk Level", row["risk_level"])

                    # Mini bar chart for this customer
                    fig_cust = go.Figure(go.Bar(
                        x=["RF v3", "LSTM", "Ensemble"],
                        y=[row["rf_score"], row["lstm_score"], row["ensemble_score"]],
                        marker_color=["#2980B9", "#8E44AD", "#E74C3C"],
                        text=[f"{row['rf_score']:.1f}", f"{row['lstm_score']:.1f}", f"{row['ensemble_score']:.1f}"],
                        textposition="auto",
                    ))
                    fig_cust.update_layout(
                        yaxis_title="Churn Score",
                        yaxis=dict(range=[0, 100]),
                        margin=dict(t=20, b=20),
                        height=300,
                    )
                    st.plotly_chart(fig_cust, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ── Section: Model Evolution ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.header("Model Evolution")
st.caption("เปรียบเทียบประสิทธิภาพของโมเดลแต่ละเวอร์ชัน — จาก v1 ถึง TX features")

model_evo_data = pd.DataFrame({
    "Model": [
        "v1 LightGBM", "v2 RF Multi", "v3 RF Extended",
        "Ensemble RF+LSTM", "v5 TX", "v7 Full TX", "v8 TX Trend",
    ],
    "Features": [33, 33, 45, "45+seq", 56, 56, 64],
    "AUC-ROC": [None, 0.7986, 0.7981, 0.7993, 0.8013, 0.8001, 0.8002],
    "Notes": [
        "Single period",
        "Multi-period",
        "25-file, production baseline",
        "Weighted ensemble",
        "+TX features (best!)",
        "20-file TX training",
        "+Multi-period trends",
    ],
})


def highlight_best_auc(row):
    """Highlight the row with the best AUC-ROC."""
    styles = [""] * len(row)
    if row["AUC-ROC"] is not None and row["AUC-ROC"] == model_evo_data["AUC-ROC"].max():
        styles = ["background-color: #d4edda; font-weight: bold"] * len(row)
    return styles


model_evo_display = model_evo_data.copy()
model_evo_display["AUC-ROC"] = model_evo_display["AUC-ROC"].apply(
    lambda x: f"{x:.4f}" if pd.notna(x) else "—"
)

st.dataframe(
    model_evo_display.style.apply(
        lambda row: [
            "background-color: #d4edda; font-weight: bold" if row["AUC-ROC"] == "0.8013" else ""
            for _ in row
        ],
        axis=1,
    ),
    hide_index=True,
    use_container_width=True,
)

# AUC-ROC bar chart
evo_chart_data = model_evo_data.dropna(subset=["AUC-ROC"]).copy()
evo_chart_data["color"] = evo_chart_data["AUC-ROC"].apply(
    lambda x: "#27AE60" if x == evo_chart_data["AUC-ROC"].max() else "#2980B9"
)
fig_evo = go.Figure(go.Bar(
    x=evo_chart_data["Model"],
    y=evo_chart_data["AUC-ROC"],
    marker_color=evo_chart_data["color"].tolist(),
    text=evo_chart_data["AUC-ROC"].apply(lambda x: f"{x:.4f}"),
    textposition="outside",
))
fig_evo.update_layout(
    yaxis_title="AUC-ROC",
    yaxis=dict(range=[0.79, 0.805]),
    xaxis_title="",
    margin=dict(t=30, b=20),
    height=400,
)
st.plotly_chart(fig_evo, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ── Section: TX Business Insights ────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.header("TX Business Insights")
st.caption("วิเคราะห์พฤติกรรมการซื้อ-ขาย (Transaction) ของลูกค้าทั้งหมด")

# Key metrics
tx_c1, tx_c2, tx_c3, tx_c4 = st.columns(4)
tx_c1.metric("ลูกค้าทั้งหมด", "3.7M")
tx_c2.metric("Peak ลูกค้า/งวด", "947K")
tx_c3.metric("Rush Buying", "42.6%")
tx_c4.metric("Cancel Rate", "5.9%")

st.markdown(
    """
    **สรุปข้อมูล TX:**
    ฐานลูกค้ารวม 3.7 ล้านราย มียอดสูงสุดถึง 947K ต่องวด —
    พบพฤติกรรม Rush Buying (ซื้อรวดเดียวแล้วหาย) สูงถึง 42.6%
    และอัตราการยกเลิก 5.9% ซึ่งเป็นสัญญาณสำคัญสำหรับการวาง Retention Strategy
    """
)

tx_chart_path = CHARTS_DIR / "tx_business_analysis.png"
if tx_chart_path.exists():
    st.image(str(tx_chart_path), caption="TX Business Analysis — ภาพรวมพฤติกรรมลูกค้า", use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ── Section: Phase Results Gallery ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.header("Phase Results Gallery")
st.caption("ผลลัพธ์จากแต่ละ Phase ของ TX Pipeline")

phase_charts = [
    {
        "path": CHARTS_DIR / "phase28_full_tx.png",
        "caption": "Phase 28 — Full TX Training: ฝึกโมเดลด้วยข้อมูล TX เต็มรูปแบบ (20 files, 56 features)",
    },
    {
        "path": CHARTS_DIR / "phase29_tx_trends.png",
        "caption": "Phase 29 — TX Trends: เพิ่ม Multi-period trend features (64 features) เพื่อจับแนวโน้มการเปลี่ยนแปลง",
    },
]

gallery_cols = st.columns(2)
for idx, chart_info in enumerate(phase_charts):
    with gallery_cols[idx]:
        if chart_info["path"].exists():
            st.image(str(chart_info["path"]), caption=chart_info["caption"], use_container_width=True)
        else:
            st.info(f"ไม่พบไฟล์: {chart_info['path'].name}")

# Show Phase 26 TX Features chart if available (full width)
phase26_path = CHARTS_DIR / "phase26_tx_features.png"
if phase26_path.exists():
    st.subheader("TX Feature Importance")
    st.image(str(phase26_path), caption="Phase 26 — TX Features: ความสำคัญของ TX features ใหม่ที่เพิ่มเข้ามา", use_container_width=True)
