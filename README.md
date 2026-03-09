# Churn Prediction & Customer Risk Scoring Pipeline

End-to-end machine learning pipeline for predicting customer churn and generating explainable risk scores — built to help marketing teams understand **why** each customer is at risk and take action accordingly.

## Presentation

[View Interactive Presentation](https://natpaphonn-ctrl.github.io/ChurnCustomer/presentation.html)

---

## Problem Statement

ทีม Marketing ต้องการทราบว่าลูกค้าคนไหนมีแนวโน้มจะหยุดซื้อ (Churn) และ **ทำไม** ถึงได้คะแนนความเสี่ยงแบบนั้น เพื่อวางแผน campaign ได้ตรงจุด

**Goal:** Predict churn probability (0–100) for every customer and provide top reasons driving each score.

---

## Pipeline Overview (11 Phases)

| Phase | Description |
|-------|-------------|
| **1. Data Preparation** | Load & clean 6 periods of purchase data (590,644 customers) |
| **2. Feature Engineering** | Create 33 behavioral features from purchase history |
| **2.5. EDA** | Explore feature distributions, correlations, and churn patterns |
| **3. Model Training** | Train & compare 4 ML models with threshold optimization |
| **4. Risk Scoring** | Calibrate churn probability into 0–100 risk scores |
| **5. Cross-Period Validation** | Validate model stability across different time periods |
| **6. Multi-Period Validation** | Extended validation across all remaining periods |
| **7. Multi-Period Retrain** | Retrain with combined data for improved generalization |
| **8. Production Scoring** | Score all 569,087 customers for the latest period (0316) |
| **9. Customer Segmentation** | Cluster customers into 8 segments by purchase patterns |
| **10. Revenue at Risk** | Quantify business impact by risk level |
| **11. SHAP Explainability** | Explain individual risk scores with per-customer top reasons |

---

## Features Engineered (33 Features)

Features are grouped into 7 categories derived from purchase history across 61 rounds:

| Category | Features | Examples |
|----------|----------|----------|
| **Lifecycle** | 3 | Tenure (rounds), total active rounds, activity rate |
| **Recency** | 4 | Bought last round, active in last 3/6 rounds |
| **Frequency & Volume** | 6 | Total items, avg items/round, max single purchase |
| **Trend** | 4 | Purchase trend slope, recent vs. historical ratio |
| **Gap & Consistency** | 6 | Avg gap between buys, max consecutive gap, std of gaps |
| **Reactivation** | 3 | Reactivation count, gap before last buy |
| **Advanced** | 7 | EWM recent activity, purchase concentration, burst score |

---

## Models & Performance

4 models were trained and compared:

| Model | AUC-ROC | AUC-PR | F1 | Accuracy |
|-------|---------|--------|----|----------|
| **LightGBM** | 0.7627 | 0.7298 | 0.6935 | 0.6931 |
| XGBoost | 0.7626 | 0.7298 | 0.6938 | 0.6934 |
| Random Forest | 0.7614 | 0.7285 | 0.6919 | 0.6921 |
| Logistic Regression | 0.7555 | 0.7203 | 0.6867 | 0.6891 |

**Cross-period AUC-ROC: 0.7848–0.7965** (spread = 0.0118) — stable across time periods.

---

## Key Findings

### Top 5 Churn Predictors (by SHAP importance)

| Rank | Feature | Mean |SHAP| | Insight |
|------|---------|-------------|---------|
| 1 | Active Last 6 Rounds | 0.4572 | **2× more important than #2** — ซื้อกี่งวดใน 6 งวดล่าสุด |
| 2 | EWM Recent Activity | 0.2186 | ค่าถ่วงน้ำหนักการซื้อล่าสุด |
| 3 | Total Active Rounds | 0.1873 | จำนวนงวดที่ซื้อทั้งหมด |
| 4 | Avg Gap Between Buys | 0.1648 | ค่าเฉลี่ยช่วงว่างระหว่างซื้อ |
| 5 | Items Last 6 Rounds | 0.1339 | จำนวนชิ้นที่ซื้อใน 6 งวดล่าสุด |

### Risk Distribution (569,087 customers)

| Risk Level | Score Range | Customers | % | Actual Churn Rate |
|------------|------------|-----------|---|-------------------|
| Critical | 76–100 | 18,896 | 16.5% | 80.2% |
| High | 51–75 | 40,180 | 35.2% | 60.9% |
| Medium | 26–50 | 34,388 | 30.1% | 36.7% |
| Low | 0–25 | 20,752 | 18.2% | 14.2% |

---

## Output Files

| File | Description |
|------|-------------|
| `charts/` | SHAP importance, beeswarm, waterfall, and risk-level comparison charts |
| `presentation.html` | Interactive 19-slide presentation for stakeholders |

> **Note:** Dataset and output CSV files are not included in this repository for data privacy.

---

## Tech Stack

- **Python** — pandas, numpy, scikit-learn
- **Models** — LightGBM, XGBoost, Random Forest, Logistic Regression
- **Explainability** — SHAP (TreeExplainer)
- **Visualization** — matplotlib, seaborn
- **Presentation** — Custom HTML/CSS/JS with interactive features

---

## Project Structure

```
ChurnCustomer/
├── churn_analysis.ipynb              # Main notebook (11 phases)
├── presentation.html                 # Interactive presentation
├── README.md
└── charts/                           # Generated chart images
    ├── shap_importance.png
    ├── shap_beeswarm.png
    ├── shap_by_risk_level.png
    └── shap_waterfall_examples.png
```
