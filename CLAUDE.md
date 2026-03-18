# ChurnCustomer — Churn Prediction Pipeline

Customer churn prediction & risk scoring pipeline for a Thai marketing team.
Predicts which customers will stop purchasing and explains **why** via SHAP.

## Key Files

| File | Purpose |
|------|---------|
| `churn_analysis.ipynb` | Main notebook — 15 phases (data prep → validation → scoring) |
| `app.py` | Streamlit dashboard for exploring risk scores |
| `models/feature_engineering.py` | Feature engineering functions (`engineer_features`, `engineer_features_v2`) |
| `models/churn_v3_extended.joblib` | Best model: Random Forest v2, 45 features, 25-file training |
| `models/churn_v2_multi.joblib` | Random Forest retrained on multi-period data (33 features) |
| `models/churn_v1_single.joblib` | LightGBM trained on single period |
| `models/scaler_multi.joblib` | Feature scaler for multi-period models |
| `charts/` | Generated visualizations (SHAP, risk distribution, validation, etc.) |
| `presentation.html` | Interactive stakeholder presentation (22 slides) |
| `mlruns/` | MLflow experiment tracking |

## Pipeline Phases (15)

1. **Data Preparation** — Load & clean purchase CSVs (~570K customers)
2. **Feature Engineering** — 33 base features from purchase history
3. **EDA** — Distribution, correlation, churn pattern analysis
4. **Model Training** — 4 models compared (LightGBM wins)
5. **Risk Scoring** — Calibrated probability → 0–100 risk score
6. **Cross-Period Validation** — Stability check across time periods
7. **Multi-Period Validation** — Extended validation across all periods
8. **Multi-Period Retrain** — Retrain with combined data (v2 model)
9. **Production Scoring** — Score all customers for latest period
10. **Customer Segmentation** — 8 clusters by purchase patterns
11. **SHAP Explainability** — Per-customer top reasons for risk score
12. **Extended Training** — 25 files, 45 features (v3 model)
13. **Export Models** — Save v1/v2/v3 as joblib + MLflow
14. **Validation** — Validate predicted scores against actual churn data
15. **Production Scoring (next period)** — Score + SHAP for next period

## Recurring Operational Flow

Every bi-weekly period follows a **2-step loop**:

### Step 1: Validate previous predictions
- **Input**: `Churn_YYYY_MMDD1_MMDD2.csv` (actual results — contains only customers who bought)
- **Compare with**: `Churn_RiskScore_YYYY_MMDD2.csv` (predicted scores from previous cycle)
- **Churn definition**: Customer in prediction file but NOT in actual file = churned
- **Output**: AUC-ROC, churn rate per risk level, confusion matrix, validation charts
- **Action**: Compare actual vs target, assess model stability

### Step 2: Score next period
- **Input**: `Churn_Pred_MMDD3.csv` (customers who bought in MMDD2, predict for MMDD3)
- **Model**: v3 (Random Forest, 45 features from `engineer_features_v2()`)
- **SHAP**: Use LightGBM proxy (train on RF predictions, 98%+ agreement, 100x faster)
- **Output**:
  - `Churn_RiskScore_YYYY_MMDD3.csv` — risk scores + recommended actions
  - `Churn_RiskScore_Explained_YYYY_MMDD3.csv` — top 5 SHAP reasons per customer
  - `charts/risk_scoring_MMDD3.png` — distribution charts
- **Action**: Set retention targets per risk level for marketing/ads/marcom teams

### Timeline Example
```
งวด 0316 ──validate──→ งวด 0401 ──validate──→ งวด 0416 ──→ ...
   ↓                      ↓                      ↓
Score 0401              Score 0416              Score 0502
```

### Input Files per Cycle (2 files)
| File | Purpose |
|------|---------|
| `Churn_YYYY_MMDD1_MMDD2.csv` | Actual results: who bought in MMDD2 (for validation) |
| `Churn_Pred_MMDD3.csv` | Customers who bought MMDD2, predict churn for MMDD3 |

### Output Files per Cycle (4 files)
| File | Purpose |
|------|---------|
| `Churn_RiskScore_YYYY_MMDD3.csv` | Risk scores (eligible + new customers score=-1) |
| `Churn_RiskScore_Explained_YYYY_MMDD3.csv` | SHAP top 5 reasons per customer |
| `charts/phaseXX_validation_MMDD2.png` | Validation charts (ROC, churn rates, confusion matrix) |
| `charts/risk_scoring_MMDD3.png` | Risk score distribution |

## Validation Results (Latest: 0316)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.7958 (training: 0.7981 — stable) |
| AUC-PR | 0.6887 |
| F1 | 0.6757 |
| Accuracy | 71.4% |

| Risk Level | Actual Churn Rate | Retention |
|------------|-------------------|-----------|
| Low (0-25) | 7.7% | 92.3% |
| Medium (26-50) | 26.1% | 73.9% |
| High (51-75) | 53.0% | 47.0% |
| Critical (76-100) | 76.1% | 23.9% |

## Risk Score Levels & Actions

| Level | Score | Action (Thai) |
|-------|-------|---------------|
| Low | 0–25 | ไม่ต้องดำเนินการ — ลูกค้าภักดี |
| Medium | 26–50 | ส่ง offer เบาๆ — รักษาความสัมพันธ์ |
| High | 51–75 | เร่งส่ง promotion — เริ่มมีสัญญาณ Churn |
| Critical | 76–100 | ติดต่อทันที — โอกาส Churn สูงมาก |

## Target Setting

Set retention targets per risk level using validated churn rates:
- **Method**: Midpoint between previous target and actual (achievable but pushes higher)
- Targets are set for marketing, ads, and marcom teams
- Compare actual vs target each cycle to assess campaign effectiveness

## Features

45 features in `engineer_features_v2()`, built on top of 33 base features from `engineer_features()`:

| Category | Count | Examples |
|----------|-------|---------|
| Lifecycle | 3 | `tenure_rounds`, `time_to_first_purchase`, `is_new` |
| Recency | 1 | `rounds_since_last_purchase` |
| Frequency & Volume | 5 | `total_active_rounds`, `purchase_frequency_ratio`, `total_items` |
| Recent Windows | 5 | `items_last_1`, `items_last_3`, `active_last_6` |
| Trend & Momentum | 5 | `trend_slope`, `recent_vs_early_ratio`, `ewm_recent` |
| Gap & Consistency | 6 | `avg_gap`, `max_gap`, `current_zero_streak`, `coeff_of_variation` |
| Reactivation | 2 | `n_reactivations`, `ever_reactivated` |
| Concentration | 3 | `gini_coefficient`, `top3_rounds_pct`, `is_declining_3` |
| Periodicity | 3 | `gap_regularity`, `purchase_autocorr_lag1`, `purchase_entropy` |
| Behavioral Shift | 3 | `freq_shift_early_to_late`, `vol_shift_early_to_late`, `mid_dip_indicator` |
| Advanced Trend | 3 | `trend_acceleration`, `trend_r_squared`, `late_momentum` |
| Engagement Depth | 3 | `purchase_amount_diversity`, `spending_quartile`, `max_active_streak` |
| Stage | 2 | `is_new`, `is_mature` |

## Models

- **Random Forest v2** (best, v3 model) — AUC-ROC 0.7981 (test), 0.7958 (validation)
- LightGBM, XGBoost, Logistic Regression also trained
- SHAP uses LightGBM proxy trained on RF predictions (agreement 98.6%, 100x faster)

## Data Format

CSV files: `Churn_YYYY_MMDD_MMDD.csv`
- Columns: `userNo` + demographics + 61-66 rounds of purchase amounts (bi-weekly)
- Each row = one customer, each round column = purchase amount
- Values: NaN = not registered, 0 = registered but didn't buy, >0 = bought
- **Important**: Validation files contain ONLY customers who bought (not full population)

`Churn_Pred_MMDD.csv`
- Same format but ALL item columns are features (no target column)
- Contains customers who bought in the previous period

## Key Parameters

- `MIN_TENURE = 3` — Exclude customers with < 3 rounds of history
- New customers (tenure < 3) get `score = -1` and `risk_level = 'New Customer'`
- Threshold for binary prediction: 0.5 (score 50)

## Tech Stack

pandas, numpy, LightGBM, XGBoost, scikit-learn, SHAP, matplotlib, seaborn, Streamlit, MLflow

## Conventions

- **Language**: Thai in presentation/README/actions (target audience = Thai marketing team)
- **Data privacy**: CSV, pkl, joblib, npy, parquet files are in `.gitignore`
- **Feature engineering**: Use vectorized NumPy operations — avoid Python loops where possible
- **MLflow**: Experiment tracking in `mlruns/`
- **Streamlit**: Run with `streamlit run app.py`
- **SHAP for RF**: Use LightGBM proxy (train on RF pseudo-labels) — RF TreeExplainer is too slow for 600K+ customers
