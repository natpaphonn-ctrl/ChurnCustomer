# ChurnCustomer — Lottery Churn Prediction Pipeline

Customer churn prediction & risk scoring pipeline for a **Thai online lottery platform**.
Predicts which customers will stop buying lottery tickets and explains **why** via SHAP.

## Business Context

- **Product**: Thai government lottery (ลอตเตอรี่) — single product, no SKU variation
- **Draw schedule**: 1st and 16th of every month (งวด 1 กับ 16)
- **Sales window**: ~12-13 days before each draw
- **Buying pattern**: Orders spike 2-3x in the last 2-3 days before the draw (rush buying)
- **"Purchase amount"** = number of lottery tickets bought (not monetary value)
- **Median basket**: 4 tickets per order, mean ~6.4
- **Cancel**: Does NOT mean dissatisfaction — means cart timeout or payment failure (~6% rate)
- **High-volume buyers** (1,000+ tickets): Likely resellers/agents, not end consumers
- **Churn**: Customer who bought in period N but does NOT buy in period N+1
- **Target audience**: Thai marketing team — all reports/actions in Thai

## Project Structure

```
ChurnCustomer/
├── app.py                    # Streamlit dashboard (entry point)
├── run_pipeline.py           # Production pipeline (bi-weekly scoring)
├── churn_analysis.ipynb      # Main notebook — Phases 1-15
├── presentation.html         # Interactive stakeholder presentation (34 slides)
├── CLAUDE.md                 # Project instructions
├── README.md                 # Project documentation
├── .gitignore
├── data/
│   ├── churn/                # Churn period CSVs (Churn_YYYY_MMDD_MMDD.csv)
│   ├── predictions/          # Churn_Pred_*.csv, Churn_RiskScore_*.csv, Churn_RiskScore_Explained_*.csv
│   └── transaction/          # TransactionOrder/ files (TX CSVs)
├── scripts/                  # Phase scripts (16-42) — historical/archival
│   ├── run_phase16_17.py     # Survival Analysis + LSTM
│   ├── run_phase18_ensemble.py / run_phase18_models.py  # Ensemble
│   ├── run_phase19.py        # Sudden Drop Detection
│   ├── run_phase20.py        # Probability Calibration
│   ├── run_phase22.py        # Drift Detection
│   ├── run_phase23.py        # A/B Testing Framework
│   ├── run_phase24.py        # CLV Priority Matrix
│   ├── run_phase25.py        # Churn Reason Clustering
│   ├── run_phase26.py        # TX Data Aggregation
│   ├── run_phase27.py        # Cross-Period TX Features
│   ├── run_phase28.py        # Full TX Training
│   ├── run_phase29.py        # Multi-Period TX Trends
│   ├── run_phase30.py        # Time-based TX Feature Engineering
│   ├── run_phase36.py        # Validation Tracking Dashboard
│   ├── run_phase37.py        # Churn Recovery Analysis
│   ├── run_phase38.py        # Segment-Specific Models
│   ├── run_phase39.py        # Prize Impact Deep Dive
│   ├── run_phase40.py        # Cross-Team Insight Report
│   ├── run_phase41.py        # Insight Catalog with Customer Examples
│   ├── run_insights_2_5.py   # Prize & Cancel Event Insights
│   ├── run_insights_6_9.py   # First Purchase & Lifecycle Insights
│   ├── run_insights_10_14.py # Buying Pattern Insights
│   ├── run_insights_15_18.py # Timing Pattern Insights
│   ├── run_insights_19_20.py # Recovery Pattern Insights
│   ├── run_winning_effect.py           # Winning effect storytelling charts
│   ├── run_winning_effect_by_type.py   # Prize-type specific winning effect
│   ├── run_winning_effect_full.py      # Full 30-period winning effect by prize type
│   ├── run_winning_effect_customers.py # Customer-level winning effect data
│   ├── run_prize_shift_analysis.py     # Prize shift estimation
│   ├── run_tx_analysis.py    # TX business analysis
│   └── score_ensemble.py     # Ensemble scoring utility
├── models/                   # Trained models, scalers, feature engineering code
├── charts/                   # Generated visualizations
├── mlruns/                   # MLflow experiment tracking
└── Prize/                    # Prize data (future use)
```

## Key Files

| File | Purpose |
|------|---------|
| `churn_analysis.ipynb` | Main notebook — 15 phases (data prep → validation → scoring) |
| `app.py` | Streamlit dashboard for exploring risk scores |
| `run_pipeline.py` | Production pipeline for bi-weekly scoring |
| `models/feature_engineering.py` | Feature functions: `engineer_features`, `_v2`, `_v3`, `aggregate_tx_features`, `compute_cross_period_tx`, `compute_multi_period_tx`, `compute_time_features` |
| `models/churn_v3_extended.joblib` | RF v3: 45 features, 25-file training (production baseline) |
| `models/churn_v5_tx.joblib` | RF v5: 45+11 TX features, AUC 0.8013 |
| `models/churn_v6_cross_tx.joblib` | RF v6: 45+11+10 cross-period TX features, AUC 0.8013 |
| `models/churn_v7_full_tx.joblib` | RF v7: 56 features, full 20-file TX training |
| `models/churn_v8_tx_trend.joblib` | RF v8: 64 features, multi-period TX trends |
| `scripts/run_phase28.py` | Phase 28: Full TX training pipeline |
| `scripts/run_phase29.py` | Phase 29: Multi-period TX trend training |
| `scripts/run_phase32.py` | Phase 32: LightGBM/XGBoost comparison |
| `scripts/run_phase33.py` | Phase 33: Feature selection & pruning |
| `scripts/run_phase34.py` | Phase 34: Demographics feature test |
| `scripts/run_phase36.py` | Phase 36: Validation tracking dashboard (25 periods) |
| `scripts/run_phase37.py` | Phase 37: Churn recovery analysis |
| `scripts/run_phase38.py` | Phase 38: Segment-specific models |
| `scripts/run_phase39.py` | Phase 39: Prize impact deep dive |
| `scripts/run_phase40.py` | Phase 40: Cross-team insight report |
| `scripts/run_phase41.py` | Phase 41: Insight catalog with customer examples |
| `scripts/run_insights_2_5.py` | Prize & cancel event insights |
| `scripts/run_insights_6_9.py` | First purchase & lifecycle insights |
| `scripts/run_insights_10_14.py` | Buying pattern insights |
| `scripts/run_insights_15_18.py` | Timing pattern insights |
| `scripts/run_insights_19_20.py` | Recovery pattern insights |
| `scripts/run_winning_effect.py` | Winning effect storytelling charts |
| `scripts/run_winning_effect_by_type.py` | Prize-type specific winning effect |
| `scripts/run_winning_effect_full.py` | Full 30-period winning effect by prize type |
| `scripts/run_winning_effect_customers.py` | Customer-level winning effect data |
| `scripts/run_prize_shift_analysis.py` | Prize shift estimation |
| `scripts/run_tx_analysis.py` | TX business analysis (30 periods) |
| `models/churn_segment_regular.joblib` | Segment model: Regular buyers |
| `models/churn_segment_fading.joblib` | Segment model: Fading buyers |
| `models/churn_segment_periodic.joblib` | Segment model: Periodic buyers |
| `data/predictions/insight_catalog.json` | 15 insights with real customer examples |
| `data/predictions/winning_effect_customers_full.csv` | Customer-level winning effect analysis |
| `data/predictions/winning_effect_shift_targets.json` | Winning effect shift estimation targets |
| `models/churn_v4_sudden.joblib` | RF v4: 52 features with sudden-drop detection |
| `models/churn_v2_multi.joblib` | RF v2: multi-period data (33 features) |
| `models/churn_v1_single.joblib` | LightGBM trained on single period |
| `models/calibrator_isotonic.joblib` | Probability calibrator (Phase 20) |
| `models/churn_clusters.joblib` | Churn reason clusters (Phase 25) |
| `models/drift_reference.json` | Drift detection reference (Phase 22) |
| `charts/insight_02` through `insight_20` | 19 purchase behavior insight charts |
| `charts/winning_effect_FIRST.png` | Winning effect by prize type: First Prize |
| `charts/winning_effect_SECOND.png` | Winning effect by prize type: Second Prize |
| `charts/winning_effect_THIRD.png` | Winning effect by prize type: Third Prize |
| `charts/winning_effect_LAST3DIGITS.png` | Winning effect by prize type: Last 3 Digits |
| `charts/winning_effect_LAST2DIGITS.png` | Winning effect by prize type: Last 2 Digits |
| `charts/winning_effect_all_types_summary.png` | Winning effect across all prize types summary |
| `charts/` | Generated visualizations (SHAP, risk distribution, validation, insights, etc.) |
| `mlruns/` | MLflow experiment tracking |

## Pipeline Phases (42)

**Phase 1-15** (notebook): Data prep → EDA → Model training → Validation → Scoring
**Phase 16**: Survival Analysis (Kaplan-Meier, Cox PH)
**Phase 17**: LSTM sequence model
**Phase 18**: 3-model Ensemble (RF+LSTM+GRU), stacking, threshold optimization, error analysis
**Phase 19**: Sudden Drop Detection — 7 new features targeting FN (loyal customers who suddenly churn)
**Phase 20**: Probability Calibration — Isotonic Regression (ensemble already well-calibrated)
**Phase 21**: Automated Pipeline (`run_pipeline.py`)
**Phase 22**: Drift Detection — PSI per feature, tenure_rounds has significant drift
**Phase 23**: A/B Testing Framework — design + backtest
**Phase 24**: Customer Lifetime Value (CLV) — priority matrix
**Phase 25**: Churn Reason Clustering — 3 behavioral clusters
**Phase 26**: TX Data Aggregation — 11 transaction features, AUC 0.7981→0.8013
**Phase 27**: Cross-Period TX Features — 10 delta features comparing consecutive periods
**Phase 28**: Full TX Training — 20-file training with TX features, v7 model AUC 0.8001
**Phase 29**: Multi-Period TX Trends — 8 trend features from 3 consecutive TX periods, v8 model AUC 0.8002
**Phase 30**: Time-based TX Features — 11 time-within-window features (early/rush buyer), v9b model AUC 0.8013
**Phase 31**: TX Coverage Analysis — verified 100% TX match (25% metric was misleading)
**Phase 32**: Model Algorithm Comparison — RF (0.8013) > XGBoost (0.7997) > LightGBM (0.7991)
**Phase 33**: Feature Selection & Pruning — RFE 67→32 features, AUC 0.8010 (unchanged), F1 improved
**Phase 34**: Demographics Feature Test — age/gender/province, MARGINAL +0.03% AUC, age 20-30 highest churn (52.4%)
**Phase 35**: Prize Feature Engineering — 12 prize features, v12 AUC 0.8050, v12b AUC 0.8081 (NEW BEST!)
**Phase 36**: Validation Tracking Dashboard — back-tested 25 periods, AUC 0.7856±0.013, model stable
**Phase 37**: Churn Recovery Analysis — 94.8% of churners return, golden window 3 periods, winners return more
**Phase 38**: Segment-Specific Models — global model robust, Fading segment benefits most (+0.0061 AUC)
**Phase 39**: Prize Impact Deep Dive — winners buy +68.2% more, ROI 2.0x, honeymoon 3 periods
**Phase 40**: Cross-Team Insight Report — demographics x churn, timing, basket, seasonality, cancel analysis
**Phase 41**: Insight Catalog with Customer Examples — 15 insights with real customer P numbers, JSON catalog
**Phase 42**: 20 Purchase Behavior Insights — customer-level purchase patterns with bar charts and real customer examples (57 examples across 19 charts)

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

| Model | Features | AUC-ROC | Notes |
|-------|----------|---------|-------|
| RF v3 (production) | 45 (Churn only) | 0.7981 | Baseline, 25-file training |
| Ensemble (RF+LSTM+GRU) | 45 + sequence | 0.7994 | Production scoring |
| RF v5 (TX) | 56 (45+11 TX) | 0.8013 | +TX single-period features |
| RF v6 (Cross-TX) | 66 (45+11+10) | 0.8013 | +TX cross-period features |
| RF v7 (Full TX) | 56 (45+11 TX) | 0.8001 | Full 20-file TX training |
| RF v8 (TX Trend) | 64 (45+11+8) | 0.8002 | +Multi-period TX trend features |
| RF v9a (Time only) | 56 (45+11 time) | 0.8011 | Time-within-window features |
| RF v9b (Full) | 67 (45+11+11) | 0.8013 | TX agg + time features combined |
| RF v10 (Pruned) | 32 (RFE selected) | 0.8010 | Half features, same AUC, better F1 |
| RF v11 (Demo) | 52 (45+7 demo) | 0.7984 | Demographics marginal (+0.03%) |
| RF v12 (Prize) | 57 (45+12 prize) | 0.8050 | Prize features +0.69% |
| RF v12b (Full) | 79 (45+11TX+11time+12prize) | 0.8081 | **NEW BEST** +1.01% |

- Models are kept **separate for comparison** — not overwritten
- New data sources (e.g., lottery prize data) will create new model versions
- SHAP uses LightGBM proxy trained on RF predictions (agreement 98.6%, 100x faster)

## Data Format

### Churn Files: `data/churn/Churn_YYYY_MMDD_MMDD.csv`
- Columns: `userNo` + demographics + 61-66 rounds of ticket purchase counts (bi-weekly = per lottery draw)
- Each row = one customer, each round column = number of tickets bought
- Values: NaN = not registered, 0 = registered but didn't buy, >0 = number of tickets
- **Important**: Validation files contain ONLY customers who bought (not full population)

### Prediction Files: `data/predictions/Churn_Pred_MMDD.csv`
- Same format but ALL item columns are features (no target column)
- Contains customers who bought in the previous period
- Risk score outputs also in `data/predictions/`: `Churn_RiskScore_*.csv`, `Churn_RiskScore_Explained_*.csv`

### Transaction Files: `data/transaction/Transaction_Order_YYYY_MMDD.csv`
- Columns: `userNo, roundDate, approvedAt_bkk, createdAt_bkk, canceledReason, status, totalItem`
- Multiple rows per customer per period (one row per order)
- `status`: SUCCESS or CANCEL (cancel = timeout/payment failure, NOT dissatisfaction)
- `totalItem`: number of lottery tickets in that order
- `canceledReason`: messy/unclean free text — do not use
- `roundDate`: draw date (1st or 16th of month)
- Each file covers ~12-13 days of sales before the draw
- ~1M rows per file, ~600K unique SUCCESS customers

### TX ↔ Churn Mapping
TX period maps to the **second-to-last** item column in the NEXT Churn file:
- TX_0102 customers = Churn_0102_0117 rows (item2026_01_02 column)
- TX_0201 customers = Churn_0201_0216 rows (item2026_02_01 column)

### Lottery Sales Pattern
- Draw dates: 1st and 16th of every month
- Sales open **3-4 days after previous draw** (gap days with no sales)
- Sales window: **11-13 days** until draw day
- Week 1 (~5 days): steady ~40K-55K orders/day
- Mid period: gradual ramp 60K-90K orders/day
- **Last 2 days: spike to 195K-235K orders/day** (3-4x normal)
- Last day: avg tickets/order jumps to 8-10 (vs normal 5-6) — rush buying before draw
- Per period: ~6-7M tickets, ~600K-650K customers, ~10-11 tickets/customer
- Cancel rate: ~6% consistent (cart timeout / payment failure)

## Key Parameters

- `MIN_TENURE = 3` — Exclude customers with < 3 rounds of history
- New customers (tenure < 3) get `score = -1` and `risk_level = 'New Customer'`
- Threshold for binary prediction: 0.5 (score 50)

## Tech Stack

pandas, numpy, LightGBM, XGBoost, scikit-learn, SHAP, matplotlib, seaborn, Streamlit, MLflow

## Conventions

- **Language**: Thai in presentation/README/actions (target audience = Thai marketing team)
- **Data privacy**: CSV, pkl, joblib, npy, parquet files and `data/` directory are in `.gitignore`
- **Data layout**: All CSVs live under `data/` (churn period files in `data/churn/`, predictions in `data/predictions/`, TX in `data/transaction/`)
- **Scripts**: Phase scripts (16-30) are in `scripts/` — run from project root: `python3 scripts/run_phaseXX.py`
- **Feature engineering**: Use vectorized NumPy operations — avoid Python loops where possible
- **MLflow**: Experiment tracking in `mlruns/`
- **Streamlit**: Run with `streamlit run app.py`
- **SHAP for RF**: Use LightGBM proxy (train on RF pseudo-labels) — RF TreeExplainer is too slow for 600K+ customers
