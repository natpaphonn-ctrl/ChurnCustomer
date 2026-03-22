#!/usr/bin/env python3
"""Run Phase 16 (Survival Analysis) + Phase 17 (LSTM) standalone."""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, 'models')
from feature_engineering import engineer_features, engineer_features_v2

MIN_TENURE = 3

# ═══════════════════════════════════════════════════════════════
# Phase 16: Survival Analysis — BG/NBD + Kaplan-Meier + Cox PH
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("Phase 16: Survival Analysis — BG/NBD + Kaplan-Meier")
print("=" * 60)

# --- Cell 1: Prepare Survival Data ---
t0 = time.time()
print("\n[1/6] Preparing survival data...")

# Use a training file with actual churn labels (last col = target)
# 0301_0316 is validation file (buyers only → churn=0%), use 0216_0301 instead
df_surv = pd.read_csv('Churn_2026_0216_0301.csv', low_memory=False)
item_cols_surv = [c for c in df_surv.columns if c.startswith('item')]
mat_surv = df_surv[item_cols_surv].apply(pd.to_numeric, errors='coerce')
n_rounds_surv = len(item_cols_surv)

print(f"  Loaded: {len(df_surv):,} customers, {n_rounds_surv} rounds")

mat_vals_surv = mat_surv.values.astype(float)
registered = ~np.isnan(mat_vals_surv)
purchased = np.where(registered, mat_vals_surv > 0, False).astype(bool)

# --- Build RFM for BG/NBD ---
# Use only feature columns (exclude last = target) for RFM
feat_cols_surv = item_cols_surv[:-1]
n_feat_rounds = len(feat_cols_surv)
mat_feat_vals = mat_vals_surv[:, :-1]
reg_feat = registered[:, :-1]
purch_feat = purchased[:, :-1]

# Churn label
churn_surv = (mat_vals_surv[:, -1] == 0).astype(int)

# Tenure filter
first_reg = np.where(reg_feat.any(axis=1), reg_feat.argmax(axis=1), n_feat_rounds)
tenure = n_feat_rounds - first_reg
valid_mask = tenure >= MIN_TENURE

print(f"  Eligible (tenure >= {MIN_TENURE}): {valid_mask.sum():,}")
print(f"  Churn rate: {churn_surv[valid_mask].mean():.1%}")

# Compute RFM per customer (vectorized where possible)
frequency = np.zeros(len(df_surv), dtype=int)
recency = np.zeros(len(df_surv), dtype=float)
T_obs = np.zeros(len(df_surv), dtype=float)

for i in range(len(df_surv)):
    if not valid_mask[i]:
        continue
    pr = np.where(purch_feat[i])[0]
    if len(pr) >= 2:
        frequency[i] = len(pr) - 1  # repeat purchases only
        recency[i] = pr[-1] - pr[0]  # last - first purchase
    elif len(pr) == 1:
        frequency[i] = 0
        recency[i] = 0
    T_obs[i] = (n_feat_rounds - 1 - pr[0]) if len(pr) >= 1 else 0

rfm_summary = pd.DataFrame({
    'userNo': df_surv['userNo'].values,
    'frequency': frequency,
    'recency': recency,
    'T': T_obs,
    'churn': churn_surv,
})
rfm_valid = rfm_summary[valid_mask & (rfm_summary['T'] > 0)].copy()

print(f"  Valid for BG/NBD: {len(rfm_valid):,}")
print(f"  frequency: mean={rfm_valid['frequency'].mean():.1f}, median={rfm_valid['frequency'].median():.0f}")
print(f"  recency:   mean={rfm_valid['recency'].mean():.1f}")
print(f"  T:         mean={rfm_valid['T'].mean():.1f}")
print(f"  ({time.time()-t0:.1f}s)")

# --- Cell 2: Fit BG/NBD ---
t0 = time.time()
print("\n[2/6] Fitting BG/NBD model...")

from lifetimes import BetaGeoFitter
from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix
from lifelines import KaplanMeierFitter, CoxPHFitter

# Subsample for fitting
rfm_fit = rfm_valid.sample(min(200_000, len(rfm_valid)), random_state=42).copy()

bgf = BetaGeoFitter(penalizer_coef=0.1)
bgf.fit(rfm_fit['frequency'], rfm_fit['recency'], rfm_fit['T'])

print("  BG/NBD Parameters:")
print(bgf.summary.to_string())

# Compute P(alive) for all valid customers
rfm_valid['p_alive'] = bgf.conditional_probability_alive(
    rfm_valid['frequency'], rfm_valid['recency'], rfm_valid['T']
)
rfm_valid['predicted_purchases_1'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    1, rfm_valid['frequency'], rfm_valid['recency'], rfm_valid['T']
)

# Check if BG/NBD produces meaningful variation
p_alive_std = rfm_valid['p_alive'].std()
print(f"\n  P(alive) distribution:")
print(f"    Mean:   {rfm_valid['p_alive'].mean():.3f}")
print(f"    Std:    {p_alive_std:.3f}")
print(f"    Min:    {rfm_valid['p_alive'].min():.3f}")
print(f"    < 0.5:  {(rfm_valid['p_alive'] < 0.5).sum():,} ({(rfm_valid['p_alive'] < 0.5).mean():.1%})")
print(f"    > 0.9:  {(rfm_valid['p_alive'] > 0.9).sum():,} ({(rfm_valid['p_alive'] > 0.9).mean():.1%})")

# Validate: P(alive) should correlate with actual churn
from sklearn.metrics import roc_auc_score
# p_alive predicts NOT churning, so flip for churn prediction
if p_alive_std > 0.01:
    auc_bgnbd = roc_auc_score(rfm_valid['churn'], 1 - rfm_valid['p_alive'])
    print(f"    AUC-ROC (P(alive) vs actual churn): {auc_bgnbd:.4f}")
else:
    auc_bgnbd = 0.5
    print(f"    P(alive) has near-zero variance — BG/NBD not discriminative for this data")

print(f"  Predicted purchases (1 period): mean={rfm_valid['predicted_purchases_1'].mean():.3f}")

# Plot matrices
fig = plt.figure(figsize=(18, 7))
try:
    plt.subplot(1, 2, 1)
    plot_frequency_recency_matrix(bgf, T=1)
    plt.title('Expected Purchases in Next Period', fontsize=13)
    plt.subplot(1, 2, 2)
    plot_probability_alive_matrix(bgf)
    plt.title('Probability Customer is Still "Alive"', fontsize=13)
except Exception as e:
    print(f"  Warning: Could not plot BG/NBD matrices: {e}")
    # Fallback: histogram of P(alive) by churn
    plt.subplot(1, 2, 1)
    plt.hist(rfm_valid.loc[rfm_valid['churn']==0, 'p_alive'], bins=50, alpha=0.5,
             color='#27AE60', label='Not Churned', density=True)
    plt.hist(rfm_valid.loc[rfm_valid['churn']==1, 'p_alive'], bins=50, alpha=0.5,
             color='#E74C3C', label='Churned', density=True)
    plt.title('P(alive) by Actual Churn', fontsize=13)
    plt.xlabel('P(alive)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist(rfm_valid['predicted_purchases_1'], bins=50, color='#3498DB', alpha=0.7)
    plt.title('Predicted Purchases (1 period)', fontsize=13)
    plt.xlabel('Expected purchases')

plt.suptitle('BG/NBD Model — Frequency/Recency Analysis', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('charts/phase16_bgnbd_matrix.png', dpi=150, bbox_inches='tight')
plt.close('all')
print(f"  Saved: charts/phase16_bgnbd_matrix.png")
print(f"  ({time.time()-t0:.1f}s)")

# --- Cell 3: Kaplan-Meier Survival Curves ---
t0 = time.time()
print("\n[3/6] Kaplan-Meier survival curves by risk level...")

# Merge with risk scores
df_risk = pd.read_csv('Churn_RiskScore_2026_0316.csv')
score_col = 'churn_score' if 'churn_score' in df_risk.columns else 'churn_score_rf_multi'
rfm_with_risk = rfm_valid.merge(
    df_risk[['userNo', score_col, 'risk_level']].rename(columns={score_col: 'churn_score'}),
    on='userNo', how='inner'
)
rfm_with_risk = rfm_with_risk[
    ~rfm_with_risk['risk_level'].isin(['New Customer', 'new_customer'])
].copy()
print(f"  Merged: {len(rfm_with_risk):,} customers with risk scores")

# For Kaplan-Meier: use tenure as duration, actual churn as event
# This shows: given a customer has been observed for X rounds, what's the survival probability?
rfm_with_risk['km_duration'] = rfm_with_risk['T'].clip(lower=1)
rfm_with_risk['km_event'] = rfm_with_risk['churn']

fig, ax = plt.subplots(figsize=(12, 7))
kmf = KaplanMeierFitter()
colors_km = {'Low': '#27AE60', 'Medium': '#F39C12', 'High': '#E67E22', 'Critical': '#E74C3C'}
median_survivals = {}

for level in ['Low', 'Medium', 'High', 'Critical']:
    mask = rfm_with_risk['risk_level'] == level
    if mask.sum() < 10:
        continue
    kmf.fit(
        rfm_with_risk.loc[mask, 'km_duration'],
        event_observed=rfm_with_risk.loc[mask, 'km_event'],
        label=f'{level} (n={mask.sum():,})'
    )
    kmf.plot_survival_function(ax=ax, color=colors_km[level], linewidth=2)
    median_survivals[level] = kmf.median_survival_time_

ax.set_title('Kaplan-Meier Survival Curves by Risk Level', fontsize=14)
ax.set_xlabel('Rounds Since First Purchase', fontsize=12)
ax.set_ylabel('Survival Probability', fontsize=12)
ax.legend(fontsize=11, loc='lower left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('charts/phase16_kaplan_meier.png', dpi=150, bbox_inches='tight')
plt.close()

print("  Median Survival Time by Risk Level:")
for level, median in median_survivals.items():
    if np.isinf(median):
        print(f"    {level:10s}: > {n_feat_rounds} งวด (> 50% survive)")
    else:
        print(f"    {level:10s}: {median:.1f} งวด")
print(f"  Saved: charts/phase16_kaplan_meier.png")
print(f"  ({time.time()-t0:.1f}s)")

# --- Cell 4: Cox Proportional Hazards ---
t0 = time.time()
print("\n[4/6] Cox Proportional Hazards...")

# Compute features for Cox PH
print(f"  Computing features for {len(rfm_with_risk):,} customers...")

# Use a subset of the Churn_2026_0216_0301 data with features
df_cox_file = 'Churn_2026_0216_0301.csv'
df_cox = pd.read_csv(df_cox_file, low_memory=False)
item_cols_cox = [c for c in df_cox.columns if c.startswith('item')]
feat_item_cols_cox = item_cols_cox[:-1]
mat_cox = df_cox[item_cols_cox].apply(pd.to_numeric, errors='coerce')

first_reg_cox = mat_cox.notna().values.argmax(axis=1)
tenure_cox = len(item_cols_cox) - first_reg_cox
exclude_cox = tenure_cox < MIN_TENURE
df_cox_elig = df_cox[~exclude_cox].copy()
mat_cox_elig = mat_cox[~exclude_cox].copy()
df_cox_elig['churn'] = (df_cox_elig[item_cols_cox[-1]] == 0).astype(int)

print(f"  Eligible: {len(df_cox_elig):,}, churn rate: {df_cox_elig['churn'].mean():.1%}")

feats_cox = engineer_features_v2(df_cox_elig, mat_cox_elig, feat_item_cols_cox)
feats_cox = feats_cox.fillna(0)

# Select features carefully — avoid collinearity
cox_features = [
    'rounds_since_last_purchase', 'current_zero_streak',
    'active_last_3', 'trend_slope',
    'purchase_frequency_ratio', 'ewm_recent',
    'avg_gap', 'gini_coefficient',
    'late_momentum', 'purchase_entropy',
]

cox_data = feats_cox[cox_features].copy()
cox_data['duration'] = feats_cox['tenure_rounds'].clip(lower=1)
cox_data['event'] = df_cox_elig['churn'].values

# Remove infinities and extreme outliers
for col in cox_features:
    cox_data[col] = cox_data[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    q99 = cox_data[col].quantile(0.99)
    q01 = cox_data[col].quantile(0.01)
    cox_data[col] = cox_data[col].clip(q01, q99)

# Subsample
if len(cox_data) > 80_000:
    cox_data = cox_data.sample(80_000, random_state=42)
    print(f"  Subsampled to {len(cox_data):,}")

# Standardize
from sklearn.preprocessing import StandardScaler
scaler_cox = StandardScaler()
cox_data[cox_features] = scaler_cox.fit_transform(cox_data[cox_features])

# Check for NaN/inf
assert cox_data.isna().sum().sum() == 0, f"NaN found: {cox_data.isna().sum()}"

# Try fitting with increasing penalizer until convergence
cph = None
for pen in [0.5, 1.0, 5.0, 10.0]:
    try:
        _cph = CoxPHFitter(penalizer=pen)
        _cph.fit(cox_data, duration_col='duration', event_col='event', )
        cph = _cph
        print(f"  Converged with penalizer={pen}")
        break
    except Exception as e:
        print(f"  penalizer={pen} failed: {str(e)[:60]}")

if cph is None:
    # Fallback: use WeibullAFTFitter (more robust than Cox for tied durations)
    print("  Trying WeibullAFT as fallback...")
    try:
        from lifelines import WeibullAFTFitter
        simple_features = ['rounds_since_last_purchase', 'active_last_3',
                           'purchase_frequency_ratio', 'avg_gap', 'trend_slope']
        aft_data = cox_data[simple_features + ['duration', 'event']].copy()
        aft = WeibullAFTFitter(penalizer=1.0)
        aft.fit(aft_data, duration_col='duration', event_col='event')
        print("  WeibullAFT converged!")
        print(aft.summary.to_string())
        cph = aft  # use aft for plotting
        cox_features = simple_features
    except Exception as e2:
        print(f"  WeibullAFT also failed: {str(e2)[:80]}")
        print("  NOTE: Cox PH / AFT not suitable for this data structure")
        print("        (tied durations, panel data → use discrete-time models instead)")
        cph = None

if cph is not None:
    print("\n  Survival Model Summary:")
    try:
        if hasattr(cph, 'summary'):
            summary = cph.summary
            if 'exp(coef)' in summary.columns:
                summary_show = summary[['exp(coef)', 'p']].sort_values('exp(coef)', ascending=False)
                for feat, row in summary_show.iterrows():
                    sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else ''
                    direction = 'เร่ง Churn' if row['exp(coef)'] > 1 else 'ชะลอ Churn'
                    print(f"    {feat:35s} HR={row['exp(coef)']:7.3f} {sig:4s} ({direction})")
            else:
                print(summary.to_string())

        fig, ax = plt.subplots(figsize=(10, 8))
        cph.plot(ax=ax)
        ax.set_title('Survival Model — Feature Effects', fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('charts/phase16_cox_hazard.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: charts/phase16_cox_hazard.png")
    except Exception as e:
        print(f"  Plotting failed: {e}")
else:
    # Create placeholder chart explaining why Cox PH didn't work
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'Cox PH / Weibull AFT did not converge\n\n'
            'Reason: Panel data with tied durations\n'
            'is not well-suited for traditional survival analysis.\n\n'
            'Alternative: Use discrete-time hazard models\n'
            'or the existing RF classifier.',
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title('Phase 16: Survival Model (Not Applicable)', fontsize=13)
    ax.axis('off')
    plt.savefig('charts/phase16_cox_hazard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved placeholder: charts/phase16_cox_hazard.png")
print(f"  ({time.time()-t0:.1f}s)")

# --- Cell 5: BG/NBD as Feature + Ensemble ---
t0 = time.time()
print("\n[5/6] BG/NBD features + RF ensemble...")

v3_artifact = joblib.load('models/churn_v3_extended.joblib')
v3_model = v3_artifact['model']
v3_feature_cols = v3_artifact['feature_cols']

all_25_files = [
    'Churn_2025_0216_0301.csv', 'Churn_2025_0301_0316.csv',
    'Churn_2025_0316_0401.csv', 'Churn_2025_0401_0416.csv',
    'Churn_2025_0416_0502.csv', 'Churn_2025_0502_0516.csv',
    'Churn_2025_0516_0601.csv', 'Churn_2025_0601_0616.csv',
    'Churn_2025_0616_0701.csv', 'Churn_2025_0701_0716.csv',
    'Churn_2025_0716_0801.csv', 'Churn_2025_0801_0816.csv',
    'Churn_2025_0816_0901.csv', 'Churn_2025_0901_0916.csv',
    'Churn_2025_0916_1001.csv', 'Churn_2025_1001_1016.csv',
    'Churn_2025_1016_1101.csv', 'Churn_2025_1101_1116.csv',
    'Churn_2025_1116_1201.csv', 'Churn_2025_1201_1216.csv',
    'Churn_2025_1216_0102.csv', 'Churn_2026_0102_0117.csv',
    'Churn_2026_0117_0201.csv',
    'Churn_2026_0201_0216.csv', 'Churn_2026_0216_0301.csv',
]
train_files = all_25_files[:20]
val_files = all_25_files[20:23]
test_files = all_25_files[23:25]


def process_file_with_bgnbd(fname, bgf_model):
    """Load a file, compute v2 features + BG/NBD features, return X, y."""
    df = pd.read_csv(fname, low_memory=False)
    item_cols = [c for c in df.columns if c.startswith('item')]
    feat_item_cols = item_cols[:-1]
    mat = df[item_cols].apply(pd.to_numeric, errors='coerce')
    mat_vals = mat.values.astype(float)
    reg = ~np.isnan(mat_vals[:, :-1])
    purch = np.where(reg, mat_vals[:, :-1] > 0, False).astype(bool)
    n_r = len(feat_item_cols)

    # Tenure filter
    first_reg = np.where(reg.any(axis=1), reg.argmax(axis=1), n_r)
    tenure = n_r - first_reg
    valid = tenure >= MIN_TENURE

    df_elig = df[valid].copy()
    mat_elig = mat[valid].copy()
    df_elig['churn'] = (df_elig[item_cols[-1]] == 0).astype(int)

    # v2 features
    feats = engineer_features_v2(df_elig, mat_elig, feat_item_cols)
    feats = feats.fillna(0)

    # BG/NBD features
    purch_v = purch[valid]
    freq = np.zeros(valid.sum())
    rec = np.zeros(valid.sum())
    T_fp = np.zeros(valid.sum())

    for i in range(valid.sum()):
        pr = np.where(purch_v[i])[0]
        if len(pr) >= 2:
            freq[i] = len(pr) - 1
            rec[i] = pr[-1] - pr[0]
        T_fp[i] = (n_r - 1 - pr[0]) if len(pr) >= 1 else 0

    p_alive = np.full(valid.sum(), 0.5)
    pred_p = np.full(valid.sum(), 0.0)
    t_mask = T_fp > 0
    if t_mask.sum() > 0:
        p_alive[t_mask] = bgf_model.conditional_probability_alive(
            freq[t_mask], rec[t_mask], T_fp[t_mask])
        pred_p[t_mask] = bgf_model.conditional_expected_number_of_purchases_up_to_time(
            1, freq[t_mask], rec[t_mask], T_fp[t_mask])

    feats['p_alive'] = p_alive
    feats['predicted_purchases_1'] = pred_p

    return feats, df_elig['churn'].values


# Use last 3 train files for speed
print("  Processing train files (last 3)...")
train_Xs, train_ys = [], []
for f in train_files[-3:]:
    X, y = process_file_with_bgnbd(f, bgf)
    train_Xs.append(X)
    train_ys.append(y)
    print(f"    {f}: {len(X):,} rows, churn={y.mean():.1%}")
X_train_surv = pd.concat(train_Xs, ignore_index=True)
y_train_surv = np.concatenate(train_ys)

print("  Processing test files...")
test_Xs, test_ys = [], []
for f in test_files:
    X, y = process_file_with_bgnbd(f, bgf)
    test_Xs.append(X)
    test_ys.append(y)
    print(f"    {f}: {len(X):,} rows, churn={y.mean():.1%}")
X_test_surv = pd.concat(test_Xs, ignore_index=True)
y_test_surv = np.concatenate(test_ys)

# Feature columns
feature_cols_surv = v3_feature_cols + ['p_alive', 'predicted_purchases_1']

# Subsample train
from sklearn.model_selection import train_test_split
if len(X_train_surv) > 1_000_000:
    X_train_surv, _, y_train_surv, _ = train_test_split(
        X_train_surv, y_train_surv, train_size=1_000_000,
        stratify=y_train_surv, random_state=42)
    print(f"  Subsampled train: {len(X_train_surv):,}")

from sklearn.ensemble import RandomForestClassifier

print(f"\n  Training RF with {len(feature_cols_surv)} features...")
rf_surv = RandomForestClassifier(
    n_estimators=300, max_depth=20, min_samples_leaf=50,
    class_weight='balanced', n_jobs=-1, random_state=42
)
rf_surv.fit(X_train_surv[feature_cols_surv].fillna(0), y_train_surv)

auc_test_surv = roc_auc_score(
    y_test_surv, rf_surv.predict_proba(X_test_surv[feature_cols_surv].fillna(0))[:, 1])
auc_test_v3_orig = roc_auc_score(
    y_test_surv, v3_model.predict_proba(X_test_surv[v3_feature_cols].fillna(0))[:, 1])

print(f"\n  {'='*50}")
print(f"  Model Comparison — AUC-ROC (Test)")
print(f"  {'='*50}")
print(f"    RF v3 (45 features):         {auc_test_v3_orig:.4f}")
print(f"    RF v3 + Survival (47 feat):  {auc_test_surv:.4f}")
print(f"    Improvement:                 {(auc_test_surv - auc_test_v3_orig)*100:+.2f}%")

# --- Cell 6: Combined Charts ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1. P(alive) by churn status
axes[0].hist(rfm_valid.loc[rfm_valid['churn']==0, 'p_alive'], bins=40, alpha=0.5,
             color='#27AE60', label='Not Churned', density=True)
axes[0].hist(rfm_valid.loc[rfm_valid['churn']==1, 'p_alive'], bins=40, alpha=0.5,
             color='#E74C3C', label='Churned', density=True)
axes[0].set_title('P(alive) by Actual Churn Status', fontsize=12)
axes[0].set_xlabel('P(alive)')
axes[0].set_ylabel('Density')
axes[0].legend()

# 2. Feature importance
importances = pd.Series(rf_surv.feature_importances_, index=feature_cols_surv)
top15 = importances.nlargest(15)
colors_bar = ['#E74C3C' if f in ['p_alive', 'predicted_purchases_1'] else '#3498DB'
              for f in top15.index]
axes[1].barh(range(len(top15)), top15.values, color=colors_bar)
axes[1].set_yticks(range(len(top15)))
axes[1].set_yticklabels(top15.index, fontsize=9)
axes[1].set_title('Top 15 Features (red = BG/NBD)', fontsize=12)
axes[1].invert_yaxis()

# 3. AUC comparison
models_comp = ['RF v3\n(45 feat)', 'RF v3+Survival\n(47 feat)']
aucs_comp = [auc_test_v3_orig, auc_test_surv]
bars = axes[2].bar(models_comp, aucs_comp, color=['#95A5A6', '#27AE60'], width=0.5)
axes[2].set_ylim(min(aucs_comp) - 0.01, max(aucs_comp) + 0.01)
axes[2].set_title('AUC-ROC Comparison', fontsize=12)
for bar, auc in zip(bars, aucs_comp):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{auc:.4f}', ha='center', fontsize=11, fontweight='bold')

plt.suptitle('Phase 16: Survival Analysis Results', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('charts/phase16_survival.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: charts/phase16_survival.png")
print(f"  ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════════════════════
# Phase 17: LSTM Sequence Model
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Phase 17: LSTM Sequence Model")
print("=" * 60)

t0 = time.time()
print("\n[1/4] Installing PyTorch + building sequences...")

import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', '--quiet'],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
print(f"  PyTorch {torch.__version__} (CPU)")

lstm_train_files = train_files[-5:]
lstm_val_files = val_files[:1]
lstm_test_files = test_files[:1]


def build_sequences(file_list, max_customers=100_000):
    """Convert item matrix → padded sequences + churn labels."""
    all_seqs, all_lengths, all_labels = [], [], []

    for fname in file_list:
        df = pd.read_csv(fname, low_memory=False)
        item_cols = [c for c in df.columns if c.startswith('item')]
        mat = df[item_cols].apply(pd.to_numeric, errors='coerce')
        mat_vals = mat.values.astype(float)
        reg = ~np.isnan(mat_vals)

        first_reg = np.where(reg.any(axis=1), reg.argmax(axis=1), len(item_cols))
        tenure = len(item_cols) - first_reg
        valid = tenure >= MIN_TENURE

        mat_valid = mat_vals[valid]
        reg_valid = reg[valid]
        labels = (mat_valid[:, -1] == 0).astype(np.float32)

        for i in range(len(mat_valid)):
            reg_start = np.argmax(reg_valid[i])
            seq = mat_valid[i, reg_start:-1]  # exclude target
            seq = np.nan_to_num(seq, nan=0.0)
            seq = np.log1p(seq).astype(np.float32)
            all_seqs.append(torch.tensor(seq).unsqueeze(-1))
            all_lengths.append(len(seq))
            all_labels.append(labels[i])

        print(f"    {fname}: {valid.sum():,} customers, churn={labels.mean():.1%}")

        if len(all_seqs) >= max_customers:
            all_seqs = all_seqs[:max_customers]
            all_lengths = all_lengths[:max_customers]
            all_labels = all_labels[:max_customers]
            break

    padded = pad_sequence(all_seqs, batch_first=True, padding_value=0.0)
    lengths = torch.tensor(all_lengths, dtype=torch.long)
    labels_t = torch.tensor(all_labels, dtype=torch.float32)
    return padded, lengths, labels_t


print("\n  Building train sequences...")
X_seq_train, len_train, y_seq_train = build_sequences(lstm_train_files, max_customers=100_000)
print(f"  Train: {X_seq_train.shape}, churn={y_seq_train.mean():.1%}")

print("  Building val sequences...")
X_seq_val, len_val, y_seq_val = build_sequences(lstm_val_files, max_customers=50_000)
print(f"  Val:   {X_seq_val.shape}, churn={y_seq_val.mean():.1%}")

print("  Building test sequences...")
X_seq_test, len_test, y_seq_test = build_sequences(lstm_test_files, max_customers=50_000)
print(f"  Test:  {X_seq_test.shape}, churn={y_seq_test.mean():.1%}")
print(f"  ({time.time()-t0:.1f}s)")


class ChurnLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(hidden).squeeze(-1)


class ChurnDataset(Dataset):
    def __init__(self, sequences, lengths, labels):
        self.sequences = sequences
        self.lengths = lengths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx], self.labels[idx]


# --- Training ---
t0 = time.time()
print("\n[2/4] Training LSTM...")

BATCH_SIZE = 512
EPOCHS = 20
LR = 0.001
PATIENCE = 4

model_lstm = ChurnLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.3)
total_params = sum(p.numel() for p in model_lstm.parameters())
print(f"  Parameters: {total_params:,}")

train_loader = DataLoader(ChurnDataset(X_seq_train, len_train, y_seq_train),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ChurnDataset(X_seq_val, len_val, y_seq_val),
                        batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(ChurnDataset(X_seq_test, len_test, y_seq_test),
                         batch_size=BATCH_SIZE, shuffle=False)

pos_weight = torch.tensor([(y_seq_train == 0).sum() / max((y_seq_train == 1).sum(), 1)])
print(f"  pos_weight: {pos_weight.item():.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

best_val_auc = 0
patience_counter = 0
train_losses, val_aucs = [], []
best_state = None

for epoch in range(1, EPOCHS + 1):
    model_lstm.train()
    epoch_loss, n_batches = 0, 0
    for seqs, lens, labels in train_loader:
        optimizer.zero_grad()
        logits = model_lstm(seqs, lens)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_lstm.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    train_losses.append(avg_loss)

    model_lstm.eval()
    val_probs, val_labels = [], []
    with torch.no_grad():
        for seqs, lens, labels in val_loader:
            probs = torch.sigmoid(model_lstm(seqs, lens))
            val_probs.append(probs.numpy())
            val_labels.append(labels.numpy())

    val_probs = np.concatenate(val_probs)
    val_labels = np.concatenate(val_labels)
    val_auc = roc_auc_score(val_labels, val_probs)
    val_aucs.append(val_auc)
    scheduler.step(-val_auc)

    print(f"  Epoch {epoch:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        best_state = {k: v.clone() for k, v in model_lstm.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

model_lstm.load_state_dict(best_state)
print(f"  Best Val AUC: {best_val_auc:.4f}")
print(f"  ({time.time()-t0:.1f}s)")

# --- Evaluation ---
t0 = time.time()
print("\n[3/4] Evaluating LSTM...")

from sklearn.metrics import average_precision_score, f1_score, roc_curve

model_lstm.eval()
test_probs_lstm, test_labels_lstm = [], []
with torch.no_grad():
    for seqs, lens, labels in test_loader:
        probs = torch.sigmoid(model_lstm(seqs, lens))
        test_probs_lstm.append(probs.numpy())
        test_labels_lstm.append(labels.numpy())

test_probs_lstm = np.concatenate(test_probs_lstm)
test_labels_lstm = np.concatenate(test_labels_lstm)
test_preds_lstm = (test_probs_lstm >= 0.5).astype(int)

auc_roc_lstm = roc_auc_score(test_labels_lstm, test_probs_lstm)
auc_pr_lstm = average_precision_score(test_labels_lstm, test_probs_lstm)
f1_lstm = f1_score(test_labels_lstm, test_preds_lstm)

print(f"\n  {'='*60}")
print(f"  Final Comparison — All Models")
print(f"  {'='*60}")
print(f"    {'Model':<30} {'AUC-ROC':>10} {'AUC-PR':>10} {'F1':>10}")
print(f"    {'-'*60}")
print(f"    {'RF v3 (45 feat)':<30} {auc_test_v3_orig:>10.4f} {'—':>10} {'—':>10}")
print(f"    {'RF v3 + Survival (47 feat)':<30} {auc_test_surv:>10.4f} {'—':>10} {'—':>10}")
print(f"    {'LSTM (sequences)':<30} {auc_roc_lstm:>10.4f} {auc_pr_lstm:>10.4f} {f1_lstm:>10.4f}")

competitive = auc_roc_lstm >= 0.75
print(f"\n    LSTM competitive (AUC > 0.75): {'Yes' if competitive else 'No'}")

# --- Charts ---
print("\n[4/4] Generating charts...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

ax1 = axes[0]
ax1_twin = ax1.twinx()
ax1.plot(range(1, len(train_losses)+1), train_losses, 'b-', linewidth=2, label='Train Loss')
ax1_twin.plot(range(1, len(val_aucs)+1), val_aucs, 'r-', linewidth=2, label='Val AUC')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='blue')
ax1_twin.set_ylabel('AUC-ROC', color='red')
ax1.set_title('LSTM Training Curves', fontsize=12)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

fpr_lstm, tpr_lstm, _ = roc_curve(test_labels_lstm, test_probs_lstm)
axes[1].plot(fpr_lstm, tpr_lstm, color='#E74C3C', linewidth=2,
             label=f'LSTM (AUC={auc_roc_lstm:.4f})')
axes[1].plot([0, 1], [0, 1], '--', color='#ccc')
axes[1].set_title('ROC Curve — LSTM', fontsize=12)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

axes[2].hist(test_probs_lstm[test_labels_lstm == 0], bins=50, alpha=0.5,
             color='#27AE60', label='Not Churned', density=True)
axes[2].hist(test_probs_lstm[test_labels_lstm == 1], bins=50, alpha=0.5,
             color='#E74C3C', label='Churned', density=True)
axes[2].set_title('LSTM Score Distribution', fontsize=12)
axes[2].set_xlabel('Churn Probability')
axes[2].set_ylabel('Density')
axes[2].legend()

plt.suptitle('Phase 17: LSTM Sequence Model Results', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('charts/phase17_lstm.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: charts/phase17_lstm.png")

if competitive:
    print("\n  LSTM is competitive — consider RF + LSTM ensemble for production")
else:
    print("\n  LSTM underperforms — RF v3 remains primary")
    print("  Next steps: multi-feature sequences (when order data arrives)")

print(f"  ({time.time()-t0:.1f}s)")

print("\n" + "=" * 60)
print("DONE — Phase 16 + 17 Complete!")
print("=" * 60)
