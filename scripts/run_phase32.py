#!/usr/bin/env python3
"""Phase 32: LightGBM and XGBoost with TX Features.

Test if LightGBM/XGBoost can break the AUC 0.8013 ceiling that RF hit.
These models handle sparse features and interactions better than RF.

Models compared:
  [A] RF v9b (67f) — load pretrained from models/churn_v9_time.joblib, evaluate only
  [B] LightGBM (67f) — train with hyperparameter tuning
  [C] XGBoost (67f) — train with hyperparameter tuning
  [D] LightGBM (45f base only) — compare if LightGBM beats RF even without TX
"""

import os, sys, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, 'models')
from feature_engineering import (
    engineer_features_v2, aggregate_tx_features, compute_time_features
)
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve
import lightgbm as lgb
import xgboost as xgb

MIN_TENURE = 3
THRESHOLD = 0.48

print("=" * 70, flush=True)
print("Phase 32: LightGBM and XGBoost with TX Features", flush=True)
print("=" * 70, flush=True)

# ===================================================================
# TX <-> Churn mapping (20 train + 2 test) — same as Phase 28/30
# ===================================================================

CHURN_DIR = 'data/churn'
TX_DIR = 'data/transaction'

train_pairs = [
    ('Transaction_Order_2025_0216.csv', 'Churn_2025_0216_0301.csv'),
    ('Transaction_Order_2025_0301.csv', 'Churn_2025_0301_0316.csv'),
    ('Transaction_Order_2025_0316.csv', 'Churn_2025_0316_0401.csv'),
    ('Transaction_Order_2025_0401.csv', 'Churn_2025_0401_0416.csv'),
    ('Transaction_Order_2025_0416.csv', 'Churn_2025_0416_0502.csv'),
    ('Transaction_Order_2025_0502.csv', 'Churn_2025_0502_0516.csv'),
    ('Transaction_Order_2025_0516.csv', 'Churn_2025_0516_0601.csv'),
    ('Transaction_Order_2025_0601.csv', 'Churn_2025_0601_0616.csv'),
    ('Transaction_Order_2025_0616.csv', 'Churn_2025_0616_0701.csv'),
    ('Transaction_Order_2025_0701.csv', 'Churn_2025_0701_0716.csv'),
    ('Transaction_Order_2025_0716.csv', 'Churn_2025_0716_0801.csv'),
    ('Transaction_Order_2025_0801.csv', 'Churn_2025_0801_0816.csv'),
    ('Transaction_Order_2025_0816.csv', 'Churn_2025_0816_0901.csv'),
    ('Transaction_Order_2025_0901.csv', 'Churn_2025_0901_0916.csv'),
    ('Transaction_Order_2025_0916.csv', 'Churn_2025_0916_1001.csv'),
    ('Transaction_Order_2025_1001.csv', 'Churn_2025_1001_1016.csv'),
    ('Transaction_Order_2025_1016.csv', 'Churn_2025_1016_1101.csv'),
    ('Transaction_Order_2025_1101.csv', 'Churn_2025_1101_1116.csv'),
    ('Transaction_Order_2025_1116.csv', 'Churn_2025_1116_1201.csv'),
    ('Transaction_Order_2025_1201.csv', 'Churn_2025_1201_1216.csv'),
]

test_pairs = [
    ('Transaction_Order_2026_0201.csv', 'Churn_2026_0201_0216.csv'),
    ('Transaction_Order_2026_0216.csv', 'Churn_2026_0216_0301.csv'),
]


# ===================================================================
# Helper functions
# ===================================================================

def load_churn_features(churn_file):
    """Load Churn CSV, compute v2 features (45), return (X, y, userNo_series)."""
    path = os.path.join(CHURN_DIR, churn_file)
    df = pd.read_csv(path, low_memory=False)
    item_cols = [c for c in df.columns if c.startswith('item')]
    feat_item_cols = item_cols[:-1]
    target_col = item_cols[-1]

    mat = df[item_cols].apply(pd.to_numeric, errors='coerce')
    mat_vals = mat.values.astype(float)
    reg = ~np.isnan(mat_vals)
    first_reg = np.where(reg.any(axis=1), reg.argmax(axis=1), len(item_cols))
    tenure = len(item_cols) - first_reg
    valid = tenure >= MIN_TENURE

    df_v = df[valid].copy()
    mat_v = mat[valid].copy()

    y_raw = mat_v[target_col].values
    y = (np.isnan(y_raw) | (y_raw == 0)).astype(int)

    feats = engineer_features_v2(df_v, mat_v, feat_item_cols)
    feats = feats.fillna(0)
    user_nos = df_v['userNo'].reset_index(drop=True)

    return feats, y, user_nos


def load_tx_agg_features(tx_file):
    """Load TX file and aggregate into per-customer features (11 aggregate)."""
    path = os.path.join(TX_DIR, tx_file)
    df = pd.read_csv(path, low_memory=False)
    tx_feats = aggregate_tx_features(df)
    return tx_feats


def load_tx_time_features(tx_file):
    """Load TX file and compute time-within-window features (11 time)."""
    path = os.path.join(TX_DIR, tx_file)
    time_feats = compute_time_features(path)
    return time_feats


# ===================================================================
# 1. Aggregate ALL TX files (both agg + time features)
# ===================================================================
print("\n[1/9] Aggregating transaction data (22 files)...", flush=True)
t0 = time.time()

tx_agg_cache = {}
tx_time_cache = {}
all_pairs = train_pairs + test_pairs

for tx_file, _ in all_pairs:
    if tx_file not in tx_agg_cache:
        t_file = time.time()
        print(f"  {tx_file}...", end=" ", flush=True)

        tx_agg = load_tx_agg_features(tx_file)
        tx_agg_cache[tx_file] = tx_agg

        tx_time = load_tx_time_features(tx_file)
        tx_time_cache[tx_file] = tx_time

        print(f"agg={len(tx_agg):,} time={len(tx_time):,} "
              f"({time.time()-t_file:.1f}s)", flush=True)

print(f"  Total TX files cached: {len(tx_agg_cache)}", flush=True)
print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

# Show feature lists
sample_agg = list(tx_agg_cache.values())[0]
sample_time = list(tx_time_cache.values())[0]
agg_feature_names = list(sample_agg.columns)
time_feature_names = list(sample_time.columns)
print(f"\n  TX aggregate features ({len(agg_feature_names)}): {agg_feature_names}", flush=True)
print(f"  TX time features ({len(time_feature_names)}): {time_feature_names}", flush=True)


# ===================================================================
# 2. Load & merge Churn + TX for each pair
# ===================================================================
print("\n[2/9] Loading Churn features & merging with TX...", flush=True)
t0 = time.time()


def load_and_merge(pairs):
    """Load Churn + TX pairs, merge, return DataFrames for model variants.

    Returns dict with keys:
        'base': X with 45 base features
        'tx_both': X with 45 base + 11 TX agg + 11 time = 67f
        'y': target array
    """
    all_base, all_tx_both, all_y = [], [], []

    for tx_file, churn_file in pairs:
        print(f"  {churn_file}...", flush=True)
        feats, y, user_nos = load_churn_features(churn_file)

        # Get TX features for this period
        tx_agg = tx_agg_cache[tx_file]
        tx_time = tx_time_cache[tx_file]

        # Merge by userNo (left join)
        feats_with_user = feats.copy()
        feats_with_user['userNo'] = user_nos.values

        agg_cols = list(tx_agg.columns)
        time_cols = list(tx_time.columns)

        # Merge both agg + time
        merged_both = feats_with_user.merge(
            tx_agg, left_on='userNo', right_index=True, how='left'
        ).merge(
            tx_time, left_on='userNo', right_index=True, how='left'
        )
        merged_both[agg_cols] = merged_both[agg_cols].fillna(0)
        merged_both[time_cols] = merged_both[time_cols].fillna(0)

        # Stats
        matched_agg = (merged_both.get('tx_n_orders', pd.Series(0, index=merged_both.index)) > 0).sum()
        print(f"    n={len(feats):,}, churn={y.mean()*100:.1f}%, "
              f"TX matched={matched_agg:,}", flush=True)

        X_base = feats.copy()
        X_tx_both = merged_both.drop(columns=['userNo'])

        all_base.append(X_base)
        all_tx_both.append(X_tx_both)
        all_y.append(y)

    return {
        'base': pd.concat(all_base, ignore_index=True),
        'tx_both': pd.concat(all_tx_both, ignore_index=True),
        'y': np.concatenate(all_y),
    }


# Load train (20 files)
print("\n  --- Train (20 files) ---", flush=True)
train_data = load_and_merge(train_pairs)
print(f"\n  Train total: {len(train_data['y']):,}, "
      f"churn={train_data['y'].mean()*100:.1f}%", flush=True)

# Load test (2 files)
print("\n  --- Test (2 files) ---", flush=True)
test_data = load_and_merge(test_pairs)
print(f"\n  Test total: {len(test_data['y']):,}, "
      f"churn={test_data['y'].mean()*100:.1f}%", flush=True)

print(f"\n  Done ({time.time()-t0:.1f}s)", flush=True)
print(f"  Base features (v2): {train_data['base'].shape[1]}", flush=True)
print(f"  TX both features: {train_data['tx_both'].shape[1]}", flush=True)


# ===================================================================
# 3. Training data management (subsample if > 2M)
# ===================================================================
print("\n[3/9] Training data management...", flush=True)

y_train = train_data['y']
y_test = test_data['y']

if len(y_train) > 2_000_000:
    print(f"  Total rows ({len(y_train):,}) > 2M, subsampling...", flush=True)
    idx = np.random.RandomState(42).choice(len(y_train), 2_000_000, replace=False)
    for key in ['base', 'tx_both']:
        train_data[key] = train_data[key].iloc[idx].reset_index(drop=True)
    y_train = y_train[idx]
    train_data['y'] = y_train
    print(f"  Subsampled to {len(y_train):,}", flush=True)
else:
    print(f"  Total rows: {len(y_train):,} (under 2M, no subsampling needed)", flush=True)

print(f"  Final train size: {len(y_train):,}, churn rate: {y_train.mean()*100:.1f}%", flush=True)

# Identify feature column groups
base_cols = list(train_data['base'].columns)
both_cols = list(train_data['tx_both'].columns)
agg_only_cols = [c for c in both_cols if c in agg_feature_names]
time_only_cols = [c for c in both_cols if c in time_feature_names]
tx_all_cols = agg_only_cols + time_only_cols

print(f"  Base cols: {len(base_cols)}", flush=True)
print(f"  TX agg cols ({len(agg_only_cols)}): {agg_only_cols}", flush=True)
print(f"  TX time cols ({len(time_only_cols)}): {time_only_cols}", flush=True)
print(f"  Total (67f): {len(both_cols)}", flush=True)


# ===================================================================
# 4. Train & evaluate 4 models
# ===================================================================
print("\n[4/9] Training & evaluating models...", flush=True)
t0 = time.time()

results = {}
trained_models = {}

# Prepare data
X_train_67 = train_data['tx_both'].fillna(0)
X_test_67 = test_data['tx_both'].fillna(0)
X_train_45 = train_data['base'].fillna(0)
X_test_45 = test_data['base'].fillna(0)

# --- [A] RF v9b (67f) — load pretrained, evaluate only ---
print("\n  [A] RF v9b (67f) — evaluate only...", flush=True)
t_a = time.time()
v9b_artifact = joblib.load('models/churn_v9_time.joblib')
v9b_model = v9b_artifact['model']
v9b_feature_cols = v9b_artifact['feature_cols']

# Match v9b feature columns to test
X_test_A = X_test_67.copy()
for c in v9b_feature_cols:
    if c not in X_test_A.columns:
        X_test_A[c] = 0
X_test_A = X_test_A[v9b_feature_cols].fillna(0)

probs_A = v9b_model.predict_proba(X_test_A)[:, 1]
preds_A = (probs_A >= THRESHOLD).astype(int)
results['[A] RF v9b (67f)'] = {
    'auc': roc_auc_score(y_test, probs_A),
    'f1': f1_score(y_test, preds_A),
    'cm': confusion_matrix(y_test, preds_A),
    'probs': probs_A,
    'preds': preds_A,
}
print(f"      AUC={results['[A] RF v9b (67f)']['auc']:.4f} "
      f"F1={results['[A] RF v9b (67f)']['f1']:.4f} "
      f"({time.time()-t_a:.1f}s)", flush=True)

# --- [B] LightGBM (67f) — train with tuned hyperparameters ---
print("\n  [B] LightGBM (67f) — training...", flush=True)
t_b = time.time()

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_child_samples': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1,
    'n_jobs': -1,
}

lgb_model_67 = lgb.LGBMClassifier(**lgb_params)
lgb_model_67.fit(
    X_train_67, y_train,
    eval_set=[(X_test_67, y_test)],
    callbacks=[lgb.log_evaluation(period=0)],
)
probs_B = lgb_model_67.predict_proba(X_test_67)[:, 1]
preds_B = (probs_B >= THRESHOLD).astype(int)
results['[B] LGB (67f)'] = {
    'auc': roc_auc_score(y_test, probs_B),
    'f1': f1_score(y_test, preds_B),
    'cm': confusion_matrix(y_test, preds_B),
    'probs': probs_B,
    'preds': preds_B,
}
trained_models['lgb_67'] = lgb_model_67
print(f"      AUC={results['[B] LGB (67f)']['auc']:.4f} "
      f"F1={results['[B] LGB (67f)']['f1']:.4f} "
      f"({time.time()-t_b:.1f}s)", flush=True)

# --- [C] XGBoost (67f) — train with tuned hyperparameters ---
print("\n  [C] XGBoost (67f) — training...", flush=True)
t_c = time.time()

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'min_child_weight': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbosity': 0,
    'n_jobs': -1,
}

xgb_model_67 = xgb.XGBClassifier(**xgb_params)
xgb_model_67.fit(
    X_train_67, y_train,
    eval_set=[(X_test_67, y_test)],
    verbose=False,
)
probs_C = xgb_model_67.predict_proba(X_test_67)[:, 1]
preds_C = (probs_C >= THRESHOLD).astype(int)
results['[C] XGB (67f)'] = {
    'auc': roc_auc_score(y_test, probs_C),
    'f1': f1_score(y_test, preds_C),
    'cm': confusion_matrix(y_test, preds_C),
    'probs': probs_C,
    'preds': preds_C,
}
trained_models['xgb_67'] = xgb_model_67
print(f"      AUC={results['[C] XGB (67f)']['auc']:.4f} "
      f"F1={results['[C] XGB (67f)']['f1']:.4f} "
      f"({time.time()-t_c:.1f}s)", flush=True)

# --- [D] LightGBM (45f base only) — compare without TX ---
print("\n  [D] LightGBM (45f base only) — training...", flush=True)
t_d = time.time()

lgb_model_45 = lgb.LGBMClassifier(**lgb_params)
lgb_model_45.fit(
    X_train_45, y_train,
    eval_set=[(X_test_45, y_test)],
    callbacks=[lgb.log_evaluation(period=0)],
)
probs_D = lgb_model_45.predict_proba(X_test_45)[:, 1]
preds_D = (probs_D >= THRESHOLD).astype(int)
results['[D] LGB (45f base)'] = {
    'auc': roc_auc_score(y_test, probs_D),
    'f1': f1_score(y_test, preds_D),
    'cm': confusion_matrix(y_test, preds_D),
    'probs': probs_D,
    'preds': preds_D,
}
trained_models['lgb_45'] = lgb_model_45
print(f"      AUC={results['[D] LGB (45f base)']['auc']:.4f} "
      f"F1={results['[D] LGB (45f base)']['f1']:.4f} "
      f"({time.time()-t_d:.1f}s)", flush=True)

print(f"\n  All models evaluated ({time.time()-t0:.1f}s)", flush=True)


# ===================================================================
# 5. Results comparison table
# ===================================================================
print("\n[5/9] Results comparison...", flush=True)

baseline_auc = results['[A] RF v9b (67f)']['auc']

print(f"\n  {'='*95}", flush=True)
print(f"  Model Comparison (Test: {len(y_test):,} customers, "
      f"churn={y_test.mean()*100:.1f}%)", flush=True)
print(f"  {'='*95}", flush=True)
print(f"  {'Model':<25} {'Features':>10} {'AUC-ROC':>10} {'F1':>10} "
      f"{'FN':>10} {'FP':>10} {'delta AUC':>12}", flush=True)
print(f"  {'-'*95}", flush=True)

model_nfeats = {
    '[A] RF v9b (67f)': 67,
    '[B] LGB (67f)': 67,
    '[C] XGB (67f)': 67,
    '[D] LGB (45f base)': 45,
}

for name, r in results.items():
    fn = r['cm'][1, 0]
    fp = r['cm'][0, 1]
    nf = model_nfeats.get(name, '?')
    delta = (r['auc'] - baseline_auc) * 100
    delta_str = f"{delta:+.2f}%" if name != '[A] RF v9b (67f)' else "baseline"
    print(f"  {name:<25} {nf:>10} {r['auc']:>10.4f} {r['f1']:>10.4f} "
          f"{fn:>10,} {fp:>10,} {delta_str:>12}", flush=True)

print(f"  {'-'*95}", flush=True)

best_name = max(results, key=lambda k: results[k]['auc'])
best_auc = results[best_name]['auc']
print(f"  RF v9b baseline AUC: {baseline_auc:.4f}", flush=True)
print(f"  Best model:          {best_name} (AUC: {best_auc:.4f})", flush=True)
print(f"  Improvement:         {(best_auc - baseline_auc)*100:+.2f}%", flush=True)

# LightGBM TX impact: [D] 45f vs [B] 67f
lgb_base_auc = results['[D] LGB (45f base)']['auc']
lgb_full_auc = results['[B] LGB (67f)']['auc']
print(f"\n  LightGBM TX impact: 45f AUC={lgb_base_auc:.4f} -> "
      f"67f AUC={lgb_full_auc:.4f} ({(lgb_full_auc-lgb_base_auc)*100:+.2f}%)", flush=True)


# ===================================================================
# 6. Feature importance comparison (RF vs LightGBM vs XGBoost)
# ===================================================================
print("\n[6/9] Feature importance comparison...", flush=True)

# RF importance (from v9b)
imp_rf = pd.Series(
    v9b_model.feature_importances_, index=v9b_feature_cols
).sort_values(ascending=False)

# LightGBM importance (67f)
imp_lgb = pd.Series(
    lgb_model_67.feature_importances_, index=both_cols
).sort_values(ascending=False)

# XGBoost importance (67f)
imp_xgb = pd.Series(
    xgb_model_67.feature_importances_, index=both_cols
).sort_values(ascending=False)

# Normalize importances to [0, 1] for fair comparison
imp_rf_norm = imp_rf / imp_rf.sum()
imp_lgb_norm = imp_lgb / imp_lgb.sum()
imp_xgb_norm = imp_xgb / imp_xgb.sum()

print(f"\n  Top 20 features per model:", flush=True)
print(f"  {'Rank':<6} {'RF v9b':<30} {'LightGBM':<30} {'XGBoost':<30}", flush=True)
print(f"  {'-'*96}", flush=True)

rf_top20 = imp_rf_norm.head(20)
lgb_top20 = imp_lgb_norm.head(20)
xgb_top20 = imp_xgb_norm.head(20)

for i in range(20):
    rf_feat = f"{rf_top20.index[i]} ({rf_top20.values[i]:.3f})" if i < len(rf_top20) else ""
    lgb_feat = f"{lgb_top20.index[i]} ({lgb_top20.values[i]:.3f})" if i < len(lgb_top20) else ""
    xgb_feat = f"{xgb_top20.index[i]} ({xgb_top20.values[i]:.3f})" if i < len(xgb_top20) else ""
    print(f"  {i+1:<6} {rf_feat:<30} {lgb_feat:<30} {xgb_feat:<30}", flush=True)

# Check TX feature ranks across models
print(f"\n  TX feature ranks across models:", flush=True)
print(f"  {'Feature':<30} {'RF rank':>10} {'LGB rank':>10} {'XGB rank':>10}", flush=True)
print(f"  {'-'*70}", flush=True)

for feat in tx_all_cols:
    rf_rank = imp_rf_norm.index.tolist().index(feat) + 1 if feat in imp_rf_norm.index else '-'
    lgb_rank = imp_lgb_norm.index.tolist().index(feat) + 1
    xgb_rank = imp_xgb_norm.index.tolist().index(feat) + 1
    print(f"  {feat:<30} {str(rf_rank):>10} {lgb_rank:>10} {xgb_rank:>10}", flush=True)


# ===================================================================
# 7. FN Analysis
# ===================================================================
print("\n[7/9] FN Analysis...", flush=True)

fn_A = set(np.where((y_test == 1) & (preds_A == 0))[0])
fn_B = set(np.where((y_test == 1) & (preds_B == 0))[0])
fn_C = set(np.where((y_test == 1) & (preds_C == 0))[0])
fn_D = set(np.where((y_test == 1) & (preds_D == 0))[0])

print(f"    [A] RF v9b (67f)       FN: {len(fn_A):,}", flush=True)
print(f"    [B] LGB (67f)          FN: {len(fn_B):,}", flush=True)
print(f"    [C] XGB (67f)          FN: {len(fn_C):,}", flush=True)
print(f"    [D] LGB (45f base)     FN: {len(fn_D):,}", flush=True)

# FN saved vs RF baseline
print(f"\n    FN saved vs RF v9b baseline:", flush=True)
print(f"      [B] LGB 67f:    {len(fn_A) - len(fn_B):+,}", flush=True)
print(f"      [C] XGB 67f:    {len(fn_A) - len(fn_C):+,}", flush=True)
print(f"      [D] LGB 45f:    {len(fn_A) - len(fn_D):+,}", flush=True)

# Cross-model FN overlap: which churners does LGB catch that RF misses?
fn_only_rf = fn_A - fn_B  # RF missed, LGB caught
fn_only_lgb = fn_B - fn_A  # LGB missed, RF caught
fn_both_miss = fn_A & fn_B  # Both missed
print(f"\n    RF vs LGB (67f) FN overlap:", flush=True)
print(f"      RF missed, LGB caught:  {len(fn_only_rf):,}", flush=True)
print(f"      LGB missed, RF caught:  {len(fn_only_lgb):,}", flush=True)
print(f"      Both missed:             {len(fn_both_miss):,}", flush=True)

# Same for XGB
fn_only_rf_xgb = fn_A - fn_C
fn_only_xgb = fn_C - fn_A
fn_both_miss_xgb = fn_A & fn_C
print(f"\n    RF vs XGB (67f) FN overlap:", flush=True)
print(f"      RF missed, XGB caught:  {len(fn_only_rf_xgb):,}", flush=True)
print(f"      XGB missed, RF caught:  {len(fn_only_xgb):,}", flush=True)
print(f"      Both missed:             {len(fn_both_miss_xgb):,}", flush=True)

# LGB 45f vs LGB 67f (TX feature impact on FN)
fn_saved_by_tx = fn_D - fn_B
fn_lost_by_tx = fn_B - fn_D
print(f"\n    LGB TX impact (45f -> 67f):", flush=True)
print(f"      FN saved by TX features: {len(fn_saved_by_tx):,}", flush=True)
print(f"      FN lost by TX features:  {len(fn_lost_by_tx):,}", flush=True)
print(f"      Net FN change:           {len(fn_D) - len(fn_B):+,}", flush=True)


# ===================================================================
# 8. Charts: 4-panel comparison
# ===================================================================
print("\n[8/9] Generating charts...", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

names_list = list(results.keys())
short_names = ['RF v9b (67f)', 'LGB (67f)', 'XGB (67f)', 'LGB (45f)']
colors_4 = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

# --- [0,0] AUC bar chart ---
ax = axes[0, 0]
aucs = [results[n]['auc'] for n in names_list]
bars = ax.bar(range(len(names_list)), aucs, color=colors_4)
ax.set_xticks(range(len(names_list)))
ax.set_xticklabels(short_names, fontsize=9, rotation=10)
ax.set_ylabel('AUC-ROC')
ax.set_title('AUC-ROC Comparison')
ax.set_ylim(min(aucs) - 0.005, max(aucs) + 0.005)
ax.axhline(y=baseline_auc, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f'{auc:.4f}', ha='center', fontsize=10, fontweight='bold')

# --- [0,1] Feature importance top 20 per model (side by side) ---
ax = axes[0, 1]

# Get union of top 20 features from all 3 models
top_feats = list(dict.fromkeys(
    list(imp_rf_norm.head(20).index) +
    list(imp_lgb_norm.head(20).index) +
    list(imp_xgb_norm.head(20).index)
))[:20]

y_pos = np.arange(len(top_feats))
w = 0.25

rf_vals = [imp_rf_norm.get(f, 0) for f in top_feats]
lgb_vals = [imp_lgb_norm.get(f, 0) for f in top_feats]
xgb_vals = [imp_xgb_norm.get(f, 0) for f in top_feats]

ax.barh(y_pos - w, rf_vals, w, label='RF v9b', color='#3498db', alpha=0.8)
ax.barh(y_pos, lgb_vals, w, label='LightGBM', color='#2ecc71', alpha=0.8)
ax.barh(y_pos + w, xgb_vals, w, label='XGBoost', color='#e74c3c', alpha=0.8)
ax.set_yticks(y_pos)

# Tag TX features in labels
feat_labels = []
for f in top_feats:
    if f in time_only_cols:
        feat_labels.append(f + ' [T]')
    elif f in agg_only_cols:
        feat_labels.append(f + ' [TX]')
    else:
        feat_labels.append(f)
ax.set_yticklabels(feat_labels, fontsize=6)
ax.invert_yaxis()
ax.set_title('Feature Importance Top 20 (normalized)')
ax.set_xlabel('Normalized Importance')
ax.legend(fontsize=8)

# --- [1,0] FN comparison ---
ax = axes[1, 0]
fn_vals = [results[n]['cm'][1, 0] for n in names_list]
fp_vals = [results[n]['cm'][0, 1] for n in names_list]
x = np.arange(len(names_list))
w = 0.35
bars_fn = ax.bar(x - w/2, fn_vals, w, label='FN (missed churn)', color='#e74c3c')
bars_fp = ax.bar(x + w/2, fp_vals, w, label='FP (false alarm)', color='#f39c12')
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=9, rotation=10)
ax.set_title('FN / FP Comparison')
ax.legend(fontsize=8)
ax.set_ylabel('Count')
for i, (fn, fp) in enumerate(zip(fn_vals, fp_vals)):
    ax.text(i - w/2, fn + max(fn_vals) * 0.02, f'{fn:,}', ha='center', fontsize=7)
    ax.text(i + w/2, fp + max(fp_vals) * 0.02, f'{fp:,}', ha='center', fontsize=7)

# --- [1,1] ROC curves ---
ax = axes[1, 1]
for name, color, short in zip(names_list, colors_4, short_names):
    fpr, tpr, _ = roc_curve(y_test, results[name]['probs'])
    auc_val = results[name]['auc']
    ax.plot(fpr, tpr, color=color, linewidth=1.5,
            label=f'{short} (AUC={auc_val:.4f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.suptitle(
    f'Phase 32: LightGBM & XGBoost with TX Features\n'
    f'RF v9b AUC={baseline_auc:.4f} | Best: {best_name} AUC={best_auc:.4f} '
    f'({(best_auc-baseline_auc)*100:+.2f}%)',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
os.makedirs('charts', exist_ok=True)
plt.savefig('charts/phase32_model_comparison.png', dpi=150, bbox_inches='tight')
print(f"  Saved: charts/phase32_model_comparison.png", flush=True)


# ===================================================================
# 9. Save best model
# ===================================================================
print("\n[9/9] Saving best model...", flush=True)

# Determine best model (excluding RF v9b baseline which is already saved)
best_new_name = max(
    [n for n in results if n != '[A] RF v9b (67f)'],
    key=lambda k: results[k]['auc']
)
best_new_auc = results[best_new_name]['auc']

model_map = {
    '[B] LGB (67f)': ('lgb_67f', lgb_model_67, both_cols),
    '[C] XGB (67f)': ('xgb_67f', xgb_model_67, both_cols),
    '[D] LGB (45f base)': ('lgb_45f', lgb_model_45, base_cols),
}

version_tag, best_model_obj, best_feature_cols = model_map[best_new_name]

# Identify which TX columns are in this model
best_tx_agg_cols = [c for c in best_feature_cols if c in agg_only_cols]
best_tx_time_cols = [c for c in best_feature_cols if c in time_only_cols]
best_base_cols = [c for c in best_feature_cols
                  if c not in agg_only_cols and c not in time_only_cols]

artifact = {
    'model': best_model_obj,
    'feature_cols': best_feature_cols,
    'tx_agg_feature_cols': best_tx_agg_cols,
    'tx_time_feature_cols': best_tx_time_cols,
    'base_feature_cols': best_base_cols,
    'version': f'v10_{version_tag}',
    'model_type': version_tag.split('_')[0],  # 'lgb' or 'xgb'
    'n_features': len(best_feature_cols),
    'n_train_files': 20,
    'n_train_samples': len(y_train),
    'auc_roc': best_new_auc,
    'threshold': THRESHOLD,
    'all_results': {
        name: {'auc': r['auc'], 'f1': r['f1']}
        for name, r in results.items()
    },
}

model_filename = f'models/churn_v10_{version_tag}.joblib'
joblib.dump(artifact, model_filename)
print(f"  Saved: {model_filename} ({version_tag}, "
      f"{len(best_feature_cols)} features, AUC={best_new_auc:.4f})", flush=True)

# If the overall best (including RF) is not RF, highlight it
if best_new_auc > baseline_auc:
    print(f"  ** New best model beats RF v9b! "
          f"AUC {baseline_auc:.4f} -> {best_new_auc:.4f} "
          f"({(best_new_auc-baseline_auc)*100:+.2f}%) **", flush=True)
else:
    print(f"  Note: RF v9b still has highest AUC ({baseline_auc:.4f}). "
          f"Best new model: {best_new_name} ({best_new_auc:.4f})", flush=True)


# ===================================================================
# Summary
# ===================================================================
print(f"\n{'='*70}", flush=True)
print(f"Phase 32: LightGBM & XGBoost with TX Features -- Summary", flush=True)
print(f"{'='*70}", flush=True)
print(f"  TX files processed        : {len(tx_agg_cache)}", flush=True)
print(f"  TX agg features           : {len(agg_only_cols)}", flush=True)
print(f"  TX time features          : {len(time_only_cols)}", flush=True)
print(f"  Train files               : 20", flush=True)
print(f"  Train samples             : {len(y_train):,}", flush=True)
print(f"  Test files                : 2", flush=True)
print(f"  Test samples              : {len(y_test):,}", flush=True)
print(f"  {'─'*50}", flush=True)

for name, r in results.items():
    nf = model_nfeats.get(name, '?')
    delta = (r['auc'] - baseline_auc) * 100
    delta_str = f"({delta:+.2f}%)" if name != '[A] RF v9b (67f)' else "(baseline)"
    print(f"  {name:<25}: AUC {r['auc']:.4f} F1 {r['f1']:.4f} {delta_str}", flush=True)

print(f"  {'─'*50}", flush=True)
print(f"  RF v9b baseline AUC       : {baseline_auc:.4f}", flush=True)
print(f"  Best new model            : {best_new_name}", flush=True)
print(f"  Best new AUC              : {best_new_auc:.4f}", flush=True)
print(f"  LGB TX impact (45f->67f)  : {(lgb_full_auc-lgb_base_auc)*100:+.2f}%", flush=True)
print(f"  FN: RF={len(fn_A):,} LGB={len(fn_B):,} XGB={len(fn_C):,}", flush=True)
print(f"  Chart                     : charts/phase32_model_comparison.png", flush=True)
print(f"  Model                     : {model_filename}", flush=True)
print(f"{'='*70}", flush=True)
