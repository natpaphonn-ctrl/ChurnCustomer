#!/usr/bin/env python3
"""Phase 30: Time-based TX Feature Engineering.

Analyze whether WHEN customers buy within the sales window (early vs rush/late)
predicts churn differently, and create new features from this analysis.

Thai lottery draws on 1st and 16th of every month. Sales window is ~12-13 days
before each draw. Rush buying happens in last 2 days (42.6% of orders).

New features (11): time-within-window (6) + buying intensity (5)
Model variants:
  - v3 baseline (45f)
  - v5 TX single (56f = 45 base + 11 TX aggregate)
  - v9a (56f = 45 base + 11 time features, no TX aggregate)
  - v9b (67f = 45 base + 11 TX aggregate + 11 time)
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

MIN_TENURE = 3

print("=" * 70, flush=True)
print("Phase 30: Time-based TX Feature Engineering", flush=True)
print("=" * 70, flush=True)

# ═══════════════════════════════════════════════════════════════
# TX ↔ Churn mapping (20 train + 2 test) — same as Phase 28
# ═══════════════════════════════════════════════════════════════

train_pairs = [
    ('TransactionOrder/Transaction_Order_2025_0216.csv', 'Churn_2025_0216_0301.csv'),
    ('TransactionOrder/Transaction_Order_2025_0301.csv', 'Churn_2025_0301_0316.csv'),
    ('TransactionOrder/Transaction_Order_2025_0316.csv', 'Churn_2025_0316_0401.csv'),
    ('TransactionOrder/Transaction_Order_2025_0401.csv', 'Churn_2025_0401_0416.csv'),
    ('TransactionOrder/Transaction_Order_2025_0416.csv', 'Churn_2025_0416_0502.csv'),
    ('TransactionOrder/Transaction_Order_2025_0502.csv', 'Churn_2025_0502_0516.csv'),
    ('TransactionOrder/Transaction_Order_2025_0516.csv', 'Churn_2025_0516_0601.csv'),
    ('TransactionOrder/Transaction_Order_2025_0601.csv', 'Churn_2025_0601_0616.csv'),
    ('TransactionOrder/Transaction_Order_2025_0616.csv', 'Churn_2025_0616_0701.csv'),
    ('TransactionOrder/Transaction_Order_2025_0701.csv', 'Churn_2025_0701_0716.csv'),
    ('TransactionOrder/Transaction_Order_2025_0716.csv', 'Churn_2025_0716_0801.csv'),
    ('TransactionOrder/Transaction_Order_2025_0801.csv', 'Churn_2025_0801_0816.csv'),
    ('TransactionOrder/Transaction_Order_2025_0816.csv', 'Churn_2025_0816_0901.csv'),
    ('TransactionOrder/Transaction_Order_2025_0901.csv', 'Churn_2025_0901_0916.csv'),
    ('TransactionOrder/Transaction_Order_2025_0916.csv', 'Churn_2025_0916_1001.csv'),
    ('TransactionOrder/Transaction_Order_2025_1001.csv', 'Churn_2025_1001_1016.csv'),
    ('TransactionOrder/Transaction_Order_2025_1016.csv', 'Churn_2025_1016_1101.csv'),
    ('TransactionOrder/Transaction_Order_2025_1101.csv', 'Churn_2025_1101_1116.csv'),
    ('TransactionOrder/Transaction_Order_2025_1116.csv', 'Churn_2025_1116_1201.csv'),
    ('TransactionOrder/Transaction_Order_2025_1201.csv', 'Churn_2025_1201_1216.csv'),
]

test_pairs = [
    ('TransactionOrder/Transaction_Order_2026_0201.csv', 'Churn_2026_0201_0216.csv'),
    ('TransactionOrder/Transaction_Order_2026_0216.csv', 'Churn_2026_0216_0301.csv'),
]


def load_churn_features(churn_file):
    """Load Churn CSV, compute v2 features (45), return (X, y, userNo_series)."""
    df = pd.read_csv(churn_file, low_memory=False)
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
    df = pd.read_csv(tx_file, low_memory=False)
    tx_feats = aggregate_tx_features(df)
    return tx_feats


def load_tx_time_features(tx_file):
    """Load TX file and compute time-within-window features (11 time)."""
    time_feats = compute_time_features(tx_file)
    return time_feats


# ═══════════════════════════════════════════════════════════════
# 1. Aggregate ALL TX files (both aggregate + time features)
# ═══════════════════════════════════════════════════════════════
print("\n[1/8] Aggregating transaction data (22 files)...", flush=True)
t0 = time.time()

tx_agg_cache = {}
tx_time_cache = {}
all_pairs = train_pairs + test_pairs

for tx_file, _ in all_pairs:
    if tx_file not in tx_agg_cache:
        t_file = time.time()
        basename = os.path.basename(tx_file)
        print(f"  {basename}...", end=" ", flush=True)

        # Aggregate TX features (same as Phase 28)
        tx_agg = load_tx_agg_features(tx_file)
        tx_agg_cache[tx_file] = tx_agg

        # Time-within-window TX features (new for Phase 30)
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


# ═══════════════════════════════════════════════════════════════
# 2. Load & merge Churn + TX for each pair
# ═══════════════════════════════════════════════════════════════
print("\n[2/8] Loading Churn features & merging with TX...", flush=True)
t0 = time.time()


def load_and_merge(pairs):
    """Load Churn + TX pairs, merge, return DataFrames for all model variants.

    Returns dict with keys:
        'base': X with 45 base features
        'tx_agg': X with 45 base + 11 TX aggregate = 56f
        'tx_time': X with 45 base + 11 time = 56f
        'tx_both': X with 45 base + 11 TX agg + 11 time = 67f
        'y': target array
    """
    all_base, all_tx_agg, all_tx_time, all_tx_both, all_y = [], [], [], [], []

    for tx_file, churn_file in pairs:
        print(f"  {os.path.basename(churn_file)}...", flush=True)
        feats, y, user_nos = load_churn_features(churn_file)

        # Get TX features for this period
        tx_agg = tx_agg_cache[tx_file]
        tx_time = tx_time_cache[tx_file]

        # Merge by userNo (left join)
        feats_with_user = feats.copy()
        feats_with_user['userNo'] = user_nos.values

        # Merge aggregate TX
        merged_agg = feats_with_user.merge(
            tx_agg, left_on='userNo', right_index=True, how='left'
        )
        agg_cols = list(tx_agg.columns)
        merged_agg[agg_cols] = merged_agg[agg_cols].fillna(0)

        # Merge time TX
        merged_time = feats_with_user.merge(
            tx_time, left_on='userNo', right_index=True, how='left'
        )
        time_cols = list(tx_time.columns)
        merged_time[time_cols] = merged_time[time_cols].fillna(0)

        # Merge both
        merged_both = feats_with_user.merge(
            tx_agg, left_on='userNo', right_index=True, how='left'
        ).merge(
            tx_time, left_on='userNo', right_index=True, how='left'
        )
        merged_both[agg_cols] = merged_both[agg_cols].fillna(0)
        merged_both[time_cols] = merged_both[time_cols].fillna(0)

        # Stats
        matched_agg = (merged_agg.get('tx_n_orders', pd.Series(0, index=merged_agg.index)) > 0).sum()
        matched_time = (merged_time.get('tx_early_buyer_ratio', pd.Series(0, index=merged_time.index)) > 0).sum()
        print(f"    n={len(feats):,}, churn={y.mean()*100:.1f}%, "
              f"TX agg matched={matched_agg:,}, time matched={matched_time:,}", flush=True)

        X_base = feats.copy()
        X_tx_agg = merged_agg.drop(columns=['userNo'])
        X_tx_time = merged_time.drop(columns=['userNo'])
        X_tx_both = merged_both.drop(columns=['userNo'])

        all_base.append(X_base)
        all_tx_agg.append(X_tx_agg)
        all_tx_time.append(X_tx_time)
        all_tx_both.append(X_tx_both)
        all_y.append(y)

    return {
        'base': pd.concat(all_base, ignore_index=True),
        'tx_agg': pd.concat(all_tx_agg, ignore_index=True),
        'tx_time': pd.concat(all_tx_time, ignore_index=True),
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
print(f"  TX agg features: {train_data['tx_agg'].shape[1]}", flush=True)
print(f"  TX time features: {train_data['tx_time'].shape[1]}", flush=True)
print(f"  TX both features: {train_data['tx_both'].shape[1]}", flush=True)


# ═══════════════════════════════════════════════════════════════
# 3. Training data management (subsample if > 2M)
# ═══════════════════════════════════════════════════════════════
print("\n[3/8] Training data management...", flush=True)

y_train = train_data['y']
y_test = test_data['y']

if len(y_train) > 2_000_000:
    print(f"  Total rows ({len(y_train):,}) > 2M, subsampling...", flush=True)
    idx = np.random.RandomState(42).choice(len(y_train), 2_000_000, replace=False)
    for key in ['base', 'tx_agg', 'tx_time', 'tx_both']:
        train_data[key] = train_data[key].iloc[idx].reset_index(drop=True)
    y_train = y_train[idx]
    train_data['y'] = y_train
    print(f"  Subsampled to {len(y_train):,}", flush=True)
else:
    print(f"  Total rows: {len(y_train):,} (under 2M, no subsampling needed)", flush=True)

print(f"  Final train size: {len(y_train):,}, churn rate: {y_train.mean()*100:.1f}%", flush=True)

# Identify feature column groups
base_cols = list(train_data['base'].columns)
agg_only_cols = [c for c in train_data['tx_agg'].columns if c not in base_cols]
time_only_cols = [c for c in train_data['tx_time'].columns if c not in base_cols]

print(f"  Base cols: {len(base_cols)}", flush=True)
print(f"  TX agg cols ({len(agg_only_cols)}): {agg_only_cols}", flush=True)
print(f"  TX time cols ({len(time_only_cols)}): {time_only_cols}", flush=True)


# ═══════════════════════════════════════════════════════════════
# 4. Train 4 models and compare
# ═══════════════════════════════════════════════════════════════
print("\n[4/8] Training & evaluating models...", flush=True)
t0 = time.time()

RF_PARAMS = dict(
    n_estimators=300, max_depth=20, min_samples_leaf=50,
    class_weight='balanced', random_state=42, n_jobs=-1,
)
THRESHOLD = 0.48

results = {}
models = {}

# --- Model A: RF v3 pretrained (45 features, baseline) ---
print("\n  [A] RF v3 pretrained (45f baseline) — evaluate only...", flush=True)
t_a = time.time()
v3_artifact = joblib.load('models/churn_v3_extended.joblib')
v3_model = v3_artifact['model']
v3_feature_cols = v3_artifact['feature_cols']

# Match v3 feature columns to test base columns
X_test_A = test_data['base'].copy()
for c in v3_feature_cols:
    if c not in X_test_A.columns:
        X_test_A[c] = 0
X_test_A = X_test_A[v3_feature_cols].fillna(0)

probs_A = v3_model.predict_proba(X_test_A)[:, 1]
preds_A = (probs_A >= THRESHOLD).astype(int)
results['v3 baseline (45f)'] = {
    'auc': roc_auc_score(y_test, probs_A),
    'f1': f1_score(y_test, preds_A),
    'cm': confusion_matrix(y_test, preds_A),
    'probs': probs_A,
    'preds': preds_A,
}
print(f"      AUC={results['v3 baseline (45f)']['auc']:.4f} ({time.time()-t_a:.1f}s)", flush=True)

# --- Model B: v5 TX aggregate (56f = 45 base + 11 TX agg) ---
print("\n  [B] v5 TX agg (56f = 45+11 agg) — training...", flush=True)
t_b = time.time()
rf_B = RandomForestClassifier(**RF_PARAMS)
rf_B.fit(train_data['tx_agg'].fillna(0), y_train)
probs_B = rf_B.predict_proba(test_data['tx_agg'].fillna(0))[:, 1]
preds_B = (probs_B >= THRESHOLD).astype(int)
results['v5 TX agg (56f)'] = {
    'auc': roc_auc_score(y_test, probs_B),
    'f1': f1_score(y_test, preds_B),
    'cm': confusion_matrix(y_test, preds_B),
    'probs': probs_B,
    'preds': preds_B,
}
models['v5'] = rf_B
print(f"      AUC={results['v5 TX agg (56f)']['auc']:.4f} ({time.time()-t_b:.1f}s)", flush=True)

# --- Model C: v9a time only (56f = 45 base + 11 time) ---
print("\n  [C] v9a time only (56f = 45+11 time) — training...", flush=True)
t_c = time.time()
rf_C = RandomForestClassifier(**RF_PARAMS)
rf_C.fit(train_data['tx_time'].fillna(0), y_train)
probs_C = rf_C.predict_proba(test_data['tx_time'].fillna(0))[:, 1]
preds_C = (probs_C >= THRESHOLD).astype(int)
results['v9a time only (56f)'] = {
    'auc': roc_auc_score(y_test, probs_C),
    'f1': f1_score(y_test, preds_C),
    'cm': confusion_matrix(y_test, preds_C),
    'probs': probs_C,
    'preds': preds_C,
}
models['v9a'] = rf_C
print(f"      AUC={results['v9a time only (56f)']['auc']:.4f} ({time.time()-t_c:.1f}s)", flush=True)

# --- Model D: v9b full (67f = 45 base + 11 TX agg + 11 time) ---
print("\n  [D] v9b full (67f = 45+11+11) — training...", flush=True)
t_d = time.time()
rf_D = RandomForestClassifier(**RF_PARAMS)
rf_D.fit(train_data['tx_both'].fillna(0), y_train)
probs_D = rf_D.predict_proba(test_data['tx_both'].fillna(0))[:, 1]
preds_D = (probs_D >= THRESHOLD).astype(int)
results['v9b full (67f)'] = {
    'auc': roc_auc_score(y_test, probs_D),
    'f1': f1_score(y_test, preds_D),
    'cm': confusion_matrix(y_test, preds_D),
    'probs': probs_D,
    'preds': preds_D,
}
models['v9b'] = rf_D
print(f"      AUC={results['v9b full (67f)']['auc']:.4f} ({time.time()-t_d:.1f}s)", flush=True)

print(f"\n  All models evaluated ({time.time()-t0:.1f}s)", flush=True)


# ═══════════════════════════════════════════════════════════════
# 5. Results table
# ═══════════════════════════════════════════════════════════════
print("\n[5/8] Results comparison...", flush=True)

baseline_auc = results['v3 baseline (45f)']['auc']

print(f"\n  {'='*90}", flush=True)
print(f"  Model Comparison (Test: {len(y_test):,} customers, "
      f"churn={y_test.mean()*100:.1f}%)", flush=True)
print(f"  {'='*90}", flush=True)
print(f"  {'Model':<30} {'Features':>10} {'AUC-ROC':>10} {'F1':>10} "
      f"{'FN':>10} {'FP':>10} {'delta AUC':>10}", flush=True)
print(f"  {'-'*90}", flush=True)

model_nfeats = {
    'v3 baseline (45f)': 45,
    'v5 TX agg (56f)': 56,
    'v9a time only (56f)': 56,
    'v9b full (67f)': 67,
}

for name, r in results.items():
    fn = r['cm'][1, 0]
    fp = r['cm'][0, 1]
    nf = model_nfeats.get(name, '?')
    delta = (r['auc'] - baseline_auc) * 100
    delta_str = f"{delta:+.2f}%" if name != 'v3 baseline (45f)' else "baseline"
    print(f"  {name:<30} {nf:>10} {r['auc']:>10.4f} {r['f1']:>10.4f} "
          f"{fn:>10,} {fp:>10,} {delta_str:>10}", flush=True)

print(f"  {'-'*90}", flush=True)

best_name = max(results, key=lambda k: results[k]['auc'])
best_auc = results[best_name]['auc']
print(f"  Baseline AUC: {baseline_auc:.4f}", flush=True)
print(f"  Best model:   {best_name} (AUC: {best_auc:.4f})", flush=True)
print(f"  Improvement:  {(best_auc - baseline_auc)*100:+.2f}%", flush=True)


# ═══════════════════════════════════════════════════════════════
# 6. Feature importance analysis
# ═══════════════════════════════════════════════════════════════
print("\n[6/8] Feature importance analysis...", flush=True)

# Use v9b (67f) for feature importance since it has all features
both_cols = list(train_data['tx_both'].columns)
importances_D = pd.Series(rf_D.feature_importances_, index=both_cols).sort_values(ascending=False)

print(f"\n  Time Feature Importance (rank / {len(both_cols)}):", flush=True)
for feat in time_only_cols:
    if feat in importances_D.index:
        rank = importances_D.index.tolist().index(feat) + 1
        print(f"    {feat:<30} importance={importances_D[feat]:.4f}  "
              f"rank={rank}/{len(both_cols)}", flush=True)

print(f"\n  TX Aggregate Feature Importance (rank / {len(both_cols)}):", flush=True)
for feat in agg_only_cols:
    if feat in importances_D.index:
        rank = importances_D.index.tolist().index(feat) + 1
        print(f"    {feat:<30} importance={importances_D[feat]:.4f}  "
              f"rank={rank}/{len(both_cols)}", flush=True)

print(f"\n  Top 15 overall features:", flush=True)
for i, (feat, imp) in enumerate(importances_D.head(15).items()):
    tag = ""
    if feat in time_only_cols:
        tag = " [TIME]"
    elif feat in agg_only_cols:
        tag = " [TX]"
    print(f"    {i+1:>2}. {feat:<35} {imp:.4f}{tag}", flush=True)


# ═══════════════════════════════════════════════════════════════
# 7. FN Analysis
# ═══════════════════════════════════════════════════════════════
print("\n[7/8] FN Analysis...", flush=True)

fn_A = set(np.where((y_test == 1) & (preds_A == 0))[0])
fn_B = set(np.where((y_test == 1) & (preds_B == 0))[0])
fn_C = set(np.where((y_test == 1) & (preds_C == 0))[0])
fn_D = set(np.where((y_test == 1) & (preds_D == 0))[0])

print(f"    [A] v3 baseline (45f)     FN: {len(fn_A):,}", flush=True)
print(f"    [B] v5 TX agg (56f)       FN: {len(fn_B):,}", flush=True)
print(f"    [C] v9a time only (56f)   FN: {len(fn_C):,}", flush=True)
print(f"    [D] v9b full (67f)        FN: {len(fn_D):,}", flush=True)
print(f"    FN saved (A->D): {len(fn_A) - len(fn_D):+,}", flush=True)
print(f"    FN saved (B->D): {len(fn_B) - len(fn_D):+,}", flush=True)

# Compare time vs agg: which captures more unique churners?
fn_saved_by_time = fn_A - fn_C  # customers saved by time features alone
fn_saved_by_agg = fn_A - fn_B   # customers saved by agg features alone
fn_saved_by_both = fn_A - fn_D  # customers saved by combining both
print(f"\n    Unique FN saves vs baseline:", flush=True)
print(f"      Time features only:  {len(fn_saved_by_time):,}", flush=True)
print(f"      TX agg features only: {len(fn_saved_by_agg):,}", flush=True)
print(f"      Both combined:       {len(fn_saved_by_both):,}", flush=True)

# Overlap analysis
fn_only_time = fn_saved_by_time - fn_saved_by_agg
fn_only_agg = fn_saved_by_agg - fn_saved_by_time
fn_overlap = fn_saved_by_time & fn_saved_by_agg
print(f"\n    Overlap analysis:", flush=True)
print(f"      Saved only by time: {len(fn_only_time):,}", flush=True)
print(f"      Saved only by agg:  {len(fn_only_agg):,}", flush=True)
print(f"      Saved by both:      {len(fn_overlap):,}", flush=True)


# ═══════════════════════════════════════════════════════════════
# 8. Charts
# ═══════════════════════════════════════════════════════════════
print("\n  Generating charts...", flush=True)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# --- [0,0] AUC bar chart (4 models) ---
ax = axes[0, 0]
names_list = list(results.keys())
aucs = [results[n]['auc'] for n in names_list]
short_names = ['v3 (45f)', 'v5 agg (56f)', 'v9a time (56f)', 'v9b both (67f)']
colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c']
bars = ax.bar(range(len(names_list)), aucs, color=colors)
ax.set_xticks(range(len(names_list)))
ax.set_xticklabels(short_names, fontsize=8, rotation=15)
ax.set_ylabel('AUC-ROC')
ax.set_title('AUC-ROC Comparison')
ax.set_ylim(min(aucs) - 0.005, max(aucs) + 0.005)
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f'{auc:.4f}', ha='center', fontsize=9, fontweight='bold')

# --- [0,1] FN/FP comparison ---
ax = axes[0, 1]
fn_vals = [results[n]['cm'][1, 0] for n in names_list]
fp_vals = [results[n]['cm'][0, 1] for n in names_list]
x = np.arange(len(names_list))
w = 0.35
bars_fn = ax.bar(x - w/2, fn_vals, w, label='FN (missed churn)', color='#e74c3c')
bars_fp = ax.bar(x + w/2, fp_vals, w, label='FP (false alarm)', color='#f39c12')
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=8, rotation=15)
ax.set_title('FN / FP Comparison')
ax.legend(fontsize=8)
ax.set_ylabel('Count')
for i, (fn, fp) in enumerate(zip(fn_vals, fp_vals)):
    ax.text(i - w/2, fn + max(fn_vals) * 0.02, f'{fn:,}', ha='center', fontsize=7)
    ax.text(i + w/2, fp + max(fp_vals) * 0.02, f'{fp:,}', ha='center', fontsize=7)

# --- [0,2] F1 bar chart ---
ax = axes[0, 2]
f1s = [results[n]['f1'] for n in names_list]
bars = ax.bar(range(len(names_list)), f1s, color=colors)
ax.set_xticks(range(len(names_list)))
ax.set_xticklabels(short_names, fontsize=8, rotation=15)
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score Comparison')
ax.set_ylim(min(f1s) - 0.01, max(f1s) + 0.01)
for bar, f1 in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{f1:.4f}', ha='center', fontsize=9, fontweight='bold')

# --- [1,0] Feature importance top 30 (color-coded) ---
ax = axes[1, 0]
top30 = importances_D.head(30)
colors_imp = []
for f in top30.index:
    if f in time_only_cols:
        colors_imp.append('#2ecc71')   # green for time
    elif f in agg_only_cols:
        colors_imp.append('#e67e22')   # orange for TX agg
    else:
        colors_imp.append('#3498db')   # blue for base
ax.barh(range(len(top30)), top30.values, color=colors_imp)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(top30.index, fontsize=6)
ax.invert_yaxis()
ax.set_title('Feature Importance (Top 30)\nBlue=Base, Orange=TX Agg, Green=Time')
ax.set_xlabel('Importance')

# --- [1,1] Time feature importance (standalone) ---
ax = axes[1, 1]
time_imp = importances_D[time_only_cols].sort_values(ascending=False)
bars_time = ax.barh(range(len(time_imp)), time_imp.values, color='#2ecc71')
ax.set_yticks(range(len(time_imp)))
ax.set_yticklabels(time_imp.index, fontsize=8)
ax.invert_yaxis()
ax.set_title('Time Feature Importance (11 features)')
ax.set_xlabel('Importance')
for i, (feat, imp) in enumerate(time_imp.items()):
    ax.text(imp + 0.0001, i, f'{imp:.4f}', va='center', fontsize=7)

# --- [1,2] Delta AUC vs baseline ---
ax = axes[1, 2]
deltas = [(results[n]['auc'] - baseline_auc) * 100 for n in names_list[1:]]
delta_names = short_names[1:]
delta_colors = colors[1:]
bars_d = ax.bar(range(len(deltas)), deltas, color=delta_colors)
ax.set_xticks(range(len(deltas)))
ax.set_xticklabels(delta_names, fontsize=8, rotation=15)
ax.set_ylabel('AUC improvement (%)')
ax.set_title('AUC Improvement vs v3 Baseline')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
for bar, d in zip(bars_d, deltas):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.01 if d >= 0 else -0.03),
            f'{d:+.2f}%', ha='center', fontsize=9, fontweight='bold')

plt.suptitle(
    f'Phase 30: Time-based TX Feature Engineering\n'
    f'Baseline AUC={baseline_auc:.4f} | Best: {best_name} AUC={best_auc:.4f} '
    f'({(best_auc-baseline_auc)*100:+.2f}%)',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig('charts/phase30_time_features.png', dpi=150, bbox_inches='tight')
print(f"  Saved: charts/phase30_time_features.png", flush=True)


# ═══════════════════════════════════════════════════════════════
# 9. Save best model
# ═══════════════════════════════════════════════════════════════
print("\n  Saving best model...", flush=True)

# Determine best model variant
best_model_map = {
    'v5 TX agg (56f)': ('v5', rf_B, list(train_data['tx_agg'].columns)),
    'v9a time only (56f)': ('v9a', rf_C, list(train_data['tx_time'].columns)),
    'v9b full (67f)': ('v9b', rf_D, list(train_data['tx_both'].columns)),
}

if best_name in best_model_map:
    version_tag, best_rf, best_feature_cols = best_model_map[best_name]
else:
    # If v3 baseline is still best, save v9b anyway for reference
    print("  Note: v3 baseline still best. Saving v9b for reference.", flush=True)
    version_tag, best_rf, best_feature_cols = 'v9b', rf_D, list(train_data['tx_both'].columns)

# Identify which TX columns are in this model
best_time_cols = [c for c in best_feature_cols if c in time_only_cols]
best_agg_cols = [c for c in best_feature_cols if c in agg_only_cols]

artifact = {
    'model': best_rf,
    'feature_cols': best_feature_cols,
    'tx_agg_feature_cols': best_agg_cols,
    'tx_time_feature_cols': best_time_cols,
    'base_feature_cols': [c for c in best_feature_cols
                          if c not in time_only_cols and c not in agg_only_cols],
    'version': f'{version_tag}_time',
    'n_features': len(best_feature_cols),
    'n_train_files': 20,
    'n_train_samples': len(y_train),
    'auc_roc': results[best_name]['auc'],
    'all_results': {
        name: {'auc': r['auc'], 'f1': r['f1']}
        for name, r in results.items()
    },
}
joblib.dump(artifact, 'models/churn_v9_time.joblib')
print(f"  Saved: models/churn_v9_time.joblib ({version_tag}, "
      f"{len(best_feature_cols)} features)", flush=True)


# ═══════════════════════════════════════════════════════════════
# 10. Summary
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}", flush=True)
print(f"Phase 30: Time-based TX Feature Engineering — Summary", flush=True)
print(f"{'='*70}", flush=True)
print(f"  TX files processed        : {len(tx_agg_cache)}", flush=True)
print(f"  TX aggregate features     : {len(agg_only_cols)}", flush=True)
print(f"  TX time features          : {len(time_only_cols)}", flush=True)
print(f"  Train files               : 20", flush=True)
print(f"  Train samples             : {len(y_train):,}", flush=True)
print(f"  Test files                : 2", flush=True)
print(f"  Test samples              : {len(y_test):,}", flush=True)
print(f"  {'─'*50}", flush=True)

for name, r in results.items():
    nf = model_nfeats.get(name, '?')
    delta = (r['auc'] - baseline_auc) * 100
    delta_str = f"({delta:+.2f}%)" if name != 'v3 baseline (45f)' else "(baseline)"
    print(f"  {name:<30}: AUC {r['auc']:.4f} {delta_str}", flush=True)

print(f"  {'─'*50}", flush=True)
print(f"  Best model                : {best_name}", flush=True)
print(f"  Best AUC                  : {best_auc:.4f}", flush=True)
print(f"  FN saved (baseline->best) : {len(fn_A) - len(fn_D):+,}", flush=True)
print(f"  Chart                     : charts/phase30_time_features.png", flush=True)
print(f"  Model                     : models/churn_v9_time.joblib", flush=True)
print(f"{'='*70}", flush=True)
