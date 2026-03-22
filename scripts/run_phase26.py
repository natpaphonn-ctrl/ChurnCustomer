#!/usr/bin/env python3
"""Phase 26: Transaction Data Aggregation & Model Testing.

Aggregates transaction-level order data into per-customer features,
merges with existing Churn features, and tests whether TX features
improve prediction performance.
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
from feature_engineering import engineer_features_v2, engineer_features_v3, aggregate_tx_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

MIN_TENURE = 3

print("=" * 60)
print("Phase 26: Transaction Data Aggregation & Model Testing")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# TX ↔ Churn mapping
# TX period → second-to-last item column in NEXT Churn file
# Train: TX_0102 → Churn_0102_0117, TX_0117 → Churn_0117_0201
# Test:  TX_0201 → Churn_0201_0216, TX_0216 → Churn_0216_0301
# ═══════════════════════════════════════════════════════════════

tx_churn_map = {
    'TransactionOrder/Transaction_Order_2026_0102.csv': 'Churn_2026_0102_0117.csv',
    'TransactionOrder/Transaction_Order_2026_0117.csv': 'Churn_2026_0117_0201.csv',
    'TransactionOrder/Transaction_Order_2026_0201.csv': 'Churn_2026_0201_0216.csv',
    'TransactionOrder/Transaction_Order_2026_0216.csv': 'Churn_2026_0216_0301.csv',
}

train_pairs = [
    ('TransactionOrder/Transaction_Order_2026_0102.csv', 'Churn_2026_0102_0117.csv'),
    ('TransactionOrder/Transaction_Order_2026_0117.csv', 'Churn_2026_0117_0201.csv'),
]
test_pairs = [
    ('TransactionOrder/Transaction_Order_2026_0201.csv', 'Churn_2026_0201_0216.csv'),
    ('TransactionOrder/Transaction_Order_2026_0216.csv', 'Churn_2026_0216_0301.csv'),
]


def load_churn_features(churn_file, version='v3'):
    """Load Churn CSV, compute features, return (X, y, userNo_series)."""
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

    if version == 'v3':
        feats = engineer_features_v3(df_v, mat_v, feat_item_cols)
    else:
        feats = engineer_features_v2(df_v, mat_v, feat_item_cols)

    feats = feats.fillna(0)
    user_nos = df_v['userNo'].reset_index(drop=True)

    return feats, y, user_nos


def load_tx_features(tx_file):
    """Load TX file and aggregate into per-customer features."""
    df = pd.read_csv(tx_file, low_memory=False)
    tx_feats = aggregate_tx_features(df)
    return tx_feats


# ═══════════════════════════════════════════════════════════════
# 1. Aggregate TX data
# ═══════════════════════════════════════════════════════════════
print("\n[1/5] Aggregating transaction data...")
t0 = time.time()

tx_cache = {}
for tx_file, _ in train_pairs + test_pairs:
    if tx_file not in tx_cache:
        print(f"  {os.path.basename(tx_file)}...", end=" ", flush=True)
        tx_feats = load_tx_features(tx_file)
        tx_cache[tx_file] = tx_feats
        print(f"{len(tx_feats):,} customers, {tx_feats.shape[1]} features")

print(f"  Done ({time.time()-t0:.1f}s)")

# Show sample TX features
sample_tx = list(tx_cache.values())[0]
print(f"\n  TX features: {list(sample_tx.columns)}")
print(f"  Sample stats:")
for col in sample_tx.columns:
    print(f"    {col:<25} mean={sample_tx[col].mean():.3f}  std={sample_tx[col].std():.3f}")


# ═══════════════════════════════════════════════════════════════
# 2. Load Churn features & merge with TX
# ═══════════════════════════════════════════════════════════════
print("\n[2/5] Loading Churn features & merging with TX...")
t0 = time.time()


def load_and_merge(pairs, version='v3'):
    """Load Churn + TX pairs, merge, return (X_base, X_merged, y)."""
    all_X_base, all_X_merged, all_y = [], [], []

    for tx_file, churn_file in pairs:
        print(f"  {os.path.basename(churn_file)}...", flush=True)
        feats, y, user_nos = load_churn_features(churn_file, version=version)

        # Get TX features for this period
        tx_feats = tx_cache[tx_file]

        # Merge by userNo (left join — keep all Churn customers)
        feats_with_user = feats.copy()
        feats_with_user['userNo'] = user_nos.values

        merged = feats_with_user.merge(
            tx_feats, left_on='userNo', right_index=True, how='left'
        )

        # Fill NaN for customers without TX data (didn't buy in this period)
        tx_cols = [c for c in tx_feats.columns]
        merged[tx_cols] = merged[tx_cols].fillna(0)

        # How many matched?
        matched = (merged['tx_n_orders'] > 0).sum()
        print(f"    n={len(feats):,}, churn={y.mean()*100:.1f}%, TX matched={matched:,} ({matched/len(feats)*100:.1f}%)")

        X_base = merged.drop(columns=['userNo'] + tx_cols)
        X_merged = merged.drop(columns=['userNo'])

        all_X_base.append(X_base)
        all_X_merged.append(X_merged)
        all_y.append(y)

    return (
        pd.concat(all_X_base, ignore_index=True),
        pd.concat(all_X_merged, ignore_index=True),
        np.concatenate(all_y),
    )


# Load train
print("\n  --- Train ---")
X_train_base, X_train_merged, y_train = load_and_merge(train_pairs, version='v3')
print(f"  Train total: {len(y_train):,}, churn={y_train.mean()*100:.1f}%")

# Subsample if too large
if len(X_train_base) > 1_000_000:
    idx = np.random.RandomState(42).choice(len(X_train_base), 1_000_000, replace=False)
    X_train_base = X_train_base.iloc[idx].reset_index(drop=True)
    X_train_merged = X_train_merged.iloc[idx].reset_index(drop=True)
    y_train = y_train[idx]
    print(f"  Subsampled to {len(y_train):,}")

# Load test
print("\n  --- Test ---")
X_test_base, X_test_merged, y_test = load_and_merge(test_pairs, version='v3')
print(f"  Test total: {len(y_test):,}, churn={y_test.mean()*100:.1f}%")

# Also load v2 features for v3 baseline comparison
print("\n  --- Test (v2 features for baseline) ---")
X_test_v2_base, X_test_v2_merged, _ = load_and_merge(test_pairs, version='v2')

print(f"\n  Done ({time.time()-t0:.1f}s)")
print(f"  Base features: {X_train_base.shape[1]}")
print(f"  Merged features: {X_train_merged.shape[1]}")
print(f"  TX features added: {X_train_merged.shape[1] - X_train_base.shape[1]}")


# ═══════════════════════════════════════════════════════════════
# 3. Train models & compare
# ═══════════════════════════════════════════════════════════════
print("\n[3/5] Training models...")
t0 = time.time()

# Load v3 model for reference
v3_artifact = joblib.load('models/churn_v3_extended.joblib')
v3_model = v3_artifact['model']
v3_feature_cols = v3_artifact['feature_cols']

# Identify feature columns
base_cols = list(X_train_base.columns)  # 52 (v3 features)
merged_cols = list(X_train_merged.columns)  # 52 + 11 TX
tx_only_cols = [c for c in merged_cols if c not in base_cols]
v2_cols = [c for c in v3_feature_cols if c in X_test_v2_base.columns]  # 45 (v2 features)
v2_plus_tx_cols = v2_cols + tx_only_cols

print(f"  v3 baseline cols: {len(v3_feature_cols)}")
print(f"  v3 retrain cols: {len(base_cols)}")
print(f"  TX cols: {tx_only_cols}")
print(f"  v3+TX cols: {len(merged_cols)}")

RF_PARAMS = dict(
    n_estimators=300, max_depth=20, min_samples_leaf=50,
    class_weight='balanced', random_state=42, n_jobs=-1,
)
THRESHOLD = 0.48

results = {}

# --- Model A: v3 pretrained (45 features, original model) ---
print("\n  [A] RF v3 pretrained (45 features)...")
v3_cols_test = [c for c in v3_feature_cols if c in X_test_v2_base.columns]
probs_A = v3_model.predict_proba(X_test_v2_base[v3_cols_test].fillna(0))[:, 1]
preds_A = (probs_A >= THRESHOLD).astype(int)
results['RF v3 pretrained (45)'] = {
    'auc': roc_auc_score(y_test, probs_A),
    'f1': f1_score(y_test, preds_A),
    'cm': confusion_matrix(y_test, preds_A),
}

# --- Model B: v3 retrained on 2026 data (52 features, no TX) ---
print("  [B] RF v3 retrained (52 features, no TX)...")
rf_B = RandomForestClassifier(**RF_PARAMS)
rf_B.fit(X_train_base.fillna(0), y_train)
probs_B = rf_B.predict_proba(X_test_base.fillna(0))[:, 1]
preds_B = (probs_B >= THRESHOLD).astype(int)
results['RF retrained (52, no TX)'] = {
    'auc': roc_auc_score(y_test, probs_B),
    'f1': f1_score(y_test, preds_B),
    'cm': confusion_matrix(y_test, preds_B),
}

# --- Model C: v2 features + TX (45+11=56 features) ---
print("  [C] RF v2+TX (56 features)...")
# Build v2+TX train set
X_train_v2_tx = X_train_merged[[c for c in X_train_merged.columns if c in v2_plus_tx_cols]].copy()
# Fill missing v2 cols
for c in v2_plus_tx_cols:
    if c not in X_train_v2_tx.columns:
        X_train_v2_tx[c] = 0
X_train_v2_tx = X_train_v2_tx[v2_plus_tx_cols]

X_test_v2_tx = X_test_v2_merged[[c for c in X_test_v2_merged.columns if c in v2_plus_tx_cols]].copy()
for c in v2_plus_tx_cols:
    if c not in X_test_v2_tx.columns:
        X_test_v2_tx[c] = 0
X_test_v2_tx = X_test_v2_tx[v2_plus_tx_cols]

rf_C = RandomForestClassifier(**RF_PARAMS)
rf_C.fit(X_train_v2_tx.fillna(0), y_train)
probs_C = rf_C.predict_proba(X_test_v2_tx.fillna(0))[:, 1]
preds_C = (probs_C >= THRESHOLD).astype(int)
results['RF v2+TX (56)'] = {
    'auc': roc_auc_score(y_test, probs_C),
    'f1': f1_score(y_test, preds_C),
    'cm': confusion_matrix(y_test, preds_C),
}

# --- Model D: v3 features + TX (52+11=63 features) ---
print("  [D] RF v3+TX (63 features)...")
rf_D = RandomForestClassifier(**RF_PARAMS)
rf_D.fit(X_train_merged.fillna(0), y_train)
probs_D = rf_D.predict_proba(X_test_merged.fillna(0))[:, 1]
preds_D = (probs_D >= THRESHOLD).astype(int)
results['RF v3+TX (63)'] = {
    'auc': roc_auc_score(y_test, probs_D),
    'f1': f1_score(y_test, preds_D),
    'cm': confusion_matrix(y_test, preds_D),
}

print(f"  Training done ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════
# 4. Results comparison
# ═══════════════════════════════════════════════════════════════
print("\n[4/5] Results comparison...")
print(f"\n  {'='*72}")
print(f"  Model Comparison (Test: {len(y_test):,} customers)")
print(f"  {'='*72}")
print(f"  {'Model':<30} {'AUC-ROC':>10} {'F1':>10} {'FN':>10} {'FP':>10}")
print(f"  {'-'*72}")

for name, r in results.items():
    fn = r['cm'][1, 0]
    fp = r['cm'][0, 1]
    print(f"  {name:<30} {r['auc']:>10.4f} {r['f1']:>10.4f} {fn:>10,} {fp:>10,}")

print(f"  {'-'*72}")

# Calculate improvements
baseline_auc = results['RF v3 pretrained (45)']['auc']
best_name = max(results, key=lambda k: results[k]['auc'])
best_auc = results[best_name]['auc']
print(f"  Baseline AUC: {baseline_auc:.4f}")
print(f"  Best model:   {best_name} (AUC: {best_auc:.4f})")
print(f"  Improvement:  {(best_auc - baseline_auc)*100:+.2f}%")

# FN analysis — does TX help catch more churners?
fn_A = set(np.where((y_test == 1) & (preds_A == 0))[0])
fn_D = set(np.where((y_test == 1) & (preds_D == 0))[0])
fn_saved = fn_A - fn_D
fn_new = fn_D - fn_A
print(f"\n  FN Analysis (v3 pretrained vs v3+TX):")
print(f"    v3 FN: {len(fn_A):,}")
print(f"    v3+TX FN: {len(fn_D):,}")
print(f"    FN saved by TX: {len(fn_saved):,}")
print(f"    New FN from TX: {len(fn_new):,}")
print(f"    Net FN change: {len(fn_A) - len(fn_D):+,}")


# ═══════════════════════════════════════════════════════════════
# 5. Feature importance & chart
# ═══════════════════════════════════════════════════════════════
print("\n[5/5] Feature importance & charts...")

# Feature importance from Model D (best with TX)
importances_D = pd.Series(rf_D.feature_importances_, index=merged_cols).sort_values(ascending=False)

print(f"\n  TX Feature Importance (in Model D, rank / {len(merged_cols)}):")
for feat in tx_only_cols:
    rank = importances_D.index.tolist().index(feat) + 1
    print(f"    {feat:<25} importance={importances_D[feat]:.4f}  rank={rank}/{len(merged_cols)}")

# Chart
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. AUC comparison bar chart
ax = axes[0, 0]
names = list(results.keys())
aucs = [results[n]['auc'] for n in names]
short_names = ['v3 (45)', 'v3 retrain (52)', 'v2+TX (56)', 'v3+TX (63)']
colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']
bars = ax.bar(range(len(names)), aucs, color=colors)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(short_names, fontsize=9)
ax.set_ylabel('AUC-ROC')
ax.set_title('AUC-ROC Comparison')
ax.set_ylim(min(aucs) - 0.005, max(aucs) + 0.005)
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f'{auc:.4f}', ha='center', fontsize=9, fontweight='bold')

# 2. FN/FP comparison
ax = axes[0, 1]
fn_vals = [results[n]['cm'][1, 0] for n in names]
fp_vals = [results[n]['cm'][0, 1] for n in names]
x = np.arange(len(names))
w = 0.35
ax.bar(x - w/2, fn_vals, w, label='FN (missed churn)', color='#e74c3c')
ax.bar(x + w/2, fp_vals, w, label='FP (false alarm)', color='#f39c12')
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=9)
ax.set_title('Error Comparison')
ax.legend()
ax.set_ylabel('Count')
for i, (fn, fp) in enumerate(zip(fn_vals, fp_vals)):
    ax.text(i - w/2, fn + 500, f'{fn:,}', ha='center', fontsize=7)
    ax.text(i + w/2, fp + 500, f'{fp:,}', ha='center', fontsize=7)

# 3. Feature importance (top 25 of Model D)
ax = axes[1, 0]
top25 = importances_D.head(25)
colors_imp = ['#e74c3c' if f in tx_only_cols else '#3498db' for f in top25.index]
ax.barh(range(len(top25)), top25.values, color=colors_imp)
ax.set_yticks(range(len(top25)))
ax.set_yticklabels(top25.index, fontsize=7)
ax.invert_yaxis()
ax.set_title('Feature Importance (Top 25)\nRed = TX Features')
ax.set_xlabel('Importance')

# 4. TX feature importance only
ax = axes[1, 1]
tx_imp = importances_D[tx_only_cols].sort_values(ascending=True)
colors_tx = ['#e74c3c'] * len(tx_imp)
ax.barh(range(len(tx_imp)), tx_imp.values, color=colors_tx)
ax.set_yticks(range(len(tx_imp)))
ax.set_yticklabels(tx_imp.index, fontsize=9)
ax.set_title('TX Feature Importance')
ax.set_xlabel('Importance')

plt.suptitle(
    f'Phase 26: Transaction Data Features\n'
    f'Baseline AUC={baseline_auc:.4f} → Best AUC={best_auc:.4f} ({(best_auc-baseline_auc)*100:+.2f}%)',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig('charts/phase26_tx_features.png', dpi=150, bbox_inches='tight')
print(f"  Saved: charts/phase26_tx_features.png")


# ═══════════════════════════════════════════════════════════════
# Save best model with TX features (if improved)
# ═══════════════════════════════════════════════════════════════
if best_auc > baseline_auc:
    best_model = rf_D
    best_cols = merged_cols
    artifact = {
        'model': best_model,
        'feature_cols': best_cols,
        'tx_feature_cols': tx_only_cols,
        'version': 'v5_tx',
        'n_features': len(best_cols),
        'auc_roc': best_auc,
    }
    joblib.dump(artifact, 'models/churn_v5_tx.joblib')
    print(f"\n  Saved improved model: models/churn_v5_tx.joblib")
else:
    print(f"\n  TX features did not improve AUC — no model saved.")

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Phase 26: Transaction Data — Summary")
print(f"{'='*60}")
print(f"  TX files processed     : {len(tx_cache)}")
print(f"  TX features created    : {len(tx_only_cols)}")
print(f"  Train samples          : {len(y_train):,}")
print(f"  Test samples           : {len(y_test):,}")
print(f"  Baseline AUC (v3 45f)  : {baseline_auc:.4f}")
for name, r in results.items():
    if name != 'RF v3 pretrained (45)':
        delta = (r['auc'] - baseline_auc) * 100
        print(f"  {name:<25}: {r['auc']:.4f} ({delta:+.2f}%)")
print(f"  Best model             : {best_name}")
print(f"  Chart                  : charts/phase26_tx_features.png")
print(f"{'='*60}")
