#!/usr/bin/env python3
"""Phase 28: Full TX Training.

Train a churn prediction model using TX (transaction) features on the FULL
20-file training set, instead of the 1-2 files used in Phase 26.

Phase 26 proved TX features help: AUC went from 0.7981 → 0.8013, but trained
on only 1 file. Now we scale up to 20 files for a more robust model.
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
from feature_engineering import engineer_features_v2, aggregate_tx_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

MIN_TENURE = 3

print("=" * 70)
print("Phase 28: Full TX Training (20-file training set)")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# TX ↔ Churn mapping (20 train + 2 test)
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


def load_tx_features(tx_file):
    """Load TX file and aggregate into per-customer features."""
    df = pd.read_csv(tx_file, low_memory=False)
    tx_feats = aggregate_tx_features(df)
    return tx_feats


# ═══════════════════════════════════════════════════════════════
# 1. Aggregate ALL TX files
# ═══════════════════════════════════════════════════════════════
print("\n[1/7] Aggregating transaction data (22 files)...")
t0 = time.time()

tx_cache = {}
all_pairs = train_pairs + test_pairs
for tx_file, _ in all_pairs:
    if tx_file not in tx_cache:
        t_file = time.time()
        print(f"  {os.path.basename(tx_file)}...", end=" ", flush=True)
        tx_feats = load_tx_features(tx_file)
        tx_cache[tx_file] = tx_feats
        print(f"{len(tx_feats):,} customers, {tx_feats.shape[1]} features ({time.time()-t_file:.1f}s)")

print(f"  Total TX files cached: {len(tx_cache)}")
print(f"  Done ({time.time()-t0:.1f}s)")

# Show TX feature list
sample_tx = list(tx_cache.values())[0]
tx_feature_names = list(sample_tx.columns)
print(f"\n  TX features ({len(tx_feature_names)}): {tx_feature_names}")


# ═══════════════════════════════════════════════════════════════
# 2. Load & merge Churn + TX for each pair
# ═══════════════════════════════════════════════════════════════
print("\n[2/7] Loading Churn features & merging with TX...")
t0 = time.time()


def load_and_merge(pairs):
    """Load Churn + TX pairs, merge, return (X_base, X_merged, y)."""
    all_X_base, all_X_merged, all_y = [], [], []

    for tx_file, churn_file in pairs:
        print(f"  {os.path.basename(churn_file)}...", flush=True)
        feats, y, user_nos = load_churn_features(churn_file)

        # Get TX features for this period
        tx_feats = tx_cache[tx_file]

        # Merge by userNo (left join — keep all Churn customers)
        feats_with_user = feats.copy()
        feats_with_user['userNo'] = user_nos.values

        merged = feats_with_user.merge(
            tx_feats, left_on='userNo', right_index=True, how='left'
        )

        # Fill NaN for customers without TX data
        tx_cols = list(tx_feats.columns)
        merged[tx_cols] = merged[tx_cols].fillna(0)

        # Stats
        matched = (merged['tx_n_orders'] > 0).sum()
        print(f"    n={len(feats):,}, churn={y.mean()*100:.1f}%, "
              f"TX matched={matched:,} ({matched/len(feats)*100:.1f}%)")

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


# Load train (20 files)
print("\n  --- Train (20 files) ---")
X_train_base, X_train_merged, y_train = load_and_merge(train_pairs)
print(f"\n  Train total: {len(y_train):,}, churn={y_train.mean()*100:.1f}%")

# Load test (2 files)
print("\n  --- Test (2 files) ---")
X_test_base, X_test_merged, y_test = load_and_merge(test_pairs)
print(f"\n  Test total: {len(y_test):,}, churn={y_test.mean()*100:.1f}%")

print(f"\n  Done ({time.time()-t0:.1f}s)")
print(f"  Base features (v2): {X_train_base.shape[1]}")
print(f"  Merged features (v2+TX): {X_train_merged.shape[1]}")
print(f"  TX features added: {X_train_merged.shape[1] - X_train_base.shape[1]}")


# ═══════════════════════════════════════════════════════════════
# 3. Training data management
# ═══════════════════════════════════════════════════════════════
print("\n[3/7] Training data management...")

if len(X_train_merged) > 2_000_000:
    print(f"  Total rows ({len(X_train_merged):,}) > 2M, subsampling...")
    idx = np.random.RandomState(42).choice(len(X_train_merged), 2_000_000, replace=False)
    X_train_base = X_train_base.iloc[idx].reset_index(drop=True)
    X_train_merged = X_train_merged.iloc[idx].reset_index(drop=True)
    y_train = y_train[idx]
    print(f"  Subsampled to {len(y_train):,}")
else:
    print(f"  Total rows: {len(y_train):,} (under 2M, no subsampling needed)")

print(f"  Final train size: {len(y_train):,}, churn rate: {y_train.mean()*100:.1f}%")

# Identify feature columns
base_cols = list(X_train_base.columns)
merged_cols = list(X_train_merged.columns)
tx_only_cols = [c for c in merged_cols if c not in base_cols]

print(f"  Base cols: {len(base_cols)}")
print(f"  Merged cols: {len(merged_cols)}")
print(f"  TX cols: {tx_only_cols}")


# ═══════════════════════════════════════════════════════════════
# 4. Train 3 models and compare
# ═══════════════════════════════════════════════════════════════
print("\n[4/7] Training & evaluating models...")
t0 = time.time()

RF_PARAMS = dict(
    n_estimators=300, max_depth=20, min_samples_leaf=50,
    class_weight='balanced', random_state=42, n_jobs=-1,
)
THRESHOLD = 0.48

results = {}

# --- Model A: RF v3 pretrained (45 features, original model) ---
print("\n  [A] RF v3 pretrained (45 features) — evaluate only...")
t_a = time.time()
v3_artifact = joblib.load('models/churn_v3_extended.joblib')
v3_model = v3_artifact['model']
v3_feature_cols = v3_artifact['feature_cols']

# Match v3 feature columns to test base columns
v3_cols_avail = [c for c in v3_feature_cols if c in X_test_base.columns]
X_test_A = X_test_base[v3_cols_avail].fillna(0)
# Add missing v3 columns as zeros
for c in v3_feature_cols:
    if c not in X_test_A.columns:
        X_test_A[c] = 0
X_test_A = X_test_A[v3_feature_cols]

probs_A = v3_model.predict_proba(X_test_A)[:, 1]
preds_A = (probs_A >= THRESHOLD).astype(int)
results['RF v3 pretrained (45f)'] = {
    'auc': roc_auc_score(y_test, probs_A),
    'f1': f1_score(y_test, preds_A),
    'cm': confusion_matrix(y_test, preds_A),
}
print(f"      AUC={results['RF v3 pretrained (45f)']['auc']:.4f} ({time.time()-t_a:.1f}s)")

# --- Model B: v5 TX (56 features, trained on 1 file from Phase 26) ---
print("\n  [B] RF v5 TX (56f, 1-file train) — evaluate only...")
t_b = time.time()
v5_artifact = joblib.load('models/churn_v5_tx.joblib')
v5_model = v5_artifact['model']
v5_feature_cols = v5_artifact['feature_cols']

# Build test set matching v5 columns
X_test_B = X_test_merged.copy()
for c in v5_feature_cols:
    if c not in X_test_B.columns:
        X_test_B[c] = 0
X_test_B = X_test_B[v5_feature_cols].fillna(0)

probs_B = v5_model.predict_proba(X_test_B)[:, 1]
preds_B = (probs_B >= THRESHOLD).astype(int)
results['RF v5 TX (56f, 1-file)'] = {
    'auc': roc_auc_score(y_test, probs_B),
    'f1': f1_score(y_test, preds_B),
    'cm': confusion_matrix(y_test, preds_B),
}
print(f"      AUC={results['RF v5 TX (56f, 1-file)']['auc']:.4f} ({time.time()-t_b:.1f}s)")

# --- Model C: NEW RF (56 features, trained on full 20 files) ---
print("\n  [C] RF v7 TX (56f, 20-file train) — training...")
t_c = time.time()
rf_C = RandomForestClassifier(**RF_PARAMS)
rf_C.fit(X_train_merged.fillna(0), y_train)
probs_C = rf_C.predict_proba(X_test_merged.fillna(0))[:, 1]
preds_C = (probs_C >= THRESHOLD).astype(int)
results['RF v7 TX (56f, 20-file)'] = {
    'auc': roc_auc_score(y_test, probs_C),
    'f1': f1_score(y_test, preds_C),
    'cm': confusion_matrix(y_test, preds_C),
}
print(f"      AUC={results['RF v7 TX (56f, 20-file)']['auc']:.4f} ({time.time()-t_c:.1f}s)")

print(f"\n  All models evaluated ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════════════════════
# 5. Results table
# ═══════════════════════════════════════════════════════════════
print("\n[5/7] Results comparison...")

baseline_auc = results['RF v3 pretrained (45f)']['auc']

print(f"\n  {'='*80}")
print(f"  Model Comparison (Test: {len(y_test):,} customers, churn={y_test.mean()*100:.1f}%)")
print(f"  {'='*80}")
print(f"  {'Model':<30} {'AUC-ROC':>10} {'F1':>10} {'FN':>10} {'FP':>10} {'Δ AUC':>10}")
print(f"  {'-'*80}")

for name, r in results.items():
    fn = r['cm'][1, 0]
    fp = r['cm'][0, 1]
    delta = (r['auc'] - baseline_auc) * 100
    delta_str = f"{delta:+.2f}%" if name != 'RF v3 pretrained (45f)' else "baseline"
    print(f"  {name:<30} {r['auc']:>10.4f} {r['f1']:>10.4f} {fn:>10,} {fp:>10,} {delta_str:>10}")

print(f"  {'-'*80}")

best_name = max(results, key=lambda k: results[k]['auc'])
best_auc = results[best_name]['auc']
print(f"  Baseline AUC: {baseline_auc:.4f}")
print(f"  Best model:   {best_name} (AUC: {best_auc:.4f})")
print(f"  Improvement:  {(best_auc - baseline_auc)*100:+.2f}%")


# ═══════════════════════════════════════════════════════════════
# 6. Feature importance for Model C
# ═══════════════════════════════════════════════════════════════
print("\n[6/7] Feature importance (Model C: v7 full TX)...")

importances_C = pd.Series(rf_C.feature_importances_, index=merged_cols).sort_values(ascending=False)

print(f"\n  TX Feature Importance (rank / {len(merged_cols)}):")
for feat in tx_only_cols:
    rank = importances_C.index.tolist().index(feat) + 1
    print(f"    {feat:<25} importance={importances_C[feat]:.4f}  rank={rank}/{len(merged_cols)}")

print(f"\n  Top 10 overall features:")
for i, (feat, imp) in enumerate(importances_C.head(10).items()):
    tag = " [TX]" if feat in tx_only_cols else ""
    print(f"    {i+1:>2}. {feat:<30} {imp:.4f}{tag}")


# ═══════════════════════════════════════════════════════════════
# 7. FN Analysis
# ═══════════════════════════════════════════════════════════════
print("\n  FN Analysis:")

fn_A = set(np.where((y_test == 1) & (preds_A == 0))[0])
fn_B = set(np.where((y_test == 1) & (preds_B == 0))[0])
fn_C = set(np.where((y_test == 1) & (preds_C == 0))[0])

print(f"    [A] v3 pretrained   FN: {len(fn_A):,}")
print(f"    [B] v5 TX (1-file)  FN: {len(fn_B):,}")
print(f"    [C] v7 TX (20-file) FN: {len(fn_C):,}")
print(f"    FN saved (A→C): {len(fn_A) - len(fn_C):+,}")
print(f"    FN saved (B→C): {len(fn_B) - len(fn_C):+,}")

fn_saved_vs_phase26 = fn_B - fn_C
fn_new_vs_phase26 = fn_C - fn_B
print(f"\n    Full training vs Phase 26 (1-file):")
print(f"      FN saved by full training: {len(fn_saved_vs_phase26):,}")
print(f"      New FN from full training: {len(fn_new_vs_phase26):,}")
print(f"      Net FN change: {len(fn_B) - len(fn_C):+,}")


# ═══════════════════════════════════════════════════════════════
# 8. Charts
# ═══════════════════════════════════════════════════════════════
print("\n  Generating charts...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- [0,0] AUC bar chart (3 models) ---
ax = axes[0, 0]
names = list(results.keys())
aucs = [results[n]['auc'] for n in names]
short_names = ['v3 (45f)', 'v5 TX 1-file (56f)', 'v7 TX 20-file (56f)']
colors = ['#3498db', '#e67e22', '#e74c3c']
bars = ax.bar(range(len(names)), aucs, color=colors)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(short_names, fontsize=9)
ax.set_ylabel('AUC-ROC')
ax.set_title('AUC-ROC Comparison')
ax.set_ylim(min(aucs) - 0.005, max(aucs) + 0.005)
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f'{auc:.4f}', ha='center', fontsize=10, fontweight='bold')

# --- [0,1] FN/FP comparison ---
ax = axes[0, 1]
fn_vals = [results[n]['cm'][1, 0] for n in names]
fp_vals = [results[n]['cm'][0, 1] for n in names]
x = np.arange(len(names))
w = 0.35
bars_fn = ax.bar(x - w/2, fn_vals, w, label='FN (missed churn)', color='#e74c3c')
bars_fp = ax.bar(x + w/2, fp_vals, w, label='FP (false alarm)', color='#f39c12')
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=9)
ax.set_title('FN / FP Comparison')
ax.legend()
ax.set_ylabel('Count')
for i, (fn, fp) in enumerate(zip(fn_vals, fp_vals)):
    ax.text(i - w/2, fn + max(fn_vals) * 0.02, f'{fn:,}', ha='center', fontsize=8)
    ax.text(i + w/2, fp + max(fp_vals) * 0.02, f'{fp:,}', ha='center', fontsize=8)

# --- [1,0] Feature importance top 25 (TX features in orange) ---
ax = axes[1, 0]
top25 = importances_C.head(25)
colors_imp = ['#e67e22' if f in tx_only_cols else '#3498db' for f in top25.index]
ax.barh(range(len(top25)), top25.values, color=colors_imp)
ax.set_yticks(range(len(top25)))
ax.set_yticklabels(top25.index, fontsize=7)
ax.invert_yaxis()
ax.set_title('Feature Importance (Top 25)\nOrange = TX Features')
ax.set_xlabel('Importance')

# --- [1,1] Training data size impact (1 file vs 20 files) ---
ax = axes[1, 1]
# Compare v5 (1-file) vs v7 (20-file) across metrics
metrics = ['AUC-ROC', 'F1', 'FN (÷1000)', 'FP (÷1000)']
v5_r = results['RF v5 TX (56f, 1-file)']
v7_r = results['RF v7 TX (56f, 20-file)']
v5_vals = [v5_r['auc'], v5_r['f1'], v5_r['cm'][1, 0] / 1000, v5_r['cm'][0, 1] / 1000]
v7_vals = [v7_r['auc'], v7_r['f1'], v7_r['cm'][1, 0] / 1000, v7_r['cm'][0, 1] / 1000]
x_m = np.arange(len(metrics))
w_m = 0.35
bars1 = ax.bar(x_m - w_m/2, v5_vals, w_m, label='v5 TX (1-file)', color='#e67e22')
bars2 = ax.bar(x_m + w_m/2, v7_vals, w_m, label='v7 TX (20-file)', color='#e74c3c')
ax.set_xticks(x_m)
ax.set_xticklabels(metrics, fontsize=9)
ax.set_title('Training Data Size Impact\n1 file vs 20 files')
ax.legend()
for bar, val in zip(bars1, v5_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', fontsize=8)
for bar, val in zip(bars2, v7_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', fontsize=8)

plt.suptitle(
    f'Phase 28: Full TX Training (20 files)\n'
    f'Baseline AUC={baseline_auc:.4f} → Best AUC={best_auc:.4f} '
    f'({(best_auc-baseline_auc)*100:+.2f}%)',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig('charts/phase28_full_tx.png', dpi=150, bbox_inches='tight')
print(f"  Saved: charts/phase28_full_tx.png")


# ═══════════════════════════════════════════════════════════════
# 9. Save model
# ═══════════════════════════════════════════════════════════════
print("\n  Saving model...")
artifact = {
    'model': rf_C,
    'feature_cols': merged_cols,
    'tx_feature_cols': tx_only_cols,
    'version': 'v7_full_tx',
    'n_features': len(merged_cols),
    'n_train_files': 20,
    'n_train_samples': len(y_train),
    'auc_roc': results['RF v7 TX (56f, 20-file)']['auc'],
}
joblib.dump(artifact, 'models/churn_v7_full_tx.joblib')
print(f"  Saved: models/churn_v7_full_tx.joblib")


# ═══════════════════════════════════════════════════════════════
# 10. Summary
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"Phase 28: Full TX Training — Summary")
print(f"{'='*70}")
print(f"  TX files processed        : {len(tx_cache)}")
print(f"  TX features               : {len(tx_only_cols)}")
print(f"  Train files               : 20")
print(f"  Train samples             : {len(y_train):,}")
print(f"  Test files                : 2")
print(f"  Test samples              : {len(y_test):,}")
print(f"  {'─'*50}")
print(f"  [A] v3 pretrained (45f)   : AUC {results['RF v3 pretrained (45f)']['auc']:.4f} (baseline)")
v5_delta = (results['RF v5 TX (56f, 1-file)']['auc'] - baseline_auc) * 100
v7_delta = (results['RF v7 TX (56f, 20-file)']['auc'] - baseline_auc) * 100
print(f"  [B] v5 TX 1-file (56f)    : AUC {results['RF v5 TX (56f, 1-file)']['auc']:.4f} ({v5_delta:+.2f}%)")
print(f"  [C] v7 TX 20-file (56f)   : AUC {results['RF v7 TX (56f, 20-file)']['auc']:.4f} ({v7_delta:+.2f}%)")
print(f"  {'─'*50}")
print(f"  Best model                : {best_name}")
print(f"  FN saved (baseline→best)  : {len(fn_A) - len(fn_C):+,}")
print(f"  FN saved (1-file→20-file) : {len(fn_B) - len(fn_C):+,}")
print(f"  Chart                     : charts/phase28_full_tx.png")
print(f"  Model                     : models/churn_v7_full_tx.joblib")
print(f"{'='*70}")
