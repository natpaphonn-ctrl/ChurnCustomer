#!/usr/bin/env python3
"""Phase 27: Cross-Period TX Features & Model Comparison.

Compares transaction behavior across consecutive periods (e.g., basket size
declining, fewer orders) and tests whether these cross-period signals improve
churn prediction beyond single-period TX features.

Models are saved separately for comparison — no existing models are overwritten.
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
    engineer_features_v2, aggregate_tx_features, compute_cross_period_tx
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

MIN_TENURE = 3
RF_PARAMS = dict(
    n_estimators=300, max_depth=20, min_samples_leaf=50,
    class_weight='balanced', random_state=42, n_jobs=-1,
)
THRESHOLD = 0.48

print("=" * 60)
print("Phase 27: Cross-Period TX Features")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# Cross-period mapping:
#   Churn file needs CURRENT TX + PREVIOUS TX
# ═══════════════════════════════════════════════════════════════
cross_period_data = {
    # churn_file: (tx_current, tx_previous)
    'Churn_2026_0117_0201.csv': (
        'TransactionOrder/Transaction_Order_2026_0117.csv',
        'TransactionOrder/Transaction_Order_2026_0102.csv',
    ),
    'Churn_2026_0201_0216.csv': (
        'TransactionOrder/Transaction_Order_2026_0201.csv',
        'TransactionOrder/Transaction_Order_2026_0117.csv',
    ),
    'Churn_2026_0216_0301.csv': (
        'TransactionOrder/Transaction_Order_2026_0216.csv',
        'TransactionOrder/Transaction_Order_2026_0201.csv',
    ),
}

train_files = ['Churn_2026_0117_0201.csv']
test_files = ['Churn_2026_0201_0216.csv', 'Churn_2026_0216_0301.csv']


# ═══════════════════════════════════════════════════════════════
# 1. Aggregate TX data (all 4 unique TX files)
# ═══════════════════════════════════════════════════════════════
print("\n[1/5] Aggregating transaction data...")
t0 = time.time()

tx_files_needed = set()
for churn_f, (tx_curr, tx_prev) in cross_period_data.items():
    tx_files_needed.add(tx_curr)
    tx_files_needed.add(tx_prev)

tx_cache = {}
for tx_file in sorted(tx_files_needed):
    print(f"  {os.path.basename(tx_file)}...", end=" ", flush=True)
    tx_feats = aggregate_tx_features(pd.read_csv(tx_file, low_memory=False))
    tx_cache[tx_file] = tx_feats
    print(f"{len(tx_feats):,} customers")

print(f"  Done ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════════════════════
# 2. Load Churn features + merge TX single + cross-period
# ═══════════════════════════════════════════════════════════════
print("\n[2/5] Loading Churn features & computing cross-period TX...")
t0 = time.time()


def load_churn_features(churn_file):
    """Load Churn CSV, compute v2 features, return (feats, y, userNos)."""
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


def build_dataset(file_list):
    """Build merged dataset with base + TX single + TX cross features."""
    all_base, all_single, all_cross, all_full, all_y = [], [], [], [], []

    for churn_file in file_list:
        print(f"  {churn_file}...")
        feats, y, user_nos = load_churn_features(churn_file)

        tx_curr_file, tx_prev_file = cross_period_data[churn_file]
        tx_curr = tx_cache[tx_curr_file]
        tx_prev = tx_cache[tx_prev_file]

        # Compute cross-period features
        cross_feats = compute_cross_period_tx(tx_curr, tx_prev)

        # Merge by userNo
        feats_u = feats.copy()
        feats_u['userNo'] = user_nos.values

        # Left join single-period TX
        merged = feats_u.merge(tx_curr, left_on='userNo', right_index=True, how='left')
        # Left join cross-period TX
        merged = merged.merge(cross_feats, left_on='userNo', right_index=True, how='left')

        tx_single_cols = [c for c in tx_curr.columns]
        tx_cross_cols = [c for c in cross_feats.columns]
        merged[tx_single_cols + tx_cross_cols] = merged[tx_single_cols + tx_cross_cols].fillna(0)

        # Match stats
        matched_single = (merged['tx_n_orders'] > 0).sum()
        matched_cross = (merged['txc_existed_prev'] > 0).sum()
        print(f"    n={len(feats):,}, churn={y.mean()*100:.1f}%, "
              f"TX matched={matched_single:,} ({matched_single/len(feats)*100:.0f}%), "
              f"cross matched={matched_cross:,} ({matched_cross/len(feats)*100:.0f}%)")

        base_cols = [c for c in feats.columns]
        X_base = merged[base_cols].copy()
        X_single = merged[base_cols + tx_single_cols].copy()
        X_cross_only = merged[base_cols + tx_cross_cols].copy()
        X_full = merged[base_cols + tx_single_cols + tx_cross_cols].copy()

        all_base.append(X_base)
        all_single.append(X_single)
        all_cross.append(X_cross_only)
        all_full.append(X_full)
        all_y.append(y)

    return {
        'base': pd.concat(all_base, ignore_index=True),
        'single': pd.concat(all_single, ignore_index=True),
        'cross_only': pd.concat(all_cross, ignore_index=True),
        'full': pd.concat(all_full, ignore_index=True),
        'y': np.concatenate(all_y),
    }


print("\n  --- Train ---")
train_data = build_dataset(train_files)
print(f"  Train total: {len(train_data['y']):,}, churn={train_data['y'].mean()*100:.1f}%")

print("\n  --- Test ---")
test_data = build_dataset(test_files)
print(f"  Test total: {len(test_data['y']):,}, churn={test_data['y'].mean()*100:.1f}%")

print(f"\n  Feature counts:")
print(f"    Base (v2):           {train_data['base'].shape[1]}")
print(f"    + TX single:         {train_data['single'].shape[1]}")
print(f"    + TX cross only:     {train_data['cross_only'].shape[1]}")
print(f"    Full (all):          {train_data['full'].shape[1]}")
print(f"  Done ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════════════════════
# 3. Train & evaluate all models
# ═══════════════════════════════════════════════════════════════
print("\n[3/5] Training models...")
t0 = time.time()

y_train = train_data['y']
y_test = test_data['y']

results = {}

# --- Model A: v3 pretrained baseline (45 features) ---
print("  [A] RF v3 pretrained (45 features)...")
v3_artifact = joblib.load('models/churn_v3_extended.joblib')
v3_model = v3_artifact['model']
v3_cols = v3_artifact['feature_cols']
v3_test_cols = [c for c in v3_cols if c in test_data['base'].columns]
probs_A = v3_model.predict_proba(test_data['base'][v3_test_cols].fillna(0))[:, 1]
preds_A = (probs_A >= THRESHOLD).astype(int)
results['A: v3 pretrained (45)'] = {
    'auc': roc_auc_score(y_test, probs_A),
    'f1': f1_score(y_test, preds_A),
    'cm': confusion_matrix(y_test, preds_A),
    'probs': probs_A, 'preds': preds_A,
}

# --- Model B: v2+TX single (56 features, same as Phase 26) ---
print("  [B] RF v2+TX single (56 features)...")
rf_B = RandomForestClassifier(**RF_PARAMS)
rf_B.fit(train_data['single'].fillna(0), y_train)
probs_B = rf_B.predict_proba(test_data['single'].fillna(0))[:, 1]
preds_B = (probs_B >= THRESHOLD).astype(int)
results['B: v2+TX single (56)'] = {
    'auc': roc_auc_score(y_test, probs_B),
    'f1': f1_score(y_test, preds_B),
    'cm': confusion_matrix(y_test, preds_B),
    'probs': probs_B, 'preds': preds_B,
}

# --- Model C: v2+TX cross only (55 features) ---
print("  [C] RF v2+TX cross only (55 features)...")
rf_C = RandomForestClassifier(**RF_PARAMS)
rf_C.fit(train_data['cross_only'].fillna(0), y_train)
probs_C = rf_C.predict_proba(test_data['cross_only'].fillna(0))[:, 1]
preds_C = (probs_C >= THRESHOLD).astype(int)
results['C: v2+CrossTX only (55)'] = {
    'auc': roc_auc_score(y_test, probs_C),
    'f1': f1_score(y_test, preds_C),
    'cm': confusion_matrix(y_test, preds_C),
    'probs': probs_C, 'preds': preds_C,
}

# --- Model D: v2+TX single+cross (66 features) ---
print("  [D] RF v2+TX+CrossTX (66 features)...")
rf_D = RandomForestClassifier(**RF_PARAMS)
rf_D.fit(train_data['full'].fillna(0), y_train)
probs_D = rf_D.predict_proba(test_data['full'].fillna(0))[:, 1]
preds_D = (probs_D >= THRESHOLD).astype(int)
results['D: v2+TX+CrossTX (66)'] = {
    'auc': roc_auc_score(y_test, probs_D),
    'f1': f1_score(y_test, preds_D),
    'cm': confusion_matrix(y_test, preds_D),
    'probs': probs_D, 'preds': preds_D,
}

print(f"  Done ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════════════════════
# 4. Results comparison
# ═══════════════════════════════════════════════════════════════
print(f"\n[4/5] Results comparison...")
print(f"\n  {'='*76}")
print(f"  Model Comparison (Test: {len(y_test):,} customers)")
print(f"  {'='*76}")
print(f"  {'Model':<30} {'AUC-ROC':>10} {'F1':>10} {'FN':>10} {'FP':>10}")
print(f"  {'-'*76}")

baseline_auc = results['A: v3 pretrained (45)']['auc']

for name, r in results.items():
    fn = r['cm'][1, 0]
    fp = r['cm'][0, 1]
    delta = (r['auc'] - baseline_auc) * 100
    marker = " ***" if r['auc'] == max(rr['auc'] for rr in results.values()) else ""
    print(f"  {name:<30} {r['auc']:>10.4f} {r['f1']:>10.4f} {fn:>10,} {fp:>10,}  ({delta:+.2f}%){marker}")

print(f"  {'-'*76}")

# Best model
best_name = max(results, key=lambda k: results[k]['auc'])
best_auc = results[best_name]['auc']
print(f"  Best model: {best_name}")
print(f"  Improvement over baseline: {(best_auc - baseline_auc)*100:+.2f}%")

# Cross-period incremental value
single_auc = results['B: v2+TX single (56)']['auc']
full_auc = results['D: v2+TX+CrossTX (66)']['auc']
print(f"\n  Cross-period incremental value:")
print(f"    TX single only AUC:      {single_auc:.4f}")
print(f"    TX single + cross AUC:   {full_auc:.4f}")
print(f"    Cross-period added:      {(full_auc - single_auc)*100:+.2f}%")

# FN analysis
fn_A = set(np.where((y_test == 1) & (preds_A == 0))[0])
fn_D = set(np.where((y_test == 1) & (preds_D == 0))[0])
fn_B = set(np.where((y_test == 1) & (preds_B == 0))[0])
print(f"\n  FN Analysis:")
print(f"    v3 baseline FN:     {len(fn_A):,}")
print(f"    TX single FN:       {len(fn_B):,}")
print(f"    TX+cross FN:        {len(fn_D):,}")
print(f"    Net FN saved (vs baseline): {len(fn_A) - len(fn_D):+,}")
print(f"    FN saved by cross (vs single): {len(fn_B) - len(fn_D):+,}")


# ═══════════════════════════════════════════════════════════════
# 5. Feature importance & charts
# ═══════════════════════════════════════════════════════════════
print(f"\n[5/5] Feature importance & charts...")

full_cols = list(train_data['full'].columns)
importances = pd.Series(rf_D.feature_importances_, index=full_cols).sort_values(ascending=False)

tx_single_cols = [c for c in full_cols if c.startswith('tx_')]
tx_cross_cols = [c for c in full_cols if c.startswith('txc_')]

print(f"\n  Cross-Period Feature Importance (rank / {len(full_cols)}):")
for feat in tx_cross_cols:
    rank = importances.index.tolist().index(feat) + 1
    print(f"    {feat:<25} importance={importances[feat]:.4f}  rank={rank}/{len(full_cols)}")

print(f"\n  Single-Period TX Feature Importance:")
for feat in tx_single_cols:
    rank = importances.index.tolist().index(feat) + 1
    print(f"    {feat:<25} importance={importances[feat]:.4f}  rank={rank}/{len(full_cols)}")

# --- Charts ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. AUC comparison
ax = axes[0, 0]
names_short = ['v3 (45)', 'v2+TX (56)', 'v2+CrossTX (55)', 'v2+TX+Cross (66)']
aucs = [results[n]['auc'] for n in results]
colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c']
bars = ax.bar(range(len(aucs)), aucs, color=colors)
ax.set_xticks(range(len(aucs)))
ax.set_xticklabels(names_short, fontsize=9, rotation=10)
ax.set_ylabel('AUC-ROC')
ax.set_title('AUC-ROC Comparison')
ax.set_ylim(min(aucs) - 0.005, max(aucs) + 0.005)
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
            f'{auc:.4f}', ha='center', fontsize=9, fontweight='bold')

# 2. FN/FP comparison
ax = axes[0, 1]
fn_vals = [results[n]['cm'][1, 0] for n in results]
fp_vals = [results[n]['cm'][0, 1] for n in results]
x = np.arange(len(results))
w = 0.35
ax.bar(x - w/2, fn_vals, w, label='FN (missed churn)', color='#e74c3c')
ax.bar(x + w/2, fp_vals, w, label='FP (false alarm)', color='#f39c12')
ax.set_xticks(x)
ax.set_xticklabels(names_short, fontsize=9, rotation=10)
ax.set_title('Error Comparison')
ax.legend()
for i, (fn, fp) in enumerate(zip(fn_vals, fp_vals)):
    ax.text(i - w/2, fn + 500, f'{fn:,}', ha='center', fontsize=7)
    ax.text(i + w/2, fp + 500, f'{fp:,}', ha='center', fontsize=7)

# 3. Top 25 feature importance
ax = axes[1, 0]
top25 = importances.head(25)
colors_imp = []
for f in top25.index:
    if f.startswith('txc_'):
        colors_imp.append('#e74c3c')  # cross = red
    elif f.startswith('tx_'):
        colors_imp.append('#e67e22')  # single TX = orange
    else:
        colors_imp.append('#3498db')  # base = blue
ax.barh(range(len(top25)), top25.values, color=colors_imp)
ax.set_yticks(range(len(top25)))
ax.set_yticklabels(top25.index, fontsize=7)
ax.invert_yaxis()
ax.set_title('Feature Importance (Top 25)\nRed=Cross-TX  Orange=Single-TX  Blue=Base')
ax.set_xlabel('Importance')

# 4. TX features only (single + cross)
ax = axes[1, 1]
all_tx = importances[tx_single_cols + tx_cross_cols].sort_values(ascending=True)
colors_tx = ['#e74c3c' if f.startswith('txc_') else '#e67e22' for f in all_tx.index]
ax.barh(range(len(all_tx)), all_tx.values, color=colors_tx)
ax.set_yticks(range(len(all_tx)))
ax.set_yticklabels(all_tx.index, fontsize=8)
ax.set_title('All TX Feature Importance\nRed=Cross-Period  Orange=Single-Period')
ax.set_xlabel('Importance')

plt.suptitle(
    f'Phase 27: Cross-Period TX Features\n'
    f'Baseline AUC={baseline_auc:.4f} → Best AUC={best_auc:.4f} ({(best_auc-baseline_auc)*100:+.2f}%)',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig('charts/phase27_cross_tx.png', dpi=150, bbox_inches='tight')
print(f"  Saved: charts/phase27_cross_tx.png")


# ═══════════════════════════════════════════════════════════════
# Save model
# ═══════════════════════════════════════════════════════════════
artifact = {
    'model': rf_D,
    'feature_cols': full_cols,
    'tx_single_cols': tx_single_cols,
    'tx_cross_cols': tx_cross_cols,
    'version': 'v6_cross_tx',
    'n_features': len(full_cols),
    'auc_roc': results['D: v2+TX+CrossTX (66)']['auc'],
}
joblib.dump(artifact, 'models/churn_v6_cross_tx.joblib')
print(f"  Saved: models/churn_v6_cross_tx.joblib")

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Phase 27: Cross-Period TX Features — Summary")
print(f"{'='*60}")
print(f"  Cross-period features  : {len(tx_cross_cols)}")
print(f"  Single-period features : {len(tx_single_cols)}")
print(f"  Base features          : {train_data['base'].shape[1]}")
print(f"  Total features (full)  : {len(full_cols)}")
print(f"  Train samples          : {len(y_train):,}")
print(f"  Test samples           : {len(y_test):,}")
print(f"  Baseline AUC           : {baseline_auc:.4f}")
print(f"  Best AUC               : {best_auc:.4f} ({(best_auc-baseline_auc)*100:+.2f}%)")
print(f"  Cross incremental      : {(full_auc - single_auc)*100:+.2f}%")
print(f"  Chart                  : charts/phase27_cross_tx.png")
print(f"  Model                  : models/churn_v6_cross_tx.joblib")
print(f"{'='*60}")
