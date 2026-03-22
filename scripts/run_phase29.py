#!/usr/bin/env python3
"""Phase 29: Multi-Period TX Trend Features & Model Comparison.

Computes 3-period TX trend features (declining baskets, order consistency,
etc.) and tests whether these multi-period signals improve churn prediction
beyond single-period TX features.

Models compared:
  A: v3 pretrained (45 features)
  B: v7 full TX (56 features) — loaded or trained inline
  C: NEW v8 (64 features = 45 base + 11 TX + 8 trend)
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
    engineer_features_v2, aggregate_tx_features, compute_multi_period_tx
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
print("Phase 29: Multi-Period TX Trend Features")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# Period order for TX files
# ═══════════════════════════════════════════════════════════════
period_order = [
    '2025_0102', '2025_0117', '2025_0201', '2025_0216', '2025_0301', '2025_0316',
    '2025_0401', '2025_0416', '2025_0502', '2025_0516', '2025_0601', '2025_0616',
    '2025_0701', '2025_0716', '2025_0801', '2025_0816', '2025_0901', '2025_0916',
    '2025_1001', '2025_1016', '2025_1101', '2025_1116', '2025_1201', '2025_1216',
    '2026_0102', '2026_0117', '2026_0201', '2026_0216', '2026_0301', '2026_0316',
]

period_idx = {p: i for i, p in enumerate(period_order)}


def tx_path(period):
    """Build TX file path from period string like '2025_0102'."""
    return f'TransactionOrder/Transaction_Order_{period}.csv'


def get_3period_chain(churn_period):
    """Given a Churn file's first date (YYYY_MMDD), return [tx_t-2, tx_t-1, tx_t0].
    Returns None if insufficient history."""
    idx = period_idx.get(churn_period)
    if idx is None or idx < 2:
        return None
    return [period_order[idx - 2], period_order[idx - 1], period_order[idx]]


# ═══════════════════════════════════════════════════════════════
# Training files: Churn_2025_0216_0301 through Churn_2025_1201_1216
# Test files: Churn_2026_0201_0216 + Churn_2026_0216_0301
# ═══════════════════════════════════════════════════════════════
all_train_churn = [
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
]
test_churn = ['Churn_2026_0201_0216.csv', 'Churn_2026_0216_0301.csv']


def churn_to_period(churn_file):
    """Extract first date period (YYYY_MMDD) from Churn filename."""
    # Churn_2025_0216_0301.csv → 2025_0216
    parts = os.path.basename(churn_file).replace('.csv', '').split('_')
    return f'{parts[1]}_{parts[2]}'


# Determine which train files have full 3-period chains
train_with_chain = []
train_no_chain = []
for f in all_train_churn:
    p = churn_to_period(f)
    chain = get_3period_chain(p)
    if chain is not None:
        train_with_chain.append(f)
    else:
        train_no_chain.append(f)

print(f"\n  Train files with 3-period chain: {len(train_with_chain)}")
print(f"  Train files without chain (skipped for trend): {len(train_no_chain)}: {train_no_chain}")

# Use only files with full chains for training
train_files = train_with_chain


# ═══════════════════════════════════════════════════════════════
# 1. Aggregate ALL TX files needed (cache)
# ═══════════════════════════════════════════════════════════════
print("\n[1/6] Aggregating transaction data...")
t0 = time.time()

# Collect all TX periods needed
tx_periods_needed = set()
for churn_file in train_files + test_churn:
    p = churn_to_period(churn_file)
    chain = get_3period_chain(p)
    if chain:
        for pp in chain:
            tx_periods_needed.add(pp)
    else:
        # Still need current TX for single-period features
        tx_periods_needed.add(p)

tx_cache = {}
for period in sorted(tx_periods_needed):
    path = tx_path(period)
    if os.path.exists(path):
        print(f"  {os.path.basename(path)}...", end=" ", flush=True)
        tx_feats = aggregate_tx_features(pd.read_csv(path, low_memory=False))
        tx_cache[period] = tx_feats
        print(f"{len(tx_feats):,} customers")
    else:
        print(f"  {os.path.basename(path)} NOT FOUND — skipping")

print(f"  Aggregated {len(tx_cache)} TX files ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════════════════════
# 2. Load Churn features + merge TX single + trend
# ═══════════════════════════════════════════════════════════════
print("\n[2/6] Loading Churn features & computing multi-period TX trends...")
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
    """Build merged dataset with base (45f) + TX single (11f) + TX trend (8f) = 64f."""
    all_X_base, all_X_single, all_X_full, all_y = [], [], [], []

    for churn_file in file_list:
        print(f"  {churn_file}...")
        feats, y, user_nos = load_churn_features(churn_file)

        p = churn_to_period(churn_file)
        chain = get_3period_chain(p)

        # Current TX features (single period)
        tx_curr = tx_cache.get(p, pd.DataFrame())

        # 3-period trend features
        if chain and all(pp in tx_cache for pp in chain):
            tx_agg_list = [tx_cache[chain[0]], tx_cache[chain[1]], tx_cache[chain[2]]]
            trend_feats = compute_multi_period_tx(tx_agg_list)
        else:
            # No chain available — empty trend features
            trend_feats = pd.DataFrame()

        # Merge by userNo
        feats_u = feats.copy()
        feats_u['userNo'] = user_nos.values

        # Left join single-period TX
        if len(tx_curr) > 0:
            merged = feats_u.merge(tx_curr, left_on='userNo', right_index=True, how='left')
        else:
            merged = feats_u.copy()

        # Left join trend TX
        if len(trend_feats) > 0:
            merged = merged.merge(trend_feats, left_on='userNo', right_index=True, how='left')

        # Identify column groups
        base_cols = [c for c in feats.columns]
        tx_single_cols = [c for c in tx_curr.columns] if len(tx_curr) > 0 else []
        tx_trend_cols = [c for c in trend_feats.columns] if len(trend_feats) > 0 else []

        # Fill NaN for unmatched
        for cols in [tx_single_cols, tx_trend_cols]:
            if cols:
                merged[cols] = merged[cols].fillna(0)

        # Match stats
        matched_single = (merged['tx_n_orders'] > 0).sum() if 'tx_n_orders' in merged.columns else 0
        matched_trend = (merged['txt_periods_present'] > 0).sum() if 'txt_periods_present' in merged.columns else 0
        print(f"    n={len(feats):,}, churn={y.mean()*100:.1f}%, "
              f"TX matched={matched_single:,} ({matched_single/len(feats)*100:.0f}%), "
              f"trend matched={matched_trend:,} ({matched_trend/len(feats)*100:.0f}%)")

        X_base = merged[base_cols].copy()
        X_single = merged[base_cols + tx_single_cols].copy() if tx_single_cols else merged[base_cols].copy()
        X_full = merged[base_cols + tx_single_cols + tx_trend_cols].copy()

        all_X_base.append(X_base)
        all_X_single.append(X_single)
        all_X_full.append(X_full)
        all_y.append(y)

    return {
        'base': pd.concat(all_X_base, ignore_index=True),
        'single': pd.concat(all_X_single, ignore_index=True),
        'full': pd.concat(all_X_full, ignore_index=True),
        'y': np.concatenate(all_y),
    }


print("\n  --- Train ---")
train_data = build_dataset(train_files)
y_train = train_data['y']
print(f"  Train total: {len(y_train):,}, churn={y_train.mean()*100:.1f}%")

# Subsample if too large
MAX_TRAIN = 2_000_000
if len(y_train) > MAX_TRAIN:
    rng = np.random.RandomState(42)
    idx = rng.choice(len(y_train), MAX_TRAIN, replace=False)
    train_data['base'] = train_data['base'].iloc[idx].reset_index(drop=True)
    train_data['single'] = train_data['single'].iloc[idx].reset_index(drop=True)
    train_data['full'] = train_data['full'].iloc[idx].reset_index(drop=True)
    y_train = y_train[idx]
    train_data['y'] = y_train
    print(f"  Subsampled to {len(y_train):,}")

print("\n  --- Test ---")
test_data = build_dataset(test_churn)
y_test = test_data['y']
print(f"  Test total: {len(y_test):,}, churn={y_test.mean()*100:.1f}%")

print(f"\n  Feature counts:")
print(f"    Base (v2):           {train_data['base'].shape[1]}")
print(f"    + TX single:         {train_data['single'].shape[1]}")
print(f"    Full (base+TX+trend):{train_data['full'].shape[1]}")
print(f"  Done ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════════════════════
# 3. Train & evaluate models
# ═══════════════════════════════════════════════════════════════
print("\n[3/6] Training models...")
t0 = time.time()

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

# --- Model B: v7 full TX (56 features) ---
print("  [B] RF v7 full TX (56 features)...")
v7_path = 'models/churn_v7_full_tx.joblib'
if os.path.exists(v7_path):
    v7_artifact = joblib.load(v7_path)
    v7_model = v7_artifact['model']
    v7_cols = v7_artifact['feature_cols']
    v7_test_cols = [c for c in v7_cols if c in test_data['single'].columns]
    # Ensure column alignment
    X_test_B = test_data['single'].reindex(columns=v7_cols, fill_value=0).fillna(0)
    probs_B = v7_model.predict_proba(X_test_B)[:, 1]
else:
    print("    v7 not found — training inline with 56 features (45 base + 11 TX)...")
    rf_B = RandomForestClassifier(**RF_PARAMS)
    rf_B.fit(train_data['single'].fillna(0), y_train)
    probs_B = rf_B.predict_proba(test_data['single'].fillna(0))[:, 1]
preds_B = (probs_B >= THRESHOLD).astype(int)
results['B: v7 full TX (56)'] = {
    'auc': roc_auc_score(y_test, probs_B),
    'f1': f1_score(y_test, preds_B),
    'cm': confusion_matrix(y_test, preds_B),
    'probs': probs_B, 'preds': preds_B,
}

# --- Model C: NEW v8 (64 features = 45 base + 11 TX + 8 trend) ---
print("  [C] RF v8 NEW (64 features = 45 base + 11 TX + 8 trend)...")
rf_C = RandomForestClassifier(**RF_PARAMS)
rf_C.fit(train_data['full'].fillna(0), y_train)
probs_C = rf_C.predict_proba(test_data['full'].fillna(0))[:, 1]
preds_C = (probs_C >= THRESHOLD).astype(int)
results['C: v8 NEW (64)'] = {
    'auc': roc_auc_score(y_test, probs_C),
    'f1': f1_score(y_test, preds_C),
    'cm': confusion_matrix(y_test, preds_C),
    'probs': probs_C, 'preds': preds_C,
}

print(f"  Done ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════════════════════
# 4. Results comparison
# ═══════════════════════════════════════════════════════════════
print(f"\n[4/6] Results comparison...")
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

best_name = max(results, key=lambda k: results[k]['auc'])
best_auc = results[best_name]['auc']
print(f"  Best model: {best_name}")
print(f"  Improvement over baseline: {(best_auc - baseline_auc)*100:+.2f}%")

# Trend incremental value
single_auc = results['B: v7 full TX (56)']['auc']
full_auc = results['C: v8 NEW (64)']['auc']
print(f"\n  Trend incremental value:")
print(f"    TX single only AUC:       {single_auc:.4f}")
print(f"    TX single + trend AUC:    {full_auc:.4f}")
print(f"    Trend features added:     {(full_auc - single_auc)*100:+.2f}%")

# FN analysis
fn_A = set(np.where((y_test == 1) & (preds_A == 0))[0])
fn_B = set(np.where((y_test == 1) & (preds_B == 0))[0])
fn_C = set(np.where((y_test == 1) & (preds_C == 0))[0])
print(f"\n  FN Analysis:")
print(f"    v3 baseline FN:        {len(fn_A):,}")
print(f"    v7 TX single FN:       {len(fn_B):,}")
print(f"    v8 TX+trend FN:        {len(fn_C):,}")
print(f"    Net FN saved (vs baseline): {len(fn_A) - len(fn_C):+,}")
print(f"    FN saved by trend (vs single): {len(fn_B) - len(fn_C):+,}")


# ═══════════════════════════════════════════════════════════════
# 5. Feature importance for trend features
# ═══════════════════════════════════════════════════════════════
print(f"\n[5/6] Feature importance & charts...")

full_cols = list(train_data['full'].columns)
importances = pd.Series(rf_C.feature_importances_, index=full_cols).sort_values(ascending=False)

tx_single_cols = [c for c in full_cols if c.startswith('tx_')]
tx_trend_cols = [c for c in full_cols if c.startswith('txt_')]

print(f"\n  Trend Feature Importance (rank / {len(full_cols)}):")
for feat in tx_trend_cols:
    rank = importances.index.tolist().index(feat) + 1
    print(f"    {feat:<25} importance={importances[feat]:.4f}  rank={rank}/{len(full_cols)}")

print(f"\n  Single-Period TX Feature Importance:")
for feat in tx_single_cols:
    rank = importances.index.tolist().index(feat) + 1
    print(f"    {feat:<25} importance={importances[feat]:.4f}  rank={rank}/{len(full_cols)}")


# ═══════════════════════════════════════════════════════════════
# 6. Charts
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. AUC comparison
ax = axes[0, 0]
names_short = ['v3 (45)', 'v7+TX (56)', 'v8+TX+Trend (64)']
aucs = [results[n]['auc'] for n in results]
colors = ['#3498db', '#2ecc71', '#e74c3c']
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
    if f.startswith('txt_'):
        colors_imp.append('#e74c3c')   # trend = red
    elif f.startswith('tx_'):
        colors_imp.append('#e67e22')   # single TX = orange
    else:
        colors_imp.append('#3498db')   # base = blue
ax.barh(range(len(top25)), top25.values, color=colors_imp)
ax.set_yticks(range(len(top25)))
ax.set_yticklabels(top25.index, fontsize=7)
ax.invert_yaxis()
ax.set_title('Feature Importance (Top 25)\nRed=Trend  Orange=Single-TX  Blue=Base')
ax.set_xlabel('Importance')

# 4. TX features only (single + trend)
ax = axes[1, 1]
all_tx = importances[tx_single_cols + tx_trend_cols].sort_values(ascending=True)
colors_tx = []
for f in all_tx.index:
    if f.startswith('txt_'):
        colors_tx.append('#e74c3c')
    else:
        colors_tx.append('#e67e22')
ax.barh(range(len(all_tx)), all_tx.values, color=colors_tx)
ax.set_yticks(range(len(all_tx)))
ax.set_yticklabels(all_tx.index, fontsize=8)
ax.set_title('All TX Feature Importance\nRed=Multi-Period Trend  Orange=Single-Period')
ax.set_xlabel('Importance')

plt.suptitle(
    f'Phase 29: Multi-Period TX Trend Features\n'
    f'Baseline AUC={baseline_auc:.4f} → Best AUC={best_auc:.4f} ({(best_auc-baseline_auc)*100:+.2f}%)',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig('charts/phase29_tx_trends.png', dpi=150, bbox_inches='tight')
print(f"  Saved: charts/phase29_tx_trends.png")


# ═══════════════════════════════════════════════════════════════
# Save model
# ═══════════════════════════════════════════════════════════════
artifact = {
    'model': rf_C,
    'feature_cols': full_cols,
    'tx_single_cols': tx_single_cols,
    'tx_trend_cols': tx_trend_cols,
    'version': 'v8_tx_trend',
    'n_features': len(full_cols),
    'auc_roc': results['C: v8 NEW (64)']['auc'],
}
joblib.dump(artifact, 'models/churn_v8_tx_trend.joblib')
print(f"  Saved: models/churn_v8_tx_trend.joblib")


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Phase 29: Multi-Period TX Trend Features — Summary")
print(f"{'='*60}")
print(f"  Trend features         : {len(tx_trend_cols)}")
print(f"  Single-period features : {len(tx_single_cols)}")
print(f"  Base features          : {train_data['base'].shape[1]}")
print(f"  Total features (full)  : {len(full_cols)}")
print(f"  Train files            : {len(train_files)}")
print(f"  Train samples          : {len(y_train):,}")
print(f"  Test samples           : {len(y_test):,}")
print(f"  Baseline AUC           : {baseline_auc:.4f}")
print(f"  Best AUC               : {best_auc:.4f} ({(best_auc-baseline_auc)*100:+.2f}%)")
print(f"  Trend incremental      : {(full_auc - single_auc)*100:+.2f}%")
print(f"  Chart                  : charts/phase29_tx_trends.png")
print(f"  Model                  : models/churn_v8_tx_trend.joblib")
print(f"{'='*60}")
