#!/usr/bin/env python3
"""Phase 35: Prize Feature Engineering.

Build 12 prize-based features from historical lottery win data and evaluate
whether winning history predicts churn.

Prize features capture: win frequency, recency, amount, type distribution,
and streaks. Research suggests win frequency > win size for retention.

Model variants:
  - v3 baseline (45f, Churn only)
  - v12 (57f = 45 base + 12 prize)
  - v12b (79f = 45 base + 11 TX agg + 11 time + 12 prize)
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
    engineer_features_v2, aggregate_tx_features, compute_time_features,
    compute_prize_features,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

MIN_TENURE = 3

print("=" * 70, flush=True)
print("Phase 35: Prize Feature Engineering", flush=True)
print("=" * 70, flush=True)

# ═══════════════════════════════════════════════════════════════
# Prize file dates (sorted chronologically)
# ═══════════════════════════════════════════════════════════════
PRIZE_DIR = 'data/prize'
all_prize_files = sorted([f for f in os.listdir(PRIZE_DIR) if f.endswith('.csv')])

# Extract period dates from prize filenames: Prize_YYYY_MMDD.csv -> "YYYY_MM_DD"
# to match item column format: item2025_01_02
def prize_file_to_item_date(fname):
    """Prize_YYYY_MMDD.csv -> 'YYYY_MM_DD' matching item column format."""
    parts = fname.replace('.csv', '').split('_')
    year = parts[1]
    mmdd = parts[2]
    mm = mmdd[:2]
    dd = mmdd[2:]
    return f"{year}_{mm}_{dd}"

prize_dates = []
for f in all_prize_files:
    prize_dates.append(prize_file_to_item_date(f))

print(f"\nPrize files available: {len(all_prize_files)}", flush=True)
print(f"  First: {all_prize_files[0]} -> {prize_dates[0]}", flush=True)
print(f"  Last:  {all_prize_files[-1]} -> {prize_dates[-1]}", flush=True)

# ═══════════════════════════════════════════════════════════════
# Churn ↔ Prize mapping
# For each Churn file, the FEATURE columns (all but last item col)
# represent periods the customer was observed. Prize data for any
# period in the feature columns can be used (no leakage).
# The TARGET column is the next period — prize data for that period
# must NOT be used.
# ═══════════════════════════════════════════════════════════════

# Train pairs: Churn files with their corresponding feature period dates
# We use the same 20 train + 2 test split as Phase 30
train_churn_files = [
    'Churn_2025_0216_0301.csv',
    'Churn_2025_0301_0316.csv',
    'Churn_2025_0316_0401.csv',
    'Churn_2025_0401_0416.csv',
    'Churn_2025_0416_0502.csv',
    'Churn_2025_0502_0516.csv',
    'Churn_2025_0516_0601.csv',
    'Churn_2025_0601_0616.csv',
    'Churn_2025_0616_0701.csv',
    'Churn_2025_0701_0716.csv',
    'Churn_2025_0716_0801.csv',
    'Churn_2025_0801_0816.csv',
    'Churn_2025_0816_0901.csv',
    'Churn_2025_0901_0916.csv',
    'Churn_2025_0916_1001.csv',
    'Churn_2025_1001_1016.csv',
    'Churn_2025_1016_1101.csv',
    'Churn_2025_1101_1116.csv',
    'Churn_2025_1116_1201.csv',
    'Churn_2025_1201_1216.csv',
]

test_churn_files = [
    'Churn_2026_0201_0216.csv',
    'Churn_2026_0216_0301.csv',
]

# TX file mapping (same as Phase 30) — for v12b model
train_tx_files = [
    'data/transaction/Transaction_Order_2025_0216.csv',
    'data/transaction/Transaction_Order_2025_0301.csv',
    'data/transaction/Transaction_Order_2025_0316.csv',
    'data/transaction/Transaction_Order_2025_0401.csv',
    'data/transaction/Transaction_Order_2025_0416.csv',
    'data/transaction/Transaction_Order_2025_0502.csv',
    'data/transaction/Transaction_Order_2025_0516.csv',
    'data/transaction/Transaction_Order_2025_0601.csv',
    'data/transaction/Transaction_Order_2025_0616.csv',
    'data/transaction/Transaction_Order_2025_0701.csv',
    'data/transaction/Transaction_Order_2025_0716.csv',
    'data/transaction/Transaction_Order_2025_0801.csv',
    'data/transaction/Transaction_Order_2025_0816.csv',
    'data/transaction/Transaction_Order_2025_0901.csv',
    'data/transaction/Transaction_Order_2025_0916.csv',
    'data/transaction/Transaction_Order_2025_1001.csv',
    'data/transaction/Transaction_Order_2025_1016.csv',
    'data/transaction/Transaction_Order_2025_1101.csv',
    'data/transaction/Transaction_Order_2025_1116.csv',
    'data/transaction/Transaction_Order_2025_1201.csv',
]

test_tx_files = [
    'data/transaction/Transaction_Order_2026_0201.csv',
    'data/transaction/Transaction_Order_2026_0216.csv',
]


def load_churn_features(churn_file):
    """Load Churn CSV, compute v2 features (45), return (feats, y, user_nos, feat_item_cols)."""
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

    return feats, y, user_nos, feat_item_cols


def get_prize_dfs_for_churn(feat_item_cols):
    """Given feature item columns, return list of prize DataFrames for periods
    covered by those columns (no leakage — excludes target period).

    Returns list of DataFrames sorted chronologically.
    """
    # Extract dates from item columns: item2025_01_02 -> 2025_01_02
    feat_dates = set()
    for col in feat_item_cols:
        date_str = col.replace('item', '')  # e.g., '2025_01_02'
        feat_dates.add(date_str)

    prize_dfs = []
    for i, pdate in enumerate(prize_dates):
        if pdate in feat_dates:
            pdf = pd.read_csv(os.path.join(PRIZE_DIR, all_prize_files[i]), low_memory=False)
            prize_dfs.append(pdf)

    return prize_dfs


# ═══════════════════════════════════════════════════════════════
# 1. Load all prize data and cache
# ═══════════════════════════════════════════════════════════════
print("\n[1/8] Loading prize data...", flush=True)
t0 = time.time()

prize_cache = {}  # filename -> DataFrame
for f in all_prize_files:
    pdf = pd.read_csv(os.path.join(PRIZE_DIR, f), low_memory=False)
    prize_cache[f] = pdf
    print(f"  {f}: {len(pdf):,} rows, {pdf['userNo'].nunique():,} users", flush=True)

print(f"  Done ({time.time()-t0:.1f}s)", flush=True)


# ═══════════════════════════════════════════════════════════════
# 2. Load Churn features + prize features for each file
# ═══════════════════════════════════════════════════════════════
print("\n[2/8] Loading Churn features + computing prize features...", flush=True)
t0 = time.time()


def get_prize_dfs_for_feat_cols(feat_item_cols):
    """Get cached prize DataFrames matching feature item columns."""
    feat_dates = set()
    for col in feat_item_cols:
        date_str = col.replace('item', '')
        feat_dates.add(date_str)

    result = []
    for i, pdate in enumerate(prize_dates):
        if pdate in feat_dates:
            result.append(prize_cache[all_prize_files[i]])
    return result


def load_and_compute(churn_files, tx_files, label=""):
    """Load Churn + prize features for a list of files.
    Returns dict with 'base', 'prize', 'full', 'y' keys.
    """
    all_base, all_prize, all_full, all_y = [], [], [], []
    all_tx_both = []

    for idx, churn_file in enumerate(churn_files):
        churn_path = f"data/churn/{churn_file}"
        print(f"  [{label}] {churn_file}...", end=" ", flush=True)
        t_f = time.time()

        feats, y, user_nos, feat_item_cols = load_churn_features(churn_path)

        # Get prize DataFrames for this period
        pz_dfs = get_prize_dfs_for_feat_cols(feat_item_cols)

        # Compute prize features
        pz_feats = compute_prize_features(pz_dfs, user_nos.values)

        # Match count
        n_winners = (pz_feats['prize_has_ever_won'] > 0).sum()
        pct = n_winners / len(feats) * 100

        print(f"n={len(feats):,}, churn={y.mean()*100:.1f}%, "
              f"prize_periods={len(pz_dfs)}, winners={n_winners:,} ({pct:.1f}%) "
              f"({time.time()-t_f:.1f}s)", flush=True)

        # Base (45 features)
        X_base = feats.copy()

        # Prize (45 + 12 = 57 features)
        X_prize = pd.concat([feats.reset_index(drop=True), pz_feats.reset_index(drop=True)], axis=1)

        all_base.append(X_base)
        all_prize.append(X_prize)
        all_y.append(y)

        # TX features for v12b model
        if tx_files is not None and idx < len(tx_files):
            tx_file = tx_files[idx]
            if os.path.exists(tx_file):
                # TX aggregate
                tx_df = pd.read_csv(tx_file, low_memory=False)
                tx_agg = aggregate_tx_features(tx_df)

                # TX time
                tx_time = compute_time_features(tx_file)

                # Merge TX features
                feats_with_user = feats.copy()
                feats_with_user['userNo'] = user_nos.values

                merged = feats_with_user.merge(
                    tx_agg, left_on='userNo', right_index=True, how='left'
                ).merge(
                    tx_time, left_on='userNo', right_index=True, how='left'
                )

                # Fill missing TX features
                for c in list(tx_agg.columns) + list(tx_time.columns):
                    if c in merged.columns:
                        merged[c] = merged[c].fillna(0)

                merged = merged.drop(columns=['userNo'])

                # Add prize features
                X_full = pd.concat([merged.reset_index(drop=True), pz_feats.reset_index(drop=True)], axis=1)
                all_full.append(X_full)
            else:
                print(f"    WARNING: TX file not found: {tx_file}", flush=True)
                all_full.append(None)
        else:
            all_full.append(None)

    result = {
        'base': pd.concat(all_base, ignore_index=True),
        'prize': pd.concat(all_prize, ignore_index=True),
        'y': np.concatenate(all_y),
    }

    # Full (base + TX + prize) — only if all files had TX
    valid_full = [x for x in all_full if x is not None]
    if len(valid_full) == len(churn_files):
        result['full'] = pd.concat(valid_full, ignore_index=True)
    else:
        result['full'] = None

    return result


print("\n  --- Train (20 files) ---", flush=True)
train_data = load_and_compute(train_churn_files, train_tx_files, "Train")
print(f"\n  Train total: {len(train_data['y']):,}, churn={train_data['y'].mean()*100:.1f}%", flush=True)

print("\n  --- Test (2 files) ---", flush=True)
test_data = load_and_compute(test_churn_files, test_tx_files, "Test")
print(f"\n  Test total: {len(test_data['y']):,}, churn={test_data['y'].mean()*100:.1f}%", flush=True)

print(f"\n  Done ({time.time()-t0:.1f}s)", flush=True)
print(f"  Base features (v2): {train_data['base'].shape[1]}", flush=True)
print(f"  Prize features: {train_data['prize'].shape[1]}", flush=True)
if train_data['full'] is not None:
    print(f"  Full features (base+TX+prize): {train_data['full'].shape[1]}", flush=True)


# ═══════════════════════════════════════════════════════════════
# 3. Prize feature statistics
# ═══════════════════════════════════════════════════════════════
print("\n[3/8] Prize feature statistics...", flush=True)

prize_cols = [c for c in train_data['prize'].columns if c.startswith('prize_')]
print(f"\n  Prize feature columns ({len(prize_cols)}):", flush=True)
for col in prize_cols:
    vals = train_data['prize'][col]
    nonzero_pct = (vals > 0).mean() * 100
    print(f"    {col:<35} mean={vals.mean():.4f}  std={vals.std():.4f}  "
          f"nonzero={nonzero_pct:.1f}%", flush=True)

# Winner vs non-winner churn rates
winners = train_data['prize']['prize_has_ever_won'] == 1
print(f"\n  Churn rate comparison:", flush=True)
print(f"    Winners:     {train_data['y'][winners.values].mean()*100:.1f}% "
      f"(n={winners.sum():,})", flush=True)
print(f"    Non-winners: {train_data['y'][~winners.values].mean()*100:.1f}% "
      f"(n={(~winners).sum():,})", flush=True)


# ═══════════════════════════════════════════════════════════════
# 4. Training data management (subsample if > 2M)
# ═══════════════════════════════════════════════════════════════
print("\n[4/8] Training data management...", flush=True)

y_train = train_data['y']
y_test = test_data['y']

if len(y_train) > 2_000_000:
    print(f"  Total rows ({len(y_train):,}) > 2M, subsampling...", flush=True)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(y_train), 2_000_000, replace=False)
    for key in ['base', 'prize']:
        train_data[key] = train_data[key].iloc[idx].reset_index(drop=True)
    if train_data['full'] is not None:
        train_data['full'] = train_data['full'].iloc[idx].reset_index(drop=True)
    y_train = y_train[idx]
    train_data['y'] = y_train
    print(f"  Subsampled to {len(y_train):,}", flush=True)
else:
    print(f"  Total rows: {len(y_train):,} (under 2M, no subsampling needed)", flush=True)

print(f"  Final train size: {len(y_train):,}, churn rate: {y_train.mean()*100:.1f}%", flush=True)


# ═══════════════════════════════════════════════════════════════
# 5. Train models and compare
# ═══════════════════════════════════════════════════════════════
print("\n[5/8] Training & evaluating models...", flush=True)
t0 = time.time()

RF_PARAMS = dict(
    n_estimators=300, max_depth=20, min_samples_leaf=50,
    class_weight='balanced', random_state=42, n_jobs=-1,
)
THRESHOLD = 0.48

results = {}
models_dict = {}

# --- Model A: RF v3 pretrained (45 features, baseline) ---
print("\n  [A] RF v3 pretrained (45f baseline) — evaluate only...", flush=True)
t_a = time.time()
v3_artifact = joblib.load('models/churn_v3_extended.joblib')
v3_model = v3_artifact['model']
v3_feature_cols = v3_artifact['feature_cols']

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

# --- Model B: v12 prize only (57f = 45 base + 12 prize) ---
print("\n  [B] v12 prize (57f = 45+12 prize) — training...", flush=True)
t_b = time.time()
rf_B = RandomForestClassifier(**RF_PARAMS)
rf_B.fit(train_data['prize'].fillna(0), y_train)
probs_B = rf_B.predict_proba(test_data['prize'].fillna(0))[:, 1]
preds_B = (probs_B >= THRESHOLD).astype(int)
results['v12 prize (57f)'] = {
    'auc': roc_auc_score(y_test, probs_B),
    'f1': f1_score(y_test, preds_B),
    'cm': confusion_matrix(y_test, preds_B),
    'probs': probs_B,
    'preds': preds_B,
}
models_dict['v12'] = rf_B
print(f"      AUC={results['v12 prize (57f)']['auc']:.4f} ({time.time()-t_b:.1f}s)", flush=True)

# --- Model C: v12b full (base + TX agg + TX time + prize) ---
if train_data['full'] is not None and test_data['full'] is not None:
    n_full_feats = train_data['full'].shape[1]
    print(f"\n  [C] v12b full ({n_full_feats}f = 45+11+11+12 prize) — training...", flush=True)
    t_c = time.time()
    rf_C = RandomForestClassifier(**RF_PARAMS)
    rf_C.fit(train_data['full'].fillna(0), y_train)
    probs_C = rf_C.predict_proba(test_data['full'].fillna(0))[:, 1]
    preds_C = (probs_C >= THRESHOLD).astype(int)
    results[f'v12b full ({n_full_feats}f)'] = {
        'auc': roc_auc_score(y_test, probs_C),
        'f1': f1_score(y_test, preds_C),
        'cm': confusion_matrix(y_test, preds_C),
        'probs': probs_C,
        'preds': preds_C,
    }
    models_dict['v12b'] = rf_C
    print(f"      AUC={results[f'v12b full ({n_full_feats}f)']['auc']:.4f} ({time.time()-t_c:.1f}s)", flush=True)
else:
    print("\n  [C] v12b full — SKIPPED (TX data not available for all files)", flush=True)
    rf_C = None

print(f"\n  All models evaluated ({time.time()-t0:.1f}s)", flush=True)


# ═══════════════════════════════════════════════════════════════
# 6. Results table
# ═══════════════════════════════════════════════════════════════
print("\n[6/8] Results comparison...", flush=True)

baseline_auc = results['v3 baseline (45f)']['auc']

print(f"\n  {'='*95}", flush=True)
print(f"  Model Comparison (Test: {len(y_test):,} customers, "
      f"churn={y_test.mean()*100:.1f}%)", flush=True)
print(f"  {'='*95}", flush=True)
print(f"  {'Model':<35} {'Features':>10} {'AUC-ROC':>10} {'F1':>10} "
      f"{'FN':>10} {'FP':>10} {'delta AUC':>10}", flush=True)
print(f"  {'-'*95}", flush=True)

model_nfeats = {
    'v3 baseline (45f)': 45,
    'v12 prize (57f)': 57,
}
if train_data['full'] is not None:
    n_full = train_data['full'].shape[1]
    for k in results:
        if 'v12b' in k:
            model_nfeats[k] = n_full

for name, r in results.items():
    fn = r['cm'][1, 0]
    fp = r['cm'][0, 1]
    nf = model_nfeats.get(name, '?')
    delta = (r['auc'] - baseline_auc) * 100
    delta_str = f"{delta:+.2f}%" if name != 'v3 baseline (45f)' else "baseline"
    print(f"  {name:<35} {nf:>10} {r['auc']:>10.4f} {r['f1']:>10.4f} "
          f"{fn:>10,} {fp:>10,} {delta_str:>10}", flush=True)

print(f"  {'-'*95}", flush=True)

best_name = max(results, key=lambda k: results[k]['auc'])
best_auc = results[best_name]['auc']
print(f"  Baseline AUC: {baseline_auc:.4f}", flush=True)
print(f"  Best model:   {best_name} (AUC: {best_auc:.4f})", flush=True)
print(f"  Improvement:  {(best_auc - baseline_auc)*100:+.2f}%", flush=True)


# ═══════════════════════════════════════════════════════════════
# 7. Feature importance analysis
# ═══════════════════════════════════════════════════════════════
print("\n[7/8] Feature importance analysis...", flush=True)

# Use v12 (57f) for prize-only importance
prize_model = rf_B
prize_feature_cols = list(train_data['prize'].columns)
importances_prize = pd.Series(
    prize_model.feature_importances_, index=prize_feature_cols
).sort_values(ascending=False)

prize_only_cols = [c for c in prize_feature_cols if c.startswith('prize_')]
base_only_cols = [c for c in prize_feature_cols if not c.startswith('prize_')]

print(f"\n  Prize Feature Importance in v12 (rank / {len(prize_feature_cols)}):", flush=True)
for feat in prize_only_cols:
    if feat in importances_prize.index:
        rank = importances_prize.index.tolist().index(feat) + 1
        print(f"    {feat:<35} importance={importances_prize[feat]:.4f}  "
              f"rank={rank}/{len(prize_feature_cols)}", flush=True)

print(f"\n  Top 20 overall features (v12):", flush=True)
for i, (feat, imp) in enumerate(importances_prize.head(20).items()):
    tag = " [PRIZE]" if feat in prize_only_cols else ""
    print(f"    {i+1:>2}. {feat:<35} {imp:.4f}{tag}", flush=True)

# If v12b exists, show its importance too
if rf_C is not None:
    full_feature_cols = list(train_data['full'].columns)
    importances_full = pd.Series(
        rf_C.feature_importances_, index=full_feature_cols
    ).sort_values(ascending=False)

    tx_agg_cols = [c for c in full_feature_cols if c.startswith('tx_') and not c.startswith('tx_early') and not c.startswith('tx_rush') and not c.startswith('tx_first_order') and not c.startswith('tx_last_order') and not c.startswith('tx_order_spread') and not c.startswith('tx_avg_order') and not c.startswith('tx_items_') and not c.startswith('tx_peak_')]
    tx_time_cols = [c for c in full_feature_cols if c.startswith('tx_') and c not in tx_agg_cols]

    print(f"\n  Prize Feature Importance in v12b full (rank / {len(full_feature_cols)}):", flush=True)
    for feat in prize_only_cols:
        if feat in importances_full.index:
            rank = importances_full.index.tolist().index(feat) + 1
            print(f"    {feat:<35} importance={importances_full[feat]:.4f}  "
                  f"rank={rank}/{len(full_feature_cols)}", flush=True)

    print(f"\n  Top 20 overall features (v12b full):", flush=True)
    for i, (feat, imp) in enumerate(importances_full.head(20).items()):
        tag = ""
        if feat in prize_only_cols:
            tag = " [PRIZE]"
        elif feat.startswith('tx_'):
            tag = " [TX]"
        print(f"    {i+1:>2}. {feat:<35} {imp:.4f}{tag}", flush=True)


# ═══════════════════════════════════════════════════════════════
# 8. FN Analysis + Charts
# ═══════════════════════════════════════════════════════════════
print("\n[8/8] FN Analysis & Charts...", flush=True)

fn_A = set(np.where((y_test == 1) & (preds_A == 0))[0])
fn_B = set(np.where((y_test == 1) & (preds_B == 0))[0])

print(f"\n  FN Comparison:", flush=True)
print(f"    [A] v3 baseline (45f)     FN: {len(fn_A):,}", flush=True)
print(f"    [B] v12 prize (57f)       FN: {len(fn_B):,}", flush=True)

fn_saved_by_prize = fn_A - fn_B
fn_lost_by_prize = fn_B - fn_A
print(f"    FN saved by prize:  {len(fn_saved_by_prize):,}", flush=True)
print(f"    FN lost by prize:   {len(fn_lost_by_prize):,}", flush=True)
print(f"    Net FN change:      {len(fn_A) - len(fn_B):+,}", flush=True)

if rf_C is not None:
    preds_C_arr = preds_C
    fn_C = set(np.where((y_test == 1) & (preds_C_arr == 0))[0])
    print(f"    [C] v12b full             FN: {len(fn_C):,}", flush=True)
    print(f"    FN saved (A->C):    {len(fn_A) - len(fn_C):+,}", flush=True)

# --- Charts ---
print("\n  Generating charts...", flush=True)

n_models = len(results)
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# --- [0,0] AUC bar chart ---
ax = axes[0, 0]
names_list = list(results.keys())
aucs = [results[n]['auc'] for n in names_list]
short_names = [n.split('(')[0].strip() for n in names_list]
colors_bar = ['#3498db', '#9b59b6', '#e74c3c'][:n_models]
bars = ax.bar(range(n_models), aucs, color=colors_bar)
ax.set_xticks(range(n_models))
ax.set_xticklabels(short_names, fontsize=8, rotation=15)
ax.set_ylabel('AUC-ROC')
ax.set_title('AUC-ROC Comparison')
ax.set_ylim(min(aucs) - 0.005, max(aucs) + 0.005)
for bar, auc_val in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f'{auc_val:.4f}', ha='center', fontsize=9, fontweight='bold')

# --- [0,1] FN/FP comparison ---
ax = axes[0, 1]
fn_vals = [results[n]['cm'][1, 0] for n in names_list]
fp_vals = [results[n]['cm'][0, 1] for n in names_list]
x = np.arange(n_models)
w = 0.35
ax.bar(x - w/2, fn_vals, w, label='FN (missed churn)', color='#e74c3c')
ax.bar(x + w/2, fp_vals, w, label='FP (false alarm)', color='#f39c12')
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
bars = ax.bar(range(n_models), f1s, color=colors_bar)
ax.set_xticks(range(n_models))
ax.set_xticklabels(short_names, fontsize=8, rotation=15)
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score Comparison')
ax.set_ylim(min(f1s) - 0.01, max(f1s) + 0.01)
for bar, f1_val in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{f1_val:.4f}', ha='center', fontsize=9, fontweight='bold')

# --- [1,0] Feature importance top 30 (v12, color-coded) ---
ax = axes[1, 0]
top30 = importances_prize.head(30)
colors_imp = ['#9b59b6' if f in prize_only_cols else '#3498db' for f in top30.index]
ax.barh(range(len(top30)), top30.values, color=colors_imp)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(top30.index, fontsize=6)
ax.invert_yaxis()
ax.set_title('Feature Importance v12 (Top 30)\nBlue=Base, Purple=Prize')
ax.set_xlabel('Importance')

# --- [1,1] Prize feature importance (standalone) ---
ax = axes[1, 1]
prize_imp = importances_prize[prize_only_cols].sort_values(ascending=False)
bars_p = ax.barh(range(len(prize_imp)), prize_imp.values, color='#9b59b6')
ax.set_yticks(range(len(prize_imp)))
ax.set_yticklabels(prize_imp.index, fontsize=8)
ax.invert_yaxis()
ax.set_title('Prize Feature Importance (12 features)')
ax.set_xlabel('Importance')
for i, (feat, imp) in enumerate(prize_imp.items()):
    ax.text(imp + 0.0001, i, f'{imp:.4f}', va='center', fontsize=7)

# --- [1,2] Winner vs non-winner churn analysis ---
ax = axes[1, 2]
test_prize_data = test_data['prize']
test_winners = test_prize_data['prize_has_ever_won'] == 1
churn_winner = y_test[test_winners.values].mean() * 100
churn_non = y_test[~test_winners.values].mean() * 100
bars_churn = ax.bar(['Winners', 'Non-winners'], [churn_winner, churn_non],
                     color=['#9b59b6', '#95a5a6'])
ax.set_ylabel('Churn Rate (%)')
ax.set_title('Churn Rate: Winners vs Non-winners (Test)')
for bar, val in zip(bars_churn, [churn_winner, churn_non]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.text(0.5, 0.95, f'Winners: {test_winners.sum():,} | Non-winners: {(~test_winners).sum():,}',
        transform=ax.transAxes, ha='center', fontsize=9, style='italic')

plt.suptitle(
    f'Phase 35: Prize Feature Engineering\n'
    f'Baseline AUC={baseline_auc:.4f} | Best: {best_name} AUC={best_auc:.4f} '
    f'({(best_auc-baseline_auc)*100:+.2f}%)',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig('charts/phase35_prize_features.png', dpi=150, bbox_inches='tight')
print(f"  Saved: charts/phase35_prize_features.png", flush=True)


# ═══════════════════════════════════════════════════════════════
# 9. Save models
# ═══════════════════════════════════════════════════════════════
print("\n  Saving models...", flush=True)

# Save v12 (prize only)
artifact_v12 = {
    'model': rf_B,
    'feature_cols': list(train_data['prize'].columns),
    'prize_feature_cols': prize_only_cols,
    'base_feature_cols': base_only_cols,
    'version': 'v12_prize',
    'n_features': len(train_data['prize'].columns),
    'n_train_files': 20,
    'n_train_samples': len(y_train),
    'auc_roc': results['v12 prize (57f)']['auc'],
}
joblib.dump(artifact_v12, 'models/churn_v12_prize.joblib')
print(f"  Saved: models/churn_v12_prize.joblib (57 features)", flush=True)

# Save v12b if available
if rf_C is not None:
    full_key = [k for k in results if 'v12b' in k][0]
    artifact_v12b = {
        'model': rf_C,
        'feature_cols': list(train_data['full'].columns),
        'prize_feature_cols': prize_only_cols,
        'version': 'v12b_full_prize',
        'n_features': len(train_data['full'].columns),
        'n_train_files': 20,
        'n_train_samples': len(y_train),
        'auc_roc': results[full_key]['auc'],
    }
    joblib.dump(artifact_v12b, 'models/churn_v12b_full_prize.joblib')
    print(f"  Saved: models/churn_v12b_full_prize.joblib ({len(train_data['full'].columns)} features)", flush=True)


# ═══════════════════════════════════════════════════════════════
# 10. Summary
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}", flush=True)
print(f"Phase 35: Prize Feature Engineering — Summary", flush=True)
print(f"{'='*70}", flush=True)
print(f"  Prize files processed     : {len(all_prize_files)}", flush=True)
print(f"  Prize features created    : {len(prize_only_cols)}", flush=True)
print(f"  Train files               : 20", flush=True)
print(f"  Train samples             : {len(y_train):,}", flush=True)
print(f"  Test files                : 2", flush=True)
print(f"  Test samples              : {len(y_test):,}", flush=True)
print(f"  {'─'*50}", flush=True)

for name, r in results.items():
    delta = (r['auc'] - baseline_auc) * 100
    delta_str = f"({delta:+.2f}%)" if name != 'v3 baseline (45f)' else "(baseline)"
    print(f"  {name:<35}: AUC {r['auc']:.4f} {delta_str}", flush=True)

print(f"  {'─'*50}", flush=True)
print(f"  Best model                : {best_name}", flush=True)
print(f"  Best AUC                  : {best_auc:.4f}", flush=True)
print(f"  FN saved (baseline->v12)  : {len(fn_A) - len(fn_B):+,}", flush=True)
print(f"  Chart                     : charts/phase35_prize_features.png", flush=True)
print(f"  Model (v12)               : models/churn_v12_prize.joblib", flush=True)
if rf_C is not None:
    print(f"  Model (v12b)              : models/churn_v12b_full_prize.joblib", flush=True)
print(f"{'='*70}", flush=True)
