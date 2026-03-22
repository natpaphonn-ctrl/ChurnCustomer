#!/usr/bin/env python3
"""Phase 33: Feature Selection & Pruning.

Reduce 67 features to the most impactful subset. Many TX features rank 50-67
and add noise. A leaner model may perform better and will be faster/simpler
for production.

Methods:
  1. Importance-based selection (top-K sweep: 20, 25, 30, 35, 40, 45, 50)
  2. Recursive Feature Elimination (remove bottom 5, retrain, repeat)
  3. Correlation-based pruning (drop one of highly correlated pairs)

Output:
  - charts/phase33_feature_selection.png
  - models/churn_v10_pruned.joblib
"""

import os, sys, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sys.path.insert(0, 'models')
from feature_engineering import engineer_features_v2, aggregate_tx_features, compute_time_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

MIN_TENURE = 3

print("=" * 70, flush=True)
print("Phase 33: Feature Selection & Pruning", flush=True)
print("=" * 70, flush=True)

# ═══════════════════════════════════════════════════════════════
# TX <-> Churn mapping (20 train + 2 test) — same as Phase 28/30
# ═══════════════════════════════════════════════════════════════

train_pairs = [
    ('data/transaction/Transaction_Order_2025_0216.csv', 'data/churn/Churn_2025_0216_0301.csv'),
    ('data/transaction/Transaction_Order_2025_0301.csv', 'data/churn/Churn_2025_0301_0316.csv'),
    ('data/transaction/Transaction_Order_2025_0316.csv', 'data/churn/Churn_2025_0316_0401.csv'),
    ('data/transaction/Transaction_Order_2025_0401.csv', 'data/churn/Churn_2025_0401_0416.csv'),
    ('data/transaction/Transaction_Order_2025_0416.csv', 'data/churn/Churn_2025_0416_0502.csv'),
    ('data/transaction/Transaction_Order_2025_0502.csv', 'data/churn/Churn_2025_0502_0516.csv'),
    ('data/transaction/Transaction_Order_2025_0516.csv', 'data/churn/Churn_2025_0516_0601.csv'),
    ('data/transaction/Transaction_Order_2025_0601.csv', 'data/churn/Churn_2025_0601_0616.csv'),
    ('data/transaction/Transaction_Order_2025_0616.csv', 'data/churn/Churn_2025_0616_0701.csv'),
    ('data/transaction/Transaction_Order_2025_0701.csv', 'data/churn/Churn_2025_0701_0716.csv'),
    ('data/transaction/Transaction_Order_2025_0716.csv', 'data/churn/Churn_2025_0716_0801.csv'),
    ('data/transaction/Transaction_Order_2025_0801.csv', 'data/churn/Churn_2025_0801_0816.csv'),
    ('data/transaction/Transaction_Order_2025_0816.csv', 'data/churn/Churn_2025_0816_0901.csv'),
    ('data/transaction/Transaction_Order_2025_0901.csv', 'data/churn/Churn_2025_0901_0916.csv'),
    ('data/transaction/Transaction_Order_2025_0916.csv', 'data/churn/Churn_2025_0916_1001.csv'),
    ('data/transaction/Transaction_Order_2025_1001.csv', 'data/churn/Churn_2025_1001_1016.csv'),
    ('data/transaction/Transaction_Order_2025_1016.csv', 'data/churn/Churn_2025_1016_1101.csv'),
    ('data/transaction/Transaction_Order_2025_1101.csv', 'data/churn/Churn_2025_1101_1116.csv'),
    ('data/transaction/Transaction_Order_2025_1116.csv', 'data/churn/Churn_2025_1116_1201.csv'),
    ('data/transaction/Transaction_Order_2025_1201.csv', 'data/churn/Churn_2025_1201_1216.csv'),
]

test_pairs = [
    ('data/transaction/Transaction_Order_2026_0201.csv', 'data/churn/Churn_2026_0201_0216.csv'),
    ('data/transaction/Transaction_Order_2026_0216.csv', 'data/churn/Churn_2026_0216_0301.csv'),
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
print("\n[1/9] Aggregating transaction data (22 files)...", flush=True)
t0 = time.time()

tx_agg_cache = {}
tx_time_cache = {}
all_pairs = train_pairs + test_pairs

for tx_file, _ in all_pairs:
    if tx_file not in tx_agg_cache:
        t_file = time.time()
        basename = os.path.basename(tx_file)
        print(f"  {basename}...", end=" ", flush=True)

        tx_agg = load_tx_agg_features(tx_file)
        tx_agg_cache[tx_file] = tx_agg

        tx_time = load_tx_time_features(tx_file)
        tx_time_cache[tx_file] = tx_time

        print(f"agg={len(tx_agg):,} time={len(tx_time):,} "
              f"({time.time()-t_file:.1f}s)", flush=True)

print(f"  Total TX files cached: {len(tx_agg_cache)}", flush=True)
print(f"  Done ({time.time()-t0:.1f}s)", flush=True)


# ═══════════════════════════════════════════════════════════════
# 2. Load & merge Churn + TX for each pair (67 features)
# ═══════════════════════════════════════════════════════════════
print("\n[2/9] Loading Churn features & merging with TX (67f)...", flush=True)
t0 = time.time()


def load_and_merge_67f(pairs):
    """Load Churn + TX pairs, merge into 67-feature dataset."""
    all_X, all_y = [], []

    for tx_file, churn_file in pairs:
        print(f"  {os.path.basename(churn_file)}...", flush=True)
        feats, y, user_nos = load_churn_features(churn_file)

        tx_agg = tx_agg_cache[tx_file]
        tx_time = tx_time_cache[tx_file]

        feats_with_user = feats.copy()
        feats_with_user['userNo'] = user_nos.values

        # Merge both TX aggregate + time features
        merged = feats_with_user.merge(
            tx_agg, left_on='userNo', right_index=True, how='left'
        ).merge(
            tx_time, left_on='userNo', right_index=True, how='left'
        )

        agg_cols = list(tx_agg.columns)
        time_cols = list(tx_time.columns)
        merged[agg_cols] = merged[agg_cols].fillna(0)
        merged[time_cols] = merged[time_cols].fillna(0)

        X = merged.drop(columns=['userNo'])

        print(f"    n={len(X):,}, features={X.shape[1]}, churn={y.mean()*100:.1f}%", flush=True)

        all_X.append(X)
        all_y.append(y)

    return pd.concat(all_X, ignore_index=True), np.concatenate(all_y)


print("\n  --- Train (20 files) ---", flush=True)
X_train_full, y_train_full = load_and_merge_67f(train_pairs)
print(f"\n  Train total: {len(y_train_full):,}, churn={y_train_full.mean()*100:.1f}%", flush=True)

print("\n  --- Test (2 files) ---", flush=True)
X_test, y_test = load_and_merge_67f(test_pairs)
print(f"\n  Test total: {len(y_test):,}, churn={y_test.mean()*100:.1f}%", flush=True)

all_feature_cols = list(X_train_full.columns)
print(f"\n  Total features: {len(all_feature_cols)}", flush=True)
print(f"  Done ({time.time()-t0:.1f}s)", flush=True)


# ═══════════════════════════════════════════════════════════════
# 3. Create subsample for fast feature selection search
# ═══════════════════════════════════════════════════════════════
print("\n[3/9] Creating subsample for feature selection search...", flush=True)

SUBSAMPLE_SIZE = 800_000
rng = np.random.RandomState(42)

if len(y_train_full) > SUBSAMPLE_SIZE:
    idx_sub = rng.choice(len(y_train_full), SUBSAMPLE_SIZE, replace=False)
    X_train_sub = X_train_full.iloc[idx_sub].reset_index(drop=True)
    y_train_sub = y_train_full[idx_sub]
    print(f"  Subsampled train: {len(y_train_sub):,} (from {len(y_train_full):,})", flush=True)
else:
    X_train_sub = X_train_full
    y_train_sub = y_train_full
    print(f"  Train size {len(y_train_sub):,} <= {SUBSAMPLE_SIZE:,}, no subsampling", flush=True)

print(f"  Subsample churn rate: {y_train_sub.mean()*100:.1f}%", flush=True)

# Cap full training data at 2M for final model retraining
FULL_TRAIN_CAP = 2_000_000
if len(y_train_full) > FULL_TRAIN_CAP:
    idx_full = rng.choice(len(y_train_full), FULL_TRAIN_CAP, replace=False)
    X_train_capped = X_train_full.iloc[idx_full].reset_index(drop=True)
    y_train_capped = y_train_full[idx_full]
    print(f"  Capped full train: {len(y_train_capped):,} (from {len(y_train_full):,})", flush=True)
else:
    X_train_capped = X_train_full
    y_train_capped = y_train_full
    print(f"  Full train: {len(y_train_capped):,} (no capping needed)", flush=True)


# ═══════════════════════════════════════════════════════════════
# 4. Baseline: Train full 67f model on subsample
# ═══════════════════════════════════════════════════════════════
print("\n[4/9] Training baseline 67f model on subsample...", flush=True)
t0 = time.time()

RF_PARAMS = dict(
    n_estimators=300, max_depth=20, min_samples_leaf=50,
    class_weight='balanced', random_state=42, n_jobs=-1,
)
THRESHOLD = 0.48

rf_baseline = RandomForestClassifier(**RF_PARAMS)
rf_baseline.fit(X_train_sub.fillna(0), y_train_sub)

probs_baseline = rf_baseline.predict_proba(X_test.fillna(0))[:, 1]
preds_baseline = (probs_baseline >= THRESHOLD).astype(int)
auc_baseline = roc_auc_score(y_test, probs_baseline)
f1_baseline = f1_score(y_test, preds_baseline)

print(f"  Baseline 67f: AUC={auc_baseline:.4f}, F1={f1_baseline:.4f}", flush=True)
print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

# Get feature importances from baseline model
importances = pd.Series(
    rf_baseline.feature_importances_, index=all_feature_cols
).sort_values(ascending=False)

print(f"\n  Feature importances (top 10):", flush=True)
for i, (feat, imp) in enumerate(importances.head(10).items()):
    print(f"    {i+1:>2}. {feat:<35} {imp:.4f}", flush=True)

print(f"\n  Feature importances (bottom 10):", flush=True)
for i, (feat, imp) in enumerate(importances.tail(10).items()):
    rank = len(importances) - 9 + i
    print(f"    {rank:>2}. {feat:<35} {imp:.4f}", flush=True)


# ═══════════════════════════════════════════════════════════════
# 5. Method 1: Importance-based top-K selection
# ═══════════════════════════════════════════════════════════════
print("\n[5/9] Method 1: Importance-based top-K selection...", flush=True)
t0 = time.time()

K_values = [20, 25, 30, 35, 40, 45, 50, 55, 60, 67]
topk_results = {}

ranked_features = importances.index.tolist()

for K in K_values:
    t_k = time.time()
    selected = ranked_features[:K]

    rf_k = RandomForestClassifier(**RF_PARAMS)
    rf_k.fit(X_train_sub[selected].fillna(0), y_train_sub)

    probs_k = rf_k.predict_proba(X_test[selected].fillna(0))[:, 1]
    preds_k = (probs_k >= THRESHOLD).astype(int)
    auc_k = roc_auc_score(y_test, probs_k)
    f1_k = f1_score(y_test, preds_k)
    cm_k = confusion_matrix(y_test, preds_k)

    topk_results[K] = {
        'auc': auc_k,
        'f1': f1_k,
        'cm': cm_k,
        'features': selected,
        'fn': cm_k[1, 0],
        'fp': cm_k[0, 1],
    }

    delta = (auc_k - auc_baseline) * 100
    print(f"  K={K:>2}: AUC={auc_k:.4f} ({delta:+.2f}%), F1={f1_k:.4f}, "
          f"FN={cm_k[1,0]:,}, FP={cm_k[0,1]:,} ({time.time()-t_k:.1f}s)", flush=True)

# Find optimal K (best AUC, or smallest K within 0.1% of best)
best_k_auc = max(topk_results, key=lambda k: topk_results[k]['auc'])
best_auc_topk = topk_results[best_k_auc]['auc']

# Find the smallest K whose AUC is within 0.001 of the best
optimal_k = best_k_auc
for K in sorted(K_values):
    if topk_results[K]['auc'] >= best_auc_topk - 0.001:
        optimal_k = K
        break

print(f"\n  Best K by AUC: K={best_k_auc} (AUC={best_auc_topk:.4f})", flush=True)
print(f"  Optimal K (within 0.001 of best): K={optimal_k} "
      f"(AUC={topk_results[optimal_k]['auc']:.4f})", flush=True)
print(f"  Done ({time.time()-t0:.1f}s)", flush=True)


# ═══════════════════════════════════════════════════════════════
# 6. Method 2: Recursive Feature Elimination (simple)
# ═══════════════════════════════════════════════════════════════
print("\n[6/9] Method 2: Recursive Feature Elimination...", flush=True)
t0 = time.time()

REMOVE_PER_STEP = 5
MIN_FEATURES = 15
AUC_DROP_THRESHOLD = 0.005  # stop if AUC drops more than 0.5%

rfe_results = []
current_features = ranked_features.copy()

# Record starting point (all 67)
rfe_results.append({
    'n_features': len(current_features),
    'auc': auc_baseline,
    'f1': f1_baseline,
    'features': current_features.copy(),
    'removed': [],
})
print(f"  Start: {len(current_features)} features, AUC={auc_baseline:.4f}", flush=True)

prev_auc = auc_baseline
stop_reason = None

while len(current_features) > MIN_FEATURES:
    t_step = time.time()

    # Train model on current features to get fresh importances
    rf_rfe = RandomForestClassifier(**RF_PARAMS)
    rf_rfe.fit(X_train_sub[current_features].fillna(0), y_train_sub)

    # Get importances for current feature set
    step_imp = pd.Series(
        rf_rfe.feature_importances_, index=current_features
    ).sort_values(ascending=True)

    # Remove bottom REMOVE_PER_STEP features
    n_remove = min(REMOVE_PER_STEP, len(current_features) - MIN_FEATURES)
    if n_remove <= 0:
        break

    to_remove = step_imp.index[:n_remove].tolist()
    remaining = [f for f in current_features if f not in to_remove]

    # Evaluate remaining features
    rf_eval = RandomForestClassifier(**RF_PARAMS)
    rf_eval.fit(X_train_sub[remaining].fillna(0), y_train_sub)

    probs_rfe = rf_eval.predict_proba(X_test[remaining].fillna(0))[:, 1]
    preds_rfe = (probs_rfe >= THRESHOLD).astype(int)
    auc_rfe = roc_auc_score(y_test, probs_rfe)
    f1_rfe = f1_score(y_test, preds_rfe)

    rfe_results.append({
        'n_features': len(remaining),
        'auc': auc_rfe,
        'f1': f1_rfe,
        'features': remaining.copy(),
        'removed': to_remove,
    })

    delta = (auc_rfe - auc_baseline) * 100
    print(f"  {len(remaining):>2}f: AUC={auc_rfe:.4f} ({delta:+.2f}%), "
          f"F1={f1_rfe:.4f}, removed: {to_remove} ({time.time()-t_step:.1f}s)", flush=True)

    # Check stop condition
    if auc_rfe < auc_baseline - AUC_DROP_THRESHOLD:
        stop_reason = f"AUC dropped below threshold ({auc_rfe:.4f} < {auc_baseline - AUC_DROP_THRESHOLD:.4f})"
        print(f"  STOP: {stop_reason}", flush=True)
        break

    current_features = remaining
    prev_auc = auc_rfe

if stop_reason is None:
    stop_reason = f"Reached minimum features ({MIN_FEATURES})"
    print(f"  STOP: {stop_reason}", flush=True)

# Find best RFE step
best_rfe = max(rfe_results, key=lambda r: r['auc'])
print(f"\n  Best RFE: {best_rfe['n_features']}f, AUC={best_rfe['auc']:.4f}", flush=True)

# Find smallest feature set within 0.001 of best
rfe_optimal = best_rfe
for r in sorted(rfe_results, key=lambda x: x['n_features']):
    if r['auc'] >= best_rfe['auc'] - 0.001:
        rfe_optimal = r
        break

print(f"  Optimal RFE (within 0.001): {rfe_optimal['n_features']}f, "
      f"AUC={rfe_optimal['auc']:.4f}", flush=True)
print(f"  Done ({time.time()-t0:.1f}s)", flush=True)


# ═══════════════════════════════════════════════════════════════
# 7. Method 3: Correlation-based pruning
# ═══════════════════════════════════════════════════════════════
print("\n[7/9] Method 3: Correlation-based pruning...", flush=True)
t0 = time.time()

CORR_THRESHOLD = 0.9

# Compute correlation matrix on subsample
corr_matrix = X_train_sub.fillna(0).corr()

# Find highly correlated pairs
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = []

for col in upper.columns:
    for idx in upper.index:
        val = upper.loc[idx, col]
        if pd.notna(val) and abs(val) > CORR_THRESHOLD:
            high_corr_pairs.append((idx, col, val))

print(f"  Found {len(high_corr_pairs)} highly correlated pairs (|r| > {CORR_THRESHOLD}):", flush=True)
for f1_name, f2_name, corr_val in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
    imp1 = importances.get(f1_name, 0)
    imp2 = importances.get(f2_name, 0)
    keep = f1_name if imp1 >= imp2 else f2_name
    drop = f2_name if imp1 >= imp2 else f1_name
    print(f"    {f1_name} <-> {f2_name}: r={corr_val:.3f} "
          f"(keep={keep}, drop={drop})", flush=True)

# Determine which features to drop
features_to_drop_corr = set()
for f1_name, f2_name, corr_val in high_corr_pairs:
    imp1 = importances.get(f1_name, 0)
    imp2 = importances.get(f2_name, 0)
    # Drop the one with lower importance
    drop = f2_name if imp1 >= imp2 else f1_name
    # Only drop if the higher-importance one is not already dropped
    keep = f1_name if imp1 >= imp2 else f2_name
    if keep not in features_to_drop_corr:
        features_to_drop_corr.add(drop)

corr_pruned_features = [f for f in all_feature_cols if f not in features_to_drop_corr]

print(f"\n  Features dropped by correlation: {len(features_to_drop_corr)}", flush=True)
print(f"  Dropped: {sorted(features_to_drop_corr)}", flush=True)
print(f"  Remaining: {len(corr_pruned_features)} features", flush=True)

# Train model with correlation-pruned features
rf_corr = RandomForestClassifier(**RF_PARAMS)
rf_corr.fit(X_train_sub[corr_pruned_features].fillna(0), y_train_sub)

probs_corr = rf_corr.predict_proba(X_test[corr_pruned_features].fillna(0))[:, 1]
preds_corr = (probs_corr >= THRESHOLD).astype(int)
auc_corr = roc_auc_score(y_test, probs_corr)
f1_corr = f1_score(y_test, preds_corr)
cm_corr = confusion_matrix(y_test, preds_corr)

delta_corr = (auc_corr - auc_baseline) * 100
print(f"\n  Correlation-pruned ({len(corr_pruned_features)}f): "
      f"AUC={auc_corr:.4f} ({delta_corr:+.2f}%), F1={f1_corr:.4f}", flush=True)
print(f"  Done ({time.time()-t0:.1f}s)", flush=True)


# ═══════════════════════════════════════════════════════════════
# 8. Compare all methods & select best pruned set
# ═══════════════════════════════════════════════════════════════
print("\n[8/9] Comparing all methods...", flush=True)

# Gather candidates
candidates = {
    f'Full 67f (baseline)': {
        'n_features': 67,
        'auc': auc_baseline,
        'f1': f1_baseline,
        'features': all_feature_cols,
    },
    f'Top-K optimal (K={optimal_k})': {
        'n_features': optimal_k,
        'auc': topk_results[optimal_k]['auc'],
        'f1': topk_results[optimal_k]['f1'],
        'features': topk_results[optimal_k]['features'],
    },
    f'RFE optimal ({rfe_optimal["n_features"]}f)': {
        'n_features': rfe_optimal['n_features'],
        'auc': rfe_optimal['auc'],
        'f1': rfe_optimal['f1'],
        'features': rfe_optimal['features'],
    },
    f'Corr-pruned ({len(corr_pruned_features)}f)': {
        'n_features': len(corr_pruned_features),
        'auc': auc_corr,
        'f1': f1_corr,
        'features': corr_pruned_features,
    },
}

print(f"\n  {'='*80}", flush=True)
print(f"  Method Comparison (Test: {len(y_test):,} customers)", flush=True)
print(f"  {'='*80}", flush=True)
print(f"  {'Method':<35} {'Features':>10} {'AUC-ROC':>10} {'F1':>10} {'delta AUC':>12}", flush=True)
print(f"  {'-'*80}", flush=True)

for name, c in candidates.items():
    delta = (c['auc'] - auc_baseline) * 100
    delta_str = f"{delta:+.2f}%" if 'baseline' not in name else "baseline"
    print(f"  {name:<35} {c['n_features']:>10} {c['auc']:>10.4f} "
          f"{c['f1']:>10.4f} {delta_str:>12}", flush=True)

print(f"  {'-'*80}", flush=True)

# Select the best pruned candidate (excluding baseline)
pruned_candidates = {k: v for k, v in candidates.items() if 'baseline' not in k}
best_pruned_name = max(pruned_candidates, key=lambda k: pruned_candidates[k]['auc'])
best_pruned = pruned_candidates[best_pruned_name]

# If multiple candidates have same AUC (within 0.0005), pick the one with fewer features
for name, c in sorted(pruned_candidates.items(), key=lambda x: x[1]['n_features']):
    if c['auc'] >= best_pruned['auc'] - 0.0005:
        best_pruned_name = name
        best_pruned = c
        break

print(f"\n  Best pruned method: {best_pruned_name}", flush=True)
print(f"  Features: {best_pruned['n_features']}, AUC: {best_pruned['auc']:.4f}", flush=True)

# Show which features survived and which were dropped
final_features = best_pruned['features']
dropped_features = [f for f in all_feature_cols if f not in final_features]

print(f"\n  Surviving features ({len(final_features)}):", flush=True)
for i, feat in enumerate(final_features):
    imp = importances.get(feat, 0)
    print(f"    {i+1:>2}. {feat:<35} importance={imp:.4f}", flush=True)

print(f"\n  Dropped features ({len(dropped_features)}):", flush=True)
for feat in dropped_features:
    imp = importances.get(feat, 0)
    print(f"    - {feat:<35} importance={imp:.4f}", flush=True)


# ═══════════════════════════════════════════════════════════════
# 9. Retrain final model on full data & save
# ═══════════════════════════════════════════════════════════════
print("\n[9/9] Retraining final pruned model on full data...", flush=True)
t0 = time.time()

rf_final = RandomForestClassifier(**RF_PARAMS)
rf_final.fit(X_train_capped[final_features].fillna(0), y_train_capped)

probs_final = rf_final.predict_proba(X_test[final_features].fillna(0))[:, 1]
preds_final = (probs_final >= THRESHOLD).astype(int)
auc_final = roc_auc_score(y_test, probs_final)
f1_final = f1_score(y_test, preds_final)
cm_final = confusion_matrix(y_test, preds_final)

print(f"  Final pruned model ({len(final_features)}f on {len(y_train_capped):,} samples):", flush=True)
print(f"    AUC={auc_final:.4f}, F1={f1_final:.4f}", flush=True)
print(f"    FN={cm_final[1,0]:,}, FP={cm_final[0,1]:,}", flush=True)
print(f"    vs baseline 67f: AUC delta={(auc_final - auc_baseline)*100:+.2f}%", flush=True)

# Get final feature importances
final_importances = pd.Series(
    rf_final.feature_importances_, index=final_features
).sort_values(ascending=False)

# Save model artifact
artifact = {
    'model': rf_final,
    'feature_cols': final_features,
    'version': 'v10_pruned',
    'n_features': len(final_features),
    'n_features_original': 67,
    'n_features_dropped': len(dropped_features),
    'dropped_features': dropped_features,
    'pruning_method': best_pruned_name,
    'n_train_files': 20,
    'n_train_samples': len(y_train_capped),
    'auc_roc': auc_final,
    'f1': f1_final,
    'auc_baseline_67f': auc_baseline,
    'feature_importances': final_importances.to_dict(),
    'all_method_results': {
        name: {'n_features': c['n_features'], 'auc': c['auc'], 'f1': c['f1']}
        for name, c in candidates.items()
    },
}
joblib.dump(artifact, 'models/churn_v10_pruned.joblib')
print(f"\n  Saved: models/churn_v10_pruned.joblib", flush=True)
print(f"  Done ({time.time()-t0:.1f}s)", flush=True)


# ═══════════════════════════════════════════════════════════════
# Charts: charts/phase33_feature_selection.png
# ═══════════════════════════════════════════════════════════════
print("\n  Generating charts...", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# --- Panel 1: AUC vs number of features (top-K curve + RFE curve) ---
ax = axes[0, 0]

# Top-K results
k_vals = sorted(topk_results.keys())
k_aucs = [topk_results[k]['auc'] for k in k_vals]
ax.plot(k_vals, k_aucs, 'o-', color='#3498db', linewidth=2, markersize=6, label='Top-K Selection')

# RFE results
rfe_nf = [r['n_features'] for r in rfe_results]
rfe_aucs = [r['auc'] for r in rfe_results]
ax.plot(rfe_nf, rfe_aucs, 's--', color='#e74c3c', linewidth=2, markersize=6, label='RFE')

# Correlation pruning point
ax.scatter([len(corr_pruned_features)], [auc_corr], marker='D', s=100,
           color='#2ecc71', zorder=5, label=f'Corr-pruned ({len(corr_pruned_features)}f)')

# Baseline line
ax.axhline(y=auc_baseline, color='gray', linestyle=':', linewidth=1, label=f'Baseline 67f ({auc_baseline:.4f})')

# Mark optimal K
ax.scatter([optimal_k], [topk_results[optimal_k]['auc']], marker='*', s=200,
           color='#f39c12', zorder=6, label=f'Optimal K={optimal_k}')

ax.set_xlabel('Number of Features')
ax.set_ylabel('AUC-ROC')
ax.set_title('AUC vs Number of Features')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)

# Annotate key points
for k in [20, 30, optimal_k, 67]:
    if k in topk_results:
        ax.annotate(f'K={k}\n{topk_results[k]["auc"]:.4f}',
                    xy=(k, topk_results[k]['auc']),
                    textcoords="offset points", xytext=(0, 12),
                    ha='center', fontsize=7)

# --- Panel 2: Surviving features (final importances bar chart) ---
ax = axes[0, 1]

# Identify feature categories for color coding
sample_agg = list(tx_agg_cache.values())[0]
sample_time = list(tx_time_cache.values())[0]
agg_only_cols = list(sample_agg.columns)
time_only_cols = list(sample_time.columns)

top_n = min(len(final_importances), 35)
top_final = final_importances.head(top_n)
bar_colors = []
for feat in top_final.index:
    if feat in time_only_cols:
        bar_colors.append('#2ecc71')   # green = time
    elif feat in agg_only_cols:
        bar_colors.append('#e67e22')   # orange = TX agg
    else:
        bar_colors.append('#3498db')   # blue = base

ax.barh(range(len(top_final)), top_final.values, color=bar_colors)
ax.set_yticks(range(len(top_final)))
ax.set_yticklabels(top_final.index, fontsize=6)
ax.invert_yaxis()
ax.set_title(f'Surviving Features ({len(final_features)}f)\n'
             f'Blue=Base, Orange=TX Agg, Green=Time')
ax.set_xlabel('Importance')

# --- Panel 3: Correlation heatmap of top features ---
ax = axes[1, 0]

top_corr_n = min(25, len(final_features))
top_corr_feats = final_importances.head(top_corr_n).index.tolist()
corr_sub = X_train_sub[top_corr_feats].fillna(0).corr()

mask = np.triu(np.ones_like(corr_sub, dtype=bool), k=1)
sns.heatmap(corr_sub, mask=mask, cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=ax, square=True,
            xticklabels=True, yticklabels=True,
            cbar_kws={'shrink': 0.6, 'label': 'Correlation'})
ax.set_xticklabels(ax.get_xticklabels(), fontsize=5, rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)
ax.set_title(f'Correlation Heatmap (Top {top_corr_n} Features)')

# --- Panel 4: Before vs after comparison ---
ax = axes[1, 1]

comparison_labels = ['Full 67f\n(baseline)', f'Pruned {len(final_features)}f\n(v10)']
comparison_aucs = [auc_baseline, auc_final]
comparison_f1s = [f1_baseline, f1_final]

x_pos = np.arange(len(comparison_labels))
w = 0.3

bars_auc = ax.bar(x_pos - w/2, comparison_aucs, w, label='AUC-ROC', color='#3498db')
bars_f1 = ax.bar(x_pos + w/2, comparison_f1s, w, label='F1 Score', color='#e74c3c')

ax.set_xticks(x_pos)
ax.set_xticklabels(comparison_labels, fontsize=10)
ax.set_ylabel('Score')
ax.set_title('Before vs After Feature Pruning')
ax.legend(fontsize=9)

# Set y-axis to show differences clearly
all_vals = comparison_aucs + comparison_f1s
y_min = min(all_vals) - 0.02
y_max = max(all_vals) + 0.02
ax.set_ylim(y_min, y_max)

for bar in bars_auc:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{bar.get_height():.4f}', ha='center', fontsize=9, fontweight='bold')
for bar in bars_f1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{bar.get_height():.4f}', ha='center', fontsize=9, fontweight='bold')

# Add delta annotation
delta_auc_pct = (auc_final - auc_baseline) * 100
delta_f1_pct = (f1_final - f1_baseline) * 100
ax.text(0.5, 0.05,
        f'AUC delta: {delta_auc_pct:+.2f}%  |  F1 delta: {delta_f1_pct:+.2f}%\n'
        f'Features reduced: 67 -> {len(final_features)} (-{67 - len(final_features)})',
        transform=ax.transAxes, ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle(
    f'Phase 33: Feature Selection & Pruning\n'
    f'Baseline 67f AUC={auc_baseline:.4f} | '
    f'Pruned {len(final_features)}f AUC={auc_final:.4f} '
    f'({delta_auc_pct:+.2f}%) | Method: {best_pruned_name}',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()

os.makedirs('charts', exist_ok=True)
plt.savefig('charts/phase33_feature_selection.png', dpi=150, bbox_inches='tight')
print(f"  Saved: charts/phase33_feature_selection.png", flush=True)


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}", flush=True)
print(f"Phase 33: Feature Selection & Pruning — Summary", flush=True)
print(f"{'='*70}", flush=True)
print(f"  Original features         : 67", flush=True)
print(f"  Pruned features           : {len(final_features)}", flush=True)
print(f"  Features dropped          : {67 - len(final_features)}", flush=True)
print(f"  Pruning method            : {best_pruned_name}", flush=True)
print(f"  {'─'*50}", flush=True)
print(f"  Baseline 67f (subsample)  : AUC={auc_baseline:.4f}, F1={f1_baseline:.4f}", flush=True)
print(f"  Final pruned (full train) : AUC={auc_final:.4f}, F1={f1_final:.4f}", flush=True)
print(f"  AUC delta                 : {delta_auc_pct:+.2f}%", flush=True)
print(f"  FN (final)                : {cm_final[1,0]:,}", flush=True)
print(f"  FP (final)                : {cm_final[0,1]:,}", flush=True)
print(f"  {'─'*50}", flush=True)
print(f"  Method 1 (Top-K)          : optimal K={optimal_k}, AUC={topk_results[optimal_k]['auc']:.4f}", flush=True)
print(f"  Method 2 (RFE)            : optimal {rfe_optimal['n_features']}f, AUC={rfe_optimal['auc']:.4f}", flush=True)
print(f"  Method 3 (Corr-pruned)    : {len(corr_pruned_features)}f, AUC={auc_corr:.4f}", flush=True)
print(f"  {'─'*50}", flush=True)
print(f"  High-corr pairs (|r|>0.9) : {len(high_corr_pairs)}", flush=True)
print(f"  Corr-dropped features     : {len(features_to_drop_corr)}", flush=True)
print(f"  {'─'*50}", flush=True)
print(f"  Chart                     : charts/phase33_feature_selection.png", flush=True)
print(f"  Model                     : models/churn_v10_pruned.joblib", flush=True)
print(f"  Train samples (final)     : {len(y_train_capped):,}", flush=True)
print(f"  Test samples              : {len(y_test):,}", flush=True)
print(f"{'='*70}", flush=True)
