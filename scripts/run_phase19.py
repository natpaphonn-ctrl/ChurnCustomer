#!/usr/bin/env python3
"""Phase 19: Sudden Drop Detection — new features to catch FN (loyal customers who suddenly churn)."""

import os, sys, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, 'models')
from feature_engineering import engineer_features_v2, engineer_features_v3
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix

MIN_TENURE = 3

print("=" * 60)
print("Phase 19: Sudden Drop Detection")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# File list (same as Phase 12/18)
# ═══════════════════════════════════════════════════════════════
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
test_files = all_25_files[23:25]


def load_and_engineer(files, version='v3', max_per_file=None):
    """Load CSV files and compute features."""
    all_X, all_y = [], []
    eng_func = engineer_features_v3 if version == 'v3' else engineer_features_v2

    for fname in files:
        if not os.path.exists(fname):
            print(f"  SKIP: {fname}")
            continue
        df = pd.read_csv(fname, low_memory=False)
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

        feats = eng_func(df_v, mat_v, feat_item_cols)
        feats = feats.fillna(0)

        if max_per_file and len(feats) > max_per_file:
            idx = np.random.RandomState(42).choice(len(feats), max_per_file, replace=False)
            feats = feats.iloc[idx]
            y = y[idx]

        all_X.append(feats)
        all_y.append(y)
        churn_rate = y.mean() * 100
        print(f"    {fname}: {len(feats):,} rows, churn={churn_rate:.1f}%")

    X = pd.concat(all_X, ignore_index=True)
    y = np.concatenate(all_y)
    return X, y


# ═══════════════════════════════════════════════════════════════
# 1. Load v3 baseline for comparison
# ═══════════════════════════════════════════════════════════════
print("\n[1/4] Loading v3 baseline...")
t0 = time.time()
v3_artifact = joblib.load('models/churn_v3_extended.joblib')
v3_model = v3_artifact['model']
v3_feature_cols = v3_artifact['feature_cols']
print(f"  RF v3: {len(v3_feature_cols)} features")

# ═══════════════════════════════════════════════════════════════
# 2. Build training data with v3 features (52 features)
# ═══════════════════════════════════════════════════════════════
print("\n[2/4] Building training data (v3 features)...")
print("  Train files (last 3 for speed):")
X_train, y_train = load_and_engineer(train_files[-3:], version='v3', max_per_file=400000)
print(f"  Total train: {len(X_train):,}, churn={y_train.mean()*100:.1f}%")

# Subsample for training speed
if len(X_train) > 1_000_000:
    idx = np.random.RandomState(42).choice(len(X_train), 1_000_000, replace=False)
    X_train = X_train.iloc[idx]
    y_train = y_train[idx]
    print(f"  Subsampled to {len(X_train):,}")

print(f"  ({time.time()-t0:.1f}s)")

print("\n  Test files:")
X_test, y_test = load_and_engineer(test_files, version='v3')
print(f"  Total test: {len(X_test):,}, churn={y_test.mean()*100:.1f}%")

# Also get v2 features for test (baseline comparison)
print("\n  Loading v2 test features for baseline...")
X_test_v2, _ = load_and_engineer(test_files, version='v2')

# ═══════════════════════════════════════════════════════════════
# 3. Train RF v4 with 52 features
# ═══════════════════════════════════════════════════════════════
print("\n[3/4] Training RF v4 (52 features)...")
t0 = time.time()

v4_feature_cols = list(X_train.columns)
print(f"  Features: {len(v4_feature_cols)}")

# New features added
new_features = [c for c in v4_feature_cols if c not in v3_feature_cols]
print(f"  New features: {new_features}")

rf_v4 = RandomForestClassifier(
    n_estimators=300, max_depth=20, min_samples_leaf=50,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf_v4.fit(X_train[v4_feature_cols].fillna(0), y_train)
print(f"  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════
# 4. Evaluate and compare
# ═══════════════════════════════════════════════════════════════
print("\n[4/4] Evaluating...")
t0 = time.time()

# v3 baseline on test
v3_cols_available = [c for c in v3_feature_cols if c in X_test_v2.columns]
rf_v3_probs = v3_model.predict_proba(X_test_v2[v3_cols_available].fillna(0))[:, 1]
v3_auc = roc_auc_score(y_test, rf_v3_probs)
v3_preds = (rf_v3_probs >= 0.48).astype(int)
v3_f1 = f1_score(y_test, v3_preds)
v3_cm = confusion_matrix(y_test, v3_preds)

# v4 on test
rf_v4_probs = rf_v4.predict_proba(X_test[v4_feature_cols].fillna(0))[:, 1]
v4_auc = roc_auc_score(y_test, rf_v4_probs)
v4_preds = (rf_v4_probs >= 0.48).astype(int)
v4_f1 = f1_score(y_test, v4_preds)
v4_cm = confusion_matrix(y_test, v4_preds)

print(f"\n  {'='*60}")
print(f"  Model Comparison (Test: {len(y_test):,} customers)")
print(f"  {'='*60}")
print(f"    {'Model':<35} {'AUC-ROC':>10} {'F1':>10} {'FN':>10}")
print(f"    {'-'*65}")
print(f"    {'RF v3 (45 features)':<35} {v3_auc:>10.4f} {v3_f1:>10.4f} {v3_cm[1,0]:>10,}")
print(f"    {'RF v4 (52 features)':<35} {v4_auc:>10.4f} {v4_f1:>10.4f} {v4_cm[1,0]:>10,}")
print(f"    {'-'*65}")
print(f"    AUC improvement: {(v4_auc - v3_auc)*100:+.2f}%")
print(f"    FN reduction:    {v3_cm[1,0] - v4_cm[1,0]:+,}")

# Feature importance for new features
importances = pd.Series(rf_v4.feature_importances_, index=v4_feature_cols).sort_values(ascending=False)
print(f"\n  New Feature Importance (rank / {len(v4_feature_cols)}):")
for feat in new_features:
    rank = (importances.index.tolist().index(feat) + 1)
    print(f"    {feat:<30} importance={importances[feat]:.4f}  rank={rank}")

# FN profile analysis
fn_mask_v3 = (y_test == 1) & (v3_preds == 0)
fn_mask_v4 = (y_test == 1) & (v4_preds == 0)
print(f"\n  FN Analysis:")
print(f"    v3 FN count: {fn_mask_v3.sum():,}")
print(f"    v4 FN count: {fn_mask_v4.sum():,}")
print(f"    FN saved:    {fn_mask_v3.sum() - fn_mask_v4.sum():,}")

# Among v3's FN, how many does v4 catch?
v3_fn_idx = np.where(fn_mask_v3)[0]
fn_mask_v4_arr = np.array(fn_mask_v4)
v4_catches = (~fn_mask_v4_arr[v3_fn_idx]).sum() if len(v3_fn_idx) > 0 else 0
print(f"    v4 catches {v4_catches:,} of v3's {len(v3_fn_idx):,} FN ({v4_catches/max(len(v3_fn_idx),1)*100:.1f}%)")

print(f"  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════
# Save model and charts
# ═══════════════════════════════════════════════════════════════
print("\nSaving artifacts...")

# Save v4 model
v4_artifact = {
    'model': rf_v4,
    'feature_cols': v4_feature_cols,
    'version': 'v4_sudden_drop',
    'n_features': len(v4_feature_cols),
    'auc_roc': v4_auc,
}
joblib.dump(v4_artifact, 'models/churn_v4_sudden.joblib')
print(f"  Saved: models/churn_v4_sudden.joblib")

# Chart
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. Feature importance (top 20)
top20 = importances.head(20)
colors = ['#e74c3c' if f in new_features else '#3498db' for f in top20.index]
axes[0].barh(range(len(top20)), top20.values, color=colors)
axes[0].set_yticks(range(len(top20)))
axes[0].set_yticklabels(top20.index, fontsize=8)
axes[0].invert_yaxis()
axes[0].set_title('Feature Importance (Top 20)\nRed = New Sudden-Drop Features')
axes[0].set_xlabel('Importance')

# 2. Confusion matrix comparison
labels = ['v3 (45 feat)', 'v4 (52 feat)']
fn_vals = [v3_cm[1, 0], v4_cm[1, 0]]
fp_vals = [v3_cm[0, 1], v4_cm[0, 1]]
x = np.arange(2)
w = 0.35
axes[1].bar(x - w/2, fn_vals, w, label='FN (missed churn)', color='#e74c3c')
axes[1].bar(x + w/2, fp_vals, w, label='FP (false alarm)', color='#f39c12')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].set_title('Error Comparison')
axes[1].legend()
axes[1].set_ylabel('Count')
for i, (fn, fp) in enumerate(zip(fn_vals, fp_vals)):
    axes[1].text(i - w/2, fn + 1000, f'{fn:,}', ha='center', fontsize=8)
    axes[1].text(i + w/2, fp + 1000, f'{fp:,}', ha='center', fontsize=8)

# 3. New feature distributions for FN vs TP
if len(new_features) > 0:
    tp_mask = (y_test == 1) & (v4_preds == 1)
    fn_mask = fn_mask_v4
    feat_data = []
    for f in new_features:
        if f in X_test.columns:
            tp_mean = X_test[f].values[tp_mask].mean()
            fn_mean = X_test[f].values[fn_mask].mean()
            feat_data.append((f, tp_mean, fn_mean))
    if feat_data:
        names, tp_means, fn_means = zip(*feat_data)
        y_pos = np.arange(len(names))
        axes[2].barh(y_pos - 0.2, tp_means, 0.4, label='TP (caught churn)', color='#27ae60')
        axes[2].barh(y_pos + 0.2, fn_means, 0.4, label='FN (missed churn)', color='#e74c3c')
        axes[2].set_yticks(y_pos)
        axes[2].set_yticklabels(names, fontsize=8)
        axes[2].set_title('New Feature Values: TP vs FN')
        axes[2].legend()
        axes[2].invert_yaxis()

plt.suptitle(f'Phase 19: Sudden Drop Detection\nv3 AUC={v3_auc:.4f} → v4 AUC={v4_auc:.4f} ({(v4_auc-v3_auc)*100:+.2f}%)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/phase19_sudden_drop.png', dpi=150, bbox_inches='tight')
print(f"  Saved: charts/phase19_sudden_drop.png")

print(f"\n{'='*60}")
print(f"DONE — Phase 19 Complete!")
print(f"{'='*60}")
