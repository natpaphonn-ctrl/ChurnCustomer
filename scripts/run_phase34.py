#!/usr/bin/env python3
"""Phase 34: Demographics Feature Test.

Test whether age, gender, province help predict churn.
Data quality issues: age outliers (< 20, > 100), province unreliable (migration).

Features:
  - is_male, is_female (binary; UNSPECIFIED = both 0)
  - age_clipped (20-80, outliers clipped)
  - age_suspicious (age < 20 or > 80 flag)
  - age_group (categorical → ordinal: 20-30, 31-40, 41-50, 51-60, 60+)
  - is_bangkok (binary)
  - region (กทม/ปริมณฑล/เหนือ/อีสาน/กลาง/ใต้/ตะวันออก → ordinal)

Output:
  - charts/phase34_demographics.png
  - models/churn_v11_demo.joblib
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
from feature_engineering import engineer_features_v2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

MIN_TENURE = 3

print("=" * 70, flush=True)
print("Phase 34: Demographics Feature Test", flush=True)
print("=" * 70, flush=True)

# ═══════════════════════════════════════════════════════════════
# Province → Region mapping
# ═══════════════════════════════════════════════════════════════

BANGKOK_METRO = {'กรุงเทพมหานคร', 'นนนทบุรี', 'นนทบุรี', 'ปทุมธานี', 'สมุทรปราการ',
                 'สมุทรสาคร', 'นครปฐม'}

NORTH = {'เชียงใหม่', 'เชียงราย', 'ลำปาง', 'ลำพูน', 'แม่ฮ่องสอน', 'น่าน',
         'พะเยา', 'แพร่', 'อุตรดิตถ์', 'ตาก', 'สุโขทัย', 'พิษณุโลก',
         'พิจิตร', 'กำแพงเพชร', 'เพชรบูรณ์', 'นครสวรรค์', 'อุทัยธานี', 'ชัยนาท'}

NORTHEAST = {'นครราชสีมา', 'ขอนแก่น', 'อุดรธานี', 'อุบลราชธานี', 'บุรีรัมย์',
             'สุรินทร์', 'ศรีสะเกษ', 'ร้อยเอ็ด', 'ชัยภูมิ', 'มหาสารคาม',
             'กาฬสินธุ์', 'สกลนคร', 'นครพนม', 'มุกดาหาร', 'เลย', 'หนองคาย',
             'หนองบัวลำภู', 'บึงกาฬ', 'ยโสธร', 'อำนาจเจริญ'}

CENTRAL = {'อยุธยา', 'พระนครศรีอยุธยา', 'อ่างทอง', 'ลพบุรี', 'สิงห์บุรี',
           'สระบุรี', 'สุพรรณบุรี', 'กาญจนบุรี', 'ราชบุรี', 'เพชรบุรี',
           'ประจวบคีรีขันธ์', 'สมุทรสงคราม', 'นครนายก', 'ปราจีนบุรี',
           'สระแก้ว', 'ชัยนาท'}

EAST = {'ชลบุรี', 'ระยอง', 'จันทบุรี', 'ตราด', 'ฉะเชิงเทรา'}

SOUTH = {'สงขลา', 'ภูเก็ต', 'สุราษฎร์ธานี', 'นครศรีธรรมราช', 'กระบี่',
         'พังงา', 'ตรัง', 'พัทลุง', 'สตูล', 'ชุมพร', 'ระนอง',
         'ยะลา', 'ปัตตานี', 'นราธิวาส'}


def get_region(province):
    """Map province to region code (0-6)."""
    if province in BANGKOK_METRO:
        return 1  # กทม+ปริมณฑล
    elif province in NORTH:
        return 2
    elif province in NORTHEAST:
        return 3
    elif province in CENTRAL:
        return 4
    elif province in EAST:
        return 5
    elif province in SOUTH:
        return 6
    else:
        return 0  # ไม่ระบุ / unknown


def compute_demo_features(df):
    """Compute demographic features from age, gender, province columns."""
    demo = pd.DataFrame(index=df.index)

    # Gender
    demo['is_male'] = (df['gender'] == 'MALE').astype(int)
    demo['is_female'] = (df['gender'] == 'FEMALE').astype(int)

    # Age — clip to 20-80, flag suspicious
    age = df['age'].copy()
    demo['age_suspicious'] = ((age < 20) | (age > 80)).astype(int)
    demo['age_clipped'] = age.clip(20, 80)

    # Age group (ordinal)
    age_c = demo['age_clipped']
    demo['age_group'] = pd.cut(age_c, bins=[0, 30, 40, 50, 60, 100],
                                labels=[1, 2, 3, 4, 5]).astype(float).fillna(3)

    # Province → Region
    demo['region'] = df['province'].map(get_region).fillna(0).astype(int)
    demo['is_bangkok'] = (demo['region'] == 1).astype(int)

    return demo


# ═══════════════════════════════════════════════════════════════
# Train/Test pairs (same as Phase 33 — use v2 features + demo)
# ═══════════════════════════════════════════════════════════════

train_files = [
    'data/churn/Churn_2025_0216_0301.csv', 'data/churn/Churn_2025_0301_0316.csv',
    'data/churn/Churn_2025_0316_0401.csv', 'data/churn/Churn_2025_0401_0416.csv',
    'data/churn/Churn_2025_0416_0502.csv', 'data/churn/Churn_2025_0502_0516.csv',
    'data/churn/Churn_2025_0516_0601.csv', 'data/churn/Churn_2025_0601_0616.csv',
    'data/churn/Churn_2025_0616_0701.csv', 'data/churn/Churn_2025_0701_0716.csv',
    'data/churn/Churn_2025_0716_0801.csv', 'data/churn/Churn_2025_0801_0816.csv',
    'data/churn/Churn_2025_0816_0901.csv', 'data/churn/Churn_2025_0901_0916.csv',
    'data/churn/Churn_2025_0916_1001.csv', 'data/churn/Churn_2025_1001_1016.csv',
    'data/churn/Churn_2025_1016_1101.csv', 'data/churn/Churn_2025_1101_1116.csv',
    'data/churn/Churn_2025_1116_1201.csv', 'data/churn/Churn_2025_1201_1216.csv',
]

test_files = [
    'data/churn/Churn_2026_0201_0216.csv',
    'data/churn/Churn_2026_0216_0301.csv',
]


def load_features_with_demo(churn_file, include_demo=True):
    """Load Churn CSV, compute v2 features (45) + optional demo features."""
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

    df_v = df[valid].copy().reset_index(drop=True)
    mat_v = mat[valid].copy().reset_index(drop=True)

    y_raw = mat_v[target_col].values
    y = (np.isnan(y_raw) | (y_raw == 0)).astype(int)

    feats = engineer_features_v2(df_v, mat_v, feat_item_cols)
    feats = feats.fillna(0)

    if include_demo:
        demo = compute_demo_features(df_v)
        feats = pd.concat([feats, demo], axis=1)

    return feats, y


# ═══════════════════════════════════════════════════════════════
# 1. Data quality analysis
# ═══════════════════════════════════════════════════════════════
print("\n[1/5] Data quality analysis...", flush=True)
t0 = time.time()

df_sample = pd.read_csv(test_files[0], usecols=['age', 'gender', 'province'], low_memory=False)
n_total = len(df_sample)

age_stats = {
    'valid_20_80': ((df_sample['age'] >= 20) & (df_sample['age'] <= 80)).sum(),
    'under_20': (df_sample['age'] < 20).sum(),
    'over_80': (df_sample['age'] > 80).sum(),
    'over_100': (df_sample['age'] > 100).sum(),
    'median': df_sample['age'].median(),
    'mean_clipped': df_sample['age'].clip(20, 80).mean(),
}

gender_stats = df_sample['gender'].value_counts()
province_stats = df_sample['province'].value_counts()

print(f"  Sample: {n_total:,} customers", flush=True)
print(f"  Age valid (20-80): {age_stats['valid_20_80']:,} ({age_stats['valid_20_80']/n_total*100:.1f}%)", flush=True)
print(f"  Age < 20: {age_stats['under_20']:,}, > 80: {age_stats['over_80']:,}, > 100: {age_stats['over_100']:,}", flush=True)
print(f"  Age median: {age_stats['median']:.0f}, mean (clipped): {age_stats['mean_clipped']:.1f}", flush=True)
print(f"  Gender: M={gender_stats.get('MALE',0):,}, F={gender_stats.get('FEMALE',0):,}, "
      f"Unspec={gender_stats.get('UNSPECIFIED',0):,}", flush=True)
print(f"  Provinces: {df_sample['province'].nunique()}, top=กรุงเทพมหานคร "
      f"({province_stats.iloc[0]:,})", flush=True)
print(f"  Done ({time.time()-t0:.1f}s)", flush=True)


# ═══════════════════════════════════════════════════════════════
# 2. Load training data (subsample for speed)
# ═══════════════════════════════════════════════════════════════
print("\n[2/5] Loading training data (last 5 files for speed)...", flush=True)
t0 = time.time()

# Use last 5 train files (most recent, most representative)
train_subset = train_files[-5:]
SUBSAMPLE = 1_000_000
rng = np.random.RandomState(42)

# Load WITH demo
all_X_demo, all_y_demo = [], []
all_X_base, all_y_base = [], []

for f in train_subset:
    print(f"  {os.path.basename(f)}...", end=" ", flush=True)
    feats_demo, y = load_features_with_demo(f, include_demo=True)
    feats_base, _ = load_features_with_demo(f, include_demo=False)
    print(f"n={len(y):,}, churn={y.mean()*100:.1f}%, demo_cols={feats_demo.shape[1]-feats_base.shape[1]}", flush=True)

    all_X_demo.append(feats_demo)
    all_y_demo.append(y)
    all_X_base.append(feats_base)
    all_y_base.append(y)

X_train_demo = pd.concat(all_X_demo, ignore_index=True)
y_train_demo = np.concatenate(all_y_demo)
X_train_base = pd.concat(all_X_base, ignore_index=True)
y_train_base = np.concatenate(all_y_base)

print(f"\n  Train total: {len(y_train_demo):,}, features (base): {X_train_base.shape[1]}, "
      f"features (demo): {X_train_demo.shape[1]}", flush=True)

# Subsample
if len(y_train_demo) > SUBSAMPLE:
    idx = rng.choice(len(y_train_demo), SUBSAMPLE, replace=False)
    X_train_demo = X_train_demo.iloc[idx].reset_index(drop=True)
    y_train_demo = y_train_demo[idx]
    X_train_base = X_train_base.iloc[idx].reset_index(drop=True)
    y_train_base = y_train_base[idx]
    print(f"  Subsampled to {SUBSAMPLE:,}", flush=True)

print(f"  Done ({time.time()-t0:.1f}s)", flush=True)


# ═══════════════════════════════════════════════════════════════
# 3. Load test data
# ═══════════════════════════════════════════════════════════════
print("\n[3/5] Loading test data...", flush=True)
t0 = time.time()

all_X_test_demo, all_y_test = [], []
all_X_test_base = []

for f in test_files:
    print(f"  {os.path.basename(f)}...", end=" ", flush=True)
    feats_demo, y = load_features_with_demo(f, include_demo=True)
    feats_base, _ = load_features_with_demo(f, include_demo=False)
    print(f"n={len(y):,}, churn={y.mean()*100:.1f}%", flush=True)

    all_X_test_demo.append(feats_demo)
    all_y_test.append(y)
    all_X_test_base.append(feats_base)

X_test_demo = pd.concat(all_X_test_demo, ignore_index=True)
y_test = np.concatenate(all_y_test)
X_test_base = pd.concat(all_X_test_base, ignore_index=True)

print(f"  Test total: {len(y_test):,}, churn={y_test.mean()*100:.1f}%", flush=True)
print(f"  Done ({time.time()-t0:.1f}s)", flush=True)


# ═══════════════════════════════════════════════════════════════
# 4. Train & compare models
# ═══════════════════════════════════════════════════════════════
print("\n[4/5] Training models...", flush=True)
t0 = time.time()

RF_PARAMS = dict(
    n_estimators=300, max_depth=20, min_samples_leaf=50,
    class_weight='balanced', random_state=42, n_jobs=-1,
)
THRESHOLD = 0.48

demo_cols = ['is_male', 'is_female', 'age_clipped', 'age_suspicious', 'age_group', 'region', 'is_bangkok']

# Model A: Base 45f (no demo)
print("\n  Training Model A: Base 45f (no demo)...", flush=True)
t_a = time.time()
rf_base = RandomForestClassifier(**RF_PARAMS)
rf_base.fit(X_train_base.fillna(0), y_train_base)
probs_base = rf_base.predict_proba(X_test_base.fillna(0))[:, 1]
preds_base = (probs_base >= THRESHOLD).astype(int)
auc_base = roc_auc_score(y_test, probs_base)
f1_base = f1_score(y_test, preds_base)
cm_base = confusion_matrix(y_test, preds_base)
print(f"    AUC={auc_base:.4f}, F1={f1_base:.4f}, FN={cm_base[1,0]:,} ({time.time()-t_a:.1f}s)", flush=True)

# Model B: Base 45f + Demo 7f = 52f
print("\n  Training Model B: Base + Demo (52f)...", flush=True)
t_b = time.time()
rf_demo = RandomForestClassifier(**RF_PARAMS)
rf_demo.fit(X_train_demo.fillna(0), y_train_demo)
probs_demo = rf_demo.predict_proba(X_test_demo.fillna(0))[:, 1]
preds_demo = (probs_demo >= THRESHOLD).astype(int)
auc_demo = roc_auc_score(y_test, probs_demo)
f1_demo = f1_score(y_test, preds_demo)
cm_demo = confusion_matrix(y_test, preds_demo)
print(f"    AUC={auc_demo:.4f}, F1={f1_demo:.4f}, FN={cm_demo[1,0]:,} ({time.time()-t_b:.1f}s)", flush=True)

# Model C: Demo only (7f) — how predictive are demographics alone?
print("\n  Training Model C: Demo only (7f)...", flush=True)
t_c = time.time()
rf_demo_only = RandomForestClassifier(**RF_PARAMS)
rf_demo_only.fit(X_train_demo[demo_cols].fillna(0), y_train_demo)
probs_demo_only = rf_demo_only.predict_proba(X_test_demo[demo_cols].fillna(0))[:, 1]
preds_demo_only = (probs_demo_only >= THRESHOLD).astype(int)
auc_demo_only = roc_auc_score(y_test, probs_demo_only)
f1_demo_only = f1_score(y_test, preds_demo_only)
print(f"    AUC={auc_demo_only:.4f}, F1={f1_demo_only:.4f} ({time.time()-t_c:.1f}s)", flush=True)

print(f"\n  Done ({time.time()-t0:.1f}s)", flush=True)

# Feature importance for demo features
demo_importances = pd.Series(
    rf_demo.feature_importances_, index=X_train_demo.columns
).sort_values(ascending=False)

print(f"\n  Feature importances (demo features):", flush=True)
for feat in demo_cols:
    imp = demo_importances.get(feat, 0)
    rank = list(demo_importances.index).index(feat) + 1
    print(f"    {feat:<25} importance={imp:.4f}  rank={rank}/{len(demo_importances)}", flush=True)


# ═══════════════════════════════════════════════════════════════
# 5. Comparison & insights
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}", flush=True)
print(f"  Model Comparison (Test: {len(y_test):,} customers)", flush=True)
print(f"{'='*70}", flush=True)
print(f"  {'Model':<35} {'Features':>10} {'AUC-ROC':>10} {'F1':>10} {'delta AUC':>12}", flush=True)
print(f"  {'-'*70}", flush=True)

delta_demo = (auc_demo - auc_base) * 100
delta_demo_only = (auc_demo_only - auc_base) * 100

print(f"  {'Base 45f (no demo)':<35} {X_train_base.shape[1]:>10} {auc_base:>10.4f} {f1_base:>10.4f} {'baseline':>12}", flush=True)
print(f"  {'Base + Demo (52f)':<35} {X_train_demo.shape[1]:>10} {auc_demo:>10.4f} {f1_demo:>10.4f} {delta_demo:>+11.2f}%", flush=True)
print(f"  {'Demo only (7f)':<35} {7:>10} {auc_demo_only:>10.4f} {f1_demo_only:>10.4f} {delta_demo_only:>+11.2f}%", flush=True)
print(f"  {'-'*70}", flush=True)

# FN analysis
fn_base_set = set(np.where((y_test == 1) & (preds_base == 0))[0])
fn_demo_set = set(np.where((y_test == 1) & (preds_demo == 0))[0])
fn_saved = fn_base_set - fn_demo_set
fn_lost = fn_demo_set - fn_base_set

print(f"\n  FN Analysis:", flush=True)
print(f"    Base FN: {len(fn_base_set):,}", flush=True)
print(f"    Demo FN: {len(fn_demo_set):,}", flush=True)
print(f"    FN saved by demo: {len(fn_saved):,}", flush=True)
print(f"    FN lost by demo: {len(fn_lost):,}", flush=True)
print(f"    Net FN change: {len(fn_demo_set) - len(fn_base_set):+,}", flush=True)

# Churn rate by demographics
print(f"\n  Churn Rate by Demographics (test set):", flush=True)

test_demo = compute_demo_features(pd.read_csv(test_files[0], low_memory=False))
# Use the full test demo data
test_full_demo = X_test_demo[demo_cols].copy()
test_full_demo['churn'] = y_test

print(f"\n    By Gender:", flush=True)
for g_val, g_name in [(1, 'Male'), (0, 'Female (is_male=0, is_female=1)')]:
    if g_val == 1:
        mask = test_full_demo['is_male'] == 1
    else:
        mask = test_full_demo['is_female'] == 1
    if mask.sum() > 0:
        cr = test_full_demo.loc[mask, 'churn'].mean() * 100
        print(f"      {g_name}: {cr:.1f}% churn (n={mask.sum():,})", flush=True)

# Unspecified
mask_unspec = (test_full_demo['is_male'] == 0) & (test_full_demo['is_female'] == 0)
if mask_unspec.sum() > 0:
    cr = test_full_demo.loc[mask_unspec, 'churn'].mean() * 100
    print(f"      Unspecified: {cr:.1f}% churn (n={mask_unspec.sum():,})", flush=True)

print(f"\n    By Age Group:", flush=True)
for ag in sorted(test_full_demo['age_group'].unique()):
    mask = test_full_demo['age_group'] == ag
    if mask.sum() > 100:
        cr = test_full_demo.loc[mask, 'churn'].mean() * 100
        labels = {1: '20-30', 2: '31-40', 3: '41-50', 4: '51-60', 5: '60+'}
        print(f"      {labels.get(ag, ag)}: {cr:.1f}% churn (n={mask.sum():,})", flush=True)

print(f"\n    By Region:", flush=True)
region_names = {0: 'ไม่ระบุ', 1: 'กทม+ปริมณฑล', 2: 'เหนือ', 3: 'อีสาน',
                4: 'กลาง', 5: 'ตะวันออก', 6: 'ใต้'}
for r in sorted(test_full_demo['region'].unique()):
    mask = test_full_demo['region'] == r
    if mask.sum() > 100:
        cr = test_full_demo.loc[mask, 'churn'].mean() * 100
        print(f"      {region_names.get(r, r)}: {cr:.1f}% churn (n={mask.sum():,})", flush=True)

print(f"\n    Bangkok vs Non-Bangkok:", flush=True)
for bkk_val, bkk_name in [(1, 'Bangkok+Metro'), (0, 'Non-Bangkok')]:
    mask = test_full_demo['is_bangkok'] == bkk_val
    if mask.sum() > 0:
        cr = test_full_demo.loc[mask, 'churn'].mean() * 100
        print(f"      {bkk_name}: {cr:.1f}% churn (n={mask.sum():,})", flush=True)


# ═══════════════════════════════════════════════════════════════
# Save model & chart
# ═══════════════════════════════════════════════════════════════
print(f"\n  Saving artifacts...", flush=True)

artifact = {
    'model': rf_demo,
    'feature_cols': list(X_train_demo.columns),
    'demo_cols': demo_cols,
    'version': 'v11_demo',
    'n_features': X_train_demo.shape[1],
    'auc_roc': auc_demo,
    'f1': f1_demo,
    'auc_base': auc_base,
    'auc_demo_only': auc_demo_only,
    'demo_importances': {feat: float(demo_importances.get(feat, 0)) for feat in demo_cols},
}
joblib.dump(artifact, 'models/churn_v11_demo.joblib')
print(f"  Saved: models/churn_v11_demo.joblib", flush=True)

# Chart
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: AUC comparison bar chart
ax = axes[0, 0]
models = ['Base 45f', f'Base+Demo\n{X_train_demo.shape[1]}f', 'Demo only\n7f']
aucs = [auc_base, auc_demo, auc_demo_only]
colors = ['#3498db', '#e74c3c', '#95a5a6']
bars = ax.bar(models, aucs, color=colors, width=0.5)
ax.set_ylabel('AUC-ROC')
ax.set_title('AUC: Base vs Base+Demo vs Demo Only')
y_min = min(aucs) - 0.05
y_max = max(aucs) + 0.02
ax.set_ylim(y_min, y_max)
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{auc:.4f}', ha='center', fontweight='bold', fontsize=10)

# Panel 2: Demo feature importances
ax = axes[0, 1]
demo_imp_vals = [demo_importances.get(f, 0) for f in demo_cols]
demo_colors = ['#3498db' if v > 0.005 else '#bdc3c7' for v in demo_imp_vals]
ax.barh(range(len(demo_cols)), demo_imp_vals, color=demo_colors)
ax.set_yticks(range(len(demo_cols)))
ax.set_yticklabels(demo_cols, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Importance')
ax.set_title('Demographic Feature Importances')

# Panel 3: Churn rate by age group
ax = axes[1, 0]
age_labels, age_churn_rates, age_counts = [], [], []
for ag in sorted(test_full_demo['age_group'].unique()):
    mask = test_full_demo['age_group'] == ag
    if mask.sum() > 100:
        labels = {1: '20-30', 2: '31-40', 3: '41-50', 4: '51-60', 5: '60+'}
        age_labels.append(labels.get(ag, str(ag)))
        age_churn_rates.append(test_full_demo.loc[mask, 'churn'].mean() * 100)
        age_counts.append(mask.sum())

bars_age = ax.bar(age_labels, age_churn_rates, color='#e74c3c', alpha=0.8)
ax.set_ylabel('Churn Rate (%)')
ax.set_xlabel('Age Group')
ax.set_title('Churn Rate by Age Group')
for bar, cr, cnt in zip(bars_age, age_churn_rates, age_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{cr:.1f}%\n(n={cnt:,})', ha='center', fontsize=8)

# Panel 4: Churn rate by region
ax = axes[1, 1]
reg_labels, reg_churn_rates = [], []
for r in sorted(test_full_demo['region'].unique()):
    mask = test_full_demo['region'] == r
    if mask.sum() > 100:
        reg_labels.append(region_names.get(r, str(r)))
        reg_churn_rates.append(test_full_demo.loc[mask, 'churn'].mean() * 100)

bars_reg = ax.barh(range(len(reg_labels)), reg_churn_rates, color='#2ecc71', alpha=0.8)
ax.set_yticks(range(len(reg_labels)))
ax.set_yticklabels(reg_labels, fontsize=9)
ax.set_xlabel('Churn Rate (%)')
ax.set_title('Churn Rate by Region')
for bar, cr in zip(bars_reg, reg_churn_rates):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{cr:.1f}%', va='center', fontsize=9)

plt.suptitle(
    f'Phase 34: Demographics Feature Test\n'
    f'Base AUC={auc_base:.4f} | +Demo AUC={auc_demo:.4f} '
    f'({"+" if delta_demo >= 0 else ""}{delta_demo:.2f}%) | '
    f'Demo only AUC={auc_demo_only:.4f}',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()

os.makedirs('charts', exist_ok=True)
plt.savefig('charts/phase34_demographics.png', dpi=150, bbox_inches='tight')
print(f"  Saved: charts/phase34_demographics.png", flush=True)


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}", flush=True)
print(f"Phase 34: Demographics Feature Test — Summary", flush=True)
print(f"{'='*70}", flush=True)
print(f"  Base features             : {X_train_base.shape[1]}", flush=True)
print(f"  Demo features added       : {len(demo_cols)}", flush=True)
print(f"  Total features            : {X_train_demo.shape[1]}", flush=True)
print(f"  {'─'*50}", flush=True)
print(f"  Base AUC                  : {auc_base:.4f}", flush=True)
print(f"  Base + Demo AUC           : {auc_demo:.4f} ({delta_demo:+.2f}%)", flush=True)
print(f"  Demo only AUC             : {auc_demo_only:.4f}", flush=True)
print(f"  {'─'*50}", flush=True)
print(f"  Data quality:", flush=True)
print(f"    Age suspicious (< 20 or > 80): {age_stats['over_80'] + age_stats['under_20']:,} ({(age_stats['over_80'] + age_stats['under_20'])/n_total*100:.1f}%)", flush=True)
print(f"    Gender UNSPECIFIED            : {gender_stats.get('UNSPECIFIED', 0):,} ({gender_stats.get('UNSPECIFIED', 0)/n_total*100:.1f}%)", flush=True)
print(f"  {'─'*50}", flush=True)
verdict = "HELPS" if delta_demo > 0.05 else "MARGINAL" if delta_demo > 0 else "NO HELP"
print(f"  Verdict                   : {verdict}", flush=True)
print(f"  Chart                     : charts/phase34_demographics.png", flush=True)
print(f"  Model                     : models/churn_v11_demo.joblib", flush=True)
print(f"{'='*70}", flush=True)
