#!/usr/bin/env python3
"""
Phase 24: Customer Lifetime Value (CLV) Scoring & Priority Matrix
=================================================================
Computes CLV scores for customers and creates a risk x CLV priority matrix
to guide marketing actions based on both churn risk and customer value.
"""

import time
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')
start_time = time.time()

# ── Data Loading ──────────────────────────────────────────────────────────────

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
test_files = all_25_files[23:25]

sys.path.insert(0, 'models')
from feature_engineering import engineer_features_v2

print("=" * 70)
print("Phase 24: Customer Lifetime Value (CLV) & Priority Matrix")
print("=" * 70)

# ── Process Test Files ────────────────────────────────────────────────────────

all_feats = []
all_y = []
all_userno = []

for fname in test_files:
    print(f"\nLoading {fname}...")
    df = pd.read_csv(fname)
    item_cols = [c for c in df.columns if c.startswith('item')]
    feat_item_cols = item_cols[:-1]
    target_col = item_cols[-1]

    mat = df[item_cols].apply(pd.to_numeric, errors='coerce')
    mat_vals = mat.values.astype(float)
    reg = ~np.isnan(mat_vals)
    first_reg = np.where(reg.any(axis=1), reg.argmax(axis=1), len(item_cols))
    tenure = len(item_cols) - first_reg
    valid = tenure >= 3

    df_valid = df.loc[valid].reset_index(drop=True)
    mat_valid = mat.loc[valid].reset_index(drop=True)

    # Target: churn = NaN or 0 in last item col
    target_vals = pd.to_numeric(mat_valid[target_col], errors='coerce').values
    y = np.where(np.isnan(target_vals) | (target_vals == 0), 1, 0)

    # Features
    feats = engineer_features_v2(df_valid, mat_valid, feat_item_cols)
    feats['userNo'] = df_valid['userNo'].values

    all_feats.append(feats)
    all_y.append(y)
    all_userno.append(df_valid['userNo'].values)
    print(f"  Customers: {len(feats):,} | Churn rate: {y.mean():.1%}")

feats_all = pd.concat(all_feats, ignore_index=True)
y_all = np.concatenate(all_y)
print(f"\nTotal customers: {len(feats_all):,} | Overall churn rate: {y_all.mean():.1%}")

# ── CLV Computation ───────────────────────────────────────────────────────────

print("\n── Computing CLV Scores ──")

avg_purchase_value = feats_all['avg_items_per_active_round'].values
purchase_frequency = feats_all['purchase_frequency_ratio'].values
rounds_since_last = feats_all['rounds_since_last_purchase'].values

expected_lifetime = np.maximum(12 - rounds_since_last * 2, 1).astype(float)
expected_lifetime = np.minimum(expected_lifetime, 12)

clv_score = avg_purchase_value * purchase_frequency * expected_lifetime

# Percentile ranking
clv_percentile = np.zeros(len(clv_score))
mask_pos = clv_score > 0
if mask_pos.sum() > 0:
    from scipy.stats import rankdata
    ranks = rankdata(clv_score[mask_pos], method='average')
    clv_percentile[mask_pos] = (ranks / ranks.max()) * 100
# Customers with clv_score == 0 stay at percentile 0

feats_all['clv_score'] = clv_score
feats_all['clv_percentile'] = clv_percentile

# CLV Tiers
conditions = [
    clv_percentile <= 25,
    (clv_percentile > 25) & (clv_percentile <= 50),
    (clv_percentile > 50) & (clv_percentile <= 75),
    clv_percentile > 75,
]
clv_tiers = np.select(conditions, ['Low', 'Medium', 'High', 'Premium'], default='Low')
feats_all['clv_tier'] = clv_tiers

print(f"  CLV Score — Mean: {clv_score.mean():.2f}, Median: {np.median(clv_score):.2f}, "
      f"Std: {clv_score.std():.2f}")
print(f"  CLV Score — Min: {clv_score.min():.2f}, Max: {clv_score.max():.2f}")
print(f"\n  CLV Tier Distribution:")
for tier in ['Low', 'Medium', 'High', 'Premium']:
    n = (clv_tiers == tier).sum()
    print(f"    {tier:>8}: {n:>8,} ({n/len(clv_tiers)*100:.1f}%)")

# ── Load RF v3 Model for Risk Scores ─────────────────────────────────────────

print("\n── Loading v3 Model for Risk Scoring ──")

v3_artifact = joblib.load('models/churn_v3_extended.joblib')
v3_model = v3_artifact['model']
v3_feature_cols = v3_artifact['feature_cols']

rf_probs = v3_model.predict_proba(feats_all[v3_feature_cols].fillna(0))[:, 1]
risk_score = np.clip(np.round(rf_probs * 100, 1), 0, 100)
feats_all['risk_score'] = risk_score

# Risk levels
risk_conditions = [
    risk_score <= 25,
    (risk_score > 25) & (risk_score <= 50),
    (risk_score > 50) & (risk_score <= 75),
    risk_score > 75,
]
risk_levels = np.select(risk_conditions, ['Low', 'Medium', 'High', 'Critical'], default='Low')
feats_all['risk_level'] = risk_levels

print(f"  Risk Score — Mean: {risk_score.mean():.1f}, Median: {np.median(risk_score):.1f}")
print(f"\n  Risk Level Distribution:")
for level in ['Low', 'Medium', 'High', 'Critical']:
    n = (risk_levels == level).sum()
    print(f"    {level:>10}: {n:>8,} ({n/len(risk_levels)*100:.1f}%)")

# ── Priority Matrix ──────────────────────────────────────────────────────────

print("\n── Priority Matrix (Risk Level x CLV Tier) ──")

PRIORITY_ACTIONS = {
    ('Critical', 'Premium'): 'ติดต่อด่วน — ลูกค้ามูลค่าสูงมาก',
    ('Critical', 'High'): 'ติดต่อด่วน — ลูกค้ามูลค่าสูง',
    ('Critical', 'Medium'): 'เร่งติดต่อ — ลูกค้ากำลังจะหาย',
    ('Critical', 'Low'): 'ติดตาม — มูลค่าต่ำ',
    ('High', 'Premium'): 'ส่ง promotion พิเศษ — VIP เริ่มมีสัญญาณ',
    ('High', 'High'): 'ส่ง promotion — รักษาลูกค้าดี',
    ('High', 'Medium'): 'ส่ง offer — เริ่มมีสัญญาณ',
    ('High', 'Low'): 'ส่ง offer เบาๆ',
    ('Medium', 'Premium'): 'ดูแลพิเศษ — VIP ต้องรักษา',
    ('Medium', 'High'): 'รักษาสัมพันธ์ — ลูกค้าสำคัญ',
    ('Medium', 'Medium'): 'ส่ง offer เบาๆ',
    ('Medium', 'Low'): 'ไม่ต้องดำเนินการ',
    ('Low', 'Premium'): 'VIP Care — รักษาความภักดี',
    ('Low', 'High'): 'ขอบคุณลูกค้า — loyalty program',
    ('Low', 'Medium'): 'ไม่ต้องดำเนินการ',
    ('Low', 'Low'): 'ไม่ต้องดำเนินการ',
}

feats_all['priority_action'] = [
    PRIORITY_ACTIONS.get((r, c), 'ไม่ระบุ')
    for r, c in zip(feats_all['risk_level'], feats_all['clv_tier'])
]
feats_all['actual_churn'] = y_all

# Build matrix counts
risk_order = ['Critical', 'High', 'Medium', 'Low']
clv_order = ['Premium', 'High', 'Medium', 'Low']

matrix_counts = np.zeros((4, 4), dtype=int)
matrix_churn_rate = np.zeros((4, 4))

header = 'Risk \\ CLV'
print(f"\n  {header:>12} | {'Premium':>10} | {'High':>10} | {'Medium':>10} | {'Low':>10} |")
print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+")

for i, risk in enumerate(risk_order):
    row_str = f"  {risk:>12} |"
    for j, clv in enumerate(clv_order):
        mask = (feats_all['risk_level'] == risk) & (feats_all['clv_tier'] == clv)
        count = mask.sum()
        churn_rate = y_all[mask].mean() if count > 0 else 0
        matrix_counts[i, j] = count
        matrix_churn_rate[i, j] = churn_rate
        row_str += f" {count:>6} ({churn_rate:.0%}) |"
    print(row_str)

print(f"\n  Priority Actions with Actual Churn Rates:")
for (risk, clv), action in PRIORITY_ACTIONS.items():
    mask = (feats_all['risk_level'] == risk) & (feats_all['clv_tier'] == clv)
    count = mask.sum()
    churn_rate = y_all[mask].mean() if count > 0 else 0
    if count > 0:
        print(f"    [{risk:>8} x {clv:<7}] n={count:>6,} churn={churn_rate:.1%} → {action}")

# ── Generate Charts ───────────────────────────────────────────────────────────

print("\n── Generating Charts ──")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Phase 24: Customer Lifetime Value & Priority Matrix', fontsize=14, fontweight='bold')

# Subplot 1: CLV Score Distribution
ax1 = axes[0]
ax1.hist(clv_score[clv_score > 0], bins=50, color='#2196F3', edgecolor='white', alpha=0.8)
ax1.set_xlabel('CLV Score')
ax1.set_ylabel('Count')
ax1.set_title('CLV Score Distribution')
ax1.axvline(np.median(clv_score[clv_score > 0]), color='red', linestyle='--',
            label=f'Median: {np.median(clv_score[clv_score > 0]):.1f}')
ax1.axvline(np.mean(clv_score[clv_score > 0]), color='orange', linestyle='--',
            label=f'Mean: {np.mean(clv_score[clv_score > 0]):.1f}')
ax1.legend(fontsize=9)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# Subplot 2: Priority Matrix Heatmap
ax2 = axes[1]
cmap = LinearSegmentedColormap.from_list('custom', ['#E8F5E9', '#FFEB3B', '#FF9800', '#F44336'])
im = ax2.imshow(matrix_counts, cmap=cmap, aspect='auto')

ax2.set_xticks(range(4))
ax2.set_xticklabels(clv_order, fontsize=9)
ax2.set_yticks(range(4))
ax2.set_yticklabels(risk_order, fontsize=9)
ax2.set_xlabel('CLV Tier')
ax2.set_ylabel('Risk Level')
ax2.set_title('Priority Matrix (Count per Cell)')

for i in range(4):
    for j in range(4):
        count = matrix_counts[i, j]
        churn = matrix_churn_rate[i, j]
        text_color = 'white' if count > matrix_counts.max() * 0.6 else 'black'
        ax2.text(j, i, f'{count:,}\n({churn:.0%})',
                 ha='center', va='center', fontsize=8, fontweight='bold', color=text_color)

plt.colorbar(im, ax=ax2, shrink=0.8)

# Subplot 3: CLV by Risk Level (box plot)
ax3 = axes[2]
risk_order_plot = ['Low', 'Medium', 'High', 'Critical']
colors = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']
data_by_risk = []
for risk in risk_order_plot:
    mask = feats_all['risk_level'] == risk
    data_by_risk.append(clv_score[mask])

bp = ax3.boxplot(data_by_risk, labels=risk_order_plot, patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax3.set_xlabel('Risk Level')
ax3.set_ylabel('CLV Score')
ax3.set_title('CLV Distribution by Risk Level')

# Add median labels
for i, data in enumerate(data_by_risk):
    med = np.median(data)
    ax3.text(i + 1, med, f'{med:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('charts/phase24_clv.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: charts/phase24_clv.png")

# ── Summary ───────────────────────────────────────────────────────────────────

elapsed = time.time() - start_time
print(f"\n{'=' * 70}")
print(f"Phase 24 Complete — {elapsed:.1f}s")
print(f"  Customers scored: {len(feats_all):,}")
print(f"  CLV Score: mean={clv_score.mean():.1f}, median={np.median(clv_score):.1f}")
print(f"  Output: charts/phase24_clv.png")
print(f"{'=' * 70}")
