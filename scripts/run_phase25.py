#!/usr/bin/env python3
"""
Phase 25: Churn Reason Clustering
==================================
Clusters churned customers by behavioral patterns to identify distinct
churn profiles (e.g., gradual decline, sudden stop, irregular buyers).
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
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
print("Phase 25: Churn Reason Clustering")
print("=" * 70)

# ── Process Test Files ────────────────────────────────────────────────────────

all_feats = []
all_y = []

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

    # Subsample to 100K per file for speed (clustering doesn't need full dataset)
    max_per_file = 100_000
    if len(df_valid) > max_per_file:
        idx = np.random.RandomState(42).choice(len(df_valid), max_per_file, replace=False)
        idx.sort()
        df_valid = df_valid.iloc[idx].reset_index(drop=True)
        mat_valid = mat_valid.iloc[idx].reset_index(drop=True)
        y = y[idx]
        print(f"  Subsampled to {max_per_file:,}")

    # Features
    feats = engineer_features_v2(df_valid, mat_valid, feat_item_cols)

    all_feats.append(feats)
    all_y.append(y)
    print(f"  Customers: {len(feats):,} | Churn rate: {y.mean():.1%}")

feats_all = pd.concat(all_feats, ignore_index=True)
y_all = np.concatenate(all_y)
print(f"\nTotal customers: {len(feats_all):,} | Overall churn rate: {y_all.mean():.1%}")

# ── Filter Churned Customers ─────────────────────────────────────────────────

churned_mask = y_all == 1
feats_churned = feats_all.loc[churned_mask].reset_index(drop=True)
print(f"\nChurned customers for clustering: {len(feats_churned):,}")

# ── Select Clustering Features ────────────────────────────────────────────────

cluster_features = [
    'trend_slope',
    'current_zero_streak',
    'gap_regularity',
    'active_last_3',
    'active_last_6',
    'late_momentum',
    'purchase_acceleration',
    'n_reactivations',
    'tenure_rounds',
    'purchase_frequency_ratio',
]

print(f"\nClustering features ({len(cluster_features)}):")
for f in cluster_features:
    vals = feats_churned[f].values
    print(f"  {f:>30}: mean={np.nanmean(vals):.3f}, std={np.nanstd(vals):.3f}")

X_cluster = feats_churned[cluster_features].fillna(0).values

# ── StandardScaler ────────────────────────────────────────────────────────────

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# ── KMeans: Try K=3,4,5,6 ────────────────────────────────────────────────────

print("\n── Finding Optimal K ──")

k_range = [3, 4, 5, 6]
inertias = []
silhouette_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil)
    print(f"  K={k}: Silhouette={sil:.4f}, Inertia={km.inertia_:,.0f}")

best_k_idx = np.argmax(silhouette_scores)
best_k = k_range[best_k_idx]
best_sil = silhouette_scores[best_k_idx]
print(f"\n  Best K = {best_k} (Silhouette = {best_sil:.4f})")

# ── Final KMeans with Best K ─────────────────────────────────────────────────

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
cluster_labels = kmeans.fit_predict(X_scaled)
feats_churned['cluster'] = cluster_labels

# ── Interpret Clusters ────────────────────────────────────────────────────────

print("\n── Cluster Interpretation ──")

centroids = kmeans.cluster_centers_  # shape: (best_k, 10) — in scaled space
centroids_original = scaler.inverse_transform(centroids)  # back to original space
centroids_df = pd.DataFrame(centroids_original, columns=cluster_features)

# Assign Thai names based on most distinctive feature per cluster
# Normalize centroids to 0-1 for comparison
centroid_min = centroids_original.min(axis=0)
centroid_max = centroids_original.max(axis=0)
centroid_range = centroid_max - centroid_min
centroid_range[centroid_range == 0] = 1  # avoid division by zero
centroids_norm = (centroids_original - centroid_min) / centroid_range

# Feature-to-name mapping for distinctive patterns
pattern_names = {
    'trend_slope': ('ค่อยๆ ลดลง (Gradual Decline)', 'min'),  # most negative slope
    'current_zero_streak': ('หยุดกะทันหัน (Sudden Stop)', 'max'),  # highest zero streak
    'gap_regularity': ('ซื้อไม่สม่ำเสมอ (Irregular)', 'max'),  # highest gap regularity
    'n_reactivations': ('กลับมาแล้วหาย (Reactivated & Lost)', 'max'),  # highest reactivations
    'tenure_rounds': ('ลูกค้าเก่าหายไป (Long-term Lost)', 'max'),  # highest tenure
    'purchase_frequency_ratio': ('ซื้อบ่อยแต่หยุด (Frequent Buyer Stopped)', 'max'),
    'late_momentum': ('โมเมนตัมตก (Momentum Drop)', 'min'),  # most negative momentum
    'active_last_3': ('เพิ่งหยุดซื้อ (Recent Drop)', 'min'),  # lowest recent activity
    'active_last_6': ('ไม่ active นาน (Long Inactive)', 'min'),  # lowest activity
    'purchase_acceleration': ('ชะลอตัว (Decelerating)', 'min'),  # most negative acceleration
}

cluster_names = {}
assigned_features = set()

for _ in range(best_k):
    best_score = -np.inf
    best_cluster = None
    best_feature = None
    best_name = None

    for feat_idx, feat in enumerate(cluster_features):
        if feat not in pattern_names or feat in assigned_features:
            continue
        name, direction = pattern_names[feat]
        for c in range(best_k):
            if c in cluster_names:
                continue
            if direction == 'max':
                score = centroids_original[c, feat_idx]
            else:  # min
                score = -centroids_original[c, feat_idx]
            if score > best_score:
                best_score = score
                best_cluster = c
                best_feature = feat
                best_name = name

    if best_cluster is not None and best_feature is not None:
        cluster_names[best_cluster] = best_name
        assigned_features.add(best_feature)

# Fill any remaining unnamed clusters
for c in range(best_k):
    if c not in cluster_names:
        cluster_names[c] = f'กลุ่ม {c+1} (Cluster {c+1})'

feats_churned['cluster_name'] = feats_churned['cluster'].map(cluster_names)

# Print cluster profiles
print(f"\n  Cluster Profiles (K={best_k}):")
print(f"  {'':>3} {'Name':<40} {'Size':>8} {'Pct':>6}")
print(f"  {'-'*3} {'-'*40} {'-'*8} {'-'*6}")

for c in range(best_k):
    mask = cluster_labels == c
    size = mask.sum()
    pct = size / len(cluster_labels) * 100
    print(f"  {c:>3} {cluster_names[c]:<40} {size:>8,} {pct:>5.1f}%")

print(f"\n  Feature Means per Cluster:")
header = f"  {'Feature':>30}"
for c in range(best_k):
    header += f" | C{c}:{cluster_names[c][:12]:>12}"
print(header)
print(f"  {'-' * (32 + best_k * 17)}")

for feat_idx, feat in enumerate(cluster_features):
    row = f"  {feat:>30}"
    for c in range(best_k):
        row += f" | {centroids_original[c, feat_idx]:>15.3f}"
    print(row)

# ── Save Model ────────────────────────────────────────────────────────────────

print("\n── Saving Cluster Model ──")
cluster_artifact = {
    'kmeans': kmeans,
    'scaler': scaler,
    'cluster_features': cluster_features,
    'cluster_names': cluster_names,
    'best_k': best_k,
    'silhouette_score': best_sil,
}
joblib.dump(cluster_artifact, 'models/churn_clusters.joblib')
print("  Saved: models/churn_clusters.joblib")

# ── Generate Charts ───────────────────────────────────────────────────────────

print("\n── Generating Charts ──")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle('Phase 25: Churn Reason Clustering', fontsize=14, fontweight='bold')

# Subplot 1: Radar/Spider Chart of Cluster Centroids (normalized)
ax1 = axes[0]
ax1.set_axis_off()
# Use polar subplot
ax_radar = fig.add_axes(ax1.get_position(), polar=True)

n_features = len(cluster_features)
angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

colors_radar = ['#F44336', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']

for c in range(best_k):
    values = centroids_norm[c].tolist()
    values += values[:1]  # close
    ax_radar.plot(angles, values, 'o-', linewidth=2, label=cluster_names[c][:20],
                  color=colors_radar[c % len(colors_radar)], markersize=4)
    ax_radar.fill(angles, values, alpha=0.1, color=colors_radar[c % len(colors_radar)])

# Shorten feature names for readability
short_names = [
    'trend_slope', 'zero_streak', 'gap_reg', 'act_last3', 'act_last6',
    'late_mom', 'purch_accel', 'n_react', 'tenure', 'freq_ratio'
]
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(short_names, fontsize=7)
ax_radar.set_title('Cluster Profiles (Normalized)', fontsize=11, pad=20)
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=7)

# Subplot 2: Cluster Size Bar Chart
ax2 = axes[1]
cluster_sizes = [np.sum(cluster_labels == c) for c in range(best_k)]
bar_colors = [colors_radar[c % len(colors_radar)] for c in range(best_k)]
bar_labels = [cluster_names[c] for c in range(best_k)]

bars = ax2.barh(range(best_k), cluster_sizes, color=bar_colors, edgecolor='white', alpha=0.8)
ax2.set_yticks(range(best_k))
ax2.set_yticklabels([f'C{c}' for c in range(best_k)], fontsize=9)
ax2.set_xlabel('Number of Customers')
ax2.set_title('Cluster Sizes')

# Add labels
for i, (bar, name, size) in enumerate(zip(bars, bar_labels, cluster_sizes)):
    ax2.text(bar.get_width() + max(cluster_sizes) * 0.01, i,
             f'{name} ({size:,})', va='center', fontsize=8)

ax2.set_xlim(0, max(cluster_sizes) * 1.6)

# Subplot 3: Elbow Plot + Silhouette Scores
ax3 = axes[2]
color_inertia = '#2196F3'
color_sil = '#F44336'

ax3.plot(k_range, inertias, 'o-', color=color_inertia, linewidth=2, markersize=8, label='Inertia')
ax3.set_xlabel('K (Number of Clusters)')
ax3.set_ylabel('Inertia', color=color_inertia)
ax3.tick_params(axis='y', labelcolor=color_inertia)
ax3.set_title('Elbow Plot & Silhouette Scores')

ax3_twin = ax3.twinx()
ax3_twin.plot(k_range, silhouette_scores, 's-', color=color_sil, linewidth=2, markersize=8,
              label='Silhouette')
ax3_twin.set_ylabel('Silhouette Score', color=color_sil)
ax3_twin.tick_params(axis='y', labelcolor=color_sil)

# Mark best K
ax3_twin.axvline(best_k, color='green', linestyle='--', alpha=0.7, label=f'Best K={best_k}')

# Combined legend
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)

ax3.set_xticks(k_range)

plt.tight_layout()
plt.savefig('charts/phase25_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: charts/phase25_clusters.png")

# ── Summary ───────────────────────────────────────────────────────────────────

elapsed = time.time() - start_time
print(f"\n{'=' * 70}")
print(f"Phase 25 Complete — {elapsed:.1f}s")
print(f"  Churned customers clustered: {len(feats_churned):,}")
print(f"  Optimal K: {best_k} (Silhouette: {best_sil:.4f})")
print(f"  Clusters:")
for c in range(best_k):
    size = np.sum(cluster_labels == c)
    print(f"    C{c}: {cluster_names[c]} — {size:,} customers ({size/len(cluster_labels)*100:.1f}%)")
print(f"  Output: charts/phase25_clusters.png")
print(f"  Model:  models/churn_clusters.joblib")
print(f"{'=' * 70}")
