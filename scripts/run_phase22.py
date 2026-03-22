"""
Phase 22: Drift Detection
=========================
Analyze feature distribution drift across 25 time periods using PSI
(Population Stability Index) and track churn rate changes over time.

Output:
  - charts/phase22_drift_heatmap.png
  - models/drift_reference.json
"""

import sys
import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')

sys.path.insert(0, 'models')
from feature_engineering import engineer_features_v2

# ── Config ──────────────────────────────────────────────────────────────────

SUBSAMPLE = 50_000
REFERENCE_PERIODS = 10  # first N files used as reference distribution
MIN_TENURE = 3

TOP_FEATURES = [
    'purchase_frequency_ratio', 'active_last_3', 'active_last_6',
    'items_last_3', 'avg_gap', 'ewm_recent', 'trend_slope',
    'gini_coefficient', 'max_active_streak', 'tenure_rounds',
    'items_last_6', 'current_zero_streak', 'recent_gap_vs_avg',
    'late_momentum', 'trend_r_squared',
]

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


def extract_period_label(filename):
    """Extract short period label from filename, e.g. '0216' from 'Churn_2025_0216_0301.csv'."""
    parts = filename.replace('.csv', '').split('_')
    return parts[2]  # e.g. '0216'


def compute_psi(expected, actual, bins=10):
    """Compute Population Stability Index between two distributions."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[-1] = np.inf
    breakpoints[0] = -np.inf
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi


def load_and_process(filepath, subsample=SUBSAMPLE):
    """Load a CSV, compute features and churn label. Returns (feats_df, y, n_total)."""
    df = pd.read_csv(filepath)
    item_cols = [c for c in df.columns if c.startswith('item')]
    feat_item_cols = item_cols[:-1]
    target_col = item_cols[-1]

    mat = df[item_cols].apply(pd.to_numeric, errors='coerce')
    mat_vals = mat.values.astype(float)
    reg = ~np.isnan(mat_vals)
    first_reg = np.where(reg.any(axis=1), reg.argmax(axis=1), len(item_cols))
    tenure = len(item_cols) - first_reg
    valid = tenure >= MIN_TENURE

    df_valid = df[valid].reset_index(drop=True)
    mat_valid = mat[valid].reset_index(drop=True)
    y_raw = mat_valid[target_col].values
    y = (np.isnan(y_raw) | (y_raw == 0)).astype(int)

    n_total = len(df_valid)

    # Subsample for speed
    if subsample and len(df_valid) > subsample:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(df_valid), subsample, replace=False)
        idx.sort()
        df_valid = df_valid.iloc[idx].reset_index(drop=True)
        mat_valid = mat_valid.iloc[idx].reset_index(drop=True)
        y = y[idx]

    feats = engineer_features_v2(df_valid, mat_valid, feat_item_cols)
    return feats, y, n_total


# ── Phase 1: Process all 25 files ──────────────────────────────────────────

print("=" * 70)
print("PHASE 22: DRIFT DETECTION")
print("=" * 70)

period_labels = [extract_period_label(f) for f in all_25_files]
period_stats = []        # dict per period: means, stds, churn_rate, n_customers
period_feat_values = {}  # feature_name -> list of arrays (one per period)

# Initialize storage for feature values
for feat in TOP_FEATURES:
    period_feat_values[feat] = []

total_start = time.time()

for i, filename in enumerate(all_25_files):
    t0 = time.time()
    label = period_labels[i]
    filepath = filename  # assume running from project root

    if not os.path.exists(filepath):
        print(f"  [{i+1:2d}/25] {label} — FILE NOT FOUND: {filename}, skipping")
        period_stats.append(None)
        for feat in TOP_FEATURES:
            period_feat_values[feat].append(None)
        continue

    print(f"  [{i+1:2d}/25] {label} — loading {filename}...", end=' ', flush=True)
    feats, y, n_total = load_and_process(filepath)

    churn_rate = float(y.mean())
    n_customers = n_total

    # Compute feature statistics
    means = {}
    stds = {}
    for feat in TOP_FEATURES:
        if feat in feats.columns:
            vals = feats[feat].values.astype(float)
            vals = vals[~np.isnan(vals)]
            means[feat] = float(np.mean(vals)) if len(vals) > 0 else 0.0
            stds[feat] = float(np.std(vals)) if len(vals) > 0 else 0.0
            period_feat_values[feat].append(vals)
        else:
            means[feat] = 0.0
            stds[feat] = 0.0
            period_feat_values[feat].append(None)

    period_stats.append({
        'label': label,
        'filename': filename,
        'churn_rate': churn_rate,
        'n_customers': n_customers,
        'means': means,
        'stds': stds,
    })

    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s) — n={n_customers:,}, churn={churn_rate:.3f}")

total_elapsed = time.time() - total_start
print(f"\nAll files processed in {total_elapsed:.1f}s")


# ── Phase 2: Compute PSI ───────────────────────────────────────────────────

print("\n" + "-" * 70)
print("Computing PSI (reference = first 10 periods)...")
print("-" * 70)

# Build reference distribution from first REFERENCE_PERIODS files
ref_values = {}
for feat in TOP_FEATURES:
    arrays = []
    for j in range(REFERENCE_PERIODS):
        if period_feat_values[feat][j] is not None:
            arrays.append(period_feat_values[feat][j])
    if arrays:
        ref_values[feat] = np.concatenate(arrays)
    else:
        ref_values[feat] = None

# Compute PSI for each period (including reference periods for completeness)
psi_matrix = np.zeros((len(TOP_FEATURES), len(all_25_files)))
psi_matrix[:] = np.nan

for fi, feat in enumerate(TOP_FEATURES):
    if ref_values[feat] is None:
        continue
    for pi in range(len(all_25_files)):
        vals = period_feat_values[feat][pi]
        if vals is not None and len(vals) > 10:
            psi_matrix[fi, pi] = compute_psi(ref_values[feat], vals)

# Print PSI summary
print(f"\n{'Feature':<30s} {'Mean PSI (ref)':>14s} {'Mean PSI (late)':>15s} {'Max PSI':>10s}")
print("-" * 72)
for fi, feat in enumerate(TOP_FEATURES):
    ref_psi = psi_matrix[fi, :REFERENCE_PERIODS]
    late_psi = psi_matrix[fi, REFERENCE_PERIODS:]
    ref_mean = np.nanmean(ref_psi) if not np.all(np.isnan(ref_psi)) else 0.0
    late_mean = np.nanmean(late_psi) if not np.all(np.isnan(late_psi)) else 0.0
    max_psi = np.nanmax(psi_matrix[fi, :]) if not np.all(np.isnan(psi_matrix[fi, :])) else 0.0
    flag = " *** DRIFT" if max_psi >= 0.25 else (" *" if max_psi >= 0.1 else "")
    print(f"  {feat:<28s} {ref_mean:>12.4f}   {late_mean:>13.4f}   {max_psi:>8.4f}{flag}")


# ── Phase 3: Generate charts ───────────────────────────────────────────────

print("\n" + "-" * 70)
print("Generating charts...")
print("-" * 70)

fig, axes = plt.subplots(3, 1, figsize=(18, 20), gridspec_kw={'height_ratios': [3, 1, 1]})
fig.suptitle('Phase 22: Feature Drift Detection (PSI) — 25 Periods',
             fontsize=16, fontweight='bold', y=0.98)

# --- Subplot 1: PSI Heatmap ---
ax1 = axes[0]

# Custom colormap: green < 0.1, yellow < 0.25, red >= 0.25
cmap_colors = [
    (0.0, '#2d8a4e'),    # green — no drift
    (0.1 / 0.5, '#2d8a4e'),
    (0.1 / 0.5 + 0.001, '#f0c929'),  # yellow — moderate
    (0.25 / 0.5, '#f0c929'),
    (0.25 / 0.5 + 0.001, '#d32f2f'),  # red — significant
    (1.0, '#d32f2f'),
]
drift_cmap = LinearSegmentedColormap.from_list('drift', cmap_colors)

# Mask NaN values
psi_display = psi_matrix.copy()
masked_psi = np.ma.masked_invalid(psi_display)

im = ax1.imshow(masked_psi, aspect='auto', cmap=drift_cmap, vmin=0, vmax=0.5,
                interpolation='nearest')
cbar = fig.colorbar(im, ax=ax1, shrink=0.8, pad=0.02)
cbar.set_label('PSI Value', fontsize=11)
cbar.ax.axhline(y=0.1, color='black', linewidth=1, linestyle='--')
cbar.ax.axhline(y=0.25, color='black', linewidth=1, linestyle='--')

ax1.set_xticks(range(len(period_labels)))
ax1.set_xticklabels(period_labels, rotation=45, ha='right', fontsize=8)
ax1.set_yticks(range(len(TOP_FEATURES)))
ax1.set_yticklabels(TOP_FEATURES, fontsize=9)
ax1.set_xlabel('Period', fontsize=11)
ax1.set_ylabel('Feature', fontsize=11)
ax1.set_title('PSI Heatmap (green < 0.1, yellow < 0.25, red >= 0.25)', fontsize=13)

# Add PSI values as text on cells
for fi in range(len(TOP_FEATURES)):
    for pi in range(len(period_labels)):
        val = psi_matrix[fi, pi]
        if not np.isnan(val):
            color = 'white' if val >= 0.25 else 'black'
            ax1.text(pi, fi, f'{val:.2f}', ha='center', va='center',
                     fontsize=5.5, color=color)

# Add vertical line separating reference from comparison
ax1.axvline(x=REFERENCE_PERIODS - 0.5, color='white', linewidth=2, linestyle='--')
ax1.text(REFERENCE_PERIODS - 0.5, -0.8, 'ref|compare', ha='center', va='bottom',
         fontsize=8, color='gray', style='italic')

# --- Subplot 2: Churn Rate Over Time ---
ax2 = axes[1]

valid_labels = []
churn_rates = []
n_customers_list = []
for ps in period_stats:
    if ps is not None:
        valid_labels.append(ps['label'])
        churn_rates.append(ps['churn_rate'])
        n_customers_list.append(ps['n_customers'])
    else:
        valid_labels.append('?')
        churn_rates.append(np.nan)
        n_customers_list.append(0)

x_pos = range(len(valid_labels))
ax2.plot(x_pos, churn_rates, 'o-', color='#d32f2f', linewidth=2, markersize=5, label='Churn Rate')
ax2.axhline(y=np.nanmean(churn_rates), color='gray', linewidth=1, linestyle='--',
            label=f'Mean: {np.nanmean(churn_rates):.3f}')
ax2.fill_between(x_pos, churn_rates, alpha=0.15, color='#d32f2f')

ax2.set_xticks(list(x_pos))
ax2.set_xticklabels(valid_labels, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Churn Rate', fontsize=11)
ax2.set_title('Churn Rate Over Time (25 Periods)', fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)

# Annotate with customer counts
for xi, (cr, nc) in enumerate(zip(churn_rates, n_customers_list)):
    if not np.isnan(cr):
        ax2.annotate(f'{cr:.2%}', (xi, cr), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=6, color='#d32f2f')

# --- Subplot 3: Top-5 Feature Means Over Time ---
ax3 = axes[2]

# Pick top 5 features (first 5 from TOP_FEATURES)
top5 = TOP_FEATURES[:5]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for fi, feat in enumerate(top5):
    feat_means = []
    for ps in period_stats:
        if ps is not None and feat in ps['means']:
            feat_means.append(ps['means'][feat])
        else:
            feat_means.append(np.nan)
    ax3.plot(x_pos, feat_means, 'o-', color=colors[fi], linewidth=1.5,
             markersize=4, label=feat, alpha=0.85)

ax3.set_xticks(list(x_pos))
ax3.set_xticklabels(valid_labels, rotation=45, ha='right', fontsize=8)
ax3.set_ylabel('Feature Mean Value', fontsize=11)
ax3.set_title('Top-5 Feature Mean Values Over Time', fontsize=13)
ax3.legend(fontsize=8, loc='upper left', ncol=2)
ax3.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
chart_path = 'charts/phase22_drift_heatmap.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {chart_path}")


# ── Phase 4: Save reference distributions ──────────────────────────────────

print("\n" + "-" * 70)
print("Saving reference distributions...")
print("-" * 70)

# Compute reference means and stds from first 10 periods
ref_means = {}
ref_stds = {}
for feat in TOP_FEATURES:
    ref_vals_list = []
    for j in range(REFERENCE_PERIODS):
        if period_stats[j] is not None and feat in period_stats[j]['means']:
            ref_vals_list.append(period_stats[j]['means'][feat])
    if ref_vals_list:
        ref_means[feat] = float(np.mean(ref_vals_list))
        ref_stds[feat] = float(np.std(ref_vals_list))
    else:
        ref_means[feat] = 0.0
        ref_stds[feat] = 0.0

# Build per-period metrics
per_period_metrics = []
for i, ps in enumerate(period_stats):
    if ps is None:
        per_period_metrics.append(None)
        continue
    psi_vals = {}
    for fi, feat in enumerate(TOP_FEATURES):
        val = psi_matrix[fi, i]
        psi_vals[feat] = float(val) if not np.isnan(val) else None
    per_period_metrics.append({
        'label': ps['label'],
        'filename': ps['filename'],
        'churn_rate': ps['churn_rate'],
        'n_customers': ps['n_customers'],
        'feature_means': ps['means'],
        'feature_stds': ps['stds'],
        'psi': psi_vals,
    })

drift_ref = {
    'description': 'Phase 22 Drift Detection — reference distributions and PSI results',
    'reference_periods': REFERENCE_PERIODS,
    'reference_files': all_25_files[:REFERENCE_PERIODS],
    'monitored_features': TOP_FEATURES,
    'reference_means': ref_means,
    'reference_stds': ref_stds,
    'psi_thresholds': {
        'no_drift': 0.1,
        'moderate_drift': 0.25,
        'significant_drift': 0.25,
    },
    'per_period_metrics': per_period_metrics,
}

ref_path = 'models/drift_reference.json'
with open(ref_path, 'w', encoding='utf-8') as f:
    json.dump(drift_ref, f, indent=2, ensure_ascii=False)
print(f"  Saved: {ref_path}")


# ── Phase 5: Summary ───────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("DRIFT DETECTION SUMMARY")
print("=" * 70)

# Features with significant drift
print("\nFeatures with significant drift (PSI >= 0.25):")
sig_drift_found = False
for fi, feat in enumerate(TOP_FEATURES):
    max_psi = np.nanmax(psi_matrix[fi, :]) if not np.all(np.isnan(psi_matrix[fi, :])) else 0.0
    if max_psi >= 0.25:
        # Find which periods
        drifted_periods = []
        for pi in range(len(period_labels)):
            if not np.isnan(psi_matrix[fi, pi]) and psi_matrix[fi, pi] >= 0.25:
                drifted_periods.append(period_labels[pi])
        print(f"  - {feat}: max PSI = {max_psi:.4f} (periods: {', '.join(drifted_periods)})")
        sig_drift_found = True

if not sig_drift_found:
    print("  (none — all features stable)")

# Features with moderate drift
print("\nFeatures with moderate drift (0.1 <= PSI < 0.25):")
mod_drift_found = False
for fi, feat in enumerate(TOP_FEATURES):
    max_psi = np.nanmax(psi_matrix[fi, :]) if not np.all(np.isnan(psi_matrix[fi, :])) else 0.0
    if 0.1 <= max_psi < 0.25:
        print(f"  - {feat}: max PSI = {max_psi:.4f}")
        mod_drift_found = True

if not mod_drift_found:
    print("  (none)")

# Churn rate trend
valid_churn = [cr for cr in churn_rates if not np.isnan(cr)]
if len(valid_churn) >= 2:
    first_half = np.mean(valid_churn[:len(valid_churn)//2])
    second_half = np.mean(valid_churn[len(valid_churn)//2:])
    trend_dir = "INCREASING" if second_half > first_half else "DECREASING"
    change_pct = abs(second_half - first_half) / first_half * 100
    print(f"\nChurn Rate Trend: {trend_dir}")
    print(f"  First half avg:  {first_half:.4f}")
    print(f"  Second half avg: {second_half:.4f}")
    print(f"  Change:          {change_pct:.1f}%")

print(f"\nTotal customers across 25 periods: {sum(n_customers_list):,}")
print(f"Average churn rate: {np.nanmean(churn_rates):.4f}")
print(f"\nOutputs:")
print(f"  - {chart_path}")
print(f"  - {ref_path}")
print("=" * 70)
