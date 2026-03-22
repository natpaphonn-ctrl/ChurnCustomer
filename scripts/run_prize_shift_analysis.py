#!/usr/bin/env python3
"""
Prize Shift Analysis — ผลกระทบของการถูกรางวัลต่อพฤติกรรมซื้อและ Risk Level
============================================================================
Quantifies how winning a prize shifts customer behavior:
  Part 1: Ticket uplift per prize type (before vs after winning)
  Part 2: Risk level transition matrix (before → after winning)
  Part 3: Revenue forecast from winning effect
  Part 4: Actionable shift targets per risk level × prize type

Run: python3 scripts/run_prize_shift_analysis.py
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PRIZE_DIR = ROOT / 'data' / 'prize'
CHURN_DIR = ROOT / 'data' / 'churn'
PRED_DIR  = ROOT / 'data' / 'predictions'
CHART_DIR = ROOT / 'charts'
CHART_DIR.mkdir(exist_ok=True)

# Thai fonts
for fp in ['/System/Library/Fonts/Supplemental/Angsana New.ttf',
           '/System/Library/Fonts/Thonburi.ttc',
           '/usr/share/fonts/truetype/thai/']:
    if os.path.exists(fp):
        from matplotlib import font_manager
        font_manager.fontManager.addfont(fp) if fp.endswith(('.ttf','.ttc')) else None

plt.rcParams.update({
    'figure.dpi': 150,
    'axes.unicode_minus': False,
    'font.size': 10,
})

# ── prize type metadata ────────────────────────────────────────────────────
PRIZE_META = {
    'LAST2DIGITS':     {'amount': 2_000,     'label': 'เลขท้าย 2 ตัว (2K)'},
    'FIRST3DIGITS':    {'amount': 4_000,     'label': 'เลขหน้า 3 ตัว (4K)'},
    'LAST3DIGITS':     {'amount': 4_000,     'label': 'เลขท้าย 3 ตัว (4K)'},
    'FIFTH':           {'amount': 20_000,    'label': 'รางวัลที่ 5 (20K)'},
    'FOURTH':          {'amount': 40_000,    'label': 'รางวัลที่ 4 (40K)'},
    'THIRD':           {'amount': 80_000,    'label': 'รางวัลที่ 3 (80K)'},
    'SECOND':          {'amount': 200_000,   'label': 'รางวัลที่ 2 (200K)'},
    'NEARBY':          {'amount': 100_000,   'label': 'รางวัลข้างเคียง (100K)'},
    'FIRST':           {'amount': 6_000_000, 'label': 'รางวัลที่ 1 (6M)'},
}

PRIZE_ORDER = ['LAST2DIGITS','FIRST3DIGITS','LAST3DIGITS','FIFTH','FOURTH',
               'THIRD','SECOND','NEARBY','FIRST']

RISK_LEVELS = ['Low', 'Medium', 'High', 'Critical']
RISK_COLORS = {'Low': '#27ae60', 'Medium': '#f39c12', 'High': '#e67e22', 'Critical': '#e74c3c'}

TICKET_PRICE = 120  # THB per ticket


# ══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def prize_rounddate_to_item_col(rounddate_str):
    """Convert prize rounddate '2025-01-02' → 'item2025_01_02'."""
    return 'item' + rounddate_str.replace('-', '_')


def load_latest_churn_file():
    """Load the churn file with the most item columns (latest)."""
    files = sorted(CHURN_DIR.glob('Churn_20*_*.csv'))
    # exclude special files
    files = [f for f in files if 'Check' not in f.name and 'old_validation' not in f.name]
    if not files:
        raise FileNotFoundError("No churn files found")
    # pick the one with most columns (most periods)
    best_file = None
    best_ncols = 0
    for f in files[-5:]:  # check last 5
        cols = pd.read_csv(f, nrows=0).columns
        item_cols = [c for c in cols if c.startswith('item')]
        if len(item_cols) > best_ncols:
            best_ncols = len(item_cols)
            best_file = f
    return best_file, best_ncols


def classify_risk_by_behavior(items_recent_3, items_recent_6, active_recent_3, active_recent_6):
    """
    Simple risk proxy based on recent purchase behavior.
    - Low:      active 5-6 of last 6 AND avg items >= 3
    - Medium:   active 3-4 of last 6
    - High:     active 1-2 of last 6
    - Critical: active 0 of last 6
    """
    if active_recent_6 >= 5 and items_recent_6 > 0:
        return 'Low'
    elif active_recent_6 >= 3:
        return 'Medium'
    elif active_recent_6 >= 1:
        return 'High'
    else:
        return 'Critical'


# ══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("Prize Shift Analysis — ผลกระทบของการถูกรางวัลต่อพฤติกรรมลูกค้า")
print("=" * 70)

# 1. Load all prize files
print("\n[1] Loading prize data...")
prize_dfs = []
for f in sorted(PRIZE_DIR.glob('Prize_*.csv')):
    df = pd.read_csv(f)
    prize_dfs.append(df)
prize_all = pd.concat(prize_dfs, ignore_index=True)
print(f"    Total prize records: {len(prize_all):,}")
print(f"    Unique winners: {prize_all['userNo'].nunique():,}")
print(f"    Prize types: {sorted(prize_all['type'].unique())}")
print(f"    Period range: {prize_all['rounddate'].min()} → {prize_all['rounddate'].max()}")

# 2. Load the latest churn file (has most periods)
print("\n[2] Loading churn data (latest file for maximum period coverage)...")
churn_file, n_item_cols = load_latest_churn_file()
print(f"    Using: {churn_file.name} ({n_item_cols} periods)")

# Load in chunks to manage memory — only keep userNo + item columns
churn_cols = pd.read_csv(churn_file, nrows=0).columns.tolist()
item_cols = [c for c in churn_cols if c.startswith('item')]
use_cols = ['userNo'] + item_cols
churn_df = pd.read_csv(churn_file, usecols=use_cols)
print(f"    Customers loaded: {len(churn_df):,}")
print(f"    Item columns: {len(item_cols)} ({item_cols[0]} → {item_cols[-1]})")

# Create ordered period list from item columns
# item2023_07_01 → '2023-07-01'
periods = []
for c in item_cols:
    parts = c.replace('item', '').split('_')
    periods.append(f"{parts[0]}-{parts[1]}-{parts[2]}")
period_to_idx = {p: i for i, p in enumerate(periods)}
print(f"    Period range: {periods[0]} → {periods[-1]}")

# Get unique prize rounddates and map to item column indices
prize_dates = sorted(prize_all['rounddate'].unique())
print(f"\n    Prize dates: {len(prize_dates)} periods")

# Filter prize dates that exist in churn item columns
valid_prize_dates = [d for d in prize_dates if d in period_to_idx]
print(f"    Prize dates in churn data: {len(valid_prize_dates)}")

# We need at least 3 periods before and 3 after for analysis
# So exclude first 3 and last 3 periods
usable_prize_dates = [d for d in valid_prize_dates
                      if period_to_idx[d] >= 3 and period_to_idx[d] <= len(periods) - 4]
print(f"    Usable prize dates (3 before + 3 after): {len(usable_prize_dates)}")
if usable_prize_dates:
    print(f"    Usable range: {usable_prize_dates[0]} → {usable_prize_dates[-1]}")


# ══════════════════════════════════════════════════════════════════════════
# PART 1: TICKET UPLIFT PER PRIZE TYPE
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Part 1: Ticket Uplift per Prize Type")
print("=" * 70)

# For each winner in a usable period, get their ticket counts before/after
# Convert item columns to numpy for fast access
item_matrix = churn_df[item_cols].values  # shape: (n_customers, n_periods)
user_index = pd.Series(range(len(churn_df)), index=churn_df['userNo'])

# Filter prize records to usable dates
prize_usable = prize_all[prize_all['rounddate'].isin(usable_prize_dates)].copy()
print(f"\nAnalyzing {len(prize_usable):,} prize records ({prize_usable['userNo'].nunique():,} unique winners)")

# For each winner, compute before/after metrics
results = []
matched = 0
unmatched = 0

for _, row in prize_usable.iterrows():
    user = row['userNo']
    rdate = row['rounddate']
    ptype = row['type']

    if user not in user_index.index:
        unmatched += 1
        continue

    uid = user_index[user]
    # handle duplicates — take first
    if isinstance(uid, pd.Series):
        uid = uid.iloc[0]

    pidx = period_to_idx[rdate]

    # Get 3 periods before and 3 periods after the win period
    before_indices = list(range(max(0, pidx - 3), pidx))
    after_indices = list(range(pidx + 1, min(len(periods), pidx + 4)))

    if len(before_indices) < 3 or len(after_indices) < 3:
        continue

    before_vals = item_matrix[uid, before_indices]
    after_vals = item_matrix[uid, after_indices]
    win_val = item_matrix[uid, pidx]

    # Replace NaN with 0 for calculation (NaN = not registered = no purchase)
    before_vals = np.where(np.isnan(before_vals), 0, before_vals)
    after_vals = np.where(np.isnan(after_vals), 0, after_vals)
    win_val = 0 if np.isnan(win_val) else win_val

    # Count active periods
    before_active = np.sum(before_vals > 0)
    after_active = np.sum(after_vals > 0)

    results.append({
        'userNo': user,
        'prize_type': ptype,
        'rounddate': rdate,
        'total_prize': row['total_prize'],
        'before_avg': np.mean(before_vals),
        'after_avg': np.mean(after_vals),
        'win_period_items': win_val,
        'before_active': before_active,
        'after_active': after_active,
        'before_items_sum': np.sum(before_vals),
        'after_items_sum': np.sum(after_vals),
    })
    matched += 1

print(f"    Matched in churn data: {matched:,}")
print(f"    Not found in churn data: {unmatched:,}")

results_df = pd.DataFrame(results)
results_df['uplift'] = results_df['after_avg'] - results_df['before_avg']
results_df['uplift_pct'] = np.where(
    results_df['before_avg'] > 0,
    (results_df['uplift'] / results_df['before_avg'] * 100),
    0
)

# Aggregate by prize type
uplift_summary = results_df.groupby('prize_type').agg(
    n_winners=('userNo', 'nunique'),
    n_records=('userNo', 'count'),
    before_avg=('before_avg', 'mean'),
    after_avg=('after_avg', 'mean'),
    win_period_avg=('win_period_items', 'mean'),
    uplift_mean=('uplift', 'mean'),
    uplift_median=('uplift', 'median'),
    uplift_pct_mean=('uplift_pct', 'mean'),
    before_active_avg=('before_active', 'mean'),
    after_active_avg=('after_active', 'mean'),
).reindex(PRIZE_ORDER).dropna(how='all')

# Count winners per period (average)
n_usable_periods = len(usable_prize_dates)
uplift_summary['winners_per_period'] = uplift_summary['n_records'] / n_usable_periods
uplift_summary['prize_amount'] = [PRIZE_META.get(t, {}).get('amount', 0) for t in uplift_summary.index]

# Duration estimate: how many periods does the effect last?
# Simple: compare after_active vs before_active
uplift_summary['activity_change'] = uplift_summary['after_active_avg'] - uplift_summary['before_active_avg']

# Total uplift = uplift per winner × winners per period
uplift_summary['total_uplift_per_period'] = uplift_summary['uplift_mean'] * uplift_summary['winners_per_period']
uplift_summary['annual_uplift'] = uplift_summary['total_uplift_per_period'] * 24  # 24 periods/year

print(f"\n{'Prize Type':<16} | {'Winners/Period':>14} | {'Before':>8} | {'After':>8} | {'Uplift':>10} | {'Uplift%':>8} | {'Total/Period':>12}")
print("-" * 100)
for ptype in uplift_summary.index:
    r = uplift_summary.loc[ptype]
    print(f"{ptype:<16} | {r['winners_per_period']:>14.0f} | {r['before_avg']:>8.1f} | {r['after_avg']:>8.1f} | "
          f"{r['uplift_mean']:>+10.2f} | {r['uplift_pct_mean']:>+7.1f}% | {r['total_uplift_per_period']:>12,.0f}")


# ══════════════════════════════════════════════════════════════════════════
# PART 2: RISK LEVEL TRANSITION MATRIX
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Part 2: Risk Level Transition Matrix (Before → After Winning)")
print("=" * 70)

# Classify risk before and after for each winner
def get_risk_level(vals_6):
    """
    Classify risk from 6 period values, accounting for tenure.
    Only count periods where customer was registered (non-NaN).
    Use active RATIO (active/registered) not raw count.
    - Low:      active ratio >= 80%  AND registered >= 3
    - Medium:   active ratio >= 50%
    - High:     active ratio >= 20%  OR active >= 1
    - Critical: active ratio < 20%  AND active == 0
    """
    registered = np.sum(~np.isnan(vals_6))
    vals_clean = np.where(np.isnan(vals_6), 0, vals_6)
    active = np.sum(vals_clean > 0)

    if registered == 0:
        return 'Critical'

    ratio = active / registered

    if ratio >= 0.8 and registered >= 3:
        return 'Low'
    elif ratio >= 0.5:
        return 'Medium'
    elif active >= 1:
        return 'High'
    else:
        return 'Critical'


risk_results = []
for _, row in prize_usable.iterrows():
    user = row['userNo']
    rdate = row['rounddate']
    ptype = row['type']

    if user not in user_index.index:
        continue

    uid = user_index[user]
    if isinstance(uid, pd.Series):
        uid = uid.iloc[0]

    pidx = period_to_idx[rdate]

    # Need 6 periods before and 6 after for robust risk classification
    # Use 6 before win for "before risk" and 6 after win for "after risk"
    # If not enough, use what we have (min 3)
    before_start = max(0, pidx - 6)
    after_end = min(len(periods), pidx + 7)

    before_vals = item_matrix[uid, before_start:pidx]
    after_vals = item_matrix[uid, pidx+1:after_end]

    if len(before_vals) < 3 or len(after_vals) < 3:
        continue

    # Pad to 6 if shorter
    if len(before_vals) < 6:
        before_vals = np.pad(before_vals, (6 - len(before_vals), 0), constant_values=np.nan)
    if len(after_vals) < 6:
        after_vals = np.pad(after_vals, (0, 6 - len(after_vals)), constant_values=np.nan)

    # Take last 6 for before, first 6 for after
    before_6 = before_vals[-6:]
    after_6 = after_vals[:6]

    risk_before = get_risk_level(before_6)
    risk_after = get_risk_level(after_6)

    risk_results.append({
        'userNo': user,
        'prize_type': ptype,
        'risk_before': risk_before,
        'risk_after': risk_after,
        'total_prize': row['total_prize'],
    })

risk_df = pd.DataFrame(risk_results)
print(f"\nRisk classifications: {len(risk_df):,} records")

# Overall transition matrix
print("\n--- Overall Transition Matrix (All Prize Types) ---")
transition = pd.crosstab(
    risk_df['risk_before'], risk_df['risk_after'],
    normalize='index'
) * 100

# Reindex to ensure correct order
transition = transition.reindex(index=RISK_LEVELS, columns=RISK_LEVELS, fill_value=0)

print(f"\n{'':>12} →  {'Low':>8}  {'Medium':>8}  {'High':>8}  {'Critical':>8}  | {'Count':>8}")
print("-" * 70)
for rl in RISK_LEVELS:
    counts = len(risk_df[risk_df['risk_before'] == rl])
    vals = [f"{transition.loc[rl, c]:>7.1f}%" for c in RISK_LEVELS]
    print(f"From {rl:<8}  {'  '.join(vals)}  | {counts:>8,}")

# Transition matrix by prize type
print("\n--- Shift Rate by Prize Type (% who shifted DOWN at least 1 level) ---")
risk_order = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
risk_df['before_rank'] = risk_df['risk_before'].map(risk_order)
risk_df['after_rank'] = risk_df['risk_after'].map(risk_order)
risk_df['shifted_down'] = risk_df['after_rank'] < risk_df['before_rank']
risk_df['shifted_up'] = risk_df['after_rank'] > risk_df['before_rank']

shift_by_type = risk_df.groupby('prize_type').agg(
    n=('userNo', 'count'),
    pct_shifted_down=('shifted_down', 'mean'),
    pct_shifted_up=('shifted_up', 'mean'),
    avg_rank_change=('after_rank', lambda x: x.mean()),
).reindex(PRIZE_ORDER).dropna(how='all')

shift_by_type['before_avg_rank'] = risk_df.groupby('prize_type')['before_rank'].mean().reindex(PRIZE_ORDER)
shift_by_type['rank_change'] = shift_by_type['avg_rank_change'] - shift_by_type['before_avg_rank']

print(f"\n{'Prize Type':<16} | {'N':>7} | {'Shifted Down':>12} | {'Shifted Up':>12} | {'Rank Change':>12}")
print("-" * 75)
for ptype in shift_by_type.index:
    r = shift_by_type.loc[ptype]
    print(f"{ptype:<16} | {r['n']:>7,.0f} | {r['pct_shifted_down']:>11.1%} | {r['pct_shifted_up']:>11.1%} | {r['rank_change']:>+11.3f}")


# ══════════════════════════════════════════════════════════════════════════
# PART 3: REVENUE FORECAST
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Part 3: Revenue Forecast — รายได้เพิ่มจาก Winning Effect")
print("=" * 70)

# Assume uplift lasts ~3 periods on average (conservative)
EFFECT_DURATION = 3  # periods

revenue = uplift_summary[['winners_per_period', 'uplift_mean', 'prize_amount']].copy()
revenue['additional_tickets_per_winner'] = revenue['uplift_mean'] * EFFECT_DURATION
revenue['additional_tickets_per_period'] = revenue['uplift_mean'] * revenue['winners_per_period']
revenue['annual_additional_tickets'] = revenue['additional_tickets_per_period'] * 24
revenue['annual_revenue_thb'] = revenue['annual_additional_tickets'] * TICKET_PRICE

print(f"\n{'Prize Type':<16} | {'Winners/Pd':>10} | {'Uplift/Pd':>10} | {'Add. Tickets/Pd':>15} | {'Annual Tickets':>14} | {'Annual Revenue':>14}")
print("-" * 100)
total_annual_tickets = 0
total_annual_revenue = 0
for ptype in revenue.index:
    r = revenue.loc[ptype]
    print(f"{ptype:<16} | {r['winners_per_period']:>10,.0f} | {r['uplift_mean']:>+10.2f} | "
          f"{r['additional_tickets_per_period']:>15,.0f} | {r['annual_additional_tickets']:>14,.0f} | "
          f"{r['annual_revenue_thb']:>13,.0f}฿")
    total_annual_tickets += r['annual_additional_tickets']
    total_annual_revenue += r['annual_revenue_thb']

print("-" * 100)
print(f"{'TOTAL':<16} | {'':>10} | {'':>10} | {'':>15} | {total_annual_tickets:>14,.0f} | {total_annual_revenue:>13,.0f}฿")
print(f"\nTotal annual additional revenue from winning effect: {total_annual_revenue:,.0f} THB ({total_annual_revenue/1e6:,.1f}M)")


# ══════════════════════════════════════════════════════════════════════════
# PART 4: ACTIONABLE SHIFT TARGETS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Part 4: Shift Targets — ตารางประเมินการ Shift ของลูกค้า")
print("=" * 70)

# For each (current_risk, prize_type) → what % shift to each new level?
print(f"\n{'Current Risk':<12} | {'Prize Type':<16} | {'Expected Shift':<25} | {'Probability':>11} | {'N':>6}")
print("-" * 85)

shift_targets = []
for rl in ['Medium', 'High', 'Critical']:  # Low doesn't need to shift
    for ptype in PRIZE_ORDER:
        subset = risk_df[(risk_df['risk_before'] == rl) & (risk_df['prize_type'] == ptype)]
        if len(subset) < 5:  # skip tiny samples
            continue

        # What's the most likely new risk level?
        after_dist = subset['risk_after'].value_counts(normalize=True)
        most_likely = after_dist.index[0]
        most_likely_pct = after_dist.iloc[0]

        # % that shifted down
        pct_down = subset['shifted_down'].mean()

        # Most common downward shift
        down_subset = subset[subset['shifted_down']]
        if len(down_subset) > 0:
            target_level = down_subset['risk_after'].value_counts().index[0]
            shift_desc = f"{rl} → {target_level}"
        else:
            shift_desc = f"{rl} → {rl} (ไม่เปลี่ยน)"

        print(f"{rl:<12} | {ptype:<16} | {shift_desc:<25} | {pct_down:>10.1%} | {len(subset):>6,}")

        shift_targets.append({
            'current_risk': rl,
            'prize_type': ptype,
            'n': len(subset),
            'pct_shifted_down': pct_down,
            'most_likely_after': most_likely,
            'shift_description': shift_desc,
        })

shift_targets_df = pd.DataFrame(shift_targets)

# Example customers
print("\n--- ตัวอย่างลูกค้าที่ Shift ลง (จริง) ---")
shifted_examples = risk_df[risk_df['shifted_down']].head(10)
for _, ex in shifted_examples.iterrows():
    print(f"  {ex['userNo']}: {ex['risk_before']} → {ex['risk_after']} (ถูก{ex['prize_type']}, เงินรางวัล {ex['total_prize']:,.0f}฿)")


# ══════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Generating charts...")
print("=" * 70)

fig = plt.figure(figsize=(20, 24))
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3,
                       left=0.07, right=0.95, top=0.95, bottom=0.03)

# ── Panel 1: Ticket Uplift by Prize Type ──
ax1 = fig.add_subplot(gs[0, 0])
valid_types = [t for t in PRIZE_ORDER if t in uplift_summary.index]
uplift_vals = [uplift_summary.loc[t, 'uplift_mean'] for t in valid_types]
labels = [PRIZE_META.get(t, {}).get('label', t) for t in valid_types]
amounts = [PRIZE_META.get(t, {}).get('amount', 0) for t in valid_types]

# Color gradient by prize amount
norm = plt.Normalize(min(amounts), max(amounts))
cmap = plt.cm.YlOrRd
colors = [cmap(norm(a)) for a in amounts]

bars = ax1.barh(range(len(valid_types)), uplift_vals, color=colors, edgecolor='white', height=0.7)
ax1.set_yticks(range(len(valid_types)))
ax1.set_yticklabels(labels, fontsize=9)
ax1.set_xlabel('Additional Tickets/Period (Net Uplift)')
ax1.set_title('Ticket Uplift by Prize Type\n(Average change in tickets/period after winning)', fontsize=12, fontweight='bold')
ax1.axvline(x=0, color='grey', linestyle='--', alpha=0.5)

# Annotate
for i, (v, ptype) in enumerate(zip(uplift_vals, valid_types)):
    n = int(uplift_summary.loc[ptype, 'n_winners'])
    ax1.text(v + (0.1 if v >= 0 else -0.1), i,
             f'{v:+.2f} (n={n:,})',
             va='center', ha='left' if v >= 0 else 'right', fontsize=8)

ax1.invert_yaxis()

# ── Panel 2: Risk Level Transition Matrix (Heatmap) ──
ax2 = fig.add_subplot(gs[0, 1])

# Custom colormap: green (good=shift down) to red (bad=shift up)
# For transition matrix, diagonal = neutral, above = shifted down (good), below = shifted up (bad)
trans_data = transition.values

# Create annotation strings
annot = np.empty_like(trans_data, dtype=object)
for i in range(4):
    for j in range(4):
        annot[i, j] = f'{trans_data[i, j]:.1f}%'

# Color: use a diverging scheme based on position relative to diagonal
# Cells where column < row index = shifted down (green), column > row = shifted up (red)
color_matrix = np.zeros_like(trans_data)
for i in range(4):
    for j in range(4):
        if j < i:  # shifted down (improvement)
            color_matrix[i, j] = trans_data[i, j]  # positive = green
        elif j > i:  # shifted up (worse)
            color_matrix[i, j] = -trans_data[i, j]  # negative = red
        else:
            color_matrix[i, j] = 0  # diagonal = neutral

im = ax2.imshow(trans_data, cmap='YlOrRd_r', aspect='auto', vmin=0, vmax=100)
ax2.set_xticks(range(4))
ax2.set_xticklabels(RISK_LEVELS, fontsize=10)
ax2.set_yticks(range(4))
ax2.set_yticklabels(RISK_LEVELS, fontsize=10)
ax2.set_xlabel('After Winning (Risk Level)')
ax2.set_ylabel('Before Winning (Risk Level)')
ax2.set_title('Risk Level Migration After Winning\n(All Prize Types Combined)', fontsize=12, fontweight='bold')

# Add text annotations with dynamic color
for i in range(4):
    for j in range(4):
        val = trans_data[i, j]
        color = 'white' if val > 50 else 'black'
        ax2.text(j, i, f'{val:.1f}%', ha='center', va='center',
                 fontsize=11, fontweight='bold', color=color)

        # Add green/red border for shift cells
        if j < i and val > 5:  # shifted down (good)
            ax2.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                         fill=False, edgecolor='#27ae60', linewidth=2))
        elif j > i and val > 5:  # shifted up (bad)
            ax2.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                         fill=False, edgecolor='#e74c3c', linewidth=1.5, linestyle='--'))

plt.colorbar(im, ax=ax2, shrink=0.8, label='% of customers')

# ── Panel 3: Shift Down Rate by Prize Type ──
ax3 = fig.add_subplot(gs[1, 0])
valid_shift = shift_by_type.dropna()
sorted_types = valid_shift.sort_values('pct_shifted_down', ascending=True).index

shift_vals = [valid_shift.loc[t, 'pct_shifted_down'] * 100 for t in sorted_types]
shift_labels = [PRIZE_META.get(t, {}).get('label', t) for t in sorted_types]
shift_n = [int(valid_shift.loc[t, 'n']) for t in sorted_types]
shift_colors = [cmap(norm(PRIZE_META.get(t, {}).get('amount', 0))) for t in sorted_types]

bars3 = ax3.barh(range(len(sorted_types)), shift_vals, color=shift_colors, edgecolor='white', height=0.7)
ax3.set_yticks(range(len(sorted_types)))
ax3.set_yticklabels(shift_labels, fontsize=9)
ax3.set_xlabel('% Shifted Down At Least 1 Risk Level')
ax3.set_title('% Customers Risk Reduced After Winning\nby Prize Type', fontsize=12, fontweight='bold')

for i, (v, n) in enumerate(zip(shift_vals, shift_n)):
    ax3.text(v + 0.5, i, f'{v:.1f}% (n={n:,})', va='center', fontsize=8)

# ── Panel 4: Revenue Forecast ──
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

# Revenue infographic
rev_text = "REVENUE FORECAST\nfrom Winning Effect\n"
rev_text += "=" * 40 + "\n\n"

for ptype in revenue.index:
    r = revenue.loc[ptype]
    label = PRIZE_META.get(ptype, {}).get('label', ptype)
    annual_t = r['annual_additional_tickets']
    annual_r = r['annual_revenue_thb']
    if abs(annual_t) > 100:  # skip negligible
        rev_text += f"{label}\n"
        rev_text += f"  Winners/period: {r['winners_per_period']:,.0f}\n"
        rev_text += f"  Uplift: {r['uplift_mean']:+.2f} tickets/period\n"
        rev_text += f"  Annual: {annual_t:+,.0f} tickets = {annual_r:+,.0f} THB\n\n"

rev_text += "=" * 40 + "\n"
rev_text += f"TOTAL ANNUAL IMPACT\n"
rev_text += f"  Additional tickets: {total_annual_tickets:+,.0f}\n"
rev_text += f"  Additional revenue: {total_annual_revenue:+,.0f} THB\n"
rev_text += f"  ({total_annual_revenue/1e6:+,.1f}M THB)"

ax4.text(0.05, 0.95, rev_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

# ── Panel 5: Shift Probability Table ──
ax5 = fig.add_subplot(gs[2, 0])
ax5.axis('off')

if len(shift_targets_df) > 0:
    # Build a pivot: rows = current_risk, columns = prize_type, values = pct_shifted_down
    pivot = shift_targets_df.pivot_table(
        index='current_risk', columns='prize_type',
        values='pct_shifted_down', aggfunc='first'
    )
    # Reindex
    pivot = pivot.reindex(index=['Medium', 'High', 'Critical'])
    pivot_cols = [c for c in PRIZE_ORDER if c in pivot.columns]
    pivot = pivot[pivot_cols]

    # Create table
    cell_text = []
    for rl in pivot.index:
        row = []
        for pt in pivot.columns:
            v = pivot.loc[rl, pt]
            if pd.isna(v):
                row.append('-')
            else:
                row.append(f'{v:.0%}')
        cell_text.append(row)

    col_labels = [PRIZE_META.get(t, {}).get('label', t).split('(')[0].strip()[:12] for t in pivot.columns]

    table = ax5.table(cellText=cell_text,
                      rowLabels=list(pivot.index),
                      colLabels=col_labels,
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)

    # Color cells
    for i, rl in enumerate(pivot.index):
        for j, pt in enumerate(pivot.columns):
            v = pivot.loc[rl, pt]
            if not pd.isna(v):
                if v >= 0.3:
                    table[i+1, j].set_facecolor('#a8e6cf')
                elif v >= 0.2:
                    table[i+1, j].set_facecolor('#dcedc1')
                elif v >= 0.1:
                    table[i+1, j].set_facecolor('#ffd3b6')
                else:
                    table[i+1, j].set_facecolor('#ffaaa5')

ax5.set_title('Shift Down Probability\nCurrent Risk x Prize Type', fontsize=12, fontweight='bold', pad=20)

# ── Panel 6: Key Recommendations ──
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')

# Find the best shift types
best_shift = shift_by_type.sort_values('pct_shifted_down', ascending=False)
if len(best_shift) > 0:
    best_type = best_shift.index[0]
    best_pct = best_shift.iloc[0]['pct_shifted_down']
else:
    best_type = 'FIFTH'
    best_pct = 0.25

# Get uplift info
if len(uplift_summary) > 0:
    biggest_uplift_type = uplift_summary['uplift_mean'].idxmax()
    biggest_uplift_val = uplift_summary.loc[biggest_uplift_type, 'uplift_mean']
else:
    biggest_uplift_type = 'LAST2DIGITS'
    biggest_uplift_val = 0

reco_text = "KEY RECOMMENDATIONS\n"
reco_text += "=" * 50 + "\n\n"

reco_text += "ADS TEAM:\n"
reco_text += f"  - Focus retargeting on High/Critical customers\n"
reco_text += f"    who recently won prizes\n"
reco_text += f"  - Best shift rate: {best_type}\n"
reco_text += f"    ({best_pct:.0%} shifted down at least 1 level)\n"
reco_text += f"  - Winning creates a re-engagement window\n\n"

reco_text += "SALES TEAM:\n"
reco_text += f"  - Prize winners buy {biggest_uplift_val:+.1f} tickets/period\n"
reco_text += f"    more after winning ({biggest_uplift_type})\n"
reco_text += f"  - Upsell within 3 periods of win\n"
reco_text += f"  - Annual impact: {total_annual_revenue/1e6:+,.1f}M THB\n\n"

reco_text += "MARCOM TEAM:\n"
reco_text += f"  - Send congratulations + special offer\n"
reco_text += f"    within 1 period of winning\n"
reco_text += f"  - Prize winners have lower churn risk\n"
reco_text += f"    for ~3 periods after winning\n"
reco_text += f"  - Target {prize_all['userNo'].nunique():,} unique winners\n"
reco_text += f"    across 30 periods\n\n"

reco_text += "PRIORITY MATRIX:\n"
reco_text += f"  1. Critical + won FIFTH/FOURTH → immediate engage\n"
reco_text += f"  2. High + won any prize → send offer in 1 period\n"
reco_text += f"  3. Medium + won → maintain relationship\n"

ax6.text(0.05, 0.95, reco_text, transform=ax6.transAxes,
         fontsize=9.5, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#e8f4fd', alpha=0.8))

# Save
chart_path = CHART_DIR / 'prize_shift_analysis.png'
fig.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nChart saved: {chart_path}")


# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nData coverage:")
print(f"  Prize files: 30 periods ({prize_all['rounddate'].min()} → {prize_all['rounddate'].max()})")
print(f"  Unique winners analyzed: {len(results_df['userNo'].unique()):,}")
print(f"  Risk classifications: {len(risk_df):,}")

print(f"\nKey findings:")
if len(uplift_summary) > 0:
    avg_uplift = results_df['uplift'].mean()
    print(f"  Overall average uplift: {avg_uplift:+.2f} tickets/period after winning")
    max_type = uplift_summary['uplift_mean'].idxmax()
    max_val = uplift_summary.loc[max_type, 'uplift_mean']
    print(f"  Highest uplift: {max_type} ({max_val:+.2f} tickets/period)")
    min_type = uplift_summary['uplift_mean'].idxmin()
    min_val = uplift_summary.loc[min_type, 'uplift_mean']
    print(f"  Lowest uplift: {min_type} ({min_val:+.2f} tickets/period)")

if len(risk_df) > 0:
    overall_shift_down = risk_df['shifted_down'].mean()
    overall_shift_up = risk_df['shifted_up'].mean()
    print(f"\n  Overall risk shift down: {overall_shift_down:.1%}")
    print(f"  Overall risk shift up: {overall_shift_up:.1%}")
    print(f"  Net shift direction: {'POSITIVE (more shift down)' if overall_shift_down > overall_shift_up else 'NEGATIVE (more shift up)'}")

print(f"\n  Annual revenue impact: {total_annual_revenue:+,.0f} THB ({total_annual_revenue/1e6:+,.1f}M)")
print(f"  Ticket price assumption: {TICKET_PRICE} THB/ticket")

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
