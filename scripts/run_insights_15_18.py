#!/usr/bin/env python3
"""Insights 15-18: Purchase Behavior Insights with Customer Examples.

Insight 15: Rush buyer vs Early buyer → retention ต่างกัน
Insight 16: เปลี่ยนจาก rush → early buyer → ยอดเพิ่มขึ้นมั้ย
Insight 17: ซื้อกลางวัน vs กลางคืน → ซื้อต่อเนื่องต่างกัน
Insight 18: สั่ง 1 ครั้ง/งวด vs สั่งหลายครั้ง/งวด → ใคร loyal กว่า

Run: python3 scripts/run_insights_15_18.py
"""

import os, sys, glob, warnings, re
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Thai font support
for fp in ['TH Sarabun New', 'Tahoma', 'Angsana New', 'Garuda', 'Loma',
           'Norasi', 'DejaVu Sans']:
    try:
        matplotlib.rcParams['font.family'] = fp
        break
    except:
        pass
matplotlib.rcParams['axes.unicode_minus'] = False

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
CHURN_DIR = 'data/churn'
TX_DIR    = 'data/transaction'
CHART_DIR = 'charts'
os.makedirs(CHART_DIR, exist_ok=True)

COLOR_BOUGHT     = '#F97316'   # orange — bought tickets
COLOR_ZERO       = '#E5E7EB'   # light gray — didn't buy
COLOR_NAN        = '#1F2937'   # dark — not registered
COLOR_EVENT_MARK = '#DC2626'   # red star
COLOR_EVENT_LINE = '#F59E0B'   # gold dashed line

# Slightly different shades for rush/early if possible
COLOR_RUSH  = '#EA580C'  # darker orange for rush periods
COLOR_EARLY = '#34D399'  # green for early periods

print("=" * 70, flush=True)
print("Insights 15-18: Purchase Behavior — Rush/Early, Day/Night, Order Freq", flush=True)
print("=" * 70, flush=True)


# ═══════════════════════════════════════════════════════════════
# Load Data
# ═══════════════════════════════════════════════════════════════

def load_churn_history():
    """Load the latest churn file for purchase history."""
    target = os.path.join(CHURN_DIR, 'Churn_2026_0301_0316.csv')
    if not os.path.exists(target):
        files = sorted(glob.glob(os.path.join(CHURN_DIR, 'Churn_*.csv')))
        files = [f for f in files if 'old_validation' not in f and 'Check' not in f]
        target = files[-1]

    print(f"  Loading churn: {os.path.basename(target)}", flush=True)
    churn = pd.read_csv(target, low_memory=False)
    item_cols = sorted([c for c in churn.columns if c.startswith('item')])
    print(f"  Customers: {len(churn):,}, Periods: {len(item_cols)}", flush=True)
    return churn, item_cols


def load_tx_data(n_files=None):
    """Load TX files. If n_files specified, load last n_files."""
    files = sorted(glob.glob(os.path.join(TX_DIR, 'Transaction_Order_*.csv')))
    if n_files:
        files = files[-n_files:]
    print(f"  Loading {len(files)} TX files...", flush=True)

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        # Extract period from filename
        base = os.path.basename(f)  # Transaction_Order_YYYY_MMDD.csv
        parts = base.replace('.csv', '').split('_')
        period_str = f"{parts[2]}_{parts[3]}"  # YYYY_MMDD
        df['period'] = period_str
        frames.append(df)
        print(f"    {base}: {len(df):,} rows", flush=True)

    tx = pd.concat(frames, ignore_index=True)
    # Parse datetimes — strip timezone to make all tz-naive for comparison
    tx['approvedAt_bkk'] = pd.to_datetime(tx['approvedAt_bkk'], errors='coerce', utc=True).dt.tz_localize(None)
    tx['createdAt_bkk'] = pd.to_datetime(tx['createdAt_bkk'], errors='coerce', utc=True).dt.tz_localize(None)
    tx['roundDate'] = pd.to_datetime(tx['roundDate'], errors='coerce')

    # Filter SUCCESS only
    tx_success = tx[tx['status'] == 'SUCCESS'].copy()
    print(f"  Total TX: {len(tx):,}, SUCCESS: {len(tx_success):,}", flush=True)
    return tx_success


def compute_churn_from_history(churn, item_cols):
    """Compute churn: bought in second-to-last but NOT in last period."""
    last_col = item_cols[-1]
    prev_col = item_cols[-2]

    # Bought in previous period
    bought_prev = churn[prev_col].fillna(0) > 0
    # Bought in last period
    bought_last = churn[last_col].fillna(0) > 0

    churn_df = churn[bought_prev].copy()
    churn_df['churned'] = (~bought_last[bought_prev]).astype(int)
    return churn_df


# ═══════════════════════════════════════════════════════════════
# Bar Chart Drawing Utility
# ═══════════════════════════════════════════════════════════════

def draw_customer_bar(ax, user_row, item_cols, title, event_indices=None,
                      event_labels=None, color_map=None):
    """Draw a single customer's purchase history bar chart.

    color_map: dict {period_index: color} for custom coloring per bar.
    """
    history = []
    for col in item_cols:
        val = user_row[col] if col in user_row.index else np.nan
        try:
            val = float(val)
        except (ValueError, TypeError):
            val = np.nan
        history.append(val)

    n_periods = len(history)
    bar_colors = []
    bar_heights = []

    for i, val in enumerate(history):
        if pd.isna(val) or (isinstance(val, str) and val.lower() == 'null'):
            bar_heights.append(0.3)
            bar_colors.append(COLOR_NAN)
        elif val == 0:
            bar_heights.append(0.3)
            bar_colors.append(COLOR_ZERO)
        else:
            bar_heights.append(val)
            if color_map and i in color_map:
                bar_colors.append(color_map[i])
            else:
                bar_colors.append(COLOR_BOUGHT)

    bars = ax.bar(range(n_periods), bar_heights, color=bar_colors, width=0.7,
                  edgecolor='white', linewidth=0.3)

    # Mark events
    if event_indices:
        max_height = max(bar_heights) if bar_heights else 1
        for j, eidx in enumerate(event_indices):
            if 0 <= eidx < n_periods:
                marker_y = bar_heights[eidx] + max_height * 0.08
                ax.plot(eidx, marker_y, marker='*', color=COLOR_EVENT_MARK,
                        markersize=14, zorder=5)
                lbl = event_labels[j] if event_labels and j < len(event_labels) else '★'
                ax.text(eidx, marker_y + max_height * 0.06, lbl,
                        ha='center', va='bottom', fontsize=7, color=COLOR_EVENT_MARK,
                        fontweight='bold')
                ax.axvline(x=eidx, color=COLOR_EVENT_LINE, linestyle='--',
                           alpha=0.6, linewidth=1, zorder=1)

    # X-axis labels
    period_labels = []
    for col in item_cols:
        parts = col.replace('item', '').split('_')
        if len(parts) == 3:
            period_labels.append(f"{parts[1]}/{parts[2]}")
        else:
            period_labels.append(col[-5:])

    step = max(1, n_periods // 20)
    tick_positions = list(range(0, n_periods, step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([period_labels[i] for i in tick_positions],
                       rotation=45, ha='right', fontsize=6)

    ax.set_xlim(-0.5, n_periods - 0.5)
    ax.set_ylabel('ใบ', fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.tick_params(axis='y', labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def add_stats_text(ax, text):
    """Add stats text below the chart."""
    ax.text(0.5, -0.18, text, transform=ax.transAxes,
            ha='center', va='top', fontsize=7, color='#6B7280')


def item_col_to_period_key(col):
    """Convert item2025_01_02 -> 2025_0102."""
    parts = col.replace('item', '').split('_')
    if len(parts) == 3:
        return f"{parts[0]}_{parts[1]}{parts[2]}"
    return col


def period_key_to_item_col(period):
    """Convert 2025_0102 -> item2025_01_02."""
    # period is YYYY_MMDD
    y = period[:4]
    md = period[5:]
    m = md[:2]
    d = md[2:]
    return f"item{y}_{m}_{d}"


# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════

print("\n[1/5] Loading data...", flush=True)
churn, item_cols = load_churn_history()

# Load at least 10 TX files for robust analysis
tx = load_tx_data(n_files=10)

# Map TX periods to item columns
tx_periods = sorted(tx['period'].unique())
print(f"  TX periods: {len(tx_periods)} — {tx_periods[0]} to {tx_periods[-1]}", flush=True)

# Build period -> item_col mapping
period_to_item = {}
for p in tx_periods:
    ic = period_key_to_item_col(p)
    if ic in item_cols:
        period_to_item[p] = ic

print(f"  TX periods mapped to churn columns: {len(period_to_item)}", flush=True)

# Item col -> index mapping
item_col_idx = {c: i for i, c in enumerate(item_cols)}

# Compute churn labels for the last period
churn_df = compute_churn_from_history(churn, item_cols)
churn_labels = churn_df.set_index('userNo')['churned']
print(f"  Churn dataset: {len(churn_df):,} customers, churn rate: {churn_df['churned'].mean():.1%}", flush=True)


# ═══════════════════════════════════════════════════════════════
# INSIGHT 15: Rush buyer vs Early buyer
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70, flush=True)
print("Insight 15: Rush buyer vs Early buyer → retention ต่างกัน", flush=True)
print("=" * 70, flush=True)

# For each customer-period, compute days before draw of each order
# Rush = majority of orders in last 2 days, Early = majority 5+ days before

def classify_rush_early(tx_df):
    """Classify each customer-period as rush or early buyer."""
    tx_df = tx_df.copy()

    # Days before draw
    tx_df['days_before_draw'] = (tx_df['roundDate'] - tx_df['createdAt_bkk']).dt.total_seconds() / 86400
    # Some may be negative (same day) -> clip to 0
    tx_df['days_before_draw'] = tx_df['days_before_draw'].clip(lower=0)

    # Classify each order
    tx_df['is_rush'] = tx_df['days_before_draw'] <= 2
    tx_df['is_early'] = tx_df['days_before_draw'] >= 5

    # Per customer-period: count rush vs early orders
    user_period = tx_df.groupby(['userNo', 'period']).agg(
        n_orders=('totalItem', 'count'),
        total_items=('totalItem', 'sum'),
        n_rush=('is_rush', 'sum'),
        n_early=('is_early', 'sum'),
        avg_days_before=('days_before_draw', 'mean'),
    ).reset_index()

    # Classify: rush if >50% rush orders, early if >50% early orders
    user_period['pct_rush'] = user_period['n_rush'] / user_period['n_orders']
    user_period['pct_early'] = user_period['n_early'] / user_period['n_orders']
    user_period['buyer_type'] = 'mixed'
    user_period.loc[user_period['pct_rush'] > 0.5, 'buyer_type'] = 'rush'
    user_period.loc[user_period['pct_early'] > 0.5, 'buyer_type'] = 'early'

    return user_period


print("\n  Classifying rush vs early buyers...", flush=True)
user_period_class = classify_rush_early(tx)

# Overall classification: majority across all periods
user_majority = user_period_class.groupby('userNo').agg(
    n_periods=('period', 'count'),
    n_rush_periods=('buyer_type', lambda x: (x == 'rush').sum()),
    n_early_periods=('buyer_type', lambda x: (x == 'early').sum()),
    avg_items=('total_items', 'mean'),
    total_orders=('n_orders', 'sum'),
).reset_index()

user_majority['dominant_type'] = 'mixed'
user_majority.loc[user_majority['n_rush_periods'] > user_majority['n_periods'] / 2, 'dominant_type'] = 'rush'
user_majority.loc[user_majority['n_early_periods'] > user_majority['n_periods'] / 2, 'dominant_type'] = 'early'

# Merge with churn labels
user_majority = user_majority.merge(churn_labels, on='userNo', how='inner')

# Stats
print("\n  === สถิติ Rush vs Early Buyer ===", flush=True)
for btype in ['early', 'rush', 'mixed']:
    subset = user_majority[user_majority['dominant_type'] == btype]
    if len(subset) == 0:
        continue
    churn_rate = subset['churned'].mean()
    avg_items = subset['avg_items'].mean()
    avg_periods = subset['n_periods'].mean()
    print(f"  {btype:>6}: {len(subset):>8,} คน | Churn {churn_rate:.1%} | "
          f"เฉลี่ย {avg_items:.1f} ใบ/งวด | เห็น {avg_periods:.1f} งวด", flush=True)

early_churn = user_majority[user_majority['dominant_type'] == 'early']['churned'].mean()
rush_churn = user_majority[user_majority['dominant_type'] == 'rush']['churned'].mean()
diff = rush_churn - early_churn
print(f"\n  → Rush buyer มี churn สูงกว่า Early buyer {diff:+.1%}", flush=True)

# Find 3 examples
churn_indexed = churn.set_index('userNo')

# 1) Loyal early buyer (early, not churned, high tenure)
early_loyal = user_majority[(user_majority['dominant_type'] == 'early') &
                            (user_majority['churned'] == 0) &
                            (user_majority['n_periods'] >= 3)]
early_loyal = early_loyal.sort_values('avg_items', ascending=False)
ex15_1 = early_loyal.iloc[min(5, len(early_loyal)-1)]['userNo'] if len(early_loyal) > 0 else None

# 2) Churned rush buyer
rush_churned = user_majority[(user_majority['dominant_type'] == 'rush') &
                             (user_majority['churned'] == 1) &
                             (user_majority['n_periods'] >= 3)]
rush_churned = rush_churned.sort_values('avg_items', ascending=False)
ex15_2 = rush_churned.iloc[min(5, len(rush_churned)-1)]['userNo'] if len(rush_churned) > 0 else None

# 3) Mixed buyer
mixed_buyers = user_majority[(user_majority['dominant_type'] == 'mixed') &
                             (user_majority['n_periods'] >= 3)]
mixed_buyers = mixed_buyers.sort_values('avg_items', ascending=False)
ex15_3 = mixed_buyers.iloc[min(5, len(mixed_buyers)-1)]['userNo'] if len(mixed_buyers) > 0 else None

examples_15 = [ex15_1, ex15_2, ex15_3]
labels_15 = [
    "Early buyer ภักดี — ซื้อล่วงหน้า ยังซื้อต่อเนื่อง",
    "Rush buyer ที่ churn — ซื้อ 2 วันสุดท้าย แล้วหายไป",
    "Mixed buyer — สลับระหว่าง early/rush",
]

# Build color map for each example (rush vs early per period)
def build_rush_color_map(user, user_period_class, item_cols):
    """Color rush periods different from early periods."""
    up = user_period_class[user_period_class['userNo'] == user]
    cmap = {}
    for _, row in up.iterrows():
        ic = period_key_to_item_col(row['period'])
        if ic in item_cols:
            idx = item_cols.index(ic)
            if row['buyer_type'] == 'rush':
                cmap[idx] = COLOR_RUSH
            elif row['buyer_type'] == 'early':
                cmap[idx] = COLOR_EARLY
    return cmap

print("\n  ตัวอย่างลูกค้า:", flush=True)
for i, (user, label) in enumerate(zip(examples_15, labels_15)):
    if user and user in churn_indexed.index:
        row = churn_indexed.loc[user]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        age = row.get('age', '?')
        gender = row.get('gender', '?')
        province = row.get('province', '?')
        churned = churn_labels.get(user, '?')
        up = user_majority[user_majority['userNo'] == user].iloc[0]
        print(f"  {i+1}. {user} ({gender}, {age}, {province}) — {label}", flush=True)
        print(f"     Type: {up['dominant_type']} | Rush งวด: {up['n_rush_periods']}/{up['n_periods']} | "
              f"เฉลี่ย {up['avg_items']:.0f} ใบ/งวด | Churned: {churned}", flush=True)

# Chart 15
print("\n  Creating chart: insight_15_rush_vs_early.png", flush=True)
fig, axes = plt.subplots(3, 1, figsize=(20, 18), dpi=150)
fig.suptitle("Insight 15: Rush buyer (2 วันสุดท้าย) vs Early buyer → retention ต่างกัน\n"
             f"Early churn: {early_churn:.1%} | Rush churn: {rush_churn:.1%} | "
             f"ต่าง: {diff:+.1%}",
             fontsize=14, fontweight='bold', y=0.98)

for i, (user, label) in enumerate(zip(examples_15, labels_15)):
    if user and user in churn_indexed.index:
        row = churn_indexed.loc[user]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        cmap = build_rush_color_map(user, user_period_class, item_cols)
        draw_customer_bar(axes[i], row, item_cols, f"{user}: {label}", color_map=cmap)
        up = user_majority[user_majority['userNo'] == user].iloc[0]
        stats = (f"Type: {up['dominant_type']} | Rush: {up['n_rush_periods']}/{up['n_periods']} งวด | "
                 f"เฉลี่ย {up['avg_items']:.0f} ใบ/งวด | Churned: {int(up['churned'])}")
        add_stats_text(axes[i], stats)
    else:
        axes[i].text(0.5, 0.5, 'ไม่พบข้อมูล', transform=axes[i].transAxes,
                     ha='center', va='center', fontsize=14)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLOR_EARLY, label='Early buyer (5+ วันก่อน)'),
    Patch(facecolor=COLOR_RUSH, label='Rush buyer (2 วันสุดท้าย)'),
    Patch(facecolor=COLOR_BOUGHT, label='ซื้อ (ไม่มี TX data)'),
    Patch(facecolor=COLOR_ZERO, label='ไม่ซื้อ'),
    Patch(facecolor=COLOR_NAN, label='ยังไม่สมัคร'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(os.path.join(CHART_DIR, 'insight_15_rush_vs_early.png'),
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Saved: charts/insight_15_rush_vs_early.png", flush=True)


# ═══════════════════════════════════════════════════════════════
# INSIGHT 16: Rush → Early transition
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70, flush=True)
print("Insight 16: เปลี่ยนจาก rush → early buyer → ยอดเพิ่มขึ้นมั้ย", flush=True)
print("=" * 70, flush=True)

# Per customer: track buyer_type sequence across periods
def find_timing_transitions(user_period_class):
    """Find customers who switched from rush->early or early->rush."""
    # Sort by period per user
    upc = user_period_class.sort_values(['userNo', 'period'])

    # Only keep users with 3+ periods
    user_counts = upc.groupby('userNo')['period'].count()
    valid_users = user_counts[user_counts >= 3].index
    upc = upc[upc['userNo'].isin(valid_users)]

    transitions = []
    for user, group in upc.groupby('userNo'):
        types = group['buyer_type'].values
        periods = group['period'].values
        items = group['total_items'].values

        for j in range(1, len(types)):
            if types[j-1] == 'rush' and types[j] == 'early':
                # Rush -> Early transition
                avg_before = items[:j].mean()
                avg_after = items[j:].mean()
                transitions.append({
                    'userNo': user,
                    'transition': 'rush→early',
                    'transition_period': periods[j],
                    'transition_idx': j,
                    'avg_before': avg_before,
                    'avg_after': avg_after,
                    'change_pct': (avg_after - avg_before) / max(avg_before, 1) * 100,
                    'n_periods': len(types),
                })
            elif types[j-1] == 'early' and types[j] == 'rush':
                avg_before = items[:j].mean()
                avg_after = items[j:].mean()
                transitions.append({
                    'userNo': user,
                    'transition': 'early→rush',
                    'transition_period': periods[j],
                    'transition_idx': j,
                    'avg_before': avg_before,
                    'avg_after': avg_after,
                    'change_pct': (avg_after - avg_before) / max(avg_before, 1) * 100,
                    'n_periods': len(types),
                })

    return pd.DataFrame(transitions)


print("\n  Finding timing transitions...", flush=True)
transitions = find_timing_transitions(user_period_class)

if len(transitions) > 0:
    transitions = transitions.merge(churn_labels, on='userNo', how='left')

    for ttype in ['rush→early', 'early→rush']:
        subset = transitions[transitions['transition'] == ttype]
        if len(subset) == 0:
            continue
        churn_rate = subset['churned'].mean() if 'churned' in subset.columns else 0
        avg_change = subset['change_pct'].mean()
        increased = (subset['change_pct'] > 0).mean()
        print(f"\n  {ttype}: {len(subset):,} transitions", flush=True)
        print(f"    Volume change: เฉลี่ย {avg_change:+.1f}% | เพิ่มขึ้น {increased:.1%}", flush=True)
        print(f"    Churn rate: {churn_rate:.1%}", flush=True)

    # Also find switchers (back and forth)
    user_trans_counts = transitions.groupby('userNo')['transition'].count()
    switchers = user_trans_counts[user_trans_counts >= 2].index
    print(f"\n  Switchers (สลับไปมา 2+ ครั้ง): {len(switchers):,} คน", flush=True)

    # Find examples
    # 1) Rush -> Early who INCREASED volume
    re_increased = transitions[(transitions['transition'] == 'rush→early') &
                               (transitions['change_pct'] > 10)]
    re_increased = re_increased.sort_values('change_pct', ascending=False)
    ex16_1 = re_increased.iloc[min(3, len(re_increased)-1)]['userNo'] if len(re_increased) > 0 else None

    # 2) Early -> Rush who DECREASED
    er_decreased = transitions[(transitions['transition'] == 'early→rush') &
                               (transitions['change_pct'] < -10)]
    er_decreased = er_decreased.sort_values('change_pct', ascending=True)
    ex16_2 = er_decreased.iloc[min(3, len(er_decreased)-1)]['userNo'] if len(er_decreased) > 0 else None

    # 3) Switcher (back and forth)
    ex16_3 = None
    if len(switchers) > 0:
        switcher_list = transitions[transitions['userNo'].isin(switchers)]
        switcher_users = switcher_list.groupby('userNo').size().sort_values(ascending=False)
        for su in switcher_users.index:
            if su in churn_indexed.index:
                ex16_3 = su
                break
else:
    ex16_1 = ex16_2 = ex16_3 = None
    print("  No transitions found.", flush=True)

examples_16 = [ex16_1, ex16_2, ex16_3]
labels_16 = [
    "Rush→Early: ยอดเพิ่มขึ้นหลังเปลี่ยน",
    "Early→Rush: ยอดลดลง — สัญญาณเตือน",
    "Switcher: สลับไปมา early/rush",
]

# Build transition event indices for chart
def get_transition_event(user, transitions, item_cols):
    """Get the item_col index where transition happened."""
    ut = transitions[transitions['userNo'] == user]
    events = []
    labels = []
    for _, row in ut.iterrows():
        ic = period_key_to_item_col(row['transition_period'])
        if ic in item_cols:
            events.append(item_cols.index(ic))
            labels.append(f"★ {row['transition']}")
    return events, labels

print("\n  ตัวอย่างลูกค้า:", flush=True)
for i, (user, label) in enumerate(zip(examples_16, labels_16)):
    if user and user in churn_indexed.index:
        row = churn_indexed.loc[user]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        age = row.get('age', '?')
        gender = row.get('gender', '?')
        province = row.get('province', '?')
        ut = transitions[transitions['userNo'] == user].iloc[0]
        print(f"  {i+1}. {user} ({gender}, {age}, {province}) — {label}", flush=True)
        print(f"     Transition: {ut['transition']} at {ut['transition_period']} | "
              f"Volume change: {ut['change_pct']:+.1f}%", flush=True)

# Chart 16
print("\n  Creating chart: insight_16_timing_shift.png", flush=True)
fig, axes = plt.subplots(3, 1, figsize=(20, 18), dpi=150)

re_stats = transitions[transitions['transition'] == 'rush→early'] if len(transitions) > 0 else pd.DataFrame()
er_stats = transitions[transitions['transition'] == 'early→rush'] if len(transitions) > 0 else pd.DataFrame()
re_avg = re_stats['change_pct'].mean() if len(re_stats) > 0 else 0
er_avg = er_stats['change_pct'].mean() if len(er_stats) > 0 else 0

fig.suptitle("Insight 16: เปลี่ยนจาก Rush → Early buyer → ยอดเพิ่มขึ้นมั้ย\n"
             f"Rush→Early: volume {re_avg:+.1f}% | Early→Rush: volume {er_avg:+.1f}%",
             fontsize=14, fontweight='bold', y=0.98)

for i, (user, label) in enumerate(zip(examples_16, labels_16)):
    if user and user in churn_indexed.index:
        row = churn_indexed.loc[user]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        cmap = build_rush_color_map(user, user_period_class, item_cols)
        events, elabels = get_transition_event(user, transitions, item_cols)
        draw_customer_bar(axes[i], row, item_cols, f"{user}: {label}",
                          event_indices=events, event_labels=elabels, color_map=cmap)
        ut = transitions[transitions['userNo'] == user].iloc[0]
        stats = f"Transition: {ut['transition']} | Volume change: {ut['change_pct']:+.1f}%"
        add_stats_text(axes[i], stats)
    else:
        axes[i].text(0.5, 0.5, 'ไม่พบข้อมูล', transform=axes[i].transAxes,
                     ha='center', va='center', fontsize=14)

legend_elements = [
    Patch(facecolor=COLOR_EARLY, label='Early buyer (5+ วันก่อน)'),
    Patch(facecolor=COLOR_RUSH, label='Rush buyer (2 วันสุดท้าย)'),
    Patch(facecolor=COLOR_BOUGHT, label='ซื้อ (ไม่มี TX data)'),
    Patch(facecolor=COLOR_ZERO, label='ไม่ซื้อ'),
    Patch(facecolor=COLOR_NAN, label='ยังไม่สมัคร'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(os.path.join(CHART_DIR, 'insight_16_timing_shift.png'),
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Saved: charts/insight_16_timing_shift.png", flush=True)


# ═══════════════════════════════════════════════════════════════
# INSIGHT 17: Day buyer vs Night buyer
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70, flush=True)
print("Insight 17: ซื้อกลางวัน vs กลางคืน → ซื้อต่อเนื่องต่างกัน", flush=True)
print("=" * 70, flush=True)

def classify_day_night(tx_df):
    """Classify each customer-period as day or night buyer."""
    tx_df = tx_df.copy()

    # Use createdAt_bkk for purchase time (when they placed order)
    # Handle timezone: if naive, assume already BKK
    tx_df['hour'] = tx_df['createdAt_bkk'].dt.hour

    # Day: 06:00-17:59, Night: 18:00-05:59
    tx_df['is_day'] = (tx_df['hour'] >= 6) & (tx_df['hour'] < 18)
    tx_df['is_night'] = ~tx_df['is_day']

    # Per customer-period
    user_period = tx_df.groupby(['userNo', 'period']).agg(
        n_orders=('totalItem', 'count'),
        total_items=('totalItem', 'sum'),
        n_day=('is_day', 'sum'),
        n_night=('is_night', 'sum'),
        avg_hour=('hour', 'mean'),
    ).reset_index()

    user_period['pct_day'] = user_period['n_day'] / user_period['n_orders']
    user_period['time_type'] = 'mixed'
    user_period.loc[user_period['pct_day'] > 0.7, 'time_type'] = 'day'
    user_period.loc[user_period['pct_day'] < 0.3, 'time_type'] = 'night'

    return user_period


print("\n  Classifying day vs night buyers...", flush=True)
user_period_time = classify_day_night(tx)

# Overall classification per user: consistent day/night?
user_time_summary = user_period_time.groupby('userNo').agg(
    n_periods=('period', 'count'),
    n_day_periods=('time_type', lambda x: (x == 'day').sum()),
    n_night_periods=('time_type', lambda x: (x == 'night').sum()),
    avg_items=('total_items', 'mean'),
    avg_hour=('avg_hour', 'mean'),
).reset_index()

# Consistent = same type in 70%+ of periods
user_time_summary['dominant_time'] = 'mixed'
user_time_summary.loc[user_time_summary['n_day_periods'] > user_time_summary['n_periods'] * 0.7,
                      'dominant_time'] = 'consistent_day'
user_time_summary.loc[user_time_summary['n_night_periods'] > user_time_summary['n_periods'] * 0.7,
                      'dominant_time'] = 'consistent_night'

user_time_summary = user_time_summary.merge(churn_labels, on='userNo', how='inner')

print("\n  === สถิติ Day vs Night Buyer ===", flush=True)
for ttype in ['consistent_day', 'consistent_night', 'mixed']:
    subset = user_time_summary[user_time_summary['dominant_time'] == ttype]
    if len(subset) == 0:
        continue
    churn_rate = subset['churned'].mean()
    avg_items = subset['avg_items'].mean()
    avg_periods = subset['n_periods'].mean()
    avg_hour = subset['avg_hour'].mean()
    print(f"  {ttype:>18}: {len(subset):>8,} คน | Churn {churn_rate:.1%} | "
          f"เฉลี่ย {avg_items:.1f} ใบ/งวด | ชม.เฉลี่ย {avg_hour:.1f} | "
          f"เห็น {avg_periods:.1f} งวด", flush=True)

day_churn = user_time_summary[user_time_summary['dominant_time'] == 'consistent_day']['churned'].mean()
night_churn = user_time_summary[user_time_summary['dominant_time'] == 'consistent_night']['churned'].mean()
dn_diff = night_churn - day_churn
print(f"\n  → Night buyer {'มี' if dn_diff > 0 else 'มี'}churn {'สูงกว่า' if dn_diff > 0 else 'ต่ำกว่า'} Day buyer {abs(dn_diff):.1%}", flush=True)

# Examples
# 1) Consistent day buyer (not churned)
day_loyal = user_time_summary[(user_time_summary['dominant_time'] == 'consistent_day') &
                              (user_time_summary['churned'] == 0) &
                              (user_time_summary['n_periods'] >= 3)]
day_loyal = day_loyal.sort_values('avg_items', ascending=False)
ex17_1 = day_loyal.iloc[min(5, len(day_loyal)-1)]['userNo'] if len(day_loyal) > 0 else None

# 2) Consistent night buyer
night_users = user_time_summary[(user_time_summary['dominant_time'] == 'consistent_night') &
                                (user_time_summary['n_periods'] >= 3)]
night_users = night_users.sort_values('avg_items', ascending=False)
ex17_2 = night_users.iloc[min(5, len(night_users)-1)]['userNo'] if len(night_users) > 0 else None

# 3) Mixed / switches
mixed_time = user_time_summary[(user_time_summary['dominant_time'] == 'mixed') &
                               (user_time_summary['n_periods'] >= 4)]
mixed_time = mixed_time.sort_values('avg_items', ascending=False)
ex17_3 = mixed_time.iloc[min(5, len(mixed_time)-1)]['userNo'] if len(mixed_time) > 0 else None

examples_17 = [ex17_1, ex17_2, ex17_3]
labels_17 = [
    "Consistent Day buyer — ซื้อกลางวัน ภักดี",
    "Consistent Night buyer — ซื้อกลางคืน",
    "Mixed — สลับ กลางวัน/กลางคืน",
]

COLOR_DAY   = '#FBBF24'  # yellow-ish for day
COLOR_NIGHT = '#6366F1'  # indigo for night

def build_daynight_color_map(user, user_period_time, item_cols):
    """Color day periods vs night periods."""
    up = user_period_time[user_period_time['userNo'] == user]
    cmap = {}
    for _, row in up.iterrows():
        ic = period_key_to_item_col(row['period'])
        if ic in item_cols:
            idx = item_cols.index(ic)
            if row['time_type'] == 'day':
                cmap[idx] = COLOR_DAY
            elif row['time_type'] == 'night':
                cmap[idx] = COLOR_NIGHT
            else:
                cmap[idx] = COLOR_BOUGHT  # mixed stays orange
    return cmap

print("\n  ตัวอย่างลูกค้า:", flush=True)
for i, (user, label) in enumerate(zip(examples_17, labels_17)):
    if user and user in churn_indexed.index:
        row = churn_indexed.loc[user]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        age = row.get('age', '?')
        gender = row.get('gender', '?')
        province = row.get('province', '?')
        ut = user_time_summary[user_time_summary['userNo'] == user].iloc[0]
        print(f"  {i+1}. {user} ({gender}, {age}, {province}) — {label}", flush=True)
        print(f"     Type: {ut['dominant_time']} | Day: {ut['n_day_periods']}/{ut['n_periods']} งวด | "
              f"ชม.เฉลี่ย {ut['avg_hour']:.1f} | เฉลี่ย {ut['avg_items']:.0f} ใบ/งวด | "
              f"Churned: {int(ut['churned'])}", flush=True)

# Chart 17
print("\n  Creating chart: insight_17_day_vs_night.png", flush=True)
fig, axes = plt.subplots(3, 1, figsize=(20, 18), dpi=150)
fig.suptitle("Insight 17: ซื้อกลางวัน vs กลางคืน → ซื้อต่อเนื่องต่างกัน\n"
             f"Day churn: {day_churn:.1%} | Night churn: {night_churn:.1%} | "
             f"ต่าง: {dn_diff:+.1%}",
             fontsize=14, fontweight='bold', y=0.98)

for i, (user, label) in enumerate(zip(examples_17, labels_17)):
    if user and user in churn_indexed.index:
        row = churn_indexed.loc[user]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        cmap = build_daynight_color_map(user, user_period_time, item_cols)
        draw_customer_bar(axes[i], row, item_cols, f"{user}: {label}", color_map=cmap)
        ut = user_time_summary[user_time_summary['userNo'] == user].iloc[0]
        stats = (f"Type: {ut['dominant_time']} | Day: {ut['n_day_periods']}/{ut['n_periods']} งวด | "
                 f"ชม.เฉลี่ย: {ut['avg_hour']:.1f} | เฉลี่ย {ut['avg_items']:.0f} ใบ/งวด | "
                 f"Churned: {int(ut['churned'])}")
        add_stats_text(axes[i], stats)
    else:
        axes[i].text(0.5, 0.5, 'ไม่พบข้อมูล', transform=axes[i].transAxes,
                     ha='center', va='center', fontsize=14)

legend_elements = [
    Patch(facecolor=COLOR_DAY, label='Day buyer (06:00-18:00)'),
    Patch(facecolor=COLOR_NIGHT, label='Night buyer (18:00-06:00)'),
    Patch(facecolor=COLOR_BOUGHT, label='ซื้อ (mixed/ไม่มี TX)'),
    Patch(facecolor=COLOR_ZERO, label='ไม่ซื้อ'),
    Patch(facecolor=COLOR_NAN, label='ยังไม่สมัคร'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(os.path.join(CHART_DIR, 'insight_17_day_vs_night.png'),
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Saved: charts/insight_17_day_vs_night.png", flush=True)


# ═══════════════════════════════════════════════════════════════
# INSIGHT 18: Single-order vs Multi-order per period
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70, flush=True)
print("Insight 18: สั่ง 1 ครั้ง/งวด vs สั่งหลายครั้ง/งวด → ใคร loyal กว่า", flush=True)
print("=" * 70, flush=True)

def classify_order_frequency(tx_df):
    """Classify each customer-period as single-order or multi-order."""
    user_period = tx_df.groupby(['userNo', 'period']).agg(
        n_orders=('totalItem', 'count'),
        total_items=('totalItem', 'sum'),
    ).reset_index()

    user_period['order_type'] = 'single'
    user_period.loc[user_period['n_orders'] >= 2, 'order_type'] = 'multi'

    return user_period


print("\n  Classifying order frequency...", flush=True)
user_period_orders = classify_order_frequency(tx)

# Overall classification per user
user_order_summary = user_period_orders.groupby('userNo').agg(
    n_periods=('period', 'count'),
    n_single=('order_type', lambda x: (x == 'single').sum()),
    n_multi=('order_type', lambda x: (x == 'multi').sum()),
    avg_items=('total_items', 'mean'),
    avg_orders=('n_orders', 'mean'),
).reset_index()

user_order_summary['dominant_order'] = 'mixed'
user_order_summary.loc[user_order_summary['n_single'] > user_order_summary['n_periods'] * 0.7,
                       'dominant_order'] = 'single'
user_order_summary.loc[user_order_summary['n_multi'] > user_order_summary['n_periods'] * 0.7,
                       'dominant_order'] = 'multi'

user_order_summary = user_order_summary.merge(churn_labels, on='userNo', how='inner')

print("\n  === สถิติ Single vs Multi-order ===", flush=True)
for otype in ['single', 'multi', 'mixed']:
    subset = user_order_summary[user_order_summary['dominant_order'] == otype]
    if len(subset) == 0:
        continue
    churn_rate = subset['churned'].mean()
    avg_items = subset['avg_items'].mean()
    avg_orders = subset['avg_orders'].mean()
    avg_periods = subset['n_periods'].mean()
    print(f"  {otype:>7}: {len(subset):>8,} คน | Churn {churn_rate:.1%} | "
          f"เฉลี่ย {avg_items:.1f} ใบ/งวด | {avg_orders:.1f} ออเดอร์/งวด | "
          f"เห็น {avg_periods:.1f} งวด", flush=True)

single_churn = user_order_summary[user_order_summary['dominant_order'] == 'single']['churned'].mean()
multi_churn = user_order_summary[user_order_summary['dominant_order'] == 'multi']['churned'].mean()
om_diff = single_churn - multi_churn
print(f"\n  → Single-order มี churn {'สูงกว่า' if om_diff > 0 else 'ต่ำกว่า'} Multi-order {abs(om_diff):.1%}", flush=True)
print(f"  → Multi-order = engagement สูงกว่า → loyal กว่า" if om_diff > 0 else
      f"  → ข้อสรุป: ความถี่สั่งซื้อไม่ใช่ปัจจัยหลัก", flush=True)

# Examples
# 1) Single-order loyal
single_loyal = user_order_summary[(user_order_summary['dominant_order'] == 'single') &
                                  (user_order_summary['churned'] == 0) &
                                  (user_order_summary['n_periods'] >= 3)]
single_loyal = single_loyal.sort_values('avg_items', ascending=False)
ex18_1 = single_loyal.iloc[min(5, len(single_loyal)-1)]['userNo'] if len(single_loyal) > 0 else None

# 2) Multi-order loyal
multi_loyal = user_order_summary[(user_order_summary['dominant_order'] == 'multi') &
                                 (user_order_summary['churned'] == 0) &
                                 (user_order_summary['n_periods'] >= 3)]
multi_loyal = multi_loyal.sort_values('avg_items', ascending=False)
ex18_2 = multi_loyal.iloc[min(5, len(multi_loyal)-1)]['userNo'] if len(multi_loyal) > 0 else None

# 3) Multi-order who churned
multi_churned = user_order_summary[(user_order_summary['dominant_order'] == 'multi') &
                                   (user_order_summary['churned'] == 1) &
                                   (user_order_summary['n_periods'] >= 3)]
multi_churned = multi_churned.sort_values('avg_items', ascending=False)
ex18_3 = multi_churned.iloc[min(5, len(multi_churned)-1)]['userNo'] if len(multi_churned) > 0 else None

examples_18 = [ex18_1, ex18_2, ex18_3]
labels_18 = [
    "Single-order ภักดี — สั่ง 1 ครั้ง/งวด แต่ซื้อต่อเนื่อง",
    "Multi-order ภักดี — สั่งหลายครั้ง/งวด engagement สูง",
    "Multi-order ที่ churn — สั่งบ่อยแต่ก็หายไป",
]

COLOR_SINGLE = '#F97316'  # orange for single
COLOR_MULTI  = '#8B5CF6'  # purple for multi

def build_order_color_map(user, user_period_orders, item_cols):
    """Color single vs multi order periods."""
    up = user_period_orders[user_period_orders['userNo'] == user]
    cmap = {}
    for _, row in up.iterrows():
        ic = period_key_to_item_col(row['period'])
        if ic in item_cols:
            idx = item_cols.index(ic)
            if row['order_type'] == 'single':
                cmap[idx] = COLOR_SINGLE
            elif row['order_type'] == 'multi':
                cmap[idx] = COLOR_MULTI
    return cmap

print("\n  ตัวอย่างลูกค้า:", flush=True)
for i, (user, label) in enumerate(zip(examples_18, labels_18)):
    if user and user in churn_indexed.index:
        row = churn_indexed.loc[user]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        age = row.get('age', '?')
        gender = row.get('gender', '?')
        province = row.get('province', '?')
        uo = user_order_summary[user_order_summary['userNo'] == user].iloc[0]
        print(f"  {i+1}. {user} ({gender}, {age}, {province}) — {label}", flush=True)
        print(f"     Type: {uo['dominant_order']} | Multi: {uo['n_multi']}/{uo['n_periods']} งวด | "
              f"เฉลี่ย {uo['avg_orders']:.1f} ออเดอร์/งวด | {uo['avg_items']:.0f} ใบ/งวด | "
              f"Churned: {int(uo['churned'])}", flush=True)

# Chart 18
print("\n  Creating chart: insight_18_order_frequency.png", flush=True)
fig, axes = plt.subplots(3, 1, figsize=(20, 18), dpi=150)
fig.suptitle("Insight 18: สั่ง 1 ครั้ง/งวด vs สั่งหลายครั้ง/งวด → ใคร loyal กว่า\n"
             f"Single churn: {single_churn:.1%} | Multi churn: {multi_churn:.1%} | "
             f"ต่าง: {om_diff:+.1%}",
             fontsize=14, fontweight='bold', y=0.98)

for i, (user, label) in enumerate(zip(examples_18, labels_18)):
    if user and user in churn_indexed.index:
        row = churn_indexed.loc[user]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        cmap = build_order_color_map(user, user_period_orders, item_cols)
        draw_customer_bar(axes[i], row, item_cols, f"{user}: {label}", color_map=cmap)
        uo = user_order_summary[user_order_summary['userNo'] == user].iloc[0]
        stats = (f"Type: {uo['dominant_order']} | Multi: {uo['n_multi']}/{uo['n_periods']} งวด | "
                 f"เฉลี่ย {uo['avg_orders']:.1f} ออเดอร์/งวด | {uo['avg_items']:.0f} ใบ/งวด | "
                 f"Churned: {int(uo['churned'])}")
        add_stats_text(axes[i], stats)
    else:
        axes[i].text(0.5, 0.5, 'ไม่พบข้อมูล', transform=axes[i].transAxes,
                     ha='center', va='center', fontsize=14)

legend_elements = [
    Patch(facecolor=COLOR_SINGLE, label='Single-order (1 ครั้ง/งวด)'),
    Patch(facecolor=COLOR_MULTI, label='Multi-order (2+ ครั้ง/งวด)'),
    Patch(facecolor=COLOR_BOUGHT, label='ซื้อ (ไม่มี TX data)'),
    Patch(facecolor=COLOR_ZERO, label='ไม่ซื้อ'),
    Patch(facecolor=COLOR_NAN, label='ยังไม่สมัคร'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(os.path.join(CHART_DIR, 'insight_18_order_frequency.png'),
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Saved: charts/insight_18_order_frequency.png", flush=True)


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70, flush=True)
print("สรุป Insights 15-18", flush=True)
print("=" * 70, flush=True)

print(f"""
Insight 15: Rush buyer vs Early buyer
  • Early buyer (ซื้อล่วงหน้า 5+ วัน) churn {early_churn:.1%}
  • Rush buyer (ซื้อ 2 วันสุดท้าย) churn {rush_churn:.1%}
  • Rush buyer มี churn สูงกว่า {diff:+.1%}
  → Action: ส่ง reminder ก่อนงวดเริ่มขาย ดึงลูกค้าให้ซื้อล่วงหน้า

Insight 16: Rush → Early transition
  • Rush→Early: volume เฉลี่ย {re_avg:+.1f}%
  • Early→Rush: volume เฉลี่ย {er_avg:+.1f}%
  → Action: ลูกค้าที่เปลี่ยนจาก early→rush เป็นสัญญาณเตือน

Insight 17: Day vs Night buyer
  • Day buyer (06:00-18:00) churn {day_churn:.1%}
  • Night buyer (18:00-06:00) churn {night_churn:.1%}
  → Action: ส่ง notification ตามเวลาที่ลูกค้าเคยซื้อ

Insight 18: Single vs Multi-order
  • Single-order churn {single_churn:.1%}
  • Multi-order churn {multi_churn:.1%}
  → Action: {'สั่งหลายครั้ง = engagement สูง → ให้สิทธิพิเศษ multi-order' if om_diff > 0 else 'ความถี่สั่งซื้อส่งผลน้อย'}

Charts saved:
  • charts/insight_15_rush_vs_early.png
  • charts/insight_16_timing_shift.png
  • charts/insight_17_day_vs_night.png
  • charts/insight_18_order_frequency.png
""", flush=True)

print("Done!", flush=True)
