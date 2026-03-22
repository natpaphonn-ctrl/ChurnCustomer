#!/usr/bin/env python3
"""Insights 6-9: Purchase Behavior Deep-Dive with Customer Examples.

Insight 6: ซื้อครั้งแรกกี่ใบ → อยู่กับเรากี่งวด
Insight 7: ซื้อครั้งแรกวันไหนของรอบขาย → retention ต่างกัน
Insight 8: ลูกค้าใหม่ 3 งวดแรกเป็นยังไง → ทำนาย lifetime ได้
Insight 9: อยู่ครบกี่งวดถึง "safe zone" → ไม่ churn แล้ว

Output:
  - charts/insight_06_first_purchase.png
  - charts/insight_07_first_day.png
  - charts/insight_08_first_3_periods.png
  - charts/insight_09_safe_zone.png
"""

import os, sys, warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

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
DATA_DIR = 'data'
CHURN_DIR = os.path.join(DATA_DIR, 'churn')
TX_DIR = os.path.join(DATA_DIR, 'transaction')
CHART_DIR = 'charts'
os.makedirs(CHART_DIR, exist_ok=True)

CHURN_LATEST = os.path.join(CHURN_DIR, 'Churn_2026_0301_0316.csv')

# Colors
C_BOUGHT = '#F97316'    # orange
C_ZERO = '#E5E7EB'      # gray
C_NAN = '#1F2937'        # dark
C_STAR = '#EF4444'       # red for star
C_GOLD = '#F59E0B'       # gold dashed line

# ═══════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════

def load_churn():
    """Load latest churn file."""
    print("=" * 70)
    print("Loading churn data...")
    df = pd.read_csv(CHURN_LATEST, low_memory=False)
    item_cols = sorted([c for c in df.columns if c.startswith('item')])
    print(f"  {len(df):,} customers, {len(item_cols)} periods")
    return df, item_cols


def col_to_label(col):
    """Convert item2025_01_02 → 01/02."""
    parts = col.replace('item', '').split('_')
    return f"{parts[1]}/{parts[2]}"


def plot_customer_bar(ax, row, item_cols, event_idx=None, event_label=None,
                      zone_start=None, zone_end=None, zone_color=None):
    """Plot a single customer's purchase history as a bar chart."""
    labels = [col_to_label(c) for c in item_cols]
    values = row[item_cols].values.astype(float)
    n = len(values)

    colors = []
    for v in values:
        if np.isnan(v):
            colors.append(C_NAN)
        elif v == 0:
            colors.append(C_ZERO)
        else:
            colors.append(C_BOUGHT)

    # Plot bars (NaN → show small bar for visibility)
    bar_heights = np.where(np.isnan(values), 0.3, np.where(values == 0, 0.3, values))
    bars = ax.bar(range(n), bar_heights, color=colors, edgecolor='white', linewidth=0.3, width=0.8)

    # Zone highlight (for insight 8)
    if zone_start is not None and zone_end is not None:
        ax.axvspan(zone_start - 0.5, zone_end + 0.5, alpha=0.15, color=zone_color or '#3B82F6',
                   zorder=0)

    # Event marker
    if event_idx is not None:
        ymax = max(np.nanmax(values[values > 0]) if np.any(values > 0) else 1, 1)
        ax.axvline(x=event_idx, color=C_GOLD, linestyle='--', linewidth=2, alpha=0.8, zorder=3)
        ax.text(event_idx, ymax * 1.05, '★', fontsize=20, color=C_STAR,
                ha='center', va='bottom', fontweight='bold', zorder=4)
        if event_label:
            ax.text(event_idx, ymax * 1.25, event_label, fontsize=9, color=C_STAR,
                    ha='center', va='bottom', zorder=4)

    # X-axis labels (show every 5th)
    ax.set_xticks(range(n))
    ax.set_xticklabels([labels[i] if i % 5 == 0 else '' for i in range(n)],
                       fontsize=7, rotation=45)
    ax.set_ylabel('จำนวนใบ', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(-0.5, n - 0.5)

    return ax


def get_customer_stats(row, item_cols):
    """Get basic stats for a customer."""
    vals = row[item_cols].values.astype(float)
    registered = ~np.isnan(vals)
    bought = (vals > 0) & registered
    tenure = int(registered.sum())
    active = int(bought.sum())
    total_tickets = int(np.nansum(vals))
    # First purchase
    first_idx = None
    first_amount = 0
    for i, v in enumerate(vals):
        if not np.isnan(v) and v > 0:
            first_idx = i
            first_amount = int(v)
            break
    # Last purchase
    last_idx = None
    for i in range(len(vals) - 1, -1, -1):
        if not np.isnan(vals[i]) and vals[i] > 0:
            last_idx = i
            break
    return {
        'tenure': tenure, 'active': active, 'total_tickets': total_tickets,
        'first_idx': first_idx, 'first_amount': first_amount, 'last_idx': last_idx,
        'userNo': row['userNo'],
        'age': row.get('age', ''),
        'gender': row.get('gender', ''),
        'province': row.get('province', ''),
    }


# ═══════════════════════════════════════════════════════════════
# INSIGHT 6: ซื้อครั้งแรกกี่ใบ → อยู่กับเรากี่งวด
# ═══════════════════════════════════════════════════════════════
def run_insight_6(df, item_cols):
    print("\n" + "=" * 70)
    print("INSIGHT 6: ซื้อครั้งแรกกี่ใบ → อยู่กับเรากี่งวด")
    print("=" * 70)

    # Compute first purchase amount and tenure for each customer
    vals = df[item_cols].values.astype(float)
    n_cust, n_periods = vals.shape

    first_amount = np.full(n_cust, np.nan)
    first_idx_arr = np.full(n_cust, -1, dtype=int)
    tenure_arr = np.zeros(n_cust, dtype=int)
    active_arr = np.zeros(n_cust, dtype=int)
    total_tickets = np.zeros(n_cust, dtype=float)

    registered = ~np.isnan(vals)
    bought = (vals > 0) & registered
    tenure_arr = registered.sum(axis=1)
    active_arr = bought.sum(axis=1)
    total_tickets = np.nansum(np.where(vals > 0, vals, 0), axis=1)

    # First purchase index & amount
    for i in range(n_cust):
        for j in range(n_periods):
            if not np.isnan(vals[i, j]) and vals[i, j] > 0:
                first_amount[i] = vals[i, j]
                first_idx_arr[i] = j
                break

    has_purchase = ~np.isnan(first_amount)
    print(f"  ลูกค้าที่เคยซื้อ: {has_purchase.sum():,} / {n_cust:,}")

    # Create analysis dataframe
    adf = pd.DataFrame({
        'first_amount': first_amount[has_purchase],
        'tenure': tenure_arr[has_purchase],
        'active': active_arr[has_purchase],
        'total_tickets': total_tickets[has_purchase],
    })

    # Group by first purchase size
    bins = [0, 2, 5, 10, 20, 50, 99999]
    labels_g = ['1-2 ใบ', '3-5 ใบ', '6-10 ใบ', '11-20 ใบ', '21-50 ใบ', '50+ ใบ']
    adf['group'] = pd.cut(adf['first_amount'], bins=bins, labels=labels_g)

    summary = adf.groupby('group', observed=True).agg(
        count=('tenure', 'size'),
        avg_tenure=('tenure', 'mean'),
        avg_active=('active', 'mean'),
        avg_total_tickets=('total_tickets', 'mean'),
        median_active=('active', 'median'),
    ).reset_index()

    print("\n  กลุ่มตามจำนวนซื้อครั้งแรก:")
    print(f"  {'กลุ่ม':<12} {'จำนวน':>10} {'Avg Tenure':>12} {'Avg Active':>12} {'Avg Tickets':>13}")
    print("  " + "-" * 62)
    for _, r in summary.iterrows():
        print(f"  {r['group']:<12} {r['count']:>10,} {r['avg_tenure']:>12.1f} "
              f"{r['avg_active']:>12.1f} {r['avg_total_tickets']:>13.0f}")

    # Find examples
    mask_purchased = has_purchase
    df_with = df[mask_purchased].copy()
    df_with['_first_amount'] = first_amount[mask_purchased]
    df_with['_first_idx'] = first_idx_arr[mask_purchased]
    df_with['_active'] = active_arr[mask_purchased]
    df_with['_tenure'] = tenure_arr[mask_purchased]
    df_with['_total'] = total_tickets[mask_purchased]

    # Example 1: bought 1-2 first, churned fast (active <= 3)
    ex1_pool = df_with[(df_with['_first_amount'] <= 2) & (df_with['_active'] <= 3) &
                       (df_with['_tenure'] >= 5)]
    ex1 = ex1_pool.sample(1, random_state=42).iloc[0] if len(ex1_pool) > 0 else df_with.iloc[0]

    # Example 2: bought 10+ first, stayed long (active >= 15)
    ex2_pool = df_with[(df_with['_first_amount'] >= 10) & (df_with['_active'] >= 15)]
    ex2 = ex2_pool.sample(1, random_state=42).iloc[0] if len(ex2_pool) > 0 else df_with.iloc[1]

    # Example 3: interesting — bought many first time but churned, or bought few but stayed
    ex3_pool = df_with[(df_with['_first_amount'] >= 20) & (df_with['_active'] <= 5) &
                       (df_with['_tenure'] >= 10)]
    if len(ex3_pool) == 0:
        ex3_pool = df_with[(df_with['_first_amount'] <= 3) & (df_with['_active'] >= 20)]
    ex3 = ex3_pool.sample(1, random_state=42).iloc[0] if len(ex3_pool) > 0 else df_with.iloc[2]

    examples = [ex1, ex2, ex3]
    ex_titles = [
        'ซื้อน้อยครั้งแรก → หายไปเร็ว',
        'ซื้อเยอะครั้งแรก → อยู่นาน',
        'Pattern น่าสนใจ'
    ]

    for i, ex in enumerate(examples):
        s = get_customer_stats(ex, item_cols)
        print(f"\n  ตัวอย่าง {i+1}: {ex_titles[i]}")
        print(f"    {s['userNo']} | อายุ {s['age']} | {s['gender']} | {s['province']}")
        print(f"    ซื้อครั้งแรก: {s['first_amount']} ใบ (งวดที่ {col_to_label(item_cols[s['first_idx']])})")
        print(f"    Tenure: {s['tenure']} งวด | Active: {s['active']} งวด | รวม: {s['total_tickets']:,} ใบ")

    # ── Chart ──
    fig, axes = plt.subplots(4, 1, figsize=(20, 18), gridspec_kw={'height_ratios': [1.2, 1, 1, 1]})
    fig.suptitle('Insight 6: ซื้อครั้งแรกกี่ใบ → อยู่กับเรากี่งวด', fontsize=18, fontweight='bold', y=0.98)

    # Summary bar chart
    ax0 = axes[0]
    x_pos = range(len(summary))
    bars_tenure = ax0.bar([p - 0.2 for p in x_pos], summary['avg_active'], width=0.35,
                          color=C_BOUGHT, label='Avg Active Periods', edgecolor='white')
    ax0_2 = ax0.twinx()
    bars_tickets = ax0_2.bar([p + 0.2 for p in x_pos], summary['avg_total_tickets'], width=0.35,
                             color='#3B82F6', alpha=0.7, label='Avg Total Tickets', edgecolor='white')
    ax0.set_xticks(list(x_pos))
    ax0.set_xticklabels(summary['group'].values, fontsize=11)
    ax0.set_ylabel('Avg Active Periods', fontsize=11, color=C_BOUGHT)
    ax0_2.set_ylabel('Avg Total Tickets', fontsize=11, color='#3B82F6')
    ax0.set_title('ค่าเฉลี่ย Active Periods และ Total Tickets ตามจำนวนซื้อครั้งแรก', fontsize=13)
    # Add count labels
    for idx, r in summary.iterrows():
        ax0.text(list(x_pos)[idx], -0.8, f'n={int(r["count"]):,}', ha='center', fontsize=8, color='gray')
    lines1, labels1 = ax0.get_legend_handles_labels()
    lines2, labels2 = ax0_2.get_legend_handles_labels()
    ax0.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    ax0.grid(axis='y', alpha=0.3)

    # Customer examples
    for i, ex in enumerate(examples):
        ax = axes[i + 1]
        s = get_customer_stats(ex, item_cols)
        plot_customer_bar(ax, ex, item_cols, event_idx=s['first_idx'],
                          event_label='ซื้อครั้งแรก')
        ax.set_title(f"{ex_titles[i]}: {s['userNo']} | ซื้อครั้งแรก {s['first_amount']} ใบ | "
                     f"Active {s['active']}/{s['tenure']} งวด | รวม {s['total_tickets']:,} ใบ",
                     fontsize=11, pad=10)

    # Legend
    legend_patches = [
        mpatches.Patch(color=C_BOUGHT, label='ซื้อ (มีใบ)'),
        mpatches.Patch(color=C_ZERO, label='ไม่ซื้อ (0 ใบ)'),
        mpatches.Patch(color=C_NAN, label='ยังไม่ลงทะเบียน'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = os.path.join(CHART_DIR, 'insight_06_first_purchase.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  ✅ Saved: {out}")

    return summary


# ═══════════════════════════════════════════════════════════════
# INSIGHT 7: ซื้อครั้งแรกวันไหนของรอบขาย → retention ต่างกัน
# ═══════════════════════════════════════════════════════════════
def run_insight_7(df, item_cols):
    print("\n" + "=" * 70)
    print("INSIGHT 7: ซื้อครั้งแรกวันไหนของรอบขาย → retention ต่างกัน")
    print("=" * 70)

    # Load all TX files to find customers' FIRST ever order
    print("  Loading TX files to find first-ever order day...")
    tx_frames = []
    tx_files = sorted([f for f in os.listdir(TX_DIR) if f.startswith('Transaction_Order')])
    for f in tx_files:
        t = pd.read_csv(os.path.join(TX_DIR, f), low_memory=False,
                        usecols=['userNo', 'roundDate', 'createdAt_bkk', 'status'])
        t = t[t['status'] == 'SUCCESS'].copy()
        t['roundDate'] = pd.to_datetime(t['roundDate'], utc=True).dt.tz_localize(None)
        t['createdAt'] = pd.to_datetime(t['createdAt_bkk'], errors='coerce', utc=True).dt.tz_localize(None)
        tx_frames.append(t[['userNo', 'roundDate', 'createdAt']])
    tx_all = pd.concat(tx_frames, ignore_index=True)
    print(f"  Total TX rows (SUCCESS): {len(tx_all):,}")

    # For each customer: find their FIRST ever order
    tx_all = tx_all.dropna(subset=['createdAt'])
    first_orders = tx_all.sort_values('createdAt').groupby('userNo').first().reset_index()
    first_orders = first_orders.rename(columns={'roundDate': 'first_round', 'createdAt': 'first_created'})

    # Calculate day of sales window
    # Sales window starts ~3-4 days after previous draw
    # roundDate is the draw date (1st or 16th)
    # Day of window = (createdAt - (roundDate - 13 days)).days + 1 approximately
    # Better: day_of_window = (roundDate - createdAt).days is days before draw
    # We want day 1 = first day of sales, day 13 = draw day
    first_orders['days_before_draw'] = (first_orders['first_round'] - first_orders['first_created'].dt.normalize()).dt.days
    first_orders['day_of_window'] = 13 - first_orders['days_before_draw']
    first_orders['day_of_window'] = first_orders['day_of_window'].clip(1, 13)

    # Group: Early (1-5), Mid (6-10), Rush (11-13)
    def day_group(d):
        if d <= 5:
            return 'Early (day 1-5)'
        elif d <= 10:
            return 'Mid (day 6-10)'
        else:
            return 'Rush (day 11-13)'

    first_orders['timing_group'] = first_orders['day_of_window'].apply(day_group)

    # Merge with churn data to get tenure/activity
    vals = df[item_cols].values.astype(float)
    registered = ~np.isnan(vals)
    bought = (vals > 0) & registered

    df_stats = pd.DataFrame({
        'userNo': df['userNo'],
        'tenure': registered.sum(axis=1),
        'active': bought.sum(axis=1),
        'total_tickets': np.nansum(np.where(vals > 0, vals, 0), axis=1),
    })

    # Churn within first 3 periods
    # Find first registered period, check if active in next 2
    first_reg = np.argmax(registered, axis=1)
    churned_early = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        fr = first_reg[i]
        if fr + 3 < vals.shape[1]:
            # Active in first 3 periods after registration
            window = bought[i, fr:fr+3]
            if window.sum() <= 1:  # only bought once in first 3
                churned_early[i] = True
    df_stats['churned_early'] = churned_early

    merged = first_orders.merge(df_stats, on='userNo', how='inner')
    print(f"  Matched customers: {len(merged):,}")

    # Summary
    grp = merged.groupby('timing_group').agg(
        count=('tenure', 'size'),
        avg_tenure=('tenure', 'mean'),
        avg_active=('active', 'mean'),
        avg_frequency=('active', lambda x: (x / merged.loc[x.index, 'tenure']).mean()),
        churn_early_rate=('churned_early', 'mean'),
    ).reset_index()

    print("\n  กลุ่มตามวันซื้อครั้งแรก:")
    print(f"  {'กลุ่ม':<20} {'จำนวน':>10} {'Avg Tenure':>12} {'Avg Active':>12} {'Churn Early %':>14}")
    print("  " + "-" * 70)
    for _, r in grp.iterrows():
        print(f"  {r['timing_group']:<20} {r['count']:>10,} {r['avg_tenure']:>12.1f} "
              f"{r['avg_active']:>12.1f} {r['churn_early_rate']:>13.1%}")

    # Find examples
    merged_with_churn = merged.merge(df[['userNo'] + item_cols], on='userNo', how='inner')

    # Ex1: Early first-timer who stayed (active >= 15)
    ex1_pool = merged_with_churn[(merged_with_churn['timing_group'] == 'Early (day 1-5)') &
                                  (merged_with_churn['active'] >= 15)]
    ex1 = ex1_pool.sample(1, random_state=42).iloc[0] if len(ex1_pool) > 0 else merged_with_churn.iloc[0]

    # Ex2: Rush first-timer who churned (active <= 3, tenure >= 5)
    ex2_pool = merged_with_churn[(merged_with_churn['timing_group'] == 'Rush (day 11-13)') &
                                  (merged_with_churn['active'] <= 3) &
                                  (merged_with_churn['tenure'] >= 5)]
    ex2 = ex2_pool.sample(1, random_state=42).iloc[0] if len(ex2_pool) > 0 else merged_with_churn.iloc[1]

    # Ex3: Mid who stayed long
    ex3_pool = merged_with_churn[(merged_with_churn['timing_group'] == 'Mid (day 6-10)') &
                                  (merged_with_churn['active'] >= 12)]
    ex3 = ex3_pool.sample(1, random_state=42).iloc[0] if len(ex3_pool) > 0 else merged_with_churn.iloc[2]

    examples = [ex1, ex2, ex3]
    ex_titles = [
        'Early First-Timer → อยู่นาน',
        'Rush First-Timer → หายไปเร็ว',
        'Mid First-Timer → อยู่ต่อ'
    ]

    for i, ex in enumerate(examples):
        s = get_customer_stats(ex, item_cols)
        day_val = ex.get('day_of_window', '?')
        print(f"\n  ตัวอย่าง {i+1}: {ex_titles[i]}")
        print(f"    {s['userNo']} | ซื้อครั้งแรกวันที่ {day_val} ของรอบขาย")
        print(f"    Active: {s['active']}/{s['tenure']} งวด | รวม: {s['total_tickets']:,} ใบ")

    # ── Chart ──
    fig, axes = plt.subplots(4, 1, figsize=(20, 18), gridspec_kw={'height_ratios': [1.2, 1, 1, 1]})
    fig.suptitle('Insight 7: ซื้อครั้งแรกวันไหนของรอบขาย → retention ต่างกัน',
                 fontsize=18, fontweight='bold', y=0.98)

    # Summary grouped bar
    ax0 = axes[0]
    groups_ordered = ['Early (day 1-5)', 'Mid (day 6-10)', 'Rush (day 11-13)']
    grp_o = grp.set_index('timing_group').reindex(groups_ordered).reset_index()
    x_pos = range(len(grp_o))
    colors_grp = ['#22C55E', '#3B82F6', '#EF4444']

    bars = ax0.bar(x_pos, grp_o['avg_active'], color=colors_grp, edgecolor='white', width=0.5)
    for idx, r in grp_o.iterrows():
        ax0.text(idx, r['avg_active'] + 0.2, f"{r['avg_active']:.1f} periods",
                 ha='center', fontsize=10, fontweight='bold')
        ax0.text(idx, -0.8, f"n={int(r['count']):,}\nChurn Early: {r['churn_early_rate']:.1%}",
                 ha='center', fontsize=9, color='gray')
    ax0.set_xticks(list(x_pos))
    ax0.set_xticklabels(grp_o['timing_group'].values, fontsize=12)
    ax0.set_ylabel('Avg Active Periods', fontsize=11)
    ax0.set_title('ค่าเฉลี่ย Active Periods ตามช่วงเวลาซื้อครั้งแรก', fontsize=13)
    ax0.grid(axis='y', alpha=0.3)

    # Customer examples
    for i, ex in enumerate(examples):
        ax = axes[i + 1]
        s = get_customer_stats(ex, item_cols)
        plot_customer_bar(ax, ex, item_cols, event_idx=s['first_idx'],
                          event_label='ซื้อครั้งแรก')
        day_val = ex.get('day_of_window', '?')
        ax.set_title(f"{ex_titles[i]}: {s['userNo']} | Day {day_val} ของรอบขาย | "
                     f"Active {s['active']}/{s['tenure']} งวด",
                     fontsize=11, pad=10)

    legend_patches = [
        mpatches.Patch(color=C_BOUGHT, label='ซื้อ (มีใบ)'),
        mpatches.Patch(color=C_ZERO, label='ไม่ซื้อ (0 ใบ)'),
        mpatches.Patch(color=C_NAN, label='ยังไม่ลงทะเบียน'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = os.path.join(CHART_DIR, 'insight_07_first_day.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  ✅ Saved: {out}")

    return grp


# ═══════════════════════════════════════════════════════════════
# INSIGHT 8: ลูกค้าใหม่ 3 งวดแรกเป็นยังไง → ทำนาย lifetime ได้
# ═══════════════════════════════════════════════════════════════
def run_insight_8(df, item_cols):
    print("\n" + "=" * 70)
    print("INSIGHT 8: ลูกค้าใหม่ 3 งวดแรกเป็นยังไง → ทำนาย lifetime ได้")
    print("=" * 70)

    vals = df[item_cols].values.astype(float)
    n_cust, n_periods = vals.shape

    registered = ~np.isnan(vals)
    bought = (vals > 0) & registered

    # Find first registered period
    # Only consider customers with tenure >= 10
    tenure = registered.sum(axis=1)
    active = bought.sum(axis=1)
    total_tickets = np.nansum(np.where(vals > 0, vals, 0), axis=1)

    mask_tenure10 = tenure >= 10

    first_reg_idx = np.full(n_cust, -1, dtype=int)
    for i in range(n_cust):
        for j in range(n_periods):
            if registered[i, j]:
                first_reg_idx[i] = j
                break

    # For tenure >= 10 customers, check first 3 periods pattern
    pattern = np.full(n_cust, '', dtype='U10')
    first3_active = np.zeros(n_cust, dtype=int)

    for i in range(n_cust):
        if not mask_tenure10[i] or first_reg_idx[i] < 0:
            continue
        fr = first_reg_idx[i]
        count = 0
        for j in range(fr, min(fr + 3, n_periods)):
            if bought[i, j]:
                count += 1
        first3_active[i] = count

    # Create analysis df
    adf = pd.DataFrame({
        'userNo': df['userNo'],
        'tenure': tenure,
        'active': active,
        'total_tickets': total_tickets,
        'first_reg_idx': first_reg_idx,
        'first3_active': first3_active,
        'mask': mask_tenure10,
    })
    adf = adf[adf['mask']].copy()
    # Filter: first_reg_idx must allow 3 periods
    adf = adf[adf['first_reg_idx'] + 3 <= n_periods].copy()

    print(f"  ลูกค้า tenure >= 10 งวด: {len(adf):,}")

    # Patterns
    patterns = {
        'A: ซื้อ 3/3 งวดแรก': adf[adf['first3_active'] == 3],
        'B: ซื้อ 2/3 งวดแรก': adf[adf['first3_active'] == 2],
        'C: ซื้อ 1/3 งวดแรก': adf[adf['first3_active'] == 1],
    }

    print(f"\n  {'Pattern':<25} {'จำนวน':>10} {'Avg Active':>12} {'Avg Tickets':>13}")
    print("  " + "-" * 62)
    pattern_stats = []
    for name, grp in patterns.items():
        avg_act = grp['active'].mean()
        avg_tick = grp['total_tickets'].mean()
        pattern_stats.append({'pattern': name, 'count': len(grp),
                              'avg_active': avg_act, 'avg_tickets': avg_tick})
        print(f"  {name:<25} {len(grp):>10,} {avg_act:>12.1f} {avg_tick:>13.0f}")

    # Find examples
    df_full = df.copy()
    df_full['_active'] = active
    df_full['_tenure'] = tenure
    df_full['_total'] = total_tickets
    df_full['_first_reg_idx'] = first_reg_idx
    df_full['_first3_active'] = first3_active

    # Ex1: Pattern A (3/3) who became loyal (active >= 25)
    ex1_pool = df_full[(df_full['_first3_active'] == 3) & (df_full['_tenure'] >= 10) &
                        (df_full['_active'] >= 25) & (df_full['_first_reg_idx'] >= 0) &
                        (df_full['_first_reg_idx'] + 3 <= n_periods)]
    if len(ex1_pool) == 0:
        ex1_pool = df_full[(df_full['_first3_active'] == 3) & (df_full['_active'] >= 15)]
    ex1 = ex1_pool.sample(1, random_state=42).iloc[0] if len(ex1_pool) > 0 else df_full.iloc[0]

    # Ex2: Pattern C (1/3) who churned (active <= 5)
    ex2_pool = df_full[(df_full['_first3_active'] == 1) & (df_full['_tenure'] >= 10) &
                        (df_full['_active'] <= 5) & (df_full['_first_reg_idx'] >= 0) &
                        (df_full['_first_reg_idx'] + 3 <= n_periods)]
    if len(ex2_pool) == 0:
        ex2_pool = df_full[(df_full['_first3_active'] == 1) & (df_full['_active'] <= 8)]
    ex2 = ex2_pool.sample(1, random_state=42).iloc[0] if len(ex2_pool) > 0 else df_full.iloc[1]

    # Ex3: Pattern B (2/3) mid behavior (active 8-15)
    ex3_pool = df_full[(df_full['_first3_active'] == 2) & (df_full['_tenure'] >= 10) &
                        (df_full['_active'] >= 8) & (df_full['_active'] <= 15) &
                        (df_full['_first_reg_idx'] >= 0) &
                        (df_full['_first_reg_idx'] + 3 <= n_periods)]
    ex3 = ex3_pool.sample(1, random_state=42).iloc[0] if len(ex3_pool) > 0 else df_full.iloc[2]

    examples = [ex1, ex2, ex3]
    ex_titles = [
        'Pattern A (3/3) → ลูกค้าภักดี',
        'Pattern C (1/3) → หายไป',
        'Pattern B (2/3) → พฤติกรรมปานกลาง'
    ]

    for i, ex in enumerate(examples):
        s = get_customer_stats(ex, item_cols)
        print(f"\n  ตัวอย่าง {i+1}: {ex_titles[i]}")
        print(f"    {s['userNo']} | อายุ {s['age']} | {s['gender']} | {s['province']}")
        print(f"    Active: {s['active']}/{s['tenure']} งวด | รวม: {s['total_tickets']:,} ใบ")

    # ── Chart ──
    fig, axes = plt.subplots(4, 1, figsize=(20, 18), gridspec_kw={'height_ratios': [1.2, 1, 1, 1]})
    fig.suptitle('Insight 8: ลูกค้าใหม่ 3 งวดแรกเป็นยังไง → ทำนาย lifetime ได้',
                 fontsize=18, fontweight='bold', y=0.98)

    # Summary bar
    ax0 = axes[0]
    ps = pd.DataFrame(pattern_stats)
    colors_p = ['#22C55E', '#F59E0B', '#EF4444']
    bars = ax0.bar(range(len(ps)), ps['avg_active'], color=colors_p, edgecolor='white', width=0.5)
    for idx, r in ps.iterrows():
        ax0.text(idx, r['avg_active'] + 0.3, f"{r['avg_active']:.1f} periods",
                 ha='center', fontsize=11, fontweight='bold')
        ax0.text(idx, -1.0, f"n={int(r['count']):,}", ha='center', fontsize=9, color='gray')
    ax0.set_xticks(range(len(ps)))
    ax0.set_xticklabels(ps['pattern'].values, fontsize=12)
    ax0.set_ylabel('Avg Active Periods', fontsize=11)
    ax0.set_title('ค่าเฉลี่ย Active Periods ตาม Pattern 3 งวดแรก (tenure >= 10)', fontsize=13)
    ax0.grid(axis='y', alpha=0.3)

    # Customer examples
    for i, ex in enumerate(examples):
        ax = axes[i + 1]
        s = get_customer_stats(ex, item_cols)
        fr = int(ex.get('_first_reg_idx', s['first_idx'] or 0))
        zone_end = min(fr + 2, n_periods - 1)
        zone_color = colors_p[i] if i < len(colors_p) else '#3B82F6'
        plot_customer_bar(ax, ex, item_cols, event_idx=None,
                          zone_start=fr, zone_end=zone_end, zone_color=zone_color)
        # Add zone label
        mid_zone = (fr + zone_end) / 2
        ymax = max(np.nanmax(ex[item_cols].values.astype(float)[ex[item_cols].values.astype(float) > 0])
                   if np.any(ex[item_cols].values.astype(float) > 0) else 1, 1)
        ax.text(mid_zone, ymax * 1.1, '★ 3 งวดแรก', fontsize=12, color=C_STAR,
                ha='center', fontweight='bold')
        f3 = int(ex.get('_first3_active', 0))
        ax.set_title(f"{ex_titles[i]}: {s['userNo']} | ซื้อ {f3}/3 งวดแรก | "
                     f"Active {s['active']}/{s['tenure']} งวด | รวม {s['total_tickets']:,} ใบ",
                     fontsize=11, pad=10)

    legend_patches = [
        mpatches.Patch(color=C_BOUGHT, label='ซื้อ (มีใบ)'),
        mpatches.Patch(color=C_ZERO, label='ไม่ซื้อ (0 ใบ)'),
        mpatches.Patch(color=C_NAN, label='ยังไม่ลงทะเบียน'),
        mpatches.Patch(color='#3B82F6', alpha=0.3, label='3 งวดแรก (zone)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = os.path.join(CHART_DIR, 'insight_08_first_3_periods.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  ✅ Saved: {out}")

    return pd.DataFrame(pattern_stats)


# ═══════════════════════════════════════════════════════════════
# INSIGHT 9: อยู่ครบกี่งวดถึง "safe zone" → ไม่ churn แล้ว
# ═══════════════════════════════════════════════════════════════
def run_insight_9(df, item_cols):
    print("\n" + "=" * 70)
    print("INSIGHT 9: อยู่ครบกี่งวดถึง \"safe zone\" → ไม่ churn แล้ว")
    print("=" * 70)

    vals = df[item_cols].values.astype(float)
    n_cust, n_periods = vals.shape

    registered = ~np.isnan(vals)
    bought = (vals > 0) & registered

    # For churn probability by tenure:
    # At each tenure level t, count customers who had exactly t active periods
    # and then check if they were active in the LAST available period
    # Better approach: for customers at each accumulated active-period count,
    # what fraction churned (didn't buy in the last period)?

    # Simpler: For each customer, compute their tenure_at_each_point and track if they churned
    # Churn = didn't buy in last period (item_cols[-1])
    last_col_vals = vals[:, -1]
    churned = np.isnan(last_col_vals) | (last_col_vals == 0)  # didn't buy in last period

    # Only consider registered customers (not NaN in last col and registered before)
    active_periods = bought.sum(axis=1)
    tenure_periods = registered.sum(axis=1)

    # For customers registered in last period
    registered_last = registered[:, -1]

    # Compute: at tenure t, what % churned?
    # tenure = number of periods they were registered
    churn_by_tenure = []
    for t in range(1, int(tenure_periods.max()) + 1):
        mask = (tenure_periods == t) & registered_last
        if mask.sum() >= 50:  # need enough samples
            churn_rate = churned[mask].mean()
            churn_by_tenure.append({
                'tenure': t, 'count': int(mask.sum()),
                'churn_rate': churn_rate, 'retention_rate': 1 - churn_rate
            })

    cbt = pd.DataFrame(churn_by_tenure)

    # Better metric: churn by ACTIVE periods (not just registered)
    churn_by_active = []
    for a in range(1, int(active_periods.max()) + 1):
        mask = (active_periods == a) & registered_last
        if mask.sum() >= 50:
            churn_rate = churned[mask].mean()
            churn_by_active.append({
                'active_periods': a, 'count': int(mask.sum()),
                'churn_rate': churn_rate
            })
    cba = pd.DataFrame(churn_by_active)

    # Find safe zone thresholds
    threshold_20 = None
    threshold_15 = None
    threshold_10 = None
    for _, r in cba.iterrows():
        if threshold_20 is None and r['churn_rate'] < 0.20:
            threshold_20 = int(r['active_periods'])
        if threshold_15 is None and r['churn_rate'] < 0.15:
            threshold_15 = int(r['active_periods'])
        if threshold_10 is None and r['churn_rate'] < 0.10:
            threshold_10 = int(r['active_periods'])

    print(f"\n  Churn rate < 20%: Active periods >= {threshold_20}")
    print(f"  Churn rate < 15%: Active periods >= {threshold_15}")
    print(f"  Churn rate < 10%: Active periods >= {threshold_10}")
    safe_zone = threshold_20 or 10  # fallback

    print(f"\n  Safe Zone Threshold: {safe_zone} active periods")
    print(f"  ถ้ารักษาลูกค้าได้ถึง {safe_zone} งวด → Churn rate ต่ำกว่า 20%")

    print(f"\n  Churn Rate by Active Periods:")
    print(f"  {'Active':>8} {'Count':>10} {'Churn %':>10}")
    print("  " + "-" * 30)
    for _, r in cba.head(20).iterrows():
        marker = " ← SAFE ZONE" if int(r['active_periods']) == safe_zone else ""
        print(f"  {int(r['active_periods']):>8} {int(r['count']):>10,} {r['churn_rate']:>9.1%}{marker}")

    # Find examples
    df_full = df.copy()
    df_full['_active'] = active_periods
    df_full['_tenure'] = tenure_periods
    df_full['_churned'] = churned
    df_full['_total'] = np.nansum(np.where(vals > 0, vals, 0), axis=1)
    df_full['_registered_last'] = registered_last

    # Ex1: Crossed safe zone and stayed (active >= safe_zone, not churned)
    ex1_pool = df_full[(df_full['_active'] >= safe_zone + 5) & (~df_full['_churned']) &
                        (df_full['_registered_last'])]
    ex1 = ex1_pool.sample(1, random_state=42).iloc[0] if len(ex1_pool) > 0 else df_full.iloc[0]

    # Ex2: Churned just before safe zone (active = safe_zone - 1 or safe_zone - 2)
    ex2_pool = df_full[(df_full['_active'] >= max(safe_zone - 2, 1)) &
                        (df_full['_active'] <= safe_zone) &
                        (df_full['_churned']) & (df_full['_registered_last'])]
    if len(ex2_pool) == 0:
        ex2_pool = df_full[(df_full['_active'] <= safe_zone) & (df_full['_churned'])]
    ex2 = ex2_pool.sample(1, random_state=42).iloc[0] if len(ex2_pool) > 0 else df_full.iloc[1]

    # Ex3: Long tenure loyal customer (active >= 30)
    ex3_pool = df_full[(df_full['_active'] >= 30) & (~df_full['_churned']) &
                        (df_full['_registered_last'])]
    if len(ex3_pool) == 0:
        ex3_pool = df_full[df_full['_active'] >= 25]
    ex3 = ex3_pool.sample(1, random_state=42).iloc[0] if len(ex3_pool) > 0 else df_full.iloc[2]

    examples = [ex1, ex2, ex3]
    ex_titles = [
        f'ข้าม Safe Zone (>= {safe_zone} งวด) → อยู่ต่อ',
        f'Churn ก่อน Safe Zone',
        'ลูกค้าระยะยาว — ภักดีตลอด'
    ]

    for i, ex in enumerate(examples):
        s = get_customer_stats(ex, item_cols)
        ch = 'Churned' if ex.get('_churned', False) else 'Active'
        print(f"\n  ตัวอย่าง {i+1}: {ex_titles[i]}")
        print(f"    {s['userNo']} | อายุ {s['age']} | {s['gender']} | {s['province']}")
        print(f"    Active: {s['active']}/{s['tenure']} งวด | รวม: {s['total_tickets']:,} ใบ | Status: {ch}")

    # ── Chart ──
    fig, axes = plt.subplots(4, 1, figsize=(20, 18), gridspec_kw={'height_ratios': [1.3, 1, 1, 1]})
    fig.suptitle('Insight 9: อยู่ครบกี่งวดถึง "Safe Zone" → ไม่ churn แล้ว',
                 fontsize=18, fontweight='bold', y=0.98)

    # Summary: churn rate curve
    ax0 = axes[0]
    ax0.bar(cba['active_periods'], cba['churn_rate'] * 100, color=C_BOUGHT, edgecolor='white',
            width=0.8, alpha=0.8)
    ax0.axhline(y=20, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='20% threshold')
    ax0.axhline(y=15, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='15% threshold')
    ax0.axhline(y=10, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='10% threshold')
    if safe_zone:
        ax0.axvline(x=safe_zone, color=C_GOLD, linestyle='--', linewidth=2.5, alpha=0.9)
        ax0.text(safe_zone + 0.3, ax0.get_ylim()[1] * 0.85,
                 f'★ Safe Zone\n(>= {safe_zone} active periods)',
                 fontsize=11, color=C_STAR, fontweight='bold')
    ax0.set_xlabel('Active Periods (จำนวนงวดที่ซื้อ)', fontsize=11)
    ax0.set_ylabel('Churn Rate (%)', fontsize=11)
    ax0.set_title('Churn Rate ตามจำนวน Active Periods — หา Safe Zone', fontsize=13)
    ax0.legend(fontsize=9, loc='upper right')
    ax0.grid(axis='y', alpha=0.3)
    ax0.set_xlim(0, min(40, cba['active_periods'].max() + 1))

    # Customer examples — mark the safe zone period in their history
    for i, ex in enumerate(examples):
        ax = axes[i + 1]
        s = get_customer_stats(ex, item_cols)

        # Find the period index where they hit safe_zone active periods
        vals_ex = ex[item_cols].values.astype(float)
        cumactive = np.cumsum((~np.isnan(vals_ex)) & (vals_ex > 0))
        safe_idx = None
        for j in range(len(cumactive)):
            if cumactive[j] >= safe_zone:
                safe_idx = j
                break

        plot_customer_bar(ax, ex, item_cols, event_idx=safe_idx,
                          event_label=f'Safe Zone ({safe_zone} งวด)')

        ch = 'Churned' if ex.get('_churned', False) else 'Active'
        ax.set_title(f"{ex_titles[i]}: {s['userNo']} | Active {s['active']}/{s['tenure']} งวด | "
                     f"รวม {s['total_tickets']:,} ใบ | {ch}",
                     fontsize=11, pad=10)

    legend_patches = [
        mpatches.Patch(color=C_BOUGHT, label='ซื้อ (มีใบ)'),
        mpatches.Patch(color=C_ZERO, label='ไม่ซื้อ (0 ใบ)'),
        mpatches.Patch(color=C_NAN, label='ยังไม่ลงทะเบียน'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = os.path.join(CHART_DIR, 'insight_09_safe_zone.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  ✅ Saved: {out}")

    return cba


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  INSIGHTS 6-9: Purchase Behavior Deep-Dive")
    print("=" * 70)

    df, item_cols = load_churn()

    s6 = run_insight_6(df, item_cols)
    s7 = run_insight_7(df, item_cols)
    s8 = run_insight_8(df, item_cols)
    s9 = run_insight_9(df, item_cols)

    print("\n" + "=" * 70)
    print("  ALL 4 INSIGHTS COMPLETE")
    print("=" * 70)
    print("  Charts saved:")
    print("    - charts/insight_06_first_purchase.png")
    print("    - charts/insight_07_first_day.png")
    print("    - charts/insight_08_first_3_periods.png")
    print("    - charts/insight_09_safe_zone.png")


if __name__ == '__main__':
    main()
