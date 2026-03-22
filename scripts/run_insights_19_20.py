#!/usr/bin/env python3
"""Insights 19-20: Gap Recovery & Return Purchase Size Analysis.

Insight 19: หายไปกี่งวดแล้วกลับมา → ยอดซื้อเท่าเดิมมั้ย
Insight 20: กลับมาครั้งแรกซื้อกี่ใบ → ทำนายได้ว่าจะอยู่ต่อมั้ย

Output:
  - charts/insight_19_gap_recovery.png
  - charts/insight_20_return_size.png
  - Console: Summary stats + examples in Thai
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
CHART_DIR = 'charts'
os.makedirs(CHART_DIR, exist_ok=True)

CHURN_LATEST = os.path.join(CHURN_DIR, 'Churn_2026_0301_0316.csv')
COLOR_BUY = '#F97316'      # orange — bought
COLOR_ZERO = '#E5E7EB'     # gray — didn't buy
COLOR_NAN = '#1F2937'      # dark — not registered
COLOR_STAR = '#EF4444'     # red — key event marker
COLOR_GOLD = '#F59E0B'     # gold — dashed line


def load_data():
    """Load latest churn file and extract item columns."""
    print("=" * 70)
    print("INSIGHTS 19-20: Gap Recovery & Return Purchase Analysis")
    print("=" * 70)
    print("\nLoading data...", flush=True)

    df = pd.read_csv(CHURN_LATEST, low_memory=False)
    item_cols = sorted([c for c in df.columns if c.startswith('item')])
    print(f"  Loaded: {len(df):,} customers, {len(item_cols)} periods")
    print(f"  Periods: {item_cols[0]} → {item_cols[-1]}")
    return df, item_cols


def find_gaps_and_returns(df, item_cols):
    """Find all gap-return events for each customer.

    A gap is 1+ consecutive periods of 0 (registered but didn't buy),
    preceded by at least 1 purchase and followed by a purchase (return).

    Returns a list of dicts with gap info.
    """
    data = df[item_cols].values  # (n_customers, n_periods)
    user_nos = df['userNo'].values
    ages = df['age'].values
    genders = df['gender'].values
    provinces = df['province'].values

    events = []
    n_customers, n_periods = data.shape

    for i in range(n_customers):
        row = data[i]
        # Find purchase indices (>0, not NaN)
        purchase_mask = (~np.isnan(row)) & (row > 0)
        purchase_indices = np.where(purchase_mask)[0]

        if len(purchase_indices) < 2:
            continue

        # Find registered mask (not NaN)
        registered_mask = ~np.isnan(row)

        # Walk through purchase indices looking for gaps
        for pi in range(1, len(purchase_indices)):
            prev_buy_idx = purchase_indices[pi - 1]
            curr_buy_idx = purchase_indices[pi]
            gap_length = curr_buy_idx - prev_buy_idx - 1

            if gap_length < 1:
                continue

            # Verify the gap periods are all 0 (registered but didn't buy)
            gap_periods = row[prev_buy_idx + 1: curr_buy_idx]
            if not all(registered_mask[prev_buy_idx + 1: curr_buy_idx]):
                continue  # some NaN in gap — not a clean gap

            # Compute pre-gap avg (all purchases before the gap)
            pre_indices = purchase_indices[:pi]
            pre_avg = np.mean(row[pre_indices])

            # Compute post-return avg (all purchases from return onward)
            post_indices = purchase_indices[pi:]
            post_avg = np.mean(row[post_indices])

            # Return purchase amount
            return_amount = row[curr_buy_idx]

            # How many periods stayed after return
            if pi < len(purchase_indices) - 1:
                # Check how far the last purchase is from return
                last_after_return = purchase_indices[-1]
                periods_after_return = last_after_return - curr_buy_idx
            else:
                periods_after_return = 0

            # Count active periods after return
            active_after = len(post_indices)

            # Check if churned again within 2 periods after return
            # (no purchase in the 2 periods immediately after return)
            periods_remaining = n_periods - curr_buy_idx - 1
            if periods_remaining >= 2:
                next2 = row[curr_buy_idx + 1: curr_buy_idx + 3]
                churned_within_2 = all(
                    (np.isnan(v) or v == 0) for v in next2
                )
            else:
                churned_within_2 = None  # can't tell

            # Stayed 3+ more periods?
            if periods_remaining >= 3:
                next3_indices = range(curr_buy_idx + 1, min(curr_buy_idx + 4, n_periods))
                buys_in_3 = sum(1 for j in next3_indices
                                if not np.isnan(row[j]) and row[j] > 0)
                stayed_3_plus = buys_in_3 >= 1  # at least 1 purchase in next 3 periods
            else:
                stayed_3_plus = None

            events.append({
                'userNo': user_nos[i],
                'age': ages[i],
                'gender': genders[i],
                'province': provinces[i],
                'row_idx': i,
                'gap_start': prev_buy_idx + 1,
                'gap_end': curr_buy_idx - 1,
                'return_idx': curr_buy_idx,
                'gap_length': gap_length,
                'pre_avg': pre_avg,
                'post_avg': post_avg,
                'return_amount': return_amount,
                'recovery_ratio': post_avg / pre_avg if pre_avg > 0 else np.nan,
                'periods_after_return': periods_after_return,
                'active_after': active_after,
                'churned_within_2': churned_within_2,
                'stayed_3_plus': stayed_3_plus,
            })

    return pd.DataFrame(events)


def plot_customer_bar(ax, row_data, item_cols, return_idx, title, subtitle=''):
    """Plot a single customer's purchase history as a bar chart."""
    n = len(item_cols)
    values = row_data.copy()

    colors = []
    for j in range(n):
        v = values[j]
        if np.isnan(v):
            colors.append(COLOR_NAN)
            values[j] = 0.3  # small bar for NaN
        elif v == 0:
            colors.append(COLOR_ZERO)
            values[j] = 0.3
        else:
            colors.append(COLOR_BUY)

    x = np.arange(n)
    bars = ax.bar(x, values, color=colors, edgecolor='white', linewidth=0.3, width=0.8)

    # Mark return period with star
    if 0 <= return_idx < n:
        ax.axvline(x=return_idx, color=COLOR_GOLD, linestyle='--', linewidth=2, alpha=0.8)
        ymax = max(values) if max(values) > 1 else 5
        ax.text(return_idx, ymax * 0.95, '★', fontsize=22, color=COLOR_STAR,
                ha='center', va='top', fontweight='bold')

    # Labels every 5 periods
    period_labels = [c.replace('item', '').replace('_', '/') for c in item_cols]
    tick_positions = list(range(0, n, 5))
    if (n - 1) not in tick_positions:
        tick_positions.append(n - 1)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([period_labels[i] for i in tick_positions],
                       rotation=45, ha='right', fontsize=7)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=9, color='#6B7280', style='italic')

    ax.set_ylabel('Tickets', fontsize=9)
    ax.set_xlim(-0.5, n - 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=COLOR_BUY, label='Bought'),
        Patch(facecolor=COLOR_ZERO, label='0 (no buy)'),
        Patch(facecolor=COLOR_NAN, label='NaN (not registered)'),
    ]
    ax.legend(handles=legend_items, loc='upper left', fontsize=7, framealpha=0.8)


# ═══════════════════════════════════════════════════════════════
# INSIGHT 19
# ═══════════════════════════════════════════════════════════════
def insight_19(df, item_cols, events_df):
    """Gap recovery analysis: does purchase volume recover after absence?"""
    print("\n" + "=" * 70)
    print("INSIGHT 19: หายไปกี่งวดแล้วกลับมา → ยอดซื้อเท่าเดิมมั้ย")
    print("=" * 70)

    if len(events_df) == 0:
        print("  No gap-return events found!")
        return

    # Group by gap length
    def gap_group(g):
        if g == 1: return '1 งวด'
        elif g == 2: return '2 งวด'
        elif g == 3: return '3 งวด'
        elif 4 <= g <= 6: return '4-6 งวด'
        else: return '7+ งวด'

    events_df = events_df.copy()
    events_df['gap_group'] = events_df['gap_length'].apply(gap_group)

    group_order = ['1 งวด', '2 งวด', '3 งวด', '4-6 งวด', '7+ งวด']
    valid = events_df.dropna(subset=['recovery_ratio'])

    print(f"\n  Total gap-return events: {len(events_df):,}")
    print(f"  Unique customers with gaps: {events_df['userNo'].nunique():,}")
    print(f"\n  {'Gap Length':<12} {'Events':>8} {'Avg Pre':>10} {'Avg Post':>10} "
          f"{'Recovery%':>10} {'>=80%':>8} {'<50%':>8}")
    print("  " + "-" * 70)

    group_stats = {}
    for g in group_order:
        gdf = valid[valid['gap_group'] == g]
        if len(gdf) == 0:
            continue
        avg_pre = gdf['pre_avg'].mean()
        avg_post = gdf['post_avg'].mean()
        avg_recovery = gdf['recovery_ratio'].mean() * 100
        pct_80 = (gdf['recovery_ratio'] >= 0.8).mean() * 100
        pct_50 = (gdf['recovery_ratio'] < 0.5).mean() * 100
        group_stats[g] = {
            'count': len(gdf), 'avg_pre': avg_pre, 'avg_post': avg_post,
            'avg_recovery': avg_recovery, 'pct_80': pct_80, 'pct_50': pct_50
        }
        print(f"  {g:<12} {len(gdf):>8,} {avg_pre:>10.1f} {avg_post:>10.1f} "
              f"{avg_recovery:>9.1f}% {pct_80:>7.1f}% {pct_50:>7.1f}%")

    # Key finding
    if group_stats:
        short = group_stats.get('1 งวด', {})
        long_ = group_stats.get('7+ งวด', {})
        if short and long_:
            print(f"\n  KEY FINDING:")
            print(f"  หายไป 1 งวด: กลับมาซื้อเฉลี่ย {short['avg_recovery']:.0f}% ของเดิม "
                  f"({short['pct_80']:.0f}% กลับมา >= 80%)")
            print(f"  หายไป 7+ งวด: กลับมาซื้อเฉลี่ย {long_['avg_recovery']:.0f}% ของเดิม "
                  f"({long_['pct_80']:.0f}% กลับมา >= 80%)")

    # ─── Find 3 example customers ─────────────────────────────
    data = df[item_cols].values

    # Example 1: Short gap (1 period), came back at same level
    ex1_candidates = valid[
        (valid['gap_length'] == 1) &
        (valid['recovery_ratio'] >= 0.8) &
        (valid['recovery_ratio'] <= 1.3) &
        (valid['pre_avg'] >= 3) &
        (valid['active_after'] >= 3)
    ].sort_values('active_after', ascending=False)

    # Example 2: Medium gap (3 periods), came back buying less
    ex2_candidates = valid[
        (valid['gap_length'] == 3) &
        (valid['recovery_ratio'] < 0.6) &
        (valid['pre_avg'] >= 4) &
        (valid['active_after'] >= 2)
    ].sort_values('pre_avg', ascending=False)

    # Example 3: Long gap (6+ periods), came back with high purchase
    ex3_candidates = valid[
        (valid['gap_length'] >= 6) &
        (valid['recovery_ratio'] >= 0.9) &
        (valid['pre_avg'] >= 3) &
        (valid['active_after'] >= 2)
    ].sort_values('gap_length', ascending=False)

    examples = []
    labels = []

    if len(ex1_candidates) > 0:
        ex = ex1_candidates.iloc[0]
        examples.append(ex)
        labels.append(
            f"หายไป {int(ex['gap_length'])} งวด → กลับมาซื้อเท่าเดิม "
            f"(Recovery {ex['recovery_ratio']*100:.0f}%)"
        )
        print(f"\n  Example 1: {ex['userNo']} — หายไป 1 งวด, กลับมาซื้อเท่าเดิม")
        print(f"    Pre-gap avg: {ex['pre_avg']:.1f}, Post-return avg: {ex['post_avg']:.1f}, "
              f"Recovery: {ex['recovery_ratio']*100:.0f}%")
    else:
        # Fallback: relax criteria
        ex1_fb = valid[
            (valid['gap_length'] == 1) & (valid['recovery_ratio'] >= 0.7) & (valid['pre_avg'] >= 2)
        ].sort_values('active_after', ascending=False)
        if len(ex1_fb) > 0:
            ex = ex1_fb.iloc[0]
            examples.append(ex)
            labels.append(f"หายไป 1 งวด → Recovery {ex['recovery_ratio']*100:.0f}%")
            print(f"\n  Example 1 (fallback): {ex['userNo']}")

    if len(ex2_candidates) > 0:
        ex = ex2_candidates.iloc[0]
        examples.append(ex)
        labels.append(
            f"หายไป {int(ex['gap_length'])} งวด → กลับมาซื้อน้อยลง "
            f"(Recovery {ex['recovery_ratio']*100:.0f}%)"
        )
        print(f"\n  Example 2: {ex['userNo']} — หายไป 3 งวด, กลับมาซื้อน้อยลง")
        print(f"    Pre-gap avg: {ex['pre_avg']:.1f}, Post-return avg: {ex['post_avg']:.1f}, "
              f"Recovery: {ex['recovery_ratio']*100:.0f}%")
    else:
        ex2_fb = valid[
            (valid['gap_length'].between(2, 4)) & (valid['recovery_ratio'] < 0.7)
        ].sort_values('pre_avg', ascending=False)
        if len(ex2_fb) > 0:
            ex = ex2_fb.iloc[0]
            examples.append(ex)
            labels.append(f"หายไป {int(ex['gap_length'])} งวด → Recovery {ex['recovery_ratio']*100:.0f}%")
            print(f"\n  Example 2 (fallback): {ex['userNo']}")

    if len(ex3_candidates) > 0:
        ex = ex3_candidates.iloc[0]
        examples.append(ex)
        labels.append(
            f"หายไป {int(ex['gap_length'])} งวด → กลับมาซื้อเยอะ! "
            f"(Recovery {ex['recovery_ratio']*100:.0f}%)"
        )
        print(f"\n  Example 3: {ex['userNo']} — หายไป {int(ex['gap_length'])} งวด, กลับมาซื้อเยอะ!")
        print(f"    Pre-gap avg: {ex['pre_avg']:.1f}, Post-return avg: {ex['post_avg']:.1f}, "
              f"Recovery: {ex['recovery_ratio']*100:.0f}%")
    else:
        ex3_fb = valid[
            (valid['gap_length'] >= 5) & (valid['recovery_ratio'] >= 0.7)
        ].sort_values('gap_length', ascending=False)
        if len(ex3_fb) > 0:
            ex = ex3_fb.iloc[0]
            examples.append(ex)
            labels.append(f"หายไป {int(ex['gap_length'])} งวด → Recovery {ex['recovery_ratio']*100:.0f}%")
            print(f"\n  Example 3 (fallback): {ex['userNo']}")

    if len(examples) < 3:
        print("  WARNING: Could not find 3 distinct examples. Filling with available.")
        # Fill remaining from any events
        remaining = valid.sort_values('gap_length', ascending=False)
        used = {e['userNo'] for e in examples}
        for _, row_e in remaining.iterrows():
            if row_e['userNo'] not in used:
                examples.append(row_e)
                labels.append(f"Gap {int(row_e['gap_length'])} งวด → Recovery {row_e['recovery_ratio']*100:.0f}%")
                used.add(row_e['userNo'])
            if len(examples) >= 3:
                break

    # ─── Summary bar chart (top area) + 3 customer charts ────
    fig = plt.figure(figsize=(20, 18))
    fig.suptitle('Insight 19: หายไปกี่งวดแล้วกลับมา → ยอดซื้อเท่าเดิมมั้ย',
                 fontsize=18, fontweight='bold', y=0.98)

    # Summary subplot at top
    gs = fig.add_gridspec(4, 1, hspace=0.45, top=0.94, bottom=0.05,
                          height_ratios=[1.2, 1, 1, 1])

    ax_summary = fig.add_subplot(gs[0])

    # Summary bar: before vs after by gap group
    groups_with_data = [g for g in group_order if g in group_stats]
    x_pos = np.arange(len(groups_with_data))
    width = 0.3
    pre_vals = [group_stats[g]['avg_pre'] for g in groups_with_data]
    post_vals = [group_stats[g]['avg_post'] for g in groups_with_data]
    recovery_vals = [group_stats[g]['avg_recovery'] for g in groups_with_data]
    counts = [group_stats[g]['count'] for g in groups_with_data]

    bars1 = ax_summary.bar(x_pos - width/2, pre_vals, width, color='#3B82F6',
                           label='Avg ก่อนหาย', edgecolor='white')
    bars2 = ax_summary.bar(x_pos + width/2, post_vals, width, color=COLOR_BUY,
                           label='Avg หลังกลับมา', edgecolor='white')

    # Add recovery % and count labels
    for j, g in enumerate(groups_with_data):
        ax_summary.text(j, max(pre_vals[j], post_vals[j]) + 0.3,
                        f'{recovery_vals[j]:.0f}%\n(n={counts[j]:,})',
                        ha='center', fontsize=9, fontweight='bold')

    ax_summary.set_xticks(x_pos)
    ax_summary.set_xticklabels(groups_with_data, fontsize=11)
    ax_summary.set_ylabel('Avg Tickets/Period', fontsize=11)
    ax_summary.set_title('เปรียบเทียบยอดซื้อก่อน vs หลังกลับมา (แต่ละกลุ่ม Gap Length)',
                         fontsize=13, fontweight='bold')
    ax_summary.legend(fontsize=10)
    ax_summary.spines['top'].set_visible(False)
    ax_summary.spines['right'].set_visible(False)

    # 3 customer bar charts
    for idx, (ex, label) in enumerate(zip(examples[:3], labels[:3])):
        ax = fig.add_subplot(gs[idx + 1])
        row_idx = int(ex['row_idx'])
        row_data = data[row_idx].copy()
        return_idx = int(ex['return_idx'])

        user_info = (f"{ex['userNo']} | {ex['gender']} อายุ {int(ex['age'])} ปี | "
                     f"{ex['province']}")
        subtitle = (f"Pre-gap avg: {ex['pre_avg']:.1f} ใบ | "
                    f"Post-return avg: {ex['post_avg']:.1f} ใบ | "
                    f"★ = กลับมาซื้อ (Period {return_idx+1})")

        plot_customer_bar(ax, row_data, item_cols, return_idx,
                          f"Ex {idx+1}: {label}", subtitle=user_info)
        # Add extra subtitle below title
        ax.text(0.5, -0.12, subtitle, transform=ax.transAxes,
                ha='center', fontsize=9, color='#6B7280')

    plt.savefig(os.path.join(CHART_DIR, 'insight_19_gap_recovery.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Chart saved: charts/insight_19_gap_recovery.png")

    return events_df


# ═══════════════════════════════════════════════════════════════
# INSIGHT 20
# ═══════════════════════════════════════════════════════════════
def insight_20(df, item_cols, events_df):
    """Return purchase size predicts retention."""
    print("\n" + "=" * 70)
    print("INSIGHT 20: กลับมาครั้งแรกซื้อกี่ใบ → ทำนายได้ว่าจะอยู่ต่อมั้ย")
    print("=" * 70)

    # Filter: gap >= 2 periods, and we can measure retention after return
    valid = events_df[
        (events_df['gap_length'] >= 2) &
        (events_df['stayed_3_plus'].notna())
    ].copy()

    print(f"\n  Gap >= 2 periods with measurable retention: {len(valid):,} events")
    print(f"  Unique customers: {valid['userNo'].nunique():,}")

    # Group by return purchase size
    def size_group(v):
        if v <= 2: return '1-2 ใบ'
        elif v <= 5: return '3-5 ใบ'
        elif v <= 10: return '6-10 ใบ'
        elif v <= 20: return '11-20 ใบ'
        else: return '20+ ใบ'

    valid['size_group'] = valid['return_amount'].apply(size_group)
    size_order = ['1-2 ใบ', '3-5 ใบ', '6-10 ใบ', '11-20 ใบ', '20+ ใบ']

    print(f"\n  {'Return Size':<12} {'Events':>8} {'Stayed 3+':>12} {'Churned<2':>12} "
          f"{'Avg Return':>12}")
    print("  " + "-" * 60)

    group_stats = {}
    for sg in size_order:
        sgdf = valid[valid['size_group'] == sg]
        if len(sgdf) == 0:
            continue
        stayed = sgdf['stayed_3_plus'].mean() * 100
        churned_mask = sgdf['churned_within_2'].notna()
        if churned_mask.sum() > 0:
            churned = sgdf.loc[churned_mask, 'churned_within_2'].mean() * 100
        else:
            churned = np.nan
        avg_ret = sgdf['return_amount'].mean()

        group_stats[sg] = {
            'count': len(sgdf), 'stayed_pct': stayed,
            'churned_pct': churned, 'avg_return': avg_ret
        }
        churned_str = f"{churned:.1f}%" if not np.isnan(churned) else "N/A"
        print(f"  {sg:<12} {len(sgdf):>8,} {stayed:>11.1f}% {churned_str:>12} "
              f"{avg_ret:>12.1f}")

    # Key finding
    small = group_stats.get('1-2 ใบ', {})
    big = group_stats.get('20+ ใบ', {})
    if small and big:
        print(f"\n  KEY FINDING:")
        print(f"  กลับมาซื้อ 1-2 ใบ: อยู่ต่อ 3+ งวด = {small['stayed_pct']:.0f}%, "
              f"Churn ภายใน 2 งวด = {small['churned_pct']:.0f}%")
        print(f"  กลับมาซื้อ 20+ ใบ: อยู่ต่อ 3+ งวด = {big['stayed_pct']:.0f}%, "
              f"Churn ภายใน 2 งวด = {big['churned_pct']:.0f}%")
        print(f"\n  ACTIONABLE: ถ้าลูกค้ากลับมาซื้อ 1-2 ใบ → ต้อง nurture ทันที!")
        print(f"  ถ้ากลับมาซื้อเยอะ → สัญญาณดี ไม่ต้องใช้งบ marketing มาก")

    # ─── Find 3 example customers ─────────────────────────────
    data = df[item_cols].values

    # Example 1: Returned with 1 ticket, churned again quickly
    ex1_candidates = valid[
        (valid['return_amount'] <= 2) &
        (valid['churned_within_2'] == True) &
        (valid['gap_length'] >= 2) &
        (valid['pre_avg'] >= 3)
    ].sort_values('pre_avg', ascending=False)

    # Example 2: Returned with 10+ tickets, stayed 10+ periods
    ex2_candidates = valid[
        (valid['return_amount'] >= 10) &
        (valid['stayed_3_plus'] == True) &
        (valid['active_after'] >= 5)
    ].sort_values('active_after', ascending=False)

    # Example 3: Returned with moderate amount, stayed but declining
    ex3_candidates = valid[
        (valid['return_amount'].between(3, 8)) &
        (valid['stayed_3_plus'] == True) &
        (valid['recovery_ratio'] < 0.8) &
        (valid['active_after'] >= 3)
    ].sort_values('active_after', ascending=False)

    examples = []
    labels = []

    if len(ex1_candidates) > 0:
        ex = ex1_candidates.iloc[0]
        examples.append(ex)
        labels.append(f"กลับมาซื้อ {int(ex['return_amount'])} ใบ → Churn อีกใน 2 งวด")
        print(f"\n  Example 1: {ex['userNo']} — กลับมาซื้อ {int(ex['return_amount'])} ใบ, Churn อีกรอบ")
        print(f"    Gap: {int(ex['gap_length'])} งวด, Pre-gap avg: {ex['pre_avg']:.1f}")
    else:
        ex1_fb = valid[
            (valid['return_amount'] <= 3) & (valid['churned_within_2'] == True)
        ].sort_values('gap_length', ascending=False)
        if len(ex1_fb) > 0:
            ex = ex1_fb.iloc[0]
            examples.append(ex)
            labels.append(f"กลับมาซื้อ {int(ex['return_amount'])} ใบ → Churn อีก")
            print(f"\n  Example 1 (fallback): {ex['userNo']}")

    if len(ex2_candidates) > 0:
        ex = ex2_candidates.iloc[0]
        examples.append(ex)
        labels.append(f"กลับมาซื้อ {int(ex['return_amount'])} ใบ → อยู่ต่อ {int(ex['active_after'])} งวด!")
        print(f"\n  Example 2: {ex['userNo']} — กลับมาซื้อ {int(ex['return_amount'])} ใบ, อยู่ต่อยาว!")
        print(f"    Gap: {int(ex['gap_length'])} งวด, Active after: {int(ex['active_after'])} งวด")
    else:
        ex2_fb = valid[
            (valid['return_amount'] >= 5) & (valid['stayed_3_plus'] == True)
        ].sort_values('active_after', ascending=False)
        if len(ex2_fb) > 0:
            ex = ex2_fb.iloc[0]
            examples.append(ex)
            labels.append(f"กลับมาซื้อ {int(ex['return_amount'])} ใบ → อยู่ต่อ!")
            print(f"\n  Example 2 (fallback): {ex['userNo']}")

    if len(ex3_candidates) > 0:
        ex = ex3_candidates.iloc[0]
        examples.append(ex)
        labels.append(
            f"กลับมาซื้อ {int(ex['return_amount'])} ใบ → อยู่ต่อแต่ยอดลด "
            f"(Recovery {ex['recovery_ratio']*100:.0f}%)"
        )
        print(f"\n  Example 3: {ex['userNo']} — กลับมาปานกลาง, อยู่แต่ซื้อน้อยลง")
        print(f"    Return: {int(ex['return_amount'])} ใบ, Recovery: {ex['recovery_ratio']*100:.0f}%")
    else:
        ex3_fb = valid[
            (valid['return_amount'].between(3, 10)) & (valid['stayed_3_plus'] == True)
        ].sort_values('active_after', ascending=False)
        if len(ex3_fb) > 0:
            ex = ex3_fb.iloc[0]
            examples.append(ex)
            labels.append(f"กลับมาซื้อ {int(ex['return_amount'])} ใบ → อยู่ต่อ")
            print(f"\n  Example 3 (fallback): {ex['userNo']}")

    # Fill if needed
    if len(examples) < 3:
        used = {e['userNo'] for e in examples}
        remaining = valid.sort_values('return_amount', ascending=False)
        for _, row_e in remaining.iterrows():
            if row_e['userNo'] not in used:
                examples.append(row_e)
                labels.append(f"กลับมาซื้อ {int(row_e['return_amount'])} ใบ")
                used.add(row_e['userNo'])
            if len(examples) >= 3:
                break

    # ─── Chart ────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 18))
    fig.suptitle('Insight 20: กลับมาครั้งแรกซื้อกี่ใบ → ทำนายได้ว่าจะอยู่ต่อมั้ย',
                 fontsize=18, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(4, 1, hspace=0.45, top=0.94, bottom=0.05,
                          height_ratios=[1.2, 1, 1, 1])

    # Summary: stayed vs churned by return size group
    ax_summary = fig.add_subplot(gs[0])

    groups_with_data = [g for g in size_order if g in group_stats]
    x_pos = np.arange(len(groups_with_data))
    width = 0.3

    stayed_vals = [group_stats[g]['stayed_pct'] for g in groups_with_data]
    churned_vals = [group_stats[g]['churned_pct'] if not np.isnan(group_stats[g]['churned_pct'])
                    else 0 for g in groups_with_data]
    counts = [group_stats[g]['count'] for g in groups_with_data]

    bars1 = ax_summary.bar(x_pos - width/2, stayed_vals, width, color='#10B981',
                           label='อยู่ต่อ 3+ งวด', edgecolor='white')
    bars2 = ax_summary.bar(x_pos + width/2, churned_vals, width, color='#EF4444',
                           label='Churn ภายใน 2 งวด', edgecolor='white')

    for j, g in enumerate(groups_with_data):
        ax_summary.text(j, max(stayed_vals[j], churned_vals[j]) + 1.5,
                        f'n={counts[j]:,}', ha='center', fontsize=9, fontweight='bold')

    ax_summary.set_xticks(x_pos)
    ax_summary.set_xticklabels(groups_with_data, fontsize=11)
    ax_summary.set_ylabel('%', fontsize=11)
    ax_summary.set_title('ยอดซื้อครั้งแรกที่กลับมา vs อัตราการอยู่ต่อ/Churn ซ้ำ',
                         fontsize=13, fontweight='bold')
    ax_summary.legend(fontsize=10)
    ax_summary.spines['top'].set_visible(False)
    ax_summary.spines['right'].set_visible(False)

    # 3 customer bar charts
    for idx, (ex, label) in enumerate(zip(examples[:3], labels[:3])):
        ax = fig.add_subplot(gs[idx + 1])
        row_idx = int(ex['row_idx'])
        row_data = data[row_idx].copy()
        return_idx = int(ex['return_idx'])

        user_info = (f"{ex['userNo']} | {ex['gender']} อายุ {int(ex['age'])} ปี | "
                     f"{ex['province']}")
        subtitle = (f"Gap: {int(ex['gap_length'])} งวด | Return: {int(ex['return_amount'])} ใบ | "
                    f"★ = กลับมาซื้อครั้งแรก")

        plot_customer_bar(ax, row_data, item_cols, return_idx,
                          f"Ex {idx+1}: {label}", subtitle=user_info)
        ax.text(0.5, -0.12, subtitle, transform=ax.transAxes,
                ha='center', fontsize=9, color='#6B7280')

    plt.savefig(os.path.join(CHART_DIR, 'insight_20_return_size.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Chart saved: charts/insight_20_return_size.png")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    df, item_cols = load_data()

    print("\nFinding gap-return events (this may take a minute)...", flush=True)
    events_df = find_gaps_and_returns(df, item_cols)
    print(f"  Found {len(events_df):,} gap-return events across "
          f"{events_df['userNo'].nunique():,} customers")

    insight_19(df, item_cols, events_df)
    insight_20(df, item_cols, events_df)

    print("\n" + "=" * 70)
    print("DONE — Insights 19-20 complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
