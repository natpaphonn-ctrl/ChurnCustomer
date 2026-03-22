#!/usr/bin/env python3
"""
Winning Effect — Full 30 Period Analysis
=========================================
Analyzes the effect of winning lottery prizes on customer purchasing behavior
across ALL 30 periods of prize data and 66 periods of purchase history.

Creates:
  - 8 charts: charts/winning_effect_{TYPE}.png (one per prize type group)
  - 1 summary: charts/winning_effect_all_types_summary.png
  - 1 CSV:    data/predictions/winning_effect_customers_full.csv
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Font config — Thai support
# ---------------------------------------------------------------------------
try:
    import matplotlib.font_manager as fm
    thai_fonts = [f.name for f in fm.fontManager.ttflist
                  if 'sarabun' in f.name.lower()]
    if thai_fonts:
        THAI_FONT = 'Sarabun'
    else:
        THAI_FONT = 'Tahoma'
    plt.rcParams['font.family'] = THAI_FONT
except Exception:
    THAI_FONT = 'Tahoma'
    plt.rcParams['font.family'] = THAI_FONT

plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRIZE_DIR = 'data/prize'
CHURN_DIR = 'data/churn'
CHART_DIR = 'charts'
PRED_DIR  = 'data/predictions'

os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# Prize type config — name_th, per-ticket prize amount, chart color
PRIZE_CONFIG = {
    'FIRST':        {'name_th': 'รางวัลที่ 1',       'amount': 6_000_000, 'color': '#DC2626'},
    'SECOND':       {'name_th': 'รางวัลที่ 2',       'amount': 200_000,   'color': '#EA580C'},
    'NEARBY':       {'name_th': 'รางวัลข้างเคียงรางวัลที่ 1', 'amount': 100_000, 'color': '#D97706'},
    'THIRD':        {'name_th': 'รางวัลที่ 3',       'amount': 80_000,    'color': '#CA8A04'},
    'FOURTH':       {'name_th': 'รางวัลที่ 4',       'amount': 40_000,    'color': '#65A30D'},
    'FIFTH':        {'name_th': 'รางวัลที่ 5',       'amount': 20_000,    'color': '#0D9488'},
    'FIRST3DIGITS': {'name_th': 'เลขหน้า 3 ตัว',    'amount': 4_000,     'color': '#2563EB'},
    'LAST3DIGITS':  {'name_th': 'เลขท้าย 3 ตัว',    'amount': 4_000,     'color': '#7C3AED'},
    'LAST2DIGITS':  {'name_th': 'เลขท้าย 2 ตัว',    'amount': 2_000,     'color': '#DB2777'},
}

# Chart groupings (combine some types)
CHART_GROUPS = [
    {'key': 'FIRST',        'types': ['FIRST'],              'label': 'FIRST'},
    {'key': 'SECOND',       'types': ['SECOND', 'NEARBY'],   'label': 'SECOND'},
    {'key': 'THIRD',        'types': ['THIRD'],              'label': 'THIRD'},
    {'key': 'FOURTH',       'types': ['FOURTH'],             'label': 'FOURTH'},
    {'key': 'FIFTH',        'types': ['FIFTH'],              'label': 'FIFTH'},
    {'key': 'FIRST3DIGITS', 'types': ['FIRST3DIGITS'],       'label': 'FIRST3DIGITS'},
    {'key': 'LAST3DIGITS',  'types': ['LAST3DIGITS'],        'label': 'LAST3DIGITS'},
    {'key': 'LAST2DIGITS',  'types': ['LAST2DIGITS'],        'label': 'LAST2DIGITS'},
]

# Bar colors
COLOR_BOUGHT   = '#F97316'   # orange — bought tickets
COLOR_ZERO     = '#E5E7EB'   # light gray — registered, didn't buy
COLOR_NAN      = '#D1D5DB'   # slightly darker gray — not yet registered
COLOR_WIN_MARK = '#DC2626'   # red star marker
COLOR_WIN_LINE = '#F59E0B'   # gold dashed line

# ---------------------------------------------------------------------------
# 1. Load ALL prize data
# ---------------------------------------------------------------------------
def load_all_prizes():
    """Load and merge all 30 prize files."""
    files = sorted(glob.glob(os.path.join(PRIZE_DIR, 'Prize_*.csv')))
    if not files:
        print("ERROR: No prize files found in", PRIZE_DIR)
        sys.exit(1)

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    prizes = pd.concat(dfs, ignore_index=True)
    prizes['rounddate'] = pd.to_datetime(prizes['rounddate'])
    print(f"  Loaded {len(files)} prize files — {len(prizes):,} winner records")
    print(f"  Prize types: {sorted(prizes['type'].unique())}")
    print(f"  Periods: {prizes['rounddate'].nunique()} unique draw dates")
    return prizes


# ---------------------------------------------------------------------------
# 2. Load churn data (purchase history)
# ---------------------------------------------------------------------------
def load_churn_history():
    """Load the latest (most complete) churn file for purchase history."""
    # Use the file with the most item columns = latest file
    target = os.path.join(CHURN_DIR, 'Churn_2026_0301_0316.csv')
    if not os.path.exists(target):
        # Fallback: find the largest file
        files = sorted(glob.glob(os.path.join(CHURN_DIR, 'Churn_*.csv')))
        target = files[-1]

    print(f"  Loading churn history from: {os.path.basename(target)}")
    churn = pd.read_csv(target, low_memory=False)

    item_cols = sorted([c for c in churn.columns if c.startswith('item')])
    print(f"  Customers: {len(churn):,}, Item columns (periods): {len(item_cols)}")

    return churn, item_cols


# ---------------------------------------------------------------------------
# 3. Map rounddate to item column name
# ---------------------------------------------------------------------------
def rounddate_to_item_col(rd):
    """Convert a rounddate (datetime or string) to item column name."""
    if isinstance(rd, str):
        rd = pd.to_datetime(rd)
    return f"item{rd.strftime('%Y_%m_%d')}"


# ---------------------------------------------------------------------------
# 4. Build winner profiles
# ---------------------------------------------------------------------------
def build_winner_profiles(prizes, churn, item_cols):
    """For each winner, extract purchase history and compute before/after stats."""

    # Create a set of valid item columns
    valid_items = set(item_cols)

    # Map each prize rounddate to an item column
    prizes['item_col'] = prizes['rounddate'].apply(rounddate_to_item_col)
    prizes['item_col_valid'] = prizes['item_col'].isin(valid_items)

    n_valid = prizes['item_col_valid'].sum()
    print(f"  Prize records with matching item column: {n_valid:,} / {len(prizes):,}")

    # Get unique winner userNos
    winner_users = prizes['userNo'].unique()
    print(f"  Unique winner userNos: {len(winner_users):,}")

    # Filter churn to only winners (memory optimization)
    churn_idx = churn.set_index('userNo')
    winners_in_churn = set(winner_users) & set(churn_idx.index)
    print(f"  Winners found in churn file: {len(winners_in_churn):,}")

    # Get item column order (chronological)
    item_col_order = item_cols  # already sorted
    item_col_to_idx = {c: i for i, c in enumerate(item_col_order)}

    # Build profiles
    profiles = []

    for _, row in prizes[prizes['userNo'].isin(winners_in_churn)].iterrows():
        user = row['userNo']
        prize_type = row['type']
        prize_total = row['total_prize']
        prize_count = row['count_prize']
        win_col = row['item_col']

        if win_col not in valid_items:
            continue

        if user not in churn_idx.index:
            continue

        user_data = churn_idx.loc[user]
        if isinstance(user_data, pd.DataFrame):
            user_data = user_data.iloc[0]

        win_idx = item_col_to_idx.get(win_col)
        if win_idx is None:
            continue

        # Extract purchase history
        history = user_data[item_col_order].values.astype(float)

        # Compute before/after (3 periods)
        before_start = max(0, win_idx - 3)
        before_vals = history[before_start:win_idx]
        before_vals = before_vals[~np.isnan(before_vals)]
        before_vals = before_vals[before_vals >= 0]

        after_start = win_idx + 1
        after_end = min(len(history), win_idx + 4)
        after_vals = history[after_start:after_end]
        after_vals = after_vals[~np.isnan(after_vals)]
        after_vals = after_vals[after_vals >= 0]

        before_avg = np.mean(before_vals) if len(before_vals) > 0 else 0
        after_avg = np.mean(after_vals) if len(after_vals) > 0 else 0

        if before_avg > 0:
            change_pct = ((after_avg - before_avg) / before_avg) * 100
        elif after_avg > 0:
            change_pct = 100.0
        else:
            change_pct = 0.0

        # Total lifetime tickets
        valid_hist = history[~np.isnan(history)]
        total_lifetime = np.sum(valid_hist[valid_hist > 0])

        # Activity stats
        last_3 = history[-3:]
        active_last_3 = np.sum((~np.isnan(last_3)) & (last_3 > 0))
        last_6 = history[-6:]
        active_last_6 = np.sum((~np.isnan(last_6)) & (last_6 > 0))
        tickets_last_3 = np.nansum(np.where(last_3 > 0, last_3, 0))

        # Before/after raw values for display
        before_raw = history[before_start:win_idx].tolist()
        after_raw = history[after_start:after_end].tolist()

        profiles.append({
            'userNo': user,
            'gender': user_data.get('gender', ''),
            'age': user_data.get('age', ''),
            'province': user_data.get('province', ''),
            'prize_type': prize_type,
            'prize_amount': prize_total,
            'prize_per_ticket': prize_total / max(prize_count, 1),
            'prize_count': prize_count,
            'prize_period': row['rounddate'],
            'win_col': win_col,
            'win_idx': win_idx,
            'history': history,
            'tickets_before_avg': round(before_avg, 1),
            'tickets_after_avg': round(after_avg, 1),
            'change_pct': round(change_pct, 1),
            'total_lifetime_tickets': int(total_lifetime),
            'active_last_3': int(active_last_3),
            'active_last_6': int(active_last_6),
            'tickets_last_3': int(tickets_last_3),
            'before_raw': before_raw,
            'after_raw': after_raw,
        })

    print(f"  Winner profiles built: {len(profiles):,}")
    return profiles, item_col_order


# ---------------------------------------------------------------------------
# 5. Select diverse customer examples
# ---------------------------------------------------------------------------
def select_examples(profiles_for_type, n=3):
    """Pick 3 diverse examples: increased, maintained, interesting."""
    if len(profiles_for_type) == 0:
        return []
    if len(profiles_for_type) <= n:
        return profiles_for_type

    df = pd.DataFrame(profiles_for_type)

    examples = []

    # 1. Best increase after winning (filter for meaningful before_avg)
    candidates = df[df['tickets_before_avg'] > 1].copy()
    if len(candidates) > 0:
        best_increase = candidates.nlargest(10, 'change_pct')
        # Pick one with reasonable history (not too extreme)
        for _, row in best_increase.iterrows():
            if row['change_pct'] > 20 and row['tickets_before_avg'] < 500:
                examples.append(row['userNo'])
                break
        if not examples:
            examples.append(best_increase.iloc[0]['userNo'])
    else:
        examples.append(df.iloc[0]['userNo'])

    # 2. Maintained level (change_pct closest to 0, with decent history)
    remaining = df[~df['userNo'].isin(examples)]
    if len(remaining) > 0:
        candidates = remaining[remaining['tickets_before_avg'] > 1].copy()
        if len(candidates) == 0:
            candidates = remaining.copy()
        candidates['abs_change'] = candidates['change_pct'].abs()
        maintained = candidates.nsmallest(5, 'abs_change')
        # Pick one with longest history
        best = maintained.iloc[0]
        for _, row in maintained.iterrows():
            hist = row['history']
            active_count = np.sum((~np.isnan(hist)) & (hist > 0))
            if active_count > np.sum((~np.isnan(best['history'])) & (best['history'] > 0)):
                best = row
        examples.append(best['userNo'])

    # 3. Interesting pattern — multi-winner or declining-then-recovery
    remaining = df[~df['userNo'].isin(examples)]
    if len(remaining) > 0:
        # Look for multi-period winners first
        user_win_counts = df.groupby('userNo').size()
        multi_winners = user_win_counts[user_win_counts > 1].index
        multi_in_remaining = remaining[remaining['userNo'].isin(multi_winners)]

        if len(multi_in_remaining) > 0:
            examples.append(multi_in_remaining.iloc[0]['userNo'])
        else:
            # Pick someone with high total lifetime
            interesting = remaining.nlargest(5, 'total_lifetime_tickets')
            examples.append(interesting.iloc[0]['userNo'])

    # Get the full profiles for selected examples
    selected = []
    for user in examples[:n]:
        matches = [p for p in profiles_for_type if p['userNo'] == user]
        if matches:
            selected.append(matches[0])

    return selected


# ---------------------------------------------------------------------------
# 6. Draw customer bar chart
# ---------------------------------------------------------------------------
def draw_customer_bar(ax, profile, item_cols, all_win_indices=None):
    """Draw a single customer's purchase history bar chart."""
    history = profile['history']
    win_idx = profile['win_idx']
    n_periods = len(history)

    # Determine bar colors
    bar_colors = []
    bar_heights = []

    for i, val in enumerate(history):
        if np.isnan(val):
            bar_heights.append(0.3)   # small bar for NaN (not registered)
            bar_colors.append(COLOR_NAN)
        elif val == 0:
            bar_heights.append(0.3)   # small bar for 0 (didn't buy)
            bar_colors.append(COLOR_ZERO)
        else:
            bar_heights.append(val)
            bar_colors.append(COLOR_BOUGHT)

    bars = ax.bar(range(n_periods), bar_heights, color=bar_colors, width=0.7,
                  edgecolor='white', linewidth=0.3)

    # Mark ALL winning periods for this user
    if all_win_indices is None:
        all_win_indices = [win_idx]

    max_height = max(bar_heights) if bar_heights else 1

    for widx in all_win_indices:
        if 0 <= widx < n_periods:
            marker_y = bar_heights[widx] + max_height * 0.08
            ax.plot(widx, marker_y, marker='*', color=COLOR_WIN_MARK,
                    markersize=14, zorder=5)
            ax.text(widx, marker_y + max_height * 0.06, 'ถูกรางวัล',
                    ha='center', va='bottom', fontsize=7, color=COLOR_WIN_MARK,
                    fontweight='bold')
            ax.axvline(x=widx, color=COLOR_WIN_LINE, linestyle='--',
                       alpha=0.6, linewidth=1, zorder=1)

    # X-axis labels (period dates in MM/DD format)
    period_labels = []
    for col in item_cols:
        # item2025_01_02 -> 01/02
        parts = col.replace('item', '').split('_')
        if len(parts) == 3:
            period_labels.append(f"{parts[1]}/{parts[2]}")
        else:
            period_labels.append(col)

    # Show every Nth label to avoid crowding
    step = max(1, n_periods // 20)
    tick_positions = list(range(0, n_periods, step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([period_labels[i] for i in tick_positions],
                       rotation=45, ha='right', fontsize=6)

    ax.set_xlim(-0.5, n_periods - 0.5)
    ax.set_ylabel('ใบ', fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Bottom stats
    p = profile
    trend = "▲ เพิ่มขึ้น" if p['change_pct'] > 10 else ("▼ ลดลง" if p['change_pct'] < -10 else "— คงที่")
    stats_text = (f"Active 3 งวดล่าสุด: {p['active_last_3']}/3 | "
                  f"Active 6 งวดล่าสุด: {p['active_last_6']}/6 | "
                  f"ใบที่ซื้อ 3 งวดล่าสุด: {p['tickets_last_3']:,} | "
                  f"แนวโน้ม: {trend}")
    ax.text(0.5, -0.18, stats_text, transform=ax.transAxes,
            ha='center', va='top', fontsize=7, color='#6B7280')


# ---------------------------------------------------------------------------
# 7. Create per-type chart
# ---------------------------------------------------------------------------
def create_type_chart(group_key, type_list, profiles, item_cols, all_profiles):
    """Create a chart for one prize type group with 3 customer examples."""

    # Filter profiles for this type group
    type_profiles = [p for p in profiles if p['prize_type'] in type_list]

    if not type_profiles:
        print(f"    No profiles for {group_key} — skipping")
        return None

    # Stats for header
    n_winners = len(type_profiles)
    unique_winners = len(set(p['userNo'] for p in type_profiles))

    # Average change
    changes = [p['change_pct'] for p in type_profiles
                if p['tickets_before_avg'] > 0 and not np.isinf(p['change_pct'])]
    avg_change = np.mean(changes) if changes else 0
    median_change = np.median(changes) if changes else 0

    # Effect duration: how many periods after winning do they stay active?
    duration_vals = []
    for p in type_profiles:
        widx = p['win_idx']
        hist = p['history']
        consecutive = 0
        for i in range(widx + 1, len(hist)):
            if not np.isnan(hist[i]) and hist[i] > 0:
                consecutive += 1
            else:
                break
        duration_vals.append(consecutive)
    avg_duration = np.mean(duration_vals) if duration_vals else 0

    # Select 3 examples
    examples = select_examples(type_profiles, n=3)
    n_examples = len(examples)

    if n_examples == 0:
        print(f"    No valid examples for {group_key} — skipping")
        return None

    # Prize info
    primary_type = type_list[0]
    cfg = PRIZE_CONFIG[primary_type]

    # Create figure
    fig_height = 4 + n_examples * 3.5
    fig = plt.figure(figsize=(20, fig_height))

    # Use gridspec for layout
    gs = fig.add_gridspec(n_examples + 1, 5, height_ratios=[1.2] + [3] * n_examples,
                          hspace=0.45, wspace=0.3,
                          left=0.05, right=0.97, top=0.95, bottom=0.05)

    # ---- Header section ----
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')

    # Combine type names for grouped types
    type_names = []
    total_amount_str = ""
    for t in type_list:
        c = PRIZE_CONFIG[t]
        type_names.append(c['name_th'])
        total_amount_str = f"{c['amount']:,} บาท/ใบ"

    header_name = " / ".join(type_names)
    if len(type_list) > 1:
        amounts = [f"{PRIZE_CONFIG[t]['amount']:,}" for t in type_list]
        total_amount_str = " / ".join(amounts) + " บาท/ใบ"

    change_direction = "เพิ่มขึ้น" if avg_change > 0 else "ลดลง"

    ax_header.text(0.5, 0.85, f"ผลของการถูกรางวัล — {header_name}",
                   ha='center', va='center', fontsize=18, fontweight='bold',
                   color='#1F2937', transform=ax_header.transAxes)

    summary_text = (f"รางวัล {total_amount_str}  |  "
                    f"ผู้ถูกรางวัล {unique_winners:,} ราย (30 งวด)  |  "
                    f"เฉลี่ย{change_direction} {abs(avg_change):.1f}%  |  "
                    f"ซื้อต่อเนื่องหลังถูก ~{avg_duration:.1f} งวด")
    ax_header.text(0.5, 0.35, summary_text,
                   ha='center', va='center', fontsize=12, color='#4B5563',
                   transform=ax_header.transAxes)

    # Horizontal line
    ax_header.axhline(y=0.05, xmin=0.1, xmax=0.9, color='#D1D5DB', linewidth=1)

    # ---- Customer rows ----
    example_labels = ['ตัวอย่างที่ 1 — ซื้อเพิ่มหลังถูกรางวัล',
                      'ตัวอย่างที่ 2 — ซื้อคงที่',
                      'ตัวอย่างที่ 3 — รูปแบบน่าสนใจ']

    for i, ex in enumerate(examples):
        row = i + 1

        # Left: Customer info
        ax_info = fig.add_subplot(gs[row, 0])
        ax_info.axis('off')

        gender_th = 'ชาย' if str(ex.get('gender', '')).lower() in ['m', 'male', 'ชาย'] else (
            'หญิง' if str(ex.get('gender', '')).lower() in ['f', 'female', 'หญิง'] else str(ex.get('gender', '')))

        prize_date = pd.to_datetime(ex['prize_period']).strftime('%d/%m/%Y') if not isinstance(ex['prize_period'], str) else ex['prize_period']

        info_lines = [
            f"{example_labels[min(i, len(example_labels)-1)]}",
            "",
            f"{ex['userNo']}",
            f"{gender_th} / อายุ {ex.get('age', 'N/A')} / {ex.get('province', 'N/A')}",
            "",
            f"ถูกรางวัล: {prize_date}",
            f"เงินรางวัล: {int(ex['prize_amount']):,} บาท",
            f"({int(ex['prize_count'])} ใบ)",
            "",
            f"ก่อนถูก: {ex['tickets_before_avg']:.1f} ใบ/งวด",
            f"หลังถูก: {ex['tickets_after_avg']:.1f} ใบ/งวด",
            f"เปลี่ยนแปลง: {ex['change_pct']:+.1f}%",
            "",
            f"รวมตลอด: {ex['total_lifetime_tickets']:,} ใบ",
        ]

        for j, line in enumerate(info_lines):
            y = 0.95 - j * 0.065
            fs = 9 if j == 0 else 8
            fw = 'bold' if j in [0, 2] else 'normal'
            color = '#1F2937' if j in [0, 2] else '#4B5563'
            if 'เปลี่ยนแปลง' in line:
                color = '#16A34A' if ex['change_pct'] > 0 else '#DC2626' if ex['change_pct'] < 0 else '#4B5563'
            ax_info.text(0.05, y, line, ha='left', va='top',
                        fontsize=fs, fontweight=fw, color=color,
                        transform=ax_info.transAxes)

        # Right: Bar chart (spans columns 1-4)
        ax_bar = fig.add_subplot(gs[row, 1:])

        # Get ALL winning indices for this user
        user_wins = [p['win_idx'] for p in type_profiles if p['userNo'] == ex['userNo']]
        # Also check other prize types
        all_user_wins = [p['win_idx'] for p in all_profiles if p['userNo'] == ex['userNo']]

        draw_customer_bar(ax_bar, ex, item_cols, all_win_indices=all_user_wins)

        # Title for the bar chart
        ax_bar.set_title(f"{ex['userNo']} — {cfg['name_th']}",
                        fontsize=10, fontweight='bold', color='#374151', pad=8)

    # Legend at bottom
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_BOUGHT, edgecolor='white', label='ซื้อ (>0 ใบ)'),
        mpatches.Patch(facecolor=COLOR_ZERO, edgecolor='white', label='ไม่ซื้อ (0 ใบ)'),
        mpatches.Patch(facecolor=COLOR_NAN, edgecolor='white', label='ยังไม่ลงทะเบียน'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COLOR_WIN_MARK,
               markersize=12, label='ถูกรางวัล'),
        Line2D([0], [0], color=COLOR_WIN_LINE, linestyle='--', label='งวดที่ถูกรางวัล'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))

    # Save
    filepath = os.path.join(CHART_DIR, f'winning_effect_{group_key}.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {filepath}")

    return {
        'group_key': group_key,
        'types': type_list,
        'n_winners': n_winners,
        'unique_winners': unique_winners,
        'avg_change': avg_change,
        'median_change': median_change,
        'avg_duration': avg_duration,
        'examples': examples,
    }


# ---------------------------------------------------------------------------
# 8. Create summary chart
# ---------------------------------------------------------------------------
def create_summary_chart(group_results, profiles, item_cols):
    """Create an overview chart with one representative per prize type."""

    valid_results = [r for r in group_results if r is not None]
    n_groups = len(valid_results)

    if n_groups == 0:
        print("  No valid results for summary chart")
        return

    fig_height = 3 + n_groups * 3.2
    fig = plt.figure(figsize=(20, fig_height))

    gs = fig.add_gridspec(n_groups + 1, 5,
                          height_ratios=[1.5] + [3] * n_groups,
                          hspace=0.45, wspace=0.25,
                          left=0.05, right=0.97, top=0.96, bottom=0.04)

    # Header
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    ax_header.text(0.5, 0.7, 'ผลของการถูกรางวัลต่อพฤติกรรมการซื้อ — สรุปทุกประเภทรางวัล',
                   ha='center', va='center', fontsize=18, fontweight='bold',
                   color='#1F2937', transform=ax_header.transAxes)

    total_all = sum(r['unique_winners'] for r in valid_results)
    ax_header.text(0.5, 0.25,
                   f'ข้อมูล 30 งวด | ผู้ถูกรางวัลทั้งหมด {total_all:,} ราย | 66 งวดประวัติการซื้อ',
                   ha='center', va='center', fontsize=12, color='#6B7280',
                   transform=ax_header.transAxes)
    ax_header.axhline(y=0.0, xmin=0.05, xmax=0.95, color='#D1D5DB', linewidth=1)

    for i, result in enumerate(valid_results):
        row = i + 1
        primary_type = result['types'][0]
        cfg = PRIZE_CONFIG[primary_type]

        # Left info
        ax_info = fig.add_subplot(gs[row, 0])
        ax_info.axis('off')

        change_sym = "▲" if result['avg_change'] > 0 else "▼" if result['avg_change'] < 0 else "—"
        change_color = '#16A34A' if result['avg_change'] > 0 else '#DC2626' if result['avg_change'] < 0 else '#4B5563'

        info_lines = [
            (cfg['name_th'], 11, 'bold', '#1F2937'),
            (f"{cfg['amount']:,} บาท/ใบ", 9, 'normal', cfg['color']),
            ("", 8, 'normal', '#4B5563'),
            (f"ผู้ถูกรางวัล: {result['unique_winners']:,} ราย", 8, 'normal', '#4B5563'),
            (f"{change_sym} เฉลี่ย {abs(result['avg_change']):.1f}%", 9, 'bold', change_color),
            (f"ซื้อต่อ ~{result['avg_duration']:.1f} งวด", 8, 'normal', '#4B5563'),
        ]

        for j, (text, fs, fw, color) in enumerate(info_lines):
            y = 0.90 - j * 0.13
            ax_info.text(0.05, y, text, ha='left', va='top',
                        fontsize=fs, fontweight=fw, color=color,
                        transform=ax_info.transAxes)

        # Bar chart — pick the first example
        ax_bar = fig.add_subplot(gs[row, 1:])

        if result['examples']:
            ex = result['examples'][0]
            all_user_wins = [p['win_idx'] for p in profiles if p['userNo'] == ex['userNo']]
            draw_customer_bar(ax_bar, ex, item_cols, all_win_indices=all_user_wins)
            ax_bar.set_title(f"{ex['userNo']} — {cfg['name_th']} ({int(ex['prize_amount']):,} บาท)",
                            fontsize=9, fontweight='bold', color='#374151', pad=6)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_BOUGHT, edgecolor='white', label='ซื้อ (>0 ใบ)'),
        mpatches.Patch(facecolor=COLOR_ZERO, edgecolor='white', label='ไม่ซื้อ (0 ใบ)'),
        mpatches.Patch(facecolor=COLOR_NAN, edgecolor='white', label='ยังไม่ลงทะเบียน'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COLOR_WIN_MARK,
               markersize=12, label='ถูกรางวัล'),
        Line2D([0], [0], color=COLOR_WIN_LINE, linestyle='--', label='งวดที่ถูกรางวัล'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))

    filepath = os.path.join(CHART_DIR, 'winning_effect_all_types_summary.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved summary: {filepath}")


# ---------------------------------------------------------------------------
# 9. Export CSV
# ---------------------------------------------------------------------------
def export_csv(profiles):
    """Export all winner profiles to CSV."""
    rows = []
    for p in profiles:
        rows.append({
            'userNo': p['userNo'],
            'gender': p.get('gender', ''),
            'age': p.get('age', ''),
            'province': p.get('province', ''),
            'prize_type': p['prize_type'],
            'prize_amount': p['prize_amount'],
            'prize_per_ticket': p['prize_per_ticket'],
            'prize_count': p['prize_count'],
            'prize_period': p['prize_period'],
            'tickets_before_avg': p['tickets_before_avg'],
            'tickets_after_avg': p['tickets_after_avg'],
            'change_pct': p['change_pct'],
            'total_lifetime_tickets': p['total_lifetime_tickets'],
            'active_last_3': p['active_last_3'],
            'active_last_6': p['active_last_6'],
            'tickets_last_3': p['tickets_last_3'],
        })

    df = pd.DataFrame(rows)
    filepath = os.path.join(PRED_DIR, 'winning_effect_customers_full.csv')
    df.to_csv(filepath, index=False)
    print(f"  Saved CSV: {filepath} ({len(df):,} rows)")
    return df


# ---------------------------------------------------------------------------
# 10. Console report
# ---------------------------------------------------------------------------
def print_report(group_results, total_profiles):
    """Print detailed console report."""

    print("\n" + "=" * 60)
    print("Winning Effect — Full 30 Period Analysis")
    print("=" * 60)

    # Sort by prize amount descending
    sorted_results = sorted(
        [r for r in group_results if r is not None],
        key=lambda r: PRIZE_CONFIG[r['types'][0]]['amount'],
        reverse=True
    )

    for result in sorted_results:
        primary = result['types'][0]
        cfg = PRIZE_CONFIG[primary]

        type_names = " / ".join([PRIZE_CONFIG[t]['name_th'] for t in result['types']])
        print(f"\n{type_names} ({cfg['amount']:,} บาท) — {result['unique_winners']:,} ราย")

        change_dir = "เพิ่มขึ้น" if result['avg_change'] > 0 else "ลดลง"
        print(f"  ซื้อก่อนถูก → หลังถูก: เฉลี่ย{change_dir} {abs(result['avg_change']):.1f}% (มัธยฐาน {result['median_change']:+.1f}%)")
        print(f"  ซื้อต่อเนื่องหลังถูก: ~{result['avg_duration']:.1f} งวด")

        if result['examples']:
            print(f"  ตัวอย่าง:")
            for ex in result['examples']:
                gender_th = 'ชาย' if str(ex.get('gender', '')).lower() in ['m', 'male', 'ชาย'] else (
                    'หญิง' if str(ex.get('gender', '')).lower() in ['f', 'female', 'หญิง'] else str(ex.get('gender', '')))

                before_str = ",".join([f"{v:.0f}" if not np.isnan(v) else "-" for v in ex['before_raw']])
                after_str = ",".join([f"{v:.0f}" if not np.isnan(v) else "-" for v in ex['after_raw']])

                print(f"    {ex['userNo']} ({gender_th}/{ex.get('age','?')}/{ex.get('province','?')}) "
                      f"— ก่อน: [{before_str}] → หลัง: [{after_str}] ({ex['change_pct']:+.1f}%)")

    total_unique = len(set(p['userNo'] for p in total_profiles))
    print(f"\nTotal winners (30 periods): {total_unique:,} ราย ({len(total_profiles):,} รายการถูกรางวัล)")
    print(f"Chart files saved: {CHART_DIR}/winning_effect_*.png")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)

    # 1. Load prize data
    prizes = load_all_prizes()

    # 2. Load churn history
    churn, item_cols = load_churn_history()

    # 3. Build winner profiles
    print("\nBuilding winner profiles...")
    profiles, item_col_order = build_winner_profiles(prizes, churn, item_cols)

    # Free memory
    del churn
    import gc; gc.collect()

    if not profiles:
        print("ERROR: No winner profiles could be built. Check data alignment.")
        sys.exit(1)

    # 4. Generate per-type charts
    print("\nGenerating per-type charts...")
    group_results = []
    for group in CHART_GROUPS:
        print(f"  {group['key']}...")
        result = create_type_chart(
            group['key'], group['types'], profiles, item_col_order, profiles
        )
        group_results.append(result)

    # 5. Summary chart
    print("\nGenerating summary chart...")
    create_summary_chart(group_results, profiles, item_col_order)

    # 6. Export CSV
    print("\nExporting CSV...")
    export_csv(profiles)

    # 7. Console report
    print_report(group_results, profiles)


if __name__ == '__main__':
    main()
