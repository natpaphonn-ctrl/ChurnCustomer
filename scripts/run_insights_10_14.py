#!/usr/bin/env python3
"""
Insights 10-14: Purchase Behavior Insights with Customer Examples and Bar Charts
================================================================================
10. ซื้อติดกันกี่งวด (streak) → พอขาดจะกลับมามั้ย
11. ซื้อเพิ่ม 2x ผิดปกติ (spike) → รักษาระดับได้กี่งวด
12. ซื้อลดลง 50% (dip) → churn ภายในกี่งวด (early warning)
13. ซื้อสม่ำเสมอ vs ซื้อแบบ burst → ใครอยู่นานกว่า
14. ซื้อเพิ่มขึ้น 3 งวดติด → จะเพิ่มต่อหรือ plateau
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Thai font setup
plt.rcParams['font.family'] = 'Tahoma'
for ff in ['TH Sarabun New', 'Sarabun', 'Tahoma', 'Garuda', 'Loma', 'Norasi']:
    try:
        plt.rcParams['font.family'] = ff
        break
    except:
        continue

ROOT = Path(__file__).resolve().parent.parent
CHART_DIR = ROOT / 'charts'
CHART_DIR.mkdir(exist_ok=True)

# Colors
C_BOUGHT = '#F97316'   # orange
C_ZERO   = '#E5E7EB'   # gray
C_NAN    = '#1F2937'   # dark
C_STAR   = '#EF4444'   # red for star
C_GOLD   = '#F59E0B'   # gold dashed line


def load_data():
    """Load the latest/largest churn file."""
    churn_dir = ROOT / 'data' / 'churn'
    files = sorted(churn_dir.glob('Churn_*_*_*.csv'))
    # Pick the file with the most item columns
    best_file, best_count = None, 0
    for f in files:
        if 'Check' in f.name or 'old' in f.name:
            continue
        cols = pd.read_csv(f, nrows=0).columns
        item_count = sum(1 for c in cols if c.startswith('item'))
        if item_count > best_count:
            best_count = item_count
            best_file = f
    print(f"Loading {best_file.name} ({best_count} item columns)...")
    df = pd.read_csv(best_file)
    item_cols = sorted([c for c in df.columns if c.startswith('item')])
    print(f"  Rows: {len(df):,}, Item columns: {len(item_cols)}")
    return df, item_cols


def get_purchase_matrix(df, item_cols):
    """Return purchase matrix (NaN preserved) and period labels."""
    mat = df[item_cols].values.astype(float)
    labels = [c.replace('item', '') for c in item_cols]
    return mat, labels


def fmt_period(label):
    """Format period label for display: 2025_01_02 -> 25/01/02"""
    parts = label.split('_')
    if len(parts) == 3:
        return f"{parts[0][2:]}/{parts[1]}/{parts[2]}"
    return label


def plot_customer_bars(ax, values, labels, title, mark_idx=None, mark_label=None):
    """Plot a single customer bar chart."""
    n = len(values)
    colors = []
    for v in values:
        if np.isnan(v):
            colors.append(C_NAN)
        elif v == 0:
            colors.append(C_ZERO)
        else:
            colors.append(C_BOUGHT)

    bars = ax.bar(range(n), [v if not np.isnan(v) else 0.3 for v in values],
                  color=colors, edgecolor='white', linewidth=0.3, width=0.8)

    # For NaN bars, use hatching
    for i, v in enumerate(values):
        if np.isnan(v):
            bars[i].set_hatch('///')
            bars[i].set_edgecolor('#6B7280')

    # Mark key event
    if mark_idx is not None and 0 <= mark_idx < n:
        ax.axvline(x=mark_idx, color=C_GOLD, linestyle='--', linewidth=2, alpha=0.8)
        ymax = max([v for v in values if not np.isnan(v)] or [1])
        ax.text(mark_idx, ymax * 1.05, '★',
                ha='center', va='bottom', fontsize=18, color=C_STAR, fontweight='bold')
        if mark_label:
            ax.text(mark_idx, ymax * 1.25, mark_label,
                    ha='center', va='bottom', fontsize=8, color=C_STAR)

    # X labels - show every 5th
    tick_pos = list(range(0, n, max(1, n // 12)))
    if n - 1 not in tick_pos:
        tick_pos.append(n - 1)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([fmt_period(labels[i]) for i in tick_pos],
                       rotation=45, ha='right', fontsize=7)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel('จำนวนใบ', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def make_legend():
    """Create standard legend patches."""
    return [
        mpatches.Patch(color=C_BOUGHT, label='ซื้อ (มีจำนวน)'),
        mpatches.Patch(color=C_ZERO, label='ไม่ซื้อ (0)'),
        mpatches.Patch(color=C_NAN, label='ยังไม่ลงทะเบียน (NaN)'),
        mpatches.Patch(color=C_GOLD, label='★ จุดสำคัญ'),
    ]


def save_figure(fig, filename, suptitle):
    """Save figure with suptitle and legend."""
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.98)
    fig.legend(handles=make_legend(), loc='upper right', fontsize=9,
               bbox_to_anchor=(0.98, 0.96), framealpha=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = CHART_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# INSIGHT 10: Buying Streak → Comeback after break
# =============================================================================
def insight_10(df, item_cols, mat, labels):
    print("\n" + "=" * 80)
    print("Insight 10: ซื้อติดกันกี่งวด (streak) → พอขาดจะกลับมามั้ย")
    print("=" * 80)

    n_cust, n_periods = mat.shape

    # For each customer, find longest buying streak and what happens after break
    streak_data = []  # (user_idx, streak_len, streak_end_idx, came_back, periods_to_comeback)

    for i in range(n_cust):
        row = mat[i]
        # Find first non-NaN
        valid = np.where(~np.isnan(row))[0]
        if len(valid) < 4:
            continue

        # Find all buying streaks
        bought = row[valid[0]:] > 0
        nan_mask = np.isnan(row[valid[0]:])
        start_offset = valid[0]

        best_streak = 0
        best_streak_end = 0
        curr_streak = 0

        for j in range(len(bought)):
            if nan_mask[j]:
                continue
            if bought[j]:
                curr_streak += 1
            else:
                if curr_streak > best_streak:
                    best_streak = curr_streak
                    best_streak_end = start_offset + j - 1  # last buy period
                curr_streak = 0
        # Check final streak (still active)
        if curr_streak > best_streak:
            best_streak = curr_streak
            best_streak_end = n_periods - 1
            # Still active streak
            streak_data.append((i, best_streak, best_streak_end, None, None))
            continue

        if best_streak < 1:
            continue

        # Streak broke at best_streak_end + 1
        break_idx = best_streak_end + 1
        if break_idx >= n_periods:
            continue

        # Did they come back?
        came_back = False
        periods_to_comeback = None
        for j in range(break_idx + 1, n_periods):
            if not np.isnan(row[j]) and row[j] > 0:
                came_back = True
                periods_to_comeback = j - break_idx
                break

        streak_data.append((i, best_streak, best_streak_end, came_back, periods_to_comeback))

    sdf = pd.DataFrame(streak_data, columns=['idx', 'streak_len', 'streak_end', 'came_back', 'periods_to_comeback'])

    # Group analysis
    bins = [(1, 3, '1-3'), (4, 6, '4-6'), (7, 10, '7-10'), (11, 999, '10+')]
    print(f"\n{'กลุ่ม streak':<15} {'จำนวน':>8} {'กลับมา%':>10} {'เฉลี่ยกี่งวด':>12}")
    print("-" * 50)
    for lo, hi, label in bins:
        group = sdf[(sdf.streak_len >= lo) & (sdf.streak_len <= hi) & (sdf.came_back.notna())]
        if len(group) == 0:
            continue
        comeback_pct = group.came_back.mean() * 100
        avg_periods = group.loc[group.came_back == True, 'periods_to_comeback'].mean()
        print(f"  {label:<13} {len(group):>8,} {comeback_pct:>9.1f}% {avg_periods:>11.1f}")

    # Still active streaks
    still_active = sdf[sdf.came_back.isna()]
    print(f"\n  Streak ยังไม่ขาด (active): {len(still_active):,} คน")
    print(f"  เฉลี่ย streak ที่ยังไม่ขาด: {still_active.streak_len.mean():.1f} งวด")

    # Find 3 examples
    # 1. Long streak broke and came back
    ex1_pool = sdf[(sdf.came_back == True) & (sdf.streak_len >= 8) & (sdf.periods_to_comeback <= 3)]
    if len(ex1_pool) == 0:
        ex1_pool = sdf[(sdf.came_back == True) & (sdf.streak_len >= 5)]
    ex1 = ex1_pool.sort_values('streak_len', ascending=False).iloc[0] if len(ex1_pool) > 0 else None

    # 2. Streak broke and churned (didn't come back)
    ex2_pool = sdf[(sdf.came_back == False) & (sdf.streak_len >= 6)]
    if len(ex2_pool) == 0:
        ex2_pool = sdf[(sdf.came_back == False) & (sdf.streak_len >= 3)]
    ex2 = ex2_pool.sort_values('streak_len', ascending=False).iloc[0] if len(ex2_pool) > 0 else None

    # 3. Very long streak still active
    ex3 = still_active.sort_values('streak_len', ascending=False).iloc[0] if len(still_active) > 0 else None

    # Print examples
    examples = []
    for tag, ex in [('กลับมาซื้อ', ex1), ('Churn ไปเลย', ex2), ('ซื้อต่อเนื่อง', ex3)]:
        if ex is None:
            continue
        idx = int(ex['idx'])
        uid = df.iloc[idx]['userNo']
        sl = int(ex['streak_len'])
        print(f"\n  ตัวอย่าง ({tag}): {uid}")
        print(f"    streak ยาว {sl} งวด", end='')
        if ex['came_back'] is not None:
            if ex['came_back']:
                print(f" → ขาด → กลับมาใน {int(ex['periods_to_comeback'])} งวด")
            else:
                print(f" → ขาด → ไม่กลับมา (churn)")
        else:
            print(f" → ยังซื้อต่อเนื่อง")
        examples.append((idx, tag, ex))

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(20, 18))
    for ax_i, (idx, tag, ex) in enumerate(examples):
        uid = df.iloc[idx]['userNo']
        vals = mat[idx]
        mark_idx = int(ex['streak_end']) + 1 if ex['came_back'] is not None else None
        mark_label = 'streak ขาด' if mark_idx else None
        plot_customer_bars(axes[ax_i], vals, labels,
                          f"{uid} — {tag} (streak {int(ex['streak_len'])} งวด)",
                          mark_idx=mark_idx, mark_label=mark_label)

    # Add summary text
    broken = sdf[sdf.came_back.notna()]
    comeback_rate = broken.came_back.mean() * 100 if len(broken) > 0 else 0
    summary = (f"สรุป: ลูกค้าที่ streak ขาด {len(broken):,} คน → กลับมาซื้อ {comeback_rate:.1f}%\n"
               f"streak ยิ่งยาว ยิ่งมีแนวโน้มกลับมา | streak ยังไม่ขาด {len(still_active):,} คน")
    fig.text(0.5, 0.01, summary, ha='center', fontsize=12, style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF7ED', alpha=0.8))

    save_figure(fig, 'insight_10_streak.png',
                'Insight 10: ซื้อติดกันกี่งวด (streak) → พอขาดจะกลับมามั้ย')


# =============================================================================
# INSIGHT 11: Spike (2x) → sustain or drop
# =============================================================================
def insight_11(df, item_cols, mat, labels):
    print("\n" + "=" * 80)
    print("Insight 11: ซื้อเพิ่ม 2x ผิดปกติ (spike) → รักษาระดับได้กี่งวด")
    print("=" * 80)

    n_cust, n_periods = mat.shape
    spike_data = []  # (user_idx, spike_idx, spike_val, avg_before, sustain_periods, outcome)

    for i in range(n_cust):
        row = mat[i]
        for j in range(3, n_periods - 1):
            prev3 = row[j-3:j]
            valid_prev = prev3[~np.isnan(prev3)]
            if len(valid_prev) < 2:
                continue
            avg_prev = valid_prev.mean()
            if avg_prev < 1:
                continue
            if np.isnan(row[j]) or row[j] == 0:
                continue
            if row[j] >= 2 * avg_prev:
                # Found spike. Check aftermath
                sustain = 0
                threshold_80 = row[j] * 0.8
                for k in range(j + 1, min(j + 7, n_periods)):
                    if np.isnan(row[k]):
                        break
                    if row[k] >= threshold_80:
                        sustain += 1
                    else:
                        break

                # Outcome classification
                if sustain >= 3:
                    outcome = 'sustained'
                elif j + 1 < n_periods and not np.isnan(row[j+1]) and row[j+1] > row[j]:
                    outcome = 'increased'
                else:
                    outcome = 'dropped'

                spike_data.append((i, j, row[j], avg_prev, sustain, outcome))
                break  # One spike per customer

    sdf = pd.DataFrame(spike_data, columns=['idx', 'spike_idx', 'spike_val', 'avg_before', 'sustain', 'outcome'])

    total = len(sdf)
    print(f"\nพบ spike (ซื้อ >= 2x เฉลี่ย 3 งวดก่อน): {total:,} ครั้ง")

    for outcome, thai in [('sustained', 'รักษาระดับ (3+ งวด)'), ('dropped', 'กลับสู่ระดับเดิม'), ('increased', 'เพิ่มขึ้นอีก')]:
        cnt = (sdf.outcome == outcome).sum()
        print(f"  {thai}: {cnt:,} ({cnt/total*100:.1f}%)")

    print(f"\n  เฉลี่ย sustain: {sdf.sustain.mean():.1f} งวด")
    print(f"  Spike เฉลี่ย: {sdf.spike_val.mean():.1f} ใบ (เฉลี่ยก่อนหน้า {sdf.avg_before.mean():.1f})")

    # Examples
    # 1. Sustained
    ex1_pool = sdf[sdf.outcome == 'sustained'].sort_values('sustain', ascending=False)
    ex1 = ex1_pool.iloc[0] if len(ex1_pool) > 0 else None

    # 2. Dropped back
    ex2_pool = sdf[(sdf.outcome == 'dropped') & (sdf.spike_val >= sdf.spike_val.median())]
    ex2 = ex2_pool.iloc[0] if len(ex2_pool) > 0 else None

    # 3. Increased even more
    ex3_pool = sdf[sdf.outcome == 'increased'].sort_values('spike_val', ascending=False)
    ex3 = ex3_pool.iloc[0] if len(ex3_pool) > 0 else None

    examples = []
    for tag, ex in [('Spike → รักษาระดับ', ex1), ('Spike → กลับที่เดิม', ex2), ('Spike → เพิ่มขึ้นอีก', ex3)]:
        if ex is None:
            continue
        idx = int(ex['idx'])
        uid = df.iloc[idx]['userNo']
        print(f"\n  ตัวอย่าง ({tag}): {uid}")
        print(f"    เฉลี่ยก่อน spike: {ex['avg_before']:.1f} → spike: {int(ex['spike_val'])} ใบ (x{ex['spike_val']/ex['avg_before']:.1f})")
        print(f"    รักษาระดับได้ {int(ex['sustain'])} งวด → {tag}")
        examples.append((idx, tag, ex))

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(20, 18))
    for ax_i, (idx, tag, ex) in enumerate(examples):
        uid = df.iloc[idx]['userNo']
        vals = mat[idx]
        spike_idx = int(ex['spike_idx'])
        plot_customer_bars(axes[ax_i], vals, labels,
                          f"{uid} — {tag} (spike x{ex['spike_val']/ex['avg_before']:.1f})",
                          mark_idx=spike_idx, mark_label='Spike 2x+')

    sustained_pct = (sdf.outcome == 'sustained').mean() * 100
    dropped_pct = (sdf.outcome == 'dropped').mean() * 100
    summary = (f"สรุป: พบ spike {total:,} ครั้ง → รักษาระดับได้ {sustained_pct:.1f}% | "
               f"กลับที่เดิม {dropped_pct:.1f}%\n"
               f"เฉลี่ยรักษาระดับ {sdf.sustain.mean():.1f} งวด")
    fig.text(0.5, 0.01, summary, ha='center', fontsize=12, style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF7ED', alpha=0.8))

    save_figure(fig, 'insight_11_spike.png',
                'Insight 11: ซื้อเพิ่ม 2x ผิดปกติ (spike) → รักษาระดับได้กี่งวด')


# =============================================================================
# INSIGHT 12: 50% Dip → churn warning
# =============================================================================
def insight_12(df, item_cols, mat, labels):
    print("\n" + "=" * 80)
    print("Insight 12: ซื้อลดลง 50% (dip) → churn ภายในกี่งวด (early warning)")
    print("=" * 80)

    n_cust, n_periods = mat.shape
    dip_data = []

    for i in range(n_cust):
        row = mat[i]
        for j in range(3, n_periods):
            prev3 = row[j-3:j]
            valid_prev = prev3[~np.isnan(prev3)]
            if len(valid_prev) < 2:
                continue
            avg_prev = valid_prev.mean()
            if avg_prev < 2:
                continue
            if np.isnan(row[j]):
                continue
            if row[j] <= 0.5 * avg_prev and row[j] > 0:
                # Found dip. How many periods until churn (0)?
                periods_to_churn = None
                for k in range(j + 1, n_periods):
                    if np.isnan(row[k]):
                        break
                    if row[k] == 0:
                        periods_to_churn = k - j
                        break

                # Did they recover?
                recovered = False
                for k in range(j + 1, min(j + 4, n_periods)):
                    if not np.isnan(row[k]) and row[k] >= avg_prev * 0.8:
                        recovered = True
                        break

                dip_data.append((i, j, row[j], avg_prev, periods_to_churn, recovered))
                break  # One dip per customer

    sdf = pd.DataFrame(dip_data, columns=['idx', 'dip_idx', 'dip_val', 'avg_before', 'periods_to_churn', 'recovered'])

    total = len(sdf)
    has_churn = sdf[sdf.periods_to_churn.notna()]
    print(f"\nพบ dip (ซื้อลดลง >= 50%): {total:,} ครั้ง")
    print(f"  Churn (ซื้อ 0) หลัง dip: {len(has_churn):,} ({len(has_churn)/total*100:.1f}%)")

    # Churn within 1/2/3 periods
    for p in [1, 2, 3]:
        cnt = (has_churn.periods_to_churn <= p).sum()
        print(f"  Churn ภายใน {p} งวด: {cnt:,} ({cnt/total*100:.1f}%)")

    recovered_cnt = sdf.recovered.sum()
    print(f"\n  ฟื้นตัว (กลับ >= 80% ภายใน 3 งวด): {recovered_cnt:,} ({recovered_cnt/total*100:.1f}%)")

    avg_periods = has_churn.periods_to_churn.mean()
    print(f"  เฉลี่ย churn หลัง dip: {avg_periods:.1f} งวด")

    # Examples
    # 1. Dip → churn quickly
    ex1_pool = has_churn[(has_churn.periods_to_churn <= 2) & (has_churn.avg_before >= 5)]
    ex1 = ex1_pool.sort_values('avg_before', ascending=False).iloc[0] if len(ex1_pool) > 0 else None

    # 2. Dip → recovered
    ex2_pool = sdf[(sdf.recovered == True) & (sdf.avg_before >= 5)]
    if len(ex2_pool) == 0:
        ex2_pool = sdf[sdf.recovered == True]
    ex2 = ex2_pool.sort_values('avg_before', ascending=False).iloc[0] if len(ex2_pool) > 0 else None

    # 3. Dip → gradual decline then churn
    ex3_pool = has_churn[(has_churn.periods_to_churn >= 3) & (has_churn.avg_before >= 4)]
    ex3 = ex3_pool.sort_values('periods_to_churn', ascending=False).iloc[0] if len(ex3_pool) > 0 else None

    examples = []
    for tag, ex in [('Dip → Churn เร็ว', ex1), ('Dip → ฟื้นตัว', ex2), ('Dip → ค่อยๆ ลด → Churn', ex3)]:
        if ex is None:
            continue
        idx = int(ex['idx'])
        uid = df.iloc[idx]['userNo']
        print(f"\n  ตัวอย่าง ({tag}): {uid}")
        print(f"    เฉลี่ยก่อน dip: {ex['avg_before']:.1f} → dip: {int(ex['dip_val'])} ใบ ({ex['dip_val']/ex['avg_before']*100:.0f}% ของเฉลี่ย)")
        if ex['periods_to_churn'] is not None and not (isinstance(ex['periods_to_churn'], float) and np.isnan(ex['periods_to_churn'])):
            print(f"    Churn ใน {int(ex['periods_to_churn'])} งวดหลัง dip")
        else:
            print(f"    ยังไม่ churn (ฟื้นตัวได้)")
        examples.append((idx, tag, ex))

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(20, 18))
    for ax_i, (idx, tag, ex) in enumerate(examples):
        uid = df.iloc[idx]['userNo']
        vals = mat[idx]
        dip_idx = int(ex['dip_idx'])
        plot_customer_bars(axes[ax_i], vals, labels,
                          f"{uid} — {tag} (ลดลง {ex['dip_val']/ex['avg_before']*100:.0f}%)",
                          mark_idx=dip_idx, mark_label='Dip 50%+')

    churn_1 = (has_churn.periods_to_churn <= 1).sum() / total * 100
    churn_3 = (has_churn.periods_to_churn <= 3).sum() / total * 100
    summary = (f"สรุป: ลูกค้าซื้อลดลง 50%+ → {len(has_churn)/total*100:.1f}% churn ในที่สุด\n"
               f"Churn ภายใน 1 งวด {churn_1:.1f}% | ภายใน 3 งวด {churn_3:.1f}% | "
               f"ฟื้นตัว {recovered_cnt/total*100:.1f}%")
    fig.text(0.5, 0.01, summary, ha='center', fontsize=12, style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF7ED', alpha=0.8))

    save_figure(fig, 'insight_12_dip_warning.png',
                'Insight 12: ซื้อลดลง 50% (dip) → churn ภายในกี่งวด (early warning)')


# =============================================================================
# INSIGHT 13: Consistent vs Burst buyers
# =============================================================================
def insight_13(df, item_cols, mat, labels):
    print("\n" + "=" * 80)
    print("Insight 13: ซื้อสม่ำเสมอ vs ซื้อแบบ burst → ใครอยู่นานกว่า")
    print("=" * 80)

    n_cust, n_periods = mat.shape

    stats = []
    for i in range(n_cust):
        row = mat[i]
        valid = row[~np.isnan(row)]
        if len(valid) < 5:
            continue
        purchases = valid[valid > 0]
        if len(purchases) < 3:
            continue

        tenure = len(valid)
        mean_p = purchases.mean()
        std_p = purchases.std()
        cv = std_p / mean_p if mean_p > 0 else 0
        total_tickets = purchases.sum()
        active_pct = len(purchases) / len(valid)
        # Check if last period is 0 (churned in latest)
        last_val = row[-1] if not np.isnan(row[-1]) else row[-2] if not np.isnan(row[-2]) else 0
        churned = 1 if last_val == 0 else 0

        # Classification
        if cv < 0.5:
            cat = 'Consistent'
        elif cv > 1.0:
            cat = 'Burst'
        else:
            cat = 'Middle'

        stats.append((i, cv, cat, tenure, total_tickets, active_pct, churned, mean_p))

    sdf = pd.DataFrame(stats, columns=['idx', 'cv', 'cat', 'tenure', 'total_tickets', 'active_pct', 'churned', 'mean_purchase'])

    print(f"\n{'ประเภท':<15} {'จำนวน':>8} {'เฉลี่ย tenure':>14} {'Churn rate':>12} {'เฉลี่ยใบรวม':>12} {'Active%':>10}")
    print("-" * 75)
    for cat in ['Consistent', 'Middle', 'Burst']:
        g = sdf[sdf.cat == cat]
        if len(g) == 0:
            continue
        print(f"  {cat:<13} {len(g):>8,} {g.tenure.mean():>13.1f} {g.churned.mean()*100:>11.1f}% {g.total_tickets.mean():>11.1f} {g.active_pct.mean()*100:>9.1f}%")

    # Examples
    # 1. Very consistent buyer (low CV, meaningful purchases — mean >= 5, active >= 70%)
    ex1_pool = sdf[(sdf.cat == 'Consistent') & (sdf.tenure >= 30) & (sdf.churned == 0)
                   & (sdf.mean_purchase >= 5) & (sdf.active_pct >= 0.7)]
    if len(ex1_pool) == 0:
        ex1_pool = sdf[(sdf.cat == 'Consistent') & (sdf.tenure >= 20) & (sdf.churned == 0) & (sdf.mean_purchase >= 3)]
    # Sort by active_pct desc then cv asc for best example
    ex1 = ex1_pool.sort_values(['active_pct', 'cv'], ascending=[False, True]).iloc[0] if len(ex1_pool) > 0 else None

    # 2. Burst buyer (high CV but not extreme — mean_purchase between 5-100 for readable chart)
    ex2_pool = sdf[(sdf.cat == 'Burst') & (sdf.tenure >= 20) & (sdf.mean_purchase >= 5) & (sdf.mean_purchase <= 100)
                   & (sdf.active_pct >= 0.3)]
    if len(ex2_pool) == 0:
        ex2_pool = sdf[(sdf.cat == 'Burst') & (sdf.tenure >= 15) & (sdf.total_tickets >= 50)]
    ex2 = ex2_pool.sort_values('cv', ascending=False).iloc[0] if len(ex2_pool) > 0 else None

    # 3. Changed from burst to consistent (high CV in first half, low in second half)
    change_candidates = []
    ex2_idx = int(ex2['idx']) if ex2 is not None else -1
    for _, r in sdf[(sdf.tenure >= 20) & (sdf.mean_purchase >= 3)].iterrows():
        idx = int(r['idx'])
        if idx == ex2_idx:
            continue  # avoid same customer as burst example
        row = mat[idx]
        valid = row[~np.isnan(row)]
        mid = len(valid) // 2
        first_half = valid[:mid]
        second_half = valid[mid:]
        p1 = first_half[first_half > 0]
        p2 = second_half[second_half > 0]
        if len(p1) < 3 or len(p2) < 3:
            continue
        cv1 = p1.std() / p1.mean() if p1.mean() > 0 else 0
        cv2 = p2.std() / p2.mean() if p2.mean() > 0 else 0
        if cv1 > 0.8 and cv2 < 0.5:
            change_candidates.append((idx, cv1, cv2, cv1 - cv2))

    ex3 = None
    if change_candidates:
        change_candidates.sort(key=lambda x: x[3], reverse=True)
        ex3_idx = change_candidates[0][0]
        ex3 = sdf[sdf.idx == ex3_idx].iloc[0]
        print(f"\n  พบลูกค้าเปลี่ยนจาก burst → consistent: {len(change_candidates):,} คน")
        print(f"    ตัวอย่าง: CV ครึ่งแรก {change_candidates[0][1]:.2f} → ครึ่งหลัง {change_candidates[0][2]:.2f}")

    examples = []
    for tag, ex in [('สม่ำเสมอ (Consistent)', ex1), ('Burst (ขึ้นๆ ลงๆ)', ex2), ('Burst → Consistent', ex3)]:
        if ex is None:
            continue
        idx = int(ex['idx'])
        uid = df.iloc[idx]['userNo']
        print(f"\n  ตัวอย่าง ({tag}): {uid}")
        print(f"    CV: {ex['cv']:.2f}, tenure: {int(ex['tenure'])} งวด, รวม {int(ex['total_tickets'])} ใบ")
        examples.append((idx, tag, ex))

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(20, 18))
    for ax_i, (idx, tag, ex) in enumerate(examples):
        uid = df.iloc[idx]['userNo']
        vals = mat[idx]
        plot_customer_bars(axes[ax_i], vals, labels,
                          f"{uid} — {tag} (CV={ex['cv']:.2f})",
                          mark_idx=None)

    cons_churn = sdf[sdf.cat == 'Consistent'].churned.mean() * 100
    burst_churn = sdf[sdf.cat == 'Burst'].churned.mean() * 100
    summary = (f"สรุป: Consistent (CV<0.5) churn {cons_churn:.1f}% vs Burst (CV>1.0) churn {burst_churn:.1f}%\n"
               f"ลูกค้าสม่ำเสมอมี churn ต่ำกว่า {burst_churn - cons_churn:.1f}pp | "
               f"tenure เฉลี่ย Consistent {sdf[sdf.cat=='Consistent'].tenure.mean():.0f} vs Burst {sdf[sdf.cat=='Burst'].tenure.mean():.0f} งวด")
    fig.text(0.5, 0.01, summary, ha='center', fontsize=12, style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF7ED', alpha=0.8))

    save_figure(fig, 'insight_13_consistency.png',
                'Insight 13: ซื้อสม่ำเสมอ vs ซื้อแบบ burst → ใครอยู่นานกว่า')


# =============================================================================
# INSIGHT 14: 3 consecutive increases → continue or plateau
# =============================================================================
def insight_14(df, item_cols, mat, labels):
    print("\n" + "=" * 80)
    print("Insight 14: ซื้อเพิ่มขึ้น 3 งวดติด → จะเพิ่มต่อหรือ plateau")
    print("=" * 80)

    n_cust, n_periods = mat.shape
    growth_data = []

    for i in range(n_cust):
        row = mat[i]
        for j in range(3, n_periods - 1):
            # Check 3 consecutive increases: j-2 < j-1 < j (all > 0, not NaN)
            vals = [row[j-2], row[j-1], row[j]]
            if any(np.isnan(v) for v in vals):
                continue
            if not all(v > 0 for v in vals):
                continue
            if vals[0] < vals[1] < vals[2]:
                # 3 consecutive increases found at index j
                # What happens next?
                if j + 1 >= n_periods or np.isnan(row[j + 1]):
                    outcome = 'unknown'
                elif row[j + 1] > row[j]:
                    outcome = 'continued'
                elif row[j + 1] >= row[j] * 0.85:
                    outcome = 'plateau'
                else:
                    outcome = 'declined'

                growth_data.append((i, j, vals[0], vals[1], vals[2],
                                   row[j+1] if j+1 < n_periods and not np.isnan(row[j+1]) else None,
                                   outcome))
                break  # One per customer

    sdf = pd.DataFrame(growth_data, columns=['idx', 'growth_end', 'v1', 'v2', 'v3', 'v4', 'outcome'])

    known = sdf[sdf.outcome != 'unknown']
    total = len(known)
    print(f"\nพบลูกค้าซื้อเพิ่ม 3 งวดติด: {len(sdf):,} คน (outcome ทราบ: {total:,})")

    for outcome, thai in [('continued', 'เพิ่มต่อ (งวดที่ 4)'), ('plateau', 'คงที่ (plateau)'), ('declined', 'ลดลง')]:
        cnt = (known.outcome == outcome).sum()
        print(f"  {thai}: {cnt:,} ({cnt/total*100:.1f}%)")

    # Average growth rate
    known_with_v4 = known[known.v4.notna()]
    if len(known_with_v4) > 0:
        avg_change = ((known_with_v4.v4 - known_with_v4.v3) / known_with_v4.v3).mean() * 100
        print(f"\n  เฉลี่ยเปลี่ยนแปลง งวดที่ 4: {avg_change:+.1f}%")

    # Examples
    # 1. Continued growth
    ex1_pool = known[(known.outcome == 'continued') & (known.v3 >= 5)]
    ex1 = ex1_pool.sort_values('v3', ascending=False).iloc[0] if len(ex1_pool) > 0 else None

    # 2. Plateau
    ex2_pool = known[(known.outcome == 'plateau') & (known.v3 >= 5)]
    if len(ex2_pool) == 0:
        ex2_pool = known[known.outcome == 'plateau']
    ex2 = ex2_pool.sort_values('v3', ascending=False).iloc[0] if len(ex2_pool) > 0 else None

    # 3. Declined after growth
    ex3_pool = known[(known.outcome == 'declined') & (known.v3 >= 5)]
    if len(ex3_pool) == 0:
        ex3_pool = known[known.outcome == 'declined']
    ex3 = ex3_pool.sort_values('v3', ascending=False).iloc[0] if len(ex3_pool) > 0 else None

    examples = []
    for tag, ex in [('เพิ่มต่อ', ex1), ('Plateau', ex2), ('ลดลงหลังเพิ่ม', ex3)]:
        if ex is None:
            continue
        idx = int(ex['idx'])
        uid = df.iloc[idx]['userNo']
        print(f"\n  ตัวอย่าง ({tag}): {uid}")
        print(f"    เพิ่มติด 3 งวด: {int(ex['v1'])} → {int(ex['v2'])} → {int(ex['v3'])}", end='')
        if ex['v4'] is not None:
            print(f" → {int(ex['v4'])} ({tag})")
        else:
            print()
        examples.append((idx, tag, ex))

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(20, 18))
    for ax_i, (idx, tag, ex) in enumerate(examples):
        uid = df.iloc[idx]['userNo']
        vals = mat[idx]
        growth_end = int(ex['growth_end'])
        plot_customer_bars(axes[ax_i], vals, labels,
                          f"{uid} — {tag} ({int(ex['v1'])}→{int(ex['v2'])}→{int(ex['v3'])})",
                          mark_idx=growth_end, mark_label='3 งวดเพิ่มติด')

    cont_pct = (known.outcome == 'continued').mean() * 100
    plat_pct = (known.outcome == 'plateau').mean() * 100
    dec_pct = (known.outcome == 'declined').mean() * 100
    summary = (f"สรุป: ซื้อเพิ่ม 3 งวดติด {len(sdf):,} คน → เพิ่มต่อ {cont_pct:.1f}% | "
               f"คงที่ {plat_pct:.1f}% | ลดลง {dec_pct:.1f}%\n"
               f"Growth streak ≠ การันตีว่าจะเพิ่มต่อ — ส่วนใหญ่ลดลงหรือคงที่")
    fig.text(0.5, 0.01, summary, ha='center', fontsize=12, style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF7ED', alpha=0.8))

    save_figure(fig, 'insight_14_growth_streak.png',
                'Insight 14: ซื้อเพิ่มขึ้น 3 งวดติด → จะเพิ่มต่อหรือ plateau')


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("Insights 10-14: Purchase Behavior Patterns")
    print("=" * 80)

    df, item_cols = load_data()
    mat, labels = get_purchase_matrix(df, item_cols)

    insight_10(df, item_cols, mat, labels)
    insight_11(df, item_cols, mat, labels)
    insight_12(df, item_cols, mat, labels)
    insight_13(df, item_cols, mat, labels)
    insight_14(df, item_cols, mat, labels)

    print("\n" + "=" * 80)
    print("All insights 10-14 complete!")
    print("=" * 80)
