#!/usr/bin/env python3
"""
Insights #2-5: Purchase behavior insights with customer examples and bar charts.

Insight 2: ถูกรางวัลซ้ำ 2+ ครั้งติด → effect ทบต้นมั้ย
Insight 3: ถูกรางวัลแล้วไม่ถูกอีก → ยอดตกกลับเมื่อไหร่
Insight 4: Cancel แล้วสั่งใหม่สำเร็จ → ซื้อต่อเนื่องมั้ย
Insight 5: Cancel แล้วไม่สั่งใหม่ → churn ภายในกี่งวด
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
import warnings, sys, time
warnings.filterwarnings('ignore')

# Thai font setup
for fname in ['Tahoma', 'TH Sarabun New', 'Angsana New', 'Garuda', 'Loma',
              'Norasi', 'Sawasdee', 'DejaVu Sans']:
    try:
        plt.rcParams['font.family'] = fname
        fig_test = plt.figure(figsize=(1,1))
        fig_test.text(0.5, 0.5, 'test')
        plt.close(fig_test)
        break
    except:
        continue

BASE = Path('/Users/nongnat/Desktop/ChurnCustomer')
PRIZE_DIR = BASE / 'data' / 'prize'
CHURN_DIR = BASE / 'data' / 'churn'
TX_DIR = BASE / 'data' / 'transaction'
CHART_DIR = BASE / 'charts'
CHART_DIR.mkdir(exist_ok=True)

# Colors
C_BOUGHT = '#F97316'
C_NOBUY = '#E5E7EB'
C_NOREG = '#1F2937'
C_EVENT = '#DC2626'
C_GOLD = '#DAA520'
C_CANCEL = '#7C3AED'

def flush_print(msg):
    print(msg)
    sys.stdout.flush()

# ─────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────
def load_churn_history():
    """Load the latest churn file."""
    churn_files = sorted([f for f in CHURN_DIR.glob('Churn_20*.csv')
                          if 'Check' not in f.name and 'old_validation' not in f.name])
    latest = churn_files[-1]
    flush_print(f"  Loading: {latest.name}")
    t0 = time.time()
    df = pd.read_csv(latest, low_memory=False)
    item_cols = [c for c in df.columns if c.startswith('item')]
    flush_print(f"  Loaded {len(df):,} rows, {len(item_cols)} periods in {time.time()-t0:.1f}s")
    return df, item_cols

def parse_period_label(col):
    parts = col.replace('item', '').split('_')
    return f"{parts[1]}/{parts[2]}"

def parse_period_date(col):
    parts = col.replace('item', '').split('_')
    return f"{parts[0]}_{parts[1]}{parts[2]}"

def load_prize_data():
    """Load all prize files, return DataFrame with userNo + period_key."""
    prize_files = sorted(PRIZE_DIR.glob('Prize_*.csv'))
    all_prizes = []
    for f in prize_files:
        df = pd.read_csv(f, usecols=['userNo', 'rounddate', 'type', 'total_prize'])
        parts = f.stem.split('_')
        df['period_key'] = f"{parts[1]}_{parts[2]}"
        all_prizes.append(df)
    all_df = pd.concat(all_prizes, ignore_index=True)
    flush_print(f"  Prize: {len(prize_files)} files, {len(all_df):,} records, {all_df['userNo'].nunique():,} unique winners")
    return all_df

def load_tx_cancel_info():
    """Load TX files, extract cancel/success per user per period."""
    tx_files = sorted(TX_DIR.glob('Transaction_Order_*.csv'))
    results = []
    t0 = time.time()
    for fi, f in enumerate(tx_files):
        parts = f.stem.split('_')
        period_key = f"{parts[2]}_{parts[3]}"
        df = pd.read_csv(f, usecols=['userNo', 'status'])
        # Create binary columns for fast groupby
        df['is_cancel'] = (df['status'] == 'CANCEL').astype(int)
        df['is_success'] = (df['status'] == 'SUCCESS').astype(int)
        agg = df.groupby('userNo').agg(
            cancel_count=('is_cancel', 'sum'),
            success_count=('is_success', 'sum')
        ).reset_index()
        agg['has_cancel'] = agg['cancel_count'] > 0
        agg['has_success'] = agg['success_count'] > 0
        agg['period_key'] = period_key
        agg['cancel_then_success'] = agg['has_cancel'] & agg['has_success']
        agg['cancel_only'] = agg['has_cancel'] & ~agg['has_success']
        results.append(agg[['userNo', 'period_key', 'has_cancel', 'has_success',
                            'cancel_then_success', 'cancel_only']])
        if (fi + 1) % 10 == 0:
            flush_print(f"    ... {fi+1}/{len(tx_files)} files loaded")
    all_tx = pd.concat(results, ignore_index=True)
    flush_print(f"  TX: {len(tx_files)} files, {len(all_tx):,} user-period records in {time.time()-t0:.1f}s")
    return all_tx


# ─────────────────────────────────────────────────────
# Chart helper
# ─────────────────────────────────────────────────────
def plot_customer_bar(ax, user_row, item_cols, event_periods=None, cancel_periods=None,
                      user_info_text="", baseline_val=None):
    labels = [parse_period_label(c) for c in item_cols]
    values = user_row[item_cols].values.astype(float)

    colors = []
    for v in values:
        if np.isnan(v):
            colors.append(C_NOREG)
        elif v == 0:
            colors.append(C_NOBUY)
        else:
            colors.append(C_BOUGHT)

    bar_values = np.where(np.isnan(values), 0.3, values)
    bar_values = np.where((~np.isnan(values)) & (values == 0), 0.3, bar_values)

    x = np.arange(len(labels))
    ax.bar(x, bar_values, color=colors, edgecolor='white', linewidth=0.3, width=0.7)

    max_bar = max(bar_values) if len(bar_values) > 0 else 1

    if event_periods:
        for ep in event_periods:
            if 0 <= ep < len(values):
                y_pos = max(bar_values[ep], 0.5) + max_bar * 0.08
                ax.text(ep, y_pos, '\u2605', color=C_EVENT, fontsize=16, ha='center', va='bottom',
                        fontweight='bold')
                ax.axvline(x=ep, color=C_GOLD, linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)

    if cancel_periods:
        for cp in cancel_periods:
            if 0 <= cp < len(values):
                y_pos = max(bar_values[cp], 0.5) + max_bar * 0.08
                ax.text(cp, y_pos, '\u2715', color=C_CANCEL, fontsize=14, ha='center', va='bottom',
                        fontweight='bold')

    if baseline_val is not None and baseline_val > 0:
        ax.axhline(y=baseline_val, color='#6B7280', linestyle=':', linewidth=1.5, alpha=0.6)

    ax.set_xticks(x[::2])  # Show every other label to avoid crowding
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), 2)], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('tickets', fontsize=9)
    ax.set_title(user_info_text, fontsize=10, loc='left', pad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-0.5, len(labels) - 0.5)


def map_item_to_prize(item_cols):
    mapping = {}
    for col in item_cols:
        pk = parse_period_date(col)
        mapping[col] = pk
    return mapping


# ═══════════════════════════════════════════════════════════════
# INSIGHT 2: ถูกรางวัลซ้ำ 2+ ครั้งติด
# ═══════════════════════════════════════════════════════════════
def insight_02(churn_df, item_cols, prize_df):
    flush_print("\n" + "="*70)
    flush_print("Insight 2: ถูกรางวัลซ้ำ 2+ ครั้งติด → effect ทบต้นมั้ย")
    flush_print("="*70)

    col_to_prize = map_item_to_prize(item_cols)
    # Reverse: period_key → col index
    pk_to_idx = {}
    for idx, col in enumerate(item_cols):
        pk = col_to_prize[col]
        pk_to_idx[pk] = idx

    # Get unique winners per period
    winners_per_period = prize_df.groupby('period_key')['userNo'].apply(set).to_dict()

    # Build: userNo → sorted list of win col-indices
    prize_periods = sorted([pk for pk in winners_per_period if pk in pk_to_idx])
    user_win_indices = {}
    for pk in prize_periods:
        idx = pk_to_idx[pk]
        for user in winners_per_period[pk]:
            if user not in user_win_indices:
                user_win_indices[user] = []
            user_win_indices[user].append(idx)

    # Find consecutive streaks
    churn_user_set = set(churn_df['userNo'].values)
    consecutive_winners = []

    for user, indices in user_win_indices.items():
        if user not in churn_user_set:
            continue
        indices = sorted(set(indices))
        if len(indices) < 2:
            continue
        # Find streaks of consecutive indices
        streaks = []
        current = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current.append(indices[i])
            else:
                if len(current) >= 2:
                    streaks.append(current)
                current = [indices[i]]
        if len(current) >= 2:
            streaks.append(current)

        if streaks:
            consecutive_winners.append({
                'userNo': user,
                'streaks': streaks,
                'max_streak': max(len(s) for s in streaks),
                'all_win_indices': indices
            })

    flush_print(f"\n  ลูกค้าที่ถูกรางวัล (อยู่ใน Churn): {len(user_win_indices):,} คน")
    flush_print(f"  ถูกซ้ำ 2+ งวดติด: {len(consecutive_winners):,} คน")

    # Analyze compounding effect using vectorized lookup
    churn_indexed = churn_df.set_index('userNo')
    item_vals = churn_indexed[item_cols]

    after_1st_changes = []
    after_2nd_changes = []

    for cw in consecutive_winners:
        user = cw['userNo']
        if user not in item_vals.index:
            continue
        vals = item_vals.loc[user].values.astype(float)

        for streak in cw['streaks']:
            if len(streak) < 2:
                continue
            pre_idx = streak[0] - 1
            if pre_idx < 0:
                continue
            pre_val = vals[pre_idx]
            if np.isnan(pre_val) or pre_val <= 0:
                continue

            # During 1st win
            v1 = vals[streak[0]]
            if not np.isnan(v1):
                after_1st_changes.append((v1 - pre_val) / pre_val * 100)

            # After 2nd win (the period AFTER the 2nd consecutive win)
            post2_idx = streak[1] + 1
            if post2_idx < len(vals):
                v2 = vals[post2_idx]
                if not np.isnan(v2):
                    after_2nd_changes.append((v2 - pre_val) / pre_val * 100)

    if after_1st_changes:
        flush_print(f"\n  ยอดซื้อเปลี่ยนแปลง:")
        flush_print(f"    หลังถูกครั้งที่ 1: เฉลี่ย {np.mean(after_1st_changes):+.1f}% (n={len(after_1st_changes)})")
    if after_2nd_changes:
        flush_print(f"    หลังถูกครั้งที่ 2: เฉลี่ย {np.mean(after_2nd_changes):+.1f}% (n={len(after_2nd_changes)})")
        compound = np.mean(after_2nd_changes) > np.mean(after_1st_changes)
        flush_print(f"    ทบต้น: {'ใช่ — ยอดเพิ่มขึ้นอีก!' if compound else 'ไม่ — ยอดไม่ได้เพิ่มต่อ'}")

    # Sort by longest streak, pick 3 with data
    consecutive_winners.sort(key=lambda c: (-c['max_streak'], -len(c['all_win_indices'])))
    examples = []
    for cw in consecutive_winners[:100]:
        user = cw['userNo']
        if user not in item_vals.index:
            continue
        vals = item_vals.loc[user].values.astype(float)
        streak = max(cw['streaks'], key=len)
        has_data = np.sum(~np.isnan(vals[max(0, streak[0]-3):min(len(vals), streak[-1]+4)]))
        if has_data >= 4:
            examples.append(cw)
        if len(examples) >= 3:
            break

    for i, ex in enumerate(examples):
        user = ex['userNo']
        vals = item_vals.loc[user].values.astype(float)
        streak = max(ex['streaks'], key=len)
        streak_labels = [parse_period_label(item_cols[s]) for s in streak]
        pre_idx = streak[0] - 1
        pre_val = vals[pre_idx] if pre_idx >= 0 and not np.isnan(vals[pre_idx]) else 0
        during = [vals[s] for s in streak if not np.isnan(vals[s])]
        flush_print(f"\n  ตัวอย่าง {i+1}: {user}")
        flush_print(f"    ถูก {len(streak)} งวดติด: {', '.join(streak_labels)}")
        flush_print(f"    ก่อนถูก: {pre_val:.0f} ใบ → ระหว่างถูก: {[f'{v:.0f}' for v in during]} ใบ")

    # Chart
    fig, axes = plt.subplots(3, 1, figsize=(20, 18))
    fig.suptitle('Insight 2: ถูกรางวัลซ้ำ 2+ ครั้งติด → effect ทบต้นมั้ย',
                 fontsize=16, fontweight='bold', y=0.98)

    for i, ex in enumerate(examples):
        user = ex['userNo']
        row = churn_df[churn_df['userNo'] == user].iloc[0]
        vals = item_vals.loc[user].values.astype(float)
        streak = max(ex['streaks'], key=len)
        pre_idx = streak[0] - 1
        pre_val = vals[pre_idx] if pre_idx >= 0 and not np.isnan(vals[pre_idx]) else 0
        info = (f"{user} | อายุ: {row.get('age', '?')} | เพศ: {row.get('gender', '?')} | "
                f"ถูกรางวัล {len(streak)} งวดติด | ก่อนถูก: {pre_val:.0f} ใบ")
        plot_customer_bar(axes[i], row, item_cols, event_periods=ex['all_win_indices'],
                         user_info_text=info)

    legend_elements = [
        mpatches.Patch(facecolor=C_BOUGHT, label='ซื้อ'),
        mpatches.Patch(facecolor=C_NOBUY, label='ไม่ซื้อ (0)'),
        mpatches.Patch(facecolor=C_NOREG, label='ยังไม่สมัคร'),
        Line2D([0], [0], marker='*', color=C_EVENT, markersize=15, linestyle='None', label='★ ถูกรางวัล'),
        Line2D([0], [0], color=C_GOLD, linestyle='--', linewidth=2, label='งวดที่ถูกรางวัล'),
    ]
    summary = [f"ถูกซ้ำ 2+ งวดติด: {len(consecutive_winners):,} คน"]
    if after_1st_changes:
        summary.append(f"หลังถูกครั้งที่ 1: {np.mean(after_1st_changes):+.1f}%")
    if after_2nd_changes:
        summary.append(f"หลังถูกครั้งที่ 2: {np.mean(after_2nd_changes):+.1f}%")

    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))
    fig.text(0.5, 0.04, ' | '.join(summary), ha='center', fontsize=11,
             style='italic', color='#374151')
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    out = CHART_DIR / 'insight_02_multi_win.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    flush_print(f"\n  บันทึก: {out}")


# ═══════════════════════════════════════════════════════════════
# INSIGHT 3: ถูกรางวัลแล้วไม่ถูกอีก → ยอดตกกลับเมื่อไหร่
# ═══════════════════════════════════════════════════════════════
def insight_03(churn_df, item_cols, prize_df):
    flush_print("\n" + "="*70)
    flush_print("Insight 3: ถูกรางวัลแล้วไม่ถูกอีก → ยอดตกกลับเมื่อไหร่")
    flush_print("="*70)

    col_to_prize = map_item_to_prize(item_cols)
    pk_to_idx = {col_to_prize[col]: idx for idx, col in enumerate(item_cols)}

    winners_per_period = prize_df.groupby('period_key')['userNo'].apply(set).to_dict()
    churn_user_set = set(churn_df['userNo'].values)

    # Build win indices per user
    user_win_indices = {}
    for pk, users in winners_per_period.items():
        if pk not in pk_to_idx:
            continue
        idx = pk_to_idx[pk]
        for user in users:
            if user in churn_user_set:
                if user not in user_win_indices:
                    user_win_indices[user] = []
                user_win_indices[user].append(idx)

    # Find users who won once then didn't win for 5+ periods
    churn_indexed = churn_df.set_index('userNo')
    item_vals = churn_indexed[item_cols]

    single_win_users = []
    for user, indices in user_win_indices.items():
        indices = sorted(set(indices))
        for i, w_idx in enumerate(indices):
            next_win = indices[i+1] if i+1 < len(indices) else len(item_cols)
            if next_win - w_idx >= 5:
                single_win_users.append({'userNo': user, 'win_idx': w_idx})
                break

    flush_print(f"\n  ถูก 1 ครั้งแล้วไม่ถูกอีก 5+ งวด: {len(single_win_users):,} คน")

    # Analyze decay
    decay_periods = []
    return_within = {3: 0, 6: 0, 10: 0}
    total_analyzable = 0

    for sw in single_win_users:
        user = sw['userNo']
        win_idx = sw['win_idx']
        if user not in item_vals.index:
            continue
        vals = item_vals.loc[user].values.astype(float)

        pre_start = max(0, win_idx - 3)
        pre_vals = vals[pre_start:win_idx]
        pre_bought = pre_vals[(~np.isnan(pre_vals)) & (pre_vals > 0)]
        if len(pre_bought) == 0:
            continue
        baseline = np.mean(pre_bought)
        win_val = vals[win_idx] if not np.isnan(vals[win_idx]) else 0
        if win_val <= baseline:
            continue

        total_analyzable += 1
        returned = False
        for offset in range(1, min(11, len(vals) - win_idx)):
            pv = vals[win_idx + offset]
            if np.isnan(pv):
                continue
            if pv <= baseline * 1.1:
                decay_periods.append(offset)
                for t in [3, 6, 10]:
                    if offset <= t:
                        return_within[t] += 1
                returned = True
                break
        if not returned:
            decay_periods.append(None)

    non_none = [d for d in decay_periods if d is not None]
    flush_print(f"  วิเคราะห์ได้: {total_analyzable:,} คน (ยอดเพิ่มหลังถูก)")
    if non_none:
        flush_print(f"  กลับ baseline เฉลี่ย: {np.mean(non_none):.1f} งวด (median: {np.median(non_none):.0f})")
    if total_analyzable > 0:
        flush_print(f"\n  อัตราการกลับสู่ baseline:")
        for t in [3, 6, 10]:
            pct = return_within[t] / total_analyzable * 100
            flush_print(f"    ภายใน {t} งวด: {return_within[t]:,} คน ({pct:.1f}%)")
        never = sum(1 for d in decay_periods if d is None)
        flush_print(f"    ไม่กลับเลย: {never:,} คน ({never/total_analyzable*100:.1f}%)")

    # Find 3 examples
    examples = []
    for sw in single_win_users:
        if len(examples) >= 3:
            break
        user = sw['userNo']
        win_idx = sw['win_idx']
        if user not in item_vals.index:
            continue
        vals = item_vals.loc[user].values.astype(float)

        pre_start = max(0, win_idx - 3)
        pre_vals = vals[pre_start:win_idx]
        pre_bought = pre_vals[(~np.isnan(pre_vals)) & (pre_vals > 0)]
        if len(pre_bought) == 0:
            continue
        baseline = np.mean(pre_bought)
        win_val = vals[win_idx] if not np.isnan(vals[win_idx]) else 0
        if win_val <= baseline * 1.2:
            continue

        post_vals = vals[win_idx+1:min(win_idx+11, len(vals))]
        if np.sum(~np.isnan(post_vals)) < 3:
            continue

        return_idx = None
        for offset in range(1, min(11, len(vals) - win_idx)):
            pv = vals[win_idx + offset]
            if not np.isnan(pv) and pv <= baseline * 1.1:
                return_idx = win_idx + offset
                break

        examples.append({
            'userNo': user, 'win_idx': win_idx, 'baseline': baseline,
            'win_val': win_val, 'return_idx': return_idx,
            'spike': win_val / baseline
        })

    for i, ex in enumerate(examples):
        flush_print(f"\n  ตัวอย่าง {i+1}: {ex['userNo']}")
        flush_print(f"    ถูกรางวัลงวด: {parse_period_label(item_cols[ex['win_idx']])}")
        flush_print(f"    Baseline: {ex['baseline']:.0f} ใบ → ตอนถูก: {ex['win_val']:.0f} ใบ ({ex['spike']:.1f}x)")
        if ex['return_idx']:
            flush_print(f"    กลับ baseline: {parse_period_label(item_cols[ex['return_idx']])} ({ex['return_idx']-ex['win_idx']} งวด)")
        else:
            flush_print(f"    ยังไม่กลับ baseline")

    # Chart
    fig, axes = plt.subplots(3, 1, figsize=(20, 18))
    fig.suptitle('Insight 3: ถูกรางวัลแล้วไม่ถูกอีก → ยอดตกกลับเมื่อไหร่',
                 fontsize=16, fontweight='bold', y=0.98)

    for i, ex in enumerate(examples):
        user = ex['userNo']
        row = churn_df[churn_df['userNo'] == user].iloc[0]
        info = (f"{user} | Baseline: {ex['baseline']:.0f} ใบ → ถูกรางวัล: {ex['win_val']:.0f} ใบ ({ex['spike']:.1f}x)")
        if ex['return_idx']:
            info += f" | กลับ baseline ใน {ex['return_idx']-ex['win_idx']} งวด"
        else:
            info += " | ยังไม่กลับ baseline"

        plot_customer_bar(axes[i], row, item_cols, event_periods=[ex['win_idx']],
                         user_info_text=info, baseline_val=ex['baseline'])

        if ex['return_idx']:
            axes[i].annotate('กลับ baseline', xy=(ex['return_idx'], ex['baseline']),
                           fontsize=9, color='#059669', fontweight='bold',
                           xytext=(ex['return_idx']+1, ex['baseline']*1.8 if ex['baseline'] > 0 else 2),
                           arrowprops=dict(arrowstyle='->', color='#059669'))

    legend_elements = [
        mpatches.Patch(facecolor=C_BOUGHT, label='ซื้อ'),
        mpatches.Patch(facecolor=C_NOBUY, label='ไม่ซื้อ (0)'),
        mpatches.Patch(facecolor=C_NOREG, label='ยังไม่สมัคร'),
        Line2D([0], [0], marker='*', color=C_EVENT, markersize=15, linestyle='None', label='★ ถูกรางวัล'),
        Line2D([0], [0], color='#6B7280', linestyle=':', linewidth=2, label='Baseline'),
    ]

    summary = [f"ถูก 1 ครั้ง ไม่ถูกอีก 5+ งวด: {len(single_win_users):,} คน"]
    if non_none:
        summary.append(f"กลับ baseline เฉลี่ย {np.mean(non_none):.1f} งวด")
    if total_analyzable > 0:
        summary.append(f"ภายใน 3 งวด: {return_within[3]/total_analyzable*100:.0f}% | 6 งวด: {return_within[6]/total_analyzable*100:.0f}%")

    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))
    fig.text(0.5, 0.04, ' | '.join(summary), ha='center', fontsize=11,
             style='italic', color='#374151')
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    out = CHART_DIR / 'insight_03_win_decay.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    flush_print(f"\n  บันทึก: {out}")


# ═══════════════════════════════════════════════════════════════
# INSIGHT 4: Cancel แล้วสั่งใหม่สำเร็จ → ซื้อต่อเนื่องมั้ย
# ═══════════════════════════════════════════════════════════════
def insight_04(churn_df, item_cols, tx_data):
    flush_print("\n" + "="*70)
    flush_print("Insight 4: Cancel แล้วสั่งใหม่สำเร็จ → ซื้อต่อเนื่องมั้ย")
    flush_print("="*70)

    # Map TX period_key to item col index
    col_to_pk = {col: parse_period_date(col) for col in item_cols}
    pk_to_col_idx = {pk: i for i, (col, pk) in enumerate(col_to_pk.items())}

    cts = tx_data[tx_data['cancel_then_success']]
    no_cancel = tx_data[~tx_data['has_cancel'] & tx_data['has_success']]

    flush_print(f"\n  Cancel แล้วสั่งใหม่: {cts['userNo'].nunique():,} คน")
    flush_print(f"  ไม่เคย Cancel: {no_cancel['userNo'].nunique():,} คน")

    churn_indexed = churn_df.set_index('userNo')
    item_vals = churn_indexed[item_cols]
    churn_user_set = set(churn_df['userNo'].values)

    def calc_continuation(user_groups, label, max_users=20000):
        """Calculate continuation rate for a group of users."""
        rates = []
        count = 0
        for user, sub in user_groups:
            if count >= max_users:
                break
            if user not in churn_user_set or user not in item_vals.index:
                count += 1
                continue
            vals = item_vals.loc[user].values.astype(float)
            # First period for this user
            first_pk = sub['period_key'].iloc[0]
            if first_pk not in pk_to_col_idx:
                count += 1
                continue
            idx = pk_to_col_idx[first_pk]
            post = vals[idx+1:]
            valid = post[~np.isnan(post)]
            if len(valid) > 0:
                rates.append(np.sum(valid > 0) / len(valid))
            count += 1
        return rates

    cts_groups = cts.groupby('userNo')
    nc_groups = no_cancel.groupby('userNo')

    flush_print("  คำนวณ continuation rates...")
    cts_rates = calc_continuation(iter(cts_groups), "CTS")
    nc_rates = calc_continuation(iter(nc_groups), "NC")

    if cts_rates and nc_rates:
        avg_cts = np.mean(cts_rates) * 100
        avg_nc = np.mean(nc_rates) * 100
        flush_print(f"\n  อัตราซื้อต่อเนื่อง (% งวดที่ซื้อหลังเหตุการณ์):")
        flush_print(f"    Cancel แล้วสั่งใหม่: {avg_cts:.1f}%")
        flush_print(f"    ไม่เคย Cancel:      {avg_nc:.1f}%")
        diff = avg_cts - avg_nc
        flush_print(f"    ผลต่าง: {diff:+.1f}% {'(ซื้อต่อเนื่องกว่า!)' if diff > 0 else '(ซื้อน้อยกว่า)'}")

    # Find 3 examples
    cts_user_periods = cts.groupby('userNo')['period_key'].apply(set).to_dict()
    examples = []
    for user, periods in cts_user_periods.items():
        if len(examples) >= 3:
            break
        if user not in churn_user_set or user not in item_vals.index:
            continue
        vals = item_vals.loc[user].values.astype(float)
        active = np.sum((~np.isnan(vals)) & (vals > 0))
        if active < 8:
            continue

        cancel_indices = []
        for pk in periods:
            if pk in pk_to_col_idx:
                cancel_indices.append(pk_to_col_idx[pk])
        if cancel_indices:
            total_valid = np.sum(~np.isnan(vals))
            examples.append({
                'userNo': user,
                'cancel_indices': sorted(cancel_indices),
                'active': int(active),
                'total': int(total_valid)
            })

    for i, ex in enumerate(examples):
        cancel_labels = [parse_period_label(item_cols[ci]) for ci in ex['cancel_indices']]
        flush_print(f"\n  ตัวอย่าง {i+1}: {ex['userNo']}")
        flush_print(f"    Cancel+Retry ในงวด: {', '.join(cancel_labels)}")
        flush_print(f"    ซื้อต่อเนื่อง: {ex['active']}/{ex['total']} งวด ({ex['active']/ex['total']*100:.0f}%)")

    # Chart
    fig, axes = plt.subplots(3, 1, figsize=(20, 18))
    fig.suptitle('Insight 4: Cancel แล้วสั่งใหม่สำเร็จ → ซื้อต่อเนื่องมั้ย',
                 fontsize=16, fontweight='bold', y=0.98)

    for i, ex in enumerate(examples):
        user = ex['userNo']
        row = churn_df[churn_df['userNo'] == user].iloc[0]
        info = (f"{user} | อายุ: {row.get('age', '?')} | Cancel+Retry: {len(ex['cancel_indices'])} งวด | "
                f"ซื้อต่อเนื่อง: {ex['active']}/{ex['total']} งวด ({ex['active']/ex['total']*100:.0f}%)")
        plot_customer_bar(axes[i], row, item_cols, cancel_periods=ex['cancel_indices'],
                         user_info_text=info)

    legend_elements = [
        mpatches.Patch(facecolor=C_BOUGHT, label='ซื้อ'),
        mpatches.Patch(facecolor=C_NOBUY, label='ไม่ซื้อ (0)'),
        mpatches.Patch(facecolor=C_NOREG, label='ยังไม่สมัคร'),
        Line2D([0], [0], marker='X', color=C_CANCEL, markersize=12, linestyle='None', label='✕ Cancel+Retry'),
    ]
    summary = [f"Cancel แล้วสั่งใหม่: {cts['userNo'].nunique():,} คน"]
    if cts_rates and nc_rates:
        summary.append(f"ซื้อต่อเนื่อง: {np.mean(cts_rates)*100:.1f}% vs ไม่ Cancel: {np.mean(nc_rates)*100:.1f}%")

    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))
    fig.text(0.5, 0.04, ' | '.join(summary), ha='center', fontsize=11,
             style='italic', color='#374151')
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    out = CHART_DIR / 'insight_04_cancel_retry.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    flush_print(f"\n  บันทึก: {out}")


# ═══════════════════════════════════════════════════════════════
# INSIGHT 5: Cancel แล้วไม่สั่งใหม่ → churn ภายในกี่งวด
# ═══════════════════════════════════════════════════════════════
def insight_05(churn_df, item_cols, tx_data):
    flush_print("\n" + "="*70)
    flush_print("Insight 5: Cancel แล้วไม่สั่งใหม่ → churn ภายในกี่งวด")
    flush_print("="*70)

    col_to_pk = {col: parse_period_date(col) for col in item_cols}
    pk_to_col_idx = {pk: i for i, (col, pk) in enumerate(col_to_pk.items())}

    cancel_only = tx_data[tx_data['cancel_only']]
    success_only = tx_data[~tx_data['has_cancel'] & tx_data['has_success']]

    flush_print(f"\n  Cancel ไม่สั่งใหม่: {cancel_only['userNo'].nunique():,} คน")
    flush_print(f"  สั่งสำเร็จ (ไม่ cancel): {success_only['userNo'].nunique():,} คน")

    churn_indexed = churn_df.set_index('userNo')
    item_vals = churn_indexed[item_cols]
    churn_user_set = set(churn_df['userNo'].values)

    def measure_churn_time(user_groups, max_users=20000):
        """Measure periods until churn (2+ consecutive zeros) for user group."""
        times = []
        count = 0
        for user, sub in user_groups:
            if count >= max_users:
                break
            if user not in churn_user_set or user not in item_vals.index:
                count += 1
                continue
            vals = item_vals.loc[user].values.astype(float)
            first_pk = sub['period_key'].iloc[0]
            if first_pk not in pk_to_col_idx:
                count += 1
                continue
            idx = pk_to_col_idx[first_pk]
            # Find 2 consecutive zeros after event
            consec_zero = 0
            for offset in range(1, len(vals) - idx):
                v = vals[idx + offset]
                if np.isnan(v):
                    continue
                if v == 0:
                    consec_zero += 1
                    if consec_zero >= 2:
                        times.append(offset - 1)
                        break
                else:
                    consec_zero = 0
            count += 1
        return times

    flush_print("  คำนวณ churn timing...")
    co_groups = cancel_only.groupby('userNo')
    sl_groups = success_only.groupby('userNo')

    co_times = measure_churn_time(iter(co_groups))
    sl_times = measure_churn_time(iter(sl_groups))

    if co_times:
        flush_print(f"\n  ระยะเวลาจนถึง Churn:")
        flush_print(f"    Cancel ไม่สั่งใหม่: เฉลี่ย {np.mean(co_times):.1f} งวด (median: {np.median(co_times):.0f})")
        for t in [1, 3, 5]:
            pct = sum(1 for x in co_times if x <= t) / len(co_times) * 100
            flush_print(f"    Churn ภายใน {t} งวด: {pct:.1f}%")
    if sl_times:
        flush_print(f"    สั่งสำเร็จ: เฉลี่ย {np.mean(sl_times):.1f} งวด (median: {np.median(sl_times):.0f})")
        if co_times:
            diff = np.mean(co_times) - np.mean(sl_times)
            flush_print(f"    ผลต่าง: {diff:+.1f} งวด {'(Cancel-only churn เร็วกว่า)' if diff < 0 else ''}")

    # Find 3 examples
    co_user_periods = cancel_only.groupby('userNo')['period_key'].apply(set).to_dict()
    examples = []
    for user, periods in co_user_periods.items():
        if len(examples) >= 3:
            break
        if user not in churn_user_set or user not in item_vals.index:
            continue
        vals = item_vals.loc[user].values.astype(float)
        active = np.sum((~np.isnan(vals)) & (vals > 0))
        if active < 5:
            continue

        cancel_indices = []
        for pk in periods:
            if pk in pk_to_col_idx:
                cancel_indices.append(pk_to_col_idx[pk])
        if not cancel_indices:
            continue

        last_cancel_idx = max(cancel_indices)
        post = vals[last_cancel_idx+1:]
        valid = post[~np.isnan(post)]
        active_after = int(np.sum(valid > 0))
        total_after = len(valid)
        if total_after >= 2 and active_after / max(total_after, 1) < 0.5:
            examples.append({
                'userNo': user,
                'cancel_indices': sorted(cancel_indices),
                'last_cancel_idx': last_cancel_idx,
                'active_after': active_after,
                'total_after': total_after,
            })

    for i, ex in enumerate(examples):
        cancel_labels = [parse_period_label(item_cols[ci]) for ci in ex['cancel_indices']]
        flush_print(f"\n  ตัวอย่าง {i+1}: {ex['userNo']}")
        flush_print(f"    Cancel (ไม่สั่งใหม่) ในงวด: {', '.join(cancel_labels)}")
        flush_print(f"    Last cancel: {parse_period_label(item_cols[ex['last_cancel_idx']])}")
        flush_print(f"    หลัง cancel: ซื้อ {ex['active_after']}/{ex['total_after']} งวด")

    # Chart
    fig, axes = plt.subplots(3, 1, figsize=(20, 18))
    fig.suptitle('Insight 5: Cancel แล้วไม่สั่งใหม่ → churn ภายในกี่งวด',
                 fontsize=16, fontweight='bold', y=0.98)

    for i, ex in enumerate(examples):
        user = ex['userNo']
        row = churn_df[churn_df['userNo'] == user].iloc[0]
        info = (f"{user} | Cancel(ไม่สั่งใหม่): {len(ex['cancel_indices'])} งวด | "
                f"Last cancel: {parse_period_label(item_cols[ex['last_cancel_idx']])} | "
                f"หลัง cancel ซื้อ {ex['active_after']}/{ex['total_after']} งวด")
        event_periods = [ex['last_cancel_idx']]
        other_cancels = [ci for ci in ex['cancel_indices'] if ci != ex['last_cancel_idx']]
        plot_customer_bar(axes[i], row, item_cols, event_periods=event_periods,
                         cancel_periods=other_cancels, user_info_text=info)

    legend_elements = [
        mpatches.Patch(facecolor=C_BOUGHT, label='ซื้อ'),
        mpatches.Patch(facecolor=C_NOBUY, label='ไม่ซื้อ (0)'),
        mpatches.Patch(facecolor=C_NOREG, label='ยังไม่สมัคร'),
        Line2D([0], [0], marker='*', color=C_EVENT, markersize=15, linestyle='None', label='★ Last Cancel'),
        Line2D([0], [0], marker='X', color=C_CANCEL, markersize=12, linestyle='None', label='✕ Cancel อื่น'),
    ]
    summary = [f"Cancel ไม่สั่งใหม่: {cancel_only['userNo'].nunique():,} คน"]
    if co_times:
        summary.append(f"Churn เฉลี่ย {np.mean(co_times):.1f} งวด")
        within3 = sum(1 for t in co_times if t <= 3) / len(co_times) * 100
        summary.append(f"Churn ภายใน 3 งวด: {within3:.0f}%")
    if sl_times and co_times:
        summary.append(f"vs สั่งสำเร็จ: {np.mean(sl_times):.1f} งวด")

    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))
    fig.text(0.5, 0.04, ' | '.join(summary), ha='center', fontsize=11,
             style='italic', color='#374151')
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    out = CHART_DIR / 'insight_05_cancel_churn.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    flush_print(f"\n  บันทึก: {out}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    flush_print("="*70)
    flush_print("Insights #2-5: Purchase Behavior Analysis")
    flush_print("="*70)

    flush_print("\nโหลดข้อมูล Churn...")
    churn_df, item_cols = load_churn_history()
    flush_print(f"  {len(churn_df):,} ลูกค้า, {len(item_cols)} งวด")

    flush_print("\nโหลดข้อมูล Prize...")
    prize_df = load_prize_data()

    flush_print("\nโหลดข้อมูล TX...")
    tx_data = load_tx_cancel_info()

    insight_02(churn_df, item_cols, prize_df)
    insight_03(churn_df, item_cols, prize_df)
    insight_04(churn_df, item_cols, tx_data)
    insight_05(churn_df, item_cols, tx_data)

    flush_print("\n" + "="*70)
    flush_print("เสร็จสิ้น Insights #2-5!")
    flush_print("="*70)
    flush_print("\nCharts saved:")
    for c in ['insight_02_multi_win.png', 'insight_03_win_decay.png',
              'insight_04_cancel_retry.png', 'insight_05_cancel_churn.png']:
        flush_print(f"  charts/{c}")

if __name__ == '__main__':
    main()
