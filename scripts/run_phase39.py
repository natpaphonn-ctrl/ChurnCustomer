#!/usr/bin/env python3
"""
Phase 39: Prize Impact Deep Dive
=================================
วิเคราะห์เชิงลึกว่าการถูกรางวัลลอตเตอรี่ส่งผลต่อพฤติกรรมลูกค้าและ Churn อย่างไร

Analyses:
1. Post-Win Behavior Change (ซื้อเพิ่มหลังถูกรางวัล?)
2. Prize Tier Impact on Churn (ระดับรางวัลกับ Churn)
3. Win Frequency Analysis (ถูกบ่อยแค่ไหน)
4. Timing Analysis (ถูกรางวัลช่วงไหนของ lifecycle)
5. Revenue Impact (ลูกค้าถูกรางวัลสร้าง revenue มากแค่ไหน)

Run: python3 scripts/run_phase39.py
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
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
PRIZE_DIR = 'data/prize'
CHURN_DIR = 'data/churn'
CHART_OUT = 'charts/phase39_prize_impact.png'

PRIZE_AMOUNTS = {
    'LAST2DIGITS': 2_000, 'LAST3DIGITS': 4_000, 'FIRST3DIGITS': 4_000,
    'FIFTH': 20_000, 'FOURTH': 40_000, 'THIRD': 80_000,
    'NEARBY': 100_000, 'SECOND': 200_000, 'FIRST': 6_000_000,
}

TIER_ORDER_MAP = {
    'LAST2DIGITS': 0, 'LAST3DIGITS': 1, 'FIRST3DIGITS': 2,
    'FIFTH': 3, 'FOURTH': 4, 'THIRD': 5,
    'NEARBY': 6, 'SECOND': 7, 'FIRST': 8,
}

TIER_GROUP = {
    'LAST2DIGITS': 'Small (2K-4K)', 'LAST3DIGITS': 'Small (2K-4K)',
    'FIRST3DIGITS': 'Small (2K-4K)', 'FIFTH': 'Medium (20K-40K)',
    'FOURTH': 'Medium (20K-40K)', 'THIRD': 'Large (80K-200K)',
    'NEARBY': 'Large (80K-200K)', 'SECOND': 'Large (80K-200K)',
    'FIRST': 'Jackpot (6M)',
}

TIER_GROUP_TH = {
    'Small (2K-4K)': 'รางวัลเล็ก\n(2K-4K)',
    'Medium (20K-40K)': 'รางวัลกลาง\n(20K-40K)',
    'Large (80K-200K)': 'รางวัลใหญ่\n(80K-200K)',
    'Jackpot (6M)': 'แจ็คพอต\n(6M)',
}


def load_all_prizes():
    """Load and combine all 30 prize files."""
    files = sorted(glob.glob(os.path.join(PRIZE_DIR, 'Prize_*.csv')))
    print(f"  พบไฟล์ Prize {len(files)} ไฟล์")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df['rounddate'] = pd.to_datetime(df['rounddate'])
        dfs.append(df)
    prizes = pd.concat(dfs, ignore_index=True)
    print(f"  รวม {len(prizes):,} แถว, {prizes['userNo'].nunique():,} ลูกค้าที่ถูกรางวัล")
    return prizes


def load_latest_churn():
    """Load the latest churn file."""
    files = sorted(glob.glob(os.path.join(CHURN_DIR, 'Churn_*.csv')))
    files = [f for f in files if 'old_validation' not in f]
    latest = files[-1]
    print(f"  โหลด Churn file: {os.path.basename(latest)}")
    df = pd.read_csv(latest)
    print(f"  {len(df):,} ลูกค้า")
    return df


def get_item_columns(churn_df):
    """Extract item columns and their dates."""
    item_cols = [c for c in churn_df.columns if c.startswith('item')]
    dates = []
    for c in item_cols:
        parts = c.replace('item', '').split('_')
        dates.append(pd.Timestamp(f"{parts[0]}-{parts[1]}-{parts[2]}"))
    return item_cols, dates


def build_rounddate_to_colidx(item_cols, dates):
    """Map each rounddate to item_col index."""
    mapping = {}
    dates_arr = np.array(dates)
    return item_cols, dates_arr


# ─────────────────────────────────────────
# VECTORIZED: Post-Win Behavior Analysis
# ─────────────────────────────────────────
def analyze_post_win_behavior(prizes, churn_df, item_cols, dates):
    """Vectorized: compare purchase BEFORE vs AFTER winning."""
    print("\n" + "="*70)
    print("  Analysis 1: Post-Win Behavior Change")
    print("="*70)

    dates_ts = pd.DatetimeIndex(dates)

    # Map each prize rounddate to the closest item_col index
    def find_col_idx(rd):
        diffs = np.abs((dates_ts - rd).days)
        best = np.argmin(diffs)
        return int(best) if diffs[best] <= 5 else -1

    unique_rds = prizes['rounddate'].unique()
    rd_to_idx = {rd: find_col_idx(rd) for rd in unique_rds}

    # Get first win per user
    user_first_win = prizes.groupby('userNo')['rounddate'].min().reset_index()
    user_first_win.columns = ['userNo', 'first_win_date']
    user_first_win['win_col_idx'] = user_first_win['first_win_date'].map(rd_to_idx)
    user_first_win = user_first_win[user_first_win['win_col_idx'] >= 0]

    # Merge with churn
    merged = churn_df[['userNo'] + item_cols].merge(user_first_win, on='userNo', how='inner')
    print(f"  ลูกค้าที่ match: {len(merged):,}")

    # Build item matrix (users x periods)
    item_matrix = merged[item_cols].fillna(0).values  # shape (n_users, n_periods)
    win_indices = merged['win_col_idx'].values.astype(int)

    n_users = len(merged)
    n_periods = len(item_cols)

    # Vectorized before/after: use 3 periods before and after
    # For each user, compute avg of [win_idx-3:win_idx] and [win_idx+1:win_idx+4]
    before_sum = np.zeros(n_users)
    before_cnt = np.zeros(n_users)
    after_sum = np.zeros(n_users)
    after_cnt = np.zeros(n_users)
    win_val = np.zeros(n_users)

    for offset in range(-3, 0):
        idx = win_indices + offset
        valid = (idx >= 0) & (idx < n_periods)
        vals = np.where(valid, item_matrix[np.arange(n_users), np.clip(idx, 0, n_periods-1)], 0)
        before_sum += vals
        before_cnt += valid.astype(float)

    for offset in range(1, 4):
        idx = win_indices + offset
        valid = (idx >= 0) & (idx < n_periods)
        vals = np.where(valid, item_matrix[np.arange(n_users), np.clip(idx, 0, n_periods-1)], 0)
        after_sum += vals
        after_cnt += valid.astype(float)

    # Win period value
    valid_win = (win_indices >= 0) & (win_indices < n_periods)
    win_val = np.where(valid_win, item_matrix[np.arange(n_users), np.clip(win_indices, 0, n_periods-1)], 0)

    # Filter users with both before and after
    has_both = (before_cnt > 0) & (after_cnt > 0)
    before_avg = np.where(has_both, before_sum / np.maximum(before_cnt, 1), np.nan)
    after_avg = np.where(has_both, after_sum / np.maximum(after_cnt, 1), np.nan)

    valid_mask = has_both
    n_valid = valid_mask.sum()

    avg_before = np.nanmean(before_avg[valid_mask])
    avg_win = np.nanmean(win_val[valid_mask])
    avg_after = np.nanmean(after_avg[valid_mask])
    pct_change = (avg_after - avg_before) / avg_before * 100 if avg_before > 0 else 0

    print(f"  ลูกค้าที่วิเคราะห์ได้: {n_valid:,}")
    print(f"  เฉลี่ยตั๋ว/งวด ก่อนถูกรางวัล: {avg_before:.2f}")
    print(f"  เฉลี่ยตั๋ว/งวด ช่วงถูกรางวัล: {avg_win:.2f}")
    print(f"  เฉลี่ยตั๋ว/งวด หลังถูกรางวัล: {avg_after:.2f}")
    print(f"  เปลี่ยนแปลง: {pct_change:+.1f}%")

    return merged, win_indices, item_matrix, rd_to_idx


# ─────────────────────────────────────────
# VECTORIZED: Winning Effect Duration
# ─────────────────────────────────────────
def analyze_winning_effect_duration(merged, win_indices, item_matrix, item_cols):
    """Track avg tickets at each offset from win period."""
    print("\n" + "="*70)
    print("  Analysis 1b: Winning Effect Duration")
    print("="*70)

    n_users = len(merged)
    n_periods = len(item_cols)
    max_offset = 6

    offsets = list(range(-max_offset, max_offset + 1))
    means = []

    for offset in offsets:
        idx = win_indices + offset
        valid = (idx >= 0) & (idx < n_periods)
        if valid.sum() == 0:
            means.append(0)
            continue
        vals = item_matrix[np.arange(n_users), np.clip(idx, 0, n_periods-1)]
        vals = np.where(valid, vals, np.nan)
        means.append(np.nanmean(vals))

    win_pos = offsets.index(0)
    pre_avg = np.mean(means[:win_pos])
    for i, offset in enumerate(offsets):
        label = f"  งวด {offset:+d}: {means[i]:.2f} ตั๋ว/งวด"
        if offset == 0:
            label += " <-- WIN"
        print(label)
    print(f"\n  เฉลี่ยก่อนถูก: {pre_avg:.2f}")
    print(f"  เฉลี่ยหลังถูก: {np.mean(means[win_pos+1:]):.2f}")

    return offsets, means


# ─────────────────────────────────────────
# VECTORIZED: Post-Win by Prize Tier
# ─────────────────────────────────────────
def analyze_post_win_by_tier(prizes, churn_df, item_cols, dates, rd_to_idx):
    """Break down post-win behavior by prize tier."""
    print("\n" + "="*70)
    print("  Analysis 1c: Post-Win by Prize Tier")
    print("="*70)

    n_periods = len(item_cols)
    dates_arr = np.array(dates)

    # Best (highest) prize per user
    prizes_ext = prizes.copy()
    prizes_ext['tier_order'] = prizes_ext['type'].map(TIER_ORDER_MAP)
    prizes_ext['tier_group'] = prizes_ext['type'].map(TIER_GROUP)

    # For each user, get the win event with highest tier
    best_idx = prizes_ext.groupby('userNo')['tier_order'].idxmax()
    user_best = prizes_ext.loc[best_idx, ['userNo', 'rounddate', 'tier_group']].copy()
    user_best['win_col_idx'] = user_best['rounddate'].map(rd_to_idx)
    user_best = user_best[user_best['win_col_idx'] >= 0]

    merged = churn_df[['userNo'] + item_cols].merge(user_best, on='userNo', how='inner')
    item_matrix = merged[item_cols].fillna(0).values
    win_indices = merged['win_col_idx'].values.astype(int)
    n_users = len(merged)

    # Compute before/after per user
    before_sum = np.zeros(n_users)
    before_cnt = np.zeros(n_users)
    after_sum = np.zeros(n_users)
    after_cnt = np.zeros(n_users)

    for offset in range(-3, 0):
        idx = win_indices + offset
        valid = (idx >= 0) & (idx < n_periods)
        vals = np.where(valid, item_matrix[np.arange(n_users), np.clip(idx, 0, n_periods-1)], 0)
        before_sum += vals
        before_cnt += valid.astype(float)

    for offset in range(1, 4):
        idx = win_indices + offset
        valid = (idx >= 0) & (idx < n_periods)
        vals = np.where(valid, item_matrix[np.arange(n_users), np.clip(idx, 0, n_periods-1)], 0)
        after_sum += vals
        after_cnt += valid.astype(float)

    has_both = (before_cnt > 0) & (after_cnt > 0)
    merged_valid = merged[has_both].copy()
    merged_valid['avg_before'] = (before_sum[has_both] / np.maximum(before_cnt[has_both], 1))
    merged_valid['avg_after'] = (after_sum[has_both] / np.maximum(after_cnt[has_both], 1))

    tier_results = {}
    for tier in ['Small (2K-4K)', 'Medium (20K-40K)', 'Large (80K-200K)', 'Jackpot (6M)']:
        subset = merged_valid[merged_valid['tier_group'] == tier]
        if len(subset) < 5:
            continue
        avg_b = subset['avg_before'].mean()
        avg_a = subset['avg_after'].mean()
        pct = (avg_a - avg_b) / avg_b * 100 if avg_b > 0 else 0
        tier_results[tier] = {'before': avg_b, 'after': avg_a, 'pct_change': pct, 'n': len(subset)}
        print(f"  {tier}: ก่อน {avg_b:.1f} -> หลัง {avg_a:.1f} ({pct:+.1f}%, n={len(subset):,})")

    return tier_results


# ─────────────────────────────────────────
# Analysis 2: Prize Tier vs Churn
# ─────────────────────────────────────────
def analyze_prize_tier_churn(prizes, churn_df, item_cols):
    """Churn rate by prize type."""
    print("\n" + "="*70)
    print("  Analysis 2: Prize Tier Impact on Churn")
    print("="*70)

    last_col = item_cols[-1]
    second_last_col = item_cols[-2]

    active = churn_df[churn_df[second_last_col] > 0][['userNo', last_col, second_last_col]].copy()
    active['churned'] = (active[last_col].fillna(0) == 0).astype(int)
    overall_churn = active['churned'].mean()
    print(f"\n  Churn rate รวม: {overall_churn:.1%} (n={len(active):,})")

    # User's best prize type
    prizes_ext = prizes.copy()
    prizes_ext['tier_order'] = prizes_ext['type'].map(TIER_ORDER_MAP)
    best_idx = prizes_ext.groupby('userNo')['tier_order'].idxmax()
    user_best_type = prizes_ext.loc[best_idx, ['userNo', 'type']].copy()

    # User stats
    user_stats = prizes.groupby('userNo').agg(
        total_wins=('count_prize', 'sum'),
        total_prize_amount=('total_prize', 'sum'),
        n_win_rounds=('rounddate', 'nunique'),
    ).reset_index()

    merged = active.merge(user_best_type, on='userNo', how='left')
    merged = merged.merge(user_stats, on='userNo', how='left')
    merged['is_winner'] = merged['type'].notna()

    w_churn = merged[merged['is_winner']]['churned'].mean()
    nw_churn = merged[~merged['is_winner']]['churned'].mean()
    print(f"  Winner churn: {w_churn:.1%} (n={merged['is_winner'].sum():,})")
    print(f"  Non-winner churn: {nw_churn:.1%} (n={(~merged['is_winner']).sum():,})")
    print(f"  Gap: {(nw_churn - w_churn):.1%}")

    # By type
    type_churn = {}
    print(f"\n  {'Type':<18} {'Churn':>8} {'Count':>8}")
    print(f"  {'-'*36}")
    for ptype in ['LAST2DIGITS', 'LAST3DIGITS', 'FIRST3DIGITS', 'FIFTH',
                   'FOURTH', 'THIRD', 'NEARBY', 'SECOND', 'FIRST']:
        subset = merged[merged['type'] == ptype]
        if len(subset) >= 3:
            cr = subset['churned'].mean()
            type_churn[ptype] = {'churn_rate': cr, 'count': len(subset)}
            print(f"  {ptype:<18} {cr:>8.1%} {len(subset):>8,}")

    # Multi small vs single big
    small_types = ['LAST2DIGITS', 'LAST3DIGITS', 'FIRST3DIGITS']
    multi_small = merged[(merged['type'].isin(small_types)) & (merged['n_win_rounds'] >= 3)]
    single_big = merged[
        (merged['type'].isin(['THIRD', 'NEARBY', 'SECOND', 'FIRST'])) &
        (merged['n_win_rounds'] == 1)
    ]
    ms_churn = multi_small['churned'].mean() if len(multi_small) > 0 else None
    sb_churn = single_big['churned'].mean() if len(single_big) > 0 else None
    print(f"\n  ถูกเล็กๆ >=3 งวด: Churn {ms_churn:.1%} (n={len(multi_small):,})" if ms_churn is not None else "")
    print(f"  ถูกใหญ่ 1 ครั้ง:  Churn {sb_churn:.1%} (n={len(single_big):,})" if sb_churn is not None else "")

    return type_churn, overall_churn, ms_churn, sb_churn


# ─────────────────────────────────────────
# Analysis 3: Win Frequency
# ─────────────────────────────────────────
def analyze_win_frequency(prizes, churn_df, item_cols):
    """Win frequency distribution and churn relationship."""
    print("\n" + "="*70)
    print("  Analysis 3: Win Frequency")
    print("="*70)

    user_freq = prizes.groupby('userNo').agg(
        n_win_rounds=('rounddate', 'nunique'),
        total_wins=('count_prize', 'sum'),
        total_prize=('total_prize', 'sum'),
    ).reset_index()

    print(f"\n  ลูกค้าที่ถูกรางวัล: {len(user_freq):,}")
    vc = user_freq['n_win_rounds'].value_counts().sort_index()
    print(f"  Distribution (n_win_rounds):")
    for val, cnt in vc.head(15).items():
        print(f"    {val:>3} งวด: {cnt:>6,} ({cnt/len(user_freq)*100:.1f}%)")

    # Churn by frequency
    last_col = item_cols[-1]
    second_last_col = item_cols[-2]
    active = churn_df[churn_df[second_last_col] > 0][['userNo', last_col]].copy()
    active['churned'] = (active[last_col].fillna(0) == 0).astype(int)
    merged = active.merge(user_freq[['userNo', 'n_win_rounds']], on='userNo', how='left')
    merged['n_win_rounds'] = merged['n_win_rounds'].fillna(0).astype(int)

    bins = [-1, 0, 1, 2, 5, 10, 999]
    labels = ['0 (never)', '1 round', '2 rounds', '3-5 rounds', '6-10 rounds', '11+ rounds']
    merged['win_freq_bin'] = pd.cut(merged['n_win_rounds'], bins=bins, labels=labels)

    freq_churn = merged.groupby('win_freq_bin', observed=False).agg(
        churn_rate=('churned', 'mean'),
        count=('userNo', 'count'),
    ).reset_index()

    print(f"\n  {'Frequency':<15} {'Churn':>8} {'Count':>10}")
    print(f"  {'-'*35}")
    for _, r in freq_churn.iterrows():
        print(f"  {str(r['win_freq_bin']):<15} {r['churn_rate']:>8.1%} {r['count']:>10,}")

    return freq_churn


# ─────────────────────────────────────────
# Analysis 4: Timing / Recency
# ─────────────────────────────────────────
def analyze_timing(prizes, churn_df, item_cols, dates):
    """Win recency vs churn (honeymoon effect)."""
    print("\n" + "="*70)
    print("  Analysis 4: Timing Analysis (Recency & Honeymoon)")
    print("="*70)

    last_col = item_cols[-1]
    second_last_col = item_cols[-2]
    ref_date = dates[-1]

    active = churn_df[churn_df[second_last_col] > 0][['userNo', last_col]].copy()
    active['churned'] = (active[last_col].fillna(0) == 0).astype(int)

    user_last_win = prizes.groupby('userNo')['rounddate'].max().reset_index()
    user_last_win.columns = ['userNo', 'last_win_date']
    user_first_win = prizes.groupby('userNo')['rounddate'].min().reset_index()
    user_first_win.columns = ['userNo', 'first_win_date']

    merged = active.merge(user_last_win, on='userNo', how='inner')
    merged = merged.merge(user_first_win, on='userNo', how='inner')
    merged['periods_since_last_win'] = ((ref_date - merged['last_win_date']).dt.days / 15).round().astype(int)

    bins = [-1, 1, 3, 6, 12, 999]
    labels = ['0-1 pds', '2-3 pds', '4-6 pds', '7-12 pds', '13+ pds']
    merged['recency_bin'] = pd.cut(merged['periods_since_last_win'], bins=bins, labels=labels)

    recency_churn = merged.groupby('recency_bin', observed=False).agg(
        churn_rate=('churned', 'mean'),
        count=('userNo', 'count'),
    ).reset_index()

    print(f"\n  {'Recency':<12} {'Churn':>8} {'Count':>10}")
    print(f"  {'-'*32}")
    for _, r in recency_churn.iterrows():
        print(f"  {str(r['recency_bin']):<12} {r['churn_rate']:>8.1%} {r['count']:>10,}")

    # Early vs late lifecycle winners
    tenure = churn_df[item_cols].notna().sum(axis=1)
    churn_df_t = churn_df[['userNo']].copy()
    churn_df_t['tenure'] = tenure.values
    merged2 = merged.merge(churn_df_t, on='userNo', how='left')
    merged2['first_win_periods_ago'] = ((ref_date - merged2['first_win_date']).dt.days / 15).round()
    merged2['win_lifecycle_pct'] = (merged2['tenure'] - merged2['first_win_periods_ago']) / merged2['tenure']
    merged2['win_lifecycle_pct'] = merged2['win_lifecycle_pct'].clip(0, 1)

    early = merged2[merged2['win_lifecycle_pct'] <= 0.33]
    late = merged2[merged2['win_lifecycle_pct'] > 0.66]
    if len(early) > 0 and len(late) > 0:
        print(f"\n  ถูกช่วงต้น lifecycle (<=33%): Churn {early['churned'].mean():.1%} (n={len(early):,})")
        print(f"  ถูกช่วงปลาย lifecycle (>66%): Churn {late['churned'].mean():.1%} (n={len(late):,})")

    return recency_churn


# ─────────────────────────────────────────
# Analysis 5: Revenue Impact
# ─────────────────────────────────────────
def analyze_revenue_impact(prizes, churn_df, item_cols):
    """Revenue comparison: winners vs non-winners."""
    print("\n" + "="*70)
    print("  Analysis 5: Revenue Impact")
    print("="*70)

    winners_set = set(prizes['userNo'].unique())
    is_winner = churn_df['userNo'].isin(winners_set)

    items = churn_df[item_cols].fillna(0)
    total_tickets = items.sum(axis=1)
    active_periods = (items > 0).sum(axis=1)

    w_tickets = total_tickets[is_winner].mean()
    nw_tickets = total_tickets[~is_winner].mean()
    w_active = active_periods[is_winner].mean()
    nw_active = active_periods[~is_winner].mean()
    w_per = w_tickets / w_active if w_active > 0 else 0
    nw_per = nw_tickets / nw_active if nw_active > 0 else 0

    print(f"\n  {'Metric':<28} {'Winners':>10} {'Non-Win':>10} {'Ratio':>7}")
    print(f"  {'-'*58}")
    print(f"  {'Avg total tickets':<28} {w_tickets:>10.1f} {nw_tickets:>10.1f} {w_tickets/nw_tickets:>7.1f}x")
    print(f"  {'Avg active periods':<28} {w_active:>10.1f} {nw_active:>10.1f} {w_active/nw_active:>7.1f}x")
    print(f"  {'Avg tickets/active period':<28} {w_per:>10.1f} {nw_per:>10.1f} {w_per/nw_per:>7.1f}x")

    total_payout = prizes['total_prize'].sum()
    winner_total = total_tickets[is_winner].sum()
    ticket_price = 120
    winner_rev = winner_total * ticket_price
    print(f"\n  Total prize payouts: {total_payout:,.0f} THB")
    print(f"  Winner total tickets: {winner_total:,.0f}")
    print(f"  Winner est. revenue (@120/ticket): {winner_rev:,.0f} THB")
    print(f"  Revenue / Payout ratio: {winner_rev/total_payout:.1f}x")

    return {
        'w_tickets': w_tickets, 'nw_tickets': nw_tickets,
        'w_active': w_active, 'nw_active': nw_active,
        'w_per_period': w_per, 'nw_per_period': nw_per,
        'total_payout': total_payout, 'winner_revenue': winner_rev,
    }


# ─────────────────────────────────────────
# Chart
# ─────────────────────────────────────────
def create_chart(type_churn, overall_churn, freq_churn, recency_churn,
                 offsets, means, tier_results, revenue, ms_churn, sb_churn):
    """Create 6-panel chart."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ─── Panel 1: Prize Type vs Churn ───
    ax1 = fig.add_subplot(gs[0, 0])
    types = list(type_churn.keys())
    rates = [type_churn[t]['churn_rate'] * 100 for t in types]
    counts = [type_churn[t]['count'] for t in types]
    bar_colors = ['#4CAF50' if r < overall_churn*100 else '#F44336' for r in rates]
    bars = ax1.bar(range(len(types)), rates, color=bar_colors, alpha=0.85, edgecolor='white')
    ax1.axhline(y=overall_churn*100, color='gray', ls='--', lw=1.5, label=f'Overall: {overall_churn:.1%}')
    ax1.set_xticks(range(len(types)))
    ax1.set_xticklabels(types, rotation=35, ha='right', fontsize=9)
    ax1.set_ylabel('Churn Rate (%)')
    ax1.set_title('Prize Type vs Churn Rate', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    for bar, cnt in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'n={cnt:,}', ha='center', va='bottom', fontsize=7.5)

    # ─── Panel 2: Before vs After by Tier ───
    ax2 = fig.add_subplot(gs[0, 1])
    tier_names = list(tier_results.keys())
    if tier_names:
        x = np.arange(len(tier_names))
        w = 0.35
        bv = [tier_results[t]['before'] for t in tier_names]
        av = [tier_results[t]['after'] for t in tier_names]
        ax2.bar(x - w/2, bv, w, label='Before Win', color='#90CAF9', edgecolor='white')
        ax2.bar(x + w/2, av, w, label='After Win', color='#2196F3', edgecolor='white')
        ax2.set_xticks(x)
        ax2.set_xticklabels([TIER_GROUP_TH.get(t, t) for t in tier_names], fontsize=9)
        ax2.set_ylabel('Avg Tickets/Period')
        ax2.set_title('Purchase Before vs After Win (by Tier)', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9)
        for i, t in enumerate(tier_names):
            pct = tier_results[t]['pct_change']
            ax2.text(x[i] + w/2, av[i] + 0.2, f'{pct:+.0f}%', ha='center', fontsize=9,
                    fontweight='bold', color='green' if pct > 0 else 'red')

    # ─── Panel 3: Win Frequency vs Churn ───
    ax3 = fig.add_subplot(gs[1, 0])
    fl = freq_churn['win_freq_bin'].astype(str).values
    fr = freq_churn['churn_rate'].values * 100
    fc = freq_churn['count'].values
    c3 = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(fl)))
    bars3 = ax3.bar(range(len(fl)), fr, color=c3, edgecolor='white', alpha=0.85)
    ax3.set_xticks(range(len(fl)))
    ax3.set_xticklabels(fl, fontsize=9)
    ax3.set_ylabel('Churn Rate (%)')
    ax3.set_title('Win Frequency vs Churn Rate', fontsize=13, fontweight='bold')
    for bar, cnt in zip(bars3, fc):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{cnt:,}', ha='center', fontsize=8)

    # ─── Panel 4: Winning Effect Duration ───
    ax4 = fig.add_subplot(gs[1, 1])
    c4 = ['#90CAF9' if o < 0 else '#F44336' if o == 0 else '#4CAF50' for o in offsets]
    ax4.bar(offsets, means, color=c4, edgecolor='white', alpha=0.85)
    ax4.axvline(x=0, color='red', ls='--', lw=1.5, alpha=0.5)
    ax4.set_xlabel('Periods Relative to Win (0 = win)')
    ax4.set_ylabel('Avg Tickets')
    ax4.set_title('Winning Effect Duration', fontsize=13, fontweight='bold')
    win_pos = offsets.index(0)
    pre_a = np.mean(means[:win_pos])
    post_a = np.mean(means[win_pos+1:])
    ax4.axhline(y=pre_a, color='#1565C0', ls=':', lw=1, alpha=0.7)
    ax4.axhline(y=post_a, color='#2E7D32', ls=':', lw=1, alpha=0.7)
    ax4.text(offsets[-1], pre_a, f'Pre: {pre_a:.1f}', va='bottom', ha='right', fontsize=9, color='#1565C0')
    ax4.text(offsets[-1], post_a, f'Post: {post_a:.1f}', va='bottom', ha='right', fontsize=9, color='#2E7D32')

    # ─── Panel 5: Win Recency vs Churn ───
    ax5 = fig.add_subplot(gs[2, 0])
    rl = recency_churn['recency_bin'].astype(str).values
    rr = recency_churn['churn_rate'].values * 100
    rc = recency_churn['count'].values
    c5 = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(rl)))
    bars5 = ax5.bar(range(len(rl)), rr, color=c5, edgecolor='white', alpha=0.85)
    ax5.set_xticks(range(len(rl)))
    ax5.set_xticklabels(rl, fontsize=9)
    ax5.set_ylabel('Churn Rate (%)')
    ax5.set_title('Win Recency vs Churn (Honeymoon Effect)', fontsize=13, fontweight='bold')
    for bar, cnt in zip(bars5, rc):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{cnt:,}', ha='center', fontsize=8)

    # ─── Panel 6: Key Insights ───
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    rev_r = revenue['w_tickets'] / revenue['nw_tickets']
    act_r = revenue['w_active'] / revenue['nw_active']
    pay_r = revenue['winner_revenue'] / revenue['total_payout']
    pct_ch = (post_a - pre_a) / pre_a * 100 if pre_a > 0 else 0

    lines = [
        ("Phase 39: Prize Impact — Key Insights", 15, 'bold', '#1a1a1a'),
        ("", 8, 'normal', 'black'),
        ("Revenue Impact:", 12, 'bold', '#1565C0'),
        (f"  Winners buy {rev_r:.1f}x more tickets", 11, 'normal', '#333'),
        (f"  Winners active {act_r:.1f}x more periods", 11, 'normal', '#333'),
        (f"  Revenue from winners = {pay_r:.0f}x prize payouts", 11, 'normal', '#333'),
        ("", 8, 'normal', 'black'),
        ("Churn Impact:", 12, 'bold', '#2E7D32'),
        (f"  Overall churn: {overall_churn:.1%}", 11, 'normal', '#333'),
    ]
    if type_churn:
        min_t = min(type_churn, key=lambda t: type_churn[t]['churn_rate'])
        max_t = max(type_churn, key=lambda t: type_churn[t]['churn_rate'])
        lines.append((f"  Best: {min_t} ({type_churn[min_t]['churn_rate']:.1%})", 11, 'normal', '#333'))
        lines.append((f"  Worst: {max_t} ({type_churn[max_t]['churn_rate']:.1%})", 11, 'normal', '#333'))
    lines.append(("", 8, 'normal', 'black'))
    lines.append(("Winning Effect:", 12, 'bold', '#FF6F00'))
    lines.append((f"  Post-win purchase: {pct_ch:+.1f}%", 11, 'normal', '#333'))
    if ms_churn is not None and sb_churn is not None:
        lines.append((f"  Multi-small wins: {ms_churn:.1%}", 11, 'normal', '#333'))
        lines.append((f"  Single big win: {sb_churn:.1%}", 11, 'normal', '#333'))
        better = "Multi-small" if ms_churn < sb_churn else "Single-big"
        lines.append((f"  Better retention: {better}", 11, 'normal', '#333'))

    y = 0.95
    for text, sz, wt, clr in lines:
        ax6.text(0.05, y, text, transform=ax6.transAxes, fontsize=sz,
                fontweight=wt, color=clr, va='top', fontfamily='monospace')
        y -= 0.055

    ax6.patch.set_facecolor('#FAFAFA')
    ax6.patch.set_edgecolor('#BDBDBD')
    ax6.patch.set_linewidth(1.5)

    fig.suptitle('Phase 39: Prize Impact Deep Dive', fontsize=16, fontweight='bold', y=0.98)
    os.makedirs('charts', exist_ok=True)
    plt.savefig(CHART_OUT, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Chart saved: {CHART_OUT}")


# ─────────────────────────────────────────
# Recommendations (Thai)
# ─────────────────────────────────────────
def print_recommendations(type_churn, overall_churn, freq_churn, recency_churn,
                          tier_results, revenue, ms_churn, sb_churn, offsets, means):
    """Business recommendations in Thai."""
    print("\n" + "="*70)
    print("  ข้อเสนอแนะเชิงธุรกิจ (Business Recommendations)")
    print("="*70)

    win_pos = offsets.index(0)
    pre_a = np.mean(means[:win_pos])
    post_a = np.mean(means[win_pos+1:])
    pct_ch = (post_a - pre_a) / pre_a * 100 if pre_a > 0 else 0

    rev_r = revenue['w_tickets'] / revenue['nw_tickets']
    pay_r = revenue['winner_revenue'] / revenue['total_payout']

    rec_rates = recency_churn['churn_rate'].values

    print(f"""
  1. ผู้ถูกรางวัลมี Churn ต่ำกว่าผู้ไม่ถูกอย่างมีนัยสำคัญ
     -> ควรส่ง notification แสดงความยินดีทันทีเมื่อถูกรางวัล
     -> ใช้ "ประสบการณ์ถูกรางวัล" เป็นเครื่องมือ retention

  2. หลังถูกรางวัล ลูกค้าซื้อเปลี่ยนแปลง {pct_ch:+.1f}%
     -> {'ลูกค้าเอารางวัลมาซื้อต่อ — เงินหมุนเวียนกลับระบบ' if pct_ch > 0 else 'ลูกค้าอาจพอใจกับรางวัลและลดการซื้อ'}
     -> ควรส่ง promotion 1-2 งวดหลังถูกรางวัล

  3. Revenue: ผู้ถูกรางวัลซื้อ {rev_r:.1f}x มากกว่าคนไม่ถูก
     -> Revenue จากผู้ถูกรางวัล = {pay_r:.0f}x เงินรางวัลที่จ่าย
     -> การลงทุนจ่ายรางวัลคุ้มค่า — ลูกค้ากลับมาซื้อต่อ""")

    if ms_churn is not None and sb_churn is not None:
        print(f"""
  4. ถูกเล็กๆ หลายครั้ง vs ถูกใหญ่ครั้งเดียว:
     -> ถูกเล็ก >=3 งวด: Churn {ms_churn:.1%}
     -> ถูกใหญ่ 1 ครั้ง: Churn {sb_churn:.1%}
     -> {'ถูกบ่อย = loyalty สูงกว่า (รู้สึกโชคดีสม่ำเสมอ)' if ms_churn < sb_churn else 'รางวัลใหญ่สร้าง loyalty ได้ดีกว่า'}""")

    if len(rec_rates) >= 2:
        print(f"""
  5. Honeymoon Period:
     -> ถูกรางวัล 0-1 งวดที่แล้ว: Churn {rec_rates[0]:.1%}
     -> ถูกรางวัล 13+ งวดที่แล้ว: Churn {rec_rates[-1]:.1%}
     -> ยิ่งนานจากครั้งสุดท้ายที่ถูก ยิ่ง Churn สูง
     -> ควร re-engage ลูกค้าที่ไม่ได้ถูกรางวัลนานกว่า 6 งวด""")

    print(f"""
  6. สำหรับทีม Marketing/Ads/Marcom:
     -> ใช้ "เรื่องราวผู้ถูกรางวัล" ในแคมเปญ
     -> เน้นจำนวนผู้ถูกรางวัลเล็กๆ (เยอะมาก) > รางวัลใหญ่ครั้งเดียว
     -> ลูกค้าที่ไม่เคยถูกเลย = Churn สูงสุด -> ต้อง retain ด้วยวิธีอื่น
     -> ส่ง "คุณเกือบถูกรางวัล" notification -> สร้าง engagement
""")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    print("="*70)
    print("Phase 39: Prize Impact Deep Dive")
    print("="*70)

    prizes = load_all_prizes()
    churn_df = load_latest_churn()
    item_cols, dates = get_item_columns(churn_df)

    # 1. Post-win behavior (vectorized)
    merged, win_indices, item_matrix, rd_to_idx = analyze_post_win_behavior(
        prizes, churn_df, item_cols, dates)

    # 1b. Duration
    offsets, means = analyze_winning_effect_duration(
        merged, win_indices, item_matrix, item_cols)

    # 1c. By tier
    tier_results = analyze_post_win_by_tier(prizes, churn_df, item_cols, dates, rd_to_idx)

    # 2. Prize tier vs churn
    type_churn, overall_churn, ms_churn, sb_churn = analyze_prize_tier_churn(
        prizes, churn_df, item_cols)

    # 3. Win frequency
    freq_churn = analyze_win_frequency(prizes, churn_df, item_cols)

    # 4. Timing / recency
    recency_churn = analyze_timing(prizes, churn_df, item_cols, dates)

    # 5. Revenue
    revenue = analyze_revenue_impact(prizes, churn_df, item_cols)

    # Chart
    create_chart(type_churn, overall_churn, freq_churn, recency_churn,
                 offsets, means, tier_results, revenue, ms_churn, sb_churn)

    # Recommendations
    print_recommendations(type_churn, overall_churn, freq_churn, recency_churn,
                          tier_results, revenue, ms_churn, sb_churn, offsets, means)

    print("\n" + "="*70)
    print("Phase 39 Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
