"""
Phase: Prize Behavior Distribution Analysis
============================================
For each prize type (THIRD=80K, FOURTH=40K, FIFTH/LAST2DIGITS/LAST3DIGITS=20K),
find ALL winners and categorize their behavior in the NEXT 1-3 periods after winning:
  1. ซื้อเพิ่มขึ้น (bought MORE)
  2. ซื้อเท่าเดิม (bought about the same, ±20%)
  3. ซื้อลดลง (bought LESS)
  4. หายไปเลย (churned)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

ROOT = Path('/Users/nongnat/Desktop/ChurnCustomer')
PRIZE_DIR = ROOT / 'data' / 'prize'
CHURN_DIR = ROOT / 'data' / 'churn'

# Prize types of interest and their prize amounts
PRIZE_TYPES = {
    'THIRD': 80_000,
    'FOURTH': 40_000,
    'FIFTH': 20_000,
    'LAST2DIGITS': 20_000,
    'LAST3DIGITS': 20_000,
}

# ── Helpers ──────────────────────────────────────────────────────────

def rounddate_to_item_col(rounddate_str):
    """Convert prize rounddate '2025-01-02' → 'item2025_01_02'."""
    return 'item' + rounddate_str.replace('-', '_')


def get_item_columns(df):
    """Return sorted list of item columns from a churn DataFrame."""
    return sorted([c for c in df.columns if c.startswith('item')])


def load_latest_churn():
    """Load the churn file with the most columns (latest file has most history)."""
    churn_files = sorted(CHURN_DIR.glob('Churn_202*_*_*.csv'))
    # Use the latest file — it has the most item columns
    latest = churn_files[-1]
    print(f"Loading churn file: {latest.name}")
    df = pd.read_csv(latest, low_memory=False)
    return df


def load_all_churn_merged():
    """Load all churn files and merge to get the widest customer x period matrix.

    Each churn file may have different customers. We need the union of all
    customers + all item columns. The latest file has the most columns, but
    earlier files may have customers that dropped off.
    """
    churn_files = sorted(CHURN_DIR.glob('Churn_202*_*_*.csv'))

    # The latest file has the widest column set — start with it
    latest = churn_files[-1]
    print(f"Primary churn file: {latest.name}")
    df = pd.read_csv(latest, low_memory=False)
    item_cols = get_item_columns(df)

    # Build a lookup: userNo → {item_col: value}
    # We only need userNo + item columns
    result = df[['userNo'] + item_cols].copy()
    result = result.set_index('userNo')
    known_users = set(result.index)

    # Go through older files to pick up customers not in the latest file
    for f in reversed(churn_files[:-1]):
        df2 = pd.read_csv(f, usecols=lambda c: c == 'userNo' or c.startswith('item'), low_memory=False)
        df2 = df2.set_index('userNo')
        new_users = df2.index.difference(result.index)
        if len(new_users) > 0:
            # Only add rows for users we don't already have
            new_rows = df2.loc[new_users]
            # Reindex to match our columns (fill missing with NaN)
            new_rows = new_rows.reindex(columns=result.columns)
            result = pd.concat([result, new_rows])
            known_users.update(new_users)

    print(f"Total unique customers in churn data: {len(result):,}")
    print(f"Item columns: {len(item_cols)} ({item_cols[0]} → {item_cols[-1]})")
    return result


def categorize_behavior(before, after):
    """Categorize post-win behavior.

    before: purchase amount in the winning period
    after: purchase amount in the next period (NaN or 0 = churn)

    Returns one of: 'ซื้อเพิ่มขึ้น', 'ซื้อเท่าเดิม', 'ซื้อลดลง', 'หายไปเลย'
    """
    if pd.isna(after) or after == 0:
        return 'หายไปเลย'
    if before == 0 or pd.isna(before):
        # Edge case: won but didn't buy in the winning period?
        # They must have bought to win, so treat this as data issue
        if after > 0:
            return 'ซื้อเพิ่มขึ้น'
        return 'หายไปเลย'

    ratio = after / before
    if ratio > 1.20:
        return 'ซื้อเพิ่มขึ้น'
    elif ratio >= 0.80:
        return 'ซื้อเท่าเดิม'
    else:
        return 'ซื้อลดลง'


# ── Main Analysis ────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("PRIZE BEHAVIOR DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # 1. Load all prize data
    prize_files = sorted(PRIZE_DIR.glob('Prize_*.csv'))
    print(f"\nLoading {len(prize_files)} prize files...")
    all_prizes = []
    for f in prize_files:
        df = pd.read_csv(f)
        all_prizes.append(df)
    prizes = pd.concat(all_prizes, ignore_index=True)
    print(f"Total prize records: {len(prizes):,}")
    print(f"Prize types: {prizes['type'].value_counts().to_dict()}")

    # Filter to our prize types of interest
    prizes = prizes[prizes['type'].isin(PRIZE_TYPES.keys())].copy()
    print(f"Filtered to target types: {len(prizes):,} records")

    # 2. Load churn data (merged from all files)
    print()
    churn = load_all_churn_merged()
    item_cols = sorted([c for c in churn.columns if c.startswith('item')])

    # Build ordered list of period columns
    period_order = item_cols  # already sorted chronologically
    period_to_idx = {col: i for i, col in enumerate(period_order)}

    # 3. For each prize record, find the winning period column and next 1-3 periods
    print("\nMapping prize rounddates to item columns...")
    prizes['item_col'] = prizes['rounddate'].apply(rounddate_to_item_col)

    # Check which prize periods exist in churn data
    available_periods = set(item_cols)
    prizes['period_exists'] = prizes['item_col'].isin(available_periods)
    n_match = prizes['period_exists'].sum()
    print(f"Prize records with matching churn period: {n_match:,} / {len(prizes):,}")

    # Filter to matching periods
    prizes = prizes[prizes['period_exists']].copy()

    # For each winner, get purchase in winning period + next 1-3 periods
    results = []

    # Group by rounddate for efficiency
    for rounddate, group in prizes.groupby('rounddate'):
        win_col = rounddate_to_item_col(rounddate)
        win_idx = period_to_idx.get(win_col)
        if win_idx is None:
            continue

        # Get next 1-3 period columns
        next_cols = []
        for offset in range(1, 4):
            nxt_idx = win_idx + offset
            if nxt_idx < len(period_order):
                next_cols.append((offset, period_order[nxt_idx]))

        if not next_cols:
            continue  # No future periods available

        # Get unique users who won in this period
        users_in_group = group['userNo'].unique()
        users_in_churn = churn.index.intersection(users_in_group)

        if len(users_in_churn) == 0:
            continue

        # Get purchase data for these users
        cols_needed = [win_col] + [c for _, c in next_cols]
        user_data = churn.loc[users_in_churn, cols_needed]

        # Merge with prize type info
        prize_info = group[['userNo', 'type', 'total_prize', 'count_prize']].drop_duplicates('userNo')
        # If a user won multiple types in same period, keep highest prize
        prize_info = prize_info.sort_values('total_prize', ascending=False).drop_duplicates('userNo')
        prize_info = prize_info.set_index('userNo')

        merged = user_data.join(prize_info, how='inner')

        for _, row in merged.iterrows():
            win_amount = row[win_col] if not pd.isna(row[win_col]) else 0
            prize_type = row['type']
            total_prize = row['total_prize']

            for offset, next_col in next_cols:
                next_amount = row[next_col] if not pd.isna(row[next_col]) else 0
                behavior = categorize_behavior(win_amount, next_amount)
                results.append({
                    'userNo': row.name,
                    'rounddate': rounddate,
                    'prize_type': prize_type,
                    'total_prize': total_prize,
                    'win_period_purchase': win_amount,
                    'next_period_purchase': next_amount,
                    'periods_after': offset,
                    'behavior': behavior,
                })

    results_df = pd.DataFrame(results)
    print(f"\nTotal analysis records: {len(results_df):,}")
    print(f"Unique winners analyzed: {results_df['userNo'].nunique():,}")

    # ── Report Results ───────────────────────────────────────────────

    behavior_order = ['ซื้อเพิ่มขึ้น', 'ซื้อเท่าเดิม', 'ซื้อลดลง', 'หายไปเลย']

    print("\n" + "=" * 80)
    print("RESULTS BY PRIZE TYPE AND PERIOD OFFSET")
    print("=" * 80)

    for prize_type in PRIZE_TYPES:
        subset = results_df[results_df['prize_type'] == prize_type]
        if len(subset) == 0:
            print(f"\n{'─' * 60}")
            print(f"  {prize_type} (฿{PRIZE_TYPES[prize_type]:,}) — NO DATA")
            continue

        print(f"\n{'─' * 60}")
        print(f"  {prize_type} (฿{PRIZE_TYPES[prize_type]:,})")
        print(f"  Total unique winners: {subset['userNo'].nunique():,}")
        print(f"{'─' * 60}")

        for offset in [1, 2, 3]:
            period_data = subset[subset['periods_after'] == offset]
            if len(period_data) == 0:
                continue

            total = len(period_data)
            print(f"\n  Period +{offset} (n={total:,}):")

            for beh in behavior_order:
                count = (period_data['behavior'] == beh).sum()
                pct = count / total * 100
                bar = '█' * int(pct / 2)
                print(f"    {beh:15s}: {count:>7,} ({pct:5.1f}%) {bar}")

            # Mean purchase change
            buyers = period_data[period_data['win_period_purchase'] > 0]
            if len(buyers) > 0:
                buyers_next = buyers[buyers['next_period_purchase'] > 0]
                if len(buyers_next) > 0:
                    avg_change = (buyers_next['next_period_purchase'] / buyers_next['win_period_purchase']).median()
                    print(f"    Median purchase ratio (non-churned): {avg_change:.2f}x")

    # ── Sustained Increase Analysis ──────────────────────────────────

    print("\n" + "=" * 80)
    print("SUSTAINED INCREASE ANALYSIS (2+ consecutive periods of increase)")
    print("=" * 80)

    for prize_type in PRIZE_TYPES:
        subset = results_df[results_df['prize_type'] == prize_type]
        if len(subset) == 0:
            continue

        # Vectorized: pivot to get behavior by period offset per user
        p1 = subset[subset['periods_after'] == 1][['userNo', 'rounddate', 'behavior']].copy()
        p2 = subset[subset['periods_after'] == 2][['userNo', 'rounddate', 'behavior']].copy()

        if len(p1) == 0 or len(p2) == 0:
            continue

        # Merge on userNo + rounddate to match same winning event
        merged_sus = p1.merge(p2, on=['userNo', 'rounddate'], suffixes=('_p1', '_p2'))
        total = len(merged_sus)
        sustained = ((merged_sus['behavior_p1'] == 'ซื้อเพิ่มขึ้น') & (merged_sus['behavior_p2'] == 'ซื้อเพิ่มขึ้น')).sum()

        pct = sustained / total * 100 if total > 0 else 0
        print(f"\n  {prize_type}: {sustained:,} / {total:,} ({pct:.1f}%) sustained increase for 2+ periods")

    # ── Churn Despite Winning ────────────────────────────────────────

    print("\n" + "=" * 80)
    print("CHURN DESPITE WINNING (หายไปเลย in period +1)")
    print("=" * 80)

    p1_data = results_df[results_df['periods_after'] == 1]

    for prize_type in PRIZE_TYPES:
        subset = p1_data[p1_data['prize_type'] == prize_type]
        if len(subset) == 0:
            continue
        churned = (subset['behavior'] == 'หายไปเลย').sum()
        total = len(subset)
        pct = churned / total * 100
        print(f"  {prize_type:15s}: {churned:>7,} / {total:>7,} ({pct:.1f}%) churned right after winning")

    # ── Statistical Test: Chi-squared across prize types ─────────────

    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON BETWEEN PRIZE TYPES (Period +1)")
    print("=" * 80)

    # Build contingency table
    contingency_data = {}
    for prize_type in PRIZE_TYPES:
        subset = p1_data[p1_data['prize_type'] == prize_type]
        if len(subset) == 0:
            continue
        counts = {}
        for beh in behavior_order:
            counts[beh] = (subset['behavior'] == beh).sum()
        contingency_data[prize_type] = counts

    if len(contingency_data) >= 2:
        cont_df = pd.DataFrame(contingency_data).T
        print("\nContingency table:")
        print(cont_df.to_string())

        chi2, p_value, dof, expected = stats.chi2_contingency(cont_df.values)
        print(f"\nChi-squared test: χ² = {chi2:.1f}, df = {dof}, p-value = {p_value:.2e}")
        if p_value < 0.05:
            print("→ SIGNIFICANT difference between prize types (p < 0.05)")
        else:
            print("→ No significant difference between prize types (p >= 0.05)")

    # ── Pairwise comparisons for churn rate ──────────────────────────

    print("\nPairwise churn rate comparison (Fisher exact test):")
    types_with_data = list(contingency_data.keys())
    for i in range(len(types_with_data)):
        for j in range(i+1, len(types_with_data)):
            t1, t2 = types_with_data[i], types_with_data[j]
            s1 = p1_data[p1_data['prize_type'] == t1]
            s2 = p1_data[p1_data['prize_type'] == t2]

            churned1 = (s1['behavior'] == 'หายไปเลย').sum()
            not_churned1 = len(s1) - churned1
            churned2 = (s2['behavior'] == 'หายไปเลย').sum()
            not_churned2 = len(s2) - churned2

            table = [[churned1, not_churned1], [churned2, not_churned2]]
            # Use chi2 for large samples
            chi2, p_val, _, _ = stats.chi2_contingency(table)

            rate1 = churned1 / len(s1) * 100
            rate2 = churned2 / len(s2) * 100
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  {t1} ({rate1:.1f}%) vs {t2} ({rate2:.1f}%): p={p_val:.4f} {sig}")

    # ── Summary Table ────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("SUMMARY TABLE (Period +1)")
    print("=" * 80)

    summary_rows = []
    for prize_type in PRIZE_TYPES:
        subset = p1_data[p1_data['prize_type'] == prize_type]
        if len(subset) == 0:
            continue
        total = len(subset)
        row = {'Prize Type': prize_type, 'Prize Amount': f"฿{PRIZE_TYPES[prize_type]:,}", 'n': total}
        for beh in behavior_order:
            count = (subset['behavior'] == beh).sum()
            row[beh] = f"{count / total * 100:.1f}%"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # ── Multi-period behavior flow ───────────────────────────────────

    print("\n" + "=" * 80)
    print("BEHAVIOR FLOW: Period +1 → Period +2 → Period +3")
    print("=" * 80)

    for prize_type in PRIZE_TYPES:
        subset = results_df[results_df['prize_type'] == prize_type]
        if len(subset) == 0:
            continue

        print(f"\n  {prize_type} (฿{PRIZE_TYPES[prize_type]:,}):")

        for offset in [1, 2, 3]:
            pd_offset = subset[subset['periods_after'] == offset]
            if len(pd_offset) == 0:
                continue
            total = len(pd_offset)

            pcts = []
            for beh in behavior_order:
                count = (pd_offset['behavior'] == beh).sum()
                pcts.append(f"{beh}={count/total*100:.1f}%")

            print(f"    +{offset}: {' | '.join(pcts)}  (n={total:,})")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
