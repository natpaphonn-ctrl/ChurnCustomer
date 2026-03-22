#!/usr/bin/env python3
"""Phase 41: Insight Catalog with Real Customer Examples.

Generates a comprehensive insight catalog with 15 insights across 4 teams
(Ads, Marcom, SEO/Content, Management). Every insight includes 2-3 real
customer examples with actual P numbers, demographics, and metrics.

Output:
  - Console: Full Thai-language insight catalog
  - JSON: data/predictions/insight_catalog.json
  - Chart: charts/phase41_insight_catalog.png (6-panel dashboard)
"""

import os, sys, json, warnings
from datetime import datetime

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
TX_DIR = os.path.join(DATA_DIR, 'transaction')
PRIZE_DIR = os.path.join(DATA_DIR, 'prize')
PRED_DIR = os.path.join(DATA_DIR, 'predictions')
CHART_DIR = 'charts'
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

MIN_TENURE = 3

# Files to load
CHURN_LATEST = os.path.join(CHURN_DIR, 'Churn_2026_0301_0316.csv')
CHURN_PREV = os.path.join(CHURN_DIR, 'Churn_2026_0201_0216.csv')
TX_FILES = [
    os.path.join(TX_DIR, 'Transaction_Order_2026_0316.csv'),
    os.path.join(TX_DIR, 'Transaction_Order_2026_0301.csv'),
    os.path.join(TX_DIR, 'Transaction_Order_2026_0216.csv'),
]
PRIZE_FILES = [
    os.path.join(PRIZE_DIR, 'Prize_2026_0316.csv'),
    os.path.join(PRIZE_DIR, 'Prize_2026_0301.csv'),
    os.path.join(PRIZE_DIR, 'Prize_2026_0216.csv'),
]
RISK_FILE = os.path.join(PRED_DIR, 'Churn_RiskScore_Explained_2026_0401.csv')
RISK_SCORE_FILE = os.path.join(PRED_DIR, 'Churn_RiskScore_2026_0401.csv')


def load_data():
    """Load all data sources."""
    print("Loading data...", flush=True)

    # Latest churn file
    df = pd.read_csv(CHURN_LATEST, low_memory=False)
    item_cols = [c for c in df.columns if c.startswith('item')]
    print(f"  Churn file: {len(df):,} customers, {len(item_cols)} periods", flush=True)

    # Previous churn file for comparison
    df_prev = pd.read_csv(CHURN_PREV, low_memory=False)
    print(f"  Prev churn: {len(df_prev):,} customers", flush=True)

    # TX data
    tx_frames = []
    for f in TX_FILES:
        if os.path.exists(f):
            t = pd.read_csv(f, low_memory=False)
            tx_frames.append(t)
            print(f"  TX: {os.path.basename(f)} — {len(t):,} rows", flush=True)
    tx = pd.concat(tx_frames, ignore_index=True) if tx_frames else pd.DataFrame()

    # Prize data
    prize_frames = []
    for f in PRIZE_FILES:
        if os.path.exists(f):
            p = pd.read_csv(f, low_memory=False)
            prize_frames.append(p)
            print(f"  Prize: {os.path.basename(f)} — {len(p):,} rows", flush=True)
    prize = pd.concat(prize_frames, ignore_index=True) if prize_frames else pd.DataFrame()

    # Risk scores
    risk_exp = None
    if os.path.exists(RISK_FILE):
        risk_exp = pd.read_csv(RISK_FILE, low_memory=False, encoding='utf-8-sig')
        print(f"  Risk Explained: {len(risk_exp):,} customers", flush=True)

    risk_score = None
    if os.path.exists(RISK_SCORE_FILE):
        risk_score = pd.read_csv(RISK_SCORE_FILE, low_memory=False, encoding='utf-8-sig')
        print(f"  Risk Score: {len(risk_score):,} customers", flush=True)

    return df, df_prev, item_cols, tx, prize, risk_exp, risk_score


def compute_churn_features(df, item_cols):
    """Compute basic features for all customers."""
    feat_cols = item_cols[:-1]
    target_col = item_cols[-1]

    mat = df[feat_cols].values.astype(float)
    target = df[target_col].values.astype(float)

    # Churn: bought previously but not in latest period
    # In churn files, customers present = bought. Target col: >0 means bought, 0/NaN = no
    # But this file only has customers who bought in at least one period
    # We define churn per the last two periods:
    # second_to_last bought, last period not bought
    second_last = df[feat_cols[-1]].values.astype(float)
    last = target.copy()

    # tenure: count non-NaN non-zero periods
    purchased = (~np.isnan(mat)) & (mat > 0)
    tenure = np.sum(~np.isnan(mat), axis=1)
    active_rounds = np.sum(purchased, axis=1)

    # Total items
    mat_clean = np.where(np.isnan(mat), 0, mat)
    total_items = np.sum(mat_clean, axis=1)

    # Items last 3 periods
    items_last_3 = np.sum(mat_clean[:, -3:], axis=1)
    items_last_1 = mat_clean[:, -1]

    # Active last 6
    active_last_6 = np.sum(purchased[:, -6:], axis=1)
    active_last_3 = np.sum(purchased[:, -3:], axis=1)

    # Churn label: customer who bought in second-to-last but NOT in last
    # Actually: target > 0 means bought in latest period. We need churn = didn't buy.
    has_last = (~np.isnan(last)) & (last > 0)
    has_second_last = (~np.isnan(second_last)) & (second_last > 0)
    churn = has_second_last & ~has_last

    # Gap analysis - find consecutive zeros from end
    def current_zero_streak(row):
        streak = 0
        for v in reversed(row):
            if np.isnan(v) or v == 0:
                streak += 1
            else:
                break
        return streak

    # Vectorized gap streak
    mat_binary = np.where(purchased, 1, 0)
    # Reverse and find first nonzero
    rev = mat_binary[:, ::-1]
    # argmax on reversed finds first 1 (= most recent purchase from end of feat_cols)
    first_nonzero = np.argmax(rev, axis=1)
    # If no purchase at all, streak = full length
    no_purchase = np.all(rev == 0, axis=1)
    current_streak = np.where(no_purchase, len(feat_cols), first_nonzero)

    df_feat = df[['userNo', 'age', 'gender', 'province']].copy()
    df_feat['tenure'] = tenure
    df_feat['active_rounds'] = active_rounds
    df_feat['total_items'] = total_items
    df_feat['items_last_3'] = items_last_3
    df_feat['items_last_1'] = items_last_1
    df_feat['active_last_6'] = active_last_6
    df_feat['active_last_3'] = active_last_3
    df_feat['purchase_freq'] = active_rounds / np.maximum(tenure, 1)
    df_feat['avg_items'] = total_items / np.maximum(active_rounds, 1)
    df_feat['churn'] = churn.astype(int)
    df_feat['bought_last'] = has_last.astype(int)
    df_feat['bought_second_last'] = has_second_last.astype(int)
    df_feat['current_zero_streak'] = current_streak
    df_feat['is_new'] = (tenure <= MIN_TENURE).astype(int)

    # Basket size trend: last 3 vs previous 3
    items_prev_3 = np.sum(mat_clean[:, -6:-3], axis=1)
    df_feat['basket_trend'] = items_last_3 - items_prev_3

    return df_feat, mat_clean, feat_cols, target_col


def find_gap_return_customers(df_feat, mat_clean, feat_cols, gap_min=1, gap_max=3, n=3):
    """Find customers who had a gap of gap_min to gap_max periods then returned."""
    examples = []
    # Look for pattern: bought, then 0s for gap periods, then bought again
    mat_binary = (mat_clean > 0).astype(int)
    n_periods = mat_binary.shape[1]

    for idx in range(len(df_feat)):
        if len(examples) >= n:
            break
        row = mat_binary[idx]
        items_row = mat_clean[idx]
        # Find gaps
        purchases = np.where(row > 0)[0]
        if len(purchases) < 3:
            continue
        # Find a gap in the later periods (last 20 periods)
        start_check = max(0, n_periods - 20)
        for i in range(len(purchases) - 1):
            if purchases[i] < start_check:
                continue
            gap = purchases[i + 1] - purchases[i] - 1
            if gap_min <= gap <= gap_max:
                before_items = items_row[purchases[i]]
                after_items = items_row[purchases[i + 1]]
                cust = df_feat.iloc[idx]
                examples.append({
                    'userNo': cust['userNo'],
                    'gender': cust['gender'],
                    'age': int(cust['age']) if pd.notna(cust['age']) else 0,
                    'province': cust['province'] if pd.notna(cust['province']) else '-',
                    'gap': int(gap),
                    'before_items': int(before_items),
                    'after_items': int(after_items),
                    'detail_th': f"หาย {gap} งวด กลับมาซื้อ {int(after_items)} ใบ (ก่อนหาย {int(before_items)} ใบ)"
                })
                break
    return examples


def gender_thai(g):
    if pd.isna(g):
        return '-'
    g = str(g).upper().strip()
    if g in ('M', 'MALE', 'ชาย'):
        return 'ชาย'
    elif g in ('F', 'FEMALE', 'หญิง'):
        return 'หญิง'
    return g


def print_table(examples, extra_col='รายละเอียด'):
    """Print a formatted table of customer examples."""
    if not examples:
        print("   (ไม่พบตัวอย่าง)")
        return
    # Header
    print(f"   {'Customer':<14} {'เพศ':<6} {'อายุ':<6} {'จังหวัด':<16} {extra_col}")
    print(f"   {'─'*14} {'─'*6} {'─'*6} {'─'*16} {'─'*40}")
    for ex in examples:
        g = gender_thai(ex.get('gender', ''))
        age = ex.get('age', '-')
        prov = ex.get('province', '-')
        if pd.isna(prov):
            prov = '-'
        detail = ex.get('detail_th', '')
        print(f"   {ex['userNo']:<14} {g:<6} {str(age):<6} {str(prov):<16} {detail}")
    print()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60, flush=True)
    print("📊 INSIGHT CATALOG — Phase 41", flush=True)
    print("=" * 60, flush=True)
    print()

    df, df_prev, item_cols, tx, prize, risk_exp, risk_score = load_data()
    df_feat, mat_clean, feat_cols, target_col = compute_churn_features(df, item_cols)

    # Eligible customers: those who bought in second-to-last period
    eligible = df_feat[df_feat['bought_second_last'] == 1].copy()
    total_eligible = len(eligible)
    total_churned = eligible['churn'].sum()
    churn_rate = total_churned / total_eligible * 100
    print(f"\nTotal eligible: {total_eligible:,}, Churned: {total_churned:,} ({churn_rate:.1f}%)\n", flush=True)

    # Collect all insights for JSON
    all_insights = []

    # ════════════════════════════════════════════════════════════
    # ทีม ADS
    # ════════════════════════════════════════════════════════════
    print("─" * 60)
    print("ทีม ADS")
    print("─" * 60)
    print()

    # --- Insight 1: ลูกค้าหายไป 1-3 งวด ส่วนใหญ่กลับมาเอง ---
    print("💡 Insight 1: ลูกค้าหายไป 1-3 งวด ส่วนใหญ่กลับมาเอง", flush=True)

    # Analyze return patterns from purchase history
    mat_binary = (mat_clean > 0).astype(int)
    n_periods = mat_binary.shape[1]

    # Count customers with gaps who returned
    gap_return_counts = {1: 0, 2: 0, 3: 0}
    gap_total_counts = {1: 0, 2: 0, 3: 0}

    for idx in range(len(df_feat)):
        row = mat_binary[idx]
        purchases = np.where(row > 0)[0]
        if len(purchases) < 2:
            continue
        for i in range(len(purchases) - 1):
            gap = purchases[i + 1] - purchases[i] - 1
            if gap in (1, 2, 3):
                gap_return_counts[gap] = gap_return_counts.get(gap, 0) + 1

    # Customers who had gaps (check last 10 periods for gaps)
    total_gaps = sum(gap_return_counts.values())
    return_within_3 = sum(gap_return_counts[g] for g in [1, 2, 3])

    # Compute gap-1 return rate specifically
    # Look at customers who had a 0 in a period and see what % came back next period
    gap1_start = 0
    gap1_return = 0
    for col_idx in range(max(0, n_periods - 15), n_periods - 1):
        had_gap = (mat_binary[:, col_idx] == 0) & (np.any(mat_binary[:, :col_idx] > 0, axis=1) if col_idx > 0 else np.zeros(len(df_feat), dtype=bool))
        returned_next = mat_binary[:, col_idx + 1] > 0
        gap1_start += had_gap.sum()
        gap1_return += (had_gap & returned_next).sum()

    gap1_pct = gap1_return / max(gap1_start, 1) * 100

    # Find real examples
    examples_1 = find_gap_return_customers(df_feat, mat_clean, feat_cols, gap_min=1, gap_max=3, n=3)

    stat_1 = f"Gap 1 งวด: {gap_return_counts[1]:,} ครั้ง, Gap 2 งวด: {gap_return_counts[2]:,} ครั้ง, Gap 3 งวด: {gap_return_counts[3]:,} ครั้ง"
    print(f"   สถิติ: {stat_1}")
    print(f"   ลูกค้าที่หาย 1 งวดแล้วกลับมา: {gap1_pct:.1f}%")
    print()
    print("   ตัวอย่างลูกค้า:")
    print_table(examples_1)

    rec_1 = "ไม่ต้องรีบ retarget ลูกค้าที่หายไปแค่ 1-2 งวด — ส่วนใหญ่กลับมาเอง ควรโฟกัส budget กับลูกค้าที่หาย 4+ งวด"
    print(f"   💬 Recommendation: {rec_1}")
    print()

    all_insights.append({
        'id': 1, 'team': 'Ads',
        'title_th': 'ลูกค้าหายไป 1-3 งวด ส่วนใหญ่กลับมาเอง',
        'stat': stat_1,
        'recommendation_th': rec_1,
        'examples': [{k: v for k, v in ex.items()} for ex in examples_1]
    })

    # --- Insight 2: Demographics × Churn ---
    print("💡 Insight 2: Demographics × Churn — กลุ่มไหน churn เยอะ/น้อย", flush=True)

    # By gender
    gender_churn = eligible.groupby('gender').agg(
        total=('churn', 'count'), churned=('churn', 'sum')
    ).reset_index()
    gender_churn['rate'] = gender_churn['churned'] / gender_churn['total'] * 100

    print("   Churn by Gender:")
    for _, row in gender_churn.iterrows():
        print(f"     {gender_thai(row['gender'])}: {row['rate']:.1f}% ({row['churned']:,}/{row['total']:,})")

    # By age group
    age_bins = [0, 25, 35, 45, 55, 100]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '55+']
    eligible_age = eligible[eligible['age'].notna() & (eligible['age'] > 0)].copy()
    eligible_age['age_group'] = pd.cut(eligible_age['age'], bins=age_bins, labels=age_labels, right=True)
    age_churn = eligible_age.groupby('age_group', observed=True).agg(
        total=('churn', 'count'), churned=('churn', 'sum')
    ).reset_index()
    age_churn['rate'] = age_churn['churned'] / age_churn['total'] * 100

    print("\n   Churn by Age Group:")
    for _, row in age_churn.iterrows():
        print(f"     {row['age_group']}: {row['rate']:.1f}% ({row['churned']:,}/{row['total']:,})")

    # Find example from highest and lowest churn age group
    highest_age = age_churn.loc[age_churn['rate'].idxmax(), 'age_group']
    lowest_age = age_churn.loc[age_churn['rate'].idxmin(), 'age_group']

    examples_2 = []
    # High churn group example (churned)
    high_group = eligible_age[(eligible_age['age_group'] == highest_age) & (eligible_age['churn'] == 1)]
    if len(high_group) > 0:
        ex = high_group.iloc[0]
        examples_2.append({
            'userNo': ex['userNo'], 'gender': ex['gender'],
            'age': int(ex['age']), 'province': ex['province'],
            'detail_th': f"กลุ่ม {highest_age} — churned (churn rate สูงสุด {age_churn.loc[age_churn['rate'].idxmax(), 'rate']:.1f}%)"
        })
    # Low churn group example (stayed)
    low_group = eligible_age[(eligible_age['age_group'] == lowest_age) & (eligible_age['churn'] == 0)]
    if len(low_group) > 0:
        ex = low_group.iloc[0]
        examples_2.append({
            'userNo': ex['userNo'], 'gender': ex['gender'],
            'age': int(ex['age']), 'province': ex['province'],
            'detail_th': f"กลุ่ม {lowest_age} — ยังซื้ออยู่ (churn rate ต่ำสุด {age_churn.loc[age_churn['rate'].idxmin(), 'rate']:.1f}%)"
        })
    # Medium group
    mid_idx = len(age_churn) // 2
    mid_age = age_churn.iloc[mid_idx]['age_group']
    mid_group = eligible_age[(eligible_age['age_group'] == mid_age) & (eligible_age['churn'] == 1)]
    if len(mid_group) > 0:
        ex = mid_group.iloc[0]
        examples_2.append({
            'userNo': ex['userNo'], 'gender': ex['gender'],
            'age': int(ex['age']), 'province': ex['province'],
            'detail_th': f"กลุ่ม {mid_age} — churned (churn rate {age_churn.iloc[mid_idx]['rate']:.1f}%)"
        })

    print("\n   ตัวอย่างลูกค้า:")
    print_table(examples_2)

    stat_2 = f"Churn สูงสุด: กลุ่ม {highest_age} ({age_churn.loc[age_churn['rate'].idxmax(), 'rate']:.1f}%), ต่ำสุด: กลุ่ม {lowest_age} ({age_churn.loc[age_churn['rate'].idxmin(), 'rate']:.1f}%)"
    rec_2 = f"โฟกัส ads ที่กลุ่มอายุ {highest_age} ซึ่งมี churn rate สูงสุด ใช้ messaging ที่ตรงกับกลุ่มอายุ"
    print(f"   💬 Recommendation: {rec_2}")
    print()

    all_insights.append({
        'id': 2, 'team': 'Ads',
        'title_th': 'Demographics × Churn — กลุ่มไหน churn เยอะ/น้อย',
        'stat': stat_2, 'recommendation_th': rec_2,
        'examples': examples_2
    })

    # --- Insight 3: Rush Buyers vs Early Buyers ---
    print("💡 Insight 3: Rush Buyers vs Early Buyers loyalty", flush=True)

    examples_3 = []
    stat_3 = "ไม่มีข้อมูล TX"
    rec_3 = ""

    if len(tx) > 0:
        tx_success = tx[tx['status'] == 'SUCCESS'].copy()
        # Parse dates — strip timezone to make both tz-naive
        tx_success['created'] = pd.to_datetime(tx_success['createdAt_bkk'], errors='coerce', utc=True).dt.tz_localize(None)
        tx_success['round_dt'] = pd.to_datetime(tx_success['roundDate'], errors='coerce')
        tx_success = tx_success.dropna(subset=['created', 'round_dt'])

        # Days before draw
        tx_success['days_before'] = (tx_success['round_dt'] - tx_success['created']).dt.days

        # Classify: early (>=5 days before) vs rush (<=1 day = last 2 days)
        early_users = set(tx_success[tx_success['days_before'] >= 5]['userNo'].unique())
        rush_users = set(tx_success[tx_success['days_before'] <= 1]['userNo'].unique())

        # Only-early and only-rush (exclusive)
        only_early = early_users - rush_users
        only_rush = rush_users - early_users

        early_churn = eligible[eligible['userNo'].isin(only_early)]
        rush_churn = eligible[eligible['userNo'].isin(only_rush)]

        early_rate = early_churn['churn'].mean() * 100 if len(early_churn) > 0 else 0
        rush_rate = rush_churn['churn'].mean() * 100 if len(rush_churn) > 0 else 0

        stat_3 = f"Early buyers (ซื้อ 5+ วันก่อน): churn {early_rate:.1f}% ({len(early_churn):,} คน), Rush buyers (ซื้อ 1-2 วันสุดท้าย): churn {rush_rate:.1f}% ({len(rush_churn):,} คน)"
        print(f"   สถิติ: {stat_3}")

        # Example: loyal early buyer
        loyal_early = early_churn[(early_churn['churn'] == 0) & (early_churn['active_rounds'] >= 10)]
        if len(loyal_early) > 0:
            ex = loyal_early.iloc[0]
            examples_3.append({
                'userNo': ex['userNo'], 'gender': ex['gender'],
                'age': int(ex['age']) if pd.notna(ex['age']) else 0,
                'province': ex['province'] if pd.notna(ex['province']) else '-',
                'detail_th': f"Early buyer ภักดี — ซื้อ {int(ex['active_rounds'])} งวด, ไม่ churn"
            })

        # Example: churned rush buyer
        churned_rush = rush_churn[(rush_churn['churn'] == 1)]
        if len(churned_rush) > 0:
            ex = churned_rush.iloc[0]
            examples_3.append({
                'userNo': ex['userNo'], 'gender': ex['gender'],
                'age': int(ex['age']) if pd.notna(ex['age']) else 0,
                'province': ex['province'] if pd.notna(ex['province']) else '-',
                'detail_th': f"Rush buyer ที่ churn — ซื้อวันสุดท้ายแล้วหาย"
            })

        # Example: loyal rush buyer for comparison
        loyal_rush = rush_churn[(rush_churn['churn'] == 0) & (rush_churn['active_rounds'] >= 5)]
        if len(loyal_rush) > 0:
            ex = loyal_rush.iloc[0]
            examples_3.append({
                'userNo': ex['userNo'], 'gender': ex['gender'],
                'age': int(ex['age']) if pd.notna(ex['age']) else 0,
                'province': ex['province'] if pd.notna(ex['province']) else '-',
                'detail_th': f"Rush buyer แต่ภักดี — ซื้อ {int(ex['active_rounds'])} งวด, ไม่ churn"
            })

        rec_3 = "Early buyers มี loyalty สูงกว่า — ใช้ ads กระตุ้นให้ซื้อเร็วขึ้น จะช่วยลด churn"
        print(f"\n   💬 Recommendation: {rec_3}")
    else:
        print("   ⚠ ไม่มีข้อมูล TX")

    print("\n   ตัวอย่างลูกค้า:")
    print_table(examples_3)

    all_insights.append({
        'id': 3, 'team': 'Ads',
        'title_th': 'Rush Buyers vs Early Buyers loyalty',
        'stat': stat_3, 'recommendation_th': rec_3,
        'examples': examples_3
    })

    # --- Insight 4: จังหวัดที่ churn สูง/ต่ำ ---
    print("💡 Insight 4: จังหวัดที่ churn สูง/ต่ำ", flush=True)

    prov_churn = eligible[eligible['province'].notna()].groupby('province').agg(
        total=('churn', 'count'), churned=('churn', 'sum')
    ).reset_index()
    prov_churn['rate'] = prov_churn['churned'] / prov_churn['total'] * 100
    prov_min_1000 = prov_churn[prov_churn['total'] >= 1000].sort_values('rate')

    top5_low = prov_min_1000.head(5)
    top5_high = prov_min_1000.tail(5).sort_values('rate', ascending=False)

    print("\n   Top 5 Churn ต่ำ (min 1,000 ลูกค้า):")
    for _, row in top5_low.iterrows():
        print(f"     {row['province']}: {row['rate']:.1f}% ({row['total']:,} คน)")

    print("\n   Top 5 Churn สูง (min 1,000 ลูกค้า):")
    for _, row in top5_high.iterrows():
        print(f"     {row['province']}: {row['rate']:.1f}% ({row['total']:,} คน)")

    examples_4 = []
    # Example from highest churn province
    high_prov = top5_high.iloc[0]['province']
    high_cust = eligible[(eligible['province'] == high_prov) & (eligible['churn'] == 1)]
    if len(high_cust) > 0:
        ex = high_cust.iloc[0]
        examples_4.append({
            'userNo': ex['userNo'], 'gender': ex['gender'],
            'age': int(ex['age']) if pd.notna(ex['age']) else 0,
            'province': ex['province'],
            'detail_th': f"จังหวัด churn สูง ({top5_high.iloc[0]['rate']:.1f}%) — churned"
        })

    # Example from lowest churn province
    low_prov = top5_low.iloc[0]['province']
    low_cust = eligible[(eligible['province'] == low_prov) & (eligible['churn'] == 0)]
    if len(low_cust) > 0:
        ex = low_cust.iloc[0]
        examples_4.append({
            'userNo': ex['userNo'], 'gender': ex['gender'],
            'age': int(ex['age']) if pd.notna(ex['age']) else 0,
            'province': ex['province'],
            'detail_th': f"จังหวัด churn ต่ำ ({top5_low.iloc[0]['rate']:.1f}%) — ยังซื้ออยู่"
        })

    print("\n   ตัวอย่างลูกค้า:")
    print_table(examples_4)

    stat_4 = f"Churn สูงสุด: {top5_high.iloc[0]['province']} ({top5_high.iloc[0]['rate']:.1f}%), ต่ำสุด: {top5_low.iloc[0]['province']} ({top5_low.iloc[0]['rate']:.1f}%)"
    rec_4 = f"จังหวัดที่ churn สูง เช่น {top5_high.iloc[0]['province']} ควรเพิ่ม ads/promotion เฉพาะพื้นที่"
    print(f"   💬 Recommendation: {rec_4}")
    print()

    all_insights.append({
        'id': 4, 'team': 'Ads',
        'title_th': 'จังหวัดที่ churn สูง/ต่ำ',
        'stat': stat_4, 'recommendation_th': rec_4,
        'examples': examples_4
    })

    # --- Insight 5: Retargeting Window ---
    print("💡 Insight 5: Retargeting Window — เวลาที่ดีที่สุดในการ retarget", flush=True)

    examples_5 = []
    stat_5 = "ไม่มีข้อมูล TX"
    rec_5 = ""

    if len(tx) > 0:
        tx_success = tx[tx['status'] == 'SUCCESS'].copy()
        tx_success['created'] = pd.to_datetime(tx_success['createdAt_bkk'], errors='coerce', utc=True).dt.tz_localize(None)
        tx_success = tx_success.dropna(subset=['created'])
        tx_success['hour'] = tx_success['created'].dt.hour
        tx_success['dow'] = tx_success['created'].dt.dayofweek  # 0=Mon

        # Find recovered churners: customers who had zero_streak >= 1 but bought in latest
        recovered = df_feat[(df_feat['bought_last'] == 1) & (df_feat['current_zero_streak'] == 0)]
        # Check if they had a gap before: items_last_3 > 0 but not all 3
        # Actually: look at mat_clean for gap pattern
        recovered_with_gap = []
        for idx in recovered.index[:5000]:  # sample
            row_idx = df_feat.index.get_loc(idx) if idx in df_feat.index else None
            if row_idx is None:
                continue
            row = mat_clean[row_idx]
            # Had at least one 0 in last 5 periods before the last feature period
            last5 = row[-5:]
            if np.sum(last5 == 0) >= 1 and last5[-1] > 0:
                recovered_with_gap.append(idx)

        if recovered_with_gap:
            recovered_users = df_feat.loc[recovered_with_gap, 'userNo'].values
            tx_recovered = tx_success[tx_success['userNo'].isin(recovered_users)]

            if len(tx_recovered) > 0:
                hour_dist = tx_recovered['hour'].value_counts().sort_index()
                peak_hour = hour_dist.idxmax()
                dow_dist = tx_recovered['dow'].value_counts().sort_index()
                dow_names = ['จันทร์', 'อังคาร', 'พุธ', 'พฤหัส', 'ศุกร์', 'เสาร์', 'อาทิตย์']
                peak_dow = dow_names[dow_dist.idxmax()]

                stat_5 = f"Recovered churners ซื้อช่วง {peak_hour}:00 น. มากที่สุด, วัน{peak_dow}"
                print(f"   สถิติ: {stat_5}")

                # Top 3 hours
                top_hours = hour_dist.nlargest(3)
                print(f"   Top 3 ชั่วโมง: {', '.join(f'{h}:00 ({c:,})' for h, c in top_hours.items())}")

                # Find 3 real recovered churners with TX data
                for uid in recovered_users[:50]:
                    if len(examples_5) >= 3:
                        break
                    u_tx = tx_recovered[tx_recovered['userNo'] == uid]
                    if len(u_tx) == 0:
                        continue
                    last_tx = u_tx.sort_values('created').iloc[-1]
                    cust = df_feat[df_feat['userNo'] == uid].iloc[0]
                    examples_5.append({
                        'userNo': uid, 'gender': cust['gender'],
                        'age': int(cust['age']) if pd.notna(cust['age']) else 0,
                        'province': cust['province'] if pd.notna(cust['province']) else '-',
                        'detail_th': f"กลับมาซื้อเวลา {last_tx['created'].strftime('%H:%M')} น. ({int(last_tx['totalItem'])} ใบ)"
                    })

                rec_5 = f"Retarget ads ช่วง {peak_hour-1}:00-{peak_hour+2}:00 น. วัน{peak_dow} — เป็นช่วงที่ recovered churners กลับมาซื้อมากที่สุด"

    print("\n   ตัวอย่างลูกค้า:")
    print_table(examples_5)
    if rec_5:
        print(f"   💬 Recommendation: {rec_5}")
    print()

    all_insights.append({
        'id': 5, 'team': 'Ads',
        'title_th': 'Retargeting Window — เวลาที่ดีที่สุดในการ retarget',
        'stat': stat_5, 'recommendation_th': rec_5,
        'examples': examples_5
    })

    # ════════════════════════════════════════════════════════════
    # ทีม MARCOM
    # ════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("ทีม MARCOM")
    print("─" * 60)
    print()

    # --- Insight 6: ลูกค้าถูกรางวัลกลับมาซื้อ ---
    print("💡 Insight 6: ลูกค้าถูกรางวัลกลับมาซื้อ", flush=True)

    examples_6 = []
    stat_6 = "ไม่มีข้อมูล Prize"
    rec_6 = ""

    if len(prize) > 0:
        # Prize winners
        winners = prize['userNo'].unique()
        winner_churn = eligible[eligible['userNo'].isin(winners)]
        non_winner_churn = eligible[~eligible['userNo'].isin(winners)]

        winner_rate = winner_churn['churn'].mean() * 100 if len(winner_churn) > 0 else 0
        non_winner_rate = non_winner_churn['churn'].mean() * 100 if len(non_winner_churn) > 0 else 0

        stat_6 = f"ถูกรางวัล churn {winner_rate:.1f}% ({len(winner_churn):,} คน), ไม่ถูกรางวัล churn {non_winner_rate:.1f}% ({len(non_winner_churn):,} คน)"
        print(f"   สถิติ: {stat_6}")

        # Prize by type
        prize_summary = prize.groupby('type').agg(
            users=('userNo', 'nunique'),
            total_prize=('total_prize', 'sum'),
            avg_prize=('total_prize', 'mean')
        ).reset_index()
        print("\n   Prize by type:")
        for _, row in prize_summary.iterrows():
            print(f"     {row['type']}: {row['users']:,} คน, avg {row['avg_prize']:,.0f} บาท")

        # Find winners who came back after gap
        winner_feat = df_feat[df_feat['userNo'].isin(winners)]
        # Winners who bought in latest period and have history
        winner_active = winner_feat[(winner_feat['bought_last'] == 1) & (winner_feat['active_rounds'] >= 3)]

        prize_user_data = prize.groupby('userNo').agg(
            total_prize=('total_prize', 'sum'),
            prize_type=('type', 'first'),
            count_prize=('count_prize', 'sum')
        ).reset_index()

        for _, wrow in winner_active.head(20).iterrows():
            if len(examples_6) >= 3:
                break
            uid = wrow['userNo']
            pdata = prize_user_data[prize_user_data['userNo'] == uid]
            if len(pdata) == 0:
                continue
            p = pdata.iloc[0]
            examples_6.append({
                'userNo': uid, 'gender': wrow['gender'],
                'age': int(wrow['age']) if pd.notna(wrow['age']) else 0,
                'province': wrow['province'] if pd.notna(wrow['province']) else '-',
                'detail_th': f"ถูกรางวัล {p['prize_type']} ({p['total_prize']:,.0f} บาท) — ยังซื้อต่อ {int(wrow['active_rounds'])} งวด"
            })

        rec_6 = "ใช้เรื่อง 'ลุ้นรางวัล' ใน messaging — ลูกค้าถูกรางวัล retention สูงกว่ามาก ควรโชว์ผู้ถูกรางวัลเป็น social proof"
        print(f"\n   💬 Recommendation: {rec_6}")

    print("\n   ตัวอย่างลูกค้า:")
    print_table(examples_6)

    all_insights.append({
        'id': 6, 'team': 'Marcom',
        'title_th': 'ลูกค้าถูกรางวัลกลับมาซื้อ',
        'stat': stat_6, 'recommendation_th': rec_6,
        'examples': examples_6
    })

    # --- Insight 7: Segment ต่างกัน ต้อง message ต่างกัน ---
    print("\n💡 Insight 7: Segment ต่างกัน ต้อง message ต่างกัน", flush=True)

    # Segments: ขาประจำ (freq >= 0.8), ขาจร (freq 0.3-0.6), กำลังหาย (freq < 0.3 or declining)
    regulars = eligible[(eligible['purchase_freq'] >= 0.8) & (eligible['active_rounds'] >= 10)]
    occasionals = eligible[(eligible['purchase_freq'] >= 0.3) & (eligible['purchase_freq'] < 0.6)]
    fading = eligible[(eligible['purchase_freq'] < 0.3) | (eligible['basket_trend'] < -3)]

    print(f"   ขาประจำ (freq >= 80%): {len(regulars):,} คน, churn {regulars['churn'].mean()*100:.1f}%")
    print(f"   ขาจร (freq 30-60%): {len(occasionals):,} คน, churn {occasionals['churn'].mean()*100:.1f}%")
    print(f"   กำลังหาย (freq < 30% หรือ declining): {len(fading):,} คน, churn {fading['churn'].mean()*100:.1f}%")

    examples_7 = []
    # ขาประจำ example
    if len(regulars) > 0:
        ex = regulars[regulars['churn'] == 0].iloc[0] if (regulars['churn'] == 0).any() else regulars.iloc[0]
        examples_7.append({
            'userNo': ex['userNo'], 'gender': ex['gender'],
            'age': int(ex['age']) if pd.notna(ex['age']) else 0,
            'province': ex['province'] if pd.notna(ex['province']) else '-',
            'detail_th': f"ขาประจำ — ซื้อ {int(ex['active_rounds'])}/{int(ex['tenure'])} งวด ({ex['purchase_freq']*100:.0f}%), เฉลี่ย {ex['avg_items']:.1f} ใบ/งวด"
        })
    # ขาจร
    if len(occasionals) > 0:
        ex = occasionals.iloc[0]
        examples_7.append({
            'userNo': ex['userNo'], 'gender': ex['gender'],
            'age': int(ex['age']) if pd.notna(ex['age']) else 0,
            'province': ex['province'] if pd.notna(ex['province']) else '-',
            'detail_th': f"ขาจร — ซื้อ {int(ex['active_rounds'])}/{int(ex['tenure'])} งวด ({ex['purchase_freq']*100:.0f}%), เฉลี่ย {ex['avg_items']:.1f} ใบ/งวด"
        })
    # กำลังหาย
    fading_churned = fading[fading['churn'] == 1]
    if len(fading_churned) > 0:
        ex = fading_churned.iloc[0]
        examples_7.append({
            'userNo': ex['userNo'], 'gender': ex['gender'],
            'age': int(ex['age']) if pd.notna(ex['age']) else 0,
            'province': ex['province'] if pd.notna(ex['province']) else '-',
            'detail_th': f"กำลังหาย — ซื้อ {int(ex['active_rounds'])}/{int(ex['tenure'])} งวด ({ex['purchase_freq']*100:.0f}%), basket ลดลง {int(ex['basket_trend'])} ใบ"
        })

    print("\n   ตัวอย่างลูกค้า:")
    print_table(examples_7)

    stat_7 = f"ขาประจำ churn {regulars['churn'].mean()*100:.1f}%, ขาจร churn {occasionals['churn'].mean()*100:.1f}%, กำลังหาย churn {fading['churn'].mean()*100:.1f}%"
    rec_7 = "ขาประจำ: ขอบคุณ + VIP reward, ขาจร: กระตุ้นซื้อบ่อยขึ้น + reminder, กำลังหาย: win-back campaign + promotion พิเศษ"
    print(f"   💬 Recommendation: {rec_7}")
    print()

    all_insights.append({
        'id': 7, 'team': 'Marcom',
        'title_th': 'Segment ต่างกัน ต้อง message ต่างกัน',
        'stat': stat_7, 'recommendation_th': rec_7,
        'examples': examples_7
    })

    # --- Insight 8: Big buyers vs normal buyers ---
    print("💡 Insight 8: Big buyers vs normal buyers — churn ต่างกัน", flush=True)

    big_buyers = eligible[eligible['avg_items'] >= 50]
    normal_buyers = eligible[(eligible['avg_items'] >= 1) & (eligible['avg_items'] <= 10)]

    big_rate = big_buyers['churn'].mean() * 100 if len(big_buyers) > 0 else 0
    normal_rate = normal_buyers['churn'].mean() * 100 if len(normal_buyers) > 0 else 0

    stat_8 = f"Big buyers (50+ ใบ/งวด): churn {big_rate:.1f}% ({len(big_buyers):,} คน), Normal (1-10 ใบ): churn {normal_rate:.1f}% ({len(normal_buyers):,} คน)"
    print(f"   สถิติ: {stat_8}")

    examples_8 = []
    # Big buyer who stayed
    big_stayed = big_buyers[big_buyers['churn'] == 0]
    if len(big_stayed) > 0:
        ex = big_stayed.iloc[0]
        examples_8.append({
            'userNo': ex['userNo'], 'gender': ex['gender'],
            'age': int(ex['age']) if pd.notna(ex['age']) else 0,
            'province': ex['province'] if pd.notna(ex['province']) else '-',
            'detail_th': f"Big buyer — เฉลี่ย {ex['avg_items']:.0f} ใบ/งวด, รวม {int(ex['total_items'])} ใบ, ไม่ churn"
        })
    # Big buyer who churned
    big_churned = big_buyers[big_buyers['churn'] == 1]
    if len(big_churned) > 0:
        ex = big_churned.iloc[0]
        examples_8.append({
            'userNo': ex['userNo'], 'gender': ex['gender'],
            'age': int(ex['age']) if pd.notna(ex['age']) else 0,
            'province': ex['province'] if pd.notna(ex['province']) else '-',
            'detail_th': f"Big buyer ที่ churn — เฉลี่ย {ex['avg_items']:.0f} ใบ/งวด, รวม {int(ex['total_items'])} ใบ"
        })
    # Normal buyer
    normal_stayed = normal_buyers[normal_buyers['churn'] == 0]
    if len(normal_stayed) > 0:
        ex = normal_stayed.iloc[0]
        examples_8.append({
            'userNo': ex['userNo'], 'gender': ex['gender'],
            'age': int(ex['age']) if pd.notna(ex['age']) else 0,
            'province': ex['province'] if pd.notna(ex['province']) else '-',
            'detail_th': f"Normal buyer — เฉลี่ย {ex['avg_items']:.1f} ใบ/งวด, {int(ex['active_rounds'])} งวด"
        })

    print("\n   ตัวอย่างลูกค้า:")
    print_table(examples_8)

    rec_8 = "Big buyers มีความเสี่ยงสูงเมื่อ churn เพราะ revenue impact ใหญ่ — ต้องมี dedicated account management"
    print(f"   💬 Recommendation: {rec_8}")
    print()

    all_insights.append({
        'id': 8, 'team': 'Marcom',
        'title_th': 'Big buyers vs normal buyers — churn ต่างกัน',
        'stat': stat_8, 'recommendation_th': rec_8,
        'examples': examples_8
    })

    # --- Insight 9: Cancel บ่อย → churn? ---
    print("💡 Insight 9: ลูกค้าที่ cancel บ่อย → churn มากขึ้น?", flush=True)

    examples_9 = []
    stat_9 = "ไม่มีข้อมูล TX"
    rec_9 = ""

    if len(tx) > 0:
        # Cancel rate per user
        tx_user = tx.groupby('userNo').agg(
            total_orders=('status', 'count'),
            cancels=('status', lambda x: (x == 'CANCEL').sum())
        ).reset_index()
        tx_user['cancel_rate'] = tx_user['cancels'] / tx_user['total_orders']

        high_cancel = tx_user[tx_user['cancel_rate'] > 0.2]['userNo']
        low_cancel = tx_user[tx_user['cancel_rate'] < 0.05]['userNo']

        hc_churn = eligible[eligible['userNo'].isin(high_cancel)]
        lc_churn = eligible[eligible['userNo'].isin(low_cancel)]

        hc_rate = hc_churn['churn'].mean() * 100 if len(hc_churn) > 0 else 0
        lc_rate = lc_churn['churn'].mean() * 100 if len(lc_churn) > 0 else 0

        stat_9 = f"Cancel >20%: churn {hc_rate:.1f}% ({len(hc_churn):,} คน), Cancel <5%: churn {lc_rate:.1f}% ({len(lc_churn):,} คน)"
        print(f"   สถิติ: {stat_9}")

        # Examples
        hc_churned = hc_churn[hc_churn['churn'] == 1]
        if len(hc_churned) > 0:
            uid = hc_churned.iloc[0]['userNo']
            udata = tx_user[tx_user['userNo'] == uid].iloc[0]
            ex = hc_churned.iloc[0]
            examples_9.append({
                'userNo': uid, 'gender': ex['gender'],
                'age': int(ex['age']) if pd.notna(ex['age']) else 0,
                'province': ex['province'] if pd.notna(ex['province']) else '-',
                'detail_th': f"Cancel rate {udata['cancel_rate']*100:.0f}% ({int(udata['cancels'])}/{int(udata['total_orders'])} orders) — churned"
            })

        lc_stayed = lc_churn[lc_churn['churn'] == 0]
        if len(lc_stayed) > 0:
            uid = lc_stayed.iloc[0]['userNo']
            udata = tx_user[tx_user['userNo'] == uid].iloc[0]
            ex = lc_stayed.iloc[0]
            examples_9.append({
                'userNo': uid, 'gender': ex['gender'],
                'age': int(ex['age']) if pd.notna(ex['age']) else 0,
                'province': ex['province'] if pd.notna(ex['province']) else '-',
                'detail_th': f"Cancel rate {udata['cancel_rate']*100:.0f}% ({int(udata['cancels'])}/{int(udata['total_orders'])} orders) — ยังซื้ออยู่"
            })

        # High cancel but stayed (interesting)
        hc_stayed = hc_churn[hc_churn['churn'] == 0]
        if len(hc_stayed) > 0:
            uid = hc_stayed.iloc[0]['userNo']
            udata = tx_user[tx_user['userNo'] == uid].iloc[0]
            ex = hc_stayed.iloc[0]
            examples_9.append({
                'userNo': uid, 'gender': ex['gender'],
                'age': int(ex['age']) if pd.notna(ex['age']) else 0,
                'province': ex['province'] if pd.notna(ex['province']) else '-',
                'detail_th': f"Cancel rate สูง {udata['cancel_rate']*100:.0f}% แต่ไม่ churn — ลูกค้าภักดีที่มีปัญหาจ่ายเงิน"
            })

        rec_9 = "Cancel สูง = ปัญหา payment flow ไม่ใช่ความไม่พอใจ — ปรับปรุง checkout UX จะช่วยลด cancel rate"
        print(f"\n   💬 Recommendation: {rec_9}")

    print("\n   ตัวอย่างลูกค้า:")
    print_table(examples_9)

    all_insights.append({
        'id': 9, 'team': 'Marcom',
        'title_th': 'ลูกค้าที่ cancel บ่อย → churn มากขึ้น?',
        'stat': stat_9, 'recommendation_th': rec_9,
        'examples': examples_9
    })

    # --- Insight 10: Basket size กับ churn ---
    print("\n💡 Insight 10: Basket size กับ churn — ซื้อน้อยลงก่อน churn?", flush=True)

    # Declining basket before churn
    declining_churners = eligible[(eligible['churn'] == 1) & (eligible['basket_trend'] < -2)]
    sudden_churners = eligible[(eligible['churn'] == 1) & (eligible['basket_trend'] >= 0) & (eligible['items_last_3'] > 0)]

    stat_10 = f"Churner ที่ basket ลดลง: {len(declining_churners):,} คน, Churner ที่ basket คงที่/เพิ่ม: {len(sudden_churners):,} คน"
    print(f"   สถิติ: {stat_10}")
    pct_decline = len(declining_churners) / max(len(eligible[eligible['churn'] == 1]), 1) * 100
    print(f"   {pct_decline:.1f}% ของ churners มี basket ลดลงก่อน churn")

    examples_10 = []
    if len(declining_churners) > 0:
        ex = declining_churners.sort_values('basket_trend').iloc[0]
        examples_10.append({
            'userNo': ex['userNo'], 'gender': ex['gender'],
            'age': int(ex['age']) if pd.notna(ex['age']) else 0,
            'province': ex['province'] if pd.notna(ex['province']) else '-',
            'detail_th': f"Basket ลดลง {int(abs(ex['basket_trend']))} ใบ (3 งวดล่าสุด vs ก่อนหน้า) — churned"
        })
        ex2 = declining_churners.sort_values('basket_trend').iloc[min(1, len(declining_churners)-1)]
        if ex2['userNo'] != ex['userNo']:
            examples_10.append({
                'userNo': ex2['userNo'], 'gender': ex2['gender'],
                'age': int(ex2['age']) if pd.notna(ex2['age']) else 0,
                'province': ex2['province'] if pd.notna(ex2['province']) else '-',
                'detail_th': f"Basket ลดลง {int(abs(ex2['basket_trend']))} ใบ — churned"
            })

    if len(sudden_churners) > 0:
        ex = sudden_churners.sort_values('items_last_3', ascending=False).iloc[0]
        examples_10.append({
            'userNo': ex['userNo'], 'gender': ex['gender'],
            'age': int(ex['age']) if pd.notna(ex['age']) else 0,
            'province': ex['province'] if pd.notna(ex['province']) else '-',
            'detail_th': f"Basket คงที่ (trend +{int(ex['basket_trend'])}) แต่ churn ทันที — sudden churn"
        })

    print("\n   ตัวอย่างลูกค้า:")
    print_table(examples_10)

    rec_10 = "Monitor basket trend เป็น early warning signal — ลูกค้าที่ basket ลดลง 2+ งวดติดต่อกัน ควรส่ง offer ก่อนที่จะ churn"
    print(f"   💬 Recommendation: {rec_10}")
    print()

    all_insights.append({
        'id': 10, 'team': 'Marcom',
        'title_th': 'Basket size กับ churn — ซื้อน้อยลงก่อน churn?',
        'stat': stat_10, 'recommendation_th': rec_10,
        'examples': examples_10
    })

    # ════════════════════════════════════════════════════════════
    # ทีม SEO/Content
    # ════════════════════════════════════════════════════════════
    print("─" * 60)
    print("ทีม SEO/Content")
    print("─" * 60)
    print()

    # --- Insight 11: ช่วงเปิดขายวันแรกๆ ลูกค้าน้อย ---
    print("💡 Insight 11: ช่วงเปิดขายวันแรกๆ ลูกค้าน้อย", flush=True)

    examples_11 = []
    stat_11 = "ไม่มีข้อมูล TX"
    rec_11 = ""

    if len(tx) > 0:
        tx_success = tx[tx['status'] == 'SUCCESS'].copy()
        tx_success['created'] = pd.to_datetime(tx_success['createdAt_bkk'], errors='coerce', utc=True).dt.tz_localize(None)
        tx_success['round_dt'] = pd.to_datetime(tx_success['roundDate'], errors='coerce')
        tx_success = tx_success.dropna(subset=['created', 'round_dt'])
        tx_success['days_before'] = (tx_success['round_dt'] - tx_success['created']).dt.days

        daily_counts = tx_success.groupby('days_before').agg(
            customers=('userNo', 'nunique'),
            orders=('userNo', 'count')
        ).sort_index(ascending=False)

        print("   ลูกค้าตามจำนวนวันก่อน draw:")
        for days, row in daily_counts.iterrows():
            if 0 <= days <= 13:
                bar = '█' * max(1, int(row['customers'] / daily_counts['customers'].max() * 30))
                print(f"     {days:2d} วันก่อน: {row['customers']:>8,} คน {bar}")

        # Day 1 buyers (buy on first day of window = 12-13 days before)
        day1_users = tx_success[tx_success['days_before'] >= 10]['userNo'].unique()
        last_day_users = tx_success[tx_success['days_before'] <= 0]['userNo'].unique()

        stat_11 = f"วันแรก (10+ วันก่อน draw): {len(day1_users):,} คน, วันสุดท้าย: {len(last_day_users):,} คน (มากกว่า {len(last_day_users)/max(len(day1_users),1):.1f}x)"

        # Find consistent day-1 buyer
        for uid in day1_users[:30]:
            if len(examples_11) >= 2:
                break
            cust_data = df_feat[df_feat['userNo'] == uid]
            if len(cust_data) > 0:
                cust = cust_data.iloc[0]
                examples_11.append({
                    'userNo': uid, 'gender': cust['gender'],
                    'age': int(cust['age']) if pd.notna(cust['age']) else 0,
                    'province': cust['province'] if pd.notna(cust['province']) else '-',
                    'detail_th': f"ซื้อวันแรกของ window (10+ วันก่อน draw) — active {int(cust['active_rounds'])} งวด"
                })

        for uid in last_day_users[:30]:
            if len(examples_11) >= 3:
                break
            cust_data = df_feat[df_feat['userNo'] == uid]
            if len(cust_data) > 0:
                cust = cust_data.iloc[0]
                examples_11.append({
                    'userNo': uid, 'gender': cust['gender'],
                    'age': int(cust['age']) if pd.notna(cust['age']) else 0,
                    'province': cust['province'] if pd.notna(cust['province']) else '-',
                    'detail_th': f"ซื้อวันสุดท้าย (วัน draw) — active {int(cust['active_rounds'])} งวด"
                })

        rec_11 = "สร้าง content กระตุ้นซื้อเร็วขึ้น: 'ซื้อก่อนได้ก่อน', 'เลขดังหมดเร็ว' — ลด server load วันท้ายและเพิ่ม retention"
        print(f"\n   💬 Recommendation: {rec_11}")

    print("\n   ตัวอย่างลูกค้า:")
    print_table(examples_11)

    all_insights.append({
        'id': 11, 'team': 'SEO/Content',
        'title_th': 'ช่วงเปิดขายวันแรกๆ ลูกค้าน้อย',
        'stat': stat_11, 'recommendation_th': rec_11,
        'examples': examples_11
    })

    # --- Insight 12: ลูกค้าใหม่ churn สูงมาก ---
    print("\n💡 Insight 12: ลูกค้าใหม่ (tenure <= 3) churn สูงมาก", flush=True)

    new_cust = eligible[eligible['is_new'] == 1]
    mature_cust = eligible[eligible['is_new'] == 0]

    new_rate = new_cust['churn'].mean() * 100 if len(new_cust) > 0 else 0
    mature_rate = mature_cust['churn'].mean() * 100 if len(mature_cust) > 0 else 0

    stat_12 = f"ใหม่ (tenure <= 3): churn {new_rate:.1f}% ({len(new_cust):,} คน), เก่า: churn {mature_rate:.1f}% ({len(mature_cust):,} คน)"
    print(f"   สถิติ: {stat_12}")

    examples_12 = []
    # New customers who churned
    new_churned = new_cust[new_cust['churn'] == 1]
    for _, ex in new_churned.head(3).iterrows():
        examples_12.append({
            'userNo': ex['userNo'], 'gender': ex['gender'],
            'age': int(ex['age']) if pd.notna(ex['age']) else 0,
            'province': ex['province'] if pd.notna(ex['province']) else '-',
            'detail_th': f"ลูกค้าใหม่ (tenure {int(ex['tenure'])}) — ซื้อ {int(ex['total_items'])} ใบแล้ว churn"
        })
    # New customers who stayed
    new_stayed = new_cust[new_cust['churn'] == 0]
    for _, ex in new_stayed.head(3).iterrows():
        examples_12.append({
            'userNo': ex['userNo'], 'gender': ex['gender'],
            'age': int(ex['age']) if pd.notna(ex['age']) else 0,
            'province': ex['province'] if pd.notna(ex['province']) else '-',
            'detail_th': f"ลูกค้าใหม่ (tenure {int(ex['tenure'])}) — ซื้อ {int(ex['total_items'])} ใบ ยังอยู่! items_last_1={int(ex['items_last_1'])}"
        })

    # Limit to 3 most interesting
    examples_12 = examples_12[:3]

    print("\n   ตัวอย่างลูกค้า:")
    print_table(examples_12)

    # What differentiates?
    if len(new_churned) > 0 and len(new_stayed) > 0:
        print(f"   ลูกค้าใหม่ที่อยู่ vs ที่ churn:")
        print(f"     ที่อยู่: avg items = {new_stayed['total_items'].mean():.1f}, freq = {new_stayed['purchase_freq'].mean()*100:.0f}%")
        print(f"     ที่ churn: avg items = {new_churned['total_items'].mean():.1f}, freq = {new_churned['purchase_freq'].mean()*100:.0f}%")

    rec_12 = "สร้าง onboarding content สำหรับลูกค้าใหม่: วิธีเลือกเลข, ขั้นตอนซื้อ, FAQ — ช่วย retention ใน 3 งวดแรก"
    print(f"\n   💬 Recommendation: {rec_12}")
    print()

    all_insights.append({
        'id': 12, 'team': 'SEO/Content',
        'title_th': 'ลูกค้าใหม่ (tenure <= 3) churn สูงมาก',
        'stat': stat_12, 'recommendation_th': rec_12,
        'examples': examples_12
    })

    # --- Insight 13: ลูกค้าที่ซื้อสม่ำเสมอแทบไม่ churn ---
    print("💡 Insight 13: ลูกค้าที่ซื้อสม่ำเสมอ (low gap variance) แทบไม่ churn", flush=True)

    # Very consistent: purchase_freq >= 0.9 and tenure >= 10
    consistent = eligible[(eligible['purchase_freq'] >= 0.9) & (eligible['tenure'] >= 10)]
    inconsistent = eligible[(eligible['purchase_freq'] < 0.5) & (eligible['tenure'] >= 10)]

    con_rate = consistent['churn'].mean() * 100 if len(consistent) > 0 else 0
    incon_rate = inconsistent['churn'].mean() * 100 if len(inconsistent) > 0 else 0

    stat_13 = f"สม่ำเสมอ (freq >= 90%, tenure >= 10): churn {con_rate:.1f}% ({len(consistent):,} คน), ไม่สม่ำเสมอ (freq < 50%): churn {incon_rate:.1f}% ({len(inconsistent):,} คน)"
    print(f"   สถิติ: {stat_13}")

    examples_13 = []
    con_stayed = consistent[consistent['churn'] == 0].sort_values('active_rounds', ascending=False)
    for _, ex in con_stayed.head(3).iterrows():
        examples_13.append({
            'userNo': ex['userNo'], 'gender': ex['gender'],
            'age': int(ex['age']) if pd.notna(ex['age']) else 0,
            'province': ex['province'] if pd.notna(ex['province']) else '-',
            'detail_th': f"ซื้อ {int(ex['active_rounds'])}/{int(ex['tenure'])} งวด ({ex['purchase_freq']*100:.0f}%) — เฉลี่ย {ex['avg_items']:.1f} ใบ/งวด, ไม่เคย churn"
        })

    print("\n   ตัวอย่างลูกค้า:")
    print_table(examples_13)

    rec_13 = "สร้าง content ที่กระตุ้น routine: 'อย่าลืมงวดนี้', 'เตือนซื้อลอตเตอรี่' — ช่วยให้ลูกค้ารักษาความสม่ำเสมอ"
    print(f"   💬 Recommendation: {rec_13}")
    print()

    all_insights.append({
        'id': 13, 'team': 'SEO/Content',
        'title_th': 'ลูกค้าที่ซื้อสม่ำเสมอ (low gap variance) แทบไม่ churn',
        'stat': stat_13, 'recommendation_th': rec_13,
        'examples': examples_13
    })

    # ════════════════════════════════════════════════════════════
    # ทีม Management/Strategy
    # ════════════════════════════════════════════════════════════
    print("─" * 60)
    print("ทีม Management/Strategy")
    print("─" * 60)
    print()

    # --- Insight 14: Revenue at Risk ---
    print("💡 Insight 14: Revenue at Risk — ลูกค้า Critical ซื้อเฉลี่ยเท่าไหร่", flush=True)

    # Merge risk scores if available
    if risk_score is not None:
        merged = eligible.merge(risk_score[['userNo', 'churn_score', 'risk_level']], on='userNo', how='left')
    elif risk_exp is not None:
        merged = eligible.merge(risk_exp[['userNo', 'churn_score', 'risk_level']], on='userNo', how='left')
    else:
        # Create approximate risk levels from purchase_freq
        merged = eligible.copy()
        merged['risk_level'] = pd.cut(
            1 - merged['purchase_freq'],
            bins=[0, 0.25, 0.5, 0.75, 1.01],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        merged['churn_score'] = (1 - merged['purchase_freq']) * 100

    risk_stats = merged.groupby('risk_level').agg(
        count=('total_items', 'count'),
        avg_items=('avg_items', 'mean'),
        total_items=('total_items', 'sum'),
        churn_rate=('churn', 'mean')
    ).reset_index()

    print("   Revenue by Risk Level:")
    for _, row in risk_stats.iterrows():
        print(f"     {row['risk_level']}: {row['count']:,} คน, avg {row['avg_items']:.1f} ใบ/งวด, churn {row['churn_rate']*100:.1f}%")

    # Revenue concentration (Pareto)
    sorted_items = eligible.sort_values('total_items', ascending=False)
    top_10pct_n = len(sorted_items) // 10
    top_10pct_items = sorted_items.head(top_10pct_n)['total_items'].sum()
    total_all_items = sorted_items['total_items'].sum()
    pareto_pct = top_10pct_items / total_all_items * 100

    stat_14 = f"Top 10% ลูกค้า = {pareto_pct:.1f}% ของ volume ทั้งหมด"
    print(f"\n   {stat_14}")

    examples_14 = []
    # High-value at-risk
    if 'churn_score' in merged.columns:
        at_risk_high_value = merged[
            (merged['risk_level'].isin(['Critical', 'High'])) &
            (merged['total_items'] >= merged['total_items'].quantile(0.9))
        ].sort_values('total_items', ascending=False)

        for _, ex in at_risk_high_value.head(3).iterrows():
            score_str = f"score {ex['churn_score']:.0f}" if pd.notna(ex.get('churn_score')) else ex['risk_level']
            examples_14.append({
                'userNo': ex['userNo'], 'gender': ex['gender'],
                'age': int(ex['age']) if pd.notna(ex['age']) else 0,
                'province': ex['province'] if pd.notna(ex['province']) else '-',
                'detail_th': f"Risk {ex['risk_level']} ({score_str}) — รวม {int(ex['total_items'])} ใบ, avg {ex['avg_items']:.0f} ใบ/งวด"
            })

    print("\n   ตัวอย่างลูกค้า High-value At-risk:")
    print_table(examples_14)

    rec_14 = "ลูกค้า top 10% สร้าง revenue เกือบครึ่ง — ต้อง priority retention สำหรับ high-value + high-risk customers"
    print(f"   💬 Recommendation: {rec_14}")
    print()

    all_insights.append({
        'id': 14, 'team': 'Management',
        'title_th': 'Revenue at Risk — ลูกค้า Critical ซื้อเฉลี่ยเท่าไหร่',
        'stat': stat_14, 'recommendation_th': rec_14,
        'examples': examples_14
    })

    # --- Insight 15: Customer Lifetime ---
    print("💡 Insight 15: Customer Lifetime — ลูกค้าอยู่กับเราเฉลี่ยกี่งวด", flush=True)

    avg_tenure = df_feat['tenure'].mean()
    median_tenure = df_feat['tenure'].median()
    avg_active = df_feat['active_rounds'].mean()
    avg_lifetime_items = df_feat['total_items'].mean()

    stat_15 = f"Avg tenure: {avg_tenure:.1f} งวด, median: {median_tenure:.0f} งวด, avg active: {avg_active:.1f} งวด, avg lifetime tickets: {avg_lifetime_items:.0f} ใบ"
    print(f"   สถิติ: {stat_15}")

    # Longest active
    longest = df_feat.sort_values('active_rounds', ascending=False).iloc[0]
    most_items = df_feat.sort_values('total_items', ascending=False).iloc[0]

    # Tenure distribution
    tenure_dist = df_feat['tenure'].describe()
    print(f"\n   Tenure distribution:")
    print(f"     Min: {tenure_dist['min']:.0f}, 25%: {tenure_dist['25%']:.0f}, 50%: {tenure_dist['50%']:.0f}, 75%: {tenure_dist['75%']:.0f}, Max: {tenure_dist['max']:.0f}")

    examples_15 = []
    examples_15.append({
        'userNo': longest['userNo'], 'gender': longest['gender'],
        'age': int(longest['age']) if pd.notna(longest['age']) else 0,
        'province': longest['province'] if pd.notna(longest['province']) else '-',
        'detail_th': f"ลูกค้าที่ active มากที่สุด — {int(longest['active_rounds'])} งวด จาก {int(longest['tenure'])} งวด, รวม {int(longest['total_items'])} ใบ"
    })
    if most_items['userNo'] != longest['userNo']:
        examples_15.append({
            'userNo': most_items['userNo'], 'gender': most_items['gender'],
            'age': int(most_items['age']) if pd.notna(most_items['age']) else 0,
            'province': most_items['province'] if pd.notna(most_items['province']) else '-',
            'detail_th': f"ลูกค้าซื้อมากที่สุด — {int(most_items['total_items'])} ใบ, active {int(most_items['active_rounds'])} งวด"
        })

    # Average customer
    avg_cust = df_feat.iloc[(df_feat['active_rounds'] - avg_active).abs().argsort().iloc[0]]
    examples_15.append({
        'userNo': avg_cust['userNo'], 'gender': avg_cust['gender'],
        'age': int(avg_cust['age']) if pd.notna(avg_cust['age']) else 0,
        'province': avg_cust['province'] if pd.notna(avg_cust['province']) else '-',
        'detail_th': f"ลูกค้าทั่วไป — {int(avg_cust['active_rounds'])} งวด, {int(avg_cust['total_items'])} ใบ (ใกล้เคียง average)"
    })

    print("\n   ตัวอย่างลูกค้า:")
    print_table(examples_15)

    rec_15 = f"Customer lifetime เฉลี่ย {avg_active:.0f} งวด active จาก {avg_tenure:.0f} งวด — โฟกัสเพิ่ม active ratio ผ่าน regular engagement"
    print(f"   💬 Recommendation: {rec_15}")
    print()

    all_insights.append({
        'id': 15, 'team': 'Management',
        'title_th': 'Customer Lifetime — ลูกค้าอยู่กับเราเฉลี่ยกี่งวด',
        'stat': stat_15, 'recommendation_th': rec_15,
        'examples': examples_15
    })

    # ════════════════════════════════════════════════════════════
    # Save JSON
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Saving outputs...", flush=True)

    # Clean examples for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if pd.isna(obj):
            return None
        return obj

    for ins in all_insights:
        for key in ['stat', 'recommendation_th', 'title_th']:
            ins[key] = clean_for_json(ins.get(key, ''))
        for ex in ins.get('examples', []):
            for k, v in ex.items():
                ex[k] = clean_for_json(v)

    catalog = {
        'generated': datetime.now().strftime('%Y-%m-%d'),
        'total_insights': len(all_insights),
        'total_customers_analyzed': int(total_eligible),
        'churn_rate': round(float(churn_rate), 1),
        'insights': all_insights
    }

    json_path = os.path.join(PRED_DIR, 'insight_catalog.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2, default=str)
    print(f"  JSON saved: {json_path}", flush=True)

    # ════════════════════════════════════════════════════════════
    # Generate Charts
    # ════════════════════════════════════════════════════════════
    print("  Generating charts...", flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Panel 1: Age × Gender churn heatmap
    ax = axes[0, 0]
    if len(eligible_age) > 0:
        pivot = eligible_age.groupby(['age_group', 'gender'], observed=True)['churn'].mean().unstack(fill_value=0) * 100
        im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([gender_thai(g) for g in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f'{pivot.values[i, j]:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Churn Rate: Age x Gender (%)', fontsize=11, fontweight='bold')

    # Panel 2: Purchase hour distribution
    ax = axes[0, 1]
    if len(tx) > 0:
        tx_s = tx[tx['status'] == 'SUCCESS'].copy()
        tx_s['created'] = pd.to_datetime(tx_s['createdAt_bkk'], errors='coerce', utc=True).dt.tz_localize(None)
        tx_s = tx_s.dropna(subset=['created'])
        hour_counts = tx_s['created'].dt.hour.value_counts().sort_index()
        ax.bar(hour_counts.index, hour_counts.values, color='steelblue', alpha=0.8)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Orders')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
    ax.set_title('Purchase Timing (Hour of Day)', fontsize=11, fontweight='bold')

    # Panel 3: Basket size by risk level
    ax = axes[0, 2]
    if 'risk_level' in merged.columns:
        risk_order = ['Low', 'Medium', 'High', 'Critical', 'New Customer']
        risk_available = [r for r in risk_order if r in merged['risk_level'].values]
        box_data = [merged[merged['risk_level'] == r]['avg_items'].dropna().clip(upper=50).values for r in risk_available]
        if box_data and any(len(d) > 0 for d in box_data):
            bp = ax.boxplot(box_data, labels=risk_available, patch_artist=True)
            colors_map = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e67e22', 'Critical': '#e74c3c', 'New Customer': '#95a5a6'}
            for patch, label in zip(bp['boxes'], risk_available):
                patch.set_facecolor(colors_map.get(label, '#bdc3c7'))
    ax.set_title('Avg Items/Period by Risk Level', fontsize=11, fontweight='bold')
    ax.set_ylabel('Items per Period')

    # Panel 4: Province churn (top/bottom 10)
    ax = axes[1, 0]
    if len(prov_min_1000) > 0:
        top_bottom = pd.concat([prov_min_1000.head(5), prov_min_1000.tail(5)])
        top_bottom = top_bottom.sort_values('rate')
        colors_prov = ['#2ecc71'] * 5 + ['#e74c3c'] * 5
        bars = ax.barh(range(len(top_bottom)), top_bottom['rate'].values, color=colors_prov)
        ax.set_yticks(range(len(top_bottom)))
        ax.set_yticklabels(top_bottom['province'].values, fontsize=8)
        for i, (_, row) in enumerate(top_bottom.iterrows()):
            ax.text(row['rate'] + 0.3, i, f"{row['rate']:.1f}%", va='center', fontsize=8)
    ax.set_title('Province Churn Rate (Top/Bottom 5)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Churn Rate (%)')

    # Panel 5: New vs Mature churn
    ax = axes[1, 1]
    labels_nm = ['New\n(tenure<=3)', 'Mature\n(tenure>3)']
    rates_nm = [new_rate, mature_rate]
    counts_nm = [len(new_cust), len(mature_cust)]
    bars = ax.bar(labels_nm, rates_nm, color=['#e74c3c', '#2ecc71'], alpha=0.8)
    for bar, rate, count in zip(bars, rates_nm, counts_nm):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rate:.1f}%\n({count:,})', ha='center', fontsize=10)
    ax.set_title('New vs Mature Customer Churn', fontsize=11, fontweight='bold')
    ax.set_ylabel('Churn Rate (%)')

    # Panel 6: Revenue concentration (Pareto)
    ax = axes[1, 2]
    sorted_total = np.sort(eligible['total_items'].values)[::-1]
    cumsum = np.cumsum(sorted_total)
    cumsum_pct = cumsum / cumsum[-1] * 100
    x_pct = np.arange(1, len(cumsum) + 1) / len(cumsum) * 100
    ax.plot(x_pct, cumsum_pct, 'b-', linewidth=2)
    ax.axhline(80, color='red', linestyle='--', alpha=0.5, label='80% revenue')
    # Find x where cumsum reaches 80%
    idx_80 = np.searchsorted(cumsum_pct, 80)
    pct_80 = x_pct[idx_80] if idx_80 < len(x_pct) else 100
    ax.axvline(pct_80, color='red', linestyle='--', alpha=0.5)
    ax.text(pct_80 + 1, 75, f'Top {pct_80:.0f}% = 80% volume', fontsize=9, color='red')
    ax.set_xlabel('% of Customers (sorted by volume)')
    ax.set_ylabel('% of Total Volume')
    ax.set_title('Revenue Concentration (Pareto)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

    plt.suptitle('Phase 41: Insight Catalog Dashboard', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    chart_path = os.path.join(CHART_DIR, 'phase41_insight_catalog.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {chart_path}", flush=True)

    # ════════════════════════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY", flush=True)
    print("=" * 60)
    print(f"  Total insights: {len(all_insights)}")
    print(f"  Total examples: {sum(len(ins.get('examples', [])) for ins in all_insights)}")
    print(f"  Teams covered: Ads (5), Marcom (5), SEO/Content (3), Management (2)")
    print(f"  JSON: {json_path}")
    print(f"  Chart: {chart_path}")
    print("=" * 60)
    print("Phase 41 complete!", flush=True)


if __name__ == '__main__':
    main()
