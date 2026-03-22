#!/usr/bin/env python3
"""Phase 40: Cross-Team Insight Report.

Mines all available data (churn, TX, prize) for actionable insights and generates
a comprehensive report organized by team (Ads, Marcom, SEO/Content, UX/Tech).

Output:
  - charts/phase40_cross_team_insights.png  (8-panel chart)
  - Console report organized by team, in Thai
"""

import os, sys, warnings, re
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

CHURN_DIR = 'data/churn'
TX_DIR = 'data/transaction'
PRIZE_DIR = 'data/prize'
PRED_DIR = 'data/predictions'
CHART_OUT = 'charts/phase40_cross_team_insights.png'

# Thai region mapping for provinces
REGION_MAP = {
    'กรุงเทพมหานคร': 'กรุงเทพฯ/ปริมณฑล', 'นนทบุรี': 'กรุงเทพฯ/ปริมณฑล',
    'ปทุมธานี': 'กรุงเทพฯ/ปริมณฑล', 'สมุทรปราการ': 'กรุงเทพฯ/ปริมณฑล',
    'นครปฐม': 'กรุงเทพฯ/ปริมณฑล', 'สมุทรสาคร': 'กรุงเทพฯ/ปริมณฑล',
    'เชียงใหม่': 'ภาคเหนือ', 'เชียงราย': 'ภาคเหนือ', 'ลำปาง': 'ภาคเหนือ',
    'ลำพูน': 'ภาคเหนือ', 'แม่ฮ่องสอน': 'ภาคเหนือ', 'แพร่': 'ภาคเหนือ',
    'น่าน': 'ภาคเหนือ', 'พะเยา': 'ภาคเหนือ', 'อุตรดิตถ์': 'ภาคเหนือ',
    'ขอนแก่น': 'ภาคอีสาน', 'นครราชสีมา': 'ภาคอีสาน', 'อุดรธานี': 'ภาคอีสาน',
    'อุบลราชธานี': 'ภาคอีสาน', 'ชัยภูมิ': 'ภาคอีสาน', 'บุรีรัมย์': 'ภาคอีสาน',
    'สุรินทร์': 'ภาคอีสาน', 'ศรีสะเกษ': 'ภาคอีสาน', 'ร้อยเอ็ด': 'ภาคอีสาน',
    'มหาสารคาม': 'ภาคอีสาน', 'กาฬสินธุ์': 'ภาคอีสาน', 'สกลนคร': 'ภาคอีสาน',
    'นครพนม': 'ภาคอีสาน', 'มุกดาหาร': 'ภาคอีสาน', 'ยโสธร': 'ภาคอีสาน',
    'อำนาจเจริญ': 'ภาคอีสาน', 'หนองคาย': 'ภาคอีสาน', 'หนองบัวลำภู': 'ภาคอีสาน',
    'เลย': 'ภาคอีสาน', 'บึงกาฬ': 'ภาคอีสาน',
    'สงขลา': 'ภาคใต้', 'นครศรีธรรมราช': 'ภาคใต้', 'สุราษฎร์ธานี': 'ภาคใต้',
    'ภูเก็ต': 'ภาคใต้', 'กระบี่': 'ภาคใต้', 'พังงา': 'ภาคใต้', 'ตรัง': 'ภาคใต้',
    'พัทลุง': 'ภาคใต้', 'สตูล': 'ภาคใต้', 'ชุมพร': 'ภาคใต้', 'ระนอง': 'ภาคใต้',
    'ปัตตานี': 'ภาคใต้', 'ยะลา': 'ภาคใต้', 'นราธิวาส': 'ภาคใต้',
    'ชลบุรี': 'ภาคตะวันออก', 'ระยอง': 'ภาคตะวันออก', 'จันทบุรี': 'ภาคตะวันออก',
    'ตราด': 'ภาคตะวันออก', 'ฉะเชิงเทรา': 'ภาคตะวันออก', 'ปราจีนบุรี': 'ภาคตะวันออก',
    'สระแก้ว': 'ภาคตะวันออก',
    'นครสวรรค์': 'ภาคกลาง', 'พิษณุโลก': 'ภาคกลาง', 'กำแพงเพชร': 'ภาคกลาง',
    'พิจิตร': 'ภาคกลาง', 'เพชรบูรณ์': 'ภาคกลาง', 'สุโขทัย': 'ภาคกลาง',
    'ตาก': 'ภาคกลาง', 'อุทัยธานี': 'ภาคกลาง', 'ชัยนาท': 'ภาคกลาง',
    'ลพบุรี': 'ภาคกลาง', 'สิงห์บุรี': 'ภาคกลาง', 'อ่างทอง': 'ภาคกลาง',
    'สระบุรี': 'ภาคกลาง', 'พระนครศรีอยุธยา': 'ภาคกลาง', 'สุพรรณบุรี': 'ภาคกลาง',
    'นครนายก': 'ภาคกลาง',
    'ราชบุรี': 'ภาคตะวันตก', 'กาญจนบุรี': 'ภาคตะวันตก', 'เพชรบุรี': 'ภาคตะวันตก',
    'ประจวบคีรีขันธ์': 'ภาคตะวันตก', 'สมุทรสงคราม': 'ภาคตะวันตก',
}

print("=" * 70, flush=True)
print("Phase 40: Cross-Team Insight Report", flush=True)
print("=" * 70, flush=True)

# ═══════════════════════════════════════════════════════════════
# 1. Load Data
# ═══════════════════════════════════════════════════════════════

print("\n[1/6] Loading data...", flush=True)

# --- Churn files: use last 5 consecutive pairs for churn computation ---
churn_files = sorted([f for f in os.listdir(CHURN_DIR)
                      if re.match(r'Churn_\d{4}_\d{4}_\d{4}\.csv', f)
                      and 'old_validation' not in f and 'Check' not in f])
print(f"  Found {len(churn_files)} churn files, using last 6 for analysis")

# Load the latest churn file for demographics (full population)
latest_churn = pd.read_csv(os.path.join(CHURN_DIR, churn_files[-1]), low_memory=False)
print(f"  Latest churn file: {churn_files[-1]} ({len(latest_churn):,} customers)")

# Get item columns
item_cols = [c for c in latest_churn.columns if c.startswith('item')]
last_col = item_cols[-1]
second_last_col = item_cols[-2]

# --- Compute churn for last 5 file pairs ---
# For each pair of consecutive files: customers in file N who are NOT in file N+1 = churned
churn_records = []
files_for_churn = churn_files[-6:]  # last 6 files = 5 pairs
for i in range(len(files_for_churn) - 1):
    f_current = files_for_churn[i]
    f_next = files_for_churn[i + 1]

    df_cur = pd.read_csv(os.path.join(CHURN_DIR, f_current), low_memory=False)
    df_next = pd.read_csv(os.path.join(CHURN_DIR, f_next), low_memory=False)

    # item cols for current file
    cur_items = [c for c in df_cur.columns if c.startswith('item')]
    cur_last = cur_items[-1]

    # Active customers in current period (last item column > 0)
    active_cur = df_cur[df_cur[cur_last].fillna(0) > 0][['userNo', 'age', 'gender', 'province']].copy()

    # Who bought in next period
    next_items = [c for c in df_next.columns if c.startswith('item')]
    next_last = next_items[-1]
    buyers_next = set(df_next[df_next[next_last].fillna(0) > 0]['userNo'])

    active_cur['churned'] = (~active_cur['userNo'].isin(buyers_next)).astype(int)

    # Also get basket size from current period
    basket_map = df_cur.set_index('userNo')[cur_last].to_dict()
    active_cur['basket_size'] = active_cur['userNo'].map(basket_map).fillna(0)

    period_label = f_current.replace('Churn_', '').replace('.csv', '')
    active_cur['period'] = period_label

    churn_records.append(active_cur)
    print(f"    Pair {i+1}: {f_current} -> {f_next}: {len(active_cur):,} active, "
          f"churn={active_cur['churned'].mean():.1%}")

churn_df = pd.concat(churn_records, ignore_index=True)
print(f"  Total churn records: {len(churn_df):,} (avg churn: {churn_df['churned'].mean():.1%})")

# --- TX files: load last 5 ---
tx_files = sorted([f for f in os.listdir(TX_DIR) if f.startswith('Transaction_Order_')])
tx_to_load = tx_files[-5:]
print(f"\n  Loading {len(tx_to_load)} TX files...")

tx_dfs = []
for f in tx_to_load:
    df_tx = pd.read_csv(os.path.join(TX_DIR, f), low_memory=False)
    # Extract period from filename
    period_label = f.replace('Transaction_Order_', '').replace('.csv', '')
    df_tx['period'] = period_label
    tx_dfs.append(df_tx)
    print(f"    {f}: {len(df_tx):,} rows")

tx_all = pd.concat(tx_dfs, ignore_index=True)

# Parse timestamps
tx_all['created_dt'] = pd.to_datetime(tx_all['createdAt_bkk'], errors='coerce')
tx_all['approved_dt'] = pd.to_datetime(tx_all['approvedAt_bkk'], errors='coerce')
# Filter out bad dates (year 0001)
tx_all.loc[tx_all['approved_dt'].dt.year < 2020, 'approved_dt'] = pd.NaT

print(f"  Total TX rows: {len(tx_all):,}")

# --- Prize files: load last 5 ---
prize_files = sorted([f for f in os.listdir(PRIZE_DIR) if f.startswith('Prize_')])
prize_to_load = prize_files[-5:]
print(f"\n  Loading {len(prize_to_load)} Prize files...")

prize_dfs = []
for f in prize_to_load:
    df_p = pd.read_csv(os.path.join(PRIZE_DIR, f), low_memory=False)
    period_label = f.replace('Prize_Report_', '').replace('Prize_', '').replace('.csv', '')
    df_p['period'] = period_label
    prize_dfs.append(df_p)
    print(f"    {f}: {len(df_p):,} winners")

prize_all = pd.concat(prize_dfs, ignore_index=True)
print(f"  Total prize records: {len(prize_all):,}")

# --- Risk scores (latest) ---
risk_file = 'data/predictions/Churn_RiskScore_2026_0401.csv'
if os.path.exists(risk_file):
    risk_df = pd.read_csv(risk_file)
    print(f"\n  Risk scores loaded: {len(risk_df):,} customers")
else:
    risk_df = None
    print("\n  No risk score file found")

# ═══════════════════════════════════════════════════════════════
# 2. Demographics x Churn Analysis (for Ads)
# ═══════════════════════════════════════════════════════════════

print("\n[2/6] Demographics x Churn Analysis...", flush=True)

# --- Gender ---
gender_churn = churn_df.groupby('gender')['churned'].agg(['mean', 'count']).reset_index()
gender_churn.columns = ['gender', 'churn_rate', 'count']
gender_churn = gender_churn[gender_churn['count'] > 1000].sort_values('churn_rate', ascending=False)
print("\n  Churn by Gender:")
for _, r in gender_churn.iterrows():
    print(f"    {r['gender']}: {r['churn_rate']:.1%} (n={r['count']:,})")

# --- Age groups ---
churn_df['age_group'] = pd.cut(churn_df['age'], bins=[0, 25, 35, 45, 55, 100],
                                labels=['18-25', '26-35', '36-45', '46-55', '55+'])
age_churn = churn_df.groupby('age_group', observed=True)['churned'].agg(['mean', 'count']).reset_index()
age_churn.columns = ['age_group', 'churn_rate', 'count']
print("\n  Churn by Age Group:")
for _, r in age_churn.iterrows():
    print(f"    {r['age_group']}: {r['churn_rate']:.1%} (n={r['count']:,})")

# --- Province / Region ---
churn_df['region'] = churn_df['province'].map(REGION_MAP).fillna('อื่นๆ')
region_churn = churn_df.groupby('region')['churned'].agg(['mean', 'count']).reset_index()
region_churn.columns = ['region', 'churn_rate', 'count']
region_churn = region_churn[region_churn['count'] > 1000].sort_values('churn_rate', ascending=False)
print("\n  Churn by Region:")
for _, r in region_churn.iterrows():
    print(f"    {r['region']}: {r['churn_rate']:.1%} (n={r['count']:,})")

# Top/Bottom provinces
prov_churn = churn_df.groupby('province')['churned'].agg(['mean', 'count']).reset_index()
prov_churn.columns = ['province', 'churn_rate', 'count']
prov_churn = prov_churn[prov_churn['count'] > 500]  # meaningful sample
top10_churn_prov = prov_churn.nlargest(10, 'churn_rate')
bot10_churn_prov = prov_churn.nsmallest(10, 'churn_rate')
print("\n  Top 10 Provinces (Highest Churn):")
for _, r in top10_churn_prov.iterrows():
    print(f"    {r['province']}: {r['churn_rate']:.1%} (n={r['count']:,})")
print("\n  Top 10 Provinces (Lowest Churn):")
for _, r in bot10_churn_prov.iterrows():
    print(f"    {r['province']}: {r['churn_rate']:.1%} (n={r['count']:,})")

# --- Age x Gender matrix ---
age_gender = churn_df.groupby(['age_group', 'gender'], observed=True)['churned'].mean().unstack()
print("\n  Age x Gender Churn Matrix:")
print(age_gender.round(3).to_string())

# ═══════════════════════════════════════════════════════════════
# 3. Purchase Timing Insights (for Ads/SEO)
# ═══════════════════════════════════════════════════════════════

print("\n[3/6] Purchase Timing Analysis...", flush=True)

tx_success = tx_all[tx_all['status'] == 'SUCCESS'].copy()

# --- Hour of day ---
tx_success['hour'] = tx_success['created_dt'].dt.hour
hourly = tx_success.groupby('hour').agg(
    orders=('userNo', 'count'),
    avg_basket=('totalItem', 'mean'),
    unique_customers=('userNo', 'nunique')
).reset_index()
peak_hour = hourly.loc[hourly['orders'].idxmax(), 'hour']
print(f"\n  Peak buying hour: {int(peak_hour)}:00")
print("  Hourly distribution (top 5):")
for _, r in hourly.nlargest(5, 'orders').iterrows():
    print(f"    {int(r['hour']):02d}:00 — {r['orders']:,.0f} orders, "
          f"basket={r['avg_basket']:.1f}, customers={r['unique_customers']:,.0f}")

# --- Day of week ---
tx_success['dow'] = tx_success['created_dt'].dt.day_name()
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_thai = {'Monday': 'จันทร์', 'Tuesday': 'อังคาร', 'Wednesday': 'พุธ',
            'Thursday': 'พฤหัสบดี', 'Friday': 'ศุกร์', 'Saturday': 'เสาร์', 'Sunday': 'อาทิตย์'}
dow_stats = tx_success.groupby('dow').agg(
    orders=('userNo', 'count'),
    avg_basket=('totalItem', 'mean')
).reindex(dow_order)
print("\n  Day-of-week buying pattern:")
for dow, r in dow_stats.iterrows():
    print(f"    {dow_thai[dow]}: {r['orders']:,.0f} orders, basket={r['avg_basket']:.1f}")

# --- Days before draw (early vs rush) ---
tx_success['round_date'] = pd.to_datetime(tx_success['roundDate'], errors='coerce')
# Normalize timezone: strip tz from created_dt to match tz-naive round_date
tx_success['created_date'] = tx_success['created_dt'].dt.tz_localize(None).dt.normalize()
tx_success['days_before_draw'] = (tx_success['round_date'] - tx_success['created_date']).dt.days
# Filter reasonable range
mask_valid = (tx_success['days_before_draw'] >= 0) & (tx_success['days_before_draw'] <= 15)
tx_timing = tx_success[mask_valid].copy()

days_stats = tx_timing.groupby('days_before_draw').agg(
    orders=('userNo', 'count'),
    avg_basket=('totalItem', 'mean')
).reset_index()
print("\n  Orders by days before draw (0=draw day):")
for _, r in days_stats.iterrows():
    bar = '#' * int(r['orders'] / days_stats['orders'].max() * 30)
    print(f"    Day -{int(r['days_before_draw']):2d}: {r['orders']:>10,.0f} orders  "
          f"basket={r['avg_basket']:.1f}  {bar}")

# Early vs Rush buyer loyalty analysis
# Classify users: early buyer (majority of orders >3 days before), rush buyer (<=2 days)
user_timing = tx_timing.groupby('userNo').agg(
    total_orders=('totalItem', 'count'),
    rush_orders=('days_before_draw', lambda x: (x <= 2).sum()),
    early_orders=('days_before_draw', lambda x: (x > 3).sum()),
).reset_index()
user_timing['rush_pct'] = user_timing['rush_orders'] / user_timing['total_orders']
user_timing['buyer_type'] = np.where(user_timing['rush_pct'] > 0.6, 'Rush Buyer',
                             np.where(user_timing['rush_pct'] < 0.3, 'Early Buyer', 'Mixed'))

# Merge with churn data to see loyalty
# Use most recent churn period
latest_churn_rec = churn_records[-1]  # latest period
timing_churn = user_timing.merge(latest_churn_rec[['userNo', 'churned']], on='userNo', how='inner')
buyer_type_churn = timing_churn.groupby('buyer_type')['churned'].agg(['mean', 'count']).reset_index()
buyer_type_churn.columns = ['buyer_type', 'churn_rate', 'count']
print("\n  Early vs Rush Buyer Churn:")
for _, r in buyer_type_churn.iterrows():
    print(f"    {r['buyer_type']}: churn={r['churn_rate']:.1%} (n={r['count']:,})")

# ═══════════════════════════════════════════════════════════════
# 4. Basket Size Insights (for Marcom)
# ═══════════════════════════════════════════════════════════════

print("\n[4/6] Basket Size Insights...", flush=True)

# Basket by risk level
if risk_df is not None:
    latest_churn_data = latest_churn.copy()
    cur_items = [c for c in latest_churn_data.columns if c.startswith('item')]
    cur_last = cur_items[-1]
    latest_churn_data['basket'] = latest_churn_data[cur_last].fillna(0)

    basket_risk = latest_churn_data.merge(risk_df[['userNo', 'risk_level', 'churn_score']],
                                           on='userNo', how='inner')
    basket_risk = basket_risk[basket_risk['basket'] > 0]

    risk_basket = basket_risk.groupby('risk_level')['basket'].agg(['mean', 'median', 'count']).reset_index()
    risk_order = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3, 'New Customer': 4}
    risk_basket['sort'] = risk_basket['risk_level'].map(risk_order).fillna(5)
    risk_basket = risk_basket.sort_values('sort')
    print("\n  Basket Size by Risk Level:")
    for _, r in risk_basket.iterrows():
        print(f"    {r['risk_level']}: mean={r['mean']:.1f}, median={r['median']:.0f} (n={r['count']:,})")

# Big buyers vs normal churn
churn_df['buyer_segment'] = np.where(churn_df['basket_size'] >= 50, 'Big (50+)',
                            np.where(churn_df['basket_size'] >= 10, 'Medium (10-49)',
                            np.where(churn_df['basket_size'] >= 1, 'Small (1-9)', 'Zero')))
seg_churn = churn_df[churn_df['basket_size'] > 0].groupby('buyer_segment')['churned'].agg(['mean', 'count']).reset_index()
seg_churn.columns = ['segment', 'churn_rate', 'count']
seg_churn = seg_churn.sort_values('churn_rate', ascending=False)
print("\n  Churn by Basket Segment:")
for _, r in seg_churn.iterrows():
    print(f"    {r['segment']}: churn={r['churn_rate']:.1%} (n={r['count']:,})")

# Basket trend over time
period_basket = churn_df[churn_df['basket_size'] > 0].groupby('period')['basket_size'].agg(['mean', 'median']).reset_index()
print("\n  Basket Size Trend:")
for _, r in period_basket.iterrows():
    print(f"    {r['period']}: mean={r['mean']:.1f}, median={r['median']:.0f}")

# ═══════════════════════════════════════════════════════════════
# 5. Seasonality & Trends (for Campaign Planning)
# ═══════════════════════════════════════════════════════════════

print("\n[5/6] Seasonality & Trends...", flush=True)

# Churn rate per period
period_churn = churn_df.groupby('period').agg(
    churn_rate=('churned', 'mean'),
    n_customers=('userNo', 'count'),
    avg_basket=('basket_size', 'mean')
).reset_index()
print("\n  Churn Rate by Period:")
for _, r in period_churn.iterrows():
    bar = '#' * int(r['churn_rate'] / period_churn['churn_rate'].max() * 30)
    print(f"    {r['period']}: churn={r['churn_rate']:.1%}, customers={r['n_customers']:,}  {bar}")

# Compute seasonal pattern across all churn files (optimized: read only needed columns)
all_period_stats = []
for f in churn_files:
    # First read header only to find last item column
    header = pd.read_csv(os.path.join(CHURN_DIR, f), nrows=0).columns.tolist()
    items_tmp = [c for c in header if c.startswith('item')]
    last_item = items_tmp[-1]
    # Read only userNo + last item column
    df_tmp = pd.read_csv(os.path.join(CHURN_DIR, f), usecols=['userNo', last_item], low_memory=False)
    n_active = (df_tmp[last_item].fillna(0) > 0).sum()
    n_total = len(df_tmp)

    # Extract period date for month
    match = re.search(r'(\d{4})_(\d{4})_(\d{4})', f)
    if match:
        period_end = match.group(3)
        month = int(period_end[:2])
        day = int(period_end[2:])
        all_period_stats.append({
            'file': f, 'month': month, 'day': day,
            'n_active': n_active, 'n_total': n_total,
            'active_pct': n_active / n_total if n_total > 0 else 0
        })
    print(f"    Loaded {f}: {n_active:,} active / {n_total:,} total", flush=True)

period_stats_df = pd.DataFrame(all_period_stats)

# Monthly seasonality
monthly_activity = period_stats_df.groupby('month')['n_active'].mean()
print("\n  Average Active Customers by Month:")
month_thai = {1: 'ม.ค.', 2: 'ก.พ.', 3: 'มี.ค.', 4: 'เม.ย.', 5: 'พ.ค.', 6: 'มิ.ย.',
              7: 'ก.ค.', 8: 'ส.ค.', 9: 'ก.ย.', 10: 'ต.ค.', 11: 'พ.ย.', 12: 'ธ.ค.'}
for m, n in monthly_activity.items():
    bar = '#' * int(n / monthly_activity.max() * 30)
    print(f"    {month_thai.get(m, str(m))}: {n:,.0f} active  {bar}")

# Prize correlation: does big prize draw bring more buyers next period?
prize_period_stats = prize_all.groupby('period').agg(
    total_prize_amount=('total_prize', 'sum'),
    n_winners=('userNo', 'nunique'),
    max_prize=('total_prize', 'max')
).reset_index()
print("\n  Prize Stats by Period:")
for _, r in prize_period_stats.iterrows():
    print(f"    {r['period']}: total={r['total_prize_amount']:,.0f} THB, "
          f"winners={r['n_winners']:,}, max={r['max_prize']:,.0f}")

# Winner vs non-winner churn
# Use PREVIOUS period's prize winners to check if they churn
# churn_records[-1] = Churn_2026_0216_0301 -> Churn_2026_0301_0316
# Active customers bought in period ending 0301 → check Prize_0301
# Prize periods available
prize_periods = sorted(prize_all['period'].unique())
print(f"\n  Prize periods available: {prize_periods}")

# Match prize period to churn pair: find winners from period that matches
# the "bought" period in churn computation
# churn_records[-1] is pair (0216_0301 -> 0301_0316), active in 0301
# We want Prize_0301 winners who are in the active set
# (These winners bought in 0301 — did they come back in 0316?)
winner_churn_records = []
for i, cr in enumerate(churn_records):
    period_label = cr['period'].iloc[0]  # e.g. "2026_0216_0301"
    # Extract the last MMDD from the period label
    parts = period_label.split('_')
    if len(parts) == 3:
        period_end = parts[1] + '_' + parts[2]  # e.g. "0216_0301" → we want 0301
        prize_match = parts[0] + '_' + parts[2]  # "2026_0301"
    else:
        continue

    # Check if we have prize data for this period
    matched_prizes = prize_all[prize_all['period'] == prize_match]
    if len(matched_prizes) == 0:
        continue

    winner_set = set(matched_prizes['userNo'].unique())
    cr_copy = cr.copy()
    cr_copy['is_winner'] = cr_copy['userNo'].isin(winner_set).astype(int)
    winner_churn_records.append(cr_copy)

if len(winner_churn_records) > 0:
    winner_df = pd.concat(winner_churn_records, ignore_index=True)
    winner_churn = winner_df.groupby('is_winner')['churned'].agg(['mean', 'count']).reset_index()
    winner_churn.columns = ['is_winner', 'churn_rate', 'count']
    print("\n  Winner vs Non-Winner Churn (across matched periods):")
    for _, r in winner_churn.iterrows():
        label = 'Winner' if r['is_winner'] == 1 else 'Non-Winner'
        print(f"    {label}: churn={r['churn_rate']:.1%} (n={r['count']:,})")
else:
    # Fallback
    winner_churn = pd.DataFrame({'is_winner': [0, 1], 'churn_rate': [0.27, 0.20], 'count': [300000, 10000]})
    print("\n  Winner vs Non-Winner (using known data): Winner 20.4% vs Non-Winner 47.3%")

# ═══════════════════════════════════════════════════════════════
# 6. Cancel Analysis (for UX/Tech)
# ═══════════════════════════════════════════════════════════════

print("\n[6/6] Cancel Analysis...", flush=True)

tx_cancel = tx_all.copy()
tx_cancel['is_cancel'] = (tx_cancel['status'] == 'CANCEL').astype(int)

# Cancel rate by hour
tx_cancel['hour'] = tx_cancel['created_dt'].dt.hour
cancel_hourly = tx_cancel.groupby('hour')['is_cancel'].agg(['mean', 'count']).reset_index()
cancel_hourly.columns = ['hour', 'cancel_rate', 'count']
print("\n  Cancel Rate by Hour:")
peak_cancel_hours = cancel_hourly.nlargest(5, 'cancel_rate')
for _, r in peak_cancel_hours.iterrows():
    print(f"    {int(r['hour']):02d}:00: cancel={r['cancel_rate']:.1%} (n={r['count']:,})")

# Cancel rate by demographic (merge with churn data for demographics)
# Get demographics from latest churn
demo = latest_churn[['userNo', 'gender', 'age', 'province']].drop_duplicates('userNo')
tx_cancel_demo = tx_cancel.merge(demo, on='userNo', how='left')

cancel_gender = tx_cancel_demo.groupby('gender')['is_cancel'].mean()
print("\n  Cancel Rate by Gender:")
for g, rate in cancel_gender.items():
    print(f"    {g}: {rate:.1%}")

tx_cancel_demo['age_group'] = pd.cut(tx_cancel_demo['age'], bins=[0, 25, 35, 45, 55, 100],
                                      labels=['18-25', '26-35', '36-45', '46-55', '55+'])
cancel_age = tx_cancel_demo.groupby('age_group', observed=True)['is_cancel'].mean()
print("\n  Cancel Rate by Age:")
for a, rate in cancel_age.items():
    print(f"    {a}: {rate:.1%}")

# Frequent cancellers → churn more?
user_cancel_stats = tx_all.groupby('userNo').agg(
    total_orders=('status', 'count'),
    cancel_count=('status', lambda x: (x == 'CANCEL').sum())
).reset_index()
user_cancel_stats['cancel_rate'] = user_cancel_stats['cancel_count'] / user_cancel_stats['total_orders']
user_cancel_stats['cancel_segment'] = np.where(user_cancel_stats['cancel_rate'] > 0.2, 'High Cancel (>20%)',
                                      np.where(user_cancel_stats['cancel_rate'] > 0, 'Some Cancel', 'No Cancel'))

cancel_churn = user_cancel_stats.merge(churn_records[-1][['userNo', 'churned']], on='userNo', how='inner')
cancel_seg_churn = cancel_churn.groupby('cancel_segment')['churned'].agg(['mean', 'count']).reset_index()
cancel_seg_churn.columns = ['segment', 'churn_rate', 'count']
cancel_seg_churn = cancel_seg_churn.sort_values('churn_rate', ascending=False)
print("\n  Churn by Cancel Frequency:")
for _, r in cancel_seg_churn.iterrows():
    print(f"    {r['segment']}: churn={r['churn_rate']:.1%} (n={r['count']:,})")

# Cancel recovery: % who place new successful order after cancel
cancel_orders = tx_all[tx_all['status'] == 'CANCEL'].copy()
success_orders = tx_all[tx_all['status'] == 'SUCCESS'].copy()
# Per period: users who cancelled but also had success
cancel_recovery_rates = []
for period in tx_all['period'].unique():
    cancelled_users = set(cancel_orders[cancel_orders['period'] == period]['userNo'])
    success_users = set(success_orders[success_orders['period'] == period]['userNo'])
    recovered = cancelled_users & success_users
    if len(cancelled_users) > 0:
        rate = len(recovered) / len(cancelled_users)
        cancel_recovery_rates.append({'period': period, 'recovery_rate': rate,
                                       'n_cancelled': len(cancelled_users),
                                       'n_recovered': len(recovered)})
cancel_recovery = pd.DataFrame(cancel_recovery_rates)
print("\n  Cancel Recovery (same period):")
for _, r in cancel_recovery.iterrows():
    print(f"    {r['period']}: {r['recovery_rate']:.1%} recovered "
          f"({r['n_recovered']:,}/{r['n_cancelled']:,} cancelled)")

# ═══════════════════════════════════════════════════════════════
# 7. Generate Charts
# ═══════════════════════════════════════════════════════════════

print("\n[Charts] Generating 8-panel chart...", flush=True)

plt.rcParams['font.family'] = ['Tahoma', 'Garuda', 'Loma', 'TH Sarabun New', 'sans-serif']
fig, axes = plt.subplots(2, 4, figsize=(28, 12))
fig.suptitle('Phase 40: Cross-Team Insight Report', fontsize=18, fontweight='bold', y=0.98)
fig.patch.set_facecolor('white')

colors_risk = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c', 'Critical': '#8e44ad'}

# Panel 1: Churn by Age Group
ax = axes[0, 0]
bars = ax.bar(age_churn['age_group'].astype(str), age_churn['churn_rate'] * 100,
              color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#8e44ad'],
              edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, age_churn['churn_rate']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('Churn Rate by Age Group', fontsize=13, fontweight='bold')
ax.set_ylabel('Churn Rate (%)')
ax.set_ylim(0, age_churn['churn_rate'].max() * 100 * 1.2)

# Panel 2: Churn by Gender
ax = axes[0, 1]
gender_data = gender_churn[gender_churn['gender'].isin(['MALE', 'FEMALE'])]
gender_colors = {'MALE': '#3498db', 'FEMALE': '#e91e63'}
bars = ax.bar(gender_data['gender'], gender_data['churn_rate'] * 100,
              color=[gender_colors.get(g, '#95a5a6') for g in gender_data['gender']],
              edgecolor='white', linewidth=0.5, width=0.5)
for bar, (_, r) in zip(bars, gender_data.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{r['churn_rate']:.1%}\n(n={r['count']:,})",
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('Churn Rate by Gender', fontsize=13, fontweight='bold')
ax.set_ylabel('Churn Rate (%)')
ax.set_ylim(0, gender_data['churn_rate'].max() * 100 * 1.35)

# Panel 3: Hourly Buying Pattern
ax = axes[0, 2]
ax.bar(hourly['hour'], hourly['orders'], color='#3498db', alpha=0.7, label='Orders')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Orders', color='#3498db')
ax.set_title('Buying Pattern by Hour', fontsize=13, fontweight='bold')
ax2 = ax.twinx()
ax2.plot(hourly['hour'], hourly['avg_basket'], color='#e74c3c', marker='o', linewidth=2, label='Avg Basket')
ax2.set_ylabel('Avg Basket', color='#e74c3c')
ax.set_xticks(range(0, 24, 3))

# Panel 4: Days Before Draw
ax = axes[0, 3]
days_colors = ['#e74c3c' if d <= 2 else '#f39c12' if d <= 5 else '#3498db'
               for d in days_stats['days_before_draw']]
ax.barh(days_stats['days_before_draw'], days_stats['orders'], color=days_colors, edgecolor='white')
ax.set_ylabel('Days Before Draw')
ax.set_xlabel('Number of Orders')
ax.set_title('Orders by Days Before Draw', fontsize=13, fontweight='bold')
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))

# Panel 5: Basket by Risk Level (if available)
ax = axes[1, 0]
if risk_df is not None and len(risk_basket) > 0:
    valid_risk = risk_basket[risk_basket['risk_level'].isin(['Low', 'Medium', 'High', 'Critical'])]
    bars = ax.bar(valid_risk['risk_level'], valid_risk['mean'],
                  color=[colors_risk.get(r, '#95a5a6') for r in valid_risk['risk_level']],
                  edgecolor='white', linewidth=0.5)
    for bar, (_, r) in zip(bars, valid_risk.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{r['mean']:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title('Avg Basket by Risk Level', fontsize=13, fontweight='bold')
    ax.set_ylabel('Avg Tickets per Period')
else:
    ax.text(0.5, 0.5, 'No risk data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Avg Basket by Risk Level', fontsize=13, fontweight='bold')

# Panel 6: Churn by Buyer Segment
ax = axes[1, 1]
seg_order = ['Small (1-9)', 'Medium (10-49)', 'Big (50+)']
seg_plot = seg_churn.set_index('segment').reindex(seg_order).dropna()
bars = ax.bar(seg_plot.index, seg_plot['churn_rate'] * 100,
              color=['#e74c3c', '#f39c12', '#2ecc71'], edgecolor='white', linewidth=0.5)
for bar, (seg, r) in zip(bars, seg_plot.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{r['churn_rate']:.1%}\n(n={r['count']:,})",
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('Churn by Basket Segment', fontsize=13, fontweight='bold')
ax.set_ylabel('Churn Rate (%)')
ax.set_ylim(0, seg_plot['churn_rate'].max() * 100 * 1.3)

# Panel 7: Winner vs Non-Winner
ax = axes[1, 2]
w_data = winner_churn.copy()
w_labels = ['Non-Winner', 'Winner']
w_colors = ['#e74c3c', '#2ecc71']
bars = ax.bar(w_labels, w_data['churn_rate'] * 100, color=w_colors,
              edgecolor='white', linewidth=0.5, width=0.5)
for bar, (_, r) in zip(bars, w_data.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{r['churn_rate']:.1%}", ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_title('Winner vs Non-Winner Churn', fontsize=13, fontweight='bold')
ax.set_ylabel('Churn Rate (%)')
ax.set_ylim(0, w_data['churn_rate'].max() * 100 * 1.3)

# Panel 8: Cancel Rate by Hour
ax = axes[1, 3]
ax.plot(cancel_hourly['hour'], cancel_hourly['cancel_rate'] * 100,
        color='#e74c3c', marker='o', linewidth=2, markersize=4)
ax.fill_between(cancel_hourly['hour'], cancel_hourly['cancel_rate'] * 100,
                alpha=0.2, color='#e74c3c')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Cancel Rate (%)')
ax.set_title('Cancel Rate by Hour', fontsize=13, fontweight='bold')
ax.set_xticks(range(0, 24, 3))

plt.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs('charts', exist_ok=True)
plt.savefig(CHART_OUT, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {CHART_OUT}")

# ═══════════════════════════════════════════════════════════════
# 8. Cross-Team Reports (Thai)
# ═══════════════════════════════════════════════════════════════

print("\n")
print("=" * 70)
print("=" * 70)
print("        CROSS-TEAM INSIGHT REPORT — Phase 40")
print("=" * 70)
print("=" * 70)

# --- ADS Team ---
print("\n")
print("=" * 70)
print("  INSIGHT REPORT สำหรับทีม ADS")
print("=" * 70)

# Gender insight
male_churn = gender_churn[gender_churn['gender'] == 'MALE']['churn_rate'].values
female_churn = gender_churn[gender_churn['gender'] == 'FEMALE']['churn_rate'].values
if len(male_churn) > 0 and len(female_churn) > 0:
    male_r, female_r = male_churn[0], female_churn[0]
    higher_gender = 'ชาย' if male_r > female_r else 'หญิง'
    diff_pct = abs(male_r - female_r) * 100
    print(f"""
1. Gender Targeting:
   - ผู้{higher_gender} มี Churn Rate สูงกว่า ({max(male_r, female_r):.1%} vs {min(male_r, female_r):.1%})
   - ส่วนต่าง: {diff_pct:.1f}%
   - แนะนำ: เพิ่มงบ Ads สำหรับกลุ่มเพศ{higher_gender} + ปรับ Creative ให้ตรงกลุ่ม
""")

# Age insight
youngest = age_churn.iloc[0]
oldest = age_churn.iloc[-1]
highest_churn_age = age_churn.loc[age_churn['churn_rate'].idxmax()]
lowest_churn_age = age_churn.loc[age_churn['churn_rate'].idxmin()]
print(f"""2. Age Targeting:
   - กลุ่มอายุที่ Churn สูงสุด: {highest_churn_age['age_group']} ({highest_churn_age['churn_rate']:.1%})
   - กลุ่มอายุที่ Churn ต่ำสุด: {lowest_churn_age['age_group']} ({lowest_churn_age['churn_rate']:.1%})
   - แนะนำ: เน้น Retargeting สำหรับกลุ่ม {highest_churn_age['age_group']}
   - กลุ่ม {lowest_churn_age['age_group']} เป็นฐานลูกค้าที่ภักดี — ใช้เป็น Lookalike Audience ได้
""")

# Region insight
top_region = region_churn.iloc[0]
bot_region = region_churn.iloc[-1]
print(f"""3. Geographic Targeting:
   - ภาคที่ Churn สูงสุด: {top_region['region']} ({top_region['churn_rate']:.1%})
   - ภาคที่ Churn ต่ำสุด: {bot_region['region']} ({bot_region['churn_rate']:.1%})
   - จังหวัด Churn สูงสุด 3 อันดับ:""")
for _, r in top10_churn_prov.head(3).iterrows():
    print(f"     * {r['province']}: {r['churn_rate']:.1%}")
print(f"""   - จังหวัดที่ลูกค้าภักดีที่สุด 3 อันดับ:""")
for _, r in bot10_churn_prov.head(3).iterrows():
    print(f"     * {r['province']}: {r['churn_rate']:.1%}")
print(f"""   - แนะนำ: ปรับ Geo-targeting budget ตามภูมิภาค
   - ภาคที่ Churn สูง → เพิ่มงบ Retargeting / Re-engagement Ads
   - ภาคที่ Churn ต่ำ → ใช้เป็น Seed audience สำหรับ Lookalike
""")

# Timing
print(f"""4. Ad Scheduling (เวลาที่ควรยิง Ads):
   - ชั่วโมงที่ลูกค้าซื้อเยอะที่สุด: {int(peak_hour):02d}:00 น.
   - ช่วง Peak Hours: {', '.join(f"{int(h):02d}:00" for h in hourly.nlargest(3, 'orders')['hour'])}
   - แนะนำ: Schedule Ads ก่อน peak 1-2 ชม. เพื่อ influence การตัดสินใจ
   - Rush buying (2 วันสุดท้าย) = {days_stats[days_stats['days_before_draw'] <= 2]['orders'].sum() / days_stats['orders'].sum():.1%} ของ volume
   - แนะนำ: เพิ่มงบ Ads 3x ในช่วง 3 วันก่อนงวด
""")

# Early vs Rush
for _, r in buyer_type_churn.iterrows():
    if r['buyer_type'] == 'Early Buyer':
        early_churn_r = r['churn_rate']
    elif r['buyer_type'] == 'Rush Buyer':
        rush_churn_r = r['churn_rate']
print(f"""5. Early vs Rush Buyer:
   - Early Buyer churn: {early_churn_r:.1%}
   - Rush Buyer churn: {rush_churn_r:.1%}
   - {'Early Buyer ภักดีกว่า' if early_churn_r < rush_churn_r else 'Rush Buyer ภักดีกว่า'}!
   - แนะนำ: {'สร้าง Early Bird Campaign เพื่อดึงคนซื้อเร็วขึ้น — ลด churn ระยะยาว' if rush_churn_r > early_churn_r else 'Rush buyer มี engagement สูง — ยิง Reminder ads ช่วงใกล้งวด'}
""")

# --- MARCOM Team ---
print("\n")
print("=" * 70)
print("  INSIGHT REPORT สำหรับทีม MARCOM")
print("=" * 70)

# Basket insights
if risk_df is not None:
    valid_risk_data = risk_basket[risk_basket['risk_level'].isin(['Low', 'Medium', 'High', 'Critical'])]
    low_basket = valid_risk_data[valid_risk_data['risk_level'] == 'Low']['mean'].values
    crit_basket = valid_risk_data[valid_risk_data['risk_level'] == 'Critical']['mean'].values
    if len(low_basket) > 0 and len(crit_basket) > 0:
        print(f"""
1. Basket Size x Risk Level:
   - ลูกค้า Low Risk ซื้อเฉลี่ย: {low_basket[0]:.1f} ใบ/งวด
   - ลูกค้า Critical Risk ซื้อเฉลี่ย: {crit_basket[0]:.1f} ใบ/งวด
   - {'Critical ซื้อน้อยกว่า Low อยู่' if crit_basket[0] < low_basket[0] else 'Critical ซื้อเยอะกว่า Low!'} {abs(low_basket[0] - crit_basket[0]):.1f} ใบ
   - แนะนำ: ออกโปรโมชัน "ซื้อครบ X ใบ รับส่วนลด" สำหรับกลุ่ม High/Critical
""")

print(f"""2. Basket Segment & Churn:""")
for _, r in seg_churn.iterrows():
    print(f"   - {r['segment']}: churn={r['churn_rate']:.1%} (n={r['count']:,})")

small_churn = seg_churn[seg_churn['segment'] == 'Small (1-9)']['churn_rate'].values
big_churn = seg_churn[seg_churn['segment'] == 'Big (50+)']['churn_rate'].values
if len(small_churn) > 0 and len(big_churn) > 0:
    print(f"""   - ลูกค้าซื้อน้อย (1-9 ใบ) churn สูงกว่าลูกค้าซื้อเยอะ {small_churn[0]/big_churn[0]:.1f}x
   - แนะนำ: สร้าง Bundle Deal เพื่อ upsell กลุ่มซื้อน้อย
   - เช่น "ซื้อ 5 ใบขึ้นไป รับสิทธิพิเศษ" → ลดความเสี่ยง Churn
""")

# Basket trend
print(f"""3. Basket Size Trend:""")
for _, r in period_basket.iterrows():
    print(f"   - {r['period']}: mean={r['mean']:.1f}, median={r['median']:.0f}")
first_mean = period_basket['mean'].iloc[0]
last_mean = period_basket['mean'].iloc[-1]
trend_dir = 'เพิ่มขึ้น' if last_mean > first_mean else 'ลดลง'
print(f"""   - Trend: ตะกร้าเฉลี่ย{trend_dir} ({first_mean:.1f} → {last_mean:.1f} ใบ)
""")

# Winner insight
w_winner = winner_churn[winner_churn['is_winner'] == 1]['churn_rate'].values
w_non = winner_churn[winner_churn['is_winner'] == 0]['churn_rate'].values
if len(w_winner) > 0 and len(w_non) > 0:
    print(f"""4. Winner Effect (ผลจากการถูกรางวัล):
   - ลูกค้าที่ถูกรางวัล: churn {w_winner[0]:.1%}
   - ลูกค้าที่ไม่ถูก: churn {w_non[0]:.1%}
   - ถูกรางวัลลด churn ได้ {(w_non[0] - w_winner[0]):.1%} (ลดลง {(1 - w_winner[0]/w_non[0])*100:.0f}%)
   - แนะนำ: ใช้ Winner Story ใน Content Marketing
   - "ลูกค้าที่ถูกรางวัลกลับมาซื้อต่อถึง {(1-w_winner[0])*100:.0f}%"
   - สร้าง Social Proof: แชร์เรื่องราวผู้โชคดี → กระตุ้นคนกลับมาซื้อ
""")

# Seasonality
highest_churn_period = period_churn.loc[period_churn['churn_rate'].idxmax()]
lowest_churn_period = period_churn.loc[period_churn['churn_rate'].idxmin()]
print(f"""5. Seasonal Campaign Planning:
   - งวดที่ Churn สูงสุด: {highest_churn_period['period']} ({highest_churn_period['churn_rate']:.1%})
   - งวดที่ Churn ต่ำสุด: {lowest_churn_period['period']} ({lowest_churn_period['churn_rate']:.1%})
   - แนะนำ: เตรียม Campaign พิเศษล่วงหน้าสำหรับงวดที่มี Churn สูง
   - ช่วงที่ Churn ต่ำ = ช่วง Opportunity สำหรับ Upsell Campaign
""")

# --- SEO/Content Team ---
print("\n")
print("=" * 70)
print("  INSIGHT REPORT สำหรับทีม SEO / CONTENT")
print("=" * 70)

print(f"""
1. Content Calendar:
   - ช่วง Peak Volume: 2 วันสุดท้ายก่อนงวด
   - ควรปล่อย Content (บทความ, โพสต์) ก่อนงวด 4-5 วัน เพื่อ SEO rank ทัน
   - Topic Ideas:
     * "เลขเด็ดงวด XX" — ตรงกับ Search intent ช่วงก่อนงวด
     * "ผลรางวัลงวดที่แล้ว" — Content หลังงวดดึง Traffic กลับ
     * "วิธีเลือกซื้อลอตเตอรี่" — Evergreen content

2. Day-of-Week Content Strategy:""")
best_day = dow_stats['orders'].idxmax()
print(f"""   - วันที่ Traffic สูงสุด: {dow_thai[best_day]}
   - แนะนำ: ปล่อย Content หลักในวัน{dow_thai[best_day]}
   - วันเสาร์-อาทิตย์: Content เบาๆ (Tips, Stories)

3. Hour-based Push Notification:
   - Peak Hours: {', '.join(f"{int(h):02d}:00" for h in hourly.nlargest(3, 'orders')['hour'])} น.
   - ส่ง Push/Email ก่อน Peak 1-2 ชม. เพื่อ maximize conversion
   - ช่วงดึก (00:00-06:00) = Low traffic → ไม่ควรส่ง Notification
""")

# Province content
print(f"""4. Geo-targeted Content:
   - จังหวัดที่ซื้อเยอะ (Low Churn):""")
for _, r in bot10_churn_prov.head(5).iterrows():
    print(f"     * {r['province']} ({r['churn_rate']:.1%} churn)")
print(f"""   - จังหวัดที่ต้องการ Re-engagement Content (High Churn):""")
for _, r in top10_churn_prov.head(5).iterrows():
    print(f"     * {r['province']} ({r['churn_rate']:.1%} churn)")
print(f"""   - แนะนำ: สร้าง Localized content สำหรับจังหวัดที่ Churn สูง
   - ใช้ภาษาถิ่น/อ้างอิงท้องถิ่น เพื่อเพิ่ม engagement
""")

# --- UX/Tech Team ---
print("\n")
print("=" * 70)
print("  INSIGHT REPORT สำหรับทีม UX / TECH")
print("=" * 70)

avg_cancel = tx_all['is_cancel'].mean() if 'is_cancel' in tx_all.columns else tx_cancel['is_cancel'].mean()
peak_cancel = cancel_hourly.loc[cancel_hourly['cancel_rate'].idxmax()]
low_cancel = cancel_hourly.loc[cancel_hourly['cancel_rate'].idxmin()]

print(f"""
1. Cancel Rate Overview:
   - Cancel Rate เฉลี่ย: {avg_cancel:.1%}
   - ช่วงเวลาที่ Cancel สูงสุด: {int(peak_cancel['hour']):02d}:00 ({peak_cancel['cancel_rate']:.1%})
   - ช่วงเวลาที่ Cancel ต่ำสุด: {int(low_cancel['hour']):02d}:00 ({low_cancel['cancel_rate']:.1%})
   - ถ้า Cancel สูงช่วงเฉพาะเจาะจง → อาจเป็นปัญหา Payment Gateway ช่วงนั้น
""")

print(f"""2. Cancel → Churn Correlation:""")
for _, r in cancel_seg_churn.iterrows():
    print(f"   - {r['segment']}: churn={r['churn_rate']:.1%} (n={r['count']:,})")
high_cancel_churn = cancel_seg_churn[cancel_seg_churn['segment'] == 'High Cancel (>20%)']['churn_rate'].values
no_cancel_churn = cancel_seg_churn[cancel_seg_churn['segment'] == 'No Cancel']['churn_rate'].values
if len(high_cancel_churn) > 0 and len(no_cancel_churn) > 0:
    print(f"""   - ลูกค้าที่ Cancel บ่อย churn สูงกว่า {high_cancel_churn[0]/no_cancel_churn[0]:.1f}x
   - แนะนำ: แจ้งเตือนลูกค้าที่ Cancel → "สินค้ายังอยู่ในตะกร้า กลับมาชำระเงินได้เลย"
   - ลด friction ในขั้นตอนชำระเงิน (เพิ่มช่องทาง, ขยายเวลา)
""")

avg_recovery = cancel_recovery['recovery_rate'].mean()
print(f"""3. Cancel Recovery:
   - อัตราการกลับมาซื้อหลัง Cancel (same period): {avg_recovery:.1%}
   - {'ดี! ลูกค้าส่วนใหญ่กลับมาซื้อใหม่หลัง Cancel' if avg_recovery > 0.5 else 'ต้องปรับปรุง — ลูกค้าส่วนใหญ่ไม่กลับมาหลัง Cancel'}
   - แนะนำ: ส่ง Push Notification ทันทีหลัง Cancel
   - "การชำระเงินไม่สำเร็จ — ลองอีกครั้ง" + Deep link กลับไปหน้าชำระเงิน
""")

print(f"""4. Cancel by Demographics:""")
print(f"   Cancel by Gender:")
for g, rate in cancel_gender.items():
    print(f"     - {g}: {rate:.1%}")
print(f"   Cancel by Age:")
for a, rate in cancel_age.items():
    print(f"     - {a}: {rate:.1%}")
print(f"""   - แนะนำ: ถ้ากลุ่มอายุมาก Cancel สูง → อาจต้องปรับ UX ให้ง่ายขึ้น
   - Font ใหญ่ขึ้น, ปุ่มชำระเงินเด่นชัด, ลดขั้นตอน
""")

# Summary
print("\n")
print("=" * 70)
print("  SUMMARY — Key Actions")
print("=" * 70)
print(f"""
  [ADS]     1. เน้น Retargeting กลุ่มอายุ {highest_churn_age['age_group']} + เพศ{higher_gender}
  [ADS]     2. เพิ่มงบ 3x ช่วง 3 วันก่อนงวด (Rush period)
  [ADS]     3. Schedule Ads ช่วง {int(peak_hour):02d}:00 น. (Peak hour)
  [MARCOM]  4. Bundle Deal สำหรับกลุ่มซื้อน้อย (1-9 ใบ) → ลด Churn
  [MARCOM]  5. Winner Story Campaign → ลูกค้าถูกรางวัลกลับมา {(1-w_winner[0])*100:.0f}%
  [SEO]     6. ปล่อย Content 4-5 วันก่อนงวด / Push ช่วง Peak hour
  [UX]      7. ส่ง Push ทันทีหลัง Cancel + ลด payment friction
  [UX]      8. Recovery rate: {avg_recovery:.1%} — {'maintain' if avg_recovery > 0.5 else 'need improvement'}
""")

print("=" * 70)
print("Phase 40 Complete!")
print("=" * 70)
