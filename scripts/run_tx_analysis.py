"""
TX Business Analysis — Thai Online Lottery Platform
=====================================================
Analyzes 30 Transaction Order files to generate business insights.

Output:
  - charts/tx_business_analysis.png (6-panel dashboard)
  - Console summary of key findings
"""

import os
import time
import warnings
from glob import glob
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
TX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TransactionOrder')
CHART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'charts')
OUTPUT_PATH = os.path.join(CHART_DIR, 'tx_business_analysis.png')
os.makedirs(CHART_DIR, exist_ok=True)

# Discover and sort TX files chronologically
tx_files = sorted(glob(os.path.join(TX_DIR, 'Transaction_Order_*.csv')))
assert len(tx_files) == 30, f"Expected 30 TX files, found {len(tx_files)}"

# Extract period labels (MMDD) from filenames
def period_label(filepath):
    # Transaction_Order_YYYY_MMDD.csv -> MMDD
    base = os.path.basename(filepath)
    parts = base.replace('.csv', '').split('_')
    return parts[-1]  # MMDD

def period_date(filepath):
    """Extract full date from filename for sorting."""
    base = os.path.basename(filepath)
    parts = base.replace('.csv', '').split('_')
    return f"{parts[-2]}-{parts[-1][:2]}-{parts[-1][2:]}"

period_labels = [period_label(f) for f in tx_files]
period_dates = [period_date(f) for f in tx_files]

print(f"Found {len(tx_files)} TX files")
print(f"Period range: {period_dates[0]} to {period_dates[-1]}")
print()

# ── Storage for per-period metrics ─────────────────────────────────────────
metrics = {
    'total_customers': [],
    'total_orders': [],
    'total_tickets': [],
    'avg_tickets_per_order': [],
    'avg_tickets_per_customer': [],
    'cancel_rate': [],
    'rush_ratio': [],
    'avg_tickets_early': [],
    'avg_tickets_late': [],
}

# For Panel 4 (loyalty) — collect userNo sets per period
user_period_sets = []

# ── Process each file ──────────────────────────────────────────────────────
total_start = time.time()

for i, filepath in enumerate(tx_files):
    t0 = time.time()
    label = period_labels[i]
    print(f"Processing period {label} ({i+1}/30)...", end=' ')

    # Read full file for panels 1-3, 5-6
    df = pd.read_csv(filepath, usecols=['userNo', 'createdAt_bkk', 'status', 'totalItem'])

    total_orders_all = len(df)
    cancel_count = (df['status'] == 'CANCEL').sum()
    cancel_rate = cancel_count / total_orders_all if total_orders_all > 0 else 0
    metrics['cancel_rate'].append(cancel_rate)

    # Panel 4: collect unique users (SUCCESS only for loyalty — they actually bought)
    success_users = set(df.loc[df['status'] == 'SUCCESS', 'userNo'].unique())
    user_period_sets.append(success_users)

    # Filter to SUCCESS for business metrics
    df = df[df['status'] == 'SUCCESS'].copy()

    total_orders = len(df)
    total_customers = df['userNo'].nunique()
    total_tickets = df['totalItem'].sum()

    metrics['total_customers'].append(total_customers)
    metrics['total_orders'].append(total_orders)
    metrics['total_tickets'].append(total_tickets)
    metrics['avg_tickets_per_order'].append(total_tickets / total_orders if total_orders > 0 else 0)
    metrics['avg_tickets_per_customer'].append(total_tickets / total_customers if total_customers > 0 else 0)

    # Parse timestamps for rush buying and early/late analysis
    df['created_dt'] = pd.to_datetime(df['createdAt_bkk'], format='mixed', utc=True)
    df['created_date'] = df['created_dt'].dt.date

    # Determine sales window
    min_date = df['created_date'].min()
    max_date = df['created_date'].max()
    total_days = (max_date - min_date).days + 1

    # Panel 3: Rush buying — orders in last 2 days
    from datetime import timedelta
    cutoff_rush = max_date - timedelta(days=1)  # last 2 days = max_date and max_date-1
    rush_orders = (df['created_date'] >= cutoff_rush).sum()
    rush_ratio = rush_orders / total_orders if total_orders > 0 else 0
    metrics['rush_ratio'].append(rush_ratio)

    # Panel 6: Early vs Late buyers
    midpoint = min_date + timedelta(days=total_days // 2)
    early_mask = df['created_date'] < midpoint
    late_mask = df['created_date'] >= midpoint

    avg_early = df.loc[early_mask, 'totalItem'].mean() if early_mask.any() else 0
    avg_late = df.loc[late_mask, 'totalItem'].mean() if late_mask.any() else 0
    metrics['avg_tickets_early'].append(avg_early)
    metrics['avg_tickets_late'].append(avg_late)

    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s) — {total_customers:,} customers, {total_orders:,} orders")

print(f"\nAll files processed in {time.time() - total_start:.1f}s")

# ── Panel 4: Customer Loyalty Distribution ─────────────────────────────────
print("\nComputing customer loyalty distribution...")
t0 = time.time()

# Count how many periods each customer appeared in
from collections import Counter
customer_period_count = Counter()
for user_set in user_period_sets:
    for user in user_set:
        customer_period_count[user] += 1

total_unique_customers = len(customer_period_count)
loyalty_counts = list(customer_period_count.values())
loyalty_hist = Counter(loyalty_counts)  # {num_periods: count_of_customers}

pct_one = loyalty_hist.get(1, 0) / total_unique_customers * 100
pct_all = loyalty_hist.get(30, 0) / total_unique_customers * 100
median_periods = np.median(loyalty_counts)
mean_periods = np.mean(loyalty_counts)

print(f"  Total unique customers: {total_unique_customers:,}")
print(f"  Bought only 1 period: {loyalty_hist.get(1, 0):,} ({pct_one:.1f}%)")
print(f"  Bought all 30 periods: {loyalty_hist.get(30, 0):,} ({pct_all:.1f}%)")
print(f"  Median periods: {median_periods:.0f}, Mean: {mean_periods:.1f}")
print(f"  Loyalty computed in {time.time() - t0:.1f}s")

# ── Plotting ───────────────────────────────────────────────────────────────
print("\nGenerating charts...")

fig, axes = plt.subplots(3, 2, figsize=(20, 16))
fig.suptitle('TX Business Analysis — Thai Lottery Platform (30 Periods)',
             fontsize=16, fontweight='bold', y=0.98)

x = np.arange(len(period_labels))
# Show every 3rd label to avoid crowding
tick_positions = x[::3]
tick_labels = [period_labels[i] for i in range(0, len(period_labels), 3)]

# ── Panel 1: Platform Overview Trends (top-left) ──────────────────────────
ax1 = axes[0, 0]
ax1_twin = ax1.twinx()

customers_k = [c / 1000 for c in metrics['total_customers']]
orders_k = [o / 1000 for o in metrics['total_orders']]
tickets_m = [t / 1_000_000 for t in metrics['total_tickets']]

ln1 = ax1.plot(x, customers_k, 'o-', color='#2196F3', markersize=3, linewidth=1.5,
               label='Customers (K)')
ln2 = ax1.plot(x, orders_k, 's-', color='#4CAF50', markersize=3, linewidth=1.5,
               label='Orders (K)')
ln3 = ax1_twin.plot(x, tickets_m, '^-', color='#FF9800', markersize=3, linewidth=1.5,
                    label='Tickets (M)')

ax1.set_xlabel('Period')
ax1.set_ylabel('Customers / Orders (thousands)')
ax1_twin.set_ylabel('Tickets (millions)')
ax1.set_title('Panel 1: Platform Overview Trends', fontweight='bold')
ax1.set_xticks(tick_positions)
ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

# Combined legend
lns = ln1 + ln2 + ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# ── Panel 2: Average Basket & Tickets per Customer (top-right) ────────────
ax2 = axes[0, 1]
ax2.plot(x, metrics['avg_tickets_per_order'], 'o-', color='#E91E63', markersize=3,
         linewidth=1.5, label='Avg tickets/order')
ax2.plot(x, metrics['avg_tickets_per_customer'], 's-', color='#9C27B0', markersize=3,
         linewidth=1.5, label='Avg tickets/customer')
ax2.set_xlabel('Period')
ax2.set_ylabel('Tickets')
ax2.set_title('Panel 2: Average Basket & Tickets per Customer', fontweight='bold')
ax2.set_xticks(tick_positions)
ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# ── Panel 3: Rush Buying Pattern (middle-left) ────────────────────────────
ax3 = axes[1, 0]
colors_rush = ['#FF5722' if r > np.mean(metrics['rush_ratio']) else '#607D8B'
               for r in metrics['rush_ratio']]
ax3.bar(x, [r * 100 for r in metrics['rush_ratio']], color=colors_rush, alpha=0.8)
ax3.axhline(y=np.mean(metrics['rush_ratio']) * 100, color='red', linestyle='--',
            linewidth=1, alpha=0.7, label=f"Avg: {np.mean(metrics['rush_ratio'])*100:.1f}%")
ax3.set_xlabel('Period')
ax3.set_ylabel('% of orders in last 2 days')
ax3.set_title('Panel 3: Rush Buying Pattern (Last 2 Days)', fontweight='bold')
ax3.set_xticks(tick_positions)
ax3.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis='y')

# ── Panel 4: Customer Loyalty Distribution (middle-right) ─────────────────
ax4 = axes[1, 1]
bins = np.arange(0.5, 31.5, 1)
ax4.hist(loyalty_counts, bins=bins, color='#3F51B5', alpha=0.8, edgecolor='white',
         linewidth=0.5)
ax4.set_xlabel('Number of periods purchased (out of 30)')
ax4.set_ylabel('Number of customers')
ax4.set_title('Panel 4: Customer Loyalty Distribution', fontweight='bold')
ax4.axvline(x=median_periods, color='red', linestyle='--', linewidth=1.5,
            label=f'Median: {median_periods:.0f} periods')
ax4.axvline(x=mean_periods, color='orange', linestyle='--', linewidth=1.5,
            label=f'Mean: {mean_periods:.1f} periods')

# Annotate key stats
textstr = (f"Total: {total_unique_customers:,} customers\n"
           f"1 period only: {pct_one:.1f}%\n"
           f"All 30 periods: {pct_all:.1f}%")
ax4.text(0.97, 0.97, textstr, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis='y')
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: f'{val/1000:.0f}K'))

# ── Panel 5: Cancel Rate Trend (bottom-left) ──────────────────────────────
ax5 = axes[2, 0]
cancel_pcts = [r * 100 for r in metrics['cancel_rate']]
ax5.plot(x, cancel_pcts, 'o-', color='#F44336', markersize=4, linewidth=1.5)
ax5.fill_between(x, cancel_pcts, alpha=0.15, color='#F44336')
ax5.axhline(y=np.mean(cancel_pcts), color='gray', linestyle='--', linewidth=1,
            alpha=0.7, label=f"Avg: {np.mean(cancel_pcts):.2f}%")
ax5.set_xlabel('Period')
ax5.set_ylabel('Cancel Rate (%)')
ax5.set_title('Panel 5: Cancel Rate Trend', fontweight='bold')
ax5.set_xticks(tick_positions)
ax5.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# ── Panel 6: Early vs Late Buyer Profile (bottom-right) ───────────────────
ax6 = axes[2, 1]
bar_width = 0.35
ax6.bar(x - bar_width/2, metrics['avg_tickets_early'], bar_width, color='#00BCD4',
        alpha=0.8, label='Early (1st half)')
ax6.bar(x + bar_width/2, metrics['avg_tickets_late'], bar_width, color='#FF5722',
        alpha=0.8, label='Late (2nd half)')
ax6.set_xlabel('Period')
ax6.set_ylabel('Avg tickets per order')
ax6.set_title('Panel 6: Early vs Late Buyer — Avg Tickets per Order', fontweight='bold')
ax6.set_xticks(tick_positions)
ax6.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3, axis='y')

# ── Save ──────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nChart saved to: {OUTPUT_PATH}")

# ── Console Summary ───────────────────────────────────────────────────────
first_customers = metrics['total_customers'][0]
last_customers = metrics['total_customers'][-1]
customer_growth = (last_customers - first_customers) / first_customers * 100

first_orders = metrics['total_orders'][0]
last_orders = metrics['total_orders'][-1]
order_growth = (last_orders - first_orders) / first_orders * 100

first_tickets = metrics['total_tickets'][0]
last_tickets = metrics['total_tickets'][-1]
ticket_growth = (last_tickets - first_tickets) / first_tickets * 100

avg_tpo_first5 = np.mean(metrics['avg_tickets_per_order'][:5])
avg_tpo_last5 = np.mean(metrics['avg_tickets_per_order'][-5:])
avg_tpc_first5 = np.mean(metrics['avg_tickets_per_customer'][:5])
avg_tpc_last5 = np.mean(metrics['avg_tickets_per_customer'][-5:])

avg_rush = np.mean(metrics['rush_ratio']) * 100
rush_trend = "increasing" if metrics['rush_ratio'][-1] > metrics['rush_ratio'][0] else "decreasing"

avg_cancel = np.mean(cancel_pcts)
cancel_first5 = np.mean(cancel_pcts[:5])
cancel_last5 = np.mean(cancel_pcts[-5:])

avg_early_all = np.mean(metrics['avg_tickets_early'])
avg_late_all = np.mean(metrics['avg_tickets_late'])
late_premium = (avg_late_all - avg_early_all) / avg_early_all * 100

# Loyalty segments
pct_loyal = sum(1 for c in loyalty_counts if c >= 20) / total_unique_customers * 100
pct_occasional = sum(1 for c in loyalty_counts if 2 <= c <= 10) / total_unique_customers * 100

print()
print("=" * 60)
print("TX Business Analysis — Key Insights")
print("=" * 60)

print(f"\nPlatform Growth (Period {period_labels[0]} → {period_labels[-1]}):")
print(f"  Customers: {first_customers:,} → {last_customers:,} ({customer_growth:+.1f}%)")
print(f"  Orders:    {first_orders:,} → {last_orders:,} ({order_growth:+.1f}%)")
print(f"  Tickets:   {first_tickets:,.0f} → {last_tickets:,.0f} ({ticket_growth:+.1f}%)")
print(f"  Peak customers: {max(metrics['total_customers']):,} (period {period_labels[np.argmax(metrics['total_customers'])]})")
print(f"  Peak orders:    {max(metrics['total_orders']):,} (period {period_labels[np.argmax(metrics['total_orders'])]})")

print(f"\nBuying Behavior:")
print(f"  Avg tickets/order:    {avg_tpo_first5:.2f} (first 5) → {avg_tpo_last5:.2f} (last 5)")
print(f"  Avg tickets/customer: {avg_tpc_first5:.2f} (first 5) → {avg_tpc_last5:.2f} (last 5)")
print(f"  Overall avg tickets/order:    {np.mean(metrics['avg_tickets_per_order']):.2f}")
print(f"  Overall avg tickets/customer: {np.mean(metrics['avg_tickets_per_customer']):.2f}")

print(f"\nRush Buying (last 2 days of sales window):")
print(f"  Average rush ratio: {avg_rush:.1f}% of orders")
print(f"  Trend: {rush_trend} ({metrics['rush_ratio'][0]*100:.1f}% → {metrics['rush_ratio'][-1]*100:.1f}%)")
print(f"  Max rush: {max(metrics['rush_ratio'])*100:.1f}% (period {period_labels[np.argmax(metrics['rush_ratio'])]})")

print(f"\nCustomer Loyalty ({total_unique_customers:,} unique customers across 30 periods):")
print(f"  ซื้อแค่ 1 งวด:  {loyalty_hist.get(1, 0):,} ({pct_one:.1f}%)")
print(f"  ซื้อทุกงวด (30): {loyalty_hist.get(30, 0):,} ({pct_all:.1f}%)")
print(f"  ซื้อ ≥20 งวด:   {sum(1 for c in loyalty_counts if c >= 20):,} ({pct_loyal:.1f}%)")
print(f"  ซื้อ 2-10 งวด:  {sum(1 for c in loyalty_counts if 2 <= c <= 10):,} ({pct_occasional:.1f}%)")
print(f"  Median: {median_periods:.0f} periods, Mean: {mean_periods:.1f} periods")

print(f"\nCancel Rate:")
print(f"  Average: {avg_cancel:.2f}%")
print(f"  First 5 periods avg: {cancel_first5:.2f}%")
print(f"  Last 5 periods avg:  {cancel_last5:.2f}%")
print(f"  Trend: {'increasing' if cancel_last5 > cancel_first5 else 'decreasing'}")
print(f"  (Note: cancels = cart timeout/payment failure, NOT dissatisfaction)")

print(f"\nEarly vs Late Buyers:")
print(f"  Avg tickets/order (early, 1st half): {avg_early_all:.2f}")
print(f"  Avg tickets/order (late, 2nd half):  {avg_late_all:.2f}")
print(f"  Late buyer premium: {late_premium:+.1f}%")
if late_premium > 0:
    print(f"  → Late buyers purchase more tickets per order")
else:
    print(f"  → Early buyers purchase more tickets per order")

print(f"\n{'=' * 60}")
print(f"Total runtime: {time.time() - total_start:.1f}s")
print(f"Chart saved: {OUTPUT_PATH}")
print(f"{'=' * 60}")
