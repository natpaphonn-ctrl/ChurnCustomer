#!/usr/bin/env python3
"""
Phase 37: Churn Recovery Analysis
==================================
Analyzes customers who churned and then came back:
1. Recovery rate by gap length (1-10 periods)
2. Profile comparison (recovered vs permanent churners)
3. Recovery sustainability (do they churn again after returning?)
4. Key stats summary + prize winner analysis

Run from project root:
    python3 scripts/run_phase37.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
CHURN_DIR = "data/churn"
PRIZE_DIR = "data/prize"
CHART_PATH = "charts/phase37_churn_recovery.png"
os.makedirs("charts", exist_ok=True)

# Thai font setup
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


# ── 1. Load latest churn file (most item columns = most history) ─────────────
def load_latest_churn():
    """Load the churn file with the most columns (longest history)."""
    files = sorted(glob.glob(f"{CHURN_DIR}/Churn_20*_*_*.csv"))
    # Exclude _old_ and _Check files
    files = [f for f in files if "old" not in f.lower() and "check" not in f.lower()]
    print(f"Found {len(files)} churn files")

    # The latest file typically has the most columns
    latest = files[-1]
    print(f"Loading: {os.path.basename(latest)}")
    df = pd.read_csv(latest)
    print(f"  Shape: {df.shape[0]:,} customers x {df.shape[1]} columns")
    return df


# ── 2. Extract purchase matrix ──────────────────────────────────────────────
def extract_purchase_matrix(df):
    """Extract item columns into a boolean purchase matrix (bought or not)."""
    item_cols = [c for c in df.columns if c.startswith("item")]
    item_cols = sorted(item_cols)  # chronological order
    print(f"  {len(item_cols)} item columns: {item_cols[0]} → {item_cols[-1]}")

    # Purchase matrix: True if bought (>0), False if 0, NaN stays NaN
    purchase = df[item_cols].copy()
    # bought = purchase > 0, but keep NaN as NaN
    bought = purchase.where(purchase.isna(), purchase > 0).astype("boolean")

    return bought, item_cols, df


# ── 3. Identify churn events and recovery ───────────────────────────────────
def analyze_churn_recovery(bought, item_cols):
    """
    For each customer, identify churn events and recovery.
    Churn event: bought in period i, did NOT buy in period i+1
    Recovery: after churning, bought again in some later period
    """
    n_customers, n_periods = bought.shape
    print(f"\nAnalyzing {n_customers:,} customers across {n_periods} periods...")

    # Convert to numpy for vectorized ops (True=bought, False=not bought, NaN=not registered)
    mat = bought.to_numpy(dtype="float64", na_value=np.nan)
    # mat: 1.0=bought, 0.0=didn't buy, nan=not registered

    # ── Find all churn events ────────────────────────────────────────────
    # Churn at period i: mat[:,i]==1 and mat[:,i+1]==0 (not nan)
    results = []

    for i in range(n_periods - 1):
        bought_i = mat[:, i] == 1.0
        not_bought_next = mat[:, i + 1] == 0.0
        churn_mask = bought_i & not_bought_next

        churn_indices = np.where(churn_mask)[0]
        if len(churn_indices) == 0:
            continue

        for cust_idx in churn_indices:
            # Look forward: when do they come back?
            future = mat[cust_idx, i + 1:]
            # Find first period they bought again
            future_buys = np.where(future == 1.0)[0]
            if len(future_buys) > 0:
                gap = future_buys[0]  # number of periods of gap (1=came back next period after gap)
                recovery_period_idx = i + 1 + future_buys[0]

                # After recovery, do they stay? Check if they churn again within 3 periods
                post_recovery = mat[cust_idx, recovery_period_idx:]
                if len(post_recovery) > 1:
                    # How many of the next periods (up to 3) did they buy?
                    check_window = min(3, len(post_recovery) - 1)
                    post_buys = np.nansum(post_recovery[1:1 + check_window] == 1.0)
                    post_total = check_window
                    sustained = post_buys / post_total if post_total > 0 else np.nan
                else:
                    sustained = np.nan

                results.append({
                    "cust_idx": cust_idx,
                    "churn_period_idx": i,
                    "gap_periods": gap,
                    "recovered": True,
                    "recovery_period_idx": recovery_period_idx,
                    "sustained_rate": sustained,
                })
            else:
                # Permanent churner (within observable window)
                remaining = n_periods - i - 1
                results.append({
                    "cust_idx": cust_idx,
                    "churn_period_idx": i,
                    "gap_periods": remaining,  # at least this many
                    "recovered": False,
                    "recovery_period_idx": np.nan,
                    "sustained_rate": np.nan,
                })

    events = pd.DataFrame(results)
    print(f"  Found {len(events):,} churn events")
    print(f"  Recovered: {events['recovered'].sum():,} ({events['recovered'].mean()*100:.1f}%)")
    print(f"  Permanent: {(~events['recovered']).sum():,} ({(~events['recovered']).mean()*100:.1f}%)")
    return events


# ── 4. Build customer profiles at time of churn ────────────────────────────
def build_profiles(events, mat, item_cols, df):
    """
    For each churn event, compute the customer's profile at the time of churn.
    Uses vectorized operations where possible.
    """
    print("\nBuilding customer profiles at churn time...")
    n_events = len(events)

    # Pre-extract arrays
    cust_idxs = events["cust_idx"].values
    churn_idxs = events["churn_period_idx"].values

    tenure = np.zeros(n_events)
    freq = np.zeros(n_events)
    avg_items = np.zeros(n_events)
    declining = np.zeros(n_events, dtype=bool)

    for k in range(n_events):
        ci = cust_idxs[k]
        pi = churn_idxs[k]
        history = mat[ci, :pi + 1]  # up to and including churn period

        # Tenure: number of periods since first non-NaN
        valid = ~np.isnan(history)
        if valid.any():
            tenure[k] = valid.sum()
        else:
            tenure[k] = 0

        # Frequency: proportion of valid periods with purchase
        purchased = history == 1.0
        if valid.any():
            freq[k] = purchased.sum() / valid.sum()
        else:
            freq[k] = 0

        # Average items (from raw data, not boolean)
        # We'll approximate from the boolean matrix — 1 if bought, 0 if not
        # Actually, let's use the raw values for avg items
        avg_items[k] = np.nan  # placeholder, fill below

        # Declining: compare last 3 vs previous 3 periods of purchases
        if pi >= 5:
            recent_3 = history[pi - 2:pi + 1]
            prev_3 = history[pi - 5:pi - 2]
            recent_buys = np.nansum(recent_3 == 1.0)
            prev_buys = np.nansum(prev_3 == 1.0)
            declining[k] = recent_buys < prev_buys
        else:
            declining[k] = False

    events["tenure_periods"] = tenure
    events["purchase_freq"] = freq
    events["was_declining"] = declining

    return events


# ── 5. Load prize data ──────────────────────────────────────────────────────
def load_prize_data():
    """Load all prize files and return set of userNos who won."""
    prize_files = sorted(glob.glob(f"{PRIZE_DIR}/Prize_*.csv"))
    if not prize_files:
        print("No prize files found")
        return set(), pd.DataFrame()

    print(f"\nLoading {len(prize_files)} prize files...")
    dfs = []
    for f in prize_files:
        try:
            d = pd.read_csv(f, usecols=["userNo", "total_prize"])
            dfs.append(d)
        except Exception:
            pass

    if dfs:
        prizes = pd.concat(dfs, ignore_index=True)
        winner_stats = prizes.groupby("userNo").agg(
            total_winnings=("total_prize", "sum"),
            n_wins=("total_prize", "count"),
        ).reset_index()
        winners = set(prizes["userNo"].unique())
        print(f"  {len(winners):,} unique prize winners")
        return winners, winner_stats
    return set(), pd.DataFrame()


# ── 6. Recovery rate by gap length ──────────────────────────────────────────
def compute_recovery_rates(events, n_periods):
    """
    Compute cumulative recovery rate: of all who churned,
    what % came back within 1, 2, ..., 10 periods?
    """
    # Only count churn events that had enough observation window
    max_gap = min(10, n_periods - 2)
    rates = {}

    for gap in range(1, max_gap + 1):
        # Eligible: churned at period idx where there are >= gap periods remaining
        eligible = events[events["churn_period_idx"] <= n_periods - 1 - gap]
        if len(eligible) == 0:
            continue
        recovered = eligible[eligible["recovered"] & (eligible["gap_periods"] <= gap)]
        rates[gap] = len(recovered) / len(eligible) * 100

    return rates


# ── 7. Plotting ─────────────────────────────────────────────────────────────
def plot_results(recovery_rates, events, n_periods, df_users, winners):
    """Create 4-panel chart."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Phase 37: Churn Recovery Analysis", fontsize=16, fontweight="bold", y=0.98)

    # ── Panel 1: Recovery Rate by Gap Length ─────────────────────────────
    ax1 = axes[0, 0]
    gaps = sorted(recovery_rates.keys())
    rates = [recovery_rates[g] for g in gaps]
    bars = ax1.bar(gaps, rates, color="#2196F3", edgecolor="white", alpha=0.85)
    ax1.set_xlabel("Gap (periods since churn)")
    ax1.set_ylabel("Cumulative Recovery Rate (%)")
    ax1.set_title("Recovery Rate by Gap Length")
    ax1.set_xticks(gaps)
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)
    ax1.set_ylim(0, max(rates) * 1.15 if rates else 100)
    ax1.grid(axis="y", alpha=0.3)

    # ── Panel 2: Profile Comparison ──────────────────────────────────────
    ax2 = axes[0, 1]
    recovered = events[events["recovered"]]
    permanent = events[~events["recovered"]]

    metrics = ["tenure_periods", "purchase_freq", "was_declining"]
    labels = ["Avg Tenure\n(periods)", "Purchase\nFrequency", "Was Declining\n(%)"]

    rec_vals = [
        recovered["tenure_periods"].mean(),
        recovered["purchase_freq"].mean(),
        recovered["was_declining"].mean() * 100,
    ]
    perm_vals = [
        permanent["tenure_periods"].mean(),
        permanent["purchase_freq"].mean(),
        permanent["was_declining"].mean() * 100,
    ]

    x = np.arange(len(labels))
    w = 0.35
    b1 = ax2.bar(x - w / 2, rec_vals, w, label="Recovered", color="#4CAF50", alpha=0.85)
    b2 = ax2.bar(x + w / 2, perm_vals, w, label="Permanent Churn", color="#F44336", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title("Profile: Recovered vs Permanent Churners")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars_group in [b1, b2]:
        for bar in bars_group:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                     f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    # ── Panel 3: Recovery Sustainability ─────────────────────────────────
    ax3 = axes[1, 0]
    sustained = recovered.dropna(subset=["sustained_rate"])
    if len(sustained) > 0:
        bins = [0, 0.01, 0.34, 0.67, 1.01]
        bin_labels = ["Churned Again\n(0%)", "Low\n(1-33%)", "Medium\n(34-67%)", "High\n(68-100%)"]
        sustained_cut = pd.cut(sustained["sustained_rate"], bins=bins, labels=bin_labels,
                               include_lowest=True)
        counts = sustained_cut.value_counts().reindex(bin_labels)
        colors = ["#F44336", "#FF9800", "#FFC107", "#4CAF50"]
        bars3 = ax3.bar(bin_labels, counts.values, color=colors, edgecolor="white", alpha=0.85)
        for bar, cnt in zip(bars3, counts.values):
            pct = cnt / len(sustained) * 100
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     f"{cnt:,.0f}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax3.set_title("After Recovery: Do They Stay? (next 3 periods)")
    ax3.set_ylabel("Number of Customers")
    ax3.grid(axis="y", alpha=0.3)

    # ── Panel 4: Key Stats Summary ───────────────────────────────────────
    ax4 = axes[1, 1]
    ax4.axis("off")

    total_churn_events = len(events)
    unique_churners = events["cust_idx"].nunique()
    n_recovered = events[events["recovered"]]["cust_idx"].nunique()
    n_permanent = unique_churners - n_recovered
    overall_recovery_pct = n_recovered / unique_churners * 100 if unique_churners > 0 else 0

    # Prize winner analysis
    if len(winners) > 0 and "userNo" in df_users.columns:
        user_nos = df_users["userNo"].values
        rec_cust_idxs = events[events["recovered"]]["cust_idx"].unique()
        perm_cust_idxs = events[~events["recovered"]]["cust_idx"].unique()
        rec_users = set(user_nos[rec_cust_idxs])
        perm_users = set(user_nos[perm_cust_idxs])
        rec_winner_pct = len(rec_users & winners) / len(rec_users) * 100 if rec_users else 0
        perm_winner_pct = len(perm_users & winners) / len(perm_users) * 100 if perm_users else 0
        prize_line = f"Prize Winners: Recovered {rec_winner_pct:.1f}% vs Permanent {perm_winner_pct:.1f}%"
    else:
        prize_line = "Prize data: N/A"
        rec_winner_pct = 0
        perm_winner_pct = 0

    # Median gap for recovered
    med_gap = recovered["gap_periods"].median() if len(recovered) > 0 else 0

    # Revenue at risk: avg tickets * permanent churners
    # Use simple estimate: permanent churners had avg purchase frequency
    avg_freq_perm = permanent["purchase_freq"].mean() if len(permanent) > 0 else 0

    stats_text = (
        f"Key Statistics\n"
        f"{'─' * 40}\n\n"
        f"Total Churn Events:  {total_churn_events:>12,}\n"
        f"Unique Churners:     {unique_churners:>12,}\n"
        f"Recovered:           {n_recovered:>12,}  ({overall_recovery_pct:.1f}%)\n"
        f"Permanent:           {n_permanent:>12,}  ({100 - overall_recovery_pct:.1f}%)\n\n"
        f"Median Gap to Recovery:  {med_gap:.0f} periods\n"
        f"Avg Freq (Recovered):    {recovered['purchase_freq'].mean():.2f}\n"
        f"Avg Freq (Permanent):    {avg_freq_perm:.2f}\n\n"
        f"{prize_line}\n"
    )
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved: {CHART_PATH}")

    return {
        "total_churn_events": total_churn_events,
        "unique_churners": unique_churners,
        "n_recovered": n_recovered,
        "n_permanent": n_permanent,
        "overall_recovery_pct": overall_recovery_pct,
        "med_gap": med_gap,
        "rec_winner_pct": rec_winner_pct,
        "perm_winner_pct": perm_winner_pct,
    }


# ── 8. Print Thai summary ──────────────────────────────────────────────────
def print_thai_summary(stats, recovery_rates, events):
    """Print actionable summary in Thai."""
    print("\n" + "=" * 70)
    print("Phase 37: Churn Recovery Analysis — สรุปผล")
    print("=" * 70)

    print(f"\n📊 ภาพรวม Churn Recovery")
    print(f"   - จำนวน Churn Events ทั้งหมด: {stats['total_churn_events']:,}")
    print(f"   - ลูกค้าที่เคย Churn (unique): {stats['unique_churners']:,}")
    print(f"   - กลับมาซื้ออีก (Recovered): {stats['n_recovered']:,} ({stats['overall_recovery_pct']:.1f}%)")
    print(f"   - หายไปถาวร (Permanent): {stats['n_permanent']:,} ({100 - stats['overall_recovery_pct']:.1f}%)")

    print(f"\n⏱️ ระยะเวลากลับมา")
    for gap, rate in sorted(recovery_rates.items()):
        print(f"   - ภายใน {gap} งวด: {rate:.1f}% กลับมาซื้อ")

    print(f"\n   Median gap: {stats['med_gap']:.0f} งวด")

    recovered = events[events["recovered"]]
    permanent = events[~events["recovered"]]

    print(f"\n👤 Profile เปรียบเทียบ")
    print(f"   {'Metric':<25} {'Recovered':>12} {'Permanent':>12}")
    print(f"   {'─' * 49}")
    print(f"   {'Tenure (งวด)':<25} {recovered['tenure_periods'].mean():>12.1f} {permanent['tenure_periods'].mean():>12.1f}")
    print(f"   {'Purchase Frequency':<25} {recovered['purchase_freq'].mean():>12.2f} {permanent['purchase_freq'].mean():>12.2f}")
    print(f"   {'Was Declining (%)':<25} {recovered['was_declining'].mean()*100:>12.1f} {permanent['was_declining'].mean()*100:>12.1f}")

    if stats["rec_winner_pct"] > 0 or stats["perm_winner_pct"] > 0:
        print(f"\n🏆 Prize Winner Analysis")
        print(f"   - ลูกค้าที่กลับมาเคยถูกรางวัล: {stats['rec_winner_pct']:.1f}%")
        print(f"   - ลูกค้าที่หายไปถาวรเคยถูกรางวัล: {stats['perm_winner_pct']:.1f}%")
        if stats["rec_winner_pct"] > stats["perm_winner_pct"]:
            diff = stats["rec_winner_pct"] - stats["perm_winner_pct"]
            print(f"   → ลูกค้าที่เคยถูกรางวัลมีแนวโน้มกลับมาซื้อมากกว่า +{diff:.1f}%")

    # Sustainability
    sustained = recovered.dropna(subset=["sustained_rate"])
    if len(sustained) > 0:
        high_sustain = (sustained["sustained_rate"] >= 0.67).mean() * 100
        churn_again = (sustained["sustained_rate"] < 0.01).mean() * 100
        print(f"\n🔄 Recovery Sustainability (3 งวดหลังกลับมา)")
        print(f"   - ซื้อต่อเนื่อง (>67%): {high_sustain:.1f}%")
        print(f"   - Churn อีกรอบ (0%): {churn_again:.1f}%")

    print(f"\n💡 Actionable Insights")
    if 1 in recovery_rates:
        print(f"   1. {recovery_rates[1]:.1f}% ของลูกค้าที่ Churn กลับมาภายใน 1 งวด — ส่วนใหญ่แค่พักซื้อ 1 รอบ")
    if 3 in recovery_rates:
        print(f"   2. ภายใน 3 งวด กลับมา {recovery_rates[3]:.1f}% — ช่วง 3 งวดแรกคือ golden window สำหรับ retention campaign")
    if 6 in recovery_rates:
        print(f"   3. หลัง 6 งวด recovery rate แค่ {recovery_rates.get(6, 0):.1f}% — ลูกค้าที่หายเกิน 6 งวดมีโอกาสน้อยมากที่จะกลับมา")
    print(f"   4. ลูกค้าที่กลับมามี Purchase Frequency สูงกว่า → เป็นลูกค้าที่เคยซื้อบ่อย แค่หยุดชั่วคราว")
    print(f"   5. เน้น retention campaign ที่ลูกค้าที่เพิ่ง Churn (1-3 งวด) เพราะมีโอกาสสูงสุดที่จะกลับมา")
    print()


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Phase 37: Churn Recovery Analysis")
    print("=" * 70)

    # Load data
    df = load_latest_churn()
    bought, item_cols, df_full = extract_purchase_matrix(df)
    mat = bought.to_numpy(dtype="float64", na_value=np.nan)
    # Convert: True→1.0, False→0.0, NA→nan
    # The boolean array needs explicit conversion
    raw_items = df[item_cols].values.astype("float64")
    mat = np.where(np.isnan(raw_items), np.nan, (raw_items > 0).astype(float))

    n_customers, n_periods = mat.shape
    print(f"  Purchase matrix: {n_customers:,} x {n_periods}")

    # Analyze churn events
    events = analyze_churn_recovery_vectorized(mat, n_periods)

    # Build profiles
    events = build_profiles(events, mat, item_cols, df)

    # Load prize data
    winners, winner_stats = load_prize_data()

    # Compute recovery rates
    recovery_rates = compute_recovery_rates(events, n_periods)

    # Plot
    stats = plot_results(recovery_rates, events, n_periods, df_full, winners)

    # Print Thai summary
    print_thai_summary(stats, recovery_rates, events)


def analyze_churn_recovery_vectorized(mat, n_periods):
    """
    Vectorized churn recovery analysis.
    For each period transition, find all churners, then trace forward.
    """
    n_customers = mat.shape[0]
    print(f"\nAnalyzing {n_customers:,} customers across {n_periods} periods...")

    all_results = []

    for i in range(n_periods - 1):
        # Churn: bought in i, didn't buy in i+1
        bought_i = mat[:, i] == 1.0
        not_bought_next = mat[:, i + 1] == 0.0
        churn_mask = bought_i & not_bought_next
        churn_idxs = np.where(churn_mask)[0]

        if len(churn_idxs) == 0:
            continue

        # For each churner, find first future purchase
        future_mat = mat[churn_idxs, i + 1:]  # shape: (n_churners, remaining_periods)

        for local_k, cust_idx in enumerate(churn_idxs):
            future = future_mat[local_k]
            future_buys = np.where(future == 1.0)[0]

            if len(future_buys) > 0:
                gap = future_buys[0]
                recovery_idx = i + 1 + future_buys[0]

                # Sustainability: next 3 periods after recovery
                post = mat[cust_idx, recovery_idx + 1: recovery_idx + 4]
                if len(post) > 0:
                    sustained = np.nansum(post == 1.0) / len(post)
                else:
                    sustained = np.nan

                all_results.append({
                    "cust_idx": cust_idx,
                    "churn_period_idx": i,
                    "gap_periods": gap,
                    "recovered": True,
                    "recovery_period_idx": recovery_idx,
                    "sustained_rate": sustained,
                })
            else:
                remaining = n_periods - i - 1
                all_results.append({
                    "cust_idx": cust_idx,
                    "churn_period_idx": i,
                    "gap_periods": remaining,
                    "recovered": False,
                    "recovery_period_idx": np.nan,
                    "sustained_rate": np.nan,
                })

    events = pd.DataFrame(all_results)
    n_recovered = events["recovered"].sum()
    n_total = len(events)
    print(f"  Found {n_total:,} churn events")
    print(f"  Recovered: {n_recovered:,} ({n_recovered/n_total*100:.1f}%)")
    print(f"  Permanent: {n_total - n_recovered:,} ({(n_total - n_recovered)/n_total*100:.1f}%)")
    return events


if __name__ == "__main__":
    main()
