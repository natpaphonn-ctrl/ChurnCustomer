#!/usr/bin/env python3
"""
Winning Effect by Prize Type — Before/After Purchase Behavior Analysis
======================================================================
For each prize type, tracks average tickets purchased:
  - 3 periods BEFORE winning
  - Period of winning (period 0)
  - 6 periods AFTER winning
Produces 2 charts + console summary.
"""

import glob
import os
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

# ── Thai font setup ──────────────────────────────────────────────────
for font_name in [
    "TH Sarabun New", "Tahoma", "Angsana New", "Browallia New",
    "Garuda", "Loma", "Norasi", "TlwgTypo", "Sawasdee",
]:
    try:
        matplotlib.rcParams["font.family"] = font_name
        fig_test = plt.figure(figsize=(1, 1))
        fig_test.text(0.5, 0.5, "ทดสอบ")
        fig_test.savefig("/dev/null", format="png")
        plt.close(fig_test)
        THAI_FONT = font_name
        break
    except Exception:
        plt.close("all")
        continue
else:
    THAI_FONT = None
    matplotlib.rcParams["font.family"] = "sans-serif"

plt.rcParams.update({"axes.unicode_minus": False, "figure.max_open_warning": 0})

# ── Constants ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIZE_DIR = os.path.join(PROJECT_ROOT, "data", "prize")
CHURN_DIR = os.path.join(PROJECT_ROOT, "data", "churn")
CHART_DIR = os.path.join(PROJECT_ROOT, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# Prize type metadata — amount per ticket
PRIZE_META = {
    "LAST2DIGITS":   {"amount": 2_000,     "thai": "เลขท้าย 2 ตัว",     "short": "ท้าย2"},
    "LAST3DIGITS":   {"amount": 4_000,     "thai": "เลขท้าย 3 ตัว",     "short": "ท้าย3"},
    "FIRST3DIGITS":  {"amount": 4_000,     "thai": "เลขหน้า 3 ตัว",     "short": "หน้า3"},
    "FIFTH":         {"amount": 20_000,    "thai": "รางวัลที่ 5",       "short": "ที่5"},
    "FOURTH":        {"amount": 40_000,    "thai": "รางวัลที่ 4",       "short": "ที่4"},
    "THIRD":         {"amount": 80_000,    "thai": "รางวัลที่ 3",       "short": "ที่3"},
    "SECOND":        {"amount": 200_000,   "thai": "รางวัลที่ 2",       "short": "ที่2"},
    "NEARBY":        {"amount": 100_000,   "thai": "รางวัลข้างเคียง",   "short": "ข้างเคียง"},
    "FIRST":         {"amount": 6_000_000, "thai": "รางวัลที่ 1",       "short": "ที่1"},
}

# Sort order by prize amount
PRIZE_ORDER = sorted(PRIZE_META.keys(), key=lambda x: PRIZE_META[x]["amount"])

# Colors — gradient from light (low prize) to dark (high prize), red for jackpot
PRIZE_COLORS = {
    "LAST2DIGITS":  "#93C5FD",
    "LAST3DIGITS":  "#60A5FA",
    "FIRST3DIGITS": "#3B82F6",
    "FIFTH":        "#2563EB",
    "FOURTH":       "#1D4ED8",
    "THIRD":        "#1E40AF",
    "SECOND":       "#1E3A8A",
    "NEARBY":       "#312E81",
    "FIRST":        "#DC2626",
}

# Before/after window
N_BEFORE = 3
N_AFTER = 6


def parse_item_date(col_name):
    """Convert item column name like 'item2025_01_02' to a comparable date string."""
    parts = col_name.replace("item", "").split("_")
    return f"{parts[0]}-{parts[1]}-{parts[2]}"


def rounddate_to_item_col(rounddate_str):
    """Convert rounddate like '2025-01-02' to item column name 'item2025_01_02'."""
    return "item" + rounddate_str.replace("-", "_")


def load_churn_data():
    """Load the latest churn file (most item columns = most history)."""
    churn_files = sorted(glob.glob(os.path.join(CHURN_DIR, "Churn_202*_*_*.csv")))
    # Filter out _Check and _old files
    churn_files = [f for f in churn_files if "_Check" not in f and "_old" not in f]
    if not churn_files:
        print("ERROR: No churn files found")
        sys.exit(1)

    # Use the latest file (most history)
    latest = churn_files[-1]
    print(f"Loading churn data from: {os.path.basename(latest)}")
    df = pd.read_csv(latest, low_memory=False)

    # Identify item columns
    item_cols = [c for c in df.columns if c.startswith("item")]
    item_cols_sorted = sorted(item_cols, key=lambda c: parse_item_date(c))

    print(f"  Customers: {len(df):,}")
    print(f"  Item columns: {len(item_cols_sorted)} ({item_cols_sorted[0]} → {item_cols_sorted[-1]})")

    return df, item_cols_sorted


def load_prize_data():
    """Load all prize files and combine."""
    prize_files = sorted(glob.glob(os.path.join(PRIZE_DIR, "Prize_*.csv")))
    if not prize_files:
        print("ERROR: No prize files found")
        sys.exit(1)

    dfs = []
    for f in prize_files:
        df = pd.read_csv(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nLoaded {len(prize_files)} prize files: {len(combined):,} prize records")
    print(f"  Prize types: {combined['type'].nunique()}")
    print(f"  Unique winners: {combined['userNo'].nunique():,}")
    print(f"  Period range: {combined['rounddate'].min()} → {combined['rounddate'].max()}")

    return combined


def build_timeline_data(churn_df, item_cols, prize_df):
    """
    For each winner and prize type, extract the ticket purchase timeline
    relative to their winning period.
    """
    # Build item column → index mapping
    col_to_idx = {col: i for i, col in enumerate(item_cols)}

    # Build rounddate → item column mapping
    rd_to_col = {}
    for col in item_cols:
        date_str = parse_item_date(col)
        rd_to_col[date_str] = col

    # Get unique rounddates from prize data that map to item columns
    prize_rounddates = prize_df["rounddate"].unique()
    valid_rounddates = [rd for rd in prize_rounddates if rd in rd_to_col]
    print(f"\n  Prize rounddates mappable to item columns: {len(valid_rounddates)}/{len(prize_rounddates)}")

    # Set userNo as index for fast lookup
    churn_indexed = churn_df.set_index("userNo")

    # Extract item values as numpy for speed
    item_values = churn_indexed[item_cols].values  # shape: (n_customers, n_periods)
    user_index = churn_indexed.index

    # Create user → row index mapping
    user_to_row = pd.Series(range(len(user_index)), index=user_index)

    # For each prize type, collect timelines
    results = {}

    for prize_type in PRIZE_ORDER:
        subset = prize_df[
            (prize_df["type"] == prize_type) &
            (prize_df["rounddate"].isin(valid_rounddates))
        ]
        if len(subset) == 0:
            continue

        timelines = []

        for _, row in subset.iterrows():
            user = row["userNo"]
            rd = row["rounddate"]

            if user not in user_to_row.index:
                continue

            user_row_idx = user_to_row[user]
            win_col = rd_to_col[rd]
            win_col_idx = col_to_idx[win_col]

            # Need N_BEFORE periods before and N_AFTER periods after
            start_idx = win_col_idx - N_BEFORE
            end_idx = win_col_idx + N_AFTER + 1  # inclusive

            if start_idx < 0 or end_idx > len(item_cols):
                continue

            # Extract the window
            window = item_values[user_row_idx, start_idx:end_idx]  # length = N_BEFORE + 1 + N_AFTER

            timelines.append({
                "userNo": user,
                "rounddate": rd,
                "window": window,
            })

        if timelines:
            results[prize_type] = timelines
            print(f"  {prize_type:15s}: {len(timelines):6,} winner-period pairs with full timeline")

    return results


def compute_baseline(churn_df, item_cols, prize_df):
    """Compute baseline average tickets for customers who NEVER won any prize."""
    all_winners = set(prize_df["userNo"].unique())
    non_winners = churn_df[~churn_df["userNo"].isin(all_winners)]
    print(f"\n  Non-winners (baseline): {len(non_winners):,} customers")

    # Average tickets across all periods (ignoring NaN = not registered)
    item_values = non_winners[item_cols].values
    # Replace 0 with NaN for "bought" average? No — 0 means registered but didn't buy
    # We want the average INCLUDING zeros (same as winners)
    baseline = np.nanmean(item_values)
    print(f"  Baseline avg tickets/period: {baseline:.2f}")
    return baseline


def analyze_timelines(results, baseline):
    """Compute summary statistics per prize type."""
    period_labels = list(range(-N_BEFORE, N_AFTER + 1))  # -3, -2, -1, 0, 1, 2, ..., 6

    summary = {}
    avg_timelines = {}

    for prize_type in PRIZE_ORDER:
        if prize_type not in results:
            continue

        timelines = results[prize_type]
        n = len(timelines)

        # Stack all windows: shape (n, N_BEFORE + 1 + N_AFTER)
        windows = np.array([t["window"] for t in timelines], dtype=float)

        # Replace NaN with 0 for averaging (NaN = not registered, treat as 0 tickets)
        windows_filled = np.where(np.isnan(windows), 0, windows)

        # Average across all winners at each relative period
        avg = np.nanmean(windows_filled, axis=0)

        avg_timelines[prize_type] = avg

        # Pre-win baseline (average of periods -3, -2, -1)
        pre_win_avg = np.mean(avg[:N_BEFORE])

        # Post-win values
        post_1 = avg[N_BEFORE + 1] if N_BEFORE + 1 < len(avg) else np.nan  # +1
        post_3 = avg[N_BEFORE + 3] if N_BEFORE + 3 < len(avg) else np.nan  # +3
        post_6 = avg[N_BEFORE + 6] if N_BEFORE + 6 < len(avg) else np.nan  # +6
        peak = np.max(avg[N_BEFORE:])  # max from period 0 onward
        peak_period = np.argmax(avg[N_BEFORE:])  # relative to period 0

        # % increase at peak
        pct_increase = ((peak - pre_win_avg) / pre_win_avg * 100) if pre_win_avg > 0 else 0

        # Duration: how many periods after win until back within 10% of pre-win baseline
        threshold = pre_win_avg * 1.10
        duration = 0
        for i in range(1, N_AFTER + 1):
            idx = N_BEFORE + i
            if idx < len(avg) and avg[idx] > threshold:
                duration = i
            else:
                break

        summary[prize_type] = {
            "n_winners": n,
            "pre_win_avg": pre_win_avg,
            "win_period_avg": avg[N_BEFORE],
            "post_1": post_1,
            "post_3": post_3,
            "post_6": post_6,
            "peak": peak,
            "peak_period": peak_period,
            "pct_increase": pct_increase,
            "duration": duration,
            "amount": PRIZE_META[prize_type]["amount"],
        }

    return summary, avg_timelines


def print_summary(summary):
    """Print console summary table."""
    print("\n" + "=" * 80)
    print("Winning Effect by Prize Type — Summary")
    print("=" * 80)
    print(f"{'Prize Type':<16s} | {'Amount':>10s} | {'Winners':>7s} | {'Before':>7s} | {'After+1':>8s} | {'% Up':>7s} | {'Duration'}")
    print("-" * 80)

    for pt in PRIZE_ORDER:
        if pt not in summary:
            continue
        s = summary[pt]
        pct_str = f"+{s['pct_increase']:.0f}%" if s['pct_increase'] > 0 else f"{s['pct_increase']:.0f}%"
        dur_str = f"{s['duration']}+ periods" if s['duration'] >= N_AFTER else f"{s['duration']} periods"
        print(f"{pt:<16s} | {s['amount']:>10,} | {s['n_winners']:>7,} | {s['pre_win_avg']:>7.1f} | {s['post_1']:>8.1f} | {pct_str:>7s} | {dur_str}")

    print("=" * 80)


def print_customer_examples(results, churn_df, item_cols):
    """Print real customer examples for top 3 most common prize types."""
    top_types = ["LAST2DIGITS", "LAST3DIGITS", "FIFTH"]

    print("\n" + "=" * 80)
    print("Customer Examples — Before/After Winning")
    print("=" * 80)

    for pt in top_types:
        if pt not in results:
            continue

        timelines = results[pt]
        thai_name = PRIZE_META[pt]["thai"]
        print(f"\n--- {pt} ({thai_name}, {PRIZE_META[pt]['amount']:,} บาท/ใบ) ---")

        # Find an "increased" buyer and a "maintained" buyer
        windows = np.array([t["window"] for t in timelines], dtype=float)
        windows_filled = np.where(np.isnan(windows), 0, windows)

        pre_avgs = np.mean(windows_filled[:, :N_BEFORE], axis=1)
        post_avgs = np.mean(windows_filled[:, N_BEFORE + 1:], axis=1)

        # Filter out zero pre-average (new customers)
        valid_mask = pre_avgs > 0
        if valid_mask.sum() < 2:
            print("  (Not enough customers with pre-win history)")
            continue

        pct_changes = np.where(valid_mask, (post_avgs - pre_avgs) / pre_avgs * 100, 0)

        # Find one who increased the most
        increased_idx = np.argmax(np.where(valid_mask, pct_changes, -999))
        # Find one who maintained (closest to 0% change)
        maintained_candidates = np.where(valid_mask, np.abs(pct_changes), 999)
        maintained_idx = np.argmin(maintained_candidates)

        for label, idx in [("Increased buyer", increased_idx), ("Maintained buyer", maintained_idx)]:
            t = timelines[idx]
            w = windows_filled[idx]
            pre = pre_avgs[idx]
            post = post_avgs[idx]
            pct = pct_changes[idx]

            period_strs = [f"{int(v):>3d}" for v in w]
            timeline_str = " → ".join(period_strs[:N_BEFORE]) + " | " + f"[{int(w[N_BEFORE]):>3d}]" + " | " + " → ".join(period_strs[N_BEFORE + 1:])
            print(f"  {label}: {t['userNo']} (won {t['rounddate']})")
            print(f"    Before → [Win] → After: {timeline_str}")
            print(f"    Pre avg: {pre:.1f}, Post avg: {post:.1f} ({pct:+.0f}%)")


# ── CHART 1: Main storytelling chart ─────────────────────────────────

def plot_main_chart(summary, avg_timelines, baseline):
    """Create the main multi-panel chart."""
    period_labels = list(range(-N_BEFORE, N_AFTER + 1))

    fig = plt.figure(figsize=(20, 28))
    gs = gridspec.GridSpec(6, 2, figure=fig, height_ratios=[1.2, 1.0, 0.8, 1.0, 1.0, 0.5],
                           hspace=0.35, wspace=0.3)

    # ── Panel A: Timeline overlay (top, full width) ──────────────────
    ax_a = fig.add_subplot(gs[0, :])

    for pt in PRIZE_ORDER:
        if pt not in avg_timelines:
            continue
        avg = avg_timelines[pt]
        color = PRIZE_COLORS[pt]
        thai = PRIZE_META[pt]["thai"]
        amt = PRIZE_META[pt]["amount"]
        n = summary[pt]["n_winners"]
        label = f"{thai} ({amt:,}฿) [n={n:,}]"

        lw = 2.5 if pt == "FIRST" else 1.8
        marker_size = 8 if pt == "FIRST" else 5

        ax_a.plot(period_labels, avg, color=color, linewidth=lw,
                  marker="o", markersize=marker_size, label=label, zorder=3)

        # Special marker at period 0 (winning)
        ax_a.plot(0, avg[N_BEFORE], marker="*", markersize=14, color=color,
                  zorder=5, markeredgecolor="white", markeredgewidth=0.5)

    # Baseline
    ax_a.axhline(y=baseline, color="gray", linestyle="--", linewidth=1.5,
                 alpha=0.7, label=f"Baseline (never won): {baseline:.1f}")

    # Vertical line at period 0
    ax_a.axvline(x=0, color="red", linestyle="--", linewidth=1.0, alpha=0.5)
    ax_a.text(0.05, ax_a.get_ylim()[1] * 0.95 if ax_a.get_ylim()[1] > 0 else 20,
              "งวดที่ถูกรางวัล", color="red", fontsize=10, ha="left", va="top")

    ax_a.set_xlabel("งวดสัมพัทธ์ (0 = งวดที่ถูกรางวัล)", fontsize=12)
    ax_a.set_ylabel("จำนวนตั๋วเฉลี่ย/งวด", fontsize=12)
    ax_a.set_title("พฤติกรรมการซื้อก่อน-หลังถูกรางวัล แยกตามประเภท", fontsize=16, fontweight="bold")
    ax_a.set_xticks(period_labels)
    ax_a.set_xticklabels([str(p) if p != 0 else "0\n(Win)" for p in period_labels])
    ax_a.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_xlim(-N_BEFORE - 0.3, N_AFTER + 0.3)

    # ── Panel B: Grouped bar chart ───────────────────────────────────
    ax_b = fig.add_subplot(gs[1, :])

    types_in_order = [pt for pt in PRIZE_ORDER if pt in summary]
    x = np.arange(len(types_in_order))
    width = 0.18

    before_vals = [summary[pt]["pre_win_avg"] for pt in types_in_order]
    after1_vals = [summary[pt]["post_1"] for pt in types_in_order]
    after3_vals = [summary[pt]["post_3"] for pt in types_in_order]
    after6_vals = [summary[pt]["post_6"] for pt in types_in_order]

    bars_before = ax_b.bar(x - 1.5 * width, before_vals, width, label="Before (avg -3 to -1)", color="#9CA3AF")
    bars_a1 = ax_b.bar(x - 0.5 * width, after1_vals, width, label="After +1 งวด", color="#3B82F6")
    bars_a3 = ax_b.bar(x + 0.5 * width, after3_vals, width, label="After +3 งวด", color="#93C5FD")
    bars_a6 = ax_b.bar(x + 1.5 * width, after6_vals, width, label="After +6 งวด", color="#DBEAFE")

    ax_b.set_xlabel("ประเภทรางวัล (เรียงตามเงินรางวัล)", fontsize=12)
    ax_b.set_ylabel("จำนวนตั๋วเฉลี่ย/งวด", fontsize=12)
    ax_b.set_title("ยอดซื้อเฉลี่ย: ก่อนถูก vs หลังถูก 1/3/6 งวด", fontsize=14, fontweight="bold")
    ax_b.set_xticks(x)
    x_labels = [f"{PRIZE_META[pt]['thai']}\n({PRIZE_META[pt]['amount']:,}฿)" for pt in types_in_order]
    ax_b.set_xticklabels(x_labels, fontsize=9)
    ax_b.legend(fontsize=10)
    ax_b.grid(True, alpha=0.3, axis="y")

    # ── Panel C: Horizontal bar — Duration ───────────────────────────
    ax_c = fig.add_subplot(gs[2, :])

    types_rev = list(reversed(types_in_order))
    y_pos = np.arange(len(types_rev))
    durations = [summary[pt]["duration"] for pt in types_rev]
    colors_c = [PRIZE_COLORS[pt] for pt in types_rev]

    bars_c = ax_c.barh(y_pos, durations, color=colors_c, edgecolor="white", linewidth=0.5)

    for i, pt in enumerate(types_rev):
        pct = summary[pt]["pct_increase"]
        dur = summary[pt]["duration"]
        ax_c.text(dur + 0.15, i, f"+{pct:.0f}% at peak", va="center", fontsize=9, color="#374151")

    ax_c.set_yticks(y_pos)
    y_labels = [f"{PRIZE_META[pt]['thai']} ({PRIZE_META[pt]['amount']:,}฿)" for pt in types_rev]
    ax_c.set_yticklabels(y_labels, fontsize=10)
    ax_c.set_xlabel("จำนวนงวดที่ Winning Effect คงอยู่", fontsize=12)
    ax_c.set_title("Winning Effect คงอยู่กี่งวด?", fontsize=14, fontweight="bold")
    ax_c.set_xlim(0, max(durations) + 2)
    ax_c.grid(True, alpha=0.3, axis="x")

    # ── Panel D: Heatmap ─────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[3, 0])

    # Build heatmap data: % change from pre-win baseline
    heatmap_data = []
    heatmap_labels = []
    for pt in types_in_order:
        avg = avg_timelines[pt]
        pre_avg = summary[pt]["pre_win_avg"]
        if pre_avg > 0:
            pct_change = (avg - pre_avg) / pre_avg * 100
        else:
            pct_change = np.zeros_like(avg)
        heatmap_data.append(pct_change)
        heatmap_labels.append(PRIZE_META[pt]["short"])

    heatmap_arr = np.array(heatmap_data)

    # Custom colormap: red-white-green
    cmap = LinearSegmentedColormap.from_list("rwg", ["#EF4444", "#FFFFFF", "#22C55E"])

    # Clip for better visualization
    vmax = min(np.nanmax(np.abs(heatmap_arr)), 200)
    im = ax_d.imshow(heatmap_arr, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ax_d.set_xticks(range(len(period_labels)))
    ax_d.set_xticklabels([str(p) for p in period_labels], fontsize=9)
    ax_d.set_yticks(range(len(heatmap_labels)))
    ax_d.set_yticklabels(heatmap_labels, fontsize=10)
    ax_d.set_xlabel("งวดสัมพัทธ์", fontsize=11)
    ax_d.set_title("% เปลี่ยนแปลงยอดซื้อ (เทียบ baseline ก่อนถูก)", fontsize=12, fontweight="bold")

    # Annotate cells
    for i in range(len(heatmap_data)):
        for j in range(len(period_labels)):
            val = heatmap_arr[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax_d.text(j, i, f"{val:+.0f}%", ha="center", va="center",
                          fontsize=7, color=color)

    plt.colorbar(im, ax=ax_d, shrink=0.8, label="% change")

    # ── Panel E: Summary table ───────────────────────────────────────
    ax_e = fig.add_subplot(gs[3, 1])
    ax_e.axis("off")

    col_labels = ["ประเภท", "เงินรางวัล", "จำนวน\nผู้ถูก", "ซื้อก่อน\nถูก", "ซื้อหลัง\nถูก(peak)", "% เพิ่ม", "คงอยู่\nกี่งวด"]
    table_data = []
    for pt in types_in_order:
        s = summary[pt]
        dur_str = f"{s['duration']}+" if s['duration'] >= N_AFTER else str(s['duration'])
        table_data.append([
            PRIZE_META[pt]["thai"],
            f"{s['amount']:,}",
            f"{s['n_winners']:,}",
            f"{s['pre_win_avg']:.1f}",
            f"{s['peak']:.1f}",
            f"+{s['pct_increase']:.0f}%",
            dur_str,
        ])

    table = ax_e.table(cellText=table_data, colLabels=col_labels,
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1E40AF")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(len(table_data)):
        color = "#F3F4F6" if i % 2 == 0 else "#FFFFFF"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)

    ax_e.set_title("สรุป Winning Effect แต่ละประเภท", fontsize=12, fontweight="bold", pad=20)

    # ── Panel F: Key Recommendations ─────────────────────────────────
    ax_f = fig.add_subplot(gs[4:, :])
    ax_f.axis("off")

    recommendations = [
        "Key Recommendations — ข้อเสนอแนะเชิงปฏิบัติ",
        "",
        "1. ลูกค้าถูกรางวัลที่ 1-3: follow up ทันที — buying effect สูงมากแต่อาจไม่ยั่งยืน",
        "   → ส่ง VIP promotion ภายใน 1 สัปดาห์หลังประกาศผล เพื่อรักษา momentum",
        "",
        "2. ลูกค้าถูกเลขท้าย 2 ตัว: จำนวนเยอะที่สุด — ส่ง promotion ภายใน 3 งวดหลังถูก",
        "   → กลุ่มนี้มีมากที่สุดแต่เงินรางวัลน้อย อาจต้อง offer เบาๆ เพื่อรักษา engagement",
        "",
        "3. ลูกค้าถูกเลขท้าย 3 ตัว / รางวัลที่ 4-5: sweet spot — effect คงอยู่นานและมีจำนวนมาก",
        "   → กลุ่มที่คุ้มค่าที่สุดในการทำ retention campaign",
        "",
        "4. ลูกค้าถูกรางวัลข้างเคียง: เงินรางวัลพอสมควร — ใกล้รางวัลที่ 1 อาจรู้สึก 'เกือบถูก'",
        "   → ส่ง message เชิง 'คราวหน้าอาจจะถูกรางวัลที่ 1' เพื่อ motivate",
    ]

    text = "\n".join(recommendations)
    ax_f.text(0.05, 0.95, text, transform=ax_f.transAxes,
              fontsize=11, verticalalignment="top", fontfamily="monospace" if not THAI_FONT else THAI_FONT,
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF3C7", edgecolor="#F59E0B", alpha=0.9),
              linespacing=1.5)

    # Main title
    fig.suptitle("Winning Effect Analysis — ผลกระทบของการถูกรางวัลต่อพฤติกรรมการซื้อ",
                 fontsize=18, fontweight="bold", y=0.995)

    out_path = os.path.join(CHART_DIR, "winning_effect_by_prize_type.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


# ── CHART 2: Simple version for slide ────────────────────────────────

def plot_simple_chart(summary, avg_timelines, baseline):
    """Create the simple 16:9 slide version."""
    period_labels = list(range(-N_BEFORE, N_AFTER + 1))

    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    for pt in PRIZE_ORDER:
        if pt not in avg_timelines:
            continue
        avg = avg_timelines[pt]
        color = PRIZE_COLORS[pt]
        thai = PRIZE_META[pt]["thai"]
        amt = PRIZE_META[pt]["amount"]
        n = summary[pt]["n_winners"]
        label = f"{thai} ({amt:,}฿) [n={n:,}]"

        lw = 3.0 if pt == "FIRST" else 2.0
        marker_size = 10 if pt == "FIRST" else 6

        ax.plot(period_labels, avg, color=color, linewidth=lw,
                marker="o", markersize=marker_size, label=label, zorder=3)

        # Star marker at period 0
        ax.plot(0, avg[N_BEFORE], marker="*", markersize=16, color=color,
                zorder=5, markeredgecolor="white", markeredgewidth=0.8)

    # Baseline
    ax.axhline(y=baseline, color="gray", linestyle="--", linewidth=2.0,
               alpha=0.7, label=f"Baseline (never won): {baseline:.1f}")

    # Vertical line at period 0
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, alpha=0.4)

    ax.set_xlabel("งวดสัมพัทธ์ (0 = งวดที่ถูกรางวัล)", fontsize=14)
    ax.set_ylabel("จำนวนตั๋วเฉลี่ย/งวด", fontsize=14)
    ax.set_title("พฤติกรรมการซื้อก่อน-หลังถูกรางวัล แยกตามประเภท",
                 fontsize=18, fontweight="bold", pad=15)
    ax.text(0.5, 1.02, "แต่ละเส้นแสดงจำนวนตั๋วเฉลี่ยที่ซื้อ 3 งวดก่อนถูกรางวัล → งวดที่ถูก → 6 งวดหลังถูก",
            transform=ax.transAxes, fontsize=12, ha="center", va="bottom", color="#6B7280")

    ax.set_xticks(period_labels)
    ax.set_xticklabels([str(p) if p != 0 else "0\n(Win)" for p in period_labels], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-N_BEFORE - 0.3, N_AFTER + 0.3)

    # Footer
    fig.text(0.5, 0.01,
             "Data: 30 prize files (2025-01 → 2026-03) × 66 purchase periods | Churn file: Churn_2026_0301_0316",
             ha="center", fontsize=9, color="#9CA3AF")

    out_path = os.path.join(CHART_DIR, "winning_effect_simple.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── MAIN ─────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("Winning Effect by Prize Type — Before/After Purchase Behavior")
    print("=" * 80)

    # Load data
    churn_df, item_cols = load_churn_data()
    prize_df = load_prize_data()

    # Build timelines
    print("\nBuilding purchase timelines around winning events...")
    results = build_timeline_data(churn_df, item_cols, prize_df)

    # Compute baseline
    print("\nComputing non-winner baseline...")
    baseline = compute_baseline(churn_df, item_cols, prize_df)

    # Analyze
    print("\nAnalyzing timelines...")
    summary, avg_timelines = analyze_timelines(results, baseline)

    # Print summary
    print_summary(summary)

    # Print customer examples
    print_customer_examples(results, churn_df, item_cols)

    # Plot charts
    print("\nGenerating charts...")
    plot_main_chart(summary, avg_timelines, baseline)
    plot_simple_chart(summary, avg_timelines, baseline)

    print("\nDone!")


if __name__ == "__main__":
    main()
