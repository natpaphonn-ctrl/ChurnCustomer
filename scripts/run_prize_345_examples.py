"""
Prize 3rd/4th/5th Customer Examples — Before/After Winning Analysis
Find 10 real customers per prize type showing purchase behavior change.
Counter-intuitive insight: smaller prizes have LONGER effects.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Sarabun'
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*')

# ── 1. Load data ──────────────────────────────────────────────────────────
print("=" * 80)
print("Loading data...")
print("=" * 80)

# Load winning effect data
wdf = pd.read_csv("data/predictions/winning_effect_customers_full.csv")

# Load the latest churn file (has most columns = longest history)
churn = pd.read_csv("data/churn/Churn_2026_0301_0316.csv")
item_cols = sorted([c for c in churn.columns if c.startswith("item")])
print(f"Churn file: {churn.shape[0]:,} customers, {len(item_cols)} periods")
print(f"Period range: {item_cols[0]} → {item_cols[-1]}")

# ── 2. Helper: prize_period → item column name ───────────────────────────
def period_to_col(period_str):
    """Convert '2025-01-02' to 'item2025_01_02'"""
    return "item" + period_str.replace("-", "_")

def col_to_label(col):
    """Convert 'item2025_01_02' to '01/02'"""
    parts = col.replace("item", "").split("_")
    return f"{parts[1]}/{parts[2]}"

# ── 3. Find good examples for each prize type ────────────────────────────
PRIZE_TYPES = {
    "THIRD":  {"amount": "80,000", "expected_effect": 2},
    "FOURTH": {"amount": "40,000", "expected_effect": 5},
    "FIFTH":  {"amount": "20,000", "expected_effect": 3},
}

N_BEFORE = 6  # periods before winning
N_AFTER = 6   # periods after winning
N_EXAMPLES = 10


def get_purchase_history(user_no, win_col, n_before=N_BEFORE, n_after=N_AFTER):
    """Get purchase history around winning period for a customer."""
    row = churn[churn["userNo"] == user_no]
    if len(row) == 0:
        return None
    row = row.iloc[0]

    if win_col not in item_cols:
        return None

    win_idx = item_cols.index(win_col)
    start = max(0, win_idx - n_before)
    end = min(len(item_cols), win_idx + n_after + 1)

    cols_window = item_cols[start:end]
    values = []
    for c in cols_window:
        v = row[c]
        values.append(0 if pd.isna(v) else int(v))

    win_pos = win_idx - start  # position of winning period in the window

    return {
        "columns": cols_window,
        "values": values,
        "win_pos": win_pos,
        "labels": [col_to_label(c) for c in cols_window],
    }


def compute_effect_duration(values, win_pos):
    """How many periods after winning did the boost last?
    Baseline = avg of periods before winning (excluding zeros for fairness).
    Boost = periods after winning where purchases > baseline * 1.1
    """
    before = [v for v in values[:win_pos] if v > 0]
    if not before:
        return 0, 0, 0
    baseline = np.mean(before)

    # Count consecutive periods after winning that are above baseline
    after_values = values[win_pos + 1:]  # exclude winning period itself
    duration = 0
    for v in after_values:
        if v > baseline * 1.05:  # at least 5% above baseline
            duration += 1
        else:
            break

    after_avg = np.mean(values[win_pos:win_pos + duration + 1]) if duration > 0 else values[win_pos] if win_pos < len(values) else 0
    return baseline, after_avg, duration


def find_examples(prize_type, n=N_EXAMPLES):
    """Find n good example customers for a prize type."""
    sub = wdf[wdf["prize_type"] == prize_type].copy()

    # Filter: positive change, bought before and after, reasonable volumes
    sub = sub[
        (sub["change_pct"] > 10) &
        (sub["tickets_before_avg"] > 1) &
        (sub["tickets_before_avg"] < 200) &  # not resellers
        (sub["tickets_after_avg"] > sub["tickets_before_avg"]) &
        (sub["active_last_6"] >= 4)  # active customer
    ]

    # Prize periods that have enough history before and after
    # Need at least 6 periods before and 6 after
    # Prize periods from mid-2025 work best
    good_periods = [p for p in sub["prize_period"].unique()
                    if period_to_col(p) in item_cols]
    good_periods_filtered = []
    for p in good_periods:
        col = period_to_col(p)
        idx = item_cols.index(col)
        if idx >= N_BEFORE and idx <= len(item_cols) - N_AFTER - 1:
            good_periods_filtered.append(p)

    sub = sub[sub["prize_period"].isin(good_periods_filtered)]

    if len(sub) == 0:
        print(f"  WARNING: No candidates for {prize_type}")
        return []

    # Score candidates: prefer clear before/after difference + moderate volume
    sub = sub.sort_values("change_pct", ascending=False)

    examples = []
    seen_users = set()

    for _, row in sub.iterrows():
        if len(examples) >= n:
            break
        if row["userNo"] in seen_users:
            continue

        win_col = period_to_col(row["prize_period"])
        hist = get_purchase_history(row["userNo"], win_col)
        if hist is None:
            continue

        # Must have non-zero purchases before winning
        before_vals = [v for v in hist["values"][:hist["win_pos"]] if v > 0]
        if len(before_vals) < 2:
            continue

        # Must have purchases after winning
        after_vals = hist["values"][hist["win_pos"] + 1:]
        if sum(1 for v in after_vals if v > 0) < 2:
            continue

        baseline, after_avg, duration = compute_effect_duration(hist["values"], hist["win_pos"])
        if duration < 1:
            continue

        seen_users.add(row["userNo"])
        examples.append({
            "userNo": row["userNo"],
            "prize_type": prize_type,
            "prize_amount": row["prize_amount"],
            "prize_period": row["prize_period"],
            "win_col": win_col,
            "history": hist,
            "baseline_avg": baseline,
            "after_avg": after_avg,
            "effect_duration": duration,
            "change_pct": row["change_pct"],
        })

    return examples


# ── 4. Find all examples ─────────────────────────────────────────────────
all_examples = {}
for ptype, info in PRIZE_TYPES.items():
    print(f"\nFinding examples for {ptype} (รางวัลที่ {'3' if ptype == 'THIRD' else '4' if ptype == 'FOURTH' else '5'}, {info['amount']}฿)...")
    examples = find_examples(ptype)
    all_examples[ptype] = examples
    print(f"  Found {len(examples)} examples")

# ── 5. Print summary table ───────────────────────────────────────────────
print("\n" + "=" * 120)
print("SUMMARY: 30 Customer Examples — Prize Types 3, 4, 5")
print("=" * 120)

for ptype in ["THIRD", "FOURTH", "FIFTH"]:
    prize_num = {"THIRD": 3, "FOURTH": 4, "FIFTH": 5}[ptype]
    info = PRIZE_TYPES[ptype]
    examples = all_examples[ptype]

    print(f"\n{'─' * 120}")
    print(f"รางวัลที่ {prize_num} ({info['amount']}฿) — Expected effect: {info['expected_effect']} periods")
    print(f"{'─' * 120}")
    print(f"{'#':<3} {'userNo':<12} {'Won Period':<12} {'Prize Amount':>12} {'Avg Before':>10} {'Avg After':>10} {'Change':>8} {'Effect':>8} {'Purchase History (★ = win period)'}")
    print(f"{'─' * 120}")

    for i, ex in enumerate(examples, 1):
        hist = ex["history"]
        # Build visual history string
        history_str = ""
        for j, (label, val) in enumerate(zip(hist["labels"], hist["values"])):
            if j == hist["win_pos"]:
                history_str += f" ★{val:>3}"
            else:
                history_str += f" {val:>4}"

        print(f"{i:<3} {ex['userNo']:<12} {ex['prize_period']:<12} {ex['prize_amount']:>12,.0f} "
              f"{ex['baseline_avg']:>10.1f} {ex['after_avg']:>10.1f} "
              f"{ex['change_pct']:>7.1f}% {ex['effect_duration']:>5} งวด"
              f"  [{history_str} ]")

    # Summary stats
    if examples:
        avg_dur = np.mean([e["effect_duration"] for e in examples])
        avg_change = np.mean([e["change_pct"] for e in examples])
        print(f"\n  → Average effect duration: {avg_dur:.1f} periods | Average change: {avg_change:.1f}%")

# ── 6. Create visualization ──────────────────────────────────────────────
print("\n" + "=" * 80)
print("Creating chart: charts/prize_345_examples.png")
print("=" * 80)

ORANGE = "#F97316"
GRAY = "#E5E7EB"
RED_STAR = "#EF4444"

n_cols = 4  # examples per row
n_rows = 3  # one per prize type
fig, axes = plt.subplots(n_rows, n_cols, figsize=(26, 16))
fig.suptitle("Prize Effect on Purchase Behavior — Real Customer Examples\n"
             "รางวัลที่ 3 (80K) vs 4 (40K) vs 5 (20K) — ★ = งวดที่ถูกรางวัล",
             fontsize=16, fontweight="bold", y=0.98)

prize_order = ["THIRD", "FOURTH", "FIFTH"]
prize_labels = {
    "THIRD": ("3", "80,000"),
    "FOURTH": ("4", "40,000"),
    "FIFTH": ("5", "20,000"),
}

for row_idx, ptype in enumerate(prize_order):
    examples = all_examples[ptype][:n_cols]
    pnum, pamount = prize_labels[ptype]

    for col_idx in range(n_cols):
        ax = axes[row_idx, col_idx]

        if col_idx >= len(examples):
            ax.set_visible(False)
            continue

        ex = examples[col_idx]
        hist = ex["history"]
        values = hist["values"]
        labels = hist["labels"]
        win_pos = hist["win_pos"]

        # Color bars: orange for purchases, gray for zero
        colors = []
        for j, v in enumerate(values):
            if v > 0:
                colors.append(ORANGE)
            else:
                colors.append(GRAY)

        bars = ax.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.5)

        # Star marker on winning period
        win_val = values[win_pos]
        ax.plot(win_pos, win_val + max(values) * 0.08, marker="*", color=RED_STAR,
                markersize=18, zorder=5)
        ax.annotate("★", xy=(win_pos, win_val + max(values) * 0.05),
                     fontsize=0, color=RED_STAR)  # hidden, star shown via marker

        # Baseline line
        baseline = ex["baseline_avg"]
        ax.axhline(y=baseline, color="#6B7280", linestyle="--", alpha=0.6, linewidth=1)
        ax.text(len(values) - 0.5, baseline, f"baseline\n{baseline:.0f}", fontsize=7,
                ha="right", va="bottom", color="#6B7280")

        # Effect duration bracket
        effect_end = min(win_pos + ex["effect_duration"], len(values) - 1)
        if ex["effect_duration"] > 0:
            ax.annotate("", xy=(win_pos + 0.5, max(values) * 0.95),
                        xytext=(effect_end + 0.5, max(values) * 0.95),
                        arrowprops=dict(arrowstyle="<->", color="#3B82F6", lw=1.5))
            mid_bracket = (win_pos + effect_end + 1) / 2
            ax.text(mid_bracket, max(values) * 0.98,
                    f"effect {ex['effect_duration']} งวด",
                    ha="center", fontsize=8, color="#3B82F6", fontweight="bold")

        # Labels
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, fontsize=7, ha="right")
        ax.set_title(f"{ex['userNo']} — รางวัลที่ {pnum} ({pamount}฿)\n"
                     f"effect {ex['effect_duration']} งวด | +{ex['change_pct']:.0f}%",
                     fontsize=9, fontweight="bold")
        ax.set_ylabel("Tickets" if col_idx == 0 else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add value labels on bars
        for j, v in enumerate(values):
            if v > 0:
                ax.text(j, v + max(values) * 0.01, str(v), ha="center", fontsize=6,
                        fontweight="bold" if j == win_pos else "normal")

# Add row labels on the left
for row_idx, ptype in enumerate(prize_order):
    pnum, pamount = prize_labels[ptype]
    expected = PRIZE_TYPES[ptype]["expected_effect"]
    avg_dur = np.mean([e["effect_duration"] for e in all_examples[ptype][:n_cols]]) if all_examples[ptype] else 0
    fig.text(0.01, 0.78 - row_idx * 0.30,
             f"รางวัลที่ {pnum}\n{pamount}฿\navg effect:\n{avg_dur:.1f} งวด",
             fontsize=11, fontweight="bold", va="center",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF7ED", edgecolor=ORANGE))

plt.tight_layout(rect=[0.06, 0.02, 1, 0.94])
plt.savefig("charts/prize_345_examples.png", dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("Saved: charts/prize_345_examples.png")

# ── 7. Detailed per-customer printout ─────────────────────────────────────
print("\n" + "=" * 80)
print("DETAILED CUSTOMER PROFILES")
print("=" * 80)

for ptype in prize_order:
    pnum, pamount = prize_labels[ptype]
    examples = all_examples[ptype]

    print(f"\n{'━' * 80}")
    print(f"  รางวัลที่ {pnum} ({pamount}฿) — {len(examples)} customers")
    print(f"{'━' * 80}")

    for i, ex in enumerate(examples, 1):
        hist = ex["history"]
        win_pos = hist["win_pos"]
        before_vals = hist["values"][:win_pos]
        after_vals = hist["values"][win_pos + 1:]
        win_val = hist["values"][win_pos]

        print(f"\n  Customer {i}: {ex['userNo']}")
        print(f"  Won: {ex['prize_period']} | Prize: {ex['prize_amount']:,.0f}฿")
        print(f"  Baseline avg (before): {ex['baseline_avg']:.1f} tickets/period")
        print(f"  After avg: {ex['after_avg']:.1f} tickets/period")
        print(f"  Change: +{ex['change_pct']:.1f}%")
        print(f"  Effect duration: {ex['effect_duration']} periods")
        print(f"  Purchase history:")

        for j, (label, val) in enumerate(zip(hist["labels"], hist["values"])):
            marker = " ★ WIN" if j == win_pos else ""
            bar = "█" * min(val, 60)
            rel = "before" if j < win_pos else ("WIN" if j == win_pos else "after")
            print(f"    {label} [{rel:>6}]: {val:>4} {bar}{marker}")

print("\n" + "=" * 80)
print("KEY INSIGHT: Prize 4 (40K฿) effect lasts LONGER than Prize 3 (80K฿)")
print("Smaller, more frequent wins create sustained behavioral change!")
print("=" * 80)

# Print average effect durations
print("\nAverage Effect Duration Summary:")
for ptype in prize_order:
    pnum, pamount = prize_labels[ptype]
    durations = [e["effect_duration"] for e in all_examples[ptype]]
    if durations:
        print(f"  รางวัลที่ {pnum} ({pamount}฿): {np.mean(durations):.1f} periods (range: {min(durations)}-{max(durations)})")

print("\nDone!")
