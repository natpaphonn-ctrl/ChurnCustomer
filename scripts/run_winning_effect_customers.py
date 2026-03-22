#!/usr/bin/env python3
"""
Winning Effect — Customer-Level Analysis
=========================================
For every customer who won a prize in the last 6 periods,
track their buying behavior before and after winning.

Output:
  - data/predictions/winning_effect_customers.csv
  - data/predictions/winning_effect_shift_targets.json
  - charts/winning_effect_customers.png

Run: python3 scripts/run_winning_effect_customers.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ---------- Thai font setup ----------
for fp in [
    "/System/Library/Fonts/Thonburi.ttc",
    "/System/Library/Fonts/Supplemental/Thonburi.ttc",
    "/Library/Fonts/Thonburi.ttc",
]:
    if os.path.exists(fp):
        from matplotlib import font_manager
        font_manager.fontManager.addfont(fp)
        matplotlib.rcParams["font.family"] = "Thonburi"
        break
plt.rcParams.update({"axes.unicode_minus": False})

DATA_DIR = "data"
PRIZE_DIR = os.path.join(DATA_DIR, "prize")
CHURN_DIR = os.path.join(DATA_DIR, "churn")
PRED_DIR = os.path.join(DATA_DIR, "predictions")
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# ============================================================
# 1. Load prize data for last 6 periods
# ============================================================
print("=" * 60)
print("Winning Effect — พฤติกรรมการซื้อหลังถูกรางวัล")
print("=" * 60)
print()

prize_files = sorted([f for f in os.listdir(PRIZE_DIR) if f.startswith("Prize_")])
last_6_prize = prize_files[-6:]  # last 6 periods
print(f"Loading prize data for 6 periods: {[f.replace('Prize_','').replace('.csv','') for f in last_6_prize]}")

all_prizes = []
for pf in last_6_prize:
    df = pd.read_csv(os.path.join(PRIZE_DIR, pf))
    all_prizes.append(df)
prizes = pd.concat(all_prizes, ignore_index=True)
print(f"  Total prize records: {len(prizes):,}")
print(f"  Unique winners: {prizes['userNo'].nunique():,}")

# Parse rounddate into item column name
def rounddate_to_item_col(rd):
    """Convert '2026-03-16' -> 'item2026_03_16'"""
    return "item" + rd.replace("-", "_")

prizes["item_col"] = prizes["rounddate"].apply(rounddate_to_item_col)

# Map prize type to Thai name and typical amount
PRIZE_THAI = {
    "FIRST": "รางวัลที่ 1 (6M)",
    "NEARBY": "รางวัลข้างเคียงที่ 1 (100K)",
    "SECOND": "รางวัลที่ 2 (200K)",
    "THIRD": "รางวัลที่ 3 (80K)",
    "FOURTH": "รางวัลที่ 4 (40K)",
    "FIFTH": "รางวัลที่ 5 (20K)",
    "FIRST3DIGITS": "เลขหน้า 3 ตัว (4K)",
    "LAST3DIGITS": "เลขท้าย 3 ตัว (4K)",
    "LAST2DIGITS": "เลขท้าย 2 ตัว (2K)",
}

# ============================================================
# 2. Load latest churn file to get purchase history + demographics
# ============================================================
churn_files = sorted([
    f for f in os.listdir(CHURN_DIR)
    if f.startswith("Churn_20") and not f.endswith("_old_validation.csv") and "_Check" not in f
])
latest_churn = churn_files[-1]
print(f"\nLoading churn data: {latest_churn}")
churn_df = pd.read_csv(os.path.join(CHURN_DIR, latest_churn))
print(f"  Customers: {len(churn_df):,}")

item_cols = [c for c in churn_df.columns if c.startswith("item")]
print(f"  Item columns: {len(item_cols)} (from {item_cols[0]} to {item_cols[-1]})")

# Build ordered list of period dates from item columns
# item2023_07_01 -> '2023-07-01'
period_dates = []
for ic in item_cols:
    parts = ic.replace("item", "").split("_")
    period_dates.append(f"{parts[0]}-{parts[1]}-{parts[2]}")

# ============================================================
# 3. Load risk scores
# ============================================================
risk_file = os.path.join(PRED_DIR, "Churn_RiskScore_2026_0401.csv")
if os.path.exists(risk_file):
    risk_df = pd.read_csv(risk_file)
    risk_map = dict(zip(risk_df["userNo"], risk_df["risk_level"]))
    score_map = dict(zip(risk_df["userNo"], risk_df["churn_score"]))
    print(f"Loaded risk scores: {len(risk_df):,} customers")
else:
    risk_map = {}
    score_map = {}
    print("WARNING: No risk score file found")

# ============================================================
# 4. For each winner, extract before/after purchase behavior
# ============================================================
print("\nBuilding customer-level winning effect table...")

# Aggregate prizes per customer per period (take max prize if multiple)
winner_records = prizes.groupby(["userNo", "rounddate", "item_col"]).agg(
    prize_type=("type", "first"),  # highest prize type (first in original order which is sorted by amount)
    prize_amount=("total_prize", "sum"),
    prize_count=("count_prize", "sum"),
).reset_index()

# For customers with multiple prize types in same period, keep the highest amount
winner_records = winner_records.sort_values("prize_amount", ascending=False).drop_duplicates(
    subset=["userNo", "rounddate"], keep="first"
).reset_index(drop=True)

print(f"  Winner-period records: {len(winner_records):,}")

# Merge with churn data for demographics + purchase history
merged = winner_records.merge(churn_df, on="userNo", how="inner")
print(f"  Matched to churn data: {len(merged):,} (some winners may not be in latest file)")

# For each record, extract tickets in periods around the win
results = []
for _, row in merged.iterrows():
    user = row["userNo"]
    win_item_col = row["item_col"]

    if win_item_col not in item_cols:
        continue

    win_idx = item_cols.index(win_item_col)

    # Extract before/after tickets
    tickets = {}
    for offset in range(-3, 4):  # -3, -2, -1, 0, +1, +2, +3
        idx = win_idx + offset
        if 0 <= idx < len(item_cols):
            val = row[item_cols[idx]]
            tickets[offset] = float(val) if pd.notna(val) else np.nan
        else:
            tickets[offset] = np.nan

    # Compute averages
    before_vals = [tickets[o] for o in [-3, -2, -1] if not np.isnan(tickets.get(o, np.nan))]
    after_vals = [tickets[o] for o in [1, 2, 3] if not np.isnan(tickets.get(o, np.nan))]

    avg_before = np.mean(before_vals) if before_vals else np.nan
    avg_after = np.mean(after_vals) if after_vals else np.nan

    if avg_before and avg_before > 0 and not np.isnan(avg_after):
        change_pct = ((avg_after - avg_before) / avg_before) * 100
    else:
        change_pct = np.nan

    current_risk = risk_map.get(user, "Unknown")
    current_score = score_map.get(user, np.nan)

    results.append({
        "userNo": user,
        "gender": row.get("gender", ""),
        "age": row.get("age", np.nan),
        "province": row.get("province", ""),
        "prize_type": row["prize_type"],
        "prize_type_thai": PRIZE_THAI.get(row["prize_type"], row["prize_type"]),
        "prize_amount": row["prize_amount"],
        "prize_period": row["rounddate"],
        "prize_period_short": row["rounddate"][5:].replace("-", ""),
        "tickets_before_3": tickets.get(-3, np.nan),
        "tickets_before_2": tickets.get(-2, np.nan),
        "tickets_before_1": tickets.get(-1, np.nan),
        "tickets_win_period": tickets.get(0, np.nan),
        "tickets_after_1": tickets.get(1, np.nan),
        "tickets_after_2": tickets.get(2, np.nan),
        "tickets_after_3": tickets.get(3, np.nan),
        "avg_before": round(avg_before, 1) if not np.isnan(avg_before) else np.nan,
        "avg_after": round(avg_after, 1) if not np.isnan(avg_after) else np.nan,
        "change_pct": round(change_pct, 1) if not np.isnan(change_pct) else np.nan,
        "current_risk": current_risk,
        "current_score": current_score,
    })

effect_df = pd.DataFrame(results)
print(f"  Complete records: {len(effect_df):,}")

# ============================================================
# 5. Compute expected shift based on historical patterns
# ============================================================
# For each prize type, what's the typical change_pct?
# Use that to estimate expected new risk level for current winners

type_stats = effect_df.dropna(subset=["change_pct"]).groupby("prize_type").agg(
    median_change=("change_pct", "median"),
    mean_change=("change_pct", "mean"),
    pct_increased=("change_pct", lambda x: (x > 5).mean() * 100),
    pct_maintained=("change_pct", lambda x: ((x >= -5) & (x <= 5)).mean() * 100),
    pct_decreased=("change_pct", lambda x: (x < -5).mean() * 100),
    count=("change_pct", "count"),
).reset_index()

print(f"\n  Prize type patterns computed for {len(type_stats)} types")

# Map expected shift: if you're High and typical winner increases 50%,
# your expected new risk should be lower
RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
RISK_NAMES = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}

def estimate_shift(current_risk, prize_type, change_pct_median):
    """Estimate new risk level based on historical winning effect."""
    if current_risk not in RISK_ORDER:
        return current_risk, 0.0

    current_idx = RISK_ORDER[current_risk]

    # If winners typically increase buying, risk should go down
    if change_pct_median > 30:
        shift = -2  # big increase -> drop 2 levels
    elif change_pct_median > 10:
        shift = -1  # moderate increase -> drop 1 level
    elif change_pct_median > -10:
        shift = 0   # maintained -> no change
    else:
        shift = 1   # decreased -> risk goes up

    new_idx = max(0, min(3, current_idx + shift))
    return RISK_NAMES[new_idx], abs(shift)

# Build type->median_change map
type_median = dict(zip(type_stats["prize_type"], type_stats["median_change"]))

# Add expected shift to effect_df
expected_risks = []
shift_probs = []
for _, row in effect_df.iterrows():
    med = type_median.get(row["prize_type"], 0)
    new_risk, shift_mag = estimate_shift(row["current_risk"], row["prize_type"], med)
    expected_risks.append(new_risk)
    # Probability based on % of winners who increased
    ts = type_stats[type_stats["prize_type"] == row["prize_type"]]
    prob = ts["pct_increased"].values[0] if len(ts) > 0 else 50.0
    shift_probs.append(round(prob, 1))

effect_df["expected_new_risk"] = expected_risks
effect_df["shift_probability"] = shift_probs

# ============================================================
# 6. Save CSV
# ============================================================
csv_cols = [
    "userNo", "gender", "age", "province", "prize_type", "prize_type_thai",
    "prize_amount", "prize_period", "prize_period_short",
    "tickets_before_3", "tickets_before_2", "tickets_before_1",
    "tickets_win_period", "tickets_after_1", "tickets_after_2", "tickets_after_3",
    "avg_before", "avg_after", "change_pct",
    "current_risk", "current_score", "expected_new_risk", "shift_probability",
]
csv_path = os.path.join(PRED_DIR, "winning_effect_customers.csv")
effect_df[csv_cols].to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path} ({len(effect_df):,} rows)")

# ============================================================
# 7. Shift Up Targets (High/Critical who recently won)
# ============================================================
# Most recent prize period
most_recent_period = sorted(effect_df["prize_period"].unique())[-1]
print(f"\nMost recent prize period: {most_recent_period}")

# Get shift targets: High/Critical in most recent 2 periods
recent_2 = sorted(effect_df["prize_period"].unique())[-2:]
shift_targets = effect_df[
    (effect_df["prize_period"].isin(recent_2)) &
    (effect_df["current_risk"].isin(["High", "Critical"]))
].copy()

# Sort by prize_amount * risk priority
shift_targets["priority_score"] = shift_targets["prize_amount"] * shift_targets["current_risk"].map(
    {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
)
shift_targets = shift_targets.sort_values("priority_score", ascending=False).reset_index(drop=True)

# Estimate expected additional tickets per customer
shift_targets["expected_additional_tickets"] = shift_targets.apply(
    lambda r: round(r["avg_before"] * (type_median.get(r["prize_type"], 0) / 100), 1)
    if pd.notna(r["avg_before"]) and r["avg_before"] > 0 else 0,
    axis=1,
)

# Save JSON
json_records = []
for _, row in shift_targets.iterrows():
    rec = {
        "userNo": row["userNo"],
        "gender": row["gender"],
        "age": int(row["age"]) if pd.notna(row["age"]) else None,
        "province": row["province"],
        "prize_type": row["prize_type"],
        "prize_type_thai": row["prize_type_thai"],
        "prize_amount": int(row["prize_amount"]),
        "prize_period": row["prize_period"],
        "current_risk": row["current_risk"],
        "current_score": round(float(row["current_score"]), 1) if pd.notna(row["current_score"]) else None,
        "expected_new_risk": row["expected_new_risk"],
        "shift_probability": row["shift_probability"],
        "avg_before": row["avg_before"],
        "expected_additional_tickets": row["expected_additional_tickets"],
    }
    json_records.append(rec)

json_path = os.path.join(PRED_DIR, "winning_effect_shift_targets.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "description": "Shift Up Targets — ลูกค้า High/Critical ที่เพิ่งถูกรางวัล",
        "recent_periods": list(recent_2),
        "total_targets": len(json_records),
        "targets": json_records,
    }, f, ensure_ascii=False, indent=2)
print(f"Saved: {json_path} ({len(json_records):,} targets)")

# ============================================================
# 8. Console Report
# ============================================================
print()
print("=" * 60)
print("Winning Effect — พฤติกรรมการซื้อหลังถูกรางวัล")
print("=" * 60)

# Overall stats
has_change = effect_df.dropna(subset=["change_pct"])
n_total = len(has_change)
n_increased = (has_change["change_pct"] > 5).sum()
n_maintained = ((has_change["change_pct"] >= -5) & (has_change["change_pct"] <= 5)).sum()
n_decreased = (has_change["change_pct"] < -5).sum()

print(f"\n  ภาพรวม: ลูกค้าถูกรางวัล 6 งวดล่าสุด")
print(f"   จำนวนทั้งหมด: {len(effect_df):,} ราย (มีข้อมูล before/after: {n_total:,} ราย)")
print(f"   ซื้อเพิ่มขึ้น: {n_increased/n_total*100:.1f}% ({n_increased:,} ราย)")
print(f"   ซื้อเท่าเดิม: {n_maintained/n_total*100:.1f}% ({n_maintained:,} ราย)")
print(f"   ซื้อลดลง: {n_decreased/n_total*100:.1f}% ({n_decreased:,} ราย)")

print(f"\n  แยกตามประเภทรางวัล:")
for _, ts in type_stats.sort_values("mean_change", ascending=False).iterrows():
    ptype = ts["prize_type"]
    thai = PRIZE_THAI.get(ptype, ptype)
    sub = has_change[has_change["prize_type"] == ptype]
    avg_before_grp = sub["avg_before"].mean()
    avg_after_grp = sub["avg_after"].mean()
    print(f"   {thai}: {int(ts['count']):,} ราย — ซื้อเพิ่ม +{ts['mean_change']:.0f}% "
          f"เฉลี่ย {avg_before_grp:.1f} -> {avg_after_grp:.1f} ใบ "
          f"(เพิ่ม {ts['pct_increased']:.0f}% / เท่าเดิม {ts['pct_maintained']:.0f}% / ลด {ts['pct_decreased']:.0f}%)")

print(f"\n  Shift Up Targets (High/Critical ที่เพิ่งถูก):")
print(f"   จำนวน: {len(shift_targets):,} ราย")
if len(shift_targets) > 0:
    n_expected_shift = (shift_targets["expected_new_risk"] != shift_targets["current_risk"]).sum()
    total_addl = shift_targets["expected_additional_tickets"].sum()
    print(f"   คาดว่า shift ลง: {n_expected_shift/len(shift_targets)*100:.0f}% ({n_expected_shift:,} ราย)")
    print(f"   Expected เพิ่ม: +{total_addl:,.0f} ใบ/งวด")

# Show example customers
print(f"\n  ตัวอย่างลูกค้า:")
examples = effect_df.dropna(subset=["change_pct", "avg_before", "avg_after"])
examples = examples[examples["avg_before"] > 0].copy()
# Pick diverse examples: different prize types, different change magnitudes
example_list = []
for ptype in ["FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "LAST2DIGITS", "LAST3DIGITS", "FIRST3DIGITS", "NEARBY"]:
    sub = examples[examples["prize_type"] == ptype]
    if len(sub) > 0:
        # Pick one with notable change
        sub_sorted = sub.sort_values("change_pct", ascending=False)
        example_list.append(sub_sorted.iloc[len(sub_sorted)//4])  # 75th percentile
        if len(example_list) >= 6:
            break

for ex in example_list[:6]:
    gender_th = "ชาย" if ex["gender"] == "MALE" else "หญิง" if ex["gender"] == "FEMALE" else ex["gender"]
    age = int(ex["age"]) if pd.notna(ex["age"]) else "?"
    prov = ex["province"]
    thai = PRIZE_THAI.get(ex["prize_type"], ex["prize_type"])
    period_short = ex["prize_period_short"]
    b3 = f"{int(ex['tickets_before_3'])}" if pd.notna(ex["tickets_before_3"]) else "-"
    b2 = f"{int(ex['tickets_before_2'])}" if pd.notna(ex["tickets_before_2"]) else "-"
    b1 = f"{int(ex['tickets_before_1'])}" if pd.notna(ex["tickets_before_1"]) else "-"
    a1 = f"{int(ex['tickets_after_1'])}" if pd.notna(ex["tickets_after_1"]) else "-"
    a2 = f"{int(ex['tickets_after_2'])}" if pd.notna(ex["tickets_after_2"]) else "-"
    a3 = f"{int(ex['tickets_after_3'])}" if pd.notna(ex["tickets_after_3"]) else "-"

    print(f"\n   {ex['userNo']} | {gender_th} {age} ปี {prov} | ถูก{thai} งวด {period_short}")
    print(f"   ก่อนถูก: {b3}, {b2}, {b1} ใบ (avg {ex['avg_before']:.1f}) "
          f"-> หลังถูก: {a1}, {a2}, {a3} ใบ (avg {ex['avg_after']:.1f}) = {ex['change_pct']:+.0f}%")
    print(f"   Risk: {ex['current_risk']} -> คาดว่า {ex['expected_new_risk']} (prob {ex['shift_probability']:.1f}%)")

# ============================================================
# 9. Charts
# ============================================================
print(f"\n\nGenerating charts...")

fig, axes = plt.subplots(3, 2, figsize=(20, 28))
fig.suptitle("Winning Effect — พฤติกรรมการซื้อหลังถูกรางวัล (รายลูกค้า)", fontsize=20, fontweight="bold", y=0.98)

# Color map for prize types
PRIZE_COLORS = {
    "FIRST": "#e74c3c",
    "SECOND": "#e67e22",
    "NEARBY": "#f39c12",
    "THIRD": "#2ecc71",
    "FOURTH": "#1abc9c",
    "FIFTH": "#3498db",
    "FIRST3DIGITS": "#9b59b6",
    "LAST3DIGITS": "#8e44ad",
    "LAST2DIGITS": "#95a5a6",
}

PRIZE_SHORT = {
    "FIRST": "ที่ 1",
    "SECOND": "ที่ 2",
    "NEARBY": "ข้างเคียง",
    "THIRD": "ที่ 3",
    "FOURTH": "ที่ 4",
    "FIFTH": "ที่ 5",
    "FIRST3DIGITS": "หน้า 3 ตัว",
    "LAST3DIGITS": "ท้าย 3 ตัว",
    "LAST2DIGITS": "ท้าย 2 ตัว",
}

# ----- Panel 1: Scatter before vs after -----
ax = axes[0, 0]
scatter_df = effect_df.dropna(subset=["avg_before", "avg_after"])
scatter_df = scatter_df[(scatter_df["avg_before"] > 0) & (scatter_df["avg_after"] >= 0)]

# Cap for visualization (99th percentile)
cap_b = scatter_df["avg_before"].quantile(0.98)
cap_a = scatter_df["avg_after"].quantile(0.98)
cap = max(cap_b, cap_a)

for ptype in PRIZE_COLORS:
    sub = scatter_df[scatter_df["prize_type"] == ptype]
    if len(sub) > 0:
        ax.scatter(
            sub["avg_before"].clip(upper=cap),
            sub["avg_after"].clip(upper=cap),
            alpha=0.3, s=15,
            color=PRIZE_COLORS[ptype],
            label=f"{PRIZE_SHORT.get(ptype, ptype)} ({len(sub):,})"
        )

ax.plot([0, cap], [0, cap], "k--", alpha=0.5, linewidth=1, label="ไม่เปลี่ยน")
ax.set_xlabel("Avg ใบ/งวด ก่อนถูกรางวัล", fontsize=12)
ax.set_ylabel("Avg ใบ/งวด หลังถูกรางวัล", fontsize=12)
ax.set_title("Panel 1: พฤติกรรมก่อน-หลังถูกรางวัล (รายลูกค้า)", fontsize=14, fontweight="bold")
ax.legend(fontsize=8, loc="upper left", ncol=2)

# Annotate % above line
above = (scatter_df["avg_after"] > scatter_df["avg_before"]).sum()
pct_above = above / len(scatter_df) * 100
ax.annotate(
    f"{pct_above:.1f}% ซื้อเพิ่มขึ้น\n(เหนือเส้นทแยง)",
    xy=(cap * 0.6, cap * 0.3), fontsize=11, fontweight="bold",
    color="#27ae60",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", alpha=0.8),
)

# ----- Panel 2: Line chart — avg trajectory by prize type -----
ax = axes[0, 1]

offsets = [-3, -2, -1, 0, 1, 2, 3]
offset_labels = ["-3", "-2", "-1", "ถูกรางวัล", "+1", "+2", "+3"]

for ptype in ["FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "LAST2DIGITS", "LAST3DIGITS", "FIRST3DIGITS"]:
    sub = effect_df[effect_df["prize_type"] == ptype]
    if len(sub) < 10:
        continue

    trajectory = []
    for off in offsets:
        if off == 0:
            col = "tickets_win_period"
        elif off < 0:
            col = f"tickets_before_{abs(off)}"
        else:
            col = f"tickets_after_{off}"

        vals = sub[col].dropna()
        trajectory.append(vals.mean() if len(vals) > 0 else np.nan)

    ax.plot(
        range(7), trajectory,
        marker="o", linewidth=2, markersize=6,
        color=PRIZE_COLORS.get(ptype, "#333"),
        label=f"{PRIZE_SHORT.get(ptype, ptype)}",
        alpha=0.8,
    )

ax.set_xticks(range(7))
ax.set_xticklabels(offset_labels, fontsize=10)
ax.axvline(x=3, color="red", linestyle="--", alpha=0.4, label="งวดถูกรางวัล")
ax.set_xlabel("งวดสัมพัทธ์ (เทียบกับงวดที่ถูกรางวัล)", fontsize=12)
ax.set_ylabel("Avg ใบ/งวด", fontsize=12)
ax.set_title("Panel 2: ยอดซื้อรายงวด แยกตามประเภทรางวัล", fontsize=14, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)

# ----- Panel 3: Stacked bar — % increased/maintained/decreased -----
ax = axes[1, 0]

bar_data = type_stats.sort_values("count", ascending=True).copy()
bar_data["prize_short"] = bar_data["prize_type"].map(PRIZE_SHORT)

y_pos = range(len(bar_data))
bars_inc = bar_data["pct_increased"].values
bars_mnt = bar_data["pct_maintained"].values
bars_dec = bar_data["pct_decreased"].values

ax.barh(y_pos, bars_inc, color="#27ae60", label="ซื้อเพิ่ม (>+5%)")
ax.barh(y_pos, bars_mnt, left=bars_inc, color="#f1c40f", label="เท่าเดิม (+-5%)")
ax.barh(y_pos, bars_dec, left=bars_inc + bars_mnt, color="#e74c3c", label="ซื้อลดลง (<-5%)")

ax.set_yticks(y_pos)
ax.set_yticklabels([f"{bar_data.iloc[i]['prize_short']} (n={int(bar_data.iloc[i]['count']):,})" for i in range(len(bar_data))], fontsize=10)
ax.set_xlabel("% ของลูกค้า", fontsize=12)
ax.set_title("Panel 3: % ลูกค้าที่ซื้อเพิ่มขึ้นหลังถูกรางวัล", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")

# Annotate percentages on bars
for i in range(len(bar_data)):
    ax.text(bars_inc[i] / 2, i, f"{bars_inc[i]:.0f}%", ha="center", va="center", fontsize=9, fontweight="bold", color="white")

ax.set_xlim(0, 100)

# ----- Panel 4: Shift Up targets by prize type -----
ax = axes[1, 1]

if len(shift_targets) > 0:
    st_counts = shift_targets.groupby("prize_type").agg(
        count=("userNo", "count"),
        total_addl=("expected_additional_tickets", "sum"),
    ).reset_index()
    st_counts["prize_short"] = st_counts["prize_type"].map(PRIZE_SHORT)
    st_counts = st_counts.sort_values("count", ascending=True)

    colors = [PRIZE_COLORS.get(pt, "#95a5a6") for pt in st_counts["prize_type"]]
    bars = ax.barh(range(len(st_counts)), st_counts["count"].values, color=colors)
    ax.set_yticks(range(len(st_counts)))
    ax.set_yticklabels(st_counts["prize_short"].values, fontsize=10)
    ax.set_xlabel("จำนวนลูกค้า (ราย)", fontsize=12)

    for i, (cnt, addl) in enumerate(zip(st_counts["count"], st_counts["total_addl"])):
        ax.text(cnt + 0.5, i, f"{cnt:,} ราย (+{addl:,.0f} ใบ)", va="center", fontsize=9)

    total_targets = len(shift_targets)
    total_tickets = shift_targets["expected_additional_tickets"].sum()
    ax.annotate(
        f"รวม: {total_targets:,} ราย\nExpected เพิ่ม +{total_tickets:,.0f} ใบ/งวด",
        xy=(0.95, 0.05), xycoords="axes fraction", fontsize=11, fontweight="bold",
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdebd0", alpha=0.9),
    )
else:
    ax.text(0.5, 0.5, "ไม่มีลูกค้า High/Critical ที่ถูกรางวัล", ha="center", va="center", fontsize=14)

ax.set_title("Panel 4: Shift Up Opportunities — ลูกค้า High/Critical ที่เพิ่งถูกรางวัล", fontsize=14, fontweight="bold")

# ----- Panel 5: 10 individual customer mini-charts -----
ax = axes[2, 0]
ax.set_axis_off()

# Pick 10 diverse examples
examples_for_chart = effect_df.dropna(subset=["change_pct", "avg_before", "avg_after"])
examples_for_chart = examples_for_chart[examples_for_chart["avg_before"] > 0].copy()

# Sample diverse: different prize types, different magnitudes
sample_list = []
for ptype in ["FIRST", "SECOND", "THIRD", "FIFTH", "LAST2DIGITS", "LAST3DIGITS", "FIRST3DIGITS", "FOURTH", "NEARBY"]:
    sub = examples_for_chart[examples_for_chart["prize_type"] == ptype].sort_values("change_pct", ascending=False)
    if len(sub) > 0:
        # Pick one from top quartile
        idx = max(0, len(sub) // 4 - 1)
        sample_list.append(sub.iloc[idx])
        if len(sample_list) >= 10:
            break

# If less than 10, fill from most common types
if len(sample_list) < 10:
    remaining = examples_for_chart[~examples_for_chart["userNo"].isin([s["userNo"] for s in sample_list])]
    for _, r in remaining.head(10 - len(sample_list)).iterrows():
        sample_list.append(r)

# Create mini-charts using inset axes
n_examples = min(10, len(sample_list))
n_cols_mini = 5
n_rows_mini = 2

for i, ex in enumerate(sample_list[:n_examples]):
    row_mini = i // n_cols_mini
    col_mini = i % n_cols_mini

    # Position within the panel
    left = 0.02 + col_mini * 0.195
    bottom = 0.55 - row_mini * 0.5
    width = 0.17
    height = 0.38

    inset = ax.inset_axes([left, bottom, width, height])

    vals = [
        ex["tickets_before_3"], ex["tickets_before_2"], ex["tickets_before_1"],
        ex["tickets_win_period"],
        ex["tickets_after_1"], ex["tickets_after_2"], ex["tickets_after_3"],
    ]
    x = list(range(7))
    valid_x = [xi for xi, v in zip(x, vals) if pd.notna(v)]
    valid_v = [v for v in vals if pd.notna(v)]

    color = PRIZE_COLORS.get(ex["prize_type"], "#333")
    inset.plot(valid_x, valid_v, marker="o", markersize=4, linewidth=1.5, color=color)
    inset.axvline(x=3, color="red", linestyle="--", alpha=0.3, linewidth=0.8)

    # Fill before/after
    if len(valid_x) > 1:
        before_x = [xi for xi in valid_x if xi < 3]
        before_v = [valid_v[valid_x.index(xi)] for xi in before_x]
        after_x = [xi for xi in valid_x if xi > 3]
        after_v = [valid_v[valid_x.index(xi)] for xi in after_x]
        if before_x:
            inset.fill_between(before_x, before_v, alpha=0.15, color="#3498db")
        if after_x:
            inset.fill_between(after_x, after_v, alpha=0.15, color="#e74c3c")

    pshort = PRIZE_SHORT.get(ex["prize_type"], ex["prize_type"])
    chg = ex["change_pct"]
    chg_str = f"+{chg:.0f}%" if chg > 0 else f"{chg:.0f}%"
    inset.set_title(f"{ex['userNo'][:6]}.. {pshort}\n{chg_str}", fontsize=7, fontweight="bold")
    inset.set_xticks([0, 3, 6])
    inset.set_xticklabels(["-3", "W", "+3"], fontsize=6)
    inset.tick_params(axis="y", labelsize=6)
    inset.grid(alpha=0.2)

ax.set_title("Panel 5: ตัวอย่างลูกค้า — Before vs After (แต่ละราย)", fontsize=14, fontweight="bold", pad=20)

# ----- Panel 6: Expected Shift by Current Risk Level -----
ax = axes[2, 1]

# For recent winners, show current risk -> expected risk migration
recent_winners = effect_df[effect_df["prize_period"].isin(recent_2)]
recent_with_risk = recent_winners[recent_winners["current_risk"].isin(["Low", "Medium", "High", "Critical"])]

if len(recent_with_risk) > 0:
    # Create migration matrix
    risk_levels = ["Low", "Medium", "High", "Critical"]
    migration = pd.crosstab(
        recent_with_risk["current_risk"],
        recent_with_risk["expected_new_risk"],
    ).reindex(index=risk_levels, columns=risk_levels, fill_value=0)

    # Plot as grouped bar
    x_pos = np.arange(len(risk_levels))
    width = 0.18
    risk_colors = {"Low": "#27ae60", "Medium": "#f1c40f", "High": "#e67e22", "Critical": "#e74c3c"}

    for j, target_risk in enumerate(risk_levels):
        if target_risk in migration.columns:
            vals = migration[target_risk].values
        else:
            vals = [0] * len(risk_levels)
        ax.bar(
            x_pos + j * width - 1.5 * width,
            vals, width,
            color=risk_colors[target_risk],
            label=f"-> {target_risk}",
            alpha=0.8,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"ปัจจุบัน: {r}" for r in risk_levels], fontsize=10)
    ax.set_ylabel("จำนวนลูกค้า (ราย)", fontsize=12)
    ax.legend(fontsize=9, title="คาดว่าจะเปลี่ยนเป็น", loc="upper left")

    # Annotate total shift-down
    shift_down = (recent_with_risk["expected_new_risk"] != recent_with_risk["current_risk"]).sum()
    ax.annotate(
        f"คาดว่า shift ลง: {shift_down:,} ราย ({shift_down/len(recent_with_risk)*100:.1f}%)",
        xy=(0.95, 0.95), xycoords="axes fraction", fontsize=11, fontweight="bold",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eaf2f8", alpha=0.9),
    )
else:
    ax.text(0.5, 0.5, "ไม่มีข้อมูลเพียงพอ", ha="center", va="center", fontsize=14)

ax.set_title("Panel 6: Expected Shift by Current Risk Level (ลูกค้าถูกรางวัลล่าสุด)", fontsize=14, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.96])
chart_path = os.path.join(CHART_DIR, "winning_effect_customers.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved: {chart_path}")

# ============================================================
# Final Summary
# ============================================================
print()
print("=" * 60)
print("Summary")
print("=" * 60)
print(f"  CSV:   {csv_path} ({len(effect_df):,} rows)")
print(f"  JSON:  {json_path} ({len(json_records):,} shift-up targets)")
print(f"  Chart: {chart_path}")
print(f"\nDone!")
