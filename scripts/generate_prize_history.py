"""
Generate prize history JSON for all customers with prize wins.
Processes all 30 Prize CSV files and outputs persona360_prize.json.

Usage: python3 scripts/generate_prize_history.py
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

# --- Config ---
PRIZE_DIR = Path("data/prize")
OUTPUT_PATH = Path("data/predictions/persona360_prize.json")

# Prize type hierarchy (highest value first)
PRIZE_RANK = {
    "FIRST": 0,
    "NEARBY": 1,
    "SECOND": 2,
    "THIRD": 3,
    "FOURTH": 4,
    "FIFTH": 5,
    "LAST3DIGITS": 6,
    "FIRST3DIGITS": 7,
    "LAST2DIGITS": 8,
}

# Big win threshold: FOURTH (40K) or higher
BIG_WIN_TYPES = {"FIRST", "NEARBY", "SECOND", "THIRD", "FOURTH"}


def parse_period_from_filename(fname: str) -> str:
    """Extract MMDD from Prize_YYYY_MMDD.csv filename."""
    m = re.search(r"Prize_(\d{4})_(\d{4})\.csv", fname)
    if m:
        return f"{m.group(1)}_{m.group(2)}"  # e.g. "2026_0316"
    return ""


def short_period(period: str) -> str:
    """Convert '2026_0316' -> '0316'."""
    return period.split("_")[1] if "_" in period else period


def main():
    # Discover and sort prize files chronologically
    prize_files = sorted(PRIZE_DIR.glob("Prize_*.csv"))
    print(f"Found {len(prize_files)} prize files")

    # Build ordered period list for periods_since_last_win calculation
    all_periods = []
    for f in prize_files:
        p = parse_period_from_filename(f.name)
        if p:
            all_periods.append(p)
    period_index = {p: i for i, p in enumerate(all_periods)}
    latest_period_idx = len(all_periods) - 1  # 0316 is the last

    # Accumulate per-customer data
    # For each customer: wins by period, prize types, amounts, ticket counts
    customer_wins = defaultdict(list)  # userNo -> [(period_idx, period, type, amount, tickets)]

    for fpath in prize_files:
        period = parse_period_from_filename(fpath.name)
        pidx = period_index[period]
        print(f"  Processing {fpath.name} (period {period})...")

        df = pd.read_csv(fpath)
        for _, row in df.iterrows():
            user = row["userNo"]
            ptype = row["type"]
            amount = int(row["total_prize"])
            tickets = int(row["count_prize"])
            customer_wins[user].append((pidx, period, ptype, amount, tickets))

    print(f"\nTotal customers with at least 1 win: {len(customer_wins)}")

    # Build output dict
    output = {}
    for user, wins in customer_wins.items():
        # Sort by period index ascending
        wins.sort(key=lambda x: x[0])

        # Unique periods where they won
        won_periods = sorted(set(w[0] for w in wins))
        total_wins = len(won_periods)

        # Total amount and tickets
        total_amount = sum(w[3] for w in wins)
        total_tickets = sum(w[4] for w in wins)

        # Last win period
        last_win_idx = won_periods[-1]
        last_win_period = all_periods[last_win_idx]
        periods_since = latest_period_idx - last_win_idx

        # LAST2DIGITS count (number of periods with LAST2DIGITS wins)
        l2_periods = set()
        for w in wins:
            if w[2] == "LAST2DIGITS":
                l2_periods.add(w[0])
        last2_wins = len(l2_periods)

        # Big win check
        big_win = any(w[2] in BIG_WIN_TYPES for w in wins)

        # Best prize type (lowest rank = best)
        best_rank = min(PRIZE_RANK.get(w[2], 99) for w in wins)
        best_prize = [k for k, v in PRIZE_RANK.items() if v == best_rank][0]

        # Prize history: last 5 wins (most recent first), aggregate by period+type
        # Take the most recent entries
        recent_wins = []
        for w in reversed(wins):
            recent_wins.append({
                "p": short_period(w[1]),
                "t": w[2],
                "a": w[3],
            })
            if len(recent_wins) >= 5:
                break

        output[user] = {
            "tw": total_wins,
            "ta": total_amount,
            "tt": total_tickets,
            "lw": short_period(last_win_period),
            "ps": periods_since,
            "l2": last2_wins,
            "bw": big_win,
            "bp": best_prize,
            "ph": recent_wins,
        }

    # Write compact JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"), ensure_ascii=False)

    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"Customers with prize history: {len(output):,}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")

    # Quick stats
    big_winners = sum(1 for v in output.values() if v["bw"])
    avg_amount = sum(v["ta"] for v in output.values()) / len(output)
    print(f"Big winners (>= FOURTH): {big_winners:,}")
    print(f"Average prize amount: {avg_amount:,.0f}")


if __name__ == "__main__":
    main()
