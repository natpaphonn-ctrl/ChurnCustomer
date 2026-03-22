"""
Generate TX buying behavior and optimal contact timing for 0401 scoring period.
Input: Transaction_Order_2026_0316.csv
Output: data/predictions/persona360_txbehavior.json
"""

import pandas as pd
import numpy as np
import json
import os
from collections import Counter

# Config
TX_FILE = "data/transaction/Transaction_Order_2026_0316.csv"
OUT_FILE = "data/predictions/persona360_txbehavior.json"
DRAW_DATE = pd.Timestamp("2026-03-16")
# Sales window: March 5 to March 16 (12 days)
WINDOW_START = pd.Timestamp("2026-03-05")

print("Loading TX data...")
df = pd.read_csv(TX_FILE, dtype={"userNo": str, "status": str, "totalItem": float})
df["createdAt_bkk"] = pd.to_datetime(df["createdAt_bkk"], utc=True).dt.tz_localize(None)
print(f"  Loaded {len(df):,} rows, {df['userNo'].nunique():,} unique customers")

# Calculate day-of-window (1-based: day 1 = first sales day, day 12 = draw day)
df["order_day"] = (df["createdAt_bkk"].dt.normalize() - WINDOW_START).dt.days + 1
# Clamp to valid range (some edge cases)
df["order_day"] = df["order_day"].clip(lower=1)

# Last 2 days of window = rush period (day 11 and 12 for this period: Mar 15, Mar 16)
window_length = (DRAW_DATE - WINDOW_START).days + 1  # 12
rush_start = window_length - 1  # day 11+

df["is_rush"] = df["order_day"] >= rush_start
df["hour"] = df["createdAt_bkk"].dt.hour

# Split SUCCESS and all
success = df[df["status"] == "SUCCESS"].copy()
print(f"  SUCCESS orders: {len(success):,}")
print(f"  CANCEL orders: {len(df) - len(success):,}")

# --- Per-customer aggregation ---
print("Computing per-customer metrics...")

# Total orders (for cancel rate)
total_orders = df.groupby("userNo")["status"].count().rename("total")
cancel_orders = df[df["status"] == "CANCEL"].groupby("userNo")["status"].count().rename("cancels")

# SUCCESS-only metrics
g = success.groupby("userNo")

avg_tickets = g["totalItem"].mean().rename("at")
n_orders = g["totalItem"].count().rename("no")
max_single = g["totalItem"].max().rename("mx")

# Rush buyer: >50% of SUCCESS orders in last 2 days
rush_orders = success[success["is_rush"]].groupby("userNo")["status"].count().rename("rush_n")

# Usual buy day: mode of order_day (SUCCESS only)
usual_day = g["order_day"].agg(lambda x: x.mode().iloc[0]).rename("bd")

# First order day
first_day = g["order_day"].min().rename("fd")

# Days spread
days_spread = g["order_day"].nunique().rename("ds")

# Buy time preference: mode of time bucket
def time_bucket(hour):
    if hour < 12:
        return "เช้า"
    elif hour < 18:
        return "บ่าย"
    else:
        return "ค่ำ"

success["time_bucket"] = success["hour"].apply(time_bucket)
buy_time = success.groupby("userNo")["time_bucket"].agg(lambda x: x.mode().iloc[0]).rename("bt")

# Combine
result = pd.DataFrame(index=avg_tickets.index)
result["at"] = avg_tickets.round(1)
result["no"] = n_orders.astype(int)

# Cancel rate
result = result.join(cancel_orders, how="left")
result = result.join(total_orders, how="left")
result["cr"] = (result["cancels"].fillna(0) / result["total"]).round(2)
result.drop(columns=["cancels", "total"], inplace=True)

# Rush buyer
result = result.join(rush_orders, how="left")
result["rush_n"] = result["rush_n"].fillna(0)
result["rb"] = (result["rush_n"] / result["no"]) > 0.5
result.drop(columns=["rush_n"], inplace=True)

result["mx"] = max_single.astype(int)
result["bd"] = usual_day.astype(int)
result["bt"] = buy_time
result["fd"] = first_day.astype(int)
result["ds"] = days_spread.astype(int)

# Convert to dict
print("Building JSON output...")
out = {}
for user_no, row in result.iterrows():
    out[user_no] = {
        "at": row["at"],
        "no": int(row["no"]),
        "cr": row["cr"],
        "rb": bool(row["rb"]),
        "mx": int(row["mx"]),
        "bd": int(row["bd"]),
        "bt": row["bt"],
        "fd": int(row["fd"]),
        "ds": int(row["ds"]),
    }

# Save
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

file_size = os.path.getsize(OUT_FILE)
print(f"\nDone!")
print(f"  Customers: {len(out):,}")
print(f"  File: {OUT_FILE}")
print(f"  Size: {file_size / 1024 / 1024:.1f} MB ({file_size:,} bytes)")

# Quick stats
print(f"\n--- Quick Stats ---")
print(f"  Avg tickets/order: {result['at'].mean():.1f}")
print(f"  Avg orders/customer: {result['no'].mean():.1f}")
print(f"  Rush buyers: {result['rb'].sum():,} ({result['rb'].mean()*100:.1f}%)")
print(f"  Cancel rate (mean): {result['cr'].mean()*100:.1f}%")
print(f"  Time preference distribution:")
print(f"    {result['bt'].value_counts().to_dict()}")
