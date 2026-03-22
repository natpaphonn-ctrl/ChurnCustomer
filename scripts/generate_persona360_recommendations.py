"""
Generate personalized action recommendations (Persona 360) for each customer
based on their SHAP explanations.

Reads: Churn_RiskScore_Explained_2026_0401.csv + Churn_RiskScore_2026_0401.csv
Writes: data/predictions/persona360_recommendations.json
"""

import json
import pandas as pd
import numpy as np
import time
import os
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPLAINED_PATH = os.path.join(PROJECT_ROOT, "data/predictions/Churn_RiskScore_Explained_2026_0401.csv")
RISKSCORE_PATH = os.path.join(PROJECT_ROOT, "data/predictions/Churn_RiskScore_2026_0401.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/predictions/persona360_recommendations.json")

MAX_RECS = 3


def get_action_for_reason(feature: str, value: float, direction: str) -> Optional[str]:
    """Map a SHAP reason to an actionable Thai recommendation sentence.
    Only generates actions for features that INCREASE churn risk."""
    if direction != "เพิ่ม Churn":
        return None

    v = value

    if feature == "active_last_6" and v <= 3:
        return f"⚠️ ซื้อแค่ {int(v)} ใน 6 งวดล่าสุด → ส่ง promotion ก่อนปิดขายงวดหน้า"

    if feature == "active_last_3" and v <= 1:
        return "🔴 แทบไม่ซื้อ 3 งวดล่าสุด → ติดต่อด่วนภายใน 3 วัน"

    if feature == "ewm_recent" and v < 1:
        return "📉 กำลังซื้อลดลงเรื่อยๆ → ส่ง offer พิเศษดึงกลับ"

    if feature == "trend_slope" and v < 0:
        return "📉 แนวโน้มลดลง → ส่ง LINE แจ้งเตือนก่อนเปิดขาย"

    if feature == "avg_gap" and v > 3:
        return f"⏰ เว้นช่วงซื้อเฉลี่ย {v:.1f} งวด → ส่ง reminder ทุกงวด"

    if feature == "max_gap" and v > 5:
        return f"⏰ เคยหายไปนานสุด {int(v)} งวด → ส่ง welcome back offer"

    if feature == "current_zero_streak" and v > 2:
        return f"🚨 ไม่ซื้อติดต่อกัน {int(v)} งวด → ส่ง promotion พิเศษทันที"

    if feature == "items_last_3":
        return "📦 ซื้อน้อยลง → แนะนำแพ็คสุดคุ้ม"

    if feature == "n_reactivations" and v >= 1:
        return f"🔄 หยุด-กลับมาซื้อ {int(v)} ครั้ง → พฤติกรรมไม่แน่นอน ต้องรักษาต่อเนื่อง"

    if feature == "purchase_frequency_ratio" and v < 0.5:
        return f"📊 ซื้อแค่ {v:.0%} ของงวดทั้งหมด → ส่ง offer ทุกงวดเพื่อสร้างนิสัย"

    if feature == "total_active_rounds" and v >= 10:
        return f"⭐ เคยเป็นลูกค้าภักดี ({int(v)} งวด) → ส่ง thank you + exclusive offer"

    if feature == "coeff_of_variation":
        return "📊 ซื้อไม่สม่ำเสมอ → ส่ง subscription-like reminder ทุกงวด"

    # Extended mappings for other common features
    if feature == "items_last_1":
        return "📦 ซื้อน้อยงวดล่าสุด → ส่ง offer ดึงกลับ"

    if feature == "trend_acceleration":
        return "📉 การซื้อชะลอตัวเร็วขึ้น → ส่ง promotion เร่งด่วน"

    if feature == "late_momentum":
        return "📉 โมเมนตัมช่วงหลังอ่อนแรง → ส่ง offer กระตุ้นการซื้อ"

    if feature == "freq_shift_early_to_late":
        return "📉 ความถี่ซื้อลดลงจากช่วงแรก → ส่ง reminder สม่ำเสมอ"

    if feature == "vol_shift_early_to_late":
        return "📉 ปริมาณซื้อลดลงจากช่วงแรก → แนะนำแพ็คสุดคุ้ม"

    if feature == "gini_coefficient":
        return "📊 การซื้อกระจุกตัวไม่สม่ำเสมอ → ส่ง reminder ทุกงวด"

    if feature == "gap_regularity":
        return "⏰ ช่วงเว้นซื้อไม่สม่ำเสมอ → ส่ง reminder ก่อนเปิดขายทุกงวด"

    if feature == "purchase_entropy":
        return "📊 รูปแบบการซื้อไม่แน่นอน → ส่ง notification สม่ำเสมอ"

    if feature == "mid_dip_indicator":
        return "📉 มีช่วงหยุดซื้อกลางทาง → ส่ง offer ช่วงกลางงวด"

    if feature == "is_declining_3":
        return "📉 ซื้อลดลง 3 งวดติด → ส่ง promotion ก่อนหาย"

    if feature == "purchase_autocorr_lag1":
        return "📊 พฤติกรรมซื้อเปลี่ยนจากรอบก่อน → ติดตามใกล้ชิด"

    if feature == "top3_rounds_pct":
        return "📊 ยอดซื้อกระจุกอยู่ไม่กี่งวด → ส่ง offer กระจายการซื้อ"

    if feature == "purchase_amount_diversity":
        return "📊 ซื้อจำนวนเดิมซ้ำๆ → แนะนำแพ็คใหม่เพื่อเพิ่มความหลากหลาย"

    if feature == "spending_quartile":
        return "📦 กลุ่มซื้อน้อย → ส่ง offer เพิ่มจำนวน"

    if feature == "max_active_streak":
        return "⭐ เคยซื้อติดต่อกันหลายงวด → ส่ง offer รักษา streak"

    if feature == "rounds_since_last_purchase":
        return "⏰ ไม่ได้ซื้อมาหลายงวด → ส่ง welcome back offer ทันที"

    if feature == "tenure_rounds":
        return "📊 อยู่ในระบบมานาน → ส่ง loyalty reward"

    if feature == "time_to_first_purchase":
        return "⏰ ใช้เวลานานกว่าจะเริ่มซื้อ → ส่ง onboarding guide"

    if feature == "total_items":
        return "📦 ซื้อรวมน้อย → แนะนำแพ็คสุดคุ้ม"

    if feature == "ever_reactivated":
        return "🔄 เคยหยุดแล้วกลับมา → ส่ง retention offer ป้องกันหยุดอีก"

    if feature == "trend_r_squared":
        return "📊 แนวโน้มซื้อชัดเจนว่าลดลง → ส่ง promotion ทันที"

    if feature == "is_new":
        return "🆕 ลูกค้าใหม่ → ส่ง welcome series + แนะนำวิธีซื้อ"

    if feature == "is_mature":
        return "⭐ ลูกค้าอยู่มานาน → ส่ง exclusive loyalty offer"

    return None


def generate_recommendations_vectorized(df: pd.DataFrame) -> dict:
    """Generate recommendations for all customers using vectorized operations where possible."""
    results = {}

    scores = df["churn_score"].values
    risk_levels = df["risk_level"].values
    user_nos = df["userNo"].values

    # Pre-extract all reason columns into arrays for speed
    reason_features = []
    reason_values = []
    reason_directions = []
    for i in range(1, 6):
        reason_features.append(df[f"reason_{i}"].values)
        reason_values.append(df[f"reason_{i}_value"].values)
        reason_directions.append(df[f"reason_{i}_direction"].values)

    for idx in range(len(df)):
        recs = []
        score = scores[idx]
        risk_level = risk_levels[idx]

        # Iterate through reasons 1-5
        for i in range(5):
            if len(recs) >= MAX_RECS:
                break

            feat = reason_features[i][idx]
            val = reason_values[i][idx]
            direction = reason_directions[i][idx]

            if pd.isna(feat) or pd.isna(direction):
                continue

            action = get_action_for_reason(str(feat), float(val), str(direction))
            if action and action not in recs:
                recs.append(action)

        # Low risk: add positive reinforcement
        if score < 25 and risk_level == "Low" and len(recs) < MAX_RECS:
            msg = "✅ ลูกค้าภักดี — ส่ง thank you reward เพื่อรักษาความสัมพันธ์"
            if msg not in recs:
                recs.append(msg)

        # Ensure at least 2 recommendations (use different messages to avoid duplicates)
        while len(recs) < 2:
            if score >= 75:
                fallback = "🚨 ความเสี่ยง Churn สูงมาก → ติดต่อทันทีเพื่อรักษาลูกค้า"
            elif score >= 50:
                fallback = "⚠️ เริ่มมีสัญญาณ Churn → ส่ง promotion ดึงกลับ"
            elif score >= 25:
                fallback = "📋 ส่ง offer เบาๆ เพื่อรักษาความสัมพันธ์"
            else:
                fallback = "✅ ลูกค้าภักดี — ส่ง thank you reward เพื่อรักษาความสัมพันธ์"

            if fallback not in recs:
                recs.append(fallback)
            else:
                # Second fallback to avoid duplicate
                if score < 25:
                    alt = "🎯 รักษาพฤติกรรมดีเยี่ยม — ส่ง loyalty point สะสม"
                elif score < 50:
                    alt = "🔔 ส่ง LINE reminder ก่อนเปิดขายงวดหน้า"
                elif score < 75:
                    alt = "📱 โทรสอบถามความต้องการเพิ่มเติม"
                else:
                    alt = "📞 โทรติดตามลูกค้าด่วน — เสี่ยงสูญเสียลูกค้า"
                if alt not in recs:
                    recs.append(alt)
                break  # Prevent infinite loop

        results[str(user_nos[idx])] = {"recs": recs[:MAX_RECS]}

        if (idx + 1) % 100000 == 0:
            print(f"  Processed {idx + 1:,} / {len(df):,}")

    return results


def main():
    t0 = time.time()

    print("Loading Explained file...")
    df_exp = pd.read_csv(EXPLAINED_PATH)
    print(f"  Explained customers: {len(df_exp):,}")

    print("Loading RiskScore file for New Customers...")
    df_risk = pd.read_csv(RISKSCORE_PATH)
    new_customers = df_risk[df_risk["risk_level"] == "New Customer"]["userNo"].tolist()
    print(f"  New customers: {len(new_customers):,}")

    # Generate recommendations for explained customers
    print("Generating recommendations...")
    results = generate_recommendations_vectorized(df_exp)

    # Add New Customer recommendations
    print(f"Adding {len(new_customers):,} New Customer recommendations...")
    for user_no in new_customers:
        results[str(user_no)] = {
            "recs": ["🆕 ลูกค้าใหม่ — ส่ง welcome offer + แนะนำวิธีซื้อ"]
        }

    print(f"Total customers: {len(results):,}")

    # Save
    print(f"Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, separators=(",", ":"))

    file_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s — File size: {file_size:.1f} MB")

    # Summary stats
    rec_counts = {}
    for v in results.values():
        n = len(v["recs"])
        rec_counts[n] = rec_counts.get(n, 0) + 1

    print(f"\nRecommendation count distribution:")
    for n in sorted(rec_counts.keys()):
        count = rec_counts[n]
        print(f"  {n} recs: {count:,} customers ({count / len(results) * 100:.1f}%)")

    # Examples per risk level
    print("\n" + "=" * 60)
    print("EXAMPLES PER RISK LEVEL")
    print("=" * 60)

    for level in ["Low", "Medium", "High", "Critical"]:
        subset = df_exp[df_exp["risk_level"] == level]
        if len(subset) == 0:
            continue
        sample = subset.sample(2, random_state=42)
        print(f"\n--- {level} ---")
        for _, row in sample.iterrows():
            user = str(row["userNo"])
            score = row["churn_score"]
            print(f"  {user} (score={score}):")
            for r in results[user]["recs"]:
                print(f"    {r}")

    # New customer example
    if new_customers:
        print(f"\n--- New Customer ---")
        user = str(new_customers[0])
        print(f"  {user}:")
        for r in results[user]["recs"]:
            print(f"    {r}")


if __name__ == "__main__":
    main()
