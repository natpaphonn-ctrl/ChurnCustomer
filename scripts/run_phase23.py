#!/usr/bin/env python3
"""
Phase 23: A/B Testing Framework
================================
Designs and backtests an A/B testing framework for measuring campaign effectiveness.

- ABTestDesigner: assigns customers to control/treatment groups
- ABTestAnalyzer: analyzes results with chi-squared tests and confidence intervals
- Backtest on historical data to verify no pre-existing bias
- Sample size calculation for 80% power

Usage:
    python3 run_phase23.py
"""

import os
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_TENURE = 3
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTROL_FRACTION = 0.10  # 10% control per risk level

RISK_LEVELS_ORDER = ["Low", "Medium", "High", "Critical"]

TEST_FILES = [
    "Churn_2026_0201_0216.csv",
    "Churn_2026_0216_0301.csv",
]

sys.path.insert(0, os.path.join(BASE_DIR, "models"))
from feature_engineering import engineer_features_v2


# ---------------------------------------------------------------------------
# Helper: assign risk level
# ---------------------------------------------------------------------------
def assign_risk_level(score):
    """Map numeric score to risk level string."""
    if score < 0:
        return "New Customer"
    if score <= 25:
        return "Low"
    if score <= 50:
        return "Medium"
    if score <= 75:
        return "High"
    return "Critical"


# ---------------------------------------------------------------------------
# Data loading (same pattern as pipeline)
# ---------------------------------------------------------------------------
def load_and_prepare(csv_path):
    """Load CSV with target column, filter by tenure, return prepared data."""
    df = pd.read_csv(csv_path, low_memory=False)
    item_cols = [c for c in df.columns if c.startswith("item")]
    feat_item_cols = item_cols[:-1]
    target_col = item_cols[-1]

    mat = df[item_cols].apply(pd.to_numeric, errors="coerce")
    mat_vals = mat.values.astype(float)
    reg = ~np.isnan(mat_vals)
    first_reg = np.where(reg.any(axis=1), reg.argmax(axis=1), len(item_cols))
    tenure = len(item_cols) - first_reg
    valid = tenure >= MIN_TENURE

    df_valid = df[valid].reset_index(drop=True)
    mat_valid = mat[valid].reset_index(drop=True)

    y_raw = mat_valid[target_col].values
    y = (np.isnan(y_raw) | (y_raw == 0)).astype(int)

    return df_valid, mat_valid, feat_item_cols, y


def get_risk_scores(df_valid, mat_valid, feat_item_cols, rf_model, feature_cols):
    """Compute risk scores using RF v3."""
    feats = engineer_features_v2(df_valid, mat_valid, feat_item_cols)
    feats = feats.fillna(0)

    for col in feature_cols:
        if col not in feats.columns:
            feats[col] = 0

    X = feats[feature_cols].fillna(0)
    probs = rf_model.predict_proba(X)[:, 1]
    scores = np.clip(np.round(probs * 100, 1), 0, 100)
    return scores


# ---------------------------------------------------------------------------
# ABTestDesigner
# ---------------------------------------------------------------------------
class ABTestDesigner:
    """Assigns customers to control/treatment groups for A/B testing.

    Rules:
        - Low risk: all 'baseline' (no campaign needed)
        - Medium, High, Critical: randomly assign 10% to 'control', rest to 'treatment'
        - New Customer: excluded (not assigned)
    """

    def __init__(self, control_fraction=CONTROL_FRACTION, random_state=42):
        self.control_fraction = control_fraction
        self.random_state = random_state

    def assign(self, df, score_col="churn_score", risk_col="risk_level"):
        """Add 'ab_group' column to dataframe.

        Args:
            df: DataFrame with score_col and risk_col columns
            score_col: column name for churn score
            risk_col: column name for risk level

        Returns:
            DataFrame with 'ab_group' column added
        """
        result = df.copy()
        result["ab_group"] = "excluded"

        rng = np.random.RandomState(self.random_state)

        # Low risk -> baseline
        low_mask = result[risk_col] == "Low"
        result.loc[low_mask, "ab_group"] = "baseline"

        # Medium, High, Critical -> control/treatment
        for level in ["Medium", "High", "Critical"]:
            level_mask = result[risk_col] == level
            level_idx = result.index[level_mask]
            n_level = len(level_idx)

            if n_level == 0:
                continue

            n_control = max(1, int(n_level * self.control_fraction))
            control_idx = rng.choice(level_idx, size=n_control, replace=False)

            result.loc[level_idx, "ab_group"] = "treatment"
            result.loc[control_idx, "ab_group"] = "control"

        return result

    def summary(self, df):
        """Print assignment summary."""
        print(f"\n  A/B Group Assignment Summary:")
        print(f"  {'Risk Level':<15} {'Treatment':>12} {'Control':>12} {'Baseline':>12} {'Excluded':>12}")
        print(f"  {'-' * 65}")
        for level in RISK_LEVELS_ORDER + ["New Customer"]:
            level_mask = df["risk_level"] == level
            if level_mask.sum() == 0:
                continue
            sub = df[level_mask]
            n_treat = (sub["ab_group"] == "treatment").sum()
            n_ctrl = (sub["ab_group"] == "control").sum()
            n_base = (sub["ab_group"] == "baseline").sum()
            n_excl = (sub["ab_group"] == "excluded").sum()
            print(f"  {level:<15} {n_treat:>12,} {n_ctrl:>12,} {n_base:>12,} {n_excl:>12,}")

        total = len(df)
        print(f"  {'-' * 65}")
        print(f"  {'Total':<15} {(df['ab_group']=='treatment').sum():>12,} "
              f"{(df['ab_group']=='control').sum():>12,} "
              f"{(df['ab_group']=='baseline').sum():>12,} "
              f"{(df['ab_group']=='excluded').sum():>12,}")


# ---------------------------------------------------------------------------
# ABTestAnalyzer
# ---------------------------------------------------------------------------
class ABTestAnalyzer:
    """Analyzes A/B test results with statistical tests."""

    def __init__(self):
        self.results = {}

    def analyze(self, df, actual_churn_col="actual_churn"):
        """Analyze churn rates by (risk_level, ab_group).

        Args:
            df: DataFrame with 'risk_level', 'ab_group', and actual_churn_col columns

        Returns:
            dict with per-level results
        """
        results = {}

        for level in ["Medium", "High", "Critical"]:
            level_mask = df["risk_level"] == level
            level_df = df[level_mask]

            treat_mask = level_df["ab_group"] == "treatment"
            ctrl_mask = level_df["ab_group"] == "control"

            n_treat = treat_mask.sum()
            n_ctrl = ctrl_mask.sum()

            if n_treat == 0 or n_ctrl == 0:
                results[level] = {"skip": True, "reason": "No data in one group"}
                continue

            treat_churn = level_df.loc[treat_mask, actual_churn_col].values
            ctrl_churn = level_df.loc[ctrl_mask, actual_churn_col].values

            treat_rate = treat_churn.mean()
            ctrl_rate = ctrl_churn.mean()

            # Effect size: relative reduction
            effect_size = (ctrl_rate - treat_rate) / ctrl_rate if ctrl_rate > 0 else 0.0

            # Chi-squared test
            contingency = np.array([
                [treat_churn.sum(), n_treat - treat_churn.sum()],
                [ctrl_churn.sum(), n_ctrl - ctrl_churn.sum()],
            ])

            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

            # 95% confidence intervals for proportions
            treat_ci = self._proportion_ci(treat_rate, n_treat)
            ctrl_ci = self._proportion_ci(ctrl_rate, n_ctrl)

            results[level] = {
                "skip": False,
                "n_treatment": int(n_treat),
                "n_control": int(n_ctrl),
                "treatment_churn_rate": float(treat_rate),
                "control_churn_rate": float(ctrl_rate),
                "effect_size": float(effect_size),
                "chi2": float(chi2),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "treatment_ci": treat_ci,
                "control_ci": ctrl_ci,
            }

        self.results = results
        return results

    def _proportion_ci(self, p, n, confidence=0.95):
        """Compute confidence interval for a proportion."""
        z = stats.norm.ppf((1 + confidence) / 2)
        se = np.sqrt(p * (1 - p) / max(n, 1))
        return (max(0, p - z * se), min(1, p + z * se))

    def print_results(self):
        """Print analysis results."""
        print(f"\n  A/B Test Analysis Results:")
        print(f"  {'Level':<12} {'Treatment':>12} {'Control':>12} {'Effect':>10} "
              f"{'Chi2':>8} {'p-value':>10} {'Sig?':>6}")
        print(f"  {'-' * 75}")

        for level in ["Medium", "High", "Critical"]:
            if level not in self.results:
                continue
            r = self.results[level]
            if r.get("skip"):
                print(f"  {level:<12} {'SKIPPED':>12} — {r.get('reason', '')}")
                continue

            sig = "YES" if r["significant"] else "No"
            print(f"  {level:<12} {r['treatment_churn_rate']:>11.1%} "
                  f"{r['control_churn_rate']:>11.1%} "
                  f"{r['effect_size']:>9.1%} "
                  f"{r['chi2']:>8.2f} {r['p_value']:>10.4f} {sig:>6}")
            print(f"  {'':>12} CI: [{r['treatment_ci'][0]:.3f}, {r['treatment_ci'][1]:.3f}]"
                  f"  CI: [{r['control_ci'][0]:.3f}, {r['control_ci'][1]:.3f}]")


# ---------------------------------------------------------------------------
# Sample size calculation
# ---------------------------------------------------------------------------
def compute_sample_size(baseline_rate, lift=0.05, alpha=0.05, power=0.80):
    """Compute minimum sample size per group for detecting a given lift.

    Formula: n = (Z_alpha + Z_beta)^2 * (p1*(1-p1) + p2*(1-p2)) / (p1-p2)^2

    Args:
        baseline_rate: baseline churn rate (control group)
        lift: absolute reduction in churn rate to detect
        alpha: significance level (default 0.05)
        power: statistical power (default 0.80)

    Returns:
        int: minimum sample size per group
    """
    p1 = baseline_rate  # control churn rate
    p2 = max(baseline_rate - lift, 0.001)  # treatment churn rate (lower)

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    numerator = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    denominator = (p1 - p2) ** 2

    if denominator == 0:
        return float("inf")

    n = numerator / denominator
    return int(np.ceil(n))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    print("=" * 70)
    print("  Phase 23: A/B Testing Framework")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ===================================================================
    # 1. Load RF v3 model
    # ===================================================================
    print("\n[1/5] Loading RF v3 model...")
    t0 = time.time()
    v3_path = os.path.join(BASE_DIR, "models", "churn_v3_extended.joblib")
    v3_artifact = joblib.load(v3_path)
    rf_model = v3_artifact["model"]
    feature_cols = v3_artifact["feature_cols"]
    print(f"  RF v3: {len(feature_cols)} features")
    print(f"  ({time.time() - t0:.1f}s)")

    # ===================================================================
    # 2. Load and score historical data
    # ===================================================================
    print(f"\n[2/5] Loading and scoring historical data ({len(TEST_FILES)} files)...")
    t0 = time.time()

    all_results = []
    observed_churn_rates = {}

    for fname in TEST_FILES:
        fpath = fname
        if not os.path.exists(fpath):
            fpath = os.path.join(BASE_DIR, fname)
        if not os.path.exists(fpath):
            fpath = os.path.join(BASE_DIR, "data", fname)
        if not os.path.exists(fpath):
            print(f"  SKIP: {fname} (not found)")
            continue

        print(f"\n  Processing {fname}...")
        df_valid, mat_valid, feat_item_cols, y_actual = load_and_prepare(fpath)
        print(f"    Rows: {len(df_valid):,}, Churn rate: {y_actual.mean():.1%}")

        # Get risk scores
        scores = get_risk_scores(df_valid, mat_valid, feat_item_cols, rf_model, feature_cols)
        risk_levels = [assign_risk_level(s) for s in scores]

        df_scored = pd.DataFrame({
            "userNo": df_valid["userNo"].values,
            "churn_score": scores,
            "risk_level": risk_levels,
            "actual_churn": y_actual,
            "source_file": fname,
        })

        # Compute observed churn rate per risk level
        for level in RISK_LEVELS_ORDER:
            mask = df_scored["risk_level"] == level
            if mask.sum() > 0:
                rate = df_scored.loc[mask, "actual_churn"].mean()
                if level not in observed_churn_rates:
                    observed_churn_rates[level] = []
                observed_churn_rates[level].append(rate)

        all_results.append(df_scored)

    if not all_results:
        print("  ERROR: No test files found. Exiting.")
        sys.exit(1)

    df_all = pd.concat(all_results, ignore_index=True)
    print(f"\n  Total: {len(df_all):,} customers from {len(all_results)} files")
    print(f"  Overall churn rate: {df_all['actual_churn'].mean():.1%}")
    print(f"  ({time.time() - t0:.1f}s)")

    # Average observed churn rates
    avg_churn_rates = {}
    for level in RISK_LEVELS_ORDER:
        if level in observed_churn_rates:
            avg_churn_rates[level] = np.mean(observed_churn_rates[level])

    print(f"\n  Observed churn rates (averaged across files):")
    for level in RISK_LEVELS_ORDER:
        if level in avg_churn_rates:
            print(f"    {level:<12}: {avg_churn_rates[level]:.1%}")

    # ===================================================================
    # 3. Backtest: A/B assignment + analysis
    # ===================================================================
    print(f"\n[3/5] Backtesting A/B framework...")
    t0 = time.time()

    designer = ABTestDesigner(control_fraction=CONTROL_FRACTION, random_state=42)
    df_ab = designer.assign(df_all, score_col="churn_score", risk_col="risk_level")
    designer.summary(df_ab)

    # Analyze: since no real campaign, control and treatment should have equal churn rates
    analyzer = ABTestAnalyzer()
    results = analyzer.analyze(df_ab, actual_churn_col="actual_churn")
    analyzer.print_results()

    # Verify: p-values should be > 0.05
    print(f"\n  Backtest Verification (no real campaign => groups should be equal):")
    all_pass = True
    for level in ["Medium", "High", "Critical"]:
        if level in results and not results[level].get("skip"):
            r = results[level]
            status = "PASS" if not r["significant"] else "FAIL"
            if r["significant"]:
                all_pass = False
            print(f"    {level:<12}: p={r['p_value']:.4f} -> {status} "
                  f"(expect p > 0.05)")
    if all_pass:
        print(f"    All groups PASS: no pre-existing bias detected")
    else:
        print(f"    WARNING: Some groups show unexpected significance")

    print(f"  ({time.time() - t0:.1f}s)")

    # ===================================================================
    # 4. Sample size calculation
    # ===================================================================
    print(f"\n[4/5] Sample size calculation (80% power, 5% lift detection)...")
    t0 = time.time()

    sample_sizes = {}
    print(f"\n  {'Risk Level':<12} {'Baseline Churn':>15} {'Target Churn':>14} {'Min N/group':>14}")
    print(f"  {'-' * 58}")
    for level in ["Medium", "High", "Critical"]:
        if level in avg_churn_rates:
            baseline = avg_churn_rates[level]
            n_required = compute_sample_size(baseline, lift=0.05, alpha=0.05, power=0.80)
            sample_sizes[level] = n_required
            target = max(baseline - 0.05, 0.001)
            print(f"  {level:<12} {baseline:>14.1%} {target:>13.1%} {n_required:>14,}")

    print(f"  ({time.time() - t0:.1f}s)")

    # ===================================================================
    # 5. Generate charts
    # ===================================================================
    print(f"\n[5/5] Generating charts...")
    t0 = time.time()
    os.makedirs(os.path.join(BASE_DIR, "charts"), exist_ok=True)
    chart_path = os.path.join(BASE_DIR, "charts", "phase23_ab_results.png")

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # --- Subplot 1: Backtest results — churn rate by group per risk level ---
    ax = axes[0]
    levels_plot = []
    treat_rates = []
    ctrl_rates = []
    for level in ["Medium", "High", "Critical"]:
        if level in results and not results[level].get("skip"):
            r = results[level]
            levels_plot.append(level)
            treat_rates.append(r["treatment_churn_rate"] * 100)
            ctrl_rates.append(r["control_churn_rate"] * 100)

    if levels_plot:
        x = np.arange(len(levels_plot))
        w = 0.35
        bars1 = ax.bar(x - w / 2, treat_rates, w, label="Treatment", color="#3498db",
                       edgecolor="white")
        bars2 = ax.bar(x + w / 2, ctrl_rates, w, label="Control", color="#e74c3c",
                       edgecolor="white")

        for bar, rate in zip(bars1, treat_rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{rate:.1f}%", ha="center", fontsize=9, fontweight="bold")
        for bar, rate in zip(bars2, ctrl_rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{rate:.1f}%", ha="center", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(levels_plot)

    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Backtest: Churn Rate by A/B Group\n(No campaign — expect equal rates)", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add p-values as annotations
    for i, level in enumerate(levels_plot):
        if level in results and not results[level].get("skip"):
            r = results[level]
            max_rate = max(treat_rates[i], ctrl_rates[i])
            ax.text(i, max_rate + 3, f"p={r['p_value']:.3f}",
                    ha="center", fontsize=9, style="italic",
                    color="green" if r["p_value"] > 0.05 else "red")

    # --- Subplot 2: Required sample sizes per risk level ---
    ax = axes[1]
    if sample_sizes:
        ss_levels = list(sample_sizes.keys())
        ss_values = [sample_sizes[l] for l in ss_levels]
        colors = ["#f39c12", "#e67e22", "#e74c3c"][:len(ss_levels)]
        bars = ax.bar(ss_levels, ss_values, color=colors, edgecolor="white")
        for bar, val in zip(bars, ss_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f"{val:,}", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("Min Sample Size (per group)")
        ax.set_title("Required Sample Size\n(80% power, detect 5% lift)", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    # --- Subplot 3: Thai-language summary table ---
    ax = axes[2]
    ax.axis("off")

    table_data = [
        ["ระดับความเสี่ยง", "อัตรา Churn\n(Treatment)", "อัตรา Churn\n(Control)",
         "p-value", "ผลลัพธ์"],
    ]
    for level in ["Medium", "High", "Critical"]:
        if level in results and not results[level].get("skip"):
            r = results[level]
            status = "ไม่มีนัยสำคัญ" if not r["significant"] else "มีนัยสำคัญ"
            table_data.append([
                level,
                f"{r['treatment_churn_rate']:.1%}",
                f"{r['control_churn_rate']:.1%}",
                f"{r['p_value']:.4f}",
                status,
            ])

    table_data.append(["", "", "", "", ""])
    table_data.append(["สรุป", "", "", "", ""])
    if all_pass:
        table_data.append(["", "ผ่านการทดสอบ", "", "", "ไม่พบ bias"])
    else:
        table_data.append(["", "ไม่ผ่าน", "", "", "พบ bias"])

    table_data.append(["", "", "", "", ""])
    table_data.append(["ขนาดตัวอย่างขั้นต่ำ", "", "", "", ""])
    for level in ["Medium", "High", "Critical"]:
        if level in sample_sizes:
            table_data.append([
                f"  {level}",
                f"{sample_sizes[level]:,} คน/กลุ่ม",
                "", "", "",
            ])

    table = ax.table(
        cellText=table_data,
        colWidths=[0.22, 0.22, 0.18, 0.16, 0.22],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header row
    for j in range(5):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("สรุปผล A/B Testing Framework", fontsize=12, fontweight="bold")

    plt.suptitle(
        f"Phase 23: A/B Testing Framework — Backtest Results\n"
        f"Data: {len(df_all):,} customers from {len(all_results)} periods  |  "
        f"Control fraction: {CONTROL_FRACTION:.0%}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {chart_path}")
    print(f"  ({time.time() - t0:.1f}s)")

    # ===================================================================
    # Thai-language summary
    # ===================================================================
    print("\n" + "=" * 70)
    print("  สรุปผล Phase 23: A/B Testing Framework")
    print("=" * 70)

    print(f"\n  ข้อมูลที่ใช้ทดสอบ:")
    print(f"    จำนวนไฟล์: {len(all_results)}")
    print(f"    จำนวนลูกค้ารวม: {len(df_all):,} คน")
    print(f"    อัตรา Churn รวม: {df_all['actual_churn'].mean():.1%}")

    print(f"\n  การออกแบบ A/B Test:")
    print(f"    สัดส่วน Control: {CONTROL_FRACTION:.0%} ของแต่ละระดับความเสี่ยง")
    print(f"    Low risk: ทั้งหมดเป็น baseline (ไม่ต้องทำแคมเปญ)")
    print(f"    New Customer: ไม่นับรวม")

    print(f"\n  ผลการ Backtest (ไม่มีแคมเปญจริง — ควรเท่ากัน):")
    print(f"  {'ระดับ':<12} {'Treatment':>12} {'Control':>12} {'p-value':>10} {'ผล':>15}")
    print(f"  {'-' * 63}")
    for level in ["Medium", "High", "Critical"]:
        if level in results and not results[level].get("skip"):
            r = results[level]
            status = "ไม่มีนัยสำคัญ" if not r["significant"] else "มีนัยสำคัญ!"
            print(f"  {level:<12} {r['treatment_churn_rate']:>11.1%} "
                  f"{r['control_churn_rate']:>11.1%} "
                  f"{r['p_value']:>10.4f} {status:>15}")

    if all_pass:
        print(f"\n  ผลสรุป: ผ่านการทดสอบ — ไม่พบ bias ระหว่างกลุ่ม")
        print(f"  พร้อมใช้งานจริงสำหรับวัดผลแคมเปญ")
    else:
        print(f"\n  ผลสรุป: พบ bias — ต้องตรวจสอบเพิ่มเติม")

    print(f"\n  ขนาดตัวอย่างขั้นต่ำ (80% power, detect 5% lift):")
    for level in ["Medium", "High", "Critical"]:
        if level in sample_sizes:
            print(f"    {level:<12}: {sample_sizes[level]:>10,} คน/กลุ่ม")

    print(f"\n  Output: {chart_path}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
