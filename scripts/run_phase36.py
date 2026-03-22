#!/usr/bin/env python3
"""
Phase 36: Validation Tracking Dashboard
========================================
Analyzes model performance across ALL historical validation periods using back-testing.

For each consecutive pair of churn files:
  1. Load churn file N as features (all item cols except last = features, last col = target period)
  2. Run v3 model (45 features) to get churn probabilities
  3. Validate against churn file N+1 (who actually bought in the next period)
  4. Compute AUC-ROC, AUC-PR, F1, Accuracy, churn rate per risk level

Also validates existing RiskScore files from data/predictions/.

Generates: charts/phase36_validation_tracking.png

Usage:
    python3 scripts/run_phase36.py
"""

import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_TENURE = 3
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHURN_DIR = os.path.join(BASE_DIR, "data", "churn")
PRED_DIR = os.path.join(BASE_DIR, "data", "predictions")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
MODELS_DIR = os.path.join(BASE_DIR, "models")

RISK_LEVELS = {
    "Low": (0, 25),
    "Medium": (26, 50),
    "High": (51, 75),
    "Critical": (76, 100),
}


def assign_risk_level(score):
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
# Find and sort churn files
# ---------------------------------------------------------------------------
def get_sorted_churn_files():
    """Get all churn files sorted chronologically."""
    pattern = re.compile(r"Churn_(\d{4})_(\d{4})_(\d{4})\.csv$")
    files = []
    for fname in os.listdir(CHURN_DIR):
        m = pattern.match(fname)
        if m and "Check" not in fname and "old" not in fname:
            year = int(m.group(1))
            mmdd_start = m.group(2)
            mmdd_end = m.group(3)
            # Build a sortable key: year + month + day
            month_start = int(mmdd_start[:2])
            day_start = int(mmdd_start[2:])
            # Handle year-crossing (e.g., 2025_1216_0102: start is Dec 2025)
            # Files from year 2025 with month >= 2 are in 2025; year 2026 files are later
            sort_key = (year, month_start, day_start)
            # Use short year suffix for disambiguation
            year_suffix = str(year)[-2:]
            files.append({
                "filename": fname,
                "filepath": os.path.join(CHURN_DIR, fname),
                "year": year,
                "mmdd_start": mmdd_start,
                "mmdd_end": mmdd_end,
                "sort_key": sort_key,
                "label": f"{mmdd_start}-{mmdd_end}'{year_suffix}",
            })
    files.sort(key=lambda x: x["sort_key"])
    return files


# ---------------------------------------------------------------------------
# Get existing RiskScore files
# ---------------------------------------------------------------------------
def get_riskscore_files():
    """Get all RiskScore files from predictions directory."""
    pattern = re.compile(r"Churn_RiskScore_(\d{4})_(\d{4})\.csv$")
    files = []
    for fname in os.listdir(PRED_DIR):
        m = pattern.match(fname)
        if m:
            year = m.group(1)
            mmdd = m.group(2)
            files.append({
                "filename": fname,
                "filepath": os.path.join(PRED_DIR, fname),
                "year": year,
                "mmdd": mmdd,
            })
    return files


# ---------------------------------------------------------------------------
# Back-test one pair of churn files
# ---------------------------------------------------------------------------
def backtest_one_period(churn_file_current, churn_file_next, rf_model, feature_cols, engineer_fn):
    """
    Back-test model on one period pair.

    - churn_file_current: features + target (last item col is target period)
    - churn_file_next: actual buyers in the next period (for validation)

    Returns dict with metrics or None on failure.
    """
    # Load current file
    df = pd.read_csv(churn_file_current["filepath"], low_memory=False)

    item_cols = [c for c in df.columns if c.startswith("item")]
    if len(item_cols) < 2:
        return None

    # Features = all item cols except the last one (which is target period)
    feat_item_cols = item_cols[:-1]
    target_col = item_cols[-1]

    mat = df[feat_item_cols].apply(pd.to_numeric, errors="coerce")
    mat_vals = mat.values.astype(float)
    reg = ~np.isnan(mat_vals)

    # Compute tenure
    first_reg = np.where(reg.any(axis=1), reg.argmax(axis=1), len(feat_item_cols))
    tenure = len(feat_item_cols) - first_reg
    valid = tenure >= MIN_TENURE

    n_eligible = valid.sum()
    if n_eligible == 0:
        return None

    df_elig = df[valid].copy()
    mat_elig = mat[valid].copy()

    # Compute features
    feats = engineer_fn(df_elig, mat_elig, feat_item_cols)
    feats = feats.fillna(0)

    for col in feature_cols:
        if col not in feats.columns:
            feats[col] = 0

    X = feats[feature_cols].fillna(0)
    rf_probs = rf_model.predict_proba(X)[:, 1]

    # Determine actual churn using NEXT churn file
    df_next = pd.read_csv(churn_file_next["filepath"], low_memory=False)
    actual_buyers = set(df_next["userNo"].values)

    y_true = (~df_elig["userNo"].isin(actual_buyers)).astype(int).values
    y_scores = rf_probs
    scores_100 = np.clip(np.round(rf_probs * 100, 1), 0, 100)
    y_pred = (scores_100 >= 50).astype(int)
    risk_levels = [assign_risk_level(s) for s in scores_100]

    # Metrics
    try:
        auc_roc = roc_auc_score(y_true, y_scores)
        auc_pr = average_precision_score(y_true, y_scores)
    except ValueError:
        return None

    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    churn_rate = y_true.mean()

    # Churn rate per risk level
    risk_churn = {}
    for level in ["Low", "Medium", "High", "Critical"]:
        mask = np.array([rl == level for rl in risk_levels])
        if mask.sum() > 0:
            risk_churn[level] = float(y_true[mask].mean())

    return {
        "period": churn_file_current["label"],
        "mmdd_end": churn_file_current["mmdd_end"],
        "year": churn_file_current["year"],
        "n_eligible": int(n_eligible),
        "n_total": len(df),
        "churn_rate": float(churn_rate),
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "f1": float(f1),
        "accuracy": float(acc),
        "risk_churn": risk_churn,
        "source": "backtest",
    }


# ---------------------------------------------------------------------------
# Validate existing RiskScore file
# ---------------------------------------------------------------------------
def validate_riskscore(rs_file, churn_files):
    """
    Validate an existing RiskScore file against the NEXT period's churn file.

    RiskScore_YYYY_MMDD predicts churn for period MMDD.
    To validate: find the churn file where mmdd_START == MMDD (i.e., the next period after MMDD).
    Customers in RiskScore but NOT in next churn file = churned.

    Falls back to churn file where mmdd_end == MMDD if no next-period file exists.
    """
    mmdd = rs_file["mmdd"]

    # First try: find churn file where this MMDD is the START (next period)
    # E.g., RiskScore_0401 -> Churn_*_0401_0416.csv
    matching = [f for f in churn_files if f["mmdd_start"] == mmdd]
    if not matching:
        # Fallback: find churn file where this MMDD is the END
        matching = [f for f in churn_files if f["mmdd_end"] == mmdd and str(f["year"]) == rs_file["year"]]
    if not matching:
        matching = [f for f in churn_files if f["mmdd_end"] == mmdd]
    if not matching:
        return None

    actual_file = matching[-1]  # take the most recent

    # Load RiskScore
    df_pred = pd.read_csv(rs_file["filepath"], low_memory=False, encoding="utf-8-sig")

    # Determine score column name
    score_col = None
    for col_name in ["churn_score", "churn_score_rf_multi"]:
        if col_name in df_pred.columns:
            score_col = col_name
            break
    if score_col is None:
        return None

    # Load actual buyers
    df_actual = pd.read_csv(actual_file["filepath"], low_memory=False)
    actual_buyers = set(df_actual["userNo"].values)

    # Exclude new customers
    if "risk_level" in df_pred.columns:
        eval_mask = df_pred["risk_level"] != "New Customer"
    else:
        eval_mask = df_pred[score_col] >= 0

    df_eval = df_pred[eval_mask].copy()
    if len(df_eval) == 0:
        return None

    y_true = (~df_eval["userNo"].isin(actual_buyers)).astype(int).values
    y_scores = df_eval[score_col].values / 100.0
    scores_100 = df_eval[score_col].values
    y_pred = (scores_100 >= 50).astype(int)

    if "risk_level" in df_eval.columns:
        risk_levels = df_eval["risk_level"].values
    else:
        risk_levels = [assign_risk_level(s) for s in scores_100]

    try:
        auc_roc = roc_auc_score(y_true, y_scores)
        auc_pr = average_precision_score(y_true, y_scores)
    except ValueError:
        return None

    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    churn_rate = y_true.mean()

    risk_churn = {}
    for level in ["Low", "Medium", "High", "Critical"]:
        mask = np.array([rl == level for rl in risk_levels])
        if mask.sum() > 0:
            risk_churn[level] = float(y_true[mask].mean())

    # Figure out the period label
    period_label = f"*-{mmdd}"
    for cf in churn_files:
        if cf["mmdd_end"] == mmdd:
            period_label = cf["label"]
            break

    return {
        "period": period_label,
        "mmdd_end": mmdd,
        "year": int(rs_file["year"]),
        "n_eligible": len(df_eval),
        "n_total": len(df_pred),
        "churn_rate": float(churn_rate),
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "f1": float(f1),
        "accuracy": float(acc),
        "risk_churn": risk_churn,
        "source": "riskscore",
    }


# ---------------------------------------------------------------------------
# Generate tracking chart
# ---------------------------------------------------------------------------
def generate_chart(results, chart_path):
    """Generate multi-panel validation tracking chart."""
    if len(results) < 1:
        print("  No results to chart.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    periods = [r["period"] for r in results]
    x = range(len(periods))

    # Color markers differently for backtest vs riskscore
    colors = ["#e74c3c" if r["source"] == "riskscore" else "#3498db" for r in results]

    # --- Panel 1: AUC-ROC trend ---
    ax = axes[0, 0]
    auc_vals = [r["auc_roc"] for r in results]
    ax.plot(x, auc_vals, "o-", color="#3498db", linewidth=2, markersize=6, label="AUC-ROC")
    # Highlight production RiskScore points
    rs_idx = [i for i, r in enumerate(results) if r["source"] == "riskscore"]
    if rs_idx:
        ax.plot(rs_idx, [auc_vals[i] for i in rs_idx], "D", color="#e74c3c",
                markersize=10, zorder=5, label="Production RiskScore")
    ax.axhline(y=0.78, color="#e74c3c", linestyle="--", alpha=0.7, linewidth=1.5, label="Target (0.78)")
    ax.axhline(y=np.mean(auc_vals), color="#27ae60", linestyle=":", alpha=0.7, linewidth=1.5,
               label=f"Mean ({np.mean(auc_vals):.4f})")
    ax.set_ylabel("AUC-ROC", fontsize=11)
    ax.set_title("AUC-ROC Trend Over Time", fontsize=13, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(periods, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(min(auc_vals) - 0.02, max(auc_vals) + 0.02)
    # Annotate min and max
    min_idx = np.argmin(auc_vals)
    max_idx = np.argmax(auc_vals)
    ax.annotate(f"{auc_vals[min_idx]:.4f}", xy=(min_idx, auc_vals[min_idx]),
                xytext=(0, -15), textcoords="offset points", ha="center", fontsize=8, color="#e74c3c")
    ax.annotate(f"{auc_vals[max_idx]:.4f}", xy=(max_idx, auc_vals[max_idx]),
                xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8, color="#27ae60")

    # --- Panel 2: Churn rate per risk level ---
    ax = axes[0, 1]
    level_colors = {"Low": "#27ae60", "Medium": "#f39c12", "High": "#e67e22", "Critical": "#e74c3c"}
    for level in ["Low", "Medium", "High", "Critical"]:
        vals = [r["risk_churn"].get(level, np.nan) * 100 for r in results]
        # Only plot if we have some data
        if not all(np.isnan(v) for v in vals):
            ax.plot(x, vals, "o-", color=level_colors[level], linewidth=1.5,
                    markersize=5, label=level, alpha=0.85)
    ax.set_ylabel("Churn Rate (%)", fontsize=11)
    ax.set_title("Churn Rate per Risk Level Over Time", fontsize=13, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(periods, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # --- Panel 3: Overall churn rate ---
    ax = axes[1, 0]
    churn_vals = [r["churn_rate"] * 100 for r in results]
    ax.bar(x, churn_vals, color="#9b59b6", alpha=0.7, edgecolor="white")
    ax.plot(x, churn_vals, "o-", color="#8e44ad", linewidth=1.5, markersize=5)
    ax.axhline(y=np.mean(churn_vals), color="#e74c3c", linestyle="--", alpha=0.7,
               label=f"Mean ({np.mean(churn_vals):.1f}%)")
    ax.set_ylabel("Churn Rate (%)", fontsize=11)
    ax.set_title("Overall Churn Rate Trend", fontsize=13, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(periods, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    # Annotate each bar
    for i, v in enumerate(churn_vals):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=7, color="#2c3e50")

    # --- Panel 4: Customer count ---
    ax = axes[1, 1]
    total_vals = [r["n_total"] for r in results]
    elig_vals = [r["n_eligible"] for r in results]
    ax.bar(x, total_vals, color="#bdc3c7", alpha=0.6, label="Total", edgecolor="white")
    ax.bar(x, elig_vals, color="#3498db", alpha=0.7, label="Eligible", edgecolor="white")
    ax.set_ylabel("Customer Count", fontsize=11)
    ax.set_title("Customer Count Trend", fontsize=13, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(periods, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: f"{v/1000:.0f}K"))

    plt.suptitle(
        "Phase 36: Validation Tracking Dashboard\n"
        f"Model: RF v3 (45 features) | {len(results)} Periods | "
        f"Mean AUC-ROC: {np.mean(auc_vals):.4f}",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved: {chart_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()

    print("=" * 70)
    print("  Phase 36: Validation Tracking Dashboard")
    print("=" * 70)

    # --- Load model ---
    print("\n  [1] Loading model...")
    model_path = os.path.join(MODELS_DIR, "churn_v3_extended.joblib")
    artifact = joblib.load(model_path)
    rf_model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    print(f"    Model: RF v3 ({len(feature_cols)} features)")

    # Load feature engineering
    sys.path.insert(0, MODELS_DIR)
    from feature_engineering import engineer_features_v2

    # --- Get files ---
    print("\n  [2] Scanning files...")
    churn_files = get_sorted_churn_files()
    rs_files = get_riskscore_files()
    print(f"    Churn files: {len(churn_files)}")
    print(f"    RiskScore files: {len(rs_files)}")

    results = []

    # --- Validate existing RiskScore files ---
    print("\n  [3] Validating existing RiskScore files...")
    for rs in rs_files:
        print(f"    Validating {rs['filename']}...", end=" ")
        res = validate_riskscore(rs, churn_files)
        if res:
            print(f"AUC={res['auc_roc']:.4f}, Churn={res['churn_rate']:.1%}")
            # Store for later dedup
            results.append(res)
        else:
            print("SKIP (no matching actual file)")

    # --- Back-test across all consecutive churn file pairs ---
    print(f"\n  [4] Back-testing across {len(churn_files) - 1} consecutive periods...")

    # Track which mmdd_end already have RiskScore results to avoid duplicates
    rs_periods = set(r["mmdd_end"] for r in results)

    for i in range(len(churn_files) - 1):
        current = churn_files[i]
        next_file = churn_files[i + 1]

        # Skip if already validated via RiskScore
        if current["mmdd_end"] in rs_periods:
            print(f"    [{i+1}/{len(churn_files)-1}] {current['label']} -> SKIP (already validated via RiskScore)")
            continue

        print(f"    [{i+1}/{len(churn_files)-1}] {current['label']}...", end=" ", flush=True)

        try:
            res = backtest_one_period(current, next_file, rf_model, feature_cols, engineer_features_v2)
            if res:
                print(f"AUC={res['auc_roc']:.4f}, Churn={res['churn_rate']:.1%}")
                results.append(res)
            else:
                print("SKIP (insufficient data)")
        except Exception as e:
            print(f"ERROR: {e}")

    # --- Filter out invalid results (NaN AUC, 0% churn) ---
    valid_results = [r for r in results if not np.isnan(r["auc_roc"]) and r["churn_rate"] > 0.01]
    invalid_count = len(results) - len(valid_results)
    if invalid_count > 0:
        print(f"\n  Filtered out {invalid_count} invalid results (NaN AUC or 0% churn)")
    results = valid_results

    # --- Sort results chronologically ---
    # Use churn_files order as the canonical timeline
    period_order = {f["mmdd_end"]: i for i, f in enumerate(churn_files)}

    def sort_key(r):
        mmdd = r["mmdd_end"]
        if mmdd in period_order:
            return period_order[mmdd]
        # Fallback: sort by year + month
        year = r.get("year", 2025)
        month = int(mmdd[:2])
        day = int(mmdd[2:])
        return 1000 + year * 100 + month

    results.sort(key=sort_key)

    # --- Print summary table ---
    print("\n" + "=" * 120)
    print("  VALIDATION TRACKING SUMMARY")
    print("=" * 120)
    print(f"  {'Period':<15} {'Source':<10} {'Eligible':>10} {'Churn%':>8} "
          f"{'AUC-ROC':>9} {'AUC-PR':>9} {'F1':>8} {'Acc%':>7} "
          f"{'Low%':>7} {'Med%':>7} {'High%':>7} {'Crit%':>7}")
    print(f"  {'-' * 118}")

    for r in results:
        low = r["risk_churn"].get("Low", float("nan")) * 100
        med = r["risk_churn"].get("Medium", float("nan")) * 100
        high = r["risk_churn"].get("High", float("nan")) * 100
        crit = r["risk_churn"].get("Critical", float("nan")) * 100

        source_tag = "PROD" if r["source"] == "riskscore" else "BT"

        print(f"  {r['period']:<15} {source_tag:<10} {r['n_eligible']:>10,} {r['churn_rate']*100:>7.1f}% "
              f"{r['auc_roc']:>9.4f} {r['auc_pr']:>9.4f} {r['f1']:>8.4f} {r['accuracy']*100:>6.1f}% "
              f"{low:>6.1f}% {med:>6.1f}% {high:>6.1f}% {crit:>6.1f}%")

    print(f"  {'-' * 118}")

    # Summary stats
    auc_vals = [r["auc_roc"] for r in results]
    churn_vals = [r["churn_rate"] for r in results]
    print(f"\n  Summary Statistics ({len(results)} periods):")
    print(f"    AUC-ROC:    Mean={np.mean(auc_vals):.4f}  Std={np.std(auc_vals):.4f}  "
          f"Min={np.min(auc_vals):.4f}  Max={np.max(auc_vals):.4f}")
    print(f"    Churn Rate:  Mean={np.mean(churn_vals)*100:.1f}%  Std={np.std(churn_vals)*100:.1f}%  "
          f"Min={np.min(churn_vals)*100:.1f}%  Max={np.max(churn_vals)*100:.1f}%")

    # Risk level averages
    print(f"\n  Average Churn Rate per Risk Level:")
    for level in ["Low", "Medium", "High", "Critical"]:
        vals = [r["risk_churn"].get(level, np.nan) for r in results]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            print(f"    {level:<10} {np.mean(vals)*100:>6.1f}%  (std={np.std(vals)*100:.1f}%, n={len(vals)})")

    # Periods above/below target
    target_auc = 0.78
    above = sum(1 for v in auc_vals if v >= target_auc)
    print(f"\n  AUC-ROC >= {target_auc}: {above}/{len(auc_vals)} periods ({above/len(auc_vals)*100:.0f}%)")

    # --- Generate chart ---
    print("\n  [5] Generating tracking chart...")
    os.makedirs(CHARTS_DIR, exist_ok=True)
    chart_path = os.path.join(CHARTS_DIR, "phase36_validation_tracking.png")
    generate_chart(results, chart_path)

    elapsed = time.time() - t0
    print(f"\n  Phase 36 complete ({elapsed:.1f}s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
