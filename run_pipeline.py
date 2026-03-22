#!/usr/bin/env python3
"""
Phase 21: Automated Bi-Weekly Pipeline
=======================================
Runs the full bi-weekly operational cycle: validate previous predictions + score next period.

Usage:
    python3 run_pipeline.py --validate Churn_2026_0301_0316.csv --score Churn_Pred_0416.csv
    python3 run_pipeline.py --validate Churn_2026_0301_0316.csv --score Churn_Pred_0416.csv --calibrate
    python3 run_pipeline.py --validate Churn_2026_0301_0316.csv --score Churn_Pred_0416.csv --calibrate --clv --cluster
    python3 run_pipeline.py --validate Churn_2026_0301_0316.csv --score Churn_Pred_0416.csv --use-tx
    python3 run_pipeline.py --validate Churn_2026_0301_0316.csv --score Churn_Pred_0416.csv --use-tx --tx-file TransactionOrder/Transaction_Order_2026_0416.csv

Steps:
    1. Validate previous predictions (AUC, churn per risk level, charts)
    2. Score next period (ensemble RF+LSTM, SHAP, optional calibration/CLV/cluster)
    3. Export CSV + charts
    4. Print Thai-language summary
"""

import argparse
import glob
import json
import os
import re
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, accuracy_score, roc_curve
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_TENURE = 3
BATCH_SIZE = 512
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RISK_LEVELS = {
    "Low": (0, 25),
    "Medium": (26, 50),
    "High": (51, 75),
    "Critical": (76, 100),
}

ACTIONS = {
    "Low": "ไม่ต้องดำเนินการ — ลูกค้าภักดี",
    "Medium": "ส่ง offer เบาๆ — รักษาความสัมพันธ์",
    "High": "เร่งส่ง promotion — เริ่มมีสัญญาณ Churn",
    "Critical": "ติดต่อทันที — โอกาส Churn สูงมาก",
    "New Customer": "ติดตามพฤติกรรม — ลูกค้าใหม่",
}


# ---------------------------------------------------------------------------
# LSTM model definition (must match training architecture)
# ---------------------------------------------------------------------------
class ChurnLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(32, 1),
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(hidden).squeeze(-1)


# ---------------------------------------------------------------------------
# Helper functions
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


def extract_mmdd(filename):
    """Extract the END date (MMDD) from a filename."""
    basename = os.path.basename(filename)
    # Churn_2026_0301_0316.csv -> 0316
    m = re.search(r"Churn_\d{4}_\d{4}_(\d{4})\.csv", basename)
    if m:
        return m.group(1)
    # Churn_Pred_0416.csv -> 0416
    m = re.search(r"Churn_Pred_(\d{4})\.csv", basename)
    if m:
        return m.group(1)
    return None


def find_previous_risk_scores(mmdd):
    """Find previous risk score file matching the END date."""
    patterns = [
        os.path.join(BASE_DIR, f"Churn_RiskScore_Ensemble_*_{mmdd}.csv"),
        os.path.join(BASE_DIR, f"Churn_RiskScore_*_{mmdd}.csv"),
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            # Return most recent
            matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]
    return None


def find_tx_file(mmdd):
    """Find TX file matching the given MMDD date code.

    Looks for TransactionOrder/Transaction_Order_*_{mmdd}.csv
    Returns the file path if found, None otherwise.
    """
    pattern = os.path.join(BASE_DIR, "TransactionOrder", f"Transaction_Order_*_{mmdd}.csv")
    matches = glob.glob(pattern)
    if matches:
        matches.sort(key=os.path.getmtime, reverse=True)
        return matches[0]
    return None


def load_models_all(base_dir, use_calibrate=False, use_tx=False):
    """Load RF model, LSTM, ensemble config, and optionally calibrator.

    When use_tx=True, loads v5 model (56 features incl. TX) instead of v3.
    """
    # Ensemble config
    config_path = os.path.join(base_dir, "models", "ensemble_config.json")
    with open(config_path) as f:
        config = json.load(f)

    # RF model: v5 (TX) or v3
    if use_tx:
        rf_path = os.path.join(base_dir, "models", "churn_v5_tx.joblib")
    else:
        rf_path = os.path.join(base_dir, "models", "churn_v3_extended.joblib")
    rf_artifact = joblib.load(rf_path)
    rf_model = rf_artifact["model"]
    feature_cols = rf_artifact["feature_cols"]

    # LSTM
    lstm_path = os.path.join(base_dir, "models", "lstm_churn.pt")
    lstm_params = config.get("lstm_params", {})
    lstm_model = ChurnLSTM(
        input_size=lstm_params.get("input_size", 1),
        hidden_size=lstm_params.get("hidden_size", 64),
        num_layers=lstm_params.get("num_layers", 2),
        dropout=lstm_params.get("dropout", 0.3),
    )
    lstm_model.load_state_dict(
        torch.load(lstm_path, map_location="cpu", weights_only=True)
    )
    lstm_model.eval()

    w_rf = config["w_rf"]
    w_lstm = config["w_lstm"]

    # Calibrator
    calibrator = None
    if use_calibrate:
        cal_path = os.path.join(base_dir, "models", "calibrator_isotonic.joblib")
        if os.path.exists(cal_path):
            calibrator = joblib.load(cal_path)
            print(f"  Calibrator loaded: {cal_path}")
        else:
            print(f"  WARNING: Calibrator not found at {cal_path}, skipping calibration")

    return rf_model, feature_cols, lstm_model, w_rf, w_lstm, calibrator


def compute_rf_probabilities(df_elig, mat_elig, feat_item_cols, rf_model, feature_cols):
    """Compute RF churn probabilities for eligible customers."""
    sys.path.insert(0, os.path.join(BASE_DIR, "models"))
    from feature_engineering import engineer_features_v2

    feats = engineer_features_v2(df_elig, mat_elig, feat_item_cols)
    feats = feats.fillna(0)

    for col in feature_cols:
        if col not in feats.columns:
            feats[col] = 0

    X_rf = feats[feature_cols].fillna(0)
    rf_probs = rf_model.predict_proba(X_rf)[:, 1]
    return rf_probs, feats


def compute_lstm_probabilities(mat_vals, reg, lstm_model):
    """Compute LSTM churn probabilities for eligible customers."""
    seqs = []
    lengths = []

    for i in range(len(mat_vals)):
        reg_start = np.argmax(reg[i])
        seq = mat_vals[i, reg_start:]
        seq = np.nan_to_num(seq, nan=0.0)
        seq = np.log1p(seq).astype(np.float32)
        seqs.append(torch.tensor(seq).unsqueeze(-1))
        lengths.append(len(seq))

    if len(seqs) == 0:
        return np.array([])

    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    lens_t = torch.tensor(lengths, dtype=torch.long)

    lstm_probs_list = []
    lstm_model.eval()
    with torch.no_grad():
        for start in range(0, len(padded), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(padded))
            batch_probs = torch.sigmoid(
                lstm_model(padded[start:end], lens_t[start:end])
            ).numpy()
            lstm_probs_list.append(batch_probs)

    return np.concatenate(lstm_probs_list)


def compute_shap_lgb_proxy(feats, feature_cols, rf_probs):
    """Compute SHAP values using LightGBM proxy trained on RF predictions."""
    import lightgbm as lgb
    import shap

    X = feats[feature_cols].fillna(0)
    # Train LightGBM proxy on RF predicted probabilities
    proxy = lgb.LGBMClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        num_leaves=63, min_child_samples=50,
        verbose=-1, n_jobs=-1,
    )
    y_pseudo = (rf_probs >= 0.5).astype(int)
    proxy.fit(X, y_pseudo)

    # SHAP
    explainer = shap.TreeExplainer(proxy)
    shap_values = explainer.shap_values(X)
    # For binary classification, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    return shap_vals, feature_cols


def get_top_shap_reasons(shap_vals, feature_cols, top_n=5):
    """Extract top N SHAP reasons per customer."""
    reasons = []
    for i in range(len(shap_vals)):
        row = shap_vals[i]
        sorted_idx = np.argsort(np.abs(row))[::-1][:top_n]
        row_reasons = []
        for j in sorted_idx:
            feat_name = feature_cols[j]
            shap_val = row[j]
            direction = "increases" if shap_val > 0 else "decreases"
            row_reasons.append(f"{feat_name} ({direction} risk, SHAP={shap_val:.3f})")
        reasons.append(row_reasons)
    return reasons


def compute_clv(feats):
    """Compute CLV scores: avg_purchase_value * frequency * expected_lifetime."""
    avg_purchase = feats["avg_items_per_active_round"].values.copy()
    frequency = feats["purchase_frequency_ratio"].values.copy()
    # Expected lifetime: proportional to tenure, inversely to churn risk
    tenure = feats["tenure_rounds"].values.copy()
    expected_lifetime = np.maximum(tenure / 10.0, 1.0)

    clv = avg_purchase * frequency * expected_lifetime
    clv = np.maximum(clv, 0.0)
    return clv


def assign_clv_tier(clv_scores):
    """Assign CLV tier based on quartiles."""
    tiers = np.full(len(clv_scores), "Low", dtype=object)
    pos = clv_scores > 0
    if pos.sum() > 3:
        q25, q50, q75 = np.percentile(clv_scores[pos], [25, 50, 75])
        tiers = np.where(clv_scores >= q75, "Platinum",
                np.where(clv_scores >= q50, "Gold",
                np.where(clv_scores >= q25, "Silver", "Bronze")))
    return tiers


def clv_priority_action(risk_level, clv_tier):
    """Combine risk level and CLV tier for priority action."""
    if risk_level == "New Customer":
        return "ติดตามพฤติกรรม — ลูกค้าใหม่"
    if risk_level == "Low":
        return "ไม่ต้องดำเนินการ — ลูกค้าภักดี"

    priority_map = {
        ("Critical", "Platinum"): "VIP ด่วนที่สุด — ติดต่อทันที ลูกค้ามูลค่าสูง",
        ("Critical", "Gold"): "ด่วนมาก — ติดต่อทันที",
        ("Critical", "Silver"): "ด่วน — ส่ง promotion พิเศษ",
        ("Critical", "Bronze"): "ติดต่อทันที — โอกาส Churn สูงมาก",
        ("High", "Platinum"): "สำคัญมาก — เร่งส่ง promotion ลูกค้ามูลค่าสูง",
        ("High", "Gold"): "สำคัญ — เร่งส่ง promotion",
        ("High", "Silver"): "เร่งส่ง promotion — เริ่มมีสัญญาณ Churn",
        ("High", "Bronze"): "เร่งส่ง promotion — เริ่มมีสัญญาณ Churn",
        ("Medium", "Platinum"): "รักษาลูกค้า VIP — ส่ง offer พิเศษ",
        ("Medium", "Gold"): "ส่ง offer เบาๆ — รักษาความสัมพันธ์",
        ("Medium", "Silver"): "ส่ง offer เบาๆ — รักษาความสัมพันธ์",
        ("Medium", "Bronze"): "ส่ง offer เบาๆ — รักษาความสัมพันธ์",
    }
    return priority_map.get((risk_level, clv_tier), ACTIONS.get(risk_level, ""))


# ---------------------------------------------------------------------------
# Step 1: Validate previous predictions
# ---------------------------------------------------------------------------
def step_validate(validate_file):
    """Validate previous predictions against actual churn data."""
    print("\n" + "=" * 60)
    print("  Step 1: Validate Previous Predictions")
    print("=" * 60)
    t0 = time.time()

    # Extract end date from validate file
    basename = os.path.basename(validate_file)
    m = re.search(r"Churn_\d{4}_(\d{4})_(\d{4})\.csv", basename)
    if not m:
        print(f"  ERROR: Cannot parse dates from {basename}")
        return None
    mmdd_start = m.group(1)
    mmdd_end = m.group(2)
    print(f"  Validate file: {basename}")
    print(f"  Period: {mmdd_start} -> {mmdd_end}")

    # Find previous risk scores
    prev_file = find_previous_risk_scores(mmdd_end)
    if prev_file is None:
        print(f"  WARNING: No previous risk score file found for MMDD={mmdd_end}")
        print(f"  Skipping validation step.")
        return None

    print(f"  Previous scores: {os.path.basename(prev_file)}")

    # Load actual results (contains ONLY customers who bought)
    df_actual = pd.read_csv(validate_file, low_memory=False)
    actual_buyers = set(df_actual["userNo"].values)
    print(f"  Actual buyers: {len(actual_buyers):,}")

    # Load previous predictions
    df_pred = pd.read_csv(prev_file)
    print(f"  Predicted customers: {len(df_pred):,}")

    # Determine churn: customer in prediction file but NOT in actual file = churned
    df_pred["actual_churn"] = (~df_pred["userNo"].isin(actual_buyers)).astype(int)

    # Exclude new customers from evaluation
    eval_mask = df_pred["risk_level"] != "New Customer"
    df_eval = df_pred[eval_mask].copy()
    n_eval = len(df_eval)
    n_new = (~eval_mask).sum()

    if n_eval == 0:
        print("  ERROR: No eligible customers to evaluate")
        return None

    y_true = df_eval["actual_churn"].values
    y_scores = df_eval["churn_score"].values / 100.0  # normalize to [0, 1]
    y_pred = (df_eval["churn_score"].values >= 50).astype(int)

    # Metrics
    auc_roc = roc_auc_score(y_true, y_scores)
    auc_pr = average_precision_score(y_true, y_scores)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    churn_rate = y_true.mean()

    # Churn rate per risk level
    risk_churn = {}
    for level in ["Low", "Medium", "High", "Critical"]:
        mask = df_eval["risk_level"] == level
        if mask.sum() > 0:
            level_churn = df_eval.loc[mask, "actual_churn"].mean()
            risk_churn[level] = {
                "count": int(mask.sum()),
                "churn_rate": float(level_churn),
                "retention": float(1 - level_churn),
            }

    # Print results
    print(f"\n  Validation Results:")
    print(f"    Eligible customers: {n_eval:,}")
    print(f"    New customers (excluded): {n_new:,}")
    print(f"    Overall churn rate: {churn_rate:.1%}")
    print(f"\n    AUC-ROC:  {auc_roc:.4f}")
    print(f"    AUC-PR:   {auc_pr:.4f}")
    print(f"    F1:       {f1:.4f}")
    print(f"    Accuracy: {acc:.1%}")
    print(f"\n    Confusion Matrix:")
    print(f"      TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"      FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    print(f"\n    Churn Rate per Risk Level:")
    print(f"    {'Level':<12} {'Count':>10} {'Churn Rate':>12} {'Retention':>12}")
    print(f"    {'-' * 48}")
    for level in ["Low", "Medium", "High", "Critical"]:
        if level in risk_churn:
            rc = risk_churn[level]
            print(f"    {level:<12} {rc['count']:>10,} {rc['churn_rate']:>11.1%} {rc['retention']:>11.1%}")

    # Generate validation chart
    os.makedirs(os.path.join(BASE_DIR, "charts"), exist_ok=True)
    chart_path = os.path.join(BASE_DIR, "charts", f"phase21_validation_{mmdd_end}.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    axes[0, 0].plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {auc_roc:.4f}")
    axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Churn rate per risk level
    levels = [l for l in ["Low", "Medium", "High", "Critical"] if l in risk_churn]
    churn_rates = [risk_churn[l]["churn_rate"] * 100 for l in levels]
    colors = ["#27ae60", "#f39c12", "#e67e22", "#e74c3c"][:len(levels)]
    bars = axes[0, 1].bar(levels, churn_rates, color=colors, edgecolor="white")
    for bar, rate in zip(bars, churn_rates):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{rate:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axes[0, 1].set_ylabel("Churn Rate (%)")
    axes[0, 1].set_title("Actual Churn Rate per Risk Level")
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # 3. Confusion Matrix heatmap
    im = axes[1, 0].imshow(cm, cmap="Blues", aspect="auto")
    for i_row in range(2):
        for j_col in range(2):
            axes[1, 0].text(j_col, i_row, f"{cm[i_row, j_col]:,}",
                           ha="center", va="center", fontsize=14, fontweight="bold",
                           color="white" if cm[i_row, j_col] > cm.max() / 2 else "black")
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticklabels(["Predicted\nNo Churn", "Predicted\nChurn"])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_yticklabels(["Actual\nNo Churn", "Actual\nChurn"])
    axes[1, 0].set_title("Confusion Matrix")

    # 4. Score distribution by actual churn
    axes[1, 1].hist(df_eval.loc[df_eval["actual_churn"] == 0, "churn_score"],
                    bins=20, alpha=0.6, color="#27ae60", label="No Churn", density=True)
    axes[1, 1].hist(df_eval.loc[df_eval["actual_churn"] == 1, "churn_score"],
                    bins=20, alpha=0.6, color="#e74c3c", label="Churned", density=True)
    axes[1, 1].set_xlabel("Churn Score")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Score Distribution by Actual Outcome")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Phase 21 Validation — Period {mmdd_end}\n"
        f"AUC-ROC={auc_roc:.4f}  F1={f1:.4f}  Accuracy={acc:.1%}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved: {chart_path}")
    print(f"  ({time.time() - t0:.1f}s)")

    return {
        "mmdd_end": mmdd_end,
        "n_eval": n_eval,
        "n_new": n_new,
        "churn_rate": float(churn_rate),
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "f1": float(f1),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "risk_churn": risk_churn,
        "chart_path": chart_path,
    }


# ---------------------------------------------------------------------------
# Step 2: Score next period
# ---------------------------------------------------------------------------
def step_score(score_file, use_calibrate=False, use_clv=False, use_cluster=False,
               use_tx=False, tx_file=None):
    """Score customers for the next period."""
    print("\n" + "=" * 60)
    print("  Step 2: Score Next Period")
    print("=" * 60)
    t0_total = time.time()

    # Extract MMDD from score file
    mmdd_score = extract_mmdd(score_file)
    year = datetime.now().strftime("%Y")
    print(f"  Score file: {os.path.basename(score_file)}")
    print(f"  Target period: {mmdd_score}")

    # --- Resolve TX file if --use-tx ---
    tx_path = None
    if use_tx:
        if tx_file:
            tx_path = tx_file if os.path.isabs(tx_file) else os.path.join(os.getcwd(), tx_file)
        else:
            # Auto-detect TX file based on scoring period's second-to-last MMDD
            # For Churn_Pred_MMDD.csv, TX_MMDD maps to same MMDD
            tx_path = find_tx_file(mmdd_score)

        if tx_path and os.path.exists(tx_path):
            print(f"  TX file: {os.path.basename(tx_path)}")
        else:
            print(f"  WARNING: --use-tx set but TX file not found"
                  f"{' (' + tx_path + ')' if tx_path else ''}")
            print(f"  Falling back to v3 model (no TX features)")
            use_tx = False
            tx_path = None

    # --- Load models ---
    print("\n  [2.1] Loading models...")
    t0 = time.time()
    rf_model, feature_cols, lstm_model, w_rf, w_lstm, calibrator = load_models_all(
        BASE_DIR, use_calibrate=use_calibrate, use_tx=use_tx
    )
    model_label = "v5 TX" if use_tx else "v3"
    print(f"    RF {model_label}: {len(feature_cols)} features")
    print(f"    Ensemble weights: RF={w_rf:.2f}, LSTM={w_lstm:.2f}")
    if use_calibrate and calibrator is not None:
        print(f"    Calibration: Isotonic Regression")
    print(f"    ({time.time() - t0:.1f}s)")

    # --- Load data ---
    print("\n  [2.2] Loading data...")
    t0 = time.time()
    df = pd.read_csv(score_file, low_memory=False)
    print(f"    Rows: {len(df):,}")

    item_cols = [c for c in df.columns if c.startswith("item")]
    if not item_cols:
        print("  ERROR: No item columns found")
        return None

    # For prediction files, ALL item columns are features
    feat_item_cols = item_cols

    mat = df[item_cols].apply(pd.to_numeric, errors="coerce")
    mat_vals = mat.values.astype(float)
    reg = ~np.isnan(mat_vals)

    # Compute tenure
    first_reg = np.where(reg.any(axis=1), reg.argmax(axis=1), len(item_cols))
    tenure = len(item_cols) - first_reg
    valid = tenure >= MIN_TENURE

    n_eligible = valid.sum()
    n_new = (~valid).sum()
    print(f"    Item columns: {len(item_cols)}")
    print(f"    Eligible (tenure >= {MIN_TENURE}): {n_eligible:,}")
    print(f"    New customers (tenure < {MIN_TENURE}): {n_new:,}")
    print(f"    ({time.time() - t0:.1f}s)")

    # --- Score eligible customers ---
    print("\n  [2.3] Scoring eligible customers...")
    t0 = time.time()

    all_scores = np.full(len(df), -1.0)
    all_feats = None

    if n_eligible > 0:
        df_elig = df[valid].copy()
        mat_elig = mat[valid].copy()
        mat_vals_elig = mat_vals[valid]
        reg_elig = reg[valid]

        # RF probabilities
        print("    Computing RF probabilities...")
        rf_probs, feats = compute_rf_probabilities(
            df_elig, mat_elig, feat_item_cols, rf_model, feature_cols
        )

        # Merge TX features if using v5
        if use_tx and tx_path:
            print("    Merging TX features...")
            from feature_engineering import aggregate_tx_features
            tx_df = pd.read_csv(tx_path, low_memory=False)
            tx_agg = aggregate_tx_features(tx_df)
            # tx_agg is indexed by userNo; merge into feats
            feats = feats.merge(
                tx_agg, left_on="userNo" if "userNo" in feats.columns
                else feats.index, right_index=True, how="left",
            )
            # If merge created a 'key_0' column from index merge, drop it
            if "key_0" in feats.columns:
                feats.drop(columns=["key_0"], inplace=True)
            feats = feats.fillna(0)
            # Ensure all v5 feature columns exist
            for col in feature_cols:
                if col not in feats.columns:
                    feats[col] = 0
            # Re-compute RF probs with full v5 features
            X_rf = feats[feature_cols].fillna(0)
            rf_probs = rf_model.predict_proba(X_rf)[:, 1]

        all_feats = feats

        # LSTM probabilities
        print("    Computing LSTM probabilities...")
        lstm_probs = compute_lstm_probabilities(mat_vals_elig, reg_elig, lstm_model)

        # Ensemble
        ensemble_probs = w_rf * rf_probs + w_lstm * lstm_probs

        # Calibration
        if use_calibrate and calibrator is not None:
            print("    Applying calibration...")
            ensemble_probs = calibrator.predict(ensemble_probs)

        ensemble_scores = np.clip(np.round(ensemble_probs * 100, 1), 0, 100)
        all_scores[valid] = ensemble_scores

    print(f"    ({time.time() - t0:.1f}s)")

    # --- SHAP explanations ---
    print("\n  [2.4] Computing SHAP explanations (LightGBM proxy)...")
    t0 = time.time()
    shap_reasons_all = None
    if n_eligible > 0 and all_feats is not None:
        shap_vals, shap_cols = compute_shap_lgb_proxy(all_feats, feature_cols, rf_probs)
        shap_reasons_all = get_top_shap_reasons(shap_vals, list(shap_cols), top_n=5)
        print(f"    SHAP computed for {len(shap_vals):,} customers")
    print(f"    ({time.time() - t0:.1f}s)")

    # --- Optional: CLV ---
    clv_scores = None
    clv_tiers = None
    if use_clv and all_feats is not None:
        print("\n  [2.5] Computing CLV scores...")
        t0 = time.time()
        clv_scores_elig = compute_clv(all_feats)
        clv_tiers_elig = assign_clv_tier(clv_scores_elig)
        clv_scores = np.full(len(df), 0.0)
        clv_tiers = np.full(len(df), "N/A", dtype=object)
        clv_scores[valid] = clv_scores_elig
        clv_tiers[valid] = clv_tiers_elig
        print(f"    CLV computed for {n_eligible:,} customers")
        print(f"    ({time.time() - t0:.1f}s)")

    # --- Optional: Cluster ---
    churn_patterns = None
    if use_cluster:
        print("\n  [2.6] Assigning churn pattern clusters...")
        t0 = time.time()
        cluster_path = os.path.join(BASE_DIR, "models", "churn_clusters.joblib")
        if os.path.exists(cluster_path):
            cluster_model = joblib.load(cluster_path)
            if n_eligible > 0 and all_feats is not None:
                X_cluster = all_feats[feature_cols].fillna(0)
                cluster_labels = cluster_model.predict(X_cluster)
                churn_patterns = np.full(len(df), -1, dtype=int)
                churn_patterns[valid] = cluster_labels
                print(f"    Clusters assigned for {n_eligible:,} customers")
        else:
            print(f"    WARNING: Cluster model not found at {cluster_path}, skipping")
        print(f"    ({time.time() - t0:.1f}s)")

    # --- Build output DataFrames ---
    print("\n  [2.7] Building output files...")
    t0 = time.time()

    # Main risk score file
    risk_levels = [assign_risk_level(s) for s in all_scores]
    result_df = pd.DataFrame({
        "userNo": df["userNo"].values,
        "churn_score": all_scores,
        "risk_level": risk_levels,
        "recommended_action": [ACTIONS.get(rl, "") for rl in risk_levels],
    })

    if use_clv and clv_scores is not None:
        result_df["clv_score"] = np.round(clv_scores, 2)
        result_df["clv_tier"] = clv_tiers
        result_df["priority_action"] = [
            clv_priority_action(rl, ct)
            for rl, ct in zip(risk_levels, clv_tiers)
        ]

    if use_cluster and churn_patterns is not None:
        result_df["churn_pattern"] = churn_patterns

    out_risk = os.path.join(BASE_DIR, f"Churn_RiskScore_{year}_{mmdd_score}.csv")
    result_df.to_csv(out_risk, index=False, encoding="utf-8-sig")
    print(f"    Saved: {os.path.basename(out_risk)}")

    # Explained file
    out_explained = os.path.join(BASE_DIR, f"Churn_RiskScore_Explained_{year}_{mmdd_score}.csv")
    if shap_reasons_all is not None:
        explained_rows = []
        elig_userNos = df.loc[valid, "userNo"].values
        elig_scores = all_scores[valid]
        elig_risk = [assign_risk_level(s) for s in elig_scores]

        for i in range(len(shap_reasons_all)):
            row = {
                "userNo": elig_userNos[i],
                "churn_score": elig_scores[i],
                "risk_level": elig_risk[i],
            }
            for j, reason in enumerate(shap_reasons_all[i]):
                row[f"top_reason_{j+1}"] = reason
            explained_rows.append(row)

        explained_df = pd.DataFrame(explained_rows)
        explained_df.to_csv(out_explained, index=False, encoding="utf-8-sig")
        print(f"    Saved: {os.path.basename(out_explained)}")

    # Scoring distribution chart
    chart_path = os.path.join(BASE_DIR, "charts", f"risk_scoring_{mmdd_score}.png")
    os.makedirs(os.path.join(BASE_DIR, "charts"), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Score distribution histogram
    elig_scores_arr = all_scores[valid]
    axes[0].hist(elig_scores_arr, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Churn Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Score Distribution (Eligible Customers)")
    axes[0].axvline(x=25, color="#27ae60", linestyle="--", alpha=0.7, label="Low/Medium")
    axes[0].axvline(x=50, color="#f39c12", linestyle="--", alpha=0.7, label="Medium/High")
    axes[0].axvline(x=75, color="#e74c3c", linestyle="--", alpha=0.7, label="High/Critical")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Risk level pie chart
    level_counts = result_df["risk_level"].value_counts()
    pie_labels = []
    pie_sizes = []
    pie_colors = {
        "Low": "#27ae60", "Medium": "#f39c12",
        "High": "#e67e22", "Critical": "#e74c3c",
        "New Customer": "#95a5a6",
    }
    pie_color_list = []
    for level in ["Low", "Medium", "High", "Critical", "New Customer"]:
        if level in level_counts:
            pie_labels.append(f"{level}\n({level_counts[level]:,})")
            pie_sizes.append(level_counts[level])
            pie_color_list.append(pie_colors.get(level, "#bdc3c7"))

    axes[1].pie(pie_sizes, labels=pie_labels, colors=pie_color_list,
                autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
    axes[1].set_title("Risk Level Distribution")

    plt.suptitle(
        f"Risk Scoring — Period {mmdd_score}\n"
        f"Total: {len(df):,}  Eligible: {n_eligible:,}  New: {n_new:,}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Chart saved: {os.path.basename(chart_path)}")
    print(f"    ({time.time() - t0:.1f}s)")

    # Build scoring summary
    scoring_summary = {
        "mmdd_score": mmdd_score,
        "total": len(df),
        "n_eligible": int(n_eligible),
        "n_new": int(n_new),
        "risk_distribution": {},
        "out_risk": out_risk,
        "out_explained": out_explained,
        "chart_path": chart_path,
    }
    for level in ["Low", "Medium", "High", "Critical", "New Customer"]:
        mask = result_df["risk_level"] == level
        scoring_summary["risk_distribution"][level] = int(mask.sum())

    if n_eligible > 0:
        scoring_summary["score_mean"] = float(elig_scores_arr.mean())
        scoring_summary["score_median"] = float(np.median(elig_scores_arr))
        scoring_summary["score_std"] = float(elig_scores_arr.std())

    print(f"\n  Step 2 complete ({time.time() - t0_total:.1f}s)")
    return scoring_summary


# ---------------------------------------------------------------------------
# Step 3 & 4: Summary
# ---------------------------------------------------------------------------
def print_thai_summary(validation_result, scoring_result, use_clv=False, use_tx=False):
    """Print Thai-language summary table."""
    print("\n" + "=" * 70)
    print("  สรุปผลการดำเนินงาน Pipeline — Phase 21")
    print("=" * 70)

    # Validation summary
    if validation_result:
        vr = validation_result
        print(f"\n  === ผลการ Validate (งวด {vr['mmdd_end']}) ===")
        print(f"    ลูกค้าที่ประเมิน: {vr['n_eval']:,} คน")
        print(f"    ลูกค้าใหม่ (ไม่นับ): {vr['n_new']:,} คน")
        print(f"    อัตรา Churn รวม: {vr['churn_rate']:.1%}")
        print(f"\n    {'Metric':<15} {'Value':>10}")
        print(f"    {'-' * 27}")
        print(f"    {'AUC-ROC':<15} {vr['auc_roc']:>10.4f}")
        print(f"    {'AUC-PR':<15} {vr['auc_pr']:>10.4f}")
        print(f"    {'F1':<15} {vr['f1']:>10.4f}")
        print(f"    {'Accuracy':<15} {vr['accuracy']:>9.1%}")

        print(f"\n    อัตรา Churn ตามระดับความเสี่ยง:")
        print(f"    {'ระดับ':<12} {'จำนวน':>10} {'Churn Rate':>12} {'Retention':>12}")
        print(f"    {'-' * 48}")
        for level in ["Low", "Medium", "High", "Critical"]:
            if level in vr["risk_churn"]:
                rc = vr["risk_churn"][level]
                print(f"    {level:<12} {rc['count']:>10,} {rc['churn_rate']:>11.1%} {rc['retention']:>11.1%}")
    else:
        print("\n  === ไม่มีข้อมูล Validation (ข้ามขั้นตอนนี้) ===")

    # Scoring summary
    if scoring_result:
        sr = scoring_result
        model_name = "v5 TX (56 features)" if use_tx else "v3 (45 features)"
        print(f"\n  === ผลการ Scoring (งวด {sr['mmdd_score']}) ===")
        print(f"    โมเดล: RF {model_name} + LSTM Ensemble")
        print(f"    ลูกค้าทั้งหมด: {sr['total']:,} คน")
        print(f"    ให้คะแนนได้: {sr['n_eligible']:,} คน")
        print(f"    ลูกค้าใหม่ (score=-1): {sr['n_new']:,} คน")

        if "score_mean" in sr:
            print(f"\n    Score Statistics:")
            print(f"      Mean:   {sr['score_mean']:.1f}")
            print(f"      Median: {sr['score_median']:.1f}")
            print(f"      Std:    {sr['score_std']:.1f}")

        print(f"\n    การกระจายตามระดับความเสี่ยง:")
        print(f"    {'ระดับ':<15} {'จำนวน':>10} {'%':>8} {'คำแนะนำ'}")
        print(f"    {'-' * 70}")
        rd = sr["risk_distribution"]
        total_elig = sr["n_eligible"]
        for level in ["Low", "Medium", "High", "Critical"]:
            count = rd.get(level, 0)
            pct = count / total_elig * 100 if total_elig > 0 else 0
            action = ACTIONS.get(level, "")
            print(f"    {level:<15} {count:>10,} {pct:>7.1f}% {action}")
        if rd.get("New Customer", 0) > 0:
            print(f"    {'New Customer':<15} {rd['New Customer']:>10,}          {ACTIONS['New Customer']}")

        print(f"\n    Output Files:")
        print(f"      {os.path.basename(sr['out_risk'])}")
        print(f"      {os.path.basename(sr['out_explained'])}")
        print(f"      {os.path.basename(sr['chart_path'])}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 21: Automated Bi-Weekly Pipeline — Validate + Score",
        epilog="Example: python3 run_pipeline.py --validate Churn_2026_0301_0316.csv --score Churn_Pred_0416.csv --calibrate",
    )
    parser.add_argument(
        "--validate", required=True,
        help="Actual results CSV (Churn_YYYY_MMDD1_MMDD2.csv — contains only buyers)",
    )
    parser.add_argument(
        "--score", required=True,
        help="Prediction input CSV (Churn_Pred_MMDD.csv — all item cols are features)",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Apply Isotonic Regression calibration to ensemble probabilities",
    )
    parser.add_argument(
        "--clv", action="store_true",
        help="Compute CLV scores and priority actions",
    )
    parser.add_argument(
        "--cluster", action="store_true",
        help="Assign churn pattern clusters from models/churn_clusters.joblib",
    )
    parser.add_argument(
        "--use-tx", action="store_true",
        help="Use v5 TX model (56 features) with Transaction Order data",
    )
    parser.add_argument(
        "--tx-file", default=None,
        help="Path to Transaction Order CSV (auto-detected if omitted)",
    )
    args = parser.parse_args()

    t_pipeline = time.time()

    print("=" * 70)
    print("  Phase 21: Automated Bi-Weekly Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"  Validate: {args.validate}")
    print(f"  Score:    {args.score}")
    print(f"  Options:  calibrate={args.calibrate}, clv={args.clv}, cluster={args.cluster}, use_tx={args.use_tx}")

    # Resolve file paths
    validate_path = args.validate
    if not os.path.isabs(validate_path):
        validate_path = os.path.join(os.getcwd(), validate_path)

    score_path = args.score
    if not os.path.isabs(score_path):
        score_path = os.path.join(os.getcwd(), score_path)

    # Check files exist
    for fpath, label in [(validate_path, "Validate"), (score_path, "Score")]:
        if not os.path.exists(fpath):
            print(f"\n  ERROR: {label} file not found: {fpath}", file=sys.stderr)
            sys.exit(1)

    # --- Step 1: Validate ---
    validation_result = step_validate(validate_path)

    # --- Step 2: Score ---
    scoring_result = step_score(
        score_path,
        use_calibrate=args.calibrate,
        use_clv=args.clv,
        use_cluster=args.cluster,
        use_tx=args.use_tx,
        tx_file=args.tx_file,
    )

    # --- Step 3 & 4: Summary ---
    print_thai_summary(validation_result, scoring_result, use_clv=args.clv, use_tx=args.use_tx)

    # --- Save pipeline log ---
    mmdd_score = extract_mmdd(score_path) or "unknown"
    logs_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"pipeline_{mmdd_score}.json")

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": "Phase 21",
        "validate_file": os.path.basename(validate_path),
        "score_file": os.path.basename(score_path),
        "options": {
            "calibrate": args.calibrate,
            "clv": args.clv,
            "cluster": args.cluster,
            "use_tx": args.use_tx,
            "tx_file": args.tx_file,
        },
        "validation": validation_result,
        "scoring": scoring_result,
        "elapsed_seconds": round(time.time() - t_pipeline, 1),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Pipeline log saved: {log_path}")

    elapsed = time.time() - t_pipeline
    print(f"\n  Total pipeline time: {elapsed:.1f}s")
    print(f"  Done.")


if __name__ == "__main__":
    main()
