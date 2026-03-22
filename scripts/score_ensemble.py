#!/usr/bin/env python3
"""
Production Ensemble Scoring Pipeline
=====================================
Scores customers using RF v3 + LSTM ensemble for churn prediction.

Usage:
    python3 score_ensemble.py Churn_Pred_0401.csv
    python3 score_ensemble.py Churn_Pred_0401.csv --output custom_output.csv

Outputs:
    Churn_RiskScore_Ensemble_YYYY_MMDD.csv with columns:
        userNo, churn_score, risk_level, recommended_action
"""

import argparse
import json
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

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

RECOMMENDED_ACTIONS = {
    "Low": "ไม่ต้องดำเนินการ — ลูกค้าภักดี",
    "Medium": "ส่ง offer เบาๆ — รักษาความสัมพันธ์",
    "High": "เร่งส่ง promotion — เริ่มมีสัญญาณ Churn",
    "Critical": "ติดต่อทันที — โอกาส Churn สูงมาก",
    "New Customer": "ติดตามพฤติกรรม — ลูกค้าใหม่ยังไม่มีข้อมูลเพียงพอ",
}


# ---------------------------------------------------------------------------
# LSTM model definition (must match training architecture)
# ---------------------------------------------------------------------------
class ChurnLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
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
def load_models(base_dir):
    """Load RF v3 model, LSTM model, and ensemble config."""
    # Ensemble config
    config_path = os.path.join(base_dir, "models", "ensemble_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Ensemble config not found: {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    # RF v3 model
    rf_path = os.path.join(base_dir, config["rf_model"])
    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"RF model not found: {rf_path}")
    v3_artifact = joblib.load(rf_path)
    rf_model = v3_artifact["model"]
    feature_cols = v3_artifact["feature_cols"]

    # LSTM model
    lstm_path = os.path.join(base_dir, "models", "lstm_churn.pt")
    if not os.path.exists(lstm_path):
        raise FileNotFoundError(f"LSTM model not found: {lstm_path}")

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

    return rf_model, feature_cols, lstm_model, w_rf, w_lstm


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


def compute_rf_probabilities(df_elig, mat_elig, feat_item_cols, rf_model, feature_cols):
    """Compute RF churn probabilities for eligible customers."""
    sys.path.insert(0, os.path.join(BASE_DIR, "models"))
    from feature_engineering import engineer_features_v2

    feats = engineer_features_v2(df_elig, mat_elig, feat_item_cols)
    feats = feats.fillna(0)

    # Ensure all required feature columns exist, fill missing with 0
    for col in feature_cols:
        if col not in feats.columns:
            feats[col] = 0

    X_rf = feats[feature_cols].fillna(0)
    rf_probs = rf_model.predict_proba(X_rf)[:, 1]
    return rf_probs


def compute_lstm_probabilities(mat_vals, reg, lstm_model):
    """Compute LSTM churn probabilities for eligible customers."""
    seqs = []
    lengths = []

    for i in range(len(mat_vals)):
        reg_start = np.argmax(reg[i])
        # For prediction files, use ALL columns (no target to exclude)
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


def derive_output_filename(input_file):
    """Derive output filename from input: Churn_Pred_MMDD.csv -> Churn_RiskScore_Ensemble_YYYY_MMDD.csv."""
    basename = os.path.basename(input_file)
    # Try to extract date portion from Churn_Pred_MMDD.csv
    match = re.search(r"Churn_Pred_(\d{4})\.csv", basename)
    if match:
        mmdd = match.group(1)
        from datetime import datetime

        year = datetime.now().strftime("%Y")
        return f"Churn_RiskScore_Ensemble_{year}_{mmdd}.csv"
    # Fallback
    name = os.path.splitext(basename)[0]
    return f"RiskScore_Ensemble_{name}.csv"


def print_summary(result_df):
    """Print scoring summary statistics."""
    total = len(result_df)
    new_mask = result_df["risk_level"] == "New Customer"
    eligible = result_df[~new_mask]
    n_new = new_mask.sum()
    n_eligible = len(eligible)

    print(f"\n{'=' * 60}")
    print(f"  Scoring Summary")
    print(f"{'=' * 60}")
    print(f"  Total customers:    {total:>10,}")
    print(f"  Eligible (scored):  {n_eligible:>10,}")
    print(f"  New (tenure < {MIN_TENURE}):   {n_new:>10,}")

    if n_eligible > 0:
        scores = eligible["churn_score"]
        print(f"\n  Score Statistics (eligible only):")
        print(f"    Mean:   {scores.mean():>8.1f}")
        print(f"    Median: {scores.median():>8.1f}")
        print(f"    Std:    {scores.std():>8.1f}")
        print(f"    Min:    {scores.min():>8.1f}")
        print(f"    Max:    {scores.max():>8.1f}")

        print(f"\n  Risk Level Distribution:")
        print(f"    {'Level':<15} {'Count':>10} {'Pct':>8}")
        print(f"    {'-' * 35}")
        for level in ["Low", "Medium", "High", "Critical"]:
            mask = eligible["risk_level"] == level
            count = mask.sum()
            pct = count / n_eligible * 100 if n_eligible > 0 else 0
            print(f"    {level:<15} {count:>10,} {pct:>7.1f}%")

    if n_new > 0:
        print(f"\n    {'New Customer':<15} {n_new:>10,}     (score = -1)")

    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Score customers using RF v3 + LSTM ensemble for churn prediction.",
        epilog="Example: python3 score_ensemble.py Churn_Pred_0401.csv",
    )
    parser.add_argument(
        "input_file",
        help="Input CSV file (Churn_Pred_XXXX.csv format)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output CSV filename (default: auto-derived from input)",
    )
    args = parser.parse_args()

    t_start = time.time()

    # --- Validate input file ---
    input_path = args.input_file
    if not os.path.isabs(input_path):
        input_path = os.path.join(os.getcwd(), input_path)

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_file = args.output if args.output else derive_output_filename(input_path)

    print("=" * 60)
    print("  Ensemble Scoring Pipeline (RF v3 + LSTM)")
    print("=" * 60)
    print(f"  Input:  {os.path.basename(input_path)}")
    print(f"  Output: {output_file}")

    # --- Step 1: Load models ---
    print("\n[1/4] Loading models...")
    t0 = time.time()
    try:
        rf_model, feature_cols, lstm_model, w_rf, w_lstm = load_models(BASE_DIR)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  RF v3: {len(feature_cols)} features")
    print(f"  LSTM:  loaded")
    print(f"  Weights: RF={w_rf:.2f}, LSTM={w_lstm:.2f}")
    print(f"  ({time.time() - t0:.1f}s)")

    # --- Step 2: Load and prepare data ---
    print("\n[2/4] Loading data...")
    t0 = time.time()
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Rows: {len(df):,}")

    item_cols = [c for c in df.columns if c.startswith("item")]
    if not item_cols:
        print("ERROR: No item columns found in input file.", file=sys.stderr)
        sys.exit(1)

    # For Churn_Pred files, ALL item columns are features (no target column)
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
    print(f"  Item columns: {len(item_cols)}")
    print(f"  Eligible (tenure >= {MIN_TENURE}): {n_eligible:,}")
    print(f"  New customers (tenure < {MIN_TENURE}): {n_new:,}")
    print(f"  ({time.time() - t0:.1f}s)")

    # --- Step 3: Score eligible customers ---
    print("\n[3/4] Scoring eligible customers...")
    t0 = time.time()

    # Initialize result arrays
    all_scores = np.full(len(df), -1.0)

    if n_eligible > 0:
        df_elig = df[valid].copy()
        mat_elig = mat[valid].copy()
        mat_vals_elig = mat_vals[valid]
        reg_elig = reg[valid]

        # RF probabilities
        print("  Computing RF probabilities...")
        rf_probs = compute_rf_probabilities(
            df_elig, mat_elig, feat_item_cols, rf_model, feature_cols
        )

        # LSTM probabilities
        print("  Computing LSTM probabilities...")
        lstm_probs = compute_lstm_probabilities(mat_vals_elig, reg_elig, lstm_model)

        # Ensemble
        ensemble_probs = w_rf * rf_probs + w_lstm * lstm_probs
        ensemble_scores = np.clip(np.round(ensemble_probs * 100, 1), 0, 100)

        all_scores[valid] = ensemble_scores

    print(f"  ({time.time() - t0:.1f}s)")

    # --- Step 4: Build output ---
    print("\n[4/4] Building output...")
    t0 = time.time()

    result_df = pd.DataFrame(
        {
            "userNo": df["userNo"].values,
            "churn_score": all_scores,
            "risk_level": [assign_risk_level(s) for s in all_scores],
        }
    )
    result_df["recommended_action"] = result_df["risk_level"].map(RECOMMENDED_ACTIONS)

    result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"  Saved: {output_file}")
    print(f"  ({time.time() - t0:.1f}s)")

    # --- Summary ---
    print_summary(result_df)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Done.")


if __name__ == "__main__":
    main()
