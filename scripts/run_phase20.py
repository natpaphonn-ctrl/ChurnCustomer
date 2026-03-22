"""
Phase 20: Probability Calibration
==================================
Calibrate ensemble (RF v3 + LSTM) probabilities using Isotonic Regression.

- Fit calibrator on val_files (index 20-22)
- Evaluate on test_files (index 23-24)
- Generate reliability diagram and score distribution
- Save calibrator to models/calibrator_isotonic.joblib
"""

import sys
import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MIN_TENURE = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_25_files = [
    'Churn_2025_0216_0301.csv', 'Churn_2025_0301_0316.csv',
    'Churn_2025_0316_0401.csv', 'Churn_2025_0401_0416.csv',
    'Churn_2025_0416_0502.csv', 'Churn_2025_0502_0516.csv',
    'Churn_2025_0516_0601.csv', 'Churn_2025_0601_0616.csv',
    'Churn_2025_0616_0701.csv', 'Churn_2025_0701_0716.csv',
    'Churn_2025_0716_0801.csv', 'Churn_2025_0801_0816.csv',
    'Churn_2025_0816_0901.csv', 'Churn_2025_0901_0916.csv',
    'Churn_2025_0916_1001.csv', 'Churn_2025_1001_1016.csv',
    'Churn_2025_1016_1101.csv', 'Churn_2025_1101_1116.csv',
    'Churn_2025_1116_1201.csv', 'Churn_2025_1201_1216.csv',
    'Churn_2025_1216_0102.csv', 'Churn_2026_0102_0117.csv',
    'Churn_2026_0117_0201.csv',
    'Churn_2026_0201_0216.csv', 'Churn_2026_0216_0301.csv',
]
val_files = all_25_files[20:23]
test_files = all_25_files[23:25]

# ---------------------------------------------------------------------------
# LSTM model definition
# ---------------------------------------------------------------------------
class ChurnLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(hidden).squeeze(-1)


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------
def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_pred = y_prob[mask].mean()
            avg_true = y_true[mask].mean()
            ece += mask.sum() / len(y_true) * abs(avg_pred - avg_true)
    return ece


# ---------------------------------------------------------------------------
# Reliability diagram data
# ---------------------------------------------------------------------------
def reliability_data(y_true, y_prob, n_bins=10):
    """Return (mean_pred, mean_true, counts) per bin for reliability diagrams."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mean_preds, mean_trues, counts = [], [], []
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() > 0:
            mean_preds.append(y_prob[mask].mean())
            mean_trues.append(y_true[mask].mean())
            counts.append(mask.sum())
        else:
            mean_preds.append(np.nan)
            mean_trues.append(np.nan)
            counts.append(0)
    return np.array(mean_preds), np.array(mean_trues), np.array(counts)


# ---------------------------------------------------------------------------
# Data loading and feature extraction
# ---------------------------------------------------------------------------
sys.path.insert(0, 'models')
from feature_engineering import engineer_features_v2


def load_and_prepare(csv_path):
    """Load a CSV, filter by tenure, return (df_valid, mat_valid, feat_item_cols, y)."""
    df = pd.read_csv(csv_path)
    item_cols = [c for c in df.columns if c.startswith('item')]
    feat_item_cols = item_cols[:-1]
    target_col = item_cols[-1]

    mat = df[item_cols].apply(pd.to_numeric, errors='coerce')
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


def get_rf_probs(df_valid, mat_valid, feat_item_cols, rf_model, feature_cols):
    """Get RF predicted probabilities."""
    feats = engineer_features_v2(df_valid, mat_valid, feat_item_cols)
    probs = rf_model.predict_proba(feats[feature_cols].fillna(0))[:, 1]
    return probs


def get_lstm_probs(mat_valid, feat_item_cols, lstm_model):
    """Get LSTM predicted probabilities."""
    mat_feat = mat_valid[feat_item_cols].values.astype(float)
    sequences = []
    lengths = []
    for i in range(len(mat_feat)):
        row = mat_feat[i]
        reg = ~np.isnan(row)
        if reg.any():
            start = np.argmax(reg)
            seq = row[start:]
            seq = np.where(np.isnan(seq), 0.0, seq)
            seq = np.log1p(seq)
        else:
            seq = np.array([0.0])
        sequences.append(seq)
        lengths.append(len(seq))

    max_len = max(lengths)
    padded = np.zeros((len(sequences), max_len, 1), dtype=np.float32)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq), 0] = seq

    x_tensor = torch.FloatTensor(padded).to(DEVICE)
    len_tensor = torch.LongTensor(lengths).to(DEVICE)

    lstm_model.eval()
    with torch.no_grad():
        batch_size = 4096
        all_probs = []
        for start in range(0, len(x_tensor), batch_size):
            end = min(start + batch_size, len(x_tensor))
            logits = lstm_model(x_tensor[start:end], len_tensor[start:end])
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs)


def get_ensemble_probs(csv_path, rf_model, feature_cols, lstm_model, w_rf, w_lstm):
    """Load file and compute ensemble probabilities."""
    df_valid, mat_valid, feat_item_cols, y = load_and_prepare(csv_path)

    rf_probs = get_rf_probs(df_valid, mat_valid, feat_item_cols, rf_model, feature_cols)
    lstm_probs = get_lstm_probs(mat_valid, feat_item_cols, lstm_model)

    ensemble_probs = w_rf * rf_probs + w_lstm * lstm_probs
    return y, ensemble_probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    print("=" * 60)
    print("Phase 20: Probability Calibration")
    print("=" * 60)

    # --- Load models ---
    t0 = time.time()
    print("\n[1/6] Loading models...")

    v3_dict = joblib.load('models/churn_v3_extended.joblib')
    rf_model = v3_dict['model']
    feature_cols = v3_dict['feature_cols']
    print(f"  RF v3 loaded — {len(feature_cols)} features")

    with open('models/ensemble_config.json', 'r') as f:
        ensemble_cfg = json.load(f)
    w_rf = ensemble_cfg['w_rf']
    w_lstm = ensemble_cfg['w_lstm']
    print(f"  Ensemble weights: RF={w_rf:.2f}, LSTM={w_lstm:.2f}")

    lstm_params = ensemble_cfg['lstm_params']
    lstm_model = ChurnLSTM(
        input_size=lstm_params['input_size'],
        hidden_size=lstm_params['hidden_size'],
        num_layers=lstm_params['num_layers'],
        dropout=lstm_params['dropout'],
    ).to(DEVICE)
    lstm_model.load_state_dict(torch.load('models/lstm_churn.pt', map_location=DEVICE))
    lstm_model.eval()
    print(f"  LSTM loaded — device={DEVICE}")
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Compute ensemble probs on val set (for fitting calibrator) ---
    t0 = time.time()
    print(f"\n[2/6] Computing ensemble probs on val set ({len(val_files)} files)...")
    val_y_list, val_prob_list = [], []
    for fname in val_files:
        path = fname
        if not os.path.exists(path):
            path = os.path.join('data', fname)
        print(f"  Processing {fname}...", end=' ')
        y, probs = get_ensemble_probs(path, rf_model, feature_cols, lstm_model, w_rf, w_lstm)
        val_y_list.append(y)
        val_prob_list.append(probs)
        print(f"n={len(y):,}, churn_rate={y.mean():.3f}")

    val_y = np.concatenate(val_y_list)
    val_probs = np.concatenate(val_prob_list)
    print(f"  Val total: n={len(val_y):,}, churn_rate={val_y.mean():.3f}")
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Compute ensemble probs on test set (for evaluation) ---
    t0 = time.time()
    print(f"\n[3/6] Computing ensemble probs on test set ({len(test_files)} files)...")
    test_y_list, test_prob_list = [], []
    for fname in test_files:
        path = fname
        if not os.path.exists(path):
            path = os.path.join('data', fname)
        print(f"  Processing {fname}...", end=' ')
        y, probs = get_ensemble_probs(path, rf_model, feature_cols, lstm_model, w_rf, w_lstm)
        test_y_list.append(y)
        test_prob_list.append(probs)
        print(f"n={len(y):,}, churn_rate={y.mean():.3f}")

    test_y = np.concatenate(test_y_list)
    test_probs = np.concatenate(test_prob_list)
    print(f"  Test total: n={len(test_y):,}, churn_rate={test_y.mean():.3f}")
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Fit Isotonic Regression calibrator on val set ---
    t0 = time.time()
    print("\n[4/6] Fitting Isotonic Regression calibrator on val set...")
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(val_probs, val_y)
    print(f"  Calibrator fitted on {len(val_probs):,} samples")
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Evaluate on test set ---
    t0 = time.time()
    print("\n[5/6] Evaluating calibration on test set...")

    test_probs_calibrated = calibrator.predict(test_probs)

    ece_before = compute_ece(test_y, test_probs)
    ece_after = compute_ece(test_y, test_probs_calibrated)

    auc_before = roc_auc_score(test_y, test_probs)
    auc_after = roc_auc_score(test_y, test_probs_calibrated)

    ap_before = average_precision_score(test_y, test_probs)
    ap_after = average_precision_score(test_y, test_probs_calibrated)

    f1_before = f1_score(test_y, (test_probs >= 0.5).astype(int))
    f1_after = f1_score(test_y, (test_probs_calibrated >= 0.5).astype(int))

    print(f"  ECE  before: {ece_before:.4f}  |  after: {ece_after:.4f}  "
          f"({(1 - ece_after / ece_before) * 100:+.1f}% improvement)")
    print(f"  AUC  before: {auc_before:.4f}  |  after: {auc_after:.4f}")
    print(f"  AP   before: {ap_before:.4f}  |  after: {ap_after:.4f}")
    print(f"  F1   before: {f1_before:.4f}  |  after: {f1_after:.4f}")
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Generate reliability diagram ---
    t0 = time.time()
    print("\n[6/6] Generating reliability diagram...")

    n_bins = 10
    mp_before, mt_before, cnt_before = reliability_data(test_y, test_probs, n_bins)
    mp_after, mt_after, cnt_after = reliability_data(test_y, test_probs_calibrated, n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    valid_before = ~np.isnan(mp_before)
    valid_after = ~np.isnan(mp_after)
    ax.plot(mp_before[valid_before], mt_before[valid_before],
            's-', color='#e74c3c', linewidth=2, markersize=8,
            label=f'Before (ECE={ece_before:.4f})')
    ax.plot(mp_after[valid_after], mt_after[valid_after],
            'o-', color='#2ecc71', linewidth=2, markersize=8,
            label=f'After (ECE={ece_after:.4f})')
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives (Actual)', fontsize=12)
    ax.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    # Right: Score distribution histogram
    ax = axes[1]
    bins = np.linspace(0, 1, 31)
    ax.hist(test_probs, bins=bins, alpha=0.5, color='#e74c3c',
            label='Before calibration', density=True, edgecolor='white')
    ax.hist(test_probs_calibrated, bins=bins, alpha=0.5, color='#2ecc71',
            label='After calibration', density=True, edgecolor='white')
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Score Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('charts', exist_ok=True)
    chart_path = 'charts/phase20_calibration.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {chart_path}")
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Save calibrator ---
    calibrator_path = 'models/calibrator_isotonic.joblib'
    joblib.dump(calibrator, calibrator_path)
    print(f"\nSaved calibrator to {calibrator_path}")

    # --- Update ensemble_config.json ---
    ensemble_cfg['calibrator_path'] = 'models/calibrator_isotonic.joblib'
    with open('models/ensemble_config.json', 'w') as f:
        json.dump(ensemble_cfg, f, indent=2)
    print("Updated models/ensemble_config.json with calibrator_path")

    # --- Summary ---
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("Phase 20: Probability Calibration — Summary")
    print("=" * 60)
    print(f"  Calibration method  : Isotonic Regression")
    print(f"  Val samples         : {len(val_y):,}")
    print(f"  Test samples        : {len(test_y):,}")
    print(f"  ECE before          : {ece_before:.4f}")
    print(f"  ECE after           : {ece_after:.4f}")
    print(f"  ECE improvement     : {(1 - ece_after / ece_before) * 100:.1f}%")
    print(f"  AUC-ROC (unchanged) : {auc_before:.4f} -> {auc_after:.4f}")
    print(f"  Chart               : {chart_path}")
    print(f"  Calibrator          : {calibrator_path}")
    print(f"  Total time          : {elapsed:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
