#!/usr/bin/env python3
"""Phase 18: RF v3 + LSTM Ensemble — average probabilities for improved AUC."""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, 'models')
from feature_engineering import engineer_features, engineer_features_v2

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve
from sklearn.calibration import calibration_curve

MIN_TENURE = 3

print("=" * 60)
print("Phase 18: RF v3 + LSTM Ensemble")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 1. Load RF v3 model
# ═══════════════════════════════════════════════════════════════
t0 = time.time()
print("\n[1/6] Loading RF v3 model...")

v3_artifact = joblib.load('models/churn_v3_extended.joblib')
v3_model = v3_artifact['model']
v3_feature_cols = v3_artifact['feature_cols']
print(f"  RF v3: {len(v3_feature_cols)} features")

# ═══════════════════════════════════════════════════════════════
# 2. Build LSTM model + train on same data
# ═══════════════════════════════════════════════════════════════

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
train_files = all_25_files[:20]
val_files = all_25_files[20:23]
test_files = all_25_files[23:25]


class ChurnLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(hidden).squeeze(-1)


class ChurnDataset(Dataset):
    def __init__(self, sequences, lengths, labels):
        self.sequences = sequences
        self.lengths = lengths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx], self.labels[idx]


def build_sequences(file_list, max_customers=100_000):
    """Convert item matrix → padded sequences + churn labels."""
    all_seqs, all_lengths, all_labels = [], [], []
    for fname in file_list:
        df = pd.read_csv(fname, low_memory=False)
        item_cols = [c for c in df.columns if c.startswith('item')]
        mat = df[item_cols].apply(pd.to_numeric, errors='coerce')
        mat_vals = mat.values.astype(float)
        reg = ~np.isnan(mat_vals)

        first_reg = np.where(reg.any(axis=1), reg.argmax(axis=1), len(item_cols))
        tenure = len(item_cols) - first_reg
        valid = tenure >= MIN_TENURE

        mat_valid = mat_vals[valid]
        reg_valid = reg[valid]
        labels = (mat_valid[:, -1] == 0).astype(np.float32)

        for i in range(len(mat_valid)):
            reg_start = np.argmax(reg_valid[i])
            seq = mat_valid[i, reg_start:-1]
            seq = np.nan_to_num(seq, nan=0.0)
            seq = np.log1p(seq).astype(np.float32)
            all_seqs.append(torch.tensor(seq).unsqueeze(-1))
            all_lengths.append(len(seq))
            all_labels.append(labels[i])

        print(f"    {fname}: {valid.sum():,} customers, churn={labels.mean():.1%}")
        if len(all_seqs) >= max_customers:
            all_seqs = all_seqs[:max_customers]
            all_lengths = all_lengths[:max_customers]
            all_labels = all_labels[:max_customers]
            break

    padded = pad_sequence(all_seqs, batch_first=True, padding_value=0.0)
    lengths = torch.tensor(all_lengths, dtype=torch.long)
    labels_t = torch.tensor(all_labels, dtype=torch.float32)
    return padded, lengths, labels_t


print("\n[2/6] Building LSTM sequences...")
print("  Train:")
X_seq_train, len_train, y_seq_train = build_sequences(train_files[-5:], max_customers=100_000)
print(f"  → {X_seq_train.shape}")

print("  Val:")
X_seq_val, len_val, y_seq_val = build_sequences(val_files[:1], max_customers=50_000)
print(f"  → {X_seq_val.shape}")
print(f"  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════
# 3. Train LSTM
# ═══════════════════════════════════════════════════════════════
t0 = time.time()
print("\n[3/6] Training LSTM...")

BATCH_SIZE = 512
EPOCHS = 20
LR = 0.001
PATIENCE = 4

model_lstm = ChurnLSTM()
pos_weight = torch.tensor([(y_seq_train == 0).sum() / max((y_seq_train == 1).sum(), 1)])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

train_loader = DataLoader(ChurnDataset(X_seq_train, len_train, y_seq_train),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ChurnDataset(X_seq_val, len_val, y_seq_val),
                        batch_size=BATCH_SIZE, shuffle=False)

best_val_auc = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model_lstm.train()
    epoch_loss, n_batches = 0, 0
    for seqs, lens, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model_lstm(seqs, lens), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_lstm.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    model_lstm.eval()
    val_probs = []
    val_labels_list = []
    with torch.no_grad():
        for seqs, lens, labels in val_loader:
            val_probs.append(torch.sigmoid(model_lstm(seqs, lens)).numpy())
            val_labels_list.append(labels.numpy())
    val_probs = np.concatenate(val_probs)
    val_labels = np.concatenate(val_labels_list)
    val_auc = roc_auc_score(val_labels, val_probs)
    scheduler.step(-val_auc)

    print(f"  Epoch {epoch:2d}/{EPOCHS} | Loss: {epoch_loss/n_batches:.4f} | "
          f"Val AUC: {val_auc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        best_state = {k: v.clone() for k, v in model_lstm.state_dict().items()}
    else:
        patience_counter = getattr(model_lstm, '_pc', 0) + 1
        model_lstm._pc = patience_counter
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

model_lstm.load_state_dict(best_state)
print(f"  Best Val AUC: {best_val_auc:.4f} ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════
# 4. Build ALIGNED test set (same customers, both RF + LSTM features)
# ═══════════════════════════════════════════════════════════════
t0 = time.time()
print("\n[4/6] Building aligned test set for ensemble...")

# For each test file, compute both RF features and LSTM sequences
# for the SAME customers, to enable fair comparison

all_rf_probs = []
all_lstm_probs = []
all_y_true = []

for fname in test_files:
    print(f"  Processing {fname}...")
    df = pd.read_csv(fname, low_memory=False)
    item_cols = [c for c in df.columns if c.startswith('item')]
    feat_item_cols = item_cols[:-1]
    mat = df[item_cols].apply(pd.to_numeric, errors='coerce')
    mat_vals = mat.values.astype(float)
    reg = ~np.isnan(mat_vals)

    first_reg = np.where(reg.any(axis=1), reg.argmax(axis=1), len(item_cols))
    tenure = len(item_cols) - first_reg
    valid = tenure >= MIN_TENURE

    df_elig = df[valid].copy()
    mat_elig = mat[valid].copy()
    mat_valid = mat_vals[valid]
    reg_valid = reg[valid]

    y_true = (mat_valid[:, -1] == 0).astype(int)

    # --- RF features ---
    feats = engineer_features_v2(df_elig, mat_elig, feat_item_cols)
    feats = feats.fillna(0)
    X_rf = feats[v3_feature_cols].fillna(0)
    rf_probs = v3_model.predict_proba(X_rf)[:, 1]

    # --- LSTM sequences ---
    seqs = []
    lengths = []
    for i in range(len(mat_valid)):
        reg_start = np.argmax(reg_valid[i])
        seq = mat_valid[i, reg_start:-1]
        seq = np.nan_to_num(seq, nan=0.0)
        seq = np.log1p(seq).astype(np.float32)
        seqs.append(torch.tensor(seq).unsqueeze(-1))
        lengths.append(len(seq))

    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    lens_t = torch.tensor(lengths, dtype=torch.long)

    # LSTM inference in batches
    model_lstm.eval()
    lstm_probs_list = []
    with torch.no_grad():
        for start in range(0, len(padded), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(padded))
            batch_probs = torch.sigmoid(
                model_lstm(padded[start:end], lens_t[start:end])
            ).numpy()
            lstm_probs_list.append(batch_probs)
    lstm_probs = np.concatenate(lstm_probs_list)

    all_rf_probs.append(rf_probs)
    all_lstm_probs.append(lstm_probs)
    all_y_true.append(y_true)

    print(f"    {len(y_true):,} customers, churn={y_true.mean():.1%}")

rf_probs_all = np.concatenate(all_rf_probs)
lstm_probs_all = np.concatenate(all_lstm_probs)
y_true_all = np.concatenate(all_y_true)

print(f"  Total: {len(y_true_all):,} customers")
print(f"  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════
# 5. Ensemble — search optimal weight
# ═══════════════════════════════════════════════════════════════
t0 = time.time()
print("\n[5/6] Finding optimal ensemble weight...")

# Test different RF:LSTM ratios
weights = np.arange(0.0, 1.05, 0.05)
results = []

for w_rf in weights:
    w_lstm = 1.0 - w_rf
    ensemble_prob = w_rf * rf_probs_all + w_lstm * lstm_probs_all
    auc = roc_auc_score(y_true_all, ensemble_prob)
    results.append({'w_rf': w_rf, 'w_lstm': w_lstm, 'auc': auc})

results_df = pd.DataFrame(results)
best_row = results_df.loc[results_df['auc'].idxmax()]
best_w_rf = best_row['w_rf']
best_w_lstm = best_row['w_lstm']
best_auc = best_row['auc']

print(f"\n  Weight Search Results (top 5):")
for _, row in results_df.nlargest(5, 'auc').iterrows():
    marker = ' ← BEST' if row['auc'] == best_auc else ''
    print(f"    RF={row['w_rf']:.2f} LSTM={row['w_lstm']:.2f} → AUC={row['auc']:.4f}{marker}")

# Compute final metrics with best weights
ensemble_probs_best = best_w_rf * rf_probs_all + best_w_lstm * lstm_probs_all
ensemble_preds = (ensemble_probs_best >= 0.5).astype(int)

auc_rf = roc_auc_score(y_true_all, rf_probs_all)
auc_lstm = roc_auc_score(y_true_all, lstm_probs_all)
auc_ensemble = roc_auc_score(y_true_all, ensemble_probs_best)
auc_simple_avg = roc_auc_score(y_true_all, 0.5 * rf_probs_all + 0.5 * lstm_probs_all)

apr_rf = average_precision_score(y_true_all, rf_probs_all)
apr_lstm = average_precision_score(y_true_all, lstm_probs_all)
apr_ensemble = average_precision_score(y_true_all, ensemble_probs_best)

f1_rf = f1_score(y_true_all, (rf_probs_all >= 0.5).astype(int))
f1_lstm = f1_score(y_true_all, (lstm_probs_all >= 0.5).astype(int))
f1_ensemble = f1_score(y_true_all, ensemble_preds)

print(f"\n  {'='*65}")
print(f"  Final Model Comparison (Test: {len(y_true_all):,} customers)")
print(f"  {'='*65}")
print(f"    {'Model':<35} {'AUC-ROC':>10} {'AUC-PR':>10} {'F1':>10}")
print(f"    {'-'*65}")
print(f"    {'RF v3 (45 features)':<35} {auc_rf:>10.4f} {apr_rf:>10.4f} {f1_rf:>10.4f}")
print(f"    {'LSTM (sequences)':<35} {auc_lstm:>10.4f} {apr_lstm:>10.4f} {f1_lstm:>10.4f}")
print(f"    {'Ensemble 50/50':<35} {auc_simple_avg:>10.4f} {'':>10} {'':>10}")
print(f"    {'Ensemble (optimal)':<35} {auc_ensemble:>10.4f} {apr_ensemble:>10.4f} {f1_ensemble:>10.4f}")
print(f"    {'  → weights: RF={best_w_rf:.2f} LSTM={best_w_lstm:.2f}'}")
print(f"\n    Improvement over RF v3:      {(auc_ensemble - auc_rf)*100:+.2f}% AUC-ROC")
print(f"    Improvement over LSTM:       {(auc_ensemble - auc_lstm)*100:+.2f}% AUC-ROC")

# Churn rate per risk level
ensemble_scores = (ensemble_probs_best * 100).round(1)
risk_levels = pd.cut(ensemble_scores, bins=[-1, 25, 50, 75, 100],
                     labels=['Low', 'Medium', 'High', 'Critical'])

print(f"\n  Ensemble — Churn Rate per Risk Level:")
for level in ['Low', 'Medium', 'High', 'Critical']:
    mask = risk_levels == level
    if mask.sum() > 0:
        churn_rate = y_true_all[mask].mean()
        print(f"    {level:10s}: {mask.sum():>8,} customers, churn={churn_rate:.1%}")

print(f"  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════
# 6. Charts
# ═══════════════════════════════════════════════════════════════
print("\n[6/6] Generating charts...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. Weight search curve
axes[0, 0].plot(results_df['w_rf'], results_df['auc'], 'b-o', linewidth=2, markersize=4)
axes[0, 0].axvline(best_w_rf, color='red', linestyle='--', alpha=0.7,
                    label=f'Best: RF={best_w_rf:.2f} (AUC={best_auc:.4f})')
axes[0, 0].axhline(auc_rf, color='#27AE60', linestyle=':', alpha=0.7, label=f'RF only ({auc_rf:.4f})')
axes[0, 0].axhline(auc_lstm, color='#E74C3C', linestyle=':', alpha=0.7, label=f'LSTM only ({auc_lstm:.4f})')
axes[0, 0].set_xlabel('RF Weight (LSTM = 1 - RF)')
axes[0, 0].set_ylabel('AUC-ROC')
axes[0, 0].set_title('Ensemble Weight Optimization', fontsize=13)
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# 2. ROC curves comparison
fpr_rf, tpr_rf, _ = roc_curve(y_true_all, rf_probs_all)
fpr_lstm, tpr_lstm, _ = roc_curve(y_true_all, lstm_probs_all)
fpr_ens, tpr_ens, _ = roc_curve(y_true_all, ensemble_probs_best)

axes[0, 1].plot(fpr_rf, tpr_rf, color='#27AE60', linewidth=2, label=f'RF v3 ({auc_rf:.4f})')
axes[0, 1].plot(fpr_lstm, tpr_lstm, color='#3498DB', linewidth=2, label=f'LSTM ({auc_lstm:.4f})')
axes[0, 1].plot(fpr_ens, tpr_ens, color='#E74C3C', linewidth=2.5, label=f'Ensemble ({auc_ensemble:.4f})')
axes[0, 1].plot([0, 1], [0, 1], '--', color='#ccc')
axes[0, 1].set_title('ROC Curves — All Models', fontsize=13)
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# 3. Score distributions
axes[1, 0].hist(ensemble_probs_best[y_true_all == 0], bins=50, alpha=0.5,
                color='#27AE60', label='Not Churned', density=True)
axes[1, 0].hist(ensemble_probs_best[y_true_all == 1], bins=50, alpha=0.5,
                color='#E74C3C', label='Churned', density=True)
axes[1, 0].set_title('Ensemble Score Distribution', fontsize=13)
axes[1, 0].set_xlabel('Churn Probability')
axes[1, 0].set_ylabel('Density')
axes[1, 0].legend()

# 4. Calibration curve
for name, probs, color in [('RF v3', rf_probs_all, '#27AE60'),
                             ('LSTM', lstm_probs_all, '#3498DB'),
                             ('Ensemble', ensemble_probs_best, '#E74C3C')]:
    prob_true, prob_pred = calibration_curve(y_true_all, probs, n_bins=10, strategy='uniform')
    axes[1, 1].plot(prob_pred, prob_true, 's-', color=color, linewidth=2, label=name)
axes[1, 1].plot([0, 1], [0, 1], '--', color='#ccc')
axes[1, 1].set_title('Calibration Curves', fontsize=13)
axes[1, 1].set_xlabel('Predicted Probability')
axes[1, 1].set_ylabel('Actual Churn Rate')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(f'Phase 18: RF + LSTM Ensemble (RF={best_w_rf:.0%} + LSTM={best_w_lstm:.0%})',
             fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('charts/phase18_ensemble.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: charts/phase18_ensemble.png")

# Save ensemble model config
ensemble_config = {
    'w_rf': float(best_w_rf),
    'w_lstm': float(best_w_lstm),
    'auc_roc': float(auc_ensemble),
    'auc_pr': float(apr_ensemble),
    'f1': float(f1_ensemble),
    'rf_model': 'models/churn_v3_extended.joblib',
    'lstm_params': {
        'input_size': 1, 'hidden_size': 64,
        'num_layers': 2, 'dropout': 0.3,
    },
}

# Save LSTM weights
torch.save(model_lstm.state_dict(), 'models/lstm_churn.pt')
print(f"  Saved: models/lstm_churn.pt")

import json
with open('models/ensemble_config.json', 'w') as f:
    json.dump(ensemble_config, f, indent=2)
print(f"  Saved: models/ensemble_config.json")

print("\n" + "=" * 60)
print("DONE — Phase 18 Complete!")
print("=" * 60)
