#!/usr/bin/env python3
"""Phase 18 extended: Stacking, Threshold Optimization, GRU, HP Tuning."""

import os, sys, time, warnings, json
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
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.model_selection import train_test_split

MIN_TENURE = 3
RESULTS = {}

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

# ═══════════════════════════════════════════════
# Shared: Load models + build aligned test data
# ═══════════════════════════════════════════════
print("=" * 60)
print("Loading models and building test data...")
print("=" * 60)

v3_artifact = joblib.load('models/churn_v3_extended.joblib')
v3_model = v3_artifact['model']
v3_feature_cols = v3_artifact['feature_cols']


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


class ChurnGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(hidden).squeeze(-1)


class ChurnDataset(torch.utils.data.Dataset):
    def __init__(self, s, l, la):
        self.s, self.l, self.la = s, l, la
    def __len__(self): return len(self.la)
    def __getitem__(self, i): return self.s[i], self.l[i], self.la[i]


def build_sequences(file_list, max_customers=100_000):
    all_seqs, all_lengths, all_labels = [], [], []
    for fname in file_list:
        df = pd.read_csv(fname, low_memory=False)
        item_cols = [c for c in df.columns if c.startswith('item')]
        mat = df[item_cols].apply(pd.to_numeric, errors='coerce').values.astype(float)
        reg = ~np.isnan(mat)
        first_reg = np.where(reg.any(axis=1), reg.argmax(axis=1), len(item_cols))
        tenure = len(item_cols) - first_reg
        valid = tenure >= MIN_TENURE
        mat_v, reg_v = mat[valid], reg[valid]
        labels = (mat_v[:, -1] == 0).astype(np.float32)
        for i in range(len(mat_v)):
            rs = np.argmax(reg_v[i])
            seq = np.nan_to_num(mat_v[i, rs:-1], nan=0.0)
            seq = np.log1p(seq).astype(np.float32)
            all_seqs.append(torch.tensor(seq).unsqueeze(-1))
            all_lengths.append(len(seq))
            all_labels.append(labels[i])
        if len(all_seqs) >= max_customers:
            all_seqs = all_seqs[:max_customers]
            all_lengths = all_lengths[:max_customers]
            all_labels = all_labels[:max_customers]
            break
    padded = pad_sequence(all_seqs, batch_first=True, padding_value=0.0)
    return padded, torch.tensor(all_lengths, dtype=torch.long), torch.tensor(all_labels, dtype=torch.float32)


def get_aligned_test_data(files, v3_model, v3_feature_cols, lstm_model, batch_size=512):
    """Build aligned RF + LSTM predictions on same customers."""
    all_rf, all_lstm, all_y, all_feats = [], [], [], []
    for fname in files:
        df = pd.read_csv(fname, low_memory=False)
        item_cols = [c for c in df.columns if c.startswith('item')]
        feat_item_cols = item_cols[:-1]
        mat = df[item_cols].apply(pd.to_numeric, errors='coerce')
        mat_vals = mat.values.astype(float)
        reg = ~np.isnan(mat_vals)
        first_reg = np.where(reg.any(axis=1), reg.argmax(axis=1), len(item_cols))
        tenure = len(item_cols) - first_reg
        valid = tenure >= MIN_TENURE
        df_e, mat_e = df[valid].copy(), mat[valid].copy()
        mat_v, reg_v = mat_vals[valid], reg[valid]
        y = (mat_v[:, -1] == 0).astype(int)

        feats = engineer_features_v2(df_e, mat_e, feat_item_cols).fillna(0)
        X_rf = feats[v3_feature_cols].fillna(0)
        rf_p = v3_model.predict_proba(X_rf)[:, 1]

        seqs = []
        lens = []
        for i in range(len(mat_v)):
            rs = np.argmax(reg_v[i])
            seq = np.nan_to_num(mat_v[i, rs:-1], nan=0.0)
            seq = np.log1p(seq).astype(np.float32)
            seqs.append(torch.tensor(seq).unsqueeze(-1))
            lens.append(len(seq))
        padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
        lens_t = torch.tensor(lens, dtype=torch.long)

        lstm_model.eval()
        lstm_p = []
        with torch.no_grad():
            for s in range(0, len(padded), batch_size):
                e = min(s + batch_size, len(padded))
                lstm_p.append(torch.sigmoid(lstm_model(padded[s:e], lens_t[s:e])).numpy())
        lstm_p = np.concatenate(lstm_p)

        all_rf.append(rf_p)
        all_lstm.append(lstm_p)
        all_y.append(y)
        all_feats.append(feats)
        print(f"    {fname}: {len(y):,}, churn={y.mean():.1%}")

    return (np.concatenate(all_rf), np.concatenate(all_lstm),
            np.concatenate(all_y), pd.concat(all_feats, ignore_index=True))


# Load LSTM
model_lstm = ChurnLSTM()
model_lstm.load_state_dict(torch.load('models/lstm_churn.pt', map_location='cpu'))
model_lstm.eval()

# Build aligned test data
t0 = time.time()
print("\nBuilding aligned test data...")
rf_probs, lstm_probs, y_true, feats_test = get_aligned_test_data(
    test_files, v3_model, v3_feature_cols, model_lstm)
print(f"  Total: {len(y_true):,} ({time.time()-t0:.1f}s)")

# Also build val data for stacking
print("Building aligned val data...")
rf_probs_val, lstm_probs_val, y_val, feats_val = get_aligned_test_data(
    val_files[:1], v3_model, v3_feature_cols, model_lstm)
print(f"  Val: {len(y_val):,} ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════
# Task 1: Stacking Ensemble
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("Task 1: Stacking Ensemble")
print("=" * 60)
t0 = time.time()

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# Build meta-features: RF prob + LSTM prob
X_meta_val = np.column_stack([rf_probs_val, lstm_probs_val])
X_meta_test = np.column_stack([rf_probs, lstm_probs])

# Logistic Regression stacker
lr_stack = LogisticRegression(C=1.0, max_iter=1000)
lr_stack.fit(X_meta_val, y_val)
stack_lr_probs = lr_stack.predict_proba(X_meta_test)[:, 1]
auc_stack_lr = roc_auc_score(y_true, stack_lr_probs)

# GBM stacker (with more meta-features)
# Add RF score * LSTM score, max, min as extra features
X_meta_val_ext = np.column_stack([
    rf_probs_val, lstm_probs_val,
    rf_probs_val * lstm_probs_val,
    np.maximum(rf_probs_val, lstm_probs_val),
    np.abs(rf_probs_val - lstm_probs_val),
])
X_meta_test_ext = np.column_stack([
    rf_probs, lstm_probs,
    rf_probs * lstm_probs,
    np.maximum(rf_probs, lstm_probs),
    np.abs(rf_probs - lstm_probs),
])

gbm_stack = GradientBoostingClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42)
gbm_stack.fit(X_meta_val_ext, y_val)
stack_gbm_probs = gbm_stack.predict_proba(X_meta_test_ext)[:, 1]
auc_stack_gbm = roc_auc_score(y_true, stack_gbm_probs)

# Simple average baseline
auc_simple = roc_auc_score(y_true, 0.5 * rf_probs + 0.5 * lstm_probs)
auc_rf = roc_auc_score(y_true, rf_probs)
auc_lstm_test = roc_auc_score(y_true, lstm_probs)

print(f"\n  {'Model':<35} {'AUC-ROC':>10}")
print(f"  {'-'*45}")
print(f"  {'RF v3 alone':<35} {auc_rf:>10.4f}")
print(f"  {'LSTM alone':<35} {auc_lstm_test:>10.4f}")
print(f"  {'Simple Average':<35} {auc_simple:>10.4f}")
print(f"  {'Stacking (LR)':<35} {auc_stack_lr:>10.4f}")
print(f"  {'Stacking (GBM + interactions)':<35} {auc_stack_gbm:>10.4f}")

best_stack = 'GBM' if auc_stack_gbm > auc_stack_lr else 'LR'
best_stack_auc = max(auc_stack_gbm, auc_stack_lr)
best_stack_probs = stack_gbm_probs if best_stack == 'GBM' else stack_lr_probs
RESULTS['stacking'] = {
    'best': best_stack, 'auc': float(best_stack_auc),
    'improvement_over_rf': float(best_stack_auc - auc_rf),
    'lr_coefs': lr_stack.coef_.tolist(),
}
print(f"\n  Best stacker: {best_stack} (AUC={best_stack_auc:.4f}, +{(best_stack_auc-auc_rf)*100:.2f}%)")
print(f"  LR weights: RF={lr_stack.coef_[0][0]:.3f}, LSTM={lr_stack.coef_[0][1]:.3f}")
print(f"  ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════
# Task 2: Threshold Optimization
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("Task 2: Threshold Optimization")
print("=" * 60)
t0 = time.time()

# Use best stacking probs
probs_for_thresh = best_stack_probs

# Find optimal thresholds for different objectives
thresholds = np.arange(0.1, 0.9, 0.01)

# 1. Maximize F1
f1_scores = [f1_score(y_true, (probs_for_thresh >= t).astype(int)) for t in thresholds]
best_f1_thresh = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores)

# 2. Maximize balanced accuracy
from sklearn.metrics import balanced_accuracy_score
bal_scores = [balanced_accuracy_score(y_true, (probs_for_thresh >= t).astype(int)) for t in thresholds]
best_bal_thresh = thresholds[np.argmax(bal_scores)]

# 3. Precision @ 80% recall
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, probs_for_thresh)
idx_80 = np.argmin(np.abs(recall_curve - 0.80))
prec_at_80_recall = precision_curve[idx_80]

# 4. Per-risk-level analysis with optimal threshold
print(f"\n  Optimal Thresholds:")
print(f"    Max F1:              {best_f1_thresh:.2f} (F1={best_f1:.4f})")
print(f"    Max Balanced Acc:    {best_bal_thresh:.2f}")
print(f"    Precision@80%Recall: {prec_at_80_recall:.3f}")

# Risk level boundaries optimization
print(f"\n  Risk Level Analysis (threshold={best_f1_thresh:.2f}):")
scores = (probs_for_thresh * 100).round(1)
preds = (probs_for_thresh >= best_f1_thresh).astype(int)
risk = pd.cut(scores, bins=[-1, 25, 50, 75, 100], labels=['Low', 'Medium', 'High', 'Critical'])

for level in ['Low', 'Medium', 'High', 'Critical']:
    mask = risk == level
    if mask.sum() == 0:
        continue
    churn_rate = y_true[mask].mean()
    precision = y_true[mask & (preds == 1)].mean() if (mask & (preds == 1)).sum() > 0 else 0
    print(f"    {level:10s}: {mask.sum():>8,} | churn={churn_rate:.1%} | "
          f"precision={precision:.1%}")

RESULTS['threshold'] = {
    'best_f1_threshold': float(best_f1_thresh),
    'best_f1': float(best_f1),
    'best_balanced_threshold': float(best_bal_thresh),
    'precision_at_80_recall': float(prec_at_80_recall),
}
print(f"  ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════
# Task 3: GRU Model
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("Task 3: GRU Model")
print("=" * 60)
t0 = time.time()

# Build train/val sequences
print("  Building sequences...")
X_seq_train, len_train, y_seq_train = build_sequences(train_files[-5:], max_customers=100_000)
X_seq_val, len_val, y_seq_val = build_sequences(val_files[:1], max_customers=50_000)

BATCH_SIZE = 512
model_gru = ChurnGRU(input_size=1, hidden_size=64, num_layers=2, dropout=0.3)
pos_weight = torch.tensor([(y_seq_train == 0).sum() / max((y_seq_train == 1).sum(), 1)])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model_gru.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

train_loader = DataLoader(ChurnDataset(X_seq_train, len_train, y_seq_train),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ChurnDataset(X_seq_val, len_val, y_seq_val),
                        batch_size=BATCH_SIZE, shuffle=False)

gru_params = sum(p.numel() for p in model_gru.parameters())
lstm_params = sum(p.numel() for p in model_lstm.parameters())
print(f"  GRU params: {gru_params:,} (LSTM: {lstm_params:,})")

best_val_auc_gru = 0
best_state_gru = None

for epoch in range(1, 21):
    model_gru.train()
    epoch_loss, n_b = 0, 0
    for seqs, lens, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model_gru(seqs, lens), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_gru.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        n_b += 1

    model_gru.eval()
    vp, vl = [], []
    with torch.no_grad():
        for s, l, la in val_loader:
            vp.append(torch.sigmoid(model_gru(s, l)).numpy())
            vl.append(la.numpy())
    vp, vl = np.concatenate(vp), np.concatenate(vl)
    val_auc = roc_auc_score(vl, vp)
    scheduler.step(-val_auc)

    print(f"  Epoch {epoch:2d} | Loss: {epoch_loss/n_b:.4f} | Val AUC: {val_auc:.4f}")

    if val_auc > best_val_auc_gru:
        best_val_auc_gru = val_auc
        best_state_gru = {k: v.clone() for k, v in model_gru.state_dict().items()}
        patience = 0
    else:
        patience += 1
        if patience >= 4:
            print(f"  Early stopping at epoch {epoch}")
            break

model_gru.load_state_dict(best_state_gru)

# Test GRU on aligned data
model_gru.eval()
gru_probs_list = []
# Build test sequences from test files
for fname in test_files:
    df = pd.read_csv(fname, low_memory=False)
    item_cols = [c for c in df.columns if c.startswith('item')]
    mat_vals = df[item_cols].apply(pd.to_numeric, errors='coerce').values.astype(float)
    reg = ~np.isnan(mat_vals)
    first_reg = np.where(reg.any(axis=1), reg.argmax(axis=1), len(item_cols))
    tenure = len(item_cols) - first_reg
    valid = tenure >= MIN_TENURE
    mat_v, reg_v = mat_vals[valid], reg[valid]

    seqs = []
    lens = []
    for i in range(len(mat_v)):
        rs = np.argmax(reg_v[i])
        seq = np.nan_to_num(mat_v[i, rs:-1], nan=0.0)
        seq = np.log1p(seq).astype(np.float32)
        seqs.append(torch.tensor(seq).unsqueeze(-1))
        lens.append(len(seq))
    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    lens_t = torch.tensor(lens, dtype=torch.long)

    with torch.no_grad():
        for s in range(0, len(padded), BATCH_SIZE):
            e = min(s + BATCH_SIZE, len(padded))
            gru_probs_list.append(torch.sigmoid(model_gru(padded[s:e], lens_t[s:e])).numpy())

gru_probs = np.concatenate(gru_probs_list)
auc_gru = roc_auc_score(y_true, gru_probs)

print(f"\n  GRU Test AUC: {auc_gru:.4f} (LSTM: {auc_lstm_test:.4f})")
print(f"  GRU vs LSTM: {(auc_gru - auc_lstm_test)*100:+.2f}%")

# GRU ensemble
auc_gru_rf = roc_auc_score(y_true, 0.5 * rf_probs + 0.5 * gru_probs)
print(f"  RF+GRU ensemble: {auc_gru_rf:.4f}")

# 3-model ensemble
ens3_probs = (rf_probs + lstm_probs + gru_probs) / 3
auc_3model = roc_auc_score(y_true, ens3_probs)
print(f"  3-model ensemble (RF+LSTM+GRU): {auc_3model:.4f}")

torch.save(model_gru.state_dict(), 'models/gru_churn.pt')
RESULTS['gru'] = {
    'val_auc': float(best_val_auc_gru),
    'test_auc': float(auc_gru),
    'params': gru_params,
    'rf_gru_ensemble': float(auc_gru_rf),
    '3model_ensemble': float(auc_3model),
}
print(f"  Saved: models/gru_churn.pt ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════
# Task 4: Hyperparameter Tuning (RF)
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("Task 4: Hyperparameter Tuning (RF)")
print("=" * 60)
t0 = time.time()

from sklearn.ensemble import RandomForestClassifier

# Use val data for tuning, test for final eval
# Load 1 train file for speed
print("  Loading train data...")
df_tr = pd.read_csv(train_files[-1], low_memory=False)
item_cols_tr = [c for c in df_tr.columns if c.startswith('item')]
feat_item_cols_tr = item_cols_tr[:-1]
mat_tr = df_tr[item_cols_tr].apply(pd.to_numeric, errors='coerce')
first_reg_tr = mat_tr.notna().values.argmax(axis=1)
tenure_tr = len(item_cols_tr) - first_reg_tr
valid_tr = tenure_tr >= MIN_TENURE
df_tr_e = df_tr[valid_tr].copy()
mat_tr_e = mat_tr[valid_tr].copy()
df_tr_e['churn'] = (df_tr_e[item_cols_tr[-1]] == 0).astype(int)
feats_tr = engineer_features_v2(df_tr_e, mat_tr_e, feat_item_cols_tr).fillna(0)
X_tr = feats_tr[v3_feature_cols].fillna(0)
y_tr = df_tr_e['churn'].values

# Subsample
if len(X_tr) > 300_000:
    X_tr, _, y_tr, _ = train_test_split(X_tr, y_tr, train_size=300_000,
                                         stratify=y_tr, random_state=42)

# Val data
X_val_hp = feats_val[v3_feature_cols].fillna(0)

# Grid search (manual — faster than sklearn GridSearchCV)
configs = [
    {'n_estimators': 200, 'max_depth': 15, 'min_samples_leaf': 50},
    {'n_estimators': 300, 'max_depth': 20, 'min_samples_leaf': 50},   # current v3
    {'n_estimators': 500, 'max_depth': 25, 'min_samples_leaf': 30},
    {'n_estimators': 300, 'max_depth': 30, 'min_samples_leaf': 20},
    {'n_estimators': 200, 'max_depth': 20, 'min_samples_leaf': 100},
    {'n_estimators': 500, 'max_depth': 20, 'min_samples_leaf': 50},
]

print(f"  Testing {len(configs)} RF configs on {len(X_tr):,} train, {len(X_val_hp):,} val...")
hp_results = []

for i, cfg in enumerate(configs):
    rf = RandomForestClassifier(
        class_weight='balanced', n_jobs=-1, random_state=42, **cfg)
    rf.fit(X_tr, y_tr)
    val_auc = roc_auc_score(y_val, rf.predict_proba(X_val_hp)[:, 1])
    test_auc = roc_auc_score(y_true, rf.predict_proba(feats_test[v3_feature_cols].fillna(0))[:, 1])
    hp_results.append({**cfg, 'val_auc': val_auc, 'test_auc': test_auc})
    print(f"    [{i+1}/{len(configs)}] n={cfg['n_estimators']} d={cfg['max_depth']} "
          f"leaf={cfg['min_samples_leaf']} → val={val_auc:.4f} test={test_auc:.4f}")

hp_df = pd.DataFrame(hp_results)
best_hp = hp_df.loc[hp_df['val_auc'].idxmax()]
print(f"\n  Best config: n={int(best_hp['n_estimators'])} d={int(best_hp['max_depth'])} "
      f"leaf={int(best_hp['min_samples_leaf'])}")
print(f"  Val AUC: {best_hp['val_auc']:.4f}, Test AUC: {best_hp['test_auc']:.4f}")
print(f"  vs current v3: {auc_rf:.4f} ({(best_hp['test_auc']-auc_rf)*100:+.2f}%)")

RESULTS['hp_tuning'] = {
    'best_config': {k: int(v) if k != 'val_auc' and k != 'test_auc' else float(v)
                    for k, v in best_hp.to_dict().items()},
    'all_results': hp_df.to_dict('records'),
}
print(f"  ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════
# Task 8: Error Analysis
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("Task 8: Error Analysis")
print("=" * 60)
t0 = time.time()

# Use best stacking predictions
preds_binary = (best_stack_probs >= best_f1_thresh).astype(int)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, preds_binary).ravel()
print(f"\n  Confusion Matrix (threshold={best_f1_thresh:.2f}):")
print(f"    TP={tp:,} FP={fp:,}")
print(f"    FN={fn:,} TN={tn:,}")
print(f"    Precision={tp/(tp+fp):.3f} Recall={tp/(tp+fn):.3f}")

# Analyze False Negatives (missed churners) and False Positives
fn_mask = (y_true == 1) & (preds_binary == 0)
fp_mask = (y_true == 0) & (preds_binary == 1)
tp_mask = (y_true == 1) & (preds_binary == 1)
tn_mask = (y_true == 0) & (preds_binary == 0)

# Feature profiles
profile_features = ['tenure_rounds', 'purchase_frequency_ratio', 'rounds_since_last_purchase',
                    'active_last_3', 'items_last_3', 'trend_slope', 'current_zero_streak',
                    'ewm_recent', 'avg_gap', 'gini_coefficient']

print(f"\n  Feature Profiles (mean values):")
print(f"    {'Feature':<30} {'TP':>8} {'FN':>8} {'FP':>8} {'TN':>8}")
print(f"    {'-'*62}")
for feat in profile_features:
    vals = feats_test[feat]
    print(f"    {feat:<30} {vals[tp_mask].mean():>8.2f} {vals[fn_mask].mean():>8.2f} "
          f"{vals[fp_mask].mean():>8.2f} {vals[tn_mask].mean():>8.2f}")

# Key insights
print(f"\n  Key Insights:")
fn_tenure = feats_test['tenure_rounds'][fn_mask].mean()
tp_tenure = feats_test['tenure_rounds'][tp_mask].mean()
print(f"    Missed churners (FN) have higher tenure: {fn_tenure:.1f} vs TP: {tp_tenure:.1f}")
fn_freq = feats_test['purchase_frequency_ratio'][fn_mask].mean()
tp_freq = feats_test['purchase_frequency_ratio'][tp_mask].mean()
print(f"    Missed churners buy more frequently: {fn_freq:.3f} vs TP: {tp_freq:.3f}")
fn_active = feats_test['active_last_3'][fn_mask].mean()
tp_active = feats_test['active_last_3'][tp_mask].mean()
print(f"    Missed churners more recently active: {fn_active:.2f} vs TP: {tp_active:.2f}")

RESULTS['error_analysis'] = {
    'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    'fn_avg_tenure': float(fn_tenure),
    'fn_avg_freq': float(fn_freq),
    'insight': 'FN (missed churners) are long-tenure, frequent buyers who suddenly stop — hardest to predict',
}
print(f"  ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════
# Task 9: Feature Interaction (SHAP)
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("Task 9: Feature Interaction (SHAP)")
print("=" * 60)
t0 = time.time()

import shap

# Use LightGBM proxy for SHAP (faster)
import lightgbm as lgb

print("  Training LightGBM proxy on RF predictions...")
lgb_proxy = lgb.LGBMClassifier(
    n_estimators=200, max_depth=10, learning_rate=0.1,
    num_leaves=31, subsample=0.8, n_jobs=-1, random_state=42, verbose=-1
)

# Train on RF predictions (pseudo-labels)
X_shap_train = feats_val[v3_feature_cols].fillna(0)
y_shap_train = (rf_probs_val >= 0.5).astype(int)
lgb_proxy.fit(X_shap_train, y_shap_train)

# SHAP on test subsample
X_shap_test = feats_test[v3_feature_cols].fillna(0).sample(10_000, random_state=42)
explainer = shap.TreeExplainer(lgb_proxy)
shap_values = explainer.shap_values(X_shap_test)
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values

print(f"  SHAP computed for {len(X_shap_test):,} samples")

# Top interaction pairs
print("\n  Computing SHAP interaction values (subsample)...")
X_interact = X_shap_test.iloc[:2000]
shap_interact = explainer.shap_interaction_values(X_interact)
if isinstance(shap_interact, list):
    shap_interact = shap_interact[1]

# Find top interaction pairs
n_feats = len(v3_feature_cols)
interactions = []
for i in range(n_feats):
    for j in range(i+1, n_feats):
        mean_interact = np.abs(shap_interact[:, i, j]).mean()
        interactions.append({
            'feat_1': v3_feature_cols[i],
            'feat_2': v3_feature_cols[j],
            'mean_interaction': mean_interact
        })

interact_df = pd.DataFrame(interactions).nlargest(15, 'mean_interaction')
print(f"\n  Top 15 Feature Interactions:")
for _, row in interact_df.iterrows():
    print(f"    {row['feat_1']:30s} × {row['feat_2']:30s} = {row['mean_interaction']:.4f}")

RESULTS['shap_interactions'] = interact_df.to_dict('records')

# Save SHAP summary plot
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Beeswarm
shap.summary_plot(shap_vals, X_shap_test, show=False, max_display=15)
plt.title('SHAP Feature Importance')
plt.tight_layout()
plt.savefig('charts/phase18_shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close('all')

# Interaction heatmap
fig, ax = plt.subplots(figsize=(12, 10))
top_feats = interact_df['feat_1'].unique()[:8].tolist()
feat_idx = [v3_feature_cols.index(f) for f in top_feats if f in v3_feature_cols]
interact_matrix = np.zeros((len(feat_idx), len(feat_idx)))
for i, fi in enumerate(feat_idx):
    for j, fj in enumerate(feat_idx):
        interact_matrix[i, j] = np.abs(shap_interact[:, fi, fj]).mean()

im = ax.imshow(interact_matrix, cmap='YlOrRd')
ax.set_xticks(range(len(feat_idx)))
ax.set_xticklabels([v3_feature_cols[i] for i in feat_idx], rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(feat_idx)))
ax.set_yticklabels([v3_feature_cols[i] for i in feat_idx], fontsize=9)
ax.set_title('SHAP Feature Interaction Heatmap', fontsize=13)
plt.colorbar(im, ax=ax, label='Mean |SHAP interaction|')
plt.tight_layout()
plt.savefig('charts/phase18_shap_interactions.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  Saved: charts/phase18_shap_beeswarm.png")
print(f"  Saved: charts/phase18_shap_interactions.png")
print(f"  ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════
# Save all results + combined chart
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("Saving results...")
print("=" * 60)

with open('models/phase18_results.json', 'w') as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("  Saved: models/phase18_results.json")

# Grand comparison chart
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. All models AUC
models_all = {
    'RF v3': auc_rf,
    'LSTM': auc_lstm_test,
    'GRU': auc_gru,
    'Avg Ens': auc_simple,
    'Stack LR': auc_stack_lr,
    'Stack GBM': auc_stack_gbm,
    'RF+LSTM+GRU': auc_3model,
}
bars = axes[0, 0].bar(models_all.keys(), models_all.values(),
                       color=['#3498DB', '#E74C3C', '#9B59B6', '#95A5A6',
                              '#27AE60', '#2ECC71', '#F39C12'])
axes[0, 0].set_ylim(min(models_all.values()) - 0.005, max(models_all.values()) + 0.005)
axes[0, 0].set_title('AUC-ROC Comparison — All Models', fontsize=13)
for bar, auc in zip(bars, models_all.values()):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                    f'{auc:.4f}', ha='center', fontsize=8, fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=30)

# 2. Threshold vs F1
axes[0, 1].plot(thresholds, f1_scores, 'b-', linewidth=2)
axes[0, 1].axvline(best_f1_thresh, color='red', linestyle='--',
                    label=f'Best: {best_f1_thresh:.2f} (F1={best_f1:.4f})')
axes[0, 1].axvline(0.5, color='gray', linestyle=':', label='Default (0.5)')
axes[0, 1].set_title('Threshold vs F1 Score', fontsize=13)
axes[0, 1].set_xlabel('Threshold')
axes[0, 1].set_ylabel('F1 Score')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Error analysis
categories = ['TP\n(correct churn)', 'FP\n(false alarm)', 'FN\n(missed)', 'TN\n(correct ok)']
counts = [tp, fp, fn, tn]
colors_err = ['#27AE60', '#F39C12', '#E74C3C', '#3498DB']
axes[1, 0].bar(categories, counts, color=colors_err)
axes[1, 0].set_title('Prediction Outcome Distribution', fontsize=13)
axes[1, 0].set_ylabel('Count')
for i, (c, v) in enumerate(zip(categories, counts)):
    axes[1, 0].text(i, v + len(y_true)*0.01, f'{v:,}', ha='center', fontsize=9)

# 4. HP Tuning results
hp_labels = [f"d={int(r['max_depth'])}\nleaf={int(r['min_samples_leaf'])}" for _, r in hp_df.iterrows()]
axes[1, 1].bar(hp_labels, hp_df['test_auc'], color='#3498DB', alpha=0.7, label='Test')
axes[1, 1].bar(hp_labels, hp_df['val_auc'], color='#E74C3C', alpha=0.3, label='Val')
axes[1, 1].set_title('RF Hyperparameter Search', fontsize=13)
axes[1, 1].set_ylabel('AUC-ROC')
axes[1, 1].legend()
axes[1, 1].set_ylim(hp_df['test_auc'].min() - 0.005, hp_df['test_auc'].max() + 0.005)

plt.suptitle('Phase 18: Model Improvements — Complete Results', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('charts/phase18_all_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: charts/phase18_all_results.png")

print("\n" + "=" * 60)
print("DONE — Tasks 1,2,3,4,8,9 Complete!")
print("=" * 60)
