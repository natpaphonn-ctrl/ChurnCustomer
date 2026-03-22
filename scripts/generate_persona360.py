"""
Generate persona360 segments for 0401 scoring period.
- Task 1: Customer segment labels (rule-based)
- Task 2: CLV priority
- Task 3: Previous period (0316) score comparison
Output: data/predictions/persona360_segments.json
"""

import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.feature_engineering import engineer_features_v2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Load data ──
print("Loading 0401 risk scores...")
rs401 = pd.read_csv(os.path.join(ROOT, 'data/predictions/Churn_RiskScore_2026_0401.csv'),
                     encoding='utf-8-sig')
print(f"  Risk scores: {len(rs401):,} rows, columns: {list(rs401.columns[:6])}")

print("Loading Churn_Pred_0401.csv...")
pred = pd.read_csv(os.path.join(ROOT, 'data/predictions/Churn_Pred_0401.csv'),
                    encoding='utf-8-sig')
print(f"  Pred: {len(pred):,} rows, {len(pred.columns)} columns")

# ── Engineer features from purchase history ──
item_cols = [c for c in pred.columns if c.startswith('item')]
print(f"  Item columns: {len(item_cols)} ({item_cols[0]} ... {item_cols[-1]})")

# All item columns are features (no target in pred file)
feat_item_cols = item_cols
mat = pred[['userNo'] + feat_item_cols].set_index('userNo')
df_in = pred.set_index('userNo')

print("Engineering features (v2)...")
feats = engineer_features_v2(df_in, mat, feat_item_cols)
feats.index = mat.index
print(f"  Features: {feats.shape}")

# ── Task 1: Segmentation (rule-based, priority order) ──
print("\nTask 1: Segmenting customers...")

tenure = feats['tenure_rounds'].values
pf = feats['purchase_frequency_ratio'].values
al6 = feats['active_last_6'].values
il3 = feats['items_last_3'].values

segments = np.full(len(feats), '', dtype=object)

# Priority order (first match wins)
# 1. Power Buyer
mask = (il3 >= 15) & (al6 >= 5)
segments[mask & (segments == '')] = 'ตัวยง'

# 2. Loyal Regular
mask = (tenure >= 30) & (pf >= 0.7) & (al6 >= 5)
segments[mask & (segments == '')] = 'ขาประจำ'

# 3. Fading (before Occasional to catch fading customers)
mask = (tenure >= 20) & (al6 <= 2)
segments[mask & (segments == '')] = 'กำลังหาย'

# 4. Newcomer
mask = tenure < 10
segments[mask & (segments == '')] = 'มาใหม่'

# 5. Occasional
mask = (tenure >= 10) & (pf >= 0.3) & (pf < 0.7) & (al6 >= 2)
segments[mask & (segments == '')] = 'ขาจร'

# 6. Sporadic
mask = (tenure >= 10) & (pf < 0.3)
segments[mask & (segments == '')] = 'ซื้อเป็นช่วง'

# 7. Remaining (catch-all: tenure>=10, pf>=0.7 but al6<5, etc.)
segments[segments == ''] = 'ขาจร'

feats['segment'] = segments

seg_counts = pd.Series(segments).value_counts()
print("  Segment distribution:")
for seg, cnt in seg_counts.items():
    print(f"    {seg}: {cnt:,} ({cnt/len(feats)*100:.1f}%)")

# ── Task 2: CLV Priority ──
print("\nTask 2: CLV Priority...")
total_items = feats['total_items'].values
tenure_vals = feats['tenure_rounds'].values

p80 = np.percentile(total_items, 80)
p40 = np.percentile(total_items, 40)
print(f"  total_items percentiles: p40={p40:.1f}, p80={p80:.1f}")

clv = np.full(len(feats), 'Low', dtype=object)
clv[(total_items >= p40) | (tenure_vals >= 15)] = 'Medium'
clv[(total_items >= p80) & (tenure_vals >= 20)] = 'High'

feats['clv'] = clv

clv_counts = pd.Series(clv).value_counts()
print("  CLV distribution:")
for c, cnt in clv_counts.items():
    print(f"    {c}: {cnt:,} ({cnt/len(feats)*100:.1f}%)")

# ── Task 3: Previous Period Comparison ──
print("\nTask 3: Previous period comparison...")
prev_path = os.path.join(ROOT, 'data/predictions/Churn_RiskScore_2026_0316.csv')
has_prev = os.path.exists(prev_path)

# Build lookup for 0401 scores
score_401 = rs401.set_index('userNo')['churn_score']

prev_scores = {}
if has_prev:
    print("  Found 0316 risk scores, loading...")
    rs316 = pd.read_csv(prev_path, encoding='utf-8-sig')
    # 0316 has churn_score_rf_multi
    score_col = 'churn_score_rf_multi' if 'churn_score_rf_multi' in rs316.columns else 'churn_score'
    print(f"  Using column: {score_col}")
    score_316 = rs316.set_index('userNo')[score_col]
    prev_scores = score_316.to_dict()
    print(f"  0316 customers: {len(prev_scores):,}")

    # Find overlap
    overlap = set(score_401.index) & set(score_316.index)
    print(f"  Overlap with 0401: {len(overlap):,}")
else:
    print("  0316 file not found, skipping comparison.")

# ── Build output JSON ──
print("\nBuilding output JSON...")
score_401_dict = score_401.to_dict()
user_list = feats.index.tolist()

result = {}
n_improved = 0
n_worsened = 0
n_stable = 0
n_no_prev = 0

for user in user_list:
    entry = {
        'seg': feats.loc[user, 'segment'],
        'clv': feats.loc[user, 'clv'],
    }

    s401 = score_401_dict.get(user)
    s316 = prev_scores.get(user)

    if s316 is not None and s401 is not None:
        try:
            s316_f = float(s316)
            s401_f = float(s401)
            sc = round(s401_f - s316_f, 1)
            entry['ps'] = round(s316_f, 1)
            entry['sc'] = sc
            if abs(sc) < 3:
                entry['dir'] = 'คงที่'
                n_stable += 1
            elif sc < 0:
                entry['dir'] = 'ดีขึ้น'
                n_improved += 1
            else:
                entry['dir'] = 'แย่ลง'
                n_worsened += 1
        except (ValueError, TypeError):
            entry['ps'] = None
            entry['sc'] = None
            entry['dir'] = None
            n_no_prev += 1
    else:
        entry['ps'] = None
        entry['sc'] = None
        entry['dir'] = None
        n_no_prev += 1

    result[user] = entry

print(f"\nTotal customers: {len(result):,}")
if has_prev:
    print(f"  ดีขึ้น (improved): {n_improved:,}")
    print(f"  แย่ลง (worsened): {n_worsened:,}")
    print(f"  คงที่ (stable): {n_stable:,}")
    print(f"  No previous score: {n_no_prev:,}")

# ── Save ──
out_path = os.path.join(ROOT, 'data/predictions/persona360_segments.json')
print(f"\nSaving to {out_path}...")
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, separators=(',', ':'))

file_size = os.path.getsize(out_path) / (1024 * 1024)
print(f"  File size: {file_size:.1f} MB")
print("Done!")
