"""
Generate persona360_0401.json for the Persona 360 slide.
Reads 3 CSV files and produces a compact JSON with ALL 645K customers.

Compression strategies:
- Province names -> index into lookup table
- SHAP reason names -> index into lookup table
- Gender -> single char (M/F/O)
- Direction -> 0 (ลด) / 1 (เพิ่ม)
- Round numbers aggressively
- Items array: NaN->null, integers only
- Omit null/zero fields where possible
"""
import pandas as pd
import numpy as np
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(BASE, 'data', 'predictions')

# Short action map (index-based: 0-4)
ACTIONS = ['ไม่ต้องดำเนินการ', 'ส่ง offer เบาๆ', 'เร่งส่ง promotion', 'ติดต่อทันที', '-']
ACTION_MAP = {
    'ไม่ต้องดำเนินการ — ลูกค้าภักดี': 0,
    'ส่ง offer เบาๆ — รักษาความสัมพันธ์': 1,
    'เร่งส่ง promotion — เริ่มมีสัญญาณ Churn': 2,
    'ติดต่อทันที — โอกาส Churn สูงมาก': 3,
}

# Risk level short codes
RISK_MAP = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3, 'New Customer': 4}
RISK_LABELS = ['Low', 'Medium', 'High', 'Critical', 'New']

GENDER_MAP = {'MALE': 0, 'FEMALE': 1, 'OTHER': 2}

print("Loading CSVs...")
risk = pd.read_csv(os.path.join(PRED_DIR, 'Churn_RiskScore_2026_0401.csv'))
explained = pd.read_csv(os.path.join(PRED_DIR, 'Churn_RiskScore_Explained_2026_0401.csv'))
pred = pd.read_csv(os.path.join(PRED_DIR, 'Churn_Pred_0401.csv'))

print(f"  RiskScore: {len(risk):,} rows")
print(f"  Explained: {len(explained):,} rows")
print(f"  Pred:      {len(pred):,} rows")

# Build province lookup
print("Building province lookup...")
provinces_list = sorted(pred['province'].dropna().unique().tolist())
province_to_idx = {p: i for i, p in enumerate(provinces_list)}

# Build reason name lookup
print("Building reason name lookup...")
reason_names = set()
for i in range(1, 6):
    vals = explained[f'reason_{i}_th'].dropna().unique()
    reason_names.update(vals)
reason_names = sorted(reason_names)
reason_to_idx = {r: i for i, r in enumerate(reason_names)}

# Build explained lookup: userNo -> list of compact reason arrays
# Each reason: [name_idx, val_rounded, shap_rounded, dir(0=ลด,1=เพิ่ม)]
print("Building SHAP reasons lookup...")
explained_dict = {}
for _, row in explained.iterrows():
    reasons = []
    for i in range(1, 6):
        th = row.get(f'reason_{i}_th', '')
        if pd.isna(th) or th == '':
            continue
        val = row.get(f'reason_{i}_value', 0)
        shap = row.get(f'reason_{i}_shap', 0)
        direction = row.get(f'reason_{i}_direction', '')

        name_idx = reason_to_idx.get(th, -1)
        val_r = round(float(val), 2) if not pd.isna(val) else 0
        shap_r = round(float(shap), 3) if not pd.isna(shap) else 0
        dir_code = 1 if (not pd.isna(direction) and 'เพิ่ม' in str(direction)) else 0
        reasons.append([name_idx, val_r, shap_r, dir_code])
    explained_dict[row['userNo']] = reasons

# Build pred lookup
print("Building prediction data lookup...")
item_cols = [c for c in pred.columns if c.startswith('item')]
last_30_cols = item_cols[-30:]

pred_age = {}
pred_gender = {}
pred_province = {}
pred_reg = {}
pred_items = {}

for _, row in pred.iterrows():
    uid = row['userNo']
    pred_age[uid] = int(row['age']) if not pd.isna(row.get('age')) else None
    g = row.get('gender', '')
    pred_gender[uid] = GENDER_MAP.get(g, None) if not pd.isna(g) else None
    prov = row.get('province', '')
    pred_province[uid] = province_to_idx.get(prov, None) if not pd.isna(prov) else None
    reg = row.get('RegisteredRoundDate', '')
    pred_reg[uid] = str(reg) if not pd.isna(reg) else None

    items = []
    for c in last_30_cols:
        v = row[c]
        if pd.isna(v):
            items.append(None)
        else:
            items.append(int(v))
    pred_items[uid] = items

# Build final data dict
# Each entry: [score, risk_idx, age, gender_idx, province_idx, reg, tenure, freq, a3, a6, i3, slope, action_idx, hist, reasons]
print("Building output JSON...")
data = {}
for _, row in risk.iterrows():
    uid = row['userNo']
    score = round(float(row['churn_score']), 1) if not pd.isna(row['churn_score']) else -1
    risk_idx = RISK_MAP.get(row['risk_level'], 4)
    tenure = int(row['tenure_rounds']) if not pd.isna(row['tenure_rounds']) else 0
    freq = round(float(row['purchase_freq']), 3) if not pd.isna(row['purchase_freq']) else 0
    a3 = int(row['active_last_3']) if not pd.isna(row['active_last_3']) else 0
    a6 = int(row['active_last_6']) if not pd.isna(row['active_last_6']) else 0
    i3 = round(float(row['items_last_3']), 1) if not pd.isna(row['items_last_3']) else 0
    slope = round(float(row['trend_slope']), 4) if not pd.isna(row['trend_slope']) else 0

    action_full = row['recommended_action'] if not pd.isna(row['recommended_action']) else ''
    action_idx = ACTION_MAP.get(action_full, 4)

    age = pred_age.get(uid)
    gender = pred_gender.get(uid)
    province = pred_province.get(uid)
    reg = pred_reg.get(uid)
    items = pred_items.get(uid, [])
    reasons = explained_dict.get(uid, [])

    data[uid] = [score, risk_idx, age, gender, province, reg, tenure, freq, a3, a6, i3, slope, action_idx, items, reasons]

output = {
    "p": "0401",
    "t": len(data),
    "risks": RISK_LABELS,
    "actions": ACTIONS,
    "genders": ["M", "F", "O"],
    "provinces": provinces_list,
    "reasons": reason_names,
    "rc": {
        "Critical": int((risk.risk_level == 'Critical').sum()),
        "High": int((risk.risk_level == 'High').sum()),
        "Medium": int((risk.risk_level == 'Medium').sum()),
        "Low": int((risk.risk_level == 'Low').sum()),
        "New": int((risk.risk_level == 'New Customer').sum()),
    },
    "d": data
}

out_path = os.path.join(PRED_DIR, 'persona360_0401.json')
print(f"Writing to {out_path}...")
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, separators=(',', ':'))

size_mb = os.path.getsize(out_path) / (1024 * 1024)
print(f"Done! File size: {size_mb:.1f} MB ({len(data):,} customers)")
print(f"  Provinces: {len(provinces_list)} unique")
print(f"  Reason names: {len(reason_names)} unique")
