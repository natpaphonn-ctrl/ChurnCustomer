"""Merge additional JSON files into persona360_0401.json."""
import json
import os

BASE = '/Users/nongnat/Desktop/ChurnCustomer/data/predictions'

# Load main file
with open(os.path.join(BASE, 'persona360_0401.json'), 'r') as f:
    main = json.load(f)

# Load additional files
with open(os.path.join(BASE, 'persona360_prize.json'), 'r') as f:
    prize = json.load(f)
with open(os.path.join(BASE, 'persona360_txbehavior.json'), 'r') as f:
    txbeh = json.load(f)
with open(os.path.join(BASE, 'persona360_segments.json'), 'r') as f:
    segs = json.load(f)
with open(os.path.join(BASE, 'persona360_recommendations.json'), 'r') as f:
    recs = json.load(f)

print(f"Main data: {len(main['d'])} customers")
print(f"Prize data: {len(prize)} customers")
print(f"TX behavior: {len(txbeh)} customers")
print(f"Segments: {len(segs)} customers")
print(f"Recommendations: {len(recs)} customers")

# Convert main data from array format to dict format with named keys
# Current array: [score, riskIdx, age, genderIdx, provIdx, reg, tenure, freq, a3, a6, i3, slope, actionIdx, items, reasons]
# We need to keep the array as-is (for backward compat) and add new named fields

# Strategy: Change each entry from array to object with 'a' (array) + new fields
merged_count = {'prize': 0, 'txbeh': 0, 'segs': 0, 'recs': 0}

for uid in main['d']:
    arr = main['d'][uid]
    # Convert to object: keep array as 'a', add new data as named fields
    entry = {'a': arr}

    if uid in prize:
        entry['pr'] = prize[uid]
        merged_count['prize'] += 1

    if uid in txbeh:
        entry['tx'] = txbeh[uid]
        merged_count['txbeh'] += 1

    if uid in segs:
        entry['sg'] = segs[uid]
        merged_count['segs'] += 1

    if uid in recs:
        entry['rc2'] = recs[uid]  # rc2 to avoid conflict with existing 'rc' key
        merged_count['recs'] += 1

    main['d'][uid] = entry

print(f"\nMerge counts:")
for k, v in merged_count.items():
    print(f"  {k}: {v} customers matched")

# Save
out_path = os.path.join(BASE, 'persona360_0401.json')
with open(out_path, 'w') as f:
    json.dump(main, f, ensure_ascii=False, separators=(',', ':'))

print(f"\nSaved to {out_path}")
# Verify
with open(out_path, 'r') as f:
    check = json.load(f)
sample_key = list(check['d'].keys())[0]
print(f"Sample entry ({sample_key}): {list(check['d'][sample_key].keys())}")
print(f"File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")
