"""
Generate churn_patterns.json using K-Means clustering on behavioral features.
Clusters churned customers by purchase behavior, picks closest-to-centroid examples.

Usage: python3 scripts/generate_churn_patterns.py
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=RuntimeWarning)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHURN_DIR = os.path.join(BASE, 'data', 'churn')
PRED_DIR = os.path.join(BASE, 'data', 'predictions')

PERIODS = [
    ('0401', 'Churn_2025_0316_0401.csv'),
    ('0316', 'Churn_2025_0301_0316.csv'),
]

COLORS = ['#EF4444', '#F97316', '#F59E0B', '#10B981', '#0EA5E9',
          '#6366F1', '#A855F7', '#EC4899', '#8B5CF6', '#14B8A6']

# Domain-driven K=8: enough for actionable segmentation
FIXED_K = 8


def extract_features(raw_matrix):
    """Extract 8 behavioral features from 20-period purchase vectors."""
    n_customers, n_periods = raw_matrix.shape
    half = n_periods // 2
    features = np.zeros((n_customers, 8), dtype=np.float32)

    for i in range(n_customers):
        vals = raw_matrix[i]
        active = vals > 0
        active_count = active.sum()
        total = vals.sum()
        active_vals = vals[active]

        # 1. Activity ratio (0-1)
        features[i, 0] = active_count / n_periods

        # 2. Trend slope (normalized by mean)
        if active_count >= 3:
            slope = np.polyfit(np.arange(n_periods), vals, 1)[0]
            mean_val = vals.mean()
            features[i, 1] = slope / max(mean_val, 0.01)
        else:
            features[i, 1] = 0

        # 3. Consistency: 1/(1+CV) of active periods
        if active_count >= 2:
            cv = np.std(active_vals) / np.mean(active_vals)
            features[i, 2] = 1.0 / (1.0 + cv)
        else:
            features[i, 2] = 0

        # 4. First-half weight: proportion of total in first half
        if total > 0:
            features[i, 3] = vals[:half].sum() / total
        else:
            features[i, 3] = 0.5

        # 5. Volume (log-scaled)
        features[i, 4] = np.log1p(total)

        # 6. Burstiness: max / mean of active periods
        if active_count >= 2:
            features[i, 5] = np.max(active_vals) / np.mean(active_vals)
        else:
            features[i, 5] = 1.0

        # 7. Late acceleration: last 5 periods avg / first 5 periods avg
        first5 = vals[:5].mean()
        last5 = vals[-5:].mean()
        if first5 > 0:
            features[i, 6] = last5 / first5
        elif last5 > 0:
            features[i, 6] = 5.0  # cap: only late activity
        else:
            features[i, 6] = 1.0

        # 8. Avg gap between purchases
        active_indices = np.where(active)[0]
        if len(active_indices) >= 2:
            features[i, 7] = np.mean(np.diff(active_indices))
        else:
            features[i, 7] = n_periods

    return features


FEATURE_NAMES = ['activity', 'trend', 'consistency', 'first_half_wt',
                 'volume', 'burstiness', 'late_accel', 'avg_gap']


def label_cluster(feat_means):
    """Label cluster based on feature means. Returns (base_label, name_th, icon, desc, action).
    base_label is used for deduplication."""
    activity = feat_means[0]
    trend = feat_means[1]
    consistency = feat_means[2]
    first_half_wt = feat_means[3]
    volume = feat_means[4]
    burstiness = feat_means[5]
    late_accel = feat_means[6]
    avg_gap = feat_means[7]

    # Score each pattern and return the best match
    # Format: (score, base_label, name_th, icon, desc, action)
    candidates = []

    # --- VIP / Loyal ---
    if activity >= 0.6 and consistency > 0.5 and volume > 4.0:
        s = activity * 10 + consistency * 5 + volume
        candidates.append((s, 'vip', 'ขาประจำหายไป', '🏆',
            'ซื้อทุกงวดสม่ำเสมอ ยอดสูง แล้วหายไปเลย',
            'VIP! โทรสอบถามทันที — อาจย้ายแพลตฟอร์ม?'))

    # --- Sudden Stop (consistent then gone) ---
    if activity >= 0.5 and consistency > 0.45 and abs(trend) < 0.04:
        s = activity * 8 + consistency * 6
        candidates.append((s, 'sudden_stop', 'ซื้อสม่ำเสมอแล้วหาย', '🛑',
            'ซื้อสม่ำเสมอหลายงวดแล้วหยุดกะทันหัน',
            'ไม่มีสัญญาณก่อนหยุด — ติดต่อเชิงรุก'))

    # --- Gradual Decline ---
    if trend < -0.03 and first_half_wt > 0.52 and activity >= 0.3:
        s = abs(trend) * 20 + first_half_wt * 5
        candidates.append((s, 'decline', 'ค่อยๆ ลดลง', '📉',
            'ยอดซื้อลดลงต่อเนื่องจนหยุดซื้อ',
            'จับสัญญาณตั้งแต่ยอดลดครั้งที่ 2 — ส่ง offer เพิ่มยอดกลับ'))

    # --- Escalator (rising then gone) ---
    if trend > 0.05 and first_half_wt < 0.4 and activity >= 0.2:
        s = trend * 20 + (1 - first_half_wt) * 5
        candidates.append((s, 'escalator', 'ยอดขึ้นแล้วหาย', '📈',
            'ยอดซื้อเพิ่มขึ้นเรื่อยๆ แล้วจู่ๆ หายไป',
            'ยอดขึ้นไม่ได้แปลว่าจะอยู่ — ติดตามทันที'))

    # --- Spike & Gone (bursty) ---
    if burstiness > 2.3 and activity >= 0.3:
        s = burstiness * 3
        candidates.append((s, 'spike', 'พุ่งแล้วหาย', '🚀',
            'มียอดพุ่งสูงผิดปกติบางงวด ไม่ต่อเนื่อง',
            'อาจซื้อพิเศษ (ฝากซื้อ/เลขเด็ด) — ส่ง offer หลังซื้อ'))

    # --- New customer (recent, low activity) ---
    if activity < 0.2 and first_half_wt < 0.3:
        s = (1 - activity) * 5 + (1 - first_half_wt) * 3
        candidates.append((s, 'new', 'เพิ่งเริ่มซื้อ', '🌱',
            'เพิ่งเริ่มซื้อ 2-3 งวดล่าสุดแล้วหยุด',
            'ลูกค้าใหม่ที่หลุด — ส่ง welcome back ทันที'))

    # --- Gone long ago ---
    if activity < 0.2 and first_half_wt > 0.5:
        s = (1 - activity) * 5 + first_half_wt * 3
        candidates.append((s, 'ghost', 'หายไปนานแล้ว', '💨',
            'เคยซื้อช่วงแรกๆ แล้วหายไปตั้งนาน',
            'หายนานเกิน — ใช้ automated campaign'))

    # --- Sparse irregular ---
    if activity < 0.2 and avg_gap > 5:
        s = avg_gap
        candidates.append((s, 'sparse', 'แทบไม่เคยซื้อ', '👻',
            'ซื้อแค่ 2-3 งวดกระจายๆ ไม่เคยเป็นลูกค้าประจำ',
            'ลูกค้าขาจร — ใช้ automated campaign'))

    # --- On-off pattern ---
    if 0.2 <= activity < 0.5 and avg_gap > 2.5:
        s = avg_gap * 2
        candidates.append((s, 'onoff', 'ซื้อบ้างหยุดบ้าง', '🔀',
            'ซื้อไม่สม่ำเสมอ เว้นบ่อย',
            'ส่ง offer ตรงงวดที่เคยซื้อ'))

    # --- Moderate irregular ---
    if 0.3 <= activity < 0.6:
        s = activity * 3
        candidates.append((s, 'moderate', 'ซื้อบ่อยพอสมควร', '🔶',
            'ซื้อราวครึ่งหนึ่งของงวด ยอดไม่คงที่',
            'ส่ง offer ต่อเนื่องเพื่อสร้างความสม่ำเสมอ'))

    # --- Frequent but volatile ---
    if activity >= 0.5 and consistency < 0.5:
        s = activity * 3 + burstiness
        candidates.append((s, 'volatile', 'ซื้อบ่อยแต่ผันผวน', '⚡',
            'ซื้อหลายงวดแต่ยอดผันผวนมาก',
            'ส่ง offer ต่อเนื่องเพื่อสร้าง loyalty'))

    if candidates:
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1:]  # (base_label, name_th, icon, desc, action)

    # Default
    return ('other', 'อื่นๆ', '💫', 'ไม่เข้ากลุ่มไหนชัดเจน', 'ใช้ general retention campaign')


def deduplicate_labels(cluster_metas, features_per_cluster):
    """Resolve duplicate base_labels by adding distinguishing suffixes."""
    # Group by base_label
    label_groups = {}
    for c_idx, meta in cluster_metas.items():
        base = meta['base_label']
        if base not in label_groups:
            label_groups[base] = []
        label_groups[base].append(c_idx)

    for base, indices in label_groups.items():
        if len(indices) <= 1:
            continue

        # Find most distinguishing feature between duplicates
        feats = {idx: features_per_cluster[idx] for idx in indices}

        # Compare: what's most different?
        feat_arrays = np.array([feats[idx] for idx in indices])
        feat_range = feat_arrays.max(axis=0) - feat_arrays.min(axis=0)
        # Normalize by mean to get relative range
        feat_mean = feat_arrays.mean(axis=0)
        feat_mean[feat_mean == 0] = 1
        relative_range = feat_range / np.abs(feat_mean)
        best_feat_idx = np.argmax(relative_range)
        best_feat_name = FEATURE_NAMES[best_feat_idx]

        # Add distinguishing suffix
        suffixes = {
            'activity': ('ซื้อน้อยกว่า', 'ซื้อบ่อยกว่า'),
            'trend': ('ยอดลดลง', 'ยอดเพิ่มขึ้น'),
            'consistency': ('ผันผวน', 'สม่ำเสมอ'),
            'first_half_wt': ('ช่วงหลัง', 'ช่วงแรก'),
            'volume': ('ยอดน้อย', 'ยอดมาก'),
            'burstiness': ('คงที่', 'เป็นพักๆ'),
            'late_accel': ('ช้าลง', 'เร็วขึ้น'),
            'avg_gap': ('ถี่', 'ห่าง'),
        }

        low_suffix, high_suffix = suffixes.get(best_feat_name, ('แบบ A', 'แบบ B'))

        # Sort by the distinguishing feature
        sorted_idx = sorted(indices, key=lambda idx: feats[idx][best_feat_idx])
        for rank, idx in enumerate(sorted_idx):
            suffix = low_suffix if rank < len(sorted_idx) / 2 else high_suffix
            old_name = cluster_metas[idx]['name_th']
            cluster_metas[idx]['name_th'] = f"{old_name} ({suffix})"
            cluster_metas[idx]['desc'] = cluster_metas[idx]['desc'].replace(
                old_name, cluster_metas[idx]['name_th'], 1)

    return cluster_metas


def get_labels(item_cols, n=20):
    labels = []
    for c in item_cols[-n:]:
        parts = c.replace('item', '').split('_')
        if len(parts) == 3:
            labels.append(f"{parts[1]}/{parts[2]}")
    return labels


def process_period(period, fname):
    print(f"\n{'='*60}")
    print(f"Processing period {period}: {fname}")
    print(f"{'='*60}")
    df = pd.read_csv(os.path.join(CHURN_DIR, fname))
    item_cols = [c for c in df.columns if c.startswith('item')]
    N = len(item_cols)

    display_cols = item_cols[max(0, N-20):N]
    labels = get_labels(display_cols, len(display_cols))

    classify_cols = item_cols[max(0, N-21):N-1]
    n_periods = len(classify_cols)
    print(f"  Item columns: {N}, classify periods: {n_periods}")

    churned = df[df[item_cols[-1]] == 0].copy()
    total_churned = len(churned)
    print(f"  Total churned: {total_churned:,}")

    raw_matrix = churned[classify_cols].fillna(0).values.astype(np.float32)

    # Filter: at least 2 purchases
    active_counts = (raw_matrix > 0).sum(axis=1)
    has_purchase = active_counts >= 2
    raw_filtered = raw_matrix[has_purchase]
    churned_filtered = churned[has_purchase].copy()
    n_filtered = len(raw_filtered)
    n_excluded = total_churned - n_filtered
    print(f"  With 2+ purchases: {n_filtered:,} (excluded: {n_excluded:,})")

    # Extract & scale features
    print("  Extracting behavioral features...")
    features = extract_features(raw_filtered)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # K-Means with fixed K
    k = FIXED_K
    print(f"\n  Fitting MiniBatchKMeans K={k} on {n_filtered:,} customers...")
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=10000, n_init=20, max_iter=500)
    cluster_labels = km.fit_predict(X_scaled)

    # Silhouette score
    sil = silhouette_score(X_scaled, cluster_labels,
                           sample_size=min(30000, n_filtered), random_state=42)
    print(f"  Silhouette score: {sil:.4f}")

    # Sort clusters by size (descending)
    cluster_sizes = np.bincount(cluster_labels, minlength=k)
    sorted_indices = np.argsort(-cluster_sizes)
    remap = np.zeros(k, dtype=int)
    for new_idx, old_idx in enumerate(sorted_indices):
        remap[old_idx] = new_idx
    cluster_labels = remap[cluster_labels]
    cluster_sizes = cluster_sizes[sorted_indices]

    # Distances for example selection
    centroids_scaled = km.cluster_centers_[sorted_indices]
    distances = np.zeros(n_filtered, dtype=np.float32)
    for c_idx in range(k):
        mask = cluster_labels == c_idx
        if mask.sum() > 0:
            diff = X_scaled[mask] - centroids_scaled[c_idx]
            distances[mask] = np.sqrt((diff ** 2).sum(axis=1))

    # Label clusters
    cluster_meta = {}
    features_per_cluster = {}
    for c_idx in range(k):
        mask = cluster_labels == c_idx
        feat_means = features[mask].mean(axis=0)
        features_per_cluster[c_idx] = feat_means
        base_label, name_th, icon, desc, action = label_cluster(feat_means)
        color = COLORS[c_idx % len(COLORS)]
        cluster_meta[c_idx] = {
            'base_label': base_label,
            'name_th': name_th, 'icon': icon, 'color': color,
            'desc': f'{icon} {name_th}: {desc}',
            'action': f'🎯 Action: {action}'
        }

    # Deduplicate labels
    cluster_meta = deduplicate_labels(cluster_meta, features_per_cluster)

    print(f"\n  Cluster results (K={k}):")
    print(f"  {'─'*70}")
    for c_idx in range(k):
        meta = cluster_meta[c_idx]
        count = int(cluster_sizes[c_idx])
        pct = count / total_churned * 100
        feat_means = features_per_cluster[c_idx]
        feat_str = ' | '.join(f'{fn}={fv:.2f}' for fn, fv in zip(FEATURE_NAMES, feat_means))
        print(f"    {meta['icon']} [{c_idx}] {meta['name_th']}: {count:,} ({pct:.1f}%)")
        print(f"        {feat_str}")

    # Build examples
    userNos = churned_filtered['userNo'].values
    patterns_data = {}

    for c_idx in range(k):
        mask = cluster_labels == c_idx
        indices = np.where(mask)[0]
        dists = distances[indices]
        sort_order = np.argsort(dists)
        closest_indices = indices[sort_order[:30]]

        examples = []
        for idx in closest_indices:
            raw_vals = raw_filtered[idx].tolist()
            # Display: 19 pre-churn + 0 for churn period = 20 bars
            display_start = max(0, n_periods - 19)
            display_vals = list(raw_vals[display_start:]) + [0]
            if len(display_vals) < 20:
                display_vals = [0] * (20 - len(display_vals)) + display_vals
            display_vals = display_vals[-20:]

            uid = str(userNos[idx])
            masked = uid[:6] + 'xxx' if uid.startswith('P') else 'P' + uid[:5] + 'xxx'
            active_vals = [v for v in raw_vals if v > 0]
            examples.append({
                'uid': masked,
                'tenure': n_periods,
                'active': int(sum(1 for v in raw_vals if v > 0)),
                'avg': round(float(np.mean(active_vals)), 1) if active_vals else 0,
                'last_amount': int(raw_vals[-1]),
                'vals': [int(v) for v in display_vals]
            })

        meta = cluster_meta[c_idx]
        count = int(cluster_sizes[c_idx])
        pat_key = f"cluster_{c_idx}"
        avg_last = round(np.mean([e['last_amount'] for e in examples])) if examples else 0
        total_tickets = sum(sum(e['vals']) for e in examples)

        patterns_data[pat_key] = {
            'count': count,
            'pct': f"{count/total_churned*100:.1f}%",
            'name_th': meta['name_th'],
            'desc': meta['desc'],
            'icon': meta['icon'],
            'color': meta['color'],
            'action': meta['action'],
            'avg_last': avg_last,
            'total_tickets': total_tickets,
            'examples': examples
        }

    # Add excluded group
    if n_excluded > 0:
        pat_key = f"cluster_{k}"
        patterns_data[pat_key] = {
            'count': n_excluded,
            'pct': f"{n_excluded/total_churned*100:.1f}%",
            'name_th': 'ซื้อครั้งเดียว/ไม่เคยซื้อ',
            'desc': '⬜ ซื้อครั้งเดียว/ไม่เคยซื้อ: ซื้อแค่ 0-1 งวดจาก 20 งวด',
            'icon': '⬜',
            'color': '#D1D5DB',
            'action': '🎯 Action: ลูกค้าขาจร — ใช้ automated campaign เท่านั้น',
            'avg_last': 0,
            'total_tickets': 0,
            'examples': []
        }

    return {
        'labels': labels,
        'period': period,
        'total_churned': total_churned,
        'patterns': patterns_data
    }


if __name__ == '__main__':
    output = {}
    for period, fname in PERIODS:
        output[period] = process_period(period, fname)

    out_path = os.path.join(PRED_DIR, 'churn_patterns.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, separators=(',', ':'))

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n{'='*60}")
    print(f"Saved to {out_path} ({size_kb:.0f} KB)")
    print(f"{'='*60}")
