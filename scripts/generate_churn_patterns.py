"""
Generate churn_patterns.json for the Churn Behavioral Patterns section.
Classifies all churned customers into 9 behavioral patterns with examples.

Usage: python3 scripts/generate_churn_patterns.py
"""
import pandas as pd
import numpy as np
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHURN_DIR = os.path.join(BASE, 'data', 'churn')
PRED_DIR = os.path.join(BASE, 'data', 'predictions')

# Period configs: (period_label, churn_file)
PERIODS = [
    ('0401', 'Churn_2025_0316_0401.csv'),
    ('0316', 'Churn_2025_0301_0316.csv'),
]

PATTERN_META = {
    'whale_gone': {
        'name_th': 'ขาใหญ่หายไป', 'icon': '🐋', 'color': '#0EA5E9',
        'desc': '🐋 ขาใหญ่หายไป (Whale Gone): ซื้อรวม 500+ ใบใน 20 งวด — ลูกค้า 1 คนเสีย = ขาจร 100 คน',
        'action': '🎯 Action: VIP treatment ทันที — โทรติดตาม ส่ง exclusive offer สูญเสียรายได้สูงมาก!'
    },
    'loyal_veteran': {
        'name_th': 'ขาประจำหายไป', 'icon': '🏆', 'color': '#F97316',
        'desc': '🏆 ขาประจำหายไป (Loyal Veteran): ซื้อทุกงวด 19-20/20 งวด ยอดคงที่สม่ำเสมอ (CV < 0.35) แล้วหายไปเลย',
        'action': '🎯 Action: ไม่มีสัญญาณเตือน! ต้องเช็คปัจจัยภายนอก — ย้ายแพลตฟอร์ม? ปัญหาบัญชี? โทรสอบถาม'
    },
    'escalator': {
        'name_th': 'ยอดขึ้นเรื่อยๆ แล้วหาย', 'icon': '📈', 'color': '#10B981',
        'desc': '📈 ยอดขึ้นเรื่อยๆ แล้วหาย (Escalator): ยอดซื้อเพิ่มขึ้นต่อเนื่อง ครึ่งหลัง > 1.5x ครึ่งแรก แล้วจู่ๆ หายไป!',
        'action': '🎯 Action: Counterintuitive! ยอดขึ้นไม่ได้แปลว่าจะอยู่ — อาจเป็นตัวแทน/ฝากซื้อที่หมดลูกค้า ติดตามทันที'
    },
    'gradual_decline': {
        'name_th': 'ค่อยๆ ลดลง', 'icon': '📉', 'color': '#F59E0B',
        'desc': '📉 ค่อยๆ ลดลง (Gradual Decline): ซื้อ 10+ งวด ยอดลดลงต่อเนื่อง — ครึ่งหลังเหลือไม่ถึง 60% ของครึ่งแรก (R² ≥ 0.25)',
        'action': '🎯 Action: จับสัญญาณตั้งแต่ยอดลดครั้งที่ 2 — ส่ง offer เพิ่มยอดกลับ ยังทันรักษา'
    },
    'u_shape': {
        'name_th': 'กลับมาแล้วหายอีก', 'icon': '🫥', 'color': '#A855F7',
        'desc': '🫥 กลับมาแล้วหายอีก (U-Shape): ซื้อช่วงแรก → หายไป 7+ งวด → กลับมาซื้อ → แล้วหายอีก!',
        'action': '🎯 Action: "ดึงกลับ" อย่างเดียวไม่พอ — ต้อง "รักษาหลังกลับ" ด้วย ส่ง offer ต่อเนื่อง 3 งวดหลังกลับมา'
    },
    'sudden_stop': {
        'name_th': 'หยุดกะทันหัน', 'icon': '🛑', 'color': '#EF4444',
        'desc': '🛑 หยุดกะทันหัน (Sudden Stop): ซื้อสม่ำเสมอ 7+ งวด แล้วหายไปเลย — CV < 0.4 ไม่มีสัญญาณเตือน',
        'action': '🎯 Action: ต้องติดต่อเชิงรุก — ลูกค้ากลุ่มนี้ไม่มีสัญญาณก่อนหยุด อาจมีปัจจัยภายนอก'
    },
    'spike_gone': {
        'name_th': 'พุ่งแล้วหาย', 'icon': '🚀', 'color': '#EC4899',
        'desc': '🚀 พุ่งแล้วหาย (Spike & Gone): งวดสุดท้ายซื้อเยอะผิดปกติ (2x+ ค่าเฉลี่ย, 10+ ใบ) แล้วหายไป',
        'action': '🎯 Action: อาจเป็นการซื้อพิเศษ (ฝากซื้อ/เลขเด็ด) — ส่ง offer หลังซื้อเพื่อดึงกลับ'
    },
    'last_breath': {
        'name_th': 'แทบไม่เคยซื้อ', 'icon': '👻', 'color': '#6B7280',
        'desc': '👻 แทบไม่เคยซื้อ (Last Breath): ซื้อแค่ 1-2 งวดจาก 10 งวดล่าสุด — แทบไม่ใช่ลูกค้าประจำ',
        'action': '🎯 Action: ลูกค้าขาจร — ROI ต่ำในการรักษา ใช้ automated campaign เท่านั้น'
    },
    'irregular_gone': {
        'name_th': 'ซื้อไม่สม่ำเสมอ', 'icon': '🔀', 'color': '#3B82F6',
        'desc': '🔀 ซื้อไม่สม่ำเสมอ (Irregular): ไม่เข้ากลุ่มไหน — ซื้อบ้างหยุดบ้าง ไม่มี pattern ชัดเจน',
        'action': '🎯 Action: ใช้ general retention campaign — ไม่ต้องเจาะจงมาก'
    }
}

PATTERN_ORDER = ['whale_gone', 'loyal_veteran', 'escalator', 'gradual_decline',
                 'u_shape', 'sudden_stop', 'spike_gone', 'last_breath', 'irregular_gone']


def classify(vals20):
    """Classify a churned customer into one of 9 behavioral patterns."""
    vals10 = vals20[-10:]
    active20 = sum(1 for v in vals20 if v > 0)
    active10 = sum(1 for v in vals10 if v > 0)
    active_vals20 = [v for v in vals20 if v > 0]
    active_vals10 = [v for v in vals10 if v > 0]
    total = sum(vals20)
    last_val = vals20[-1]

    # 1. LAST BREATH
    if active10 <= 2:
        return 'last_breath'

    # 2. SPIKE GONE
    if len(active_vals10) > 1:
        other_mean = np.mean(active_vals10[:-1])
        spike = last_val / other_mean if other_mean > 0 else 1
        if spike >= 2.0 and last_val >= 10:
            return 'spike_gone'

    # 3. WHALE GONE
    if total >= 500:
        return 'whale_gone'

    # 4. LOYAL VETERAN
    if active20 >= 19:
        mean_val = np.mean(active_vals20)
        cv = np.std(active_vals20) / mean_val if mean_val > 0 else 0
        if cv < 0.35:
            return 'loyal_veteran'

    # 5. ESCALATOR
    if active20 >= 10:
        f_sum = sum(vals20[:10])
        s_sum = sum(vals20[10:])
        if f_sum > 0 and s_sum > f_sum * 1.5:
            positions = [i for i, v in enumerate(vals20) if v > 0]
            values = [vals20[i] for i in positions]
            if len(values) >= 4:
                slope = np.polyfit(range(len(values)), values, 1)[0]
                if slope > 0.3:
                    return 'escalator'

    # 6. GRADUAL DECLINE
    if active20 >= 10:
        f_half = np.mean(vals20[:10])
        s_half = np.mean(vals20[10:])
        if f_half >= 3 and s_half < f_half * 0.6:
            slope, intercept = np.polyfit(range(20), vals20, 1)
            if slope < -0.1:
                predicted = [slope * x + intercept for x in range(20)]
                ss_res = sum((vals20[i] - predicted[i])**2 for i in range(20))
                ss_tot = sum((vals20[i] - np.mean(vals20))**2 for i in range(20))
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                if r2 >= 0.25:
                    return 'gradual_decline'

    # 7. U-SHAPE
    early = sum(1 for v in vals20[:7] if v > 0)
    mid = sum(1 for v in vals20[7:14] if v > 0)
    late = sum(1 for v in vals20[14:] if v > 0)
    if early >= 4 and mid <= 1 and late >= 2:
        return 'u_shape'

    # 8. SUDDEN STOP
    if active10 >= 5:
        mean_active = np.mean(active_vals10) if active_vals10 else 0
        cv = np.std(active_vals10) / mean_active if mean_active > 0 else 0
        positions = [i for i, v in enumerate(vals10) if v > 0]
        values = [vals10[i] for i in positions]
        slope10 = np.polyfit(range(len(values)), values, 1)[0] if len(values) >= 2 else 0
        if active10 >= 7 and cv < 0.4 and abs(slope10) < 0.5:
            return 'sudden_stop'
        elif active10 >= 7 and slope10 >= -0.3:
            return 'sudden_stop'

    return 'irregular_gone'


def get_labels(item_cols, n=20):
    labels = []
    for c in item_cols[-n:]:
        parts = c.replace('item', '').split('_')
        if len(parts) == 3:
            labels.append(f"{parts[1]}/{parts[2]}")
    return labels


def pick_decline_display(candidates, n=30):
    scored = []
    for c in candidates:
        vals = c['vals']
        g = [np.mean(vals[i*5:(i+1)*5]) for i in range(4)]
        mono = sum(1 for i in range(3) if g[i] > g[i+1] * 0.85)
        if mono < 3 or g[0] < 6 or g[3] > 5:
            continue
        has_spike = any(
            v > g[gi] * 2.5 and v > 8
            for gi in range(4) if g[gi] > 0
            for v in vals[gi*5:(gi+1)*5]
        )
        if has_spike or vals[-1] > 5:
            continue
        mono_strict = sum(1 for i in range(3) if g[i] > g[i+1])
        score = c.get('r2', 0.5) * 30 + mono_strict * 10 + min(g[0] / max(g[3], 0.1), 30) + c.get('active', 15)
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    return [s[1] for s in scored[:n]]


def process_period(period, fname):
    print(f"\nProcessing period {period}: {fname}")
    df = pd.read_csv(os.path.join(CHURN_DIR, fname))
    item_cols = [c for c in df.columns if c.startswith('item')]
    N = len(item_cols)
    labels = get_labels(item_cols, 20)

    churned = df[df[item_cols[-1]] == 0].copy()
    total_churned = len(churned)
    print(f"  Churned: {total_churned:,}")

    use20 = item_cols[max(0, N-21):N-1]

    pattern_counts = {p: 0 for p in PATTERN_ORDER}
    pattern_examples = {p: [] for p in PATTERN_ORDER}
    decline_candidates = []

    for _, row in churned.iterrows():
        vals20 = [int(row[c]) if not pd.isna(row[c]) else 0 for c in use20]
        pat = classify(vals20)
        pattern_counts[pat] += 1

        uid = row['userNo']
        masked = uid[:6] + 'xxx' if uid.startswith('P') else 'P' + uid[:5] + 'xxx'
        active_vals = [v for v in vals20 if v > 0]
        entry = {
            'uid': masked, 'tenure': 20,
            'active': sum(1 for v in vals20 if v > 0),
            'avg': round(np.mean(active_vals), 1) if active_vals else 0,
            'last_amount': vals20[-1], 'vals': vals20
        }

        if pat == 'gradual_decline':
            slope, intercept = np.polyfit(range(20), vals20, 1)
            predicted = [slope * x + intercept for x in range(20)]
            ss_res = sum((vals20[i] - predicted[i])**2 for i in range(20))
            ss_tot = sum((vals20[i] - np.mean(vals20))**2 for i in range(20))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            decline_candidates.append({**entry, 'r2': r2})

        if len(pattern_examples[pat]) < 100:
            pattern_examples[pat].append(entry)

    # Pick best display examples
    decline_display = pick_decline_display(decline_candidates, 30)
    if decline_display:
        pattern_examples['gradual_decline'] = decline_display

    pattern_examples['whale_gone'].sort(key=lambda e: -sum(e['vals']))
    pattern_examples['whale_gone'] = pattern_examples['whale_gone'][:30]

    for e in pattern_examples['loyal_veteran']:
        v = [x for x in e['vals'] if x > 0]
        e['_cv'] = np.std(v) / np.mean(v) if np.mean(v) > 0 else 1
    pattern_examples['loyal_veteran'].sort(key=lambda e: e.pop('_cv'))
    pattern_examples['loyal_veteran'] = pattern_examples['loyal_veteran'][:30]

    for e in pattern_examples['escalator']:
        e['_growth'] = sum(e['vals'][10:]) / max(sum(e['vals'][:10]), 1)
    pattern_examples['escalator'].sort(key=lambda e: -e.pop('_growth'))
    pattern_examples['escalator'] = pattern_examples['escalator'][:30]

    pattern_examples['u_shape'].sort(key=lambda e: -(sum(1 for v in e['vals'][:7] if v > 0)))
    pattern_examples['u_shape'] = pattern_examples['u_shape'][:30]

    pattern_examples['sudden_stop'].sort(key=lambda e: -e['active'])
    pattern_examples['sudden_stop'] = pattern_examples['sudden_stop'][:30]

    pattern_examples['spike_gone'].sort(key=lambda e: -e['last_amount'])
    pattern_examples['spike_gone'] = pattern_examples['spike_gone'][:30]

    for pat in ['last_breath', 'irregular_gone']:
        pattern_examples[pat] = pattern_examples[pat][:30]

    patterns_data = {}
    for pat_key in PATTERN_ORDER:
        meta = PATTERN_META[pat_key]
        count = pattern_counts[pat_key]
        pct = f"{count/total_churned*100:.1f}%"
        exs = pattern_examples[pat_key]
        avg_last = round(np.mean([e['last_amount'] for e in exs])) if exs else 0
        total_tickets = sum(e['last_amount'] for e in exs)
        patterns_data[pat_key] = {
            'count': count, 'pct': pct,
            'name_th': meta['name_th'], 'desc': meta['desc'],
            'icon': meta['icon'], 'color': meta['color'],
            'action': meta['action'],
            'avg_last': avg_last, 'total_tickets': total_tickets,
            'examples': exs
        }

    print(f"  Patterns:")
    for pat_key in PATTERN_ORDER:
        c = pattern_counts[pat_key]
        print(f"    {PATTERN_META[pat_key]['icon']} {pat_key}: {c:,} ({c/total_churned*100:.1f}%)")

    return {
        'labels': labels, 'period': period,
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
    print(f"\nSaved to {out_path} ({size_kb:.0f} KB)")
