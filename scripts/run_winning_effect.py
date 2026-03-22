#!/usr/bin/env python3
"""
Phase 41 — Winning Effect Visualization
Creates storytelling charts about how winning prizes changes customer behavior.

Run from project root:
    python3 scripts/run_winning_effect.py
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from collections import defaultdict

warnings.filterwarnings('ignore')

# ── Font Setup ──────────────────────────────────────────────────────────
THAI_FONT = None
for font_path in [
    '/System/Library/Fonts/Supplemental/Thonburi.ttc',
    '/System/Library/Fonts/Supplemental/Tahoma.ttf',
]:
    if os.path.exists(font_path):
        THAI_FONT = fm.FontProperties(fname=font_path)
        break

if THAI_FONT:
    fe = fm.FontEntry(fname=THAI_FONT.get_file(), name='ThaiFont')
    fm.fontManager.ttflist.insert(0, fe)
    plt.rcParams['font.family'] = fe.name
    print(f"[Font] Using Thai font: {THAI_FONT.get_file()}")
else:
    print("[Font] No Thai font found — Thai text may not render correctly")

# ── Color Scheme ────────────────────────────────────────────────────────
C_BLUE    = '#2563EB'
C_GREEN   = '#059669'
C_RED     = '#DC2626'
C_GRAY    = '#6B7280'
C_DARK    = '#111827'
C_AMBER   = '#D97706'
C_PURPLE  = '#7C3AED'
C_BLUE_L  = '#DBEAFE'
C_GREEN_L = '#D1FAE5'
C_RED_L   = '#FEE2E2'

CHARTS_DIR = 'charts'
os.makedirs(CHARTS_DIR, exist_ok=True)


def churn_gradient(n):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('rg', [C_RED, C_AMBER, C_GREEN])
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


# ── Data Loading ────────────────────────────────────────────────────────

def load_all_prizes():
    files = sorted(glob.glob('data/prize/Prize_*.csv'))
    print(f"[Data] Loading {len(files)} prize files...")
    dfs = [pd.read_csv(f) for f in files]
    prizes = pd.concat(dfs, ignore_index=True)
    prizes['rounddate'] = pd.to_datetime(prizes['rounddate'])
    print(f"  -> {len(prizes):,} prize records, {prizes['userNo'].nunique():,} unique winners")
    return prizes


def load_main_churn_file():
    """Load the main churn file (Churn_2026_0301_0316.csv).
    This file has item2026_03_16 where 0 = churned, >0 = retained."""
    fpath = 'data/churn/Churn_2026_0301_0316.csv'
    print(f"[Data] Loading {fpath}...")
    df = pd.read_csv(fpath)
    print(f"  -> {len(df):,} customers")
    return df


def get_item_cols_and_dates(df):
    item_cols = [c for c in df.columns if c.startswith('item')]
    dates = []
    for c in item_cols:
        parts = c.replace('item', '').split('_')
        dates.append(f"{parts[0]}-{parts[1]}-{parts[2]}")
    return item_cols, dates


def build_data(prizes, df, item_cols, dates):
    """Build purchase matrix and winner index.
    Churn = item in last column is 0 or NaN."""
    purchase_matrix = df[item_cols].values.astype(float)
    user_nos = df['userNo'].values

    # Map round dates to column indices
    round_to_idx = {}
    for i, d in enumerate(dates):
        round_to_idx[d] = i

    # Build per-user win events (round_idx + amount)
    prize_agg = prizes.groupby(['userNo', 'rounddate']).agg(
        total_prize=('total_prize', 'sum'),
        n_wins=('count_prize', 'sum'),
    ).reset_index()

    # Also keep per-row type for per-type analysis later
    user_wins = defaultdict(list)
    for _, row in prize_agg.iterrows():
        rd = row['rounddate'].strftime('%Y-%m-%d')
        if rd in round_to_idx:
            user_wins[row['userNo']].append({
                'round_idx': round_to_idx[rd],
                'round_date': rd,
                'amount': row['total_prize'],
                'n_wins': row['n_wins'],
            })

    # User index for fast lookup
    user_idx = {u: i for i, u in enumerate(user_nos)}

    # Churn vector: last column 0 or NaN = churned
    last_vals = purchase_matrix[:, -1]
    is_churned = (np.isnan(last_vals)) | (last_vals == 0)

    n_churn = is_churned.sum()
    n_total = len(is_churned)
    print(f"[Data] Churn: {n_churn:,}/{n_total:,} = {n_churn/n_total*100:.1f}%")
    print(f"[Data] Winners in purchase data: {sum(1 for u in user_wins if u in user_idx):,}")

    return purchase_matrix, user_nos, user_wins, user_idx, is_churned


# ── Analysis ────────────────────────────────────────────────────────────

def analyze_before_after(user_wins, purchase_matrix, user_nos, user_idx):
    """Average tickets BEFORE vs AFTER first win."""
    before_list, after_list = [], []
    examples = []

    for user, wins in user_wins.items():
        if user not in user_idx:
            continue
        i = user_idx[user]
        sorted_wins = sorted(wins, key=lambda w: w['round_idx'])
        first_idx = sorted_wins[0]['round_idx']
        purch = purchase_matrix[i]

        # Before: up to 3 rounds before first win, must have bought (>0)
        bstart = max(0, first_idx - 3)
        bvals = purch[bstart:first_idx]
        bvals = bvals[(~np.isnan(bvals)) & (bvals > 0)]

        # After: up to 3 rounds after first win
        aend = min(len(purch), first_idx + 1 + 3)
        avals = purch[first_idx + 1:aend]
        avals = avals[(~np.isnan(avals)) & (avals > 0)]

        if len(bvals) >= 1 and len(avals) >= 1:
            b, a = np.mean(bvals), np.mean(avals)
            before_list.append(b)
            after_list.append(a)
            if b > 0:
                examples.append({
                    'userNo': user, 'before_avg': b, 'after_avg': a,
                    'pct_change': (a - b) / b * 100,
                    'first_win_amount': sorted_wins[0]['amount'],
                })

    return np.mean(before_list), np.mean(after_list), examples


def analyze_timeline(user_wins, purchase_matrix, user_nos, user_idx):
    """Purchase timeline: periods -3..+6 relative to first win."""
    timeline = defaultdict(list)
    never_won = []

    for i, user in enumerate(user_nos):
        purch = purchase_matrix[i]
        if user not in user_wins:
            valid = purch[(~np.isnan(purch)) & (purch > 0)]
            if len(valid) >= 3:
                never_won.append(np.mean(valid))
            continue

        sorted_wins = sorted(user_wins[user], key=lambda w: w['round_idx'])
        first_idx = sorted_wins[0]['round_idx']
        for offset in range(-3, 7):
            idx = first_idx + offset
            if 0 <= idx < len(purch):
                v = purch[idx]
                if not np.isnan(v) and v > 0:
                    timeline[offset].append(v)

    means = {k: np.mean(v) for k, v in timeline.items() if len(v) >= 50}
    baseline = np.mean(never_won) if never_won else 0
    return means, baseline


def analyze_win_frequency_churn(user_wins, user_nos, user_idx, is_churned):
    """Churn rate by number of winning rounds."""
    bins = [(0, '0'), (1, '1'), (2, '2'), (5, '3-5'), (10, '6-10'), (999, '11+')]

    def get_bin(n):
        if n == 0: return '0'
        if n == 1: return '1'
        if n == 2: return '2'
        if n <= 5: return '3-5'
        if n <= 10: return '6-10'
        return '11+'

    counts = defaultdict(lambda: {'churned': 0, 'total': 0})
    for i, user in enumerate(user_nos):
        n_win_rounds = len(user_wins.get(user, []))
        label = get_bin(n_win_rounds)
        counts[label]['total'] += 1
        if is_churned[i]:
            counts[label]['churned'] += 1

    order = ['0', '1', '2', '3-5', '6-10', '11+']
    result = {}
    for label in order:
        t = counts[label]['total']
        result[label] = counts[label]['churned'] / t * 100 if t > 0 else 0
    return result


def analyze_honeymoon(user_wins, user_nos, user_idx, is_churned, n_rounds):
    """Churn rate by periods since last win."""
    bin_edges = [(0, 1), (2, 3), (4, 5), (6, 8), (9, 12), (13, 99)]
    bin_labels = ['0-1', '2-3', '4-5', '6-8', '9-12', '13+']
    counts = {l: {'churned': 0, 'total': 0} for l in bin_labels}

    last_round_idx = n_rounds - 1
    for i, user in enumerate(user_nos):
        if user not in user_wins:
            continue
        sorted_wins = sorted(user_wins[user], key=lambda w: w['round_idx'])
        last_win_idx = sorted_wins[-1]['round_idx']
        periods_since = last_round_idx - last_win_idx

        for (lo, hi), label in zip(bin_edges, bin_labels):
            if lo <= periods_since <= hi:
                counts[label]['total'] += 1
                if is_churned[i]:
                    counts[label]['churned'] += 1
                break

    return {l: counts[l]['churned'] / counts[l]['total'] * 100
            if counts[l]['total'] > 0 else 0 for l in bin_labels}


def analyze_small_vs_big(prizes, user_idx, is_churned):
    """Multiple small wins vs one big win."""
    small_types = {'LAST2DIGITS', 'LAST3DIGITS', 'FIRST3DIGITS'}
    user_small = defaultdict(int)
    user_big = defaultdict(int)
    for _, row in prizes.iterrows():
        u = row['userNo']
        if row['type'] in small_types:
            user_small[u] += 1
        else:
            user_big[u] += 1

    ga = {'churned': 0, 'total': 0}  # 3+ small, 0 big
    gb = {'churned': 0, 'total': 0}  # 1 big, <=1 small

    for user in set(list(user_small.keys()) + list(user_big.keys())):
        if user not in user_idx:
            continue
        i = user_idx[user]

        if user_small[user] >= 3 and user_big[user] == 0:
            ga['total'] += 1
            if is_churned[i]:
                ga['churned'] += 1

        if user_big[user] >= 1 and user_small[user] <= 1:
            gb['total'] += 1
            if is_churned[i]:
                gb['churned'] += 1

    ca = ga['churned'] / ga['total'] * 100 if ga['total'] > 0 else 0
    cb = gb['churned'] / gb['total'] * 100 if gb['total'] > 0 else 0
    return ca, ga['total'], cb, gb['total']


def analyze_churn_by_type(prizes, user_idx, is_churned):
    """Churn rate by best prize type per user."""
    type_rank = {'FIRST': 9, 'SECOND': 8, 'THIRD': 7, 'NEARBY': 6, 'FOURTH': 5,
                 'FIFTH': 4, 'LAST3DIGITS': 3, 'FIRST3DIGITS': 2, 'LAST2DIGITS': 1}
    user_best = {}
    for _, row in prizes.iterrows():
        u, t = row['userNo'], row['type']
        if u not in user_best or type_rank.get(t, 0) > type_rank.get(user_best[u], 0):
            user_best[u] = t

    tc = defaultdict(lambda: {'churned': 0, 'total': 0})
    for user, best in user_best.items():
        if user not in user_idx:
            continue
        i = user_idx[user]
        tc[best]['total'] += 1
        if is_churned[i]:
            tc[best]['churned'] += 1

    return {t: {'churn_rate': d['churned'] / d['total'] * 100 if d['total'] > 0 else 0,
                'total': d['total']} for t, d in tc.items()}


def analyze_revenue(prizes, purchase_matrix, user_nos, user_idx):
    """Revenue from winners vs prize payouts."""
    TICKET_PRICE = 120
    total_payout = prizes['total_prize'].sum()

    total_tickets = 0
    for user in prizes['userNo'].unique():
        if user not in user_idx:
            continue
        i = user_idx[user]
        purch = purchase_matrix[i]
        valid = purch[(~np.isnan(purch)) & (purch > 0)]
        total_tickets += np.sum(valid)

    total_revenue = total_tickets * TICKET_PRICE
    roi = total_revenue / total_payout if total_payout > 0 else 0
    return total_payout, total_revenue, roi


def find_examples(user_wins, purchase_matrix, user_nos, user_idx, is_churned, dates, prizes):
    """Find real customers matching 5 archetype patterns."""
    n_rounds = len(dates)
    examples = {}

    # Build per-user type counts
    user_types_count = defaultdict(lambda: defaultdict(int))
    for _, row in prizes.iterrows():
        user_types_count[row['userNo']][row['type']] += 1

    # Pattern A: Multiple small wins, never churn, buys every round
    for user in user_nos:
        if user not in user_wins:
            continue
        i = user_idx[user]
        if is_churned[i]:
            continue
        wins = user_wins[user]
        small_wins = sum(1 for w in wins if w['amount'] < 10000)
        if small_wins < 5:
            continue
        purch = purchase_matrix[i]
        active = np.sum((~np.isnan(purch)) & (purch > 0))
        if active >= n_rounds * 0.75:
            examples['A'] = {
                'userNo': user, 'n_small_wins': small_wins,
                'active_rounds': int(active), 'total_rounds': n_rounds,
                'desc': f"ถูกเลขท้าย {small_wins} ครั้ง -> ซื้อ {int(active)}/{n_rounds} งวด ไม่เคย churn"
            }
            break

    # Pattern B: Won big prize, purchase increased
    for user in user_nos:
        if user not in user_wins:
            continue
        i = user_idx[user]
        wins = sorted(user_wins[user], key=lambda w: w['round_idx'])
        big = [w for w in wins if w['amount'] >= 20000]
        if not big:
            continue
        bw = big[0]
        widx = bw['round_idx']
        purch = purchase_matrix[i]
        bvals = purch[max(0, widx-3):widx]
        bvals = bvals[(~np.isnan(bvals)) & (bvals > 0)]
        avals = purch[widx+1:min(len(purch), widx+4)]
        avals = avals[(~np.isnan(avals)) & (avals > 0)]
        if len(bvals) >= 1 and len(avals) >= 1:
            b, a = np.mean(bvals), np.mean(avals)
            if a > b * 1.5 and 2 <= b < 10:
                examples['B'] = {
                    'userNo': user, 'before_avg': round(b, 1), 'after_avg': round(a, 1),
                    'win_amount': bw['amount'],
                    'desc': f"ถูกรางวัล {bw['amount']:,.0f} บาท -> ซื้อเพิ่มจาก {b:.0f} ใบเป็น {a:.0f} ใบ"
                }
                break

    # Pattern C: Never won, churned
    for i, user in enumerate(user_nos):
        if user in user_wins:
            continue
        if not is_churned[i]:
            continue
        purch = purchase_matrix[i]
        active = np.sum((~np.isnan(purch)) & (purch > 0))
        if 5 <= active <= 12:
            examples['C'] = {
                'userNo': user, 'active_rounds': int(active),
                'desc': f"ไม่เคยถูกรางวัล -> churn หลัง {int(active)} งวด"
            }
            break

    # Pattern D: Won long ago, honeymoon expired, churned
    last_round_idx = n_rounds - 1
    for user in user_nos:
        if user not in user_wins:
            continue
        i = user_idx[user]
        if not is_churned[i]:
            continue
        wins = sorted(user_wins[user], key=lambda w: w['round_idx'])
        last_win = wins[-1]
        ps = last_round_idx - last_win['round_idx']
        if ps >= 8:
            examples['D'] = {
                'userNo': user, 'periods_since_win': ps,
                'last_win_amount': last_win['amount'],
                'desc': f"ถูกรางวัลเมื่อ {ps} งวดก่อน -> churn แล้ว (honeymoon หมด)"
            }
            break

    # Pattern E: Jackpot/big winner who buys a lot
    for user in user_nos:
        if user not in user_wins:
            continue
        i = user_idx[user]
        wins = user_wins[user]
        big = [w for w in wins if w['amount'] >= 100000]
        if not big:
            continue
        purch = purchase_matrix[i]
        active = purch[(~np.isnan(purch)) & (purch > 0)]
        if len(active) >= 3 and np.mean(active) > 15:
            max_prize = max(w['amount'] for w in big)
            examples['E'] = {
                'userNo': user, 'win_amount': max_prize,
                'avg_tickets': round(np.mean(active), 1),
                'desc': f"ถูกรางวัลใหญ่ {max_prize:,.0f} บาท -> ซื้อเฉลี่ย {np.mean(active):.0f} ใบ/งวด"
            }
            break

    return examples


# ── Plotting ────────────────────────────────────────────────────────────

def tp(text, ax, x, y, **kwargs):
    """Text with Thai font properties."""
    kwargs.setdefault('fontproperties', THAI_FONT)
    return ax.text(x, y, text, **kwargs)


def panel_title(ax, title):
    ax.set_title(title, fontsize=14, fontweight='bold', color=C_DARK, pad=12,
                 fontproperties=THAI_FONT, loc='left')


def plot_p1(ax, before_avg, after_avg, examples):
    """Before vs After win."""
    bars = ax.bar(['ก่อนถูกรางวัล\n(Before)', 'หลังถูกรางวัล\n(After)'],
                  [before_avg, after_avg], color=[C_GRAY, C_BLUE],
                  width=0.5, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, [before_avg, after_avg]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.1f} ใบ', ha='center', fontsize=13, fontweight='bold',
                color=C_DARK, fontproperties=THAI_FONT)

    pct = (after_avg - before_avg) / before_avg * 100
    ax.annotate(f'+{pct:.1f}%', xy=(1, after_avg * 0.85),
                xytext=(0, before_avg * 1.15),
                fontsize=18, fontweight='bold', color=C_GREEN, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=2.5),
                fontproperties=THAI_FONT)

    # Top 3 examples
    top3 = sorted(examples, key=lambda x: x['pct_change'], reverse=True)[:3]
    for j, ex in enumerate(top3):
        txt = f"  {ex['userNo'][:10]}: {ex['before_avg']:.0f}->{ex['after_avg']:.0f} ใบ (+{ex['pct_change']:.0f}%)"
        ax.text(1.35, after_avg * (0.45 - j * 0.1), txt, fontsize=7.5,
                color=C_DARK, fontproperties=THAI_FONT,
                bbox=dict(boxstyle='round,pad=0.2', facecolor=C_BLUE_L, alpha=0.7, edgecolor='none'))

    ax.set_ylabel('เฉลี่ยใบ/งวด', fontsize=10, fontproperties=THAI_FONT)
    ax.set_ylim(0, after_avg * 1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    panel_title(ax, '1. ก่อน vs หลังถูกรางวัล')


def plot_p2(ax, timeline, baseline):
    """Timeline relative to win."""
    offsets = sorted(timeline.keys())
    vals = [timeline[o] for o in offsets]

    ax.plot(offsets, vals, 'o-', color=C_BLUE, lw=2.5, ms=7, zorder=3)
    ax.fill_between(offsets, vals, alpha=0.12, color=C_BLUE)
    ax.axhline(y=baseline, color=C_RED, ls='--', lw=1.5, alpha=0.7)
    ax.text(max(offsets) - 0.5, baseline + 0.3,
            f'Baseline (ไม่เคยถูก) = {baseline:.1f}',
            fontsize=8.5, color=C_RED, ha='right', fontproperties=THAI_FONT)
    ax.axvline(x=0, color=C_GREEN, lw=2, alpha=0.5)
    ax.text(0.15, max(vals) * 0.95, 'ถูกรางวัล!', fontsize=10, color=C_GREEN,
            fontweight='bold', fontproperties=THAI_FONT)

    ax.set_xlabel('งวดเทียบกับตอนถูกรางวัล', fontsize=10, fontproperties=THAI_FONT)
    ax.set_ylabel('เฉลี่ยใบ/งวด', fontsize=10, fontproperties=THAI_FONT)
    ax.set_xticks(offsets)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    panel_title(ax, '2. Winning Effect Timeline')


def plot_p3(ax, freq_churn):
    """Win frequency vs churn rate."""
    labels = list(freq_churn.keys())
    vals = list(freq_churn.values())
    colors = churn_gradient(len(labels))[::-1]

    bars = ax.bar(labels, vals, color=colors, width=0.6, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold',
                color=C_DARK, fontproperties=THAI_FONT)

    ax.set_xlabel('จำนวนงวดที่ถูกรางวัล', fontsize=10, fontproperties=THAI_FONT)
    ax.set_ylabel('Churn Rate (%)', fontsize=10, fontproperties=THAI_FONT)
    ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(0.98, 0.92, 'ยิ่งถูกบ่อย = ยิ่งไม่หยุดซื้อ!',
            transform=ax.transAxes, fontsize=10, color=C_GREEN, fontweight='bold',
            ha='right', fontproperties=THAI_FONT,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=C_GREEN_L, edgecolor=C_GREEN, alpha=0.8))
    panel_title(ax, '3. ยิ่งถูกบ่อย ยิ่งไม่หยุดซื้อ')


def plot_p4(ax, honeymoon):
    """Honeymoon period."""
    labels = list(honeymoon.keys())
    vals = list(honeymoon.values())

    ax.plot(range(len(labels)), vals, 'o-', color=C_RED, lw=2.5, ms=8, zorder=3)
    ax.fill_between(range(len(labels)), vals, alpha=0.1, color=C_RED)
    ax.axvspan(-0.5, 1.5, alpha=0.15, color=C_GREEN, zorder=0)
    ax.text(0.5, max(vals) * 0.85 if max(vals) > 0 else 5, 'Honeymoon\nZone',
            fontsize=10, color=C_GREEN, fontweight='bold', ha='center',
            fontproperties=THAI_FONT)

    for i, val in enumerate(vals):
        offset = max(vals) * 0.03 if max(vals) > 0 else 0.3
        ax.text(i, val + offset, f'{val:.1f}%', ha='center', fontsize=9.5,
                fontweight='bold', color=C_DARK, fontproperties=THAI_FONT)

    ax.set_xlabel('งวดนับจากถูกรางวัลครั้งสุดท้าย', fontsize=10, fontproperties=THAI_FONT)
    ax.set_ylabel('Churn Rate (%)', fontsize=10, fontproperties=THAI_FONT)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(0.98, 0.08, 'ถูกรางวัลไม่เกิน 3 งวดก่อน = churn ต่ำมาก',
            transform=ax.transAxes, fontsize=9, color=C_GREEN, fontweight='bold',
            ha='right', fontproperties=THAI_FONT,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=C_GREEN_L, edgecolor=C_GREEN, alpha=0.8))
    panel_title(ax, '4. Honeymoon Period')


def plot_p5(ax, churn_s, n_s, churn_b, n_b):
    """Small vs big wins."""
    cats = ['ถูกเลขท้าย 3+ ครั้ง\n(Multiple Small)', 'ถูกรางวัลใหญ่ 1+ ครั้ง\n(Big Win)']
    vals = [churn_s, churn_b]

    bars = ax.barh(cats, vals, color=[C_GREEN, C_RED], height=0.45,
                   edgecolor='white', linewidth=2)
    for bar, val, n in zip(bars, vals, [n_s, n_b]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%  (n={n:,})', va='center', fontsize=12,
                fontweight='bold', color=C_DARK, fontproperties=THAI_FONT)

    ax.set_xlabel('Churn Rate (%)', fontsize=10, fontproperties=THAI_FONT)
    ax.set_xlim(0, max(vals) * 1.6 if max(vals) > 0 else 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    diff = abs(churn_b - churn_s)
    ax.text(0.98, 0.15, f'ถูกบ่อยเล็กๆ churn น้อยกว่า {diff:.1f}%',
            transform=ax.transAxes, fontsize=10, color=C_GREEN, fontweight='bold',
            ha='right', fontproperties=THAI_FONT,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=C_GREEN_L, edgecolor=C_GREEN, alpha=0.8))
    panel_title(ax, '5. ถูกบ่อยเล็กๆ ดีกว่าถูกครั้งเดียวใหญ่')


def plot_p6(ax, type_churn):
    """Churn by prize type."""
    sorted_tc = sorted(type_churn.items(), key=lambda x: x[1]['churn_rate'], reverse=True)
    labels, vals, counts = [], [], []
    for t, d in sorted_tc:
        if d['total'] >= 10:
            labels.append(t)
            vals.append(d['churn_rate'])
            counts.append(d['total'])

    colors = churn_gradient(len(labels))
    bars = ax.barh(range(len(labels)), vals, color=colors, height=0.6,
                   edgecolor='white', linewidth=1.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()

    for i, (bar, val, cnt) in enumerate(zip(bars, vals, counts)):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}% (n={cnt:,})', va='center', fontsize=9,
                fontweight='bold', color=C_DARK, fontproperties=THAI_FONT)

    ax.set_xlabel('Churn Rate (%)', fontsize=10, fontproperties=THAI_FONT)
    ax.set_xlim(0, max(vals) * 1.35 if max(vals) > 0 else 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    panel_title(ax, '6. Churn Rate ตามประเภทรางวัล')


def plot_p7(ax, total_payout, total_revenue, roi):
    """Revenue infographic."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    pm = total_payout / 1e6
    rm = total_revenue / 1e6

    rect1 = FancyBboxPatch((0.3, 4.5), 3.5, 4.0, boxstyle="round,pad=0.3",
                            facecolor=C_RED_L, edgecolor=C_RED, linewidth=2)
    ax.add_patch(rect1)
    ax.text(2.05, 7.5, 'เงินรางวัลจ่าย', fontsize=11, ha='center', color=C_RED,
            fontweight='bold', fontproperties=THAI_FONT)
    ax.text(2.05, 6.1, f'{pm:,.0f} ล้าน', fontsize=20, ha='center', color=C_RED,
            fontweight='bold', fontproperties=THAI_FONT)
    ax.text(2.05, 5.2, '(Prize Payouts)', fontsize=8, ha='center', color=C_GRAY)

    rect2 = FancyBboxPatch((5.5, 4.5), 4.0, 4.0, boxstyle="round,pad=0.3",
                            facecolor=C_GREEN_L, edgecolor=C_GREEN, linewidth=2)
    ax.add_patch(rect2)
    ax.text(7.5, 7.5, 'รายได้จากผู้ชนะ', fontsize=11, ha='center', color=C_GREEN,
            fontweight='bold', fontproperties=THAI_FONT)
    ax.text(7.5, 6.1, f'{rm:,.0f} ล้าน', fontsize=20, ha='center', color=C_GREEN,
            fontweight='bold', fontproperties=THAI_FONT)
    ax.text(7.5, 5.2, '(Winner Revenue)', fontsize=8, ha='center', color=C_GRAY)

    ax.annotate('', xy=(5.3, 6.5), xytext=(4.0, 6.5),
                arrowprops=dict(arrowstyle='->', color=C_DARK, lw=2))

    rect3 = FancyBboxPatch((2.5, 0.8), 4.5, 2.8, boxstyle="round,pad=0.3",
                            facecolor=C_BLUE_L, edgecolor=C_BLUE, linewidth=2.5)
    ax.add_patch(rect3)
    ax.text(4.75, 2.7, 'ROI', fontsize=12, ha='center', color=C_BLUE, fontweight='bold')
    ax.text(4.75, 1.6, f'{roi:.1f}x', fontsize=28, ha='center', color=C_BLUE, fontweight='bold')

    panel_title(ax, '7. Revenue คุ้มค่า — รางวัลคือการลงทุน')


def plot_p8(ax, examples):
    """Customer profiles."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    markers = {'A': ('●', C_GREEN), 'B': ('▲', C_BLUE), 'C': ('✕', C_RED),
               'D': ('◆', C_AMBER), 'E': ('★', C_PURPLE)}
    profiles = []
    for key in ['A', 'B', 'C', 'D', 'E']:
        if key in examples:
            m, c = markers[key]
            profiles.append((m, c, examples[key]['userNo'], examples[key]['desc']))

    for i, (marker, color, uid, desc) in enumerate(profiles):
        y = 8.2 - i * 1.7
        rect = FancyBboxPatch((0.2, y - 0.6), 9.3, 1.4, boxstyle="round,pad=0.2",
                              facecolor='white', edgecolor=color, linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(0.8, y + 0.15, marker, fontsize=18, color=color, ha='center',
                va='center', fontweight='bold')
        ax.text(1.5, y + 0.25, uid, fontsize=10, color=C_DARK, fontweight='bold', va='center')
        ax.text(1.5, y - 0.2, desc, fontsize=9, color=C_GRAY, va='center',
                fontproperties=THAI_FONT)

    panel_title(ax, '8. ตัวอย่างลูกค้าจริง (Real Customer Profiles)')


def create_summary(timeline, baseline, before_avg, after_avg, freq_churn, honeymoon, roi, n_cust):
    """Single 'money shot' chart for slide decks."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')

    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    offsets = sorted(timeline.keys())
    vals = [timeline[o] for o in offsets]

    ax.fill_between(offsets, vals, baseline, alpha=0.12, color=C_BLUE)
    ax.fill_between(offsets, [baseline] * len(offsets), alpha=0.05, color=C_RED)
    ax.plot(offsets, vals, 'o-', color=C_BLUE, lw=3.5, ms=10, zorder=5,
            label='ผู้ถูกรางวัล (Winners)')
    ax.axhline(y=baseline, color=C_RED, ls='--', lw=2, alpha=0.8)
    ax.text(max(offsets) + 0.2, baseline,
            f'Baseline: {baseline:.1f} ใบ\n(ไม่เคยถูกรางวัล)',
            fontsize=11, color=C_RED, va='center', fontproperties=THAI_FONT,
            fontweight='bold')

    ax.axvline(x=0, color=C_GREEN, lw=3, alpha=0.4)
    if 0 in timeline:
        ax.plot(0, timeline[0], 's', color=C_GREEN, ms=15, zorder=6)
        ax.text(0.2, timeline[0] + 0.5, f'ถูกรางวัล!\n{timeline[0]:.1f} ใบ',
                fontsize=12, color=C_GREEN, fontweight='bold', fontproperties=THAI_FONT)

    for o, v in zip(offsets, vals):
        if o != 0:
            ax.text(o, v + 0.4, f'{v:.1f}', fontsize=9, ha='center',
                    color=C_BLUE, fontweight='bold')

    ax.set_xlabel('งวดเทียบกับตอนถูกรางวัล', fontsize=13, fontproperties=THAI_FONT)
    ax.set_ylabel('เฉลี่ยใบลอตเตอรี่ต่องวด', fontsize=13, fontproperties=THAI_FONT)
    ax.set_xticks(offsets)
    xlabels = []
    for o in offsets:
        if o < 0:
            xlabels.append(f'{o} งวดก่อน')
        elif o == 0:
            xlabels.append('ถูกรางวัล')
        else:
            xlabels.append(f'+{o} งวดหลัง')
    ax.set_xticklabels(xlabels, fontsize=9, fontproperties=THAI_FONT)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Key stats overlay
    pct_inc = (after_avg - before_avg) / before_avg * 100
    freq_vals = list(freq_churn.values())
    hm_vals = list(honeymoon.values())

    rect = FancyBboxPatch((0.68, 0.25), 0.30, 0.68, boxstyle="round,pad=0.02",
                          facecolor='white', edgecolor=C_GRAY, linewidth=1,
                          alpha=0.92, transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(0.83, 0.88, 'Key Insights', fontsize=14, fontweight='bold',
            color=C_DARK, transform=ax.transAxes, ha='center')

    stats = [
        (f'+{pct_inc:.0f}%', 'ซื้อเพิ่มหลังถูกรางวัล', C_GREEN),
        (f'{freq_vals[0]:.0f}% -> {freq_vals[-1]:.0f}%', 'Churn ลดตามจำนวนครั้งที่ถูก', C_BLUE),
        (f'{hm_vals[0]:.1f}%', 'Churn ใน Honeymoon (0-1 งวด)', C_GREEN),
        (f'{hm_vals[-1]:.1f}%', 'Churn หลัง 13+ งวด', C_RED),
        (f'{roi:.1f}x', 'ROI รายได้ vs เงินรางวัล', C_BLUE),
    ]
    for i, (num, desc, color) in enumerate(stats):
        y = 0.78 - i * 0.11
        ax.text(0.70, y, num, fontsize=16, fontweight='bold', color=color,
                transform=ax.transAxes)
        ax.text(0.70, y - 0.04, desc, fontsize=8.5, color=C_GRAY,
                transform=ax.transAxes, fontproperties=THAI_FONT)

    ax.set_title('Winning Effect: ถูกรางวัลเปลี่ยนพฤติกรรมลูกค้าอย่างไร',
                 fontsize=18, fontweight='bold', color=C_DARK, pad=20,
                 fontproperties=THAI_FONT)

    fig.text(0.5, 0.02,
             f'Data: 30 งวด (ม.ค. 2025 — มี.ค. 2026) | {n_cust:,} ลูกค้า | ChurnCustomer Phase 41',
             fontsize=9, ha='center', color=C_GRAY, fontproperties=THAI_FONT, style='italic')

    out = os.path.join(CHARTS_DIR, 'winning_effect_summary.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Saved: {out}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Phase 41: Winning Effect Visualization")
    print("=" * 70)

    prizes = load_all_prizes()
    df = load_main_churn_file()
    item_cols, dates = get_item_cols_and_dates(df)
    purchase_matrix, user_nos, user_wins, user_idx, is_churned = build_data(
        prizes, df, item_cols, dates)

    n_rounds = len(dates)
    n_cust = len(user_nos)

    # ── Analyses ──
    print("\n--- 1: Before vs After ---")
    before_avg, after_avg, ba_examples = analyze_before_after(
        user_wins, purchase_matrix, user_nos, user_idx)
    pct = (after_avg - before_avg) / before_avg * 100
    print(f"  Before: {before_avg:.1f}, After: {after_avg:.1f}, Change: +{pct:.1f}%")

    print("\n--- 2: Timeline ---")
    timeline, baseline = analyze_timeline(user_wins, purchase_matrix, user_nos, user_idx)
    for o in sorted(timeline.keys()):
        print(f"  Period {o:+d}: {timeline[o]:.1f}")
    print(f"  Baseline: {baseline:.1f}")

    print("\n--- 3: Win Frequency -> Churn ---")
    freq_churn = analyze_win_frequency_churn(user_wins, user_nos, user_idx, is_churned)
    for l, r in freq_churn.items():
        print(f"  {l} wins: {r:.1f}% churn")

    print("\n--- 4: Honeymoon ---")
    honeymoon = analyze_honeymoon(user_wins, user_nos, user_idx, is_churned, n_rounds)
    for l, r in honeymoon.items():
        print(f"  {l} periods: {r:.1f}% churn")

    print("\n--- 5: Small vs Big ---")
    cs, ns, cb, nb = analyze_small_vs_big(prizes, user_idx, is_churned)
    print(f"  Small (3+): {cs:.1f}% (n={ns:,})")
    print(f"  Big (1+):   {cb:.1f}% (n={nb:,})")

    print("\n--- 6: Churn by Type ---")
    type_churn = analyze_churn_by_type(prizes, user_idx, is_churned)
    for t, d in sorted(type_churn.items(), key=lambda x: x[1]['churn_rate']):
        print(f"  {t:15s}: {d['churn_rate']:.1f}% (n={d['total']:,})")

    print("\n--- 7: Revenue ---")
    payout, revenue, roi = analyze_revenue(prizes, purchase_matrix, user_nos, user_idx)
    print(f"  Payout: {payout/1e6:,.0f}M, Revenue: {revenue/1e6:,.0f}M, ROI: {roi:.1f}x")

    print("\n--- 8: Customer Examples ---")
    examples = find_examples(user_wins, purchase_matrix, user_nos, user_idx,
                             is_churned, dates, prizes)
    for key, ex in examples.items():
        print(f"  [{key}] {ex['userNo']}: {ex['desc']}")

    # ── Charts ──
    print("\n" + "=" * 70)
    print("  Creating charts...")
    print("=" * 70)

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')

    fig = plt.figure(figsize=(22, 28))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.28,
                  top=0.93, bottom=0.04, left=0.07, right=0.95)

    fig.text(0.5, 0.97,
             'Winning Effect: ถูกรางวัลเปลี่ยนพฤติกรรมลูกค้าอย่างไร',
             fontsize=22, fontweight='bold', ha='center', color=C_DARK,
             fontproperties=THAI_FONT)
    fig.text(0.5, 0.955, 'Insight สำหรับทีม Ads / Marcom / Management',
             fontsize=13, ha='center', color=C_GRAY, fontproperties=THAI_FONT)

    plot_p1(fig.add_subplot(gs[0, 0]), before_avg, after_avg, ba_examples)
    plot_p2(fig.add_subplot(gs[0, 1]), timeline, baseline)
    plot_p3(fig.add_subplot(gs[1, 0]), freq_churn)
    plot_p4(fig.add_subplot(gs[1, 1]), honeymoon)
    plot_p5(fig.add_subplot(gs[2, 0]), cs, ns, cb, nb)
    plot_p6(fig.add_subplot(gs[2, 1]), type_churn)
    plot_p7(fig.add_subplot(gs[3, 0]), payout, revenue, roi)
    plot_p8(fig.add_subplot(gs[3, 1]), examples)

    fig.text(0.5, 0.015,
             f'Data: 30 งวด (ม.ค. 2025 — มี.ค. 2026) | {n_cust:,} ลูกค้า | ChurnCustomer Phase 41',
             fontsize=10, ha='center', color=C_GRAY, fontproperties=THAI_FONT, style='italic')

    out = os.path.join(CHARTS_DIR, 'winning_effect_story.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n[OK] Saved: {out}")

    create_summary(timeline, baseline, before_avg, after_avg, freq_churn, honeymoon, roi, n_cust)

    print("\n" + "=" * 70)
    print("  Done! Charts saved to charts/")
    print("=" * 70)


if __name__ == '__main__':
    main()
