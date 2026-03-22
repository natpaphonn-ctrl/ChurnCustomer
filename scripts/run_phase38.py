#!/usr/bin/env python3
"""
Phase 38: Segment-Specific Models
==================================
Train separate churn prediction models per customer segment and compare
with the global v3 model.

Segments (rule-based):
    - New (มาใหม่): tenure_rounds <= 5
    - Loyal (ตัวยง): tenure_rounds >= 40 AND purchase_frequency_ratio >= 0.6
    - Regular (ขาประจำ): purchase_frequency_ratio >= 0.4 AND active_last_6 >= 4
    - Fading (กำลังหาย): active_last_6 <= 1 AND tenure_rounds >= 10
    - Periodic (ซื้อเป็นช่วง): n_reactivations >= 2
    - Occasional (ขาจร): everything else

Usage:
    python3 scripts/run_phase38.py
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_TENURE = 3
MIN_SEGMENT_SIZE = 1000  # minimum samples to train a segment model

# RF hyperparameters — same as production v3
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=50,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

# Segment definitions (Thai + English)
SEGMENT_NAMES = {
    "New": "มาใหม่",
    "Loyal": "ตัวยง",
    "Regular": "ขาประจำ",
    "Fading": "กำลังหาย",
    "Periodic": "ซื้อเป็นช่วง",
    "Occasional": "ขาจร",
}

SEGMENT_COLORS = {
    "New": "#3498db",
    "Loyal": "#e74c3c",
    "Regular": "#27ae60",
    "Fading": "#9b59b6",
    "Periodic": "#f39c12",
    "Occasional": "#95a5a6",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def load_churn_file(filepath):
    """Load a churn CSV and return df, item_cols."""
    df = pd.read_csv(filepath, low_memory=False)
    item_cols = sorted([c for c in df.columns if c.startswith("item")])
    return df, item_cols


def prepare_features_and_target(df, item_cols):
    """Compute v2 features and binary churn target from a churn file.

    Target: last item column (1 = churned if value is 0 or NaN after registration).
    Features: all item columns except the last one.
    """
    sys.path.insert(0, os.path.join(BASE_DIR, "models"))
    from feature_engineering import engineer_features_v2

    feat_item_cols = item_cols[:-1]  # all but last = features
    target_col = item_cols[-1]       # last column = target

    mat = df[item_cols].apply(pd.to_numeric, errors="coerce")
    mat_vals = mat.values.astype(float)

    # Tenure from feature columns
    reg_feat = ~np.isnan(mat[feat_item_cols].values.astype(float))
    has_any = reg_feat.any(axis=1)
    first_reg = np.where(has_any, reg_feat.argmax(axis=1), len(feat_item_cols))
    tenure = len(feat_item_cols) - first_reg

    # Eligible: tenure >= MIN_TENURE
    valid = tenure >= MIN_TENURE

    # Target: churned = did NOT buy in target period
    # NaN in target means not registered yet -> exclude (these are new to the platform)
    target_vals = mat[target_col].values.astype(float)
    # Registered customers who didn't buy = churn (value == 0)
    # NaN = not yet registered, exclude
    reg_in_feat = has_any  # registered in feature period
    target_valid = ~np.isnan(target_vals) | reg_in_feat  # registered somewhere
    # Churn = registered in features period AND (target is 0 or NaN-but-was-registered)
    churn = np.where(
        reg_in_feat,
        np.where(np.isnan(target_vals), 1, (target_vals == 0).astype(int)),
        np.nan  # not registered at all
    )

    # Combined mask: eligible AND has valid target
    mask = valid & reg_in_feat

    df_elig = df[mask].copy().reset_index(drop=True)
    mat_elig = mat[mask][feat_item_cols].copy().reset_index(drop=True)
    y = churn[mask].astype(int)

    feats = engineer_features_v2(df_elig, mat_elig, feat_item_cols)
    feats = feats.fillna(0)

    return feats, y, df_elig


def assign_segment(feats):
    """Assign customer segment based on feature rules. Returns Series of segment labels."""
    n = len(feats)
    segments = pd.Series("Occasional", index=feats.index)

    # Order matters — more specific rules first, last assignment wins for ties
    # We apply in reverse priority so higher-priority segments overwrite

    # Periodic: n_reactivations >= 2
    mask_periodic = feats["n_reactivations"] >= 2
    segments[mask_periodic] = "Periodic"

    # Fading: active_last_6 <= 1 AND tenure_rounds >= 10
    mask_fading = (feats["active_last_6"] <= 1) & (feats["tenure_rounds"] >= 10)
    segments[mask_fading] = "Fading"

    # Regular: purchase_frequency_ratio >= 0.4 AND active_last_6 >= 4
    mask_regular = (feats["purchase_frequency_ratio"] >= 0.4) & (feats["active_last_6"] >= 4)
    segments[mask_regular] = "Regular"

    # Loyal: tenure_rounds >= 40 AND purchase_frequency_ratio >= 0.6
    mask_loyal = (feats["tenure_rounds"] >= 40) & (feats["purchase_frequency_ratio"] >= 0.6)
    segments[mask_loyal] = "Loyal"

    # New: tenure_rounds <= 5
    mask_new = feats["tenure_rounds"] <= 5
    segments[mask_new] = "New"

    return segments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0_total = time.time()
    print("=" * 70)
    print("  Phase 38: Segment-Specific Models")
    print("=" * 70)

    # ── 1. Identify churn files ──
    churn_dir = os.path.join(BASE_DIR, "data", "churn")
    all_files = sorted([
        f for f in os.listdir(churn_dir)
        if f.startswith("Churn_") and "_old_" not in f and "_Check" not in f
        and len(f.split("_")) >= 4  # Churn_YYYY_MMDD_MMDD.csv format
    ])
    # Filter to proper format with two MMDD parts
    churn_files = [f for f in all_files if len(f.replace(".csv", "").split("_")) == 4]
    print(f"\n  Found {len(churn_files)} churn files")

    # Use last 5 files: 3 for training, 2 for test
    n_train = 3
    n_test = 2
    n_total = n_train + n_test
    if len(churn_files) < n_total:
        print(f"  ERROR: Need at least {n_total} files, found {len(churn_files)}")
        return

    selected = churn_files[-n_total:]
    train_files = selected[:n_train]
    test_files = selected[-n_test:]

    print(f"\n  Training files ({n_train}):")
    for f in train_files:
        print(f"    {f}")
    print(f"\n  Test files ({n_test}):")
    for f in test_files:
        print(f"    {f}")

    # ── 2. Load and build features ──
    print("\n  [Step 1] Loading data and computing features...")
    t0 = time.time()

    def load_and_prepare(file_list, label=""):
        all_feats = []
        all_y = []
        for fname in file_list:
            fpath = os.path.join(churn_dir, fname)
            df, item_cols = load_churn_file(fpath)
            feats, y, _ = prepare_features_and_target(df, item_cols)
            all_feats.append(feats)
            all_y.append(y)
            print(f"    {fname}: {len(feats):,} rows, churn={y.mean():.1%}")

        X = pd.concat(all_feats, ignore_index=True)
        y = np.concatenate(all_y)
        return X, y

    print("\n  Training data:")
    X_train, y_train = load_and_prepare(train_files, "train")
    print(f"\n  Total training: {len(X_train):,} rows, churn={y_train.mean():.1%}")

    print("\n  Test data:")
    X_test, y_test = load_and_prepare(test_files, "test")
    print(f"\n  Total test: {len(X_test):,} rows, churn={y_test.mean():.1%}")
    print(f"  ({time.time() - t0:.1f}s)")

    # ── 3. Load global v3 model ──
    print("\n  [Step 2] Loading global v3 model...")
    v3_artifact = joblib.load(os.path.join(BASE_DIR, "models", "churn_v3_extended.joblib"))
    global_model = v3_artifact["model"]
    feature_cols = v3_artifact["feature_cols"]
    print(f"    Features: {len(feature_cols)}")

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in X_train.columns:
            X_train[col] = 0
        if col not in X_test.columns:
            X_test[col] = 0

    # ── 4. Assign segments ──
    print("\n  [Step 3] Assigning customer segments...")
    seg_train = assign_segment(X_train)
    seg_test = assign_segment(X_test)

    print(f"\n  Training segment distribution:")
    for seg in SEGMENT_NAMES:
        mask = seg_train == seg
        n = mask.sum()
        churn_r = y_train[mask].mean() if n > 0 else 0
        print(f"    {seg:<12} ({SEGMENT_NAMES[seg]:<12}): {n:>8,} ({n/len(X_train)*100:>5.1f}%)  churn={churn_r:.1%}")

    print(f"\n  Test segment distribution:")
    for seg in SEGMENT_NAMES:
        mask = seg_test == seg
        n = mask.sum()
        churn_r = y_test[mask].mean() if n > 0 else 0
        print(f"    {seg:<12} ({SEGMENT_NAMES[seg]:<12}): {n:>8,} ({n/len(X_test)*100:>5.1f}%)  churn={churn_r:.1%}")

    # ── 5. Global model predictions on test set ──
    print("\n  [Step 4] Global model predictions on test set...")
    X_test_rf = X_test[feature_cols].fillna(0)
    global_probs = global_model.predict_proba(X_test_rf)[:, 1]
    global_auc = roc_auc_score(y_test, global_probs)
    print(f"    Global AUC-ROC: {global_auc:.4f}")

    # ── 6. Train segment-specific models ──
    print("\n  [Step 5] Training segment-specific models...")
    t0 = time.time()

    segment_results = {}
    segment_models = {}

    for seg_name in SEGMENT_NAMES:
        mask_tr = seg_train == seg_name
        mask_te = seg_test == seg_name
        n_tr = mask_tr.sum()
        n_te = mask_te.sum()

        print(f"\n  --- {seg_name} ({SEGMENT_NAMES[seg_name]}) ---")
        print(f"    Train: {n_tr:,}  Test: {n_te:,}")

        # Global model AUC on this segment
        if n_te > 0 and len(np.unique(y_test[mask_te])) == 2:
            global_seg_auc = roc_auc_score(y_test[mask_te], global_probs[mask_te])
            global_seg_f1 = f1_score(y_test[mask_te], (global_probs[mask_te] >= 0.5).astype(int))
        else:
            global_seg_auc = np.nan
            global_seg_f1 = np.nan
        print(f"    Global model on segment — AUC: {global_seg_auc:.4f}, F1: {global_seg_f1:.4f}")

        # Train segment model if enough data
        seg_auc = np.nan
        seg_f1 = np.nan
        if n_tr >= MIN_SEGMENT_SIZE and n_te >= 100:
            X_seg_tr = X_train.loc[mask_tr, feature_cols].fillna(0)
            y_seg_tr = y_train[mask_tr]
            X_seg_te = X_test.loc[mask_te, feature_cols].fillna(0)
            y_seg_te = y_test[mask_te]

            if len(np.unique(y_seg_tr)) < 2:
                print(f"    SKIP: Only one class in training data")
                segment_results[seg_name] = {
                    "n_train": int(n_tr), "n_test": int(n_te),
                    "global_auc": global_seg_auc, "global_f1": global_seg_f1,
                    "segment_auc": np.nan, "segment_f1": np.nan,
                    "trained": False,
                }
                continue

            rf_seg = RandomForestClassifier(**RF_PARAMS)
            rf_seg.fit(X_seg_tr, y_seg_tr)
            seg_probs = rf_seg.predict_proba(X_seg_te)[:, 1]

            if len(np.unique(y_seg_te)) == 2:
                seg_auc = roc_auc_score(y_seg_te, seg_probs)
                seg_f1 = f1_score(y_seg_te, (seg_probs >= 0.5).astype(int))
            print(f"    Segment model        — AUC: {seg_auc:.4f}, F1: {seg_f1:.4f}")

            delta = seg_auc - global_seg_auc if not np.isnan(seg_auc) and not np.isnan(global_seg_auc) else np.nan
            print(f"    Delta AUC: {delta:+.4f}" if not np.isnan(delta) else "    Delta AUC: N/A")

            segment_models[seg_name] = rf_seg

            # Feature importance: top 10
            imp = rf_seg.feature_importances_
            top_idx = np.argsort(imp)[::-1][:10]
            print(f"    Top 10 features:")
            for rank, idx in enumerate(top_idx):
                print(f"      {rank+1:>2}. {feature_cols[idx]:<35} {imp[idx]:.4f}")
        else:
            reason = f"train={n_tr:,}" if n_tr < MIN_SEGMENT_SIZE else f"test={n_te:,}"
            print(f"    SKIP: Too few samples ({reason})")

        segment_results[seg_name] = {
            "n_train": int(n_tr),
            "n_test": int(n_te),
            "churn_rate_train": float(y_train[mask_tr].mean()) if n_tr > 0 else 0.0,
            "churn_rate_test": float(y_test[mask_te].mean()) if n_te > 0 else 0.0,
            "global_auc": float(global_seg_auc) if not np.isnan(global_seg_auc) else None,
            "global_f1": float(global_seg_f1) if not np.isnan(global_seg_f1) else None,
            "segment_auc": float(seg_auc) if not np.isnan(seg_auc) else None,
            "segment_f1": float(seg_f1) if not np.isnan(seg_f1) else None,
            "trained": seg_name in segment_models,
        }

    print(f"\n  ({time.time() - t0:.1f}s)")

    # ── 7. Summary comparison table ──
    print("\n" + "=" * 90)
    print("  COMPARISON: Global v3 vs Segment-Specific Models")
    print("=" * 90)
    print(f"  {'Segment':<14} {'(Thai)':<14} {'Train':>8} {'Test':>8} {'Churn%':>7} {'Global AUC':>11} {'Seg AUC':>9} {'Delta':>8} {'Winner':>10}")
    print(f"  {'-' * 88}")

    for seg_name in SEGMENT_NAMES:
        r = segment_results[seg_name]
        g_auc = f"{r['global_auc']:.4f}" if r['global_auc'] is not None else "N/A"
        s_auc = f"{r['segment_auc']:.4f}" if r['segment_auc'] is not None else "N/A"

        if r['global_auc'] is not None and r['segment_auc'] is not None:
            delta = r['segment_auc'] - r['global_auc']
            delta_str = f"{delta:+.4f}"
            winner = "Segment" if delta > 0.001 else ("Global" if delta < -0.001 else "Tie")
        else:
            delta_str = "N/A"
            winner = "N/A"

        churn_pct = f"{r['churn_rate_test']*100:.1f}%" if r['churn_rate_test'] is not None else "N/A"
        print(f"  {seg_name:<14} {SEGMENT_NAMES[seg_name]:<14} {r['n_train']:>8,} {r['n_test']:>8,} {churn_pct:>7} {g_auc:>11} {s_auc:>9} {delta_str:>8} {winner:>10}")

    print(f"\n  Global model overall AUC-ROC: {global_auc:.4f}")

    # ── 8. Save best segment models ──
    print("\n  [Step 6] Saving segment models that beat global...")
    saved_count = 0
    for seg_name, model in segment_models.items():
        r = segment_results[seg_name]
        if r['segment_auc'] is not None and r['global_auc'] is not None:
            if r['segment_auc'] > r['global_auc'] + 0.001:
                out_path = os.path.join(BASE_DIR, "models", f"churn_segment_{seg_name.lower()}.joblib")
                artifact = {
                    "model": model,
                    "model_name": f"RF Segment ({seg_name})",
                    "feature_cols": feature_cols,
                    "segment": seg_name,
                    "segment_thai": SEGMENT_NAMES[seg_name],
                    "segment_auc": r["segment_auc"],
                    "global_auc": r["global_auc"],
                    "delta_auc": r["segment_auc"] - r["global_auc"],
                    "n_train": r["n_train"],
                    "n_test": r["n_test"],
                }
                joblib.dump(artifact, out_path)
                print(f"    Saved: {os.path.basename(out_path)} (AUC {r['segment_auc']:.4f}, +{r['segment_auc']-r['global_auc']:.4f})")
                saved_count += 1
    if saved_count == 0:
        print("    No segment model beat the global model significantly.")

    # ── 9. Generate chart ──
    print("\n  [Step 7] Generating charts...")

    # Collect global feature importances for comparison
    global_imp = global_model.feature_importances_

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # --- Panel 1: AUC comparison by segment ---
    ax1 = axes[0, 0]
    segments_with_both = [s for s in SEGMENT_NAMES if
                          segment_results[s]['global_auc'] is not None and
                          segment_results[s]['segment_auc'] is not None]
    x_pos = np.arange(len(segments_with_both))
    bar_w = 0.35
    global_aucs = [segment_results[s]['global_auc'] for s in segments_with_both]
    seg_aucs = [segment_results[s]['segment_auc'] for s in segments_with_both]

    bars1 = ax1.bar(x_pos - bar_w/2, global_aucs, bar_w, label="Global v3",
                    color="#3498db", edgecolor="white", alpha=0.85)
    bars2 = ax1.bar(x_pos + bar_w/2, seg_aucs, bar_w, label="Segment-Specific",
                    color="#e74c3c", edgecolor="white", alpha=0.85)

    for i, (g, s) in enumerate(zip(global_aucs, seg_aucs)):
        ax1.text(i - bar_w/2, g + 0.002, f"{g:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax1.text(i + bar_w/2, s + 0.002, f"{s:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{s}\n({SEGMENT_NAMES[s]})" for s in segments_with_both], fontsize=8)
    ax1.set_ylabel("AUC-ROC")
    ax1.set_title("AUC-ROC: Global v3 vs Segment-Specific Models", fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3, axis="y")
    # Set y-axis to show meaningful range
    all_aucs = global_aucs + seg_aucs
    y_min = max(0.5, min(all_aucs) - 0.03)
    y_max = min(1.0, max(all_aucs) + 0.03)
    ax1.set_ylim(y_min, y_max)

    # --- Panel 2: Feature importance differences ---
    ax2 = axes[0, 1]
    # Show top 10 global features and how they differ across segments
    top_global_idx = np.argsort(global_imp)[::-1][:10]
    top_feat_names = [feature_cols[i] for i in top_global_idx]

    # For each trained segment model, get its importance for these features
    seg_labels = [s for s in segment_models]
    if len(seg_labels) > 0:
        importance_matrix = np.zeros((len(top_feat_names), len(seg_labels) + 1))
        importance_matrix[:, 0] = global_imp[top_global_idx]

        for j, seg_name in enumerate(seg_labels):
            seg_imp = segment_models[seg_name].feature_importances_
            importance_matrix[:, j + 1] = seg_imp[top_global_idx]

        y_feat = np.arange(len(top_feat_names))
        h = 0.8 / (len(seg_labels) + 1)
        colors_list = ["#3498db"] + [SEGMENT_COLORS.get(s, "#95a5a6") for s in seg_labels]
        labels_list = ["Global"] + [f"{s}" for s in seg_labels]

        for j in range(len(seg_labels) + 1):
            ax2.barh(y_feat + j * h - 0.4 + h/2, importance_matrix[:, j], h,
                     label=labels_list[j], color=colors_list[j], alpha=0.8, edgecolor="white")

        ax2.set_yticks(y_feat)
        ax2.set_yticklabels(top_feat_names, fontsize=8)
        ax2.set_xlabel("Feature Importance")
        ax2.legend(fontsize=7, loc="lower right")
    else:
        ax2.text(0.5, 0.5, "No segment models trained", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12)
    ax2.set_title("Top 10 Feature Importance: Global vs Segments", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")

    # --- Panel 3: Segment size distribution ---
    ax3 = axes[1, 0]
    seg_names_ordered = list(SEGMENT_NAMES.keys())
    test_sizes = [segment_results[s]["n_test"] for s in seg_names_ordered]
    colors_ordered = [SEGMENT_COLORS[s] for s in seg_names_ordered]

    bars = ax3.bar(range(len(seg_names_ordered)), test_sizes, color=colors_ordered,
                   edgecolor="white", alpha=0.85)
    for i, (bar, sz) in enumerate(zip(bars, test_sizes)):
        pct = sz / sum(test_sizes) * 100 if sum(test_sizes) > 0 else 0
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(test_sizes)*0.01,
                 f"{sz:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax3.set_xticks(range(len(seg_names_ordered)))
    ax3.set_xticklabels([f"{s}\n({SEGMENT_NAMES[s]})" for s in seg_names_ordered], fontsize=8)
    ax3.set_ylabel("Number of Customers (Test Set)")
    ax3.set_title("Segment Size Distribution", fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # --- Panel 4: Churn rate by segment ---
    ax4 = axes[1, 1]
    churn_rates = [segment_results[s]["churn_rate_test"] * 100 for s in seg_names_ordered]
    bars = ax4.bar(range(len(seg_names_ordered)), churn_rates, color=colors_ordered,
                   edgecolor="white", alpha=0.85)
    for i, (bar, cr) in enumerate(zip(bars, churn_rates)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{cr:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax4.set_xticks(range(len(seg_names_ordered)))
    ax4.set_xticklabels([f"{s}\n({SEGMENT_NAMES[s]})" for s in seg_names_ordered], fontsize=8)
    ax4.set_ylabel("Churn Rate (%)")
    ax4.set_title("Churn Rate by Segment (Test Set)", fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.axhline(y=y_test.mean() * 100, color="black", linestyle="--", alpha=0.5,
                label=f"Overall: {y_test.mean()*100:.1f}%")
    ax4.legend(fontsize=9)

    plt.suptitle(
        f"Phase 38: Segment-Specific Models\n"
        f"Global v3 AUC={global_auc:.4f} | {len(segment_models)} segment models trained | "
        f"{saved_count} models saved",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    charts_dir = os.path.join(BASE_DIR, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    chart_path = os.path.join(charts_dir, "phase38_segment_models.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Chart saved: {chart_path}")

    # ── 10. Final insights ──
    print("\n" + "=" * 70)
    print("  INSIGHTS")
    print("=" * 70)

    for seg_name in SEGMENT_NAMES:
        r = segment_results[seg_name]
        if r['segment_auc'] is not None and r['global_auc'] is not None:
            delta = r['segment_auc'] - r['global_auc']
            if delta > 0.005:
                print(f"  + {seg_name} ({SEGMENT_NAMES[seg_name]}): Segment model WINS by {delta:+.4f} AUC")
            elif delta < -0.005:
                print(f"  - {seg_name} ({SEGMENT_NAMES[seg_name]}): Global model better by {-delta:+.4f} AUC")
            else:
                print(f"  = {seg_name} ({SEGMENT_NAMES[seg_name]}): Roughly tied (delta={delta:+.4f})")
        elif not r['trained']:
            print(f"  ? {seg_name} ({SEGMENT_NAMES[seg_name]}): Not enough data to train")

    elapsed = time.time() - t0_total
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Done.")


if __name__ == "__main__":
    main()
