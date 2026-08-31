"""
Fidelity benchmarking for RAPTOR's synthetic attack data.

Three metrics:
  1. KS test per feature  — Kolmogorov-Smirnov statistic comparing legit vs fraud
     marginal distributions. Stat < 0.3 means healthy overlap; stat > 0.7 means
     the feature is trivially separable (synthetic artifact, not real signal).
     We report mean stat, % features with realistic overlap (< 0.5), and flag
     zero-overlap features (stat ≥ 0.95).

  2. DCR ratio  — Distance to Closest Record (intra vs cross class).
     For each fraud test event, find:
       d_intra: L2 distance to nearest fraud event in the training set
       d_cross: L2 distance to nearest legit event in the training set
     DCR ratio = mean(d_cross) / mean(d_intra).
     Ratio > 1 means fraud events are genuinely more similar to other fraud
     than to legit — the joint distribution is separable even when marginals
     overlap. Proves generation not memorization when d_intra is also large
     relative to the feature scale.

  3. TSTR (Train on Synthetic, Test on Synthetic)  — PR-AUC on the held-out
     test set, read from blue/results/summary.json. This is our TSTR score:
     if the detector trained on synthetic data achieves high PR-AUC on held-out
     synthetic, the synthetic data is realistic enough to support learning.

Run:
    python -m red.fidelity
Outputs:
    blue/results/fidelity.json
"""

import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "blue" / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.json"


def _ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic (pure numpy — no scipy required)."""
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    all_vals = np.concatenate([a_sorted, b_sorted])
    cdf_a = np.searchsorted(a_sorted, all_vals, side='right') / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, all_vals, side='right') / len(b_sorted)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _dcr_ratio(X_fraud_test: np.ndarray,
               X_fraud_train: np.ndarray,
               X_legit_train: np.ndarray,
               max_samples: int = 400) -> dict:
    """
    Subsample, then compute mean distance from each fraud test event to its
    nearest fraud-train neighbor (d_intra) and nearest legit-train neighbor
    (d_cross). Returns ratio and raw means.
    """
    rng = np.random.default_rng(42)

    def _subsample(X, n):
        if len(X) <= n:
            return X
        return X[rng.choice(len(X), n, replace=False)]

    Xft = _subsample(X_fraud_test, max_samples)
    Xfr = _subsample(X_fraud_train, max_samples)
    Xlr = _subsample(X_legit_train, max_samples)

    if len(Xft) == 0 or len(Xfr) == 0 or len(Xlr) == 0:
        return {"dcr_ratio": None, "d_intra_mean": None, "d_cross_mean": None}

    # Normalise columns to [0,1] range so no single feature dominates L2
    all_data = np.vstack([Xft, Xfr, Xlr])
    col_min = all_data.min(axis=0)
    col_max = all_data.max(axis=0)
    col_range = np.where(col_max - col_min > 0, col_max - col_min, 1.0)
    Xft_n = (Xft - col_min) / col_range
    Xfr_n = (Xfr - col_min) / col_range
    Xlr_n = (Xlr - col_min) / col_range

    # Chunked nearest-neighbour to avoid OOM on large arrays
    def _mean_nn_dist(query, reference, chunk=100):
        dists = []
        for i in range(0, len(query), chunk):
            q = query[i:i+chunk]  # (chunk, F)
            diff = q[:, None, :] - reference[None, :, :]  # (chunk, ref, F)
            d = np.sqrt((diff ** 2).sum(axis=-1))          # (chunk, ref)
            dists.append(d.min(axis=1))
        return float(np.concatenate(dists).mean())

    d_intra = _mean_nn_dist(Xft_n, Xfr_n)
    d_cross = _mean_nn_dist(Xft_n, Xlr_n)
    ratio = d_cross / d_intra if d_intra > 1e-9 else None

    return {
        "dcr_ratio": round(ratio, 3) if ratio is not None else None,
        "d_intra_mean": round(d_intra, 4),
        "d_cross_mean": round(d_cross, 4),
    }


def run_fidelity() -> dict:
    from blue.featurize import load_channel, get_feature_matrix
    from blue.train import chronological_split

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError("Run blue stack first: python -m blue.run_blue")

    summary = json.loads(SUMMARY_PATH.read_text())
    results = {}

    channels = ["txn-sequence", "kyc-session", "agent-payment", "chat-call"]
    for channel in channels:
        print(f"\n── fidelity: {channel} ──")
        try:
            df = load_channel(channel)
            X, y, meta, feat_cols = get_feature_matrix(df, channel)
        except Exception as e:
            print(f"  skip ({e})")
            continue

        train_idx, _, test_idx = chronological_split(meta)
        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        X_fraud_train = X_train[y_train == 1]
        X_legit_train = X_train[y_train == 0]
        X_fraud_test  = X_test[y_test == 1]
        X_legit_test  = X_test[y_test == 0]

        # ── 1. KS test per feature ────────────────────────────────────
        n_feats = len(feat_cols)
        ks_stats = []
        zero_overlap_feats = []
        for i, col in enumerate(feat_cols):
            fraud_vals = X[y == 1, i]
            legit_vals = X[y == 0, i]
            if len(fraud_vals) == 0 or len(legit_vals) == 0:
                continue
            stat = _ks_stat(fraud_vals, legit_vals)
            ks_stats.append({"feature": col, "ks_stat": round(stat, 4)})
            if stat >= 0.95:
                zero_overlap_feats.append(col)

        ks_values = [r["ks_stat"] for r in ks_stats]
        mean_ks = round(float(np.mean(ks_values)), 4) if ks_values else 0
        pct_overlap = round(float(np.mean([s < 0.5 for s in ks_values]) * 100), 1)
        pct_near_perfect = round(float(np.mean([s >= 0.95 for s in ks_values]) * 100), 1)

        print(f"  KS: mean={mean_ks:.3f}  overlap(<0.5)={pct_overlap}%  "
              f"near-perfect-sep(≥0.95)={pct_near_perfect}%  "
              f"zero-overlap feats: {len(zero_overlap_feats)}")

        # ── 2. DCR ratio ──────────────────────────────────────────────
        dcr = _dcr_ratio(X_fraud_test, X_fraud_train, X_legit_train)
        print(f"  DCR: ratio={dcr['dcr_ratio']}  "
              f"d_intra={dcr['d_intra_mean']}  d_cross={dcr['d_cross_mean']}")

        # ── 3. TSTR from summary ──────────────────────────────────────
        ch_key = channel.replace("-", "_") if channel.replace("-", "_") in summary else channel
        ch_summary = summary.get(channel, summary.get(ch_key, {}))
        tstr_prauc = ch_summary.get("prauc_combined", {}).get("mean")
        tstr_roc   = ch_summary.get("roc_auc_combined")
        print(f"  TSTR PR-AUC={tstr_prauc}  ROC-AUC={tstr_roc}")

        results[channel] = {
            "n_features": n_feats,
            "ks_mean": mean_ks,
            "ks_pct_realistic_overlap": pct_overlap,
            "ks_pct_near_perfect_separation": pct_near_perfect,
            "ks_zero_overlap_features": zero_overlap_feats,
            "ks_per_feature": sorted(ks_stats, key=lambda r: r["ks_stat"], reverse=True),
            "dcr_ratio": dcr["dcr_ratio"],
            "dcr_d_intra_mean": dcr["d_intra_mean"],
            "dcr_d_cross_mean": dcr["d_cross_mean"],
            "tstr_prauc": tstr_prauc,
            "tstr_roc_auc": tstr_roc,
            "n_fraud_train": int(len(X_fraud_train)),
            "n_legit_train": int(len(X_legit_train)),
            "n_fraud_test": int(len(X_fraud_test)),
        }

    out_path = RESULTS_DIR / "fidelity.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nFidelity results → {out_path}")
    return results


if __name__ == "__main__":
    run_fidelity()
