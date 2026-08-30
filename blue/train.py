"""
Training pipeline with:
  - Chronological 70 / 15 / 15 split (not random)
  - Zero-day label masking: V011, V012, V013 are held out entirely from training
  - Supervised classifier (XGBoost with RF fallback)
  - OOD detector (IsolationForest) trained on legit-only training events
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ZERO_DAY_VECTORS = {"V011", "V012", "V013"}
ARTIFACTS_DIR = Path(__file__).parent / "results"


def save_artifacts(channel: str, clf, vector_clfs: dict, iso, op_threshold: float,
                    feat_cols: list[str]) -> Path:
    """
    Persist everything needed to score a brand-new event later without
    retraining: the cross-vector classifier, per-vector specialists, the
    OOD forest (with its calibration range baked in via train_ood), the
    operating threshold, and the exact feature column order. This is what
    lets the dashboard's live-simulation endpoint run real inference.
    """
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    path = ARTIFACTS_DIR / f"{channel.replace('-', '_')}_model.joblib"
    joblib.dump({
        "clf": clf,
        "vector_clfs": vector_clfs,
        "iso": iso,
        "op_threshold": op_threshold,
        "feat_cols": feat_cols,
    }, path)
    return path


def chronological_split(meta: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    """
    Split PER VECTOR by time percentile, then union across vectors.

    A single global timestamp cutoff across all vectors mixed together means
    which vectors get any test-set representation at all is an accident of
    how their individual timelines happen to interleave — a vector whose
    attacks all land early in the window never gets measured, even though
    plenty of vectors with later-landing attacks do. Splitting each vector
    on its own 70/15/15 (using its position in the already time-sorted
    `meta`) guarantees every vector with enough fraud actors gets genuine
    held-out representation, while still holding out each vector's most
    recent events specifically.

    meta must be sorted by timestamp already. Returns (train_idx, val_idx,
    test_idx) as np.ndarray of integer positions into the original array.
    """
    train_parts, val_parts, test_parts = [], [], []
    positions = np.arange(len(meta))
    for vid in meta["vector_id"].unique():
        idx = positions[meta["vector_id"].values == vid]  # preserves time order
        n = len(idx)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))
        train_parts.append(idx[:train_end])
        val_parts.append(idx[train_end:val_end])
        test_parts.append(idx[val_end:])

    train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int)
    val_idx = np.sort(np.concatenate(val_parts)) if val_parts else np.array([], dtype=int)
    test_idx = np.sort(np.concatenate(test_parts)) if test_parts else np.array([], dtype=int)
    return train_idx, val_idx, test_idx


def mask_zero_day(y: np.ndarray, vector_ids: pd.Series) -> np.ndarray:
    """
    In the training/val set, suppress labels for held-out vectors so the
    supervised model never sees them as fraud. Returns a copy of y.
    """
    y_masked = y.copy()
    mask = vector_ids.isin(ZERO_DAY_VECTORS).values
    y_masked[mask] = 0
    return y_masked


def train_classifier(X_train: np.ndarray, y_train: np.ndarray):
    """
    Train LightGBM if available, else HistGradientBoostingClassifier.
    Both are gradient boosting; LightGBM is faster and more accurate.
    """
    pos = y_train.sum()
    neg = len(y_train) - pos
    weight_pos = min(neg / max(pos, 1), 10.0)

    try:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            scale_pos_weight=weight_pos,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        clf.fit(X_train, y_train)
        print(f"  classifier: LightGBM  pos={int(pos)}  neg={int(neg)}  "
              f"scale_pos_weight={weight_pos:.1f}")
        return clf
    except Exception:
        pass

    from sklearn.ensemble import HistGradientBoostingClassifier
    sample_weight = np.where(y_train == 1, weight_pos, 1.0)
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05,
        max_leaf_nodes=63, random_state=42, early_stopping=False,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    print(f"  classifier: HistGradBoost  pos={int(pos)}  neg={int(neg)}  "
          f"weight_pos={weight_pos:.1f}")
    return clf


def train_ood(X_train_legit: np.ndarray):
    """
    IsolationForest trained only on legit training events. Also computes a
    FIXED calibration range (min/max of raw decision_function on the training
    legit pool) so that ood_norm is comparable across any future batch
    (train/val/test/diagnostic) — normalizing against the batch itself would
    make the same absolute anomaly score map to different values depending
    on what else is in that batch.
    """
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train_legit)
    raw_train = iso.decision_function(X_train_legit)
    iso._ood_cal_min = float(raw_train.min())
    iso._ood_cal_max = float(raw_train.max())
    print(f"  OOD (IsolationForest): trained on {len(X_train_legit)} legit events")
    return iso


def _ood_norm(iso, X: np.ndarray) -> np.ndarray:
    """Normalise raw_ood using the FIXED train-legit calibration range."""
    raw_ood = iso.decision_function(X)
    lo, hi = iso._ood_cal_min, iso._ood_cal_max
    ood_norm = 1.0 - (raw_ood - lo) / (hi - lo + 1e-9)
    return np.clip(ood_norm, 0.0, 1.0)


def find_operating_point(clf, iso, X_val, y_val_true, target_fpr=0.01,
                          alpha=0.6):
    """
    Find the LOWEST threshold where FPR ≤ target_fpr on the validation set.
    Scanning from high threshold (strict) to low (permissive): each step we
    lower T as long as FPR stays within budget, stopping when FPR exceeds it.
    """
    scores = combined_score(clf, iso, X_val, alpha)
    neg_mask = y_val_true == 0
    pos_mask = y_val_true == 1

    if not neg_mask.any() or not pos_mask.any():
        print("  operating point: val set has no fraud or no legit events — "
              "defaulting threshold to 0.5")
        return 0.5

    # Scan from high → low; keep lowering threshold while FPR ≤ budget
    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    best_thresh = float(scores.max())   # most conservative default (flag nothing)

    for t in sorted(thresholds, reverse=True):  # high to low
        fpr = (scores[neg_mask] >= t).mean()
        if fpr <= target_fpr:
            best_thresh = t   # still feasible → try going lower
        else:
            break             # FPR exceeded budget → stop, use last best_thresh

    preds = (scores >= best_thresh).astype(int)
    achieved_fpr = preds[neg_mask].mean() if neg_mask.any() else 0.0
    achieved_recall = preds[pos_mask].mean() if pos_mask.any() else 0.0
    print(f"  operating point: threshold={best_thresh:.3f}  "
          f"FPR={achieved_fpr:.3%}  recall={achieved_recall:.3%}")
    return best_thresh


def train_vector_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    meta_train: pd.DataFrame,
    min_fraud: int = 10,
) -> dict:
    """
    Train one LightGBM per vector_id: each specialist sees only its own
    vector's fraud events + the full legit pool.  Returns {vector_id: clf}.
    Vectors with fewer than min_fraud training examples are skipped.
    """
    try:
        import lightgbm as lgb
        _backend = "lgb"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier as _HGBC
        _backend = "hgbc"

    vector_clfs = {}
    legit_mask = y_train == 0
    X_legit = X_train[legit_mask]
    y_legit = y_train[legit_mask]

    for vid in sorted(meta_train["vector_id"].unique()):
        fraud_mask = (meta_train["vector_id"].values == vid) & (y_train == 1)
        n_fraud = fraud_mask.sum()
        if n_fraud < min_fraud:
            continue

        X_v = np.vstack([X_legit, X_train[fraud_mask]])
        y_v = np.concatenate([y_legit, y_train[fraud_mask]])
        # Cap sample weight lower than the cross-vector classifier — small
        # per-vector fraud counts (as low as min_fraud) make aggressive
        # reweighting prone to overfitting spurious splits that then fire
        # confidently on unrelated legit events (see V008 regression).
        weight = min(len(y_legit) / max(n_fraud, 1), 8.0)
        # Shallower trees / fewer rounds for the same reason.
        num_leaves = 7 if n_fraud < 20 else 15

        try:
            if _backend == "lgb":
                clf = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    num_leaves=num_leaves,
                    min_child_samples=max(5, n_fraud // 2),
                    scale_pos_weight=weight,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1,
                )
            else:
                from sklearn.ensemble import HistGradientBoostingClassifier
                clf = HistGradientBoostingClassifier(
                    max_iter=100, learning_rate=0.05,
                    max_leaf_nodes=num_leaves, random_state=42,
                    early_stopping=False,
                )
            sw = np.where(y_v == 1, weight, 1.0)
            clf.fit(X_v, y_v) if _backend == "lgb" else clf.fit(X_v, y_v, sample_weight=sw)

            # Reject specialists that fire on >0.5% of the legit pool at
            # prob > 0.5 — a mis-calibrated specialist that broadly flags
            # unrelated legit events would otherwise poison the ensemble max.
            legit_probs = clf.predict_proba(X_legit)[:, 1]
            legit_fpr = (legit_probs > 0.5).mean()
            if legit_fpr > 0.005:
                print(f"    Rejecting specialist {vid}: legit FPR@0.5 = {legit_fpr:.3%} (too high)")
                continue

            vector_clfs[vid] = clf
        except Exception as e:
            print(f"    Warning: specialist for {vid} failed ({e})")

    print(f"  per-vector specialists: {len(vector_clfs)} trained "
          f"({sorted(vector_clfs.keys())})")
    return vector_clfs


def ensemble_clf_score(
    cross_clf,
    vector_clfs: dict,
    X: np.ndarray,
) -> np.ndarray:
    """
    Element-wise max across the cross-vector classifier and all per-vector
    specialists.  Running all specialists at inference is legitimate — we
    don't use the vector_id label to select which specialist to run.
    """
    scores = [cross_clf.predict_proba(X)[:, 1]]
    for clf in vector_clfs.values():
        scores.append(clf.predict_proba(X)[:, 1])
    return np.max(scores, axis=0)


def combined_score(clf, iso, X: np.ndarray, alpha=0.6) -> np.ndarray:
    """
    Combined fraud score = alpha * clf_prob + (1 - alpha) * ood_score_norm.
    Uses the FIXED train-legit calibration range set by train_ood(), so
    scores are comparable across train/val/test batches of any size.
    """
    clf_prob = clf.predict_proba(X)[:, 1]
    ood_norm = _ood_norm(iso, X)
    return alpha * clf_prob + (1 - alpha) * ood_norm


def combined_score_ensemble(
    cross_clf,
    vector_clfs: dict,
    iso,
    X: np.ndarray,
    alpha=0.6,
) -> tuple:
    """
    Returns (ensemble_clf_scores, combined_scores) — the ensemble replaces
    the single cross-vector classifier in the combined formula. Uses the
    same fixed train-legit OOD calibration as combined_score().
    """
    ens = ensemble_clf_score(cross_clf, vector_clfs, X)
    ood_norm = _ood_norm(iso, X)
    return ens, alpha * ens + (1 - alpha) * ood_norm
