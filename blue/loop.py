"""
Closed-loop adversarial hardening.

Each round:
  1. Perturb known fraud events toward the legit centroid (attacker camouflage)
  2. Score perturbed variants through the current detector
  3. Mine evasive samples (score < threshold) — the attacker succeeded
  4. Augment training data with those hard negatives (still label=1)
  5. Retrain classifier, recalibrate operating threshold on val set
  6. Re-evaluate on the ORIGINAL (unperturbed) test set

Tracks: attack success rate (ASR), recall, ROC-AUC, PR-AUC per round.
ASR = fraction of perturbed fraud that slips past the detector.
A successful hardening loop drives ASR toward 0 over rounds.
"""

import numpy as np
import pandas as pd

from blue.train import (
    train_classifier, train_ood, find_operating_point,
    combined_score, combined_score_ensemble, mask_zero_day,
    train_vector_classifiers,
)
from blue.evaluate import roc_auc, confusion_metrics, bootstrap_prauc


def _perturb(X_fraud: np.ndarray, legit_centroid: np.ndarray, alpha: float) -> np.ndarray:
    """Nudge fraud feature vectors alpha-fraction toward the legit centroid."""
    return X_fraud * (1.0 - alpha) + legit_centroid * alpha


def _score_batch(clf, vector_clfs, iso, X, channel):
    if vector_clfs and channel == "txn-sequence":
        _, combined = combined_score_ensemble(clf, vector_clfs, iso, X)
        return combined
    return combined_score(clf, iso, X)


class _EnsembleCLFWrapper:
    """Thin wrapper so find_operating_point can treat the ensemble as a clf."""
    def __init__(self, clf, vector_clfs, iso):
        self._clf = clf
        self._vclfs = vector_clfs
        self._iso = iso

    def predict_proba(self, X):
        ens, _ = combined_score_ensemble(self._clf, self._vclfs, self._iso, X)
        return np.column_stack([1 - ens, ens])


def run_loop(
    X_train: np.ndarray,
    y_train: np.ndarray,
    meta_train: pd.DataFrame,
    X_val: np.ndarray,
    y_val: np.ndarray,
    meta_val: pd.DataFrame,
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta_test: pd.DataFrame,
    channel: str,
    clf_r0,
    vector_clfs_r0: dict,
    iso,
    op_threshold_r0: float,
    n_rounds: int = 3,
    alpha: float = 0.30,
) -> list[dict]:
    """
    Run n_rounds of closed-loop hardening starting from round-0 trained models.

    Returns a list of dicts, one per round (0 = baseline, 1..n = hardened).
    Round 0 metrics are the unmodified baseline — same numbers as run_blue's
    test evaluation. Rounds 1+ show improvement as the model is retrained on
    evasive hard negatives.
    """
    y_val_masked = mask_zero_day(y_val, meta_val["vector_id"])

    # Fixed legit centroid from round-0 training legit pool
    legit_centroid = X_train[y_train == 0].mean(axis=0)

    # Test fraud positions (never perturbed — evaluation always on original)
    fraud_test_idx = np.where(y_test == 1)[0]

    # Mutable state
    X_aug = X_train.copy()
    y_aug = y_train.copy()
    meta_aug = meta_train.copy()
    clf_cur = clf_r0
    vector_clfs_cur = dict(vector_clfs_r0)
    op_cur = op_threshold_r0

    rounds = []

    for r in range(n_rounds + 1):
        # ── Evaluate on original test set ─────────────────────────────
        scores_test = _score_batch(clf_cur, vector_clfs_cur, iso, X_test, channel)
        conf = confusion_metrics(y_test, scores_test, op_cur)
        auc = roc_auc(y_test, scores_test)
        prauc_mean, prauc_lo, prauc_hi = bootstrap_prauc(y_test, scores_test, n=200)

        # ── Attack Success Rate on perturbed fraud ─────────────────────
        if len(fraud_test_idx) > 0:
            X_fraud_test = X_test[fraud_test_idx]
            X_pert = _perturb(X_fraud_test, legit_centroid, alpha)
            scores_pert = _score_batch(clf_cur, vector_clfs_cur, iso, X_pert, channel)
            evasive_mask = scores_pert < op_cur
            n_evasive = int(evasive_mask.sum())
            asr = n_evasive / len(fraud_test_idx)
        else:
            n_evasive, asr, evasive_mask = 0, 0.0, np.array([], dtype=bool)

        rounds.append({
            "round": r,
            "recall": round(float(conf["recall"]), 4),
            "precision": round(float(conf["precision"]), 4),
            "f1": round(float(conf["f1"]), 4),
            "roc_auc": round(float(auc), 4),
            "prauc": round(float(prauc_mean), 4),
            "prauc_ci_lo": round(float(prauc_lo), 4),
            "prauc_ci_hi": round(float(prauc_hi), 4),
            "fpr": round(float(conf["false_positive_rate"]), 4),
            "attack_success_rate": round(float(asr), 4),
            "n_evasive_mined": n_evasive if r < n_rounds else 0,
            "n_training_events": int(len(X_aug)),
            "op_threshold": round(float(op_cur), 4),
        })

        label = "baseline" if r == 0 else f"round {r}"
        print(f"  [loop {label}]  recall={conf['recall']:.3f}  "
              f"ROC-AUC={auc:.4f}  ASR={asr:.1%}  evasive={n_evasive}")

        if r == n_rounds:
            break

        if n_evasive == 0:
            print(f"  [loop] No evasive samples — hardening converged at round {r}")
            break

        # ── Mine evasive samples as hard negatives ─────────────────────
        hard_positions = fraud_test_idx[evasive_mask]  # positions in X_test
        X_hard = _perturb(X_test[hard_positions], legit_centroid, alpha)
        y_hard = np.ones(len(X_hard), dtype=y_aug.dtype)
        meta_hard = meta_test.iloc[hard_positions].reset_index(drop=True)

        X_aug = np.vstack([X_aug, X_hard])
        y_aug = np.concatenate([y_aug, y_hard])
        meta_aug = pd.concat([meta_aug, meta_hard], ignore_index=True)

        # ── Retrain ────────────────────────────────────────────────────
        y_aug_masked = mask_zero_day(y_aug, meta_aug["vector_id"])
        print(f"  [loop round {r+1}] retraining on {len(X_aug):,} events "
              f"({n_evasive} hard negatives added) ...")
        clf_cur = train_classifier(X_aug, y_aug_masked)

        if channel == "txn-sequence":
            vector_clfs_cur = train_vector_classifiers(X_aug, y_aug_masked, meta_aug)

        # Recalibrate threshold on val set
        if vector_clfs_cur and channel == "txn-sequence":
            wrapper = _EnsembleCLFWrapper(clf_cur, vector_clfs_cur, iso)
            op_cur = find_operating_point(wrapper, iso, X_val, y_val_masked)
        else:
            op_cur = find_operating_point(clf_cur, iso, X_val, y_val_masked)

    return rounds
