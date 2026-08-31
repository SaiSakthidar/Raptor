"""
Blue stack orchestrator.

Usage:
    cd mastercard-hackathon
    python -m blue.run_blue

Runs per channel:
    1. Load + featurize
    2. Chronological split (70/15/15)
    3. Zero-day label masking on train+val
    4. Train XGBoost + IsolationForest OOD
    5. Find operating point on val set (target FPR ≤ 1%)
    6. Evaluate on test set: per-vector recall, coverage matrix,
       bootstrap PR-AUC, expected loss, policy distribution
    7. Save results to blue/results/<channel>_results.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score, classification_report

from blue.featurize import load_channel, get_feature_matrix, CHANNEL_FEATURES
from blue.train import (
    chronological_split, mask_zero_day, train_classifier,
    train_vector_classifiers, combined_score_ensemble,
    train_ood, find_operating_point, combined_score, ZERO_DAY_VECTORS,
    save_artifacts,
)
from blue.evaluate import (
    per_vector_recall, coverage_matrix, bootstrap_prauc, roc_auc,
    confusion_metrics, friction_cost,
    expected_loss, ood_vs_supervised_breakdown, apply_policy,
)
from blue.loop import run_loop

RESULTS_DIR = Path(__file__).parent / "results"
CHANNELS = ["txn-sequence", "kyc-session", "agent-payment", "chat-call"]


def run_channel(channel: str) -> dict:
    print(f"\n{'='*60}")
    print(f"CHANNEL: {channel}")
    print(f"{'='*60}")

    # ── 1. Load + featurize ────────────────────────────────────────
    df = load_channel(channel)
    X, y, meta, feat_cols = get_feature_matrix(df, channel)
    print(f"  total events: {len(X):,}  fraud: {y.sum():,}  "
          f"legit: {(y==0).sum():,}")

    # ── 2. Chronological split ─────────────────────────────────────
    # meta is already sorted by timestamp (load_channel sorts it)
    train_idx, val_idx, test_idx = chronological_split(meta)
    print(f"  split: train={len(train_idx):,}  val={len(val_idx):,}  "
          f"test={len(test_idx):,}")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    meta_train = meta.iloc[train_idx].reset_index(drop=True)
    meta_val = meta.iloc[val_idx].reset_index(drop=True)
    meta_test = meta.iloc[test_idx].reset_index(drop=True)

    # ── 3. Zero-day label masking ─────────────────────────────────
    y_train_masked = mask_zero_day(y_train, meta_train["vector_id"])
    y_val_masked = mask_zero_day(y_val, meta_val["vector_id"])

    zd_train_suppressed = int((y_train - y_train_masked).sum())
    print(f"  zero-day masking: {zd_train_suppressed} fraud labels suppressed in train")

    # ── 4. Train classifier + per-vector specialists + OOD ───────
    clf = train_classifier(X_train, y_train_masked)

    # Per-vector specialists only for txn-sequence (KYC/agent already high recall)
    vector_clfs = {}
    if channel == "txn-sequence":
        vector_clfs = train_vector_classifiers(X_train, y_train_masked, meta_train)

    X_train_legit = X_train[y_train_masked == 0]
    iso = train_ood(X_train_legit)

    # ── 5. Operating point (val set, ensemble scores) ─────────────
    if vector_clfs:
        _, combined_val = combined_score_ensemble(clf, vector_clfs, iso, X_val)
        # reuse find_operating_point by faking clf with a lambda wrapper
        class _EnsembleCLF:
            def predict_proba(self, X):
                ens, _ = combined_score_ensemble(clf, vector_clfs, iso, X)
                return np.column_stack([1 - ens, ens])
        op_threshold = find_operating_point(_EnsembleCLF(), iso, X_val, y_val_masked)
    else:
        op_threshold = find_operating_point(clf, iso, X_val, y_val_masked)

    # ── 6. Test set scores ────────────────────────────────────────
    if vector_clfs:
        clf_probs_test, combined_scores_test = combined_score_ensemble(
            clf, vector_clfs, iso, X_test)
    else:
        clf_probs_test = clf.predict_proba(X_test)[:, 1]
        combined_scores_test = combined_score(clf, iso, X_test)

    # ── 7. Evaluate ───────────────────────────────────────────────
    # PR-AUC (combined score, true labels)
    prauc_mean, prauc_lo, prauc_hi = bootstrap_prauc(y_test, combined_scores_test)
    prauc_clf_mean, _, _ = bootstrap_prauc(y_test, clf_probs_test)
    roc_auc_combined = roc_auc(y_test, combined_scores_test)
    roc_auc_clf = roc_auc(y_test, clf_probs_test)
    print(f"\n  PR-AUC (combined): {prauc_mean:.3f}  "
          f"90% CI [{prauc_lo:.3f}, {prauc_hi:.3f}]")
    print(f"  PR-AUC (supervised only): {prauc_clf_mean:.3f}")
    print(f"  ROC-AUC (combined): {roc_auc_combined:.3f}  "
          f"(supervised only: {roc_auc_clf:.3f})")

    # Confusion matrix / precision / recall / F1 / FPR at the operating point
    conf = confusion_metrics(y_test, combined_scores_test, op_threshold)
    print(f"\n  Confusion matrix @ operating point (threshold={op_threshold:.3f}):")
    print(f"    TP={conf['tp']}  FP={conf['fp']}  TN={conf['tn']}  FN={conf['fn']}")
    print(f"    precision={conf['precision']:.3f}  recall={conf['recall']:.3f}  "
          f"f1={conf['f1']:.3f}  FPR={conf['false_positive_rate']:.3%}  "
          f"specificity={conf['specificity']:.3f}")

    # Friction cost: legit volume wrongly held/declined (mirror of expected_loss)
    friction = friction_cost(meta_test, combined_scores_test, op_threshold)
    print(f"\n  Friction: {friction['legit_events_blocked']} of "
          f"{friction.get('total_legit_events', 0)} legitimate events blocked")
    print(f"  Friction cost: ₹{friction['friction_inr']:,.0f}  (${friction['friction_usd']:,.0f} USD)")

    # Per-vector recall
    pvr = per_vector_recall(meta_test, clf_probs_test,
                             combined_scores_test, op_threshold)
    print(f"\n  Per-vector recall (test set):")
    if pvr.empty:
        print("  (no fraud actors landed in the test window for this channel)")
    else:
        print(pvr.to_string(index=False))

    # OOD story: zero-day + V009
    ood_story = ood_vs_supervised_breakdown(
        meta_test, clf_probs_test, combined_scores_test, op_threshold)
    if not ood_story.empty:
        print(f"\n  OOD lift (zero-day + model-evasion vectors):")
        print(ood_story.to_string(index=False))

    # Coverage matrix
    cov = coverage_matrix(meta_test, combined_scores_test, op_threshold)
    print(f"\n  Coverage matrix (modality × vector):")
    if cov.empty:
        print("  (no fraud actors landed in the test window for this channel)")
    else:
        print(cov.to_string())

    # Expected loss
    loss = expected_loss(meta_test, combined_scores_test, op_threshold)
    print(f"\n  Expected loss: {loss['approved_fraud_events']} fraud events approved")
    print(f"  Loss: ₹{loss['loss_inr']:,.0f}  (${loss['loss_usd']:,.0f} USD)")

    # Policy distribution on ALL test fraud events
    policies = apply_policy(combined_scores_test, op_threshold)
    policy_counts = pd.Series(policies).value_counts()
    print(f"\n  Policy distribution (all test events):")
    print(policy_counts.to_string())

    # Feature importances
    try:
        fi = clf.feature_importances_
        fi_index = feat_cols[:len(fi)]  # guard against length mismatch after dedup
        fi_df = pd.Series(fi, index=fi_index).sort_values(ascending=False).head(10)
        print(f"\n  Top 10 features:")
        print(fi_df.to_string())
    except AttributeError:
        pass

    # ── 8. Persist model artifacts for live inference ─────────────
    artifact_path = save_artifacts(channel, clf, vector_clfs, iso, op_threshold, feat_cols)
    print(f"\n  Model artifacts → {artifact_path}")

    # ── 8b. Closed-loop adversarial hardening (3 rounds) ──────────
    print(f"\n  {'─'*50}")
    print(f"  CLOSED-LOOP HARDENING  (channel={channel}, rounds=3, alpha=0.30)")
    print(f"  {'─'*50}")
    loop_rounds = run_loop(
        X_train=X_train, y_train=y_train, meta_train=meta_train,
        X_val=X_val, y_val=y_val, meta_val=meta_val,
        X_test=X_test, y_test=y_test, meta_test=meta_test,
        channel=channel,
        clf_r0=clf, vector_clfs_r0=vector_clfs, iso=iso,
        op_threshold_r0=op_threshold,
        n_rounds=3, alpha=0.30,
    )
    print(f"\n  Hardening summary:")
    for rd in loop_rounds:
        asr_str = f"{rd['attack_success_rate']:.1%}"
        print(f"    Round {rd['round']}: recall={rd['recall']:.3f}  "
              f"ROC-AUC={rd['roc_auc']:.4f}  ASR={asr_str}  "
              f"evasive_mined={rd['n_evasive_mined']}")

    # ── 9. Save results ────────────────────────────────────────────
    result = {
        "channel": channel,
        "n_total": len(X),
        "n_fraud": int(y.sum()),
        "operating_point": float(op_threshold),
        "prauc_combined": {"mean": prauc_mean, "ci_lo": prauc_lo, "ci_hi": prauc_hi},
        "prauc_supervised": prauc_clf_mean,
        "roc_auc_combined": roc_auc_combined,
        "roc_auc_supervised": roc_auc_clf,
        "confusion_matrix": conf,
        "per_vector_recall": pvr.to_dict(orient="records"),
        "ood_story": ood_story.to_dict(orient="records") if not ood_story.empty else [],
        "expected_loss": loss,
        "friction_cost": friction,
        "policy_distribution": policy_counts.to_dict(),
        "hardening_loop": loop_rounds,
    }
    return result


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = {}

    for channel in CHANNELS:
        try:
            result = run_channel(channel)
            all_results[channel] = result
            out_path = RESULTS_DIR / f"{channel.replace('-','_')}_results.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n  Saved → {out_path}")
        except Exception as e:
            print(f"\n  ERROR on {channel}: {e}")
            import traceback; traceback.print_exc()

    # Combined summary
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nAll results saved to blue/results/")
    print(f"Combined summary: {summary_path}")

    # Fidelity benchmarking
    print(f"\n{'='*60}")
    print("FIDELITY BENCHMARKING")
    print(f"{'='*60}")
    try:
        from red.fidelity import run_fidelity
        run_fidelity()
    except Exception as e:
        print(f"  Fidelity benchmark failed: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
