"""
Trains a classifier on the featurized transactions and reports how well it
catches the structuring/threshold-evasion pattern.

Split by ACCOUNT, not by row: all of one account's transactions go entirely
into train or entirely into test. If you split by row instead, the model
could see 3 of an account's 5 burst transactions in training and "cheat" on
the other 2 - splitting by account is what proves it generalizes to accounts
it has never seen before, which is the actual claim you want to make.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

df = pd.read_csv("transactions_features.csv", parse_dates=["timestamp"])

feature_cols = [
    "amount",
    "amount_to_threshold_ratio",
    "account_age_days",
    "txn_count_last_1hr",
    "sum_amount_last_1hr",
    "distinct_merchants_last_1hr",
    "same_mcc_count_last_1hr",
    "time_since_prev_txn_sec",
]

rng = np.random.default_rng(0)
accounts = df["account_id"].unique()
rng.shuffle(accounts)
split = int(len(accounts) * 0.8)
train_accounts, test_accounts = set(accounts[:split]), set(accounts[split:])

train = df[df["account_id"].isin(train_accounts)]
test = df[df["account_id"].isin(test_accounts)]

X_train, y_train = train[feature_cols], train["label"]
X_test, y_test = test[feature_cols], test["label"]

clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0, class_weight="balanced")
clf.fit(X_train, y_train)

probs = clf.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

print("=== Held-out accounts the model never saw during training ===")
print(f"train accounts: {len(train_accounts)}, test accounts: {len(test_accounts)}")
print(f"test fraud rows: {y_test.sum()} / {len(y_test)}\n")

print("=== Classification report (threshold = 0.5) ===")
print(classification_report(y_test, preds, target_names=["legit", "structuring_fraud"]))

print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}\n")

print("=== Feature importances (what the model actually keyed on) ===")
importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importances.to_string())

print("\n=== A few flagged transactions from the held-out set ===")
test_out = test.copy()
test_out["fraud_score"] = probs
flagged = test_out[test_out["fraud_score"] >= 0.5].sort_values("fraud_score", ascending=False)
cols_to_show = ["account_id", "timestamp", "amount", "txn_count_last_1hr", "sum_amount_last_1hr", "label", "fraud_score"]
print(flagged[cols_to_show].head(8).to_string(index=False))
