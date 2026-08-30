"""
Turns raw transaction rows into the numeric feature vector a classifier
actually trains on. This is the "adapter" step from the architecture
discussion: envelope -> feature vector.

Key idea: no single transaction in the structuring burst looks bad on its
own (each is under the Rs 50,000 threshold). The signal only shows up when
you look at the account's own recent transaction VELOCITY - so every feature
here is computed from a rolling window over that account's own prior history.
"""

import pandas as pd

THRESHOLD = 50_000

df = pd.read_csv("transactions_raw.csv", parse_dates=["timestamp"])
df = df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)

features = []
for account_id, grp in df.groupby("account_id"):
    grp = grp.reset_index(drop=True)
    timestamps = grp["timestamp"].tolist()
    amounts = grp["amount"].tolist()
    merchants = grp["merchant_id"].tolist()
    mccs = grp["mcc"].tolist()

    for i in range(len(grp)):
        window_start = timestamps[i] - pd.Timedelta(hours=1)
        # all prior/current txns for this account within the last hour
        window_idx = [j for j in range(i + 1) if timestamps[j] > window_start]

        txn_count_last_1hr = len(window_idx)
        sum_amount_last_1hr = sum(amounts[j] for j in window_idx)
        distinct_merchants_last_1hr = len(set(merchants[j] for j in window_idx))
        same_mcc_count_last_1hr = sum(1 for j in window_idx if mccs[j] == mccs[i])
        time_since_prev_txn_sec = (
            (timestamps[i] - timestamps[i - 1]).total_seconds() if i > 0 else 999999
        )

        features.append({
            "account_id": account_id,
            "timestamp": timestamps[i],
            "amount": amounts[i],
            "amount_to_threshold_ratio": amounts[i] / THRESHOLD,
            "account_age_days": grp.loc[i, "account_age_days"],
            "txn_count_last_1hr": txn_count_last_1hr,
            "sum_amount_last_1hr": sum_amount_last_1hr,
            "distinct_merchants_last_1hr": distinct_merchants_last_1hr,
            "same_mcc_count_last_1hr": same_mcc_count_last_1hr,
            "time_since_prev_txn_sec": min(time_since_prev_txn_sec, 999999),
            "label": grp.loc[i, "label"],
        })

out = pd.DataFrame(features)
out.to_csv("transactions_features.csv", index=False)
print(f"Wrote {len(out)} feature rows to transactions_features.csv")
print(out.groupby("label")[["txn_count_last_1hr", "sum_amount_last_1hr", "amount_to_threshold_ratio"]].mean())
