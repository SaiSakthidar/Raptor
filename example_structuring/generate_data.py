"""
Generates a synthetic transaction dataset for ONE attack vector:
"Structuring / threshold evasion" - splitting a large payment into several
transactions that each stay just under a bank's auto-review threshold
(here: Rs 50,000), fired in quick succession, same device, same merchant category.

This is transaction-level tabular data - the kind you'd get from a real
card/payments processor - so it's a realistic stand-in for BAF/IEEE-CIS-style
data until you swap in the real dataset.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

THRESHOLD = 50_000
N_ACCOUNTS_LEGIT = 50
N_ACCOUNTS_FRAUD = 10
SIM_DAYS = 30
MERCHANTS = [f"M{i:03d}" for i in range(40)]
MCCS = ["5411", "5812", "5732", "5942", "4111", "5999", "5814", "5691"]
DEVICES_PER_ACCOUNT = 1  # most customers use ~1 primary device

rows = []

def rand_ts(day):
    hour = rng.integers(6, 23)
    minute = rng.integers(0, 60)
    second = rng.integers(0, 60)
    return pd.Timestamp("2026-07-01") + pd.Timedelta(days=day, hours=hour, minutes=minute, seconds=second)

# ---- Legit accounts: normal, spread-out spending ----
for acc_idx in range(N_ACCOUNTS_LEGIT):
    account_id = f"ACC{acc_idx:04d}"
    device_id = f"DEV{acc_idx:04d}"
    account_age_days = int(rng.integers(60, 2000))
    n_txns = rng.integers(15, 45)  # over 30 days
    days = rng.integers(0, SIM_DAYS, size=n_txns)
    for day in sorted(days):
        amount = float(np.round(rng.lognormal(mean=7.5, sigma=1.0), 2))  # typically hundreds to a few thousand
        amount = min(amount, 45000)  # legit spend rarely near the threshold
        merchant = rng.choice(MERCHANTS)
        mcc = rng.choice(MCCS)
        rows.append({
            "account_id": account_id,
            "device_id": device_id,
            "timestamp": rand_ts(day),
            "amount": amount,
            "merchant_id": merchant,
            "mcc": mcc,
            "account_age_days": account_age_days,
            "label": 0,
        })

# ---- Fraud accounts: mostly normal, but contain one structuring burst ----
for acc_idx in range(N_ACCOUNTS_FRAUD):
    account_id = f"FRAUD{acc_idx:04d}"
    device_id = f"FDEV{acc_idx:04d}"
    account_age_days = int(rng.integers(10, 400))  # fraud accounts tend to be newer
    # some normal-looking cover transactions
    n_normal = rng.integers(5, 15)
    days = rng.integers(0, SIM_DAYS, size=n_normal)
    for day in sorted(days):
        amount = float(np.round(rng.lognormal(mean=7.0, sigma=0.8), 2))
        amount = min(amount, 40000)
        rows.append({
            "account_id": account_id,
            "device_id": device_id,
            "timestamp": rand_ts(day),
            "amount": amount,
            "merchant_id": rng.choice(MERCHANTS),
            "mcc": rng.choice(MCCS),
            "account_age_days": account_age_days,
            "label": 0,
        })

    # the structuring burst: 4-6 transactions, each just under THRESHOLD,
    # same device (already fixed), same MCC, different merchants,
    # all within ~90 minutes
    burst_day = int(rng.integers(0, SIM_DAYS))
    burst_start = rand_ts(burst_day)
    burst_mcc = rng.choice(MCCS)  # attacker reuses one category (e.g. "electronics")
    n_burst = rng.integers(4, 7)
    t = burst_start
    for _ in range(n_burst):
        amount = float(np.round(rng.uniform(THRESHOLD * 0.80, THRESHOLD * 0.99), 2))
        t = t + pd.Timedelta(minutes=int(rng.integers(8, 25)))
        rows.append({
            "account_id": account_id,
            "device_id": device_id,
            "timestamp": t,
            "amount": amount,
            "merchant_id": rng.choice(MERCHANTS),
            "mcc": burst_mcc,
            "account_age_days": account_age_days,
            "label": 1,
        })

df = pd.DataFrame(rows).sort_values(["account_id", "timestamp"]).reset_index(drop=True)
df.to_csv("transactions_raw.csv", index=False)
print(f"Wrote {len(df)} transactions ({df['label'].sum()} labeled fraud) to transactions_raw.csv")
print(df["label"].value_counts())
