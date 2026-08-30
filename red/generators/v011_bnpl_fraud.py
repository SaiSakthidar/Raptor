"""
V011 — BNPL Synthetic Identity Bust-Out
Signal: brand-new account, credit inquiry burst, multiple BNPL merchants
within 7 days, zero prior transaction history, total BNPL near credit limit.
Channel: kyc-session (account-open event + subsequent BNPL usage events).
"""

import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope

BNPL_MERCHANTS = [f"BNPL_MERCH_{i:03d}" for i in range(30)]
CATEGORIES = ["electronics", "fashion", "furniture", "travel", "jewelry"]


class V011Generator(BaseGenerator):
    VECTOR_ID = "V011"
    CHANNEL = "kyc-session"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        merch_lo, merch_hi = p.get("bnpl_merchants_range", [3, 8])
        bust_window = p.get("bust_window_days", 7)
        acct_age_lo, acct_age_hi = p.get("account_age_fraud_range", [1, 14])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        # --- Legit ---
        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V011", i)
            acct_age = int(self.rng.integers(3, 1500))  # some legit customers are also new
            credit_limit = float(np.round(self.rng.uniform(20_000, 200_000), 2))
            events = []
            # Account-open event (long ago)
            doc_age_days = int(self.rng.integers(180, 3650))
            credit_inquiry_count_7d = int(self.rng.integers(0, 2))
            prior_txn_count = int(self.rng.integers(20, 200))
            events.append({
                "timestamp": str(base_ts - pd.Timedelta(days=acct_age)),
                "event_type": "account_open",
                "doc_age_days": doc_age_days,
                "credit_inquiry_count_7d": credit_inquiry_count_7d,
                "prior_txn_count": prior_txn_count,
                "label": 0,
            })
            # BNPL purchases — occasional, spread out
            for _ in range(int(self.rng.integers(1, 5))):
                day = int(self.rng.integers(0, 30))
                ts = base_ts + pd.Timedelta(days=day, hours=int(self.rng.integers(9,21)))
                amount = float(np.round(self.rng.uniform(1000, credit_limit * 0.3), 2))
                events.append({
                    "timestamp": str(ts),
                    "event_type": "bnpl_purchase",
                    "amount": amount,
                    "merchant_id": str(self.rng.choice(BNPL_MERCHANTS)),
                    "category": str(self.rng.choice(CATEGORIES)),
                    "account_age_at_purchase_days": acct_age + day,
                    "bnpl_merchant_count_7d": 1,
                    "total_bnpl_30d": amount,
                    "credit_utilization": float(np.round(amount / credit_limit, 3)),
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": acct_age,
                                  "credit_limit": credit_limit,
                                  "doc_age_days": doc_age_days,
                                  "credit_inquiry_count_7d": credit_inquiry_count_7d,
                                  "prior_txn_count": prior_txn_count},
                generation_params={},
            ))

        # --- Fraud ---
        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V011", i)
            acct_age = int(self.rng.integers(acct_age_lo, acct_age_hi + 1))
            credit_limit = float(np.round(self.rng.uniform(50_000, 300_000), 2))
            n_merchants = int(self.rng.integers(merch_lo, merch_hi + 1))
            events = []

            # Account-open (very recent, AI-generated identity)
            doc_age_days = int(self.rng.integers(1, 14))
            credit_inquiry_count_7d = int(self.rng.integers(3, 8))  # inquiry burst
            prior_txn_count = 0
            events.append({
                "timestamp": str(base_ts - pd.Timedelta(days=acct_age)),
                "event_type": "account_open",
                "doc_age_days": doc_age_days,
                "credit_inquiry_count_7d": credit_inquiry_count_7d,
                "prior_txn_count": prior_txn_count,
                "label": 0,
            })

            # Rapid bust-out within days of account open. 75% of episodes land
            # early (train-friendly); 25% land in the held-out tail so this
            # vector still gets genuine test-set representation. account_age
            # stays small either way — it's the account's age, not calendar day.
            episode_early = self.rng.random() < 0.75
            total_bnpl = 0.0
            chosen_merchants = self.rng.choice(BNPL_MERCHANTS, size=n_merchants, replace=False)
            for j, merchant in enumerate(chosen_merchants):
                day = int(self.rng.integers(0, bust_window)) if episode_early else int(self.rng.integers(26, 29))
                ts = base_ts + pd.Timedelta(days=day, hours=int(self.rng.integers(9,21)))
                amount = float(np.round(
                    self.rng.uniform(credit_limit * 0.10, credit_limit * 0.25), 2))
                total_bnpl += amount
                events.append({
                    "timestamp": str(ts),
                    "event_type": "bnpl_purchase",
                    "amount": amount,
                    "merchant_id": str(merchant),
                    "category": str(self.rng.choice(CATEGORIES)),
                    "account_age_at_purchase_days": acct_age + day,
                    "bnpl_merchant_count_7d": j + 1,
                    "total_bnpl_30d": round(total_bnpl, 2),
                    "credit_utilization": float(np.round(total_bnpl / credit_limit, 3)),
                    "label": 1,
                })

            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": acct_age,
                                  "credit_limit": credit_limit,
                                  "n_bnpl_merchants": n_merchants,
                                  "doc_age_days": doc_age_days,
                                  "credit_inquiry_count_7d": credit_inquiry_count_7d,
                                  "prior_txn_count": prior_txn_count},
                generation_params={"bust_window_days": bust_window},
            ))

        return envelopes
