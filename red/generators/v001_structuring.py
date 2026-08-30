"""
V001 — Structuring / Threshold Evasion
Signal: burst of 4-6 txns in <90 min, each 80-99% of threshold, same MCC.
Fix vs. example_structuring: amount and age distributions OVERLAP between
legit and fraud — the burst *structure* is the only real signal.
"""

import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope

MCCS = ["5411", "5812", "5732", "5942", "4111", "5999", "5814", "5691"]
MERCHANTS = [f"M{i:03d}" for i in range(60)]


class V001Generator(BaseGenerator):
    VECTOR_ID = "V001"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        threshold = p.get("threshold", 50_000)
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        n_burst_lo, n_burst_hi = p.get("n_burst_range", [4, 7])
        amt_lo, amt_hi = p.get("amount_pct_range", [0.80, 0.99])
        burst_window = p.get("burst_window_minutes", 90)

        envelopes = []

        # --- Legit accounts ---
        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V001", i)
            # Overlapping age: 30-2000 days (fraud range will also be 30-1000)
            age = int(self.rng.integers(30, 2001))
            n_txns = int(self.rng.integers(10, 40))
            base_ts = pd.Timestamp("2026-07-01")
            events = []
            for _ in range(n_txns):
                day = int(self.rng.integers(0, 30))
                hour = int(self.rng.integers(6, 23))
                ts = base_ts + pd.Timedelta(days=day, hours=hour,
                                            minutes=int(self.rng.integers(0, 60)))
                # Legit: broad lognormal — occasionally near threshold
                amount = float(np.round(self.rng.lognormal(mean=8.5, sigma=1.2), 2))
                amount = min(amount, threshold * 1.5)
                events.append({
                    "timestamp": str(ts),
                    "amount": amount,
                    "merchant_id": str(self.rng.choice(MERCHANTS)),
                    "mcc": str(self.rng.choice(MCCS)),
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={"threshold": threshold},
            ))

        # --- Fraud accounts ---
        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V001", i)
            age = int(self.rng.integers(30, 1001))  # overlapping with legit
            base_ts = pd.Timestamp("2026-07-01")
            events = []

            # Cover transactions (normal looking)
            n_cover = int(self.rng.integers(4, 12))
            for _ in range(n_cover):
                day = int(self.rng.integers(0, 25))
                hour = int(self.rng.integers(6, 23))
                ts = base_ts + pd.Timedelta(days=day, hours=hour,
                                            minutes=int(self.rng.integers(0, 60)))
                amount = float(np.round(self.rng.lognormal(mean=8.0, sigma=1.0), 2))
                amount = min(amount, threshold * 1.2)
                events.append({
                    "timestamp": str(ts),
                    "amount": amount,
                    "merchant_id": str(self.rng.choice(MERCHANTS)),
                    "mcc": str(self.rng.choice(MCCS)),
                    "label": 0,
                })

            # The structuring burst
            burst_day = int(self.rng.integers(5, 22)) if self.rng.random() < 0.75 else int(self.rng.integers(26, 29))
            burst_hour = int(self.rng.integers(8, 20))
            burst_ts = base_ts + pd.Timedelta(days=burst_day, hours=burst_hour)
            burst_mcc = str(self.rng.choice(MCCS))
            n_burst = int(self.rng.integers(n_burst_lo, n_burst_hi + 1))
            t = burst_ts
            for _ in range(n_burst):
                amount = float(np.round(
                    self.rng.uniform(threshold * amt_lo, threshold * amt_hi), 2))
                t = t + pd.Timedelta(minutes=int(self.rng.integers(8, 25)))
                events.append({
                    "timestamp": str(t),
                    "amount": amount,
                    "merchant_id": str(self.rng.choice(MERCHANTS)),
                    "mcc": burst_mcc,
                    "label": 1,
                })

            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={"threshold": threshold, "burst_mcc": burst_mcc},
            ))

        return envelopes
