"""
V012 — Digital Arrest / Authority-Impersonation Coercion
Signal: call immediately precedes large transfer (<30 min gap),
new beneficiary, amount >> 5x 30d max, unusual hour, first-ever large wire.
"""

import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V012Generator(BaseGenerator):
    VECTOR_ID = "V012"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        coerce_lo, coerce_hi = p.get("coercion_amount_range", [500_000, 10_000_000])
        call_lo, call_hi = p.get("call_to_transfer_minutes_range", [5, 25])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        # --- Legit ---
        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V012", i)
            age = int(self.rng.integers(180, 3000))
            max_hist = float(np.round(self.rng.uniform(10_000, 200_000), 2))
            events = []
            for _ in range(int(self.rng.integers(10, 40))):
                ts = base_ts + pd.Timedelta(
                    days=int(self.rng.integers(0, 30)),
                    hours=int(self.rng.integers(9, 20)),
                    minutes=int(self.rng.integers(0, 60)))
                amount = float(np.round(self.rng.uniform(1000, max_hist * 0.4), 2))
                events.append({
                    "timestamp": str(ts),
                    "amount": amount,
                    "beneficiary_is_new": 0,
                    "call_precedes_transfer_minutes": -1,  # no call
                    "amount_vs_30d_max_ratio": float(np.round(amount / max_hist, 3)),
                    "hour_of_day": ts.hour,
                    "prior_large_transfer_count": int(self.rng.integers(0, 10)),
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age,
                                  "max_30d_transfer": max_hist},
                generation_params={},
            ))

        # --- Fraud ---
        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V012", i)
            age = int(self.rng.integers(90, 2000))
            max_hist = float(np.round(self.rng.uniform(5_000, 100_000), 2))
            events = []

            # Normal history
            for _ in range(int(self.rng.integers(5, 20))):
                ts = base_ts + pd.Timedelta(
                    days=int(self.rng.integers(0, 25)),
                    hours=int(self.rng.integers(9, 20)))
                amount = float(np.round(self.rng.uniform(1000, max_hist * 0.3), 2))
                events.append({
                    "timestamp": str(ts),
                    "amount": amount,
                    "beneficiary_is_new": 0,
                    "call_precedes_transfer_minutes": -1,
                    "amount_vs_30d_max_ratio": float(np.round(amount / max_hist, 3)),
                    "hour_of_day": ts.hour,
                    "prior_large_transfer_count": 0,
                    "label": 0,
                })

            # The coerced transfer — late night / early morning
            coerce_hour = int(self.rng.choice([1, 2, 3, 22, 23]))
            ts = base_ts + pd.Timedelta(
                days=int(self.rng.integers(26, 30)),
                hours=coerce_hour,
                minutes=int(self.rng.integers(0, 30)))
            coerce_amount = float(np.round(self.rng.uniform(coerce_lo, coerce_hi), 2))
            call_gap = int(self.rng.integers(call_lo, call_hi + 1))
            events.append({
                "timestamp": str(ts),
                "amount": coerce_amount,
                "beneficiary_is_new": 1,
                "call_precedes_transfer_minutes": call_gap,
                "amount_vs_30d_max_ratio": float(np.round(coerce_amount / max_hist, 3)),
                "hour_of_day": coerce_hour,
                "prior_large_transfer_count": 0,
                "label": 1,
            })

            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age,
                                  "max_30d_transfer": max_hist},
                generation_params={"coerce_amount": coerce_amount,
                                    "call_gap_minutes": call_gap},
            ))

        return envelopes
