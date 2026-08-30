"""
V017 — Grandparent / Family Emergency Voice Clone
Signal: call from unknown number, immediate modest transfer to new beneficiary,
victim is an older established account, transfer follows call within 30 min.
"""
import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V017Generator(BaseGenerator):
    VECTOR_ID = "V017"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        amt_lo, amt_hi = p.get("emergency_amount_range", [5000, 100000])
        call_lo, call_hi = p.get("call_to_transfer_minutes_range", [5, 25])
        age_lo, age_hi = p.get("victim_account_age_range", [1000, 4000])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V017", i)
            age = int(self.rng.integers(age_lo, age_hi))
            events = []
            for _ in range(int(self.rng.integers(8, 30))):
                ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 30)),
                                             hours=int(self.rng.integers(9, 20)))
                amount = float(np.round(self.rng.lognormal(mean=8.5, sigma=0.8), 2))
                events.append({
                    "timestamp": str(ts), "amount": amount,
                    "call_from_unknown_number": 0,
                    "beneficiary_is_new": 0,
                    "transfer_follows_call_minutes": -1,
                    "account_age_days": age,
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={},
            ))

        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V017", i)
            age = int(self.rng.integers(age_lo, age_hi))
            events = []

            for _ in range(int(self.rng.integers(6, 20))):
                ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 27)),
                                             hours=int(self.rng.integers(9, 18)))
                amount = float(np.round(self.rng.lognormal(mean=8.0, sigma=0.7), 2))
                events.append({
                    "timestamp": str(ts), "amount": amount,
                    "call_from_unknown_number": 0,
                    "beneficiary_is_new": 0,
                    "transfer_follows_call_minutes": -1,
                    "account_age_days": age,
                    "label": 0,
                })

            fraud_ts = base_ts + pd.Timedelta(days=int(self.rng.integers(5, 22)) if self.rng.random() < 0.75 else int(self.rng.integers(26, 29)),
                                               hours=int(self.rng.integers(9, 21)))
            fraud_amount = float(np.round(self.rng.uniform(amt_lo, amt_hi), 2))
            call_gap = int(self.rng.integers(call_lo, call_hi + 1))
            events.append({
                "timestamp": str(fraud_ts),
                "amount": fraud_amount,
                "call_from_unknown_number": 1,
                "beneficiary_is_new": 1,
                "transfer_follows_call_minutes": call_gap,
                "account_age_days": age,
                "label": 1,
            })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={"fraud_amount": fraud_amount},
            ))
        return envelopes
