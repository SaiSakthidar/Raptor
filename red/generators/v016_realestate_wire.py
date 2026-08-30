"""
V016 — Real-Estate Closing Wire Fraud
Signal: first-ever large outbound wire, brand-new beneficiary, amount in
real-estate range, within 60 min of a "closing" call event.
"""
import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V016Generator(BaseGenerator):
    VECTOR_ID = "V016"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        close_lo, close_hi = p.get("closing_amount_range", [1_000_000, 50_000_000])
        call_lo, call_hi = p.get("call_to_wire_minutes_range", [10, 55])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V016", i)
            age = int(self.rng.integers(365, 3000))
            events = []
            prior_large = int(self.rng.integers(1, 5))
            for _ in range(int(self.rng.integers(8, 25))):
                ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 30)),
                                             hours=int(self.rng.integers(9, 17)))
                amount = float(np.round(self.rng.lognormal(mean=11, sigma=1.0), 2))
                events.append({
                    "timestamp": str(ts), "amount": amount,
                    "beneficiary_tenure_days": int(self.rng.integers(30, 800)),
                    "first_large_outbound_ever": 0,
                    "call_precedes_wire_minutes": -1,
                    "amount_in_realestate_range": int(amount > 500_000),
                    "days_since_prior_large_transfer": int(self.rng.integers(1, 150)),
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age, "prior_large_transfers": prior_large},
                generation_params={},
            ))

        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V016", i)
            age = int(self.rng.integers(180, 2000))
            events = []

            # Normal spend history — no large transfers
            for _ in range(int(self.rng.integers(6, 18))):
                ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 25)),
                                             hours=int(self.rng.integers(9, 17)))
                amount = float(np.round(self.rng.lognormal(mean=9, sigma=0.8), 2))
                events.append({
                    "timestamp": str(ts), "amount": amount,
                    "beneficiary_tenure_days": int(self.rng.integers(30, 500)),
                    "first_large_outbound_ever": 0,
                    "call_precedes_wire_minutes": -1,
                    "amount_in_realestate_range": 0,
                    "days_since_prior_large_transfer": int(self.rng.integers(80, 400)),  # long ago or never, but not a sentinel
                    "label": 0,
                })

            # The fraudulent closing wire
            fraud_ts = base_ts + pd.Timedelta(days=int(self.rng.integers(5, 22)) if self.rng.random() < 0.75 else int(self.rng.integers(26, 29)),
                                               hours=int(self.rng.integers(10, 15)))
            fraud_amount = float(np.round(self.rng.uniform(close_lo, close_hi), 2))
            call_gap = int(self.rng.integers(call_lo, call_hi + 1))
            events.append({
                "timestamp": str(fraud_ts),
                "amount": fraud_amount,
                "beneficiary_tenure_days": 0,
                "first_large_outbound_ever": 1,
                "call_precedes_wire_minutes": call_gap,
                "amount_in_realestate_range": 1,
                "days_since_prior_large_transfer": int(self.rng.integers(80, 400)),  # long ago or never, but not a sentinel
                "label": 1,
            })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age, "prior_large_transfers": 0},
                generation_params={"fraud_amount": fraud_amount, "call_gap": call_gap},
            ))
        return envelopes
