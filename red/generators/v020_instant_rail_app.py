"""
V020 — Pix / FedNow Instant Rail APP Fraud
Signal: instant rail transfer, new beneficiary, amount >> 5× 30d avg,
within minutes of a social-engineering contact event, no prior instant transfers.
"""
import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V020Generator(BaseGenerator):
    VECTOR_ID = "V020"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        amt_lo, amt_hi = p.get("instant_amount_range", [10000, 500000])
        contact_lo, contact_hi = p.get("contact_to_transfer_minutes_range", [1, 8])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V020", i)
            age = int(self.rng.integers(90, 2000))
            avg_txn = float(np.round(self.rng.uniform(500, 20000), 2))
            # Legit users occasionally use instant rails for known payees —
            # but some are also genuinely first-time instant-rail users
            prior_instant = int(self.rng.integers(2, 10)) if self.rng.random() < 0.85 else 0
            events = []
            for _ in range(int(self.rng.integers(10, 35))):
                ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 30)),
                                             hours=int(self.rng.integers(8, 22)))
                amount = float(np.round(self.rng.lognormal(np.log(avg_txn), 0.6), 2))
                rail = "instant" if self.rng.random() < 0.2 else "standard"
                events.append({
                    "timestamp": str(ts), "amount": amount,
                    "rail_type": rail,
                    "beneficiary_is_new": 0,
                    "amount_vs_30d_avg_ratio": round(amount / avg_txn, 3),
                    "contact_to_transfer_minutes": -1,
                    "prior_instant_rail_transfers": prior_instant,
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age, "avg_txn_30d": avg_txn},
                generation_params={},
            ))

        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V020", i)
            age = int(self.rng.integers(90, 1500))
            avg_txn = float(np.round(self.rng.uniform(500, 15000), 2))
            events = []

            # Normal history — standard rail only
            for _ in range(int(self.rng.integers(8, 20))):
                ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 25)),
                                             hours=int(self.rng.integers(9, 20)))
                amount = float(np.round(self.rng.lognormal(np.log(avg_txn), 0.5), 2))
                events.append({
                    "timestamp": str(ts), "amount": amount,
                    "rail_type": "standard",
                    "beneficiary_is_new": 0,
                    "amount_vs_30d_avg_ratio": round(amount / avg_txn, 3),
                    "contact_to_transfer_minutes": -1,
                    "prior_instant_rail_transfers": 0,
                    "label": 0,
                })

            # Instant rail fraud transfer
            fraud_ts = base_ts + pd.Timedelta(days=int(self.rng.integers(5, 22)) if self.rng.random() < 0.75 else int(self.rng.integers(26, 29)),
                                               hours=int(self.rng.integers(9, 21)))
            fraud_amount = float(np.round(self.rng.uniform(amt_lo, amt_hi), 2))
            contact_gap = int(self.rng.integers(contact_lo, contact_hi + 1))
            events.append({
                "timestamp": str(fraud_ts),
                "amount": fraud_amount,
                "rail_type": "instant",
                "beneficiary_is_new": 1,
                "amount_vs_30d_avg_ratio": round(fraud_amount / avg_txn, 3),
                "contact_to_transfer_minutes": contact_gap,
                # occasionally the victim has used instant rail once or twice before
                "prior_instant_rail_transfers": int(self.rng.integers(0, 3)) if self.rng.random() < 0.15 else 0,
                "label": 1,
            })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age, "avg_txn_30d": avg_txn},
                generation_params={"fraud_amount": fraud_amount, "contact_gap": contact_gap},
            ))
        return envelopes
