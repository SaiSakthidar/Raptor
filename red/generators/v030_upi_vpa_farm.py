"""V030 — UPI VPA Farm Hop Laundering"""
import numpy as np
import pandas as pd
from red.base_generator import BaseGenerator
from red.envelope import Envelope

class V030Generator(BaseGenerator):
    VECTOR_ID = "V030"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        hop_lo, hop_hi = p.get("hop_count_range", [4, 7])
        amt_lo, amt_hi = p.get("amount_range", [2000, 80000])
        base_ts = pd.Timestamp("2026-07-01")
        envelopes = []

        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V030", i)
            age = int(self.rng.integers(30, 2000))
            n_txns = int(self.rng.integers(8, 30))
            events = []
            for _ in range(n_txns):
                day = int(self.rng.integers(0, 25))
                ts = base_ts + pd.Timedelta(days=day, hours=int(self.rng.integers(7, 22)),
                                            minutes=int(self.rng.integers(0, 60)))
                events.append({
                    "timestamp": str(ts),
                    "amount": float(np.round(self.rng.lognormal(8.3, 1.0), 2)),
                    "is_new_vpa": int(self.rng.random() < 0.05),
                    "vpa_age_hours": float(self.rng.uniform(48, 8760)),
                    "upi_transfers_15min": int(self.rng.integers(0, 3)),
                    "vpa_count_per_device": int(self.rng.integers(1, 3)),
                    "is_new_beneficiary": int(self.rng.random() < 0.1),
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id, channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={},
            ))

        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V030", i)
            age = int(self.rng.integers(30, 1200))
            events = []
            # Cover history
            for _ in range(int(self.rng.integers(3, 10))):
                day = int(self.rng.integers(0, 22))
                ts = base_ts + pd.Timedelta(days=day, hours=int(self.rng.integers(8, 21)),
                                            minutes=int(self.rng.integers(0, 60)))
                events.append({
                    "timestamp": str(ts),
                    "amount": float(np.round(self.rng.lognormal(8.0, 0.8), 2)),
                    "is_new_vpa": 0,
                    "vpa_age_hours": float(self.rng.uniform(200, 5000)),
                    "upi_transfers_15min": int(self.rng.integers(0, 2)),
                    "vpa_count_per_device": int(self.rng.integers(1, 2)),
                    "is_new_beneficiary": 0,
                    "label": 0,
                })
            # Hop chain: rapid transfers through fresh VPAs
            hops = int(self.rng.integers(hop_lo, hop_hi + 1))
            attack_day = int(self.rng.integers(26, 30))
            base_amount = float(self.rng.uniform(amt_lo, amt_hi))
            for h in range(hops):
                ts = base_ts + pd.Timedelta(days=attack_day,
                                            hours=int(self.rng.integers(9, 21)),
                                            minutes=int(h * self.rng.uniform(1.5, 3.5)))
                events.append({
                    "timestamp": str(ts),
                    "amount": float(np.round(base_amount * self.rng.uniform(0.95, 1.0), 2)),
                    "is_new_vpa": 1,
                    "vpa_age_hours": float(self.rng.uniform(0.5, 48)),
                    "upi_transfers_15min": hops,
                    "vpa_count_per_device": int(self.rng.integers(3, 8)),
                    "is_new_beneficiary": 1,
                    "label": 1,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id, channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={"hops": hops},
            ))

        return envelopes
