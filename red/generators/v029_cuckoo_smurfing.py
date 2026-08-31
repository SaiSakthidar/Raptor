"""V029 — Cuckoo Smurfing"""
import numpy as np
import pandas as pd
from red.base_generator import BaseGenerator
from red.envelope import Envelope

class V029Generator(BaseGenerator):
    VECTOR_ID = "V029"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        amt_lo, amt_hi = p.get("amount_range", [5000, 100000])
        pt_lo, pt_hi = p.get("pass_through_pct_range", [0.92, 1.0])
        base_ts = pd.Timestamp("2026-07-01")
        envelopes = []

        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V029", i)
            age = int(self.rng.integers(30, 2000))
            n_txns = int(self.rng.integers(10, 35))
            events = []
            for _ in range(n_txns):
                day = int(self.rng.integers(0, 25))
                ts = base_ts + pd.Timedelta(days=day, hours=int(self.rng.integers(7, 22)),
                                            minutes=int(self.rng.integers(0, 60)))
                # Legit: known senders, low pass-through
                events.append({
                    "timestamp": str(ts),
                    "amount": float(np.round(self.rng.lognormal(8.5, 1.0), 2)),
                    "inbound_sender_count_7d": int(self.rng.integers(1, 5)),
                    "inbound_unknown_sender_pct": float(self.rng.uniform(0, 0.2)),
                    "pass_through_ratio": float(self.rng.uniform(0.0, 0.4)),
                    "time_inbound_to_outbound_hours": float(self.rng.uniform(12, 168)),
                    "outbound_recipient_count_7d": int(self.rng.integers(1, 4)),
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id, channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={},
            ))

        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V029", i)
            age = int(self.rng.integers(60, 1500))
            events = []
            # Normal history
            for _ in range(int(self.rng.integers(5, 15))):
                day = int(self.rng.integers(0, 22))
                ts = base_ts + pd.Timedelta(days=day, hours=int(self.rng.integers(7, 22)),
                                            minutes=int(self.rng.integers(0, 60)))
                events.append({
                    "timestamp": str(ts),
                    "amount": float(np.round(self.rng.lognormal(8.0, 0.9), 2)),
                    "inbound_sender_count_7d": int(self.rng.integers(1, 4)),
                    "inbound_unknown_sender_pct": float(self.rng.uniform(0, 0.15)),
                    "pass_through_ratio": float(self.rng.uniform(0.0, 0.3)),
                    "time_inbound_to_outbound_hours": float(self.rng.uniform(24, 120)),
                    "outbound_recipient_count_7d": int(self.rng.integers(1, 3)),
                    "label": 0,
                })
            # Attack: unknown senders, instant pass-through
            n_smurfs = int(self.rng.integers(3, 9))
            attack_day = int(self.rng.integers(26, 30))
            for j in range(n_smurfs):
                ts = base_ts + pd.Timedelta(days=attack_day,
                                            hours=int(self.rng.integers(8, 20)),
                                            minutes=int(self.rng.integers(0, 60)))
                events.append({
                    "timestamp": str(ts),
                    "amount": float(np.round(self.rng.uniform(amt_lo, amt_hi), 2)),
                    "inbound_sender_count_7d": int(self.rng.integers(5, 15)),
                    "inbound_unknown_sender_pct": float(self.rng.uniform(0.8, 1.0)),
                    "pass_through_ratio": float(self.rng.uniform(pt_lo, pt_hi)),
                    "time_inbound_to_outbound_hours": float(self.rng.uniform(0.5, 6.0)),
                    "outbound_recipient_count_7d": int(self.rng.integers(3, 8)),
                    "label": 1,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id, channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={"n_smurfs": n_smurfs},
            ))

        return envelopes
