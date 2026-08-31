"""V028 — MFA Push-Fatigue Attack"""
import numpy as np
import pandas as pd
from red.base_generator import BaseGenerator
from red.envelope import Envelope

class V028Generator(BaseGenerator):
    VECTOR_ID = "V028"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        push_lo, push_hi = p.get("push_count_range", [6, 25])
        amt_lo, amt_hi = p.get("amount_range", [10000, 200000])
        base_ts = pd.Timestamp("2026-07-01")
        envelopes = []

        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V028", i)
            age = int(self.rng.integers(60, 2000))
            n_txns = int(self.rng.integers(8, 30))
            events = []
            for _ in range(n_txns):
                day = int(self.rng.integers(0, 25))
                ts = base_ts + pd.Timedelta(days=day, hours=int(self.rng.integers(7, 22)),
                                            minutes=int(self.rng.integers(0, 60)))
                # Legit: 1-2 push attempts, fast approval
                events.append({
                    "timestamp": str(ts),
                    "amount": float(np.round(self.rng.lognormal(8.5, 1.1), 2)),
                    "push_count_before_approval": int(self.rng.integers(1, 3)),
                    "response_time_seconds": float(self.rng.uniform(5, 45)),
                    "approval_attempt_number": int(self.rng.integers(1, 3)),
                    "call_precedes_transfer_minutes": float(self.rng.uniform(0, 5)) if self.rng.random() < 0.1 else 0.0,
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id, channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={},
            ))

        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V028", i)
            age = int(self.rng.integers(90, 1800))
            events = []
            # Cover txns
            for _ in range(int(self.rng.integers(3, 8))):
                day = int(self.rng.integers(0, 22))
                ts = base_ts + pd.Timedelta(days=day, hours=int(self.rng.integers(7, 22)),
                                            minutes=int(self.rng.integers(0, 60)))
                events.append({
                    "timestamp": str(ts),
                    "amount": float(np.round(self.rng.lognormal(8.0, 0.9), 2)),
                    "push_count_before_approval": int(self.rng.integers(1, 3)),
                    "response_time_seconds": float(self.rng.uniform(5, 40)),
                    "approval_attempt_number": int(self.rng.integers(1, 2)),
                    "call_precedes_transfer_minutes": 0.0,
                    "label": 0,
                })
            # Attack event: many pushes, eventual tired approval, large immediate transfer
            push_count = int(self.rng.integers(push_lo, push_hi + 1))
            attack_day = int(self.rng.integers(26, 30))
            ts_attack = base_ts + pd.Timedelta(days=attack_day, hours=int(self.rng.integers(10, 22)))
            events.append({
                "timestamp": str(ts_attack),
                "amount": float(np.round(self.rng.uniform(amt_lo, amt_hi), 2)),
                "push_count_before_approval": push_count,
                "response_time_seconds": float(self.rng.uniform(0.5, 4.0)),  # tired, fast tap
                "approval_attempt_number": push_count,
                "call_precedes_transfer_minutes": float(self.rng.uniform(2, 15)),
                "label": 1,
            })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id, channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={"push_count": push_count},
            ))

        return envelopes
