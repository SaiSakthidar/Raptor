"""V031 — Agentic Token Replay"""
import numpy as np
import pandas as pd
from red.base_generator import BaseGenerator
from red.envelope import Envelope

class V031Generator(BaseGenerator):
    VECTOR_ID = "V031"
    CHANNEL = "agent-payment"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        token_age_lo, token_age_hi = p.get("token_age_range", [3600, 86400])
        amt_lo, amt_hi = p.get("amount_range", [1000, 50000])
        base_ts = pd.Timestamp("2026-07-01")
        envelopes = []

        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V031", i)
            age = int(self.rng.integers(60, 2000))
            n_txns = int(self.rng.integers(5, 20))
            events = []
            for _ in range(n_txns):
                day = int(self.rng.integers(0, 25))
                ts = base_ts + pd.Timedelta(days=day, hours=int(self.rng.integers(8, 22)),
                                            minutes=int(self.rng.integers(0, 60)))
                checkout = float(np.round(self.rng.lognormal(8.5, 1.0), 2))
                events.append({
                    "timestamp": str(ts),
                    "checkout_amount": checkout,
                    "cart_amount": checkout * float(self.rng.uniform(0.98, 1.02)),
                    "token_age_seconds": float(self.rng.uniform(5, 300)),
                    "token_reuse_count": 0,
                    "session_ip_changed": 0,
                    "agent_verified": 1,
                    "merchant_age_days": int(self.rng.integers(30, 1800)),
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id, channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={},
            ))

        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V031", i)
            age = int(self.rng.integers(60, 1800))
            events = []
            # Legit cover sessions
            for _ in range(int(self.rng.integers(3, 8))):
                day = int(self.rng.integers(0, 22))
                ts = base_ts + pd.Timedelta(days=day, hours=int(self.rng.integers(8, 21)),
                                            minutes=int(self.rng.integers(0, 60)))
                checkout = float(np.round(self.rng.lognormal(8.3, 0.9), 2))
                events.append({
                    "timestamp": str(ts),
                    "checkout_amount": checkout,
                    "cart_amount": checkout * float(self.rng.uniform(0.98, 1.02)),
                    "token_age_seconds": float(self.rng.uniform(5, 280)),
                    "token_reuse_count": 0,
                    "session_ip_changed": 0,
                    "agent_verified": 1,
                    "merchant_age_days": int(self.rng.integers(30, 1500)),
                    "label": 0,
                })
            # Token replay attack
            attack_day = int(self.rng.integers(26, 30))
            ts_attack = base_ts + pd.Timedelta(days=attack_day,
                                               hours=int(self.rng.integers(2, 6)))  # off-hours
            stale_token_age = float(self.rng.uniform(token_age_lo, token_age_hi))
            checkout_amt = float(np.round(self.rng.uniform(amt_lo, amt_hi), 2))
            events.append({
                "timestamp": str(ts_attack),
                "checkout_amount": checkout_amt,
                "cart_amount": checkout_amt * float(self.rng.uniform(0.0, 0.3)),  # cart mismatch
                "token_age_seconds": stale_token_age,
                "token_reuse_count": int(self.rng.integers(1, 5)),
                "session_ip_changed": 1,
                "agent_verified": 0,
                "merchant_age_days": int(self.rng.integers(0, 10)),
                "label": 1,
            })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id, channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={"token_age_seconds": stale_token_age},
            ))

        return envelopes
