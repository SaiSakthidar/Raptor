"""
V002 — Deepfake BEC / APP Fraud
Signal: large single authorized wire to a brand-new beneficiary,
preceded by an out-of-band communication event, urgency markers.
No transaction-level anomaly — detection lives in beneficiary novelty +
request-context features.
"""

import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope

CHANNELS_CONTACT = ["whatsapp", "teams", "email", "phone"]


class V002Generator(BaseGenerator):
    VECTOR_ID = "V002"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        fraud_lo, fraud_hi = p.get("fraud_amount_range", [500_000, 5_000_000])
        legit_max = p.get("legit_max_amount", 300_000)

        envelopes = []

        # --- Legit ---
        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V002", i)
            age = int(self.rng.integers(180, 3000))
            n_txns = int(self.rng.integers(15, 50))
            base_ts = pd.Timestamp("2026-07-01")
            events = []
            known_benes = [f"BENE_{self.rng.integers(0, 20):03d}" for _ in range(5)]
            for _ in range(n_txns):
                day = int(self.rng.integers(0, 30))
                ts = base_ts + pd.Timedelta(days=day,
                                             hours=int(self.rng.integers(9, 18)),
                                             minutes=int(self.rng.integers(0, 60)))
                amount = float(np.round(self.rng.lognormal(mean=9.5, sigma=1.0), 2))
                amount = min(amount, legit_max)
                bene = str(self.rng.choice(known_benes))
                events.append({
                    "timestamp": str(ts),
                    "amount": amount,
                    "beneficiary_id": bene,
                    "beneficiary_tenure_days": int(self.rng.integers(60, 1000)),
                    "urgency_score": float(np.round(self.rng.uniform(0, 0.2), 3)),
                    "contact_channel": "email",
                    "is_new_beneficiary": 0,
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age,
                                  "max_30d_transfer": legit_max * 0.3},
                generation_params={},
            ))

        # --- Fraud ---
        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V002", i)
            age = int(self.rng.integers(90, 2000))
            base_ts = pd.Timestamp("2026-07-01")
            events = []

            # Cover: normal transactions before the attack
            known_benes = [f"BENE_{self.rng.integers(0, 20):03d}" for _ in range(5)]
            max_historical = float(np.round(self.rng.uniform(20_000, 150_000), 2))
            for _ in range(int(self.rng.integers(8, 20))):
                day = int(self.rng.integers(0, 27))
                ts = base_ts + pd.Timedelta(days=day,
                                             hours=int(self.rng.integers(9, 18)),
                                             minutes=int(self.rng.integers(0, 60)))
                amount = float(np.round(self.rng.uniform(5000, max_historical * 0.5), 2))
                events.append({
                    "timestamp": str(ts),
                    "amount": amount,
                    "beneficiary_id": str(self.rng.choice(known_benes)),
                    "beneficiary_tenure_days": int(self.rng.integers(60, 800)),
                    "urgency_score": float(np.round(self.rng.uniform(0, 0.15), 3)),
                    "contact_channel": "email",
                    "is_new_beneficiary": 0,
                    "label": 0,
                })

            # The fraudulent wire
            fraud_ts = base_ts + pd.Timedelta(days=int(self.rng.integers(5, 22)) if self.rng.random() < 0.75 else int(self.rng.integers(26, 29)),
                                               hours=int(self.rng.integers(10, 17)),
                                               minutes=int(self.rng.integers(0, 30)))
            fraud_amount = float(np.round(self.rng.uniform(fraud_lo, fraud_hi), 2))
            events.append({
                "timestamp": str(fraud_ts),
                "amount": fraud_amount,
                "beneficiary_id": f"MULE_{i:05d}",
                "beneficiary_tenure_days": 0,
                "urgency_score": float(np.round(self.rng.uniform(0.75, 1.0), 3)),
                "contact_channel": str(self.rng.choice(CHANNELS_CONTACT)),
                "is_new_beneficiary": 1,
                "label": 1,
            })

            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age,
                                  "max_30d_transfer": max_historical},
                generation_params={"fraud_amount": fraud_amount},
            ))

        return envelopes
