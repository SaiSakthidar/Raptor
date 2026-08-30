"""
V004 — Agentic Prompt Injection / Payload Poisoning
Signal: checkout amount != user-approved cart amount (delta > 0),
hidden line items, unverified agent, newly registered merchant.
Channel: agent-payment (one session event per transaction).
"""

import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V004Generator(BaseGenerator):
    VECTOR_ID = "V004"
    CHANNEL = "agent-payment"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        delta_lo, delta_hi = p.get("delta_pct_range", [0.05, 0.30])
        merch_age_lo, merch_age_hi = p.get("merchant_age_fraud_range", [1, 30])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        # --- Legit agent purchases ---
        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V004", i)
            n_sessions = int(self.rng.integers(3, 15))
            events = []
            for _ in range(n_sessions):
                ts = base_ts + pd.Timedelta(
                    days=int(self.rng.integers(0, 30)),
                    hours=int(self.rng.integers(8, 23)),
                    minutes=int(self.rng.integers(0, 60)))
                cart_amount = float(np.round(self.rng.uniform(200, 15000), 2))
                # Real checkouts aren't always penny-exact — rounding, tax,
                # or a legitimate small shipping fee shows up sometimes.
                legit_delta_pct = float(np.round(self.rng.normal(0, 0.02), 4))
                if self.rng.random() < 0.08:
                    legit_delta_pct = float(np.round(self.rng.uniform(0.03, 0.07), 4))
                checkout_amount = float(np.round(cart_amount * (1 + legit_delta_pct), 2))
                events.append({
                    "timestamp": str(ts),
                    "cart_amount": cart_amount,
                    "checkout_amount": checkout_amount,
                    "cart_checkout_delta": round(checkout_amount - cart_amount, 2),
                    "cart_checkout_delta_pct": legit_delta_pct,
                    "n_line_items": int(self.rng.integers(1, 6)),
                    "hidden_line_items": 0,
                    "agent_verified": 1,
                    "merchant_age_days": int(self.rng.integers(90, 3000)),
                    "merchant_dispute_rate": float(np.round(
                        self.rng.uniform(0, 0.02), 4)),
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={},
                generation_params={},
            ))

        # --- Fraud: injected agent sessions ---
        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V004", i)
            n_sessions = int(self.rng.integers(1, 5))
            events = []
            for _ in range(n_sessions):
                ts = base_ts + pd.Timedelta(
                    days=int(self.rng.integers(0, 30)),
                    hours=int(self.rng.integers(8, 23)),
                    minutes=int(self.rng.integers(0, 60)))
                cart_amount = float(np.round(self.rng.uniform(200, 10000), 2))
                delta_pct = float(np.round(self.rng.uniform(delta_lo, delta_hi), 4))
                checkout_amount = float(np.round(cart_amount * (1 + delta_pct), 2))
                hidden_items = int(self.rng.integers(1, 4))
                events.append({
                    "timestamp": str(ts),
                    "cart_amount": cart_amount,
                    "checkout_amount": checkout_amount,
                    "cart_checkout_delta": round(checkout_amount - cart_amount, 2),
                    "cart_checkout_delta_pct": delta_pct,
                    "n_line_items": int(self.rng.integers(1, 4)) + hidden_items,
                    "hidden_line_items": hidden_items,
                    "agent_verified": int(self.rng.random() < 0.15),
                    "merchant_age_days": int(self.rng.integers(merch_age_lo, merch_age_hi)),
                    "merchant_dispute_rate": float(np.round(
                        self.rng.uniform(0.05, 0.40), 4)),
                    "label": 1,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={},
                generation_params={"delta_pct": delta_pct},
            ))

        return envelopes
