"""
V026 — Rogue Merchant Agent Impersonation
Channel : agent-payment
Modality: AGENT

A rogue agent impersonates a trusted brand's checkout agent inside an
agent-to-agent negotiation, claiming a merchant identity it holds no valid
certificate for. The shopping agent completes the purchase believing it is
dealing with the real merchant; funds settle to an account that does not
match the claimed merchant's registered settlement account. Distinct from
V023 (a real merchant agent colluding for a kickback) — this is identity
theft of the merchant agent itself, not collusion.
"""
import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V026Generator(BaseGenerator):
    VECTOR_ID = "V026"
    CHANNEL = "agent-payment"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        amt_lo, amt_hi = p.get("checkout_amount_range", [2000, 150000])
        seen_lo, seen_hi = p.get("first_seen_hours_range", [0.5, 20])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V026", i)
            age = int(self.rng.integers(30, 1500))
            cart = float(np.round(self.rng.lognormal(7.8, 0.9), 2))
            checkout = float(np.round(cart * self.rng.uniform(0.97, 1.03), 2))
            ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 30)),
                                         hours=int(self.rng.integers(8, 22)))
            event = {
                "timestamp": str(ts),
                "cart_amount": cart,
                "checkout_amount": checkout,
                "cart_checkout_delta": round(checkout - cart, 2),
                "cart_checkout_delta_pct": round((checkout - cart) / max(cart, 1), 4),
                "merchant_identity_verified": 1,
                "settlement_account_mismatch": 0,
                "merchant_agent_first_seen_hours": float(np.round(self.rng.uniform(500, 20000), 1)),
                "agent_verified": 1,
                "label": 0,
            }
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=[event],
                entity_features={"account_age_days": age},
                generation_params={},
            ))

        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V026", i)
            age = int(self.rng.integers(30, 1200))
            cart = float(np.round(self.rng.lognormal(8.0, 0.8), 2))
            # checkout is mostly close to cart (the identity/settlement mismatch
            # is the real tell here, not the amount) but sometimes independent
            if self.rng.random() < 0.6:
                checkout = float(np.round(cart * self.rng.uniform(0.9, 1.15), 2))
            else:
                checkout = float(np.round(self.rng.uniform(amt_lo, amt_hi), 2))
            first_seen = float(np.round(self.rng.uniform(seen_lo, seen_hi), 2))
            ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 30)),
                                         hours=int(self.rng.integers(8, 22)))
            event = {
                "timestamp": str(ts),
                "cart_amount": cart,
                "checkout_amount": checkout,
                "cart_checkout_delta": round(checkout - cart, 2),
                "cart_checkout_delta_pct": round((checkout - cart) / max(cart, 1), 4),
                "merchant_identity_verified": 0,
                "settlement_account_mismatch": 1,
                "merchant_agent_first_seen_hours": first_seen,
                "agent_verified": 0,
                "label": 1,
            }
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=[event],
                entity_features={"account_age_days": age},
                generation_params={"checkout": checkout, "first_seen_hours": first_seen},
            ))
        return envelopes
