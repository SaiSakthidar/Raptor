"""
V023 — Agent-to-Agent Collusion / Kickback Fraud
Channel : agent-payment
Modality: AGENT

A consumer shopping agent and a merchant checkout agent negotiate a private
side-channel: checkout price is inflated above market, then a rebate flows
back to a wallet controlled by the shopping agent's operator — not the
customer. High negotiation-round count + inflated price + a hidden rebate
routed away from the customer is the signature.
"""
import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V023Generator(BaseGenerator):
    VECTOR_ID = "V023"
    CHANNEL = "agent-payment"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        infl_lo, infl_hi = p.get("price_inflation_ratio_range", [1.2, 1.8])
        rounds_lo, rounds_hi = p.get("negotiation_rounds_range", [5, 15])
        rebate_lo, rebate_hi = p.get("rebate_pct_of_checkout_range", [0.05, 0.25])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V023", i)
            age = int(self.rng.integers(30, 1500))
            events = []
            for _ in range(int(self.rng.integers(3, 15))):
                ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 30)),
                                             hours=int(self.rng.integers(8, 22)))
                cart = float(np.round(self.rng.lognormal(7.5, 0.8), 2))
                market_price = cart
                checkout = float(np.round(market_price * self.rng.uniform(0.95, 1.05), 2))
                events.append({
                    "timestamp": str(ts),
                    "cart_amount": cart,
                    "checkout_amount": checkout,
                    "cart_checkout_delta": round(checkout - cart, 2),
                    "cart_checkout_delta_pct": round((checkout - cart) / max(cart, 1), 4),
                    "price_vs_market_ratio": round(checkout / market_price, 3),
                    "agent_negotiation_rounds": int(self.rng.integers(1, 4)),
                    "hidden_rebate_amount": 0.0,
                    "rebate_to_non_customer_wallet": 0,
                    "merchant_agent_reputation_score": float(np.round(self.rng.uniform(0.7, 1.0), 3)),
                    "agent_verified": 1,
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
            actor_id = self._actor_id("FRAUD_V023", i)
            age = int(self.rng.integers(30, 1000))
            events = []

            # Normal purchases before the colluding merchant is used
            for _ in range(int(self.rng.integers(2, 10))):
                ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 20)),
                                             hours=int(self.rng.integers(8, 22)))
                cart = float(np.round(self.rng.lognormal(7.3, 0.7), 2))
                checkout = float(np.round(cart * self.rng.uniform(0.96, 1.04), 2))
                events.append({
                    "timestamp": str(ts),
                    "cart_amount": cart,
                    "checkout_amount": checkout,
                    "cart_checkout_delta": round(checkout - cart, 2),
                    "cart_checkout_delta_pct": round((checkout - cart) / max(cart, 1), 4),
                    "price_vs_market_ratio": round(checkout / cart, 3),
                    "agent_negotiation_rounds": int(self.rng.integers(1, 4)),
                    "hidden_rebate_amount": 0.0,
                    "rebate_to_non_customer_wallet": 0,
                    "merchant_agent_reputation_score": float(np.round(self.rng.uniform(0.6, 0.95), 3)),
                    "agent_verified": 1,
                    "label": 0,
                })

            # Colluding transaction — 75% train-window, 25% test-tail (see V024 note)
            day_lo, day_hi = (5, 22) if self.rng.random() < 0.75 else (26, 30)
            fraud_ts = base_ts + pd.Timedelta(days=int(self.rng.integers(day_lo, day_hi)),
                                               hours=int(self.rng.integers(9, 21)))
            cart = float(np.round(self.rng.lognormal(8.0, 0.7), 2))
            inflation = float(self.rng.uniform(infl_lo, infl_hi))
            checkout = float(np.round(cart * inflation, 2))
            rebate_pct = float(self.rng.uniform(rebate_lo, rebate_hi))
            rebate = float(np.round(checkout * rebate_pct, 2))
            rounds = int(self.rng.integers(rounds_lo, rounds_hi + 1))
            events.append({
                "timestamp": str(fraud_ts),
                "cart_amount": cart,
                "checkout_amount": checkout,
                "cart_checkout_delta": round(checkout - cart, 2),
                "cart_checkout_delta_pct": round((checkout - cart) / max(cart, 1), 4),
                "price_vs_market_ratio": round(inflation, 3),
                "agent_negotiation_rounds": rounds,
                "hidden_rebate_amount": rebate,
                "rebate_to_non_customer_wallet": 1,
                "merchant_agent_reputation_score": float(np.round(self.rng.uniform(0.3, 0.7), 3)),
                "agent_verified": 1,
                "label": 1,
            })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={"rebate": rebate, "rounds": rounds},
            ))
        return envelopes
