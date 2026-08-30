"""
V005 — Pig-Butchering Investment Scam Funnel
Signal: short contact_age (<60d), first large outbound wire, beneficiary is
investment-platform category, escalating transfer amounts, then sudden stop.
"""

import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V005Generator(BaseGenerator):
    VECTOR_ID = "V005"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        romance_lo, romance_hi = p.get("romance_duration_days_range", [14, 90])
        steps_lo, steps_hi = p.get("n_transfer_steps_range", [3, 8])
        final_lo, final_hi = p.get("final_amount_range", [50_000, 2_000_000])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        # --- Legit ---
        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V005", i)
            age = int(self.rng.integers(365, 3000))
            events = []
            known_benes = [f"BENE_{j:04d}" for j in self.rng.integers(0, 50, size=6)]
            for _ in range(int(self.rng.integers(10, 40))):
                day = int(self.rng.integers(0, 30))
                ts = base_ts + pd.Timedelta(days=day,
                                             hours=int(self.rng.integers(8, 22)))
                amount = float(np.round(self.rng.lognormal(mean=9.0, sigma=1.0), 2))
                # Legit users DO make investment transfers — just to known platforms
                is_investment = int(self.rng.random() < 0.1)
                events.append({
                    "timestamp": str(ts),
                    "amount": amount,
                    "beneficiary_id": str(self.rng.choice(known_benes)),
                    "contact_age_days": int(self.rng.integers(180, 3000)),
                    "is_investment_platform": is_investment,
                    "prior_transfer_count": int(self.rng.integers(5, 50)),
                    "amount_escalation_ratio": 1.0,
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={},
            ))

        # --- Fraud ---
        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V005", i)
            age = int(self.rng.integers(180, 2500))
            base_ts = pd.Timestamp("2026-07-01")
            events = []

            # Normal cover transactions
            for _ in range(int(self.rng.integers(5, 15))):
                day = int(self.rng.integers(0, 10))
                ts = base_ts + pd.Timedelta(days=day,
                                             hours=int(self.rng.integers(8, 22)))
                amount = float(np.round(self.rng.lognormal(mean=8.5, sigma=0.8), 2))
                events.append({
                    "timestamp": str(ts),
                    "amount": amount,
                    "beneficiary_id": f"BENE_{int(self.rng.integers(0, 50)):04d}",
                    "contact_age_days": int(self.rng.integers(180, 1000)),
                    "is_investment_platform": 0,
                    "prior_transfer_count": int(self.rng.integers(5, 30)),
                    "amount_escalation_ratio": 1.0,
                    "label": 0,
                })

            # Escalating transfers to fake investment platform
            romance_days = int(self.rng.integers(romance_lo, romance_hi))
            n_steps = int(self.rng.integers(steps_lo, steps_hi + 1))
            final_amount = float(np.round(self.rng.uniform(final_lo, final_hi), 2))
            # Build escalating amounts ending at final_amount
            amounts = np.geomspace(final_amount * 0.02, final_amount, n_steps)
            mule_bene = f"INVEST_SCAM_{i:04d}"
            for step, amt in enumerate(amounts):
                day = romance_days + step * int(self.rng.integers(2, 8))
                ts = base_ts + pd.Timedelta(days=day,
                                             hours=int(self.rng.integers(18, 23)))
                events.append({
                    "timestamp": str(ts),
                    "amount": float(np.round(amt, 2)),
                    "beneficiary_id": mule_bene,
                    "contact_age_days": romance_days + step,
                    "is_investment_platform": 1,
                    "prior_transfer_count": step,
                    "amount_escalation_ratio": float(np.round(
                        amt / max(amounts[0], 1), 3)),
                    "label": 1,
                })

            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age,
                                  "romance_duration_days": romance_days},
                generation_params={"n_steps": n_steps,
                                    "final_amount": final_amount},
            ))

        return envelopes
