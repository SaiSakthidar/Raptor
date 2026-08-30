"""
V021 — AI Fake Crypto / Investment Platform
Signal: transfers to a newly registered platform, escalating amounts over
multiple weeks, platform_registration_age_days < 60, total exposure >> avg.
"""
import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V021Generator(BaseGenerator):
    VECTOR_ID = "V021"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        plat_age_lo, plat_age_hi = p.get("platform_age_range", [7, 60])
        steps_lo, steps_hi = p.get("n_transfer_steps_range", [4, 10])
        final_lo, final_hi = p.get("final_amount_range", [100000, 5000000])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V021", i)
            age = int(self.rng.integers(365, 3000))
            events = []
            # Legit investors DO use platforms — but registered/established ones
            for _ in range(int(self.rng.integers(8, 30))):
                ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 30)),
                                             hours=int(self.rng.integers(9, 18)))
                amount = float(np.round(self.rng.lognormal(mean=10, sigma=1.0), 2))
                is_platform = int(self.rng.random() < 0.15)
                events.append({
                    "timestamp": str(ts), "amount": amount,
                    "beneficiary_type_is_platform": is_platform,
                    "platform_registration_age_days": int(
                        self.rng.integers(180, 3000)) if is_platform else -1,
                    "n_transfers_to_platform": int(self.rng.integers(1, 8)) if is_platform else 0,
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

        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V021", i)
            age = int(self.rng.integers(365, 2500))
            events = []
            plat_age = int(self.rng.integers(plat_age_lo, plat_age_hi + 1))
            n_steps = int(self.rng.integers(steps_lo, steps_hi + 1))
            final_amount = float(np.round(self.rng.uniform(final_lo, final_hi), 2))
            platform_id = f"FAKE_PLATFORM_{i:04d}"

            # Normal spend before the scam
            for _ in range(int(self.rng.integers(5, 15))):
                ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 10)),
                                             hours=int(self.rng.integers(9, 18)))
                amount = float(np.round(self.rng.lognormal(mean=9.5, sigma=0.8), 2))
                events.append({
                    "timestamp": str(ts), "amount": amount,
                    "beneficiary_type_is_platform": 0,
                    "platform_registration_age_days": -1,
                    "n_transfers_to_platform": 0,
                    "amount_escalation_ratio": 1.0,
                    "label": 0,
                })

            # Escalating transfers to fake platform
            amounts = np.geomspace(final_amount * 0.03, final_amount, n_steps)
            for step, amt in enumerate(amounts):
                day = 12 + step * int(self.rng.integers(2, 6))
                ts = base_ts + pd.Timedelta(days=day, hours=int(self.rng.integers(18, 23)))
                events.append({
                    "timestamp": str(ts),
                    "amount": float(np.round(amt, 2)),
                    "beneficiary_type_is_platform": 1,
                    "platform_registration_age_days": plat_age,
                    "n_transfers_to_platform": step + 1,
                    "amount_escalation_ratio": float(np.round(amt / max(amounts[0], 1), 3)),
                    "label": 1,
                })

            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age,
                                  "platform_id": platform_id,
                                  "platform_age_days": plat_age},
                generation_params={"n_steps": n_steps, "final_amount": final_amount},
            ))
        return envelopes
