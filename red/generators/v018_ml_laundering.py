"""
V018 — AI Money-Laundering-as-a-Service / Layering
Signal: very high recipient count, amounts clustered near threshold × 0.65,
unnaturally regular inter-transaction timing (AI-scheduled), huge total volume.
"""
import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope

THRESHOLD = 50_000


class V018Generator(BaseGenerator):
    VECTOR_ID = "V018"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        recip_lo, recip_hi = p.get("n_recipients_range", [30, 100])
        pct_lo, pct_hi = p.get("amount_pct_of_threshold_range", [0.55, 0.75])
        reg_lo, reg_hi = p.get("scheduling_regularity_range", [0.85, 0.99])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V018", i)
            age = int(self.rng.integers(90, 2000))
            events = []
            for _ in range(int(self.rng.integers(10, 40))):
                ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 30)),
                                             hours=int(self.rng.integers(7, 22)),
                                             minutes=int(self.rng.integers(0, 60)))
                amount = float(np.round(self.rng.lognormal(mean=9, sigma=1.2), 2))
                events.append({
                    "timestamp": str(ts), "amount": amount,
                    "recipients_7d": int(self.rng.integers(1, 8)),
                    "amount_pct_of_threshold": round(amount / THRESHOLD, 4),
                    "scheduling_regularity": float(np.round(self.rng.uniform(0.1, 0.72), 3)),  # some legit habits are fairly regular too
                    "total_volume_7d": amount * int(self.rng.integers(1, 10)),
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
            actor_id = self._actor_id("FRAUD_V018", i)
            age = int(self.rng.integers(30, 180))  # relatively new shell account
            n_recipients = int(self.rng.integers(recip_lo, recip_hi + 1))
            amount_pct = float(np.round(self.rng.uniform(pct_lo, pct_hi), 4))
            regularity = float(np.round(self.rng.uniform(reg_lo, reg_hi), 3))
            events = []

            # AI-scheduled uniform batch of layering transactions
            burst_start = base_ts + pd.Timedelta(days=int(self.rng.integers(5, 25)),
                                                  hours=int(self.rng.integers(0, 4)))
            interval_min = int(self.rng.integers(8, 20))  # very regular
            t = burst_start
            total_volume = 0.0
            for j in range(n_recipients):
                # Small jitter around the regular interval
                jitter = int(self.rng.normal(0, interval_min * (1 - regularity)))
                t = t + pd.Timedelta(minutes=max(1, interval_min + jitter))
                amount = float(np.round(THRESHOLD * amount_pct * self.rng.uniform(0.97, 1.03), 2))
                total_volume += amount
                events.append({
                    "timestamp": str(t),
                    "amount": amount,
                    "recipients_7d": j + 1,
                    "amount_pct_of_threshold": round(amount / THRESHOLD, 4),
                    "scheduling_regularity": regularity,
                    "total_volume_7d": round(total_volume, 2),
                    "label": 1,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age, "n_recipients": n_recipients},
                generation_params={"n_recipients": n_recipients, "regularity": regularity},
            ))
        return envelopes
