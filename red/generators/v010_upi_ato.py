"""
V010 — UPI / Fast-Payment Account Takeover
Signal: recent device change, burst of P2P UPI transfers within 15 minutes,
all to new VPAs (beneficiary Virtual Payment Addresses never seen before),
transfer amounts far above 30-day average.
"""

import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V010Generator(BaseGenerator):
    VECTOR_ID = "V010"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        drain_lo, drain_hi = p.get("drain_txn_count_range", [3, 8])
        drain_window = p.get("drain_window_minutes", 15)
        multiplier_lo, multiplier_hi = p.get("amount_vs_avg_multiplier_range", [3, 15])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        # --- Legit ---
        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V010", i)
            age = int(self.rng.integers(90, 2000))
            known_vpas = [f"user{self.rng.integers(0,1000)}@upi" for _ in range(8)]
            avg_txn = float(np.round(self.rng.uniform(500, 20_000), 2))
            events = []
            for _ in range(int(self.rng.integers(10, 40))):
                ts = base_ts + pd.Timedelta(
                    days=int(self.rng.integers(0, 30)),
                    hours=int(self.rng.integers(7, 22)),
                    minutes=int(self.rng.integers(0, 60)))
                amount = float(np.round(self.rng.lognormal(
                    mean=np.log(avg_txn), sigma=0.5), 2))
                events.append({
                    "timestamp": str(ts),
                    "amount": amount,
                    "beneficiary_vpa": str(self.rng.choice(known_vpas)),
                    "is_new_vpa": 0,
                    "device_changed_flag": 0,
                    "amount_vs_30d_avg_ratio": float(np.round(amount / avg_txn, 3)),
                    "upi_transfers_15min": int(self.rng.integers(0, 2)),
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age,
                                  "avg_txn_amount_30d": avg_txn},
                generation_params={},
            ))

        # --- Fraud: ATO drain ---
        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V010", i)
            age = int(self.rng.integers(90, 1500))
            avg_txn = float(np.round(self.rng.uniform(500, 15_000), 2))
            events = []

            # Normal history
            known_vpas = [f"user{self.rng.integers(0,1000)}@upi" for _ in range(6)]
            for _ in range(int(self.rng.integers(8, 25))):
                ts = base_ts + pd.Timedelta(
                    days=int(self.rng.integers(0, 25)),
                    hours=int(self.rng.integers(7, 22)))
                amount = float(np.round(self.rng.lognormal(
                    mean=np.log(avg_txn), sigma=0.5), 2))
                events.append({
                    "timestamp": str(ts),
                    "amount": amount,
                    "beneficiary_vpa": str(self.rng.choice(known_vpas)),
                    "is_new_vpa": 0,
                    "device_changed_flag": 0,
                    "amount_vs_30d_avg_ratio": float(np.round(amount / avg_txn, 3)),
                    "upi_transfers_15min": 1,
                    "label": 0,
                })

            # Device change event (day 26-28)
            device_change_ts = base_ts + pd.Timedelta(
                days=int(self.rng.integers(5, 22)) if self.rng.random() < 0.75 else int(self.rng.integers(26, 29)),
                hours=int(self.rng.integers(1, 5)))  # middle of night

            # Drain burst: within 15 minutes of device change
            n_drain = int(self.rng.integers(drain_lo, drain_hi + 1))
            multiplier = float(self.rng.uniform(multiplier_lo, multiplier_hi))
            t = device_change_ts + pd.Timedelta(minutes=int(self.rng.integers(2, 8)))
            for j in range(n_drain):
                amount = float(np.round(avg_txn * multiplier / n_drain, 2))
                events.append({
                    "timestamp": str(t),
                    "amount": amount,
                    "beneficiary_vpa": f"mule{i}_{j}@upi",
                    "is_new_vpa": 1,
                    "device_changed_flag": 1,
                    "amount_vs_30d_avg_ratio": float(np.round(amount / avg_txn, 3)),
                    "upi_transfers_15min": n_drain,
                    "label": 1,
                })
                t = t + pd.Timedelta(minutes=int(self.rng.integers(1, 4)))

            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age,
                                  "avg_txn_amount_30d": avg_txn},
                generation_params={"n_drain_txns": n_drain,
                                    "drain_window_minutes": drain_window},
            ))

        return envelopes
