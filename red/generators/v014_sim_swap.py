"""
V014 — SIM-Swap → Account Takeover
Signal: SIM swap event closely preceding large transfer to new beneficiary,
new device, OTP source changed, amount >> 30d avg.
"""
import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V014Generator(BaseGenerator):
    VECTOR_ID = "V014"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        swap_lo, swap_hi = p.get("swap_to_drain_hours_range", [1, 12])
        mult_lo, mult_hi = p.get("drain_amount_multiplier_range", [5, 20])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V014", i)
            age = int(self.rng.integers(180, 2500))
            avg_txn = float(np.round(self.rng.uniform(1000, 30000), 2))
            events = []
            for _ in range(int(self.rng.integers(10, 40))):
                ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 30)),
                                             hours=int(self.rng.integers(7, 22)))
                amount = float(np.round(self.rng.lognormal(np.log(avg_txn), 0.5), 2))
                events.append({
                    "timestamp": str(ts), "amount": amount,
                    "sim_swap_preceding_transfer_hours": -1,
                    "otp_source_changed": 0, "device_is_new": 0,
                    "beneficiary_is_new": 0,
                    "amount_vs_30d_avg_ratio": round(amount / avg_txn, 3),
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age, "avg_txn_30d": avg_txn},
                generation_params={},
            ))

        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V014", i)
            age = int(self.rng.integers(180, 2000))
            avg_txn = float(np.round(self.rng.uniform(1000, 20000), 2))
            events = []

            # Normal history
            for _ in range(int(self.rng.integers(8, 20))):
                ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 25)),
                                             hours=int(self.rng.integers(8, 20)))
                amount = float(np.round(self.rng.lognormal(np.log(avg_txn), 0.5), 2))
                events.append({
                    "timestamp": str(ts), "amount": amount,
                    "sim_swap_preceding_transfer_hours": -1,
                    "otp_source_changed": 0, "device_is_new": 0,
                    "beneficiary_is_new": 0,
                    "amount_vs_30d_avg_ratio": round(amount / avg_txn, 3),
                    "label": 0,
                })

            # SIM swap then drain
            swap_ts = base_ts + pd.Timedelta(days=int(self.rng.integers(26, 30)),
                                              hours=int(self.rng.integers(1, 5)))
            swap_hours = float(np.round(self.rng.uniform(swap_lo, swap_hi), 2))
            drain_ts = swap_ts + pd.Timedelta(hours=swap_hours)
            multiplier = float(self.rng.uniform(mult_lo, mult_hi))
            drain_amount = float(np.round(avg_txn * multiplier, 2))
            events.append({
                "timestamp": str(drain_ts),
                "amount": drain_amount,
                "sim_swap_preceding_transfer_hours": swap_hours,
                "otp_source_changed": 1,
                "device_is_new": 1,
                "beneficiary_is_new": 1,
                "amount_vs_30d_avg_ratio": round(drain_amount / avg_txn, 3),
                "label": 1,
            })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age, "avg_txn_30d": avg_txn},
                generation_params={"drain_amount": drain_amount, "swap_hours": swap_hours},
            ))
        return envelopes
