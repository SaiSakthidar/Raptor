"""
V007 — AI-Orchestrated BIN Attack / Card Testing
Signal: burst of micro-transactions (<₹50) across many online merchants,
then a single large exploit transaction. All probes same online MCC.
"""

import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope

ONLINE_MCC = "5734"  # Computer Software Stores (common for online fraud probes)


class V007Generator(BaseGenerator):
    VECTOR_ID = "V007"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        probe_lo, probe_hi = p.get("probe_count_range", [8, 25])
        probe_amt_lo, probe_amt_hi = p.get("probe_amount_range", [1, 50])
        exploit_lo, exploit_hi = p.get("exploit_amount_range", [5000, 50000])
        probe_window = p.get("probe_window_minutes", 45)

        MCCS = ["5411", "5812", "5942", "4111", "5999", "5814", "5691"]
        MERCHANTS = [f"M{i:03d}" for i in range(100)]

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        # --- Legit ---
        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V007", i)
            age = int(self.rng.integers(3, 2000))  # some legit cardholders are also new
            events = []
            for _ in range(int(self.rng.integers(10, 40))):
                ts = base_ts + pd.Timedelta(
                    days=int(self.rng.integers(0, 30)),
                    hours=int(self.rng.integers(6, 23)),
                    minutes=int(self.rng.integers(0, 60)))
                amount = float(np.round(self.rng.lognormal(mean=8.0, sigma=1.0), 2))
                events.append({
                    "timestamp": str(ts),
                    "amount": amount,
                    "merchant_id": str(self.rng.choice(MERCHANTS)),
                    "mcc": str(self.rng.choice(MCCS)),
                    "is_online": int(self.rng.random() < 0.3),
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
            actor_id = self._actor_id("FRAUD_V007", i)
            age = int(self.rng.integers(1, 30))  # fresh card
            events = []

            # Optional cover: a couple of normal-looking transactions before
            for _ in range(int(self.rng.integers(0, 3))):
                ts = base_ts + pd.Timedelta(
                    days=int(self.rng.integers(0, 20)),
                    hours=int(self.rng.integers(8, 18)))
                events.append({
                    "timestamp": str(ts),
                    "amount": float(np.round(self.rng.lognormal(mean=7.5, sigma=0.8), 2)),
                    "merchant_id": str(self.rng.choice(MERCHANTS)),
                    "mcc": str(self.rng.choice(MCCS)),
                    "is_online": 0,
                    "label": 0,
                })

            # The probe burst
            probe_start = base_ts + pd.Timedelta(
                days=int(self.rng.integers(5, 22)) if self.rng.random() < 0.75 else int(self.rng.integers(26, 29)),
                hours=int(self.rng.integers(2, 6)))   # odd hours
            n_probes = int(self.rng.integers(probe_lo, probe_hi + 1))
            t = probe_start
            probe_merchants = self.rng.choice(MERCHANTS, size=n_probes, replace=False)
            for j in range(n_probes):
                t = t + pd.Timedelta(seconds=int(self.rng.integers(30, 180)))
                events.append({
                    "timestamp": str(t),
                    "amount": float(np.round(
                        self.rng.uniform(probe_amt_lo, probe_amt_hi), 2)),
                    "merchant_id": str(probe_merchants[j]),
                    "mcc": ONLINE_MCC,
                    "is_online": 1,
                    "label": 0,  # probe itself isn't flagged as fraud row
                })

            # The exploit
            exploit_ts = t + pd.Timedelta(minutes=int(self.rng.integers(2, 15)))
            events.append({
                "timestamp": str(exploit_ts),
                "amount": float(np.round(self.rng.uniform(exploit_lo, exploit_hi), 2)),
                "merchant_id": str(self.rng.choice(MERCHANTS)),
                "mcc": ONLINE_MCC,
                "is_online": 1,
                "label": 1,
            })

            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age, "n_probes": n_probes},
                generation_params={"n_probes": n_probes,
                                    "probe_window_minutes": probe_window},
            ))

        return envelopes
