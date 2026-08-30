"""
V008 — AI-Recruited Mule Account Network
Signal: high fan-in sender count, high fan-out recipient count,
low dwell time between receive and forward, near-zero retail spending,
pass-through ratio close to 1.0.
"""

import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V008Generator(BaseGenerator):
    VECTOR_ID = "V008"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        fan_in_lo, fan_in_hi = p.get("fan_in_range", [3, 12])
        fan_out_lo, fan_out_hi = p.get("fan_out_range", [2, 8])
        dwell_lo, dwell_hi = p.get("dwell_hours_range", [0.5, 6])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        # --- Legit ---
        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V008", i)
            age = int(self.rng.integers(10, 2000))  # some legit accounts are also fairly new
            # Normal accounts: mostly outbound retail, few inbound transfers
            inbound_senders = int(self.rng.integers(1, 4))
            outbound_recipients = int(self.rng.integers(1, 5))
            events = []
            for _ in range(int(self.rng.integers(10, 40))):
                ts = base_ts + pd.Timedelta(
                    days=int(self.rng.integers(0, 30)),
                    hours=int(self.rng.integers(8, 22)))
                is_inbound = int(self.rng.random() < 0.2)
                amount = float(np.round(self.rng.lognormal(mean=8.5, sigma=1.0), 2))
                dwell = float(np.round(self.rng.uniform(48, 720), 2))  # days to weeks
                events.append({
                    "timestamp": str(ts),
                    "amount": amount,
                    "direction": "inbound" if is_inbound else "outbound",
                    "inbound_sender_count_7d": inbound_senders,
                    "outbound_recipient_count_7d": outbound_recipients,
                    "avg_dwell_hours": dwell,
                    "retail_txn_ratio": float(np.round(self.rng.uniform(0.5, 0.9), 3)),
                    "pass_through_ratio": float(np.round(self.rng.uniform(0.0, 0.3), 3)),
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={},
            ))

        # --- Fraud mule accounts ---
        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V008", i)
            age = int(self.rng.integers(7, 90))  # new mule accounts
            fan_in = int(self.rng.integers(fan_in_lo, fan_in_hi + 1))
            fan_out = int(self.rng.integers(fan_out_lo, fan_out_hi + 1))
            dwell = float(np.round(self.rng.uniform(dwell_lo, dwell_hi), 2))
            events = []

            # Inbound transfers (receive stolen funds)
            total_inbound = 0.0
            for j in range(fan_in):
                ts = base_ts + pd.Timedelta(
                    days=int(self.rng.integers(0, 28)),
                    hours=int(self.rng.integers(0, 23)),
                    minutes=int(self.rng.integers(0, 60)))
                amount = float(np.round(self.rng.uniform(10_000, 500_000), 2))
                total_inbound += amount
                events.append({
                    "timestamp": str(ts),
                    "amount": amount,
                    "direction": "inbound",
                    "inbound_sender_count_7d": fan_in,
                    "outbound_recipient_count_7d": fan_out,
                    "avg_dwell_hours": dwell,
                    "retail_txn_ratio": float(np.round(self.rng.uniform(0.0, 0.05), 3)),
                    "pass_through_ratio": float(np.round(
                        self.rng.uniform(0.88, 0.99), 3)),
                    "label": 0,  # receiving isn't the fraud row
                })

            # Outbound forwards (fan out), shortly after inbound
            for j in range(fan_out):
                last_inbound_ts = max(e["timestamp"] for e in events
                                      if e["direction"] == "inbound")
                ts = pd.Timestamp(last_inbound_ts) + pd.Timedelta(hours=dwell)
                ts = ts + pd.Timedelta(minutes=int(self.rng.integers(0, 60)))
                amount = float(np.round(total_inbound / fan_out, 2))
                events.append({
                    "timestamp": str(ts),
                    "amount": amount,
                    "direction": "outbound",
                    "inbound_sender_count_7d": fan_in,
                    "outbound_recipient_count_7d": fan_out,
                    "avg_dwell_hours": dwell,
                    "retail_txn_ratio": float(np.round(self.rng.uniform(0.0, 0.05), 3)),
                    "pass_through_ratio": float(np.round(
                        self.rng.uniform(0.88, 0.99), 3)),
                    "label": 1,
                })

            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age,
                                  "fan_in": fan_in, "fan_out": fan_out},
                generation_params={"dwell_hours": dwell},
            ))

        return envelopes
