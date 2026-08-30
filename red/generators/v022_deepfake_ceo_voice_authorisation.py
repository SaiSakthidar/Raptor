"""
V022 — Deepfake CEO Voice Authorisation
Channel : txn-sequence
Modality: CONTEXT

Attacker clones the CEO/CFO voice from <60s of public audio (earnings calls,
interviews). Calls finance urgently authorising a large wire "off-channel" —
no email trail. Transfer to a new beneficiary, amount >> 30d average,
initiated within 2 hours of an unscheduled inbound call from an external
number.

Signal: inbound_call_from_external_number=1, call_to_wire_hours < 2,
amount_vs_30d_avg_ratio > 8x, beneficiary_is_new=1, no_email_approval_trail=1.
"""
import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V022Generator(BaseGenerator):
    VECTOR_ID = "V022"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        wire_lo, wire_hi = p.get("wire_amount_range", [500000, 10000000])
        call_lo, call_hi = p.get("call_to_wire_hours_range", [0.25, 2.0])
        ratio_lo, ratio_hi = p.get("amount_ratio_range", [8, 30])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V022", i)
            age = int(self.rng.integers(365, 3000))
            avg_wire = float(np.round(self.rng.uniform(20000, 300000), 2))
            events = []
            for _ in range(int(self.rng.integers(8, 25))):
                ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 30)),
                                             hours=int(self.rng.integers(9, 18)))
                amount = float(np.round(self.rng.lognormal(np.log(avg_wire), 0.6), 2))
                events.append({
                    "timestamp": str(ts), "amount": amount,
                    "inbound_call_from_external_number": 0,
                    "call_to_wire_hours": -1,
                    "amount_vs_30d_avg_ratio": round(amount / avg_wire, 3),
                    "beneficiary_is_new": 0,
                    "no_email_approval_trail": 0,
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age, "avg_wire_30d": avg_wire},
                generation_params={},
            ))

        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V022", i)
            age = int(self.rng.integers(365, 2500))
            avg_wire = float(np.round(self.rng.uniform(20000, 200000), 2))
            events = []

            # Normal wire history before the attack
            for _ in range(int(self.rng.integers(6, 18))):
                ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 22)),
                                             hours=int(self.rng.integers(9, 18)))
                amount = float(np.round(self.rng.lognormal(np.log(avg_wire), 0.5), 2))
                events.append({
                    "timestamp": str(ts), "amount": amount,
                    "inbound_call_from_external_number": 0,
                    "call_to_wire_hours": -1,
                    "amount_vs_30d_avg_ratio": round(amount / avg_wire, 3),
                    "beneficiary_is_new": 0,
                    "no_email_approval_trail": 0,
                    "label": 0,
                })

            # Deepfake voice call → urgent off-channel wire
            fraud_ts = base_ts + pd.Timedelta(days=int(self.rng.integers(5, 22)) if self.rng.random() < 0.75 else int(self.rng.integers(26, 29)),
                                               hours=int(self.rng.integers(9, 20)))
            call_gap_hours = float(np.round(self.rng.uniform(call_lo, call_hi), 3))
            ratio = float(self.rng.uniform(ratio_lo, ratio_hi))
            fraud_amount = float(np.round(min(avg_wire * ratio,
                                               self.rng.uniform(wire_lo, wire_hi)), 2))
            events.append({
                "timestamp": str(fraud_ts),
                "amount": fraud_amount,
                "inbound_call_from_external_number": 1,
                "call_to_wire_hours": call_gap_hours,
                "amount_vs_30d_avg_ratio": round(fraud_amount / avg_wire, 3),
                "beneficiary_is_new": 1,
                "no_email_approval_trail": 1,
                "label": 1,
            })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age, "avg_wire_30d": avg_wire},
                generation_params={"fraud_amount": fraud_amount, "call_gap_hours": call_gap_hours},
            ))
        return envelopes
