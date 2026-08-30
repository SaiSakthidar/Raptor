"""
V013 — Voice-Clone IVR / Bank Contact-Center Bypass
Signal: high voice_auth_confidence from an unrecognized number,
immediately followed by account change or large transfer,
device not previously seen.
Channel: kyc-session (one call session event).
"""

import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V013Generator(BaseGenerator):
    VECTOR_ID = "V013"
    CHANNEL = "kyc-session"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        transfer_pct = p.get("post_auth_transfer_pct", 0.85)
        conf_fraud_lo, conf_fraud_hi = p.get("voice_confidence_fraud_range", [0.80, 0.99])
        conf_legit_lo, conf_legit_hi = p.get("voice_confidence_legit_range", [0.60, 0.98])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        # --- Legit call sessions ---
        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V013", i)
            age = int(self.rng.integers(90, 2000))
            n_calls = int(self.rng.integers(1, 5))
            events = []
            for _ in range(n_calls):
                ts = base_ts + pd.Timedelta(
                    days=int(self.rng.integers(0, 30)),
                    hours=int(self.rng.integers(9, 20)))
                # Legit: known number, real voice, variable confidence
                confidence = float(np.round(
                    self.rng.uniform(conf_legit_lo, conf_legit_hi), 4))
                transfer_amount = 0.0
                if self.rng.random() < 0.15:
                    transfer_amount = float(np.round(self.rng.uniform(500, 50_000), 2))
                events.append({
                    "timestamp": str(ts),
                    "voice_auth_confidence": confidence,
                    # a small fraction of legit callers use an unlisted/new number
                    "caller_number_known": int(self.rng.random() > 0.08),
                    "device_is_known": 1,
                    "account_change_flag": 0,
                    "post_auth_transfer_amount": transfer_amount,
                    "call_duration_sec": int(self.rng.integers(60, 600)),
                    "post_auth_action_count": int(self.rng.integers(1, 4)),
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={},
            ))

        # --- Fraud call sessions ---
        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V013", i)
            age = int(self.rng.integers(90, 1500))
            events = []
            ts = base_ts + pd.Timedelta(
                days=int(self.rng.integers(0, 30)),
                hours=int(self.rng.integers(8, 22)))
            # Cloned voice passes with high confidence
            confidence = float(np.round(
                self.rng.uniform(conf_fraud_lo, conf_fraud_hi), 4))
            does_transfer = self.rng.random() < transfer_pct
            transfer_amount = 0.0
            if does_transfer:
                transfer_amount = float(np.round(
                    self.rng.uniform(50_000, 2_000_000), 2))
            events.append({
                "timestamp": str(ts),
                "voice_auth_confidence": confidence,
                # sophisticated spoofing sometimes shows as a "known" number
                "caller_number_known": int(self.rng.random() < 0.15),
                "device_is_known": 0,
                "account_change_flag": int(not does_transfer or self.rng.random() < 0.5),
                "post_auth_transfer_amount": transfer_amount,
                "call_duration_sec": int(self.rng.integers(30, 120)),  # shorter
                "post_auth_action_count": int(self.rng.integers(3, 8)),  # more actions
                "label": 1,
            })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=[events[0]],
                entity_features={"account_age_days": age},
                generation_params={"voice_confidence": confidence,
                                    "transfer_amount": transfer_amount},
            ))

        return envelopes
