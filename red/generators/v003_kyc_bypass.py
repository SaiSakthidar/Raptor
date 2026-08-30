"""
V003 — Synthetic Identity KYC Bypass
Signal: AI-generated docs (all same low age), emulator/VPN device,
low liveness score, poor selfie-to-document consistency.
Channel: kyc-session (one event per session, not a transaction sequence).
"""

import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V003Generator(BaseGenerator):
    VECTOR_ID = "V003"
    CHANNEL = "kyc-session"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        emulator_fraud_pct = p.get("emulator_fraud_pct", 0.80)
        vpn_fraud_pct = p.get("vpn_fraud_pct", 0.75)
        liveness_lo, liveness_hi = p.get("liveness_fraud_score_range", [0.3, 0.65])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        # --- Legit KYC sessions ---
        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V003", i)
            # Real people: old documents, real device, high liveness
            doc_age = int(self.rng.integers(5, 3650))  # some legit users also have new documents
            liveness = float(np.round(self.rng.uniform(0.68, 0.99), 4))  # some legit sessions have weaker liveness too
            selfie_consistency = float(np.round(self.rng.uniform(0.62, 0.99), 4))
            ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 30)),
                                         hours=int(self.rng.integers(8, 22)))
            event = {
                "timestamp": str(ts),
                "doc_age_days": doc_age,
                "liveness_score": liveness,
                "selfie_consistency_score": selfie_consistency,
                "device_is_emulator": 0,
                "ip_is_vpn": 0,
                "doc_metadata_age_variance": float(np.round(
                    self.rng.uniform(50, 3000), 1)),  # multiple real docs, different ages
                "n_documents_submitted": int(self.rng.integers(2, 4)),
                "session_duration_sec": int(self.rng.integers(90, 600)),
                "label": 0,
            }
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=[event],
                entity_features={"doc_age_days": doc_age},
                generation_params={},
            ))

        # --- Fraud KYC sessions ---
        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V003", i)
            # AI-generated docs: all freshly created (same timestamp window),
            # emulator device, VPN, low liveness
            doc_age = int(self.rng.integers(1, 14))  # fresh docs
            liveness = float(np.round(self.rng.uniform(liveness_lo, liveness_hi), 4))
            selfie_consistency = float(np.round(self.rng.uniform(0.40, 0.68), 4))
            ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 30)),
                                         hours=int(self.rng.integers(1, 5)))  # odd hours
            event = {
                "timestamp": str(ts),
                "doc_age_days": doc_age,
                "liveness_score": liveness,
                "selfie_consistency_score": selfie_consistency,
                "device_is_emulator": int(self.rng.random() < emulator_fraud_pct),
                "ip_is_vpn": int(self.rng.random() < vpn_fraud_pct),
                # All AI-generated docs have nearly zero age variance
                "doc_metadata_age_variance": float(np.round(
                    self.rng.uniform(0, 3), 1)),
                "n_documents_submitted": int(self.rng.integers(2, 4)),
                "session_duration_sec": int(self.rng.integers(20, 90)),  # too fast
                "label": 1,
            }
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=[event],
                entity_features={"doc_age_days": doc_age},
                generation_params={"emulator_fraud_pct": emulator_fraud_pct},
            ))

        return envelopes
