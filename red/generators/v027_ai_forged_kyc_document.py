"""
V027 — AI-Forged KYC Document
Channel : kyc-session
Modality: MEDIA

Attacker submits an AI-generated fake government ID (passport/licence) at
account opening — a diffusion/GAN-generated image with a matching synthetic
selfie, engineered so basic liveness and face-match checks pass. Naive
face-match is defeated by design (the selfie and the forged document persona
were generated to match each other); catching this requires forensic image
analysis: font/kerning consistency, security-feature detection, and an
issuance-registry lookup.
"""
import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V027Generator(BaseGenerator):
    VECTOR_ID = "V027"
    CHANNEL = "kyc-session"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        forgery_lo, forgery_hi = p.get("forgery_score_range", [0.7, 0.98])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V027", i)
            doc_age = int(self.rng.integers(5, 3650))  # some legit users also have new documents
            ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 30)),
                                         hours=int(self.rng.integers(8, 22)))
            event = {
                "timestamp": str(ts),
                "doc_age_days": doc_age,
                "liveness_score": float(np.round(self.rng.uniform(0.82, 0.99), 4)),
                "selfie_consistency_score": float(np.round(self.rng.uniform(0.80, 0.99), 4)),
                "doc_forgery_score": float(np.round(self.rng.uniform(0.0, 0.15), 3)),
                "doc_security_feature_match": 1,
                "doc_issuance_registry_match": 1,
                "selfie_doc_face_match_score": float(np.round(self.rng.uniform(0.85, 0.99), 4)),
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

        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V027", i)
            doc_age = int(self.rng.integers(1, 20))
            forgery_score = float(np.round(self.rng.uniform(forgery_lo, forgery_hi), 3))
            ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 30)),
                                         hours=int(self.rng.integers(1, 6)))
            event = {
                "timestamp": str(ts),
                "doc_age_days": doc_age,
                "liveness_score": float(np.round(self.rng.uniform(0.75, 0.95), 4)),
                # The whole point of this attack: face-match still looks fine —
                # the selfie was generated to match the forged document.
                "selfie_consistency_score": float(np.round(self.rng.uniform(0.80, 0.98), 4)),
                "doc_forgery_score": forgery_score,
                "doc_security_feature_match": 0,
                "doc_issuance_registry_match": 0,
                "selfie_doc_face_match_score": float(np.round(self.rng.uniform(0.85, 0.99), 4)),
                "session_duration_sec": int(self.rng.integers(30, 120)),
                "label": 1,
            }
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=[event],
                entity_features={"doc_age_days": doc_age},
                generation_params={"forgery_score": forgery_score},
            ))
        return envelopes
