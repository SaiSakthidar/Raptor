"""
V006 — LLM-Generated Chargeback / Friendly Fraud
Signal: high dispute rate, low text perplexity (AI-authored), low
sentence-length variance, disputes filed at optimal timing window.
"""

import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V006Generator(BaseGenerator):
    VECTOR_ID = "V006"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        disputes_lo, disputes_hi = p.get("disputes_per_fraud_range", [5, 20])
        ai_perp_lo, ai_perp_hi = p.get("ai_perplexity_range", [10, 35])
        human_perp_lo, human_perp_hi = p.get("human_perplexity_range", [55, 150])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        # --- Legit ---
        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V006", i)
            age = int(self.rng.integers(180, 3000))
            # a small fraction of legit customers genuinely dispute often too
            n_disputes = int(self.rng.integers(0, 8)) if self.rng.random() < 0.15 else int(self.rng.integers(0, 3))
            events = []
            for d in range(n_disputes):
                txn_day = int(self.rng.integers(0, 20))
                dispute_day = txn_day + int(self.rng.integers(5, 30))
                ts = base_ts + pd.Timedelta(days=dispute_day)
                events.append({
                    "timestamp": str(ts),
                    "dispute_amount": float(np.round(self.rng.uniform(100, 5000), 2)),
                    "days_since_txn": dispute_day - txn_day,
                    "dispute_text_perplexity": float(np.round(
                        self.rng.uniform(human_perp_lo, human_perp_hi), 2)),
                    "dispute_text_sentence_len_variance": float(np.round(
                        self.rng.uniform(15, 80), 2)),
                    "has_ai_evidence_image": 0,
                    "dispute_rate_30d": float(np.round(n_disputes / 30, 4)),
                    "label": 0,
                })
            if not events:  # accounts with no disputes
                ts = base_ts
                events.append({
                    "timestamp": str(ts),
                    "dispute_amount": 0.0,
                    "days_since_txn": 0,
                    "dispute_text_perplexity": 0.0,
                    "dispute_text_sentence_len_variance": 0.0,
                    "has_ai_evidence_image": 0,
                    "dispute_rate_30d": 0.0,
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
            actor_id = self._actor_id("FRAUD_V006", i)
            age = int(self.rng.integers(60, 1000))
            n_disputes = int(self.rng.integers(disputes_lo, disputes_hi + 1))
            events = []
            # Fraudster files disputes in tight window, just before deadline
            for d in range(n_disputes):
                txn_day = int(self.rng.integers(0, 15))
                # Optimal filing: day 55-60 of a 60-day dispute window
                dispute_day = int(self.rng.integers(16, 22)) if self.rng.random() < 0.75 else int(self.rng.integers(26, 29))
                ts = base_ts + pd.Timedelta(days=dispute_day)
                events.append({
                    "timestamp": str(ts),
                    "dispute_amount": float(np.round(self.rng.uniform(500, 8000), 2)),
                    "days_since_txn": dispute_day - txn_day,
                    "dispute_text_perplexity": float(np.round(
                        self.rng.uniform(ai_perp_lo, ai_perp_hi), 2)),
                    "dispute_text_sentence_len_variance": float(np.round(
                        self.rng.uniform(0.5, 5.0), 2)),  # very uniform = AI
                    "has_ai_evidence_image": int(self.rng.random() < 0.65),
                    "dispute_rate_30d": float(np.round(n_disputes / 30, 4)),
                    "label": 1,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age,
                                  "total_disputes": n_disputes},
                generation_params={"n_disputes": n_disputes},
            ))

        return envelopes
