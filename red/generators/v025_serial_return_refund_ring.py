"""
V025 — Serial Return / Refund Fraud Ring
Channel : txn-sequence
Modality: PROCEDURAL

An AI-coordinated ring buys goods, uses them, then files near-identical
LLM-generated "defective item" refund claims with AI-generated evidence
photos, spreading claims across many merchants to dodge any single
merchant's abuse threshold. Signature: high text-similarity-to-ring-template,
long use-before-return window, elevated AI-evidence score, many merchants
targeted, and a coordinated cluster of ring members filing in the same
narrow time window (same generation pattern used for V019).
"""
import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V025Generator(BaseGenerator):
    VECTOR_ID = "V025"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        cluster_lo, cluster_hi = p.get("cluster_size_range", [6, 18])
        amt_lo, amt_hi = p.get("refund_amount_range", [500, 25000])
        merch_lo, merch_hi = p.get("merchants_targeted_range", [6, 20])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V025", i)
            age = int(self.rng.integers(90, 2500))
            events = []
            for _ in range(int(self.rng.integers(1, 6))):
                ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 30)),
                                             hours=int(self.rng.integers(9, 20)))
                amount = float(np.round(self.rng.lognormal(7.5, 0.9), 2))
                events.append({
                    "timestamp": str(ts),
                    "amount": amount,
                    "item_used_days_before_return": int(self.rng.integers(0, 5)),
                    "refund_claim_text_similarity_score": float(np.round(self.rng.uniform(0.0, 0.3), 3)),
                    "ai_generated_evidence_score": float(np.round(self.rng.uniform(0.0, 0.2), 3)),
                    "n_refund_claims_30d": int(self.rng.integers(0, 2)),
                    "distinct_merchants_targeted_30d": int(self.rng.integers(1, 3)),
                    # occasionally a real product recall causes several genuine
                    # customers to return the same item around the same time
                    "actor_cluster_size": int(self.rng.integers(2, 4)) if self.rng.random() < 0.1 else 1,
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={},
            ))

        # Fraud actors organised into coordinated rings (same pattern as V019)
        n_clusters = max(1, n_fraud // 8)
        actor_idx = 0
        for c in range(n_clusters):
            cluster_size = int(self.rng.integers(cluster_lo, cluster_hi + 1))
            similarity = float(np.round(self.rng.uniform(0.85, 0.99), 3))
            ai_evidence = float(np.round(self.rng.uniform(0.7, 0.97), 3))
            n_merchants = int(self.rng.integers(merch_lo, merch_hi + 1))
            # Most rings file in the train-friendly window; some land in the
            # test tail so recall is measurable on held-out clusters too.
            day_lo, day_hi = (5, 22) if self.rng.random() < 0.75 else (25, 29)
            cluster_base_ts = base_ts + pd.Timedelta(days=int(self.rng.integers(day_lo, day_hi)))

            for j in range(min(cluster_size, n_fraud - actor_idx)):
                actor_id = self._actor_id("FRAUD_V025", actor_idx)
                age = int(self.rng.integers(30, 800))
                events = []
                n_claims = int(self.rng.integers(6, 20))
                for k in range(n_claims):
                    jitter_hours = float(self.rng.normal(0, (1 - similarity) * 48))
                    ts = cluster_base_ts + pd.Timedelta(
                        hours=jitter_hours, minutes=int(self.rng.integers(0, 60)))
                    amount = float(np.round(self.rng.uniform(amt_lo, amt_hi), 2))
                    events.append({
                        "timestamp": str(ts),
                        "amount": amount,
                        "item_used_days_before_return": int(self.rng.integers(20, 90)),
                        "refund_claim_text_similarity_score": similarity,
                        "ai_generated_evidence_score": ai_evidence,
                        "n_refund_claims_30d": n_claims,
                        "distinct_merchants_targeted_30d": n_merchants,
                        "actor_cluster_size": cluster_size,
                        "label": 1,
                    })
                envelopes.append(Envelope(
                    vector_id=self.VECTOR_ID, actor_id=actor_id,
                    channel=self.CHANNEL, label=1,
                    event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                    entity_features={"account_age_days": age, "cluster_id": c},
                    generation_params={"cluster_size": cluster_size, "similarity": similarity},
                ))
                actor_idx += 1
                if actor_idx >= n_fraud:
                    break
            if actor_idx >= n_fraud:
                break

        return envelopes
