"""
V019 — Label Poisoning via Dispute Feedback Loop (MODEL modality)
Signal: cluster of coordinated accounts all filing false "not fraud" disputes
on known-bad transactions — high timing similarity, high false-claim rate,
actor cluster size > 5. Targets the retraining pipeline, not individual txns.
"""
import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V019Generator(BaseGenerator):
    VECTOR_ID = "V019"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        cluster_lo, cluster_hi = p.get("cluster_size_range", [5, 20])
        claim_lo, claim_hi = p.get("false_claim_rate_range", [0.80, 0.99])
        sim_lo, sim_hi = p.get("timing_similarity_range", [0.85, 0.99])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V019", i)
            age = int(self.rng.integers(90, 2000))
            events = []
            for _ in range(int(self.rng.integers(3, 15))):
                ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 30)),
                                             hours=int(self.rng.integers(9, 18)))
                events.append({
                    "timestamp": str(ts),
                    "dispute_false_claim_rate": float(np.round(self.rng.uniform(0.0, 0.4), 3)),  # some legit customers dispute often too
                    "dispute_timing_similarity_score": float(np.round(self.rng.uniform(0.0, 0.3), 3)),
                    "dispute_success_rate": float(np.round(self.rng.uniform(0.5, 0.9), 3)),
                    "actor_cluster_size": 1,
                    "n_disputes_30d": int(self.rng.integers(0, 3)),
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={},
            ))

        # Fraud actors in coordinated clusters
        n_clusters = max(1, n_fraud // 5)
        actor_idx = 0
        for c in range(n_clusters):
            cluster_size = int(self.rng.integers(cluster_lo, cluster_hi + 1))
            false_claim_rate = float(np.round(self.rng.uniform(claim_lo, claim_hi), 3))
            timing_similarity = float(np.round(self.rng.uniform(sim_lo, sim_hi), 3))
            # All cluster actors file disputes in the same narrow window
            cluster_base_ts = base_ts + pd.Timedelta(days=int(self.rng.integers(15, 28)))

            for j in range(min(cluster_size, n_fraud - actor_idx)):
                actor_id = self._actor_id("FRAUD_V019", actor_idx)
                age = int(self.rng.integers(30, 365))
                events = []
                n_disputes = int(self.rng.integers(3, 10))
                for d in range(n_disputes):
                    # Tiny jitter around cluster base — very similar timing
                    jitter_hours = float(self.rng.normal(0, (1 - timing_similarity) * 24))
                    ts = cluster_base_ts + pd.Timedelta(hours=jitter_hours,
                                                         minutes=int(self.rng.integers(0, 30)))
                    events.append({
                        "timestamp": str(ts),
                        "dispute_false_claim_rate": false_claim_rate,
                        "dispute_timing_similarity_score": timing_similarity,
                        "dispute_success_rate": float(np.round(self.rng.uniform(0.85, 0.99), 3)),
                        "actor_cluster_size": cluster_size,
                        "n_disputes_30d": n_disputes,
                        "label": 1,
                    })
                envelopes.append(Envelope(
                    vector_id=self.VECTOR_ID, actor_id=actor_id,
                    channel=self.CHANNEL, label=1,
                    event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                    entity_features={"account_age_days": age, "cluster_id": c},
                    generation_params={"cluster_size": cluster_size,
                                       "false_claim_rate": false_claim_rate},
                ))
                actor_idx += 1
                if actor_idx >= n_fraud:
                    break
            if actor_idx >= n_fraud:
                break

        return envelopes
