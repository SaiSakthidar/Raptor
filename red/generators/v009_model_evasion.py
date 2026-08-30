"""
V009 — Adversarial Evasion of Fraud Classifier (MODEL modality)
Fraudster has probed or extracted the decision boundary and crafts
transactions that score just below the alert threshold. These look normal
to the supervised layer but are anomalous to an OOD detector because
their features are unnaturally precise / clustered near the boundary.
"""

import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope

MCCS = ["5411", "5812", "5732", "5942", "4111", "5999", "5814", "5691"]
MERCHANTS = [f"M{i:03d}" for i in range(60)]

# Approximate "safe zone" feature values that evade a structuring detector
# (tuned to sit just below the alert boundary)
EVASION_TEMPLATE = {
    "txn_count_last_1hr": 3,          # one below typical burst trigger of 4
    "same_mcc_count_last_1hr": 2,
    "amount_to_threshold_ratio": 0.79, # just below 0.80 probe range
    "time_since_prev_txn_sec": 1800,   # 30 min — not fast enough to trigger velocity
}


class V009Generator(BaseGenerator):
    VECTOR_ID = "V009"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        perturb_std = p.get("perturbation_std", 0.02)
        THRESHOLD = 50_000

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        # --- Legit: normal transactions (no particular clustering near boundary) ---
        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V009", i)
            age = int(self.rng.integers(30, 2000))
            n_txns = int(self.rng.integers(10, 40))
            events = []
            for _ in range(n_txns):
                ts = base_ts + pd.Timedelta(
                    days=int(self.rng.integers(0, 30)),
                    hours=int(self.rng.integers(6, 23)),
                    minutes=int(self.rng.integers(0, 60)))
                amount = float(np.round(self.rng.lognormal(mean=8.5, sigma=1.2), 2))
                amount = min(amount, THRESHOLD * 1.5)
                events.append({
                    "timestamp": str(ts),
                    "amount": amount,
                    "amount_to_threshold_ratio": round(amount / THRESHOLD, 4),
                    "txn_count_last_1hr": int(self.rng.integers(1, 5)),
                    "same_mcc_count_last_1hr": int(self.rng.integers(1, 4)),
                    "time_since_prev_txn_sec": int(self.rng.integers(300, 86400)),
                    "merchant_id": str(self.rng.choice(MERCHANTS)),
                    "mcc": str(self.rng.choice(MCCS)),
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={},
            ))

        # --- Fraud: adversarially crafted to cluster near decision boundary ---
        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V009", i)
            age = int(self.rng.integers(30, 1000))
            events = []

            # Cover: a few totally normal transactions
            for _ in range(int(self.rng.integers(3, 8))):
                ts = base_ts + pd.Timedelta(
                    days=int(self.rng.integers(0, 20)),
                    hours=int(self.rng.integers(8, 18)))
                amount = float(np.round(self.rng.lognormal(mean=8.0, sigma=0.9), 2))
                events.append({
                    "timestamp": str(ts),
                    "amount": amount,
                    "amount_to_threshold_ratio": round(amount / THRESHOLD, 4),
                    "txn_count_last_1hr": 1,
                    "same_mcc_count_last_1hr": 1,
                    "time_since_prev_txn_sec": int(self.rng.integers(3600, 86400)),
                    "merchant_id": str(self.rng.choice(MERCHANTS)),
                    "mcc": str(self.rng.choice(MCCS)),
                    "label": 0,
                })

            # The adversarially crafted burst: features are unnaturally precise
            n_evasion = int(self.rng.integers(4, 8))
            mcc = str(self.rng.choice(MCCS))
            t = base_ts + pd.Timedelta(days=int(self.rng.integers(25, 30)),
                                        hours=int(self.rng.integers(8, 20)))
            for _ in range(n_evasion):
                # Add tiny Gaussian noise — but variance is much smaller than legit
                ratio = EVASION_TEMPLATE["amount_to_threshold_ratio"] + \
                    float(self.rng.normal(0, perturb_std))
                ratio = np.clip(ratio, 0.70, 0.85)
                amount = float(np.round(ratio * THRESHOLD, 2))
                txn_count = max(1, EVASION_TEMPLATE["txn_count_last_1hr"] +
                                int(self.rng.integers(-1, 2)))
                t = t + pd.Timedelta(
                    seconds=EVASION_TEMPLATE["time_since_prev_txn_sec"] +
                    int(self.rng.normal(0, 120)))
                events.append({
                    "timestamp": str(t),
                    "amount": amount,
                    "amount_to_threshold_ratio": round(ratio, 4),
                    "txn_count_last_1hr": txn_count,
                    "same_mcc_count_last_1hr": min(txn_count,
                        EVASION_TEMPLATE["same_mcc_count_last_1hr"]),
                    "time_since_prev_txn_sec": EVASION_TEMPLATE["time_since_prev_txn_sec"],
                    "merchant_id": str(self.rng.choice(MERCHANTS)),
                    "mcc": mcc,
                    "label": 1,
                })

            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={"perturbation_std": perturb_std,
                                    "evasion_template": EVASION_TEMPLATE},
            ))

        return envelopes
