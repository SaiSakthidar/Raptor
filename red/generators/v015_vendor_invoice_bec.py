"""
V015 — LLM Vendor Invoice BEC (text-only)
Signal: known vendor (high tenure), bank details changed, new beneficiary
account, amount matches historical invoice pattern, no voice/video confirmation.
"""
import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V015Generator(BaseGenerator):
    VECTOR_ID = "V015"
    CHANNEL = "txn-sequence"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        ven_lo, ven_hi = p.get("vendor_tenure_range", [180, 1500])
        amt_lo, amt_hi = p.get("invoice_amount_range", [50000, 2000000])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V015", i)
            age = int(self.rng.integers(365, 3000))
            vendor_tenure = int(self.rng.integers(ven_lo, ven_hi))
            events = []
            for _ in range(int(self.rng.integers(5, 20))):
                ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 30)),
                                             hours=int(self.rng.integers(9, 17)))
                amount = float(np.round(self.rng.uniform(amt_lo * 0.1, amt_hi * 0.5), 2))
                events.append({
                    "timestamp": str(ts), "amount": amount,
                    "vendor_tenure_days": vendor_tenure,
                    "bank_details_changed": 0,
                    "beneficiary_account_age_days": int(self.rng.integers(180, 1000)),
                    "request_via_email_only": int(self.rng.random() < 0.4),
                    "phone_confirmation": 1,
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={},
            ))

        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V015", i)
            age = int(self.rng.integers(365, 2500))
            vendor_tenure = int(self.rng.integers(ven_lo, ven_hi))
            events = []

            # Normal vendor payments (known relationship)
            for _ in range(int(self.rng.integers(4, 12))):
                ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 25)),
                                             hours=int(self.rng.integers(9, 17)))
                amount = float(np.round(self.rng.uniform(amt_lo * 0.1, amt_lo * 0.5), 2))
                events.append({
                    "timestamp": str(ts), "amount": amount,
                    "vendor_tenure_days": vendor_tenure,
                    "bank_details_changed": 0,
                    "beneficiary_account_age_days": int(self.rng.integers(180, 800)),
                    "request_via_email_only": 0,
                    "phone_confirmation": 1,
                    "label": 0,
                })

            # The fraudulent invoice redirect
            fraud_ts = base_ts + pd.Timedelta(days=int(self.rng.integers(5, 22)) if self.rng.random() < 0.75 else int(self.rng.integers(26, 29)),
                                               hours=int(self.rng.integers(10, 15)))
            fraud_amount = float(np.round(self.rng.uniform(amt_lo, amt_hi), 2))
            events.append({
                "timestamp": str(fraud_ts),
                "amount": fraud_amount,
                "vendor_tenure_days": vendor_tenure,  # same vendor name
                "bank_details_changed": 1,             # but NEW bank details
                "beneficiary_account_age_days": int(self.rng.integers(1, 30)),
                "request_via_email_only": 1,
                "phone_confirmation": 0,
                "label": 1,
            })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age,
                                  "vendor_tenure_days": vendor_tenure},
                generation_params={"fraud_amount": fraud_amount},
            ))
        return envelopes
