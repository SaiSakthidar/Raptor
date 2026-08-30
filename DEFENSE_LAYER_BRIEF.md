# Defense Layer — Briefing for Blue Team

**Project:** Mastercard Innovation Challenge 2026  
**Your role:** Build the detection/defense layer (blue stack) that catches the 13 attack types the red team generates.

---

## What the red team gives you

The red generators produce **JSONL files** in `red/output/`. Each line is one actor's data in a standard envelope:

```json
{
  "vector_id": "V001",
  "actor_id": "FRAUD_V001_00003",
  "channel": "txn-sequence",
  "label": 1,
  "event_sequence": [
    {"timestamp": "2026-07-28 14:03:00", "amount": 47200.0, "mcc": "5732", "label": 0},
    {"timestamp": "2026-07-28 14:19:00", "amount": 44800.0, "mcc": "5732", "label": 0},
    {"timestamp": "2026-07-28 14:31:00", "amount": 48900.0, "mcc": "5732", "label": 1},
    {"timestamp": "2026-07-28 14:52:00", "amount": 46100.0, "mcc": "5732", "label": 1}
  ],
  "entity_features": {"account_age_days": 342},
  "generation_params": {"threshold": 50000, "burst_mcc": "5732"}
}
```

`actor_id` is the account/session. `label` on the envelope = is this actor a fraud actor. `label` inside each event = is this specific event the fraud event.

---

## The 13 attack vectors

| ID | Name | Channel | Modality | What to detect |
|----|------|---------|----------|----------------|
| V001 | Structuring / Threshold Evasion | txn-sequence | TXN | Burst of 4-6 txns within 90 min, each 80-99% of ₹50k, same MCC |
| V002 | Deepfake BEC / APP Fraud | txn-sequence | BENEFICIARY | Large wire to brand-new beneficiary, high urgency_score, amount >> 30d max |
| V003 | Synthetic Identity KYC Bypass | kyc-session | KYC | Low doc_age, emulator device, VPN, poor liveness score |
| V004 | Agentic Prompt Injection | agent-payment | AGENT | checkout_amount > cart_amount, hidden line items, unverified agent |
| V005 | Pig-Butchering Investment Scam | txn-sequence | BENEFICIARY | contact_age < 60d, first large wire to investment platform, escalating amounts |
| V006 | LLM Chargeback Fraud | txn-sequence | TXN | High dispute rate, AI-low perplexity text, disputes filed at deadline |
| V007 | AI BIN Attack / Card Testing | txn-sequence | TXN | Micro-txn burst (>10 probes) then exploit, all online MCC |
| V008 | Mule Account Network | txn-sequence | BENEFICIARY | High fan-in AND fan-out, dwell < 6 hours, pass_through_ratio > 0.9 |
| V009 | Adversarial Model Evasion | txn-sequence | MODEL | Features unnaturally clustered near decision boundary (OOD anomaly) |
| V010 | UPI Account Takeover | txn-sequence | TXN | Device changed, then 3+ UPI transfers in 15 min to new VPAs |
| V011 | BNPL Bust-Out | kyc-session | KYC | Account age < 14d, 3+ BNPL merchants in 7d, zero prior history |
| V012 | Digital Arrest Coercion | txn-sequence | CONTEXT | Call precedes transfer by < 30 min, new beneficiary, amount >> 5x max |
| V013 | Voice IVR Bypass | kyc-session | MEDIA | High voice_auth_confidence from unknown number, immediate account change |

---

## Your blue stack architecture

Build **three detection layers** — each catches different things:

### Layer 1 — Rules (fast, zero latency)
Hard threshold rules that catch the most obvious patterns.

```python
# Examples:
if txn_count_last_1hr >= 4 and same_mcc_count >= 3 and amount_to_threshold_ratio > 0.80:
    flag("V001_structuring_rule")

if is_new_beneficiary and amount > 500_000 and urgency_score > 0.7:
    flag("V002_bec_rule")

if device_changed_flag and upi_transfers_15min >= 3:
    flag("V010_upi_ato_rule")
```

Rules are fast and explainable. They'll catch ~60% of fraud with good precision. Anything the rules miss goes to Layer 2.

### Layer 2 — Supervised Classifier
Train one classifier **per channel** (not per vector — they share a feature space):

| Channel | Feature extractor needed | Vectors it covers |
|---------|--------------------------|-------------------|
| `txn-sequence` | Rolling-window featurizer (like featurize.py) | V001, V002, V005, V006, V007, V008, V009, V010, V012 |
| `kyc-session` | Single-row features (no window needed) | V003, V011, V013 |
| `agent-payment` | Per-session features | V004 |

**Model:** RandomForest or XGBoost, `class_weight="balanced"`, split by account (not by row).

**Crucial: split by account, not by row.** If you split by row, the model sees some transactions from a fraud burst in training and cheats on the rest. Account-level split proves it generalizes.

### Layer 3 — OOD / Anomaly Detector
**This is the most important layer for the judging criteria.**

The supervised classifier cannot catch V009 (adversarial evasion) or any held-out vector it never saw in training. The OOD layer catches them because their features are statistically anomalous even if the supervised score is low.

**Options:**
- **Isolation Forest** — fast, no labels needed, works well for tabular data
- **Autoencoder** — reconstruct the input; high reconstruction error = OOD
- Train only on legit data, or on the training split vectors. Test on held-out vectors.

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_train_legit)  # train only on legit examples

anomaly_scores = iso.decision_function(X_test)
# More negative = more anomalous
```

---

## What the eval harness should report

These are the judging metrics. Build eval code that outputs all of these.

| Metric | How to compute |
|--------|----------------|
| **Detection rate per vector** | Recall on each V001..V013 separately |
| **False positive rate** | % legit actors flagged as fraud |
| **Precision @ fixed FPR** | At FPR=1%, what is precision? (real-world metric) |
| **Coverage matrix** | Heatmap: modality × vector, detection rate in each cell |
| **Held-out vector improvement** | Train on V001-V010, hold out V011-V013; show OOD catches them |
| **Fidelity (optional)** | Real-vs-synthetic discriminator AUC (need a real dataset for this) |

---

## Example: what generated data looks like

### V001 — Structuring (fraud actor)
```
actor_id: FRAUD_V001_00003  label: 1
event 1:  2026-07-28 14:03  amount=47,200  mcc=5732  label=0
event 2:  2026-07-28 14:19  amount=44,800  mcc=5732  label=0
event 3:  2026-07-28 14:31  amount=48,900  mcc=5732  label=1
event 4:  2026-07-28 14:52  amount=46,100  mcc=5732  label=1
  → rolling features: txn_count_last_1hr=4, same_mcc=4, avg_amount=46,750
```

### V002 — BEC App Fraud (fraud actor)
```
actor_id: FRAUD_V002_00001  label: 1
events 1-12:  normal transfers ₹8,000-₹90,000 to known beneficiaries  label=0
event 13: 2026-07-29 15:22  amount=2,400,000  beneficiary_id=MULE_00001
          beneficiary_tenure_days=0  urgency_score=0.94  is_new_beneficiary=1  label=1
  → signal: first-ever transfer to this beneficiary, 26x historical max
```

### V008 — Mule Network (fraud actor)
```
actor_id: FRAUD_V008_00007  label: 1
inbound:  6 transfers from 6 different senders  total=₹840,000
outbound: 4 transfers to 4 recipients  4.5 hours later  total=₹820,000
  → pass_through_ratio=0.976, retail_txn_ratio=0.02, dwell=4.5h
```

### V009 — Adversarial Evasion
```
actor_id: FRAUD_V009_00002  label: 1
All transactions have amount_to_threshold_ratio ≈ 0.79 (±0.02)
txn_count_last_1hr always = 3, time_since_prev always ≈ 1800s
  → supervised classifier scores ~0.38 (just below alert)
  → OOD score: high (feature variance is unnaturally low)
```

---

## Run the red generators first

```bash
cd mastercard-hackathon
pip install pyyaml numpy pandas scikit-learn
python -m red.run_all
```

This creates `red/output/V001_structuring.jsonl` ... `V013_voice_ivr.jsonl` and a flat `ALL_vectors_flat.csv` you can load directly into pandas.

Then build your channel featurizer on top of that CSV, train per-channel classifiers, and add the IsolationForest OOD layer on top.

---

## Key gotchas

1. **Don't split by row** — split by `actor_id`. All events from one actor go to train or test, never both.
2. **V009 is a MODEL-modality attack** — the supervised classifier will miss it by design. Only your OOD layer should catch it. This is intentional and is the "closed loop" story.
3. **V003 and V013 are kyc-session channel** — they're single-event sessions, not time-series. No rolling window needed. Treat each event as one sample.
4. **Class imbalance is heavy** — use `class_weight="balanced"` or oversample fraud. Report precision and recall, not accuracy.
5. **The `label` field inside each event is the event-level label.** The `label` on the envelope is the actor-level label. For training the classifier, use event-level labels if you want event-level detection; use actor-level if you want account-level detection.
