# RAPTOR — Real-time Adversarial Payment Threat Orchestrator

**Mastercard Innovation Challenge 2026** · Submitted by Team Arintra

RAPTOR is an end-to-end adversarial AI platform that (1) catalogs 27 novel GenAI-powered payment fraud attacks, (2) generates realistic synthetic attack simulations at scale, and (3) defends against them with a three-layer detection stack — all wired together with a live interactive dashboard.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [System Architecture](#system-architecture)
3. [Red Team — Attack Generation](#red-team--attack-generation)
4. [Blue Team — Defense Stack](#blue-team--defense-stack)
5. [Attack Catalog](#attack-catalog)
6. [Results](#results)
7. [Live Dashboard](#live-dashboard)
8. [Getting Started](#getting-started)
9. [Repository Layout](#repository-layout)
10. [Team](#team)

---

## Problem Statement

Generative AI has fundamentally changed the attacker's toolkit. Fraudsters can now synthesize KYC documents, clone voices, inject prompts into payment agents, and train evasion models against production classifiers — all at near-zero marginal cost. Existing fraud detection systems were built for rule-based or statistical anomalies; they have no simulation infrastructure to discover what the next GenAI attack even looks like.

RAPTOR closes that gap.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          RAPTOR PLATFORM                              │
│                                                                      │
│  ┌────────────────────────────────────┐                              │
│  │          RED TEAM (Simulation)     │                              │
│  │                                    │                              │
│  │  attack_catalog.yaml               │                              │
│  │       ↓                            │                              │
│  │  red/generators/v001…v027.py       │                              │
│  │       ↓  Emission Envelope Schema  │                              │
│  │  red/output/*.jsonl  (per-vector)  │                              │
│  └──────────────────┬─────────────────┘                             │
│                     │  JSONL streams                                 │
│  ┌──────────────────▼─────────────────┐                             │
│  │        BLUE TEAM (Detection)       │                              │
│  │                                    │                              │
│  │  blue/featurize.py  (per-channel)  │                              │
│  │       ↓                            │                              │
│  │  Layer 1: Cross-vector LightGBM    │  ← trained on all vectors   │
│  │  Layer 2: Per-vector specialists   │  ← auto-gated on legit FPR  │
│  │  Layer 3: IsolationForest OOD      │  ← catches zero-days        │
│  │       ↓                            │                              │
│  │  4-way policy (APPROVE/DECLINE/    │                              │
│  │    STEP_UP/HOLD)                   │                              │
│  │  blue/results/*.json  + *.joblib   │                              │
│  └──────────────────┬─────────────────┘                             │
│                     │                                                │
│  ┌──────────────────▼─────────────────┐                             │
│  │        DASHBOARD (FastAPI + HTML)  │                              │
│  │                                    │                              │
│  │  POST /api/simulate/{vector_id}    │  ← live inference            │
│  │  GET  /api/results                 │                              │
│  │  GET  /api/catalog                 │                              │
│  └────────────────────────────────────┘                             │
└──────────────────────────────────────────────────────────────────────┘
```

### Four Channels

| Channel | Transport | Vectors |
|---|---|---|
| `txn-sequence` | Card / UPI transaction stream | V001, V002, V005–V010, V012, V014–V022, V025 |
| `kyc-session` | Onboarding / KYC events | V003, V011, V013, V027 |
| `agent-payment` | Agentic payment rail | V004, V023, V026 |
| `chat-call` | Support chat / IVR call | V024 |

---

## Red Team — Attack Generation

### Emission Envelope Schema

Every generator emits the same JSON structure, regardless of attack type:

```json
{
  "vector_id":        "V001",
  "actor_id":         "ACT_V001_0042",
  "channel":          "txn-sequence",
  "label":            1,
  "event_sequence":   [{"event_type": "...", "ts_day": 3, ...}],
  "entity_features":  {"account_age_days": 14, ...},
  "generation_params": {"seed": 42, "n_fraud": 1}
}
```

This uniform interface means the blue stack does not need to know which generator produced an envelope.

### Key Design Decisions

- **Per-vector chronological split** — each vector gets its own 70 / 15 / 15 (train / val / test) cut by its own timeline, then rows are unioned. Prevents all-in-train skew for low-volume vectors.
- **Overlapping legit / fraud distributions** — legit and fraud features overlap on marginals; only joint patterns or sequences separate them. Verified by systematic zero-overlap scan across all 27 vectors.
- **Zero-day withholding** — V011, V012, V013 labels are withheld entirely from supervised training. Only the OOD layer may catch them.
- **Adversarial timing** — V009 (Model Evasion) mimics known-good behavior for 25 days then strikes at the tail; zero-days inherit this pattern.
- **Auto-discovery** — `red/run_all.py` uses `importlib` + `inspect` to find all `v*.py` generators; no manual registration needed.

### Scaffold Generator

Add a new attack vector in 30 seconds:

```bash
python -m red.scaffold_generator V028 --list   # browse catalog
python -m red.scaffold_generator V028          # generates runnable boilerplate
```

---

## Blue Team — Defense Stack

### Layer 1 — Cross-Vector LightGBM (Supervised)

One gradient-boosted classifier per channel, trained on all known vectors. Features are channel-specific:

- **txn-sequence**: 63 features — velocity, amount patterns, UPI flags, terminal diversity, escalation ratios, V014–V025 specific signals
- **kyc-session**: 22 features — liveness scores, document age, biometric consistency, BNPL utilization, entity-level counts
- **agent-payment**: 19 features — checkout amount, merchant verification, agent collusion signals (V023/V026)
- **chat-call**: 7 features — IVR interaction flags, vishing cadence, caller trust score

### Layer 2 — Per-Vector Specialist Classifiers

One LightGBM per attack vector, auto-gated: any specialist where legit FPR @ 0.5 > 0.5% is rejected to prevent overfitting. The ensemble score is the max across all non-rejected specialists. Applied to `txn-sequence` only (highest vector diversity).

### Layer 3 — IsolationForest OOD

Trained on legit-only data. Calibrated at train time using fixed `[cal_min, cal_max]` bounds derived from the training legit set — so the same absolute anomaly score maps to the same OOD probability regardless of what's in the inference batch. Zero-day vectors (V011, V012, V013) are the primary target of this layer.

### Operating Point

Threshold = lowest value where FPR ≤ 1% on the validation set. Combined score = 0.7 × supervised + 0.3 × OOD.

### 4-Way Policy Engine

| Score | Action | Rationale |
|---|---|---|
| ≥ threshold + 0.2 | DECLINE | High-confidence fraud |
| threshold … +0.2 | STEP_UP | MFA / additional verification |
| threshold − 0.05 … threshold | HOLD | Human review queue |
| < threshold − 0.05 | APPROVE | Clear |

---

## Attack Catalog

26 measured vectors + 1 zero-day (V012 withheld from supervised labels).

### Transaction Sequence

| ID | Name | Description |
|---|---|---|
| V001 | Cash-out Structuring | LLM-generated micro-transactions just below reporting thresholds, timed to avoid velocity rules |
| V002 | BEC App Fraud | Synthetic invoice + voice clone directs victim to redirect payroll; AI-written BEC email chain |
| V005 | Pig Butchering | Multi-week trust escalation with AI persona; gradual legitimate deposits then large withdrawal |
| V006 | Chargeback Fraud | AI-generated dispute scripts targeting refund policies at scale |
| V007 | BIN Enumeration Attack | ML-guided card testing — smallest viable amount, rotating BINs, GPU-speed probing |
| V008 | Mule Network Layering | AI-recruited money mules with synthetic identities; funds layered across accounts |
| V009 | Model Evasion (Adversarial) | Queries production model, reconstructs decision boundary, crafts transactions in blind spot |
| V010 | UPI Account Takeover | SIM swap + deepfake liveness defeat; UPI PIN reset via AI voice call to carrier |
| V012 | Synthetic Identity Bust-Out | **Zero-day** — credit-build phase with clean history, then simultaneous bust-out across lenders |
| V014 | SIM Swap + Instant Rail | Automated SIM swap followed by instant rail drain before OTP timeout |
| V015 | Vendor Invoice BEC | AI lookalike domain + forged invoice PDF; wire redirect at B2B payment step |
| V016 | Real-Estate Wire Fraud | LLM-crafted closing instruction spoof sent to title company; high-value single transaction |
| V017 | Grandparent Scam Industrialized | AI voice clones grandchild in distress; call center scales across thousands of elderly targets |
| V018 | ML-Assisted Money Laundering | Reinforcement-learning agent routes funds through layering chains to minimize detection probability |
| V019 | Training Data Poisoning | Fraudster submits dispute-labeled transactions to poison retraining pipeline; degrades future recall |
| V020 | Instant Rail APP Fraud | AI-crafted social engineering convinces victim to initiate their own transfer (authorized push) |
| V021 | Fake Investment Platform | AI-generated trading dashboard with fabricated returns; mass phishing with personalized lures |
| V022 | Deepfake CEO Voice Authorization | Real-time voice clone of CFO/CEO instructs treasury to authorize emergency wire transfer |
| V025 | Serial Return-Refund Ring | Coordinated product return fraud; AI optimizes claim wording and timing across a merchant ring |

### KYC Session

| ID | Name | Description |
|---|---|---|
| V003 | AI KYC Bypass | Generative video passes liveness check; synthetic selfie + manipulated ID document |
| V011 | BNPL Ghost Identity | **Zero-day** — synthetic identity exploits BNPL onboarding; entity-level signals withheld from labels |
| V013 | Voice IVR Vishing | **Zero-day** — deepfake voice defeats IVR biometrics; caller number spoofed via VoIP |
| V027 | AI-Forged KYC Document | Diffusion model generates photorealistic government ID; font/seal metadata pass OCR checks |

### Agent Payment

| ID | Name | Description |
|---|---|---|
| V004 | Agentic Payment Injection | Prompt injection into AI payment agent via malicious merchant description; covert fund diversion |
| V023 | Agent Collusion Kickback | Compromised merchant agent colludes with buyer agent; inflated checkout + kickback split |
| V026 | Rogue Merchant Agent | AI merchant agent accepts payments for non-existent goods; cryptographic receipt spoofing |

### Chat / Call

| ID | Name | Description |
|---|---|---|
| V024 | Support Chatbot Prompt Injection | User injects adversarial prompt into support session; chatbot exfiltrates account details or initiates transfer |

---

## Results

### Channel-Level Performance

| Channel | Vectors | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| txn-sequence | 19 | 0.873 (CI: 0.856–0.890) | 0.960 | 88.0% | 85.8% | 86.9% |
| kyc-session | 4 | 0.407 (CI: 0.321–0.493) | 0.579 | 81.3% | 19.4% | 31.3% |
| agent-payment | 3 | 1.000 | 1.000 | 89.5% | 100.0% | 94.4% |
| chat-call | 1 | 1.000 | 1.000 | 100.0% | 100.0% | 100.0% |

### Zero-Day Detection (OOD Layer)

| Vector | Attack | Supervised Recall | Combined Recall | OOD Lift |
|---|---|---|---|---|
| V012 | Synthetic Identity Bust-Out | 100% (label leaked in test) | 100% | — |
| V013 | Voice IVR Vishing | 0% | 20% | **+20pp** |
| V011 | BNPL Ghost Identity | 0% | 0% | 0% (known gap) |

V013 is detected solely by the IsolationForest layer — supervised recall is zero because labels were withheld. V011 remains an open gap documented as a known limitation (signal lives on onboarding event, not fraud event row).

### Business Impact (txn-sequence, test set)

| Metric | Value |
|---|---|
| Expected fraud loss approved | ₹60.8L / $72.4K |
| Friction cost (legit blocked) | ₹32.2L / $38.3K |
| Fraud events blocked | 85.8% of test set |
| Legit FPR | 0.93% |
| Policy: APPROVE | 92.7% |
| Policy: DECLINE | 7.4% |
| Policy: STEP_UP / HOLD | mix |

---

## Live Dashboard

```
http://localhost:8080
```

Features:
- **Filterable attack ledger** — search by name, filter by modality chip, sort by worst recall; press `/` to focus search
- **Expandable rows** — full attack description, expected signal, observed-by tags, per-vector precision/recall/F1 bars
- **Live simulation** — click any vector → backend generates a fresh actor with a new seed, runs real model inference, streams per-event scores to an animated console, shows CATCH / MISS verdict
- **Zero-day callouts** — OOD-only detection highlighted separately
- **Policy distribution** — stacked bar per channel
- **Auto-refresh** every 10 seconds

---

## Getting Started

### Option A — Docker (recommended)

```bash
docker build -t raptor .
docker run -p 8080:8080 raptor         # runs full pipeline + dashboard
```

### Option B — Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn lightgbm pyyaml joblib fastapi "uvicorn[standard]"

# 1. Generate all 26 attack simulations
bash run.sh --red

# 2. Train detectors and evaluate
bash run.sh                            # default: blue stack only

# 3. Launch dashboard
bash run.sh --ui
```

### Option C — Step by step

```bash
# Generate
python red/run_all.py

# Train + evaluate all channels
python blue/run_blue.py

# Dashboard
uvicorn dashboard.app:app --host 0.0.0.0 --port 8080 --reload
```

### Adding a New Attack Vector

```bash
# 1. Add entry to attack_catalog.yaml (vector_id, name, modality, channel, description, …)
# 2. Generate boilerplate generator
python -m red.scaffold_generator V028
# 3. Implement generate() in red/generators/v028_*.py
# 4. Re-run red + blue — auto-discovery picks it up automatically
```

---

## Repository Layout

```
.
├── attack_catalog.yaml          # source-of-truth: 27 attack vectors
├── run.sh                       # orchestration (--red / --ui / --all)
├── Dockerfile
│
├── red/                         # attack simulation (Red Team)
│   ├── envelope.py              # uniform emission schema
│   ├── base_generator.py        # abstract base class
│   ├── run_all.py               # auto-discovers + runs all generators
│   ├── scaffold_generator.py    # CLI: generate boilerplate for new vector
│   └── generators/
│       ├── v001_structuring.py
│       ├── v002_bec_app_fraud.py
│       └── … v027_ai_forged_kyc_document.py
│
├── blue/                        # fraud detection (Blue Team)
│   ├── featurize.py             # channel-specific feature engineering
│   ├── train.py                 # LightGBM + IsolationForest training
│   ├── evaluate.py              # PR-AUC, ROC-AUC, per-vector recall/F1
│   ├── run_blue.py              # pipeline entrypoint
│   ├── infer.py                 # live single-actor inference (used by dashboard)
│   └── results/
│       ├── summary.json         # full metrics (all channels)
│       ├── *_model.joblib       # persisted model artifacts
│       └── *_results.json       # per-channel detailed results
│
├── dashboard/
│   └── app.py                   # FastAPI + embedded HTML/CSS/JS
│
└── RAPTOR_Submission.docx        # formal submission document
```

---

## Team

| Name | Role | GitHub |
|---|---|---|
| Sai Sakthidar | Red team architecture, Blue stack, Dashboard | [@SaiSakthidar](https://github.com/SaiSakthidar) |
| Sneha Samanta | Defense layer, Evaluation framework | [@nthsneha](https://github.com/nthsneha) |

---

*Submitted for Mastercard Innovation Challenge 2026 · Global Fintech Festival, Mumbai · 8–11 September 2026*
