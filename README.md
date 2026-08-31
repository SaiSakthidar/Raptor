# RAPTOR — Real-time Adversarial Payment Threat Orchestrator

**Mastercard Innovation Challenge 2026** · Team Arintra · GFF Mumbai, 8–11 September 2026

---

Fraud used to be someone stealing your card details and buying stuff online. GenAI changed the game completely. Now an attacker can clone a CEO's voice on a Monday morning call and get a ₹2 crore wire approved before lunch. They can generate a photorealistic passport that passes KYC in four seconds. They can inject a single sentence into a merchant's AI checkout agent and silently redirect every payment to their own account. They can spam your phone with MFA push notifications until you tap approve out of exhaustion.

The problem is not just that these attacks exist. It is that nobody has actually built them and measured whether detection systems can catch them. Everyone is defending against yesterday's fraud.

**RAPTOR builds the attacks first.**

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Red Team — Attack Simulation](#red-team--attack-simulation)
3. [Blue Team — Defense Stack](#blue-team--defense-stack)
4. [Closed-Loop Hardening](#closed-loop-hardening)
5. [Data Fidelity](#data-fidelity)
6. [Attack Catalog](#attack-catalog)
7. [Results](#results)
8. [Live Dashboard](#live-dashboard)
9. [Getting Started](#getting-started)
10. [Repository Layout](#repository-layout)
11. [Team](#team)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAPTOR PLATFORM                          │
│                                                                 │
│  ┌──────────────────────────────────────┐                       │
│  │  RED TEAM  (31 attack generators)    │                       │
│  │                                      │                       │
│  │  attack_catalog.yaml                 │                       │
│  │       ↓                              │                       │
│  │  red/generators/v001…v031.py         │                       │
│  │       ↓  Uniform Envelope Schema     │                       │
│  │  red/output/*.jsonl                  │                       │
│  └──────────────────┬───────────────────┘                       │
│                     │  JSONL streams (4 channels)               │
│  ┌──────────────────▼───────────────────┐                       │
│  │  BLUE TEAM  (3-layer detection)      │                       │
│  │                                      │                       │
│  │  Layer 1: Cross-vector LightGBM      │ ← all 31 vectors      │
│  │  Layer 2: Per-vector specialists     │ ← FPR-gated ensemble  │
│  │  Layer 3: IsolationForest OOD        │ ← catches zero-days   │
│  │       ↓                              │                       │
│  │  Closed-loop hardening (3 rounds)    │ ← ASR 55% → 0%       │
│  │       ↓                              │                       │
│  │  4-way policy engine                 │                       │
│  │  blue/results/*.json + *.joblib      │                       │
│  └──────────────────┬───────────────────┘                       │
│                     │                                           │
│  ┌──────────────────▼───────────────────┐                       │
│  │  DASHBOARD  (FastAPI + HTML)         │                       │
│  │  POST /api/simulate/{vector_id}      │ ← live inference      │
│  │  GET  /api/results  /api/fidelity    │                       │
│  └──────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

### Four Channels

| Channel | Transport | Vectors |
|---|---|---|
| `txn-sequence` | Card / UPI transaction stream | V001–V002, V005–V010, V012, V014–V022, V025, V028–V030 |
| `kyc-session` | Onboarding / KYC events | V003, V011, V013, V027 |
| `agent-payment` | Agentic payment rail | V004, V023, V026, V031 |
| `chat-call` | Support chat / IVR call | V024 |

---

## Red Team — Attack Simulation

### Emission Envelope Schema

Every generator emits the same JSON structure regardless of attack type:

```json
{
  "vector_id":        "V028",
  "actor_id":         "FRAUD_V028_0012",
  "channel":          "txn-sequence",
  "label":            1,
  "event_sequence":   [{"timestamp": "...", "push_count_before_approval": 14, ...}],
  "entity_features":  {"account_age_days": 342},
  "generation_params": {"push_count": 14}
}
```

The uniform interface means the blue stack does not need to know which generator produced an envelope — new vectors plug in automatically.

### Key Design Decisions

- **Per-vector chronological split** — each vector gets its own 70/15/15 cut by its own timeline, then rows are unioned. Guarantees every vector gets test-set representation regardless of how timelines interleave.
- **Overlapping marginal distributions** — legit and fraud features overlap on individual marginals; only the joint pattern across events is the true signal. Verified by KS scan across all 31 vectors.
- **Zero-day withholding** — V011, V012, V013 labels withheld entirely from supervised training. Only the OOD layer may catch them.
- **Auto-discovery** — `red/run_all.py` discovers generators via `importlib`/`inspect`; no manual registration.

### Scaffold Generator

```bash
python -m red.scaffold_generator V032 --list   # browse catalog
python -m red.scaffold_generator V032          # generates runnable boilerplate
```

---

## Blue Team — Defense Stack

### Layer 1 — Cross-Vector LightGBM

One gradient-boosted classifier per channel trained on all known vectors. Feature counts per channel:

| Channel | Features | Key signals |
|---|---|---|
| txn-sequence | 70 | velocity, amount patterns, push fatigue, VPA age, pass-through ratio |
| kyc-session | 24 | liveness score, doc forgery score, BNPL utilization, credit inquiry rate |
| agent-payment | 22 | token age, token reuse, session IP change, cart/checkout delta |
| chat-call | 7 | injection pattern, sensitive action flag, authentication level |

### Layer 2 — Per-Vector Specialist Classifiers

One LightGBM per attack vector on `txn-sequence`. Any specialist where legit FPR @ 0.5 > 0.5% is automatically rejected. Ensemble score = max across non-rejected specialists.

### Layer 3 — IsolationForest OOD

Trained on legit-only events. Calibration bounds (`cal_min`, `cal_max`) fixed at train time from the training legit pool — so the same absolute anomaly score maps to the same OOD probability regardless of what else is in the inference batch. Zero-day vectors are the primary target.

### Operating Point

Lowest threshold where FPR ≤ 1% on the validation set. Combined score = 0.6 × supervised + 0.4 × OOD.

### 4-Way Policy Engine

| Score | Decision | Rationale |
|---|---|---|
| ≥ threshold | DECLINE | High-confidence fraud |
| ≥ 0.9 × threshold | HOLD | Human review queue |
| ≥ 0.7 × threshold | STEP_UP | MFA / step-up auth |
| < 0.7 × threshold | APPROVE | Clear |

---

## Closed-Loop Hardening

Once the detector is trained, the red team generates evasive variants by nudging fraud feature vectors 30% toward the centroid of the legitimate distribution — simulating an attacker who has reverse-engineered roughly where the decision boundary sits. Variants that slip through (score < threshold) become hard negatives. The detector retrains on the augmented dataset, the threshold recalibrates to hold FPR ≤ 1%, and the cycle repeats.

**txn-sequence results:**

| Round | Recall | ROC-AUC | Attack Success Rate | Hard negatives added |
|---|---|---|---|---|
| 0 (baseline) | 86.1% | 0.9754 | 55.1% | — |
| 1 (hardened) | 86.1% | 0.9858 | **0.0%** | +787 |

ASR dropped from 55.1% to 0% in a single hardening round. The system does not just defend against known attacks — it learns to defend against the next mutation of each attack.

---

## Data Fidelity

Three independent probes verify the synthetic attack data is realistic:

| Channel | KS mean ↓ | Overlap < 0.5 | DCR ratio ↑ | TSTR PR-AUC |
|---|---|---|---|---|
| txn-sequence | **0.104** | 93.3% | 1.18× | 0.862 |
| kyc-session | 0.265 | 84.0% | 12.05× | 0.407 |
| agent-payment | 0.319 | 80.0% | 1.73× | 1.000 |
| chat-call | 0.448 | 71.4% | 18.22× | 1.000 |

- **KS mean** — Kolmogorov-Smirnov statistic comparing fraud vs legit marginal distributions. Lower = more realistic overlap. 0.104 on txn-sequence means fraud and legit are nearly indistinguishable on individual features — the joint pattern is the only true signal.
- **DCR ratio** — distance from each fraud test event to its nearest fraud training neighbor, divided by distance to nearest legit neighbor. Ratio > 1 proves the joint distribution is genuinely separable even when marginals overlap. Ratio 12× on kyc-session means KYC fraud is highly internally consistent — a realistic family structure.
- **TSTR PR-AUC** — Train on Synthetic, Test on (held-out) Synthetic. The detector's PR-AUC on the held-out test set is the synthetic data quality score.

Zero zero-overlap features detected on txn-sequence, kyc-session, and agent-payment channels.

---

## Attack Catalog

31 vectors across 4 channels. **Bold** = zero-day (labels withheld from supervised training).

### Transaction Sequence (23 vectors)

| ID | Name | Description |
|---|---|---|
| V001 | Cash-out Structuring | LLM-generated micro-transactions below reporting thresholds, timed to avoid velocity rules |
| V002 | BEC App Fraud | Synthetic invoice + voice clone redirects payroll; AI-written email chain |
| V005 | Pig Butchering | Multi-week AI persona trust escalation; gradual deposits then large withdrawal |
| V006 | Chargeback Fraud | AI-generated dispute scripts targeting refund policies at scale |
| V007 | BIN Enumeration | ML-guided card testing — smallest viable amount, rotating BINs, GPU-speed probing |
| V008 | Mule Network Layering | AI-recruited mules with synthetic identities; funds layered across accounts |
| V009 | Model Evasion | Queries production model, reconstructs decision boundary, crafts transactions in blind spot |
| V010 | UPI Account Takeover | SIM swap + deepfake liveness; UPI PIN reset via AI voice call to carrier |
| **V012** | **Synthetic Identity Bust-Out** | **Zero-day — credit-build phase then simultaneous bust-out across lenders** |
| V014 | SIM Swap + Instant Rail | Automated SIM swap followed by instant rail drain before OTP timeout |
| V015 | Vendor Invoice BEC | AI lookalike domain + forged invoice PDF; wire redirect at B2B payment step |
| V016 | Real-Estate Wire Fraud | LLM-crafted closing instruction spoof sent to title company |
| V017 | Grandparent Scam Industrialized | AI voice clones grandchild in distress; call center scales across thousands of elderly targets |
| V018 | ML-Assisted Money Laundering | RL agent routes funds through layering chains to minimise detection probability |
| V019 | Training Data Poisoning | Fraudster submits dispute-labeled transactions to poison retraining pipeline |
| V020 | Instant Rail APP Fraud | AI-crafted social engineering convinces victim to initiate their own transfer |
| V021 | Fake Investment Platform | AI-generated trading dashboard with fabricated returns; mass phishing with personalised lures |
| V022 | Deepfake CEO Voice Auth | Real-time voice clone of CFO instructs treasury to authorise emergency wire |
| V025 | Serial Return-Refund Ring | Coordinated return fraud; AI optimises claim wording across a merchant ring |
| V028 | MFA Push-Fatigue | Attacker spams MFA push notifications until victim accidentally approves; immediate drain |
| V029 | Cuckoo Smurfing | Criminal funds deposited into unwitting third-party accounts expecting legitimate remittance |
| V030 | UPI VPA Farm | Mass-created VPA addresses on payments banks; 4–7 hop chain in under 15 minutes |

### KYC Session (4 vectors)

| ID | Name | Description |
|---|---|---|
| V003 | AI KYC Bypass | Generative video passes liveness; synthetic selfie + manipulated ID document |
| **V011** | **BNPL Ghost Identity** | **Zero-day — synthetic identity exploits BNPL onboarding; entity-level signals withheld** |
| **V013** | **Voice IVR Vishing** | **Zero-day — deepfake voice defeats IVR biometrics; caller number spoofed via VoIP** |
| V027 | AI-Forged KYC Document | Diffusion-generated government ID; font/seal metadata passes OCR; defeats face-match |

### Agent Payment (4 vectors)

| ID | Name | Description |
|---|---|---|
| V004 | Agentic Payment Injection | Prompt injection into AI payment agent via malicious merchant description |
| V023 | Agent Collusion Kickback | Compromised merchant agent colludes with buyer agent; inflated checkout + kickback split |
| V026 | Rogue Merchant Agent | AI merchant agent accepts payments for non-existent goods; cryptographic receipt spoofing |
| V031 | Agentic Token Replay | Replays expired session token from prior legitimate agent session to initiate new payment |

### Chat / Call (1 vector)

| ID | Name | Description |
|---|---|---|
| V024 | Support Chatbot Prompt Injection | Adversarial prompt injected into support session; chatbot exfiltrates account details |

---

## Results

### Channel-Level Performance (test set, operating point FPR ≤ 1%)

| Channel | Vectors | PR-AUC (90% CI) | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| txn-sequence | 23 | 0.873 [0.856–0.890] | 0.960 | 87.9% | 85.8% | 86.9% |
| kyc-session | 4 | 0.407 [0.321–0.493] | 0.579 | 81.3% | 19.4% | 31.3% |
| agent-payment | 4 | 1.000 [1.000–1.000] | 1.000 | 89.5% | 100.0% | 94.4% |
| chat-call | 1 | 1.000 [1.000–1.000] | 1.000 | 100.0% | 100.0% | 100.0% |

### Zero-Day Detection (OOD layer only)

| Vector | Attack | Supervised Recall | Combined Recall | OOD Lift |
|---|---|---|---|---|
| V013 | Voice IVR Vishing | 0% | 20% | **+20pp** |
| V012 | Synthetic Identity Bust-Out | 100%* | 100% | — |
| V011 | BNPL Ghost Identity | 0% | 0% | Known gap (signal on wrong event row) |

*V012 supervised recall is high because it shares feature space with structuring vectors — a partial label leak; V013's 20% is purely OOD.

### Business Impact (txn-sequence, test set)

| Metric | Value |
|---|---|
| Fraud blocked (recall) | 85.8% |
| Expected fraud loss approved | ₹60.8L / $72.4K |
| Friction cost (legit blocked) | ₹32.2L / $38.3K |
| Legit FPR at operating point | 0.93% |
| Attack Success Rate after hardening | 0.0% |

---

## Live Dashboard

```
http://localhost:8080
```

- **Attack ledger** — all 31 vectors, filterable by modality, searchable, expandable with full description + per-vector precision/recall/F1 bars
- **Live simulation** — pick any vector, generate a fresh synthetic actor with a new random seed, watch the detector score each event in real time with an animated console
- **Fidelity panel** — KS mean, DCR ratio, TSTR PR-AUC per channel
- **Hardening dashboard** — ASR vs recall per round across channels
- **Zero-day callouts** — OOD-only detections highlighted separately
- **Policy distribution** — APPROVE / STEP_UP / HOLD / DECLINE breakdown per channel
- Auto-refreshes every 10 seconds

---

## Getting Started

### Local

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn lightgbm pyyaml joblib fastapi "uvicorn[standard]"

# 1. Generate all 31 attack simulations
python -m red.run_all

# 2. Train detectors, run hardening loop, compute fidelity
python -m blue.run_blue

# 3. Launch dashboard
uvicorn dashboard.app:app --host 0.0.0.0 --port 8080 --reload
```

### Docker

```bash
docker build -t raptor .
docker run -p 8080:8080 raptor
```

### Add a New Attack Vector

```bash
# 1. Add entry to attack_catalog.yaml
# 2. Generate boilerplate
python -m red.scaffold_generator V032
# 3. Implement generate() in red/generators/v032_*.py
# 4. Re-run — auto-discovery picks it up, no registration needed
python -m red.run_all && python -m blue.run_blue
```

---

## Repository Layout

```
.
├── attack_catalog.yaml              # source of truth: 31 attack vectors
├── run.sh                           # orchestration (--red / --ui / --all)
├── Dockerfile
│
├── red/                             # attack simulation (Red Team)
│   ├── envelope.py                  # uniform emission schema
│   ├── base_generator.py            # abstract base class
│   ├── run_all.py                   # auto-discovers + runs all generators
│   ├── scaffold_generator.py        # CLI: boilerplate for new vector
│   ├── fidelity.py                  # KS test, DCR ratio, TSTR benchmarks
│   └── generators/
│       ├── v001_structuring.py
│       └── … v031_agentic_token_replay.py
│
├── blue/                            # fraud detection (Blue Team)
│   ├── featurize.py                 # channel-specific feature engineering (70/24/22/7 cols)
│   ├── train.py                     # LightGBM + IsolationForest training
│   ├── evaluate.py                  # PR-AUC, ROC-AUC, per-vector recall/F1/precision
│   ├── loop.py                      # closed-loop adversarial hardening
│   ├── infer.py                     # live single-actor inference (dashboard backend)
│   ├── run_blue.py                  # pipeline entrypoint
│   └── results/
│       ├── summary.json             # full metrics (all channels)
│       ├── fidelity.json            # KS / DCR / TSTR per channel
│       ├── *_model.joblib           # persisted model artifacts
│       └── *_results.json           # per-channel detailed results
│
└── dashboard/
    └── app.py                       # FastAPI + embedded HTML/CSS/JS
```

---

## Team

| Name | Role | GitHub |
|---|---|---|
| Sai Sakthidar | Red team architecture, Blue stack, Dashboard, Hardening loop | [@SaiSakthidar](https://github.com/SaiSakthidar) |
| Sneha Samanta | Defense layer, Evaluation framework, Fidelity benchmarking | [@nthsneha](https://github.com/nthsneha) |

---

*Mastercard Innovation Challenge 2026 · Global Fintech Festival, Mumbai · 8–11 September 2026*
