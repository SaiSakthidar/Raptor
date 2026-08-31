<div align="center">

# RAPTOR

### Real-time Adversarial Payment Threat Orchestrator

**Mastercard Innovation Challenge 2026 · Team Arintra · GFF Mumbai, 8–11 Sept 2026**

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)
![Vectors](https://img.shields.io/badge/Attack%20Vectors-31-red?style=flat-square)
![Channels](https://img.shields.io/badge/Channels-4-orange?style=flat-square)
![PR--AUC](https://img.shields.io/badge/PR--AUC-0.873-brightgreen?style=flat-square)
![ASR](https://img.shields.io/badge/Attack%20Success%20Rate-0%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

</div>

---

> Fraud used to be someone stealing your card details. GenAI changed the game completely.
>
> A CFO's voice cloned on Monday morning gets a ₹2 crore wire approved before lunch. A photorealistic passport passes KYC in four seconds. A single injected sentence into a merchant's AI checkout agent silently reroutes every payment. MFA push spam continues until exhaustion forces a tap-approve.
>
> The problem is not just that these attacks exist. It is that **nobody has actually built them and measured whether detection systems can catch them.** Everyone is defending against yesterday's fraud.
>
> **RAPTOR builds the attacks first.**

---

## At a Glance

<table>
<tr>
<td align="center"><b>31</b><br><sub>attack vectors</sub></td>
<td align="center"><b>4</b><br><sub>payment channels</sub></td>
<td align="center"><b>87.3%</b><br><sub>PR-AUC (txn-sequence)</sub></td>
<td align="center"><b>0.0%</b><br><sub>Attack Success Rate<br>after hardening</sub></td>
<td align="center"><b>0.104</b><br><sub>KS fidelity score<br>(lower = better)</sub></td>
</tr>
</table>

---

## Table of Contents

1. [Architecture](#architecture)
2. [Red Team — Attack Simulation](#red-team--attack-simulation)
3. [Blue Team — Defense Stack](#blue-team--defense-stack)
4. [Closed-Loop Hardening](#closed-loop-hardening)
5. [Threat Intelligence Ingest](#threat-intelligence-ingest)
6. [Data Fidelity](#data-fidelity)
7. [Attack Catalog](#attack-catalog)
8. [Results](#results)
9. [Live Dashboard](#live-dashboard)
10. [Getting Started](#getting-started)
11. [Repository Layout](#repository-layout)
12. [Team](#team)

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                         RAPTOR PLATFORM                              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌─────────────────────────────────────────────────────────────────┐ ║
║  │  RED TEAM   ·   31 attack generators across 4 channels          │ ║
║  │                                                                 │ ║
║  │   attack_catalog.yaml  ──►  red/generators/v001…v031.py        │ ║
║  │                                    │                            │ ║
║  │                         Uniform Envelope Schema                 │ ║
║  │                         (vector_id · actor_id · channel         │ ║
║  │                          label · events[] · entity_features)   │ ║
║  │                                    │                            │ ║
║  │                         red/output/*.jsonl                      │ ║
║  └──────────────────────────────┬──────────────────────────────────┘ ║
║                                 │  4 channel streams                 ║
║  ┌──────────────────────────────▼──────────────────────────────────┐ ║
║  │  BLUE TEAM  ·   3-layer detection stack                         │ ║
║  │                                                                 │ ║
║  │   Layer 1 ──  Cross-vector LightGBM        ← all 31 vectors    │ ║
║  │   Layer 2 ──  Per-vector specialists       ← FPR-gated         │ ║
║  │   Layer 3 ──  IsolationForest OOD          ← zero-day only     │ ║
║  │                      │                                          │ ║
║  │   score = 0.6 × supervised + 0.4 × OOD                         │ ║
║  │                      │                                          │ ║
║  │   Closed-loop hardening  (3 rounds · ASR 55% → 0%)             │ ║
║  │                      │                                          │ ║
║  │   4-way policy  APPROVE / STEP_UP / HOLD / DECLINE             │ ║
║  └──────────────────────────────┬──────────────────────────────────┘ ║
║                                 │                                    ║
║  ┌──────────────────────────────▼──────────────────────────────────┐ ║
║  │  DASHBOARD  ·  FastAPI + HTML/JS                                │ ║
║  │                                                                 │ ║
║  │   GET  /              Live threat intelligence report            │ ║
║  │   POST /api/simulate/{vector_id}   Proof-on-demand simulation   │ ║
║  │   GET  /api/results  /api/fidelity  /api/catalog                │ ║
║  │   POST /api/ingest   AI threat ingestion agent                  │ ║
║  └─────────────────────────────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Four Payment Channels

| Channel | Transport Layer | Vectors |
|---|---|---|
| `txn-sequence` | Card / UPI / Instant Rail | V001–V002, V005–V010, V012, V014–V022, V025, V028–V030 |
| `kyc-session` | Onboarding / KYC events | V003, V011, V013, V027 |
| `agent-payment` | AI agentic payment rail | V004, V023, V026, V031 |
| `chat-call` | Support chat / IVR | V024 |

---

## Red Team — Attack Simulation

### Uniform Emission Envelope

Every generator emits one schema regardless of attack type. The blue stack never needs to know which generator produced an envelope — new vectors plug in automatically.

```json
{
  "vector_id":        "V028",
  "actor_id":         "FRAUD_V028_0012",
  "channel":          "txn-sequence",
  "label":            1,
  "event_sequence":   [
    {"timestamp": "2026-07-28 09:14:33", "push_count_before_approval": 14,
     "response_time_seconds": 1.2, "amount": 82000, ...}
  ],
  "entity_features":  {"account_age_days": 342},
  "generation_params": {}
}
```

### Design Principles

| Principle | Implementation |
|---|---|
| Chronological split | Each vector gets its own 70/15/15 cut by its own timeline, then unioned. Every vector gets test-set representation. |
| Overlapping marginals | Legit and fraud overlap on individual features; only the **joint** pattern is the signal. Verified via KS scan across all 31 vectors. |
| Zero-day withholding | V011, V012, V013 labels fully withheld from supervised training — only the OOD layer may catch them. |
| Auto-discovery | `red/run_all.py` discovers generators via `importlib`/`inspect`. No manual registration. |

### Adding a New Vector

```bash
# 1. Scaffold boilerplate
python -m red.scaffold_generator V032

# 2. Implement generate() in red/generators/v032_your_attack.py
# 3. Auto-discovered on next run — no registration needed
python -m red.run_all && python -m blue.run_blue
```

---

## Blue Team — Defense Stack

### Layer 1 · Cross-Vector LightGBM

One gradient-boosted classifier per channel, trained on all known vectors.

| Channel | Feature count | Notable signals |
|---|---|---|
| `txn-sequence` | 70 | velocity, amount ratios, push-fatigue count, VPA age, pass-through ratio |
| `kyc-session` | 24 | liveness score, doc forgery score, BNPL utilization, credit inquiry rate |
| `agent-payment` | 22 | token age, token reuse count, session IP change, cart/checkout delta |
| `chat-call` | 7 | injection pattern match, sensitive action flag, authentication level |

### Layer 2 · Per-Vector Specialist Ensemble

One LightGBM per `txn-sequence` vector. Any specialist where legit FPR @ threshold > 0.5% is auto-rejected. Ensemble score = `max` across non-rejected specialists. Specialists learn the *idiosyncratic* signature of each attack on top of the cross-vector baseline.

### Layer 3 · IsolationForest OOD Detector

Trained on legit-only events. Calibration bounds (`cal_min`, `cal_max`) frozen at train time from the legit pool — the same absolute anomaly score maps to the same OOD probability at any inference time, independent of batch composition.

**Final score** = `0.6 × supervised + 0.4 × OOD`

### 4-Way Policy Engine

| Threshold band | Decision | Intent |
|---|---|---|
| ≥ τ | **DECLINE** | High-confidence fraud — block |
| ≥ 0.9 τ | **HOLD** | Human review queue |
| ≥ 0.7 τ | **STEP_UP** | Require MFA / step-up auth |
| < 0.7 τ | **APPROVE** | Clear |

Operating point τ = lowest threshold where validation FPR ≤ 1%.

---

## Closed-Loop Hardening

The red team attacks its own trained detector. Fraud feature vectors are nudged 30% toward the legit centroid — simulating an attacker who has reverse-engineered the decision boundary. Events that slip through become **hard negatives**. The detector retrains on the augmented corpus, threshold recalibrates to hold FPR ≤ 1%, and the loop repeats.

### txn-sequence Channel

| Round | Recall | ROC-AUC | PR-AUC | Attack Success Rate | Hard negatives |
|---|---|---|---|---|---|
| 0 · baseline | 86.1% | 0.9754 | 0.865 | 🔴 **55.1%** | — |
| 1 · hardened | 86.1% | 0.9858 | 0.885 | 🟢 **0.0%** | +787 |

**ASR dropped from 55.1% to zero in a single round** — without sacrificing recall or breaching the FPR budget.

> Traditional systems train once and evaluate on held-out data. That answers: *can we catch yesterday's attacks?*
> Closed-loop hardening answers: *can we catch the next mutation of each attack?*

---

## Threat Intelligence Ingest

Paste any free-text threat description — a news article, red-team note, or incident report. The agent:

1. **Extracts** structured attack metadata (name, channel, modality, key features) via LLM
2. **Deduplicates** against all 31 existing vectors (Jaccard keyword overlap; >35% = duplicate, halted)
3. **Assigns** the next vector ID and appends to `attack_catalog.yaml`
4. **Generates** a complete, runnable Python simulator via LLM (same `BaseGenerator` interface)
5. **Runs** the generator immediately and writes JSONL output
6. **Returns** the result live to the dashboard — new vector appears in the ledger within seconds

```bash
# CLI
python -m red.threat_ingest "Attackers are using compromised AI shopping agents 
to replay stale OAuth tokens across multiple merchant checkouts..."

# Dashboard
POST /api/ingest   {"text": "..."}
```

The new vector is auto-discovered on the next `python -m blue.run_blue` run — no manual registration.

---

## Data Fidelity

Three independent probes verify the synthetic attack data is realistic enough for the trained models to transfer to real-world patterns.

| Channel | KS mean ↓ | Realistic overlap | DCR ratio ↑ | TSTR PR-AUC |
|---|---|---|---|---|
| `txn-sequence` | **0.104** | 93.3% of features | 1.18× | 0.862 |
| `kyc-session` | 0.265 | 84.0% | 12.05× | 0.407 |
| `agent-payment` | 0.319 | 80.0% | 1.73× | 1.000 |
| `chat-call` | 0.448 | 71.4% | 18.22× | 1.000 |

**KS mean 0.104** on `txn-sequence` — fraud and legit are nearly indistinguishable feature-by-feature. The joint pattern across events is the only true signal, which is exactly how real fraud is structured.

**DCR ratio > 1** on all channels proves the joint distribution is genuinely separable even when marginals overlap. Ratio 12× on `kyc-session` means KYC fraud has strong internal consistency — a realistic synthetic family structure.

**Zero zero-overlap features** detected on `txn-sequence`, `kyc-session`, and `agent-payment`.

---

## Attack Catalog

31 vectors across 9 modalities and 4 channels. **`ZERO-DAY`** = labels withheld from supervised training.

### Transaction Sequence · 23 vectors

| ID | Name | Modality | Attack |
|---|---|---|---|
| V001 | Cash-out Structuring | TXN | LLM-generated micro-txns below reporting thresholds, timed to evade velocity rules |
| V002 | BEC App Fraud | MEDIA | Deepfake video + synthetic invoice redirects payroll wire |
| V005 | Pig Butchering | CONTEXT | Multi-week AI persona; gradual deposit escalation then single large withdrawal |
| V006 | Chargeback Fraud | PROCEDURAL | AI-generated dispute scripts at scale targeting refund policy gaps |
| V007 | BIN Enumeration | TXN | ML-guided card testing — minimal viable amounts, rotating BINs, GPU-speed |
| V008 | Mule Network | BENEFICIARY | AI chatbots mass-recruit mules via job ads; synthetic-identity accounts |
| V009 | Model Evasion | MODEL | Queries production model, reconstructs decision boundary, crafts blind-spot txns |
| V010 | UPI Account Takeover | ATO | SIM swap + deepfake liveness; UPI PIN reset via AI voice call to carrier |
| **V012** | **Synthetic Identity Bust-Out** | **TXN** | `ZERO-DAY` — credit-build phase then simultaneous bust-out across all lenders |
| V014 | SIM Swap + Instant Rail | ATO | Automated SIM swap followed by instant rail drain before OTP window closes |
| V015 | Vendor Invoice BEC | TXN | AI lookalike domain + forged invoice PDF; wire redirect at B2B payment step |
| V016 | Real-Estate Wire Fraud | TXN | LLM-crafted closing instruction spoof sent directly to title company |
| V017 | Grandparent Scam | MEDIA | AI voice clone of grandchild in distress; call center scales across thousands of targets |
| V018 | ML Money Laundering | TXN | RL agent routes funds through layering chains to minimise detection |
| V019 | Training Data Poisoning | PROCEDURAL | Dispute-labeled submissions poison the retraining feedback loop |
| V020 | Instant Rail APP Fraud | TXN | AI-crafted social engineering convinces victim to initiate their own transfer |
| V021 | Fake Investment Platform | TXN | AI-generated trading dashboard with fabricated returns; mass personalised phishing |
| V022 | Deepfake CEO Voice Auth | MEDIA | Real-time CFO voice clone instructs treasury to authorise emergency wire |
| V025 | Serial Return-Refund Ring | PROCEDURAL | Coordinated return fraud; AI optimises claim wording across merchant ring |
| V028 | MFA Push-Fatigue | ATO | Pushes MFA notifications until fatigue forces victim approval; immediate drain |
| V029 | Cuckoo Smurfing | BENEFICIARY | Criminal funds deposited into unwitting third-party accounts expecting legitimate remittance |
| V030 | UPI VPA Farm | TXN | Mass-created VPAs on payments banks; 4–7 hop chain completed in under 15 minutes |

### KYC Session · 4 vectors

| ID | Name | Modality | Attack |
|---|---|---|---|
| V003 | AI KYC Bypass | KYC | Generative video passes liveness; synthetic selfie + manipulated government ID |
| **V011** | **BNPL Ghost Identity** | **KYC** | `ZERO-DAY` — synthetic identity exploits BNPL onboarding; entity-level signals withheld |
| **V013** | **Voice IVR Vishing** | **MEDIA** | `ZERO-DAY` — deepfake voice defeats IVR biometrics; caller number spoofed via VoIP |
| V027 | AI-Forged KYC Document | KYC | Diffusion-generated government ID; font/seal metadata passes OCR and face-match |

### Agent Payment · 4 vectors

| ID | Name | Modality | Attack |
|---|---|---|---|
| V004 | Agentic Prompt Injection | AGENT | Hidden instructions in merchant product description hijack AI checkout agent |
| V023 | Agent Collusion Kickback | AGENT | Compromised merchant agent colludes with buyer agent; inflated checkout + kickback |
| V026 | Rogue Merchant Agent | AGENT | Rogue agent impersonates trusted brand; accepts payments for non-existent goods |
| V031 | Agentic Token Replay | AGENT | Replays expired OAuth session token after IP change to initiate high-value cart |

### Chat / Call · 1 vector

| ID | Name | Modality | Attack |
|---|---|---|---|
| V024 | Support Chatbot Injection | CONTEXT | Adversarial prompt in support session; chatbot exfiltrates account data |

---

## Results

### Channel-Level Performance · test set · FPR ≤ 1%

| Channel | Vectors | PR-AUC (90% CI) | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| `txn-sequence` | 23 | **0.873** [0.856–0.890] | 0.960 | 87.9% | 85.8% | 86.9% |
| `kyc-session` | 4 | 0.407 [0.321–0.493] | 0.579 | 81.3% | 19.4% | 31.3% |
| `agent-payment` | 4 | **1.000** [1.000–1.000] | 1.000 | 89.5% | 100.0% | 94.4% |
| `chat-call` | 1 | **1.000** [1.000–1.000] | 1.000 | 100.0% | 100.0% | 100.0% |

> `kyc-session` numbers are low by design, not by failure. Three of four KYC vectors are zero-day withholdings — the supervised classifier is intentionally blind to them. The anomaly layer provides the only defense.

### Zero-Day Detection · OOD layer · test set

| Vector | Attack | Supervised | Combined | OOD lift |
|---|---|---|---|---|
| V013 | Voice IVR Vishing | 0% | **20%** | +20pp |
| V012 | Synthetic Identity Bust-Out | 100%* | 100% | — |
| V011 | BNPL Ghost Identity | 0% | 0% | Known gap — signal lands on wrong event row |

*V012 shares feature space with structuring vectors — partial label overlap; V013's 20% is purely anomaly-layer detection.

### Business Impact · txn-sequence · test set

| Metric | Value |
|---|---|
| Fraud blocked | 85.8% recall |
| Expected fraud loss approved | ₹60.8L / $72.4K |
| Friction cost (legit incorrectly blocked) | ₹32.2L / $38.3K |
| Legit FPR at operating point | 0.93% |
| Attack Success Rate after hardening | **0.0%** |

---

## Live Dashboard

```
http://localhost:8080
```

Every number on the dashboard was computed by the actual running system — nothing is pre-baked.

| Section | What it shows |
|---|---|
| **Live Simulation** | Pick any vector, generate a fresh actor with a new random seed, watch the detector score each event live with an animated console |
| **Attack Ledger** | All 31 vectors, filterable by modality, searchable, expandable with description + per-vector precision/recall/F1 bars |
| **Zero-Day** | OOD-only detections highlighted, supervised vs combined recall delta |
| **Hardening** | ASR vs recall across rounds per channel |
| **Fidelity** | KS mean, DCR ratio, TSTR PR-AUC per channel |
| **Policy** | APPROVE / STEP_UP / HOLD / DECLINE breakdown |
| **Threat Ingest** | Paste any threat description, watch the agent generate a new live vector |

Auto-refreshes every 10 seconds.

---

## Getting Started

### Local

```bash
git clone https://github.com/SaiSakthidar/Raptor
cd Raptor

python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn lightgbm pyyaml joblib fastapi "uvicorn[standard]"

# Generate all 31 attack simulations
python -m red.run_all

# Train detectors · run hardening loop · compute fidelity
python -m blue.run_blue

# Launch dashboard
uvicorn dashboard.app:app --host 0.0.0.0 --port 8080 --reload
# → http://localhost:8080
```

### Docker

```bash
docker build -t raptor .
docker run -p 8080:8080 raptor
```

### Threat Intelligence Ingest (requires Anthropic API key)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn dashboard.app:app --host 0.0.0.0 --port 8080

# CLI ingest
python -m red.threat_ingest "Attackers are using AI-generated voice calls to..."
```

---

## Repository Layout

```
.
├── attack_catalog.yaml          # source of truth — 31 attack vectors, channels, signals
├── Dockerfile
├── entrypoint.sh
│
├── red/                         # Red Team — attack simulation
│   ├── envelope.py              # uniform emission schema
│   ├── base_generator.py        # abstract base class for all generators
│   ├── run_all.py               # auto-discovers and runs all generators
│   ├── scaffold_generator.py    # CLI: boilerplate for a new attack vector
│   ├── fidelity.py              # KS test, DCR ratio, TSTR benchmarks
│   ├── threat_ingest.py         # AI threat intelligence ingestion agent
│   └── generators/
│       ├── v001_structuring.py
│       ├── v002_bec_app_fraud.py
│       └── …  v031_agentic_token_replay.py
│
├── blue/                        # Blue Team — detection stack
│   ├── featurize.py             # channel feature engineering (70/24/22/7 cols)
│   ├── train.py                 # LightGBM + IsolationForest training
│   ├── evaluate.py              # PR-AUC, ROC-AUC, per-vector metrics
│   ├── loop.py                  # closed-loop adversarial hardening engine
│   ├── infer.py                 # live single-actor inference for dashboard
│   ├── run_blue.py              # full pipeline entrypoint
│   └── results/
│       ├── summary.json         # all channel metrics
│       ├── fidelity.json        # KS / DCR / TSTR per channel
│       └── *.joblib             # persisted model artifacts
│
└── dashboard/
    └── app.py                   # FastAPI + embedded HTML/CSS/JS — single file
```

---

## Why RAPTOR Wins

| Capability | RAPTOR | Typical system |
|---|---|---|
| Attack generation | 31 GenAI-native vectors built and running | Static historical datasets |
| Evaluation | Closed-loop: red team attacks its own detector | One-shot holdout |
| Zero-day coverage | IsolationForest OOD catches unseen attack families | Supervised only — zero-day = zero recall |
| Live proof | Generate a new synthetic actor on demand, score it live | Pre-computed numbers |
| Threat ingestion | LLM converts any article/note into a live simulator | Manual analyst work |
| Fidelity | KS, DCR, TSTR — three independent realism probes | None |

---

## Team

| Name | Contribution |
|---|---|
| **Sai Sakthidar** | Red team architecture · Blue stack · Hardening loop · Dashboard · Threat ingest agent |
| **Sneha Samanta** | Defense layer · Evaluation framework · Fidelity benchmarking |

---

<div align="center">

*Mastercard Innovation Challenge 2026 · Global Fintech Festival, Mumbai · 8–11 September 2026*

</div>
