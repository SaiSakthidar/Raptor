"""
Threat Intelligence Ingestion Agent.

Give it any free-text description of a new attack (news article, threat intel
report, internal incident note). It will:

  1. Extract structured attack metadata via Claude
  2. Deduplicate against existing attack_catalog.yaml
  3. If novel: assign next vector ID, append to catalog, generate a working
     Python generator via Claude, write the JSONL output
  4. Return a status dict the dashboard can render immediately

Usage (CLI):
    python -m red.threat_ingest "Attackers are using deepfake video calls..."

Usage (API):
    from red.threat_ingest import ingest
    result = ingest("...")
"""

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml

CATALOG_PATH = Path(__file__).parent.parent / "attack_catalog.yaml"
GENERATORS_DIR = Path(__file__).parent / "generators"
OUTPUT_DIR = Path(__file__).parent / "output"
CHANNELS = ["txn-sequence", "kyc-session", "agent-payment", "chat-call"]
MODALITIES = ["TXN", "BENEFICIARY", "KYC", "MEDIA", "AGENT", "MODEL",
              "CONTEXT", "ATO", "PROCEDURAL"]


def _llm(prompt: str, system: str = "") -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    msgs = [{"role": "user", "content": prompt}]
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system or "You are a payment fraud expert and Python engineer.",
        messages=msgs,
    )
    return resp.content[0].text.strip()


def _load_catalog() -> list[dict]:
    return yaml.safe_load(CATALOG_PATH.read_text()).get("vectors", [])


def _next_vector_id(vectors: list[dict]) -> str:
    ids = [int(v["vector_id"][1:]) for v in vectors if v.get("vector_id", "").startswith("V")]
    return f"V{max(ids) + 1:03d}"


def _dedup_check(description: str, vectors: list[dict]) -> dict | None:
    """Simple keyword overlap dedup — fast, no embeddings needed."""
    desc_words = set(re.findall(r'\b\w{4,}\b', description.lower()))
    best_score, best_match = 0.0, None
    for v in vectors:
        existing = (v.get("name", "") + " " + v.get("description", "")).lower()
        ex_words = set(re.findall(r'\b\w{4,}\b', existing))
        if not ex_words:
            continue
        overlap = len(desc_words & ex_words) / len(desc_words | ex_words)
        if overlap > best_score:
            best_score, best_match = overlap, v
    if best_score > 0.35:
        return best_match
    return None


def _extract_metadata(text: str) -> dict:
    prompt = f"""Extract attack metadata from this threat description and return ONLY valid JSON.

Description:
{text}

Return JSON with exactly these keys:
{{
  "name": "short attack name (3-6 words)",
  "modality": "one of: TXN, BENEFICIARY, KYC, MEDIA, AGENT, MODEL, CONTEXT, ATO, PROCEDURAL",
  "channel": "one of: txn-sequence, kyc-session, agent-payment, chat-call",
  "observed_by": ["list", "of", "issuer/acquirer/network"],
  "description": "2-3 sentence description of how the attack works",
  "expected_signal": "key features that would detect this attack (comma separated)",
  "key_features": ["feature_name_1", "feature_name_2", "feature_name_3"]
}}

Return ONLY the JSON object, no markdown."""
    raw = _llm(prompt)
    raw = re.sub(r"^```(?:json)?\s*", "", raw).rstrip("` \n")
    return json.loads(raw)


def _generate_code(vector_id: str, meta: dict) -> str:
    class_name = "".join(w.title() for w in re.split(r'[\s_\-]+', meta["name"])) + "Generator"
    filename_stem = f"{vector_id.lower()}_{re.sub(r'[^a-z0-9]+', '_', meta['name'].lower()).strip('_')}"

    prompt = f"""Write a complete Python generator class for this payment fraud attack vector.

Vector ID: {vector_id}
Name: {meta['name']}
Channel: {meta['channel']}
Description: {meta['description']}
Expected signals: {meta['expected_signal']}
Key features to include: {meta.get('key_features', [])}

Write the FULL Python file. Follow this exact pattern:

```python
\"\"\"
{vector_id} — {meta['name']}
Signal: <one line describing the key signal>
\"\"\"
import numpy as np
import pandas as pd
from red.base_generator import BaseGenerator
from red.envelope import Envelope

class {class_name}(BaseGenerator):
    VECTOR_ID = "{vector_id}"
    CHANNEL = "{meta['channel']}"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        base_ts = pd.Timestamp("2026-07-01")
        envelopes = []

        # --- Legit actors ---
        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_{vector_id}", i)
            age = int(self.rng.integers(30, 2000))
            events = []
            for _ in range(int(self.rng.integers(8, 25))):
                day = int(self.rng.integers(0, 25))
                ts = base_ts + pd.Timedelta(days=day, hours=int(self.rng.integers(7,22)), minutes=int(self.rng.integers(0,60)))
                events.append({{
                    "timestamp": str(ts),
                    # ADD FEATURE FIELDS HERE — legit values (overlapping with fraud on marginals)
                    "label": 0,
                }})
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id, channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={{"account_age_days": age}},
                generation_params={{}},
            ))

        # --- Fraud actors ---
        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_{vector_id}", i)
            age = int(self.rng.integers(30, 1500))
            events = []
            # Cover transactions (normal-looking)
            for _ in range(int(self.rng.integers(3, 10))):
                day = int(self.rng.integers(0, 22))
                ts = base_ts + pd.Timedelta(days=day, hours=int(self.rng.integers(7,22)), minutes=int(self.rng.integers(0,60)))
                events.append({{
                    "timestamp": str(ts),
                    # ADD FEATURE FIELDS HERE — legit-looking values
                    "label": 0,
                }})
            # Attack event (days 26-29)
            attack_day = int(self.rng.integers(26, 30))
            ts_attack = base_ts + pd.Timedelta(days=attack_day, hours=int(self.rng.integers(9,21)))
            events.append({{
                "timestamp": str(ts_attack),
                # ADD FEATURE FIELDS HERE — fraud signal values
                "label": 1,
            }})
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id, channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={{"account_age_days": age}},
                generation_params={{}},
            ))

        return envelopes
```

Fill in ALL feature fields with realistic numpy-generated values.
Legit and fraud must OVERLAP on marginal distributions — the joint pattern is the signal.
Return ONLY the Python code, no markdown fences."""

    code = _llm(prompt)
    code = re.sub(r"^```(?:python)?\s*", "", code).rstrip("` \n")
    filename_stem = f"{vector_id.lower()}_{re.sub(r'[^a-z0-9]+', '_', meta['name'].lower()).strip('_')}"
    return code, filename_stem


def ingest(text: str) -> dict:
    """
    Main entry point. Returns a status dict:
      {status: "added"|"duplicate"|"error", vector_id, name, message, jsonl_rows}
    """
    try:
        vectors = _load_catalog()

        # 1. Extract metadata
        print("  [ingest] extracting metadata...")
        meta = _extract_metadata(text)

        # 2. Dedup check
        print("  [ingest] checking for duplicates...")
        dup = _dedup_check(meta["description"] + " " + meta["name"], vectors)
        if dup:
            return {
                "status": "duplicate",
                "vector_id": dup["vector_id"],
                "name": dup["name"],
                "message": f"Too similar to existing vector {dup['vector_id']} — {dup['name']}",
                "jsonl_rows": 0,
            }

        # 3. Assign ID and append to catalog
        vector_id = _next_vector_id(vectors)
        print(f"  [ingest] new vector → {vector_id}: {meta['name']}")

        catalog_entry = {
            "vector_id": vector_id,
            "name": meta["name"],
            "modality": meta.get("modality", "TXN"),
            "channel": meta.get("channel", "txn-sequence"),
            "observed_by": meta.get("observed_by", ["issuer"]),
            "description": meta["description"],
            "expected_signal": meta.get("expected_signal", ""),
            "generator_params": {"n_legit": 200, "n_fraud": 40},
        }

        # Append to YAML
        raw_yaml = CATALOG_PATH.read_text()
        entry_yaml = "\n" + textwrap.indent(
            yaml.dump([catalog_entry], default_flow_style=False, allow_unicode=True,
                      sort_keys=False).strip(),
            "  "
        )
        CATALOG_PATH.write_text(raw_yaml.rstrip() + "\n" + entry_yaml + "\n")

        # 4. Generate Python code
        print("  [ingest] generating code...")
        code, filename_stem = _generate_code(vector_id, meta)
        gen_path = GENERATORS_DIR / f"{filename_stem}.py"
        gen_path.write_text(code)
        print(f"  [ingest] wrote {gen_path.name}")

        # 5. Run generator
        print("  [ingest] generating attack data...")
        result = subprocess.run(
            [sys.executable, "-m", "red.run_all", "--only", vector_id],
            capture_output=True, text=True,
            cwd=CATALOG_PATH.parent,
        )
        # Fallback: run generator directly if --only flag not supported
        if result.returncode != 0:
            import importlib.util
            spec = importlib.util.spec_from_file_location("gen_module", gen_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cls = next(v for k, v in vars(mod).items()
                       if isinstance(v, type) and hasattr(v, "VECTOR_ID") and v.VECTOR_ID == vector_id)
            gen = cls(seed=42, params={"n_legit": 200, "n_fraud": 40})
            envelopes = gen.generate()
            out_path = OUTPUT_DIR / f"{vector_id}_{filename_stem.split('_', 1)[1]}.jsonl"
            with open(out_path, "w") as f:
                for env in envelopes:
                    f.write(env.to_json() + "\n")
            jsonl_rows = len(envelopes)
        else:
            jsonl_rows = result.stdout.count(vector_id)

        return {
            "status": "added",
            "vector_id": vector_id,
            "name": meta["name"],
            "channel": meta["channel"],
            "modality": meta["modality"],
            "message": f"Added {vector_id} — {meta['name']}. Generator written. {jsonl_rows} envelopes generated.",
            "jsonl_rows": jsonl_rows,
            "catalog_entry": catalog_entry,
        }

    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or input("Paste threat description: ")
    result = ingest(text)
    print(json.dumps(result, indent=2))
