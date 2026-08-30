"""
Generator scaffold tool.

Reads attack_catalog.yaml and creates a runnable generator file for any
vector that doesn't have one yet.  The scaffold produces legit actors with
random features and empty fraud actors — you fill in the fraud pattern.

Usage:
    python -m red.scaffold_generator           # scaffold all missing vectors
    python -m red.scaffold_generator V022      # scaffold a specific vector
    python -m red.scaffold_generator --list    # show which vectors are missing

The scaffold is immediately runnable: `python -m red.run_all` will pick it
up automatically without any edits to run_all.py.
"""

import re
import sys
import yaml
from pathlib import Path

CATALOG_PATH = Path(__file__).parent.parent / "attack_catalog.yaml"
GENERATORS_DIR = Path(__file__).parent / "generators"

# Feature sets per channel — we fill legit events with zeros for unknown cols
CHANNEL_DEFAULT_FEATURES: dict[str, list[str]] = {
    "txn-sequence": ["amount", "beneficiary_is_new", "amount_vs_30d_avg_ratio"],
    "kyc-session":  ["liveness_score", "selfie_consistency_score", "doc_age_days"],
    "agent-payment": ["cart_amount", "checkout_amount", "cart_checkout_delta"],
}


# ── template ────────────────────────────────────────────────────────────────

_TEMPLATE = '''\
"""
{vector_id} — {name}
Channel : {channel}
Modality: {modality}

{description}

Signal (from catalog):
{expected_signal}

TODO: implement the fraud pattern in the n_fraud loop below.
      The legit loop is pre-filled and runnable as-is.
"""
import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class {class_name}(BaseGenerator):
    VECTOR_ID = "{vector_id}"
    CHANNEL   = "{channel}"

    def generate(self) -> list[Envelope]:
        p      = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud",  40)
{param_lines}
        envelopes = []
        base_ts   = pd.Timestamp("2026-07-01")

        # ── legit actors ─────────────────────────────────────────────
        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_{vector_id}", i)
            age      = int(self.rng.integers(180, 2500))
            events   = []
            for _ in range(int(self.rng.integers(8, 25))):
                ts     = base_ts + pd.Timedelta(
                    days =int(self.rng.integers(0, 30)),
                    hours=int(self.rng.integers(8, 22)),
                )
                amount = float(np.round(self.rng.lognormal(9.0, 0.8), 2))
                events.append({{
                    "timestamp": str(ts),
                    "amount":    amount,
{legit_feature_lines}                    "label": 0,
                }})
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL,     label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={{"account_age_days": age}},
                generation_params={{}},
            ))

        # ── fraud actors — TODO: implement signal ────────────────────
        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_{vector_id}", i)
            age      = int(self.rng.integers(30, 1000))
            events   = []

            # ----------------------------------------------------------------
            # FRAUD PATTERN — replace this block with the real signal.
            # Refer to the catalog expected_signal above for guidance.
            # ----------------------------------------------------------------
            fraud_ts  = base_ts + pd.Timedelta(
                days =int(self.rng.integers(26, 30)),
                hours=int(self.rng.integers(9, 21)),
            )
            fraud_amt = float(np.round(self.rng.uniform(1000, 50000), 2))
            events.append({{
                "timestamp": str(fraud_ts),
                "amount":    fraud_amt,
{fraud_feature_lines}                "label": 1,
            }})
            # ----------------------------------------------------------------

            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL,     label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={{"account_age_days": age}},
                generation_params={{}},
            ))

        return envelopes
'''


# ── helpers ──────────────────────────────────────────────────────────────────

def _vector_id_to_class(vector_id: str, name: str) -> str:
    """V022 + 'Foo Bar Baz' → V022Generator"""
    return f"V{vector_id[1:]}Generator"


def _vector_id_to_filename(vector_id: str, name: str) -> str:
    """V022 + 'Foo Bar Baz' → v022_foo_bar_baz"""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"{vector_id.lower()}_{slug}"


def _existing_vector_ids() -> set[str]:
    """Return set of vector IDs that already have a generator file."""
    ids = set()
    for path in GENERATORS_DIR.glob("v*.py"):
        if path.name == "__init__.py":
            continue
        # Extract vector ID from filename: v001_xxx → V001
        parts = path.stem.split("_", 1)
        ids.add(parts[0].upper())
    return ids


def _indent(text: str, spaces: int = 8) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in text.splitlines())


def _param_lines(generator_params: dict) -> str:
    """Generate param extraction lines from generator_params dict."""
    if not generator_params:
        return ""
    lines = []
    skip = {"n_legit", "n_fraud"}
    for k, v in generator_params.items():
        if k in skip:
            continue
        if isinstance(v, list) and len(v) == 2:
            lo, hi = v
            lines.append(f'        {k}_lo, {k}_hi = p.get("{k}", [{lo}, {hi}])')
        else:
            lines.append(f'        {k} = p.get("{k}", {repr(v)})')
    return "\n".join(lines) + ("\n" if lines else "")


def _feature_lines(channel: str, indent: int = 20) -> tuple[str, str]:
    """Return (legit_lines, fraud_lines) for default channel features."""
    pad = " " * indent
    feats = CHANNEL_DEFAULT_FEATURES.get(channel, [])
    legit_lines = "".join(f'{pad}"{f}": 0.0,\n' for f in feats)
    fraud_lines = "".join(f'{pad}"{f}": 0.0,  # TODO\n' for f in feats)
    return legit_lines, fraud_lines


def scaffold(vector: dict, dry_run: bool = False) -> Path:
    """Generate a scaffold file for one vector. Returns the output path."""
    vid  = vector["vector_id"]
    name = vector["name"]
    fname = _vector_id_to_filename(vid, name)
    out_path = GENERATORS_DIR / f"{fname}.py"

    legit_lines, fraud_lines = _feature_lines(vector.get("channel", "txn-sequence"))

    content = _TEMPLATE.format(
        vector_id=vid,
        name=name,
        channel=vector.get("channel", "txn-sequence"),
        modality=vector.get("modality", "TXN"),
        description=_indent(vector.get("description", "").strip(), 0),
        expected_signal=_indent(vector.get("expected_signal", "").strip(), 0),
        class_name=_vector_id_to_class(vid, name),
        param_lines=_param_lines(vector.get("generator_params", {})),
        legit_feature_lines=legit_lines,
        fraud_feature_lines=fraud_lines,
    )

    if dry_run:
        print(f"  [dry-run] would write → {out_path}")
    else:
        out_path.write_text(content)
        print(f"  Wrote → {out_path}")

    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    with open(CATALOG_PATH) as f:
        catalog = yaml.safe_load(f)

    vectors = catalog["vectors"]
    existing = _existing_vector_ids()

    args = sys.argv[1:]

    if "--list" in args:
        print("Vectors with generators:")
        for v in vectors:
            status = "✓" if v["vector_id"] in existing else "✗ MISSING"
            print(f"  {v['vector_id']:6s} {status:10s} {v['name']}")
        return

    if args and not args[0].startswith("--"):
        # Specific vector ID(s) requested
        requested = {a.upper() for a in args}
        targets = [v for v in vectors if v["vector_id"] in requested]
        if not targets:
            print(f"No vectors found for: {args}")
            sys.exit(1)
    else:
        # Default: all vectors missing a generator
        targets = [v for v in vectors if v["vector_id"] not in existing]

    if not targets:
        print("All vectors already have generators. Nothing to scaffold.")
        return

    print(f"Scaffolding {len(targets)} generator(s)...")
    for v in targets:
        scaffold(v)

    print("\nDone. Next steps for each scaffolded file:")
    print("  1. Open the file and implement the fraud pattern in the # FRAUD PATTERN block")
    print("  2. Run `python -m red.run_all` — the new generator is picked up automatically")


if __name__ == "__main__":
    main()
