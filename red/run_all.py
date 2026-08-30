"""
Run all red generators and write per-vector JSONL files to output/.
Each line in the JSONL is one actor's Envelope (serialized as JSON).

Generators are auto-discovered from red/generators/v*.py — no manual
registration needed.  Add a new vector to attack_catalog.yaml, scaffold
it with `python -m red.scaffold_generator`, implement the fraud pattern,
then re-run this script.

Usage:
    python -m red.run_all

Output:
    red/output/V001_structuring.jsonl  …  red/output/V0NN_xxx.jsonl
    red/output/ALL_vectors_flat.csv   (flattened event rows)
"""

import importlib
import inspect
import json
import yaml
import pandas as pd
from pathlib import Path

from red.base_generator import BaseGenerator

CATALOG_PATH = Path(__file__).parent.parent / "attack_catalog.yaml"
OUTPUT_DIR   = Path(__file__).parent / "output"
GEN_DIR      = Path(__file__).parent / "generators"


def discover_generators() -> tuple[dict, dict]:
    """
    Scan red/generators/v*.py, import each module, find the BaseGenerator
    subclass, and return (GENERATOR_MAP, FILENAMES) dicts keyed by vector_id.
    """
    generator_map: dict[str, type] = {}
    filenames: dict[str, str] = {}

    for path in sorted(GEN_DIR.glob("v*.py")):
        if path.name.startswith("__"):
            continue
        module_name = f"red.generators.{path.stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            print(f"  Warning: could not import {module_name}: {e}")
            continue

        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (issubclass(obj, BaseGenerator)
                    and obj is not BaseGenerator
                    and hasattr(obj, "VECTOR_ID")):
                vid = obj.VECTOR_ID
                # Derive JSONL filename from path stem: v001_foo → V001_foo
                stem_parts = path.stem.split("_", 1)
                jsonl_name = stem_parts[0].upper() + ("_" + stem_parts[1] if len(stem_parts) > 1 else "")
                generator_map[vid] = obj
                filenames[vid] = jsonl_name
                break

    return generator_map, filenames


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(CATALOG_PATH) as f:
        catalog = yaml.safe_load(f)

    params_by_id = {v["vector_id"]: v.get("generator_params", {})
                    for v in catalog["vectors"]}

    generator_map, filenames = discover_generators()
    print(f"Discovered {len(generator_map)} generators: {sorted(generator_map.keys())}\n")

    all_flat_rows = []
    summary_rows  = []

    for vid in sorted(generator_map.keys()):
        GenClass = generator_map[vid]
        params   = params_by_id.get(vid, {})
        gen      = GenClass(params=params, seed=42)
        envelopes = gen.generate()

        out_path = OUTPUT_DIR / f"{filenames[vid]}.jsonl"
        with open(out_path, "w") as f:
            for env in envelopes:
                f.write(env.to_json() + "\n")

        n_legit      = sum(1 for e in envelopes if e.label == 0)
        n_fraud      = sum(1 for e in envelopes if e.label == 1)
        total_events = sum(len(e.event_sequence) for e in envelopes)
        fraud_events = sum(
            sum(1 for ev in e.event_sequence if ev.get("label", e.label) == 1)
            for e in envelopes
        )

        for env in envelopes:
            for ev in env.event_sequence:
                all_flat_rows.append({
                    "vector_id":  env.vector_id,
                    "actor_id":   env.actor_id,
                    "channel":    env.channel,
                    "actor_label": env.label,
                    **ev,
                    **{f"ef_{k}": v for k, v in env.entity_features.items()},
                })

        summary_rows.append({
            "vector_id":      vid,
            "n_actors_legit": n_legit,
            "n_actors_fraud": n_fraud,
            "total_events":   total_events,
            "fraud_events":   fraud_events,
            "fraud_event_pct": round(fraud_events / max(total_events, 1) * 100, 1),
            "output_file":    out_path.name,
        })

        print(f"  {vid}  actors={len(envelopes):4d} "
              f"(legit={n_legit}, fraud={n_fraud})  "
              f"events={total_events:6d}  fraud_events={fraud_events}")

    flat_df  = pd.DataFrame(all_flat_rows)
    flat_path = OUTPUT_DIR / "ALL_vectors_flat.csv"
    flat_df.to_csv(flat_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    print("\n=== Generation Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nFlat CSV: {flat_path}  ({len(flat_df):,} rows)")


if __name__ == "__main__":
    main()
