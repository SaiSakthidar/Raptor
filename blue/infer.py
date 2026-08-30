"""
Live inference — the "prove it" path.

Generates exactly ONE brand-new synthetic actor for a given attack vector
using the real red-team generator (fresh random seed, never seen during
training), scores it through the persisted trained models, and returns a
verdict. Nothing here is pre-computed: every call produces a different
actor and re-runs the actual classifier + OOD forest.

Requires `python -m blue.run_blue` to have been run at least once, since
that is what writes the model artifacts this module loads.
"""
import importlib
import inspect
import random
from pathlib import Path

import joblib
import yaml

from red.base_generator import BaseGenerator
from blue.featurize import envelopes_to_df, get_feature_matrix
from blue.train import combined_score_ensemble, combined_score
from blue.evaluate import apply_policy

CATALOG_PATH = Path(__file__).parent.parent / "attack_catalog.yaml"
ARTIFACTS_DIR = Path(__file__).parent / "results"
GEN_DIR = Path(__file__).parent.parent / "red" / "generators"

_artifact_cache: dict[str, dict] = {}
_generator_cache: dict[str, type] = {}
_catalog_cache: dict | None = None


def _load_catalog() -> dict:
    global _catalog_cache
    if _catalog_cache is None:
        with open(CATALOG_PATH) as f:
            _catalog_cache = yaml.safe_load(f)
    return _catalog_cache


def _discover_generators() -> dict[str, type]:
    if _generator_cache:
        return _generator_cache
    for path in sorted(GEN_DIR.glob("v*.py")):
        if path.name.startswith("__"):
            continue
        try:
            mod = importlib.import_module(f"red.generators.{path.stem}")
        except Exception:
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (issubclass(obj, BaseGenerator) and obj is not BaseGenerator
                    and hasattr(obj, "VECTOR_ID")):
                _generator_cache[obj.VECTOR_ID] = obj
                break
    return _generator_cache


def _load_artifacts(channel: str) -> dict:
    if channel not in _artifact_cache:
        path = ARTIFACTS_DIR / f"{channel.replace('-', '_')}_model.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model artifacts for channel '{channel}'. "
                f"Run `python -m blue.run_blue` first."
            )
        _artifact_cache[channel] = joblib.load(path)
    return _artifact_cache[channel]


def available_vectors() -> list[dict]:
    """Vectors that have both a catalog entry and an implemented generator."""
    catalog = _load_catalog()
    known = _discover_generators()
    return [v for v in catalog["vectors"] if v["vector_id"] in known]


def simulate(vector_id: str) -> dict:
    """
    Generate one fresh actor for vector_id and score it live. Raises
    KeyError for an unknown vector, FileNotFoundError if the channel's
    model hasn't been trained yet.
    """
    catalog = _load_catalog()
    vector = next((v for v in catalog["vectors"] if v["vector_id"] == vector_id), None)
    if vector is None:
        raise KeyError(f"Unknown vector_id: {vector_id}")

    generators = _discover_generators()
    if vector_id not in generators:
        raise KeyError(f"No generator implemented for {vector_id}")

    channel = vector["channel"]
    artifacts = _load_artifacts(channel)

    # Fresh seed every call — a genuinely new actor, not a replay.
    params = dict(vector.get("generator_params", {}))
    params["n_legit"] = 0
    params["n_fraud"] = 1
    seed = random.randint(0, 2_000_000_000)
    gen = generators[vector_id](params=params, seed=seed)
    raw_envelopes = gen.generate()
    if not raw_envelopes:
        raise RuntimeError(f"Generator for {vector_id} produced no actor this run")
    envelope = raw_envelopes[0].to_dict()

    df = envelopes_to_df([envelope], channel)
    X, y, meta, feat_cols = get_feature_matrix(df, channel)

    clf = artifacts["clf"]
    vector_clfs = artifacts.get("vector_clfs") or {}
    iso = artifacts["iso"]
    op_threshold = artifacts["op_threshold"]

    if vector_clfs:
        clf_scores, combined_scores = combined_score_ensemble(clf, vector_clfs, iso, X)
    else:
        clf_scores = clf.predict_proba(X)[:, 1]
        combined_scores = combined_score(clf, iso, X)

    policies = apply_policy(combined_scores, op_threshold)

    events = []
    for i, ev in enumerate(envelope["event_sequence"]):
        events.append({
            "timestamp": str(ev.get("timestamp")),
            "amount": ev.get("amount"),
            "label": int(ev.get("label", envelope["label"])),
            "clf_score": round(float(clf_scores[i]), 4),
            "combined_score": round(float(combined_scores[i]), 4),
            "policy": policies[i],
        })

    max_score = float(combined_scores.max()) if len(combined_scores) else 0.0
    caught = bool(max_score >= op_threshold)
    max_idx = int(combined_scores.argmax()) if len(combined_scores) else 0
    caught_by = None
    if caught:
        caught_by = "supervised" if clf_scores[max_idx] >= op_threshold else "anomaly-layer"

    return {
        "vector_id": vector_id,
        "name": vector["name"],
        "modality": vector["modality"],
        "channel": channel,
        "description": vector["description"],
        "actor_id": envelope["actor_id"],
        "seed": seed,
        "events": events,
        "max_score": round(max_score, 4),
        "operating_threshold": round(float(op_threshold), 4),
        "caught": caught,
        "caught_by": caught_by,
        "used_specialist": vector_id in vector_clfs,
    }
