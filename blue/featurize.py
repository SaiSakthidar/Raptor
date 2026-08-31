"""
Channel adapters: JSONL envelopes → per-channel feature DataFrames.
One DataFrame per channel type (txn-sequence, kyc-session, agent-payment).
"""

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "red" / "output"
CATALOG_PATH = Path(__file__).parent.parent / "attack_catalog.yaml"

# ── feature columns per channel ────────────────────────────────────────────

TXN_FEATURES = [
    "amount", "amount_to_threshold_ratio",
    "txn_count_last_1hr", "sum_amount_last_1hr",
    "same_mcc_count_last_1hr", "time_since_prev_txn_sec",
    "is_new_beneficiary", "beneficiary_tenure_days", "urgency_score",
    "contact_age_days", "is_investment_platform",
    "prior_transfer_count", "amount_escalation_ratio",
    "dispute_text_perplexity", "dispute_text_sentence_len_variance",
    "has_ai_evidence_image", "dispute_rate_30d",
    "is_online",
    "inbound_sender_count_7d", "outbound_recipient_count_7d",
    "avg_dwell_hours", "retail_txn_ratio", "pass_through_ratio",
    "is_new_vpa", "device_changed_flag",
    "amount_vs_30d_avg_ratio", "upi_transfers_15min",
    "beneficiary_is_new", "call_precedes_transfer_minutes",
    "amount_vs_30d_max_ratio", "hour_of_day", "prior_large_transfer_count",
    "ef_account_age_days",
    # V014 – SIM-swap ATO
    "sim_swap_preceding_transfer_hours", "otp_source_changed", "device_is_new",
    # V015 – Vendor Invoice BEC
    "vendor_tenure_days", "bank_details_changed", "request_via_email_only", "phone_confirmation",
    # V016 – Real-estate wire
    "first_large_outbound_ever", "call_precedes_wire_minutes", "amount_in_realestate_range",
    "days_since_prior_large_transfer",
    # V017 – Grandparent scam
    "call_from_unknown_number", "transfer_follows_call_minutes",
    # V018 – ML laundering
    "recipients_7d", "amount_pct_of_threshold", "scheduling_regularity", "total_volume_7d",
    # V019 – Label poisoning
    "dispute_false_claim_rate", "dispute_timing_similarity_score",
    "dispute_success_rate", "actor_cluster_size", "n_disputes_30d",
    # V020 – Instant rail APP
    "prior_instant_rail_transfers", "contact_to_transfer_minutes",
    # V021 – Fake investment platform
    "beneficiary_type_is_platform", "platform_registration_age_days",
    "n_transfers_to_platform",
    # V022 – Deepfake CEO voice authorisation
    "inbound_call_from_external_number", "call_to_wire_hours",
    "no_email_approval_trail",
    # V025 – Serial return/refund fraud ring
    "item_used_days_before_return", "refund_claim_text_similarity_score",
    "ai_generated_evidence_score", "n_refund_claims_30d",
    "distinct_merchants_targeted_30d",
    # V028 – MFA push-fatigue
    "push_count_before_approval", "response_time_seconds", "approval_attempt_number",
    # V029 – Cuckoo smurfing
    "inbound_unknown_sender_pct", "time_inbound_to_outbound_hours",
    # V030 – UPI VPA farm hop laundering
    "vpa_age_hours", "vpa_count_per_device",
]

KYC_FEATURES = [
    "doc_age_days", "liveness_score", "selfie_consistency_score",
    "device_is_emulator", "ip_is_vpn", "doc_metadata_age_variance",
    "session_duration_sec",
    "voice_auth_confidence", "caller_number_known", "device_is_known",
    "account_change_flag", "post_auth_transfer_amount",
    "credit_inquiry_count_7d", "bnpl_merchant_count_7d",
    "account_age_at_purchase_days", "credit_utilization",
    "ef_account_age_days",
    # V011 – identity-thinness signal, repeated onto every event row via
    # entity_features so it survives on the label==1 rows that get scored
    "ef_doc_age_days", "ef_credit_inquiry_count_7d", "ef_prior_txn_count",
    "total_bnpl_30d",
    # V027 – AI-forged KYC document
    "doc_forgery_score", "doc_security_feature_match",
    "doc_issuance_registry_match", "selfie_doc_face_match_score",
]

AGENT_FEATURES = [
    "cart_amount", "checkout_amount",
    "cart_checkout_delta", "cart_checkout_delta_pct",
    "n_line_items", "hidden_line_items",
    "agent_verified", "merchant_age_days", "merchant_dispute_rate",
    # V023 – Agent-to-agent collusion / kickback fraud
    "price_vs_market_ratio", "agent_negotiation_rounds",
    "hidden_rebate_amount", "rebate_to_non_customer_wallet",
    "merchant_agent_reputation_score",
    # V026 – Rogue merchant agent impersonation
    "merchant_identity_verified", "settlement_account_mismatch",
    "merchant_agent_first_seen_hours",
    # V031 – Agentic token replay
    "token_age_seconds", "token_reuse_count", "session_ip_changed",
]

CHAT_CALL_FEATURES = [
    # V024 – Support-chatbot prompt injection
    "turn_index", "message_contains_injection_pattern",
    "requested_action_is_sensitive", "authentication_level",
    "chatbot_complied", "session_turn_count",
    "time_since_session_start_sec",
]

CHANNEL_FEATURES = {
    "txn-sequence": TXN_FEATURES,
    "kyc-session": KYC_FEATURES,
    "agent-payment": AGENT_FEATURES,
    "chat-call": CHAT_CALL_FEATURES,
}

FILENAMES = {
    "V001": "V001_structuring",
    "V002": "V002_bec_app_fraud",
    "V003": "V003_kyc_bypass",
    "V004": "V004_agent_injection",
    "V005": "V005_pig_butchering",
    "V006": "V006_chargeback_fraud",
    "V007": "V007_bin_attack",
    "V008": "V008_mule_network",
    "V009": "V009_model_evasion",
    "V010": "V010_upi_ato",
    "V011": "V011_bnpl_fraud",
    "V012": "V012_digital_arrest",
    "V013": "V013_voice_ivr",
    "V014": "V014_sim_swap",
    "V015": "V015_vendor_invoice_bec",
    "V016": "V016_realestate_wire",
    "V017": "V017_grandparent_scam",
    "V018": "V018_ml_laundering",
    "V019": "V019_label_poisoning",
    "V020": "V020_instant_rail_app",
    "V021": "V021_fake_investment_platform",
}


def _discover_filenames() -> dict[str, str]:
    """Auto-discover JSONL filenames from red/generators/v*.py VECTOR_ID attrs."""
    import importlib, inspect
    from red.base_generator import BaseGenerator
    gen_dir = OUTPUT_DIR.parent / "generators"
    filenames: dict[str, str] = {}
    for path in sorted(gen_dir.glob("v*.py")):
        if path.name.startswith("__"):
            continue
        try:
            mod = importlib.import_module(f"red.generators.{path.stem}")
        except Exception:
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, BaseGenerator) and obj is not BaseGenerator and hasattr(obj, "VECTOR_ID"):
                parts = path.stem.split("_", 1)
                filenames[obj.VECTOR_ID] = parts[0].upper() + ("_" + parts[1] if len(parts) > 1 else "")
                break
    return filenames


def _load_catalog() -> dict:
    with open(CATALOG_PATH) as f:
        return yaml.safe_load(f)


def _load_envelopes(vector_id: str, filenames: dict | None = None) -> list[dict]:
    fm = filenames or FILENAMES
    path = OUTPUT_DIR / f"{fm[vector_id]}.jsonl"
    envelopes = []
    with open(path) as f:
        for line in f:
            envelopes.append(json.loads(line.strip()))
    return envelopes


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 1-hour rolling window features per actor for txn-sequence."""
    df = df.copy()
    # Ensure timestamp is parsed
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values(["actor_id", "timestamp"]).reset_index(drop=True)

    txn_counts, sum_amounts, same_mccs, time_prev = [], [], [], []

    for _, grp in df.groupby("actor_id", sort=False):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        ts = grp["timestamp"].tolist()
        amounts = grp["amount"].fillna(0).tolist()
        mccs = grp["mcc"].tolist() if "mcc" in grp.columns else [None] * len(grp)

        for i in range(len(grp)):
            w_start = ts[i] - pd.Timedelta(hours=1)
            widx = [j for j in range(i + 1) if ts[j] > w_start]
            txn_counts.append(len(widx))
            sum_amounts.append(sum(amounts[j] for j in widx))
            same_mccs.append(
                sum(1 for j in widx if mccs[j] == mccs[i]) if mccs[i] else 0
            )
            time_prev.append(
                (ts[i] - ts[i - 1]).total_seconds() if i > 0 else 999_999
            )

    df["txn_count_last_1hr"] = txn_counts
    df["sum_amount_last_1hr"] = sum_amounts
    df["same_mcc_count_last_1hr"] = same_mccs
    df["time_since_prev_txn_sec"] = time_prev
    return df


def envelopes_to_df(envelopes: list[dict], channel: str) -> pd.DataFrame:
    """
    Expand a list of envelope dicts (from disk OR freshly generated in memory
    by a live simulation) into a flat per-event DataFrame with rolling
    features computed. This is the shared core so live inference scores
    exactly the same feature representation the models were trained on.
    """
    all_rows = []
    for env in envelopes:
        ef = {f"ef_{k}": v for k, v in env["entity_features"].items()}
        for ev in env["event_sequence"]:
            row = {
                "vector_id": env["vector_id"],
                "actor_id": env["actor_id"],
                "channel": channel,
                "actor_label": env["label"],
                **ev,
                **ef,
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

    # Normalise the label column: some event dicts use "label", some don't
    if "label" not in df.columns:
        df["label"] = df["actor_label"]

    # V006 uses "dispute_amount" — map to "amount" so rolling features work
    if "dispute_amount" in df.columns and "amount" not in df.columns:
        df["amount"] = df["dispute_amount"]

    # agent-payment vectors use "checkout_amount" — map to "amount" so
    # expected_loss / friction_cost have a $ figure to work with
    if "checkout_amount" in df.columns and "amount" not in df.columns:
        df["amount"] = df["checkout_amount"]

    # Rolling features only for txn-sequence (and only where amount exists)
    if channel == "txn-sequence" and "amount" in df.columns:
        df = _add_rolling_features(df)

    # Derive hour_of_day if not already present
    if "hour_of_day" not in df.columns:
        df["hour_of_day"] = df["timestamp"].dt.hour

    return df.sort_values("timestamp").reset_index(drop=True)


def load_channel(channel: str) -> pd.DataFrame:
    """
    Load all envelopes for the given channel from red/output/*.jsonl and
    hand off to envelopes_to_df for feature expansion.
    """
    catalog = _load_catalog()
    vector_ids = [v["vector_id"] for v in catalog["vectors"]
                  if v["channel"] == channel]

    # Use auto-discovered filenames if available, fall back to static FILENAMES
    try:
        fm = _discover_filenames()
    except Exception:
        fm = FILENAMES

    envelopes = []
    for vid in vector_ids:
        if vid not in fm:
            continue
        envelopes.extend(_load_envelopes(vid, fm))

    return envelopes_to_df(envelopes, channel)


def get_feature_matrix(df: pd.DataFrame, channel: str):
    """
    Returns (X: np.ndarray, y: np.ndarray, meta: DataFrame).
    meta keeps actor_id, vector_id, timestamp, label for eval.
    All missing feature values are filled with 0.
    """
    feat_cols = CHANNEL_FEATURES[channel]
    existing = [c for c in feat_cols if c in df.columns]
    missing = [c for c in feat_cols if c not in df.columns]

    X_df = df[existing].copy()
    for col in missing:
        X_df[col] = 0.0

    X_df = X_df[feat_cols].fillna(0).astype(float)
    y = df["label"].fillna(0).astype(int).values
    meta_cols = ["actor_id", "vector_id", "timestamp", "label", "actor_label", "channel"]
    if "amount" in df.columns:
        meta_cols.append("amount")
    meta = df[meta_cols].copy()
    return X_df.values, y, meta, feat_cols
