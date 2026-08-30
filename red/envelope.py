"""
Standard emission envelope — every red generator returns a list of these.
The channel adapter (featurize step) consumes this format, so it never needs
to know which generator produced it.
"""

from dataclasses import dataclass, field, asdict
import json
from typing import Any


@dataclass
class Envelope:
    vector_id: str           # e.g. "V001"
    actor_id: str            # unique account/actor ID for this sample
    channel: str             # txn-sequence | agent-payment | kyc-session | chat-call
    label: int               # 0 = legit, 1 = fraud
    event_sequence: list[dict] = field(default_factory=list)
    entity_features: dict[str, Any] = field(default_factory=dict)
    generation_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)
