"""
V024 — Support-Chatbot Prompt Injection
Channel : chat-call   (new channel: conversational session, turn-by-turn)
Modality: MODEL

Attacker embeds an instruction-override payload in a chat session with the
bank's own LLM support agent, coercing it into executing a sensitive action
(reset transfer limit / change beneficiary) without the MFA step that
action normally requires. Signature: injection pattern present + sensitive
action requested + insufficient auth level + the bot actually complies.
"""
import pandas as pd
import numpy as np
from red.base_generator import BaseGenerator
from red.envelope import Envelope


class V024Generator(BaseGenerator):
    VECTOR_ID = "V024"
    CHANNEL = "chat-call"

    def generate(self) -> list[Envelope]:
        p = self.params
        n_legit = p.get("n_legit", 200)
        n_fraud = p.get("n_fraud", 40)
        turns_lo, turns_hi = p.get("session_turns_range", [3, 12])

        envelopes = []
        base_ts = pd.Timestamp("2026-07-01")

        for i in range(n_legit):
            actor_id = self._actor_id("LEGIT_V024", i)
            age = int(self.rng.integers(30, 2000))
            n_turns = int(self.rng.integers(turns_lo, turns_hi + 1))
            session_ts = base_ts + pd.Timedelta(days=int(self.rng.integers(0, 30)),
                                                 hours=int(self.rng.integers(8, 22)))
            events = []
            for t in range(n_turns):
                ts = session_ts + pd.Timedelta(seconds=t * int(self.rng.integers(20, 90)))
                sensitive = self.rng.random() < 0.15
                auth = 2 if sensitive else int(self.rng.integers(0, 3))
                complied = 1 if (not sensitive or auth == 2) else 0
                events.append({
                    "timestamp": str(ts),
                    "turn_index": t,
                    "message_contains_injection_pattern": 0,
                    "requested_action_is_sensitive": int(sensitive),
                    "authentication_level": auth,
                    "chatbot_complied": complied,
                    "session_turn_count": n_turns,
                    "time_since_session_start_sec": t * 45,
                    "label": 0,
                })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=0,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={},
            ))

        for i in range(n_fraud):
            actor_id = self._actor_id("FRAUD_V024", i)
            age = int(self.rng.integers(30, 1500))
            n_turns = int(self.rng.integers(turns_lo, turns_hi + 1))
            # 75% of fraud sessions land in the train-friendly window so the
            # specialist has signal; 25% land in the test tail so recall is
            # actually measurable on held-out data.
            day_lo, day_hi = (5, 22) if self.rng.random() < 0.75 else (26, 30)
            session_ts = base_ts + pd.Timedelta(days=int(self.rng.integers(day_lo, day_hi)),
                                                 hours=int(self.rng.integers(9, 21)))
            events = []

            # Benign warm-up turns to establish a normal-looking session
            n_benign = max(1, n_turns - 1)
            for t in range(n_benign):
                ts = session_ts + pd.Timedelta(seconds=t * int(self.rng.integers(20, 90)))
                events.append({
                    "timestamp": str(ts),
                    "turn_index": t,
                    "message_contains_injection_pattern": 0,
                    "requested_action_is_sensitive": 0,
                    "authentication_level": int(self.rng.integers(0, 2)),
                    "chatbot_complied": 1,
                    "session_turn_count": n_turns,
                    "time_since_session_start_sec": t * 45,
                    "label": 0,
                })

            # The injection turn — bot bypasses MFA and executes the sensitive action
            inj_ts = session_ts + pd.Timedelta(seconds=n_benign * int(self.rng.integers(20, 90)))
            auth = int(self.rng.integers(0, 2))  # below full MFA
            events.append({
                "timestamp": str(inj_ts),
                "turn_index": n_benign,
                "message_contains_injection_pattern": 1,
                "requested_action_is_sensitive": 1,
                "authentication_level": auth,
                "chatbot_complied": 1,
                "session_turn_count": n_turns,
                "time_since_session_start_sec": n_benign * 45,
                "label": 1,
            })
            envelopes.append(Envelope(
                vector_id=self.VECTOR_ID, actor_id=actor_id,
                channel=self.CHANNEL, label=1,
                event_sequence=sorted(events, key=lambda e: e["timestamp"]),
                entity_features={"account_age_days": age},
                generation_params={"auth_level_at_injection": auth},
            ))
        return envelopes
