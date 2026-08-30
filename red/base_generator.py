"""
Abstract base class all red generators implement.
"""

from abc import ABC, abstractmethod
import numpy as np
from red.envelope import Envelope


class BaseGenerator(ABC):
    def __init__(self, params: dict, seed: int = 42):
        self.params = params
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def generate(self) -> list[Envelope]:
        """Return a mixed list of legit (label=0) and fraud (label=1) envelopes."""
        ...

    def _actor_id(self, prefix: str, idx: int) -> str:
        return f"{prefix}_{idx:05d}"
