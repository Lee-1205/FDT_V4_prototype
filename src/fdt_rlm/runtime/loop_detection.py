from __future__ import annotations

from collections import Counter

from .actions import Action


class LoopDetector:
    def __init__(self, max_repeated_action: int = 2):
        self.max_repeated_action = max_repeated_action
        self.counts: Counter[str] = Counter()

    def observe(self, action: Action) -> bool:
        key = action.fingerprint()
        self.counts[key] += 1
        return self.counts[key] > self.max_repeated_action

    def reset(self) -> None:
        self.counts.clear()
