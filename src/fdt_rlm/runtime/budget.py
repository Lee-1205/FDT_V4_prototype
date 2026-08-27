from __future__ import annotations

from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class BudgetConfig:
    max_steps: int = 8
    max_depth: int = 1
    max_subcalls: int = 3
    max_read_tokens: int = 4096
    max_generated_tokens: int = 1024
    max_repeated_action: int = 2
    max_parse_repairs: int = 2


@dataclass
class BudgetTracker:
    config: BudgetConfig
    steps: int = 0
    subcalls: int = 0
    read_tokens: int = 0
    generated_tokens: int = 0
    parse_repairs: int = 0

    def consume_step(self, generated_tokens: int = 0) -> None:
        self.steps += 1
        self.generated_tokens += max(int(generated_tokens), 0)
        self._check()

    def consume_read(self, tokens: int) -> None:
        self.read_tokens += max(int(tokens), 0)
        self._check()

    def consume_subcall(self, depth: int) -> None:
        if depth > self.config.max_depth:
            raise BudgetExceeded("maximum recursion depth exceeded")
        self.subcalls += 1
        self._check()

    def consume_repair(self) -> None:
        self.parse_repairs += 1
        self._check()

    def _check(self) -> None:
        if self.steps > self.config.max_steps:
            raise BudgetExceeded("maximum steps exceeded")
        if self.subcalls > self.config.max_subcalls:
            raise BudgetExceeded("maximum subcalls exceeded")
        if self.read_tokens > self.config.max_read_tokens:
            raise BudgetExceeded("read-token budget exceeded")
        if self.generated_tokens > self.config.max_generated_tokens:
            raise BudgetExceeded("generation-token budget exceeded")
        if self.parse_repairs > self.config.max_parse_repairs:
            raise BudgetExceeded("parse-repair budget exceeded")

    def remaining(self) -> dict[str, int]:
        return {
            "steps": self.config.max_steps - self.steps,
            "subcalls": self.config.max_subcalls - self.subcalls,
            "read_tokens": self.config.max_read_tokens - self.read_tokens,
            "generated_tokens": self.config.max_generated_tokens - self.generated_tokens,
            "parse_repairs": self.config.max_parse_repairs - self.parse_repairs,
        }
