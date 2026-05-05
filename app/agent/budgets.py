"""Per-run budgets and enforcement.

Budgets are *soft* during a turn (we let the in-flight LLM/tool call complete)
and *hard* between turns: when a budget is breached the runner stops the agent
and triggers the fallback router.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from app.agent.cost import estimate_cost
from app.config import Settings, get_settings

BudgetReason = Literal[
    "tokens", "cost", "latency", "tool_calls", "ok"
]


@dataclass
class BudgetState:
    """Live counters updated by hooks during a run."""

    started_at: float = field(default_factory=time.monotonic)
    tokens_in: int = 0
    tokens_out: int = 0
    tool_calls: int = 0
    model: str = ""

    @property
    def latency_ms(self) -> float:
        return (time.monotonic() - self.started_at) * 1000

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def cost_usd(self, settings: Settings | None = None) -> float:
        if not self.model:
            return 0.0
        return estimate_cost(self.model, self.tokens_in, self.tokens_out, settings).total_usd


@dataclass
class Budgets:
    max_total_tokens: int
    max_cost_usd: float
    max_latency_ms: int
    max_tool_calls: int

    @classmethod
    def from_settings(cls, settings: Settings | None = None, **overrides: float | int) -> Budgets:
        s = settings or get_settings()
        return cls(
            max_total_tokens=int(overrides.get("max_total_tokens", s.max_total_tokens_per_run)),
            max_cost_usd=float(overrides.get("max_cost_usd", s.max_cost_usd_per_run)),
            max_latency_ms=int(overrides.get("max_latency_ms", s.max_latency_ms_per_run)),
            max_tool_calls=int(overrides.get("max_tool_calls", s.max_tool_calls_per_run)),
        )

    def check(self, state: BudgetState, settings: Settings | None = None) -> BudgetReason:
        if state.total_tokens > self.max_total_tokens:
            return "tokens"
        if state.cost_usd(settings) > self.max_cost_usd:
            return "cost"
        if state.latency_ms > self.max_latency_ms:
            return "latency"
        if state.tool_calls > self.max_tool_calls:
            return "tool_calls"
        return "ok"
