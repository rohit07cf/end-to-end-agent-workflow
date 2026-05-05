"""Pydantic schemas for the public API surface and the agent's structured
output. These are the types that cross trust boundaries (HTTP, eval, logs)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RunRequest(BaseModel):
    user_input: str = Field(..., min_length=1, max_length=8000)
    user_id: str | None = None
    conversation_id: str | None = None
    # Per-request override; falls back to settings if missing.
    max_cost_usd: float | None = None
    max_latency_ms: int | None = None
    max_tool_calls: int | None = None


class Citation(BaseModel):
    title: str
    source: str
    snippet: str | None = None


class FinalAnswer(BaseModel):
    """Strict structured output the agent must produce."""

    answer: str
    confidence: Literal["low", "medium", "high"]
    citations: list[Citation] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class ReasoningEvent(BaseModel):
    """Safe execution trace event. Never contains hidden chain-of-thought —
    only summarised actions/decisions for observability and SSE streaming."""

    step: int
    run_id: str
    agent_name: str
    action_type: Literal[
        "agent_start",
        "agent_end",
        "tool_start",
        "tool_end",
        "llm_start",
        "llm_end",
        "handoff",
        "reasoning_summary",
        "fallback",
        "budget_warning",
        "error",
    ]
    decision_summary: str
    tool_used: str | None = None
    latency_ms: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    timestamp: datetime = Field(default_factory=_now)
    extra: dict[str, Any] = Field(default_factory=dict)


class RunMetrics(BaseModel):
    run_id: str
    duration_ms: float
    tokens_in: int
    tokens_out: int
    estimated_cost_usd: float
    tool_calls: int
    fallback_used: bool
    model: str
    error: str | None = None


class RunResponse(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    final: FinalAnswer | None = None
    events: list[ReasoningEvent] = Field(default_factory=list)
    metrics: RunMetrics
    status: Literal["ok", "partial", "error"] = "ok"
