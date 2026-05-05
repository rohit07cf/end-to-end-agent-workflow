"""Fallback router. Maps error / budget conditions to a recovery strategy.

The contract: callers raise an `AgentRunError` (or the workflow detects a
budget breach), then call `decide_fallback(...)` to pick a strategy. The
workflow executes the strategy and emits a `fallback` reasoning event.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from app.models import FinalAnswer

FailureKind = Literal[
    "model_timeout",
    "model_rate_limit",
    "tool_failure",
    "invalid_structured_output",
    "budget_exceeded",
    "tracing_export_failure",
    "eval_regression",
    "unknown",
]


class Strategy(str, Enum):
    RETRY_BACKOFF = "retry_backoff"
    SIMPLER_MODEL = "simpler_model"
    NO_TOOL_ANSWER = "no_tool_answer"
    CACHED_ANSWER = "cached_answer"
    PARTIAL_ANSWER = "partial_answer"
    HUMAN_REVIEW = "human_review"
    LOG_AND_CONTINUE = "log_and_continue"


@dataclass
class FallbackDecision:
    strategy: Strategy
    detail: str


class AgentRunError(RuntimeError):
    """Raised by the workflow to signal a recoverable failure to the router."""

    def __init__(self, kind: FailureKind, message: str = "") -> None:
        super().__init__(message or kind)
        self.kind = kind


def decide_fallback(kind: FailureKind, *, attempt: int = 1) -> FallbackDecision:
    """Pure routing logic — easy to unit-test."""
    if kind == "model_timeout":
        if attempt < 3:
            return FallbackDecision(Strategy.RETRY_BACKOFF, "transient model timeout, retry")
        return FallbackDecision(Strategy.SIMPLER_MODEL, "exhausted retries, downgrade")
    if kind == "model_rate_limit":
        if attempt < 4:
            return FallbackDecision(Strategy.RETRY_BACKOFF, "rate limited, backoff")
        return FallbackDecision(Strategy.SIMPLER_MODEL, "still rate limited, downgrade")
    if kind == "tool_failure":
        return FallbackDecision(Strategy.NO_TOOL_ANSWER, "answer without the failing tool")
    if kind == "invalid_structured_output":
        if attempt < 2:
            return FallbackDecision(Strategy.RETRY_BACKOFF, "reparse / retry once")
        return FallbackDecision(Strategy.HUMAN_REVIEW, "model unable to comply with schema")
    if kind == "budget_exceeded":
        return FallbackDecision(Strategy.PARTIAL_ANSWER, "return what we have, mark partial")
    if kind == "tracing_export_failure":
        return FallbackDecision(Strategy.LOG_AND_CONTINUE, "trace export failing, keep serving")
    if kind == "eval_regression":
        return FallbackDecision(Strategy.HUMAN_REVIEW, "regression detected — block deploy")
    return FallbackDecision(Strategy.PARTIAL_ANSWER, f"unhandled failure: {kind}")


def safe_partial_answer(reason: str) -> FinalAnswer:
    """Canonical partial answer — used when budget exceeded or model gives up."""
    return FinalAnswer(
        answer=(
            "I wasn't able to fully complete this request within the configured "
            "budget. Here is a partial answer based on what I was able to gather. "
            "Please retry, narrow the question, or escalate for human review."
        ),
        confidence="low",
        citations=[],
        caveats=[f"partial: {reason}", "human-review recommended"],
    )


def human_review_answer(reason: str) -> FinalAnswer:
    return FinalAnswer(
        answer=(
            "This request requires human review before answering. "
            "An operator has been notified."
        ),
        confidence="low",
        citations=[],
        caveats=[f"human-review: {reason}"],
    )
