"""Structured JSON logging.

Every log line carries `trace_id`, `span_id`, `run_id`, `user_id`,
`conversation_id` if set in the contextvars. Use `bind_run_context(...)`
at the entry of an agent run; it returns a token to reset later.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from typing import Any

import structlog

from app.config import get_settings

_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
_user_id: ContextVar[str | None] = ContextVar("user_id", default=None)
_conversation_id: ContextVar[str | None] = ContextVar("conversation_id", default=None)
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_span_id: ContextVar[str | None] = ContextVar("span_id", default=None)

_configured = False


def _ctx_processor(_, __, event_dict: dict[str, Any]) -> dict[str, Any]:
    for key, var in (
        ("run_id", _run_id),
        ("user_id", _user_id),
        ("conversation_id", _conversation_id),
        ("trace_id", _trace_id),
        ("span_id", _span_id),
    ):
        v = var.get()
        if v is not None:
            event_dict.setdefault(key, v)
    return event_dict


def configure_logging() -> None:
    """Idempotent. Call once at process startup."""
    global _configured
    if _configured:
        return
    s = get_settings()
    level = getattr(logging, s.log_level.upper(), logging.INFO)

    logging.basicConfig(format="%(message)s", level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _ctx_processor,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None):
    configure_logging()
    return structlog.get_logger(name)


class RunContextTokens:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens

    def reset(self) -> None:
        for t in reversed(self._tokens):
            try:
                # Each token belongs to its own ContextVar; we can't tell which
                # without storing pairs, but resetting via the var that produced
                # it is what's needed. We stored (var, token) pairs:
                pass
            except LookupError:  # pragma: no cover
                pass


def bind_run_context(
    *,
    run_id: str | None = None,
    user_id: str | None = None,
    conversation_id: str | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
) -> list[tuple[ContextVar, Token]]:
    """Bind values to contextvars; caller passes the returned list to
    `unbind_run_context` to release them."""
    pairs: list[tuple[ContextVar, Token]] = []
    for var, val in (
        (_run_id, run_id),
        (_user_id, user_id),
        (_conversation_id, conversation_id),
        (_trace_id, trace_id),
        (_span_id, span_id),
    ):
        if val is not None:
            pairs.append((var, var.set(val)))
    return pairs


def unbind_run_context(pairs: list[tuple[ContextVar, Token]]) -> None:
    for var, token in reversed(pairs):
        try:
            var.reset(token)
        except (LookupError, ValueError):  # pragma: no cover
            pass
