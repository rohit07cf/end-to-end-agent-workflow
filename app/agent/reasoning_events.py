"""Safe reasoning-event sink.

We intentionally never expose hidden chain-of-thought. Hooks call
`emit(...)` with an *action summary* — what happened, what was decided at a
high level, which tool was used, and how long/expensive it was. These events
are streamed to clients (SSE) and logged for offline replay.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from app.models import ReasoningEvent


class EventSink:
    """Per-run event sink. Buffers events and supports async streaming."""

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._step = 0
        self._events: list[ReasoningEvent] = []
        self._queue: asyncio.Queue[ReasoningEvent | None] = asyncio.Queue()
        self._closed = False

    @property
    def events(self) -> list[ReasoningEvent]:
        return list(self._events)

    @property
    def run_id(self) -> str:
        return self._run_id

    def emit(
        self,
        agent_name: str,
        action_type: str,
        decision_summary: str,
        *,
        tool_used: str | None = None,
        latency_ms: float | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ReasoningEvent:
        self._step += 1
        ev = ReasoningEvent(
            step=self._step,
            run_id=self._run_id,
            agent_name=agent_name,
            action_type=action_type,  # type: ignore[arg-type]
            decision_summary=_redact(decision_summary),
            tool_used=tool_used,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            extra=extra or {},
        )
        self._events.append(ev)
        if not self._closed:
            self._queue.put_nowait(ev)
        return ev

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(None)

    async def stream(self) -> AsyncIterator[ReasoningEvent]:
        while True:
            ev = await self._queue.get()
            if ev is None:
                return
            yield ev


_REDACT_SUBSTRINGS = ("api_key", "authorization", "bearer ", "secret")


def _redact(s: str) -> str:
    """Coarse redaction for accidental secrets. Production should plug in a
    proper PII/secret scrubber."""
    lower = s.lower()
    for needle in _REDACT_SUBSTRINGS:
        if needle in lower:
            return "[redacted summary]"
    if len(s) > 500:
        return s[:500] + "…"
    return s
