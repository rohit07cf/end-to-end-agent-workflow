"""Custom RunHooks + AgentHooks.

These hooks have one job: convert OpenAI Agents SDK lifecycle callbacks into
*safe, structured* reasoning events, plus update budget counters and metrics.

Crucially, we do NOT log raw model output / hidden CoT — we log a short
``decision_summary`` (what action class happened, which tool was used, how
big it was). Anything that could leak sensitive content goes through the
event sink's redactor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from agents import Agent
from agents.items import ModelResponse, TResponseInputItem
from agents.lifecycle import AgentHooks, RunHooks
from agents.run_context import AgentHookContext, RunContextWrapper
from agents.tool import Tool

from app.agent.budgets import BudgetState, Budgets
from app.agent.reasoning_events import EventSink
from app.observability.metrics import (
    AGENT_RUNS_TOTAL,
    ERROR_COUNT,
    FALLBACK_COUNT,
    LATENCY_MS,
    MODEL_LATENCY_MS,
    TOKENS_IN,
    TOKENS_OUT,
    TOOL_CALL_COUNT,
    TOOL_LATENCY_MS,
)


@dataclass
class HookContext:
    """The context object passed to Runner.run as `context=`."""

    run_id: str
    sink: EventSink
    budgets: Budgets
    state: BudgetState
    user_id: str | None = None
    conversation_id: str | None = None
    # Per-call timing scratch:
    _llm_started_at: float | None = None
    _tool_started_at: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self._tool_started_at = {}


class ResearchRunHooks(RunHooks):
    """Workflow-level hooks: token accounting, metrics, fallback signals."""

    async def on_agent_start(
        self, context: AgentHookContext[HookContext], agent: Agent[HookContext]
    ) -> None:
        ctx = context.context
        AGENT_RUNS_TOTAL.labels(agent=agent.name).inc()
        ctx.sink.emit(agent.name, "agent_start", f"agent {agent.name} starting")

    async def on_agent_end(
        self, context: AgentHookContext[HookContext], agent: Agent[HookContext], output: Any
    ) -> None:
        ctx = context.context
        LATENCY_MS.labels(agent=agent.name).observe(ctx.state.latency_ms)
        ctx.sink.emit(
            agent.name,
            "agent_end",
            f"agent {agent.name} produced final output",
            latency_ms=ctx.state.latency_ms,
            tokens_in=ctx.state.tokens_in,
            tokens_out=ctx.state.tokens_out,
            extra={"output_type": type(output).__name__},
        )

    async def on_llm_start(
        self,
        context: RunContextWrapper[HookContext],
        agent: Agent[HookContext],
        system_prompt: str | None,
        input_items: list[TResponseInputItem],
    ) -> None:
        ctx = context.context
        ctx._llm_started_at = time.monotonic()
        ctx.sink.emit(agent.name, "llm_start", f"calling model ({len(input_items)} input items)")

    async def on_llm_end(
        self,
        context: RunContextWrapper[HookContext],
        agent: Agent[HookContext],
        response: ModelResponse,
    ) -> None:
        ctx = context.context
        latency = (
            (time.monotonic() - ctx._llm_started_at) * 1000 if ctx._llm_started_at else None
        )
        usage = response.usage
        if usage:
            ctx.state.tokens_in += int(getattr(usage, "input_tokens", 0) or 0)
            ctx.state.tokens_out += int(getattr(usage, "output_tokens", 0) or 0)
            TOKENS_IN.labels(agent=agent.name).inc(getattr(usage, "input_tokens", 0) or 0)
            TOKENS_OUT.labels(agent=agent.name).inc(getattr(usage, "output_tokens", 0) or 0)
        if latency is not None:
            MODEL_LATENCY_MS.labels(agent=agent.name).observe(latency)
        ctx.sink.emit(
            agent.name,
            "llm_end",
            "model returned response",
            latency_ms=latency,
            tokens_in=getattr(usage, "input_tokens", 0) if usage else None,
            tokens_out=getattr(usage, "output_tokens", 0) if usage else None,
        )

    async def on_tool_start(
        self,
        context: RunContextWrapper[HookContext],
        agent: Agent[HookContext],
        tool: Tool,
    ) -> None:
        ctx = context.context
        ctx.state.tool_calls += 1
        TOOL_CALL_COUNT.labels(agent=agent.name, tool=tool.name).inc()
        assert ctx._tool_started_at is not None
        ctx._tool_started_at[tool.name] = time.monotonic()
        ctx.sink.emit(
            agent.name,
            "tool_start",
            f"calling tool {tool.name}",
            tool_used=tool.name,
        )

    async def on_tool_end(
        self,
        context: RunContextWrapper[HookContext],
        agent: Agent[HookContext],
        tool: Tool,
        result: str,
    ) -> None:
        ctx = context.context
        assert ctx._tool_started_at is not None
        started = ctx._tool_started_at.pop(tool.name, None)
        latency = (time.monotonic() - started) * 1000 if started else None
        if latency is not None:
            TOOL_LATENCY_MS.labels(agent=agent.name, tool=tool.name).observe(latency)
        # Summarise — never include the full tool output verbatim in the trace
        # for safety; we cap to first 200 chars.
        summary = (result or "")[:200]
        ctx.sink.emit(
            agent.name,
            "tool_end",
            f"tool {tool.name} returned: {summary}",
            tool_used=tool.name,
            latency_ms=latency,
        )

    async def on_handoff(
        self,
        context: RunContextWrapper[HookContext],
        from_agent: Agent[HookContext],
        to_agent: Agent[HookContext],
    ) -> None:
        ctx = context.context
        ctx.sink.emit(
            from_agent.name,
            "handoff",
            f"handing off from {from_agent.name} → {to_agent.name}",
            extra={"to": to_agent.name},
        )


class ReasoningSummaryHooks(AgentHooks):
    """Per-agent hooks. Emits one ``reasoning_summary`` event per LLM call so
    a streaming UI gets a high-level beat for every major decision step
    without exposing raw chain-of-thought.

    Also runs the per-turn budget check and emits a ``budget_warning``
    event when nearing thresholds (≥80%)."""

    def __init__(self, planning_label: str = "Researcher") -> None:
        self.planning_label = planning_label

    async def on_start(
        self, context: AgentHookContext[HookContext], agent: Agent[HookContext]
    ) -> None:
        ctx = context.context
        ctx.sink.emit(
            agent.name,
            "reasoning_summary",
            f"{self.planning_label}: planning approach to user request",
        )

    async def on_llm_end(
        self,
        context: RunContextWrapper[HookContext],
        agent: Agent[HookContext],
        response: ModelResponse,
    ) -> None:
        ctx = context.context
        # Soft budget warning. Hard enforcement happens in workflow loop.
        usage = response.usage
        cost = ctx.state.cost_usd()
        b = ctx.budgets
        warns = []
        if usage and ctx.state.total_tokens / max(1, b.max_total_tokens) >= 0.8:
            warns.append(f"tokens {ctx.state.total_tokens}/{b.max_total_tokens}")
        if cost / max(1e-9, b.max_cost_usd) >= 0.8:
            warns.append(f"cost ${cost:.4f}/${b.max_cost_usd:.4f}")
        if ctx.state.latency_ms / max(1, b.max_latency_ms) >= 0.8:
            warns.append(f"latency {ctx.state.latency_ms:.0f}/{b.max_latency_ms}ms")
        if warns:
            ctx.sink.emit(
                agent.name,
                "budget_warning",
                "approaching budget: " + ", ".join(warns),
            )

    async def on_tool_start(
        self,
        context: RunContextWrapper[HookContext],
        agent: Agent[HookContext],
        tool: Tool,
    ) -> None:
        ctx = context.context
        ctx.sink.emit(
            agent.name,
            "reasoning_summary",
            f"{self.planning_label}: chose tool '{tool.name}' to advance the task",
            tool_used=tool.name,
        )

    async def on_end(
        self, context: AgentHookContext[HookContext], agent: Agent[HookContext], output: Any
    ) -> None:
        ctx = context.context
        ctx.sink.emit(
            agent.name,
            "reasoning_summary",
            f"{self.planning_label}: produced final structured answer",
        )


def emit_error(sink: EventSink, agent_name: str, err: BaseException) -> None:
    ERROR_COUNT.labels(agent=agent_name, kind=type(err).__name__).inc()
    sink.emit(agent_name, "error", f"{type(err).__name__}: {err}"[:500])


def emit_fallback(sink: EventSink, agent_name: str, kind: str, detail: str = "") -> None:
    FALLBACK_COUNT.labels(agent=agent_name, kind=kind).inc()
    sink.emit(agent_name, "fallback", f"fallback({kind}): {detail}".strip())
