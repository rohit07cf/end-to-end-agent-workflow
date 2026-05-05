"""Hook unit tests — invoke handlers directly with synthetic context."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent.budgets import BudgetState, Budgets
from app.agent.hooks import (
    HookContext,
    ReasoningSummaryHooks,
    ResearchRunHooks,
    emit_error,
    emit_fallback,
)
from app.agent.reasoning_events import EventSink


def _ctx_objs():
    sink = EventSink("run-1")
    state = BudgetState(model="claude-haiku-4-5-20251001")
    budgets = Budgets(max_total_tokens=1000, max_cost_usd=1.0, max_latency_ms=10000, max_tool_calls=5)
    hc = HookContext(run_id="run-1", sink=sink, budgets=budgets, state=state)
    wrapper = SimpleNamespace(context=hc)
    return wrapper, sink, state


@pytest.mark.asyncio
async def test_run_hook_agent_start_emits():
    wrapper, sink, _ = _ctx_objs()
    h = ResearchRunHooks()
    agent = SimpleNamespace(name="ResearchAssistant")
    await h.on_agent_start(wrapper, agent)
    assert any(ev.action_type == "agent_start" for ev in sink.events)


@pytest.mark.asyncio
async def test_run_hook_llm_end_accumulates_tokens():
    wrapper, sink, state = _ctx_objs()
    h = ResearchRunHooks()
    agent = SimpleNamespace(name="A")
    await h.on_llm_start(wrapper, agent, "sys", [])
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=20))
    await h.on_llm_end(wrapper, agent, response)
    assert state.tokens_in == 10 and state.tokens_out == 20
    kinds = [ev.action_type for ev in sink.events]
    assert "llm_start" in kinds and "llm_end" in kinds


@pytest.mark.asyncio
async def test_run_hook_tool_increments_count():
    wrapper, sink, state = _ctx_objs()
    h = ResearchRunHooks()
    agent = SimpleNamespace(name="A")
    tool = SimpleNamespace(name="web_search")
    await h.on_tool_start(wrapper, agent, tool)
    await h.on_tool_end(wrapper, agent, tool, "result text")
    assert state.tool_calls == 1
    assert any(ev.action_type == "tool_start" for ev in sink.events)
    assert any(ev.action_type == "tool_end" for ev in sink.events)


@pytest.mark.asyncio
async def test_agent_hooks_emit_reasoning_summary():
    wrapper, sink, _ = _ctx_objs()
    h = ReasoningSummaryHooks()
    agent = SimpleNamespace(name="A")
    await h.on_start(wrapper, agent)
    await h.on_end(wrapper, agent, output=None)
    summaries = [ev for ev in sink.events if ev.action_type == "reasoning_summary"]
    assert len(summaries) == 2


@pytest.mark.asyncio
async def test_agent_hooks_warn_when_near_budget():
    wrapper, sink, state = _ctx_objs()
    state.tokens_in = 800  # 80% of 1000
    h = ReasoningSummaryHooks()
    agent = SimpleNamespace(name="A")
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=0, output_tokens=0))
    await h.on_llm_end(wrapper, agent, response)
    assert any(ev.action_type == "budget_warning" for ev in sink.events)


def test_emit_error_and_fallback():
    sink = EventSink("r")
    emit_error(sink, "A", RuntimeError("boom"))
    emit_fallback(sink, "A", "partial_answer", "budget")
    kinds = [ev.action_type for ev in sink.events]
    assert "error" in kinds and "fallback" in kinds
