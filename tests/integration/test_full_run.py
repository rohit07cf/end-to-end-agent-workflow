"""Full agent run via the OpenAI Agents SDK Runner with the mock model."""

import pytest

from app.agent.workflow import ResearchWorkflow
from app.models import ReasoningEvent, RunRequest, RunResponse


@pytest.mark.asyncio
async def test_full_run_returns_structured_answer():
    wf = ResearchWorkflow()
    resp = await wf.run(RunRequest(user_input="What is the capital of France?"))
    assert resp.status == "ok"
    assert resp.final is not None
    assert "Paris" in resp.final.answer
    assert resp.final.confidence == "high"
    # we should have at least an agent_start, llm_start, llm_end, agent_end
    kinds = {ev.action_type for ev in resp.events}
    assert {"agent_start", "agent_end", "llm_start", "llm_end"} <= kinds


@pytest.mark.asyncio
async def test_full_run_metrics_populated():
    wf = ResearchWorkflow()
    resp = await wf.run(RunRequest(user_input="What is the capital of France?"))
    assert resp.metrics.tokens_in > 0
    assert resp.metrics.tokens_out > 0
    assert resp.metrics.duration_ms > 0
    assert resp.metrics.model == "mock"
    assert resp.metrics.fallback_used is False


@pytest.mark.asyncio
async def test_streaming_emits_events_and_final():
    wf = ResearchWorkflow()
    events: list[ReasoningEvent] = []
    final: RunResponse | None = None
    async for item in wf.run_stream(RunRequest(user_input="What is the capital of France?")):
        if isinstance(item, ReasoningEvent):
            events.append(item)
        elif isinstance(item, RunResponse):
            final = item
    assert len(events) >= 4
    assert final is not None and final.status == "ok"
