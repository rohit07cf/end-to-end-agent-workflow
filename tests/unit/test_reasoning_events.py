import pytest

from app.agent.reasoning_events import EventSink


def test_emit_increments_step():
    s = EventSink("run-1")
    a = s.emit("agent", "agent_start", "starting")
    b = s.emit("agent", "agent_end", "done")
    assert a.step == 1 and b.step == 2
    assert len(s.events) == 2


def test_redacts_secrets():
    s = EventSink("run-1")
    ev = s.emit("agent", "tool_end", "result with api_key=sk-xyz")
    assert ev.decision_summary == "[redacted summary]"


def test_truncates_long_summaries():
    s = EventSink("run-1")
    long = "x" * 800
    ev = s.emit("agent", "tool_end", long)
    assert ev.decision_summary.endswith("…")
    assert len(ev.decision_summary) < 600


@pytest.mark.asyncio
async def test_stream_yields_then_closes():
    s = EventSink("run-2")
    s.emit("a", "agent_start", "go")
    s.close()
    out = [ev async for ev in s.stream()]
    assert len(out) == 1 and out[0].action_type == "agent_start"
