"""Failure-mode integration tests."""

from __future__ import annotations

import pytest

from app.agent.fallback import (
    Strategy,
    decide_fallback,
    safe_partial_answer,
)
from app.agent.providers import MockModel
from app.agent.workflow import ResearchWorkflow
from app.config import get_settings
from app.models import RunRequest


@pytest.mark.asyncio
async def test_model_error_triggers_fallback(monkeypatch):
    """If the model raises every time, the workflow should still return a
    RunResponse with status != 'ok' and fallback_used=True."""

    from app.agent import providers, workflow

    def _broken(settings=None, model_name=None):
        return providers.ProviderHandle(
            model=MockModel(name="mock", force_error=True), name="mock", is_mock=True
        )

    monkeypatch.setattr(workflow, "build_model", _broken)

    wf = ResearchWorkflow()
    resp = await wf.run(RunRequest(user_input="anything"))
    assert resp.status in ("partial", "error")
    assert resp.metrics.fallback_used is True
    # final must still be a valid FinalAnswer (so callers can render something)
    assert resp.final is not None
    assert resp.final.confidence == "low"


@pytest.mark.asyncio
async def test_budget_exceeded_returns_partial(monkeypatch):
    """Setting cost budget to nearly zero should breach after the LLM call and
    coerce the response into a partial answer."""

    s = get_settings()
    monkeypatch.setattr(s, "max_cost_usd_per_run", 0.0)
    # Token budget low enough that any positive consumption breaches.
    monkeypatch.setattr(s, "max_total_tokens_per_run", 0)
    # Force pricing to non-zero for "mock" so the cost guard triggers.
    new_pricing = dict(s.model_pricing_usd_per_mtok)
    new_pricing["mock"] = {"input": 1000.0, "output": 1000.0}
    monkeypatch.setattr(s, "model_pricing_usd_per_mtok", new_pricing)

    wf = ResearchWorkflow(settings=s)
    resp = await wf.run(RunRequest(user_input="What is the capital of France?"))
    assert resp.status == "partial"
    assert resp.metrics.fallback_used is True


def test_fallback_router_for_tool_failure():
    d = decide_fallback("tool_failure")
    assert d.strategy == Strategy.NO_TOOL_ANSWER


@pytest.mark.asyncio
async def test_partial_answer_is_valid_final_answer():
    fa = safe_partial_answer("budget")
    assert fa.confidence == "low"
    assert "human-review" in " ".join(fa.caveats)
