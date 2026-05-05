from app.models import Citation, FinalAnswer, RunMetrics, RunResponse
from evals.scorers import (
    score_correctness,
    score_cost,
    score_factual_coverage,
    score_latency,
    score_safety,
    score_tool_use,
)


def _final(answer="Paris", confidence="high", caveats=None) -> FinalAnswer:
    return FinalAnswer(
        answer=answer,
        confidence=confidence,
        citations=[Citation(title="t", source="kb://x")],
        caveats=caveats or [],
    )


def _resp(**kw) -> RunResponse:
    m = RunMetrics(
        run_id="r",
        duration_ms=kw.get("dur", 100),
        tokens_in=kw.get("ti", 10),
        tokens_out=kw.get("to", 10),
        estimated_cost_usd=kw.get("cost", 0.001),
        tool_calls=kw.get("tools", 0),
        fallback_used=False,
        model="mock",
    )
    return RunResponse(run_id="r", final=_final(), metrics=m)


def test_correctness_high_pass():
    assert score_correctness(_final(confidence="high"), "answers Paris") == 1.0


def test_correctness_low_fail():
    assert score_correctness(_final(confidence="low"), "answers Paris") == 0.5


def test_correctness_hedge_expected():
    assert score_correctness(_final(confidence="low"), "low confidence answer") == 1.0
    assert score_correctness(_final(confidence="high"), "refuse to answer") == 0.4


def test_correctness_no_answer_zero():
    assert score_correctness(None, "anything") == 0.0


def test_factual_coverage_partial():
    fa = _final(answer="The number is 42 only.")
    assert score_factual_coverage(fa, ["42", "missing"]) == 0.5


def test_factual_coverage_no_required_facts():
    assert score_factual_coverage(_final(), []) == 1.0


def test_safety_detects_forbidden():
    fa = _final(answer="here is sk-secretkey for you")
    score, hits = score_safety(fa, ["sk-"])
    assert score == 0.0 and hits == ["sk-"]


def test_safety_clean():
    score, hits = score_safety(_final(), ["sk-"])
    assert score == 1.0 and hits == []


def test_tool_use_within_budget():
    assert score_tool_use(_resp(tools=2), max_tools=3) == 1.0


def test_tool_use_over_budget():
    s = score_tool_use(_resp(tools=6), max_tools=3)
    assert 0.0 <= s < 1.0


def test_cost_within():
    assert score_cost(_resp(cost=0.01), max_cost_usd=0.05) == 1.0


def test_cost_over():
    assert score_cost(_resp(cost=0.10), max_cost_usd=0.05) < 1.0


def test_latency_within():
    assert score_latency(_resp(dur=500), max_latency_ms=1000) == 1.0
