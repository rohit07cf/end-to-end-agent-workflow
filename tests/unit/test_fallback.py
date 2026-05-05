from app.agent.fallback import (
    Strategy,
    decide_fallback,
    human_review_answer,
    safe_partial_answer,
)


def test_model_timeout_first_attempt_retries():
    d = decide_fallback("model_timeout", attempt=1)
    assert d.strategy == Strategy.RETRY_BACKOFF


def test_model_timeout_after_retries_downgrades():
    d = decide_fallback("model_timeout", attempt=3)
    assert d.strategy == Strategy.SIMPLER_MODEL


def test_rate_limit_eventually_downgrades():
    assert decide_fallback("model_rate_limit", attempt=1).strategy == Strategy.RETRY_BACKOFF
    assert decide_fallback("model_rate_limit", attempt=5).strategy == Strategy.SIMPLER_MODEL


def test_tool_failure_uses_no_tool_path():
    assert decide_fallback("tool_failure").strategy == Strategy.NO_TOOL_ANSWER


def test_invalid_output_retries_then_human():
    assert decide_fallback("invalid_structured_output", attempt=1).strategy == Strategy.RETRY_BACKOFF
    assert decide_fallback("invalid_structured_output", attempt=3).strategy == Strategy.HUMAN_REVIEW


def test_budget_returns_partial():
    assert decide_fallback("budget_exceeded").strategy == Strategy.PARTIAL_ANSWER


def test_eval_regression_human():
    assert decide_fallback("eval_regression").strategy == Strategy.HUMAN_REVIEW


def test_partial_answer_shape():
    fa = safe_partial_answer("tokens")
    assert fa.confidence == "low"
    assert any("tokens" in c for c in fa.caveats)


def test_human_review_shape():
    fa = human_review_answer("schema")
    assert fa.confidence == "low"
    assert any("human-review" in c for c in fa.caveats)
