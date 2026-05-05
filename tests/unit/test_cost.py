from app.agent.cost import estimate_cost


def test_known_model_pricing():
    cb = estimate_cost("claude-opus-4-7", 1_000_000, 1_000_000)
    # input $15 + output $75
    assert cb.total_usd == 90.0
    assert cb.input_usd == 15.0
    assert cb.output_usd == 75.0


def test_unknown_model_zero_cost():
    cb = estimate_cost("some-unknown-model", 100, 100)
    assert cb.total_usd == 0.0


def test_prefix_match():
    cb = estimate_cost("claude-opus-4-7-20990101", 1_000_000, 0)
    assert cb.input_usd == 15.0


def test_zero_tokens():
    cb = estimate_cost("claude-haiku-4-5-20251001", 0, 0)
    assert cb.total_usd == 0.0
