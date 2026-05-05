import time

from app.agent.budgets import BudgetState, Budgets


def _b(**kw):
    return Budgets(
        max_total_tokens=kw.get("tokens", 1000),
        max_cost_usd=kw.get("cost", 1.0),
        max_latency_ms=kw.get("latency", 10_000),
        max_tool_calls=kw.get("tools", 5),
    )


def test_ok():
    b = _b()
    s = BudgetState(model="claude-haiku-4-5-20251001", tokens_in=10, tokens_out=10)
    assert b.check(s) == "ok"


def test_tokens_breach():
    b = _b(tokens=5)
    s = BudgetState(model="claude-haiku-4-5-20251001", tokens_in=5, tokens_out=5)
    assert b.check(s) == "tokens"


def test_cost_breach():
    # Tokens budget large so the cost gate is the one that trips first.
    b = _b(tokens=10_000_000, cost=0.0001)
    s = BudgetState(model="claude-opus-4-7", tokens_in=1000, tokens_out=1000)
    assert b.check(s) == "cost"


def test_tool_breach():
    b = _b(tools=2)
    s = BudgetState(tool_calls=3)
    assert b.check(s) == "tool_calls"


def test_latency_breach():
    b = _b(latency=1)  # 1ms
    s = BudgetState()
    s.started_at -= 1.0  # pretend we started 1s ago
    assert b.check(s) == "latency"
