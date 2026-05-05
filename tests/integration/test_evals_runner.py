"""Eval runner integration test (mock mode)."""

import pytest

from evals.run_evals import aggregate, gates_passed, load_dataset, run_all


@pytest.mark.asyncio
async def test_evals_pass_in_mock_mode(tmp_path, monkeypatch):
    cases = load_dataset()
    results = await run_all(cases)
    assert len(results) == len(cases)
    agg = aggregate(results)
    ok, failures = gates_passed(agg)
    assert ok, f"gates failed in mock mode: {failures}"
