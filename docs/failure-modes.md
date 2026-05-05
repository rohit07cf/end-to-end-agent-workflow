# Failure modes & fallbacks

Every production agent runs on a brittle stack: the model, the tools, the
network, the budget, the schema, the trace exporter. This doc enumerates
the failures we plan for and the strategy each one maps to.

## Decision matrix

| failure                       | first response          | escalation                    |
|-------------------------------|-------------------------|-------------------------------|
| `model_timeout`               | retry with backoff      | downgrade to simpler model    |
| `model_rate_limit`            | exponential backoff     | downgrade to simpler model    |
| `tool_failure`                | answer without the tool | partial answer if no info     |
| `invalid_structured_output`   | retry once              | human-review response         |
| `budget_exceeded`             | partial answer          | (terminal — return immediately) |
| `tracing_export_failure`      | log and continue        | alert on sustained failure    |
| `eval_regression`             | block deploy            | human review                  |

Mapping logic lives in `app/agent/fallback.py` (`decide_fallback`). It is
pure — unit-tested in `tests/unit/test_fallback.py`.

## Strategies

- **`retry_backoff`** — `tenacity`-style exponential backoff using
  `RETRY_INITIAL_BACKOFF_S * 2^(attempt-1)` capped at 8s. Up to
  `RETRY_MAX_ATTEMPTS` attempts.
- **`simpler_model`** — re-run with `CLAUDE_FALLBACK_MODEL` (e.g. Haiku),
  fewer turns, tighter timeout. The model id is logged so cost reports stay
  attributable.
- **`no_tool_answer`** — re-run with `tools=[]` and an instruction to answer
  without external data. Useful when an upstream search index is down.
- **`partial_answer`** — `safe_partial_answer(reason)` returns a valid
  `FinalAnswer` with `confidence="low"` and a `caveat` explaining the
  reason. Status code `partial`.
- **`human_review`** — explicit "needs human" response. Caller should
  surface this in the UI and route to a human queue.
- **`cached_answer`** — placeholder for production cache integration. Not
  enabled in this repo.
- **`log_and_continue`** — for non-critical subsystem failures (e.g. trace
  export). Alert on sustained failure rather than failing the request.

## Boundaries

- Hard wall-clock cap: `asyncio.wait_for(...)` around `Runner.run` enforces
  `MAX_LATENCY_MS_PER_RUN`. Even a buggy tool can't keep the request hung.
- Token + cost cap: budget hooks accumulate usage; the workflow checks
  `Budgets.check(state)` after each attempt and coerces partial output if
  any cap has been breached.
- Output schema cap: `output_type=FinalAnswer` is enforced by the SDK; if
  the model can't produce schema-conformant JSON, that surfaces as
  `ModelBehaviorError` → `invalid_structured_output` → retry → human-review.

## What we don't handle (yet)

- Long-running tool calls inside a single turn (we cap *between* turns; an
  individual tool that blocks longer than `MAX_LATENCY_MS_PER_RUN` will
  trigger `model_timeout`).
- Partial tool results — tools must succeed or raise. A tool that returns
  silently degraded data will pass through.
- Distributed cache invalidation — `cached_answer` is a placeholder.
