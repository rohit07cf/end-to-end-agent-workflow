# Research Assistant Agent

Production-grade agent workflow built on the **OpenAI Agents SDK**, using
**Claude** as the reasoning model through a configurable provider
abstraction. Includes safe reasoning streams, hooks, cost/latency budgets,
fallback routing, offline evals with regression gates, observability, and
CI.

## Architecture

```
┌──────────────────────┐      ┌─────────────────────────────────────┐
│  HTTP client / SSE   │──────▶│ FastAPI (app/main.py)               │
└──────────────────────┘      │  POST /agent/run                    │
                              │  POST /agent/run/stream  (SSE)      │
                              │  GET  /health                       │
                              │  GET  /metrics  (Prometheus)        │
                              └──────────────┬──────────────────────┘
                                             │
                              ┌──────────────▼──────────────────────┐
                              │ ResearchWorkflow (app/agent/        │
                              │  workflow.py)                       │
                              │  ┌──────────────────────────────┐   │
                              │  │ OpenAI Agents SDK Runner     │   │
                              │  │  • RunHooks (token/metrics)  │   │
                              │  │  • AgentHooks (reasoning)    │   │
                              │  │  • output_type=FinalAnswer   │   │
                              │  │  • tools=[search,calc,kb]    │   │
                              │  └──────────────────────────────┘   │
                              │  budgets · cost calc · fallback     │
                              │  reasoning event sink (safe)        │
                              └──────┬───────────────┬──────────────┘
                                     │               │
                       ┌─────────────▼─┐   ┌─────────▼──────────┐
                       │ Provider:     │   │ Observability:     │
                       │  Claude (Lite-│   │  structlog (JSON)  │
                       │  LLM)  / Mock │   │  Prometheus metrics│
                       └───────────────┘   │  OpenTelemetry     │
                                           └────────────────────┘
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]' litellm    # litellm needed for the Claude provider
cp .env.example .env
# edit .env, set ANTHROPIC_API_KEY for live mode
```

## Run

```bash
# Mock mode (default in tests / CI):
MODEL_PROVIDER=mock uvicorn app.main:app --port 8080

# Live mode (Claude):
MODEL_PROVIDER=claude ANTHROPIC_API_KEY=sk-ant-... \
  uvicorn app.main:app --port 8080
```

Try it:

```bash
curl -s localhost:8080/agent/run \
  -H 'content-type: application/json' \
  -d '{"user_input":"What is the capital of France?"}' | jq .
```

Streaming (server-sent events):

```bash
curl -N localhost:8080/agent/run/stream \
  -H 'content-type: application/json' \
  -d '{"user_input":"What is the capital of France?"}'
```

You'll see `agent_start`, `reasoning_summary`, `llm_start`, `tool_start`,
`tool_end`, `llm_end`, `agent_end`, and finally a `final` event carrying
the structured answer.

## Tests

```bash
pytest                      # 60+ tests, ~2s in mock mode
pytest tests/unit           # just unit
pytest tests/integration    # full-run, fallback, eval-runner, HTTP API
```

## Evals

```bash
python -m evals.run_evals             # mock mode, exits non-zero on regression
python -m evals.run_evals --no-gate   # advisory
MODEL_PROVIDER=claude ANTHROPIC_API_KEY=... python -m evals.run_evals
```

Outputs:

- `evals/results.json` — full per-case scores + metrics
- `evals/report.md` — human-readable summary

## Regression gates (CI)

The `python -m evals.run_evals` step in `.github/workflows/ci.yml` fails if
any of:

| gate                       | env                          | default |
|----------------------------|------------------------------|---------|
| correctness                | `GATE_MIN_CORRECTNESS`       | 0.85    |
| factual coverage           | `GATE_MIN_FACTUAL_COVERAGE`  | 0.85    |
| safety                     | `GATE_MIN_SAFETY`            | 0.95    |
| p95 latency                | `GATE_MAX_P95_LATENCY_MS`    | 20000   |
| avg cost                   | `GATE_MAX_AVG_COST_USD`      | 0.10    |
| any forbidden behavior     | hard fail                    |         |

## Configuration (env vars)

See `.env.example`. Most knobs live in `app/config.py` (`Settings`).

## Docs

- `docs/observability.md` — logs, metrics, traces, reasoning events
- `docs/evals.md` — dataset, scorers, gates, modes
- `docs/failure-modes.md` — every failure path and its strategy
- `docs/production-readiness-checklist.md` — pre-deploy checklist

## Repo layout

```
app/
  main.py                 FastAPI entrypoint
  config.py               Pydantic settings
  models.py               public + structured-output schemas
  agent/
    workflow.py           ResearchWorkflow (Runner orchestration)
    providers.py          Claude (LiteLLM) + Mock model providers
    tools.py              web_search / calculator / knowledge_lookup
    hooks.py              RunHooks + AgentHooks (reasoning + budgets)
    reasoning_events.py   safe event sink + redactor
    budgets.py            per-run budget state + check
    cost.py               per-token USD calculator
    fallback.py           failure → strategy router
  observability/
    logging.py            structlog JSON + contextvars
    metrics.py            Prometheus registry & metrics
    tracing.py            OpenTelemetry init
evals/
  golden_dataset.jsonl    test cases
  scorers.py              pure scoring functions
  run_evals.py            runner → results.json + report.md
tests/
  unit/                   cost, budgets, fallback, hooks, scorers, models
  integration/            full-run, fallback, evals, HTTP API
docs/
  observability.md  evals.md  failure-modes.md  production-readiness-checklist.md
grafana/
  agent-dashboard.json    Grafana dashboard JSON
.github/workflows/ci.yml
pyproject.toml  .env.example
```
