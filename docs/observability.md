# Observability

Three signals, one trace ID per run.

## Logs (JSON)

Every log line passes through structlog → JSONRenderer and carries the
contextvars set in `app/observability/logging.py`:

| field             | source                          |
|-------------------|---------------------------------|
| `run_id`          | bound at workflow entry         |
| `user_id`         | from `RunRequest.user_id`       |
| `conversation_id` | from `RunRequest.conversation_id` |
| `trace_id`        | from OTel span (if active)      |
| `span_id`         | from OTel span (if active)      |
| `level`, `event`, `timestamp` | structlog defaults |

Bind context with `bind_run_context(...)` (returns a list of tokens) and
release with `unbind_run_context(tokens)`. The workflow does this for you.

## Metrics (Prometheus)

Exposed at `GET /metrics`. All metrics are namespaced `agent_*`:

- `agent_request_count_total{endpoint=...}` — HTTP requests
- `agent_runs_total{agent=...}` — agent runs started
- `agent_error_count_total{agent,kind}` — errors by exception class
- `agent_tool_call_count_total{agent,tool}` — tool invocations
- `agent_tokens_in_total{agent}` / `agent_tokens_out_total{agent}` — token usage
- `agent_estimated_cost_usd_total{agent,model}` — running cost (Counter)
- `agent_fallback_count_total{agent,kind}` — fallback path activations
- `agent_latency_ms`, `agent_model_latency_ms`, `agent_tool_latency_ms` — Histograms

Buckets are tuned for sub-second to ~1-min agent runs. Tweak in
`app/observability/metrics.py` if your SLO differs.

A ready-made Grafana dashboard ships at `grafana/agent-dashboard.json`.

## Traces (OpenTelemetry)

`init_tracing()` is called at app startup. If `OTEL_EXPORTER_OTLP_ENDPOINT`
is set, spans are exported via OTLP/HTTP. The agent run is wrapped in a
`agent.run` span; the OpenAI Agents SDK adds finer-grained spans for the
LLM call, tool call, etc., when its built-in tracing is enabled.

## Reasoning events (safe traces)

The custom reasoning hook (`ReasoningSummaryHooks`) emits a stream of
`ReasoningEvent` objects with high-level **decision summaries**, not hidden
chain-of-thought. The SSE endpoint streams these to clients as named SSE
events (`agent_start`, `tool_start`, `tool_end`, `llm_end`,
`reasoning_summary`, `budget_warning`, `fallback`, `error`, `final`).

Field reference:

| field              | meaning                                      |
|--------------------|----------------------------------------------|
| `step`             | per-run monotonic counter                    |
| `run_id`           | run identifier                               |
| `agent_name`       | which agent emitted the event                |
| `action_type`      | one of the enum values                       |
| `decision_summary` | short human-readable summary (redacted)      |
| `tool_used`        | tool name, if applicable                     |
| `latency_ms`       | wall-clock time                              |
| `tokens_in/out`    | usage from the SDK, when available           |

Redaction: any summary containing `api_key`, `authorization`, `bearer`, or
`secret` is replaced with `[redacted summary]`.
