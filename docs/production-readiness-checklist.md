# Production-Readiness Checklist

Use this when promoting the service to a new environment.

## Code

- [x] Type hints throughout (`mypy --strict`-ready)
- [x] Pydantic models on every trust boundary (HTTP, eval, log)
- [x] No hidden chain-of-thought emitted to clients/logs
- [x] Secrets read from env, never hardcoded
- [x] Pluggable model provider (`MODEL_PROVIDER=claude|mock`)
- [x] Pluggable cost table

## Tests

- [x] Unit tests: cost, budgets, fallback router, scorers, hooks, schemas, reasoning sink
- [x] Integration tests: full agent run, streaming, fallback paths, budget breach, eval runner, HTTP API
- [x] CI runs them on every push/PR

## Evals & gates

- [x] Golden dataset (`evals/golden_dataset.jsonl`)
- [x] Pure scorers (correctness, factual coverage, safety, tool-use, cost, latency)
- [x] Mock + live modes
- [x] CI fails on regression
- [x] Per-case `max_cost_usd` / `max_latency_ms` budgets
- [ ] **TODO:** swap heuristic correctness scorer for LLM-judge in CI

## Observability

- [x] JSON structured logs with `trace_id` / `run_id` / `user_id` / `conversation_id`
- [x] Prometheus metrics (`agent_*`)
- [x] OpenTelemetry tracing (OTLP/HTTP exporter)
- [x] Grafana dashboard (`grafana/agent-dashboard.json`)
- [x] SSE stream of safe reasoning events
- [ ] **TODO:** wire alerts (p95 latency, error rate, fallback rate, cost burn)

## Cost & latency controls

- [x] `MAX_TOTAL_TOKENS_PER_RUN`, `MAX_COST_USD_PER_RUN`, `MAX_LATENCY_MS_PER_RUN`, `MAX_TOOL_CALLS_PER_RUN`
- [x] Hard wall-clock cap via `asyncio.wait_for`
- [x] Cost calculator with auditable pricing table
- [ ] **TODO:** rate-limit middleware in FastAPI (per-user / per-org)
- [ ] **TODO:** persistent cache for `cached_answer` strategy

## Failure handling

- [x] `decide_fallback` matrix covers timeout / rate-limit / tool fail / invalid output / budget / trace fail / eval regression
- [x] Partial-answer + human-review canonical responses
- [x] Errors counted in metrics with `kind` label

## Security

- [x] Mock tools have no secrets
- [x] Reasoning summaries redacted for `api_key`/`authorization`/`bearer`/`secret`
- [ ] **TODO:** add a real PII / secret scrubber on free-text logs
- [ ] **TODO:** auth middleware (JWT / mTLS / API key) on `/agent/run*`
- [ ] **TODO:** input length / content moderation on `user_input`
- [ ] **TODO:** egress allowlist in production tools

## Deployment

- [ ] **TODO:** Dockerfile + image scanning
- [ ] **TODO:** Kubernetes manifests with resource limits, HPA, PDB
- [ ] **TODO:** secret manager integration (Vault / AWS Secrets Manager)
- [ ] **TODO:** blue/green or canary roll-outs gated on the eval suite

## Data retention

- [ ] **TODO:** decide retention for run logs, reasoning events, traces
- [ ] **TODO:** PII handling docs and DSAR support
