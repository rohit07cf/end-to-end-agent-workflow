"""Prometheus metrics. Singleton registry — counters/histograms are module-level
so labels stay aggregatable across the process lifetime."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

REQUEST_COUNT = Counter(
    "agent_request_count_total",
    "Total /agent/run requests",
    ["endpoint"],
    registry=REGISTRY,
)

AGENT_RUNS_TOTAL = Counter(
    "agent_runs_total",
    "Total agent runs started, by agent name",
    ["agent"],
    registry=REGISTRY,
)

ERROR_COUNT = Counter(
    "agent_error_count_total",
    "Errors raised during an agent run",
    ["agent", "kind"],
    registry=REGISTRY,
)

TOOL_CALL_COUNT = Counter(
    "agent_tool_call_count_total",
    "Tool calls made by the agent",
    ["agent", "tool"],
    registry=REGISTRY,
)

TOKENS_IN = Counter(
    "agent_tokens_in_total",
    "Total input tokens consumed",
    ["agent"],
    registry=REGISTRY,
)
TOKENS_OUT = Counter(
    "agent_tokens_out_total",
    "Total output tokens produced",
    ["agent"],
    registry=REGISTRY,
)

COST_USD = Counter(
    "agent_estimated_cost_usd_total",
    "Estimated USD cost of agent runs",
    ["agent", "model"],
    registry=REGISTRY,
)

FALLBACK_COUNT = Counter(
    "agent_fallback_count_total",
    "Fallback path activations",
    ["agent", "kind"],
    registry=REGISTRY,
)

# Histograms — buckets tuned for an interactive agent (sub-second to a minute).
_LATENCY_BUCKETS = (50, 100, 250, 500, 1000, 2500, 5000, 10_000, 20_000, 45_000, 60_000)

LATENCY_MS = Histogram(
    "agent_latency_ms",
    "End-to-end latency of an agent run, ms",
    ["agent"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
MODEL_LATENCY_MS = Histogram(
    "agent_model_latency_ms",
    "Per-LLM-call latency, ms",
    ["agent"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
TOOL_LATENCY_MS = Histogram(
    "agent_tool_latency_ms",
    "Per-tool-call latency, ms",
    ["agent", "tool"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
