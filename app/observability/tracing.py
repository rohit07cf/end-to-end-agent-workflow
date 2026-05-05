"""OpenTelemetry tracing init.

If OTEL_EXPORTER_OTLP_ENDPOINT is unset we use a no-op exporter — useful for
local dev, CI, and tests. Production deployments set the OTLP endpoint to
their collector (Tempo/Jaeger/Honeycomb/etc).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from app.config import get_settings

_initialised = False


def init_tracing() -> trace.Tracer:
    """Idempotent OTel init. Returns a tracer."""
    global _initialised
    if _initialised:
        return trace.get_tracer("research-agent")

    s = get_settings()
    resource = Resource.create({"service.name": s.otel_service_name})
    provider = TracerProvider(resource=resource)

    if s.otel_exporter_otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=s.otel_exporter_otlp_endpoint))
            )
        except Exception:  # pragma: no cover — degrade to console
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    # No exporter when endpoint is unset — keep traces in-memory for local dev.

    trace.set_tracer_provider(provider)
    _initialised = True
    return trace.get_tracer("research-agent")


@contextmanager
def run_span(name: str, **attrs: str | int | float | bool) -> Iterator[trace.Span]:
    tracer = init_tracing()
    with tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(k, v)
        yield span
