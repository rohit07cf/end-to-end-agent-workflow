"""FastAPI entrypoint.

Endpoints:
  POST /agent/run          -> RunResponse JSON
  POST /agent/run/stream   -> SSE: ReasoningEvent then RunResponse
  GET  /health             -> liveness + provider/model
  GET  /metrics            -> Prometheus exposition
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sse_starlette.sse import EventSourceResponse

from app.agent.workflow import ResearchWorkflow
from app.config import get_settings
from app.models import ReasoningEvent, RunRequest, RunResponse
from app.observability.logging import configure_logging, get_logger
from app.observability.metrics import REGISTRY, REQUEST_COUNT
from app.observability.tracing import init_tracing, run_span

configure_logging()
init_tracing()
log = get_logger(__name__)

app = FastAPI(title="Research Assistant Agent", version="0.1.0")
_workflow = ResearchWorkflow()


@app.get("/health")
async def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "provider": s.model_provider,
        "model": s.claude_model,
        "version": "0.1.0",
    }


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.post("/agent/run", response_model=RunResponse)
async def agent_run(req: RunRequest) -> RunResponse:
    REQUEST_COUNT.labels(endpoint="agent.run.json").inc()
    with run_span("agent.run", user_id=req.user_id or "", conversation_id=req.conversation_id or ""):
        return await _workflow.run(req)


@app.post("/agent/run/stream")
async def agent_run_stream(req: RunRequest, request: Request) -> EventSourceResponse:
    REQUEST_COUNT.labels(endpoint="agent.run.stream").inc()

    async def event_gen() -> AsyncIterator[dict]:
        try:
            async for item in _workflow.run_stream(req):
                if await request.is_disconnected():
                    log.info("sse.client_disconnected")
                    break
                if isinstance(item, ReasoningEvent):
                    yield {
                        "event": item.action_type,
                        "data": item.model_dump_json(),
                    }
                elif isinstance(item, RunResponse):
                    yield {"event": "final", "data": item.model_dump_json()}
        except Exception as e:  # noqa: BLE001
            log.exception("sse.error", error=str(e))
            yield {
                "event": "error",
                "data": json.dumps({"error": type(e).__name__, "detail": str(e)[:500]}),
            }

    return EventSourceResponse(event_gen())


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    log.exception("api.unhandled", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": type(exc).__name__},
    )
