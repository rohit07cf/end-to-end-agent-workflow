"""End-to-end agent workflow — orchestrates the OpenAI Agents SDK Runner with
our hooks, budgets, fallback router, and structured output.

The public surface is two coroutine methods:

  - ``ResearchWorkflow.run(req)``         -> RunResponse
  - ``ResearchWorkflow.run_stream(req)``  -> async iterator of ReasoningEvent

plus a final ``RunResponse`` payload available via ``last_response``.

The Agent itself is rebuilt per-run because the model identity may change
on fallback (downgrade to cheaper model)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from agents import Agent, Runner
from agents.exceptions import (
    AgentsException,
    MaxTurnsExceeded,
    ModelBehaviorError,
)
from pydantic import ValidationError

from app.agent.budgets import BudgetState, Budgets
from app.agent.fallback import (
    AgentRunError,
    Strategy,
    decide_fallback,
    human_review_answer,
    safe_partial_answer,
)
from app.agent.hooks import (
    HookContext,
    ReasoningSummaryHooks,
    ResearchRunHooks,
    emit_error,
    emit_fallback,
)
from app.agent.providers import build_model
from app.agent.reasoning_events import EventSink
from app.agent.tools import ALL_TOOLS
from app.agent.cost import estimate_cost
from app.config import Settings, get_settings
from app.models import (
    FinalAnswer,
    ReasoningEvent,
    RunMetrics,
    RunRequest,
    RunResponse,
)
from app.observability.logging import bind_run_context, get_logger, unbind_run_context
from app.observability.metrics import COST_USD, ERROR_COUNT, REQUEST_COUNT

log = get_logger(__name__)

INSTRUCTIONS = """\
You are a careful research assistant.

Your job:
1. Plan how to answer the user's question.
2. Use the provided tools (web_search, calculator, knowledge_lookup) when they
   help. Do not invent citations.
3. Produce a final answer that conforms to the FinalAnswer schema:
   - answer: clear, concise, factual
   - confidence: "low" | "medium" | "high"
   - citations: list of {title, source, snippet?}
   - caveats: any caveats the user should know

Hard rules:
- Never include private chain-of-thought in the final answer.
- If you cannot answer factually, say so and set confidence="low".
- Do not call a tool more than necessary. Stop calling tools once you have
  enough information.
"""


class ResearchWorkflow:
    """Wraps the OpenAI Agents SDK Runner with our production controls."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ---------- public ----------

    async def run(self, req: RunRequest) -> RunResponse:
        return await self._run_internal(req, stream=False)

    async def run_stream(
        self, req: RunRequest
    ) -> AsyncIterator[ReasoningEvent | RunResponse]:
        """Yields ReasoningEvent objects as they happen, then a final
        RunResponse as the last item (so SSE consumers can render the answer
        once the stream completes)."""
        run_id = str(uuid4())
        sink = EventSink(run_id=run_id)
        ctx_tokens = bind_run_context(
            run_id=run_id, user_id=req.user_id, conversation_id=req.conversation_id
        )

        # Drive the workflow concurrently with the event drain.
        task = asyncio.create_task(self._drive(req, sink, run_id))
        try:
            async for ev in sink.stream():
                yield ev
            response = await task
            yield response
        finally:
            unbind_run_context(ctx_tokens)

    # ---------- internals ----------

    async def _run_internal(self, req: RunRequest, *, stream: bool) -> RunResponse:
        run_id = str(uuid4())
        sink = EventSink(run_id=run_id)
        ctx_tokens = bind_run_context(
            run_id=run_id, user_id=req.user_id, conversation_id=req.conversation_id
        )
        try:
            return await self._drive(req, sink, run_id)
        finally:
            unbind_run_context(ctx_tokens)

    async def _drive(self, req: RunRequest, sink: EventSink, run_id: str) -> RunResponse:
        REQUEST_COUNT.labels(endpoint="agent.run").inc()
        budgets = Budgets.from_settings(
            self.settings,
            **{
                k: v
                for k, v in {
                    "max_cost_usd": req.max_cost_usd,
                    "max_latency_ms": req.max_latency_ms,
                    "max_tool_calls": req.max_tool_calls,
                }.items()
                if v is not None
            },
        )
        state = BudgetState()
        ctx = HookContext(
            run_id=run_id,
            sink=sink,
            budgets=budgets,
            state=state,
            user_id=req.user_id,
            conversation_id=req.conversation_id,
        )

        run_hooks = ResearchRunHooks()
        agent_hooks = ReasoningSummaryHooks(planning_label="Researcher")

        attempt = 0
        model_name = self.settings.claude_model
        final: FinalAnswer | None = None
        status: str = "ok"
        error_msg: str | None = None
        fallback_used = False

        try:
            while True:
                attempt += 1
                handle = build_model(self.settings, model_name=model_name)
                state.model = handle.name

                agent = Agent[HookContext](
                    name="ResearchAssistant",
                    instructions=INSTRUCTIONS,
                    tools=list(ALL_TOOLS),
                    model=handle.model,
                    output_type=FinalAnswer,
                    hooks=agent_hooks,
                )

                try:
                    # Cap wall-clock per attempt to enforce latency budget.
                    timeout_s = max(1.0, budgets.max_latency_ms / 1000.0)
                    result = await asyncio.wait_for(
                        Runner.run(
                            starting_agent=agent,
                            input=req.user_input,
                            context=ctx,
                            hooks=run_hooks,
                            max_turns=max(2, budgets.max_tool_calls + 2),
                        ),
                        timeout=timeout_s,
                    )
                    raw_output = result.final_output
                    final = _coerce_final(raw_output)
                    break
                except asyncio.TimeoutError as e:
                    emit_error(sink, agent.name, e)
                    raise AgentRunError("model_timeout", "wait_for timeout") from e
                except MaxTurnsExceeded as e:
                    emit_error(sink, agent.name, e)
                    raise AgentRunError("budget_exceeded", "max_turns exceeded") from e
                except ModelBehaviorError as e:
                    emit_error(sink, agent.name, e)
                    raise AgentRunError("invalid_structured_output", str(e)) from e
                except (ValidationError, json.JSONDecodeError) as e:
                    emit_error(sink, agent.name, e)
                    raise AgentRunError("invalid_structured_output", str(e)) from e
                except AgentsException as e:
                    emit_error(sink, agent.name, e)
                    raise AgentRunError("unknown", str(e)) from e
                except Exception as e:  # noqa: BLE001
                    emit_error(sink, agent.name, e)
                    kind = "model_rate_limit" if "rate" in str(e).lower() else "unknown"
                    raise AgentRunError(kind, str(e)) from e

        except AgentRunError as e:
            decision = decide_fallback(e.kind, attempt=attempt)
            emit_fallback(sink, "ResearchAssistant", decision.strategy.value, decision.detail)
            fallback_used = True
            final, status = await self._apply_fallback(
                decision.strategy, req, ctx, attempt
            )
            error_msg = f"{e.kind}: {e}"
            log.warning(
                "agent.fallback",
                kind=e.kind,
                strategy=decision.strategy.value,
                detail=decision.detail,
            )

        except Exception as e:  # last-resort net
            ERROR_COUNT.labels(agent="ResearchAssistant", kind=type(e).__name__).inc()
            log.exception("agent.unhandled", error=str(e))
            final = safe_partial_answer("unhandled error")
            status = "error"
            error_msg = str(e)
            fallback_used = True

        # post-run budget enforcement (in case last attempt put us over)
        breach = budgets.check(state, self.settings)
        if breach != "ok" and status == "ok":
            emit_fallback(
                sink, "ResearchAssistant", "partial_answer", f"budget breach: {breach}"
            )
            final = safe_partial_answer(breach)
            status = "partial"
            fallback_used = True

        cost = estimate_cost(state.model or model_name, state.tokens_in, state.tokens_out, self.settings).total_usd
        COST_USD.labels(agent="ResearchAssistant", model=state.model or model_name).inc(cost)

        sink.close()

        metrics = RunMetrics(
            run_id=run_id,
            duration_ms=state.latency_ms,
            tokens_in=state.tokens_in,
            tokens_out=state.tokens_out,
            estimated_cost_usd=cost,
            tool_calls=state.tool_calls,
            fallback_used=fallback_used,
            model=state.model or model_name,
            error=error_msg,
        )
        return RunResponse(
            run_id=run_id, final=final, events=sink.events, metrics=metrics, status=status  # type: ignore[arg-type]
        )

    async def _apply_fallback(
        self,
        strategy: Strategy,
        req: RunRequest,
        ctx: HookContext,
        attempt: int,
    ) -> tuple[FinalAnswer, str]:
        """Execute the fallback strategy. Returns (final, status)."""
        if strategy == Strategy.RETRY_BACKOFF:
            backoff = self.settings.retry_initial_backoff_s * (2 ** (attempt - 1))
            await asyncio.sleep(min(backoff, 8.0))
            # Caller will loop — but our control flow is single-shot at the
            # top level; surface as partial so we don't infinite-loop here.
            return safe_partial_answer("retry exhausted"), "partial"
        if strategy == Strategy.SIMPLER_MODEL:
            # Re-run once with cheaper model.
            handle = build_model(self.settings, model_name=self.settings.claude_fallback_model)
            ctx.state.model = handle.name
            agent = Agent[HookContext](
                name="ResearchAssistantFallback",
                instructions=INSTRUCTIONS,
                tools=list(ALL_TOOLS),
                model=handle.model,
                output_type=FinalAnswer,
            )
            try:
                result = await asyncio.wait_for(
                    Runner.run(
                        starting_agent=agent,
                        input=req.user_input,
                        context=ctx,
                        max_turns=4,
                    ),
                    timeout=15.0,
                )
                return _coerce_final(result.final_output), "partial"
            except Exception:  # noqa: BLE001
                return safe_partial_answer("simpler-model also failed"), "partial"
        if strategy == Strategy.NO_TOOL_ANSWER:
            handle = build_model(self.settings)
            agent = Agent[HookContext](
                name="ResearchAssistantNoTools",
                instructions=INSTRUCTIONS + "\n\nDo not call any tools.",
                tools=[],
                model=handle.model,
                output_type=FinalAnswer,
            )
            try:
                result = await asyncio.wait_for(
                    Runner.run(starting_agent=agent, input=req.user_input, context=ctx, max_turns=2),
                    timeout=15.0,
                )
                return _coerce_final(result.final_output), "partial"
            except Exception:  # noqa: BLE001
                return safe_partial_answer("no-tool fallback failed"), "partial"
        if strategy == Strategy.PARTIAL_ANSWER:
            return safe_partial_answer("budget"), "partial"
        if strategy == Strategy.HUMAN_REVIEW:
            return human_review_answer("model unable to comply"), "partial"
        if strategy == Strategy.CACHED_ANSWER:
            return safe_partial_answer("no cache configured"), "partial"
        if strategy == Strategy.LOG_AND_CONTINUE:
            return safe_partial_answer("non-fatal subsystem failure"), "partial"
        return safe_partial_answer("unknown strategy"), "partial"


def _coerce_final(output: Any) -> FinalAnswer:
    if isinstance(output, FinalAnswer):
        return output
    if isinstance(output, dict):
        return FinalAnswer.model_validate(output)
    if isinstance(output, str):
        try:
            return FinalAnswer.model_validate_json(output)
        except Exception:  # noqa: BLE001
            return FinalAnswer(
                answer=output[:2000], confidence="low", citations=[], caveats=["unstructured output"]
            )
    raise AgentRunError("invalid_structured_output", f"cannot coerce {type(output)}")
