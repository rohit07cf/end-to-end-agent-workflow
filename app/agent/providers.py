"""Model-provider abstraction.

The OpenAI Agents SDK exposes a `Model` interface; production code uses
`LitellmModel` to talk to Claude (any Anthropic model id), and tests use
`MockModel` which returns deterministic responses without network calls.

Switching providers is config-driven via `MODEL_PROVIDER` env var so a SRE
can roll between providers without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents import ModelSettings
from agents.items import ModelResponse, TResponseInputItem
from agents.models.interface import Model, ModelTracing
from agents.usage import Usage

from app.config import Settings, get_settings


@dataclass
class ProviderHandle:
    """Bundle of model + name returned to the workflow."""

    model: Model
    name: str
    is_mock: bool


def build_model(settings: Settings | None = None, *, model_name: str | None = None) -> ProviderHandle:
    """Return a `Model` instance selected by settings.

    Args:
      settings: optional override (tests inject a config).
      model_name: override the model id (used by fallback to a cheaper model).
    """
    s = settings or get_settings()
    name = model_name or s.claude_model

    if s.model_provider == "mock":
        return ProviderHandle(model=MockModel(name="mock"), name="mock", is_mock=True)

    if s.model_provider == "claude":
        if not s.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY required when MODEL_PROVIDER=claude")
        # Lazy import — keeps mock-only environments lightweight.
        from agents.extensions.models.litellm_model import LitellmModel

        litellm_id = name if "/" in name else f"anthropic/{name}"
        return ProviderHandle(
            model=LitellmModel(model=litellm_id, api_key=s.anthropic_api_key),
            name=name,
            is_mock=False,
        )

    raise ValueError(f"Unknown MODEL_PROVIDER: {s.model_provider}")


class MockModel(Model):
    """Deterministic mock for tests/evals without network.

    The mock looks at the user input to decide which path to exercise so we can
    cover branches (tool-call, structured output, error handling) reliably."""

    def __init__(self, name: str = "mock", *, force_error: bool = False) -> None:
        self._name = name
        self._force_error = force_error

    async def get_response(  # type: ignore[override]
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Any],
        output_schema: Any | None,
        handoffs: list[Any],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any | None = None,
    ) -> ModelResponse:
        if self._force_error:
            raise RuntimeError("forced mock model error")

        text = _flatten_input(input)
        # Hard-coded final answer in strict structured form.
        answer = _mock_answer_for(text)

        from openai.types.responses import (
            ResponseOutputMessage,
            ResponseOutputText,
        )

        msg = ResponseOutputMessage(
            id="msg_mock_1",
            role="assistant",
            type="message",
            status="completed",
            content=[
                ResponseOutputText(
                    type="output_text",
                    text=answer,
                    annotations=[],
                )
            ],
        )
        return ModelResponse(
            output=[msg],
            usage=Usage(
                requests=1,
                input_tokens=max(1, len(text) // 4),
                output_tokens=max(1, len(answer) // 4),
                total_tokens=max(2, (len(text) + len(answer)) // 4),
            ),
            response_id="resp_mock_1",
        )

    def stream_response(  # type: ignore[override]
        self, *args: Any, **kwargs: Any
    ):  # noqa: D401
        async def _gen():
            resp = await self.get_response(*args, **kwargs)
            from openai.types.responses import ResponseCompletedEvent

            yield ResponseCompletedEvent(
                type="response.completed",
                response={  # type: ignore[arg-type]
                    "id": resp.response_id,
                    "output": resp.output,
                    "usage": resp.usage,
                },
                sequence_number=0,
            )

        return _gen()


def _flatten_input(input: Any) -> str:
    if isinstance(input, str):
        return input
    parts: list[str] = []
    for item in input or []:
        if isinstance(item, dict):
            content = item.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and isinstance(c.get("text"), str):
                        parts.append(c["text"])
    return "\n".join(parts)


def _mock_answer_for(user_text: str) -> str:
    """Return JSON matching FinalAnswer schema. Branches are deterministic
    so eval/test outcomes are stable."""
    import json

    t = user_text.lower()
    if "capital of france" in t:
        return json.dumps(
            {
                "answer": "The capital of France is Paris.",
                "confidence": "high",
                "citations": [
                    {"title": "World Atlas", "source": "kb://geo/france", "snippet": "Paris is the capital."}
                ],
                "caveats": [],
            }
        )
    if "speed of light" in t:
        return json.dumps(
            {
                "answer": "The speed of light in a vacuum is approximately 299,792,458 m/s.",
                "confidence": "high",
                "citations": [{"title": "Physics Constants", "source": "kb://physics/c"}],
                "caveats": [],
            }
        )
    if "compute" in t or "calculate" in t or "+" in t:
        return json.dumps(
            {
                "answer": "Computed result: 42.",
                "confidence": "medium",
                "citations": [],
                "caveats": ["Mock arithmetic; verify in production."],
            }
        )
    return json.dumps(
        {
            "answer": "I don't have enough information to answer confidently.",
            "confidence": "low",
            "citations": [],
            "caveats": ["mock fallback path"],
        }
    )
