import pytest
from pydantic import ValidationError

from app.models import Citation, FinalAnswer, ReasoningEvent, RunRequest


def test_final_answer_requires_confidence_enum():
    with pytest.raises(ValidationError):
        FinalAnswer(answer="x", confidence="super-high")  # type: ignore[arg-type]


def test_final_answer_minimal():
    fa = FinalAnswer(answer="hello", confidence="low")
    assert fa.citations == [] and fa.caveats == []


def test_final_answer_with_citation():
    fa = FinalAnswer(
        answer="hi",
        confidence="high",
        citations=[Citation(title="t", source="kb://x", snippet="s")],
    )
    assert fa.citations[0].source == "kb://x"


def test_run_request_input_bounds():
    with pytest.raises(ValidationError):
        RunRequest(user_input="")
    RunRequest(user_input="ok")


def test_reasoning_event_rejects_unknown_action():
    with pytest.raises(ValidationError):
        ReasoningEvent(
            step=1,
            run_id="r",
            agent_name="a",
            action_type="explode",  # type: ignore[arg-type]
            decision_summary="x",
        )
