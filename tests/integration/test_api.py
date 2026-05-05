"""HTTP API tests using FastAPI's TestClient."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics_exposes_prom_format():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "agent_request_count_total" in r.text


def test_agent_run_returns_structured_response():
    r = client.post("/agent/run", json={"user_input": "What is the capital of France?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["final"]["confidence"] in {"low", "medium", "high"}
    assert "Paris" in body["final"]["answer"]
    assert body["metrics"]["model"] == "mock"


def test_agent_run_validation_error():
    r = client.post("/agent/run", json={"user_input": ""})
    assert r.status_code == 422


def test_stream_emits_sse_events():
    with client.stream(
        "POST", "/agent/run/stream", json={"user_input": "What is the capital of France?"}
    ) as r:
        assert r.status_code == 200
        seen_events: list[str] = []
        for line in r.iter_lines():
            if not line:
                continue
            if line.startswith("event:"):
                seen_events.append(line.split(":", 1)[1].strip())
            if "final" in seen_events:
                break
        assert "agent_start" in seen_events
        assert "final" in seen_events
