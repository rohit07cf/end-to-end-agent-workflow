# Evals

Offline evaluation suite. Designed for two modes:

| mode  | env                              | use                                  |
|-------|----------------------------------|--------------------------------------|
| mock  | `MODEL_PROVIDER=mock` (default)  | CI, fast deterministic, no spend     |
| live  | `MODEL_PROVIDER=claude` + key    | nightly / pre-deploy regression run  |

## Dataset

`evals/golden_dataset.jsonl`. One JSON object per line:

```json
{
  "id": "geo-001",
  "user_input": "What is the capital of France?",
  "expected_behavior": "answers Paris with a citation",
  "required_facts": ["Paris"],
  "forbidden_behavior": ["I don't know"],
  "max_latency_ms": 5000,
  "max_cost_usd": 0.05
}
```

Add cases freely — the runner picks them up automatically. Aim for
representative coverage of every important user journey, plus at least one
safety case and one hedge-expected case (so the scorer exercises both
correctness paths).

## Scorers

`evals/scorers.py` exposes pure functions:

- `score_correctness` — heuristic confidence-vs-expected match.
  **Replace with an LLM-judge** in production for nuanced grading.
- `score_factual_coverage` — substring coverage of `required_facts`.
- `score_safety` — 0/1 on `forbidden_behavior` substrings; production should
  use a classifier or PII detector.
- `score_tool_use`, `score_cost`, `score_latency` — budget compliance.

## Runner

```bash
python -m evals.run_evals
# or
python -m evals.run_evals --no-gate     # advisory only
```

Outputs:

- `evals/results.json` — full per-case detail (answer, metrics, scores).
- `evals/report.md` — human-readable summary with aggregate metrics and gate
  status.
- exit code `1` when any regression gate is breached (CI uses this).

## Regression gates

Configured via env (see `.env.example`):

| gate                       | default |
|----------------------------|---------|
| `GATE_MIN_CORRECTNESS`     | 0.85    |
| `GATE_MIN_FACTUAL_COVERAGE`| 0.85    |
| `GATE_MIN_SAFETY`          | 0.95    |
| `GATE_MAX_P95_LATENCY_MS`  | 20000   |
| `GATE_MAX_AVG_COST_USD`    | 0.10    |
| any forbidden behavior     | hard fail |

CI runs `python -m evals.run_evals` after the unit/integration suite. Any
gate breach fails the workflow and uploads `results.json` + `report.md` as
artifacts so reviewers can inspect what regressed.
