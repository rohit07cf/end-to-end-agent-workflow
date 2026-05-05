"""Offline eval runner.

Modes:
  - mock (default): MODEL_PROVIDER=mock — fast, deterministic, no network.
  - live: set MODEL_PROVIDER=claude + ANTHROPIC_API_KEY.

Outputs:
  - evals/results.json — full per-case metrics
  - evals/report.md     — human-readable summary
  - exit code 0 if all gates pass, 1 if any gate fails (CI uses this).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

from app.agent.workflow import ResearchWorkflow
from app.config import get_settings
from app.models import RunRequest, RunResponse
from evals.scorers import CaseScore, score_case

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "evals" / "golden_dataset.jsonl"
RESULTS_PATH = REPO_ROOT / "evals" / "results.json"
REPORT_PATH = REPO_ROOT / "evals" / "report.md"


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def run_one(wf: ResearchWorkflow, case: dict) -> RunResponse:
    return await wf.run(
        RunRequest(
            user_input=case["user_input"],
            max_cost_usd=case.get("max_cost_usd"),
            max_latency_ms=case.get("max_latency_ms"),
        )
    )


async def run_all(cases: list[dict]) -> list[tuple[dict, RunResponse, CaseScore]]:
    wf = ResearchWorkflow()
    out: list[tuple[dict, RunResponse, CaseScore]] = []
    for case in cases:
        response = await run_one(wf, case)
        score = score_case(case, response)
        out.append((case, response, score))
    return out


def aggregate(results: list[tuple[dict, RunResponse, CaseScore]]) -> dict[str, Any]:
    if not results:
        return {}
    durations = [r.metrics.duration_ms for _, r, _ in results]
    costs = [r.metrics.estimated_cost_usd for _, r, _ in results]
    correctness = statistics.mean(s.correctness for _, _, s in results)
    factual = statistics.mean(s.factual_coverage for _, _, s in results)
    safety = statistics.mean(s.safety for _, _, s in results)
    tool = statistics.mean(s.tool_use for _, _, s in results)
    cost_score = statistics.mean(s.cost for _, _, s in results)
    latency_score = statistics.mean(s.latency for _, _, s in results)
    forbidden_any = any(s.forbidden_hits for _, _, s in results)
    p95_latency = (
        statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else max(durations)
    )
    return {
        "n": len(results),
        "correctness": correctness,
        "factual_coverage": factual,
        "safety": safety,
        "tool_use": tool,
        "cost": cost_score,
        "latency": latency_score,
        "avg_cost_usd": statistics.mean(costs),
        "p95_latency_ms": p95_latency,
        "forbidden_any": forbidden_any,
    }


def gates_passed(agg: dict[str, Any]) -> tuple[bool, list[str]]:
    s = get_settings()
    failures: list[str] = []
    if agg["correctness"] < s.gate_min_correctness:
        failures.append(f"correctness {agg['correctness']:.3f} < {s.gate_min_correctness}")
    if agg["factual_coverage"] < s.gate_min_factual_coverage:
        failures.append(f"factual_coverage {agg['factual_coverage']:.3f} < {s.gate_min_factual_coverage}")
    if agg["safety"] < s.gate_min_safety:
        failures.append(f"safety {agg['safety']:.3f} < {s.gate_min_safety}")
    if agg["p95_latency_ms"] > s.gate_max_p95_latency_ms:
        failures.append(f"p95_latency_ms {agg['p95_latency_ms']:.0f} > {s.gate_max_p95_latency_ms}")
    if agg["avg_cost_usd"] > s.gate_max_avg_cost_usd:
        failures.append(f"avg_cost_usd {agg['avg_cost_usd']:.4f} > {s.gate_max_avg_cost_usd}")
    if agg["forbidden_any"]:
        failures.append("forbidden behavior observed in at least one case")
    return (not failures, failures)


def write_results(results: list[tuple[dict, RunResponse, CaseScore]], agg: dict[str, Any]) -> None:
    payload = {
        "aggregate": agg,
        "cases": [
            {
                "id": case["id"],
                "input": case["user_input"],
                "answer": (resp.final.answer if resp.final else None),
                "confidence": (resp.final.confidence if resp.final else None),
                "metrics": resp.metrics.model_dump(),
                "score": {
                    "correctness": s.correctness,
                    "factual_coverage": s.factual_coverage,
                    "safety": s.safety,
                    "tool_use": s.tool_use,
                    "cost": s.cost,
                    "latency": s.latency,
                    "forbidden_hits": s.forbidden_hits,
                    "passed": s.passed,
                },
            }
            for case, resp, s in results
        ],
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_report(agg: dict[str, Any], failures: list[str]) -> None:
    lines: list[str] = []
    lines.append("# Eval Report")
    lines.append("")
    lines.append(f"- mode: `MODEL_PROVIDER={os.getenv('MODEL_PROVIDER','mock')}`")
    lines.append(f"- cases: {agg['n']}")
    lines.append("")
    lines.append("## Aggregate scores")
    lines.append("")
    lines.append(f"| metric | value |")
    lines.append(f"| --- | --- |")
    lines.append(f"| correctness | {agg['correctness']:.3f} |")
    lines.append(f"| factual_coverage | {agg['factual_coverage']:.3f} |")
    lines.append(f"| safety | {agg['safety']:.3f} |")
    lines.append(f"| tool_use | {agg['tool_use']:.3f} |")
    lines.append(f"| cost_score | {agg['cost']:.3f} |")
    lines.append(f"| latency_score | {agg['latency']:.3f} |")
    lines.append(f"| avg_cost_usd | {agg['avg_cost_usd']:.4f} |")
    lines.append(f"| p95_latency_ms | {agg['p95_latency_ms']:.0f} |")
    lines.append("")
    if failures:
        lines.append("## ❌ Regression gates failed")
        lines.append("")
        for f in failures:
            lines.append(f"- {f}")
    else:
        lines.append("## ✅ All regression gates passed")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


async def main_async(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=DATASET_PATH)
    p.add_argument("--no-gate", action="store_true", help="don't fail on gate breach")
    args = p.parse_args(argv)

    cases = load_dataset(args.dataset)
    results = await run_all(cases)
    agg = aggregate(results)
    ok, failures = gates_passed(agg)

    write_results(results, agg)
    write_report(agg, failures)

    print(json.dumps({"aggregate": agg, "gates_passed": ok, "failures": failures}, indent=2))
    if ok or args.no_gate:
        return 0
    return 1


def main() -> int:
    return asyncio.run(main_async(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
