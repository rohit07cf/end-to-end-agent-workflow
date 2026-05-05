"""Realistic, deterministic mock tools so the system runs end-to-end without
external network calls. In production, replace bodies with real integrations
(search index, calculator, KB), but keep the function signatures stable."""

from __future__ import annotations

import math
import random

from agents import function_tool
from pydantic import BaseModel


class SearchHit(BaseModel):
    title: str
    url: str
    snippet: str


_KNOWLEDGE_BASE: dict[str, list[SearchHit]] = {
    "france": [
        SearchHit(
            title="France — Country Profile",
            url="kb://geo/france",
            snippet="France's capital is Paris. Population ~67M.",
        )
    ],
    "speed of light": [
        SearchHit(
            title="Physical Constants",
            url="kb://physics/c",
            snippet="The speed of light c = 299,792,458 m/s.",
        )
    ],
    "python": [
        SearchHit(
            title="Python Programming Language",
            url="kb://tech/python",
            snippet="Python is a high-level, interpreted language created by Guido van Rossum.",
        )
    ],
}


@function_tool
def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Search the (mock) public web index. Returns title/url/snippet hits."""
    q = (query or "").lower()
    out: list[SearchHit] = []
    for k, hits in _KNOWLEDGE_BASE.items():
        if k in q:
            out.extend(hits)
    if not out:
        # Deterministic stub result so callers can still cite *something*.
        out = [
            SearchHit(
                title=f"Search results for: {query}",
                url=f"kb://search/{abs(hash(q)) % 10_000:04d}",
                snippet="No high-confidence match found.",
            )
        ]
    return [h.model_dump() for h in out[:max_results]]


@function_tool
def calculator(expression: str) -> str:
    """Evaluate a SAFE arithmetic expression. Only digits, ops, parens, and the
    `pi` / `e` / `sqrt` symbols are allowed. Returns the numeric result."""
    allowed = set("0123456789+-*/().,% ")
    if not all(c in allowed or c.isalpha() for c in expression):
        raise ValueError("disallowed characters in expression")
    safe_globals = {"__builtins__": {}}
    safe_locals = {"pi": math.pi, "e": math.e, "sqrt": math.sqrt}
    try:
        result = eval(expression, safe_globals, safe_locals)  # noqa: S307 — sandboxed
    except Exception as e:  # pragma: no cover — surfaced to model
        raise ValueError(f"calculator error: {e}") from e
    return str(result)


@function_tool
def knowledge_lookup(topic: str) -> dict:
    """Look up a curated knowledge-base entry. Returns {found, entries}."""
    key = (topic or "").lower().strip()
    for k, hits in _KNOWLEDGE_BASE.items():
        if k in key:
            return {"found": True, "entries": [h.model_dump() for h in hits]}
    return {"found": False, "entries": []}


@function_tool
def flaky_tool(seed: int = 0) -> str:
    """Tool that fails ~50% of the time. Used by integration tests to exercise
    the tool-failure fallback path."""
    rng = random.Random(seed)
    if rng.random() < 0.5:
        raise RuntimeError("flaky tool transient failure")
    return "ok"


# Public list — workflow.py wires these into the Agent.
ALL_TOOLS = [web_search, calculator, knowledge_lookup]
