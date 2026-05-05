"""USD cost calculator. Pricing table lives in settings so it can be updated
without a code release."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings, get_settings


@dataclass(frozen=True)
class CostBreakdown:
    model: str
    tokens_in: int
    tokens_out: int
    input_usd: float
    output_usd: float
    total_usd: float


def estimate_cost(
    model: str, tokens_in: int, tokens_out: int, settings: Settings | None = None
) -> CostBreakdown:
    s = settings or get_settings()
    table = s.model_pricing_usd_per_mtok
    # Look up by exact id, else by canonical prefix (e.g. claude-opus-4-7-20251101 -> claude-opus-4-7).
    pricing = table.get(model)
    if pricing is None:
        for known, p in table.items():
            if model.startswith(known):
                pricing = p
                break
    if pricing is None:
        pricing = {"input": 0.0, "output": 0.0}

    input_usd = (tokens_in / 1_000_000) * pricing["input"]
    output_usd = (tokens_out / 1_000_000) * pricing["output"]
    return CostBreakdown(
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        input_usd=round(input_usd, 6),
        output_usd=round(output_usd, 6),
        total_usd=round(input_usd + output_usd, 6),
    )
