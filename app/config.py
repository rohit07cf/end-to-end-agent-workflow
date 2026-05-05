"""Centralised settings. All knobs live here so production deployments tune
behavior via env vars without code changes."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Model / provider
    model_provider: Literal["claude", "mock"] = "mock"
    claude_model: str = "claude-opus-4-7"
    claude_fallback_model: str = "claude-haiku-4-5-20251001"
    anthropic_api_key: str | None = None

    # Budgets — enforced per-run
    max_total_tokens_per_run: int = 20_000
    max_cost_usd_per_run: float = 0.50
    max_latency_ms_per_run: int = 45_000
    max_tool_calls_per_run: int = 8

    # Regression gates
    gate_min_correctness: float = 0.85
    gate_min_factual_coverage: float = 0.85
    gate_min_safety: float = 0.95
    gate_max_p95_latency_ms: int = 20_000
    gate_max_avg_cost_usd: float = 0.10

    # Observability
    log_level: str = "INFO"
    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "research-assistant-agent"
    prometheus_enabled: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8080

    # Retries / fallback
    retry_max_attempts: int = 3
    retry_initial_backoff_s: float = 0.5

    # Per-1M-token pricing table (USD). Keep small + auditable.
    # Values approximate public list pricing; override via code if needed.
    model_pricing_usd_per_mtok: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "claude-opus-4-7": {"input": 15.0, "output": 75.0},
            "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
            "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
            "mock": {"input": 0.0, "output": 0.0},
        }
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
