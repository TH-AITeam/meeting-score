"""Scoring weight and application configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


@dataclass
class ScoringWeights:
    """Weights for positive scoring axes."""

    issue_clarification: float = 1.3
    decision_progress: float = 1.5
    risk_detection: float = 1.2
    actionability: float = 1.3
    groundedness: float = 0.8
    novelty: float = 0.9
    summarization: float = 0.8


@dataclass
class AppConfig:
    """Application configuration."""

    weights: ScoringWeights = field(default_factory=ScoringWeights)
    context_before: int = 3
    context_after: int = 3
    llm_model: str = "gpt-5.4-mini"
    llm_max_tokens: int = 1024
    llm_max_retries: int = 3
    top_utterances_count: int = 5
    top_per_axis_count: int = 3


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load config.yaml into AppConfig."""
    config_path = Path(path) if path else _CONFIG_PATH

    if not config_path.exists():
        return AppConfig()

    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    weights_raw = raw.get("weights", {})
    weights = ScoringWeights(
        issue_clarification=weights_raw.get("issue_clarification", 1.3),
        decision_progress=weights_raw.get("decision_progress", 1.5),
        risk_detection=weights_raw.get("risk_detection", 1.2),
        actionability=weights_raw.get("actionability", 1.3),
        groundedness=weights_raw.get("groundedness", 0.8),
        novelty=weights_raw.get("novelty", 0.9),
        summarization=weights_raw.get("summarization", 0.8),
    )

    context_raw = raw.get("context", {})
    llm_raw = raw.get("llm", {})
    aggregation_raw = raw.get("aggregation", {})

    return AppConfig(
        weights=weights,
        context_before=context_raw.get("before_count", 3),
        context_after=context_raw.get("after_count", 3),
        llm_model=llm_raw.get("model", "gpt-5.4-mini"),
        llm_max_tokens=llm_raw.get("max_tokens", 1024),
        llm_max_retries=llm_raw.get("max_retries", 3),
        top_utterances_count=aggregation_raw.get("top_utterances_count", 5),
        top_per_axis_count=aggregation_raw.get("top_per_axis_count", 3),
    )
