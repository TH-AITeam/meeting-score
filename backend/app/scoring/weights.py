"""スコアリング重み設定の読み込み"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


@dataclass
class ScoringWeights:
    """評価軸の重み"""
    # 主評価軸
    issue_clarification: float = 1.3
    decision_progress: float = 1.5
    risk_detection: float = 1.2
    actionability: float = 1.3
    # 補助評価軸
    groundedness: float = 0.8
    novelty: float = 0.9
    summarization: float = 0.8


@dataclass
class AppConfig:
    """アプリケーション全体設定"""
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    context_before: int = 3
    context_after: int = 3
    llm_model: str = "gpt-5.4-mini"
    llm_max_tokens: int = 1024
    llm_max_retries: int = 3
    top_utterances_count: int = 5
    top_per_axis_count: int = 3


def load_config(path: str | Path | None = None) -> AppConfig:
    """config.yaml から設定を読み込む"""
    path = Path(path) if path else _CONFIG_PATH

    if not path.exists():
        return AppConfig()

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    w = raw.get("weights", {})
    weights = ScoringWeights(
        issue_clarification=w.get("issue_clarification", 1.3),
        decision_progress=w.get("decision_progress", 1.5),
        risk_detection=w.get("risk_detection", 1.2),
        actionability=w.get("actionability", 1.3),
        groundedness=w.get("groundedness", 0.8),
        novelty=w.get("novelty", 0.9),
        summarization=w.get("summarization", 0.8),
    )

    ctx = raw.get("context", {})
    llm = raw.get("llm", {})
    agg = raw.get("aggregation", {})

    return AppConfig(
        weights=weights,
        context_before=ctx.get("before_count", 3),
        context_after=ctx.get("after_count", 3),
        llm_model=llm.get("model", "gpt-5.4-mini"),
        llm_max_tokens=llm.get("max_tokens", 1024),
        llm_max_retries=llm.get("max_retries", 3),
        top_utterances_count=agg.get("top_utterances_count", 5),
        top_per_axis_count=agg.get("top_per_axis_count", 3),
    )
