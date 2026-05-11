"""総合スコア計算モジュール

重み付き総合点 =
  論点整理 × 1.3
  + 意思決定寄与 × 1.5
  + リスク検知 × 1.2
  + アクション化 × 1.3
  + 根拠性 × 0.8
  + 新規性 × 0.9
  + 要約・交通整理 × 0.8
  + 重複 + 冗長 + 論点逸脱 + 根拠薄い断言
"""

from __future__ import annotations

from app.schemas.models import Penalties, Scores
from app.scoring.weights import ScoringWeights


def calculate_total_score(
    scores: Scores,
    penalties: Penalties,
    weights: ScoringWeights | None = None,
) -> float:
    """重み付き総合スコアを計算する"""
    if weights is None:
        weights = ScoringWeights()

    total = (
        scores.issue_clarification * weights.issue_clarification
        + scores.decision_progress * weights.decision_progress
        + scores.risk_detection * weights.risk_detection
        + scores.actionability * weights.actionability
        + scores.groundedness * weights.groundedness
        + scores.novelty * weights.novelty
        + scores.summarization * weights.summarization
        + penalties.duplication
        + penalties.verbosity
        + penalties.off_topic
        + penalties.unsupported_assertion
    )

    return round(total, 1)
