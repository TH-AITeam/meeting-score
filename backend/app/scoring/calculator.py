"""総合スコア計算モジュール

重み付き総合点 =
  論点整理 × w_issue_clarification
  + 意思決定寄与 × w_decision_progress
  + リスク検知 × w_risk_detection
  + アクション化 × w_actionability
  + 根拠性 × w_groundedness
  + 新規性 × w_novelty
  + 要約・交通整理 × w_summarization
  + 重複 × pw_duplication
  + 冗長 × pw_verbosity
  + 論点逸脱 × pw_off_topic
  + 根拠薄い断言 × pw_unsupported_assertion

penalty 値は元々負値 (0〜-3)、penalty_weights は正の倍率。
"""

from __future__ import annotations

from app.schemas.models import Penalties, Scores
from app.scoring.weights import PenaltyWeights, ScoringWeights


def calculate_total_score(
    scores: Scores,
    penalties: Penalties,
    weights: ScoringWeights | None = None,
    penalty_weights: PenaltyWeights | None = None,
) -> float:
    """重み付き総合スコアを計算する。

    Parameters
    ----------
    scores : Scores
        各加点軸 (0〜3)
    penalties : Penalties
        各減点軸 (0〜-3、負値)
    weights : ScoringWeights | None
        加点軸の重み。None なら既定値。
    penalty_weights : PenaltyWeights | None
        減点軸の重み（正の倍率）。None なら既定値（全 1.0）。
    """
    if weights is None:
        weights = ScoringWeights()
    if penalty_weights is None:
        penalty_weights = PenaltyWeights()

    total = (
        scores.issue_clarification * weights.issue_clarification
        + scores.decision_progress * weights.decision_progress
        + scores.risk_detection * weights.risk_detection
        + scores.actionability * weights.actionability
        + scores.groundedness * weights.groundedness
        + scores.novelty * weights.novelty
        + scores.summarization * weights.summarization
        + penalties.duplication * penalty_weights.duplication
        + penalties.verbosity * penalty_weights.verbosity
        + penalties.off_topic * penalty_weights.off_topic
        + penalties.unsupported_assertion * penalty_weights.unsupported_assertion
    )

    return round(total, 1)
