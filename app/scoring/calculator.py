"""Total score calculation."""

from __future__ import annotations

from app.schemas.models import Penalties, Scores
from app.scoring.weights import ScoringWeights


def calculate_total_score(
    scores: Scores,
    penalties: Penalties,
    weights: ScoringWeights | None = None,
) -> float:
    """Calculate weighted total score from axis scores and penalties."""
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
