"""Rule-based corrections applied after LLM evaluation."""

from __future__ import annotations

from app.schemas.models import EvaluatedUtterance, Penalties
from app.scoring.calculator import calculate_total_score
from app.scoring.weights import ScoringWeights

_VERBOSITY_CHAR_THRESHOLD_MILD = 120
_VERBOSITY_CHAR_THRESHOLD_STRONG = 200
_DUPLICATE_OVERLAP_RATIO = 0.6


def _bigrams(text: str) -> set[str]:
    """Return character bigrams for a text."""
    if len(text) < 2:
        return set()
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _bigram_jaccard(a: str, b: str) -> float:
    """Calculate character-bigram Jaccard similarity."""
    if not a or not b:
        return 0.0
    set_a = _bigrams(a)
    set_b = _bigrams(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _check_duplication(
    target: EvaluatedUtterance,
    prior_texts: list[str],
) -> int:
    """Return an additional duplication penalty for repeated content."""
    for prior in prior_texts:
        if _bigram_jaccard(target.text, prior) >= _DUPLICATE_OVERLAP_RATIO:
            return -1
    return 0


def _check_verbosity(target: EvaluatedUtterance) -> int:
    """Return an additional verbosity penalty for long low-value utterances."""
    text_len = len(target.text)
    total_positive = (
        target.scores.issue_clarification
        + target.scores.decision_progress
        + target.scores.risk_detection
        + target.scores.actionability
        + target.scores.groundedness
        + target.scores.novelty
        + target.scores.summarization
    )
    if total_positive >= 4:
        return 0

    if text_len >= _VERBOSITY_CHAR_THRESHOLD_STRONG and total_positive <= 1:
        return -1
    if text_len >= _VERBOSITY_CHAR_THRESHOLD_MILD and total_positive == 0:
        return -1

    return 0


def apply_rule_corrections(
    evaluated: list[EvaluatedUtterance],
    weights: ScoringWeights | None = None,
) -> list[EvaluatedUtterance]:
    """Apply duplication and verbosity corrections, then recalculate totals."""
    corrected: list[EvaluatedUtterance] = []
    prior_texts: list[str] = []

    for eu in evaluated:
        dup_adj = _check_duplication(eu, prior_texts)
        verb_adj = _check_verbosity(eu)

        if dup_adj != 0 or verb_adj != 0:
            new_penalties = Penalties(
                duplication=max(eu.penalties.duplication + dup_adj, -3),
                verbosity=max(eu.penalties.verbosity + verb_adj, -3),
                off_topic=eu.penalties.off_topic,
                unsupported_assertion=eu.penalties.unsupported_assertion,
            )
            eu = eu.model_copy(
                update={
                    "penalties": new_penalties,
                    "total_score": calculate_total_score(eu.scores, new_penalties, weights),
                }
            )

        corrected.append(eu)
        prior_texts.append(eu.text)

    return corrected
