"""Scoring tests."""

from app.schemas.models import Penalties, Scores
from app.scoring.calculator import calculate_total_score


def test_calculate_total_score_basic():
    scores = Scores(
        issue_clarification=3,
        decision_progress=2,
        risk_detection=0,
        actionability=0,
        groundedness=1,
        novelty=2,
        summarization=1,
    )
    total = calculate_total_score(scores, Penalties())
    assert total == 10.3


def test_calculate_total_score_with_penalties():
    penalties = Penalties(
        duplication=-2,
        verbosity=-1,
        off_topic=-3,
        unsupported_assertion=0,
    )
    total = calculate_total_score(Scores(), penalties)
    assert total == -6.0


def test_calculate_total_score_mixed():
    scores = Scores(
        issue_clarification=2,
        decision_progress=1,
        risk_detection=1,
        actionability=0,
        groundedness=0,
        novelty=1,
        summarization=0,
    )
    penalties = Penalties(duplication=-1, verbosity=-1)
    total = calculate_total_score(scores, penalties)
    assert total == 4.2


def test_all_zeros():
    assert calculate_total_score(Scores(), Penalties()) == 0.0


def test_max_scores():
    scores = Scores(
        issue_clarification=3,
        decision_progress=3,
        risk_detection=3,
        actionability=3,
        groundedness=3,
        novelty=3,
        summarization=3,
    )
    assert calculate_total_score(scores, Penalties()) == 23.4
