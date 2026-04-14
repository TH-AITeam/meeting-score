"""スコアリングモジュールのテスト"""

from app.schemas.models import Penalties, Scores
from app.scoring.calculator import calculate_total_score
from app.scoring.weights import ScoringWeights


def test_calculate_total_score_basic():
    """基本的なスコア計算"""
    scores = Scores(
        issue_clarification=3,
        decision_progress=2,
        risk_detection=0,
        actionability=0,
        groundedness=1,
        novelty=2,
        summarization=1,
    )
    penalties = Penalties(
        duplication=0,
        verbosity=0,
        off_topic=0,
        unsupported_assertion=0,
    )
    total = calculate_total_score(scores, penalties)
    # 3*1.3 + 2*1.5 + 0 + 0 + 1*0.8 + 2*0.9 + 1*0.8 = 3.9+3.0+0.8+1.8+0.8 = 10.3
    assert total == 10.3


def test_calculate_total_score_with_penalties():
    """減点ありのスコア計算"""
    scores = Scores(
        issue_clarification=0,
        decision_progress=0,
        risk_detection=0,
        actionability=0,
        groundedness=0,
        novelty=0,
        summarization=0,
    )
    penalties = Penalties(
        duplication=-2,
        verbosity=-1,
        off_topic=-3,
        unsupported_assertion=0,
    )
    total = calculate_total_score(scores, penalties)
    assert total == -6.0


def test_calculate_total_score_mixed():
    """加点と減点が混在するケース"""
    scores = Scores(
        issue_clarification=2,
        decision_progress=1,
        risk_detection=1,
        actionability=0,
        groundedness=0,
        novelty=1,
        summarization=0,
    )
    penalties = Penalties(
        duplication=-1,
        verbosity=-1,
        off_topic=0,
        unsupported_assertion=0,
    )
    total = calculate_total_score(scores, penalties)
    # 2*1.3 + 1*1.5 + 1*1.2 + 0 + 0 + 1*0.9 + 0 + (-1) + (-1) = 2.6+1.5+1.2+0.9-2 = 4.2
    assert total == 4.2


def test_all_zeros():
    """全ゼロの場合"""
    total = calculate_total_score(Scores(), Penalties())
    assert total == 0.0


def test_max_scores():
    """全て最高点の場合"""
    scores = Scores(
        issue_clarification=3,
        decision_progress=3,
        risk_detection=3,
        actionability=3,
        groundedness=3,
        novelty=3,
        summarization=3,
    )
    total = calculate_total_score(scores, Penalties())
    # 3*(1.3+1.5+1.2+1.3+0.8+0.9+0.8) = 3*7.8 = 23.4
    assert total == 23.4
