"""集計モジュールのテスト"""

from app.aggregation.aggregator import (
    aggregate_by_speaker,
    extract_top_by_axis,
    extract_top_utterances,
)
from app.schemas.models import EvaluatedUtterance, Penalties, Scores


def _make_eu(uid: str, speaker: str, total: float, **kwargs) -> EvaluatedUtterance:
    scores = Scores(**{k: v for k, v in kwargs.items() if k in Scores.model_fields})
    return EvaluatedUtterance(
        utterance_id=uid,
        speaker=speaker,
        timestamp="00:00:00",
        text=f"テスト発言 {uid}",
        speech_type="情報共有",
        scores=scores,
        penalties=Penalties(),
        total_score=total,
        reason="テスト",
    )


def test_extract_top_utterances():
    evaluated = [
        _make_eu("u001", "A", 5.0),
        _make_eu("u002", "B", 8.0),
        _make_eu("u003", "A", 3.0),
        _make_eu("u004", "B", 10.0),
    ]
    top = extract_top_utterances(evaluated, top_count=2)
    assert len(top) == 2
    assert top[0].utterance_id == "u004"
    assert top[1].utterance_id == "u002"


def test_extract_top_by_axis():
    evaluated = [
        _make_eu("u001", "A", 5.0, issue_clarification=3),
        _make_eu("u002", "B", 8.0, issue_clarification=1),
        _make_eu("u003", "A", 3.0, issue_clarification=2),
    ]
    top = extract_top_by_axis(evaluated, "issue_clarification", top_count=2)
    assert top[0].utterance_id == "u001"
    assert top[1].utterance_id == "u003"


def test_extract_top_by_axis_excludes_zero():
    """スコア0の発言は軸別Topから除外される"""
    evaluated = [
        _make_eu("u001", "A", 5.0, risk_detection=2),
        _make_eu("u002", "B", 8.0, risk_detection=0),
        _make_eu("u003", "A", 3.0, risk_detection=0),
    ]
    top = extract_top_by_axis(evaluated, "risk_detection", top_count=3)
    assert len(top) == 1
    assert top[0].utterance_id == "u001"


def test_aggregate_by_speaker():
    evaluated = [
        _make_eu("u001", "A", 6.0, issue_clarification=2, decision_progress=1),
        _make_eu("u002", "A", 4.0, issue_clarification=1, decision_progress=2),
        _make_eu("u003", "B", 8.0, issue_clarification=3, decision_progress=3),
    ]
    summaries = aggregate_by_speaker(evaluated)
    assert len(summaries) == 2

    a = next(s for s in summaries if s.speaker == "A")
    assert a.utterance_count == 2
    assert a.total_contribution_score == 10.0
    assert a.average_total_score == 5.0
    assert a.average_scores.issue_clarification == 1.5
