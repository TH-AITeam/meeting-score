"""Rule correction tests."""

from app.schemas.models import EvaluatedUtterance, Penalties, Scores, SpeechType
from app.scoring.rule_corrections import apply_rule_corrections


def _make_eu(uid: str, text: str, penalties: Penalties | None = None, **score_kwargs) -> EvaluatedUtterance:
    scores = Scores(**{k: v for k, v in score_kwargs.items() if k in Scores.model_fields})
    if penalties is None:
        penalties = Penalties()
    return EvaluatedUtterance(
        utterance_id=uid,
        speaker="A",
        timestamp="00:00:00",
        text=text,
        speech_type=SpeechType.INFO_SHARING.value,
        scores=scores,
        penalties=penalties,
        total_score=0.0,
        reason="test",
    )


def test_duplicate_detection():
    evaluated = [
        _make_eu("u001", "社内利用を先に進めるべきです。セキュリティ要件が高いためです。"),
        _make_eu("u002", "社内利用を先に進めるべきです。セキュリティの要件が高いためです。"),
    ]
    corrected = apply_rule_corrections(evaluated)
    assert corrected[0].penalties.duplication == 0
    assert corrected[1].penalties.duplication <= -1


def test_verbosity_detection_long_no_value():
    long_text = "長い" * 130
    evaluated = [_make_eu("u001", long_text)]
    corrected = apply_rule_corrections(evaluated)
    assert corrected[0].penalties.verbosity <= -1


def test_no_verbosity_for_high_value():
    long_text = "長い" * 130
    evaluated = [_make_eu("u001", long_text, issue_clarification=3, decision_progress=2)]
    corrected = apply_rule_corrections(evaluated)
    assert corrected[0].penalties.verbosity == 0


def test_no_over_correction():
    penalties = Penalties(duplication=-3)
    evaluated = [
        _make_eu("u001", "先に述べた内容を繰り返します。"),
        _make_eu("u002", "先に述べた内容を繰り返します。", penalties=penalties),
    ]
    corrected = apply_rule_corrections(evaluated)
    assert corrected[1].penalties.duplication == -3


def test_different_topics_not_duplicate():
    evaluated = [
        _make_eu("u001", "今日はUIの色を決めましょう。"),
        _make_eu("u002", "今日は認証方式を決めましょう。"),
    ]
    corrected = apply_rule_corrections(evaluated)
    assert corrected[1].penalties.duplication == 0


def test_recalculates_total():
    long_text = "長い" * 130
    evaluated = [_make_eu("u001", long_text)]
    corrected = apply_rule_corrections(evaluated)
    assert corrected[0].total_score < 0
