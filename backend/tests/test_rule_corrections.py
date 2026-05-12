"""ルールベース補正のテスト"""

from app.schemas.models import EvaluatedUtterance, Penalties, Scores
from app.scoring.rule_corrections import apply_rule_corrections
from app.scoring.weights import PenaltyWeights


def _make_eu(uid: str, text: str, penalties: Penalties | None = None, **score_kwargs) -> EvaluatedUtterance:
    scores = Scores(**{k: v for k, v in score_kwargs.items() if k in Scores.model_fields})
    if penalties is None:
        penalties = Penalties()
    return EvaluatedUtterance(
        utterance_id=uid,
        speaker="A",
        timestamp="00:00:00",
        text=text,
        speech_type="情報共有",
        scores=scores,
        penalties=penalties,
        total_score=0.0,
        reason="test",
    )


def test_duplicate_detection():
    """文字集合の重複が高い発言に追加減点される"""
    evaluated = [
        _make_eu("u001", "社内利用を先にすべきだと思います。セキュリティ要件が高いため。"),
        _make_eu("u002", "社内利用を先にすべきだと思います。セキュリティの要件が高いため。"),
    ]
    corrected = apply_rule_corrections(evaluated)
    # 2件目は重複減点が追加される
    assert corrected[0].penalties.duplication == 0
    assert corrected[1].penalties.duplication <= -1


def test_verbosity_detection_long_no_value():
    """長いのに加点がない発言は冗長減点される"""
    long_text = "あ" * 250  # 200文字超え、加点0
    evaluated = [_make_eu("u001", long_text)]
    corrected = apply_rule_corrections(evaluated)
    assert corrected[0].penalties.verbosity <= -1


def test_no_verbosity_for_high_value():
    """加点が高い発言は長くても冗長減点されない"""
    long_text = "あ" * 250
    evaluated = [_make_eu("u001", long_text, issue_clarification=3, decision_progress=2)]
    corrected = apply_rule_corrections(evaluated)
    assert corrected[0].penalties.verbosity == 0


def test_no_over_correction():
    """LLM が既に -3 を付けている場合、-3 を超えない"""
    penalties = Penalties(duplication=-3)
    evaluated = [
        _make_eu("u001", "先に述べた内容を繰り返します"),
        _make_eu("u002", "先に述べた内容を繰り返します", penalties=penalties),
    ]
    corrected = apply_rule_corrections(evaluated)
    assert corrected[1].penalties.duplication == -3  # -3 + (-1) は -3 に留まる


def test_different_topics_not_duplicate():
    """別論点の発言は構文が似ていても重複扱いしない"""
    evaluated = [
        _make_eu("u001", "今日はUIの色を決めましょう。"),
        _make_eu("u002", "今日は認証方式を決めましょう。"),
    ]
    corrected = apply_rule_corrections(evaluated)
    # 別の論点なので重複減点されない
    assert corrected[1].penalties.duplication == 0


def test_recalculates_total():
    """補正後に総合スコアが再計算される"""
    long_text = "あ" * 250
    evaluated = [_make_eu("u001", long_text)]
    corrected = apply_rule_corrections(evaluated)
    # 全スコア0 で冗長 -1 なので total は -1.0
    assert corrected[0].total_score < 0


def test_penalty_weights_propagate_through_corrections():
    """補正後の total 再計算に penalty_weights が反映される (Issue #3)"""
    long_text = "あ" * 250  # verbosity 補正で -1 が入る
    evaluated = [_make_eu("u001", long_text)]

    # 既定 (weight=1.0): verbosity -1 → total -1.0
    baseline = apply_rule_corrections(evaluated)
    assert baseline[0].penalties.verbosity == -1
    assert baseline[0].total_score == -1.0

    # 重み 3.0: -1 * 3.0 → total -3.0
    weighted = apply_rule_corrections(
        evaluated, penalty_weights=PenaltyWeights(verbosity=3.0)
    )
    assert weighted[0].penalties.verbosity == -1  # penalty 値そのものは変化しない
    assert weighted[0].total_score == -3.0
