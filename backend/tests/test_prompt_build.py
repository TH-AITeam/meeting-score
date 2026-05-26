"""プロンプト構築のテスト — JSON braces で KeyError にならないことを検証"""

from app.context_builder.builder import EvaluationContext
from app.evaluators.llm_evaluator import _build_prompt
from app.schemas.models import Utterance


def _make_ctx() -> EvaluationContext:
    target = Utterance(
        utterance_id="u001",
        speaker="田中",
        timestamp="00:01:00",
        text="テスト発言です。",
    )
    return EvaluationContext(
        meeting_goal="テスト目的",
        agenda=["議題1", "議題2"],
        decision_points=["決定1"],
        current_topic="議題1",
        before_utterances=[],
        target_utterance=target,
        after_utterances=[],
    )


def test_build_prompt_no_key_error():
    """テンプレート内の JSON {} で KeyError にならない"""
    prompt = _build_prompt(_make_ctx())
    # 置換がすべて完了していること
    assert "テスト目的" in prompt
    assert "田中" in prompt
    assert "テスト発言です。" in prompt
    assert "議題1" in prompt
    # JSON 例がそのまま残っていること
    assert '"speech_type"' in prompt
    assert '"issue_clarification"' in prompt


def test_build_prompt_contains_current_topic():
    """current_topic がプロンプトに含まれる"""
    prompt = _build_prompt(_make_ctx())
    assert "現在の議題: 議題1" in prompt


def test_build_prompt_uses_unknown_topic_label_when_current_topic_empty():
    """current_topic が空なら議題不明としてプロンプトに入る"""
    ctx = _make_ctx()
    ctx.current_topic = ""

    prompt = _build_prompt(ctx)

    assert "現在の議題: (議題不明)" in prompt
