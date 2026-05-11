"""app.evaluators.llm_evaluator のテスト。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.context_builder.builder import EvaluationContext
from app.evaluators.llm_evaluator import evaluate_utterance
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
        agenda=["議題1"],
        decision_points=[],
        current_topic="議題1",
        before_utterances=[],
        target_utterance=target,
        after_utterances=[],
    )


_VALID_JSON = (
    '{"speech_type": "情報共有", '
    '"scores": {"issue_clarification": 0, "decision_progress": 0, '
    '"risk_detection": 0, "actionability": 0, "groundedness": 0, '
    '"novelty": 0, "summarization": 0}, '
    '"penalties": {"duplication": 0, "verbosity": 0, '
    '"off_topic": 0, "unsupported_assertion": 0}, '
    '"reason": "ok"}'
)


def test_evaluate_utterance_passes_temperature() -> None:
    create = Mock(return_value=SimpleNamespace(output_text=_VALID_JSON))
    client = SimpleNamespace(responses=SimpleNamespace(create=create))

    with patch("openai.OpenAI", return_value=client):
        result = evaluate_utterance(_make_ctx(), temperature=0.7)

    assert result["evaluation_failed"] is False
    assert create.call_args.kwargs["temperature"] == 0.7
