"""Analysis service tests."""

import asyncio

import pytest

import app.services.analysis as analysis
from app.evaluators.llm_evaluator import EvaluationResult
from app.schemas.models import MeetingInput, Penalties, Scores, SpeechType, Utterance
from app.scoring.weights import AppConfig


def _meeting() -> MeetingInput:
    return MeetingInput(
        meeting_id="m001",
        title="企画会議",
        goal="リリース範囲を決める",
        utterances=[
            Utterance(
                utterance_id="u001",
                speaker="A",
                timestamp="00:00:01",
                text="対象ユーザーを先に確認しましょう。",
            )
        ],
    )


def test_run_analysis_uses_explicit_config(monkeypatch):
    """run_analysis は明示的に渡された AppConfig で分析する。"""

    def _fake_evaluate_utterance(*args, **kwargs):
        assert kwargs["model"] == "test-model"
        return EvaluationResult(
            speech_type=SpeechType.ISSUE_CLARIFICATION.value,
            scores=Scores(issue_clarification=3),
            penalties=Penalties(),
            reason="論点を明確にしたため。",
        )

    monkeypatch.setattr(analysis, "evaluate_utterance", _fake_evaluate_utterance)

    result = asyncio.run(analysis.run_analysis(_meeting(), AppConfig(llm_model="test-model")))

    assert result["meeting_id"] == "m001"
    assert result["evaluated_utterances"][0]["speech_type"] == "論点整理"


def test_run_analysis_raises_when_all_evaluations_fail(monkeypatch):
    """すべての評価に失敗した場合はサービス例外を投げる。"""

    def _fake_evaluate_utterance(*args, **kwargs):
        return EvaluationResult(
            speech_type=SpeechType.INFO_SHARING.value,
            scores=Scores(),
            penalties=Penalties(),
            reason="評価を取得できませんでした。",
            evaluation_failed=True,
        )

    monkeypatch.setattr(analysis, "evaluate_utterance", _fake_evaluate_utterance)

    with pytest.raises(analysis.AnalysisEvaluationError):
        asyncio.run(analysis.run_analysis(_meeting(), AppConfig()))
