"""API route tests."""

from fastapi.testclient import TestClient

import app.services.analysis as analysis
from app.api.main import app
from app.evaluators.llm_evaluator import EvaluationResult
from app.schemas.models import Penalties, Scores, SpeechType


def _sample_payload() -> dict:
    return {
        "meeting_id": "m001",
        "title": "テスト会議",
        "goal": "初回リリース範囲を決める",
        "utterances": [
            {
                "utterance_id": "u001",
                "speaker": "A",
                "timestamp": "00:00:01",
                "text": "今日決めるべき範囲を確認しましょう。",
            }
        ],
    }


def test_analyze_returns_502_when_all_evaluations_fail(monkeypatch):
    """全発言の評価に失敗した場合は 502 を返す。"""

    def _fake_evaluate_utterance(*args, **kwargs):
        return EvaluationResult(
            speech_type=SpeechType.INFO_SHARING.value,
            scores=Scores(),
            penalties=Penalties(),
            reason="評価を取得できませんでした。",
            evaluation_failed=True,
        )

    monkeypatch.setattr(analysis, "evaluate_utterance", _fake_evaluate_utterance)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/analyze", json=_sample_payload())

    assert response.status_code == 502
    assert "LLM による評価がすべて失敗しました" in response.text


def test_analyze_returns_summary_when_evaluation_succeeds(monkeypatch):
    """評価が成功した場合は会議サマリーを返す。"""

    def _fake_evaluate_utterance(*args, **kwargs):
        return EvaluationResult(
            speech_type=SpeechType.ISSUE_CLARIFICATION.value,
            scores=Scores(issue_clarification=3, decision_progress=2),
            penalties=Penalties(),
            reason="論点を明確にし、意思決定を前に進めたため。",
        )

    monkeypatch.setattr(analysis, "evaluate_utterance", _fake_evaluate_utterance)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/analyze", json=_sample_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "テスト会議"
    assert len(data["evaluated_utterances"]) == 1
    assert data["evaluated_utterances"][0]["speech_type"] == "論点整理"


def test_analyze_returns_400_for_invalid_input():
    """入力データが不正な場合は 400 を返す。"""
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/analyze", json={"meeting_id": "m001"})

    assert response.status_code == 400


def test_static_ui_assets_are_served():
    """分割された静的UIファイルが配信される。"""
    with TestClient(app, raise_server_exceptions=False) as client:
        css_response = client.get("/css/app.css")
        js_response = client.get("/js/app.js")

    assert css_response.status_code == 200
    assert js_response.status_code == 200
