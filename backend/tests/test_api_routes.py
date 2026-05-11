"""API ルートのテスト"""

from fastapi.testclient import TestClient

import app.api.routes as routes
from app.api.main import app
from app.evaluators.base import EvaluationResult, Evaluator
from app.schemas.models import Penalties, Scores


class _FakeEvaluator(Evaluator):
    """テスト用 Evaluator。固定の EvaluationResult を返す。"""

    def __init__(self, result: EvaluationResult) -> None:
        self._result = result

    def evaluate(self, ctx):  # noqa: ANN001
        return self._result


def _patch_evaluator(monkeypatch, result: EvaluationResult) -> None:
    """routes.create_evaluator をモック化して固定 Evaluator を返す。"""
    monkeypatch.setattr(
        routes, "create_evaluator", lambda config: _FakeEvaluator(result)
    )


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
                "text": "今決めるべきは範囲です。",
            }
        ],
    }


def test_analyze_returns_502_when_all_evaluations_fail(monkeypatch):
    """全発言の評価失敗時は 502 を返す"""
    _patch_evaluator(
        monkeypatch,
        EvaluationResult(
            speech_type="情報共有",
            scores=Scores(),
            penalties=Penalties(),
            reason="評価を取得できませんでした。",
            evaluation_failed=True,
        ),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/analyze", json=_sample_payload())

    assert response.status_code == 502
    assert "LLM による評価がすべて失敗しました" in response.text


def test_analyze_returns_summary_when_evaluation_succeeds(monkeypatch):
    """評価が成功した場合は会議サマリーを返す"""
    _patch_evaluator(
        monkeypatch,
        EvaluationResult(
            speech_type="論点整理",
            scores=Scores(issue_clarification=3, decision_progress=2),
            penalties=Penalties(),
            reason="論点を明確にし、意思決定を前に進めた。",
            evaluation_failed=False,
        ),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/analyze", json=_sample_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "テスト会議"
    assert len(data["evaluated_utterances"]) == 1
    assert data["evaluated_utterances"][0]["speech_type"] == "論点整理"
