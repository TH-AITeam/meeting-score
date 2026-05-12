"""/upload_audio エンドポイントのテスト (Issue #11)。

`transcribe_to_meeting_input` を monkeypatch して、HTTP 層 (multipart 解釈、
拡張子バリデーション、エラー応答) を検証する。実 WhisperX / pyannote は呼ばない。
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.audio_routes import router as audio_router
from app.schemas.models import MeetingInput
from app.schemas.models import Utterance as SchemaUtterance
from app.scoring.weights import AppConfig, PenaltyWeights, ScoringWeights


def _make_app(monkeypatch_target_module: Any, fake_result: MeetingInput) -> FastAPI:
    """テスト用最小 FastAPI アプリ。config を inject、transcribe を stub。"""
    app = FastAPI()
    app.include_router(audio_router, prefix="/api")
    app.state.config = AppConfig(
        weights=ScoringWeights(),
        penalty_weights=PenaltyWeights(),
        llm_backend="local",
        llm_model="dummy",
    )
    app.state.config_path = None

    def _fake(*args, **kwargs):
        return fake_result

    monkeypatch_target_module.setattr("app.api.audio_routes.transcribe_to_meeting_input", _fake)
    return app


def _dummy_meeting_input(meeting_id: str = "m001") -> MeetingInput:
    return MeetingInput(
        meeting_id=meeting_id,
        title="サンプル",
        goal="ゴール",
        agenda=["議題A"],
        decision_points=[],
        utterances=[
            SchemaUtterance(
                utterance_id="u001",
                speaker="SPEAKER_00",
                timestamp="00:00:00",
                text="こんにちは",
            )
        ],
    )


def test_upload_audio_returns_meeting_json(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _make_app(monkeypatch, _dummy_meeting_input("m042"))
    client = TestClient(app)
    response = client.post(
        "/api/upload_audio",
        files={"file": ("meeting.wav", BytesIO(b"FAKE_AUDIO"), "audio/wav")},
        data={"meeting_id": "m042"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["meeting_id"] == "m042"
    assert body["title"] == "サンプル"
    assert len(body["utterances"]) == 1


def test_upload_audio_rejects_unknown_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _make_app(monkeypatch, _dummy_meeting_input())
    client = TestClient(app)
    response = client.post(
        "/api/upload_audio",
        files={"file": ("readme.txt", BytesIO(b"x"), "text/plain")},
    )
    assert response.status_code == 400
    assert "サポート外" in response.json()["detail"]


def test_upload_audio_generates_meeting_id_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """meeting_id 未指定でも m_<uuid8> で生成して 200 を返す。"""
    captured: dict[str, Any] = {}

    def _fake(audio_path, **kwargs):
        captured["meeting_id"] = kwargs["meeting_id"]
        return _dummy_meeting_input(kwargs["meeting_id"])

    app = FastAPI()
    app.include_router(audio_router, prefix="/api")
    app.state.config = AppConfig(
        weights=ScoringWeights(),
        penalty_weights=PenaltyWeights(),
        llm_model="dummy",
    )
    monkeypatch.setattr("app.api.audio_routes.transcribe_to_meeting_input", _fake)

    client = TestClient(app)
    response = client.post(
        "/api/upload_audio",
        files={"file": ("a.mp3", BytesIO(b"FAKE"), "audio/mpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["meeting_id"].startswith("m_")
    assert len(body["meeting_id"]) == len("m_") + 8


def test_upload_audio_passes_form_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """form の title / goal / num_speakers / no_meta_extract が transcribe に渡る。"""
    captured: dict[str, Any] = {}

    def _fake(audio_path, **kwargs):
        captured.update(kwargs)
        return _dummy_meeting_input(kwargs["meeting_id"])

    app = FastAPI()
    app.include_router(audio_router, prefix="/api")
    app.state.config = AppConfig(
        weights=ScoringWeights(),
        penalty_weights=PenaltyWeights(),
        llm_model="dummy",
    )
    monkeypatch.setattr("app.api.audio_routes.transcribe_to_meeting_input", _fake)

    client = TestClient(app)
    response = client.post(
        "/api/upload_audio",
        files={"file": ("a.wav", BytesIO(b"x"), "audio/wav")},
        data={
            "meeting_id": "m007",
            "title": "T",
            "goal": "G",
            "num_speakers": "3",
            "no_meta_extract": "true",
        },
    )
    assert response.status_code == 200
    assert captured["meeting_id"] == "m007"
    assert captured["default_title"] == "T"
    assert captured["default_goal"] == "G"
    assert captured["num_speakers"] == 3
    assert captured["use_meta_extractor"] is False


def test_upload_audio_returns_500_on_processing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """transcribe が例外を投げたら 500 + 詳細メッセージ。"""

    def _raise(*args, **kwargs):
        msg = "load failed"
        raise RuntimeError(msg)

    app = FastAPI()
    app.include_router(audio_router, prefix="/api")
    app.state.config = AppConfig(
        weights=ScoringWeights(),
        penalty_weights=PenaltyWeights(),
        llm_model="dummy",
    )
    monkeypatch.setattr("app.api.audio_routes.transcribe_to_meeting_input", _raise)

    client = TestClient(app)
    response = client.post(
        "/api/upload_audio",
        files={"file": ("a.wav", BytesIO(b"x"), "audio/wav")},
    )
    assert response.status_code == 500
    assert "音声処理に失敗" in response.json()["detail"]


def test_load_audio_section_helper_reads_yaml(tmp_path: Path) -> None:
    from app.api.audio_routes import _load_audio_section_from_app_state

    cfg = tmp_path / "config.yaml"
    cfg.write_text("audio:\n  asr:\n    device: cpu\n", encoding="utf-8")

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(config_path=str(cfg))))
    sec = _load_audio_section_from_app_state(request)
    assert sec["asr"]["device"] == "cpu"
