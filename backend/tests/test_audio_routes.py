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
    """テスト用最小 FastAPI アプリ。config を inject、transcribe / normalize を stub。"""
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

    # Issue #68: normalize_to_wav は本物の ffmpeg を呼ぶので mock で短絡
    def _fake_normalize(input_path, output_path, **kwargs):
        output_path.write_bytes(b"FAKE_WAV")
        return output_path

    monkeypatch_target_module.setattr("app.api.audio_routes.normalize_to_wav", _fake_normalize)
    return app


def _stub_normalize(monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #68: normalize_to_wav は ffmpeg を呼ぶので mock で短絡する共通ヘルパ。"""

    def _fake_normalize(input_path, output_path, **kwargs):
        output_path.write_bytes(b"FAKE_WAV")
        return output_path

    monkeypatch.setattr("app.api.audio_routes.normalize_to_wav", _fake_normalize)


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
    _stub_normalize(monkeypatch)

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
    _stub_normalize(monkeypatch)

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


def test_upload_audio_runs_processing_in_threadpool(monkeypatch: pytest.MonkeyPatch) -> None:
    """重い音声処理本体は event loop 上ではなく threadpool 経由で呼ぶ。"""
    captured: dict[str, Any] = {}

    def _fake(audio_path, **kwargs):
        return _dummy_meeting_input(kwargs["meeting_id"])

    async def _fake_threadpool(func, *args, **kwargs):
        # transcribe_to_meeting_input は kwargs に meeting_id を持つ。
        # normalize_to_wav 呼び出しでは meeting_id 無し (route 内の 2 つ目の呼び出しが transcribe)
        if "meeting_id" in kwargs:
            captured["func"] = func
            captured["meeting_id"] = kwargs["meeting_id"]
        return func(*args, **kwargs)

    app = FastAPI()
    app.include_router(audio_router, prefix="/api")
    app.state.config = AppConfig(
        weights=ScoringWeights(),
        penalty_weights=PenaltyWeights(),
        llm_model="dummy",
    )
    monkeypatch.setattr("app.api.audio_routes.transcribe_to_meeting_input", _fake)
    monkeypatch.setattr("app.api.audio_routes.run_in_threadpool", _fake_threadpool)
    _stub_normalize(monkeypatch)

    client = TestClient(app)
    response = client.post(
        "/api/upload_audio",
        files={"file": ("a.wav", BytesIO(b"x"), "audio/wav")},
        data={"meeting_id": "m_thread"},
    )

    assert response.status_code == 200
    assert captured["func"] is _fake
    assert captured["meeting_id"] == "m_thread"


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
    _stub_normalize(monkeypatch)

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


# --------------------------------------------------------------------------
# Issue #68: 動画拒否 / webm 受付 / 正規化失敗のケース
# --------------------------------------------------------------------------


def test_upload_audio_rejects_video_with_415(monkeypatch: pytest.MonkeyPatch) -> None:
    """動画ファイルは 415 で拒否し、frontend 抽出案内を返す。"""
    app = _make_app(monkeypatch, _dummy_meeting_input())
    client = TestClient(app)
    for ext, mime in [
        (".mp4", "video/mp4"),
        (".mov", "video/quicktime"),
        (".mkv", "video/x-matroska"),
        (".avi", "video/x-msvideo"),
    ]:
        response = client.post(
            "/api/upload_audio",
            files={"file": (f"meeting{ext}", BytesIO(b"FAKE_VIDEO"), mime)},
        )
        assert response.status_code == 415, f"{ext} should be 415"
        detail = response.json()["detail"]
        assert "動画" in detail
        # 案内文に frontend 抽出 / ffmpeg コマンド例が入る
        assert "frontend" in detail or "ffmpeg" in detail


def test_upload_audio_accepts_webm_opus(monkeypatch: pytest.MonkeyPatch) -> None:
    """frontend 抽出で生成される webm/opus を受け付ける (Issue #68 主経路)。"""
    app = _make_app(monkeypatch, _dummy_meeting_input("m_webm"))
    client = TestClient(app)
    response = client.post(
        "/api/upload_audio",
        files={"file": ("extracted.webm", BytesIO(b"OPUS_FAKE"), "audio/webm")},
        data={"meeting_id": "m_webm"},
    )
    assert response.status_code == 200
    assert response.json()["meeting_id"] == "m_webm"


def test_upload_audio_accepts_opus_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    """`.opus` も許容拡張子に入る。"""
    app = _make_app(monkeypatch, _dummy_meeting_input("m_opus"))
    client = TestClient(app)
    response = client.post(
        "/api/upload_audio",
        files={"file": ("a.opus", BytesIO(b"OPUS"), "audio/ogg")},
    )
    assert response.status_code == 200


def test_upload_audio_returns_500_when_normalize_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """normalize_to_wav (ffmpeg) が失敗したら 500 + 案内を返す。"""
    from app.asr.media import MediaError

    app = FastAPI()
    app.include_router(audio_router, prefix="/api")
    app.state.config = AppConfig(
        weights=ScoringWeights(),
        penalty_weights=PenaltyWeights(),
        llm_model="dummy",
    )
    app.state.config_path = None

    def _raise_media(*args, **kwargs):
        msg = "ffmpeg failed"
        raise MediaError(msg)

    monkeypatch.setattr("app.api.audio_routes.normalize_to_wav", _raise_media)
    # transcribe には到達しない想定だが念のため stub
    monkeypatch.setattr(
        "app.api.audio_routes.transcribe_to_meeting_input",
        lambda *a, **k: _dummy_meeting_input(),
    )

    client = TestClient(app)
    response = client.post(
        "/api/upload_audio",
        files={"file": ("a.webm", BytesIO(b"OPUS_FAKE"), "audio/webm")},
    )
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "正規化" in detail or "ffmpeg" in detail
