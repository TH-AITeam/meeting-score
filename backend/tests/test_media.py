"""media.py のテスト (Issue #68)。

subprocess.run と shutil.which を mock し、実 ffmpeg を呼ばずに
- normalize_to_wav の経路 / 引数 / 動画拒否 / 失敗時例外
- extract_audio_from_video の経路 / 音声トラック判定 / 失敗時例外
- ffmpeg 未導入時の MediaError
を検証する。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from app.asr import media
from app.asr.media import (
    MediaError,
    extract_audio_from_video,
    has_audio_stream,
    normalize_to_wav,
)


def _fake_completed(returncode: int = 0, stderr: str = "") -> Any:
    """subprocess.CompletedProcess っぽい SimpleNamespace を作る。"""

    class _P:
        pass

    p = _P()
    p.returncode = returncode
    p.stdout = ""
    p.stderr = stderr
    return p


@pytest.fixture
def _existing_audio(tmp_path: Path) -> Path:
    """ダミーの音声入力ファイル (存在チェック通過用、中身は使われない)。"""
    f = tmp_path / "input.wav"
    f.write_bytes(b"FAKE")
    return f


@pytest.fixture
def _existing_video(tmp_path: Path) -> Path:
    f = tmp_path / "input.mp4"
    f.write_bytes(b"FAKE")
    return f


# --------------------------------------------------------------------------
# normalize_to_wav
# --------------------------------------------------------------------------


def test_normalize_to_wav_invokes_ffmpeg(
    _existing_audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ffmpeg コマンドが正しい引数で呼ばれ、出力ファイルが生成されたとみなす。"""
    out = tmp_path / "out.wav"

    monkeypatch.setattr(media.shutil, "which", lambda name: f"/usr/bin/{name}")
    captured: dict[str, Any] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # 「ffmpeg が出力した」体で wav を擬似生成
        out.write_bytes(b"RIFFFAKE")
        return _fake_completed(0)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    result = normalize_to_wav(_existing_audio, out, sample_rate=16000, channels=1)
    assert result == out
    assert out.exists()
    # ffmpeg 引数の検証
    cmd = captured["cmd"]
    assert cmd[0].endswith("ffmpeg")
    assert "-vn" in cmd
    assert "-ar" in cmd and "16000" in cmd
    assert "-ac" in cmd and "1" in cmd
    assert "pcm_s16le" in cmd


def test_normalize_to_wav_rejects_video_extension(_existing_video: Path, tmp_path: Path) -> None:
    """動画拡張子は API 層に到達する前に MediaError で弾く。"""
    out = tmp_path / "out.wav"
    with pytest.raises(MediaError, match="動画拡張子"):
        normalize_to_wav(_existing_video, out)


def test_normalize_to_wav_missing_input(tmp_path: Path) -> None:
    with pytest.raises(MediaError, match="入力ファイルが存在しません"):
        normalize_to_wav(tmp_path / "nope.wav", tmp_path / "out.wav")


def test_normalize_to_wav_ffmpeg_failure_wrapped(
    _existing_audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ffmpeg が非 0 終了したら MediaError + stderr 抜粋。"""
    out = tmp_path / "out.wav"
    monkeypatch.setattr(media.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _fake_completed(1, stderr="error line 1\nerror line 2"),
    )
    with pytest.raises(MediaError, match="ffmpeg 失敗"):
        normalize_to_wav(_existing_audio, out)


def test_normalize_to_wav_when_ffmpeg_missing(
    _existing_audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """which が None を返したら MediaError + インストール案内。"""
    monkeypatch.setattr(media.shutil, "which", lambda _: None)
    with pytest.raises(MediaError, match="ffmpeg が見つかりません"):
        normalize_to_wav(_existing_audio, tmp_path / "out.wav")


def test_normalize_to_wav_empty_output_raises(
    _existing_audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ffmpeg は成功したが出力ファイルが空なら MediaError。"""
    out = tmp_path / "out.wav"
    monkeypatch.setattr(media.shutil, "which", lambda name: f"/usr/bin/{name}")
    # 出力ファイルは作らない
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _fake_completed(0))
    with pytest.raises(MediaError, match="空または存在しません"):
        normalize_to_wav(_existing_audio, out)


# --------------------------------------------------------------------------
# extract_audio_from_video
# --------------------------------------------------------------------------


def test_extract_audio_from_video_invokes_ffmpeg_with_opus(
    _existing_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out.webm"
    monkeypatch.setattr(media.shutil, "which", lambda name: f"/usr/bin/{name}")
    # has_audio_stream は ffprobe を呼ぶので別途モック
    monkeypatch.setattr(media, "has_audio_stream", lambda _p: True)
    captured: dict[str, Any] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        out.write_bytes(b"OPUS_FAKE")
        return _fake_completed(0)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    result = extract_audio_from_video(_existing_video, out)
    assert result == out
    cmd = captured["cmd"]
    # 既定で libopus + 32k + mono + 16000
    assert "libopus" in cmd
    assert "32k" in cmd
    assert "-ar" in cmd and "16000" in cmd
    assert "-ac" in cmd and "1" in cmd


def test_extract_audio_rejects_non_video(tmp_path: Path) -> None:
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"FAKE")
    with pytest.raises(MediaError, match="動画拡張子ではありません"):
        extract_audio_from_video(audio, tmp_path / "out.webm")


def test_extract_audio_no_audio_track(
    _existing_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """has_audio_stream=False のとき音声トラック無しエラー。"""
    monkeypatch.setattr(media, "has_audio_stream", lambda _p: False)
    monkeypatch.setattr(media.shutil, "which", lambda name: f"/usr/bin/{name}")
    with pytest.raises(MediaError, match="音声トラックが見つかりません"):
        extract_audio_from_video(_existing_video, tmp_path / "out.webm")


def test_extract_audio_missing_input(tmp_path: Path) -> None:
    with pytest.raises(MediaError, match="入力ファイルが存在しません"):
        extract_audio_from_video(tmp_path / "no.mp4", tmp_path / "out.webm")


# --------------------------------------------------------------------------
# has_audio_stream
# --------------------------------------------------------------------------


def test_has_audio_stream_true_when_ffprobe_returns_audio(
    _existing_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(media.shutil, "which", lambda name: f"/usr/bin/{name}")

    def _fake_run(cmd, **kwargs):
        out = _fake_completed(0)
        out.stdout = "audio\n"
        return out

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert has_audio_stream(_existing_video) is True


def test_has_audio_stream_false_when_ffprobe_returns_empty(
    _existing_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(media.shutil, "which", lambda name: f"/usr/bin/{name}")

    def _fake_run(cmd, **kwargs):
        out = _fake_completed(0)
        out.stdout = ""
        return out

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert has_audio_stream(_existing_video) is False


def test_has_audio_stream_fallback_when_ffprobe_missing(tmp_path: Path) -> None:
    """ffprobe が無くても、音声拡張子なら True を返す (フォールバック)。"""
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"FAKE")
    with patch.object(media.shutil, "which", return_value=None):
        assert has_audio_stream(audio) is True


# --------------------------------------------------------------------------
# 拡張子セットの一貫性
# --------------------------------------------------------------------------


def test_audio_and_video_extensions_dont_overlap() -> None:
    """AUDIO と VIDEO の集合は (.webm を除いて) 重ならない。"""
    assert media.AUDIO_EXTENSIONS.isdisjoint(media.VIDEO_EXTENSIONS)


def test_webm_is_treated_as_audio_for_api() -> None:
    """frontend 抽出後の典型形式 .webm は backend で音声扱い。"""
    assert ".webm" in media.AUDIO_EXTENSIONS_INCLUDING_WEBM
    assert ".webm" not in media.VIDEO_EXTENSIONS
