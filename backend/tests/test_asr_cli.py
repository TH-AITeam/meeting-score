"""app.asr.cli の統合テスト (Issue #11)。

実 WhisperX / pyannote / LLM は呼ばない。`transcribe_to_meeting_input` の
コンポーネント組み立てと `main()` の CLI 引数解釈を mock で検証する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.asr.base import Turn, Word
from app.scoring.weights import AppConfig, PenaltyWeights, ScoringWeights


def _base_cfg() -> AppConfig:
    return AppConfig(
        weights=ScoringWeights(),
        penalty_weights=PenaltyWeights(),
        llm_backend="local",
        llm_model="dummy-model",
        llm_endpoint="http://stub/v1",
        llm_max_tokens=512,
        llm_timeout=10.0,
    )


def _stub_audio_cfg() -> dict[str, Any]:
    return {
        "asr": {"device": "cpu", "compute_type": "int8", "language": "ja", "batch_size": 4},
        "diarization": {"device": "cpu", "hf_token_env": "HUGGINGFACE_HUB_TOKEN"},
        "volume": {"enabled": False},  # CPU テストでは無効化
        "pipeline": {"overlap_iou_threshold": 0.3},
    }


def test_transcribe_to_meeting_input_full_path(monkeypatch, tmp_path) -> None:
    """音声ファイル → WhisperX(mock) → pyannote(mock) → segmenter → meeting_builder
    のフルパスを mock 経由で通す。"""
    from app.asr import cli

    audio = tmp_path / "x.wav"
    audio.write_bytes(b"")
    captured: dict[str, Any] = {}

    # WhisperXTranscriber を fake に差し替え
    class _FakeTranscriber:
        def __init__(self, config) -> None:
            self.config = config
            captured["model_name"] = config.model_name

        def transcribe(self, path):
            return [
                Word("こんにちは", 0.0, 1.0),
                Word("会議", 1.0, 2.0),
            ]

    class _FakeDiarizer:
        def __init__(self, config) -> None:
            self.config = config

        def diarize(self, path, num_speakers=None):
            return [
                Turn(speaker="SPEAKER_00", start_sec=0.0, end_sec=2.0),
            ]

    monkeypatch.setattr(cli, "WhisperXTranscriber", _FakeTranscriber)
    monkeypatch.setattr(cli, "PyannoteDiarizer", _FakeDiarizer)

    mi = cli.transcribe_to_meeting_input(
        audio,
        meeting_id="m042",
        cfg=_base_cfg(),
        audio_cfg=_stub_audio_cfg(),
        use_meta_extractor=False,  # LLM 呼び出しなし
        use_volume_analyzer=False,
        default_title="サンプル会議",
        default_goal="目的サンプル",
    )
    assert mi.meeting_id == "m042"
    assert mi.title == "サンプル会議"
    assert mi.goal == "目的サンプル"
    assert len(mi.utterances) == 1
    assert mi.utterances[0].speaker == "SPEAKER_00"
    assert mi.utterances[0].text == "こんにちは会議"
    assert captured["model_name"] == "large-v3"


def test_transcribe_to_meeting_input_passes_num_speakers(monkeypatch, tmp_path) -> None:
    """num_speakers が Diarizer.diarize へ渡る。"""
    from app.asr import cli

    audio = tmp_path / "x.wav"
    audio.write_bytes(b"")
    captured: dict[str, Any] = {}

    class _FakeTranscriber:
        def __init__(self, config) -> None:
            self.config = config

        def transcribe(self, path):
            return [Word("a", 0.0, 1.0)]

    class _FakeDiarizer:
        def __init__(self, config) -> None:
            self.config = config

        def diarize(self, path, num_speakers=None):
            captured["num_speakers"] = num_speakers
            return [Turn(speaker="S0", start_sec=0.0, end_sec=1.0)]

    monkeypatch.setattr(cli, "WhisperXTranscriber", _FakeTranscriber)
    monkeypatch.setattr(cli, "PyannoteDiarizer", _FakeDiarizer)

    cli.transcribe_to_meeting_input(
        audio,
        meeting_id="m001",
        cfg=_base_cfg(),
        audio_cfg=_stub_audio_cfg(),
        num_speakers=4,
        use_meta_extractor=False,
        use_volume_analyzer=False,
    )
    assert captured["num_speakers"] == 4


def test_cli_main_writes_json(monkeypatch, tmp_path) -> None:
    """python -m app.asr.cli が JSON ファイルを出力する。"""
    from app.asr import cli

    audio = tmp_path / "x.wav"
    audio.write_bytes(b"")
    out = tmp_path / "out.json"

    class _FakeTranscriber:
        def __init__(self, config) -> None:
            self.config = config

        def transcribe(self, path):
            return [Word("発言", 0.0, 1.0)]

    class _FakeDiarizer:
        def __init__(self, config) -> None:
            self.config = config

        def diarize(self, path, num_speakers=None):
            return [Turn(speaker="S0", start_sec=0.0, end_sec=1.0)]

    monkeypatch.setattr(cli, "WhisperXTranscriber", _FakeTranscriber)
    monkeypatch.setattr(cli, "PyannoteDiarizer", _FakeDiarizer)
    monkeypatch.setattr(cli, "load_config", lambda *_: _base_cfg())
    monkeypatch.setattr(cli, "_load_audio_section", lambda *_: _stub_audio_cfg())

    exit_code = cli.main(
        [
            "--input",
            str(audio),
            "--output",
            str(out),
            "--meeting-id",
            "m999",
            "--no-meta-extract",
            "--no-volume",
            "--title",
            "x",
            "--goal",
            "y",
        ]
    )
    assert exit_code == 0
    assert out.exists()
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["meeting_id"] == "m999"
    assert parsed["title"] == "x"
    assert parsed["goal"] == "y"
    assert len(parsed["utterances"]) == 1


def test_cli_main_returns_1_when_input_missing(monkeypatch, tmp_path, capsys) -> None:
    from app.asr import cli

    monkeypatch.setattr(cli, "load_config", lambda *_: _base_cfg())
    monkeypatch.setattr(cli, "_load_audio_section", lambda *_: {})

    exit_code = cli.main(
        [
            "--input",
            str(tmp_path / "missing.wav"),
            "--output",
            str(tmp_path / "out.json"),
            "--meeting-id",
            "m000",
            "--no-meta-extract",
        ]
    )
    assert exit_code == 1


def test_load_audio_section_reads_yaml(tmp_path: Path) -> None:
    from app.asr.cli import _load_audio_section

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "audio:\n  asr:\n    device: cpu\n    batch_size: 8\n",
        encoding="utf-8",
    )
    sec = _load_audio_section(str(cfg))
    assert sec["asr"]["device"] == "cpu"
    assert sec["asr"]["batch_size"] == 8


def test_load_audio_section_missing_file_returns_empty(tmp_path: Path) -> None:
    from app.asr.cli import _load_audio_section

    sec = _load_audio_section(str(tmp_path / "nope.yaml"))
    assert sec == {}


# --------------------------------------------------------------------------
# Issue #68: CLI で動画拡張子サポート
# --------------------------------------------------------------------------


def test_cli_main_extracts_audio_for_video_input(monkeypatch, tmp_path) -> None:
    """`--input meeting.mp4` で extract→normalize→transcribe が順に走る。"""
    from app.asr import cli

    video = tmp_path / "meeting.mp4"
    video.write_bytes(b"FAKE_MP4")
    out = tmp_path / "out.json"

    extracted_paths: list[Path] = []
    normalized_paths: list[Path] = []
    transcribed_paths: list[Path] = []

    def _fake_extract(input_path, output_path, **kwargs):
        output_path.write_bytes(b"OPUS_FAKE")
        extracted_paths.append(output_path)
        return output_path

    def _fake_normalize(input_path, output_path, **kwargs):
        output_path.write_bytes(b"WAV_FAKE")
        normalized_paths.append(output_path)
        return output_path

    class _FakeTranscriber:
        def __init__(self, config) -> None:
            self.config = config

        def transcribe(self, path):
            transcribed_paths.append(Path(path))
            return [Word("動画から抽出", 0.0, 1.0)]

    class _FakeDiarizer:
        def __init__(self, config) -> None:
            self.config = config

        def diarize(self, path, num_speakers=None):
            return [Turn(speaker="S0", start_sec=0.0, end_sec=1.0)]

    monkeypatch.setattr(cli, "extract_audio_from_video", _fake_extract)
    monkeypatch.setattr(cli, "normalize_to_wav", _fake_normalize)
    monkeypatch.setattr(cli, "WhisperXTranscriber", _FakeTranscriber)
    monkeypatch.setattr(cli, "PyannoteDiarizer", _FakeDiarizer)
    monkeypatch.setattr(cli, "load_config", lambda *_: _base_cfg())
    monkeypatch.setattr(cli, "_load_audio_section", lambda *_: _stub_audio_cfg())

    exit_code = cli.main(
        [
            "--input",
            str(video),
            "--output",
            str(out),
            "--meeting-id",
            "m_video",
            "--no-meta-extract",
            "--no-volume",
        ]
    )
    assert exit_code == 0
    assert len(extracted_paths) == 1
    assert extracted_paths[0].suffix == ".webm"
    assert len(normalized_paths) == 1
    assert normalized_paths[0].suffix == ".wav"
    # ASR には正規化後の wav が渡る (webm のままではない)
    assert len(transcribed_paths) == 1
    assert transcribed_paths[0].suffix == ".wav"
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["meeting_id"] == "m_video"
    assert parsed["utterances"][0]["text"] == "動画から抽出"


def test_cli_main_extracts_audio_for_webm_video_input(monkeypatch, tmp_path) -> None:
    """`--input meeting.webm` も動画入力として extract→normalize に通す。"""
    from app.asr import cli

    video = tmp_path / "meeting.webm"
    video.write_bytes(b"FAKE_WEBM")
    out = tmp_path / "out.json"

    extracted_inputs: list[Path] = []
    normalized_inputs: list[Path] = []
    transcribed_paths: list[Path] = []

    def _fake_extract(input_path, output_path, **kwargs):
        output_path.write_bytes(b"OPUS_FAKE")
        extracted_inputs.append(input_path)
        return output_path

    def _fake_normalize(input_path, output_path, **kwargs):
        output_path.write_bytes(b"WAV_FAKE")
        normalized_inputs.append(input_path)
        return output_path

    class _FakeTranscriber:
        def __init__(self, config) -> None:
            self.config = config

        def transcribe(self, path):
            transcribed_paths.append(Path(path))
            return [Word("WebMから抽出", 0.0, 1.0)]

    class _FakeDiarizer:
        def __init__(self, config) -> None:
            self.config = config

        def diarize(self, path, num_speakers=None):
            return [Turn(speaker="S0", start_sec=0.0, end_sec=1.0)]

    monkeypatch.setattr(cli, "extract_audio_from_video", _fake_extract)
    monkeypatch.setattr(cli, "normalize_to_wav", _fake_normalize)
    monkeypatch.setattr(cli, "WhisperXTranscriber", _FakeTranscriber)
    monkeypatch.setattr(cli, "PyannoteDiarizer", _FakeDiarizer)
    monkeypatch.setattr(cli, "load_config", lambda *_: _base_cfg())
    monkeypatch.setattr(cli, "_load_audio_section", lambda *_: _stub_audio_cfg())

    exit_code = cli.main(
        [
            "--input",
            str(video),
            "--output",
            str(out),
            "--meeting-id",
            "m_webm_video",
            "--no-meta-extract",
            "--no-volume",
        ]
    )

    assert exit_code == 0
    assert extracted_inputs == [video]
    assert len(normalized_inputs) == 1
    assert normalized_inputs[0].suffix == ".webm"
    assert len(transcribed_paths) == 1
    assert transcribed_paths[0].suffix == ".wav"
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["meeting_id"] == "m_webm_video"
    assert parsed["utterances"][0]["text"] == "WebMから抽出"


def test_cli_main_returns_2_when_video_extraction_fails(monkeypatch, tmp_path) -> None:
    """動画→音声抽出に失敗したら exit code 2 を返す (transcribe へ進まない)。"""
    from app.asr import cli
    from app.asr.media import MediaError

    video = tmp_path / "broken.mp4"
    video.write_bytes(b"FAKE_BROKEN")
    out = tmp_path / "out.json"
    tmp_dir = tmp_path / "scratch_extract"

    def _raise_media(*args, **kwargs):
        msg = "ffmpeg failed for broken.mp4"
        raise MediaError(msg)

    def _fake_mkdtemp(prefix):
        tmp_dir.mkdir()
        return str(tmp_dir)

    monkeypatch.setattr(cli, "extract_audio_from_video", _raise_media)
    monkeypatch.setattr(cli.tempfile, "mkdtemp", _fake_mkdtemp)
    monkeypatch.setattr(cli, "load_config", lambda *_: _base_cfg())
    monkeypatch.setattr(cli, "_load_audio_section", lambda *_: _stub_audio_cfg())

    exit_code = cli.main(
        [
            "--input",
            str(video),
            "--output",
            str(out),
            "--meeting-id",
            "m_video_fail",
            "--no-meta-extract",
        ]
    )
    assert exit_code == 2
    assert not out.exists()
    assert not tmp_dir.exists()


def test_cli_main_returns_2_when_normalize_fails(monkeypatch, tmp_path) -> None:
    """動画→音声抽出は成功するが、その後の wav 正規化で MediaError → exit 2。"""
    from app.asr import cli
    from app.asr.media import MediaError

    video = tmp_path / "video.mp4"
    video.write_bytes(b"FAKE_MP4")
    out = tmp_path / "out.json"
    tmp_dir = tmp_path / "scratch_normalize"

    def _fake_extract(input_path, output_path, **kwargs):
        output_path.write_bytes(b"OPUS_FAKE")
        return output_path

    def _raise_normalize(*args, **kwargs):
        msg = "normalize failed for video_5min.webm"
        raise MediaError(msg)

    def _fake_mkdtemp(prefix):
        tmp_dir.mkdir()
        return str(tmp_dir)

    monkeypatch.setattr(cli, "extract_audio_from_video", _fake_extract)
    monkeypatch.setattr(cli, "normalize_to_wav", _raise_normalize)
    monkeypatch.setattr(cli.tempfile, "mkdtemp", _fake_mkdtemp)
    monkeypatch.setattr(cli, "load_config", lambda *_: _base_cfg())
    monkeypatch.setattr(cli, "_load_audio_section", lambda *_: _stub_audio_cfg())

    exit_code = cli.main(
        [
            "--input",
            str(video),
            "--output",
            str(out),
            "--meeting-id",
            "m_norm_fail",
            "--no-meta-extract",
        ]
    )
    assert exit_code == 2
    assert not out.exists()
    assert not tmp_dir.exists()


def test_cli_main_no_extraction_for_audio_input(monkeypatch, tmp_path) -> None:
    """音声入力では extract_audio_from_video / normalize_to_wav は呼ばれない (回帰防止)。"""
    from app.asr import cli

    audio = tmp_path / "x.wav"
    audio.write_bytes(b"FAKE")
    out = tmp_path / "out.json"

    extract_called = {"value": False}
    normalize_called = {"value": False}

    def _fake_extract(input_path, output_path, **kwargs):
        extract_called["value"] = True
        return output_path

    def _fake_normalize(input_path, output_path, **kwargs):
        normalize_called["value"] = True
        return output_path

    class _FakeTranscriber:
        def __init__(self, config) -> None:
            self.config = config

        def transcribe(self, path):
            return [Word("音声直接", 0.0, 1.0)]

    class _FakeDiarizer:
        def __init__(self, config) -> None:
            self.config = config

        def diarize(self, path, num_speakers=None):
            return [Turn(speaker="S0", start_sec=0.0, end_sec=1.0)]

    monkeypatch.setattr(cli, "extract_audio_from_video", _fake_extract)
    monkeypatch.setattr(cli, "normalize_to_wav", _fake_normalize)
    monkeypatch.setattr(cli, "WhisperXTranscriber", _FakeTranscriber)
    monkeypatch.setattr(cli, "PyannoteDiarizer", _FakeDiarizer)
    monkeypatch.setattr(cli, "load_config", lambda *_: _base_cfg())
    monkeypatch.setattr(cli, "_load_audio_section", lambda *_: _stub_audio_cfg())

    exit_code = cli.main(
        [
            "--input",
            str(audio),
            "--output",
            str(out),
            "--meeting-id",
            "m_aud",
            "--no-meta-extract",
            "--no-volume",
        ]
    )
    assert exit_code == 0
    assert extract_called["value"] is False
    assert normalize_called["value"] is False
