"""PyannoteDiarizer の mock テスト (Issue #11)。

実 GPU / 実 pyannote を呼ばずに、Pipeline 風の dummy を差し込んで
- HF token 検証
- load() の冪等性
- diarize() の Annotation → Turn 変換
- overlap 判定
- 失敗時の PyannoteLoadError
を検証する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

from app.asr.pyannote_diarizer import (
    PyannoteConfig,
    PyannoteDiarizer,
    PyannoteLoadError,
    _annotation_to_turns,
)


class _FakeSegment:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _FakeAnnotation:
    """`pyannote.core.Annotation` の itertracks を模す。

    tracks: list of (start_sec, end_sec, speaker_label)
    """

    def __init__(self, tracks: list[tuple[float, float, str]]) -> None:
        self._tracks = tracks

    def itertracks(self, yield_label: bool = True):
        for i, (s, e, label) in enumerate(self._tracks):
            yield (_FakeSegment(s, e), f"track_{i}", label)


class _FakePipeline:
    def __init__(self, annotation: _FakeAnnotation) -> None:
        self.annotation = annotation
        self.calls: list[dict[str, Any]] = []

    def to(self, device):
        self.device = device
        return self

    def __call__(self, audio_path, **kwargs):
        self.calls.append({"audio_path": audio_path, **kwargs})
        return self.annotation


class _FakePipelineClass:
    """from_pretrained を持つ Pipeline クラスのスタブ。"""

    last_kwargs: ClassVar[dict[str, Any]] = {}

    def __init__(self, pipeline_to_return: _FakePipeline) -> None:
        self._return = pipeline_to_return

    def from_pretrained(self, model_name, use_auth_token):
        _FakePipelineClass.last_kwargs = {
            "model_name": model_name,
            "use_auth_token": use_auth_token,
        }
        return self._return


def test_load_requires_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    diarizer = PyannoteDiarizer()
    with pytest.raises(PyannoteLoadError, match="HF token が見つかりません"):
        diarizer.load()


def test_load_passes_token_and_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hf_dummy")
    annotation = _FakeAnnotation([])
    pipeline = _FakePipeline(annotation)
    diarizer = PyannoteDiarizer(PyannoteConfig(device="cpu"))  # GPU 移動を回避
    diarizer._pyannote = _FakePipelineClass(pipeline)
    diarizer.load()
    assert _FakePipelineClass.last_kwargs["model_name"] == "pyannote/speaker-diarization-3.1"
    assert _FakePipelineClass.last_kwargs["use_auth_token"] == "hf_dummy"


def test_load_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hf_dummy")
    pipeline = _FakePipeline(_FakeAnnotation([]))
    diarizer = PyannoteDiarizer(PyannoteConfig(device="cpu"))
    diarizer._pyannote = _FakePipelineClass(pipeline)
    diarizer.load()
    first = diarizer._pipeline
    diarizer.load()
    assert diarizer._pipeline is first


def test_diarize_returns_turn_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hf_dummy")
    annotation = _FakeAnnotation(
        [
            (0.0, 5.0, "SPEAKER_00"),
            (5.0, 10.0, "SPEAKER_01"),
        ]
    )
    pipeline = _FakePipeline(annotation)
    diarizer = PyannoteDiarizer(PyannoteConfig(device="cpu"))
    diarizer._pyannote = _FakePipelineClass(pipeline)
    turns = diarizer.diarize(Path("/tmp/x.wav"))
    assert [t.speaker for t in turns] == ["SPEAKER_00", "SPEAKER_01"]
    assert turns[0].start_sec == 0.0
    assert turns[1].end_sec == 10.0
    assert all(t.overlap is False for t in turns)


def test_diarize_passes_num_speakers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hf_dummy")
    pipeline = _FakePipeline(_FakeAnnotation([]))
    diarizer = PyannoteDiarizer(PyannoteConfig(device="cpu"))
    diarizer._pyannote = _FakePipelineClass(pipeline)
    diarizer.diarize(Path("/tmp/x.wav"), num_speakers=3)
    assert pipeline.calls[0]["num_speakers"] == 3


def test_diarize_uses_default_num_speakers(monkeypatch: pytest.MonkeyPatch) -> None:
    """num_speakers 省略時は config.default_num_speakers を使う。"""
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hf_dummy")
    pipeline = _FakePipeline(_FakeAnnotation([]))
    diarizer = PyannoteDiarizer(PyannoteConfig(device="cpu", default_num_speakers=2))
    diarizer._pyannote = _FakePipelineClass(pipeline)
    diarizer.diarize(Path("/tmp/x.wav"))
    assert pipeline.calls[0]["num_speakers"] == 2


def test_annotation_to_turns_detects_overlap() -> None:
    """同じ時間帯に別 speaker のラベルがあれば overlap=True。"""
    annotation = _FakeAnnotation(
        [
            (0.0, 5.0, "SPEAKER_00"),
            (3.0, 7.0, "SPEAKER_01"),  # 3〜5 で SPEAKER_00 とオーバーラップ
        ]
    )
    turns = _annotation_to_turns(annotation)
    assert turns[0].overlap is True
    assert turns[1].overlap is True


def test_annotation_to_turns_no_overlap_when_same_speaker() -> None:
    """同一 speaker の連続 turn は overlap 扱いしない (同じ人が割り込んでいるとは見ない)。"""
    annotation = _FakeAnnotation(
        [
            (0.0, 5.0, "SPEAKER_00"),
            (3.0, 7.0, "SPEAKER_00"),  # 同じ話者なので overlap には数えない
        ]
    )
    turns = _annotation_to_turns(annotation)
    assert all(t.overlap is False for t in turns)


def test_diarize_load_failure_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hf_dummy")

    class _BrokenPipelineClass:
        def from_pretrained(self, *a, **k):
            msg = "401"
            raise RuntimeError(msg)

    diarizer = PyannoteDiarizer(PyannoteConfig(device="cpu"))
    diarizer._pyannote = _BrokenPipelineClass()
    with pytest.raises(PyannoteLoadError, match="ロードに失敗"):
        diarizer.load()
