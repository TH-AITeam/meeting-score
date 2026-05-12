"""WhisperXTranscriber の mock テスト (Issue #11)。

実 GPU / 実モデルを呼ばずに、whisperx の load / align 経路を monkeypatch して
- ロード時の引数受け渡し
- transcribe() の align 結果 → Word リスト変換
- timestamp の欠損補完
- エラー時の WhisperXLoadError
を検証する。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.asr.whisperx_transcriber import (
    WhisperXConfig,
    WhisperXLoadError,
    WhisperXTranscriber,
    _extract_words,
)


def _stub_whisperx(transcribe_return: dict, align_return: dict) -> SimpleNamespace:
    """whisperx モジュール風のスタブを組み立てる。"""
    load_calls: dict[str, Any] = {}

    def load_model(model_name, device, compute_type, language):
        load_calls["model_name"] = model_name
        load_calls["device"] = device
        load_calls["compute_type"] = compute_type
        load_calls["language"] = language

        class _Model:
            def transcribe(self, audio, batch_size, language):
                load_calls["transcribe_batch_size"] = batch_size
                return transcribe_return

        return _Model()

    def load_align_model(**kwargs):
        load_calls["align_kwargs"] = kwargs
        return ("align_model", {"metadata": True})

    def load_audio(path):
        load_calls["audio_path"] = path
        return [0.0]  # ダミー numpy 配列代わり

    def align(segments, align_model, metadata, audio, device, return_char_alignments):
        load_calls["align_called"] = True
        return align_return

    return SimpleNamespace(
        load_model=load_model,
        load_align_model=load_align_model,
        load_audio=load_audio,
        align=align,
        _calls=load_calls,
    )


def test_load_passes_config_to_whisperx() -> None:
    """WhisperXConfig の値が whisperx.load_model に渡る。"""
    stub = _stub_whisperx({"segments": []}, {"segments": []})
    transcriber = WhisperXTranscriber(
        WhisperXConfig(
            model_name="kotoba-tech/kotoba-whisper-v2.0",
            device="cuda",
            compute_type="float16",
            language="ja",
        )
    )
    transcriber._whisperx = stub
    transcriber.load()
    assert stub._calls["model_name"] == "kotoba-tech/kotoba-whisper-v2.0"
    assert stub._calls["device"] == "cuda"
    assert stub._calls["language"] == "ja"
    assert stub._calls["align_kwargs"]["language_code"] == "ja"


def test_load_uses_custom_align_model_when_specified() -> None:
    stub = _stub_whisperx({"segments": []}, {"segments": []})
    transcriber = WhisperXTranscriber(
        WhisperXConfig(align_model="jonatasgrosman/wav2vec2-large-xlsr-53-japanese")
    )
    transcriber._whisperx = stub
    transcriber.load()
    assert (
        stub._calls["align_kwargs"]["model_name"]
        == "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
    )


def test_load_is_idempotent() -> None:
    """load() を 2 回呼んでも load_model は 1 回だけ。"""
    stub = _stub_whisperx({"segments": []}, {"segments": []})
    transcriber = WhisperXTranscriber()
    transcriber._whisperx = stub
    transcriber.load()
    first_model = transcriber._model
    transcriber.load()
    assert transcriber._model is first_model


def test_transcribe_returns_word_list() -> None:
    """align 結果が Word リストに変換される。"""
    align_return = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "words": [
                    {"word": "こんにちは", "start": 0.0, "end": 1.0, "score": 0.98},
                    {"word": "会議", "start": 1.0, "end": 2.0, "score": 0.95},
                ],
            }
        ]
    }
    stub = _stub_whisperx({"segments": ["dummy"]}, align_return)
    transcriber = WhisperXTranscriber()
    transcriber._whisperx = stub
    words = transcriber.transcribe(Path("/tmp/x.wav"))
    assert [w.text for w in words] == ["こんにちは", "会議"]
    assert words[0].confidence == 0.98
    assert words[0].start_sec == 0.0
    assert words[1].end_sec == 2.0


def test_transcribe_load_failure_wrapped() -> None:
    """whisperx 側が例外を投げたら WhisperXLoadError に包む。"""

    class _BrokenStub:
        def load_model(self, *a, **k):
            msg = "boom"
            raise RuntimeError(msg)

    transcriber = WhisperXTranscriber()
    transcriber._whisperx = _BrokenStub()
    with pytest.raises(WhisperXLoadError, match="ロードに失敗"):
        transcriber.load()


def test_extract_words_fills_missing_timestamp() -> None:
    """align 出力で word の start/end が欠けても、隣接 word の値で埋める。"""
    segments = [
        {
            "start": 0.0,
            "end": 3.0,
            "words": [
                {"word": "a", "start": 0.0, "end": 1.0, "score": 0.9},
                {"word": "b", "start": None, "end": None},  # 欠損
                {"word": "c", "start": 2.0, "end": 3.0, "score": 0.8},
            ],
        }
    ]
    words = _extract_words(segments)
    assert len(words) == 3
    # b の start は前 word の end (1.0)、end は次 word の start (2.0)
    assert words[1].start_sec == 1.0
    assert words[1].end_sec == 2.0


def test_extract_words_skips_empty_text() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "words": [
                {"word": "", "start": 0.0, "end": 0.1},
                {"word": "発言", "start": 0.1, "end": 1.0, "score": 0.7},
            ],
        }
    ]
    words = _extract_words(segments)
    assert [w.text for w in words] == ["発言"]


def test_unload_clears_state() -> None:
    stub = _stub_whisperx({"segments": []}, {"segments": []})
    transcriber = WhisperXTranscriber()
    transcriber._whisperx = stub
    transcriber.load()
    assert transcriber._model is not None
    transcriber.unload()
    assert transcriber._model is None
    assert transcriber._align_model is None
