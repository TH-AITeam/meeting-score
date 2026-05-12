"""WhisperX ベースの Transcriber 実装 (Issue #11)。

ADR 0002 に基づく WhisperX (https://github.com/m-bain/whisperX) のラッパー。
モデルのロードは遅延 (最初の `load()` / `transcribe()` 呼び出し時)、
GPU メモリは `unload()` で明示開放する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.asr.base import Word

logger = logging.getLogger(__name__)


@dataclass
class WhisperXConfig:
    """WhisperX 起動パラメータ。

    `backend/config.yaml` の `audio.asr` セクションから注入する想定。
    """

    model_name: str = "large-v3"  # backbone (e.g. large-v3, kotoba-whisper-v2.0 等)
    device: str = "cuda"
    compute_type: str = "float16"  # float16 / int8_float16 / int8
    language: str = "ja"
    batch_size: int = 16
    align_model: str | None = None  # 例: "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"


class WhisperXLoadError(RuntimeError):
    """WhisperX ライブラリのロード／推論失敗を示す。"""


class WhisperXTranscriber:
    """WhisperX による ASR。

    本クラスを初期化しただけでは GPU を確保せず、`load()` または `transcribe()`
    の初回呼び出しで Whisper + アライメントモデルを GPU に載せる (遅延ロード)。
    """

    def __init__(self, config: WhisperXConfig | None = None) -> None:
        self.config = config or WhisperXConfig()
        self._model: Any = None
        self._align_model: Any = None
        self._align_metadata: Any = None
        self._whisperx: Any = None

    def _import_whisperx(self) -> Any:
        if self._whisperx is not None:
            return self._whisperx
        try:
            import whisperx
        except ImportError as e:  # pragma: no cover - 依存が無い環境
            msg = (
                "whisperx が見つかりません。`uv sync --extra audio` で audio 依存を入れてください。"
            )
            raise WhisperXLoadError(msg) from e
        self._whisperx = whisperx
        return whisperx

    def load(self) -> None:
        """Whisper モデルと wav2vec2 アライメントモデルを GPU にロードする。

        多重呼び出しは安全 (二度目以降は no-op)。
        """
        if self._model is not None:
            return
        whisperx = self._import_whisperx()
        try:
            logger.info(
                "Loading WhisperX model: %s (device=%s, compute_type=%s)",
                self.config.model_name,
                self.config.device,
                self.config.compute_type,
            )
            self._model = whisperx.load_model(
                self.config.model_name,
                device=self.config.device,
                compute_type=self.config.compute_type,
                language=self.config.language,
            )
            align_kwargs: dict[str, Any] = {
                "language_code": self.config.language,
                "device": self.config.device,
            }
            if self.config.align_model:
                align_kwargs["model_name"] = self.config.align_model
            self._align_model, self._align_metadata = whisperx.load_align_model(**align_kwargs)
        except Exception as e:
            msg = f"WhisperX のロードに失敗しました: {e}"
            raise WhisperXLoadError(msg) from e

    def transcribe(self, audio_path: Path) -> list[Word]:
        """音声ファイルから word-level timestamp 付きの単語列を返す。

        - faster-whisper backend で粗い segment を取得
        - wav2vec2 アライメントモデルで word-level timestamp を整える
        - 各単語を `Word(text, start_sec, end_sec, confidence)` に変換
        """
        self.load()
        whisperx = self._import_whisperx()
        try:
            audio = whisperx.load_audio(str(audio_path))
            result = self._model.transcribe(
                audio, batch_size=self.config.batch_size, language=self.config.language
            )
            aligned = whisperx.align(
                result["segments"],
                self._align_model,
                self._align_metadata,
                audio,
                self.config.device,
                return_char_alignments=False,
            )
        except Exception as e:
            msg = f"WhisperX 推論に失敗しました ({audio_path}): {e}"
            raise WhisperXLoadError(msg) from e

        return _extract_words(aligned.get("segments", []))

    def unload(self) -> None:
        """GPU メモリを開放する (次のモデルをロードする前など)。"""
        self._model = None
        self._align_model = None
        self._align_metadata = None
        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass


def _extract_words(segments: list[dict]) -> list[Word]:
    """WhisperX の align 結果から `Word` リストを取り出す。

    `segments` は `[{"words": [{"word": ..., "start": ..., "end": ..., "score": ...}, ...]}, ...]`
    形式。timestamp が欠ける単語は前後の値で埋める (faster-whisper では端単語で起き得る)。
    """
    words: list[Word] = []
    for seg in segments:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", seg_start)
        word_items = seg.get("words", [])
        last_end = seg_start
        for i, w in enumerate(word_items):
            text = w.get("word") or w.get("text") or ""
            if not text:
                continue
            start = w.get("start")
            end = w.get("end")
            if start is None:
                start = last_end
            if end is None:
                # 次の word の start で埋める。最後なら segment 終端
                if i + 1 < len(word_items) and word_items[i + 1].get("start") is not None:
                    end = word_items[i + 1]["start"]
                else:
                    end = max(seg_end, start)
            words.append(
                Word(
                    text=text,
                    start_sec=float(start),
                    end_sec=float(end),
                    confidence=float(w["score"]) if w.get("score") is not None else None,
                )
            )
            last_end = float(end)
    return words


__all__ = [
    "WhisperXConfig",
    "WhisperXLoadError",
    "WhisperXTranscriber",
    "_extract_words",
]
