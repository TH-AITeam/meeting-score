"""WhisperX ベースの Transcriber スケルトン (Issue #19)。

本ファイルは ADR 0002 に基づく WhisperX (https://github.com/m-bain/whisperX) の
ラッパー型骨格を提供する。実音声処理ロジックの本実装は Issue #11 で行う。

Issue #11 でやること（メモ）:
    1. `whisperx.load_model(...)` の引数 (model, device, compute_type) を config 化
    2. `whisperx.load_align_model(language_code="ja", ...)` を初期化時に取得
    3. `transcribe(audio_path)` で:
        - whisperx.load_audio → faster-whisper でセグメント取得
        - whisperx.align で word-level timestamp に整える
        - `Word(text, start_sec, end_sec, confidence)` のリストへ変換
    4. 失敗時は `WhisperXLoadError` を上に投げ、上流で握り潰さない
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.asr.base import Word


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

    Issue #19 時点ではスケルトン。`transcribe()` は `NotImplementedError` を
    投げるので、実音声処理を必要とするテストは pytest mark でスキップする。
    """

    def __init__(self, config: WhisperXConfig | None = None) -> None:
        self.config = config or WhisperXConfig()
        self._model = None  # 遅延ロード（Issue #11 で実装）
        self._align_model = None
        self._align_metadata = None

    def load(self) -> None:
        """Whisper モデルと wav2vec2 アライメントモデルを遅延ロードする。

        Issue #11 で実装する。本クラスを初期化しただけでは GPU 確保しない。
        """
        msg = (
            "WhisperXTranscriber.load() は Issue #11 で実装予定。"
            "現状は ADR 0002 のインタフェース定義のみ。"
        )
        raise NotImplementedError(msg)

    def transcribe(self, audio_path: Path) -> list[Word]:
        """音声ファイルから word-level timestamp 付きの単語列を返す。"""
        msg = "WhisperXTranscriber.transcribe() は Issue #11 で実装予定。"
        raise NotImplementedError(msg)


__all__ = ["WhisperXConfig", "WhisperXLoadError", "WhisperXTranscriber"]
