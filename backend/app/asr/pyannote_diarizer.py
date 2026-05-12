"""pyannote.audio ベースの Diarizer スケルトン (Issue #19)。

ADR 0002 で採用された `pyannote/speaker-diarization-3.1` のラッパー。
実音声処理ロジックは Issue #11 で本実装する。

Issue #11 でやること（メモ）:
    1. `pyannote.audio.Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
       use_auth_token=HF_TOKEN)` を初期化時に取得
    2. GPU 環境では `pipeline.to(torch.device("cuda"))`
    3. `diarize(audio_path, num_speakers=...)` で:
        - pipeline(audio_path, num_speakers=...) を呼ぶ
        - `Annotation` の各セグメントを `Turn(speaker, start, end, overlap)` へ変換
        - overlap 判定: pyannote の `OverlappedSpeechDetection` を別に走らせる か、
          Annotation 同士の時間重なりを直接検出
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.asr.base import Turn


@dataclass
class PyannoteConfig:
    """pyannote diarization の起動パラメータ。

    `backend/config.yaml` の `audio.diarization` セクションから注入する想定。
    """

    model_name: str = "pyannote/speaker-diarization-3.1"
    device: str = "cuda"
    hf_token_env: str = "HUGGINGFACE_HUB_TOKEN"  # 環境変数名のみ保持し値は持たない
    # 話者数を固定したい場合に使う。None なら自動推定
    default_num_speakers: int | None = None
    detect_overlap: bool = True


class PyannoteLoadError(RuntimeError):
    """pyannote 認証 / ロード失敗を示す。"""


class PyannoteDiarizer:
    """pyannote.audio による話者分離。

    Issue #19 時点ではスケルトン。`diarize()` は `NotImplementedError` を投げる。
    """

    def __init__(self, config: PyannoteConfig | None = None) -> None:
        self.config = config or PyannoteConfig()
        self._pipeline = None  # 遅延ロード

    @property
    def hf_token(self) -> str | None:
        """環境変数から HF token を解決する (config に値を持たない)。"""
        return os.environ.get(self.config.hf_token_env)

    def load(self) -> None:
        """pyannote パイプラインをロードする。HF token が無ければ
        `PyannoteLoadError` を投げる (gated repo)。Issue #11 で実装。"""
        msg = (
            "PyannoteDiarizer.load() は Issue #11 で実装予定。"
            "現状は ADR 0002 のインタフェース定義のみ。"
        )
        raise NotImplementedError(msg)

    def diarize(self, audio_path: Path, num_speakers: int | None = None) -> list[Turn]:
        """音声ファイルから話者交替区間を返す。

        Parameters
        ----------
        num_speakers : int | None
            None なら自動推定、整数なら強制固定。
        """
        _ = audio_path, num_speakers
        msg = "PyannoteDiarizer.diarize() は Issue #11 で実装予定。"
        raise NotImplementedError(msg)


__all__ = ["PyannoteConfig", "PyannoteDiarizer", "PyannoteLoadError"]
