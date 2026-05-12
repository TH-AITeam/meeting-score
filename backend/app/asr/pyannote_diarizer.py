"""pyannote.audio ベースの Diarizer 実装 (Issue #11)。

ADR 0002 で採用された `pyannote/speaker-diarization-3.1` のラッパー。

- パイプラインは遅延ロード (初回 `diarize()` 呼び出し時)
- HF token は環境変数 (既定 `HUGGINGFACE_HUB_TOKEN`) から取得し、`PyannoteConfig`
  には値を持たせない
- GPU 環境では `pipeline.to(torch.device("cuda"))`
- diarize 結果 (`pyannote.core.Annotation`) を `Turn(speaker, start, end, overlap)`
  に変換。overlap 判定は同一時刻の他話者ラベル数で
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.asr.base import Turn

logger = logging.getLogger(__name__)


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
    """pyannote.audio による話者分離。"""

    def __init__(self, config: PyannoteConfig | None = None) -> None:
        self.config = config or PyannoteConfig()
        self._pipeline: Any = None
        self._pyannote: Any = None  # テスト時の差し替え窓口

    @property
    def hf_token(self) -> str | None:
        """環境変数から HF token を解決する (config に値を持たない)。"""
        return os.environ.get(self.config.hf_token_env)

    def _import_pyannote(self) -> Any:
        if self._pyannote is not None:
            return self._pyannote
        try:
            from pyannote.audio import Pipeline
        except ImportError as e:  # pragma: no cover - 依存が無い環境
            msg = (
                "pyannote.audio が見つかりません。`uv sync --extra audio` で audio "
                "依存を入れてください。"
            )
            raise PyannoteLoadError(msg) from e
        return Pipeline

    def load(self) -> None:
        """pyannote パイプラインを HF からロードして GPU に載せる。

        - HF token が無ければ `PyannoteLoadError` (gated repo アクセス不可)
        - 多重呼び出しは安全 (二度目以降は no-op)
        """
        if self._pipeline is not None:
            return
        token = self.hf_token
        if not token:
            msg = (
                f"HF token が見つかりません ({self.config.hf_token_env})。"
                "`pyannote/speaker-diarization-3.1` は gated repo なので "
                f"`export {self.config.hf_token_env}=hf_xxx` が必要です。"
            )
            raise PyannoteLoadError(msg)

        pipeline_cls = self._import_pyannote()
        try:
            logger.info("Loading pyannote pipeline: %s", self.config.model_name)
            pipeline = pipeline_cls.from_pretrained(self.config.model_name, use_auth_token=token)
            if self.config.device.startswith("cuda"):
                try:
                    import torch

                    pipeline.to(torch.device(self.config.device))
                except ImportError:  # pragma: no cover
                    logger.warning("torch 未導入のため device 指定を無視します")
            self._pipeline = pipeline
        except Exception as e:
            msg = f"pyannote パイプラインのロードに失敗しました: {e}"
            raise PyannoteLoadError(msg) from e

    def diarize(self, audio_path: Path, num_speakers: int | None = None) -> list[Turn]:
        """音声ファイルから話者交替区間を返す。

        Parameters
        ----------
        num_speakers : int | None
            None なら config.default_num_speakers を使い、それも None なら自動推定。
        """
        self.load()
        n = num_speakers if num_speakers is not None else self.config.default_num_speakers
        try:
            kwargs: dict[str, Any] = {}
            if n is not None:
                kwargs["num_speakers"] = int(n)
            annotation = self._pipeline(str(audio_path), **kwargs)
        except Exception as e:
            msg = f"pyannote 推論に失敗しました ({audio_path}): {e}"
            raise PyannoteLoadError(msg) from e
        return _annotation_to_turns(annotation)


def _annotation_to_turns(annotation: Any) -> list[Turn]:
    """`pyannote.core.Annotation` を `Turn` リストへ変換する。

    annotation.itertracks(yield_label=True) は `(Segment, track_id, label)` を返す。
    overlap は同一時刻に複数 label がある場合に True。
    """
    raw: list[tuple[float, float, str]] = []
    for item in annotation.itertracks(yield_label=True):
        seg, _track_id, label = item
        raw.append((float(seg.start), float(seg.end), str(label)))

    raw.sort(key=lambda x: x[0])
    turns: list[Turn] = []
    for start, end, speaker in raw:
        overlap = any(
            other_speaker != speaker
            and not (other_end <= start or other_start >= end)  # 時間が重なる
            for other_start, other_end, other_speaker in raw
        )
        turns.append(Turn(speaker=speaker, start_sec=start, end_sec=end, overlap=overlap))
    return turns


__all__ = [
    "PyannoteConfig",
    "PyannoteDiarizer",
    "PyannoteLoadError",
    "_annotation_to_turns",
]
