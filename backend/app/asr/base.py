"""ASR / Diarization パイプラインの共通インタフェース (Issue #19)。

実装本体は Issue #11 (`app/asr/` 本実装) で書く。
本モジュールは型 + Protocol を確定させ、後続実装の receptacle として機能する。

データの流れ:
    音声ファイル
      → Transcriber  → list[Word]   (テキスト + word-level timestamp)
      → Diarizer    → list[Turn]   (話者ラベル付きの時間区間)
      → VolumeAnalyzer → dict[span, volume_level]
      → AudioPipeline.run → list[Utterance]  (上記を融合)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

VolumeLevel = Literal["silent", "low", "mid", "high"]


@dataclass
class Word:
    """ASR から得られる単語と timestamp。

    start_sec / end_sec は音声ファイル先頭からの秒数。
    confidence は ASR 側が出してくれる場合のみセット。
    """

    text: str
    start_sec: float
    end_sec: float
    confidence: float | None = None


@dataclass
class Turn:
    """Diarization から得られる話者交替区間。

    speaker は "SPEAKER_00" / "SPEAKER_01" のような pyannote 形式のラベル。
    overlap=True のときは同時発話が検出された区間。
    """

    speaker: str
    start_sec: float
    end_sec: float
    overlap: bool = False


@dataclass
class Utterance:
    """ASR × Diarization × Volume を融合した最終的な発言単位。

    Issue #11 の音声入力パイプラインが本クラスのリストを上流の
    `MeetingInput.utterances` 形式へ変換する想定。
    """

    utterance_id: str
    speaker: str
    start_sec: float
    end_sec: float
    text: str
    words: list[Word] = field(default_factory=list)
    overlap_with: list[str] = field(default_factory=list)  # 同時発話している他話者
    volume_level: VolumeLevel = "mid"

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


class Transcriber(Protocol):
    """音声ファイル → 単語列 (word-level timestamp 付き)。

    実装例: `app.asr.whisperx_transcriber.WhisperXTranscriber`
    """

    def transcribe(self, audio_path: Path) -> list[Word]: ...


class Diarizer(Protocol):
    """音声ファイル → 話者交替区間。

    実装例: `app.asr.pyannote_diarizer.PyannoteDiarizer`
    """

    def diarize(self, audio_path: Path, num_speakers: int | None = None) -> list[Turn]: ...


class VolumeAnalyzer(Protocol):
    """時間区間ごとの音量レベルを返す。

    実装例: `app.asr.volume_analyzer.LibrosaVolumeAnalyzer`
    """

    def classify(self, audio_path: Path, spans: list[tuple[float, float]]) -> list[VolumeLevel]: ...


__all__ = [
    "Diarizer",
    "Transcriber",
    "Turn",
    "Utterance",
    "VolumeAnalyzer",
    "VolumeLevel",
    "Word",
]
