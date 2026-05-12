"""音声処理パッケージ (Issue #19 で型定義、Issue #11 で本実装)。

- `base`: Transcriber / Diarizer / VolumeAnalyzer の Protocol と
  `Word` / `Turn` / `Utterance` データクラス
- `whisperx_transcriber`: WhisperX ベースの Transcriber スケルトン
- `pyannote_diarizer`: pyannote ベースの Diarizer スケルトン
- `volume_analyzer`: librosa / numpy による音量分析
- `pipeline`: 3 つを融合して `list[Utterance]` を返す統合
"""

from app.asr.base import (
    Diarizer,
    Transcriber,
    Turn,
    Utterance,
    VolumeAnalyzer,
    VolumeLevel,
    Word,
)
from app.asr.pipeline import AudioPipeline, assemble_utterances
from app.asr.pyannote_diarizer import PyannoteConfig, PyannoteDiarizer
from app.asr.volume_analyzer import (
    LibrosaVolumeAnalyzer,
    VolumeThresholds,
    classify_volume,
    rms_energy,
    segment_rms_means,
)
from app.asr.whisperx_transcriber import WhisperXConfig, WhisperXTranscriber

__all__ = [
    "AudioPipeline",
    "Diarizer",
    "LibrosaVolumeAnalyzer",
    "PyannoteConfig",
    "PyannoteDiarizer",
    "Transcriber",
    "Turn",
    "Utterance",
    "VolumeAnalyzer",
    "VolumeLevel",
    "VolumeThresholds",
    "WhisperXConfig",
    "WhisperXTranscriber",
    "Word",
    "assemble_utterances",
    "classify_volume",
    "rms_energy",
    "segment_rms_means",
]
