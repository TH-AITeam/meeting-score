"""音量分析の補助ロジック (Issue #19)。

ADR 0002 の「音量分析」担当。pyannote の overlap_detection を補強し、
各発言の音量レベル (silent / low / mid / high) を分類する。

実装は **numpy のみで完結する純関数**を中心に持ち、音声ファイル I/O が
必要な部分だけ librosa に依存する。これにより:
- ユニットテストは合成 numpy 配列で純関数を直接叩ける
- librosa 未インストール環境でも import 自体は通る
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.asr.base import VolumeLevel


@dataclass(frozen=True)
class VolumeThresholds:
    """RMS energy (linear) に対する閾値。

    既定値は人手調整の暫定値。Issue #19 の実測フェーズで `data/eval_audio/` の
    全体分布を見て再調整する想定。
    """

    silent_below: float = 0.005
    low_below: float = 0.02
    high_above: float = 0.10


def rms_energy(samples: np.ndarray, frame_size: int = 400, hop: int = 160) -> np.ndarray:
    """短時間 RMS energy を計算する (sr=16000 で 25ms 窓 / 10ms hop)。

    Parameters
    ----------
    samples : np.ndarray (mono, float)
    frame_size : int
        1 フレームのサンプル数。
    hop : int
        フレーム間のホップ。

    Returns
    -------
    np.ndarray of shape (n_frames,)
        各フレームの RMS 値。
    """
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=-1)
    n_frames = max(0, 1 + (samples.shape[0] - frame_size) // hop)
    if n_frames == 0:
        # 1 フレームに満たない場合は全体の RMS を 1 点で返す
        return np.array(
            [float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))],
            dtype=np.float32,
        )
    out = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        chunk = samples[i * hop : i * hop + frame_size]
        out[i] = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
    return out


def classify_volume(rms_mean: float, thresholds: VolumeThresholds) -> VolumeLevel:
    """1 セグメントの平均 RMS から音量レベルを 4 段階に分類する。"""
    if rms_mean < thresholds.silent_below:
        return "silent"
    if rms_mean < thresholds.low_below:
        return "low"
    if rms_mean >= thresholds.high_above:
        return "high"
    return "mid"


def segment_rms_means(
    samples: np.ndarray,
    sample_rate: int,
    spans: list[tuple[float, float]],
    frame_size: int = 400,
    hop: int = 160,
) -> list[float]:
    """各時間区間 (start_sec, end_sec) の平均 RMS を返す。

    pyannote の Turn 区間に対して呼び、`classify_volume` と組み合わせて使う。
    """
    rms = rms_energy(samples, frame_size=frame_size, hop=hop)
    means: list[float] = []
    frame_sec = hop / sample_rate
    n_frames = rms.shape[0]
    for start_sec, end_sec in spans:
        if end_sec <= start_sec or n_frames == 0:
            means.append(0.0)
            continue
        start_f = max(0, int(start_sec / frame_sec))
        end_f = min(n_frames, int(end_sec / frame_sec) + 1)
        if end_f <= start_f:
            means.append(float(rms[start_f]) if start_f < n_frames else 0.0)
        else:
            means.append(float(rms[start_f:end_f].mean()))
    return means


class LibrosaVolumeAnalyzer:
    """音声ファイルを librosa で読み込んで音量レベルを返す。

    librosa は optional 依存 (`uv sync --extra audio`) なので、未導入環境では
    classify() を呼んだ瞬間に `ImportError` で落とす。
    """

    def __init__(self, thresholds: VolumeThresholds | None = None) -> None:
        self.thresholds = thresholds or VolumeThresholds()

    def classify(
        self, audio_path: Path, spans: list[tuple[float, float]]
    ) -> list[VolumeLevel]:
        try:
            import librosa
        except ImportError as e:  # pragma: no cover - 依存が無い環境
            msg = (
                "librosa が見つかりません。`uv sync --extra audio` で audio "
                "依存を入れてください。"
            )
            raise ImportError(msg) from e
        samples, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        means = segment_rms_means(samples, sr, spans)
        return [classify_volume(m, self.thresholds) for m in means]


__all__ = [
    "LibrosaVolumeAnalyzer",
    "VolumeThresholds",
    "classify_volume",
    "rms_energy",
    "segment_rms_means",
]
