"""ASR × Diarization × Volume を融合する統合パイプライン (Issue #19)。

実音声 I/O 部分 (Transcriber / Diarizer / VolumeAnalyzer 実装) は Issue #11 で
完成させるが、**3 つの結果を Utterance リストに融合する純ロジック**は本 PR で
書いて単体テストを通せる状態にしておく。Issue #11 では実装側を差し込むだけで
済む形にする。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.asr.base import (
    Diarizer,
    Transcriber,
    Turn,
    Utterance,
    VolumeAnalyzer,
    VolumeLevel,
    Word,
)


@dataclass
class AudioPipeline:
    """`Transcriber + Diarizer + VolumeAnalyzer` を組み合わせて `Utterance` 列を作る。

    本クラスは I/O 副作用を 3 つの依存に閉じ込め、`run()` の戻り値は純粋に
    `list[Utterance]` のみ。テストでは fake 実装を渡せる。
    """

    transcriber: Transcriber
    diarizer: Diarizer
    volume_analyzer: VolumeAnalyzer | None = None
    overlap_iou_threshold: float = 0.3

    def run(self, audio_path: Path, num_speakers: int | None = None) -> list[Utterance]:
        words = self.transcriber.transcribe(audio_path)
        turns = self.diarizer.diarize(audio_path, num_speakers=num_speakers)
        return assemble_utterances(
            words=words,
            turns=turns,
            volumes=self._compute_volumes(audio_path, turns),
            overlap_iou_threshold=self.overlap_iou_threshold,
        )

    def _compute_volumes(self, audio_path: Path, turns: list[Turn]) -> list[VolumeLevel]:
        if self.volume_analyzer is None or not turns:
            return ["mid"] * len(turns)
        spans = [(t.start_sec, t.end_sec) for t in turns]
        return self.volume_analyzer.classify(audio_path, spans)


# --------------------------------------------------------------------------
# 結合の純関数
# --------------------------------------------------------------------------


def _interval_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """2 つの区間の重なり秒数 (>= 0)。"""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """区間の IoU。両方 0 長なら 0.0。"""
    ov = _interval_overlap(a, b)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return ov / union if union > 0 else 0.0


def _attach_words_to_turn(turn: Turn, words: list[Word]) -> list[Word]:
    """単語の中点が turn 区間に入っていれば紐づける。"""
    out: list[Word] = []
    for w in words:
        mid = (w.start_sec + w.end_sec) / 2.0
        if turn.start_sec <= mid < turn.end_sec:
            out.append(w)
    return out


def assemble_utterances(
    words: list[Word],
    turns: list[Turn],
    volumes: list[VolumeLevel] | None = None,
    overlap_iou_threshold: float = 0.3,
) -> list[Utterance]:
    """3 つの入力を融合して `Utterance` 列を返す純関数。

    Parameters
    ----------
    words : 単語列 (word-level timestamp 付き)
    turns : 話者交替区間
    volumes : 各 turn と同じ並びの音量レベル。None なら全て "mid" 扱い
    overlap_iou_threshold : 他話者の turn と IoU がこの値以上なら overlap_with に追加
    """
    if volumes is None:
        volumes = ["mid"] * len(turns)
    if len(volumes) != len(turns):
        msg = f"volumes と turns の長さが一致しません ({len(volumes)} != {len(turns)})"
        raise ValueError(msg)

    sorted_turns = sorted(enumerate(turns), key=lambda kv: kv[1].start_sec)
    utterances: list[Utterance] = []
    for seq, (orig_idx, turn) in enumerate(sorted_turns):
        attached = _attach_words_to_turn(turn, words)
        text = "".join(w.text for w in attached).strip()
        if not text and not attached:
            # 単語が一つも紐づかない turn (短い無音や非言語音) はスキップ
            continue
        overlaps = [
            t.speaker
            for j, t in enumerate(turns)
            if j != orig_idx
            and t.speaker != turn.speaker
            and _interval_iou((turn.start_sec, turn.end_sec), (t.start_sec, t.end_sec))
            >= overlap_iou_threshold
        ]
        utterances.append(
            Utterance(
                utterance_id=f"u{seq + 1:04d}",
                speaker=turn.speaker,
                start_sec=turn.start_sec,
                end_sec=turn.end_sec,
                text=text,
                words=attached,
                overlap_with=overlaps,
                volume_level=volumes[orig_idx],
            )
        )
    return utterances


__all__ = [
    "AudioPipeline",
    "assemble_utterances",
]
