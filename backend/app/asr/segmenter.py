"""発言単位への正規化 (Issue #11)。

`AudioPipeline.run()` が返す `Utterance[]` (PR #54 の `assemble_utterances` 由来) は
pyannote の `Turn` 1 つ = 1 発言になっている。これでは「同じ話者が短い間 (相槌等) で
区切られて喋っているケース」が冗長に複数 utterance になるため、

  - 同一話者の連続 turn
  - turn 間の無音が `max_silence_sec` (既定 3 秒) 未満

の条件で 1 発言にマージする。

Issue #11 仕様:
> word-ts と話者を突き合わせ、発言単位に正規化
> 同一話者連続の発話は 3 秒未満の無音まで 1 発言として結合
"""

from __future__ import annotations

from app.asr.base import Utterance


def merge_same_speaker_segments(
    utterances: list[Utterance],
    max_silence_sec: float = 3.0,
) -> list[Utterance]:
    """同一話者連続発話を結合する。

    Parameters
    ----------
    utterances : list[Utterance]
        時間順に並んでいる必要は無い。内部で `start_sec` 昇順にソートする。
    max_silence_sec : float
        この値未満の無音を挟んだ同一話者の連続発話を 1 つに結合する。

    Returns
    -------
    list[Utterance]
        結合済み Utterance。`utterance_id` は時間順に `u0001` から振り直す。
        - `text` は連結
        - `words` は連結
        - `start_sec` は先頭、`end_sec` は末尾
        - `overlap_with` は重複排除した和集合
        - `volume_level` は構成 utterance の中で最大 (silent < low < mid < high)
    """
    if not utterances:
        return []

    sorted_utts = sorted(utterances, key=lambda u: u.start_sec)
    merged: list[Utterance] = []
    for u in sorted_utts:
        if merged and _can_merge(merged[-1], u, max_silence_sec):
            merged[-1] = _merge_pair(merged[-1], u)
        else:
            merged.append(u)

    # utterance_id を u0001 から振り直す (結合前後で連番がズレるため)
    return [
        Utterance(
            utterance_id=f"u{i + 1:04d}",
            speaker=u.speaker,
            start_sec=u.start_sec,
            end_sec=u.end_sec,
            text=u.text,
            words=u.words,
            overlap_with=u.overlap_with,
            volume_level=u.volume_level,
        )
        for i, u in enumerate(merged)
    ]


def _can_merge(prev: Utterance, curr: Utterance, max_silence_sec: float) -> bool:
    """同一話者で curr の start - prev の end が max_silence_sec 未満なら結合可。"""
    if prev.speaker != curr.speaker:
        return False
    gap = curr.start_sec - prev.end_sec
    return gap < max_silence_sec


_VOLUME_RANK = {"silent": 0, "low": 1, "mid": 2, "high": 3}


def _max_volume(a: str, b: str) -> str:
    """volume level の強さで大きい方を返す。"""
    return a if _VOLUME_RANK.get(a, 2) >= _VOLUME_RANK.get(b, 2) else b


def _merge_pair(prev: Utterance, curr: Utterance) -> Utterance:
    """2 つの Utterance を 1 つに結合する。utterance_id は仮 (後で振り直し)。"""
    return Utterance(
        utterance_id=prev.utterance_id,  # 後で振り直す
        speaker=prev.speaker,
        start_sec=prev.start_sec,
        end_sec=curr.end_sec,
        text=(prev.text + curr.text).strip(),
        words=prev.words + curr.words,
        overlap_with=sorted(set(prev.overlap_with) | set(curr.overlap_with)),
        volume_level=_max_volume(prev.volume_level, curr.volume_level),  # type: ignore[arg-type]
    )


__all__ = [
    "merge_same_speaker_segments",
]
