"""発言結合ロジック (segmenter) のテスト (Issue #11)。

Issue #11 仕様: 同一話者連続発話は 3 秒未満の無音まで 1 発言として結合。
"""

from __future__ import annotations

from app.asr.base import Utterance, Word
from app.asr.segmenter import merge_same_speaker_segments


def _utt(
    speaker: str,
    start: float,
    end: float,
    text: str = "",
    volume: str = "mid",
    overlap_with: list[str] | None = None,
) -> Utterance:
    return Utterance(
        utterance_id="tmp",
        speaker=speaker,
        start_sec=start,
        end_sec=end,
        text=text or f"発言{start}",
        words=[Word(text=text or "x", start_sec=start, end_sec=end)],
        overlap_with=overlap_with or [],
        volume_level=volume,  # type: ignore[arg-type]
    )


def test_merge_same_speaker_short_gap() -> None:
    """同一話者で gap < 3秒なら結合される。"""
    utts = [
        _utt("S0", 0.0, 2.0, text="あの"),
        _utt("S0", 4.0, 6.0, text="ですね"),  # gap = 2 秒 < 3 秒
    ]
    merged = merge_same_speaker_segments(utts, max_silence_sec=3.0)
    assert len(merged) == 1
    assert merged[0].speaker == "S0"
    assert merged[0].start_sec == 0.0
    assert merged[0].end_sec == 6.0
    assert merged[0].text == "あのですね"


def test_no_merge_when_gap_exceeds_threshold() -> None:
    """gap >= 3 秒なら結合しない。"""
    utts = [
        _utt("S0", 0.0, 2.0, text="A"),
        _utt("S0", 5.0, 7.0, text="B"),  # gap = 3.0 秒、境界 (3 未満ではない)
    ]
    merged = merge_same_speaker_segments(utts, max_silence_sec=3.0)
    assert len(merged) == 2


def test_no_merge_different_speakers() -> None:
    """話者が違えば gap が短くても結合しない。"""
    utts = [
        _utt("S0", 0.0, 2.0, text="質問"),
        _utt("S1", 2.5, 4.0, text="回答"),
    ]
    merged = merge_same_speaker_segments(utts, max_silence_sec=3.0)
    assert len(merged) == 2
    assert merged[0].speaker == "S0"
    assert merged[1].speaker == "S1"


def test_merge_chain_of_three() -> None:
    """3 連続 (全て gap < 3 秒) でも 1 つに結合される。"""
    utts = [
        _utt("S0", 0.0, 1.0, text="一"),
        _utt("S0", 2.0, 3.0, text="二"),  # gap=1
        _utt("S0", 4.0, 5.0, text="三"),  # gap=1
    ]
    merged = merge_same_speaker_segments(utts, max_silence_sec=3.0)
    assert len(merged) == 1
    assert merged[0].text == "一二三"
    assert merged[0].start_sec == 0.0
    assert merged[0].end_sec == 5.0
    assert len(merged[0].words) == 3


def test_utterance_id_is_renumbered() -> None:
    """結合後の utterance_id は u0001 から振り直される。"""
    utts = [
        _utt("S0", 0.0, 1.0),
        _utt("S1", 2.0, 3.0),
        _utt("S0", 4.0, 5.0),
    ]
    merged = merge_same_speaker_segments(utts)
    assert [u.utterance_id for u in merged] == ["u0001", "u0002", "u0003"]


def test_unordered_input_is_sorted() -> None:
    """入力が時間順でなくても、内部で sort される。"""
    utts = [
        _utt("S0", 4.0, 5.0, text="後"),
        _utt("S0", 0.0, 1.0, text="先"),
    ]
    merged = merge_same_speaker_segments(utts, max_silence_sec=10.0)
    assert len(merged) == 1
    assert merged[0].text == "先後"
    assert merged[0].start_sec == 0.0


def test_overlap_with_is_union() -> None:
    """結合時に overlap_with は和集合 + 重複排除。"""
    utts = [
        _utt("S0", 0.0, 1.0, overlap_with=["S1"]),
        _utt("S0", 1.5, 2.5, overlap_with=["S1", "S2"]),
    ]
    merged = merge_same_speaker_segments(utts, max_silence_sec=3.0)
    assert len(merged) == 1
    assert merged[0].overlap_with == ["S1", "S2"]


def test_volume_level_takes_max() -> None:
    """結合時の volume_level は強い方を採用。"""
    utts = [
        _utt("S0", 0.0, 1.0, volume="low"),
        _utt("S0", 1.5, 2.5, volume="high"),
    ]
    merged = merge_same_speaker_segments(utts, max_silence_sec=3.0)
    assert merged[0].volume_level == "high"


def test_empty_input_returns_empty() -> None:
    assert merge_same_speaker_segments([]) == []


def test_threshold_boundary_strict_less_than() -> None:
    """境界条件: gap == max_silence_sec ちょうどでは結合しない (< 3 ではなく <= 3 ではないため)。"""
    utts = [
        _utt("S0", 0.0, 2.0, text="A"),
        _utt("S0", 5.0, 7.0, text="B"),  # gap = 3.0
    ]
    merged = merge_same_speaker_segments(utts, max_silence_sec=3.0)
    assert len(merged) == 2  # 結合されない (gap >= 3 はマージ不可)
