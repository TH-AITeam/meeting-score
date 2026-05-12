"""ASR / Diarization / Volume の統合パイプラインのテスト (Issue #19)。

実音声・実モデル不要のテストのみ。WhisperX や pyannote の実呼び出しは
Issue #11 で書く。本テストでは:
- assemble_utterances の純粋融合ロジック
- volume_analyzer の純関数 (rms_energy / classify_volume)
- AudioPipeline.run が fake 実装の Transcriber / Diarizer を正しく組合せること
を検証する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.asr import (
    AudioPipeline,
    Turn,
    Utterance,
    VolumeThresholds,
    Word,
    assemble_utterances,
    classify_volume,
    rms_energy,
    segment_rms_means,
)

# --------------------------------------------------------------------------
# assemble_utterances
# --------------------------------------------------------------------------


def test_assemble_utterances_attaches_words_by_midpoint() -> None:
    """単語の中点で turn に紐付くこと。"""
    words = [
        Word("こんにちは", 0.0, 1.0),
        Word("今日は", 1.0, 2.0),
        Word("会議です", 2.0, 3.0),
    ]
    turns = [
        Turn(speaker="SPEAKER_00", start_sec=0.0, end_sec=2.0),
        Turn(speaker="SPEAKER_01", start_sec=2.0, end_sec=3.0),
    ]
    utts = assemble_utterances(words, turns)
    assert len(utts) == 2
    assert utts[0].speaker == "SPEAKER_00"
    assert utts[0].text == "こんにちは今日は"
    assert utts[1].speaker == "SPEAKER_01"
    assert utts[1].text == "会議です"


def test_assemble_utterances_skips_empty_turn() -> None:
    """紐付く単語が無い turn はスキップされる。"""
    words = [Word("発言", 0.0, 1.0)]
    turns = [
        Turn(speaker="SPEAKER_00", start_sec=0.0, end_sec=1.0),
        Turn(speaker="SPEAKER_01", start_sec=10.0, end_sec=11.0),  # 単語無し
    ]
    utts = assemble_utterances(words, turns)
    assert len(utts) == 1
    assert utts[0].speaker == "SPEAKER_00"


def test_assemble_utterances_detects_overlap() -> None:
    """同時発話する別話者の turn を overlap_with に入れる。"""
    words = [Word("並行発言A", 0.0, 2.0), Word("並行発言B", 0.5, 2.5)]
    turns = [
        Turn(speaker="SPEAKER_00", start_sec=0.0, end_sec=2.0),
        Turn(speaker="SPEAKER_01", start_sec=0.5, end_sec=2.5),
    ]
    utts = assemble_utterances(words, turns, overlap_iou_threshold=0.3)
    assert "SPEAKER_01" in utts[0].overlap_with
    assert "SPEAKER_00" in utts[1].overlap_with


def test_assemble_utterances_volume_level_propagates() -> None:
    """volumes 引数の音量レベルが Utterance に反映される。"""
    words = [Word("a", 0.0, 1.0), Word("b", 1.0, 2.0)]
    turns = [
        Turn(speaker="SPEAKER_00", start_sec=0.0, end_sec=1.0),
        Turn(speaker="SPEAKER_01", start_sec=1.0, end_sec=2.0),
    ]
    utts = assemble_utterances(words, turns, volumes=["high", "low"])
    assert utts[0].volume_level == "high"
    assert utts[1].volume_level == "low"


def test_assemble_utterances_validates_volumes_length() -> None:
    """volumes と turns の長さが食い違ったら ValueError。"""
    import pytest

    words = [Word("a", 0.0, 1.0)]
    turns = [Turn(speaker="SPEAKER_00", start_sec=0.0, end_sec=1.0)]
    with pytest.raises(ValueError, match="長さが一致"):
        assemble_utterances(words, turns, volumes=["mid", "low"])


def test_utterance_id_is_sequential() -> None:
    """utterance_id が start_sec 順で u001 から振られる。"""
    words = [Word("a", 0.0, 1.0), Word("b", 2.0, 3.0), Word("c", 5.0, 6.0)]
    turns = [
        Turn(speaker="S0", start_sec=5.0, end_sec=6.0),
        Turn(speaker="S0", start_sec=0.0, end_sec=1.0),
        Turn(speaker="S0", start_sec=2.0, end_sec=3.0),
    ]
    utts = assemble_utterances(words, turns)
    assert [u.utterance_id for u in utts] == ["u001", "u002", "u003"]
    assert [u.start_sec for u in utts] == [0.0, 2.0, 5.0]


# --------------------------------------------------------------------------
# volume_analyzer
# --------------------------------------------------------------------------


def test_rms_energy_returns_one_frame_for_short_input() -> None:
    samples = np.array([0.1, -0.1, 0.1, -0.1], dtype=np.float32)
    out = rms_energy(samples, frame_size=400, hop=160)
    # 400 サンプルに満たないので 1 フレーム
    assert out.shape == (1,)
    assert abs(out[0] - 0.1) < 1e-3


def test_rms_energy_is_zero_for_silence() -> None:
    samples = np.zeros(1600, dtype=np.float32)
    out = rms_energy(samples)
    assert np.all(out == 0.0)


def test_rms_energy_grows_with_amplitude() -> None:
    quiet = (np.random.RandomState(0).randn(16000) * 0.01).astype(np.float32)
    loud = (np.random.RandomState(0).randn(16000) * 0.5).astype(np.float32)
    assert rms_energy(loud).mean() > rms_energy(quiet).mean()


def test_classify_volume_thresholds() -> None:
    th = VolumeThresholds()
    assert classify_volume(0.001, th) == "silent"
    assert classify_volume(0.01, th) == "low"
    assert classify_volume(0.05, th) == "mid"
    assert classify_volume(0.2, th) == "high"


def test_segment_rms_means_picks_segment_correctly() -> None:
    """前半が大音量、後半が無音の合成信号で区間別 RMS を取れる。"""
    sr = 16000
    samples = np.concatenate(
        [
            (np.random.RandomState(0).randn(sr) * 0.5).astype(np.float32),  # 0-1 秒
            np.zeros(sr, dtype=np.float32),  # 1-2 秒
        ]
    )
    means = segment_rms_means(samples, sr, [(0.0, 1.0), (1.0, 2.0)])
    assert means[0] > means[1]
    assert means[1] < 0.001


# --------------------------------------------------------------------------
# AudioPipeline
# --------------------------------------------------------------------------


class _FakeTranscriber:
    """テスト用: 与えられた単語列をそのまま返す。"""

    def __init__(self, words: list[Word]) -> None:
        self.words = words

    def transcribe(self, audio_path: Path) -> list[Word]:
        return list(self.words)


class _FakeDiarizer:
    def __init__(self, turns: list[Turn]) -> None:
        self.turns = turns

    def diarize(self, audio_path: Path, num_speakers: int | None = None) -> list[Turn]:
        _ = audio_path, num_speakers
        return list(self.turns)


class _FakeVolumeAnalyzer:
    def __init__(self, levels: list[str]) -> None:
        self.levels = levels

    def classify(self, audio_path: Path, spans: list[tuple[float, float]]):
        return list(self.levels[: len(spans)])


def test_pipeline_runs_with_fake_implementations(tmp_path: Path) -> None:
    audio = tmp_path / "dummy.wav"
    audio.write_bytes(b"")  # 中身は触られない
    words = [Word("会議", 0.0, 1.0), Word("はじめる", 1.0, 2.0)]
    turns = [
        Turn(speaker="SPEAKER_00", start_sec=0.0, end_sec=1.0),
        Turn(speaker="SPEAKER_01", start_sec=1.0, end_sec=2.0),
    ]
    pipeline = AudioPipeline(
        transcriber=_FakeTranscriber(words),
        diarizer=_FakeDiarizer(turns),
        volume_analyzer=_FakeVolumeAnalyzer(["high", "mid"]),
    )
    result = pipeline.run(audio)
    assert len(result) == 2
    assert isinstance(result[0], Utterance)
    assert result[0].text == "会議"
    assert result[0].volume_level == "high"
    assert result[1].text == "はじめる"


def test_pipeline_default_volume_when_analyzer_is_none(tmp_path: Path) -> None:
    audio = tmp_path / "dummy.wav"
    audio.write_bytes(b"")
    words = [Word("発言", 0.0, 1.0)]
    turns = [Turn(speaker="SPEAKER_00", start_sec=0.0, end_sec=1.0)]
    pipeline = AudioPipeline(
        transcriber=_FakeTranscriber(words),
        diarizer=_FakeDiarizer(turns),
        volume_analyzer=None,
    )
    result = pipeline.run(audio)
    assert result[0].volume_level == "mid"
