"""scripts/build_sft_dataset.py のテスト (Issue #13)。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from string import Template

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.build_sft_dataset import (
    RESPONSE_SCHEMA,
    BuildResult,
    build_arg_parser,
    classify_reject,
    collect_split,
    load_tier,
    normalized_assistant,
    run,
    split_meetings,
    validate_schema,
)

PROMPT_TEMPLATE = Template(
    "type=$meeting_type goal=$meeting_goal agenda=$agenda dp=$decision_points "
    "topic=$current_topic before=$before_utterances "
    "spk=$target_speaker ts=$target_timestamp text=$target_text after=$after_utterances"
)


def _good_label(order: int, reason: str = "具体的な理由") -> dict:
    return {
        "order": order,
        "speech_type": "提案",
        "scores": {"decision_progress": 2, "actionability": 1},
        "penalties": {},
        "reason": reason,
    }


def _utt(order: int) -> dict:
    return {
        "order": order,
        "speaker": f"話者{order}",
        "timestamp": str(order),
        "text": f"発言{order}の本文",
        "before_text": "(なし)",
        "after_text": "(なし)",
    }


def _make_meeting(root: Path, meeting_id: str, labels: list[dict]) -> None:
    """jobs/<id>.json と labels/<id>.json を tmp に作る。"""
    jobs = root / "jobs"
    labs = root / "labels"
    jobs.mkdir(parents=True, exist_ok=True)
    labs.mkdir(parents=True, exist_ok=True)
    (jobs / f"{meeting_id}.json").write_text(
        json.dumps({"issueID": meeting_id, "utterances": [_utt(lab["order"]) for lab in labels]}),
        encoding="utf-8",
    )
    (labs / f"{meeting_id}.json").write_text(
        json.dumps({"meta": {"goal": "テスト", "meeting_type": "decision"}, "labels": labels}),
        encoding="utf-8",
    )


# ---------- スキーマ検証 ----------


def test_validate_schema_accepts_normalized():
    assistant = normalized_assistant(_good_label(1))
    assert validate_schema(assistant, RESPONSE_SCHEMA)


def test_validate_schema_rejects_out_of_range_and_extra_keys():
    assert not validate_schema({"speech_type": "提案"}, RESPONSE_SCHEMA)  # required 欠落
    bad = normalized_assistant(_good_label(1))
    bad["scores"]["decision_progress"] = 9  # 範囲外
    assert not validate_schema(bad, RESPONSE_SCHEMA)
    extra = normalized_assistant(_good_label(1))
    extra["unexpected"] = 1  # additionalProperties=false
    assert not validate_schema(extra, RESPONSE_SCHEMA)


# ---------- reject loop ----------


def test_reject_empty_reason():
    label = _good_label(1, reason="  ")
    assert classify_reject(label, normalized_assistant(label)) == "empty_reason"


def test_reject_all_zero_extreme():
    label = {"order": 1, "speech_type": "情報共有", "scores": {}, "penalties": {}, "reason": "x"}
    assert classify_reject(label, normalized_assistant(label)) == "extreme_all_zero"


def test_keep_extreme_flag_keeps_all_zero():
    label = {"order": 1, "speech_type": "情報共有", "scores": {}, "penalties": {}, "reason": "x"}
    assert classify_reject(label, normalized_assistant(label), keep_extreme=True) is None


def test_good_label_not_rejected():
    label = _good_label(1)
    assert classify_reject(label, normalized_assistant(label)) is None


# ---------- ローダ ----------


def test_load_tier_builds_samples_with_meta(tmp_path):
    _make_meeting(tmp_path, "m1", [_good_label(1), _good_label(2)])
    result = BuildResult()
    load_tier(tmp_path / "jobs", tmp_path / "labels", "distilled", PROMPT_TEMPLATE, result)
    assert "m1" in result.by_meeting
    samples = result.by_meeting["m1"]
    assert len(samples) == 2
    rec = samples[0].to_record()
    assert [m["role"] for m in rec["messages"]] == ["user", "assistant"]
    assert rec["meta"] == {"source": "distilled", "meeting_id": "m1", "utterance_id": "1"}
    # user は本番プロンプトの体裁、assistant は schema 準拠
    assert "text=発言1の本文" in rec["messages"][0]["content"]
    assert validate_schema(json.loads(rec["messages"][1]["content"]), RESPONSE_SCHEMA)


def test_load_tier_skips_rejected(tmp_path):
    _make_meeting(
        tmp_path,
        "m1",
        [_good_label(1), _good_label(2, reason=""), _good_label(3)],
    )
    result = BuildResult()
    load_tier(tmp_path / "jobs", tmp_path / "labels", "distilled", PROMPT_TEMPLATE, result)
    assert len(result.by_meeting["m1"]) == 2  # 空 reason の 1 件が落ちる
    assert result.reject_counts["empty_reason"] == 1


def test_load_tier_missing_dir_noop(tmp_path):
    result = BuildResult()
    load_tier(
        tmp_path / "nope" / "jobs", tmp_path / "nope" / "labels", "gold", PROMPT_TEMPLATE, result
    )
    assert result.by_meeting == {}


def test_load_tier_merges_same_meeting_from_multiple_sources(tmp_path):
    distill = tmp_path / "distill"
    gold = tmp_path / "gold"
    _make_meeting(distill, "m1", [_good_label(1)])
    _make_meeting(gold, "m1", [_good_label(2)])

    result = BuildResult()
    load_tier(distill / "jobs", distill / "labels", "distilled", PROMPT_TEMPLATE, result)
    load_tier(gold / "jobs", gold / "labels", "gold", PROMPT_TEMPLATE, result)

    assert [s.source for s in result.by_meeting["m1"]] == ["distilled", "gold"]
    assert [s.utterance_id for s in result.by_meeting["m1"]] == ["1", "2"]
    assert result.source_meetings["m1"] == {"distilled", "gold"}


# ---------- 会議単位分割 ----------


def test_split_meetings_disjoint_and_covers_all():
    meetings = [f"m{i}" for i in range(10)]
    sets = split_meetings(meetings, val_ratio=0.2, test_ratio=0.2, seed=1)
    assert sets["train"] and sets["val"] and sets["test"]
    assert sets["train"] & sets["val"] == set()
    assert sets["train"] & sets["test"] == set()
    assert sets["val"] & sets["test"] == set()
    assert sets["train"] | sets["val"] | sets["test"] == set(meetings)


def test_split_single_meeting_all_train():
    sets = split_meetings(["only"], val_ratio=0.2, test_ratio=0.2, seed=1)
    assert sets["train"] == {"only"}
    assert sets["val"] == set() and sets["test"] == set()


def test_collect_split_routes_by_meeting(tmp_path):
    _make_meeting(tmp_path, "a", [_good_label(1)])
    _make_meeting(tmp_path, "b", [_good_label(1)])
    result = BuildResult()
    load_tier(tmp_path / "jobs", tmp_path / "labels", "distilled", PROMPT_TEMPLATE, result)
    sets = {"train": {"a"}, "val": {"b"}, "test": set()}
    splits = collect_split(result.by_meeting, sets)
    assert len(splits["train"]) == 1 and len(splits["val"]) == 1 and splits["test"] == []


def test_collect_split_sorts_meetings_for_stable_output(tmp_path):
    _make_meeting(tmp_path, "a", [_good_label(1)])
    _make_meeting(tmp_path, "b", [_good_label(1)])
    result = BuildResult()
    load_tier(tmp_path / "jobs", tmp_path / "labels", "distilled", PROMPT_TEMPLATE, result)

    splits = collect_split(result.by_meeting, {"train": {"b", "a"}, "val": set(), "test": set()})

    assert [s.meeting_id for s in splits["train"]] == ["a", "b"]


# ---------- エンドツーエンド ----------


def test_run_end_to_end(tmp_path):
    distill = tmp_path / "distill"
    for i in range(6):
        _make_meeting(distill, f"m{i}", [_good_label(1), _good_label(2)])
    out_dir = tmp_path / "sft" / "v1"
    stats_out = tmp_path / "stats.md"

    args = build_arg_parser().parse_args(
        [
            "--distill-dir",
            str(distill),
            "--gold-dir",
            str(tmp_path / "gold"),  # 存在しない -> 0 件
            "--out-dir",
            str(out_dir),
            "--stats-out",
            str(stats_out),
            "--val-ratio",
            "0.34",
            "--test-ratio",
            "0.34",
        ]
    )
    stats = run(args)

    # 出力ファイル
    for sp in ("train", "val", "test"):
        assert (out_dir / f"{sp}.jsonl").exists()
    assert (out_dir / "README.md").exists()
    assert stats_out.exists()

    # 全件 schema pass + 会議 disjoint
    meetings = {"train": set(), "val": set(), "test": set()}
    for sp in ("train", "val", "test"):
        for line in (out_dir / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            assert validate_schema(json.loads(rec["messages"][1]["content"]), RESPONSE_SCHEMA)
            meetings[sp].add(rec["meta"]["meeting_id"])
    assert meetings["val"] & meetings["train"] == set()
    assert meetings["test"] & meetings["train"] == set()
    assert meetings["val"] & meetings["test"] == set()
    assert stats["source_dist"] == {"distilled": 12}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
