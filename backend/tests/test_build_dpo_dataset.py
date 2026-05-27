"""scripts/build_dpo_dataset.py と eval_pairwise の純ロジックのテスト (Issue #15)。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from string import Template

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "lora"))

from scripts.build_dpo_dataset import (  # noqa: E402
    build_arg_parser,
    build_dpo_records,
    load_sample_index,
    normalize_winner,
    run,
    synthesize_pairs,
    total_score,
)
from scripts.build_sft_dataset import Sample  # noqa: E402

PROMPT_TEMPLATE = Template(
    "type=$meeting_type goal=$meeting_goal agenda=$agenda dp=$decision_points "
    "topic=$current_topic before=$before_utterances "
    "spk=$target_speaker ts=$target_timestamp text=$target_text after=$after_utterances"
)


def _assistant(total_hint: int) -> dict:
    """合計が total_hint 付近になる評価 JSON を作る（テスト用）。"""
    scores = {
        k: 0
        for k in [
            "issue_clarification",
            "decision_progress",
            "risk_detection",
            "actionability",
            "groundedness",
            "novelty",
            "summarization",
        ]
    }
    # 正の合計は decision_progress などに振る
    v = max(0, min(3, total_hint))
    scores["decision_progress"] = v
    scores["issue_clarification"] = max(0, min(3, total_hint - v))
    return {
        "speech_type": "提案",
        "scores": scores,
        "penalties": {
            k: 0
            for k in [
                "duplication",
                "verbosity",
                "off_topic",
                "unsupported_assertion",
                "override",
            ]
        },
        "reason": "r",
    }


def _sample(meeting: str, uid: str, total: int) -> Sample:
    return Sample(
        meeting_id=meeting,
        utterance_id=uid,
        source="distilled",
        user=f"prompt-{uid}",
        assistant=_assistant(total),
    )


# ---------- 純関数 ----------


def test_normalize_winner():
    assert normalize_winner("A_better") == "a"
    assert normalize_winner("B") == "b"
    assert normalize_winner("tie") == "tie"
    assert normalize_winner("???") is None


def test_total_score():
    assert total_score(_assistant(5)) == 5


def test_total_score_includes_override_penalty():
    assistant = _assistant(5)
    assistant["penalties"]["override"] = -2
    assert total_score(assistant) == 3


def test_synthesize_pairs_picks_high_vs_low():
    index = {
        ("m1", "1"): _sample("m1", "1", 6),
        ("m1", "2"): _sample("m1", "2", 3),
        ("m1", "3"): _sample("m1", "3", 0),
    }
    pairs = synthesize_pairs(index, min_margin=4, max_per_meeting=8)
    # 6 vs 0 が margin>=4 を満たす
    assert len(pairs) >= 1
    p = pairs[0]
    assert p["winner"] == "A_better"
    assert p["source"] == "synthetic"
    # utt_a (winner) の合計 >= utt_b (loser)
    assert total_score(index[("m1", p["utt_a"])].assistant) >= total_score(
        index[("m1", p["utt_b"])].assistant
    )


def test_build_dpo_records_winner_is_chosen():
    from collections import Counter

    index = {
        ("m1", "1"): _sample("m1", "1", 7),
        ("m1", "2"): _sample("m1", "2", 1),
    }
    pairs = [
        {"meeting_id": "m1", "utt_a": "1", "utt_b": "2", "winner": "A_better", "source": "gold"}
    ]
    counts: Counter = Counter()
    by_meeting = build_dpo_records(pairs, index, symmetric=True, result_counts=counts)
    recs = by_meeting["m1"]
    assert len(recs) == 2  # symmetric
    # 1 件目: winner(1) のプロンプト, chosen=winner評価, rejected=loser評価
    r0 = recs[0]
    assert r0["prompt"] == "prompt-1"
    assert json.loads(r0["chosen"])["scores"]["decision_progress"] >= 0
    assert r0["meta"]["chosen_id"] == "1" and r0["meta"]["rejected_id"] == "2"
    # 対称: loser(2) のプロンプトでも winner/loser の選好方向は維持する
    assert recs[1]["prompt"] == "prompt-2"
    assert recs[1]["meta"]["chosen_id"] == "1"
    assert recs[1]["meta"]["rejected_id"] == "2"
    assert counts["pairs_used"] == 1


def test_build_dpo_records_skips_tie_and_missing():
    from collections import Counter

    index = {("m1", "1"): _sample("m1", "1", 5)}
    pairs = [
        {"meeting_id": "m1", "utt_a": "1", "utt_b": "2", "winner": "tie"},
        {"meeting_id": "m1", "utt_a": "1", "utt_b": "99", "winner": "A_better"},  # 99 欠落
    ]
    counts: Counter = Counter()
    by_meeting = build_dpo_records(pairs, index, symmetric=False, result_counts=counts)
    assert by_meeting == {}
    assert counts["tie_skipped"] == 1
    assert counts["missing_utterance"] == 1


def test_build_dpo_records_no_symmetric_single_record():
    from collections import Counter

    index = {("m1", "1"): _sample("m1", "1", 7), ("m1", "2"): _sample("m1", "2", 1)}
    pairs = [{"meeting_id": "m1", "utt_a": "1", "utt_b": "2", "winner": "A_better"}]
    by_meeting = build_dpo_records(pairs, index, symmetric=False, result_counts=Counter())
    assert len(by_meeting["m1"]) == 1


# ---------- eval_pairwise 純関数 ----------


def test_eval_pairwise_metrics():
    import eval_pairwise as ep

    assistant = _assistant(5)
    assistant["penalties"]["override"] = -2
    assert ep.total_of(assistant) == 3

    totals = {("m1", "1"): 7.0, ("m1", "2"): 2.0, ("m1", "3"): 5.0}
    pairs = [
        {"meeting_id": "m1", "utt_a": "1", "utt_b": "2", "winner": "A_better"},  # 正(7>2)
        {"meeting_id": "m1", "utt_a": "2", "utt_b": "3", "winner": "A_better"},  # 誤(2<5)
        {"meeting_id": "m1", "utt_a": "1", "utt_b": "4", "winner": "A_better"},  # 同点は誤
        {"meeting_id": "m1", "utt_a": "1", "utt_b": "2", "winner": "tie"},  # 除外
    ]
    totals[("m1", "4")] = 7.0
    acc, n = ep.pairwise_accuracy(pairs, totals)
    assert n == 3
    assert acc == pytest.approx(1 / 3)

    assert ep.jaccard({"1", "2"}, {"1", "2"}) == 1.0
    assert ep.jaccard({"1", "2"}, {"3", "4"}) == 0.0
    assert ep.jaccard({"1", "2", "3"}, {"2", "3", "4"}) == 0.5
    assert ep.top_k_by_total({"a": 1, "b": 9, "c": 5}, k=2) == {"b", "c"}

    pred = {"m1": {"1", "2", "3", "4", "5"}}
    gold = {"m1": {"1", "2", "3", "9", "8"}}
    mean_j, n_meet = ep.mean_top5_jaccard(pred, gold)
    assert n_meet == 1
    assert mean_j == pytest.approx(3 / 7)


# ---------- エンドツーエンド（既存 distill ラベルで合成） ----------


def test_run_end_to_end_synthesize(tmp_path):
    # tmp に distill jobs/labels を 2 会議ぶん作る
    distill = tmp_path / "distill"
    (distill / "jobs").mkdir(parents=True)
    (distill / "labels").mkdir(parents=True)
    for mid in ("mA", "mB"):
        utts, labels = [], []
        # total=0 は build_sft_dataset の reject(extreme_all_zero)で落ちるため避け、
        # かつ min-margin(4) を満たす差をつける（6 と 2 で差 4）。
        for order, tot in [(1, 6), (2, 5), (3, 2)]:
            utts.append(
                {
                    "order": order,
                    "speaker": "s",
                    "timestamp": str(order),
                    "text": f"t{order}",
                    "before_text": "(なし)",
                    "after_text": "(なし)",
                }
            )
            labels.append({"order": order, **_assistant(tot)})
        (distill / "jobs" / f"{mid}.json").write_text(
            json.dumps({"issueID": mid, "utterances": utts}), encoding="utf-8"
        )
        (distill / "labels" / f"{mid}.json").write_text(
            json.dumps({"meta": {"goal": "g", "meeting_type": "decision"}, "labels": labels}),
            encoding="utf-8",
        )

    out_dir = tmp_path / "dpo" / "v1"
    args = build_arg_parser().parse_args(
        [
            "--distill-dir",
            str(distill),
            "--gold-dir",
            str(tmp_path / "nogold"),
            "--out-dir",
            str(out_dir),
            "--synthesize-from-labels",
            "--no-symmetric",  # 主レコードのみ → chosen 合計 >= rejected 合計が全件で成立
            "--val-ratio",
            "0.5",
        ]
    )
    stats = run(args)
    assert stats["n_input_pairs"] >= 1
    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "val.jsonl").exists()

    # 全レコードで chosen 合計 >= rejected 合計（winner=chosen の整合）
    for sp in ("train", "val"):
        for line in (out_dir / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            assert set(rec) == {"prompt", "chosen", "rejected", "meta"}
            ch = json.loads(rec["chosen"])
            rj = json.loads(rec["rejected"])
            assert total_score(ch) >= total_score(rj)


def test_load_sample_index_empty(tmp_path):
    idx = load_sample_index(tmp_path / "x", tmp_path / "y", PROMPT_TEMPLATE)
    assert idx == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
