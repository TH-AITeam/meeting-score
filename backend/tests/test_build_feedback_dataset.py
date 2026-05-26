"""scripts/build_feedback_dataset.py のテスト (Issue #80)。

完了条件:
  - 2 組織のフィードバックが同一出力ファイルに混入しない（出力ディレクトリ分離）
  - consent_to_train=false の組織からは何も出力されない
  - PII（話者名）が匿名化されている
  - 全件 JSON Schema バリデーションが通る
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_feedback_dataset import (  # noqa: E402
    MeetingEval,
    axis_flags_to_normalized,
    build_for_org,
    load_feedback_from_db,
    pairwise_to_normalized,
)
from scripts.build_sft_dataset import RESPONSE_SCHEMA, validate_schema  # noqa: E402


def _utt(uid: str, speaker: str, dp: int, ts: str) -> dict:
    """評価結果込みの発言を作る。dp で総合スコアを調整。"""
    return {
        "utterance_id": uid,
        "speaker": speaker,
        "timestamp": ts,
        # 発言本文には話者名を入れない（本文への NER は Phase2。#80 は話者フィールドを匿名化）
        "text": f"発言{uid}の内容です",
        "speech_type": "提案",
        "scores": {
            "issue_clarification": 0,
            "decision_progress": dp,
            "risk_detection": 0,
            "actionability": 0,
            "groundedness": 0,
            "novelty": 0,
            "summarization": 0,
        },
        "penalties": {"duplication": 0, "verbosity": 0, "off_topic": 0, "unsupported_assertion": 0},
        "reason": "理由",
    }


def _meeting(meeting_id: str, speakers: list[str]) -> MeetingEval:
    utts = [_utt(f"u{i + 1}", spk, dp=(i % 4), ts=f"00:0{i}") for i, spk in enumerate(speakers)]
    return MeetingEval(
        meeting_id=meeting_id,
        goal="目的",
        meeting_type="decision",
        agenda=["A"],
        decision_points=["D"],
        utterances=utts,
    )


# ---------- 純関数 ----------


def test_pairwise_to_normalized_winner_resolution():
    rows = [
        {"meeting_id": "m1", "utt_a": "u1", "utt_b": "u2", "winner": "A", "source": "manual_pair"},
        {"meeting_id": "m1", "utt_a": "u1", "utt_b": "u2", "winner": "B", "source": "top5_reorder"},
        {
            "meeting_id": "m1",
            "utt_a": "u1",
            "utt_b": "u2",
            "winner": "tie",
            "source": "manual_pair",
        },
    ]
    pairs = pairwise_to_normalized(rows)
    assert len(pairs) == 2  # tie は除外
    assert (pairs[0].winner_id, pairs[0].loser_id) == ("u1", "u2")
    assert (pairs[1].winner_id, pairs[1].loser_id) == ("u2", "u1")
    # top5_reorder が source として保持される（#78 で展開済みのものをそのまま使う）
    assert pairs[1].source == "top5_reorder"


def test_axis_flags_overrated_makes_target_loser():
    # u3 が最高スコア(dp=3 相当), u1 低い。u3 を overrated とフラグ → u3 が loser になるペア
    meeting = _meeting("m1", ["X", "Y", "Z", "W"])  # u1..u4, dp=0,1,2,3
    idx = {"m1": meeting}
    flags = [{"meeting_id": "m1", "utterance_id": "u4", "direction": "overrated"}]
    pairs = axis_flags_to_normalized(flags, idx, max_pairs=20)
    # u4 は全体最高なので「より高い発言」が無い → ペア 0
    assert pairs == []
    # u1(最低) を underrated → u1 が winner、より低い発言... 無いので 0
    flags2 = [{"meeting_id": "m1", "utterance_id": "u1", "direction": "underrated"}]
    assert axis_flags_to_normalized(flags2, idx, max_pairs=20) == []
    # u2 を overrated → u3,u4 が winner（u2 が loser）
    flags3 = [{"meeting_id": "m1", "utterance_id": "u2", "direction": "overrated"}]
    p3 = axis_flags_to_normalized(flags3, idx, max_pairs=20)
    assert len(p3) == 2
    assert all(p.loser_id == "u2" and p.source == "axis_flag_synthesized" for p in p3)


def test_axis_flag_max_pairs_cap():
    meeting = _meeting("m1", [f"S{i}" for i in range(10)])  # u1..u10
    idx = {"m1": meeting}
    flags = [{"meeting_id": "m1", "utterance_id": "u1", "direction": "underrated"}]
    pairs = axis_flags_to_normalized(flags, idx, max_pairs=3)
    assert len(pairs) <= 3


# ---------- build_for_org（DB 非依存） ----------


def _run(tmp_path, org_id, consent, pairwise_rows, axis_rows, meetings):
    return build_for_org(
        org_id,
        consent=consent,
        pairwise_rows=pairwise_rows,
        axis_rows=axis_rows,
        meeting_index=meetings,
        out_root=tmp_path,
        version="v1",
        max_pairs_per_feedback=20,
        val_ratio=0.5,
        seed=1,
    )


def test_consent_false_writes_nothing(tmp_path):
    meetings = {"m1": _meeting("m1", ["山田", "佐藤"])}
    rows = [
        {"meeting_id": "m1", "utt_a": "u1", "utt_b": "u2", "winner": "A", "source": "manual_pair"}
    ]
    summary = _run(tmp_path, "org_x", False, rows, [], meetings)
    assert summary["skipped"] == "no_consent"
    # 出力ディレクトリ自体が作られない
    assert not (tmp_path / "org_x").exists()


def test_pii_speaker_anonymized(tmp_path):
    meetings = {"m1": _meeting("m1", ["山田太郎", "佐藤花子"])}
    rows = [
        {"meeting_id": "m1", "utt_a": "u1", "utt_b": "u2", "winner": "A", "source": "manual_pair"}
    ]
    _run(tmp_path, "org_1", True, rows, [], meetings)
    text = (tmp_path / "org_1" / "dpo" / "v1" / "train.jsonl").read_text(encoding="utf-8")
    text += (tmp_path / "org_1" / "dpo" / "v1" / "val.jsonl").read_text(encoding="utf-8")
    assert "山田太郎" not in text and "佐藤花子" not in text
    # 匿名 ID が使われている
    assert "A:" in text or "B:" in text


def test_all_records_schema_valid(tmp_path):
    meetings = {"m1": _meeting("m1", ["A社", "B社", "C社"])}
    rows = [
        {"meeting_id": "m1", "utt_a": "u1", "utt_b": "u3", "winner": "B", "source": "manual_pair"},
    ]
    _run(tmp_path, "org_1", True, rows, [], meetings)
    base = tmp_path / "org_1" / "dpo" / "v1"
    n = 0
    for sp in ("train", "val"):
        for line in (base / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            assert set(rec) == {"prompt", "chosen", "rejected", "meta"}
            assert rec["meta"]["org_id"] == "org_1"
            assert validate_schema(json.loads(rec["chosen"]), RESPONSE_SCHEMA)
            assert validate_schema(json.loads(rec["rejected"]), RESPONSE_SCHEMA)
            n += 1
    assert n >= 1
    # 重み回帰 pairs も出力される
    weights = (tmp_path / "org_1" / "weights" / "v1" / "pairs.jsonl").read_text(encoding="utf-8")
    wrec = json.loads(weights.splitlines()[0])
    assert wrec["winner"] == "A_better"
    assert "scores_a" in wrec and "scores_b" in wrec


def test_two_orgs_separate_dirs_no_mixing(tmp_path):
    m1 = {"m1": _meeting("m1", ["P", "Q"])}
    m2 = {"m2": _meeting("m2", ["R", "S"])}
    rows1 = [
        {"meeting_id": "m1", "utt_a": "u1", "utt_b": "u2", "winner": "A", "source": "manual_pair"}
    ]
    rows2 = [
        {"meeting_id": "m2", "utt_a": "u1", "utt_b": "u2", "winner": "A", "source": "manual_pair"}
    ]
    _run(tmp_path, "org_1", True, rows1, [], m1)
    _run(tmp_path, "org_2", True, rows2, [], m2)

    t1 = (tmp_path / "org_1" / "dpo" / "v1" / "train.jsonl").read_text(encoding="utf-8") + (
        tmp_path / "org_1" / "dpo" / "v1" / "val.jsonl"
    ).read_text(encoding="utf-8")
    t2 = (tmp_path / "org_2" / "dpo" / "v1" / "train.jsonl").read_text(encoding="utf-8") + (
        tmp_path / "org_2" / "dpo" / "v1" / "val.jsonl"
    ).read_text(encoding="utf-8")
    # 各組織のファイルに自組織の meeting_id / org_id のみ
    assert "m1" in t1 and "m2" not in t1
    assert '"org_id": "org_1"' in t1 and '"org_id": "org_2"' not in t1
    assert "m2" in t2 and "m1" not in t2


def test_missing_meeting_skipped(tmp_path):
    rows = [
        {
            "meeting_id": "ghost",
            "utt_a": "u1",
            "utt_b": "u2",
            "winner": "A",
            "source": "manual_pair",
        }
    ]
    summary = _run(tmp_path, "org_1", True, rows, [], {})
    assert summary["dpo_train"] + summary["dpo_val"] == 0
    assert summary["counts"].get("meeting_not_found") == 1


# ---------- DB レベル（org フィルタ + consent） ----------


@pytest.fixture
def _db(tmp_path, monkeypatch):
    import app.store.db as db

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'fb.db'}")
    db.reset_engine()
    db.init_db()
    yield db
    db.reset_engine()


def test_load_feedback_from_db_org_filter_and_consent(_db):
    from sqlmodel import Session

    from app.store.feedback_models import Organization, PairwiseFeedback

    with Session(_db.get_engine()) as s:
        s.add(Organization(org_id="org_1", name="o1", consent_to_train=True))
        s.add(Organization(org_id="org_2", name="o2", consent_to_train=False))
        for org in ("org_1", "org_2"):
            s.add(
                PairwiseFeedback(
                    org_id=org,
                    meeting_id="m1",
                    utt_a="u1",
                    utt_b="u2",
                    winner="A",
                    source="manual_pair",
                )
            )
        s.commit()

    consent1, pw1, _ax1 = load_feedback_from_db("org_1", None)
    assert consent1 is True
    assert len(pw1) == 1 and pw1[0]["meeting_id"] == "m1"

    # consent=false 組織は consent False を返し、行は取らない
    consent2, pw2, ax2 = load_feedback_from_db("org_2", None)
    assert consent2 is False and pw2 == [] and ax2 == []

    # 未知組織も consent False 扱い
    consent3, _, _ = load_feedback_from_db("org_unknown", None)
    assert consent3 is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
