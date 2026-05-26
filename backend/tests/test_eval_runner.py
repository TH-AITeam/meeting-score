"""evals.runner のテスト (Issue #5)。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from app.schemas.models import Penalties, Scores
from app.scoring.weights import PenaltyWeights, ScoringWeights
from evals.protocol import EvaluationResult
from evals.runner import _evaluate_meeting, _human_ranks_from_pairs, run_eval
from evals.schema import PairwiseAnnotation

SAMPLE_MEETING = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "sample_meetings"
    / "sample_meeting_01.json"
)


class _OrderedEvaluator:
    """utterance_id (u001, u002, ...) の末尾数字に比例した点数を返す。

    score 順を utterance_id 順と一致させ、人手アノテと照合しやすくする。
    """

    def evaluate(self, ctx) -> EvaluationResult:
        idx = int(ctx.target_utterance.utterance_id.lstrip("u"))
        s = max(0, min(3, idx % 4))
        return EvaluationResult(
            speech_type="提案",
            scores=Scores(
                issue_clarification=s,
                decision_progress=s,
                risk_detection=s,
                actionability=s,
                groundedness=s,
                novelty=s,
                summarization=s,
            ),
            penalties=Penalties(),
        )


class _NeutralEvaluator:
    def evaluate(self, ctx) -> EvaluationResult:
        return EvaluationResult(
            speech_type="情報共有",
            scores=Scores(),
            penalties=Penalties(),
        )


class _DuplicationPenaltyEvaluator:
    def evaluate(self, ctx) -> EvaluationResult:
        return EvaluationResult(
            speech_type="情報共有",
            scores=Scores(),
            penalties=Penalties(duplication=-1),
        )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_human_ranks_from_pairs_aggregates_votes():
    pairs = [
        PairwiseAnnotation(meeting_id="m", utt_a="u1", utt_b="u2", winner="A_better"),
        PairwiseAnnotation(meeting_id="m", utt_a="u1", utt_b="u3", winner="A_better"),
        PairwiseAnnotation(meeting_id="m", utt_a="u2", utt_b="u3", winner="tie"),
    ]
    score = _human_ranks_from_pairs(pairs, ["u1", "u2", "u3"])
    assert score is not None
    # u1: +2 (2勝), u2: -1 (1敗, 1引き分け), u3: -1 (1敗, 1引き分け)
    assert score["u1"] == 2
    assert score["u2"] == -1
    assert score["u3"] == -1


def test_human_ranks_from_pairs_returns_none_when_no_match():
    pairs = [
        PairwiseAnnotation(meeting_id="m", utt_a="x1", utt_b="x2", winner="A_better"),
    ]
    assert _human_ranks_from_pairs(pairs, ["u1", "u2"]) is None


def test_evaluate_meeting_applies_rule_corrections(tmp_path):
    meeting_file = tmp_path / "m_dup.json"
    meeting_file.write_text(
        json.dumps(
            {
                "meeting_id": "m_dup",
                "title": "dup",
                "goal": "dup",
                "utterances": [
                    {
                        "utterance_id": "u001",
                        "speaker": "A",
                        "timestamp": "00:00:01",
                        "text": "同じ内容です。",
                    },
                    {
                        "utterance_id": "u002",
                        "speaker": "B",
                        "timestamp": "00:00:02",
                        "text": "同じ内容です。",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    meeting_id, evaluated = _evaluate_meeting(
        meeting_file,
        _NeutralEvaluator(),
        ScoringWeights(),
    )

    assert meeting_id == "m_dup"
    assert evaluated[0].penalties.duplication == 0
    assert evaluated[1].penalties.duplication == -1
    assert evaluated[1].total_score == -1.0


def test_evaluate_meeting_applies_penalty_weights_to_totals_and_corrections(tmp_path):
    meeting_file = tmp_path / "m_weighted_dup.json"
    meeting_file.write_text(
        json.dumps(
            {
                "meeting_id": "m_weighted_dup",
                "title": "dup",
                "goal": "dup",
                "utterances": [
                    {
                        "utterance_id": "u001",
                        "speaker": "A",
                        "timestamp": "00:00:01",
                        "text": "同じ内容です。",
                    },
                    {
                        "utterance_id": "u002",
                        "speaker": "B",
                        "timestamp": "00:00:02",
                        "text": "同じ内容です。",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    meeting_id, evaluated = _evaluate_meeting(
        meeting_file,
        _DuplicationPenaltyEvaluator(),
        ScoringWeights(),
        penalty_weights=PenaltyWeights(duplication=2.0),
    )

    assert meeting_id == "m_weighted_dup"
    assert evaluated[0].penalties.duplication == -1
    assert evaluated[0].total_score == -2.0
    assert evaluated[1].penalties.duplication == -2
    assert evaluated[1].total_score == -4.0


def test_run_eval_with_real_sample_meeting(tmp_path):
    """sample_meeting_01.json を OrderedEvaluator で評価し、Top5 Jaccard と
    pairwise accuracy を確認する。"""
    assert SAMPLE_MEETING.exists(), "サンプル会議データが見つかりません"

    dataset = tmp_path / "v_test"
    meetings_dir = tmp_path / "meetings"
    meetings_dir.mkdir(parents=True)
    # _find_meeting_file は `{meeting_id}.json` 形式を期待するので、
    # 元データを meeting_id 名でコピーする
    shutil.copy(SAMPLE_MEETING, meetings_dir / "m001.json")

    # 数字が大きい u020〜u023 が高スコア寄りになるよう作る
    _write_jsonl(
        dataset / "top_bottom.jsonl",
        [
            {
                "meeting_id": "m001",
                "top5": ["u023", "u022", "u021", "u020", "u019"],
                "bottom5": ["u001", "u002", "u003", "u004", "u005"],
            }
        ],
    )
    _write_jsonl(
        dataset / "pairs.jsonl",
        [
            # OrderedEvaluator のスコアは idx % 4 なので、差が出る組のみを置く
            {"meeting_id": "m001", "utt_a": "u003", "utt_b": "u001", "winner": "A_better"},
            {"meeting_id": "m001", "utt_a": "u002", "utt_b": "u006", "winner": "tie"},
        ],
    )

    report = run_eval(
        dataset,
        _OrderedEvaluator(),
        ScoringWeights(),
        meetings_dir=meetings_dir,
        model_name="stub-ordered",
    )

    assert len(report.per_meeting) == 1
    m = report.per_meeting[0]
    assert m.meeting_id == "m001"
    assert m.n_utterances == 24

    # OrderedEvaluator は u003 (score=3) > u001 (score=1) を返すはず
    assert m.pairwise is not None
    assert m.pairwise.n == 2
    # u002 (2) vs u006 (2) は同点 → tie 判定で一致
    # u003 (3) vs u001 (1) は A_better → 一致
    assert m.pairwise.accuracy == 1.0

    # Top5/Bottom5 Jaccard は 0.0 以上（厳密な値は idx % 4 の偶然に依存）
    assert m.top5_jaccard is not None
    assert 0.0 <= m.top5_jaccard <= 1.0
    assert m.bottom5_jaccard is not None
    assert 0.0 <= m.bottom5_jaccard <= 1.0

    payload = report.to_dict()
    assert "macro" in payload
    assert payload["model"] == "stub-ordered"
    assert payload["per_meeting"][0]["n_utterances"] == 24


def test_run_eval_with_no_annotations(tmp_path):
    """アノテ JSONL が空でも、meetings_dir があれば実行できる。"""
    dataset = tmp_path / "v_empty"
    dataset.mkdir(parents=True)
    meetings_dir = tmp_path / "meetings"
    meetings_dir.mkdir(parents=True)
    shutil.copy(SAMPLE_MEETING, meetings_dir / "m001.json")

    report = run_eval(
        dataset,
        _OrderedEvaluator(),
        ScoringWeights(),
        meetings_dir=meetings_dir,
    )
    # メトリクスは出ないが、会議は1本評価される
    assert len(report.per_meeting) == 1
    m = report.per_meeting[0]
    assert m.spearman is None
    assert m.pairwise is None
