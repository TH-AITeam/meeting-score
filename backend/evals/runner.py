"""eval ハーネス本体 (Issue #5)。

アノテーション済みデータセットを読み、評価器を回し、metrics dict を返す。

入力 (dataset_path / v1):
    data/annotations/gold/v1/
        tags.jsonl
        pairs.jsonl
        top_bottom.jsonl
        meetings/             # 各 meeting の元データ（任意、既定で data/sample_meetings から探す）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from app.context_builder.builder import build_contexts
from app.ingest.loader import load_meeting_from_file
from app.schemas.models import EvaluatedUtterance
from app.scoring.calculator import calculate_total_score
from app.scoring.rule_corrections import apply_rule_corrections
from evals.metrics import (
    PairwiseAccuracyReport,
    kendall_tau,
    pairwise_accuracy,
    spearman,
    top_k_jaccard,
)
from evals.schema import (
    PairwiseAnnotation,
    TopBottomAnnotation,
    load_pairwise_annotations,
    load_top_bottom_annotations,
)

if TYPE_CHECKING:
    from app.scoring.weights import ScoringWeights
    from evals.protocol import Evaluator


@dataclass
class MeetingMetrics:
    """会議1本の評価結果。"""

    meeting_id: str
    spearman: float | None = None
    kendall_tau: float | None = None
    top5_jaccard: float | None = None
    bottom5_jaccard: float | None = None
    pairwise: PairwiseAccuracyReport | None = None
    n_utterances: int = 0


@dataclass
class EvalReport:
    """eval 全体の結果。"""

    dataset: str
    model: str
    timestamp: str
    per_meeting: list[MeetingMetrics] = field(default_factory=list)

    @property
    def macro_spearman(self) -> float:
        vals = [m.spearman for m in self.per_meeting if m.spearman is not None]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def macro_kendall_tau(self) -> float:
        vals = [m.kendall_tau for m in self.per_meeting if m.kendall_tau is not None]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def macro_top5_jaccard(self) -> float:
        vals = [m.top5_jaccard for m in self.per_meeting if m.top5_jaccard is not None]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def macro_bottom5_jaccard(self) -> float:
        vals = [m.bottom5_jaccard for m in self.per_meeting if m.bottom5_jaccard is not None]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def micro_pairwise_accuracy(self) -> float:
        """全会議のペアを束ねた micro accuracy。"""
        total = sum(m.pairwise.n for m in self.per_meeting if m.pairwise)
        correct = 0
        for m in self.per_meeting:
            if not m.pairwise:
                continue
            correct += int(m.pairwise.accuracy * m.pairwise.n)
        return correct / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "model": self.model,
            "timestamp": self.timestamp,
            "macro": {
                "spearman": self.macro_spearman,
                "kendall_tau": self.macro_kendall_tau,
                "top5_jaccard": self.macro_top5_jaccard,
                "bottom5_jaccard": self.macro_bottom5_jaccard,
                "pairwise_accuracy": self.micro_pairwise_accuracy,
            },
            "per_meeting": [
                {
                    "meeting_id": m.meeting_id,
                    "spearman": m.spearman,
                    "kendall_tau": m.kendall_tau,
                    "top5_jaccard": m.top5_jaccard,
                    "bottom5_jaccard": m.bottom5_jaccard,
                    "pairwise": {
                        "accuracy": m.pairwise.accuracy if m.pairwise else None,
                        "n": m.pairwise.n if m.pairwise else 0,
                        "n_skipped": m.pairwise.n_skipped if m.pairwise else 0,
                        "by_class": m.pairwise.by_class if m.pairwise else None,
                    },
                    "n_utterances": m.n_utterances,
                }
                for m in self.per_meeting
            ],
        }


# --------------------------------------------------------------------------
# 内部ヘルパ
# --------------------------------------------------------------------------


def _evaluate_meeting(
    meeting_file: Path,
    evaluator: Evaluator,
    weights: ScoringWeights,
    context_before: int = 3,
    context_after: int = 3,
) -> tuple[str, list[EvaluatedUtterance]]:
    """1会議を Evaluator で評価し、(meeting_id, 評価済み発言リスト) を返す。"""
    meeting = load_meeting_from_file(meeting_file)
    contexts = build_contexts(meeting, before_count=context_before, after_count=context_after)
    evaluated: list[EvaluatedUtterance] = []
    for ctx in contexts:
        result = evaluator.evaluate(ctx)
        total = calculate_total_score(result.scores, result.penalties, weights)
        target = ctx.target_utterance
        evaluated.append(
            EvaluatedUtterance(
                utterance_id=target.utterance_id,
                speaker=target.speaker,
                timestamp=target.timestamp,
                text=target.text,
                speech_type=result.speech_type,
                scores=result.scores,
                penalties=result.penalties,
                total_score=total,
                reason=result.reason,
            )
        )
    return meeting.meeting_id, apply_rule_corrections(evaluated, weights)


def _system_scores_dict(evaluated: list[EvaluatedUtterance]) -> dict[str, float]:
    return {u.utterance_id: u.total_score for u in evaluated}


def _system_top_bottom(
    evaluated: list[EvaluatedUtterance], k: int = 5
) -> tuple[list[str], list[str]]:
    sorted_desc = sorted(evaluated, key=lambda u: u.total_score, reverse=True)
    top = [u.utterance_id for u in sorted_desc[:k]]
    bottom = [u.utterance_id for u in sorted_desc[-k:]] if len(sorted_desc) >= k else []
    return top, bottom


def _human_ranks_from_pairs(
    pairs: list[PairwiseAnnotation], utt_ids: list[str]
) -> dict[str, float] | None:
    """ペアワイズ多数決から人手スコア（疑似ランク）を構築する。

    A_better で +1、B_better で -1、tie で 0 を足し合わせ、最後に各発言の
    総獲得点を疑似ランクとして返す。発言数 < 2 や全 0 のときは None。
    """
    score = dict.fromkeys(utt_ids, 0.0)
    counted = False
    for p in pairs:
        if p.utt_a not in score or p.utt_b not in score:
            continue
        counted = True
        if p.winner == "A_better":
            score[p.utt_a] += 1
            score[p.utt_b] -= 1
        elif p.winner == "B_better":
            score[p.utt_a] -= 1
            score[p.utt_b] += 1
    if not counted:
        return None
    return score


# --------------------------------------------------------------------------
# 公開 API
# --------------------------------------------------------------------------


def run_eval(
    dataset_path: Path,
    evaluator: Evaluator,
    weights: ScoringWeights,
    *,
    meetings_dir: Path | None = None,
    model_name: str = "unknown",
    context_before: int = 3,
    context_after: int = 3,
) -> EvalReport:
    """データセット 1 つを評価する。

    Parameters
    ----------
    dataset_path : Path
        例: data/annotations/gold/v1
    meetings_dir : Path | None
        会議元データの探索ディレクトリ。既定は dataset_path / "meetings"。
        無ければ data/sample_meetings を探す。
    """
    dataset_path = Path(dataset_path)
    pairs = load_pairwise_annotations(dataset_path / "pairs.jsonl")
    top_bottom = load_top_bottom_annotations(dataset_path / "top_bottom.jsonl")
    tb_by_meeting: dict[str, TopBottomAnnotation] = {t.meeting_id: t for t in top_bottom}

    if meetings_dir is None:
        meetings_dir = dataset_path / "meetings"
    fallback_meetings_dir = (
        Path(__file__).resolve().parent.parent.parent / "data" / "sample_meetings"
    )

    # 評価対象 meeting_id を pairs / top_bottom から集める
    meeting_ids: set[str] = set()
    meeting_ids.update(p.meeting_id for p in pairs)
    meeting_ids.update(tb_by_meeting.keys())
    if not meeting_ids and meetings_dir.exists():
        # アノテが空でもデータセット dir 内の meeting ファイルを探す
        for f in meetings_dir.glob("*.json"):
            meeting_ids.add(f.stem)

    report = EvalReport(
        dataset=str(dataset_path),
        model=model_name,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )

    for meeting_id in sorted(meeting_ids):
        meeting_file = _find_meeting_file(meeting_id, meetings_dir, fallback_meetings_dir)
        if meeting_file is None:
            continue
        actual_id, evaluated = _evaluate_meeting(
            meeting_file, evaluator, weights, context_before, context_after
        )
        report.per_meeting.append(
            _compute_meeting_metrics(actual_id, evaluated, pairs, tb_by_meeting)
        )

    return report


def _find_meeting_file(meeting_id: str, primary: Path, fallback: Path) -> Path | None:
    for d in (primary, fallback):
        if not d.exists():
            continue
        for pat in (f"{meeting_id}.json", f"*{meeting_id}*.json"):
            for f in d.glob(pat):
                return f
    return None


def _compute_meeting_metrics(
    meeting_id: str,
    evaluated: list[EvaluatedUtterance],
    pairs: list[PairwiseAnnotation],
    tb_by_meeting: dict[str, TopBottomAnnotation],
) -> MeetingMetrics:
    utt_ids = [u.utterance_id for u in evaluated]
    system_scores = _system_scores_dict(evaluated)

    metrics = MeetingMetrics(meeting_id=meeting_id, n_utterances=len(evaluated))

    # ペアワイズに基づく Spearman / Kendall（疑似人手スコア）
    meeting_pairs = [p for p in pairs if p.meeting_id == meeting_id]
    human_score_map = _human_ranks_from_pairs(meeting_pairs, utt_ids)
    if human_score_map is not None and len(utt_ids) >= 2:
        human_vals = [human_score_map[uid] for uid in utt_ids]
        system_vals = [system_scores[uid] for uid in utt_ids]
        metrics.spearman = spearman(human_vals, system_vals)
        metrics.kendall_tau = kendall_tau(human_vals, system_vals)

    # Top-K / Bottom-K Jaccard
    tb = tb_by_meeting.get(meeting_id)
    if tb is not None:
        system_top, system_bottom = _system_top_bottom(evaluated, k=5)
        metrics.top5_jaccard = top_k_jaccard(tb.top5, system_top, k=5)
        metrics.bottom5_jaccard = top_k_jaccard(tb.bottom5, system_bottom, k=5)

    # ペアワイズ一致率
    if meeting_pairs:
        metrics.pairwise = pairwise_accuracy(meeting_pairs, system_scores)

    return metrics


__all__ = [
    "EvalReport",
    "MeetingMetrics",
    "run_eval",
]
