"""安定性評価: 同一発言を N 回採点して分散を測る (Issue #5)。

LLM は確率的な出力をするので、同じ発言を複数回採点したときの
ばらつきを軸ごと（7軸）に計測する。

- 個別発言レベル: 7軸 × N 回の値から SD と max-min
- 会議レベル: 個別発言レベルの集計（平均 SD と最大 SD）
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.context_builder.builder import EvaluationContext
    from evals.protocol import Evaluator

# Scores 上の 7軸
AXES: tuple[str, ...] = (
    "issue_clarification",
    "decision_progress",
    "risk_detection",
    "actionability",
    "groundedness",
    "novelty",
    "summarization",
)

DEFAULT_N_SAMPLES = 5


@dataclass
class UtteranceStability:
    utterance_id: str
    samples: list[dict[str, int]]  # 各 N 回の {axis: score}
    sd_per_axis: dict[str, float]  # 軸ごとの標準偏差
    range_per_axis: dict[str, int]  # 軸ごとの max - min


@dataclass
class MeetingStability:
    meeting_id: str
    utterances: list[UtteranceStability] = field(default_factory=list)

    @property
    def mean_sd_per_axis(self) -> dict[str, float]:
        """会議内の全発言について、軸ごとの平均 SD。"""
        if not self.utterances:
            return dict.fromkeys(AXES, 0.0)
        result: dict[str, float] = {}
        for axis in AXES:
            vals = [u.sd_per_axis[axis] for u in self.utterances]
            result[axis] = sum(vals) / len(vals)
        return result

    @property
    def max_sd_per_axis(self) -> dict[str, float]:
        """会議内の全発言について、軸ごとの最大 SD。"""
        if not self.utterances:
            return dict.fromkeys(AXES, 0.0)
        return {axis: max(u.sd_per_axis[axis] for u in self.utterances) for axis in AXES}


def _stdev(values: list[float]) -> float:
    """母集団標準偏差。N=1 のときは 0。"""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5


def _scores_to_dict(scores) -> dict[str, int]:
    """Scores Pydantic モデルを軸ごとの dict に変換する。"""
    return {axis: int(getattr(scores, axis)) for axis in AXES}


def _summarize_utterance(utterance_id: str, samples: list[dict[str, int]]) -> UtteranceStability:
    sd: dict[str, float] = {}
    rng: dict[str, int] = {}
    for axis in AXES:
        vals = [s[axis] for s in samples]
        sd[axis] = _stdev([float(v) for v in vals])
        rng[axis] = max(vals) - min(vals) if vals else 0
    return UtteranceStability(
        utterance_id=utterance_id,
        samples=samples,
        sd_per_axis=sd,
        range_per_axis=rng,
    )


def evaluate_stability(
    evaluator: Evaluator,
    contexts: Iterable[EvaluationContext],
    meeting_id: str,
    n_samples: int = DEFAULT_N_SAMPLES,
) -> MeetingStability:
    """同一発言を `n_samples` 回採点し、軸ごとの SD と range を算出する。

    LLM 側の temperature 制御は Evaluator 実装に委ねる（決定的でも実行可能で
    その場合は SD=0 になる）。本関数は同じ ctx を n_samples 回呼ぶだけ。
    """
    meeting = MeetingStability(meeting_id=meeting_id)
    for ctx in contexts:
        samples: list[dict[str, int]] = []
        for _ in range(n_samples):
            result = evaluator.evaluate(ctx)
            samples.append(_scores_to_dict(result.scores))
        meeting.utterances.append(_summarize_utterance(ctx.target_utterance.utterance_id, samples))
    return meeting


__all__ = [
    "AXES",
    "DEFAULT_N_SAMPLES",
    "MeetingStability",
    "UtteranceStability",
    "evaluate_stability",
]
