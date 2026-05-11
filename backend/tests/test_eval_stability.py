"""evals.stability のテスト (Issue #5)。"""

from __future__ import annotations

from app.context_builder.builder import build_contexts
from app.schemas.models import MeetingInput, Penalties, Scores, Utterance

from evals.protocol import EvaluationResult
from evals.stability import AXES, _stdev, evaluate_stability


def _meeting(n: int = 3) -> MeetingInput:
    return MeetingInput(
        meeting_id="m_stab",
        title="stab",
        goal="stab",
        utterances=[
            Utterance(
                utterance_id=f"u{i+1:03d}",
                speaker=f"s{i}",
                timestamp=f"00:0{i}:00",
                text=f"発言{i}",
            )
            for i in range(n)
        ],
    )


class _DeterministicEvaluator:
    """毎回同じスコアを返す。SD は 0 になるはず。"""

    def evaluate(self, ctx) -> EvaluationResult:  # noqa: ARG002
        return EvaluationResult(
            speech_type="提案",
            scores=Scores(
                issue_clarification=2,
                decision_progress=1,
                risk_detection=0,
                actionability=2,
                groundedness=2,
                novelty=1,
                summarization=1,
            ),
            penalties=Penalties(),
        )


class _VaryingEvaluator:
    """呼び出しごとに axis ごとに 0/1/2/3/3 と返す（既知の母標準偏差）。"""

    def __init__(self) -> None:
        self.calls = 0
        self._series = [0, 1, 2, 3, 3]

    def evaluate(self, ctx) -> EvaluationResult:  # noqa: ARG002
        v = self._series[self.calls % len(self._series)]
        self.calls += 1
        return EvaluationResult(
            speech_type="情報共有",
            scores=Scores(
                issue_clarification=v,
                decision_progress=v,
                risk_detection=v,
                actionability=v,
                groundedness=v,
                novelty=v,
                summarization=v,
            ),
            penalties=Penalties(),
        )


def test_stdev_helper():
    assert _stdev([]) == 0.0
    assert _stdev([5]) == 0.0
    # 母標準偏差: 値 1,2,3 → mean 2, var = 2/3
    assert abs(_stdev([1.0, 2.0, 3.0]) - (2 / 3) ** 0.5) < 1e-9


def test_stability_deterministic_evaluator_gives_zero_sd():
    contexts = build_contexts(_meeting(2), before_count=1, after_count=1)
    stab = evaluate_stability(_DeterministicEvaluator(), contexts, "m_stab", n_samples=4)
    assert len(stab.utterances) == 2
    for u in stab.utterances:
        for axis in AXES:
            assert u.sd_per_axis[axis] == 0.0
            assert u.range_per_axis[axis] == 0
    for v in stab.mean_sd_per_axis.values():
        assert v == 0.0


def test_stability_varying_evaluator_sd_per_axis():
    contexts = build_contexts(_meeting(1), before_count=0, after_count=0)
    stab = evaluate_stability(_VaryingEvaluator(), contexts, "m_stab", n_samples=5)
    assert len(stab.utterances) == 1
    u = stab.utterances[0]
    # 0,1,2,3,3 の母標準偏差を全軸で同じ値で確認
    expected = _stdev([0.0, 1.0, 2.0, 3.0, 3.0])
    for axis in AXES:
        assert abs(u.sd_per_axis[axis] - expected) < 1e-9
        assert u.range_per_axis[axis] == 3


def test_stability_meeting_aggregation():
    contexts = build_contexts(_meeting(3), before_count=1, after_count=1)
    stab = evaluate_stability(_DeterministicEvaluator(), contexts, "m_stab", n_samples=3)
    mean = stab.mean_sd_per_axis
    mx = stab.max_sd_per_axis
    assert set(mean.keys()) == set(AXES)
    assert set(mx.keys()) == set(AXES)
    for axis in AXES:
        assert mean[axis] == 0.0
        assert mx[axis] == 0.0
