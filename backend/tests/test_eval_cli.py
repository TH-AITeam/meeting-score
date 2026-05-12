"""evals.cli のテスト。"""

from __future__ import annotations

from types import SimpleNamespace

import evals.cli as cli
from app.context_builder.builder import EvaluationContext
from app.schemas.models import MeetingInput, Penalties, Scores, Utterance
from app.scoring.weights import PenaltyWeights, ScoringWeights


def _make_ctx() -> EvaluationContext:
    target = Utterance(
        utterance_id="u001",
        speaker="田中",
        timestamp="00:01:00",
        text="テスト発言です。",
    )
    return EvaluationContext(
        meeting_goal="テスト目的",
        agenda=["議題1"],
        decision_points=[],
        current_topic="議題1",
        before_utterances=[],
        target_utterance=target,
        after_utterances=[],
    )


def test_llm_evaluator_adapter_passes_temperature(monkeypatch) -> None:
    captured: dict = {}

    def _fake_evaluate_utterance(ctx, **kwargs):
        captured.update(kwargs)
        return {
            "speech_type": "情報共有",
            "scores": Scores(),
            "penalties": Penalties(),
            "reason": "ok",
            "evaluation_failed": False,
        }

    monkeypatch.setattr(cli, "evaluate_utterance", _fake_evaluate_utterance)

    adapter = cli.LLMEvaluatorAdapter(temperature=0.7)
    adapter.evaluate(_make_ctx())

    assert captured["temperature"] == 0.7


def test_cmd_run_passes_penalty_weights_to_runner(monkeypatch, capsys) -> None:
    captured: dict = {}
    penalty_weights = PenaltyWeights(duplication=2.0)

    class _FakeAdapter:
        def __init__(self, **kwargs) -> None:
            captured["adapter_kwargs"] = kwargs

    class _FakeReport:
        def to_dict(self) -> dict:
            return {
                "macro": {
                    "spearman": 0.0,
                    "kendall_tau": 0.0,
                    "top5_jaccard": 0.0,
                    "bottom5_jaccard": 0.0,
                    "pairwise_accuracy": 0.0,
                },
                "per_meeting": [],
            }

    def _fake_run_eval(dataset, evaluator, weights, passed_penalty_weights, **kwargs):
        captured["dataset"] = dataset
        captured["weights"] = weights
        captured["penalty_weights"] = passed_penalty_weights
        captured["runner_kwargs"] = kwargs
        return _FakeReport()

    monkeypatch.setattr(cli, "LLMEvaluatorAdapter", _FakeAdapter)
    monkeypatch.setattr(cli, "run_eval", _fake_run_eval)
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda: SimpleNamespace(
            weights=ScoringWeights(),
            penalty_weights=penalty_weights,
            llm_model="gpt-test",
            llm_max_tokens=10,
            llm_max_retries=1,
            context_before=2,
            context_after=4,
        ),
    )

    args = SimpleNamespace(
        config=None,
        model=None,
        dataset="dataset",
        meetings_dir=None,
        out=None,
    )

    assert cli._cmd_run(args) == 0
    assert captured["penalty_weights"] is penalty_weights
    assert captured["runner_kwargs"]["context_before"] == 2
    assert captured["runner_kwargs"]["context_after"] == 4

    capsys.readouterr()


def test_cmd_stability_uses_stability_temperature(monkeypatch, capsys) -> None:
    captured: dict = {}

    class _FakeAdapter:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(cli, "LLMEvaluatorAdapter", _FakeAdapter)
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda: SimpleNamespace(
            llm_model="gpt-test",
            llm_max_tokens=10,
            llm_max_retries=1,
            context_before=0,
            context_after=0,
        ),
    )
    monkeypatch.setattr(
        cli,
        "load_meeting_from_file",
        lambda path: MeetingInput(
            meeting_id="m001",
            title="test",
            goal="test",
            utterances=[],
        ),
    )
    monkeypatch.setattr(cli, "build_contexts", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        cli,
        "evaluate_stability",
        lambda *args, **kwargs: SimpleNamespace(
            meeting_id="m001",
            mean_sd_per_axis={},
            max_sd_per_axis={},
            utterances=[],
        ),
    )

    args = SimpleNamespace(
        config=None,
        model=None,
        meeting="meeting.json",
        n=5,
        out=None,
    )

    assert cli._cmd_stability(args) == 0
    assert captured["temperature"] == cli.STABILITY_TEMPERATURE

    capsys.readouterr()
