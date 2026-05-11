"""evals.cli のテスト。"""

from __future__ import annotations

from types import SimpleNamespace

from app.context_builder.builder import EvaluationContext
from app.schemas.models import MeetingInput, Penalties, Scores, Utterance

import evals.cli as cli


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

    def _fake_evaluate_utterance(ctx, **kwargs):  # noqa: ANN001, ARG001, ANN202
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


def test_cmd_stability_uses_stability_temperature(monkeypatch, capsys) -> None:
    captured: dict = {}

    class _FakeAdapter:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
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

    assert cli._cmd_stability(args) == 0  # noqa: SLF001
    assert captured["temperature"] == cli.STABILITY_TEMPERATURE

    capsys.readouterr()
