"""evals.cli のテスト。"""

from __future__ import annotations

from types import SimpleNamespace

import evals.cli as cli
from app.context_builder.builder import EvaluationContext
from app.schemas.models import MeetingInput, Utterance
from app.scoring.weights import (
    DEFAULT_OPENAI_LLM_MODEL,
    AppConfig,
    PenaltyWeights,
    ScoringWeights,
)


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


def _base_cfg() -> AppConfig:
    return AppConfig(
        weights=ScoringWeights(),
        penalty_weights=PenaltyWeights(),
        context_before=2,
        context_after=4,
        llm_backend="openai",
        llm_model="gpt-test",
        llm_max_tokens=10,
        llm_max_retries=1,
        llm_timeout=10.0,
    )


def test_build_config_applies_cli_overrides(monkeypatch) -> None:
    """--backend / --endpoint / --model / --api-key で config を上書きできる"""
    monkeypatch.setattr(cli, "load_config", lambda *_: _base_cfg())
    args = SimpleNamespace(
        config=None,
        backend="local",
        endpoint="http://127.0.0.1:8001/v1",
        model="qwen3.6-27b-bnb",
        api_key="dummy",
    )
    cfg = cli._build_config(args)
    assert cfg.llm_backend == "local"
    assert cfg.llm_endpoint == "http://127.0.0.1:8001/v1"
    assert cfg.llm_model == "qwen3.6-27b-bnb"
    assert cfg.llm_api_key == "dummy"


def test_build_config_endpoint_only_implies_local(monkeypatch) -> None:
    """--endpoint だけ指定したら backend は自動で local になる"""
    monkeypatch.setattr(cli, "load_config", lambda *_: _base_cfg())
    args = SimpleNamespace(
        config=None,
        backend=None,
        endpoint="http://127.0.0.1:8001/v1",
        model=None,
        api_key=None,
    )
    cfg = cli._build_config(args)
    assert cfg.llm_backend == "local"
    assert cfg.llm_endpoint == "http://127.0.0.1:8001/v1"


def test_build_config_openai_backend_replaces_local_model(monkeypatch) -> None:
    """--backend openai だけでも OpenAI 用の既定モデルを使う"""
    base_cfg = _base_cfg()
    base_cfg.llm_backend = "local"
    base_cfg.llm_model = "qwen3.6-35b-nvfp4"
    monkeypatch.setattr(cli, "load_config", lambda *_: base_cfg)
    args = SimpleNamespace(
        config=None,
        backend="openai",
        endpoint=None,
        model=None,
        api_key=None,
    )
    cfg = cli._build_config(args)
    assert cfg.llm_backend == "openai"
    assert cfg.llm_model == DEFAULT_OPENAI_LLM_MODEL


def test_cmd_run_passes_penalty_weights_to_runner(monkeypatch, capsys) -> None:
    """_cmd_run が config の penalty_weights を runner に渡すこと"""
    captured: dict = {}
    cfg = _base_cfg()
    cfg.penalty_weights = PenaltyWeights(duplication=2.0)
    cfg.llm_backend = "local"
    cfg.llm_endpoint = "http://stub/v1"

    class _FakeEvaluator:
        def evaluate(self, ctx):
            return SimpleNamespace()

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

    monkeypatch.setattr(cli, "load_config", lambda *_: cfg)
    monkeypatch.setattr(cli, "create_evaluator", lambda _cfg: _FakeEvaluator())
    monkeypatch.setattr(cli, "run_eval", _fake_run_eval)

    args = SimpleNamespace(
        config=None,
        backend=None,
        endpoint=None,
        model=None,
        api_key=None,
        dataset="dataset",
        meetings_dir=None,
        out=None,
    )

    assert cli._cmd_run(args) == 0
    assert captured["penalty_weights"] is cfg.penalty_weights
    assert captured["runner_kwargs"]["context_before"] == 2
    assert captured["runner_kwargs"]["context_after"] == 4

    capsys.readouterr()


def test_cmd_stability_uses_create_evaluator(monkeypatch, capsys) -> None:
    """_cmd_stability が create_evaluator で Evaluator を取得し stability を回す"""
    cfg = _base_cfg()
    cfg.llm_backend = "local"
    cfg.llm_endpoint = "http://stub/v1"

    class _FakeEvaluator:
        def evaluate(self, ctx):
            return SimpleNamespace()

    monkeypatch.setattr(cli, "load_config", lambda *_: cfg)
    monkeypatch.setattr(cli, "create_evaluator", lambda _cfg: _FakeEvaluator())
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
        backend=None,
        endpoint=None,
        model=None,
        api_key=None,
        meeting="meeting.json",
        n=5,
        out=None,
    )

    assert cli._cmd_stability(args) == 0
    capsys.readouterr()
