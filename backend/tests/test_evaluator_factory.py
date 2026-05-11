"""factory と LocalEvaluator のテスト (Issue #12)。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.context_builder.builder import EvaluationContext
from app.evaluators import (
    EvaluationResult,
    Evaluator,
    LocalEvaluator,
    OpenAIEvaluator,
    create_evaluator,
)
from app.schemas.models import Utterance
from app.scoring.weights import AppConfig


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


# ---------------------------------------------------------------------------
# create_evaluator
# ---------------------------------------------------------------------------


class TestCreateEvaluator:
    def test_openai_backend(self) -> None:
        config = AppConfig(
            llm_backend="openai",
            llm_model="gpt-4o-mini",
            llm_timeout=12.5,
        )
        ev = create_evaluator(config)
        assert isinstance(ev, OpenAIEvaluator)
        assert isinstance(ev, Evaluator)
        assert ev._timeout == 12.5  # noqa: SLF001

    def test_local_backend_requires_endpoint(self) -> None:
        config = AppConfig(llm_backend="local", llm_endpoint=None)
        with pytest.raises(ValueError, match="endpoint"):
            create_evaluator(config)

    def test_local_backend_with_endpoint(self) -> None:
        config = AppConfig(
            llm_backend="local",
            llm_endpoint="http://localhost:8000/v1",
            llm_model="qwen2.5-7b-instruct",
        )
        ev = create_evaluator(config)
        assert isinstance(ev, LocalEvaluator)

    def test_unknown_backend_raises(self) -> None:
        config = AppConfig(llm_backend="anthropic")
        with pytest.raises(ValueError, match="未対応"):
            create_evaluator(config)

    def test_backend_case_insensitive(self) -> None:
        config = AppConfig(llm_backend="OpenAI")
        ev = create_evaluator(config)
        assert isinstance(ev, OpenAIEvaluator)


# ---------------------------------------------------------------------------
# OpenAIEvaluator: クライアント生成失敗時のフォールバック
# ---------------------------------------------------------------------------


class _FailingOpenAIEvaluator(OpenAIEvaluator):
    def _get_client(self):  # noqa: ANN201
        raise RuntimeError("missing api key")


class TestOpenAIEvaluator:
    def test_client_creation_failure_returns_failed(self) -> None:
        ev = _FailingOpenAIEvaluator()
        result = ev.evaluate(_make_ctx())
        assert result.evaluation_failed is True

    def test_get_client_passes_timeout(self) -> None:
        ev = OpenAIEvaluator(timeout=12.5)
        with patch("openai.OpenAI") as openai_cls:
            ev._get_client()  # noqa: SLF001
        openai_cls.assert_called_once_with(timeout=12.5)


# ---------------------------------------------------------------------------
# LocalEvaluator: モック clientで evaluate() の挙動を検証
# ---------------------------------------------------------------------------


class _FakeChatCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):  # noqa: ANN001
        self.last_kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self.content))
            ]
        )


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(content))


_VALID_JSON = (
    '{"speech_type": "意思決定促進", '
    '"scores": {"issue_clarification": 2, "decision_progress": 3, '
    '"risk_detection": 1, "actionability": 2, "groundedness": 1, '
    '"novelty": 0, "summarization": 0}, '
    '"penalties": {"duplication": 0, "verbosity": -1, '
    '"off_topic": 0, "unsupported_assertion": 0}, '
    '"reason": "判定理由"}'
)


class TestLocalEvaluator:
    def test_evaluate_success(self) -> None:
        client = _FakeClient(_VALID_JSON)
        ev = LocalEvaluator(
            model="qwen2.5-7b-instruct",
            endpoint="http://localhost:8000/v1",
            client=client,
        )
        result = ev.evaluate(_make_ctx())
        assert isinstance(result, EvaluationResult)
        assert result.evaluation_failed is False
        assert result.speech_type == "意思決定促進"
        assert result.scores.decision_progress == 3
        assert result.penalties.verbosity == -1

    def test_evaluate_passes_json_schema(self) -> None:
        client = _FakeClient(_VALID_JSON)
        ev = LocalEvaluator(
            model="qwen",
            endpoint="http://localhost:8000/v1",
            client=client,
        )
        ev.evaluate(_make_ctx())
        kwargs = client.chat.completions.last_kwargs
        assert kwargs is not None
        assert kwargs["model"] == "qwen"
        assert kwargs["response_format"]["type"] == "json_schema"
        assert kwargs["response_format"]["json_schema"]["strict"] is True

    def test_evaluate_parse_failure_returns_failed(self) -> None:
        client = _FakeClient("not a json")
        ev = LocalEvaluator(
            model="qwen",
            endpoint="http://localhost:8000/v1",
            max_retries=2,
            client=client,
        )
        result = ev.evaluate(_make_ctx())
        assert result.evaluation_failed is True
        # デフォルト値で埋まる
        assert result.speech_type in {"情報共有"}

    def test_endpoint_trailing_slash_normalized(self) -> None:
        ev = LocalEvaluator(
            model="qwen",
            endpoint="http://localhost:8000/v1/",
            client=_FakeClient(_VALID_JSON),
        )
        # 内部で rstrip("/") している
        assert ev._endpoint == "http://localhost:8000/v1"  # noqa: SLF001


# ---------------------------------------------------------------------------
# EvaluationResult.as_dict (後方互換)
# ---------------------------------------------------------------------------


class TestEvaluationResultDict:
    def test_as_dict_keys(self) -> None:
        r = EvaluationResult.failed()
        d = r.as_dict()
        assert set(d) == {
            "speech_type", "scores", "penalties", "reason", "evaluation_failed",
        }
        assert d["evaluation_failed"] is True

    def test_default_scores_zero(self) -> None:
        r = EvaluationResult.failed()
        assert r.scores.issue_clarification == 0
        assert r.scores.decision_progress == 0
