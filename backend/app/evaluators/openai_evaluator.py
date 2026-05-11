"""OpenAI Responses API を使った Evaluator 実装 (Issue #12)。

旧 `app/evaluators/llm_evaluator.py: evaluate_utterance` の実装を Evaluator
プロトコルに準拠させたもの。
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from app.evaluators.base import EvaluationResult, Evaluator
from app.evaluators.prompt import (
    RESPONSE_SCHEMA,
    RESPONSE_SCHEMA_NAME,
    build_prompt,
    normalize_result,
    parse_response,
)

if TYPE_CHECKING:
    from app.context_builder.builder import EvaluationContext

logger = logging.getLogger(__name__)


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _extract_response_text(response: Any) -> str:
    """Responses API 応答からテキストを取り出す。"""
    output_text = getattr(response, "output_text", "")
    if output_text:
        return output_text
    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []):
            text = getattr(content, "text", None)
            if text:
                return text
    msg = "OpenAI 応答からテキストを取得できませんでした。"
    raise ValueError(msg)


class OpenAIEvaluator(Evaluator):
    """OpenAI Responses API で発言を評価する Evaluator。

    最大 `max_retries` 回までリトライし、全て失敗した場合は
    `EvaluationResult.failed()` を返す。
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        max_retries: int = 3,
        timeout: float = 30.0,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._timeout = timeout
        self._injected_client = client  # テスト用に注入可

    def _get_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        _load_dotenv_if_available()
        from openai import OpenAI

        return OpenAI(timeout=self._timeout)

    def evaluate(self, ctx: EvaluationContext) -> EvaluationResult:
        try:
            from openai import APIError
        except ImportError as e:  # pragma: no cover - sdk 不在のテスト環境
            logger.error("OpenAI SDK の読み込みに失敗しました: %s", e)
            return EvaluationResult.failed()

        try:
            client = self._get_client()
        except Exception as e:
            logger.error("OpenAI クライアント生成に失敗しました: %s: %s", type(e).__name__, e)
            return EvaluationResult.failed()

        prompt = build_prompt(ctx)

        for attempt in range(self._max_retries):
            try:
                response = client.responses.create(
                    model=self._model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                            ],
                        }
                    ],
                    max_output_tokens=self._max_tokens,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": RESPONSE_SCHEMA_NAME,
                            "strict": True,
                            "schema": RESPONSE_SCHEMA,
                        }
                    },
                )
                text = _extract_response_text(response)
                parsed = parse_response(text)
                return normalize_result(parsed)
            except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                logger.warning(
                    "LLM応答パース失敗 (attempt %d/%d): %s",
                    attempt + 1, self._max_retries, e,
                )
                continue
            except APIError as e:
                logger.error(
                    "OpenAI API エラー (attempt %d/%d): %s",
                    attempt + 1, self._max_retries, e,
                )
                continue
            except Exception as e:
                logger.error(
                    "予期しないエラー (attempt %d/%d): %s: %s",
                    attempt + 1, self._max_retries, type(e).__name__, e,
                )
                continue

        logger.error("全リトライ失敗。デフォルト値を返します。")
        return EvaluationResult.failed()


__all__ = ["OpenAIEvaluator"]
