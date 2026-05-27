"""OpenAI 互換エンドポイント経由でローカル LLM を叩く Evaluator (Issue #12)。

vLLM / sglang / TGI などの **OpenAI 互換 chat completions API** に対し、
公式 `openai` SDK の `base_url` を切り替えて推論する。

JSON 強制 (guided decoding) は `response_format={"type": "json_schema", ...}` で
vLLM の `--guided-decoding-backend xgrammar` が解釈する想定。
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

# OpenAI 互換サーバが API キーを要求するときの既定値（vLLM などは任意の文字列で良い）
DEFAULT_API_KEY_PLACEHOLDER = "EMPTY"


def _extract_chat_message_content(response: Any) -> str:
    """Chat Completions 応答からアシスタントの content 文字列を取り出す。"""
    choices = getattr(response, "choices", None)
    if not choices:
        # dict ベースのフォールバック（SDK 仕様変更耐性）
        choices = response["choices"] if isinstance(response, dict) else None
    if not choices:
        msg = "OpenAI 互換応答に choices がありません"
        raise ValueError(msg)
    first = choices[0]
    message = getattr(first, "message", None) or first.get("message")
    content = getattr(message, "content", None) or message.get("content")
    if not content:
        msg = "OpenAI 互換応答の content が空です"
        raise ValueError(msg)
    return str(content)


class LocalEvaluator(Evaluator):
    """OpenAI 互換 API でローカル LLM を叩く Evaluator。

    Parameters
    ----------
    model:
        推論サーバが配信しているモデル名（例: "Qwen/Qwen2.5-7B-Instruct"）。
        Issue #17 で選定したモデルを config 経由で渡す。
    endpoint:
        OpenAI 互換エンドポイント (例: "http://localhost:8001/v1")。
    api_key:
        サーバ側で必要な場合のみ指定。vLLM は既定で任意文字列で良い。
    max_tokens / max_retries / timeout:
        既存挙動と同じ意味。
    client:
        テスト時に openai.OpenAI 互換オブジェクトを差し込むための窓口。
    """

    def __init__(
        self,
        model: str,
        endpoint: str,
        api_key: str | None = None,
        max_tokens: int = 1024,
        max_retries: int = 3,
        timeout: float = 30.0,
        client: Any | None = None,
        fallback_model: str | None = None,
    ) -> None:
        self._model = model
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key or DEFAULT_API_KEY_PLACEHOLDER
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._timeout = timeout
        self._injected_client = client
        # 組織別 LoRA アダプタ (model) のロード失敗時に切り替えるベースモデル (Issue #83)。
        # 障害時も必ずベースモデルで応答を返し、評価を止めない。
        self._fallback_model = fallback_model

    def _get_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        from openai import OpenAI

        return OpenAI(
            base_url=self._endpoint,
            api_key=self._api_key,
            timeout=self._timeout,
        )

    def evaluate(self, ctx: EvaluationContext) -> EvaluationResult:
        try:
            from openai import APIError  # noqa: F401  (存在確認)
        except ImportError as e:  # pragma: no cover - sdk 不在のテスト環境
            logger.error("OpenAI SDK の読み込みに失敗しました: %s", e)
            return EvaluationResult.failed()

        client = self._get_client()
        prompt = build_prompt(ctx)

        # primary（組織別アダプタ等）→ 失敗したら fallback（ベースモデル）の順に試す。
        models = [self._model]
        if self._fallback_model and self._fallback_model != self._model:
            models.append(self._fallback_model)

        for i, model in enumerate(models):
            result = self._attempt_model(client, prompt, model)
            if result is not None:
                return result
            if i + 1 < len(models):
                logger.warning(
                    "モデル %s で評価に失敗。フォールバック %s で再試行します (Issue #83)。",
                    model,
                    models[i + 1],
                )

        logger.error("ローカル LLM 全リトライ失敗。デフォルト値を返します。")
        return EvaluationResult.failed()

    def _attempt_model(self, client: Any, prompt: str, model: str) -> EvaluationResult | None:
        """1 モデルでリトライ込み評価。成功で EvaluationResult、全失敗で None。"""
        from openai import APIError

        for attempt in range(self._max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self._max_tokens,
                    temperature=0.0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": RESPONSE_SCHEMA_NAME,
                            "schema": RESPONSE_SCHEMA,
                            "strict": True,
                        },
                    },
                )
                text = _extract_chat_message_content(response)
                parsed = parse_response(text)
                return normalize_result(parsed)
            except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                logger.warning(
                    "ローカル LLM 応答パース失敗 [%s] (attempt %d/%d): %s",
                    model,
                    attempt + 1,
                    self._max_retries,
                    e,
                )
                continue
            except APIError as e:
                logger.error(
                    "ローカル LLM API エラー [%s] (attempt %d/%d): %s",
                    model,
                    attempt + 1,
                    self._max_retries,
                    e,
                )
                continue
            except Exception as e:
                logger.error(
                    "予期しないエラー [%s] (attempt %d/%d): %s: %s",
                    model,
                    attempt + 1,
                    self._max_retries,
                    type(e).__name__,
                    e,
                )
                continue
        return None


__all__ = ["DEFAULT_API_KEY_PLACEHOLDER", "LocalEvaluator"]
