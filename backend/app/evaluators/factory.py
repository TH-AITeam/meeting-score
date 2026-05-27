"""バックエンド切替 factory (Issue #12 + Issue #17)。

config.llm.backend の値で Local / OpenAI を切り替えて Evaluator を返す。
既定は "local" (vLLM 等の OpenAI 互換サーバ)。OpenAI Responses API は
蒸留・ベンチマーク用途の optional 経路。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from app.evaluators.base import Evaluator
from app.evaluators.local_evaluator import LocalEvaluator
from app.evaluators.openai_evaluator import OpenAIEvaluator
from app.scoring.weights import resolve_llm_model_for_backend

if TYPE_CHECKING:
    from app.scoring.weights import AppConfig

Backend = Literal["openai", "local"]
SUPPORTED_BACKENDS: tuple[Backend, ...] = ("openai", "local")


def create_evaluator(
    config: AppConfig,
    *,
    model_override: str | None = None,
    fallback_model: str | None = None,
) -> Evaluator:
    """AppConfig から Evaluator を生成する。

    config.llm_backend == "openai" → OpenAIEvaluator
    config.llm_backend == "local"  → LocalEvaluator

    model_override / fallback_model (Issue #83):
        組織別 LoRA アダプタへのルーティング用。local バックエンドのとき
        model_override をリクエストのモデル名に使い、ロード失敗時は
        fallback_model（通常はベースモデル）に切り替える。アダプタは
        local 推論専用のため openai バックエンドでは無視する。
    """
    backend = (config.llm_backend or "local").lower()
    model = resolve_llm_model_for_backend(backend, config.llm_model)
    if backend == "openai":
        return OpenAIEvaluator(
            model=model,
            max_tokens=config.llm_max_tokens,
            max_retries=config.llm_max_retries,
            timeout=config.llm_timeout,
        )
    if backend == "local":
        if not config.llm_endpoint:
            msg = (
                "llm.backend=local の場合は llm.endpoint を config.yaml に設定してください "
                "(例: http://localhost:8001/v1)"
            )
            raise ValueError(msg)
        return LocalEvaluator(
            model=model_override or model,
            endpoint=config.llm_endpoint,
            api_key=config.llm_api_key,
            max_tokens=config.llm_max_tokens,
            max_retries=config.llm_max_retries,
            timeout=config.llm_timeout,
            fallback_model=fallback_model,
        )

    msg = f"未対応の llm.backend: {backend!r}。対応バックエンド: {', '.join(SUPPORTED_BACKENDS)}"
    raise ValueError(msg)


__all__ = ["SUPPORTED_BACKENDS", "Backend", "create_evaluator"]
