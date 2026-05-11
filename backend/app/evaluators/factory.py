"""バックエンド切替 factory (Issue #12)。

config.llm.backend の値で OpenAI / Local を切り替えて Evaluator を返す。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from app.evaluators.base import Evaluator
from app.evaluators.local_evaluator import LocalEvaluator
from app.evaluators.openai_evaluator import OpenAIEvaluator

if TYPE_CHECKING:
    from app.scoring.weights import AppConfig

Backend = Literal["openai", "local"]
SUPPORTED_BACKENDS: tuple[Backend, ...] = ("openai", "local")


def create_evaluator(config: AppConfig) -> Evaluator:
    """AppConfig から Evaluator を生成する。

    config.llm_backend == "openai" → OpenAIEvaluator
    config.llm_backend == "local"  → LocalEvaluator
    """
    backend = (config.llm_backend or "openai").lower()
    if backend == "openai":
        return OpenAIEvaluator(
            model=config.llm_model,
            max_tokens=config.llm_max_tokens,
            max_retries=config.llm_max_retries,
        )
    if backend == "local":
        if not config.llm_endpoint:
            msg = (
                "llm.backend=local の場合は llm.endpoint を config.yaml に設定してください "
                "(例: http://localhost:8000/v1)"
            )
            raise ValueError(msg)
        return LocalEvaluator(
            model=config.llm_model,
            endpoint=config.llm_endpoint,
            api_key=config.llm_api_key,
            max_tokens=config.llm_max_tokens,
            max_retries=config.llm_max_retries,
            timeout=config.llm_timeout,
        )

    msg = (
        f"未対応の llm.backend: {backend!r}。"
        f"対応バックエンド: {', '.join(SUPPORTED_BACKENDS)}"
    )
    raise ValueError(msg)


__all__ = ["Backend", "SUPPORTED_BACKENDS", "create_evaluator"]
