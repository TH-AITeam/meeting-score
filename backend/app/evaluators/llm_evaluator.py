"""LLM による発言評価モジュール（後方互換ラッパー、Issue #12 で再構成）。

実装本体は `app/evaluators/openai_evaluator.py` (OpenAI 用) と
`app/evaluators/local_evaluator.py` (ローカル LLM 用) に分離された。
本モジュールは旧来の `evaluate_utterance()` 関数 API と
`_build_prompt()` シンボルを保持する薄い互換層。

新しいコードは `app.evaluators.create_evaluator(config)` を使うこと。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.evaluators.openai_evaluator import OpenAIEvaluator
from app.evaluators.prompt import build_prompt

if TYPE_CHECKING:
    from app.context_builder.builder import EvaluationContext


# 旧名（tests/test_prompt_build.py からの参照あり）の互換エイリアス
_build_prompt = build_prompt


def evaluate_utterance(
    ctx: EvaluationContext,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> dict:
    """1発言を OpenAI で評価する（旧 API、dict 形式で返す）。

    新しいコードは `app.evaluators.create_evaluator(config)` を経由して
    Evaluator.evaluate(ctx) → EvaluationResult を直接受け取ること。
    """
    evaluator = OpenAIEvaluator(
        model=model,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )
    result = evaluator.evaluate(ctx)
    return result.as_dict()


__all__ = ["_build_prompt", "evaluate_utterance"]
