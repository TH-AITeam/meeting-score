"""LLM による発言評価モジュール（後方互換ラッパー、Issue #12 で再構成）。

.. deprecated:: Issue #17
    本モジュールは OpenAI Responses API への直接呼び出し旧 API
    (`evaluate_utterance()`) を維持する薄い互換層。新規コードは
    `app.evaluators.create_evaluator(config)` を経由してローカル推論
    バックエンド (vLLM 等) を使うこと。本モジュールはテストや蒸留・
    ベンチマーク等の限定用途でのみ残す。将来的に削除予定。

実装本体は `app/evaluators/openai_evaluator.py` (OpenAI 用) と
`app/evaluators/local_evaluator.py` (ローカル LLM 用) に分離されている。
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
    temperature: float | None = None,
) -> dict:
    """1発言を OpenAI で評価する（旧 API、dict 形式で返す）。

    新しいコードは `app.evaluators.create_evaluator(config)` を経由して
    Evaluator.evaluate(ctx) → EvaluationResult を直接受け取ること。
    """
    evaluator = OpenAIEvaluator(
        model=model,
        max_tokens=max_tokens,
        max_retries=max_retries,
        temperature=temperature,
    )
    result = evaluator.evaluate(ctx)
    return result.as_dict()


__all__ = ["_build_prompt", "evaluate_utterance"]
