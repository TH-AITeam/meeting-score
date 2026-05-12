"""発言評価器パッケージ (Issue #12)。

複数バックエンド（OpenAI / ローカル LLM）の Evaluator を factory 経由で切替できる。
旧 evaluate_utterance() API は llm_evaluator モジュールに残しており、
内部では create_evaluator() を経由する後方互換ラッパーになっている。
"""

from app.evaluators.base import EvaluationResult, Evaluator
from app.evaluators.factory import SUPPORTED_BACKENDS, Backend, create_evaluator
from app.evaluators.local_evaluator import LocalEvaluator
from app.evaluators.openai_evaluator import OpenAIEvaluator

__all__ = [
    "SUPPORTED_BACKENDS",
    "Backend",
    "EvaluationResult",
    "Evaluator",
    "LocalEvaluator",
    "OpenAIEvaluator",
    "create_evaluator",
]
