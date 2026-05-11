"""Evaluator 抽象クラスと EvaluationResult dataclass (Issue #12)。

OpenAI / ローカル LLM 等、複数バックエンドで発言評価を切り替えるための共通契約。
既存の `evaluate_utterance` が返していた dict と相互変換できる `as_dict()` を持つ。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.schemas.models import Penalties, Scores, SpeechType

if TYPE_CHECKING:
    from app.context_builder.builder import EvaluationContext


@dataclass
class EvaluationResult:
    """1発言の評価結果。"""

    speech_type: str = SpeechType.INFO_SHARING.value
    scores: Scores = field(default_factory=Scores)
    penalties: Penalties = field(default_factory=Penalties)
    reason: str = ""
    evaluation_failed: bool = False

    def as_dict(self) -> dict:
        """既存 API 互換（dict 形式）。

        既存の `evaluate_utterance` を呼んでいた callers との後方互換のため。
        """
        return {
            "speech_type": self.speech_type,
            "scores": self.scores,
            "penalties": self.penalties,
            "reason": self.reason,
            "evaluation_failed": self.evaluation_failed,
        }

    @classmethod
    def failed(cls, reason: str = "評価を取得できませんでした。") -> EvaluationResult:
        """全リトライ失敗時のデフォルト値。"""
        return cls(
            speech_type=SpeechType.INFO_SHARING.value,
            scores=Scores(),
            penalties=Penalties(),
            reason=reason,
            evaluation_failed=True,
        )


class Evaluator(ABC):
    """発言評価器の抽象基底クラス。

    具象クラスは `evaluate(ctx)` を実装し、EvaluationResult を返す。
    通信エラーやレート制限は内部でリトライしつつ、最終的に失敗したら
    `EvaluationResult.failed()` を返す責務を持つ。
    """

    @abstractmethod
    def evaluate(self, ctx: EvaluationContext) -> EvaluationResult:
        """1発言を評価する。"""
        raise NotImplementedError


__all__ = [
    "EvaluationResult",
    "Evaluator",
]
