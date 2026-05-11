"""eval ハーネス用 Evaluator プロトコル (Issue #5)。

Issue #12 (Evaluator ABC) が main にマージされるまでの間、eval ハーネスは
自前の最小プロトコルで動作する。マージ後は `app.evaluators.base.Evaluator`
が同じ shape を満たすため、CLI 側のアダプタを差し替えるだけで移行できる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from app.schemas.models import Penalties, Scores

if TYPE_CHECKING:
    from app.context_builder.builder import EvaluationContext


@dataclass
class EvaluationResult:
    """1発言の評価結果（プロトコル準拠の値）。"""

    speech_type: str
    scores: Scores
    penalties: Penalties
    reason: str = ""
    evaluation_failed: bool = False

    @classmethod
    def failed(cls, reason: str = "評価を取得できませんでした。") -> EvaluationResult:
        return cls(
            speech_type="情報共有",
            scores=Scores(),
            penalties=Penalties(),
            reason=reason,
            evaluation_failed=True,
        )


class Evaluator(Protocol):
    """評価器プロトコル。

    実装は `evaluate(ctx: EvaluationContext) -> EvaluationResult` を提供する。
    """

    def evaluate(self, ctx: EvaluationContext) -> EvaluationResult: ...


__all__ = ["EvaluationResult", "Evaluator"]
