"""ペアワイズフィードバックから軸重みを推定する。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

from app.scoring.weights import ScoringWeights

AXES = tuple(asdict(ScoringWeights()).keys())


@dataclass(frozen=True)
class PairwiseTrainingExample:
    """1 件の pairwise 比較に対応する学習例。"""

    scores_a: dict[str, float]
    scores_b: dict[str, float]
    winner: str


@dataclass(frozen=True)
class WeightRegressionResult:
    weights: ScoringWeights
    pairwise_acc: float
    n_pairs: int


def _features_and_labels(
    examples: Iterable[PairwiseTrainingExample],
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[list[float]] = []
    ys: list[float] = []
    for ex in examples:
        if ex.winner == "tie":
            continue
        xs.append(
            [float(ex.scores_a.get(axis, 0.0)) - float(ex.scores_b.get(axis, 0.0)) for axis in AXES]
        )
        ys.append(1.0 if ex.winner == "A" else 0.0)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def pairwise_accuracy(
    examples: Iterable[PairwiseTrainingExample],
    weights: ScoringWeights,
    tie_threshold: float = 1e-9,
) -> float:
    """与えた重みで pairwise の勝敗をどれだけ再現できるかを返す。"""
    w = np.asarray([getattr(weights, axis) for axis in AXES], dtype=float)
    total = 0
    correct = 0
    for ex in examples:
        diff = np.asarray(
            [
                float(ex.scores_a.get(axis, 0.0)) - float(ex.scores_b.get(axis, 0.0))
                for axis in AXES
            ],
            dtype=float,
        )
        margin = float(diff @ w)
        pred = "tie" if abs(margin) <= tie_threshold else "A" if margin > 0 else "B"
        total += 1
        correct += int(pred == ex.winner)
    return correct / total if total else 0.0


def regress_weights(
    examples: Iterable[PairwiseTrainingExample],
    *,
    base_weights: ScoringWeights | None = None,
    l2: float = 0.2,
    learning_rate: float = 0.05,
    max_iter: int = 2000,
) -> WeightRegressionResult:
    """Ridge 正則化つきロジスティック回帰で非負の重みを推定する。

    sklearn を追加依存にせず、numpy のみで小さなバッチ学習を行う。
    """
    examples = list(examples)
    x, y = _features_and_labels(examples)
    if len(y) == 0:
        weights = base_weights or ScoringWeights()
        return WeightRegressionResult(
            weights=weights, pairwise_acc=pairwise_accuracy(examples, weights), n_pairs=0
        )

    initial = base_weights or ScoringWeights()
    prior = np.asarray([getattr(initial, axis) for axis in AXES], dtype=float)
    w = prior.copy()

    for _ in range(max_iter):
        logits = np.clip(x @ w, -40.0, 40.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        grad = (x.T @ (probs - y)) / len(y) + l2 * (w - prior)
        w -= learning_rate * grad
        w = np.clip(w, 0.05, 3.0)

    learned = ScoringWeights(
        **{axis: round(float(value), 4) for axis, value in zip(AXES, w, strict=True)}
    )
    return WeightRegressionResult(
        weights=learned,
        pairwise_acc=pairwise_accuracy(examples, learned),
        n_pairs=len(examples),
    )
