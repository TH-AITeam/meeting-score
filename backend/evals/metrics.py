"""評価メトリクス (Issue #5)。

AGENT.md §15「人が見て納得」を以下の4つの定量指標に分解する。

- spearman      : 人手ランクとシステムランクの順位相関
- kendall_tau   : 同上（ペアの一致／不一致ベース、より厳しめ）
- top_k_jaccard : 人手 Top-K と システム Top-K の Jaccard 係数
- pairwise_accuracy: ペアワイズ比較の一致率（A_better / B_better / tie）

スコア差が `tie_threshold` 以下なら tie 判定する。閾値はデフォルト 0.5。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evals.schema import PairwiseAnnotation

DEFAULT_TIE_THRESHOLD: float = 0.5


def _ranks(values: list[float]) -> list[float]:
    """値の平均ランクを返す（タイは平均）。1 始まり。

    scipy 等を呼ばずに済むよう小さい実装。
    """
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based の (i+1) .. (j+1) の平均
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(human_ranks: list[float], system_ranks: list[float]) -> float:
    """Spearman 順位相関係数 (-1.0 ～ 1.0)。

    human_ranks / system_ranks は同じ長さで、同じ発言を指す並びである前提。
    スコア値そのものを渡しても良い（内部で平均ランクに変換）。
    """
    if len(human_ranks) != len(system_ranks):
        msg = "human_ranks と system_ranks の長さが一致しません"
        raise ValueError(msg)
    n = len(human_ranks)
    if n < 2:
        return 0.0
    rh = _ranks(human_ranks)
    rs = _ranks(system_ranks)
    mean_h = sum(rh) / n
    mean_s = sum(rs) / n
    num = sum((rh[i] - mean_h) * (rs[i] - mean_s) for i in range(n))
    den_h = sum((rh[i] - mean_h) ** 2 for i in range(n)) ** 0.5
    den_s = sum((rs[i] - mean_s) ** 2 for i in range(n)) ** 0.5
    if den_h == 0 or den_s == 0:
        return 0.0
    return num / (den_h * den_s)


def kendall_tau(human_ranks: list[float], system_ranks: list[float]) -> float:
    """Kendall tau-b 相関係数 (-1.0 ～ 1.0)。

    ペア (i, j) について同順 (concordant) と逆順 (discordant) の差を取る。
    """
    if len(human_ranks) != len(system_ranks):
        msg = "human_ranks と system_ranks の長さが一致しません"
        raise ValueError(msg)
    n = len(human_ranks)
    if n < 2:
        return 0.0

    concordant = 0
    discordant = 0
    tie_h = 0
    tie_s = 0
    for i in range(n):
        for j in range(i + 1, n):
            dh = human_ranks[i] - human_ranks[j]
            ds = system_ranks[i] - system_ranks[j]
            if dh == 0 and ds == 0:
                tie_h += 1
                tie_s += 1
            elif dh == 0:
                tie_h += 1
            elif ds == 0:
                tie_s += 1
            elif (dh > 0 and ds > 0) or (dh < 0 and ds < 0):
                concordant += 1
            else:
                discordant += 1

    total_pairs = n * (n - 1) / 2
    den = ((total_pairs - tie_h) * (total_pairs - tie_s)) ** 0.5
    if den == 0:
        return 0.0
    return (concordant - discordant) / den


def top_k_jaccard(
    human_top: Iterable[str],
    system_top: Iterable[str],
    k: int = 5,
) -> float:
    """Top-K Jaccard 係数 (0.0 ～ 1.0)。

    両者の上位 k 件の集合の Jaccard。両方とも空なら 1.0。
    """
    h = set(list(human_top)[:k])
    s = set(list(system_top)[:k])
    if not h and not s:
        return 1.0
    union = h | s
    return len(h & s) / len(union) if union else 0.0


def _winner_from_scores(
    score_a: float, score_b: float, tie_threshold: float
) -> str:
    if abs(score_a - score_b) <= tie_threshold:
        return "tie"
    return "A_better" if score_a > score_b else "B_better"


@dataclass
class PairwiseAccuracyReport:
    accuracy: float
    by_class: dict[str, dict]
    n: int
    n_skipped: int


def pairwise_accuracy(
    pairs: Iterable[PairwiseAnnotation],
    system_scores: dict[str, float],
    tie_threshold: float = DEFAULT_TIE_THRESHOLD,
) -> PairwiseAccuracyReport:
    """ペアワイズ判定 (A_better / B_better / tie) の一致率を計算する。

    Parameters
    ----------
    pairs : Iterable[PairwiseAnnotation]
        人手アノテのペアリスト。
    system_scores : dict[utterance_id, float]
        システムの総合スコア。
    tie_threshold : float
        2つのスコア差がこれ以下なら tie 扱い。

    Returns
    -------
    PairwiseAccuracyReport
        accuracy : 全体一致率
        by_class : クラスごとの正解数/総数/precision/recall
        n : 集計対象のペア数
        n_skipped : utt_a / utt_b のどちらかが system_scores に無くスキップした数
    """
    classes = ("A_better", "B_better", "tie")
    counts: dict[str, dict[str, int]] = {
        c: {"correct": 0, "total_in_human": 0, "total_in_system": 0} for c in classes
    }

    correct = 0
    total = 0
    n_skipped = 0

    for p in pairs:
        if p.utt_a not in system_scores or p.utt_b not in system_scores:
            n_skipped += 1
            continue
        human = p.winner
        system = _winner_from_scores(
            system_scores[p.utt_a], system_scores[p.utt_b], tie_threshold
        )
        counts[human]["total_in_human"] += 1
        counts[system]["total_in_system"] += 1
        if human == system:
            correct += 1
            counts[human]["correct"] += 1
        total += 1

    accuracy = correct / total if total else 0.0

    by_class: dict[str, dict] = {}
    for c in classes:
        tih = counts[c]["total_in_human"]
        tis = counts[c]["total_in_system"]
        cc = counts[c]["correct"]
        by_class[c] = {
            "correct": cc,
            "n_in_human": tih,
            "n_in_system": tis,
            "recall": cc / tih if tih else 0.0,
            "precision": cc / tis if tis else 0.0,
        }

    return PairwiseAccuracyReport(
        accuracy=accuracy,
        by_class=by_class,
        n=total,
        n_skipped=n_skipped,
    )


__all__ = [
    "DEFAULT_TIE_THRESHOLD",
    "PairwiseAccuracyReport",
    "kendall_tau",
    "pairwise_accuracy",
    "spearman",
    "top_k_jaccard",
]
