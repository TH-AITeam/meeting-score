"""ルールベース補正モジュール

LLM 評価後に、ルールベースで重複・冗長を追加補正し、
総合スコアを再計算する。

仕様 13.2: 直前〜会議全体の既出発言との意味重複を検出
仕様 13.3: 発言長・情報密度で冗長判定
"""

from __future__ import annotations

from app.schemas.models import EvaluatedUtterance, Penalties
from app.scoring.calculator import calculate_total_score
from app.scoring.weights import ScoringWeights

# 冗長判定の閾値（文字数）
_VERBOSITY_CHAR_THRESHOLD_MILD = 120
_VERBOSITY_CHAR_THRESHOLD_STRONG = 200

# 重複判定: 短い発言で加点もない場合に既出テキストと照合
_DUPLICATE_OVERLAP_RATIO = 0.6


def _bigrams(text: str) -> set[str]:
    """テキストから bigram(2文字組) の集合を生成する"""
    if len(text) < 2:
        return set()
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _bigram_jaccard(a: str, b: str) -> float:
    """2つの文字列間の bigram Jaccard 類似度

    文字集合ではなく bigram (連続2文字) の集合で比較することで、
    語順・文脈を考慮した意味的な重複検出を行う。
    """
    if not a or not b:
        return 0.0
    set_a = _bigrams(a)
    set_b = _bigrams(b)
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _check_duplication(
    target: EvaluatedUtterance,
    prior_texts: list[str],
) -> int:
    """既出発言との重複を検出し、追加減点を返す (0 or -1)"""
    text = target.text
    for prior in prior_texts:
        if _bigram_jaccard(text, prior) >= _DUPLICATE_OVERLAP_RATIO:
            return -1
    return 0


def _check_verbosity(target: EvaluatedUtterance) -> int:
    """発言長から冗長さを追加減点する (0 or -1)

    長さだけで判定する。LLM が既に冗長減点を付けている場合はそれを尊重し、
    ここでは LLM が見逃した「長いのに加点スコアが低い」ケースを拾う。
    """
    text_len = len(target.text)
    # 加点が十分ある発言は冗長とみなさない
    total_positive = (
        target.scores.issue_clarification
        + target.scores.decision_progress
        + target.scores.risk_detection
        + target.scores.actionability
        + target.scores.groundedness
        + target.scores.novelty
        + target.scores.summarization
    )
    if total_positive >= 4:
        return 0

    if text_len >= _VERBOSITY_CHAR_THRESHOLD_STRONG and total_positive <= 1:
        return -1
    if text_len >= _VERBOSITY_CHAR_THRESHOLD_MILD and total_positive == 0:
        return -1

    return 0


def apply_rule_corrections(
    evaluated: list[EvaluatedUtterance],
    weights: ScoringWeights | None = None,
) -> list[EvaluatedUtterance]:
    """ルールベースで重複・冗長を追加補正し、総合スコアを再計算する"""
    corrected: list[EvaluatedUtterance] = []
    prior_texts: list[str] = []

    for eu in evaluated:
        dup_adj = _check_duplication(eu, prior_texts)
        verb_adj = _check_verbosity(eu)

        needs_update = (dup_adj != 0 or verb_adj != 0)

        if needs_update:
            new_dup = max(eu.penalties.duplication + dup_adj, -3)
            new_verb = max(eu.penalties.verbosity + verb_adj, -3)
            new_penalties = Penalties(
                duplication=new_dup,
                verbosity=new_verb,
                off_topic=eu.penalties.off_topic,
                unsupported_assertion=eu.penalties.unsupported_assertion,
            )
            new_total = calculate_total_score(eu.scores, new_penalties, weights)
            eu = eu.model_copy(update={
                "penalties": new_penalties,
                "total_score": new_total,
            })

        corrected.append(eu)
        prior_texts.append(eu.text)

    return corrected
