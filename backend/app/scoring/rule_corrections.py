"""ルールベース補正モジュール

LLM 評価後に、ルールベースで重複・冗長を追加補正し、
総合スコアを再計算する。

仕様 13.2: 直前〜会議全体の既出発言との意味重複を検出
仕様 13.3: 発言長・情報密度で冗長判定
"""

from __future__ import annotations

from app.schemas.models import EvaluatedUtterance, Penalties, SpeechType
from app.scoring.calculator import calculate_total_score
from app.scoring.weights import PenaltyWeights, ScoringWeights

# 冗長判定の閾値（文字数）
_VERBOSITY_CHAR_THRESHOLD_MILD = 120
_VERBOSITY_CHAR_THRESHOLD_STRONG = 200

# 重複判定: 短い発言で加点もない場合に既出テキストと照合
_DUPLICATE_OVERLAP_RATIO = 0.6

# 上書き判定: 直前発言とほぼ語彙接点がない場合だけ拾う、緩めの初期値。
_OVERRIDE_REPLY_OVERLAP_RATIO = 0.08
_OVERRIDE_MIN_TEXT_LENGTH = 12
_ASSERTIVE_SPEECH_TYPES = {
    SpeechType.PROPOSAL.value,
    SpeechType.DECISION_PUSH.value,
}
_PRIOR_INVITATION_MARKERS = (
    "?",
    "\uff1f",
    "ませんか",
    "ましょう",
    "したい",
    "どうですか",
    "いかが",
    "確認",
    "次に",
)

_REPLY_MARKERS = (
    "同意",
    "賛成",
    "反対",
    "確かに",
    "たしかに",
    "ただ",
    "一方",
    "とはいえ",
    "しかし",
    "でも",
    "ですが",
    "ではなく",
    "じゃなく",
    "むしろ",
    "今の",
    "先ほど",
    "さっき",
    "踏まえ",
    "受けて",
    "について",
    "観点",
    "論点",
)


def _bigrams(text: str) -> set[str]:
    """テキストから bigram(2文字組) の集合を生成する"""
    if len(text) < 2:
        return set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


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


def _check_override(target: EvaluatedUtterance, prior: EvaluatedUtterance | None) -> int:
    """直前の他者発言を無視して自説を被せた発言を検出する (0 or -1)"""
    if prior is None:
        return 0
    if target.speaker == prior.speaker:
        return 0
    if target.speech_type not in _ASSERTIVE_SPEECH_TYPES:
        return 0
    if len(target.text) < _OVERRIDE_MIN_TEXT_LENGTH:
        return 0

    # 司会・他者から論点提示や提案依頼を受けた直後は、自然な応答として扱う。
    if any(marker in prior.text for marker in _PRIOR_INVITATION_MARKERS):
        return 0

    # 直前発言への明示的な参照・同意・反論・訂正がある場合は、正当な論点修正として扱う。
    if prior.speaker in target.text or any(marker in target.text for marker in _REPLY_MARKERS):
        return 0

    if _bigram_jaccard(target.text, prior.text) >= _OVERRIDE_REPLY_OVERLAP_RATIO:
        return 0

    return -1


def apply_rule_corrections(
    evaluated: list[EvaluatedUtterance],
    weights: ScoringWeights | None = None,
    penalty_weights: PenaltyWeights | None = None,
) -> list[EvaluatedUtterance]:
    """ルールベースで重複・冗長を追加補正し、総合スコアを再計算する"""
    corrected: list[EvaluatedUtterance] = []
    prior_texts: list[str] = []
    prior_utterance: EvaluatedUtterance | None = None

    for eu in evaluated:
        dup_adj = _check_duplication(eu, prior_texts)
        verb_adj = _check_verbosity(eu)
        override_adj = _check_override(eu, prior_utterance)

        needs_update = dup_adj != 0 or verb_adj != 0 or override_adj != 0

        if needs_update:
            new_dup = max(eu.penalties.duplication + dup_adj, -3)
            new_verb = max(eu.penalties.verbosity + verb_adj, -3)
            new_override = max(eu.penalties.override + override_adj, -3)
            new_penalties = Penalties(
                duplication=new_dup,
                verbosity=new_verb,
                off_topic=eu.penalties.off_topic,
                unsupported_assertion=eu.penalties.unsupported_assertion,
                override=new_override,
            )
            new_total = calculate_total_score(eu.scores, new_penalties, weights, penalty_weights)
            eu = eu.model_copy(
                update={
                    "penalties": new_penalties,
                    "total_score": new_total,
                }
            )

        corrected.append(eu)
        prior_texts.append(eu.text)
        prior_utterance = eu

    return corrected
