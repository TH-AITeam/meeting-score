"""集計モジュール

話者別サマリー・会議全体のTop発言抽出を行う。
"""

from __future__ import annotations

from collections import defaultdict

from app.schemas.models import (
    AverageScores,
    EvaluatedUtterance,
    MeetingSummary,
    SpeakerSummary,
)


def _determine_style_label(avg: AverageScores) -> str:
    """軸別平均から発言傾向ラベルを決定する"""
    axis_scores = {
        "整理型": avg.issue_clarification + avg.summarization,
        "提案型": avg.decision_progress + avg.novelty,
        "警戒型": avg.risk_detection + avg.groundedness,
        "推進型": avg.actionability + avg.decision_progress,
    }
    return max(axis_scores, key=lambda k: axis_scores[k])


def aggregate_by_speaker(
    evaluated: list[EvaluatedUtterance],
    top_count: int = 2,
) -> list[SpeakerSummary]:
    """話者別にサマリーを生成する"""
    by_speaker: dict[str, list[EvaluatedUtterance]] = defaultdict(list)
    for eu in evaluated:
        by_speaker[eu.speaker].append(eu)

    summaries: list[SpeakerSummary] = []

    for speaker, utterances in sorted(by_speaker.items()):
        n = len(utterances)
        avg = AverageScores(
            issue_clarification=round(sum(u.scores.issue_clarification for u in utterances) / n, 2),
            decision_progress=round(sum(u.scores.decision_progress for u in utterances) / n, 2),
            risk_detection=round(sum(u.scores.risk_detection for u in utterances) / n, 2),
            actionability=round(sum(u.scores.actionability for u in utterances) / n, 2),
            groundedness=round(sum(u.scores.groundedness for u in utterances) / n, 2),
            novelty=round(sum(u.scores.novelty for u in utterances) / n, 2),
            summarization=round(sum(u.scores.summarization for u in utterances) / n, 2),
        )
        sum_total = round(sum(u.total_score for u in utterances), 2)
        avg_total = round(sum_total / n, 2)

        # Top発言を total_score 降順で取得
        sorted_utts = sorted(utterances, key=lambda u: u.total_score, reverse=True)
        top_ids = [u.utterance_id for u in sorted_utts[:top_count]]

        summaries.append(
            SpeakerSummary(
                speaker=speaker,
                utterance_count=n,
                total_contribution_score=sum_total,
                average_total_score=avg_total,
                average_scores=avg,
                style_label=_determine_style_label(avg),
                top_utterances=top_ids,
            )
        )

    return summaries


def extract_top_utterances(
    evaluated: list[EvaluatedUtterance],
    top_count: int = 5,
) -> list[EvaluatedUtterance]:
    """総合点上位の発言を返す"""
    return sorted(evaluated, key=lambda u: u.total_score, reverse=True)[:top_count]


def extract_top_by_axis(
    evaluated: list[EvaluatedUtterance],
    axis: str,
    top_count: int = 3,
) -> list[EvaluatedUtterance]:
    """特定の評価軸で上位の発言を返す（スコア0の発言は除外）"""

    def _get_score(eu: EvaluatedUtterance) -> int:
        return getattr(eu.scores, axis, 0)

    positive = [eu for eu in evaluated if _get_score(eu) > 0]
    return sorted(positive, key=_get_score, reverse=True)[:top_count]


def build_meeting_summary(
    meeting_id: str,
    title: str,
    goal: str,
    evaluated: list[EvaluatedUtterance],
    top_count: int = 5,
    top_axis_count: int = 3,
) -> MeetingSummary:
    """会議全体のサマリーを組み立てる"""
    speaker_summaries = aggregate_by_speaker(evaluated)
    top_all = extract_top_utterances(evaluated, top_count)
    top_issue = extract_top_by_axis(evaluated, "issue_clarification", top_axis_count)
    top_decision = extract_top_by_axis(evaluated, "decision_progress", top_axis_count)
    top_risk = extract_top_by_axis(evaluated, "risk_detection", top_axis_count)
    top_action = extract_top_by_axis(evaluated, "actionability", top_axis_count)

    # 改善コメントの生成
    improvement_comments = _generate_improvement_comments(evaluated)

    return MeetingSummary(
        meeting_id=meeting_id,
        title=title,
        goal=goal,
        overall_comment=_generate_overall_comment(evaluated, speaker_summaries),
        top_utterances=top_all,
        top_issue_clarification=top_issue,
        top_decision_progress=top_decision,
        top_risk_detection=top_risk,
        top_actionability=top_action,
        improvement_comments=improvement_comments,
        speaker_summaries=speaker_summaries,
        evaluated_utterances=evaluated,
    )


def _generate_overall_comment(
    evaluated: list[EvaluatedUtterance],
    speakers: list[SpeakerSummary],
) -> str:
    """会議全体の簡易コメントをルールベースで生成する"""
    if not evaluated:
        return "発言データがありません。"

    avg_score = sum(u.total_score for u in evaluated) / len(evaluated)
    total_penalties = sum(
        u.penalties.duplication
        + u.penalties.verbosity
        + u.penalties.off_topic
        + u.penalties.unsupported_assertion
        + u.penalties.override
        for u in evaluated
    )

    parts = []
    if avg_score >= 4.0:
        parts.append("全体的に質の高い議論が行われました。")
    elif avg_score >= 2.0:
        parts.append("議論は概ね目的に沿って進行しました。")
    else:
        parts.append("議論の焦点が定まりにくい場面が見られました。")

    if total_penalties < -5:
        parts.append("重複や脱線が目立つ箇所がありました。")

    return "".join(parts)


def _generate_improvement_comments(
    evaluated: list[EvaluatedUtterance],
) -> list[str]:
    """改善コメントを生成する"""
    comments = []

    # 重複が多い場面の検出
    dup_utterances = [u for u in evaluated if u.penalties.duplication <= -2]
    if dup_utterances:
        comments.append(
            f"重複発言が{len(dup_utterances)}件検出されました。同じ内容の繰り返しを減らすことで会議の効率が上がります。"
        )

    # 脱線が多い場面の検出
    offtopic = [u for u in evaluated if u.penalties.off_topic <= -2]
    if offtopic:
        comments.append(
            f"論点から逸脱した発言が{len(offtopic)}件ありました。アジェンダに沿った進行を意識すると改善できます。"
        )

    # 冗長発言の検出
    verbose = [u for u in evaluated if u.penalties.verbosity <= -2]
    if verbose:
        comments.append(
            f"冗長な発言が{len(verbose)}件ありました。結論を先に述べることで議論がスムーズになります。"
        )

    # 上書き発言が多い場面の検出
    overrides = [u for u in evaluated if u.penalties.override <= -2]
    if overrides:
        comments.append(
            f"直前発言を受けずに自説を被せた発言が{len(overrides)}件ありました。前の発言への応答を明確にすると議論がつながります。"
        )

    # アクション化が弱い場合
    action_scores = [u.scores.actionability for u in evaluated]
    if action_scores and max(action_scores) <= 1:
        comments.append(
            "次アクションの明確化が不足しています。会議終了前に担当・期限を確認する時間を設けると改善できます。"
        )

    return comments
