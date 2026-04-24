"""Aggregation utilities for evaluated meeting utterances."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from app.schemas.models import (
    AverageScores,
    EvaluatedUtterance,
    MeetingSummary,
    SpeakerSummary,
    StyleLabel,
)


def _determine_style_label(avg: AverageScores) -> str:
    """軸別平均から話者の発言傾向ラベルを決める。"""
    axis_scores = {
        StyleLabel.ORGANIZER.value: avg.issue_clarification + avg.summarization,
        StyleLabel.PROPOSER.value: avg.decision_progress + avg.novelty,
        StyleLabel.CAUTIOUS.value: avg.risk_detection + avg.groundedness,
        StyleLabel.DRIVER.value: avg.actionability + avg.decision_progress,
    }
    return max(axis_scores, key=axis_scores.get)


def aggregate_by_speaker(
    evaluated: list[EvaluatedUtterance],
    top_count: int = 2,
) -> list[SpeakerSummary]:
    """話者別サマリーを生成する。"""
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
    """総合スコア上位の発言を返す。"""
    return sorted(evaluated, key=lambda u: u.total_score, reverse=True)[:top_count]


def extract_top_by_axis(
    evaluated: list[EvaluatedUtterance],
    axis: str,
    top_count: int = 3,
) -> list[EvaluatedUtterance]:
    """特定の評価軸で上位の発言を返す。スコア0の発言は除外する。"""

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
    """会議全体のサマリーを組み立てる。"""
    speaker_summaries = aggregate_by_speaker(evaluated)

    return MeetingSummary(
        meeting_id=meeting_id,
        title=title,
        goal=goal,
        overall_comment=_generate_overall_comment(evaluated),
        top_utterances=extract_top_utterances(evaluated, top_count),
        top_issue_clarification=extract_top_by_axis(evaluated, "issue_clarification", top_axis_count),
        top_decision_progress=extract_top_by_axis(evaluated, "decision_progress", top_axis_count),
        top_risk_detection=extract_top_by_axis(evaluated, "risk_detection", top_axis_count),
        top_actionability=extract_top_by_axis(evaluated, "actionability", top_axis_count),
        improvement_comments=_generate_improvement_comments(evaluated),
        speaker_summaries=speaker_summaries,
        evaluated_utterances=evaluated,
    )


def _generate_overall_comment(evaluated: list[EvaluatedUtterance]) -> str:
    """会議全体への短いコメントをルールベースで生成する。"""
    if not evaluated:
        return "発言データがありません。"

    avg_score = sum(u.total_score for u in evaluated) / len(evaluated)
    total_penalties = sum(
        u.penalties.duplication
        + u.penalties.verbosity
        + u.penalties.off_topic
        + u.penalties.unsupported_assertion
        for u in evaluated
    )

    parts = []
    if avg_score >= 4.0:
        parts.append("会議の目的に沿った貢献が多く見られました。")
    elif avg_score >= 2.0:
        parts.append("議論はおおむね目的に沿って進行しています。")
    else:
        parts.append("議論の焦点が定まりにくい場面がありました。")

    if total_penalties < -5:
        parts.append("重複、冗長さ、脱線が目立つ箇所もあります。")

    return "".join(parts)


def _generate_improvement_comments(
    evaluated: list[EvaluatedUtterance],
) -> list[str]:
    """改善コメントを生成する。"""
    comments = []

    checks: list[tuple[Callable[[EvaluatedUtterance], bool], str]] = [
        (
            lambda u: u.penalties.duplication <= -2,
            "重複した発言が{count}件検出されました。同じ内容の繰り返しを減らすと、会議の進行がより滑らかになります。",
        ),
        (
            lambda u: u.penalties.off_topic <= -2,
            "議題から離れた発言が{count}件ありました。現在の論点を確認しながら進めると改善できます。",
        ),
        (
            lambda u: u.penalties.verbosity <= -2,
            "冗長な発言が{count}件ありました。結論を先に述べると、意図が伝わりやすくなります。",
        ),
    ]

    for predicate, template in checks:
        count = len([u for u in evaluated if predicate(u)])
        if count:
            comments.append(template.format(count=count))

    action_scores = [u.scores.actionability for u in evaluated]
    if action_scores and max(action_scores) <= 1:
        comments.append(
            "次の行動につながる発言が少なめです。担当者、期限、次の確認事項を明確にすると改善できます。"
        )

    return comments
