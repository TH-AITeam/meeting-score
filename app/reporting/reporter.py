"""Format domain models for the UI response."""

from __future__ import annotations

from app.schemas.models import EvaluatedUtterance, MeetingSummary, SpeakerSummary


def format_utterance_for_ui(eu: EvaluatedUtterance) -> dict:
    """Convert an evaluated utterance to a UI-friendly dict."""
    return {
        "utterance_id": eu.utterance_id,
        "speaker": eu.speaker,
        "timestamp": eu.timestamp,
        "text": eu.text,
        "speech_type": eu.speech_type,
        "scores": eu.scores.model_dump(),
        "penalties": eu.penalties.model_dump(),
        "total_score": eu.total_score,
        "reason": eu.reason,
    }


def format_speaker_for_ui(ss: SpeakerSummary) -> dict:
    """Convert a speaker summary to a UI-friendly dict."""
    return {
        "speaker": ss.speaker,
        "utterance_count": ss.utterance_count,
        "total_contribution_score": ss.total_contribution_score,
        "average_total_score": ss.average_total_score,
        "average_scores": ss.average_scores.model_dump(),
        "style_label": ss.style_label,
        "top_utterances": ss.top_utterances,
    }


def format_meeting_summary_for_ui(ms: MeetingSummary) -> dict:
    """Convert a meeting summary to the public API response shape."""
    return {
        "meeting_id": ms.meeting_id,
        "title": ms.title,
        "goal": ms.goal,
        "overall_comment": ms.overall_comment,
        "top_utterances": [format_utterance_for_ui(u) for u in ms.top_utterances],
        "top_issue_clarification": [format_utterance_for_ui(u) for u in ms.top_issue_clarification],
        "top_decision_progress": [format_utterance_for_ui(u) for u in ms.top_decision_progress],
        "top_risk_detection": [format_utterance_for_ui(u) for u in ms.top_risk_detection],
        "top_actionability": [format_utterance_for_ui(u) for u in ms.top_actionability],
        "improvement_comments": ms.improvement_comments,
        "speaker_summaries": [format_speaker_for_ui(s) for s in ms.speaker_summaries],
        "evaluated_utterances": [format_utterance_for_ui(u) for u in ms.evaluated_utterances],
    }
