"""プロンプト構築と JSON Schema (Issue #12)。

OpenAI / ローカル LLM の両バックエンドで共通利用するため、
プロンプト構築・JSON Schema・応答パース・結果正規化を独立モジュールに切り出す。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, cast

from app.evaluators.base import EvaluationResult
from app.schemas.models import Penalties, Scores, SpeechType

if TYPE_CHECKING:
    from app.context_builder.builder import EvaluationContext

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "utterance_eval.txt"

_SPEECH_TYPE_VALUES = [s.value for s in SpeechType]

RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "speech_type": {
            "type": "string",
            "enum": _SPEECH_TYPE_VALUES,
        },
        "scores": {
            "type": "object",
            "properties": {
                "issue_clarification": {"type": "integer", "minimum": 0, "maximum": 3},
                "decision_progress": {"type": "integer", "minimum": 0, "maximum": 3},
                "risk_detection": {"type": "integer", "minimum": 0, "maximum": 3},
                "actionability": {"type": "integer", "minimum": 0, "maximum": 3},
                "groundedness": {"type": "integer", "minimum": 0, "maximum": 3},
                "novelty": {"type": "integer", "minimum": 0, "maximum": 3},
                "summarization": {"type": "integer", "minimum": 0, "maximum": 3},
            },
            "required": [
                "issue_clarification",
                "decision_progress",
                "risk_detection",
                "actionability",
                "groundedness",
                "novelty",
                "summarization",
            ],
            "additionalProperties": False,
        },
        "penalties": {
            "type": "object",
            "properties": {
                "duplication": {"type": "integer", "minimum": -3, "maximum": 0},
                "verbosity": {"type": "integer", "minimum": -3, "maximum": 0},
                "off_topic": {"type": "integer", "minimum": -3, "maximum": 0},
                "unsupported_assertion": {
                    "type": "integer",
                    "minimum": -3,
                    "maximum": 0,
                },
            },
            "required": [
                "duplication",
                "verbosity",
                "off_topic",
                "unsupported_assertion",
            ],
            "additionalProperties": False,
        },
        "reason": {"type": "string"},
    },
    "required": ["speech_type", "scores", "penalties", "reason"],
    "additionalProperties": False,
}

RESPONSE_SCHEMA_NAME = "utterance_evaluation"


def _load_prompt_template() -> Template:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    return Template(text)


def _format_utterances(utterances) -> str:
    if not utterances:
        return "(なし)"
    return "\n".join(f"[{u.timestamp}] {u.speaker}: {u.text}" for u in utterances)


_MEETING_TYPE_LABELS: dict[str, str] = {
    "decision": "意思決定会議(重視軸: 意思決定寄与・根拠性・リスク検知)",
    "brainstorming": "ブレスト会議(重視軸: 新規性・論点整理・根拠性)",
    "progress": "進捗共有・定例(重視軸: アクション化・リスク検知・要約)",
    "retrospective": "振り返り・レビュー(重視軸: 根拠性・リスク検知・要約・論点整理)",
}


def build_prompt(ctx: EvaluationContext) -> str:
    """EvaluationContext からプロンプト文字列を構築する。"""
    template = _load_prompt_template()
    meeting_type_label = (
        _MEETING_TYPE_LABELS.get(ctx.meeting_type, ctx.meeting_type)
        if ctx.meeting_type
        else "(未指定)"
    )
    return template.substitute(
        meeting_type=meeting_type_label,
        meeting_goal=ctx.meeting_goal,
        agenda="、".join(ctx.agenda) if ctx.agenda else "(なし)",
        decision_points="、".join(ctx.decision_points) if ctx.decision_points else "(なし)",
        current_topic=ctx.current_topic if ctx.current_topic else "(未設定)",
        before_utterances=_format_utterances(ctx.before_utterances),
        target_speaker=ctx.target_utterance.speaker,
        target_timestamp=ctx.target_utterance.timestamp,
        target_text=ctx.target_utterance.text,
        after_utterances=_format_utterances(ctx.after_utterances),
    )


def parse_response(text: str) -> dict[str, object]:
    """LLM 応答テキストから JSON を抽出してパースする。

    ```json ... ``` で囲まれているケースにも対応する。
    """
    if "```json" in text:
        start = text.index("```json") + len("```json")
        end = text.index("```", start)
        text = text[start:end]
    elif "```" in text:
        start = text.index("```") + len("```")
        end = text.index("```", start)
        text = text[start:end]
    return cast(dict[str, object], json.loads(text.strip()))


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _normalize_speech_type(value: str | None) -> str:
    """仕様で許可された発言タイプに正規化する。"""
    if value in _SPEECH_TYPE_VALUES:
        return value
    compact = (value or "").replace(" ", "")
    for candidate in _SPEECH_TYPE_VALUES:
        if compact == candidate.replace(" ", ""):
            return candidate
    return SpeechType.INFO_SHARING.value


def normalize_result(parsed: dict) -> EvaluationResult:
    """パース結果（dict）を EvaluationResult に正規化する。

    スコアは [0, 3]、減点は [-3, 0] にクランプ、speech_type は許容値に正規化。
    """
    scores_raw = parsed.get("scores", {})
    penalties_raw = parsed.get("penalties", {})

    scores = Scores(
        issue_clarification=_clamp(scores_raw.get("issue_clarification", 0), 0, 3),
        decision_progress=_clamp(scores_raw.get("decision_progress", 0), 0, 3),
        risk_detection=_clamp(scores_raw.get("risk_detection", 0), 0, 3),
        actionability=_clamp(scores_raw.get("actionability", 0), 0, 3),
        groundedness=_clamp(scores_raw.get("groundedness", 0), 0, 3),
        novelty=_clamp(scores_raw.get("novelty", 0), 0, 3),
        summarization=_clamp(scores_raw.get("summarization", 0), 0, 3),
    )
    penalties = Penalties(
        duplication=_clamp(penalties_raw.get("duplication", 0), -3, 0),
        verbosity=_clamp(penalties_raw.get("verbosity", 0), -3, 0),
        off_topic=_clamp(penalties_raw.get("off_topic", 0), -3, 0),
        unsupported_assertion=_clamp(penalties_raw.get("unsupported_assertion", 0), -3, 0),
    )

    return EvaluationResult(
        speech_type=_normalize_speech_type(parsed.get("speech_type")),
        scores=scores,
        penalties=penalties,
        reason=parsed.get("reason", ""),
        evaluation_failed=False,
    )


__all__ = [
    "PROMPT_PATH",
    "RESPONSE_SCHEMA",
    "RESPONSE_SCHEMA_NAME",
    "build_prompt",
    "normalize_result",
    "parse_response",
]
