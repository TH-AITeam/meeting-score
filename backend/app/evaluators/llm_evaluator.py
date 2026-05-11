"""LLM による発言評価モジュール

各発言を OpenAI Responses API で評価し、
発言タイプ・軸別スコア・減点・理由を返す。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from string import Template

from app.context_builder.builder import EvaluationContext
from app.schemas.models import Penalties, Scores, SpeechType

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "utterance_eval.txt"
_SPEECH_TYPE_VALUES = [speech_type.value for speech_type in SpeechType]
_RESPONSE_SCHEMA = {
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
                "unsupported_assertion": {"type": "integer", "minimum": -3, "maximum": 0},
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


def _load_prompt_template() -> Template:
    text = _PROMPT_PATH.read_text(encoding="utf-8")
    return Template(text)


def _format_utterances(utterances) -> str:
    if not utterances:
        return "(なし)"
    lines = []
    for u in utterances:
        lines.append(f"[{u.timestamp}] {u.speaker}: {u.text}")
    return "\n".join(lines)


def _build_prompt(ctx: EvaluationContext) -> str:
    template = _load_prompt_template()
    return template.substitute(
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


def _parse_response(text: str) -> dict:
    """LLM応答からJSONを抽出してパースする"""
    # ```json ... ``` ブロックがあれば抽出
    if "```json" in text:
        start = text.index("```json") + len("```json")
        end = text.index("```", start)
        text = text[start:end]
    elif "```" in text:
        start = text.index("```") + len("```")
        end = text.index("```", start)
        text = text[start:end]

    return json.loads(text.strip())


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _normalize_speech_type(value: str | None) -> str:
    """仕様で許可された発言タイプに正規化する"""
    if value in _SPEECH_TYPE_VALUES:
        return value

    compact = (value or "").replace(" ", "")
    for candidate in _SPEECH_TYPE_VALUES:
        if compact == candidate.replace(" ", ""):
            return candidate

    return SpeechType.INFO_SHARING.value


def _load_dotenv_if_available() -> None:
    """必要であれば .env を読み込む"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()


def _extract_response_text(response) -> str:
    """Responses API 応答からテキストを取り出す"""
    output_text = getattr(response, "output_text", "")
    if output_text:
        return output_text

    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []):
            text = getattr(content, "text", None)
            if text:
                return text

    raise ValueError("OpenAI 応答からテキストを取得できませんでした。")


def _safe_result(parsed: dict) -> dict:
    """パース結果を安全な範囲に正規化する"""
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

    return {
        "speech_type": _normalize_speech_type(parsed.get("speech_type")),
        "scores": scores,
        "penalties": penalties,
        "reason": parsed.get("reason", ""),
    }


def _default_result() -> dict:
    """パース完全失敗時のデフォルト値"""
    return {
        "speech_type": "情報共有",
        "scores": Scores(),
        "penalties": Penalties(),
        "reason": "評価を取得できませんでした。",
        "evaluation_failed": True,
    }


def evaluate_utterance(
    ctx: EvaluationContext,
    model: str = "gpt-5.4-mini",
    max_tokens: int = 1024,
    max_retries: int = 3,
    temperature: float | None = None,
) -> dict:
    """1発言を LLM で評価する

    Returns:
        {speech_type, scores: Scores, penalties: Penalties, reason, evaluation_failed}
    """
    for attempt in range(max_retries):
        try:
            _load_dotenv_if_available()
            from openai import APIError, OpenAI

            client = OpenAI()
            prompt = _build_prompt(ctx)
            request = {
                "model": model,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt,
                            }
                        ],
                    }
                ],
                "max_output_tokens": max_tokens,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "utterance_evaluation",
                        "strict": True,
                        "schema": _RESPONSE_SCHEMA,
                    }
                },
            }
            if temperature is not None:
                request["temperature"] = temperature

            response = client.responses.create(**request)
            text = _extract_response_text(response)
            parsed = _parse_response(text)
            result = _safe_result(parsed)
            result["evaluation_failed"] = False
            return result
        except ImportError as e:
            logger.error("OpenAI SDK の読み込みに失敗しました: %s", e)
            break
        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
            logger.warning("LLM応答パース失敗 (attempt %d/%d): %s", attempt + 1, max_retries, e)
            continue
        except APIError as e:
            logger.error("OpenAI API エラー (attempt %d/%d): %s", attempt + 1, max_retries, e)
            continue
        except Exception as e:
            logger.error("予期しないエラー (attempt %d/%d): %s: %s", attempt + 1, max_retries, type(e).__name__, e)
            continue

    logger.error("全リトライ失敗。デフォルト値を返します。")
    return _default_result()
