"""ASR/Diar 出力から MeetingInput JSON を組み立てる (Issue #11)。

入力: `app.asr.base.Utterance[]` (segmenter で結合済み)
出力: `app.schemas.models.MeetingInput` (既存パイプラインの受け取れる形)

メタ情報 (title / goal / agenda / decision_points) は会議全文書き起こしを
LLM に投げて 1 コールで抽出する (#18 で選定した判断モデルを使う)。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from string import Template
from typing import Any, Protocol

from app.asr.base import Utterance as AsrUtterance
from app.schemas.models import MeetingInput
from app.schemas.models import Utterance as SchemaUtterance

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "meta_extraction.txt"

_META_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "goal": {"type": "string"},
        "agenda": {"type": "array", "items": {"type": "string"}},
        "decision_points": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "goal", "agenda", "decision_points"],
    "additionalProperties": False,
}


class MetaExtractor(Protocol):
    """会議全文 → {title, goal, agenda, decision_points} を返すコントラクト。

    既定実装は OpenAI 互換 API (#18 の採用モデル) を使うが、テストでは
    任意の callable に差し替え可能。
    """

    def extract(self, transcript: str) -> dict[str, Any]: ...


def _format_timestamp(sec: float) -> str:
    """秒数を HH:MM:SS 形式に整形する。"""
    total = max(0, int(sec))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_transcript(utterances: list[AsrUtterance]) -> str:
    """LLM 抽出用の整形済みテキスト。`[HH:MM:SS] 話者: 発言` 形式。"""
    lines: list[str] = []
    for u in utterances:
        ts = _format_timestamp(u.start_sec)
        speaker = u.speaker
        lines.append(f"[{ts}] {speaker}: {u.text}")
    return "\n".join(lines)


def build_meeting_input(
    asr_utterances: list[AsrUtterance],
    meeting_id: str,
    meta_extractor: MetaExtractor | None = None,
    *,
    default_title: str = "",
    default_goal: str = "",
) -> MeetingInput:
    """ASR/Diar 出力 + LLM メタ抽出から MeetingInput を生成する。

    Parameters
    ----------
    asr_utterances : list[AsrUtterance]
        segmenter で結合済みの ASR 発言列
    meeting_id : str
        会議 ID (CLI から渡す、または UUID)
    meta_extractor : MetaExtractor | None
        メタ情報抽出器。None なら title=default_title / goal=default_goal /
        agenda=[] / decision_points=[] の空メタで構築 (LLM 呼び出しなし)
    """
    schema_utterances = [
        SchemaUtterance(
            utterance_id=u.utterance_id,
            speaker=u.speaker,
            timestamp=_format_timestamp(u.start_sec),
            text=u.text,
        )
        for u in asr_utterances
    ]

    if meta_extractor is None:
        meta = {
            "title": default_title,
            "goal": default_goal,
            "agenda": [],
            "decision_points": [],
        }
    else:
        transcript = _build_transcript(asr_utterances)
        try:
            meta = meta_extractor.extract(transcript)
        except Exception as e:
            logger.warning("メタ情報抽出に失敗、デフォルト値で構築: %s", e)
            meta = {
                "title": default_title,
                "goal": default_goal,
                "agenda": [],
                "decision_points": [],
            }

    return MeetingInput(
        meeting_id=meeting_id,
        title=str(meta.get("title", default_title) or default_title),
        goal=str(meta.get("goal", default_goal) or default_goal),
        agenda=list(meta.get("agenda", []) or []),
        decision_points=list(meta.get("decision_points", []) or []),
        utterances=schema_utterances,
    )


# --------------------------------------------------------------------------
# OpenAI 互換 API ベースのデフォルト MetaExtractor
# --------------------------------------------------------------------------


class OpenAIMetaExtractor:
    """OpenAI 互換エンドポイント (#18 で選定したローカルモデル等) で
    会議メタ情報を 1 コール抽出する MetaExtractor 実装。

    `app.scoring.weights.AppConfig` の `llm_*` 設定を使う想定。テストでは
    `client` を mock に差し替えて呼べる。
    """

    def __init__(
        self,
        *,
        model: str,
        endpoint: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._injected_client = client
        self._template: Template | None = None

    def _get_template(self) -> Template:
        if self._template is None:
            self._template = Template(_PROMPT_PATH.read_text(encoding="utf-8"))
        return self._template

    def _get_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        from openai import OpenAI

        kwargs: dict[str, Any] = {"timeout": self.timeout}
        if self.endpoint:
            kwargs["base_url"] = self.endpoint
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return OpenAI(**kwargs)

    def extract(self, transcript: str) -> dict[str, Any]:
        prompt = self._get_template().safe_substitute(transcript=transcript)
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "meeting_meta",
                    "strict": True,
                    "schema": _META_SCHEMA,
                },
            },
        )
        text = _extract_response_text(response)
        return _parse_json_relaxed(text)


def _extract_response_text(response: Any) -> str:
    """OpenAI / OpenAI 互換のレスポンスから本文文字列を取り出す。"""
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError) as e:  # pragma: no cover
        msg = f"LLM 応答の形式が想定外: {e}"
        raise ValueError(msg) from e


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json_relaxed(text: str) -> dict[str, Any]:
    """LLM が ```json ... ``` で包んでくる可能性に対応した寛容な JSON パース。"""
    s = text.strip()
    m = _JSON_FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()
    parsed: dict[str, Any] = json.loads(s)
    return parsed


__all__ = [
    "MetaExtractor",
    "OpenAIMetaExtractor",
    "build_meeting_input",
]
