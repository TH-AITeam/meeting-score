"""Meeting input loading and normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.schemas.models import MeetingInput


def load_meeting_from_file(path: str | Path) -> MeetingInput:
    """JSONファイルから会議データを読み込む。"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return _normalize(data)


def load_meeting_from_dict(data: dict[str, Any]) -> MeetingInput:
    """辞書から会議データを読み込む。"""
    return _normalize(data)


def _normalize(data: dict[str, Any]) -> MeetingInput:
    """入力データを補完し、MeetingInput として検証する。"""
    normalized = dict(data)
    normalized.setdefault("agenda", [])
    normalized.setdefault("decision_points", [])

    utterances = [dict(u) for u in normalized.get("utterances", [])]

    for i, utterance in enumerate(utterances):
        if not utterance.get("utterance_id"):
            utterance["utterance_id"] = f"u{i + 1:03d}"
        if not utterance.get("speaker"):
            utterance["speaker"] = f"Speaker {i + 1}"
        if not utterance.get("timestamp"):
            utterance["timestamp"] = "00:00:00"
        if not utterance.get("text"):
            utterance["text"] = ""

    normalized["utterances"] = utterances

    return MeetingInput(**normalized)
