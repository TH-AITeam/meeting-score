"""会議データの入力正規化モジュール"""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas.models import MeetingInput


def load_meeting_from_file(path: str | Path) -> MeetingInput:
    """JSONファイルから会議データを読み込む"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return _normalize(data)


def load_meeting_from_dict(data: dict) -> MeetingInput:
    """辞書から会議データを読み込む"""
    return _normalize(data)


def _normalize(data: dict) -> MeetingInput:
    """入力データの正規化

    - 欠損フィールドの補完
    - 発言IDの自動付番 (未設定時)
    - 入力順を時系列として保持（再ソートしない）
    """
    # agenda / decision_points がなければ空リスト
    data.setdefault("agenda", [])
    data.setdefault("decision_points", [])

    utterances = data.get("utterances", [])

    # 発言IDの自動付番と欠損フィールド補完
    for i, u in enumerate(utterances):
        if not u.get("utterance_id"):
            u["utterance_id"] = f"u{i + 1:03d}"
        if not u.get("speaker"):
            u["speaker"] = f"Speaker {i + 1}"
        if not u.get("timestamp"):
            u["timestamp"] = "00:00:00"
        if not u.get("text"):
            u["text"] = ""

    # 入力順 = 時系列順として信頼する。再ソートしない。
    data["utterances"] = utterances

    return MeetingInput(**data)
