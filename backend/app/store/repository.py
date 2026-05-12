"""JSON ファイルベースの保存済み会議リポジトリ"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.store.models import SavedMeeting, SavedMeetingMeta

logger = logging.getLogger(__name__)

_STORE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "stored_meetings"


def _store_dir() -> Path:
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    return _STORE_DIR


def _path(meeting_id: str) -> Path:
    return _store_dir() / f"{meeting_id}.json"


def generate_id() -> str:
    now = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return f"analysis_{now}"


def save(meeting: SavedMeeting) -> SavedMeeting:
    path = _path(meeting.id)
    path.write_text(meeting.model_dump_json(indent=2), encoding="utf-8")
    return meeting


def list_all() -> list[SavedMeetingMeta]:
    results: list[SavedMeetingMeta] = []
    for path in sorted(_store_dir().glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            results.append(SavedMeetingMeta.model_validate(data))
        except Exception:
            logger.warning("保存ファイルの読み込みに失敗しました: %s", path)
    return results


def get(meeting_id: str) -> SavedMeeting | None:
    path = _path(meeting_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return SavedMeeting.model_validate(data)
    except Exception:
        logger.warning("保存ファイルの読み込みに失敗しました: %s", path)
        return None


def delete(meeting_id: str) -> bool:
    path = _path(meeting_id)
    if not path.exists():
        return False
    path.unlink()
    return True
