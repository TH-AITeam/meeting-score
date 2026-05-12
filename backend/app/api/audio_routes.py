"""音声入力エンドポイント (Issue #11)。

`POST /upload_audio` で音声ファイル (multipart) を受け取り、
WhisperX + pyannote + 音量分析 + LLM メタ抽出を通して MeetingInput JSON を返す。

長時間処理 (30 分音声で 5〜10 分) を**同期で**実行する点に注意。非同期ジョブ化は
別 Issue で扱う。本エンドポイントは内部運用 / 開発検証用。
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path

import yaml
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.asr.cli import transcribe_to_meeting_input

_UPLOAD_FILE_DEFAULT = File(...)
_FORM_MEETING_ID_DEFAULT = Form(None)
_FORM_TITLE_DEFAULT = Form("")
_FORM_GOAL_DEFAULT = Form("")
_FORM_NUM_SPEAKERS_DEFAULT = Form(None)
_FORM_NO_META_DEFAULT = Form(False)

logger = logging.getLogger(__name__)

router = APIRouter()


_ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


@router.post("/upload_audio")
async def upload_audio(
    request: Request,
    file: UploadFile = _UPLOAD_FILE_DEFAULT,
    meeting_id: str | None = _FORM_MEETING_ID_DEFAULT,
    title: str = _FORM_TITLE_DEFAULT,
    goal: str = _FORM_GOAL_DEFAULT,
    num_speakers: int | None = _FORM_NUM_SPEAKERS_DEFAULT,
    no_meta_extract: bool = _FORM_NO_META_DEFAULT,
) -> dict:
    """音声ファイル → MeetingInput JSON。

    Parameters
    ----------
    file : 音声ファイル (wav / mp3 / m4a / flac / ogg)
    meeting_id : 任意。省略時は UUID で生成
    title / goal : LLM 抽出に失敗した時の default 値
    num_speakers : 話者数の強制指定 (None=自動推定)
    no_meta_extract : True なら LLM メタ抽出をスキップ (動作確認用)
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"サポート外の拡張子: {suffix}。許容: {sorted(_ALLOWED_EXTENSIONS)}",
        )

    mid = meeting_id or f"m_{uuid.uuid4().hex[:8]}"
    cfg = request.app.state.config

    # multipart の bytes を一時ファイルに書き出して WhisperX/pyannote に渡す
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        content = await file.read()
        tmp_path.write_bytes(content)

    try:
        audio_section = _load_audio_section_from_app_state(request)
        logger.info(
            "Upload received: filename=%s size=%d bytes meeting_id=%s",
            file.filename,
            len(content),
            mid,
        )
        mi = transcribe_to_meeting_input(
            tmp_path,
            meeting_id=mid,
            cfg=cfg,
            audio_cfg=audio_section,
            num_speakers=num_speakers,
            use_meta_extractor=not no_meta_extract,
            use_volume_analyzer=audio_section.get("volume", {}).get("enabled", True),
            default_title=title,
            default_goal=goal,
        )
    except Exception as e:
        logger.exception("音声処理に失敗")
        raise HTTPException(status_code=500, detail=f"音声処理に失敗しました: {e}") from e
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("tmp file 削除失敗: %s", tmp_path)

    return mi.model_dump()


def _load_audio_section_from_app_state(request: Request) -> dict:
    """`app.state.config_path` があれば直接読む。無ければ既定パスを試す。"""
    config_path = getattr(request.app.state, "config_path", None)
    path = Path(config_path) if config_path else _default_config_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("audio") or {}


def _default_config_path() -> Path:
    """`load_config` と同じ既定値を解決する。"""
    return Path(__file__).resolve().parent.parent.parent / "config.yaml"


__all__ = ["router", "upload_audio"]
