"""音声入力エンドポイント (Issue #11, 拡張 #68)。

`POST /upload_audio` で音声ファイル (multipart) を受け取り、
WhisperX + pyannote + 音量分析 + LLM メタ抽出を通して MeetingInput JSON を返す。

Issue #68 の方針:
- 動画ファイル (.mp4/.mov/.mkv/.avi) は **415 で拒否** し、frontend での
  クライアント側音声抽出を促す。
- 受信音声は `normalize_to_wav` で 16kHz mono wav に正規化してから ASR に渡す。
- backend は「動画抽出担当」ではなく「音声正規化担当」。

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
from starlette.concurrency import run_in_threadpool

from app.asr.cli import transcribe_to_meeting_input
from app.asr.media import (
    AUDIO_EXTENSIONS_INCLUDING_WEBM,
    VIDEO_EXTENSIONS,
    MediaError,
    normalize_to_wav,
)

_UPLOAD_FILE_DEFAULT = File(...)
_FORM_MEETING_ID_DEFAULT = Form(None)
_FORM_TITLE_DEFAULT = Form("")
_FORM_GOAL_DEFAULT = Form("")
_FORM_NUM_SPEAKERS_DEFAULT = Form(None)
_FORM_NO_META_DEFAULT = Form(False)

logger = logging.getLogger(__name__)

router = APIRouter()


# Issue #68: webm/opus を受け付ける。frontend が ffmpeg.wasm で抽出した形式に対応
_ALLOWED_EXTENSIONS = AUDIO_EXTENSIONS_INCLUDING_WEBM
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
_UPLOAD_CHUNK_BYTES = 1024 * 1024


def _format_mb(n_bytes: int) -> str:
    return f"{n_bytes / (1024 * 1024):.0f}MB"


def _is_video_upload(suffix: str, content_type: str | None) -> bool:
    return suffix in VIDEO_EXTENSIONS or (content_type or "").lower().startswith("video/")


def _raise_upload_too_large() -> None:
    raise HTTPException(
        status_code=413,
        detail=f"アップロード上限を超えています。上限: {_format_mb(MAX_UPLOAD_BYTES)}",
    )


async def _write_upload_with_limit(file: UploadFile, upload_path: Path) -> int:
    total = 0
    with upload_path.open("wb") as f:
        while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                _raise_upload_too_large()
            f.write(chunk)
    return total


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
    content_type = file.content_type

    # Issue #68: 動画は backend で受け付けない。frontend で音声抽出してから送らせる
    if _is_video_upload(suffix, content_type):
        raise HTTPException(
            status_code=415,
            detail=(
                f"動画ファイル ({suffix}) は受け付けません。アップロード効率化のため、"
                "frontend で音声を抽出してから送信してください "
                "(ローカルからは `ffmpeg -i input.mp4 -vn -c:a libopus -b:a 32k "
                "-ac 1 -ar 16000 output.webm` 等)。"
            ),
        )
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"サポート外の拡張子: {suffix}。許容: {sorted(_ALLOWED_EXTENSIONS)}",
        )

    mid = meeting_id or f"m_{uuid.uuid4().hex[:8]}"
    cfg = request.app.state.config

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES:
                _raise_upload_too_large()
        except ValueError:
            logger.warning("invalid content-length header: %s", content_length)

    upload_path: Path | None = None
    normalized_path: Path | None = None
    try:
        # multipart の bytes を一時ファイルに書き出す。Content-Length が無い場合も
        # chunk ごとに上限を見て、ASR/正規化に進む前に 413 で止める。
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            upload_path = Path(tmp.name)
        upload_size = await _write_upload_with_limit(file, upload_path)

        # Issue #68: 受信音声を 16kHz mono wav に正規化してから ASR に渡す
        audio_section = _load_audio_section_from_app_state(request)
        logger.info(
            "Upload received: filename=%s size=%d bytes meeting_id=%s",
            file.filename,
            upload_size,
            mid,
        )
        # wav 以外は ffmpeg で正規化する。wav の場合も正規化を通して
        # sample_rate / channels を確実に Whisper 互換にする。
        normalized_path = upload_path.with_suffix(".normalized.wav")
        try:
            await run_in_threadpool(normalize_to_wav, upload_path, normalized_path)
        except MediaError as me:
            raise HTTPException(
                status_code=500,
                detail=f"音声の正規化に失敗しました (ffmpeg): {me}",
            ) from me

        mi = await run_in_threadpool(
            transcribe_to_meeting_input,
            normalized_path,
            meeting_id=mid,
            cfg=cfg,
            audio_cfg=audio_section,
            num_speakers=num_speakers,
            use_meta_extractor=not no_meta_extract,
            use_volume_analyzer=audio_section.get("volume", {}).get("enabled", True),
            default_title=title,
            default_goal=goal,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("音声処理に失敗")
        raise HTTPException(status_code=500, detail=f"音声処理に失敗しました: {e}") from e
    finally:
        for p in (upload_path, normalized_path):
            if p is None:
                continue
            try:
                p.unlink(missing_ok=True)
            except OSError:
                logger.warning("tmp file 削除失敗: %s", p)

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
