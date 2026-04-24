"""API route definitions."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from pydantic import ValidationError

from app.ingest.loader import load_meeting_from_dict, load_meeting_from_file
from app.schemas.models import MeetingInput
from app.services.analysis import AnalysisEvaluationError, run_analysis

router = APIRouter()

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample_meetings"


@router.get("/samples")
async def list_samples():
    """サンプル会議データの一覧を返す。"""
    files = sorted(SAMPLE_DIR.glob("*.json"))
    return [{"filename": f.name, "path": str(f)} for f in files]


@router.get("/samples/{filename}")
async def get_sample(filename: str):
    """サンプル会議データの中身を返す。"""
    path = SAMPLE_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="サンプルが見つかりません。")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@router.post("/analyze")
async def analyze_meeting(request: Request):
    """会議データ JSON を受け取り、分析サマリーを返す。"""
    body = await request.json()
    try:
        meeting_data = load_meeting_from_dict(body)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"入力データのバリデーションエラー: {e}") from e
    return await _run_analysis_for_request(meeting_data, request)


@router.post("/analyze/file")
async def analyze_meeting_file(file: UploadFile, request: Request):
    """アップロードされた JSON ファイルを分析する。"""
    content = await file.read()
    try:
        data = json.loads(content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"JSONの読み込みに失敗しました: {e}") from e

    try:
        meeting = load_meeting_from_dict(data)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"入力データのバリデーションエラー: {e}") from e
    return await _run_analysis_for_request(meeting, request)


@router.post("/analyze/sample/{filename}")
async def analyze_sample(filename: str, request: Request):
    """指定されたサンプルデータを分析する。"""
    path = SAMPLE_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="サンプルが見つかりません。")

    meeting = load_meeting_from_file(path)
    return await _run_analysis_for_request(meeting, request)


async def _run_analysis_for_request(meeting_data: MeetingInput, request: Request) -> dict:
    """FastAPI state から設定を取り出し、サービス層へ渡す。"""
    config = request.app.state.config
    try:
        return await run_analysis(meeting_data, config)
    except AnalysisEvaluationError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
