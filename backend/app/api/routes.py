"""API ルート定義"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from pydantic import BaseModel, ValidationError

from app.aggregation.aggregator import build_meeting_summary
from app.context_builder.builder import build_contexts
from app.evaluators import create_evaluator
from app.ingest.loader import load_meeting_from_dict, load_meeting_from_file
from app.reporting.reporter import format_meeting_summary_for_ui
from app.schemas.models import EvaluatedUtterance, MeetingInput
from app.scoring.calculator import calculate_total_score
from app.scoring.rule_corrections import apply_rule_corrections
from app.store import repository
from app.store.models import SavedMeeting

logger = logging.getLogger(__name__)

router = APIRouter()

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "sample_meetings"


@router.get("/samples")
async def list_samples():
    """サンプル会議データの一覧を返す"""
    files = sorted(SAMPLE_DIR.glob("*.json"))
    return [{"filename": f.name, "path": str(f)} for f in files]


@router.get("/samples/{filename}")
async def get_sample(filename: str):
    """サンプル会議データの中身���返す"""
    path = SAMPLE_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="サンプルが見つかりません")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@router.post("/analyze")
async def analyze_meeting(request: Request):
    """会議データを受け取り、全発言を評価してサマリーを返す

    生の JSON dict を受け取り、ingest の正規化を通してから処理する。
    欠損フィールド (utterance_id, speaker, timestamp) があっても自動補完される。
    """
    body = await request.json()
    try:
        meeting_data = load_meeting_from_dict(body)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"入力データのバリデーションエラー: {e}") from e
    return await _run_analysis(meeting_data, request)


@router.post("/analyze/file")
async def analyze_meeting_file(file: UploadFile, request: Request):
    """JSONファイルをアップロードして分析する"""
    content = await file.read()
    try:
        data = json.loads(content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"JSONパースエラー: {e}") from e

    try:
        meeting = load_meeting_from_dict(data)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"入力データのバリデーションエラー: {e}") from e
    return await _run_analysis(meeting, request)


@router.post("/analyze/sample/{filename}")
async def analyze_sample(filename: str, request: Request):
    """サンプルデータを指定して分析する"""
    path = SAMPLE_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="サンプルが見つかりません")

    meeting = load_meeting_from_file(path)
    return await _run_analysis(meeting, request)


# ---------------------------------------------------------------------------
# 保存済み会議 CRUD
# ---------------------------------------------------------------------------

class SaveMeetingRequest(BaseModel):
    source_type: str
    input: dict
    result: dict


@router.get("/meetings")
async def list_meetings():
    """保存済み会議の一覧を返す"""
    return [m.model_dump() for m in repository.list_all()]


@router.get("/meetings/{meeting_id}")
async def get_meeting(meeting_id: str):
    """保存済み会議の詳細・分析結果を返す"""
    meeting = repository.get(meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="保存済み会議が見つかりません")
    return meeting.model_dump()


@router.post("/meetings", status_code=201)
async def save_meeting(body: SaveMeetingRequest):
    """分析結果を保存する"""
    result = body.result
    title = result.get("title", "（タイトルなし）")
    speaker_summaries = result.get("speaker_summaries", [])
    evaluated = result.get("evaluated_utterances", [])
    overall_score = (
        sum(u.get("total_score", 0) for u in evaluated) / len(evaluated)
        if evaluated
        else 0.0
    )

    meeting = SavedMeeting(
        id=repository.generate_id(),
        title=title,
        source_type=body.source_type,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        speaker_count=len(speaker_summaries),
        utterance_count=len(evaluated),
        overall_score=round(overall_score, 2),
        input=body.input,
        result=result,
    )
    saved = repository.save(meeting)
    return saved.model_dump()


@router.delete("/meetings/{meeting_id}", status_code=204)
async def delete_meeting(meeting_id: str):
    """保存済み会議を削除する"""
    deleted = repository.delete(meeting_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="保存済み会議が見つかりません")


# ---------------------------------------------------------------------------
# 分析パイプライン
# ---------------------------------------------------------------------------

async def _run_analysis(meeting_data: MeetingInput, request: Request) -> dict:
    """共通の分析パイプライン"""
    config = request.app.state.config

    # 文脈ウィンドウ生成
    contexts = build_contexts(
        meeting_data,
        before_count=config.context_before,
        after_count=config.context_after,
    )

    evaluated: list[EvaluatedUtterance] = []
    failed_count = 0

    # config.llm_backend に応じて OpenAI / Local Evaluator を生成 (Issue #12)
    evaluator = create_evaluator(config)

    for ctx in contexts:
        target = ctx.target_utterance

        # LLM 評価
        result = evaluator.evaluate(ctx)

        if result.evaluation_failed:
            failed_count += 1

        # 総合スコア計算
        total = calculate_total_score(
            result.scores,
            result.penalties,
            config.weights,
            config.penalty_weights,
        )

        evaluated.append(
            EvaluatedUtterance(
                utterance_id=target.utterance_id,
                speaker=target.speaker,
                timestamp=target.timestamp,
                text=target.text,
                speech_type=result.speech_type,
                scores=result.scores,
                penalties=result.penalties,
                total_score=total,
                reason=result.reason,
            )
        )

    # 全発言の評価が失敗した場合はエラーを返す
    total_count = len(contexts)
    if total_count > 0 and failed_count == total_count:
        raise HTTPException(
            status_code=502,
            detail="LLM による評価がすべて失敗しました。API キーの設定や API の状態を確認してください。",
        )

    # ルールベース補正（重複・冗長の追加補正）
    evaluated = apply_rule_corrections(evaluated, config.weights, config.penalty_weights)

    # 集計
    summary = build_meeting_summary(
        meeting_id=meeting_data.meeting_id,
        title=meeting_data.title,
        goal=meeting_data.goal,
        evaluated=evaluated,
        top_count=config.top_utterances_count,
        top_axis_count=config.top_per_axis_count,
    )

    return format_meeting_summary_for_ui(summary)
