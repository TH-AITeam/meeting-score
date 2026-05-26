"""フィードバック収集 API (Issue #78)

エンドポイント:
- POST /api/feedback/pairwise
- POST /api/feedback/topk
- POST /api/feedback/axis_flag
- GET  /api/feedback/stats?org_id=...

認可: 認証基盤 (ログイン・組織紐付け) は本 Epic の範囲外のため、暫定で
``X-Org-Id`` ヘッダがリクエストの ``org_id`` と一致することのみ検証する。
認証基盤が入ったら、ヘッダ検証をトークン由来の org_id 突合に差し替える。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlmodel import Session

from app.schemas.feedback import (
    FeedbackAck,
    FeedbackAxisFlag,
    FeedbackPairwise,
    FeedbackStats,
    FeedbackTopK,
)
from app.store import feedback_repository as repo
from app.store.db import get_session

router = APIRouter(prefix="/feedback", tags=["feedback"])


def _check_org(x_org_id: str, body_org_id: str) -> None:
    """リクエスト元 (X-Org-Id) と対象 org_id の一致を検証する。"""
    if x_org_id != body_org_id:
        raise HTTPException(
            status_code=403,
            detail="X-Org-Id がリクエストの org_id と一致しません",
        )


@router.post("/pairwise", response_model=FeedbackAck)
def post_pairwise(
    data: FeedbackPairwise,
    x_org_id: str = Header(..., alias="X-Org-Id"),
    session: Session = Depends(get_session),  # noqa: B008
) -> FeedbackAck:
    _check_org(x_org_id, data.org_id)
    row = repo.add_pairwise(session, data)
    return FeedbackAck(id=row.id)


@router.post("/topk", response_model=FeedbackAck)
def post_topk(
    data: FeedbackTopK,
    x_org_id: str = Header(..., alias="X-Org-Id"),
    session: Session = Depends(get_session),  # noqa: B008
) -> FeedbackAck:
    _check_org(x_org_id, data.org_id)
    topk, derived = repo.add_topk(session, data)
    return FeedbackAck(id=topk.id, generated_pairs=len(derived))


@router.post("/axis_flag", response_model=FeedbackAck)
def post_axis_flag(
    data: FeedbackAxisFlag,
    x_org_id: str = Header(..., alias="X-Org-Id"),
    session: Session = Depends(get_session),  # noqa: B008
) -> FeedbackAck:
    _check_org(x_org_id, data.org_id)
    row = repo.add_axis_flag(session, data)
    return FeedbackAck(id=row.id)


@router.get("/stats", response_model=FeedbackStats)
def get_stats(
    org_id: str = Query(...),
    x_org_id: str = Header(..., alias="X-Org-Id"),
    session: Session = Depends(get_session),  # noqa: B008
) -> FeedbackStats:
    _check_org(x_org_id, org_id)
    return repo.get_stats(session, org_id)
