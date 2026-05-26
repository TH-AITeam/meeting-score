"""フィードバックの永続化リポジトリ (Issue #78)

全関数が ``org_id`` を必須にし、組織をまたいだ参照・混入を防ぐ。
セッションは呼び出し側 (API 層の依存性注入) から渡す。
"""

from __future__ import annotations

from sqlalchemy import func
from sqlmodel import Session, select

from app.schemas.feedback import (
    FeedbackAxisFlag,
    FeedbackPairwise,
    FeedbackStats,
    FeedbackTopK,
)
from app.store.feedback_models import (
    AxisFlagFeedback,
    Organization,
    PairwiseFeedback,
    TopKFeedback,
)

# 段階昇格の閾値 (Epic #77)
STAGE1_MIN_PAIRS = 50  # 組織別重みプロファイル
STAGE2_MIN_PAIRS = 300  # 組織別 LoRA


def list_trainable_org_ids(session: Session) -> list[str]:
    """学習利用に同意している組織 ID を列挙する。"""
    stmt = select(Organization.org_id).where(Organization.consent_to_train == True)  # noqa: E712
    return list(session.exec(stmt).all())


def get_or_create_org(session: Session, org_id: str, name: str | None = None) -> Organization:
    """組織を取得。無ければ作成する (認証基盤未整備のためモック挙動)。

    外部キー制約 (feedback.org_id -> organizations.org_id) を満たすため、
    フィードバック書き込み前に必ず呼ぶ。``consent_to_train`` は既定 True。
    """
    org = session.get(Organization, org_id)
    if org is None:
        org = Organization(org_id=org_id, name=name or org_id)
        session.add(org)
        session.flush()
    return org


def add_pairwise(session: Session, data: FeedbackPairwise) -> PairwiseFeedback:
    get_or_create_org(session, data.org_id)
    row = PairwiseFeedback(
        org_id=data.org_id,
        meeting_id=data.meeting_id,
        utt_a=data.utt_a,
        utt_b=data.utt_b,
        winner=data.winner,
        source=data.source,
        annotator=data.annotator,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def derive_pairs_from_topk(
    corrected_top5: list[str], original_top5: list[str]
) -> list[tuple[str, str]]:
    """Top5 差分から「勝ち発言 vs 負け発言」のペアを生成する。

    Top 入りした発言 (corrected にあり original に無い) を、入替で押し出された
    発言 (original にあり corrected に無い) との全組合せで勝たせる。
    1 回の並べ替え操作から複数ペアを獲得できる (Issue #78 の意図)。

    純粋な順位入替 (メンバー不変) からのペア生成は行わない。必要なら学習データ
    正規化 (Issue #80) 側で TopKFeedback 行を読んで追加展開する。
    """
    original_set = set(original_top5)
    corrected_set = set(corrected_top5)
    newcomers = [u for u in corrected_top5 if u not in original_set]
    dropouts = [u for u in original_top5 if u not in corrected_set]
    return [(nc, do) for nc in newcomers for do in dropouts]


def add_topk(session: Session, data: FeedbackTopK) -> tuple[TopKFeedback, list[PairwiseFeedback]]:
    """Top5 訂正を保存し、差分から派生ペアワイズを ``source='top5_reorder'`` で展開する。"""
    get_or_create_org(session, data.org_id)
    topk = TopKFeedback(
        org_id=data.org_id,
        meeting_id=data.meeting_id,
        corrected_top5=data.corrected_top5,
        original_top5=data.original_top5,
        annotator=data.annotator,
    )
    session.add(topk)

    derived: list[PairwiseFeedback] = []
    for winner_utt, loser_utt in derive_pairs_from_topk(data.corrected_top5, data.original_top5):
        pair = PairwiseFeedback(
            org_id=data.org_id,
            meeting_id=data.meeting_id,
            utt_a=winner_utt,
            utt_b=loser_utt,
            winner="A",
            source="top5_reorder",
            annotator=data.annotator,
        )
        session.add(pair)
        derived.append(pair)

    session.commit()
    session.refresh(topk)
    for pair in derived:
        session.refresh(pair)
    return topk, derived


def add_axis_flag(session: Session, data: FeedbackAxisFlag) -> AxisFlagFeedback:
    get_or_create_org(session, data.org_id)
    row = AxisFlagFeedback(
        org_id=data.org_id,
        meeting_id=data.meeting_id,
        utterance_id=data.utterance_id,
        direction=data.direction,
        axis=data.axis,
        comment=data.comment,
        annotator=data.annotator,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


_FeedbackModel = type[PairwiseFeedback] | type[TopKFeedback] | type[AxisFlagFeedback]


def _count(session: Session, model: _FeedbackModel, org_id: str) -> int:
    stmt = select(func.count()).select_from(model).where(model.org_id == org_id)
    return int(session.exec(stmt).one())


def count_pairwise_since(session: Session, org_id: str, since: object | None = None) -> int:
    """組織の pairwise 件数を数える。since があれば差分件数。"""
    stmt = (
        select(func.count()).select_from(PairwiseFeedback).where(PairwiseFeedback.org_id == org_id)
    )
    if since is not None:
        stmt = stmt.where(PairwiseFeedback.created_at > since)
    return int(session.exec(stmt).one())


def list_pairwise(session: Session, org_id: str) -> list[PairwiseFeedback]:
    """組織の pairwise feedback を作成順に返す。"""
    stmt = (
        select(PairwiseFeedback)
        .where(PairwiseFeedback.org_id == org_id)
        .order_by(PairwiseFeedback.created_at)
    )
    return list(session.exec(stmt).all())


def get_stats(session: Session, org_id: str) -> FeedbackStats:
    """組織のフィードバック件数と段階 (0/1/2)、次段階までの不足ペア数を返す。"""
    n_pairwise = _count(session, PairwiseFeedback, org_id)
    n_topk = _count(session, TopKFeedback, org_id)
    n_axis_flag = _count(session, AxisFlagFeedback, org_id)

    if n_pairwise < STAGE1_MIN_PAIRS:
        stage = 0
        next_stage: int | None = 1
        pairs_to_next: int | None = STAGE1_MIN_PAIRS - n_pairwise
    elif n_pairwise < STAGE2_MIN_PAIRS:
        stage = 1
        next_stage = 2
        pairs_to_next = STAGE2_MIN_PAIRS - n_pairwise
    else:
        stage = 2
        next_stage = None
        pairs_to_next = None

    return FeedbackStats(
        org_id=org_id,
        n_pairwise=n_pairwise,
        n_topk=n_topk,
        n_axis_flag=n_axis_flag,
        stage=stage,
        next_stage=next_stage,
        pairs_to_next_stage=pairs_to_next,
    )
