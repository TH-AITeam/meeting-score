"""組織別フィードバックの DB テーブル定義 (SQLModel / Issue #78)

`org_id` で行レベルに分離する。SQLite ローカル / PostgreSQL 本番の両対応のため、
UUID は ``str`` (UUIDv4 文字列)、JSONB は汎用 ``JSON`` 列、TIMESTAMPTZ は
``DateTime(timezone=True)`` で表現する。

学習データ抽出 (Issue #80) では ``consent_to_train=False`` の組織を除外する。
本テーブル群は同意に関わらず書き込みを保存し、除外は抽出側の責務とする。
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import JSON as SA_JSON
from sqlalchemy import CheckConstraint, Column, DateTime, Index
from sqlmodel import Field, SQLModel


def _uuid() -> str:
    return str(uuid4())


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _created_at_column() -> Column:
    # timezone-aware にして SQLite/Postgres どちらでも TZ 情報を保持する
    return Column(DateTime(timezone=True), nullable=False)


class Organization(SQLModel, table=True):
    """組織マスタ。``consent_to_train`` が学習利用可否を表す。"""

    __tablename__ = "organizations"

    org_id: str = Field(primary_key=True)
    name: str
    # 学習にフィードバックを利用してよいか。False でも書き込みは保存し、
    # 学習データ抽出 (Issue #80) 側で除外する。
    consent_to_train: bool = Field(default=True)
    created_at: datetime = Field(default_factory=_now, sa_column=_created_at_column())


class PairwiseFeedback(SQLModel, table=True):
    """ペアワイズ比較フィードバック。

    ``source='top5_reorder'`` の行は Top5 並べ替え (TopKFeedback) から
    サーバ側で自動生成される。``manual_pair`` は明示的な 1 対比較。
    """

    __tablename__ = "feedback_pairwise"
    __table_args__ = (
        CheckConstraint("winner IN ('A','B','tie')", name="ck_pairwise_winner"),
        CheckConstraint("source IN ('top5_reorder','manual_pair')", name="ck_pairwise_source"),
        Index("ix_pairwise_org", "org_id", "created_at"),
    )

    id: str = Field(default_factory=_uuid, primary_key=True)
    org_id: str = Field(foreign_key="organizations.org_id")
    meeting_id: str
    utt_a: str
    utt_b: str
    winner: str  # 'A' | 'B' | 'tie'
    source: str  # 'top5_reorder' | 'manual_pair'
    annotator: str | None = None
    created_at: datetime = Field(default_factory=_now, sa_column=_created_at_column())


class TopKFeedback(SQLModel, table=True):
    """Top5 訂正フィードバック (差分のみ保持)。

    保存時にサーバ側で ``corrected_top5`` と ``original_top5`` の差分から
    ペアワイズを自動生成し、``PairwiseFeedback(source='top5_reorder')`` に展開する。
    """

    __tablename__ = "feedback_topk"
    __table_args__ = (Index("ix_topk_org", "org_id", "created_at"),)

    id: str = Field(default_factory=_uuid, primary_key=True)
    org_id: str = Field(foreign_key="organizations.org_id")
    meeting_id: str
    corrected_top5: list[str] = Field(sa_column=Column(SA_JSON, nullable=False))
    original_top5: list[str] = Field(sa_column=Column(SA_JSON, nullable=False))
    annotator: str | None = None
    created_at: datetime = Field(default_factory=_now, sa_column=_created_at_column())


class AxisFlagFeedback(SQLModel, table=True):
    """発言カードの 👎 (D-2)。過大/過小評価と任意の該当軸。"""

    __tablename__ = "feedback_axis_flag"
    __table_args__ = (
        CheckConstraint("direction IN ('overrated','underrated')", name="ck_axis_direction"),
        Index("ix_axis_org", "org_id", "created_at"),
    )

    id: str = Field(default_factory=_uuid, primary_key=True)
    org_id: str = Field(foreign_key="organizations.org_id")
    meeting_id: str
    utterance_id: str
    direction: str  # 'overrated' | 'underrated'
    axis: str | None = None  # 任意。'issue_clarification' 等
    comment: str | None = None
    annotator: str | None = None
    created_at: datetime = Field(default_factory=_now, sa_column=_created_at_column())
