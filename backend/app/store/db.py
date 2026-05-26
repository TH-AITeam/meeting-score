"""フィードバック DB の接続・セッション管理 (SQLModel)

接続先は環境変数 ``DATABASE_URL`` で切り替える:

- 未設定: ローカル SQLite ファイル ``data/feedback.db`` (開発・テスト用)
- 設定時: 例 ``postgresql+psycopg://user:pass@db:5432/meeting_score`` (Docker / 本番)

本番は Docker 管理の PostgreSQL を想定する。SQLModel(SQLAlchemy 2.0) が
方言差を吸収するため、接続 URL を差し替えるだけで両対応できる。
Postgres ドライバ (psycopg) は optional extra ``[postgres]`` で導入する。
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

# repo ルート直下の data/ に配置 (store/db.py -> app -> backend -> repo root)
_DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "feedback.db"

_engine: Engine | None = None


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection: object, _: object) -> None:
    """SQLite は既定で外部キー制約を無効化しているため接続ごとに有効化する。

    Postgres 等では ``sqlite3.Connection`` ではないので何もしない。
    """
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    _DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{_DEFAULT_DB_PATH}"


def get_engine() -> Engine:
    """プロセス内で共有する Engine を遅延生成して返す。"""
    global _engine
    if _engine is None:
        url = _database_url()
        # SQLite は同一スレッド制約を緩める (FastAPI のスレッドプール対応)
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args)
    return _engine


def reset_engine() -> None:
    """キャッシュした Engine を破棄する (主にテストでの DB 切り替え用)。"""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


def init_db() -> None:
    """テーブルを作成する (存在すれば何もしない)。

    SQLite ローカル開発ではアプリ起動時にこれで十分。Postgres 本番運用に
    移行し、スキーマのバージョン管理が必要になった段階で Alembic を後付けする。
    """
    # テーブル定義を SQLModel.metadata に登録するため副作用 import が必要
    from app.store import feedback_models  # noqa: F401

    SQLModel.metadata.create_all(get_engine())


def get_session() -> Iterator[Session]:
    """FastAPI 依存性注入用のセッション供給。"""
    with Session(get_engine()) as session:
        yield session
