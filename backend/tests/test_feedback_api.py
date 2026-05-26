"""フィードバック収集 API のテスト (Issue #78)

- 各エンドポイントの正常系
- 組織間でフィードバックが漏洩しないこと
- Top5 並べ替え -> ペアワイズ自動展開
- X-Org-Id と org_id 不一致の拒否
- stats の段階 (0/1/2) と次段階までの不足ペア数
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """テストごとに独立した SQLite DB を使う。

    DATABASE_URL を一時ファイルに向け、キャッシュ済み Engine を破棄する。
    TestClient の lifespan で init_db() が走りテーブルが作られる。
    """
    import app.store.db as db

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'feedback.db'}")
    db.reset_engine()
    yield
    db.reset_engine()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _headers(org_id: str) -> dict[str, str]:
    return {"X-Org-Id": org_id}


# ---------- 正常系 ----------


def test_post_pairwise_ok(client):
    body = {
        "org_id": "org_001",
        "meeting_id": "m042",
        "utt_a": "u003",
        "utt_b": "u015",
        "winner": "A",
        "source": "manual_pair",
    }
    res = client.post("/api/feedback/pairwise", json=body, headers=_headers("org_001"))
    assert res.status_code == 200
    data = res.json()
    assert data["id"]
    assert data["generated_pairs"] == 0


def test_post_axis_flag_ok(client):
    body = {
        "org_id": "org_001",
        "meeting_id": "m042",
        "utterance_id": "u003",
        "direction": "overrated",
        "axis": "issue_clarification",
        "comment": "根拠が薄い",
    }
    res = client.post("/api/feedback/axis_flag", json=body, headers=_headers("org_001"))
    assert res.status_code == 200
    assert res.json()["id"]


def test_post_topk_expands_to_pairwise(client):
    """Top入りした発言 × 入替で押し出された発言 の全組合せが pairwise になる。"""
    body = {
        "org_id": "org_001",
        "meeting_id": "m042",
        # u100, u101 が新規 Top入り / u003, u015 が押し出された
        "corrected_top5": ["u100", "u101", "u001", "u002", "u004"],
        "original_top5": ["u003", "u015", "u001", "u002", "u004"],
    }
    res = client.post("/api/feedback/topk", json=body, headers=_headers("org_001"))
    assert res.status_code == 200
    # newcomers(2) × dropouts(2) = 4 ペア
    assert res.json()["generated_pairs"] == 4

    stats = client.get(
        "/api/feedback/stats", params={"org_id": "org_001"}, headers=_headers("org_001")
    ).json()
    assert stats["n_topk"] == 1
    assert stats["n_pairwise"] == 4


def test_topk_no_change_generates_no_pairs(client):
    """並べ替えなし (同一メンバー・同一順) ならペアは生成されない。"""
    body = {
        "org_id": "org_001",
        "meeting_id": "m042",
        "corrected_top5": ["u001", "u002", "u003", "u004", "u005"],
        "original_top5": ["u001", "u002", "u003", "u004", "u005"],
    }
    res = client.post("/api/feedback/topk", json=body, headers=_headers("org_001"))
    assert res.json()["generated_pairs"] == 0


@pytest.mark.parametrize(
    ("corrected_top5", "original_top5"),
    [
        (["u001", "u002", "u003", "u004"], ["u001", "u002", "u003", "u004", "u005"]),
        (
            ["u001", "u002", "u003", "u004", "u005", "u006"],
            ["u001", "u002", "u003", "u004", "u005"],
        ),
        (["u001", "u001", "u003", "u004", "u005"], ["u001", "u002", "u003", "u004", "u005"]),
        (["u001", "u002", "u003", "u004", "u005"], ["u001", "u001", "u003", "u004", "u005"]),
    ],
)
def test_topk_rejects_non_five_or_duplicate_items(client, corrected_top5, original_top5):
    """Top5 訂正は各リストが5件固定かつ配列内ユニークでなければ拒否する。"""
    body = {
        "org_id": "org_001",
        "meeting_id": "m042",
        "corrected_top5": corrected_top5,
        "original_top5": original_top5,
    }
    res = client.post("/api/feedback/topk", json=body, headers=_headers("org_001"))
    assert res.status_code == 422


# ---------- 認可 (X-Org-Id) ----------


def test_pairwise_rejects_org_mismatch(client):
    body = {
        "org_id": "org_001",
        "meeting_id": "m042",
        "utt_a": "u003",
        "utt_b": "u015",
        "winner": "A",
        "source": "manual_pair",
    }
    # ヘッダが別組織 -> 403
    res = client.post("/api/feedback/pairwise", json=body, headers=_headers("org_999"))
    assert res.status_code == 403


def test_missing_org_header_rejected(client):
    body = {
        "org_id": "org_001",
        "meeting_id": "m042",
        "utt_a": "u003",
        "utt_b": "u015",
        "winner": "A",
        "source": "manual_pair",
    }
    res = client.post("/api/feedback/pairwise", json=body)  # ヘッダ無し
    assert res.status_code == 422


def test_invalid_winner_rejected(client):
    body = {
        "org_id": "org_001",
        "meeting_id": "m042",
        "utt_a": "u003",
        "utt_b": "u015",
        "winner": "X",  # Literal 外
        "source": "manual_pair",
    }
    res = client.post("/api/feedback/pairwise", json=body, headers=_headers("org_001"))
    assert res.status_code == 422


# ---------- 組織間漏洩なし ----------


def test_no_cross_org_leakage(client):
    """org_001 と org_002 が同時に書き込んでも互いの件数を参照できない。"""
    pair = {
        "meeting_id": "m1",
        "utt_a": "ua",
        "utt_b": "ub",
        "winner": "A",
        "source": "manual_pair",
    }
    # org_001 に 3 件
    for _ in range(3):
        client.post(
            "/api/feedback/pairwise",
            json={"org_id": "org_001", **pair},
            headers=_headers("org_001"),
        )
    # org_002 に 1 件
    client.post(
        "/api/feedback/pairwise",
        json={"org_id": "org_002", **pair},
        headers=_headers("org_002"),
    )

    stats1 = client.get(
        "/api/feedback/stats", params={"org_id": "org_001"}, headers=_headers("org_001")
    ).json()
    stats2 = client.get(
        "/api/feedback/stats", params={"org_id": "org_002"}, headers=_headers("org_002")
    ).json()

    assert stats1["n_pairwise"] == 3
    assert stats2["n_pairwise"] == 1


def test_stats_cannot_read_other_org(client):
    """他組織の stats はヘッダ不一致で拒否される。"""
    res = client.get(
        "/api/feedback/stats", params={"org_id": "org_001"}, headers=_headers("org_002")
    )
    assert res.status_code == 403


# ---------- stats 段階判定 ----------


def test_stats_initial_stage(client):
    stats = client.get(
        "/api/feedback/stats", params={"org_id": "org_new"}, headers=_headers("org_new")
    ).json()
    assert stats["stage"] == 0
    assert stats["next_stage"] == 1
    assert stats["pairs_to_next_stage"] == 50
    assert stats["n_pairwise"] == 0


def _bulk_add_pairs(org_id: str, n: int) -> None:
    """テスト用に pairwise を直接 N 件投入する (HTTP を 300 回叩かないため)。"""
    from sqlmodel import Session

    import app.store.db as db
    from app.store.feedback_models import Organization, PairwiseFeedback

    with Session(db.get_engine()) as s:
        if s.get(Organization, org_id) is None:
            s.add(Organization(org_id=org_id, name=org_id))
        for _ in range(n):
            s.add(
                PairwiseFeedback(
                    org_id=org_id,
                    meeting_id="m",
                    utt_a="a",
                    utt_b="b",
                    winner="A",
                    source="manual_pair",
                )
            )
        s.commit()


def test_stats_stage_1(client):
    """50 ペア到達で段階 1、次段階 (300) までの不足を返す。"""
    _bulk_add_pairs("org_s1", 50)
    stats = client.get(
        "/api/feedback/stats", params={"org_id": "org_s1"}, headers=_headers("org_s1")
    ).json()
    assert stats["stage"] == 1
    assert stats["next_stage"] == 2
    assert stats["pairs_to_next_stage"] == 250


def test_stats_stage_2(client):
    """300 ペア到達で段階 2、次段階は無し。"""
    _bulk_add_pairs("org_s2", 300)
    stats = client.get(
        "/api/feedback/stats", params={"org_id": "org_s2"}, headers=_headers("org_s2")
    ).json()
    assert stats["stage"] == 2
    assert stats["next_stage"] is None
    assert stats["pairs_to_next_stage"] is None
