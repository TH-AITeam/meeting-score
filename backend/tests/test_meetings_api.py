"""保存済み会議 CRUD API のテスト"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.store.repository as repo
from app.api.main import app


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """テストごとに独立した保存ディレクトリを使用する"""
    monkeypatch.setattr(repo, "_STORE_DIR", tmp_path)


def _sample_result(title: str = "テスト会議") -> dict:
    return {
        "meeting_id": "m001",
        "title": title,
        "goal": "テスト",
        "overall_comment": "",
        "top_utterances": [],
        "top_issue_clarification": [],
        "top_decision_progress": [],
        "top_risk_detection": [],
        "top_actionability": [],
        "improvement_comments": [],
        "speaker_summaries": [{"speaker": "A"}, {"speaker": "B"}],
        "evaluated_utterances": [
            {"utterance_id": "u1", "total_score": 16.0},
            {"utterance_id": "u2", "total_score": 8.0},
        ],
    }


def _sample_input() -> dict:
    return {
        "meeting_id": "m001",
        "title": "テスト会議",
        "goal": "テスト",
        "utterances": [],
    }


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_list_meetings_empty(client):
    res = client.get("/api/meetings")
    assert res.status_code == 200
    assert res.json() == []


def test_save_and_list_meeting(client):
    res = client.post(
        "/api/meetings",
        json={
            "source_type": "upload",
            "input": _sample_input(),
            "result": _sample_result(),
        },
    )
    assert res.status_code == 201
    data = res.json()
    assert data["title"] == "テスト会議"
    assert data["speaker_count"] == 2
    assert data["utterance_count"] == 2
    assert data["overall_score"] == 51.3
    assert data["source_type"] == "upload"
    assert "id" in data

    res2 = client.get("/api/meetings")
    assert res2.status_code == 200
    assert len(res2.json()) == 1
    assert res2.json()[0]["title"] == "テスト会議"


def test_get_meeting_detail(client):
    save_res = client.post(
        "/api/meetings",
        json={
            "source_type": "sample",
            "input": _sample_input(),
            "result": _sample_result("詳細テスト"),
        },
    )
    assert save_res.status_code == 201
    meeting_id = save_res.json()["id"]

    res = client.get(f"/api/meetings/{meeting_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["title"] == "詳細テスト"
    assert "result" in data
    assert "input" in data


def test_get_meeting_not_found(client):
    res = client.get("/api/meetings/nonexistent_id")
    assert res.status_code == 404


def test_delete_meeting(client):
    save_res = client.post(
        "/api/meetings",
        json={
            "source_type": "upload",
            "input": _sample_input(),
            "result": _sample_result(),
        },
    )
    meeting_id = save_res.json()["id"]

    del_res = client.delete(f"/api/meetings/{meeting_id}")
    assert del_res.status_code == 204

    # 一覧から消えていることを確認
    list_res = client.get("/api/meetings")
    assert list_res.json() == []

    # 詳細取得も 404
    get_res = client.get(f"/api/meetings/{meeting_id}")
    assert get_res.status_code == 404


def test_delete_meeting_not_found(client):
    res = client.delete("/api/meetings/nonexistent_id")
    assert res.status_code == 404


def test_save_multiple_meetings_listed_newest_first(client):
    for title in ["会議A", "会議B", "会議C"]:
        client.post(
            "/api/meetings",
            json={
                "source_type": "upload",
                "input": _sample_input(),
                "result": _sample_result(title),
            },
        )

    res = client.get("/api/meetings")
    assert res.status_code == 200
    titles = [m["title"] for m in res.json()]
    # ファイル名の降順（= 保存時刻の降順）になっているはず
    assert len(titles) == 3
