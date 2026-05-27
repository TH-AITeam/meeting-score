"""scripts/update_weight_profiles.py のテスト (Issue #81)。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.update_weight_profiles import (  # noqa: E402
    SCORE_KEYS,
    render_profile_yaml,
    update_org_profile,
)


@pytest.fixture
def _isolated_db(tmp_path, monkeypatch):
    """テストごとに独立した SQLite フィードバック DB。"""
    import app.store.db as db

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'fb.db'}")
    db.reset_engine()
    db.init_db()
    yield db
    db.reset_engine()


def _seed_org(db, org_id: str, *, consent: bool, n_pairs: int) -> None:
    """org と n_pairs 件のペアワイズ (utt_hi が常に勝ち) を投入する。"""
    from sqlmodel import Session

    from app.store.feedback_models import Organization, PairwiseFeedback

    with Session(db.get_engine()) as s:
        s.add(Organization(org_id=org_id, name=org_id, consent_to_train=consent))
        for _ in range(n_pairs):
            s.add(
                PairwiseFeedback(
                    org_id=org_id,
                    meeting_id="m1",
                    utt_a="u_hi",
                    utt_b="u_lo",
                    winner="A",
                    source="manual_pair",
                )
            )
        s.commit()


def _seed_org_mixed_meetings(
    db, org_id: str, *, consent: bool, valid_pairs: int, missing_pairs: int
) -> None:
    """解決できるペアと保存済み会議が無いペアを混ぜて投入する。"""
    from sqlmodel import Session

    from app.store.feedback_models import Organization, PairwiseFeedback

    with Session(db.get_engine()) as s:
        s.add(Organization(org_id=org_id, name=org_id, consent_to_train=consent))
        for _ in range(valid_pairs):
            s.add(
                PairwiseFeedback(
                    org_id=org_id,
                    meeting_id="m1",
                    utt_a="u_hi",
                    utt_b="u_lo",
                    winner="A",
                    source="manual_pair",
                )
            )
        for _ in range(missing_pairs):
            s.add(
                PairwiseFeedback(
                    org_id=org_id,
                    meeting_id="missing",
                    utt_a="u_hi",
                    utt_b="u_lo",
                    winner="A",
                    source="manual_pair",
                )
            )
        s.commit()


def _meetings_dir(tmp_path) -> Path:
    """u_hi(高スコア) / u_lo(低スコア) を持つ保存済み会議を作る。"""
    d = tmp_path / "meetings"
    d.mkdir()
    import json

    def _scores(dp):
        return {
            "issue_clarification": 0,
            "decision_progress": dp,
            "risk_detection": 0,
            "actionability": 0,
            "groundedness": 0,
            "novelty": 0,
            "summarization": 0,
        }

    saved = {
        "id": "analysis_1",
        "result": {
            "meeting_id": "m1",
            "goal": "g",
            "evaluated_utterances": [
                {
                    "utterance_id": "u_hi",
                    "speaker": "A",
                    "timestamp": "00:01",
                    "text": "高",
                    "scores": _scores(3),
                    "penalties": {},
                    "reason": "r",
                },
                {
                    "utterance_id": "u_lo",
                    "speaker": "B",
                    "timestamp": "00:02",
                    "text": "低",
                    "scores": _scores(0),
                    "penalties": {},
                    "reason": "r",
                },
            ],
        },
    }
    (d / "m1.json").write_text(json.dumps(saved), encoding="utf-8")
    return d


def _common_kwargs(tmp_path):
    return dict(
        meetings_dir=_meetings_dir(tmp_path),
        out_root=tmp_path / "feedback",
        profile_dir=tmp_path / "profiles",
        iters=300,
    )


# ---------- 純関数 ----------


def test_render_profile_yaml_format():
    import numpy as np

    w = np.array([1.4, 1.6, 1.1, 1.3, 0.8, 0.85, 0.8])
    text = render_profile_yaml("org_001", w, n_pairs=142, acc_reg=0.74, acc_fixed=0.69)
    parsed = yaml.safe_load(text)
    assert parsed["n_pairs"] == 142
    assert set(parsed["weights"]) == set(SCORE_KEYS)
    assert parsed["weights"]["decision_progress"] == 1.6
    assert "penalties" in parsed and "eval" in parsed
    assert parsed["eval"]["pairwise_acc"] == 0.74


# ---------- update_org_profile ----------


def test_skips_below_threshold(_isolated_db, tmp_path):
    _seed_org(_isolated_db, "org_small", consent=True, n_pairs=10)
    r = update_org_profile("org_small", min_pairs=50, **_common_kwargs(tmp_path))
    assert r["status"] == "skipped_below_threshold"
    assert not (tmp_path / "profiles" / "org_small.yaml").exists()


def test_skips_no_consent(_isolated_db, tmp_path):
    _seed_org(_isolated_db, "org_nc", consent=False, n_pairs=100)
    r = update_org_profile("org_nc", min_pairs=50, **_common_kwargs(tmp_path))
    assert r["status"] == "skipped_no_consent"
    assert not (tmp_path / "profiles" / "org_nc.yaml").exists()


def test_updates_and_writes_profile(_isolated_db, tmp_path):
    _seed_org(_isolated_db, "org_ok", consent=True, n_pairs=60)
    r = update_org_profile("org_ok", min_pairs=50, **_common_kwargs(tmp_path))
    assert r["status"] == "updated"
    profile = tmp_path / "profiles" / "org_ok.yaml"
    assert profile.exists()
    parsed = yaml.safe_load(profile.read_text(encoding="utf-8"))
    assert set(parsed["weights"]) == set(SCORE_KEYS)
    assert parsed["n_pairs"] >= 1
    # 履歴も記録される
    assert (tmp_path / "profiles" / "history.jsonl").exists()


def test_skips_when_resolved_pairs_are_below_threshold(_isolated_db, tmp_path):
    """DB 行数が閾値以上でも、解決済みペアが閾値未満なら profile を書かない。"""
    _seed_org_mixed_meetings(
        _isolated_db, "org_sparse", consent=True, valid_pairs=49, missing_pairs=11
    )
    r = update_org_profile("org_sparse", min_pairs=50, **_common_kwargs(tmp_path))
    assert r["status"] == "skipped_below_resolved_threshold"
    assert r["n_pairs"] == 49
    assert r["n_feedback_pairs"] == 60
    assert not (tmp_path / "profiles" / "org_sparse.yaml").exists()


def test_invalid_org_id_is_rejected_before_path_use(_isolated_db, tmp_path):
    """../ を含む org_id は feedback/profile パスへ使う前に拒否する。"""
    with pytest.raises(ValueError, match="invalid org_id"):
        update_org_profile("../bad", min_pairs=1, **_common_kwargs(tmp_path))
    assert not (tmp_path / "feedback").exists()
    assert not (tmp_path / "profiles").exists()


def test_org_isolation_only_target_written(_isolated_db, tmp_path):
    _seed_org(_isolated_db, "org_a", consent=True, n_pairs=60)
    _seed_org(_isolated_db, "org_b", consent=True, n_pairs=60)
    update_org_profile("org_a", min_pairs=50, **_common_kwargs(tmp_path))
    assert (tmp_path / "profiles" / "org_a.yaml").exists()
    # org_a の更新で org_b のプロファイルは作られない
    assert not (tmp_path / "profiles" / "org_b.yaml").exists()


# ---------- 管理 API トリガ ----------


def test_admin_retrain_route(monkeypatch):
    """POST /api/admin/retrain_weights が update_org_profile を呼び結果を返す。"""
    import scripts.update_weight_profiles as upm
    from fastapi.testclient import TestClient

    from app.api.main import app

    monkeypatch.setattr(
        upm,
        "update_org_profile",
        lambda org_id, **_k: {"org_id": org_id, "status": "updated", "n_pairs": 60},
    )
    with TestClient(app) as c:
        res = c.post(
            "/api/admin/retrain_weights",
            params={"org_id": "org_x"},
            headers={"X-Org-Id": "org_x"},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["org_id"] == "org_x"
    assert body["status"] == "updated"


def test_admin_retrain_route_requires_matching_org_header(monkeypatch):
    """管理APIも暫定認可として X-Org-Id と対象 org_id の一致を求める。"""
    import scripts.update_weight_profiles as upm
    from fastapi.testclient import TestClient

    from app.api.main import app

    called = False

    def fake_update(_org_id, **_k):
        nonlocal called
        called = True
        return {"status": "updated"}

    monkeypatch.setattr(upm, "update_org_profile", fake_update)
    with TestClient(app) as c:
        res = c.post(
            "/api/admin/retrain_weights",
            params={"org_id": "org_x"},
            headers={"X-Org-Id": "org_y"},
        )
    assert res.status_code == 403
    assert called is False


def test_admin_retrain_route_returns_400_for_invalid_org(monkeypatch):
    """org_id の形式不正は 500 ではなく 400 として返す。"""
    import scripts.update_weight_profiles as upm
    from fastapi.testclient import TestClient

    from app.api.main import app

    def fake_update(_org_id, **_k):
        raise ValueError("invalid org_id for filesystem path")

    monkeypatch.setattr(upm, "update_org_profile", fake_update)
    with TestClient(app) as c:
        res = c.post(
            "/api/admin/retrain_weights",
            params={"org_id": "org_x"},
            headers={"X-Org-Id": "org_x"},
        )
    assert res.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
