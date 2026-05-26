from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlmodel import Session

REPO_ROOT = Path(__file__).resolve().parents[2]
RETRAIN_SCRIPT_PATH = REPO_ROOT / "scripts" / "retrain_loras.py"
ROLLBACK_SCRIPT_PATH = REPO_ROOT / "scripts" / "rollback_lora.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def retrain(tmp_path, monkeypatch):
    import app.store.db as db

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'feedback.db'}")
    db.reset_engine()
    db.init_db()
    yield _load_module("retrain_loras", RETRAIN_SCRIPT_PATH)
    db.reset_engine()


@pytest.fixture()
def rollback_module():
    return _load_module("rollback_lora", ROLLBACK_SCRIPT_PATH)


def _add_pairs(org_id: str, n: int, *, consent: bool = True, winner: str | None = None) -> None:
    import app.store.db as db
    from app.store.feedback_models import Organization, PairwiseFeedback

    with Session(db.get_engine()) as session:
        if session.get(Organization, org_id) is None:
            session.add(Organization(org_id=org_id, name=org_id, consent_to_train=consent))
        for i in range(n):
            pair_winner = winner or ("A" if i % 2 == 0 else "B")
            session.add(
                PairwiseFeedback(
                    org_id=org_id,
                    meeting_id=f"m-{org_id}",
                    utt_a=f"{org_id}-a-{i}",
                    utt_b=f"{org_id}-b-{i}",
                    winner=pair_winner,
                    source="manual_pair",
                )
            )
        session.commit()


def _mock_trainer(plan, dataset_path, adapter_dir, base_model, dataset_hash):
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base_model}) + "\n",
        encoding="utf-8",
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"mock\n")
    (adapter_dir / "training_meta.json").write_text(
        json.dumps(
            {
                "org_id": plan.org_id,
                "dataset_path": str(dataset_path),
                "dataset_hash": dataset_hash,
                "base_model": base_model,
                "pair_count": plan.pair_count,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return adapter_dir


def test_under_300_pairs_does_not_start_training(retrain, tmp_path):
    _add_pairs("org_small", 299)
    calls = []

    def trainer(*args):
        calls.append(args)
        return _mock_trainer(*args)

    results = retrain.run_retraining(
        adapters_root=tmp_path / "adapters",
        registry_path=tmp_path / "adapters" / "registry.yaml",
        data_root=tmp_path / "data",
        trainer=trainer,
        evaluator=lambda *_: {"pairwise_acc": 0.8},
    )

    assert calls == []
    assert results == [{"org_id": "org_small", "status": "skipped", "pair_count": 299}]


def test_tie_pairs_do_not_count_toward_training_threshold(retrain, tmp_path):
    _add_pairs("org_tie", 299)
    _add_pairs("org_tie", 20, winner="tie")
    calls = []

    results = retrain.run_retraining(
        adapters_root=tmp_path / "adapters",
        registry_path=tmp_path / "adapters" / "registry.yaml",
        data_root=tmp_path / "data",
        trainer=lambda *args: calls.append(args) or _mock_trainer(*args),
        evaluator=lambda *_: {"pairwise_acc": 0.8},
    )

    assert calls == []
    assert results == [{"org_id": "org_tie", "status": "skipped", "pair_count": 299}]


def test_eval_gate_holds_registry_update_on_regression(retrain, tmp_path):
    _add_pairs("org_gate", 400)
    registry_path = tmp_path / "adapters" / "registry.yaml"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        """
organizations:
  org_gate:
    active_version: v1
    adapter_path: old/path
    created_at: "2026-05-10T00:00:00+00:00"
    pair_count: 300
    eval_metrics:
      pairwise_acc: 0.90
""".lstrip(),
        encoding="utf-8",
    )

    results = retrain.run_retraining(
        adapters_root=tmp_path / "adapters",
        registry_path=registry_path,
        data_root=tmp_path / "data",
        trainer=_mock_trainer,
        evaluator=lambda *_: {"pairwise_acc": 0.88},
        at=datetime(2026, 5, 26, tzinfo=UTC),
    )

    assert results == [{"org_id": "org_gate", "status": "held", "version": "v2"}]
    registry = retrain.load_registry(registry_path)
    assert registry["organizations"]["org_gate"]["active_version"] == "v1"
    assert registry["organizations"]["org_gate"]["held_candidate"]["version"] == "v2"
    hold_path = tmp_path / "adapters" / "org_gate" / "v2" / "deployment_hold.json"
    assert json.loads(hold_path.read_text(encoding="utf-8"))["status"] == "held_for_review"


def test_missing_eval_metrics_hold_fail_closed(retrain, tmp_path):
    _add_pairs("org_no_eval", 300)

    results = retrain.run_retraining(
        adapters_root=tmp_path / "adapters",
        registry_path=tmp_path / "adapters" / "registry.yaml",
        data_root=tmp_path / "data",
        trainer=_mock_trainer,
        evaluator=lambda *_: {"source": "missing_eval_metrics"},
    )

    assert results == [{"org_id": "org_no_eval", "status": "held", "version": "v1"}]
    registry = retrain.load_registry(tmp_path / "adapters" / "registry.yaml")
    assert registry["organizations"]["org_no_eval"]["held_candidate"]["reason"]


def test_held_candidate_is_not_retrained_until_new_pairs_arrive(retrain, tmp_path):
    _add_pairs("org_held", 300)
    registry_path = tmp_path / "adapters" / "registry.yaml"

    retrain.run_retraining(
        adapters_root=tmp_path / "adapters",
        registry_path=registry_path,
        data_root=tmp_path / "data",
        trainer=_mock_trainer,
        evaluator=lambda *_: {"source": "missing_eval_metrics"},
    )
    calls = []
    results = retrain.run_retraining(
        adapters_root=tmp_path / "adapters",
        registry_path=registry_path,
        data_root=tmp_path / "data",
        trainer=lambda *args: calls.append(args) or _mock_trainer(*args),
        evaluator=lambda *_: {"pairwise_acc": 0.9},
    )

    assert calls == []
    assert results == [{"org_id": "org_held", "status": "skipped", "pair_count": 300}]


def test_invalid_org_id_is_failed_without_path_write(retrain, tmp_path):
    _add_pairs("../bad", 300)

    results = retrain.run_retraining(
        adapters_root=tmp_path / "adapters",
        registry_path=tmp_path / "adapters" / "registry.yaml",
        data_root=tmp_path / "data",
        trainer=_mock_trainer,
        evaluator=lambda *_: {"pairwise_acc": 0.8},
    )

    assert results[0]["org_id"] == "../bad"
    assert results[0]["status"] == "failed"
    assert not (tmp_path / "bad").exists()


def test_org_failure_does_not_stop_later_orgs(retrain, tmp_path):
    _add_pairs("org_fail", 300)
    _add_pairs("org_ok", 300)

    def trainer(plan, *args):
        if plan.org_id == "org_fail":
            raise RuntimeError("boom")
        return _mock_trainer(plan, *args)

    results = retrain.run_retraining(
        adapters_root=tmp_path / "adapters",
        registry_path=tmp_path / "adapters" / "registry.yaml",
        data_root=tmp_path / "data",
        trainer=trainer,
        evaluator=lambda *_: {"pairwise_acc": 0.8},
    )

    assert [r["status"] for r in results] == ["failed", "deployed"]


def test_build_feedback_dataset_isolates_org_rows(retrain, tmp_path):
    import app.store.db as db

    _add_pairs("org_a", 3)
    _add_pairs("org_b", 2)

    with Session(db.get_engine()) as session:
        dataset_hash = retrain.build_feedback_dataset(session, "org_a", tmp_path / "org_a.jsonl")

    rows = [
        json.loads(line)
        for line in (tmp_path / "org_a.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert dataset_hash
    assert len(rows) == 3
    assert {row["meta"]["org_id"] for row in rows} == {"org_a"}
    assert all(row["chosen"].startswith("org_a-") for row in rows)


def test_incremental_runs_after_week_even_without_100_new_pairs(retrain, tmp_path):
    _add_pairs("org_weekly", 320)
    registry_path = tmp_path / "adapters" / "registry.yaml"
    registry_path.parent.mkdir(parents=True)
    old = datetime(2026, 5, 1, tzinfo=UTC)
    registry_path.write_text(
        f"""
organizations:
  org_weekly:
    active_version: v1
    adapter_path: old/path
    created_at: "{old.isoformat()}"
    pair_count: 300
    eval_metrics:
      pairwise_acc: 0.80
""".lstrip(),
        encoding="utf-8",
    )

    results = retrain.run_retraining(
        adapters_root=tmp_path / "adapters",
        registry_path=registry_path,
        data_root=tmp_path / "data",
        trainer=_mock_trainer,
        evaluator=lambda *_: {"pairwise_acc": 0.795},
        at=old + timedelta(days=8),
    )

    assert results == [{"org_id": "org_weekly", "status": "deployed", "version": "v2"}]
    registry = retrain.load_registry(registry_path)
    assert registry["organizations"]["org_weekly"]["active_version"] == "v2"


def test_rollback_restores_metadata_from_target_adapter(rollback_module, tmp_path):
    registry_path = tmp_path / "adapters" / "registry.yaml"
    adapter_dir = tmp_path / "adapters" / "org_rb" / "v1"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "training_meta.json").write_text(
        json.dumps(
            {
                "base_model": "model-v1",
                "dataset_hash": "hash-v1",
                "pair_count": 310,
                "eval_metrics": {"pairwise_acc": 0.77},
            }
        ),
        encoding="utf-8",
    )
    registry_path.write_text(
        """
organizations:
  org_rb:
    active_version: v2
    adapter_path: old/path
    pair_count: 500
    eval_metrics:
      pairwise_acc: 0.90
    held_candidate:
      version: v3
""".lstrip(),
        encoding="utf-8",
    )

    rollback_module.rollback(registry_path, tmp_path / "adapters", "org_rb", "v1")

    registry = rollback_module.load_registry(registry_path)
    entry = registry["organizations"]["org_rb"]
    assert entry["active_version"] == "v1"
    assert entry["eval_metrics"] == {"pairwise_acc": 0.77}
    assert entry["pair_count"] == 310
    assert entry["base_model"] == "model-v1"
    assert "held_candidate" not in entry
