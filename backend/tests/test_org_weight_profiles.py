from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml
from sqlmodel import Session

from app.scoring.weights import AppConfig, ScoringWeights
from app.scoring.weights_loader import load_penalty_weights, load_weights
from app.store import db, repository
from app.store.feedback_models import Organization, PairwiseFeedback
from app.store.models import SavedMeeting
from training.regress_weights import WeightRegressionResult

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "retrain_weights_per_org.py"
spec = importlib.util.spec_from_file_location("retrain_weights_per_org", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
retrain = importlib.util.module_from_spec(spec)
sys.modules["retrain_weights_per_org"] = retrain
spec.loader.exec_module(retrain)


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'feedback.db'}")
    monkeypatch.setattr(repository, "_STORE_DIR", tmp_path / "stored_meetings")
    db.reset_engine()
    db.init_db()
    yield
    db.reset_engine()


def _save_scores(
    meeting_id: str = "m",
    *,
    org_id: str = "org_a",
    saved_id: str = "saved",
    issue_score: int = 3,
) -> None:
    repository.save(
        SavedMeeting(
            id=saved_id,
            title="meeting",
            source_type="sample",
            created_at="2026-05-26T00:00:00+09:00",
            speaker_count=2,
            utterance_count=2,
            overall_score=50.0,
            input={"meeting_id": meeting_id, "org_id": org_id},
            result={
                "evaluated_utterances": [
                    {
                        "utterance_id": "a",
                        "scores": {"issue_clarification": issue_score, "decision_progress": 2},
                    },
                    {
                        "utterance_id": "b",
                        "scores": {"issue_clarification": 0, "decision_progress": 0},
                    },
                ]
            },
        )
    )


def _add_pairs(org_id: str, n: int, winner: str = "A") -> None:
    with Session(db.get_engine()) as session:
        session.add(Organization(org_id=org_id, name=org_id))
        for _ in range(n):
            session.add(
                PairwiseFeedback(
                    org_id=org_id,
                    meeting_id="m",
                    utt_a="a",
                    utt_b="b",
                    winner=winner,
                    source="manual_pair",
                )
            )
        session.commit()


def test_load_weights_prefers_org_profile(tmp_path: Path):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "org_001.yaml").write_text(
        yaml.safe_dump({"weights": {"issue_clarification": 2.2}}),
        encoding="utf-8",
    )

    config = AppConfig(
        weights=ScoringWeights(issue_clarification=1.0),
        meeting_type_weights={"decision": ScoringWeights(issue_clarification=1.5)},
    )

    assert (
        load_weights(
            config, org_id="org_001", meeting_type="decision", profile_dir=profile_dir
        ).issue_clarification
        == 2.2
    )
    assert (
        load_weights(
            config, org_id="missing", meeting_type="decision", profile_dir=profile_dir
        ).issue_clarification
        == 1.5
    )


def test_retrain_skips_org_with_less_than_50_pairs(tmp_path: Path):
    _save_scores(org_id="org_small")
    _add_pairs("org_small", 49)

    with Session(db.get_engine()) as session:
        outcome = retrain.retrain_org(
            session,
            "org_small",
            config=AppConfig(),
            profile_dir=tmp_path / "profiles",
        )

    assert outcome.status == "skipped"
    assert not (tmp_path / "profiles" / "org_small.yaml").exists()


def test_retrain_blocks_when_eval_gate_fails(tmp_path: Path, monkeypatch):
    _save_scores(org_id="org_bad")
    _add_pairs("org_bad", 50)

    def bad_regress(examples, *, base_weights=None, **_):
        examples = list(examples)
        weights = ScoringWeights(issue_clarification=0.05, decision_progress=0.05)
        return WeightRegressionResult(
            weights=weights,
            pairwise_acc=0.0,
            n_pairs=len(examples),
        )

    monkeypatch.setattr(retrain, "regress_weights", bad_regress)

    with Session(db.get_engine()) as session:
        outcome = retrain.retrain_org(
            session,
            "org_bad",
            config=AppConfig(),
            profile_dir=tmp_path / "profiles",
        )

    assert outcome.status == "blocked"
    assert not (tmp_path / "profiles" / "org_bad.yaml").exists()
    assert list((tmp_path / "profiles" / "_review" / "org_bad").glob("*.yaml"))


def test_retrain_updates_org_profiles_independently(tmp_path: Path):
    _save_scores(org_id="org_a", saved_id="saved_a")
    _save_scores(org_id="org_b", saved_id="saved_b")
    _add_pairs("org_a", 50, winner="A")
    _add_pairs("org_b", 50, winner="B")

    outcomes = retrain.retrain_all(profile_dir=tmp_path / "profiles")

    assert {o.org_id: o.status for o in outcomes} == {"org_a": "updated", "org_b": "updated"}
    assert (tmp_path / "profiles" / "org_a.yaml").exists()
    assert (tmp_path / "profiles" / "org_b.yaml").exists()
    assert list((tmp_path / "profiles" / "_history" / "org_a").glob("*.yaml"))
    assert list((tmp_path / "profiles" / "_history" / "org_b").glob("*.yaml"))


def test_load_penalty_weights_falls_back_to_config_for_missing_profile_fields(tmp_path: Path):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "org_001.yaml").write_text(
        yaml.safe_dump(
            {"weights": {"issue_clarification": 2.2}, "penalties": {"duplication": 3.0}}
        ),
        encoding="utf-8",
    )
    config = AppConfig()
    config.penalty_weights.verbosity = 0.25
    config.penalty_weights.override = 0.75

    penalties = load_penalty_weights(config, org_id="org_001", profile_dir=profile_dir)

    assert penalties.duplication == 3.0
    assert penalties.verbosity == 0.25
    assert penalties.override == 0.75


def test_dataset_join_is_scoped_by_org_id():
    _save_scores(org_id="org_a", saved_id="saved_a", issue_score=3)
    _save_scores(org_id="org_b", saved_id="saved_b", issue_score=1)
    _add_pairs("org_a", 1, winner="A")

    with Session(db.get_engine()) as session:
        pairs = retrain.feedback_repository.list_pairwise(session, "org_a")
        examples = retrain.build_feedback_dataset(pairs)

    assert len(examples) == 1
    assert examples[0].scores_a["issue_clarification"] == 3


def test_score_index_keeps_latest_duplicate_meeting_id():
    _save_scores(org_id="org_a", saved_id="analysis_202605260001", issue_score=1)
    _save_scores(org_id="org_a", saved_id="analysis_202605260002", issue_score=3)

    score_index = retrain._scores_by_meeting()

    assert score_index["org_a"]["m"]["a"]["issue_clarification"] == 3


def test_parse_generated_at_treats_invalid_value_as_missing(tmp_path: Path):
    profile = tmp_path / "profiles" / "org_a.yaml"
    profile.parent.mkdir()
    profile.write_text("generated_at: not-a-date\n", encoding="utf-8")

    assert retrain._parse_generated_at(profile) is None


def test_history_and_review_paths_sanitize_org_id(tmp_path: Path):
    payload = {
        "generated_at": "2026-05-26T00:00:00Z",
        "org_id": "../org/evil",
        "weights": {},
    }
    profile = tmp_path / "profiles" / ".._org_evil.yaml"

    retrain._write_profile_with_history(profile, payload)
    review_path = retrain._write_review(tmp_path / "profiles", "../org/evil", payload, "blocked")

    assert (tmp_path / "profiles" / "_history" / ".._org_evil").is_dir()
    assert review_path.parent == tmp_path / "profiles" / "_review" / ".._org_evil"
    assert not (tmp_path / "org").exists()


def test_write_yaml_is_atomic_for_target_path(tmp_path: Path):
    path = tmp_path / "profiles" / "org_a.yaml"

    retrain._write_yaml(path, {"generated_at": "2026-05-26T00:00:00Z"})

    assert (
        yaml.safe_load(path.read_text(encoding="utf-8"))["generated_at"] == "2026-05-26T00:00:00Z"
    )
    assert not list(path.parent.glob(".*.tmp"))
