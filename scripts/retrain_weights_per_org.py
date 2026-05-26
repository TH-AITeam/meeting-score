#!/usr/bin/env python3
"""組織別の軸重みプロファイルを更新するバッチ。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml
from sqlmodel import Session

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.scoring.weights import AppConfig, ScoringWeights, load_config  # noqa: E402
from app.scoring.weights_loader import (  # noqa: E402
    load_org_profile,
    penalties_to_dict,
    profile_path,
    sanitize_org_id,
    weights_to_dict,
)
from app.store import db, feedback_repository, repository  # noqa: E402
from app.store.feedback_models import PairwiseFeedback  # noqa: E402
from training.regress_weights import (  # noqa: E402
    PairwiseTrainingExample,
    pairwise_accuracy,
    regress_weights,
)

logger = logging.getLogger(__name__)
MIN_PAIRS = 50
GATE_DROP = 0.02
LARGE_SHIFT_THRESHOLD = 0.4


@dataclass(frozen=True)
class RetrainOutcome:
    org_id: str
    status: str
    n_pairs: int
    pairwise_acc: float | None = None
    baseline_acc: float | None = None
    path: str | None = None
    reason: str | None = None


def _parse_generated_at(path: Path) -> datetime | None:
    if not path.exists():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    value = raw.get("generated_at")
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        logger.warning("invalid generated_at in %s: %r", path, value)
        return None


ScoreIndex = dict[str, dict[str, dict[str, dict[str, float]]]]


def _scores_by_meeting() -> ScoreIndex:
    meetings: ScoreIndex = {}
    for meta in repository.list_all():
        saved = repository.get(meta.id)
        if saved is None:
            continue
        org_id = saved.input.get("org_id")
        if not org_id:
            continue
        meeting_id = saved.input.get("meeting_id") or meta.id
        utterances = saved.result.get("evaluated_utterances", [])
        scores = {
            u["utterance_id"]: {axis: float(value) for axis, value in u.get("scores", {}).items()}
            for u in utterances
            if "utterance_id" in u
        }
        if scores:
            org_scores = meetings.setdefault(str(org_id), {})
            org_scores.setdefault(str(meeting_id), scores)
    return meetings


def build_feedback_dataset(
    pairs: list[PairwiseFeedback],
    score_index: ScoreIndex | None = None,
) -> list[PairwiseTrainingExample]:
    """DB の pairwise feedback と保存済みスコアを結合する。"""
    score_index = score_index or _scores_by_meeting()
    examples: list[PairwiseTrainingExample] = []
    for pair in pairs:
        meeting_scores = score_index.get(pair.org_id, {}).get(pair.meeting_id)
        if not meeting_scores:
            continue
        scores_a = meeting_scores.get(pair.utt_a)
        scores_b = meeting_scores.get(pair.utt_b)
        if scores_a is None or scores_b is None:
            continue
        examples.append(
            PairwiseTrainingExample(scores_a=scores_a, scores_b=scores_b, winner=pair.winner)
        )
    return examples


def _profile_payload(
    *,
    org_id: str,
    config: AppConfig,
    weights: ScoringWeights,
    n_pairs: int,
    pairwise_acc_value: float,
    baseline_acc: float,
) -> dict:
    return {
        "generated_at": datetime.now(tz=UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "org_id": org_id,
        "n_pairs": n_pairs,
        "weights": weights_to_dict(weights),
        "penalties": penalties_to_dict(config.penalty_weights),
        "eval": {
            "pairwise_acc": round(pairwise_acc_value, 4),
            "baseline_pairwise_acc": round(baseline_acc, 4),
        },
    }


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def _write_profile_with_history(profile: Path, payload: dict) -> None:
    safe_org_id = sanitize_org_id(payload["org_id"])
    ts = payload["generated_at"].replace(":", "").replace("-", "")
    history = profile.parent / "_history" / safe_org_id / f"{ts}.yaml"
    _write_yaml(history, payload)
    _write_yaml(profile, payload)


def _write_review(profile_dir: Path, org_id: str, payload: dict, reason: str) -> Path:
    payload = {**payload, "status": "blocked", "reason": reason}
    ts = payload["generated_at"].replace(":", "").replace("-", "")
    path = profile_dir / "_review" / sanitize_org_id(org_id) / f"{ts}.yaml"
    _write_yaml(path, payload)
    return path


def _large_weight_shifts(base: ScoringWeights, learned: ScoringWeights) -> list[str]:
    shifts: list[str] = []
    for axis, before in asdict(base).items():
        after = getattr(learned, axis)
        if abs(float(after) - float(before)) >= LARGE_SHIFT_THRESHOLD:
            shifts.append(f"{axis}: {before:.2f} -> {after:.2f}")
    return shifts


def retrain_org(
    session: Session,
    org_id: str,
    *,
    config: AppConfig,
    profile_dir: Path,
    min_pairs: int = MIN_PAIRS,
) -> RetrainOutcome:
    profile = profile_path(org_id, profile_dir)
    since = _parse_generated_at(profile)
    diff_pairs = feedback_repository.count_pairwise_since(session, org_id, since)
    if diff_pairs < min_pairs:
        return RetrainOutcome(
            org_id=org_id, status="skipped", n_pairs=diff_pairs, reason="insufficient_pairs"
        )

    pairs = feedback_repository.list_pairwise(session, org_id)
    examples = build_feedback_dataset(pairs)
    if len(examples) < min_pairs:
        return RetrainOutcome(
            org_id=org_id, status="skipped", n_pairs=len(examples), reason="insufficient_dataset"
        )

    current_profile = load_org_profile(org_id, profile_dir)
    baseline_weights = current_profile[0] if current_profile is not None else config.weights
    baseline_acc = pairwise_accuracy(examples, baseline_weights)
    result = regress_weights(examples, base_weights=baseline_weights)
    payload = _profile_payload(
        org_id=org_id,
        config=config,
        weights=result.weights,
        n_pairs=result.n_pairs,
        pairwise_acc_value=result.pairwise_acc,
        baseline_acc=baseline_acc,
    )

    if result.pairwise_acc < baseline_acc - GATE_DROP:
        reason = "pairwise_acc_below_baseline"
        review_path = _write_review(profile_dir, org_id, payload, reason)
        return RetrainOutcome(
            org_id=org_id,
            status="blocked",
            n_pairs=result.n_pairs,
            pairwise_acc=result.pairwise_acc,
            baseline_acc=baseline_acc,
            path=str(review_path),
            reason=reason,
        )

    shifts = _large_weight_shifts(baseline_weights, result.weights)
    if shifts:
        logger.warning("large weight shift for %s: %s", org_id, ", ".join(shifts))

    _write_profile_with_history(profile, payload)
    return RetrainOutcome(
        org_id=org_id,
        status="updated",
        n_pairs=result.n_pairs,
        pairwise_acc=result.pairwise_acc,
        baseline_acc=baseline_acc,
        path=str(profile),
    )


def retrain_all(
    *,
    org_id: str | None = None,
    config_path: Path | None = None,
    profile_dir: Path | None = None,
) -> list[RetrainOutcome]:
    db.init_db()
    config = load_config(config_path)
    profile_dir = profile_dir or (REPO_ROOT / "config" / "weights_profile")
    with Session(db.get_engine()) as session:
        org_ids = [org_id] if org_id else feedback_repository.list_trainable_org_ids(session)
        return [
            retrain_org(session, oid, config=config, profile_dir=profile_dir) for oid in org_ids
        ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--org-id", help="特定組織だけ再学習する")
    parser.add_argument("--config", type=Path, default=BACKEND_DIR / "config.yaml")
    parser.add_argument(
        "--profile-dir", type=Path, default=REPO_ROOT / "config" / "weights_profile"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    outcomes = retrain_all(
        org_id=args.org_id, config_path=args.config, profile_dir=args.profile_dir
    )
    for outcome in outcomes:
        logger.info("%s", outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
