#!/usr/bin/env python3
"""組織別 LoRA 定期再学習バッチ (Issue #82)."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import yaml
from sqlalchemy import func
from sqlmodel import Session, select

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.scoring.weights import load_config  # noqa: E402
from app.store import db  # noqa: E402
from app.store.feedback_models import Organization, PairwiseFeedback  # noqa: E402
from app.store.feedback_repository import STAGE2_MIN_PAIRS  # noqa: E402

INITIAL_MIN_PAIRS = STAGE2_MIN_PAIRS
INCREMENTAL_MIN_NEW_PAIRS = 100
INCREMENTAL_MAX_AGE = timedelta(days=7)
EVAL_GATE_TOLERANCE = 0.01
ORG_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class OrgPlan:
    org_id: str
    next_version: str
    pair_count: int
    reason: str
    current_adapter: Path | None
    current_pairwise_acc: float | None


Trainer = Callable[[OrgPlan, Path, Path, str, str], Path]
Evaluator = Callable[[OrgPlan, Path, str], dict[str, Any]]


def now_utc() -> datetime:
    return datetime.now(tz=UTC)


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"organizations": {}}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("organizations", {})
    return data


@contextlib.contextmanager
def registry_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def save_registry(path: Path, registry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(registry, allow_unicode=True, sort_keys=True), encoding="utf-8")


def validate_org_id(org_id: str) -> None:
    if not ORG_ID_RE.fullmatch(org_id):
        raise ValueError(f"invalid org_id for filesystem path: {org_id!r}")


def safe_child_path(root: Path, *parts: str) -> Path:
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*parts).resolve()
    candidate.relative_to(resolved_root)
    return candidate


def org_entry(registry: dict[str, Any], org_id: str) -> dict[str, Any] | None:
    entry = registry.get("organizations", {}).get(org_id)
    return entry if isinstance(entry, dict) else None


def active_version(entry: dict[str, Any] | None) -> str | None:
    return entry.get("active_version") if entry else None


def version_number(version: str | None) -> int:
    if not version or not version.startswith("v"):
        return 0
    try:
        return int(version[1:])
    except ValueError:
        return 0


def count_pairs(session: Session, org_id: str) -> int:
    stmt = (
        select(func.count())
        .select_from(PairwiseFeedback)
        .where(PairwiseFeedback.org_id == org_id)
        .where(PairwiseFeedback.winner.in_(["A", "B"]))
    )
    return int(session.exec(stmt).one())


def list_training_orgs(session: Session) -> list[str]:
    stmt = select(Organization.org_id).where(Organization.consent_to_train.is_(True))
    return list(session.exec(stmt))


def latest_adapter_path(adapters_root: Path, org_id: str, entry: dict[str, Any] | None) -> Path | None:
    version = active_version(entry)
    if not version:
        return None
    return safe_child_path(adapters_root, org_id, version)


def current_pairwise_acc(entry: dict[str, Any] | None) -> float | None:
    if not entry:
        return None
    metrics = entry.get("eval_metrics") or {}
    value = metrics.get("pairwise_acc", metrics.get("pairwise_accuracy"))
    return float(value) if value is not None else None


def eligible_plan(
    *,
    org_id: str,
    pair_count: int,
    registry: dict[str, Any],
    adapters_root: Path,
    at: datetime,
) -> OrgPlan | None:
    entry = org_entry(registry, org_id)
    current_version = active_version(entry)
    next_version = f"v{version_number(current_version) + 1}"
    held = (entry or {}).get("held_candidate") or {}
    if held.get("version") == next_version and int(held.get("pair_count", -1)) == pair_count:
        return None

    if not current_version:
        if pair_count < INITIAL_MIN_PAIRS:
            return None
        return OrgPlan(
            org_id=org_id,
            next_version=next_version,
            pair_count=pair_count,
            reason="initial",
            current_adapter=None,
            current_pairwise_acc=None,
        )

    last_pair_count = int(entry.get("pair_count", 0))
    created_at = parse_dt(entry.get("created_at"))
    enough_new_pairs = pair_count - last_pair_count >= INCREMENTAL_MIN_NEW_PAIRS
    weekly_due = created_at is None or at - created_at >= INCREMENTAL_MAX_AGE
    if not (enough_new_pairs or weekly_due):
        return None

    reason = "new_pairs" if enough_new_pairs else "weekly"
    return OrgPlan(
        org_id=org_id,
        next_version=next_version,
        pair_count=pair_count,
        reason=reason,
        current_adapter=latest_adapter_path(adapters_root, org_id, entry),
        current_pairwise_acc=current_pairwise_acc(entry),
    )


def _dpo_row(pair: PairwiseFeedback) -> dict[str, Any] | None:
    if pair.winner == "tie":
        return None
    chosen = pair.utt_a if pair.winner == "A" else pair.utt_b
    rejected = pair.utt_b if pair.winner == "A" else pair.utt_a
    prompt = (
        "以下の会議発言を会議貢献度の観点で評価してください。"
        "同じ基準で、より有用な発言が高く評価されるようにしてください。"
    )
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "meta": {
            "org_id": pair.org_id,
            "pair_id": pair.id,
            "meeting_id": pair.meeting_id,
            "source": pair.source,
            "created_at": pair.created_at.isoformat(),
        },
    }


def build_feedback_dataset(session: Session, org_id: str, output_path: Path) -> str:
    stmt = (
        select(PairwiseFeedback)
        .where(PairwiseFeedback.org_id == org_id)
        .order_by(PairwiseFeedback.created_at, PairwiseFeedback.id)
    )
    rows = [_dpo_row(pair) for pair in session.exec(stmt)]
    rows = [row for row in rows if row is not None]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True)
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
            f.write(line + "\n")
    return digest.hexdigest()


def default_trainer(
    plan: OrgPlan, dataset_path: Path, adapter_dir: Path, base_model: str, dataset_hash: str
) -> Path:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "training" / "dpo_train_per_org.py"),
        "--org-id",
        plan.org_id,
        "--train-jsonl",
        str(dataset_path),
        "--output-dir",
        str(adapter_dir),
        "--model-id",
        base_model,
        "--dataset-hash",
        dataset_hash,
    ]
    if plan.current_adapter is not None:
        cmd.extend(["--init-adapter", str(plan.current_adapter)])
    subprocess.run(cmd, check=True)
    return adapter_dir


def default_evaluator(_: OrgPlan, adapter_dir: Path, __: str) -> dict[str, Any]:
    meta_path = adapter_dir / "training_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        metrics = meta.get("eval_metrics") or {}
        if metrics:
            return metrics
    return {"source": "missing_eval_metrics"}


def update_registry(
    registry: dict[str, Any],
    *,
    plan: OrgPlan,
    adapter_dir: Path,
    base_model: str,
    dataset_hash: str,
    eval_metrics: dict[str, Any],
    at: datetime,
) -> None:
    registry.setdefault("organizations", {})[plan.org_id] = {
        "active_version": plan.next_version,
        "adapter_path": str(adapter_dir),
        "created_at": at.isoformat(),
        "eval_metrics": eval_metrics,
        "base_model": base_model,
        "pair_count": plan.pair_count,
        "dataset_hash": dataset_hash,
        "reason": plan.reason,
    }


def hold_deployment(
    registry: dict[str, Any],
    adapter_dir: Path,
    *,
    plan: OrgPlan,
    eval_metrics: dict[str, Any],
    base_model: str,
    dataset_hash: str,
) -> None:
    meta = {
        "org_id": plan.org_id,
        "candidate_version": plan.next_version,
        "status": "held_for_review",
        "reason": "pairwise_acc_regression_or_missing",
        "current_pairwise_acc": plan.current_pairwise_acc,
        "candidate_eval_metrics": eval_metrics,
        "base_model": base_model,
        "dataset_hash": dataset_hash,
        "created_at": now_utc().isoformat(),
    }
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "deployment_hold.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    entry = registry.setdefault("organizations", {}).setdefault(plan.org_id, {})
    entry["held_candidate"] = {
        "version": plan.next_version,
        "adapter_path": str(adapter_dir),
        "created_at": meta["created_at"],
        "eval_metrics": eval_metrics,
        "base_model": base_model,
        "pair_count": plan.pair_count,
        "dataset_hash": dataset_hash,
        "reason": meta["reason"],
    }


def passes_eval_gate(plan: OrgPlan, eval_metrics: dict[str, Any]) -> bool:
    candidate = eval_metrics.get("pairwise_acc", eval_metrics.get("pairwise_accuracy"))
    if candidate is None:
        return False
    if plan.current_pairwise_acc is None:
        return True
    return float(candidate) >= plan.current_pairwise_acc - EVAL_GATE_TOLERANCE


def run_retraining(
    *,
    adapters_root: Path,
    registry_path: Path,
    data_root: Path,
    trainer: Trainer = default_trainer,
    evaluator: Evaluator = default_evaluator,
    at: datetime | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    at = at or now_utc()
    db.init_db()
    cfg = load_config()
    results: list[dict[str, Any]] = []

    with registry_lock(registry_path):
        registry = load_registry(registry_path)
        with Session(db.get_engine()) as session:
            for org_id in list_training_orgs(session):
                try:
                    validate_org_id(org_id)
                    pair_count = count_pairs(session, org_id)
                    plan = eligible_plan(
                        org_id=org_id,
                        pair_count=pair_count,
                        registry=registry,
                        adapters_root=adapters_root,
                        at=at,
                    )
                    if plan is None:
                        results.append({"org_id": org_id, "status": "skipped", "pair_count": pair_count})
                        continue
                    if dry_run:
                        results.append({"org_id": org_id, "status": "eligible", "plan": plan.reason})
                        continue

                    dataset_path = safe_child_path(data_root, org_id, f"{plan.next_version}.jsonl")
                    dataset_hash = build_feedback_dataset(session, org_id, dataset_path)
                    adapter_dir = safe_child_path(adapters_root, org_id, plan.next_version)
                    trained_dir = trainer(plan, dataset_path, adapter_dir, cfg.llm_model, dataset_hash)
                    eval_metrics = evaluator(plan, trained_dir, cfg.llm_model)

                    if not passes_eval_gate(plan, eval_metrics):
                        hold_deployment(
                            registry,
                            trained_dir,
                            plan=plan,
                            eval_metrics=eval_metrics,
                            base_model=cfg.llm_model,
                            dataset_hash=dataset_hash,
                        )
                        save_registry(registry_path, registry)
                        results.append({"org_id": org_id, "status": "held", "version": plan.next_version})
                        continue

                    update_registry(
                        registry,
                        plan=plan,
                        adapter_dir=trained_dir,
                        base_model=cfg.llm_model,
                        dataset_hash=dataset_hash,
                        eval_metrics=eval_metrics,
                        at=at,
                    )
                    save_registry(registry_path, registry)
                    results.append({"org_id": org_id, "status": "deployed", "version": plan.next_version})
                except Exception as exc:
                    results.append({"org_id": org_id, "status": "failed", "error": str(exc)})
                    continue
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="組織別 LoRA アダプタを順次再学習する")
    p.add_argument("--adapters-root", default=str(REPO_ROOT / "adapters"))
    p.add_argument("--registry", default=str(REPO_ROOT / "adapters" / "registry.yaml"))
    p.add_argument("--data-root", default=str(REPO_ROOT / "data" / "training" / "org_feedback"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results = run_retraining(
        adapters_root=Path(args.adapters_root),
        registry_path=Path(args.registry),
        data_root=Path(args.data_root),
        dry_run=args.dry_run,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
