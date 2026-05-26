#!/usr/bin/env python3
"""組織別 LoRA の active version をロールバックする (Issue #82)."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_registry(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"registry not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("organizations", {})
    return data


def rollback(registry_path: Path, adapters_root: Path, org_id: str, version: str) -> None:
    registry = load_registry(registry_path)
    orgs = registry.setdefault("organizations", {})
    if org_id not in orgs:
        raise ValueError(f"org not found in registry: {org_id}")
    adapter_path = adapters_root / org_id / version
    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter version not found: {adapter_path}")
    orgs[org_id]["active_version"] = version
    orgs[org_id]["adapter_path"] = str(adapter_path)
    orgs[org_id]["rolled_back_at"] = datetime.now(tz=UTC).isoformat()
    registry_path.write_text(
        yaml.safe_dump(registry, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA adapter registry を任意バージョンへ戻す")
    p.add_argument("--org", required=True)
    p.add_argument("--to", required=True, dest="version")
    p.add_argument("--adapters-root", default=str(REPO_ROOT / "adapters"))
    p.add_argument("--registry", default=str(REPO_ROOT / "adapters" / "registry.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rollback(Path(args.registry), Path(args.adapters_root), args.org, args.version)


if __name__ == "__main__":
    main()
