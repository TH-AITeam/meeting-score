#!/usr/bin/env python3
"""組織別重みプロファイルを履歴から復元する。"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.scoring.weights_loader import sanitize_org_id  # noqa: E402

PROFILE_DIR = REPO_ROOT / "config" / "weights_profile"
_TIMESTAMP_RE = re.compile(r"[0-9TZ]+")


def _validate_timestamp(timestamp: str) -> str:
    if not _TIMESTAMP_RE.fullmatch(timestamp):
        raise ValueError("timestamp must match [0-9TZ]+")
    return timestamp


def rollback(org_id: str, timestamp: str, profile_dir: Path = PROFILE_DIR) -> Path:
    safe_org_id = sanitize_org_id(org_id)
    safe_timestamp = _validate_timestamp(timestamp)
    history = profile_dir / "_history" / safe_org_id / f"{safe_timestamp}.yaml"
    if not history.exists():
        raise FileNotFoundError(f"history not found: {history}")
    target = profile_dir / f"{safe_org_id}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(history, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", required=True, help="組織 ID")
    parser.add_argument("--to", required=True, help="履歴 YAML のタイムスタンプ部分")
    parser.add_argument("--profile-dir", type=Path, default=PROFILE_DIR)
    args = parser.parse_args()
    target = rollback(args.org, args.to, args.profile_dir)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
