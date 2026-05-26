#!/usr/bin/env python3
"""組織別重みプロファイルを履歴から復元する。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = REPO_ROOT / "config" / "weights_profile"


def rollback(org_id: str, timestamp: str, profile_dir: Path = PROFILE_DIR) -> Path:
    safe_org_id = org_id.replace("/", "_").replace("\\", "_")
    history = profile_dir / "_history" / safe_org_id / f"{timestamp}.yaml"
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
