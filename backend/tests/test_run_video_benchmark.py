"""scripts/run_video_benchmark.py のテスト。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_script_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "run_video_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_video_benchmark", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_find_videos_includes_webm(tmp_path: Path) -> None:
    """meeting_* 配下の .webm もベンチ対象動画として検出する。"""
    bench = _load_script_module()
    meeting_dir = tmp_path / "meeting_001"
    meeting_dir.mkdir()
    mp4 = meeting_dir / "meeting.mp4"
    webm = meeting_dir / "meeting.webm"
    note = meeting_dir / "note.txt"
    mp4.write_bytes(b"MP4")
    webm.write_bytes(b"WEBM")
    note.write_text("ignore me", encoding="utf-8")

    videos = bench._find_videos(tmp_path)

    assert videos == [mp4, webm]
