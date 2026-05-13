"""動画→音声抽出のアップロード効率を計測する (Issue #68)。

`data/eval_video/meeting_*/video.*` をすべて走査し、`app.asr.media.
extract_audio_from_video` と同じパラメータ (libopus / mono / 16kHz / 32kbps)
で音声を抽出して以下を JSON に書き出す:

- 元動画サイズ / 抽出音声サイズ / 圧縮率
- 抽出にかかった wall-clock 時間
- 動画 duration (ffprobe があれば)
- 100Mbps / 10Mbps での仮想アップロード時間 (動画 vs 抽出後)

ブラウザの ffmpeg.wasm はネイティブ ffmpeg よりおよそ 2〜4 倍遅いので、
ここで得る抽出時間は **下限値の目安** として参照する。

使い方:
    python scripts/run_video_benchmark.py
    python scripts/run_video_benchmark.py --print-table
    python scripts/run_video_benchmark.py --eval-dir path/to/videos
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.asr.media import (  # noqa: E402  (sys.path 追加後の import)
    VIDEO_EXTENSIONS,
    MediaError,
    extract_audio_from_video,
)


def _probe_duration_sec(path: Path) -> float | None:
    """ffprobe が居れば動画長 (秒) を返す。失敗時は None。"""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def _format_mb(n_bytes: int) -> str:
    return f"{n_bytes / (1024 * 1024):.2f} MB"


def _upload_seconds(n_bytes: int, mbps: float) -> float:
    return n_bytes * 8 / (mbps * 1_000_000)


class BenchResult(TypedDict):
    video: str
    input_bytes: int
    input_mb: float
    output_bytes: int
    output_mb: float
    compression_ratio: float
    duration_sec: float | None
    extract_wall_sec: float
    rtf: float | None
    upload_sec_100mbps_video: float
    upload_sec_100mbps_audio: float
    upload_sec_10mbps_video: float
    upload_sec_10mbps_audio: float
    status: str
    error: str | None


def _find_videos(eval_dir: Path) -> list[Path]:
    videos: list[Path] = []
    for sub in sorted(eval_dir.glob("meeting_*")):
        if not sub.is_dir():
            continue
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(f)
    return videos


def main() -> int:
    parser = argparse.ArgumentParser(
        description="動画→音声抽出のアップロード効率ベンチ (Issue #68)"
    )
    parser.add_argument(
        "--eval-dir",
        default=str(REPO_ROOT / "data" / "eval_video"),
        help="評価動画のディレクトリ (既定: data/eval_video)",
    )
    parser.add_argument(
        "--report",
        default=str(REPO_ROOT / "reports" / "video_benchmarks" / "upload_efficiency.json"),
        help="結果 JSON の出力先 (既定: reports/video_benchmarks/upload_efficiency.json)",
    )
    parser.add_argument(
        "--print-table",
        action="store_true",
        help="標準出力にも表形式で表示する",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    videos = _find_videos(eval_dir)
    if not videos:
        print(f"ERROR: {eval_dir} に動画が見つかりませんでした。", file=sys.stderr)
        print("       data/eval_video/README.md の手順で配置してください。", file=sys.stderr)
        return 1

    results: list[BenchResult] = []
    for video in videos:
        in_bytes = video.stat().st_size
        duration = _probe_duration_sec(video)
        out_path = video.with_suffix(".bench.webm")
        try:
            t0 = time.perf_counter()
            extract_audio_from_video(video, out_path)
            extract_sec = time.perf_counter() - t0
            out_bytes = out_path.stat().st_size
            ratio = out_bytes / in_bytes if in_bytes else 0.0
            status = "ok"
            err: str | None = None
        except MediaError as e:
            extract_sec = 0.0
            out_bytes = 0
            ratio = 0.0
            status = "error"
            err = str(e)
        finally:
            with contextlib.suppress(OSError):
                out_path.unlink(missing_ok=True)

        results.append(
            {
                "video": str(video.relative_to(REPO_ROOT)),
                "input_bytes": in_bytes,
                "input_mb": round(in_bytes / (1024 * 1024), 3),
                "output_bytes": out_bytes,
                "output_mb": round(out_bytes / (1024 * 1024), 3),
                "compression_ratio": round(ratio, 4),
                "duration_sec": round(duration, 2) if duration is not None else None,
                "extract_wall_sec": round(extract_sec, 3),
                "rtf": (round(extract_sec / duration, 3) if duration else None),
                "upload_sec_100mbps_video": round(_upload_seconds(in_bytes, 100), 2),
                "upload_sec_100mbps_audio": round(_upload_seconds(out_bytes, 100), 2),
                "upload_sec_10mbps_video": round(_upload_seconds(in_bytes, 10), 2),
                "upload_sec_10mbps_audio": round(_upload_seconds(out_bytes, 10), 2),
                "status": status,
                "error": err,
            }
        )

    out_report = Path(args.report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {out_report.relative_to(REPO_ROOT)}")

    if args.print_table:
        print()
        print(
            f"{'video':<40} {'入力':>10} {'抽出後':>10} {'比率':>8} {'抽出時間':>10} {'10Mbps↑(動画)':>14} {'10Mbps↑(音声)':>14}"
        )
        print("-" * 110)
        for r in results:
            if r["status"] != "ok":
                print(f"{r['video']:<40} ERROR: {r['error']}")
                continue
            print(
                f"{r['video']:<40} "
                f"{_format_mb(r['input_bytes']):>10} "
                f"{_format_mb(r['output_bytes']):>10} "
                f"{r['compression_ratio'] * 100:>7.2f}% "
                f"{r['extract_wall_sec']:>9.2f}s "
                f"{r['upload_sec_10mbps_video']:>13.1f}s "
                f"{r['upload_sec_10mbps_audio']:>13.1f}s"
            )

    failed = [r for r in results if r["status"] != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
