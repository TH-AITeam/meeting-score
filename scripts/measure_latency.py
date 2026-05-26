"""1 発言推論のレイテンシ計測 (Issue #18)。

vLLM の OpenAI 互換エンドポイントに対し、同一発言を N 回同期呼び出しして
p50 / p95 / p99 / 平均レイテンシを取る。

Usage:
    python scripts/measure_latency.py \
        --endpoint http://127.0.0.1:8001/v1 \
        --model qwen3.6-27b-awq \
        --sample data/sample_meetings/sample_meeting_01.json \
        --n 100 \
        --out reports/model_benchmarks/qwen3.6-27b-awq/latency.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.context_builder.builder import build_contexts  # noqa: E402
from app.evaluators.local_evaluator import LocalEvaluator  # noqa: E402
from app.ingest.loader import load_meeting_from_file  # noqa: E402


def _percentile(values: list[float], p: float) -> float:
    """シンプルな線形補間 percentile (p は 0〜100)。"""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def main() -> int:
    parser = argparse.ArgumentParser(description="1 発言推論のレイテンシ計測")
    parser.add_argument("--endpoint", required=True, help="OpenAI 互換エンドポイント")
    parser.add_argument("--model", required=True, help="vLLM の --served-model-name")
    parser.add_argument(
        "--sample", required=True, help="計測に使う会議 JSON（最初の発言を使う）"
    )
    parser.add_argument("--n", type=int, default=100, help="繰り返し回数")
    parser.add_argument("--api-key", default="dummy", help="OpenAI 互換 API キー")
    parser.add_argument("--out", help="JSON 出力先")
    args = parser.parse_args()

    meeting = load_meeting_from_file(Path(args.sample))
    contexts = build_contexts(meeting, before_count=3, after_count=3)
    if not contexts:
        msg = "会議に発言がありません"
        raise SystemExit(msg)
    ctx = contexts[0]

    evaluator = LocalEvaluator(
        model=args.model,
        endpoint=args.endpoint,
        api_key=args.api_key,
        max_tokens=1024,
        max_retries=1,
        timeout=60.0,
    )

    durations: list[float] = []
    failures = 0
    print("Warming up...")
    evaluator.evaluate(ctx)  # JIT / KV cache 暖気

    print(f"Measuring {args.n} iterations...")
    for i in range(args.n):
        t0 = time.perf_counter()
        result = evaluator.evaluate(ctx)
        dt = time.perf_counter() - t0
        if result.evaluation_failed:
            failures += 1
        durations.append(dt)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{args.n}  last={dt:.3f}s")

    durations_ms = [d * 1000 for d in durations]
    payload = {
        "endpoint": args.endpoint,
        "model": args.model,
        "n": args.n,
        "failures": failures,
        "success_rate": (args.n - failures) / args.n if args.n else 0.0,
        "latency_ms": {
            "p50": _percentile(durations_ms, 50),
            "p90": _percentile(durations_ms, 90),
            "p95": _percentile(durations_ms, 95),
            "p99": _percentile(durations_ms, 99),
            "mean": statistics.mean(durations_ms),
            "stdev": statistics.stdev(durations_ms) if len(durations_ms) > 1 else 0.0,
            "min": min(durations_ms),
            "max": max(durations_ms),
        },
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Wrote {out_path}")
    else:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")

    lat = payload["latency_ms"]
    print(
        f"\nResult: p50={lat['p50']:.0f}ms p95={lat['p95']:.0f}ms "
        f"mean={lat['mean']:.0f}ms success={payload['success_rate']:.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
