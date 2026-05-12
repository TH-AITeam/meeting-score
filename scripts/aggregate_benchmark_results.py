"""ベンチマーク JSON を集計して Markdown 表に整形する (Issue #18)。

`reports/model_benchmarks/{served_name}/{ts}_*.json` を読み、
各モデルの最新ランから

- ベースライン評価 (`{ts}.json`):
    macro Spearman / Kendall τ / Top5 Jaccard / Pairwise accuracy
- 安定性 (`{ts}_stability.json`):
    7軸 mean SD
- レイテンシ (`{ts}_latency.json`):
    p50 / p95 / 成功率

を抽出して、`docs/model_selection_v1.md` に貼り付け可能な Markdown 表を標準出力に出す。

Usage:
    python scripts/aggregate_benchmark_results.py \\
        --reports-dir reports/model_benchmarks \\
        --out reports/model_benchmarks/_summary.md

    # 標準出力に出すだけ
    python scripts/aggregate_benchmark_results.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# モデル名 (served_name) → ライセンス / 量子化 / 学習しやすさ の手書きメタ。
# 候補が増えたら model_candidates.md と同期して追記する。
MODEL_META: dict[str, dict[str, str]] = {
    "qwen3.6-27b-bnb": {
        "label": "Qwen3.6-27B",
        "quant": "BnB NF4 (on-the-fly)",
        "license": "Apache 2.0",
        "trainability": "★★★",
    },
    "qwen3-14b-bf16": {
        "label": "Qwen3-14B",
        "quant": "bf16",
        "license": "Apache 2.0",
        "trainability": "★★★",
    },
    "qwen2.5-32b-awq": {
        "label": "Qwen2.5-32B-Instruct-AWQ",
        "quant": "AWQ INT4",
        "license": "Apache 2.0",
        "trainability": "★★★",
    },
    "swallow-3.1-8b-bf16": {
        "label": "Swallow-Llama-3.1-8B-Instruct",
        "quant": "bf16",
        "license": "Llama 3.1 CL + Swallow",
        "trainability": "★★☆",
    },
    "phi-4-14b-bf16": {
        "label": "Phi-4-14B",
        "quant": "bf16",
        "license": "MIT",
        "trainability": "★★★",
    },
}


@dataclass
class ModelResult:
    served_name: str
    timestamp: str | None = None
    spearman: float | None = None
    kendall_tau: float | None = None
    top5_jaccard: float | None = None
    pairwise_acc: float | None = None
    mean_sd_axes: float | None = None
    json_success_rate: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    vram_gb: float | None = None
    raw_files: list[Path] = field(default_factory=list)


def _fmt(v: Any, spec: str = ".3f", na: str = "TBD") -> str:
    if v is None:
        return na
    if isinstance(v, float):
        return format(v, spec)
    return str(v)


def _load_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _latest_timestamp(dir_path: Path) -> str | None:
    """Pick the latest `{ts}.json` / `{ts}_*.json` timestamp prefix.

    File name 例: `20260512_143012.json`, `20260512_143012_stability.json`.
    """
    seen: set[str] = set()
    for f in dir_path.glob("*.json"):
        name = f.stem
        for suffix in ("_stability", "_latency"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        seen.add(name)
    return max(seen) if seen else None


def _collect(reports_dir: Path) -> list[ModelResult]:
    out: list[ModelResult] = []
    if not reports_dir.exists():
        return out
    for sub in sorted(p for p in reports_dir.iterdir() if p.is_dir()):
        ts = _latest_timestamp(sub)
        if ts is None:
            continue
        result = ModelResult(served_name=sub.name, timestamp=ts)

        eval_file = sub / f"{ts}.json"
        if eval_file.exists():
            data = _load_json(eval_file) or {}
            macro = data.get("macro", {})
            result.spearman = macro.get("spearman")
            result.kendall_tau = macro.get("kendall_tau")
            result.top5_jaccard = macro.get("top5_jaccard")
            result.pairwise_acc = macro.get("pairwise_accuracy")
            result.raw_files.append(eval_file)

        stab_file = sub / f"{ts}_stability.json"
        if stab_file.exists():
            data = _load_json(stab_file) or {}
            mean = data.get("mean_sd_per_axis", {}) or {}
            if mean:
                result.mean_sd_axes = sum(mean.values()) / len(mean)
            result.raw_files.append(stab_file)

        lat_file = sub / f"{ts}_latency.json"
        if lat_file.exists():
            data = _load_json(lat_file) or {}
            result.json_success_rate = data.get("success_rate")
            lat = data.get("latency_ms", {}) or {}
            result.p50_ms = lat.get("p50")
            result.p95_ms = lat.get("p95")
            result.raw_files.append(lat_file)

        out.append(result)
    return out


def _format_table(results: list[ModelResult]) -> str:
    header = (
        "| モデル | 量子化 | VRAM (GB) | Spearman | Kendall τ | Top5 Jaccard | Pairwise acc | "
        "JSON 成功率 | mean SD (7軸) | p50 (ms) | p95 (ms) | ライセンス | 学習しやすさ |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )

    # docs/model_candidates.md の並び順に揃える
    desired_order = [
        "qwen3.6-27b-bnb",
        "qwen3-14b-bf16",
        "qwen2.5-32b-awq",
        "swallow-3.1-8b-bf16",
        "phi-4-14b-bf16",
    ]
    by_name = {r.served_name: r for r in results}

    rows: list[str] = []
    for name in desired_order:
        meta = MODEL_META.get(name, {})
        r = by_name.get(name)
        if r is None:
            r = ModelResult(served_name=name)
        rows.append(
            "| {label} | {quant} | {vram} | {sp} | {kt} | {tj} | {pa} | {js} | {sd} | "
            "{p50} | {p95} | {lic} | {tr} |".format(
                label=meta.get("label", name),
                quant=meta.get("quant", "?"),
                vram=_fmt(r.vram_gb, ".1f"),
                sp=_fmt(r.spearman),
                kt=_fmt(r.kendall_tau),
                tj=_fmt(r.top5_jaccard),
                pa=_fmt(r.pairwise_acc),
                js=_fmt(r.json_success_rate, ".1%") if r.json_success_rate is not None else "TBD",
                sd=_fmt(r.mean_sd_axes, ".3f"),
                p50=_fmt(r.p50_ms, ".0f"),
                p95=_fmt(r.p95_ms, ".0f"),
                lic=meta.get("license", "?"),
                tr=meta.get("trainability", "?"),
            )
        )

    # 想定外モデルが reports に居る場合は末尾に追加
    for r in results:
        if r.served_name not in desired_order:
            rows.append(
                f"| {r.served_name} | ? | TBD | {_fmt(r.spearman)} | {_fmt(r.kendall_tau)} | "
                f"{_fmt(r.top5_jaccard)} | {_fmt(r.pairwise_acc)} | "
                f"{_fmt(r.json_success_rate, '.1%') if r.json_success_rate is not None else 'TBD'} | "
                f"{_fmt(r.mean_sd_axes)} | {_fmt(r.p50_ms, '.0f')} | {_fmt(r.p95_ms, '.0f')} | ? | ? |"
            )

    return header + "\n" + "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="ベンチマーク JSON を Markdown 表に集計する")
    parser.add_argument(
        "--reports-dir",
        default="reports/model_benchmarks",
        help="reports/model_benchmarks の場所",
    )
    parser.add_argument("--out", help="Markdown を書き出す先（省略時は stdout）")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    results = _collect(reports_dir)
    if not results:
        msg = (
            f"WARN: {reports_dir} 配下に集計対象が見つかりません。\n"
            "      `bash scripts/run_model_benchmark.sh --all` を先に走らせてください。"
        )
        print(msg)
        return 0

    table = _format_table(results)
    summary = (
        "# モデル比較ベンチマーク集計結果\n\n"
        f"reports_dir: `{reports_dir}`\n\n"
        f"対象モデル数: {len(results)}\n\n"
        f"集計時刻ベース (各モデル最新): "
        + ", ".join(f"{r.served_name}={r.timestamp}" for r in results)
        + "\n\n"
        "## スコアシート\n\n"
        + table
        + "\n\n"
        "## 反映手順\n\n"
        "1. 上記表を `docs/model_selection_v1.md` の同等表に置き換える\n"
        "2. `docs/adr/0001-judgment-model.md` の Status を `Proposed` → `Accepted` に更新\n"
        "3. `docs/model_history.md` の v1 行の採用日を確定日に更新\n"
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(summary, encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
