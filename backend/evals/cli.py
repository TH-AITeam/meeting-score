"""eval ハーネス CLI (Issue #5 + Issue #17)。

既定の backend は local (vLLM 等 OpenAI 互換サーバ)。
OpenAI Responses API はベンチマーク比較・蒸留用途の optional 経路。

Examples
--------
    # ベースライン評価（既定の backend = local）
    python -m evals.cli run \\
        --dataset data/annotations/gold/v1 \\
        --out reports/eval/v1.json

    # 別ホストの vLLM サーバを叩く（Issue #18 のベンチマーク用途）
    python -m evals.cli \\
        --endpoint http://other-host:8001/v1 \\
        --model qwen3.6-27b-bnb \\
        stability \\
        --meeting data/sample_meetings/sample_meeting_01.json \\
        --n 5 \\
        --out reports/eval/stability.json

    # OpenAI クラウドと比較（API キーが必要、optional）
    python -m evals.cli --backend openai --model gpt-4o-mini run \\
        --dataset data/annotations/gold/v1 \\
        --out reports/eval/openai_baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from app.context_builder.builder import build_contexts
from app.evaluators import create_evaluator
from app.ingest.loader import load_meeting_from_file
from app.scoring.weights import AppConfig, load_config, resolve_llm_model_for_backend
from evals.runner import run_eval
from evals.stability import evaluate_stability

logger = logging.getLogger(__name__)


def _build_config(args: argparse.Namespace) -> AppConfig:
    """config.yaml + CLI 上書きで AppConfig を組み立てる。

    `--backend` / `--endpoint` / `--model` / `--api-key` で config.yaml の値を
    上書きできるようにする。これにより SSH 先で vLLM サーバを順次切り替える
    ベンチマーク (Issue #18) が config を書き換えずに走らせられる。
    """
    cfg = load_config(args.config) if args.config else load_config()
    if args.backend:
        cfg.llm_backend = args.backend
    if args.endpoint:
        cfg.llm_endpoint = args.endpoint
        # endpoint を指定したなら明示しない限り local に倒す
        if not args.backend:
            cfg.llm_backend = "local"
    if args.model:
        cfg.llm_model = args.model
    if args.api_key:
        cfg.llm_api_key = args.api_key
    cfg.llm_model = resolve_llm_model_for_backend(cfg.llm_backend, cfg.llm_model)
    return cfg


def _write_json(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    evaluator = create_evaluator(cfg)
    report = run_eval(
        Path(args.dataset),
        evaluator,
        cfg.weights,
        cfg.penalty_weights,
        meetings_dir=Path(args.meetings_dir) if args.meetings_dir else None,
        model_name=cfg.llm_model,
        context_before=cfg.context_before,
        context_after=cfg.context_after,
        meeting_type_weights=cfg.meeting_type_weights or None,
    )
    payload = report.to_dict()

    out_path = Path(args.out) if args.out else None
    if out_path:
        _write_json(out_path, payload)
        logger.info("レポートを書き出しました: %s", out_path)
    else:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, default=str)
        sys.stdout.write("\n")

    macro = payload["macro"]
    print(
        "macro: spearman={s:.3f} kendall={k:.3f} "
        "top5_jaccard={t:.3f} bottom5_jaccard={b:.3f} pairwise={p:.3f}".format(
            s=macro["spearman"],
            k=macro["kendall_tau"],
            t=macro["top5_jaccard"],
            b=macro["bottom5_jaccard"],
            p=macro["pairwise_accuracy"],
        ),
        file=sys.stderr,
    )
    return 0


def _cmd_stability(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    evaluator = create_evaluator(cfg)
    meeting = load_meeting_from_file(Path(args.meeting))
    contexts = build_contexts(
        meeting,
        before_count=cfg.context_before,
        after_count=cfg.context_after,
    )
    stability = evaluate_stability(
        evaluator,
        contexts,
        meeting_id=meeting.meeting_id,
        n_samples=args.n,
    )

    payload = {
        "meeting_id": stability.meeting_id,
        "n_samples": args.n,
        "model": cfg.llm_model,
        "backend": cfg.llm_backend,
        "mean_sd_per_axis": stability.mean_sd_per_axis,
        "max_sd_per_axis": stability.max_sd_per_axis,
        "utterances": [asdict(u) for u in stability.utterances],
    }
    out_path = Path(args.out) if args.out else None
    if out_path:
        _write_json(out_path, payload)
        logger.info("レポートを書き出しました: %s", out_path)
    else:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, default=str)
        sys.stdout.write("\n")

    mean_sd = {k: round(v, 3) for k, v in stability.mean_sd_per_axis.items()}
    max_sd = {k: round(v, 3) for k, v in stability.max_sd_per_axis.items()}
    print(f"stability: mean_sd_axes={mean_sd} max_sd_axes={max_sd}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval", description="eval ハーネス")
    parser.add_argument("--config", help="config.yaml のパス (既定: backend/config.yaml)")
    parser.add_argument("--backend", choices=("openai", "local"), help="LLM backend 上書き")
    parser.add_argument(
        "--endpoint", help="OpenAI 互換エンドポイント (例: http://127.0.0.1:8001/v1)"
    )
    parser.add_argument("--model", help="LLM モデル名 (既定: config.yaml の値)")
    parser.add_argument("--api-key", help="OpenAI 互換 API キー上書き")

    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="アノテーション済みデータセットを評価する")
    run_p.add_argument("--dataset", required=True, help="例: data/annotations/gold/v1")
    run_p.add_argument("--meetings-dir", help="会議元データの探索ディレクトリ")
    run_p.add_argument("--out", help="JSON 出力先")
    run_p.set_defaults(func=_cmd_run)

    stab_p = sub.add_parser("stability", help="同一会議を N 回採点して分散を測る")
    stab_p.add_argument("--meeting", required=True, help="会議 JSON のパス")
    stab_p.add_argument("--n", type=int, default=5, help="サンプル数 (既定 5)")
    stab_p.add_argument("--out", help="JSON 出力先")
    stab_p.set_defaults(func=_cmd_stability)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
