"""eval ハーネス CLI (Issue #5)。

Examples
--------
    # ベースライン評価
    python -m evals.cli run \\
        --dataset data/annotations/gold/v1 \\
        --out reports/eval/v1.json

    # 安定性評価 (N=5, temperature は Evaluator 側で設定)
    python -m evals.cli stability \\
        --meeting data/sample_meetings/sample_01.json \\
        --n 5 \\
        --out reports/eval/stability_sample_01.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from app.context_builder.builder import build_contexts
from app.evaluators.llm_evaluator import evaluate_utterance
from app.ingest.loader import load_meeting_from_file
from app.scoring.weights import load_config

from evals.protocol import EvaluationResult
from evals.runner import run_eval
from evals.stability import evaluate_stability

if TYPE_CHECKING:
    from app.context_builder.builder import EvaluationContext

logger = logging.getLogger(__name__)
STABILITY_TEMPERATURE = 0.7


class LLMEvaluatorAdapter:
    """既存の `evaluate_utterance` を Evaluator プロトコルに整形するアダプタ。

    Issue #12 で Evaluator ABC が main にマージされたら、ここを差し替えるだけで
    eval ハーネス本体は無改変で動く想定。
    """

    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        max_tokens: int = 1024,
        max_retries: int = 3,
        temperature: float | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.temperature = temperature

    def evaluate(self, ctx: EvaluationContext) -> EvaluationResult:
        raw = evaluate_utterance(
            ctx,
            model=self.model,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            temperature=self.temperature,
        )
        return EvaluationResult(
            speech_type=raw["speech_type"],
            scores=raw["scores"],
            penalties=raw["penalties"],
            reason=raw.get("reason", ""),
            evaluation_failed=raw.get("evaluation_failed", False),
        )


def _write_json(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else load_config()
    evaluator = LLMEvaluatorAdapter(
        model=args.model or cfg.llm_model,
        max_tokens=cfg.llm_max_tokens,
        max_retries=cfg.llm_max_retries,
    )
    report = run_eval(
        Path(args.dataset),
        evaluator,
        cfg.weights,
        meetings_dir=Path(args.meetings_dir) if args.meetings_dir else None,
        model_name=args.model or cfg.llm_model,
        context_before=cfg.context_before,
        context_after=cfg.context_after,
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
    cfg = load_config(args.config) if args.config else load_config()
    evaluator = LLMEvaluatorAdapter(
        model=args.model or cfg.llm_model,
        max_tokens=cfg.llm_max_tokens,
        max_retries=cfg.llm_max_retries,
        temperature=STABILITY_TEMPERATURE,
    )
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

    print(
        "stability: mean_sd_axes={m} max_sd_axes={x}".format(
            m={k: round(v, 3) for k, v in stability.mean_sd_per_axis.items()},
            x={k: round(v, 3) for k, v in stability.max_sd_per_axis.items()},
        ),
        file=sys.stderr,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval", description="eval ハーネス")
    parser.add_argument("--config", help="config.yaml のパス（既定: backend/config.yaml）")
    parser.add_argument("--model", help="LLM モデル名（既定: config.yaml の値）")

    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="アノテーション済みデータセットを評価する")
    run_p.add_argument("--dataset", required=True, help="例: data/annotations/gold/v1")
    run_p.add_argument("--meetings-dir", help="会議元データの探索ディレクトリ")
    run_p.add_argument("--out", help="JSON 出力先")
    run_p.set_defaults(func=_cmd_run)

    stab_p = sub.add_parser("stability", help="同一会議を N 回採点して分散を測る")
    stab_p.add_argument("--meeting", required=True, help="会議 JSON のパス")
    stab_p.add_argument("--n", type=int, default=5, help="サンプル数（既定 5）")
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
