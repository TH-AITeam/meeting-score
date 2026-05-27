#!/usr/bin/env python3
"""軸重みをペアワイズデータから回帰で推定する (Issue #16)。

`config.yaml` の軸重み (issue_clarification × 1.3 等) は現状カン。人手ペアワイズ
(#5) や組織フィードバック (#80 の weights/pairs.jsonl) から **ペアワイズ
ロジスティック回帰**で重み w を推定する。

## アプローチ A (主): ペアワイズロジスティック回帰

ペア (winner W, loser L) の軸スコア差 x = scores(W) - scores(L) に対し、
P(W が勝つ) = sigmoid(w · x) を最尤推定する（RankNet 型ペアワイズロジ損失）:

    L(w) = mean( -log sigmoid(w · x) ) + λ ||w||²        (L2 / Ridge)

得られた w がそのまま加点 7 軸の重み (ScoringWeights) になる。
- 加点軸の重みは ≥ 0 に射影（投影勾配法）。
- アノテが少ないと過学習するため Ridge 正則化 (--l2) を入れる。
- sklearn は使わず numpy で実装（依存最小）。

## 入力（どちらか）

- ``--pairs PATH``: 軸スコア同梱のペア JSONL。
  #80 形式 ``{"meeting_id","utt_a","utt_b","winner","scores_a","scores_b",...}``
  または gold 形式（scores_a/scores_b を別途付与済み）。
- ``--synthesize-from-labels``: 既存 distill ラベル (#13 の load_tier 経由) から
  会議内のスコア差でペアを合成（gold #5 未整備時のブートストラップ・GPU 不要）。

## 出力

- ``config/weights_regressed.yaml``: 推定重み + eval（固定 vs 回帰の pairwise acc）。
- ``docs/weight_regression_report.md``: 比較レポート。

使い方:
    python scripts/regress_weights.py --synthesize-from-labels
    python scripts/regress_weights.py --pairs data/feedback/org_001/weights/v1/pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from string import Template
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT))

from app.evaluators.prompt import PROMPT_PATH, RESPONSE_SCHEMA  # noqa: E402
from app.scoring.weights import ScoringWeights  # noqa: E402

# 加点 7 軸（スキーマを単一の真実とする）。回帰対象はこの 7 軸の重み。
SCORE_KEYS = list(RESPONSE_SCHEMA["properties"]["scores"]["properties"])


# ---------------------------------------------------------------------------
# ペアの読み込み / 合成（各ペアは winner/loser の軸スコアベクトルを持つ）
# ---------------------------------------------------------------------------
def _scores_vec(scores: dict[str, Any]) -> list[int]:
    return [int(scores.get(k, 0)) for k in SCORE_KEYS]


def _normalize_winner(winner: str) -> str | None:
    w = str(winner).strip().lower()
    if w in {"a", "a_better", "utt_a"}:
        return "a"
    if w in {"b", "b_better", "utt_b"}:
        return "b"
    return None  # tie ほかは除外


def load_pairs_with_scores(path: Path) -> list[tuple[list[int], list[int]]]:
    """scores_a/scores_b 同梱のペア JSONL → (winner_vec, loser_vec) のリスト。"""
    out: list[tuple[list[int], list[int]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        r = json.loads(line)
        side = _normalize_winner(r.get("winner", ""))
        if side is None or "scores_a" not in r or "scores_b" not in r:
            continue
        va, vb = _scores_vec(r["scores_a"]), _scores_vec(r["scores_b"])
        out.append((va, vb) if side == "a" else (vb, va))
    return out


def synthesize_pairs_from_labels(
    distill_dir: Path, min_margin: int, max_per_meeting: int
) -> list[tuple[list[int], list[int]]]:
    """既存 distill ラベルから会議内のスコア差でペアを合成する（#13 の load_tier 流用）。"""
    from scripts.build_sft_dataset import BuildResult, load_tier

    template = Template(PROMPT_PATH.read_text(encoding="utf-8"))
    result = BuildResult()
    load_tier(distill_dir / "jobs", distill_dir / "labels", "distilled", template, result)

    def total(scores: dict[str, Any]) -> int:
        return sum(int(scores.get(k, 0)) for k in SCORE_KEYS)

    out: list[tuple[list[int], list[int]]] = []
    for samples in result.by_meeting.values():
        ranked = sorted(samples, key=lambda s: total(s.assistant["scores"]), reverse=True)
        made, i, j = 0, 0, len(ranked) - 1
        while i < j and made < max_per_meeting:
            hi, lo = ranked[i], ranked[j]
            if total(hi.assistant["scores"]) - total(lo.assistant["scores"]) >= min_margin:
                out.append(
                    (_scores_vec(hi.assistant["scores"]), _scores_vec(lo.assistant["scores"]))
                )
                made += 1
                i += 1
                j -= 1
            else:
                j -= 1
    return out


# ---------------------------------------------------------------------------
# ペアワイズロジスティック回帰（numpy / Ridge / 加点軸 ≥ 0）
# ---------------------------------------------------------------------------
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_pairwise_logistic(
    diffs: np.ndarray,
    *,
    l2: float = 1.0,
    lr: float = 0.1,
    iters: int = 3000,
    nonneg: bool = True,
) -> np.ndarray:
    """diffs[i] = scores(winner) - scores(loser)（全て勝者向き）に対し w を最尤推定。

    損失: mean(-log sigmoid(diffs @ w)) + l2 * ||w||²。
    nonneg=True なら各ステップで w を 0 以上に射影（加点軸の重み ≥ 0 制約）。
    """
    n, d = diffs.shape
    w = np.zeros(d)
    for _ in range(iters):
        p = _sigmoid(diffs @ w)  # P(winner が勝つ)
        grad = -(diffs.T @ (1.0 - p)) / n + 2.0 * l2 * w
        w = w - lr * grad
        if nonneg:
            w = np.maximum(w, 0.0)
    return w


def normalize_weights(w: np.ndarray, target_mean: float = 1.0) -> np.ndarray:
    """重みの平均を target_mean に揃える（順位は相対値で決まるためスケール自由）。"""
    m = float(np.mean(w))
    if m <= 0:
        return np.full_like(w, target_mean)
    return w * (target_mean / m)


def pairwise_accuracy(diffs: np.ndarray, w: np.ndarray) -> float:
    """各ペアで w·(winner-loser) > 0 となる割合（順位の整合率）。"""
    if len(diffs) == 0:
        return float("nan")
    return float(np.mean((diffs @ w) > 0))


def default_weight_vector() -> np.ndarray:
    d = ScoringWeights()
    return np.array([getattr(d, k) for k in SCORE_KEYS], dtype=float)


# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------
def render_weights_yaml(w: np.ndarray, n_pairs: int, acc_fixed: float, acc_reg: float) -> str:
    lines = [
        "# 回帰で推定した軸重み (auto-generated by scripts/regress_weights.py, Issue #16)",
        "# 手で編集しないこと。config.yaml に採用する場合は値をコピーする。",
        f'generated_at: "{datetime.now(tz=UTC).isoformat()}"',
        f"n_pairs: {n_pairs}",
        "weights:",
    ]
    lines += [f"  {k}: {round(float(v), 3)}" for k, v in zip(SCORE_KEYS, w, strict=True)]
    lines += [
        "eval:",
        f"  pairwise_acc_fixed: {round(acc_fixed, 4)}",
        f"  pairwise_acc_regressed: {round(acc_reg, 4)}",
    ]
    return "\n".join(lines) + "\n"


def render_report(
    w: np.ndarray, n_train: int, n_val: int, acc_fixed: float, acc_reg: float, l2: float
) -> str:
    default = default_weight_vector()
    lines = [
        "# 軸重み回帰レポート (Issue #16)",
        "",
        "自動生成 (`scripts/regress_weights.py`)。",
        "",
        f"- 学習ペア: {n_train} / 検証ペア: {n_val}",
        f"- L2 (Ridge): {l2}",
        "",
        "## pairwise accuracy (検証ペア)",
        "",
        "| 重み | pairwise acc |",
        "|---|---:|",
        f"| 固定 (config 既定) | {acc_fixed:.4f} |",
        f"| 回帰 | {acc_reg:.4f} |",
        f"| 差分 | {acc_reg - acc_fixed:+.4f} |",
        "",
        "> 完了条件「Spearman または pairwise accuracy が +0.02 以上」のうち pairwise を計測。",
        "> Spearman (人手ランキング相関) は gold アノテ (#5) と eval ハーネスが必要で別途。",
        "",
        "## 重み比較 (固定 → 回帰、平均 1.0 に正規化)",
        "",
        "| 軸 | 固定 | 回帰 |",
        "|---|---:|---:|",
    ]
    for k, dv, rv in zip(SCORE_KEYS, default, w, strict=True):
        lines.append(f"| {k} | {dv:.2f} | {rv:.3f} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# オーケストレーション
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.pairs:
        pairs = load_pairs_with_scores(Path(args.pairs))
        source = f"file:{args.pairs}"
    elif args.synthesize_from_labels:
        pairs = synthesize_pairs_from_labels(
            Path(args.distill_dir), args.min_margin, args.max_per_meeting
        )
        source = "synthesized-from-labels"
    else:
        print("--pairs PATH か --synthesize-from-labels を指定してください。", file=sys.stderr)
        return {}

    if not pairs:
        print("ペアが 0 件です。入力を確認してください。", file=sys.stderr)
        return {}

    diffs_all = np.array([np.array(win) - np.array(lose) for win, lose in pairs], dtype=float)

    # 会議をまたぐ情報は持たないため単純シャッフルで train/val 分割。
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(diffs_all))
    n_val = max(1, round(len(idx) * args.val_ratio)) if len(idx) > 1 else 0
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train, val = diffs_all[train_idx], diffs_all[val_idx]

    w_raw = fit_pairwise_logistic(train, l2=args.l2, lr=args.lr, iters=args.iters, nonneg=True)
    w = normalize_weights(w_raw)
    w_fixed = default_weight_vector()

    eval_set = val if len(val) else train
    acc_fixed = pairwise_accuracy(eval_set, w_fixed)
    acc_reg = pairwise_accuracy(eval_set, w)

    out_yaml = Path(args.out_weights)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(render_weights_yaml(w, len(pairs), acc_fixed, acc_reg), encoding="utf-8")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(w, len(train), len(eval_set), acc_fixed, acc_reg, args.l2),
        encoding="utf-8",
    )

    print(
        f"ペア {len(pairs)} 件 ({source}) -> train {len(train)} / eval {len(eval_set)}\n"
        f"  pairwise acc: 固定 {acc_fixed:.4f} / 回帰 {acc_reg:.4f} ({acc_reg - acc_fixed:+.4f})\n"
        f"  -> {out_yaml}\n  -> {report_path}"
    )
    return {
        "n_pairs": len(pairs),
        "acc_fixed": acc_fixed,
        "acc_regressed": acc_reg,
        "weights": dict(zip(SCORE_KEYS, [round(float(v), 3) for v in w], strict=True)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--pairs", default=None, help="軸スコア同梱のペア JSONL (#80 weights/pairs.jsonl 等)"
    )
    p.add_argument(
        "--synthesize-from-labels",
        action="store_true",
        help="既存 distill ラベルからペアを合成 (gold #5 未整備時のブートストラップ)",
    )
    p.add_argument(
        "--distill-dir",
        default=str(REPO_ROOT / "data" / "annotations" / "kokkai" / "distill"),
    )
    p.add_argument("--out-weights", default=str(REPO_ROOT / "config" / "weights_regressed.yaml"))
    p.add_argument("--report", default=str(REPO_ROOT / "docs" / "weight_regression_report.md"))
    p.add_argument("--l2", type=float, default=1.0, help="Ridge 正則化係数 (小データの過学習抑制)")
    p.add_argument("--lr", type=float, default=0.1, help="勾配降下の学習率")
    p.add_argument("--iters", type=int, default=3000, help="勾配降下のイテレーション数")
    p.add_argument("--min-margin", type=int, default=4, help="合成ペアの最小合計点差")
    p.add_argument("--max-per-meeting", type=int, default=8, help="合成ペアの会議あたり上限")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=3407)
    return p


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
