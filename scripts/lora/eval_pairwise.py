#!/usr/bin/env python3
"""ペアワイズ accuracy と Top-5 Jaccard で SFT 単独 vs SFT+DPO を比較する (Issue #15)。

完了条件の指標:
  - ペアワイズ accuracy: 人手ペア（gold pairs.jsonl）について、モデルが
    勝者発言に高い合計スコアを付けられた割合。
  - Top-5 Jaccard: 会議ごとに、モデルの Top5（合計スコア上位）と gold Top5 の
    Jaccard 類似度の平均。

絶対スコア精度は `eval_lora.py`、本スクリプトは**順位の納得感**を測る。
SFT と SFT+DPO のアダプタそれぞれに対して実行し、差分を比較する:

    source scripts/lora/.venv-lora/bin/activate
    python scripts/lora/eval_pairwise.py --adapter outputs/qwen35-9b-lora   # SFT
    python scripts/lora/eval_pairwise.py --adapter outputs/qwen35-9b-dpo    # SFT+DPO

メトリクス計算（`pairwise_accuracy` / `mean_top5_jaccard` 等）はモデル非依存の
純関数で、ユニットテスト対象。モデル推論部は `eval_lora.py` と同じ Unsloth 経路。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_pairwise")

REPO_ROOT = Path(__file__).resolve().parents[2]

SCORE_AXES = [
    "issue_clarification",
    "decision_progress",
    "risk_detection",
    "actionability",
    "groundedness",
    "novelty",
    "summarization",
]
PENALTY_AXES = ["duplication", "verbosity", "off_topic", "unsupported_assertion"]

_WINNER_A = {"a", "a_better", "utt_a"}
_WINNER_B = {"b", "b_better", "utt_b"}


# ---------- 純関数（テスト対象） ----------
def clamp(v: Any, lo: int, hi: int) -> int | None:
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return None


def total_of(obj: dict) -> float:
    """評価 JSON の合計点（scores 合計 + penalties 合計、値域クランプ込み）。"""
    s = sum(clamp(obj.get("scores", {}).get(a), 0, 3) or 0 for a in SCORE_AXES)
    p = sum(clamp(obj.get("penalties", {}).get(a), -3, 0) or 0 for a in PENALTY_AXES)
    return float(s + p)


def normalize_winner(winner: str) -> str | None:
    w = str(winner).strip().lower()
    if w in _WINNER_A:
        return "a"
    if w in _WINNER_B:
        return "b"
    if w == "tie":
        return "tie"
    return None


def pairwise_accuracy(
    pairs: list[dict[str, Any]], totals: dict[tuple[str, str], float]
) -> tuple[float, int]:
    """gold ペアに対する勝者予測の正答率を返す。

    totals: (meeting_id, utterance_id) -> モデルが付けた合計スコア。
    タイは accuracy 計算から除外する。戻り値: (accuracy, 評価対象ペア数)。
    """
    correct = 0
    n = 0
    for p in pairs:
        side = normalize_winner(p.get("winner", ""))
        if side is None or side == "tie":
            continue
        key_a = (p["meeting_id"], str(p["utt_a"]))
        key_b = (p["meeting_id"], str(p["utt_b"]))
        if key_a not in totals or key_b not in totals:
            continue
        ta, tb = totals[key_a], totals[key_b]
        if ta == tb:
            continue  # モデルが同点 → 順位判定不能（不正解扱いにしないため除外）
        pred = "a" if ta > tb else "b"
        correct += int(pred == side)
        n += 1
    return (correct / n if n else float("nan"), n)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def top_k_by_total(totals: dict[str, float], k: int = 5) -> set[str]:
    """utterance_id -> total から上位 k の utterance_id 集合を返す。"""
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    return {uid for uid, _ in ranked[:k]}


def mean_top5_jaccard(
    pred_top5: dict[str, set[str]], gold_top5: dict[str, set[str]]
) -> tuple[float, int]:
    """会議ごとの Top5 Jaccard の平均を返す。戻り値: (mean, 会議数)。"""
    meetings = [m for m in gold_top5 if m in pred_top5]
    if not meetings:
        return (float("nan"), 0)
    vals = [jaccard(pred_top5[m], gold_top5[m]) for m in meetings]
    return (sum(vals) / len(vals), len(vals))


# ---------- モデル推論部（GPU / Unsloth、eval_lora.py と同経路） ----------
def extract_json(text: str) -> dict | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    p.add_argument(
        "--adapter",
        default=str(REPO_ROOT / "outputs" / "qwen35-9b-lora"),
        help="評価する LoRA アダプタ（SFT or SFT+DPO）",
    )
    p.add_argument("--no-adapter", action="store_true", help="素のベースモデルを評価")
    p.add_argument(
        "--pairs",
        default=str(REPO_ROOT / "data" / "annotations" / "gold" / "v1" / "pairs.jsonl"),
        help="ペアワイズ accuracy 用の gold ペア JSONL",
    )
    p.add_argument(
        "--eval-jsonl",
        default=str(REPO_ROOT / "data" / "sft" / "v1" / "val.jsonl"),
        help="Top5/合計推論に使う messages 形式 JSONL（meta に meeting_id/utterance_id）",
    )
    p.add_argument("--max-seq-len", type=int, default=6144)
    p.add_argument("--max-new-tokens", type=int, default=512)
    return p.parse_args()


def _generate_totals(
    args: argparse.Namespace,
) -> tuple[dict[tuple[str, str], float], dict[str, set[str]]]:
    """eval-jsonl の各発言をモデルで評価し、(meeting,utt)->total と会議別 Top5 を返す。"""
    import torch
    from unsloth import FastLanguageModel

    rows = [
        json.loads(line)
        for line in Path(args.eval_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model_name = args.model_id if args.no_adapter else args.adapter
    logger.info("ロード: %s (%d 件評価)", model_name, len(rows))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    device = getattr(model, "device", None) or next(model.parameters()).device

    totals: dict[tuple[str, str], float] = {}
    by_meeting: dict[str, dict[str, float]] = {}
    for row in rows:
        meta = row.get("meta", {})
        mid, uid = meta.get("meeting_id"), meta.get("utterance_id")
        if mid is None or uid is None:
            continue
        user_msg = next(m for m in row["messages"] if m["role"] == "user")
        text = tokenizer.apply_chat_template(
            [user_msg],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = text_tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=text_tok.eos_token_id,
            )
        gen = text_tok.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        pred = extract_json(gen)
        t = total_of(pred) if pred else 0.0
        totals[(mid, str(uid))] = t
        by_meeting.setdefault(mid, {})[str(uid)] = t

    pred_top5 = {m: top_k_by_total(v, 5) for m, v in by_meeting.items()}
    return totals, pred_top5


def _gold_top5_from_eval(eval_jsonl: Path) -> dict[str, set[str]]:
    """eval-jsonl の gold(assistant) 合計から会議別 gold Top5 を作る。"""
    rows = [
        json.loads(line)
        for line in eval_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_meeting: dict[str, dict[str, float]] = {}
    for row in rows:
        meta = row.get("meta", {})
        mid, uid = meta.get("meeting_id"), meta.get("utterance_id")
        if mid is None or uid is None:
            continue
        gold = json.loads(
            next(m for m in row["messages"] if m["role"] == "assistant")["content"]
        )
        by_meeting.setdefault(mid, {})[str(uid)] = total_of(gold)
    return {m: top_k_by_total(v, 5) for m, v in by_meeting.items()}


def main() -> None:
    args = parse_args()
    totals, pred_top5 = _generate_totals(args)

    gold_top5 = _gold_top5_from_eval(Path(args.eval_jsonl))
    jac, n_meet = mean_top5_jaccard(pred_top5, gold_top5)

    pairs_path = Path(args.pairs)
    if pairs_path.exists():
        pairs = [
            json.loads(line)
            for line in pairs_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        acc, n_pairs = pairwise_accuracy(pairs, totals)
    else:
        logger.warning(
            "gold ペアが無いため pairwise accuracy をスキップ: %s", pairs_path
        )
        acc, n_pairs = float("nan"), 0

    print("\n" + "=" * 56)
    print(f"ペアワイズ評価  (adapter={args.adapter})")
    print("=" * 56)
    print(f"ペアワイズ accuracy : {acc:.1%}  (n={n_pairs})")
    print(f"Top-5 Jaccard 平均  : {jac:.3f}  (会議 {n_meet} 件)")
    print("=" * 56)


if __name__ == "__main__":
    main()
