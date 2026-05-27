#!/usr/bin/env python3
"""DPO 学習データ構築パイプライン (Issue #15)。

ペアワイズ選好（どちらの発言が会議をより前進させたか）を DPO の
``{prompt, chosen, rejected}`` 形式に変換する。SFT (#13/#14) と同じく本番推論と
同一の ``prompt`` を使い、``chosen``/``rejected`` は本番スキーマ準拠の評価 JSON。

## DPO の組み方（このスクリプトの定式化）

ペアの勝者 W・敗者 L に対し、次の三つ組を作る:

    prompt   = W の評価プロンプト（本番 build_prompt 相当）
    chosen   = W の評価 JSON（高スコアが出るべき発言）
    rejected = L の評価 JSON（低スコアが出るべき発言）

「W のプロンプトに対しては、W 本来の（高めの）評価を、L の（低めの）評価より
選好する」ことを学習させ、順位の納得感を直接最適化する。``--symmetric``（既定 on）
では L 側プロンプトの三つ組（chosen=L 評価, rejected=W 評価）も作り、敗者を
押し下げる信号も与える。

> 注: 本タスクの DPO 定式化は研究的な選択肢が複数ある（Issue #15「コツ」）。
> 上記は実装・解釈が素直な既定。データ・SFT モデルが揃った段階で見直し可。

## 評価 JSON の出どころ

各発言の ``chosen``/``rejected`` JSON は、本来は **SFT モデルで両発言を評価して**
生成する（#15 やること）。本スクリプトは GPU 無し環境でも回せるよう、既定では
distilled/gold の **既存ラベル**（`build_sft_dataset` 経由）を評価 JSON として使う。
SFT モデルで再評価する経路は、モデルを vLLM 配信して `--eval-source` を拡張する
将来作業（モデル・GPU 依存）。

## ペアの入力

- ``--pairs PATH``: ペアワイズ JSONL（gold #5 / 合成 #7）。
  形式 ``{"meeting_id","utt_a","utt_b","winner","source"?,"pattern"?}``。
  winner は ``A_better|B_better|tie``（gold）または ``A|B|tie`` を受け付ける。
- ``--synthesize-from-labels``: 既存ラベルのスコア合計差から会議内でペアを合成
  （gold #5 が未整備でもパイプラインを検証・運用できるブートストラップ）。

使い方:
    # 既存 distill ラベルからペアを合成して DPO データを作る（GPU 不要）
    python scripts/build_dpo_dataset.py --synthesize-from-labels

    # gold/合成ペアファイルから作る
    python scripts/build_dpo_dataset.py --pairs data/annotations/gold/v1/pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from string import Template
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT))

# 本番プロンプト/スキーマ流用（#13 と同じ単一の真実）。
from app.evaluators.prompt import PROMPT_PATH  # noqa: E402

# #13 のローダ・正規化を再利用（重複実装を避ける）。
from scripts.build_sft_dataset import (  # noqa: E402
    PENALTY_KEYS,
    SCORE_KEYS,
    BuildResult,
    Sample,
    load_tier,
)

# winner ラベルの正規化: 値 -> 'a' | 'b' | 'tie'
_WINNER_A = {"a", "a_better", "utt_a"}
_WINNER_B = {"b", "b_better", "utt_b"}
DPO_PENALTY_KEYS = (
    PENALTY_KEYS if "override" in PENALTY_KEYS else [*PENALTY_KEYS, "override"]
)


def total_score(assistant: dict[str, Any]) -> int:
    """評価 JSON の合計点（scores 合計 + penalties 合計）。"""
    s = sum(assistant["scores"][k] for k in SCORE_KEYS)
    p = sum(assistant["penalties"][k] for k in DPO_PENALTY_KEYS)
    return s + p


def normalize_winner(winner: str) -> str | None:
    """winner ラベルを 'a' | 'b' | 'tie' に正規化。未知は None。"""
    w = winner.strip().lower()
    if w in _WINNER_A:
        return "a"
    if w in _WINNER_B:
        return "b"
    if w == "tie":
        return "tie"
    return None


def load_sample_index(
    distill_dir: Path, gold_dir: Path, template: Template
) -> dict[tuple[str, str], Sample]:
    """(meeting_id, utterance_id) -> Sample のインデックスを作る（#13 のローダ流用）。"""
    result = BuildResult()
    load_tier(
        distill_dir / "jobs", distill_dir / "labels", "distilled", template, result
    )
    load_tier(gold_dir / "jobs", gold_dir / "labels", "gold", template, result)
    index: dict[tuple[str, str], Sample] = {}
    for meeting_id, samples in result.by_meeting.items():
        for s in samples:
            index[(meeting_id, s.utterance_id)] = s
    return index


def synthesize_pairs(
    index: dict[tuple[str, str], Sample], min_margin: int, max_per_meeting: int
) -> list[dict[str, Any]]:
    """会議内でスコア合計差が ``min_margin`` 以上のペアを合成する。

    高合計を winner、低合計を loser とする（人手ペアの代用ブートストラップ）。
    """
    by_meeting: dict[str, list[Sample]] = {}
    for (meeting_id, _), s in index.items():
        by_meeting.setdefault(meeting_id, []).append(s)

    pairs: list[dict[str, Any]] = []
    for meeting_id, samples in by_meeting.items():
        ranked = sorted(samples, key=lambda s: total_score(s.assistant), reverse=True)
        made = 0
        # 上位と下位を突き合わせて明確な差のペアを作る
        i, j = 0, len(ranked) - 1
        while i < j and made < max_per_meeting:
            hi, lo = ranked[i], ranked[j]
            if total_score(hi.assistant) - total_score(lo.assistant) >= min_margin:
                pairs.append(
                    {
                        "meeting_id": meeting_id,
                        "utt_a": hi.utterance_id,
                        "utt_b": lo.utterance_id,
                        "winner": "A_better",
                        "source": "synthetic",
                    }
                )
                made += 1
                i += 1
                j -= 1
            else:
                # 差が足りない: 下位側を詰めて差を広げる
                j -= 1
    return pairs


def load_pairs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def build_dpo_records(
    pairs: list[dict[str, Any]],
    index: dict[tuple[str, str], Sample],
    *,
    symmetric: bool,
    result_counts: Counter,
) -> dict[str, list[dict[str, Any]]]:
    """ペア -> DPO レコード（会議単位）。winner を chosen、loser を rejected。"""
    by_meeting: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        meeting_id = pair["meeting_id"]
        side = normalize_winner(str(pair.get("winner", "")))
        if side is None:
            result_counts["bad_winner"] += 1
            continue
        if side == "tie":
            result_counts["tie_skipped"] += 1
            continue
        a = index.get((meeting_id, str(pair["utt_a"])))
        b = index.get((meeting_id, str(pair["utt_b"])))
        if a is None or b is None:
            result_counts["missing_utterance"] += 1
            continue
        winner, loser = (a, b) if side == "a" else (b, a)

        meta_base = {
            "source": pair.get("source", "unknown"),
            "meeting_id": meeting_id,
        }
        if pair.get("pattern"):
            meta_base["pattern"] = pair["pattern"]

        records = [
            {
                "prompt": winner.user,
                "chosen": json.dumps(winner.assistant, ensure_ascii=False),
                "rejected": json.dumps(loser.assistant, ensure_ascii=False),
                "meta": {
                    **meta_base,
                    "chosen_id": winner.utterance_id,
                    "rejected_id": loser.utterance_id,
                },
            }
        ]
        if symmetric:
            records.append(
                {
                    "prompt": loser.user,
                    "chosen": json.dumps(winner.assistant, ensure_ascii=False),
                    "rejected": json.dumps(loser.assistant, ensure_ascii=False),
                    "meta": {
                        **meta_base,
                        "chosen_id": winner.utterance_id,
                        "rejected_id": loser.utterance_id,
                    },
                }
            )
        by_meeting.setdefault(meeting_id, []).extend(records)
        result_counts["pairs_used"] += 1
    return by_meeting


def split_train_val(
    by_meeting: dict[str, list[dict[str, Any]]], val_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """会議単位で train/val に分割（同一会議のペアが両 split に跨らない）。"""
    import random

    meetings = list(by_meeting)
    random.Random(seed).shuffle(meetings)
    n_val = round(len(meetings) * val_ratio)
    if val_ratio > 0 and len(meetings) > 1:
        n_val = max(1, min(n_val, len(meetings) - 1))
    val_meetings = set(meetings[:n_val])
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for m, recs in by_meeting.items():
        (val if m in val_meetings else train).extend(recs)
    return train, val


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    path.write_text(body + ("\n" if body else ""), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    template = Template(PROMPT_PATH.read_text(encoding="utf-8"))
    index = load_sample_index(Path(args.distill_dir), Path(args.gold_dir), template)
    if not index:
        print(
            "評価ラベルが見つかりません（distill/gold の jobs+labels を確認）",
            file=sys.stderr,
        )
        return {}

    if args.pairs:
        pairs = load_pairs(Path(args.pairs))
        pair_source = f"file:{args.pairs}"
    elif args.synthesize_from_labels:
        pairs = synthesize_pairs(index, args.min_margin, args.max_per_meeting)
        pair_source = "synthesized-from-labels"
    else:
        print(
            "ペアの入力がありません。--pairs PATH か --synthesize-from-labels を指定してください。",
            file=sys.stderr,
        )
        return {}

    counts: Counter = Counter()
    by_meeting = build_dpo_records(
        pairs, index, symmetric=args.symmetric, result_counts=counts
    )
    train, val = split_train_val(by_meeting, args.val_ratio, args.seed)

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)

    stats = {
        "pair_source": pair_source,
        "n_input_pairs": len(pairs),
        "n_meetings": len(by_meeting),
        "n_train": len(train),
        "n_val": len(val),
        "counts": dict(counts),
        "symmetric": args.symmetric,
    }
    print(
        f"ペア {len(pairs)} 件 ({pair_source}) -> DPO train {len(train)} / val {len(val)} "
        f"(会議 {len(by_meeting)} 件)"
    )
    print(f"  内訳: {dict(counts)}")
    print(f"  -> {out_dir}/train.jsonl, {out_dir}/val.jsonl")
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--distill-dir",
        default=str(REPO_ROOT / "data" / "annotations" / "kokkai" / "distill"),
    )
    p.add_argument(
        "--gold-dir", default=str(REPO_ROOT / "data" / "annotations" / "gold" / "v1")
    )
    p.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "dpo" / "v1"))
    p.add_argument(
        "--pairs", default=None, help="ペアワイズ JSONL（gold #5 / 合成 #7）"
    )
    p.add_argument(
        "--synthesize-from-labels",
        action="store_true",
        help="既存ラベルのスコア差からペアを合成する（gold 未整備時のブートストラップ）",
    )
    p.add_argument("--min-margin", type=int, default=4, help="合成ペアの最小合計点差")
    p.add_argument(
        "--max-per-meeting", type=int, default=8, help="合成ペアの会議あたり上限"
    )
    p.add_argument(
        "--no-symmetric",
        dest="symmetric",
        action="store_false",
        help="敗者側プロンプトの三つ組を作らない",
    )
    p.set_defaults(symmetric=True)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=3407)
    return p


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
