#!/usr/bin/env python3
"""SFT 用データセット構築パイプライン (Issue #13)。

distilled tier (`jobs/` + 教師ラベル `labels/`) と gold tier (人手アノテ #5) を統合し、
**本番推論と同一の `user` プロンプト**（`backend/prompts/utterance_eval.txt` 由来）と
**本番スキーマ準拠の `assistant` JSON**（`prompt.normalize_result` 相当）を組にした
messages 形式 JSONL を、**会議単位**で train/val/test に分割して書き出す。

メッセージ形式は本番推論に合わせ `user` + `assistant` のみ（system ロールは持たない）。
train/inference のプロンプト一致を最優先する（ズレると SFT/LoRA の効果が落ちる）。

完了条件のうち「train >= 5000 件」は教師ラベリング / gold 収集（データ収集タスク）に
依存する。本スクリプトは利用可能なラベル全件をパイプライン処理する。現状データ
（distill 841 件・gold 未生成）では 5000 件に満たないが、データが増えれば同コマンドで
そのまま規模拡大できる。

使い方:
    python scripts/build_sft_dataset.py \
        --distill-dir data/annotations/kokkai/distill \
        --out-dir data/sft/v1 \
        --val-ratio 0.15 --test-ratio 0.15
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
# 本番のプロンプト・スキーマ・正規化を単一の真実として流用する。
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.evaluators.prompt import (  # noqa: E402  (sys.path 追加後の import)
    PROMPT_PATH,
    RESPONSE_SCHEMA,
    _MEETING_TYPE_LABELS,
    normalize_result,
)

SCORE_KEYS = [
    "issue_clarification",
    "decision_progress",
    "risk_detection",
    "actionability",
    "groundedness",
    "novelty",
    "summarization",
]
PENALTY_KEYS = ["duplication", "verbosity", "off_topic", "unsupported_assertion"]


@dataclass
class Sample:
    """SFT 1 サンプル（1 発言）。"""

    meeting_id: str
    utterance_id: str
    source: str  # "gold" | "distilled"
    user: str
    assistant: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": self.user},
                {
                    "role": "assistant",
                    "content": json.dumps(self.assistant, ensure_ascii=False),
                },
            ],
            "meta": {
                "source": self.source,
                "meeting_id": self.meeting_id,
                "utterance_id": self.utterance_id,
            },
        }


@dataclass
class BuildResult:
    by_meeting: dict[str, list[Sample]] = field(default_factory=dict)
    reject_counts: Counter = field(default_factory=Counter)
    source_meetings: dict[str, str] = field(
        default_factory=dict
    )  # meeting_id -> source


# ---------------------------------------------------------------------------
# プロンプト構築（本番 build_prompt と同一の user 文字列を job から再現する）
# ---------------------------------------------------------------------------
def build_user_prompt(template: Template, meta: dict, utt: dict) -> str:
    mt = meta.get("meeting_type", "")
    return template.substitute(
        meeting_type=_MEETING_TYPE_LABELS.get(mt, mt) if mt else "(未指定)",
        meeting_goal=meta.get("goal", ""),
        agenda="、".join(meta.get("agenda", [])) if meta.get("agenda") else "(なし)",
        decision_points="、".join(meta.get("decision_points", []))
        if meta.get("decision_points")
        else "(なし)",
        current_topic="(未設定)",  # 会議録に逐次トピックが無いため本番同様未設定で統一
        before_utterances=utt["before_text"],
        target_speaker=utt["speaker"],
        target_timestamp=utt["timestamp"],
        target_text=utt["text"],
        after_utterances=utt["after_text"],
    )


def normalized_assistant(raw_label: dict) -> dict[str, Any]:
    """教師/人手ラベルを本番スキーマ準拠の 4 キー dict に正規化する。"""
    r = normalize_result(raw_label)
    return {
        "speech_type": r.speech_type,
        "scores": r.scores.model_dump(),
        "penalties": r.penalties.model_dump(),
        "reason": r.reason,
    }


# ---------------------------------------------------------------------------
# reject loop（JSON Schema 違反 / 合計点が極端 / reason が空 を捨てる）
# ---------------------------------------------------------------------------
def validate_schema(obj: Any, schema: dict) -> bool:
    """RESPONSE_SCHEMA に対する最小限の JSON Schema 検証（依存追加なし）。

    type=object/integer/string、enum、minimum/maximum、required、
    additionalProperties=false を解釈する。RESPONSE_SCHEMA の構造を網羅する。
    """
    t = schema.get("type")
    if t == "object":
        if not isinstance(obj, dict):
            return False
        props = schema.get("properties", {})
        if any(req not in obj for req in schema.get("required", [])):
            return False
        if schema.get("additionalProperties") is False and any(
            k not in props for k in obj
        ):
            return False
        return all(validate_schema(obj[k], props[k]) for k in obj if k in props)
    if t == "integer":
        # bool は int のサブクラスなので明示的に弾く
        if isinstance(obj, bool) or not isinstance(obj, int):
            return False
        if "minimum" in schema and obj < schema["minimum"]:
            return False
        return not ("maximum" in schema and obj > schema["maximum"])
    if t == "string":
        if not isinstance(obj, str):
            return False
        return not ("enum" in schema and obj not in schema["enum"])
    return True


def classify_reject(
    raw_label: dict, assistant: dict, *, keep_extreme: bool = False
) -> str | None:
    """サンプルを捨てるべきか判定する。捨てる場合は理由文字列、残す場合は None。"""
    if not str(raw_label.get("reason", "")).strip():
        return "empty_reason"
    if not validate_schema(assistant, RESPONSE_SCHEMA):
        return "schema_violation"
    if not keep_extreme:
        score_sum = sum(assistant["scores"].values())
        penalty_sum = sum(assistant["penalties"].values())
        # 全軸 0 かつ減点も 0 = 「貢献も問題も皆無」は矛盾的・低シグナルとして除外
        if score_sum == 0 and penalty_sum == 0:
            return "extreme_all_zero"
        # 全軸満点も実装的に不自然（教師の振り切り）として除外
        if all(v == 3 for v in assistant["scores"].values()):
            return "extreme_all_max"
    return None


# ---------------------------------------------------------------------------
# ローダ（distilled / gold は同じ jobs+labels 構造を共有する）
# ---------------------------------------------------------------------------
def load_tier(
    jobs_dir: Path,
    labels_dir: Path,
    source: str,
    template: Template,
    result: BuildResult,
    *,
    keep_extreme: bool = False,
) -> None:
    """jobs/<id>.json と labels/<id>.json を突き合わせてサンプル化し result に蓄積する。

    labels/<id>.json 形式: {"meta": {...}, "labels": [{"order": int, "scores": {...}, ...}]}
    """
    if not jobs_dir.is_dir() or not labels_dir.is_dir():
        return
    for job_path in sorted(jobs_dir.glob("*.json")):
        meeting_id = job_path.stem
        label_path = labels_dir / f"{meeting_id}.json"
        if not label_path.exists():
            continue
        job = json.loads(job_path.read_text(encoding="utf-8"))
        labelf = json.loads(label_path.read_text(encoding="utf-8"))
        meta = labelf.get("meta", {})
        by_order = {lab["order"]: lab for lab in labelf.get("labels", [])}

        samples: list[Sample] = []
        for utt in job.get("utterances", []):
            raw_label = by_order.get(utt["order"])
            if raw_label is None:
                result.reject_counts["missing_label"] += 1
                continue
            assistant = normalized_assistant(raw_label)
            reason = classify_reject(raw_label, assistant, keep_extreme=keep_extreme)
            if reason is not None:
                result.reject_counts[reason] += 1
                continue
            samples.append(
                Sample(
                    meeting_id=meeting_id,
                    utterance_id=str(utt["order"]),
                    source=source,
                    user=build_user_prompt(template, meta, utt),
                    assistant=assistant,
                )
            )
        if samples:
            result.by_meeting[meeting_id] = samples
            result.source_meetings[meeting_id] = source


# ---------------------------------------------------------------------------
# 会議単位 train/val/test 分割（発言単位だとリークするため会議で分ける）
# ---------------------------------------------------------------------------
def split_meetings(
    meetings: list[str], val_ratio: float, test_ratio: float, seed: int
) -> dict[str, set[str]]:
    """会議 ID を train/val/test の互いに素な集合に分ける。"""
    shuffled = list(meetings)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_test = round(n * test_ratio)
    n_val = round(n * val_ratio)
    # 会議が複数あれば val/test を最低 1 件確保しつつ train を空にしない
    if n > 2:
        if test_ratio > 0:
            n_test = max(1, n_test)
        if val_ratio > 0:
            n_val = max(1, n_val)
    n_val = min(n_val, max(0, n - 1))
    n_test = min(n_test, max(0, n - 1 - n_val))
    test_set = set(shuffled[:n_test])
    val_set = set(shuffled[n_test : n_test + n_val])
    train_set = set(shuffled[n_test + n_val :])
    return {"train": train_set, "val": val_set, "test": test_set}


def collect_split(
    by_meeting: dict[str, list[Sample]], meeting_sets: dict[str, set[str]]
) -> dict[str, list[Sample]]:
    out: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for split, meetings in meeting_sets.items():
        for m in meetings:
            out[split].extend(by_meeting.get(m, []))
    return out


# ---------------------------------------------------------------------------
# 統計サマリ
# ---------------------------------------------------------------------------
def compute_stats(
    splits: dict[str, list[Sample]],
    meeting_sets: dict[str, set[str]],
    reject_counts: Counter,
) -> dict[str, Any]:
    all_samples = [s for rows in splits.values() for s in rows]
    score_dist: dict[str, Counter] = {k: Counter() for k in SCORE_KEYS}
    penalty_dist: dict[str, Counter] = {k: Counter() for k in PENALTY_KEYS}
    speech_type_dist: Counter = Counter()
    source_dist: Counter = Counter()
    for s in all_samples:
        for k in SCORE_KEYS:
            score_dist[k][s.assistant["scores"][k]] += 1
        for k in PENALTY_KEYS:
            penalty_dist[k][s.assistant["penalties"][k]] += 1
        speech_type_dist[s.assistant["speech_type"]] += 1
        source_dist[s.source] += 1
    return {
        "n_rows": {sp: len(rows) for sp, rows in splits.items()},
        "n_meetings": {sp: len(ms) for sp, ms in meeting_sets.items()},
        "score_dist": {k: dict(sorted(c.items())) for k, c in score_dist.items()},
        "penalty_dist": {k: dict(sorted(c.items())) for k, c in penalty_dist.items()},
        "speech_type_dist": dict(speech_type_dist.most_common()),
        "source_dist": dict(source_dist),
        "reject_counts": dict(reject_counts),
    }


def render_stats_md(stats: dict[str, Any]) -> str:
    lines = ["# SFT データ統計サマリ (Issue #13)", ""]
    lines.append("自動生成 (`scripts/build_sft_dataset.py`)。手で編集しないこと。\n")

    total_rows = sum(stats["n_rows"].values())
    lines.append("## 件数")
    lines.append("")
    lines.append("| split | 会議数 | 発言(行)数 |")
    lines.append("|---|---:|---:|")
    for sp in ("train", "val", "test"):
        lines.append(f"| {sp} | {stats['n_meetings'][sp]} | {stats['n_rows'][sp]} |")
    lines.append(f"| **合計** | {sum(stats['n_meetings'].values())} | {total_rows} |")
    lines.append("")
    train_rows = stats["n_rows"]["train"]
    gate = "✅ 達成" if train_rows >= 5000 else f"❌ 未達 (あと {5000 - train_rows} 件)"
    lines.append(f"> 完了条件 train >= 5000 件: **{gate}**")
    lines.append("")

    lines.append("## ソース別")
    lines.append("")
    lines.append("| source | 件数 |")
    lines.append("|---|---:|")
    for src, n in sorted(stats["source_dist"].items()):
        lines.append(f"| {src} | {n} |")
    lines.append("")

    lines.append("## speech_type 分布")
    lines.append("")
    lines.append("| speech_type | 件数 |")
    lines.append("|---|---:|")
    for st, n in stats["speech_type_dist"].items():
        lines.append(f"| {st} | {n} |")
    lines.append("")

    lines.append("## 軸スコア分布 (0-3)")
    lines.append("")
    lines.append("| 軸 | 0 | 1 | 2 | 3 |")
    lines.append("|---|---:|---:|---:|---:|")
    for k in SCORE_KEYS:
        d = stats["score_dist"][k]
        lines.append(
            f"| {k} | {d.get(0, 0)} | {d.get(1, 0)} | {d.get(2, 0)} | {d.get(3, 0)} |"
        )
    lines.append("")

    lines.append("## 減点軸分布 (-3-0)")
    lines.append("")
    lines.append("| 軸 | -3 | -2 | -1 | 0 |")
    lines.append("|---|---:|---:|---:|---:|")
    for k in PENALTY_KEYS:
        d = stats["penalty_dist"][k]
        lines.append(
            f"| {k} | {d.get(-3, 0)} | {d.get(-2, 0)} | {d.get(-1, 0)} | {d.get(0, 0)} |"
        )
    lines.append("")

    if stats["reject_counts"]:
        lines.append("## reject 内訳")
        lines.append("")
        lines.append("| 理由 | 件数 |")
        lines.append("|---|---:|")
        for reason, n in sorted(stats["reject_counts"].items()):
            lines.append(f"| {reason} | {n} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def render_readme_md(teacher_model: str, version: str) -> str:
    return f"""# data/sft: SFT 用データセット (Issue #13)

ローカル判断モデルを SFT で適合させる学習データ。`scripts/build_sft_dataset.py`
が生成する。**研究用途**であり、商用展開時は蒸留データの扱い（蒸留元 API の規約）を
別途確認すること。

## 形式

`{version}/{{train,val,test}}.jsonl`。1 行 = 1 発言。本番推論に合わせ `user` +
`assistant` のみ（system ロールは持たない）。

```json
{{"messages": [
   {{"role": "user", "content": "<本番 build_prompt() と同一のプロンプト>"}},
   {{"role": "assistant", "content": "{{\\"speech_type\\":...,\\"scores\\":{{...}},\\"penalties\\":{{...}},\\"reason\\":\\"...\\"}}"}}
 ],
 "meta": {{"source": "gold|distilled", "meeting_id": "...", "utterance_id": "..."}}}}
```

- `user` は `backend/prompts/utterance_eval.txt`（本番 `build_prompt()`）をそのまま使う。
  **train と inference のプロンプトを一致させること**（ズレると SFT/LoRA の効果が激減する）。
- `assistant` は本番 `prompt.normalize_result` 準拠（scores 0〜3 / penalties -3〜0 / speech_type 正規化）。
- chat template は採用ベースモデル (#17) に合わせて学習側で適用する。

## データソース 3 層

| Tier | 教師信号 | 状態 |
|---|---|---|
| distilled | 教師 LLM の出力（`data/annotations/kokkai/distill/`） | 利用可 |
| gold | 人手アノテ (#5) | **保留**。現行 gold アノテ (tags/pairs/top_bottom) は eval ハーネス用で、SFT の `assistant`（軸採点 JSON）形式ではない。#5/#6 が軸採点ラベルを出したら `--gold-dir` 配下に distill と同じ jobs/+labels/ 構造で投入する |
| synthetic | #7 で生成 | 未実装 |

## 蒸留元（教師）モデル

- **{teacher_model}**
- 蒸留元と採用ベースモデル (#17) が同系統だと比較が偏るため、できるだけ系統を変えること。
- 蒸留元 API の利用規約（Terms）を確認すること。

## 分割

train/val/test は **会議単位**で分割しリークを防ぐ（同一会議の発言が複数 split に跨らない）。

## 再生成

```bash
python scripts/build_sft_dataset.py --out-dir data/sft/{version} --val-ratio 0.15 --test-ratio 0.15
```

統計サマリは `docs/sft_data_stats.md` に出力される。

## 完了条件 (#13) と現状

- [ ] `train.jsonl` >= 5000 件 — **教師ラベリング / gold 収集（データ収集タスク）に依存**。
  現状 distill のラベル済み会議数に比例。会議を増やせば同コマンドで規模拡大できる。
- [x] 全件 JSON Schema バリデーション通過（`assistant` を `RESPONSE_SCHEMA` で検証）
- [x] `val`/`test` の会議が `train` に含まれない（会議単位分割）
- [x] 統計サマリ（軸スコア分布・speech_type 分布）を `docs/sft_data_stats.md` に出力
"""


# ---------------------------------------------------------------------------
# オーケストレーション
# ---------------------------------------------------------------------------
def write_jsonl(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(s.to_record(), ensure_ascii=False) for s in samples)
    path.write_text(body + ("\n" if body else ""), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    template = Template(PROMPT_PATH.read_text(encoding="utf-8"))
    result = BuildResult()

    distill_dir = Path(args.distill_dir)
    load_tier(
        distill_dir / "jobs",
        distill_dir / "labels",
        "distilled",
        template,
        result,
        keep_extreme=args.keep_extreme,
    )

    gold_dir = Path(args.gold_dir)
    load_tier(
        gold_dir / "jobs",
        gold_dir / "labels",
        "gold",
        template,
        result,
        keep_extreme=args.keep_extreme,
    )

    if not result.by_meeting:
        print(
            "ラベル付きデータがありません。"
            f"\n  distilled: {distill_dir}/{{jobs,labels}}"
            f"\n  gold:      {gold_dir}/{{jobs,labels}}",
            file=sys.stderr,
        )
        return {}

    meeting_sets = split_meetings(
        list(result.by_meeting), args.val_ratio, args.test_ratio, args.seed
    )
    splits = collect_split(result.by_meeting, meeting_sets)

    out_dir = Path(args.out_dir)
    for sp in ("train", "val", "test"):
        write_jsonl(out_dir / f"{sp}.jsonl", splits[sp])

    stats = compute_stats(splits, meeting_sets, result.reject_counts)
    Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_out).write_text(render_stats_md(stats), encoding="utf-8")
    (out_dir / "README.md").write_text(
        render_readme_md(args.teacher_model, out_dir.name), encoding="utf-8"
    )

    train_n = stats["n_rows"]["train"]
    print(
        f"会議 {len(result.by_meeting)} 件 -> "
        f"train {stats['n_rows']['train']} / val {stats['n_rows']['val']} / "
        f"test {stats['n_rows']['test']} 行"
    )
    print(f"  reject: {dict(result.reject_counts)}")
    print(f"  -> {out_dir}/{{train,val,test}}.jsonl, {out_dir}/README.md")
    print(f"  -> {args.stats_out}")
    if train_n < 5000:
        print(
            f"  注意: train {train_n} 件 < 5000（完了条件未達。ラベリング/gold 収集が必要）"
        )
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--distill-dir",
        default=str(REPO_ROOT / "data" / "annotations" / "kokkai" / "distill"),
        help="distilled tier のルート（配下に jobs/ と labels/）",
    )
    p.add_argument(
        "--gold-dir",
        default=str(REPO_ROOT / "data" / "annotations" / "gold" / "v1"),
        help="gold tier のルート（配下に jobs/ と labels/ があれば取り込む。#5 待ち）",
    )
    p.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "sft" / "v1"))
    p.add_argument("--stats-out", default=str(REPO_ROOT / "docs" / "sft_data_stats.md"))
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument(
        "--keep-extreme",
        action="store_true",
        help="全軸0/全軸満点などの極端ラベルを捨てずに残す",
    )
    p.add_argument(
        "--teacher-model",
        default="Claude (本体/サブエージェントが prompts 01/02 に従い手動ラベリング)",
        help="蒸留元（教師）モデル名。README に出典として記録する",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
