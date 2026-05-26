#!/usr/bin/env python3
"""フィードバック → 学習データ正規化スクリプト (Issue #80)。

フィードバック DB (#78: feedback_pairwise / feedback_topk / feedback_axis_flag) を
**組織別**に、後段の DPO 学習 (#15 形式) と 軸重み回帰 (#16 形式) が読める JSONL に
変換する。組織をまたいだ混入をコードレベルで禁止する（`org_id: str` 単一・ループ不可）。

出力:
    data/feedback/{org_id}/dpo/v{n}/{train,val}.jsonl   # DPO (#15 形式)
    data/feedback/{org_id}/weights/v{n}/pairs.jsonl     # 重み回帰 (#16 形式)
    data/feedback/{org_id}/stats.md                     # 統計サマリ

設計上の要点:
  - ``consent_to_train=false`` の組織は **早期 return** で何も出力しない。
  - PII: 話者名は会議内で匿名 ID (A/B/C...) に置換。自由記述コメント
    (axis_flag.comment) は学習データに **含めない**（Phase2 で NER 検討）。
  - ``feedback_topk`` の Top入り×入替のペア展開は、収集 API (#78) が保存時に
    ``feedback_pairwise(source='top5_reorder')`` へ既に展開済み。二重計上を避けるため
    本スクリプトは **topk を再展開しない**（feedback_pairwise をそのまま読む）。
  - ``feedback_axis_flag`` は同会議の上位/下位発言とのペアに合成する
    (``source='axis_flag_synthesized'``)。
  - chosen/rejected の評価 JSON は、保存済み会議 (data/stored_meetings) の
    過去評価ログを使う（Issue #80 が許容する経路。GPU 無しで回せる）。

使い方:
    python scripts/build_feedback_dataset.py --org-id org_001
    python scripts/build_feedback_dataset.py --org-id org_001 --since 2026-01-01T00:00:00
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_sft_dataset import validate_schema  # noqa: E402

from app.evaluators.prompt import (  # noqa: E402
    _MEETING_TYPE_LABELS,
    PROMPT_PATH,
    RESPONSE_SCHEMA,
)

# 軸キーは RESPONSE_SCHEMA（単一の真実）から導出し、スキーマ変更
# (例: #91 の penalties.override 追加) に追従する。build_sft_dataset の
# 定数は #13 時点で固定されており追従しないため、ここでは使わない。
SCORE_KEYS = list(RESPONSE_SCHEMA["properties"]["scores"]["properties"])
PENALTY_KEYS = list(RESPONSE_SCHEMA["properties"]["penalties"]["properties"])

DEFAULT_MEETINGS_DIR = REPO_ROOT / "data" / "stored_meetings"
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "feedback"


# ---------------------------------------------------------------------------
# 会議の評価ログ（保存済み会議 = 過去の評価結果）。chosen/rejected と prompt の素。
# ---------------------------------------------------------------------------
@dataclass
class MeetingEval:
    meeting_id: str
    goal: str
    meeting_type: str
    agenda: list[str]
    decision_points: list[str]
    # 出現順（timestamp 昇順）の発言。各要素は評価結果込み。
    utterances: list[dict[str, Any]]

    def by_id(self) -> dict[str, dict[str, Any]]:
        return {u["utterance_id"]: u for u in self.utterances}

    def anon_map(self) -> dict[str, str]:
        """会議内の話者名 → 匿名 ID (A, B, C ...)。出現順で安定に割り当てる。"""
        mapping: dict[str, str] = {}
        for u in self.utterances:
            spk = u.get("speaker", "")
            if spk not in mapping:
                mapping[spk] = chr(ord("A") + len(mapping))
        return mapping


def meeting_eval_from_saved(saved: dict[str, Any]) -> MeetingEval | None:
    """保存済み会議 JSON (SavedMeeting) から MeetingEval を作る。"""
    result = saved.get("result", {})
    utts = result.get("evaluated_utterances", [])
    meeting_id = result.get("meeting_id") or saved.get("id")
    if not meeting_id or not utts:
        return None
    # timestamp 昇順（無ければ入力順）で文脈を組めるように並べる。
    ordered = sorted(utts, key=lambda u: str(u.get("timestamp", "")))
    inp = saved.get("input", {})
    return MeetingEval(
        meeting_id=meeting_id,
        goal=result.get("goal", "") or inp.get("goal", ""),
        meeting_type=saved.get("meeting_type") or inp.get("meeting_type") or "",
        agenda=inp.get("agenda", []) or [],
        decision_points=inp.get("decision_points", []) or [],
        utterances=ordered,
    )


def load_meeting_index(meetings_dir: Path) -> dict[str, MeetingEval]:
    """stored_meetings ディレクトリを meeting_id でインデックス化する。"""
    index: dict[str, MeetingEval] = {}
    if not meetings_dir.is_dir():
        return index
    for path in meetings_dir.glob("*.json"):
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        me = meeting_eval_from_saved(saved)
        if me is not None:
            index[me.meeting_id] = me
    return index


# ---------------------------------------------------------------------------
# 評価 JSON / プロンプト構築（PII 匿名化込み）
# ---------------------------------------------------------------------------
def eval_json_of(utt: dict[str, Any]) -> dict[str, Any]:
    """保存済み評価結果から本番スキーマ準拠の 4 キー評価 JSON を作る。"""
    return {
        "speech_type": utt.get("speech_type", "情報共有"),
        "scores": {k: int(utt.get("scores", {}).get(k, 0)) for k in SCORE_KEYS},
        "penalties": {k: int(utt.get("penalties", {}).get(k, 0)) for k in PENALTY_KEYS},
        "reason": str(utt.get("reason", "")),
    }


def total_of(utt: dict[str, Any]) -> int:
    s = sum(int(utt.get("scores", {}).get(k, 0)) for k in SCORE_KEYS)
    p = sum(int(utt.get("penalties", {}).get(k, 0)) for k in PENALTY_KEYS)
    return s + p


def _fmt_context(utts: list[dict[str, Any]], anon: dict[str, str]) -> str:
    if not utts:
        return "(なし)"
    return "\n".join(
        f"[{u.get('timestamp', '')}] {anon.get(u.get('speaker', ''), '?')}: {u.get('text', '')}"
        for u in utts
    )


def build_prompt(
    meeting: MeetingEval,
    target_id: str,
    anon: dict[str, str],
    template: Template,
    context: int = 3,
) -> str | None:
    """対象発言の本番相当プロンプトを、話者名を匿名化して構築する。"""
    idx = next(
        (i for i, u in enumerate(meeting.utterances) if u["utterance_id"] == target_id),
        -1,
    )
    if idx == -1:
        return None
    target = meeting.utterances[idx]
    before = meeting.utterances[max(0, idx - context) : idx]
    after = meeting.utterances[idx + 1 : idx + 1 + context]
    mt = meeting.meeting_type
    return template.substitute(
        meeting_type=_MEETING_TYPE_LABELS.get(mt, mt) if mt else "(未指定)",
        meeting_goal=meeting.goal,
        agenda="、".join(meeting.agenda) if meeting.agenda else "(なし)",
        decision_points="、".join(meeting.decision_points) if meeting.decision_points else "(なし)",
        current_topic="(未設定)",
        before_utterances=_fmt_context(before, anon),
        target_speaker=anon.get(target.get("speaker", ""), "?"),
        target_timestamp=target.get("timestamp", ""),
        target_text=target.get("text", ""),
        after_utterances=_fmt_context(after, anon),
    )


# ---------------------------------------------------------------------------
# ペアの正規化（pairwise / axis_flag 合成）
# ---------------------------------------------------------------------------
@dataclass
class NormalizedPair:
    meeting_id: str
    winner_id: str
    loser_id: str
    source: str  # 'manual_pair' | 'top5_reorder' | 'axis_flag_synthesized'
    created_at: str = ""


def pairwise_to_normalized(rows: list[dict[str, Any]]) -> list[NormalizedPair]:
    """feedback_pairwise 行 → NormalizedPair。tie は捨てる。"""
    out: list[NormalizedPair] = []
    for r in rows:
        w = str(r.get("winner", "")).strip().upper()
        if w == "A":
            winner, loser = r["utt_a"], r["utt_b"]
        elif w == "B":
            winner, loser = r["utt_b"], r["utt_a"]
        else:  # tie ほか
            continue
        out.append(
            NormalizedPair(
                meeting_id=r["meeting_id"],
                winner_id=winner,
                loser_id=loser,
                source=r.get("source", "manual_pair"),
                created_at=str(r.get("created_at", "")),
            )
        )
    return out


def axis_flags_to_normalized(
    flags: list[dict[str, Any]],
    meeting_index: dict[str, MeetingEval],
    max_pairs: int,
) -> list[NormalizedPair]:
    """axis_flag を同会議の上位/下位発言とのペアに合成する。

    overrated: 当該発言は下位であるべき → 上位スコア群を winner、当該を loser。
    underrated: 当該発言は上位であるべき → 当該を winner、下位スコア群を loser。
    """
    out: list[NormalizedPair] = []
    for f in flags:
        meeting = meeting_index.get(f["meeting_id"])
        if meeting is None:
            continue
        by_id = meeting.by_id()
        target = by_id.get(f["utterance_id"])
        if target is None:
            continue
        t_total = total_of(target)
        others = [u for u in meeting.utterances if u["utterance_id"] != f["utterance_id"]]
        direction = f.get("direction")
        if direction == "overrated":
            # 当該より高得点の発言を winner にして「当該は下位」を学習
            partners = sorted(
                (u for u in others if total_of(u) > t_total),
                key=total_of,
                reverse=True,
            )[:max_pairs]
            out.extend(
                NormalizedPair(
                    meeting.meeting_id,
                    u["utterance_id"],
                    f["utterance_id"],
                    "axis_flag_synthesized",
                    str(f.get("created_at", "")),
                )
                for u in partners
            )
        elif direction == "underrated":
            partners = sorted((u for u in others if total_of(u) < t_total), key=total_of)[
                :max_pairs
            ]
            out.extend(
                NormalizedPair(
                    meeting.meeting_id,
                    f["utterance_id"],
                    u["utterance_id"],
                    "axis_flag_synthesized",
                    str(f.get("created_at", "")),
                )
                for u in partners
            )
    return out


# ---------------------------------------------------------------------------
# DPO / 重み回帰レコード生成
# ---------------------------------------------------------------------------
@dataclass
class BuildOutput:
    dpo: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # meeting_id -> records
    weights: list[dict[str, Any]] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)


def build_records(
    org_id: str,
    pairs: list[NormalizedPair],
    meeting_index: dict[str, MeetingEval],
    template: Template,
) -> BuildOutput:
    """NormalizedPair から DPO レコードと重み回帰 pairs を作る（org_id 単一）。"""
    out = BuildOutput()
    for pair in pairs:
        meeting = meeting_index.get(pair.meeting_id)
        if meeting is None:
            out.counts["meeting_not_found"] += 1
            continue
        by_id = meeting.by_id()
        win, lose = by_id.get(pair.winner_id), by_id.get(pair.loser_id)
        if win is None or lose is None:
            out.counts["utterance_not_found"] += 1
            continue

        anon = meeting.anon_map()
        prompt = build_prompt(meeting, pair.winner_id, anon, template)
        if prompt is None:
            out.counts["prompt_failed"] += 1
            continue

        chosen = eval_json_of(win)
        rejected = eval_json_of(lose)
        # スキーマ検証（不正は捨てる）
        if not validate_schema(chosen, RESPONSE_SCHEMA) or not validate_schema(
            rejected, RESPONSE_SCHEMA
        ):
            out.counts["schema_violation"] += 1
            continue

        dpo_rec = {
            "prompt": prompt,
            "chosen": json.dumps(chosen, ensure_ascii=False),
            "rejected": json.dumps(rejected, ensure_ascii=False),
            "meta": {
                "org_id": org_id,
                "source": pair.source,
                "meeting_id": pair.meeting_id,
                "created_at": pair.created_at,
            },
        }
        out.dpo.setdefault(pair.meeting_id, []).append(dpo_rec)

        # 重み回帰 (#16 形式): 軸スコアベクトル + winner（A_better 固定: utt_a=winner）
        out.weights.append(
            {
                "meeting_id": pair.meeting_id,
                "utt_a": pair.winner_id,
                "utt_b": pair.loser_id,
                "winner": "A_better",
                "scores_a": chosen["scores"],
                "scores_b": rejected["scores"],
                "source": pair.source,
            }
        )
        out.counts[f"pair:{pair.source}"] += 1
    return out


def split_train_val(
    by_meeting: dict[str, list[dict[str, Any]]], val_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import random

    meetings = list(by_meeting)
    random.Random(seed).shuffle(meetings)
    n_val = round(len(meetings) * val_ratio)
    if val_ratio > 0 and len(meetings) > 1:
        n_val = max(1, min(n_val, len(meetings) - 1))
    val_set = set(meetings[:n_val])
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for m, recs in by_meeting.items():
        (val if m in val_set else train).extend(recs)
    return train, val


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    path.write_text(body + ("\n" if body else ""), encoding="utf-8")


def render_stats(org_id: str, out: BuildOutput, n_train: int, n_val: int) -> str:
    source_counts = {k.split(":", 1)[1]: v for k, v in out.counts.items() if k.startswith("pair:")}
    lines = [
        f"# フィードバック学習データ統計: {org_id} (Issue #80)",
        "",
        "自動生成 (`scripts/build_feedback_dataset.py`)。",
        "",
        f"- DPO: train {n_train} / val {n_val} (会議 {len(out.dpo)} 件)",
        f"- 重み回帰 pairs: {len(out.weights)} 件",
        "",
        "## source 内訳",
        "",
        "| source | ペア数 |",
        "|---|---:|",
    ]
    for src, n in sorted(source_counts.items()):
        lines.append(f"| {src} | {n} |")
    skipped = {k: v for k, v in out.counts.items() if not k.startswith("pair:")}
    if skipped:
        lines += ["", "## スキップ内訳", "", "| 理由 | 件数 |", "|---|---:|"]
        for k, v in sorted(skipped.items()):
            lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# DB 読み出し（org_id 単一・ループ禁止）
# ---------------------------------------------------------------------------
def _to_dict(obj: Any, fields: list[str]) -> dict[str, Any]:
    return {f: getattr(obj, f) for f in fields}


def load_feedback_from_db(
    org_id: str, since: datetime | None
) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]]]:
    """DB から org の (consent, pairwise行, axis_flag行) を読む。

    戻り値の consent が False（or 組織不在）の場合、呼び出し側は早期 return する。
    """
    from sqlmodel import Session, select

    from app.store.db import get_engine
    from app.store.feedback_models import (
        AxisFlagFeedback,
        Organization,
        PairwiseFeedback,
    )

    with Session(get_engine()) as session:
        org = session.get(Organization, org_id)
        if org is None or not org.consent_to_train:
            return (False, [], [])

        def _filter(stmt: Any, model: Any) -> Any:
            stmt = stmt.where(model.org_id == org_id)
            if since is not None:
                stmt = stmt.where(model.created_at >= since)
            return stmt

        pw = session.exec(_filter(select(PairwiseFeedback), PairwiseFeedback)).all()
        ax = session.exec(_filter(select(AxisFlagFeedback), AxisFlagFeedback)).all()

    pairwise = [
        _to_dict(r, ["meeting_id", "utt_a", "utt_b", "winner", "source", "created_at"]) for r in pw
    ]
    # 自由記述コメント(comment)は意図的に取り込まない（PII / Issue #80）。
    axis = [
        _to_dict(r, ["meeting_id", "utterance_id", "direction", "axis", "created_at"]) for r in ax
    ]
    return (True, pairwise, axis)


# ---------------------------------------------------------------------------
# オーケストレーション（org_id: str 単一。複数組織のループは設計上不可）
# ---------------------------------------------------------------------------
def build_for_org(
    org_id: str,
    *,
    consent: bool,
    pairwise_rows: list[dict[str, Any]],
    axis_rows: list[dict[str, Any]],
    meeting_index: dict[str, MeetingEval],
    out_root: Path,
    version: str,
    max_pairs_per_feedback: int,
    val_ratio: float,
    seed: int,
) -> dict[str, Any]:
    """単一組織のフィードバックを DPO / 重み回帰 JSONL に正規化して書き出す。

    DB 非依存（行データを引数で受ける）。テストはここを直接叩く。
    """
    out_dir = out_root / org_id
    if not consent:
        # consent_to_train=false / 組織不在 → 何も出力しない（早期 return）。
        return {
            "org_id": org_id,
            "skipped": "no_consent",
            "dpo_train": 0,
            "dpo_val": 0,
            "weights": 0,
        }

    template = Template(PROMPT_PATH.read_text(encoding="utf-8"))
    pairs = pairwise_to_normalized(pairwise_rows)
    pairs += axis_flags_to_normalized(axis_rows, meeting_index, max_pairs_per_feedback)

    out = build_records(org_id, pairs, meeting_index, template)
    train, val = split_train_val(out.dpo, val_ratio, seed)

    write_jsonl(out_dir / "dpo" / version / "train.jsonl", train)
    write_jsonl(out_dir / "dpo" / version / "val.jsonl", val)
    write_jsonl(out_dir / "weights" / version / "pairs.jsonl", out.weights)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.md").write_text(
        render_stats(org_id, out, len(train), len(val)), encoding="utf-8"
    )

    return {
        "org_id": org_id,
        "dpo_train": len(train),
        "dpo_val": len(val),
        "weights": len(out.weights),
        "counts": dict(out.counts),
        "out_dir": str(out_dir),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    since = datetime.fromisoformat(args.since) if args.since else None
    consent, pairwise_rows, axis_rows = load_feedback_from_db(args.org_id, since)
    meeting_index = load_meeting_index(Path(args.meetings_dir))
    summary = build_for_org(
        args.org_id,
        consent=consent,
        pairwise_rows=pairwise_rows,
        axis_rows=axis_rows,
        meeting_index=meeting_index,
        out_root=Path(args.out_dir),
        version=args.version,
        max_pairs_per_feedback=args.max_pairs_per_feedback,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if summary.get("skipped"):
        print(f"[{args.org_id}] consent_to_train=false または組織不在のため出力なし")
    else:
        print(
            f"[{args.org_id}] DPO train {summary['dpo_train']} / val {summary['dpo_val']} "
            f"/ weights {summary['weights']}  内訳: {summary.get('counts', {})}"
        )
        print(f"  -> {summary['out_dir']}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--org-id", required=True, help="対象組織 ID (単一・必須)")
    p.add_argument("--since", default=None, help="この日時以降のフィードバックのみ (ISO 8601)")
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_ROOT),
        help="出力ルート (配下に {org_id}/)",
    )
    p.add_argument(
        "--meetings-dir",
        default=str(DEFAULT_MEETINGS_DIR),
        help="保存済み会議ディレクトリ",
    )
    p.add_argument(
        "--version",
        default="v1",
        help="出力バージョン (dpo/weights のサブディレクトリ)",
    )
    p.add_argument(
        "--max-pairs-per-feedback",
        type=int,
        default=20,
        help="axis_flag 1 件から合成するペアの上限 (爆発抑制)",
    )
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=3407)
    return p


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
