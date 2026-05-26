#!/usr/bin/env python3
"""国会会議録 → 蒸留ラベリング用「ジョブ」生成（機械処理パート）。

`data/kokkai/meeting_*/part_*.jsonl`（生会議録）を読み、会議（issueID）ごとに
発言をパース・ノイズ除去し、教師 LLM（= Claude 本体 / サブエージェント）が
ラベル付けするための構造化ジョブ JSON を 1 会議 1 ファイルで書き出す。

各ジョブには次が入る:
  - header: 会議名 / 院 / 日付（メタ抽出プロンプト 01 の埋め込み用）
  - transcript: 実質発言を speechOrder 順に連結した会議全文（メタ抽出用）
  - utterances: 採点対象の発言リスト。各要素に before/after 文脈を整形済みで持つ
                （教師プロンプト 02 と本番 build_prompt() の両方をそのまま組める）

ラベル付けの後段は `assemble_distill_data.py` が回収して train/val を作る。

使い方:
    python scripts/lora/build_distill_jobs.py \
        --src data/kokkai/meeting_20250522_20260522 \
        --out-dir data/annotations/kokkai/distill/jobs \
        --max-meetings 10 --context 3
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO_ROOT / "data" / "kokkai" / "meeting_20250522_20260522"
DEFAULT_OUT = REPO_ROOT / "data" / "annotations" / "kokkai" / "distill" / "jobs"

# 発言冒頭の「○委員長（藤川政人君）　」のような話者マーカーを除去する。
_SPEAKER_PREFIX = re.compile(r"^○[^　\n]*[　\s]+")

# 純粋な議事進行・採決宣言など、実質的でない定型句。これらだけで構成される
# 発言は採点対象から除外する（README のフィルタ方針）。
_PROCEDURAL_PATTERNS = [
    "ただいまから",
    "これより会議を開きます",
    "開会いたします",
    "御異議ないと認め",
    "異議ありませんか",
    "異議ございませんか",
    "異議なし",
    "挙手多数",
    "挙手少数",
    "よって、そのように決定",
    "さよう決定いたします",
    "指名いたします",
    "お諮りいたします",
    "本日はこれにて散会",
    "これにて散会",
    "休憩いたします",
    "再開いたします",
    "速記を",
]

# 氏名のみの指名「○○君。」等の極端に短い発言を弾く下限（空白除去後）。
_MIN_CHARS = 40


def clean_text(raw: str) -> str:
    """改行・全角空白を整理し、話者マーカー prefix を落とす。"""
    text = raw.replace("\r\n", "\n").strip()
    text = _SPEAKER_PREFIX.sub("", text)
    # 連続する空白・改行を 1 個に畳む（プロンプト埋め込み時の見やすさ用）。
    text = re.sub(r"[ \t　]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def is_substantive(speaker: str, text: str) -> bool:
    """採点対象にすべき実質発言かを判定する（保守的なヒューリスティック）。"""
    if speaker == "会議録情報":
        return False
    compact = text.replace(" ", "").replace("\n", "")
    if len(compact) < _MIN_CHARS:
        return False
    # 定型進行の語が含まれ、かつ短め（=進行のみ）なら除外。
    if len(compact) < 120 and any(p in compact for p in _PROCEDURAL_PATTERNS):
        return False
    return True


def build_sequence(speech_record: list[dict]) -> list[dict]:
    """会議録情報を除いた発言列を speechOrder 順で整形して返す。"""
    seq = []
    for s in sorted(speech_record, key=lambda x: x.get("speechOrder", 0)):
        if s.get("speaker") == "会議録情報":
            continue
        text = clean_text(s.get("speech", ""))
        if not text:
            continue
        seq.append(
            {
                "order": s.get("speechOrder", 0),
                "speaker": s.get("speaker", "不明"),
                "text": text,
            }
        )
    return seq


def fmt_context(items: list[dict]) -> str:
    """before/after 文脈を本番 _format_utterances と同じ体裁で整形。"""
    if not items:
        return "(なし)"
    # timestamp は会議録に存在しないため speechOrder を代用（train/inference 一致）。
    return "\n".join(f"[{u['order']}] {u['speaker']}: {u['text']}" for u in items)


def make_job(meeting: dict, context: int) -> dict | None:
    seq = build_sequence(meeting.get("speechRecord", []))
    if not seq:
        return None

    # メタ抽出用の会議全文（実質発言を連結）。長すぎる場合は呼び出し側で調整。
    transcript = "\n".join(f"{u['speaker']}: {u['text']}" for u in seq)

    utterances = []
    for i, u in enumerate(seq):
        if not is_substantive(u["speaker"], u["text"]):
            continue
        before = seq[max(0, i - context) : i]
        after = seq[i + 1 : i + 1 + context]
        utterances.append(
            {
                "order": u["order"],
                "speaker": u["speaker"],
                "timestamp": str(u["order"]),
                "text": u["text"],
                "before_text": fmt_context(before),
                "after_text": fmt_context(after),
            }
        )

    if not utterances:
        return None

    return {
        "issueID": meeting["issueID"],
        "nameOfHouse": meeting.get("nameOfHouse", ""),
        "nameOfMeeting": meeting.get("nameOfMeeting", ""),
        "issue": meeting.get("issue", ""),
        "date": meeting.get("date", ""),
        "transcript": transcript,
        "n_utterances": len(utterances),
        "utterances": utterances,
    }


def load_meeting_rows(src: Path) -> list[dict]:
    """会議データ JSONL を読む。src は単一ファイルまたは part_*.jsonl ディレクトリ。"""
    paths = sorted(src.glob("part_*.jsonl")) if src.is_dir() else [src]
    rows: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def select_meetings(rows: list[dict], max_meetings: int, min_sub: int, max_sub: int) -> list[dict]:
    """委員会審議を中心に、多様性を持たせて会議を選ぶ。

    nameOfMeeting でグループ化し各種別から順に 1 件ずつ拾うことで、
    特定委員会・本会議に偏らないパイロット集合を作る。
    """
    # まず実質発言数を見積もってフィルタ。
    candidates: dict[str, list[tuple[int, dict]]] = {}
    for r in rows:
        seq = build_sequence(r.get("speechRecord", []))
        n_sub = sum(1 for u in seq if is_substantive(u["speaker"], u["text"]))
        if min_sub <= n_sub <= max_sub:
            candidates.setdefault(r.get("nameOfMeeting", ""), []).append((n_sub, r))

    # 各種別を発言数の多い順に並べ、ラウンドロビンで多様に選ぶ。
    for lst in candidates.values():
        lst.sort(key=lambda x: -x[0])
    ordered_types = sorted(candidates, key=lambda k: -len(candidates[k]))

    selected: list[dict] = []
    idx = 0
    while len(selected) < max_meetings and any(candidates.values()):
        progressed = False
        for t in ordered_types:
            if candidates[t]:
                selected.append(candidates[t].pop(0)[1])
                progressed = True
                if len(selected) >= max_meetings:
                    break
        if not progressed:
            break
        idx += 1
    return selected


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(DEFAULT_SRC), help="会議データ JSONL ファイル、または part_*.jsonl を含むディレクトリ")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--max-meetings", type=int, default=10, help="抽出する会議数（パイロット）")
    p.add_argument("--context", type=int, default=3, help="前後文脈の件数 N")
    p.add_argument("--min-sub", type=int, default=10, help="実質発言数の下限")
    p.add_argument("--max-sub", type=int, default=45, help="実質発言数の上限")
    args = p.parse_args()

    rows = load_meeting_rows(Path(args.src))
    print(f"読み込み: {len(rows)} 会議")

    chosen = select_meetings(rows, args.max_meetings, args.min_sub, args.max_sub)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_utt = 0
    written = 0
    for m in chosen:
        job = make_job(m, args.context)
        if job is None:
            continue
        (out_dir / f"{job['issueID']}.json").write_text(
            json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        total_utt += job["n_utterances"]
        written += 1
        print(f"  {job['issueID']}  {job['nameOfHouse']}{job['nameOfMeeting']} {job['issue']}  発言{job['n_utterances']}件")

    print(f"\n書き出し: {written} 会議 / 採点対象 {total_utt} 発言 -> {out_dir}")


if __name__ == "__main__":
    main()
