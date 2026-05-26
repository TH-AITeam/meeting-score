#!/usr/bin/env python3
"""ラベル回収 → LoRA 学習データ（messages 形式）組み立て（機械処理パート）。

`build_distill_jobs.py` が出した jobs/<issueID>.json と、教師（Claude/サブ
エージェント）が付けた labels/<issueID>.json を突き合わせ、本番推論と同一の
`user` プロンプト（backend/prompts/utterance_eval.txt 由来）と、本番スキーマ
準拠に正規化した `assistant` JSON を組にして train.jsonl / val.jsonl を書く。

labels/<issueID>.json の形式:
    {
      "meta": {"title","goal","agenda":[...],"decision_points":[...],"meeting_type"},
      "labels": [
        {"order": 3, "speech_type": "...", "scores": {...}, "penalties": {...}, "reason": "..."},
        ...
      ]
    }
  labels[].order は jobs の utterances[].order と対応させること。

train/val は会議単位で分割しリークを防ぐ。

使い方:
    python scripts/lora/assemble_distill_data.py --val-ratio 0.2
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from string import Template

REPO_ROOT = Path(__file__).resolve().parents[2]
DISTILL = REPO_ROOT / "data" / "annotations" / "kokkai" / "distill"
PROMPT_PATH = REPO_ROOT / "backend" / "prompts" / "utterance_eval.txt"

# backend/app/evaluators/prompt.py::_MEETING_TYPE_LABELS と一致させる。
_MEETING_TYPE_LABELS = {
    "decision": "意思決定会議(重視軸: 意思決定寄与・根拠性・リスク検知)",
    "brainstorming": "ブレスト会議(重視軸: 新規性・論点整理・根拠性)",
    "progress": "進捗共有・定例(重視軸: アクション化・リスク検知・要約)",
    "retrospective": "振り返り・レビュー(重視軸: 根拠性・リスク検知・要約・論点整理)",
}

_SCORE_KEYS = [
    "issue_clarification",
    "decision_progress",
    "risk_detection",
    "actionability",
    "groundedness",
    "novelty",
    "summarization",
]
_PENALTY_KEYS = ["duplication", "verbosity", "off_topic", "unsupported_assertion"]
_SPEECH_TYPES = [
    "論点整理", "提案", "質問", "情報共有", "要約",
    "懸念提示", "根拠提示", "意思決定促進", "雑談/脱線",
]


def _clamp(v, lo, hi):
    try:
        v = int(v)
    except (TypeError, ValueError):
        v = 0
    return max(lo, min(hi, v))


def normalize_label(label: dict) -> dict:
    """本番 normalize_result 相当：値域クランプと speech_type 正規化。"""
    scores = {k: _clamp(label.get("scores", {}).get(k, 0), 0, 3) for k in _SCORE_KEYS}
    penalties = {k: _clamp(label.get("penalties", {}).get(k, 0), -3, 0) for k in _PENALTY_KEYS}
    st = label.get("speech_type", "")
    if st not in _SPEECH_TYPES:
        compact = st.replace(" ", "")
        st = next((c for c in _SPEECH_TYPES if c.replace(" ", "") == compact), "情報共有")
    return {
        "speech_type": st,
        "scores": scores,
        "penalties": penalties,
        "reason": str(label.get("reason", "")).strip(),
    }


def build_user_prompt(template: Template, meta: dict, utt: dict) -> str:
    """本番 build_prompt() と同一の user 文字列を再現する。"""
    mt = meta.get("meeting_type", "")
    return template.substitute(
        meeting_type=_MEETING_TYPE_LABELS.get(mt, mt) if mt else "(未指定)",
        meeting_goal=meta.get("goal", "") or "(なし)",
        agenda="、".join(meta.get("agenda", [])) if meta.get("agenda") else "(なし)",
        decision_points="、".join(meta.get("decision_points", []))
        if meta.get("decision_points") else "(なし)",
        current_topic="(未設定)",  # 会議録に逐次トピックが無いため本番同様未設定で統一
        before_utterances=utt["before_text"],
        target_speaker=utt["speaker"],
        target_timestamp=utt["timestamp"],
        target_text=utt["text"],
        after_utterances=utt["after_text"],
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs-dir", default=str(DISTILL / "jobs"))
    p.add_argument("--labels-dir", default=str(DISTILL / "labels"))
    p.add_argument("--out-train", default=str(DISTILL / "train.jsonl"))
    p.add_argument("--out-val", default=str(DISTILL / "val.jsonl"))
    p.add_argument("--val-ratio", type=float, default=0.2, help="検証に回す会議の割合")
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()

    template = Template(PROMPT_PATH.read_text(encoding="utf-8"))
    jobs_dir, labels_dir = Path(args.jobs_dir), Path(args.labels_dir)

    samples_by_meeting: dict[str, list[dict]] = {}
    skipped = 0
    for job_path in sorted(jobs_dir.glob("*.json")):
        issue_id = job_path.stem
        label_path = labels_dir / f"{issue_id}.json"
        if not label_path.exists():
            print(f"  [skip] ラベル未生成: {issue_id}")
            continue
        job = json.loads(job_path.read_text(encoding="utf-8"))
        labelf = json.loads(label_path.read_text(encoding="utf-8"))
        meta = labelf.get("meta", {})
        by_order = {lab["order"]: lab for lab in labelf.get("labels", [])}

        rows = []
        for utt in job["utterances"]:
            lab = by_order.get(utt["order"])
            if lab is None:
                skipped += 1
                continue
            user = build_user_prompt(template, meta, utt)
            assistant = json.dumps(normalize_label(lab), ensure_ascii=False)
            rows.append({"messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]})
        if rows:
            samples_by_meeting[issue_id] = rows

    if not samples_by_meeting:
        print("ラベル付きジョブがありません。labels/ を確認してください。")
        return

    # 会議単位で train/val 分割（リーク防止）。
    meetings = list(samples_by_meeting)
    random.Random(args.seed).shuffle(meetings)
    n_val = max(1, round(len(meetings) * args.val_ratio)) if len(meetings) > 1 else 0
    val_meetings = set(meetings[:n_val])

    train, val = [], []
    for m, rows in samples_by_meeting.items():
        (val if m in val_meetings else train).extend(rows)

    Path(args.out_train).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n", encoding="utf-8"
    )
    if val:
        Path(args.out_val).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in val) + "\n", encoding="utf-8"
        )

    print(f"会議 {len(samples_by_meeting)} 件 / train {len(train)} 件・val {len(val)} 件"
          f"（ラベル欠落 {skipped} 件スキップ）")
    print(f"  -> {args.out_train}")
    if val:
        print(f"  -> {args.out_val}")


if __name__ == "__main__":
    main()
