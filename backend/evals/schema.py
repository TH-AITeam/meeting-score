"""アノテーション JSONL スキーマ定義 (Issue #5)。

Issue #6 の 3 形式アノテーションを Pydantic で型定義する。

形式:
- tags.jsonl       : 各発言に複数タグを付与（multi-label）
- pairs.jsonl      : ペアワイズ比較 (A_better / B_better / tie)
- top_bottom.jsonl : 会議ごとの top5 / bottom5
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# 許容タグ（Issue #6 に基づく）
ANNOTATION_TAGS = (
    "論点設定", "提案", "深掘り質問", "情報提供", "要約", "リスク提示",
    "根拠提示", "アクション化", "決定", "雑談", "重複", "脱線", "上書き",
)

PairwiseWinner = Literal["A_better", "B_better", "tie"]


class TagAnnotation(BaseModel):
    """発言1つに対するタグ付け結果（multi-label）。"""

    meeting_id: str
    utterance_id: str
    tags: list[str] = Field(default_factory=list)
    annotator: str = "unknown"


class PairwiseAnnotation(BaseModel):
    """同一会議内の2発言の比較結果。"""

    meeting_id: str
    utt_a: str
    utt_b: str
    winner: PairwiseWinner
    annotator: str = "unknown"


class TopBottomAnnotation(BaseModel):
    """1会議の top5 / bottom5。"""

    meeting_id: str
    top5: list[str] = Field(default_factory=list)
    bottom5: list[str] = Field(default_factory=list)
    annotator: str = "unknown"


def load_jsonl(path: Path, model_cls: type[BaseModel]) -> list:
    """JSONL ファイルを1行=1モデルで読み込む。

    空行と先頭が `#` のコメント行はスキップする。
    """
    out: list = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                out.append(model_cls.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                msg = f"{path} 行 {line_no} のパース失敗: {exc}"
                raise ValueError(msg) from exc
    return out


def load_tag_annotations(path: Path) -> list[TagAnnotation]:
    return load_jsonl(path, TagAnnotation)


def load_pairwise_annotations(path: Path) -> list[PairwiseAnnotation]:
    return load_jsonl(path, PairwiseAnnotation)


def load_top_bottom_annotations(path: Path) -> list[TopBottomAnnotation]:
    return load_jsonl(path, TopBottomAnnotation)


__all__ = [
    "ANNOTATION_TAGS",
    "PairwiseAnnotation",
    "PairwiseWinner",
    "TagAnnotation",
    "TopBottomAnnotation",
    "load_jsonl",
    "load_pairwise_annotations",
    "load_tag_annotations",
    "load_top_bottom_annotations",
]
