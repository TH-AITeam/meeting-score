"""フィードバック収集 API の入出力スキーマ (Issue #78)

API 層の入力検証は ``Literal`` で厳密に行い、DB 層 (feedback_models) は
CHECK 制約で二重に守る。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Winner = Literal["A", "B", "tie"]
PairwiseSource = Literal["top5_reorder", "manual_pair"]
Direction = Literal["overrated", "underrated"]


class FeedbackPairwise(BaseModel):
    """POST /api/feedback/pairwise のリクエスト。"""

    org_id: str
    meeting_id: str
    utt_a: str
    utt_b: str
    winner: Winner
    source: PairwiseSource = "manual_pair"
    annotator: str | None = None


class FeedbackTopK(BaseModel):
    """POST /api/feedback/topk のリクエスト。

    ``corrected_top5`` は並べ替え後、``original_top5`` は元の Top5。
    サーバ側で差分からペアワイズを自動生成する。
    """

    org_id: str
    meeting_id: str
    corrected_top5: list[str]
    original_top5: list[str]
    annotator: str | None = None


class FeedbackAxisFlag(BaseModel):
    """POST /api/feedback/axis_flag のリクエスト。"""

    org_id: str
    meeting_id: str
    utterance_id: str
    direction: Direction
    axis: str | None = None
    comment: str | None = Field(default=None, max_length=80)
    annotator: str | None = None


class FeedbackAck(BaseModel):
    """書き込み系エンドポイントの共通レスポンス。"""

    id: str
    # topk から自動生成されたペアワイズ件数 (pairwise/axis_flag では 0)
    generated_pairs: int = 0


class FeedbackStats(BaseModel):
    """GET /api/feedback/stats のレスポンス。

    段階は Epic #77 の閾値に従う:
    - 段階 0: 0〜49 ペア (ベースモデル + デフォルト重み)
    - 段階 1: 50 ペア以上 (組織別重みプロファイル)
    - 段階 2: 300 ペア以上 (組織別 LoRA)
    """

    org_id: str
    n_pairwise: int  # feedback_pairwise の総数 (top5_reorder 展開分を含む)
    n_topk: int
    n_axis_flag: int
    stage: int  # 0 | 1 | 2
    next_stage: int | None  # 次の段階 (段階 2 到達後は None)
    pairs_to_next_stage: int | None  # 次段階までの不足ペア数 (段階 2 到達後は None)
