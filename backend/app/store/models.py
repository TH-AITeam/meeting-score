"""保存済み会議のデータモデル"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SavedMeetingMeta(BaseModel):
    """保存済み会議のメタ情報（一覧表示用）"""

    id: str
    title: str
    source_type: str  # "sample" | "upload"
    created_at: str
    speaker_count: int
    utterance_count: int
    overall_score: float


class SavedMeeting(SavedMeetingMeta):
    """保存済み会議の全データ"""

    input: dict = Field(default_factory=dict)
    result: dict = Field(default_factory=dict)
