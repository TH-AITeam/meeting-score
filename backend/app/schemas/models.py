"""会議貢献度スコアリング データモデル定義"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 会議タイプ
# ---------------------------------------------------------------------------
class MeetingType(StrEnum):
    DECISION = "decision"
    BRAINSTORMING = "brainstorming"
    PROGRESS = "progress"
    RETROSPECTIVE = "retrospective"


# ---------------------------------------------------------------------------
# 発言タイプ
# ---------------------------------------------------------------------------
class SpeechType(StrEnum):
    ISSUE_CLARIFICATION = "論点整理"
    PROPOSAL = "提案"
    QUESTION = "質問"
    INFO_SHARING = "情報共有"
    SUMMARY = "要約"
    CONCERN = "懸念提示"
    EVIDENCE = "根拠提示"
    DECISION_PUSH = "意思決定促進"
    OFF_TOPIC = "雑談/脱線"


# ---------------------------------------------------------------------------
# 発言傾向ラベル
# ---------------------------------------------------------------------------
class StyleLabel(StrEnum):
    ORGANIZER = "整理型"
    PROPOSER = "提案型"
    CAUTIOUS = "警戒型"
    DRIVER = "推進型"


# ---------------------------------------------------------------------------
# 入力モデル
# ---------------------------------------------------------------------------
class Utterance(BaseModel):
    """発言1件"""

    utterance_id: str
    speaker: str
    timestamp: str
    text: str
    topic: str | None = None


class TopicTransition(BaseModel):
    """議題遷移マーカー: この utterance_id 以降の議題を示す"""

    utterance_id: str
    topic: str


class MeetingInput(BaseModel):
    """会議データ入力"""

    meeting_id: str
    org_id: str | None = None
    title: str
    goal: str
    agenda: list[str] = Field(default_factory=list)
    decision_points: list[str] = Field(default_factory=list)
    topic_transitions: list[TopicTransition] = Field(default_factory=list)
    utterances: list[Utterance]
    meeting_type: MeetingType | None = None


# ---------------------------------------------------------------------------
# 評価結果モデル
# ---------------------------------------------------------------------------
class Scores(BaseModel):
    """評価軸スコア (各 0-3)"""

    issue_clarification: int = Field(0, ge=0, le=3)
    decision_progress: int = Field(0, ge=0, le=3)
    risk_detection: int = Field(0, ge=0, le=3)
    actionability: int = Field(0, ge=0, le=3)
    groundedness: int = Field(0, ge=0, le=3)
    novelty: int = Field(0, ge=0, le=3)
    summarization: int = Field(0, ge=0, le=3)


class Penalties(BaseModel):
    """減点軸 (各 -3〜0)"""

    duplication: int = Field(0, ge=-3, le=0)
    verbosity: int = Field(0, ge=-3, le=0)
    off_topic: int = Field(0, ge=-3, le=0)
    unsupported_assertion: int = Field(0, ge=-3, le=0)
    override: int = Field(0, ge=-3, le=0)


class EvaluatedUtterance(BaseModel):
    """評価済み発言"""

    utterance_id: str
    speaker: str
    timestamp: str
    text: str
    speech_type: str
    scores: Scores
    penalties: Penalties
    total_score: float
    reason: str


# ---------------------------------------------------------------------------
# 集計モデル
# ---------------------------------------------------------------------------
class AverageScores(BaseModel):
    """話者別の軸別平均スコア"""

    issue_clarification: float = 0.0
    decision_progress: float = 0.0
    risk_detection: float = 0.0
    actionability: float = 0.0
    groundedness: float = 0.0
    novelty: float = 0.0
    summarization: float = 0.0


class SpeakerSummary(BaseModel):
    """話者別サマリー"""

    speaker: str
    utterance_count: int
    total_contribution_score: float
    average_total_score: float
    average_scores: AverageScores
    style_label: str
    top_utterances: list[str] = Field(default_factory=list)


class MeetingSummary(BaseModel):
    """会議全体サマリー"""

    meeting_id: str
    title: str
    goal: str
    overall_comment: str = ""
    top_utterances: list[EvaluatedUtterance] = Field(default_factory=list)
    top_issue_clarification: list[EvaluatedUtterance] = Field(default_factory=list)
    top_decision_progress: list[EvaluatedUtterance] = Field(default_factory=list)
    top_risk_detection: list[EvaluatedUtterance] = Field(default_factory=list)
    top_actionability: list[EvaluatedUtterance] = Field(default_factory=list)
    improvement_comments: list[str] = Field(default_factory=list)
    speaker_summaries: list[SpeakerSummary] = Field(default_factory=list)
    evaluated_utterances: list[EvaluatedUtterance] = Field(default_factory=list)
