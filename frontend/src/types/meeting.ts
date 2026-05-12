export type MeetingType = 'decision' | 'brainstorming' | 'progress' | 'retrospective'

export const MEETING_TYPE_LABELS: Record<MeetingType, string> = {
  decision: '意思決定会議',
  brainstorming: 'ブレスト会議',
  progress: '進捗共有・定例',
  retrospective: '振り返り・レビュー',
}

export const MEETING_TYPE_AXES: Record<MeetingType, string> = {
  decision: '意思決定寄与・根拠性・リスク検知',
  brainstorming: '新規性・論点整理・根拠性',
  progress: 'アクション化・リスク検知・要約',
  retrospective: '根拠性・リスク検知・要約・論点整理',
}

export interface Scores {
  issue_clarification: number
  decision_progress: number
  risk_detection: number
  actionability: number
  groundedness: number
  novelty: number
  summarization: number
}

export interface Penalties {
  duplication: number
  verbosity: number
  off_topic: number
  unsupported_assertion: number
}

export interface EvaluatedUtterance {
  utterance_id: string
  speaker: string
  timestamp: string
  text: string
  speech_type: string
  scores: Scores
  penalties: Penalties
  total_score: number
  reason: string
}

export interface SpeakerSummary {
  speaker: string
  utterance_count: number
  total_contribution_score: number
  average_total_score: number
  average_scores: Scores
  style_label: string
  top_utterances: string[]
}

export interface MeetingSummary {
  meeting_id: string
  title: string
  goal: string
  overall_comment: string
  top_utterances: EvaluatedUtterance[]
  top_issue_clarification: EvaluatedUtterance[]
  top_decision_progress: EvaluatedUtterance[]
  top_risk_detection: EvaluatedUtterance[]
  top_actionability: EvaluatedUtterance[]
  improvement_comments: string[]
  speaker_summaries: SpeakerSummary[]
  evaluated_utterances: EvaluatedUtterance[]
}

export interface SampleFile {
  filename: string
  path: string
}

export type ResultTab = 'summary' | 'timeline' | 'speakers'

export interface SavedMeetingMeta {
  id: string
  title: string
  source_type: string
  meeting_type: MeetingType | null
  created_at: string
  speaker_count: number
  utterance_count: number
  overall_score: number
}

export interface SavedMeeting extends SavedMeetingMeta {
  input: Record<string, unknown>
  result: MeetingSummary
}
