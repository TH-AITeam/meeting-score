export const AXIS_LABELS: Record<string, string> = {
  issue_clarification: '論点整理',
  decision_progress: '意思決定',
  risk_detection: 'リスク',
  actionability: 'アクション',
  groundedness: '根拠性',
  novelty: '新規性',
  summarization: '要約整理',
}

export const AXIS_LABELS_FULL: Record<string, string> = {
  issue_clarification: '論点整理',
  decision_progress: '意思決定寄与',
  risk_detection: 'リスク検知',
  actionability: 'アクション化',
  groundedness: '根拠性',
  novelty: '新規性',
  summarization: '要約整理',
}

export const PENALTY_LABELS: Record<string, string> = {
  duplication: '重複',
  verbosity: '冗長',
  off_topic: '脱線',
  unsupported_assertion: '根拠薄',
  override: '上書き',
}

export function scoreClass(score: number): string {
  if (score >= 6) return 'score-high'
  if (score >= 3) return 'score-mid'
  if (score < 0) return 'score-negative'
  return 'score-low'
}
