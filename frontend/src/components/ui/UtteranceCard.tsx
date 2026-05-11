import type { EvaluatedUtterance } from '../../types/meeting'
import { AXIS_LABELS, PENALTY_LABELS } from '../../utils/labels'
import ScoreBadge from './ScoreBadge'

interface Props {
  utterance: EvaluatedUtterance
  highlight?: boolean
  showHeader?: boolean
}

export default function UtteranceCard({ utterance: u, highlight = false, showHeader = true }: Props) {
  const scoreItems = (Object.entries(u.scores) as [string, number][])
    .filter(([, v]) => v > 0)
    .map(([k, v]) => (
      <span className="score-item" key={k}>{AXIS_LABELS[k] ?? k}: {v}</span>
    ))

  const penaltyItems = (Object.entries(u.penalties) as [string, number][])
    .filter(([, v]) => v < 0)
    .map(([k, v]) => (
      <span className="score-item-penalty" key={k}>{PENALTY_LABELS[k] ?? k}: {v}</span>
    ))

  return (
    <div className={`utterance-card${highlight ? ' highlight' : ''}`}>
      {showHeader && (
        <div className="utterance-header">
          <span className="utterance-speaker">{u.speaker}</span>
          <span className="utterance-time">{u.timestamp}</span>
          <span className="type-label" data-type={u.speech_type}>{u.speech_type}</span>
          <ScoreBadge score={u.total_score} />
        </div>
      )}
      {!showHeader && (
        <div className="utterance-header">
          <span className="utterance-time">{u.timestamp}</span>
          <span className="type-label" data-type={u.speech_type}>{u.speech_type}</span>
          <ScoreBadge score={u.total_score} />
        </div>
      )}
      <div className="utterance-text">{u.text}</div>
      <div className="utterance-reason">{u.reason}</div>
      {(scoreItems.length > 0 || penaltyItems.length > 0) && (
        <div className="scores-grid">
          {scoreItems}
          {penaltyItems}
        </div>
      )}
    </div>
  )
}
