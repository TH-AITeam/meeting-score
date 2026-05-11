import type { Scores } from '../../types/meeting'
import { AXIS_LABELS_FULL } from '../../utils/labels'

const AXES = [
  'issue_clarification',
  'decision_progress',
  'risk_detection',
  'actionability',
  'groundedness',
  'novelty',
  'summarization',
] as const

interface Props {
  scores: Scores
}

export default function BarChart({ scores }: Props) {
  return (
    <div className="bar-chart">
      {AXES.map((key) => {
        const val = scores[key] ?? 0
        const pct = (val / 3) * 100
        return (
          <div className="bar-row" key={key}>
            <div className="bar-label">{AXIS_LABELS_FULL[key]}</div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${pct}%` }} />
            </div>
            <div className="bar-value">{val.toFixed(1)}</div>
          </div>
        )
      })}
    </div>
  )
}
