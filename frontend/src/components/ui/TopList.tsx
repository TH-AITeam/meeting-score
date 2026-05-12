import type { EvaluatedUtterance } from '../../types/meeting'
import ScoreBadge from './ScoreBadge'

interface Props {
  items: EvaluatedUtterance[]
}

export default function TopList({ items }: Props) {
  if (items.length === 0) return <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>該当なし</p>

  return (
    <ul className="top-list">
      {items.map((u, i) => (
        <li key={u.utterance_id}>
          <div className="top-rank">{i + 1}</div>
          <div className="top-content">
            <span className="top-speaker">{u.speaker}</span>
            {' '}
            <span className="utterance-time">{u.timestamp}</span>
            {' '}
            <ScoreBadge score={u.total_score} />
            <div className="top-text">{u.text}</div>
            <div className="top-meta">{u.reason}</div>
          </div>
        </li>
      ))}
    </ul>
  )
}
