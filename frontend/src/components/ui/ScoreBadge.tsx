import { scoreClass } from '../../utils/labels'

interface Props {
  score: number
}

export default function ScoreBadge({ score }: Props) {
  return <span className={`score-badge ${scoreClass(score)}`}>{score}</span>
}
