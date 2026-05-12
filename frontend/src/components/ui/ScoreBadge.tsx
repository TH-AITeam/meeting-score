import { scoreClass } from '../../utils/labels'

const SCORE_TIP = '各軸スコア（0〜3 の整数）× 重み（例: 1.3, 0.8）の合計のため小数になります'

interface Props {
  score: number
}

export default function ScoreBadge({ score }: Props) {
  return (
    <span className="score-tip-wrapper">
      <span className={`score-badge ${scoreClass(score)}`}>{score}</span>
      <span className="score-tip" role="tooltip">{SCORE_TIP}</span>
    </span>
  )
}
