import { useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { scoreClass } from '../../utils/labels'

const SCORE_TIP = '各軸スコア（0〜3 の整数）× 重み（例: 1.3, 0.8）の合計のため小数になります'

interface Props {
  score: number
  large?: boolean
  style?: React.CSSProperties
}

export default function ScoreBadge({ score, large, style }: Props) {
  const ref = useRef<HTMLSpanElement>(null)
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null)

  const handleMouseEnter = () => {
    if (!ref.current) return
    const r = ref.current.getBoundingClientRect()
    setPos({ top: r.top - 8, left: r.left + r.width / 2 })
  }

  const className = large
    ? 'wf-h1'
    : `score-badge ${scoreClass(score)}`

  return (
    <>
      <span
        ref={ref}
        className={className}
        style={{ cursor: 'help', ...style }}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={() => setPos(null)}
      >
        {large ? `${score >= 0 ? '+' : ''}${score.toFixed(1)}` : score}
      </span>
      {pos && createPortal(
        <div className="score-tip" style={{ top: pos.top, left: pos.left }}>
          {SCORE_TIP}
        </div>,
        document.body,
      )}
    </>
  )
}
