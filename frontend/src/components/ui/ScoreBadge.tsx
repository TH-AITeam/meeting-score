import { useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { scoreClass } from '../../utils/labels'

const SCORE_TIP = '各軸スコア（0〜3 の整数）× 重み（例: 1.3, 0.8）の合計のため小数になります'

interface Props {
  score: number
}

export default function ScoreBadge({ score }: Props) {
  const ref = useRef<HTMLSpanElement>(null)
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null)

  const handleMouseEnter = () => {
    if (!ref.current) return
    const r = ref.current.getBoundingClientRect()
    setPos({ top: r.top - 8, left: r.left + r.width / 2 })
  }

  return (
    <>
      <span
        ref={ref}
        className={`score-badge ${scoreClass(score)}`}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={() => setPos(null)}
      >
        {score}
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
