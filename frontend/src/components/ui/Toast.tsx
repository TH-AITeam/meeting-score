import { useEffect } from 'react'

interface Props {
  message: string | null
  onDone: () => void
  durationMs?: number
}

// 控えめな通知。ゲーミフィケーション（バッジ・効果音・派手な演出）は持たせない。
export default function Toast({ message, onDone, durationMs = 2400 }: Props) {
  useEffect(() => {
    if (!message) return
    const id = window.setTimeout(onDone, durationMs)
    return () => window.clearTimeout(id)
  }, [message, durationMs, onDone])

  if (!message) return null

  return (
    <div className="feedback-toast" role="status" aria-live="polite">
      {message}
    </div>
  )
}
