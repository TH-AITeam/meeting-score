import { useEffect, useState } from 'react'
import { getFeedbackStats, type FeedbackStats as Stats } from '../api/feedback'

// 段階の意味（Epic #77）。順位・バッジ等のゲーミフィケーションは出さず、淡々と表示する。
const STAGE_LABEL: Record<number, string> = {
  0: 'デフォルト重みで採点中',
  1: '組織別の重みを更新中',
  2: '組織別モデルで採点中',
}

export default function FeedbackStats() {
  const [stats, setStats] = useState<Stats | null>(null)

  useEffect(() => {
    let active = true
    getFeedbackStats()
      .then((s) => active && setStats(s))
      .catch(() => active && setStats(null))
    return () => {
      active = false
    }
  }, [])

  if (!stats) return null

  const next =
    stats.pairs_to_next_stage != null
      ? `／ 次の段階まで ${stats.pairs_to_next_stage} 件`
      : ''

  return (
    <div className="feedback-stats wf-note" aria-live="polite">
      貴社のフィードバック: 比較 {stats.n_pairwise} 件
      {' '}
      <span className="feedback-stage">段階 {stats.stage}（{STAGE_LABEL[stats.stage] ?? '—'}）</span>
      {' '}
      {next}
    </div>
  )
}
