import { useEffect, useState } from 'react'

const MESSAGES = [
  '発言データを読み込んでいます...',
  '文脈ウィンドウを生成しています...',
  '各発言をAIで評価しています...',
  'スコアを計算しています...',
  '集計レポートを生成しています...',
]

export default function LoadingView() {
  const [msgIdx, setMsgIdx] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setMsgIdx((i) => (i + 1) % MESSAGES.length), 3000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="loading-screen">
      <div className="spinner" />
      <div className="wf-h3">{MESSAGES[msgIdx]}</div>
      <div className="wf-note">会議ログを分析しています。しばらくお待ちください。</div>
    </div>
  )
}
