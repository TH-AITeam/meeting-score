import { useEffect, useState } from 'react'

const ANALYZE_MESSAGES = [
  '発言データを読み込んでいます...',
  '文脈ウィンドウを生成しています...',
  '各発言をAIで評価しています...',
  'スコアを計算しています...',
  '集計レポートを生成しています...',
]

const TRANSCRIBE_MESSAGES = [
  '音声ファイルをアップロードしています...',
  '音声を文字起こし中...',
  '話者を識別しています...',
  '文字起こし結果を整形しています...',
]

interface Props {
  step?: 'transcribing' | 'analyzing'
}

export default function LoadingView({ step }: Props) {
  const messages = step === 'transcribing' ? TRANSCRIBE_MESSAGES : ANALYZE_MESSAGES
  const note = step === 'transcribing'
    ? '音声の文字起こしに数分かかる場合があります。'
    : '会議ログを分析しています。しばらくお待ちください。'

  const [msgIdx, setMsgIdx] = useState(0)

  useEffect(() => {
    setMsgIdx(0)
    const id = setInterval(() => setMsgIdx((i) => (i + 1) % messages.length), 3000)
    return () => clearInterval(id)
  }, [step, messages.length])

  return (
    <div className="loading-screen">
      <div className="spinner" />
      <div className="wf-h3">{messages[msgIdx]}</div>
      <div className="wf-note">{note}</div>
    </div>
  )
}
