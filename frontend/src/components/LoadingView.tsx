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

const EXTRACT_MESSAGES = [
  '動画から音声を抽出しています...',
  'ffmpeg.wasm を準備しています (初回のみダウンロード)...',
  '音声を圧縮しています (libopus 32kbps)...',
]

interface Props {
  step?: 'extracting' | 'transcribing' | 'analyzing'
  /** 抽出進捗 0.0 - 1.0 (extracting のときのみ意味を持つ) */
  progress?: number
}

export default function LoadingView({ step, progress }: Props) {
  const messages =
    step === 'extracting' ? EXTRACT_MESSAGES
    : step === 'transcribing' ? TRANSCRIBE_MESSAGES
    : ANALYZE_MESSAGES
  const note =
    step === 'extracting'
      ? 'ブラウザ内で処理しているため、動画はサーバーに送信されません。'
      : step === 'transcribing'
        ? '音声の文字起こしに数分かかる場合があります。'
        : '会議ログを分析しています。しばらくお待ちください。'

  const [msgIdx, setMsgIdx] = useState(0)

  useEffect(() => {
    setMsgIdx(0)
    const id = setInterval(() => setMsgIdx((i) => (i + 1) % messages.length), 3000)
    return () => clearInterval(id)
  }, [step, messages.length])

  const pct =
    step === 'extracting' && typeof progress === 'number'
      ? Math.round(Math.max(0, Math.min(1, progress)) * 100)
      : null

  return (
    <div className="loading-screen">
      <div className="spinner" />
      <div className="wf-h3">{messages[msgIdx]}</div>
      <div className="wf-note">{note}</div>
      {pct !== null && (
        <div className="wf-note" style={{ marginTop: 8, fontVariantNumeric: 'tabular-nums' }}>
          {pct}%
        </div>
      )}
    </div>
  )
}
