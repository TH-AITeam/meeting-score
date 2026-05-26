import { useEffect, useState } from 'react'
import type { MediaStats } from '../App'

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
  /**
   * 動画 → 音声抽出後のサイズ比較 (Issue #68)。
   * extracting 完了後 (transcribing / analyzing 中) に表示してアップロード効率化を可視化する。
   */
  mediaStats?: MediaStats
}

function formatSize(mb: number): string {
  if (mb < 1) return `${(mb * 1024).toFixed(0)} KB`
  return `${mb.toFixed(1)} MB`
}

export default function LoadingView({ step, progress, mediaStats }: Props) {
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

  // 抽出が完了し transcribing / analyzing に進んだら、削減効果を可視化する。
  const showMediaStats = mediaStats && step !== 'extracting'

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
      {showMediaStats && (
        <div
          aria-label="アップロード効率化の結果"
          style={{
            marginTop: 14,
            padding: '8px 14px',
            border: '1px solid var(--rule-2)',
            borderRadius: 6,
            background: 'var(--accent-soft)',
            fontVariantNumeric: 'tabular-nums',
            display: 'inline-flex',
            alignItems: 'center',
            gap: 8,
            fontSize: 12,
          }}
        >
          <span className="wf-note" style={{ margin: 0 }}>動画</span>
          <strong>{formatSize(mediaStats.originalMB)}</strong>
          <span aria-hidden>→</span>
          <span className="wf-note" style={{ margin: 0 }}>抽出音声</span>
          <strong>{formatSize(mediaStats.extractedMB)}</strong>
          <span style={{ color: 'var(--accent)', fontWeight: 700 }}>
            (-{mediaStats.reductionPct.toFixed(1)}%)
          </span>
        </div>
      )}
    </div>
  )
}
