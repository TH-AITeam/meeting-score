import { useEffect, useRef, useState } from 'react'
import { fetchSamples } from '../api/client'
import type { SampleFile } from '../types/meeting'

interface Props {
  onAnalyzeSample: (filename: string) => void
  onAnalyzeJson: (data: unknown) => void
}

type InputMode = 'none' | 'file' | 'paste'

const CONNECTORS = [
  {
    id: 'file' as const,
    name: '文字起こしJSON',
    desc: '会議データ .json をアップロード',
    icon: '≡',
  },
  {
    id: 'paste' as const,
    name: 'テキストペースト',
    desc: 'JSON を直接貼り付け',
    icon: '✎',
  },
]

export default function InputView({ onAnalyzeSample, onAnalyzeJson }: Props) {
  const [samples, setSamples] = useState<SampleFile[]>([])
  const [samplesStatus, setSamplesStatus] = useState<'loading' | 'ok' | 'error'>('loading')
  const [mode, setMode] = useState<InputMode>('none')
  const [pasteText, setPasteText] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    fetchSamples()
      .then((data) => { setSamples(data); setSamplesStatus('ok') })
      .catch(() => setSamplesStatus('error'))
  }, [])

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    try {
      const text = await file.text()
      onAnalyzeJson(JSON.parse(text) as unknown)
    } catch (err) {
      alert(`JSONの読み込みに失敗しました: ${String(err)}`)
    }
    e.target.value = ''
  }

  const handlePasteSubmit = () => {
    try {
      onAnalyzeJson(JSON.parse(pasteText) as unknown)
    } catch (err) {
      alert(`JSONのパースに失敗しました: ${String(err)}`)
    }
  }

  return (
    <div className="wf-page" style={{ overflow: 'auto' }}>
      <div style={{ padding: 28, maxWidth: 900, margin: '0 auto', width: '100%' }}>
        <div className="wf-h1" style={{ marginBottom: 4 }}>会議ログを取り込む</div>
        <div className="wf-note" style={{ marginBottom: 22 }}>
          取り込み元を選ぶか、サンプルデータで試してください
        </div>

        {/* Connector cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 14, marginBottom: 22 }}>
          {CONNECTORS.map((c) => (
            <div
              key={c.id}
              className={`wf-box wf-pad${mode === c.id ? ' accent' : ''}`}
              style={{ cursor: 'pointer' }}
              onClick={() => {
                setMode(mode === c.id ? 'none' : c.id)
                if (c.id === 'file') fileInputRef.current?.click()
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{
                  width: 36, height: 36,
                  border: '1.5px solid var(--ink)',
                  borderRadius: 8,
                  display: 'grid', placeItems: 'center',
                  fontFamily: 'var(--hand)', fontSize: 18,
                  background: 'var(--accent-soft)',
                  flexShrink: 0,
                }}>{c.icon}</span>
                <div>
                  <div className="wf-h3">{c.name}</div>
                  <div className="wf-note">{c.desc}</div>
                </div>
              </div>
              {c.id === 'file' && (
                <div style={{ marginTop: 10, display: 'flex', justifyContent: 'flex-end' }}>
                  <span className="btn sm">ファイルを選択</span>
                </div>
              )}
              {c.id === 'paste' && (
                <div style={{ marginTop: 10, display: 'flex', justifyContent: 'flex-end' }}>
                  <span className="btn sm">ペーストする</span>
                </div>
              )}
            </div>
          ))}
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />

        {/* Paste area */}
        {mode === 'paste' && (
          <div className="wf-box wf-pad" style={{ marginBottom: 22 }}>
            <div className="wf-h3" style={{ marginBottom: 8 }}>JSONを貼り付け</div>
            <textarea
              value={pasteText}
              onChange={(e) => setPasteText(e.target.value)}
              placeholder={'{\n  "meeting_id": "m001",\n  "title": "会議名",\n  "goal": "目的",\n  "utterances": [...]\n}'}
              style={{
                width: '100%', height: 200,
                fontFamily: 'ui-monospace, monospace',
                fontSize: 12,
                padding: 10,
                border: '1px solid var(--rule-2)',
                borderRadius: 6,
                background: 'var(--bg)',
                color: 'var(--ink)',
                resize: 'vertical',
              }}
            />
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 10, gap: 8 }}>
              <button className="btn ghost sm" onClick={() => setMode('none')}>キャンセル</button>
              <button
                className="btn accent sm"
                onClick={handlePasteSubmit}
                disabled={!pasteText.trim()}
              >
                分析を開始 →
              </button>
            </div>
          </div>
        )}

        {/* Separator */}
        <div className="sep dashed" style={{ margin: '0 0 22px' }} />

        {/* Sample meetings */}
        <div className="wf-h3" style={{ marginBottom: 10 }}>サンプルデータで試す</div>
        {samplesStatus === 'loading' && (
          <div className="wf-note">読み込み中...</div>
        )}
        {samplesStatus === 'error' && (
          <div className="wf-note" style={{ color: 'var(--red)' }}>
            サンプルの取得に失敗しました
          </div>
        )}
        {samplesStatus === 'ok' && samples.length === 0 && (
          <div className="wf-note">サンプルデータがありません</div>
        )}
        {samplesStatus === 'ok' && samples.length > 0 && (
          <div className="wf-box soft">
            <table className="wf-table">
              <thead>
                <tr>
                  <th>ファイル名</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {samples.map((s) => (
                  <tr key={s.filename}>
                    <td>{s.filename.replace('.json', '')}</td>
                    <td style={{ textAlign: 'right' }}>
                      <button
                        className="btn sm"
                        onClick={() => onAnalyzeSample(s.filename)}
                      >
                        分析する →
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
