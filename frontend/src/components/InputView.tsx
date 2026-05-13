import { useEffect, useRef, useState } from 'react'
import { fetchSamples, listMeetings, getMeeting, deleteMeeting } from '../api/client'
import type { MeetingType, SampleFile, SavedMeetingMeta, MeetingSummary } from '../types/meeting'
import { MEETING_TYPE_LABELS, MEETING_TYPE_AXES } from '../types/meeting'

interface Props {
  onAnalyzeSample: (filename: string, meetingType?: MeetingType) => void
  onAnalyzeJson: (data: unknown, meetingType?: MeetingType) => void
  onAnalyzeAudio: (file: File, meetingType?: MeetingType) => void
  onRestore: (data: MeetingSummary) => void
  historyVersion: number
}

const MEETING_TYPES: MeetingType[] = ['decision', 'brainstorming', 'progress', 'retrospective']

type InputMode = 'none' | 'file' | 'paste' | 'audio'

const CONNECTORS = [
  {
    id: 'audio' as const,
    name: '音声ファイル',
    desc: '音声から自動で文字起こし・分析',
    icon: '♪',
    wip: false,
  },
  {
    id: 'video' as const,
    name: '動画ファイル',
    desc: '動画から音声を抽出して分析',
    icon: '▶',
    wip: true,
  },
  {
    id: 'file' as const,
    name: '文字起こしJSON',
    desc: '会議データ .json をアップロード',
    icon: '≡',
    wip: false,
  },
  {
    id: 'paste' as const,
    name: 'テキストペースト',
    desc: 'JSON を直接貼り付け',
    icon: '✎',
    wip: false,
  },
]

const AUDIO_EXTENSIONS = '.wav,.mp3,.m4a,.flac,.ogg'

function formatDate(iso: string) {
  try {
    return new Intl.DateTimeFormat('ja-JP', {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit',
    }).format(new Date(iso))
  } catch {
    return iso
  }
}

export default function InputView({ onAnalyzeSample, onAnalyzeJson, onAnalyzeAudio, onRestore, historyVersion }: Props) {
  const [samples, setSamples] = useState<SampleFile[]>([])
  const [samplesStatus, setSamplesStatus] = useState<'loading' | 'ok' | 'error'>('loading')
  const [mode, setMode] = useState<InputMode>('none')
  const [pasteText, setPasteText] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const audioInputRef = useRef<HTMLInputElement>(null)
  const [meetingType, setMeetingType] = useState<MeetingType | undefined>(undefined)

  const [history, setHistory] = useState<SavedMeetingMeta[]>([])
  const [historyStatus, setHistoryStatus] = useState<'loading' | 'ok' | 'error'>('loading')
  const [confirmId, setConfirmId] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)

  useEffect(() => {
    fetchSamples()
      .then((data) => { setSamples(data); setSamplesStatus('ok') })
      .catch(() => setSamplesStatus('error'))
  }, [])

  useEffect(() => {
    listMeetings()
      .then((data) => { setHistory(data); setHistoryStatus('ok') })
      .catch(() => setHistoryStatus('error'))
  }, [historyVersion])

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    try {
      const text = await file.text()
      onAnalyzeJson(JSON.parse(text) as unknown, meetingType)
    } catch (err) {
      alert(`JSONの読み込みに失敗しました: ${String(err)}`)
    }
    e.target.value = ''
  }

  const handleAudioChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    onAnalyzeAudio(file, meetingType)
    e.target.value = ''
  }

  const handlePasteSubmit = () => {
    try {
      onAnalyzeJson(JSON.parse(pasteText) as unknown, meetingType)
    } catch (err) {
      alert(`JSONのパースに失敗しました: ${String(err)}`)
    }
  }

  const handleRestore = async (id: string) => {
    try {
      const saved = await getMeeting(id)
      onRestore(saved.result as MeetingSummary)
    } catch {
      alert('分析結果の読み込みに失敗しました')
    }
  }

  const handleDelete = async (id: string) => {
    setDeletingId(id)
    setConfirmId(null)
    try {
      await deleteMeeting(id)
      setHistory((prev) => prev.filter((m) => m.id !== id))
    } catch {
      alert('削除に失敗しました')
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <div className="wf-page" style={{ overflow: 'auto' }}>
      <div style={{ padding: 28, maxWidth: 900, margin: '0 auto', width: '100%' }}>
        <div className="wf-h1" style={{ marginBottom: 4 }}>会議ログを取り込む</div>
        <div className="wf-note" style={{ marginBottom: 22 }}>
          取り込み元を選ぶか、サンプルデータで試してください
        </div>

        {/* Meeting type selector */}
        <div style={{ marginBottom: 22 }}>
          <div className="wf-h3" style={{ marginBottom: 8 }}>会議タイプ（任意）</div>
          <div className="wf-note" style={{ marginBottom: 10 }}>
            選択した会議タイプに合わせた重みで評価します。未選択時はデフォルト重みを使用します。
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 10 }}>
            {MEETING_TYPES.map((t) => (
              <div
                key={t}
                className={`wf-box wf-pad${meetingType === t ? ' accent' : ''}`}
                style={{ cursor: 'pointer', padding: '10px 14px' }}
                onClick={() => setMeetingType(meetingType === t ? undefined : t)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{
                    width: 16, height: 16,
                    border: '1.5px solid var(--ink)',
                    borderRadius: '50%',
                    display: 'grid', placeItems: 'center',
                    flexShrink: 0,
                    background: meetingType === t ? 'var(--ink)' : 'transparent',
                  }} />
                  <div>
                    <div className="wf-h3" style={{ fontSize: 13 }}>{MEETING_TYPE_LABELS[t]}</div>
                    <div className="wf-note" style={{ fontSize: 11 }}>{MEETING_TYPE_AXES[t]}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          {meetingType && (
            <div style={{ marginTop: 8, textAlign: 'right' }}>
              <button
                className="btn ghost sm"
                style={{ fontSize: 11, color: 'var(--ink-3)' }}
                onClick={() => setMeetingType(undefined)}
              >
                選択を解除
              </button>
            </div>
          )}
        </div>

        {/* Connector cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 14, marginBottom: 22 }}>
          {CONNECTORS.map((c) => (
            <div
              key={c.id}
              style={{ position: 'relative' }}
            >
              <div
                className={`wf-box wf-pad${!c.wip && mode === c.id ? ' accent' : ''}`}
                style={{ cursor: c.wip ? 'default' : 'pointer', opacity: c.wip ? 0.55 : 1 }}
                onClick={() => {
                  if (c.wip) return
                  const id = c.id as InputMode
                  setMode(mode === id ? 'none' : id)
                  if (c.id === 'file') fileInputRef.current?.click()
                  if (c.id === 'audio') audioInputRef.current?.click()
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
                {c.id === 'audio' && (
                  <div style={{ marginTop: 10, display: 'flex', justifyContent: 'flex-end' }}>
                    <span className="btn sm accent">音声を選択</span>
                  </div>
                )}
                {c.id === 'video' && (
                  <div style={{ marginTop: 10, display: 'flex', justifyContent: 'flex-end' }}>
                    <span className="btn sm">動画を選択</span>
                  </div>
                )}
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

              {c.wip && (
                <div style={{
                  position: 'absolute', inset: 0,
                  borderRadius: 6,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  backgroundImage: 'repeating-linear-gradient(135deg, transparent, transparent 6px, oklch(0.88 0.01 80 / 0.45) 6px, oklch(0.88 0.01 80 / 0.45) 7px)',
                  pointerEvents: 'none',
                }}>
                  <span style={{
                    fontFamily: 'var(--hand)',
                    fontSize: 12,
                    fontWeight: 700,
                    padding: '4px 10px',
                    borderRadius: 999,
                    background: 'var(--ink)',
                    color: 'var(--bg)',
                    letterSpacing: '0.04em',
                    boxShadow: '0 1px 4px oklch(0 0 0 / 0.15)',
                  }}>
                    機能追加中
                  </span>
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
        <input
          ref={audioInputRef}
          type="file"
          accept={AUDIO_EXTENSIONS}
          style={{ display: 'none' }}
          onChange={handleAudioChange}
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
                        onClick={() => onAnalyzeSample(s.filename, meetingType)}
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

        {/* Separator */}
        <div className="sep dashed" style={{ margin: '22px 0' }} />

        {/* History */}
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 10 }}>
          <div className="wf-h3">保存済みの分析結果</div>
          {historyStatus === 'ok' && history.length > 0 && (
            <span className="wf-note">{history.length} 件</span>
          )}
        </div>

        {historyStatus === 'loading' && (
          <div className="wf-note">読み込み中...</div>
        )}
        {historyStatus === 'error' && (
          <div className="wf-note" style={{ color: 'var(--red)' }}>履歴の取得に失敗しました</div>
        )}
        {historyStatus === 'ok' && history.length === 0 && (
          <div className="wf-note">まだ保存された分析結果はありません。分析すると自動で保存されます。</div>
        )}
        {historyStatus === 'ok' && history.length > 0 && (
          <div className="wf-box soft">
            <table className="wf-table">
              <thead>
                <tr>
                  <th>会議タイトル</th>
                  <th>タイプ</th>
                  <th>日時</th>
                  <th>話者</th>
                  <th>発言数</th>
                  <th>スコア</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {history.map((m) => (
                  <>
                    <tr key={m.id}>
                      <td>{m.title}</td>
                      <td style={{ color: 'var(--ink-3)', fontSize: 11, whiteSpace: 'nowrap' }}>
                        {m.meeting_type ? MEETING_TYPE_LABELS[m.meeting_type] : '—'}
                      </td>
                      <td style={{ whiteSpace: 'nowrap', color: 'var(--ink-3)', fontSize: 11 }}>
                        {formatDate(m.created_at)}
                      </td>
                      <td style={{ textAlign: 'center' }}>{m.speaker_count}</td>
                      <td style={{ textAlign: 'center' }}>{m.utterance_count}</td>
                      <td style={{ textAlign: 'center' }}>
                        <span className={`history-score${m.overall_score >= 70 ? ' high' : m.overall_score >= 50 ? ' mid' : ' low'}`}>
                          {m.overall_score.toFixed(1)}
                        </span>
                      </td>
                      <td style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                        <button
                          className="btn sm"
                          style={{ marginRight: 6 }}
                          onClick={() => handleRestore(m.id)}
                        >
                          再表示 →
                        </button>
                        <button
                          className="btn ghost sm"
                          style={{ color: 'var(--ink-3)' }}
                          onClick={() => setConfirmId(confirmId === m.id ? null : m.id)}
                          disabled={deletingId === m.id}
                        >
                          {deletingId === m.id ? '削除中' : '削除'}
                        </button>
                      </td>
                    </tr>
                    {confirmId === m.id && (
                      <tr key={`${m.id}-confirm`}>
                        <td colSpan={7}>
                          <div className="history-confirm">
                            <span>「{m.title}」を削除しますか？</span>
                            <button className="history-confirm-yes" onClick={() => handleDelete(m.id)}>削除する</button>
                            <button className="history-confirm-no" onClick={() => setConfirmId(null)}>キャンセル</button>
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
