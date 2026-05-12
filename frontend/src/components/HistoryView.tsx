import { useEffect, useState } from 'react'
import { listMeetings, getMeeting, deleteMeeting } from '../api/client'
import type { SavedMeetingMeta, MeetingSummary } from '../types/meeting'

interface Props {
  onRestore: (data: MeetingSummary) => void
}

export default function HistoryView({ onRestore }: Props) {
  const [meetings, setMeetings] = useState<SavedMeetingMeta[]>([])
  const [loading, setLoading] = useState(true)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [confirmId, setConfirmId] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    try {
      const data = await listMeetings()
      setMeetings(data)
    } catch {
      // silently fail
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const handleRestore = async (id: string) => {
    try {
      const saved = await getMeeting(id)
      onRestore(saved.result as MeetingSummary)
    } catch {
      alert('分析結果の読み込みに失敗しました')
    }
  }

  const handleDeleteConfirm = (id: string) => setConfirmId(id)

  const handleDelete = async (id: string) => {
    setDeletingId(id)
    setConfirmId(null)
    try {
      await deleteMeeting(id)
      setMeetings((prev) => prev.filter((m) => m.id !== id))
    } catch {
      alert('削除に失敗しました')
    } finally {
      setDeletingId(null)
    }
  }

  const formatDate = (iso: string) => {
    try {
      return new Intl.DateTimeFormat('ja-JP', {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit',
      }).format(new Date(iso))
    } catch {
      return iso
    }
  }

  if (loading) {
    return (
      <div className="history-empty">
        <p className="ink-3">読み込み中...</p>
      </div>
    )
  }

  if (meetings.length === 0) {
    return (
      <div className="history-empty">
        <p className="ink-3">保存済みの会議分析はまだありません。</p>
        <p className="ink-3" style={{ marginTop: 4 }}>分析完了後に「保存」ボタンで保存できます。</p>
      </div>
    )
  }

  return (
    <div className="history-root">
      <div className="history-header">
        <h2 className="history-title">保存済み会議一覧</h2>
        <span className="ink-3">{meetings.length} 件</span>
      </div>
      <div className="history-list">
        {meetings.map((m) => (
          <div key={m.id} className="history-card">
            <div className="history-card-main">
              <div className="history-card-title">{m.title}</div>
              <div className="history-card-meta">
                <span>{formatDate(m.created_at)}</span>
                <span className="history-dot">·</span>
                <span>{m.speaker_count} 名</span>
                <span className="history-dot">·</span>
                <span>{m.utterance_count} 発言</span>
                <span className="history-dot">·</span>
                <span className={`history-score${m.overall_score >= 70 ? ' high' : m.overall_score >= 50 ? ' mid' : ' low'}`}>
                  スコア {m.overall_score.toFixed(1)}
                </span>
                <span className="history-source">{m.source_type === 'sample' ? 'サンプル' : 'アップロード'}</span>
              </div>
            </div>
            <div className="history-card-actions">
              <button
                className="history-btn-restore"
                onClick={() => handleRestore(m.id)}
              >
                再表示
              </button>
              <button
                className="history-btn-delete"
                onClick={() => handleDeleteConfirm(m.id)}
                disabled={deletingId === m.id}
              >
                {deletingId === m.id ? '削除中...' : '削除'}
              </button>
            </div>

            {confirmId === m.id && (
              <div className="history-confirm">
                <span>「{m.title}」を削除しますか？</span>
                <button className="history-confirm-yes" onClick={() => handleDelete(m.id)}>削除する</button>
                <button className="history-confirm-no" onClick={() => setConfirmId(null)}>キャンセル</button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
