import { useMemo, useState } from 'react'
import {
  DndContext,
  KeyboardSensor,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core'
import {
  SortableContext,
  arrayMove,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import type { EvaluatedUtterance } from '../types/meeting'
import { postTopK } from '../api/feedback'
import Toast from './ui/Toast'

interface Props {
  meetingId: string
  /** 元の Top5（保存時の original_top5）。 */
  top5: EvaluatedUtterance[]
  /** 差し替え候補となる全発言。 */
  allUtterances: EvaluatedUtterance[]
  onClose: () => void
}

function snippet(text: string, n = 48): string {
  return text.length > n ? text.slice(0, n) + '…' : text
}

function SortableRow({
  utt,
  rank,
  onRemove,
}: {
  utt: EvaluatedUtterance
  rank: number
  onRemove: () => void
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: utt.utterance_id,
  })
  return (
    <li
      ref={setNodeRef}
      className="top5-edit-row"
      style={{
        transform: CSS.Transform.toString(transform),
        transition,
        opacity: isDragging ? 0.6 : 1,
      }}
    >
      <button
        type="button"
        className="top5-drag-handle"
        aria-label={`${rank}位「${snippet(utt.text, 20)}」をドラッグして並べ替え`}
        {...attributes}
        {...listeners}
      >
        ⠿
      </button>
      <span className="wf-mono top5-rank">{rank}</span>
      <div className="top5-row-body">
        <span className="top5-speaker">{utt.speaker}</span>{' '}
        <span className="wf-mono">{utt.timestamp}</span>
        <div className="top5-row-text">{snippet(utt.text)}</div>
      </div>
      <button
        type="button"
        className="btn ghost sm"
        aria-label={`${rank}位「${snippet(utt.text, 20)}」を Top5 から外す`}
        onClick={onRemove}
      >
        外す
      </button>
    </li>
  )
}

export default function Top5Editor({ meetingId, top5, allUtterances, onClose }: Props) {
  const originalIds = useMemo(() => top5.map((u) => u.utterance_id), [top5])
  const [orderIds, setOrderIds] = useState<string[]>(originalIds)
  const [showPicker, setShowPicker] = useState(false)
  const [query, setQuery] = useState('')
  const [saving, setSaving] = useState(false)
  const [toast, setToast] = useState<string | null>(null)

  const byId = useMemo(() => {
    const m = new Map<string, EvaluatedUtterance>()
    allUtterances.forEach((u) => m.set(u.utterance_id, u))
    top5.forEach((u) => m.set(u.utterance_id, u))
    return m
  }, [allUtterances, top5])

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  )

  const rows = orderIds.map((id) => byId.get(id)).filter((u): u is EvaluatedUtterance => !!u)
  const top5CountValid = orderIds.length === 5 && rows.length === 5 && originalIds.length === 5

  const changed =
    orderIds.length !== originalIds.length || orderIds.some((id, i) => id !== originalIds[i])

  const candidates = useMemo(() => {
    const inList = new Set(orderIds)
    const q = query.trim()
    return allUtterances
      .filter((u) => !inList.has(u.utterance_id))
      .filter((u) => !q || u.text.includes(q) || u.speaker.includes(q))
      .slice(0, 30)
  }, [allUtterances, orderIds, query])

  const handleDragEnd = (e: DragEndEvent) => {
    const { active, over } = e
    if (!over || active.id === over.id) return
    setOrderIds((ids) => {
      const from = ids.indexOf(String(active.id))
      const to = ids.indexOf(String(over.id))
      return from === -1 || to === -1 ? ids : arrayMove(ids, from, to)
    })
  }

  const removeAt = (id: string) => setOrderIds((ids) => ids.filter((x) => x !== id))
  const addUtterance = (id: string) => {
    setOrderIds((ids) => (ids.includes(id) || ids.length >= 5 ? ids : [...ids, id]))
    setShowPicker(false)
    setQuery('')
  }

  const handleSave = async () => {
    if (!top5CountValid) {
      setToast('Top5 は5件で保存してください')
      return
    }
    setSaving(true)
    try {
      const ack = await postTopK({
        meeting_id: meetingId,
        corrected_top5: orderIds,
        original_top5: originalIds,
      })
      setToast(
        ack.generated_pairs > 0
          ? `ありがとうございました（${ack.generated_pairs} 件の比較に反映）`
          : 'ありがとうございました',
      )
      window.setTimeout(onClose, 900)
    } catch {
      setToast('送信に失敗しました。時間をおいて再度お試しください')
      setSaving(false)
    }
  }

  return (
    <div className="feedback-modal-backdrop" role="presentation" onClick={onClose}>
      <div
        className="feedback-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Top5 を直す"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="feedback-modal-head">
          <div className="wf-h3">Top5 を直す</div>
          <button type="button" className="btn ghost sm" aria-label="閉じる" onClick={onClose}>
            ✕
          </button>
        </div>
        <p className="wf-note" style={{ margin: '0 0 8px' }}>
          ドラッグで並べ替え、「外す」と「他の発言から差し替え」で入れ替えできます。
        </p>

        <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
          <SortableContext items={orderIds} strategy={verticalListSortingStrategy}>
            <ul className="top5-edit-list">
              {rows.map((u, i) => (
                <SortableRow
                  key={u.utterance_id}
                  utt={u}
                  rank={i + 1}
                  onRemove={() => removeAt(u.utterance_id)}
                />
              ))}
            </ul>
          </SortableContext>
        </DndContext>

        {rows.length === 0 && (
          <div className="wf-note" style={{ padding: '8px 0' }}>
            発言がありません。下から追加してください。
          </div>
        )}

        <button
          type="button"
          className="btn ghost sm"
          aria-expanded={showPicker}
          disabled={rows.length >= 5}
          title={rows.length >= 5 ? 'Top5 から外してから追加してください' : undefined}
          onClick={() => setShowPicker((v) => !v)}
        >
          ＋ 他の発言から差し替え
        </button>

        {showPicker && (
          <div className="top5-picker">
            <input
              className="feedback-input"
              type="search"
              placeholder="発言・話者で検索"
              aria-label="差し替える発言を検索"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              autoFocus
            />
            <ul className="top5-picker-list">
              {candidates.map((u) => (
                <li key={u.utterance_id}>
                  <button
                    type="button"
                    className="top5-picker-item"
                    disabled={rows.length >= 5}
                    onClick={() => addUtterance(u.utterance_id)}
                    aria-label={`「${snippet(u.text, 24)}」を Top5 に追加`}
                  >
                    <span className="top5-speaker">{u.speaker}</span>{' '}
                    <span className="wf-mono">{u.timestamp}</span>
                    <div className="top5-row-text">{snippet(u.text)}</div>
                  </button>
                </li>
              ))}
              {candidates.length === 0 && (
                <li className="wf-note" style={{ padding: 6 }}>
                  該当する発言がありません
                </li>
              )}
            </ul>
          </div>
        )}

        <div className="feedback-modal-foot">
          <button type="button" className="btn" onClick={onClose} disabled={saving}>
            キャンセル
          </button>
          <button
            type="button"
            className="btn accent"
            onClick={handleSave}
            disabled={saving || !changed || !top5CountValid}
          >
            保存
          </button>
        </div>

        <Toast message={toast} onDone={() => setToast(null)} />
      </div>
    </div>
  )
}
