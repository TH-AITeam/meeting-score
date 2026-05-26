import { useState } from 'react'
import { AXIS_LABELS_FULL, PENALTY_LABELS } from '../utils/labels'
import { postAxisFlag, type FlagDirection } from '../api/feedback'
import Toast from './ui/Toast'

interface Props {
  meetingId: string
  utteranceId: string
}

// 軸セレクタの選択肢（採点軸 7 + 減点軸 4）。value は API に送るキー、label は表示名。
const AXIS_OPTIONS: { value: string; label: string }[] = [
  ...Object.entries(AXIS_LABELS_FULL).map(([value, label]) => ({ value, label })),
  ...Object.entries(PENALTY_LABELS).map(([value, label]) => ({ value, label })),
]

const COMMENT_MAX = 80

export default function FeedbackFlag({ meetingId, utteranceId }: Props) {
  const [open, setOpen] = useState(false)
  const [direction, setDirection] = useState<FlagDirection | null>(null)
  const [axes, setAxes] = useState<Set<string>>(new Set())
  const [comment, setComment] = useState('')
  const [saving, setSaving] = useState(false)
  const [toast, setToast] = useState<string | null>(null)

  const reset = () => {
    setDirection(null)
    setAxes(new Set())
    setComment('')
  }

  const close = () => {
    setOpen(false)
    reset()
  }

  const toggleAxis = (value: string) => {
    setAxes((prev) => {
      const next = new Set(prev)
      if (next.has(value)) next.delete(value)
      else next.add(value)
      return next
    })
  }

  const handleSubmit = async () => {
    if (!direction) return
    setSaving(true)
    try {
      // API は 1 軸/行なので、複数選択時は軸ごとに 1 件ずつ送る。
      const selectedAxes = axes.size > 0 ? [...axes] : [null]
      const trimmedComment = comment.trim() || null
      await Promise.all(
        selectedAxes.map((axis) =>
          postAxisFlag({
            meeting_id: meetingId,
            utterance_id: utteranceId,
            direction,
            axis,
            comment: trimmedComment,
          }),
        ),
      )
      setToast(
        selectedAxes.length > 1
          ? `ありがとうございました（${selectedAxes.length} 件に反映）`
          : 'ありがとうございました',
      )
      setOpen(false)
      reset()
    } catch {
      setToast('送信に失敗しました')
    } finally {
      setSaving(false)
    }
  }

  return (
    // 親（発言カード）の onClick(=選択) に伝播させない
    <span className="feedback-flag" onClick={(e) => e.stopPropagation()}>
      <button
        type="button"
        className="feedback-flag-trigger"
        aria-label="このスコアを評価（過大/過小評価を報告）"
        aria-expanded={open}
        onClick={() => (open ? close() : setOpen(true))}
      >
        ⚑
      </button>

      {open && (
        <div
          className="feedback-popover"
          role="dialog"
          aria-label="スコアへのフィードバック"
        >
          <div className="wf-label" style={{ marginBottom: 6 }}>
            このスコアは…
          </div>
          <div className="feedback-dir-toggle" role="radiogroup" aria-label="評価の方向">
            <button
              type="button"
              role="radio"
              aria-checked={direction === 'overrated'}
              className={`btn sm${direction === 'overrated' ? ' accent' : ''}`}
              onClick={() => setDirection('overrated')}
            >
              過大評価 ↓
            </button>
            <button
              type="button"
              role="radio"
              aria-checked={direction === 'underrated'}
              className={`btn sm${direction === 'underrated' ? ' accent' : ''}`}
              onClick={() => setDirection('underrated')}
            >
              過小評価 ↑
            </button>
          </div>

          <div className="wf-label" style={{ margin: '8px 0 4px' }}>
            該当する軸（任意・複数可）
          </div>
          <div className="feedback-axis-grid">
            {AXIS_OPTIONS.map((opt) => (
              <label key={opt.value} className="feedback-axis-opt">
                <input
                  type="checkbox"
                  checked={axes.has(opt.value)}
                  onChange={() => toggleAxis(opt.value)}
                />
                <span className="wf-note">{opt.label}</span>
              </label>
            ))}
          </div>

          <div className="wf-label" style={{ margin: '8px 0 4px' }}>
            コメント（任意・{COMMENT_MAX}字まで）
          </div>
          <textarea
            className="feedback-input"
            rows={2}
            maxLength={COMMENT_MAX}
            value={comment}
            aria-label="コメント"
            onChange={(e) => setComment(e.target.value)}
          />

          <div className="feedback-popover-foot">
            <button type="button" className="btn sm" onClick={close} disabled={saving}>
              キャンセル
            </button>
            <button
              type="button"
              className="btn accent sm"
              onClick={handleSubmit}
              disabled={saving || !direction}
            >
              送信
            </button>
          </div>
        </div>
      )}

      <Toast message={toast} onDone={() => setToast(null)} />
    </span>
  )
}
