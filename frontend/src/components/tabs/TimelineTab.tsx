import { useState } from 'react'
import type { EvaluatedUtterance, Scores } from '../../types/meeting'
import { AXIS_LABELS, AXIS_LABELS_FULL, PENALTY_LABELS } from '../../utils/labels'

interface Props {
  utterances: EvaluatedUtterance[]
}

const AXIS_COLORS: Record<keyof Scores, string> = {
  issue_clarification: 'accent',
  decision_progress: 'blue',
  risk_detection: 'red',
  actionability: 'green',
  groundedness: '',
  novelty: '',
  summarization: '',
}

const MAIN_AXES: (keyof Scores)[] = ['issue_clarification', 'decision_progress', 'risk_detection', 'actionability']
const SUB_AXES: (keyof Scores)[] = ['groundedness', 'novelty', 'summarization']

function AxisRow({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="axis-row">
      <span className="wf-label" style={{ fontSize: 11 }}>{label}</span>
      <div className={`bar${color ? ' ' + color : ''}`}>
        <i style={{ width: `${Math.min(100, (value / 3) * 100)}%` }} />
      </div>
      <span className="wf-mono" style={{ textAlign: 'right' }}>{value.toFixed(1)}</span>
    </div>
  )
}

function speechTypeTone(u: EvaluatedUtterance): string {
  if (u.total_score < 0) return 'red'
  const s = u.scores
  if (s.decision_progress >= 2 || s.issue_clarification >= 2) return 'accent'
  if (s.risk_detection >= 2) return 'red'
  if (s.actionability >= 2) return 'green'
  return ''
}

function isForward(u: EvaluatedUtterance): boolean {
  return u.total_score >= 4
}

function hasPenalty(u: EvaluatedUtterance): boolean {
  return Object.values(u.penalties).some((v) => v < 0)
}

function penaltyLabel(u: EvaluatedUtterance): string {
  const entry = (Object.entries(u.penalties) as [string, number][]).find(([, v]) => v < 0)
  if (!entry) return ''
  return PENALTY_LABELS[entry[0]] ?? entry[0]
}

export default function TimelineTab({ utterances }: Props) {
  // Unique speakers
  const speakers = [...new Set(utterances.map((u) => u.speaker))]
  // Unique types
  const types = [...new Set(utterances.map((u) => u.speech_type))]

  const [activeSpeakers, setActiveSpeakers] = useState<Set<string>>(new Set(speakers))
  const [activeTypes, setActiveTypes] = useState<Set<string>>(new Set(types))
  const [selected, setSelected] = useState<string | null>(utterances[0]?.utterance_id ?? null)
  const [hovered, setHovered] = useState<string | null>(null)

  const toggleSpeaker = (s: string) => {
    setActiveSpeakers((prev) => {
      const next = new Set(prev)
      if (next.has(s)) next.delete(s); else next.add(s)
      return next
    })
  }

  const toggleType = (t: string) => {
    setActiveTypes((prev) => {
      const next = new Set(prev)
      if (next.has(t)) next.delete(t); else next.add(t)
      return next
    })
  }

  const filtered = utterances.filter(
    (u) => activeSpeakers.has(u.speaker) && activeTypes.has(u.speech_type),
  )

  const focusId = hovered ?? selected
  const focus = utterances.find((u) => u.utterance_id === focusId) ?? utterances[0]

  const penaltyEntries = focus
    ? (Object.entries(focus.penalties) as [string, number][]).filter(([, v]) => v < 0)
    : []

  return (
    <div className="wf-page" style={{ overflow: 'hidden' }}>
      <div style={{ padding: 14, display: 'grid', gridTemplateColumns: '190px 1fr 320px', gap: 12, flex: 1, overflow: 'hidden' }}>

        {/* LEFT: filter */}
        <div className="wf-box wf-pad-s" style={{ overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 0 }}>
          <div className="wf-label" style={{ marginBottom: 8 }}>話者で絞り込み</div>
          {speakers.map((s) => (
            <label
              key={s}
              style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '5px 4px', cursor: 'pointer', borderRadius: 4 }}
            >
              <input
                type="checkbox"
                checked={activeSpeakers.has(s)}
                onChange={() => toggleSpeaker(s)}
              />
              <div className="avatar-sm">{s.slice(0, 2)}</div>
              <div>
                <div style={{ fontSize: 13, fontWeight: 500 }}>{s}</div>
              </div>
            </label>
          ))}

          <div className="sep dashed" />

          <div className="wf-label" style={{ marginBottom: 6 }}>発言タイプ</div>
          {types.map((t) => (
            <label
              key={t}
              style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '3px 0', cursor: 'pointer' }}
            >
              <input
                type="checkbox"
                checked={activeTypes.has(t)}
                onChange={() => toggleType(t)}
              />
              <span className="wf-note">{t}</span>
            </label>
          ))}

          <div className="sep dashed" />

          <div className="wf-label" style={{ marginBottom: 6 }}>表示</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
            <span
              className="chip accent dot clickable"
              onClick={() => {
                const forwardIds = new Set(utterances.filter(isForward).map((u) => u.speaker))
                setActiveSpeakers(forwardIds.size > 0 ? forwardIds : new Set(speakers))
              }}
            >
              前進発言のみ
            </span>
            <span
              className="chip red dot clickable"
              onClick={() => {
                const penaltyIds = new Set(utterances.filter(hasPenalty).map((u) => u.speaker))
                setActiveSpeakers(penaltyIds.size > 0 ? penaltyIds : new Set(speakers))
              }}
            >
              減点ありのみ
            </span>
            <span
              className="chip clickable"
              onClick={() => {
                setActiveSpeakers(new Set(speakers))
                setActiveTypes(new Set(types))
              }}
            >
              すべて表示
            </span>
          </div>

          <div className="sep dashed" />
          <span className="wf-mono">{filtered.length} / {utterances.length} 件</span>
        </div>

        {/* CENTER: chat */}
        <div className="wf-box" style={{ overflow: 'auto', padding: 14 }}>
          {filtered.map((u) => {
            const isSel = selected === u.utterance_id
            const isHov = hovered === u.utterance_id
            const fwd = isForward(u)
            const pen = hasPenalty(u)
            return (
              <div
                key={u.utterance_id}
                onMouseEnter={() => setHovered(u.utterance_id)}
                onMouseLeave={() => setHovered(null)}
                onClick={() => setSelected(u.utterance_id)}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '30px 1fr',
                  gap: 10,
                  marginBottom: 12,
                  padding: 4,
                  borderRadius: 8,
                  background: isSel ? 'var(--bg-2)' : 'transparent',
                  outline: isHov && !isSel ? '1px dashed var(--rule)' : 'none',
                  cursor: 'pointer',
                }}
              >
                <div className="avatar-sm">{u.speaker.slice(0, 2)}</div>
                <div
                  className={`wf-box wf-pad-s${fwd ? ' accent' : pen ? ' red' : ''}`}
                  style={{ borderWidth: isSel ? 2 : 1.5 }}
                >
                  <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 4, flexWrap: 'wrap' }}>
                    <span style={{ fontFamily: 'var(--hand)', fontWeight: 700, fontSize: 13 }}>{u.speaker}</span>
                    <span className={`chip${speechTypeTone(u) ? ' ' + speechTypeTone(u) : ''}`}>{u.speech_type}</span>
                    <span className="wf-mono">{u.timestamp}</span>
                    {fwd && <span className="chip accent dot">前進</span>}
                    {pen && <span className="chip red dot">{penaltyLabel(u)}</span>}
                    <span style={{ marginLeft: 'auto' }} className="wf-mono">
                      {u.total_score >= 0 ? '+' : ''}{u.total_score.toFixed(1)}
                    </span>
                  </div>
                  <div style={{ fontSize: 13 }}>{u.text}</div>
                </div>
              </div>
            )
          })}
          {filtered.length === 0 && (
            <div className="wf-note" style={{ padding: 20 }}>該当する発言がありません</div>
          )}
        </div>

        {/* RIGHT: detail */}
        <div className="wf-box wf-pad" style={{ overflow: 'auto', position: 'relative' }}>
          {focus ? (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span className="wf-label">{hovered ? 'ホバー中' : '選択中'}</span>
                <span className="wf-mono" style={{ marginLeft: 'auto' }}>{focus.utterance_id}</span>
              </div>
              <div style={{ display: 'flex', gap: 6, alignItems: 'center', margin: '6px 0' }}>
                <div className="avatar-sm">{focus.speaker.slice(0, 2)}</div>
                <span style={{ fontFamily: 'var(--hand)', fontWeight: 700 }}>{focus.speaker}</span>
                <span className={`chip${speechTypeTone(focus) ? ' ' + speechTypeTone(focus) : ''}`}>{focus.speech_type}</span>
                <span className="wf-mono">{focus.timestamp}</span>
              </div>
              <div className="wf-box soft wf-pad-s" style={{ background: 'var(--bg-2)', fontSize: 13, marginBottom: 2 }}>
                「{focus.text}」
              </div>

              <div className="sep dashed" />
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <span className="wf-label">総合スコア</span>
                <span className="wf-h1" style={{ color: focus.total_score < 0 ? 'var(--red)' : 'var(--ink)' }}>
                  {focus.total_score >= 0 ? '+' : ''}{focus.total_score.toFixed(1)}
                </span>
              </div>

              <div className="wf-label" style={{ marginTop: 8 }}>主評価軸</div>
              {MAIN_AXES.map((key) => (
                <AxisRow
                  key={key}
                  label={AXIS_LABELS[key]}
                  value={focus.scores[key]}
                  color={AXIS_COLORS[key]}
                />
              ))}

              <div className="wf-label" style={{ marginTop: 8 }}>補助評価軸</div>
              {SUB_AXES.map((key) => (
                <AxisRow key={key} label={AXIS_LABELS_FULL[key]} value={focus.scores[key]} color="" />
              ))}

              <div className="sep dashed" />
              <div className="wf-label">減点</div>
              {penaltyEntries.length === 0
                ? <div className="wf-note" style={{ marginTop: 4 }}>なし</div>
                : penaltyEntries.map(([k, v]) => (
                    <div key={k} className="wf-note" style={{ marginTop: 4 }}>
                      {PENALTY_LABELS[k] ?? k}: {v}
                    </div>
                  ))
              }

              <div className="sep dashed" />
              <div className="wf-label">理由</div>
              <div className="wf-note" style={{ marginTop: 4 }}>{focus.reason}</div>

              {hovered && hovered !== selected && (
                <div
                  className="wf-note"
                  style={{
                    position: 'sticky',
                    bottom: 0,
                    textAlign: 'center',
                    padding: '6px 10px',
                    background: 'var(--accent-soft)',
                    borderRadius: 6,
                    marginTop: 8,
                  }}
                >
                  クリックで固定 / ホバーで一時表示
                </div>
              )}
            </>
          ) : (
            <div className="wf-note">発言を選択してください</div>
          )}
        </div>
      </div>
    </div>
  )
}
