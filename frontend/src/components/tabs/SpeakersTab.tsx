import { useState } from 'react'
import type { EvaluatedUtterance, Scores, SpeakerSummary } from '../../types/meeting'
import { AXIS_LABELS } from '../../utils/labels'

type SpeakerMode = 'cards' | 'focus'

interface Props {
  summaries: SpeakerSummary[]
  utteranceMap: Map<string, EvaluatedUtterance>
  mode: SpeakerMode
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

function AxisGrid({ scores }: { scores: Scores }) {
  return (
    <div>
      {(Object.keys(AXIS_LABELS) as (keyof Scores)[]).map((key) => (
        <AxisRow key={key} label={AXIS_LABELS[key]} value={scores[key]} color={AXIS_COLORS[key]} />
      ))}
    </div>
  )
}

function RoleBadge({ role }: { role: string }) {
  const tones: Record<string, string> = {
    '整理役': 'accent',
    '推進役': 'blue',
    'リスク検知': 'red',
    '要約役': 'green',
  }
  return <span className={`chip${tones[role] ? ' ' + tones[role] : ''} dot`}>{role}</span>
}

/* ===== V1: Role cards ===== */
function RoleCards({ summaries, utteranceMap }: { summaries: SpeakerSummary[]; utteranceMap: Map<string, EvaluatedUtterance> }) {
  return (
    <div className="wf-page" style={{ overflow: 'auto' }}>
      <div style={{ padding: 18 }}>
        <div className="wf-h1" style={{ marginBottom: 4 }}>会議での役割</div>
        <div className="wf-note" style={{ marginBottom: 16 }}>
          ※ 個人の優劣ではなく、この会議で果たした役割の見え方です
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 14 }}>
          {summaries.map((s) => {
            const topUtterances = s.top_utterances
              .map((id) => utteranceMap.get(id))
              .filter((u): u is EvaluatedUtterance => u !== undefined)
            const topQuote = topUtterances[0]?.text ?? '—'
            return (
              <div key={s.speaker} className="wf-box wf-pad-l">
                <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                  <div className="avatar-md">{s.speaker.slice(0, 2)}</div>
                  <div>
                    <div className="wf-h2">{s.speaker}</div>
                    <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
                      <RoleBadge role={s.style_label} />
                      <span className="wf-mono">{s.utterance_count} 発言</span>
                    </div>
                  </div>
                  <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
                    <div className="wf-label">平均貢献</div>
                    <div className="wf-h2">{s.average_total_score.toFixed(1)}</div>
                  </div>
                </div>
                <div className="sep dashed" />
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                  <AxisGrid scores={s.average_scores} />
                  <div>
                    <div className="wf-label">代表発言</div>
                    <div className="wf-box soft wf-pad-s" style={{ marginTop: 4, fontSize: 12 }}>
                      「{topQuote}」
                    </div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

/* ===== V3: Focus detail ===== */
function FocusDetail({ summaries, utteranceMap }: { summaries: SpeakerSummary[]; utteranceMap: Map<string, EvaluatedUtterance> }) {
  const [selectedIdx, setSelectedIdx] = useState(0)
  const s = summaries[selectedIdx]

  const topUtterances = s
    ? s.top_utterances.map((id) => utteranceMap.get(id)).filter((u): u is EvaluatedUtterance => u !== undefined)
    : []

  return (
    <div className="wf-page" style={{ overflow: 'hidden' }}>
      <div style={{ padding: 16, display: 'grid', gridTemplateColumns: '240px 1fr', gap: 14, flex: 1, overflow: 'hidden' }}>

        {/* Left: speaker list */}
        <div className="wf-box wf-pad-s" style={{ overflow: 'auto' }}>
          <div className="wf-label" style={{ marginBottom: 6 }}>話者一覧</div>
          {summaries.map((sp, i) => (
            <div
              key={sp.speaker}
              onClick={() => setSelectedIdx(i)}
              style={{
                display: 'flex', gap: 8, alignItems: 'center',
                padding: 8, borderRadius: 6,
                background: i === selectedIdx ? 'var(--ink)' : 'transparent',
                color: i === selectedIdx ? 'var(--bg)' : 'var(--ink)',
                cursor: 'pointer',
                marginTop: 4,
              }}
            >
              <div
                className="avatar-sm"
                style={{ borderColor: i === selectedIdx ? 'var(--bg)' : 'var(--ink)' }}
              >
                {sp.speaker.slice(0, 2)}
              </div>
              <div>
                <div className="wf-h3" style={{ color: 'inherit' }}>{sp.speaker}</div>
                <div
                  className="wf-mono"
                  style={{ color: i === selectedIdx ? 'oklch(0.8 0.01 80)' : 'var(--ink-3)' }}
                >
                  {sp.style_label} · {sp.utterance_count}発言
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Right: detail */}
        {s && (
          <div style={{ overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
            {/* Header */}
            <div className="wf-box wf-pad-l">
              <div style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
                <div className="avatar-md">{s.speaker.slice(0, 2)}</div>
                <div>
                  <div className="wf-h1">{s.speaker}</div>
                  <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                    <RoleBadge role={s.style_label} />
                    <span className="chip">{s.utterance_count}発言</span>
                    <span className="chip">平均 {s.average_total_score.toFixed(1)}</span>
                  </div>
                </div>
                <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
                  <button
                    className="btn sm"
                    disabled={selectedIdx === 0}
                    onClick={() => setSelectedIdx((i) => Math.max(0, i - 1))}
                  >
                    ← 前の話者
                  </button>
                  <button
                    className="btn sm"
                    disabled={selectedIdx >= summaries.length - 1}
                    onClick={() => setSelectedIdx((i) => Math.min(summaries.length - 1, i + 1))}
                  >
                    次の話者 →
                  </button>
                </div>
              </div>
            </div>

            {/* Axis + type distribution */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <div className="wf-box wf-pad">
                <div className="wf-h3" style={{ marginBottom: 8 }}>軸別貢献</div>
                <AxisGrid scores={s.average_scores} />
              </div>
              <div className="wf-box wf-pad">
                <div className="wf-h3" style={{ marginBottom: 8 }}>総合貢献スコア</div>
                <div className="wf-h1">{s.total_contribution_score.toFixed(1)}</div>
                <div className="wf-note" style={{ marginTop: 4 }}>
                  発言数: {s.utterance_count}件 / 平均: {s.average_total_score.toFixed(1)}
                </div>
              </div>
            </div>

            {/* Top utterances */}
            {topUtterances.length > 0 && (
              <div className="wf-box wf-pad">
                <div className="wf-h3" style={{ marginBottom: 8 }}>代表発言</div>
                {topUtterances.map((u) => (
                  <div
                    key={u.utterance_id}
                    style={{ padding: '8px 0', borderBottom: '1px dashed var(--rule-2)' }}
                  >
                    <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
                      <span className="chip accent">{u.speech_type}</span>
                      <span className="wf-mono">{u.timestamp}</span>
                      <span className="wf-mono" style={{ marginLeft: 'auto', fontWeight: 700 }}>
                        +{u.total_score.toFixed(1)}
                      </span>
                    </div>
                    <div style={{ marginTop: 4, fontSize: 13 }}>「{u.text}」</div>
                    <div className="wf-note" style={{ marginTop: 4 }}>{u.reason}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default function SpeakersTab({ summaries, utteranceMap, mode }: Props) {
  if (summaries.length === 0) {
    return (
      <div className="wf-page" style={{ padding: 24 }}>
        <div className="wf-note">話者データがありません</div>
      </div>
    )
  }

  if (mode === 'cards') {
    return <RoleCards summaries={summaries} utteranceMap={utteranceMap} />
  }
  return <FocusDetail summaries={summaries} utteranceMap={utteranceMap} />
}
