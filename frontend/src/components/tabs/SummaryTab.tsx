import { useState } from 'react'
import type { EvaluatedUtterance, MeetingSummary, Scores } from '../../types/meeting'
import { AXIS_LABELS } from '../../utils/labels'

interface Props {
  data: MeetingSummary
}

type Filter = 'all' | 'issue' | 'decision' | 'risk' | 'action'

const FILTER_CHIPS: { id: Filter; label: string; tone: string }[] = [
  { id: 'all',      label: 'すべて',    tone: '' },
  { id: 'issue',    label: '論点整理',  tone: 'accent' },
  { id: 'decision', label: '意思決定',  tone: 'blue' },
  { id: 'risk',     label: 'リスク',    tone: 'red' },
  { id: 'action',   label: 'アクション', tone: 'green' },
]

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

function typeChipTone(u: EvaluatedUtterance): string {
  if (u.total_score < 0) return 'red'
  if (u.total_score >= 5) return 'accent'
  return ''
}

export default function SummaryTab({ data }: Props) {
  const [filter, setFilter] = useState<Filter>('all')

  const utterances = data.evaluated_utterances
  const totalCount = utterances.length
  const topCount = utterances.filter((u) => u.total_score >= 4).length
  const penaltyCount = utterances.filter(
    (u) => u.penalties.off_topic < 0 || u.penalties.duplication < 0,
  ).length

  // Average scores across all evaluated utterances
  const avgScores: Scores = (() => {
    if (utterances.length === 0) {
      return { issue_clarification: 0, decision_progress: 0, risk_detection: 0, actionability: 0, groundedness: 0, novelty: 0, summarization: 0 }
    }
    const keys = Object.keys(utterances[0].scores) as (keyof Scores)[]
    const sums = Object.fromEntries(keys.map((k) => [k, 0])) as unknown as Scores
    utterances.forEach((u) => {
      keys.forEach((k) => { sums[k] += u.scores[k] })
    })
    keys.forEach((k) => { sums[k] = Math.round((sums[k] / utterances.length) * 10) / 10 })
    return sums
  })()

  // Filter top utterances — "all" shows only genuinely forward utterances (score >= 4)
  const filteredUtterances = (() => {
    if (filter === 'all') return data.top_utterances.filter((u) => u.total_score >= 4)
    if (filter === 'issue') return data.top_issue_clarification
    if (filter === 'decision') return data.top_decision_progress
    if (filter === 'risk') return data.top_risk_detection
    return data.top_actionability
  })()

  const kpiTiles = [
    { label: '総発言',    value: String(totalCount),  tone: '' },
    { label: '前進発言',  value: String(topCount),    tone: 'accent' },
    { label: '論点整理',  value: String(utterances.filter((u) => u.scores.issue_clarification > 0).length), tone: '' },
    { label: '意思決定',  value: String(utterances.filter((u) => u.scores.decision_progress > 0).length),   tone: 'blue' },
    { label: 'リスク指摘', value: String(utterances.filter((u) => u.scores.risk_detection > 0).length),     tone: 'red' },
    { label: '脱線/重複', value: String(penaltyCount), tone: '' },
  ]

  return (
    <div className="wf-page" style={{ overflow: 'hidden' }}>
      <div style={{ padding: 18, display: 'flex', flexDirection: 'column', gap: 12, height: '100%', overflow: 'hidden' }}>

        {/* Meta strip */}
        <div className="wf-box wf-pad" style={{ display: 'flex', gap: 18, alignItems: 'center', flexWrap: 'wrap', flexShrink: 0 }}>
          <div>
            <div className="wf-label">会議</div>
            <div className="wf-h3">{data.title}</div>
          </div>
          <div style={{ width: 1, height: 28, background: 'var(--rule-2)' }} />
          <div>
            <div className="wf-label">目的</div>
            <div style={{ fontSize: 13 }}>{data.goal}</div>
          </div>
          {data.overall_comment && (
            <>
              <div style={{ width: 1, height: 28, background: 'var(--rule-2)' }} />
              <div style={{ flex: 1, fontSize: 12, color: 'var(--ink-2)', minWidth: 200 }}>{data.overall_comment}</div>
            </>
          )}
        </div>

        {/* KPI tiles */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 10, flexShrink: 0 }}>
          {kpiTiles.map((t) => (
            <div key={t.label} className={`wf-box wf-pad-s${t.tone ? ' ' + t.tone : ''}`}>
              <div className="wf-label">{t.label}</div>
              <div className="wf-h2">{t.value}</div>
            </div>
          ))}
        </div>

        {/* Main area */}
        <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 12, flex: 1, minHeight: 0 }}>

          {/* Forward progress list */}
          <div className="wf-box wf-pad" style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8, flexShrink: 0 }}>
              <div className="wf-h3">前進した発言</div>
              <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                {FILTER_CHIPS.map((c) => (
                  <span
                    key={c.id}
                    className={`chip clickable${c.tone ? ' ' + c.tone : ''}${filter === c.id ? ' ink' : ''}`}
                    onClick={() => setFilter(c.id)}
                  >
                    {c.label}
                  </span>
                ))}
              </div>
            </div>
            <div className="sep dashed" style={{ flexShrink: 0 }} />
            <div style={{ overflow: 'auto', flex: 1 }}>
              {filteredUtterances.length === 0 && (
                <div className="wf-note" style={{ padding: '12px 0' }}>該当なし</div>
              )}
              {filteredUtterances.map((u, i) => (
                <div
                  key={u.utterance_id}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '22px 30px 1fr 52px',
                    gap: 8,
                    padding: '8px 4px',
                    borderBottom: '1px dashed var(--rule-2)',
                    alignItems: 'center',
                  }}
                >
                  <span className="wf-mono">{i + 1}</span>
                  <div className="avatar-sm">{u.speaker.slice(0, 2)}</div>
                  <div style={{ minWidth: 0 }}>
                    <div style={{ display: 'flex', gap: 5, alignItems: 'center', marginBottom: 2, flexWrap: 'wrap' }}>
                      <span className={`chip${typeChipTone(u) ? ' ' + typeChipTone(u) : ''}`}>{u.speech_type}</span>
                      <span className="wf-mono">{u.timestamp}</span>
                    </div>
                    <div style={{ fontSize: 12, color: 'var(--ink-2)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{u.text}</div>
                  </div>
                  <div className="wf-mono" style={{ textAlign: 'right', fontWeight: 700, color: u.total_score < 0 ? 'var(--red)' : 'var(--ink)' }}>
                    {u.total_score >= 0 ? '+' : ''}{u.total_score.toFixed(1)}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Right column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12, overflow: 'hidden' }}>

            {/* Axis balance */}
            <div className="wf-box wf-pad" style={{ overflow: 'auto' }}>
              <div className="wf-h3" style={{ marginBottom: 8 }}>軸別バランス</div>
              {(Object.keys(AXIS_LABELS) as (keyof Scores)[]).map((key) => (
                <AxisRow
                  key={key}
                  label={AXIS_LABELS[key]}
                  value={avgScores[key]}
                  color={AXIS_COLORS[key]}
                />
              ))}
            </div>

            {/* Improvement comments */}
            {data.improvement_comments.length > 0 && (
              <div className="wf-box wf-pad tinted" style={{ overflow: 'auto', flex: 1 }}>
                <div className="wf-label" style={{ marginBottom: 6 }}>改善メモ</div>
                {data.improvement_comments.map((c, i) => (
                  <div key={i} className="wf-note" style={{ marginBottom: 6 }}>・{c}</div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
