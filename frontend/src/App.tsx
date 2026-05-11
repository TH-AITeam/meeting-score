import { useMemo, useState } from 'react'
import { analyzeJson, analyzeSample } from './api/client'
import type { MeetingSummary } from './types/meeting'
import InputView from './components/InputView'
import LoadingView from './components/LoadingView'
import SummaryTab from './components/tabs/SummaryTab'
import TimelineTab from './components/tabs/TimelineTab'
import SpeakersTab from './components/tabs/SpeakersTab'

type ResultTab = 'summary' | 'timeline' | 'speakers'
type SpeakerMode = 'cards' | 'focus'

type AppState =
  | { phase: 'upload' }
  | { phase: 'loading' }
  | { phase: 'results'; data: MeetingSummary; tab: ResultTab }

const RESULT_TABS: { id: ResultTab; label: string }[] = [
  { id: 'summary',  label: '会議サマリー' },
  { id: 'timeline', label: '発言タイムライン' },
  { id: 'speakers', label: '話者別分析' },
]

export default function App() {
  const [state, setState] = useState<AppState>({ phase: 'upload' })
  const [speakerMode, setSpeakerMode] = useState<SpeakerMode>('cards')

  const handleAnalyzeSample = async (filename: string) => {
    setState({ phase: 'loading' })
    try {
      const data = await analyzeSample(filename)
      setState({ phase: 'results', data, tab: 'summary' })
    } catch (e) {
      alert(`分析に失敗しました: ${String(e)}`)
      setState({ phase: 'upload' })
    }
  }

  const handleAnalyzeJson = async (body: unknown) => {
    setState({ phase: 'loading' })
    try {
      const data = await analyzeJson(body)
      setState({ phase: 'results', data, tab: 'summary' })
    } catch (e) {
      alert(`分析に失敗しました: ${String(e)}`)
      setState({ phase: 'upload' })
    }
  }

  const activeTab: string = state.phase === 'results' ? state.tab : 'upload'

  const utteranceMap = useMemo(() => {
    if (state.phase !== 'results') return new Map()
    const map = new Map<string, MeetingSummary['evaluated_utterances'][number]>()
    state.data.evaluated_utterances.forEach((u) => map.set(u.utterance_id, u))
    return map
  }, [state])

  const setTab = (tab: ResultTab) => {
    if (state.phase === 'results') setState({ ...state, tab })
  }

  return (
    <div className="app-root">
      {/* Top bar */}
      <div className="topbar">
        <div className="topbar-logo">MeetingScore</div>
        <div className="topbar-tabs">
          <button
            className={`topbar-tab${activeTab === 'upload' ? ' active' : ''}`}
            onClick={() => setState({ phase: 'upload' })}
          >
            アップロード
          </button>
          {state.phase === 'results' && RESULT_TABS.map((t) => (
            <button
              key={t.id}
              className={`topbar-tab${activeTab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>

        {state.phase === 'results' && activeTab === 'speakers' && (
          <div className="topbar-right">
            <span className="wf-label" style={{ fontSize: 12 }}>表示モード:</span>
            <div className="mode-toggle">
              {([['cards', '役割カード'], ['focus', '個別フォーカス']] as const).map(([id, label]) => (
                <button
                  key={id}
                  className={`mode-toggle-btn${speakerMode === id ? ' active' : ''}`}
                  onClick={() => setSpeakerMode(id)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        )}

        {state.phase === 'results' && (
          <div className={activeTab === 'speakers' ? '' : 'topbar-right'}>
            <span className="wf-mono" style={{ marginLeft: 8 }}>
              {state.data.title}
            </span>
          </div>
        )}
      </div>

      {/* Screen */}
      <div className="app-screen">
        {state.phase === 'upload' && (
          <InputView onAnalyzeSample={handleAnalyzeSample} onAnalyzeJson={handleAnalyzeJson} />
        )}
        {state.phase === 'loading' && <LoadingView />}
        {state.phase === 'results' && (
          <>
            {state.tab === 'summary'  && <SummaryTab data={state.data} />}
            {state.tab === 'timeline' && <TimelineTab utterances={state.data.evaluated_utterances} />}
            {state.tab === 'speakers' && (
              <SpeakersTab
                summaries={state.data.speaker_summaries}
                utteranceMap={utteranceMap}
                mode={speakerMode}
              />
            )}
          </>
        )}
      </div>
    </div>
  )
}
