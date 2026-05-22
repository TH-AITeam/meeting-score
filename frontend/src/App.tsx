import { useMemo, useState } from 'react'
import { analyzeJson, analyzeSample, saveMeeting, uploadAudio } from './api/client'
import type { MeetingSummary, MeetingType } from './types/meeting'
import InputView from './components/InputView'
import LoadingView from './components/LoadingView'
import SummaryTab from './components/tabs/SummaryTab'
import TimelineTab from './components/tabs/TimelineTab'
import SpeakersTab from './components/tabs/SpeakersTab'

type ResultTab = 'summary' | 'timeline' | 'speakers'
type SpeakerMode = 'cards' | 'focus'

type AppState =
  | { phase: 'upload' }
  | { phase: 'loading'; step?: 'transcribing' | 'analyzing' }
  | { phase: 'results'; data: MeetingSummary; tab: ResultTab }

const RESULT_TABS: { id: ResultTab; label: string }[] = [
  { id: 'summary',  label: '会議サマリー' },
  { id: 'timeline', label: '発言タイムライン' },
  { id: 'speakers', label: '話者別分析' },
]

export default function App() {
  const [state, setState] = useState<AppState>({ phase: 'upload' })
  const [speakerMode, setSpeakerMode] = useState<SpeakerMode>('cards')
  const [historyVersion, setHistoryVersion] = useState(0)

  const triggerHistoryRefresh = () => setHistoryVersion((v) => v + 1)

  const handleAnalyzeAudio = async (file: File, meetingType?: MeetingType) => {
    setState({ phase: 'loading', step: 'transcribing' })
    try {
      const meetingInput = await uploadAudio(file)
      setState({ phase: 'loading', step: 'analyzing' })
      const data = await analyzeJson(meetingInput, meetingType)
      setState({ phase: 'results', data, tab: 'summary' })
      saveMeeting('audio', { filename: file.name }, data, meetingType)
        .then(triggerHistoryRefresh)
        .catch(() => {/* 保存失敗はサイレントに無視 */})
    } catch (e) {
      alert(`分析に失敗しました: ${String(e)}`)
      setState({ phase: 'upload' })
    }
  }

  const handleAnalyzeSample = async (filename: string, meetingType?: MeetingType) => {
    setState({ phase: 'loading' })
    try {
      const data = await analyzeSample(filename, meetingType)
      setState({ phase: 'results', data, tab: 'summary' })
      saveMeeting('sample', { filename }, data, meetingType)
        .then(triggerHistoryRefresh)
        .catch(() => {/* 保存失敗はサイレントに無視 */})
    } catch (e) {
      alert(`分析に失敗しました: ${String(e)}`)
      setState({ phase: 'upload' })
    }
  }

  const handleAnalyzeJson = async (body: unknown, meetingType?: MeetingType) => {
    setState({ phase: 'loading' })
    // UI 未選択でも body に meeting_type が埋め込まれていればそれを使う
    const effectiveType = meetingType ?? (body as Record<string, unknown>)?.meeting_type as MeetingType | undefined
    try {
      const data = await analyzeJson(body, effectiveType)
      setState({ phase: 'results', data, tab: 'summary' })
      saveMeeting('upload', body, data, effectiveType)
        .then(triggerHistoryRefresh)
        .catch(() => {/* 保存失敗はサイレントに無視 */})
    } catch (e) {
      alert(`分析に失敗しました: ${String(e)}`)
      setState({ phase: 'upload' })
    }
  }

  const handleRestore = (data: MeetingSummary) => {
    setState({ phase: 'results', data, tab: 'summary' })
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
          <InputView
            onAnalyzeSample={handleAnalyzeSample}
            onAnalyzeJson={handleAnalyzeJson}
            onAnalyzeAudio={handleAnalyzeAudio}
            onRestore={handleRestore}
            historyVersion={historyVersion}
          />
        )}
        {state.phase === 'loading' && <LoadingView step={state.step} />}
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
