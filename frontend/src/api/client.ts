import type { MeetingSummary, SampleFile, SavedMeeting, SavedMeetingMeta } from '../types/meeting'

const API_BASE = import.meta.env.VITE_API_BASE_URL
  ? `${import.meta.env.VITE_API_BASE_URL}/api`
  : '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<T>
}

export const fetchSamples = (): Promise<SampleFile[]> =>
  request('/samples')

export const analyzeSample = (filename: string): Promise<MeetingSummary> =>
  request(`/analyze/sample/${encodeURIComponent(filename)}`, { method: 'POST' })

export const analyzeJson = (data: unknown): Promise<MeetingSummary> =>
  request('/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })

export const listMeetings = (): Promise<SavedMeetingMeta[]> =>
  request('/meetings')

export const getMeeting = (id: string): Promise<SavedMeeting> =>
  request(`/meetings/${encodeURIComponent(id)}`)

export const saveMeeting = (
  sourceType: string,
  input: unknown,
  result: unknown,
): Promise<SavedMeetingMeta> =>
  request('/meetings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_type: sourceType, input, result }),
  })

export const deleteMeeting = (id: string): Promise<void> =>
  request(`/meetings/${encodeURIComponent(id)}`, { method: 'DELETE' })
