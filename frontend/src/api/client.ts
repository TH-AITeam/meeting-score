import type { MeetingSummary, MeetingType, SampleFile, SavedMeeting, SavedMeetingMeta } from '../types/meeting'

const API_BASE = import.meta.env.VITE_API_BASE_URL
  ? `${import.meta.env.VITE_API_BASE_URL}/api`
  : '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  if (res.status === 204 || res.headers.get('content-length') === '0') {
    return undefined as T
  }
  return res.json() as Promise<T>
}

export const fetchSamples = (): Promise<SampleFile[]> =>
  request('/samples')

export const analyzeSample = (filename: string, meetingType?: MeetingType): Promise<MeetingSummary> => {
  const params = meetingType ? `?meeting_type=${encodeURIComponent(meetingType)}` : ''
  return request(`/analyze/sample/${encodeURIComponent(filename)}${params}`, { method: 'POST' })
}

export const analyzeJson = (data: unknown, meetingType?: MeetingType): Promise<MeetingSummary> => {
  const body = meetingType ? { ...(data as object), meeting_type: meetingType } : data
  return request('/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export const listMeetings = (): Promise<SavedMeetingMeta[]> =>
  request('/meetings')

export const getMeeting = (id: string): Promise<SavedMeeting> =>
  request(`/meetings/${encodeURIComponent(id)}`)

export const saveMeeting = (
  sourceType: string,
  input: unknown,
  result: unknown,
  meetingType?: MeetingType,
): Promise<SavedMeetingMeta> =>
  request('/meetings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_type: sourceType, input, result, meeting_type: meetingType ?? null }),
  })

export const deleteMeeting = (id: string): Promise<void> =>
  request(`/meetings/${encodeURIComponent(id)}`, { method: 'DELETE' })

export const uploadAudio = (file: File): Promise<unknown> => {
  const formData = new FormData()
  formData.append('file', file)
  return request('/upload_audio', { method: 'POST', body: formData })
}
