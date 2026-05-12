import type { MeetingSummary, SampleFile } from '../types/meeting'

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
