// フィードバック収集 API クライアント (Issue #79 / バックエンド #78)
//
// 認証基盤は未整備のため、org_id は暫定でローカル保存値を使い、X-Org-Id ヘッダで
// 送る（バックエンドは X-Org-Id == body.org_id を検証する）。認証導入時にここを
// トークン由来の org_id に差し替える。

const API_BASE = import.meta.env.VITE_API_BASE_URL
  ? `${import.meta.env.VITE_API_BASE_URL}/api`
  : '/api'

const ORG_STORAGE_KEY = 'meeting_score_org_id'
const DEFAULT_ORG_ID = 'org_local'

export function getOrgId(): string {
  try {
    return localStorage.getItem(ORG_STORAGE_KEY) || DEFAULT_ORG_ID
  } catch {
    // localStorage 不可（SSR / プライベートモード等）でも動く
    return DEFAULT_ORG_ID
  }
}

export function setOrgId(orgId: string): void {
  try {
    localStorage.setItem(ORG_STORAGE_KEY, orgId)
  } catch {
    // 保存できなくても致命的でないため握りつぶす
  }
}

async function postFeedback<T>(path: string, body: Record<string, unknown>): Promise<T> {
  const orgId = getOrgId()
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Org-Id': orgId },
    body: JSON.stringify({ org_id: orgId, ...body }),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<T>
}

export type PairwiseWinner = 'A' | 'B' | 'tie'
export type FlagDirection = 'overrated' | 'underrated'

export interface FeedbackAck {
  id: string
  generated_pairs: number
}

export interface FeedbackStats {
  org_id: string
  n_pairwise: number
  n_topk: number
  n_axis_flag: number
  stage: number
  next_stage: number | null
  pairs_to_next_stage: number | null
}

/** ペアワイズ比較を 1 件送る（手動ペア用）。 */
export const postPairwise = (params: {
  meeting_id: string
  utt_a: string
  utt_b: string
  winner: PairwiseWinner
}): Promise<FeedbackAck> =>
  postFeedback('/feedback/pairwise', { ...params, source: 'manual_pair' })

/**
 * Top5 訂正を送る。サーバ側で「Top入りした発言 × 押し出された発言」のペアワイズに
 * 自動展開される（generated_pairs に件数が返る）。
 */
export const postTopK = (params: {
  meeting_id: string
  corrected_top5: string[]
  original_top5: string[]
}): Promise<FeedbackAck> => postFeedback('/feedback/topk', params)

/** 発言カードの 👎（過大/過小評価 + 任意の軸 + コメント）を送る。 */
export const postAxisFlag = (params: {
  meeting_id: string
  utterance_id: string
  direction: FlagDirection
  axis?: string | null
  comment?: string | null
}): Promise<FeedbackAck> => postFeedback('/feedback/axis_flag', params)

/** 蓄積件数・段階(0/1/2)・次段階までの不足ペア数を取得する。 */
export const getFeedbackStats = async (): Promise<FeedbackStats> => {
  const orgId = getOrgId()
  const res = await fetch(
    `${API_BASE}/feedback/stats?org_id=${encodeURIComponent(orgId)}`,
    { headers: { 'X-Org-Id': orgId } },
  )
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<FeedbackStats>
}
