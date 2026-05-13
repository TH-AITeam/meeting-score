/**
 * 動画ファイルから音声を抽出する (Issue #68)。
 *
 * 方針:
 * - ffmpeg.wasm を **lazy load** する。初回呼び出し時に core (~20MB) を CDN
 *   から fetch するため、呼び元はローディング UI を出す責務を持つ。
 * - 出力は libopus / mono / 16kHz / 32kbps (webm コンテナ)。backend が WhisperX
 *   へ流す前に正規化するので圧縮優先で良い。
 * - 失敗時は `MediaExtractError` を throw する。呼び元で fallback (元動画送信)
 *   は **しない** — Issue #68 の方針通り、ユーザーに再選択を促す。
 */
import { FFmpeg } from '@ffmpeg/ffmpeg'
import { fetchFile } from '@ffmpeg/util'

const FFMPEG_CORE_VERSION = '0.12.6'
// CDN を直接渡す。toBlobURL 経由だと Worker の importScripts が blob URL を
// 読めず "failed to import ffmpeg-core.js" で落ちるブラウザがある。
// unpkg は CORS を許可しているので直接 URL でも動く。
const FFMPEG_CORE_BASE = `https://unpkg.com/@ffmpeg/core@${FFMPEG_CORE_VERSION}/dist/umd`

export class MediaExtractError extends Error {
  readonly cause?: unknown
  constructor(message: string, options?: { cause?: unknown }) {
    super(message)
    this.name = 'MediaExtractError'
    if (options?.cause !== undefined) this.cause = options.cause
  }
}

export type ExtractProgress = {
  /** 0.0 - 1.0 (ffmpeg.wasm が報告する変換進捗) */
  ratio: number
}

export type ExtractOptions = {
  /** 進捗コールバック (UI 表示用)。例外を投げないこと。 */
  onProgress?: (p: ExtractProgress) => void
  /** core のロード進捗 (初回のみ)。0.0 - 1.0。 */
  onLoadProgress?: (ratio: number) => void
  /** AbortSignal でキャンセル可能にする (将来用途)。現状は未使用。 */
  signal?: AbortSignal
}

let ffmpegInstance: FFmpeg | null = null
let loadPromise: Promise<FFmpeg> | null = null

/**
 * ffmpeg.wasm インスタンスを 1 度だけロードして共有する。
 * 並行呼び出しは同じ Promise を待つので二重 fetch しない。
 */
async function getFFmpeg(onLoadProgress?: (ratio: number) => void): Promise<FFmpeg> {
  if (ffmpegInstance) return ffmpegInstance
  if (loadPromise) return loadPromise

  loadPromise = (async () => {
    const ffmpeg = new FFmpeg()
    ffmpeg.on('log', ({ type, message }) => {
      // ffmpeg.wasm 内部のログを全部 console に出して、hang/失敗時の解析を可能にする
      console.debug(`[ffmpeg.wasm ${type}]`, message)
    })
    if (onLoadProgress) {
      ffmpeg.on('progress', ({ progress }) => onLoadProgress(progress))
    }
    try {
      await ffmpeg.load({
        coreURL: `${FFMPEG_CORE_BASE}/ffmpeg-core.js`,
        wasmURL: `${FFMPEG_CORE_BASE}/ffmpeg-core.wasm`,
      })
    } catch (e) {
      loadPromise = null
      console.error('[audioExtract] ffmpeg.load failed:', e)
      const rawMessage =
        e instanceof Error ? e.message
        : typeof e === 'string' ? e
        : JSON.stringify(e)
      throw new MediaExtractError(
        `ffmpeg.wasm のロードに失敗しました: ${rawMessage || '(原因不明: Console を確認)'}`,
        { cause: e },
      )
    }
    ffmpegInstance = ffmpeg
    return ffmpeg
  })()
  return loadPromise
}

/**
 * 動画 File → 音声 Blob (webm/opus, mono 16kHz 32kbps)。
 *
 * @throws MediaExtractError 抽出失敗時 (動画に音声トラックが無い等を含む)
 */
export async function extractAudioFromVideo(
  file: File,
  options: ExtractOptions = {},
): Promise<Blob> {
  const { onProgress, onLoadProgress } = options
  const ffmpeg = await getFFmpeg(onLoadProgress)

  const inputName = `input_${Date.now()}_${file.name.replace(/[^\w.-]/g, '_')}`
  const outputName = `output_${Date.now()}.webm`

  const progressHandler = ({ progress }: { progress: number }) => {
    onProgress?.({ ratio: Math.max(0, Math.min(1, progress)) })
  }
  if (onProgress) ffmpeg.on('progress', progressHandler)

  try {
    await ffmpeg.writeFile(inputName, await fetchFile(file))
    const exitCode = await ffmpeg.exec([
      '-i', inputName,
      '-vn',
      '-c:a', 'libopus',
      '-b:a', '32k',
      '-ac', '1',
      '-ar', '16000',
      outputName,
    ])
    if (exitCode !== 0) {
      throw new MediaExtractError(
        `ffmpeg exec failed (code=${exitCode}). 動画に音声トラックが無い可能性があります。`,
      )
    }
    const data = await ffmpeg.readFile(outputName)
    if (typeof data === 'string' || data.length === 0) {
      throw new MediaExtractError('ffmpeg が空の出力を返しました')
    }
    // SharedArrayBuffer 由来の場合に備えて新規 Uint8Array にコピーしてから Blob 化
    const view = data as Uint8Array
    const copy = new Uint8Array(view.byteLength)
    copy.set(view)
    return new Blob([copy], { type: 'audio/webm' })
  } catch (e) {
    if (e instanceof MediaExtractError) throw e
    throw new MediaExtractError(
      `動画→音声抽出に失敗しました: ${(e as Error).message}`,
      { cause: e as Error },
    )
  } finally {
    if (onProgress) ffmpeg.off('progress', progressHandler)
    // 一時ファイルを掃除する (失敗してもユーザー操作に影響しないので throw しない)
    try { await ffmpeg.deleteFile(inputName) } catch { /* noop */ }
    try { await ffmpeg.deleteFile(outputName) } catch { /* noop */ }
  }
}

/** 拡張子から動画判定する (ファイル選択 UI で使う)。 */
export const VIDEO_EXTENSIONS: ReadonlySet<string> = new Set([
  '.mp4', '.mov', '.mkv', '.avi', '.webm',
])

export function isVideoFile(file: File): boolean {
  const lower = file.name.toLowerCase()
  for (const ext of VIDEO_EXTENSIONS) {
    if (lower.endsWith(ext)) {
      // .webm は音声単独でもあり得るので、type で video かを優先判定
      if (ext === '.webm') return file.type.startsWith('video/')
      return true
    }
  }
  return file.type.startsWith('video/')
}
