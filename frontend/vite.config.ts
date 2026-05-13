import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteStaticCopy } from 'vite-plugin-static-copy'

export default defineConfig({
  plugins: [
    react(),
    // Issue #68: @ffmpeg/core を `public/` 相当に同期し、`/ffmpeg/*` で
    // 同一オリジン配信する。CDN (unpkg) 経由だと worker の importScripts が
    // cross-origin で失敗するため。
    viteStaticCopy({
      targets: [
        // umd と esm の両方を配る必要がある。@ffmpeg/ffmpeg の worker は
        // type: 'module' なので importScripts が失敗し、URL の /umd/ を
        // /esm/ に書き換えて dynamic import にフォールバックする実装になっている。
        {
          src: 'node_modules/@ffmpeg/core/dist/umd/*',
          dest: 'ffmpeg/umd',
        },
        {
          src: 'node_modules/@ffmpeg/core/dist/esm/*',
          dest: 'ffmpeg/esm',
        },
      ],
    }),
  ],
  // Issue #68: @ffmpeg/ffmpeg は内部で `new Worker(new URL('./worker.js', import.meta.url))`
  // を使う。Vite の esbuild pre-bundle はこの URL 解決を壊して dev サーバで
  // exec が無限 hang する既知の事象があるため、optimizeDeps から除外する。
  optimizeDeps: {
    exclude: ['@ffmpeg/ffmpeg', '@ffmpeg/util'],
  },
  server: {
    proxy: { '/api': 'http://localhost:8000' },
  },
})
