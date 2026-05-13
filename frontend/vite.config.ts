import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
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
