import { defineConfig } from 'vite'

// Frontend dev server. /api is proxied to the FastAPI backend so the browser
// talks to a single origin and NiiVue can fetch volumes/overlays without CORS.
export default defineConfig({
  server: {
    port: 5173,
    strictPort: true,
    allowedHosts: ['siddharth-legion-s7.tailb7f323.ts.net'],
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
