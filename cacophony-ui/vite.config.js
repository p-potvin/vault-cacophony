import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Served from the python server at /, so keep asset paths relative.
  base: './',
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    // `npm run dev` proxies to the python server so the UI can be developed
    // with hot reload while the real conversation runs behind it.
    proxy: { '/api': { target: 'http://127.0.0.1:8750', changeOrigin: true } },
  },
})
