import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import os from 'node:os';
import path from 'node:path';

const cacheDir = process.env.VITE_CACHE_DIR || path.join(os.tmpdir(), 'multirank-rag-vite-cache');

export default defineConfig({
  cacheDir,
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8765',
    },
    watch: {
      usePolling: true,
      interval: 1000,
    },
  },
});
