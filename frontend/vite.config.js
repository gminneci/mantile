import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Debug HMR configuration
const hmrDisabled = process.env.VITE_HMR === 'false';
console.log('=== VITE CONFIG DEBUG ===');
console.log('VITE_HMR env var:', process.env.VITE_HMR);
console.log('HMR disabled:', hmrDisabled);
console.log('VITE_BASE_PATH:', process.env.VITE_BASE_PATH);
console.log('========================');

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_BASE_PATH || '/',
  server: {
    host: '0.0.0.0', // Listen on all network interfaces
    port: 5173,
    strictPort: true,
    allowedHosts: true,
    hmr: hmrDisabled ? false : {
      // Use the external host for HMR WebSocket connections
      clientPort: process.env.VITE_HMR_PORT ? parseInt(process.env.VITE_HMR_PORT) : undefined,
      host: process.env.VITE_HMR_HOST || 'localhost',
      protocol: process.env.VITE_HMR_PROTOCOL || 'ws',
    },
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
