// frontend/vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      // forward /api/* -> http://localhost:9000/api/*
      "/api": {
        target: "http://localhost:9000",
        changeOrigin: true,
        secure: false,
        // IMPORTANT: do NOT rewrite; keep the /api prefix so backend receives /api/analyze
        // rewrite: (path) => path.replace(/^\/api/, ""),  <-- remove / disable this
      },
    },
  },
});
