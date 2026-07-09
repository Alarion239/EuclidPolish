import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The SPA is served BY Flask: `vite build` emits into ../static/dist and every
// asset is referenced under /static/dist/ (Flask's static_folder is web/static).
// In dev, `vite` runs its own server and proxies the Flask API + binary cube
// endpoints to the running Flask app (default :9777) so the SPA talks to real
// data without CORS. Add new backend path prefixes here as tabs are ported.
const FLASK = process.env.FLASK_ORIGIN || "http://localhost:9777";
const API_PREFIXES = [
  "/ensemble", "/viewer", "/api", "/view", "/vis", "/inspect",
  "/sky", "/tng", "/inference", "/evaluation", "/lensfinder",
  "/catalog", "/psfs", "/tracking", "/git", "/fasrc", "/cutouts",
];

export default defineConfig({
  root: __dirname,
  base: "/static/dist/",
  plugins: [react()],
  build: {
    outDir: "../static/dist",
    emptyOutDir: true,
    manifest: false,
    chunkSizeWarningLimit: 1200,
  },
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      API_PREFIXES.map((p) => [p, { target: FLASK, changeOrigin: true }]),
    ),
  },
});
