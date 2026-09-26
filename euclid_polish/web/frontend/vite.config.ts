import path from "node:path";
import type { IncomingMessage } from "node:http";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { devRoute } from "./src/app/devProxy";

// The SPA is served BY Flask: `vite build` emits into ../static/dist (committed)
// and every asset is referenced under /static/dist/ (Flask's static folder is
// web/static). `base` applies to the build only; the dev server serves at "/".
//
// Dev (`npm run dev`, http://localhost:5173): page paths from the route
// manifest (../spa_routes.json, contract C1) and Vite's own module URLs are
// served by Vite; EVERY other path (/api, /viewer, /ensemble/*.json,
// /static/dist/…, /auth, …) is proxied to Flask at
// FLASK_ORIGIN || http://localhost:9777. The Host header is kept
// (changeOrigin: false) so Flask's same-origin mutation guard sees
// Origin == Host and dev POSTs are accepted.
const FLASK = process.env.FLASK_ORIGIN || "http://localhost:9777";

/** Vite's proxy hook: a returned URL is served by Vite; undefined → Flask. */
function devBypass(req: IncomingMessage): string | undefined {
  const url = req.url ?? "/";
  const route = devRoute(url);
  if (route === "vite") return url;
  if (route === "spa") return "/index.html";
  return undefined;
}

const REACT_PKGS = /[\\/]node_modules[\\/](?:react|react-dom|scheduler|react-router|react-router-dom|@remix-run[\\/]router)[\\/]/;

export default defineConfig(({ command }) => ({
  root: __dirname,
  base: command === "build" ? "/static/dist/" : "/",
  plugins: [react()],
  build: {
    outDir: "../static/dist",
    emptyOutDir: true,
    manifest: false,
    // aladin-lite (~2.4 MB, WebGL/WASM) is a lazy chunk loaded only by the Sky
    // atlas; every eager chunk stays far below this.
    chunkSizeWarningLimit: 2600,
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (!id.includes("node_modules")) return undefined;
          if (id.includes(`${path.sep}aladin-lite${path.sep}`) || id.includes("/aladin-lite/")) return "aladin";
          if (REACT_PKGS.test(id)) return "react";
          return "vendor";
        },
      },
    },
  },
  server: {
    port: 5173,
    // The manifest lives one level above the Vite root.
    fs: { allow: [__dirname, path.resolve(__dirname, "../spa_routes.json")] },
    proxy: {
      "^/": { target: FLASK, changeOrigin: false, ws: false, bypass: devBypass },
    },
  },
}));
