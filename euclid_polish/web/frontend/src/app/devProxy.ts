/* Which server answers a request on the Vite dev server (`npm run dev`).
 *
 *   "vite"  — Vite's own module URLs (/@vite/*, /src/*, /node_modules/*, …)
 *   "spa"   — page paths from the route manifest (C1) and legacy page URLs
 *             (the SPA redirects those client-side, like Flask's 308)
 *   "flask" — everything else: /api, /viewer, /ensemble/*.json, /static/*,
 *             /auth, /app/* (Flask's 308), … proxied to FLASK_ORIGIN.
 *
 * Pure so it can be unit-tested; vite.config.ts wires it into server.proxy. */
import { isPagePath, redirectTarget } from "./manifest";

export type DevRoute = "vite" | "spa" | "flask";

const VITE_OWN = /^\/(?:@vite\/|@id\/|@fs\/|@react-refresh|src\/|node_modules\/|__vite|index\.html$)/;

export function devRoute(url: string): DevRoute {
  const q = url.indexOf("?");
  const pathname = q < 0 ? url : url.slice(0, q);
  if (VITE_OWN.test(pathname)) return "vite";
  if (isPagePath(pathname)) return "spa";
  if (!pathname.startsWith("/app/") && redirectTarget(pathname) != null) return "spa";
  return "flask";
}
