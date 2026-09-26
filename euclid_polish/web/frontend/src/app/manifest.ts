/* Route manifest (contract C1) — typed access + the page matcher.
 *
 * `euclid_polish/web/spa_routes.json` is the single source of truth for page
 * URLs. Flask reads it through `spa_routes.py`; the SPA router, the rail, the
 * command palette and the Vite dev proxy read it through this module. The
 * matcher below mirrors `spa_routes.py` exactly (trailing slash ignored,
 * params substituted from their allowed values, one optional tab segment,
 * exact-path redirects with the query preserved, `/app/<rest>` → `/<rest>`
 * with every leading slash/backslash/control character collapsed so the
 * target never leaves the host). manifest.test.ts ports the cases of
 * tests/test_spa_routes.py, including the open-redirect ones.
 *
 * Pure (no DOM, no React) so `vite.config.ts` can import it in Node.
 */
import raw from "../../../spa_routes.json";

export type WorkspaceDef = {
  id: string;
  label: string;
  /** Path pattern, e.g. "/sky" or "/ensemble/:mode". */
  path: string;
  /** Allowed values for each `:param` in `path`. */
  params?: Record<string, string[]>;
  defaultParams?: Record<string, string>;
  tabs: string[];
  defaultTab?: string;
};

export type RouteManifest = {
  version: number;
  workspaces: WorkspaceDef[];
  redirects: Record<string, string>;
};

export type PageMatch = {
  workspace: string;
  params: Record<string, string>;
  tab: string | null;
  /** The concrete workspace path (params substituted, no tab). */
  base: string;
};

export const MANIFEST: RouteManifest = raw as RouteManifest;

const APP_PREFIX = "/app";

/** `/sky/` → `/sky`; the root stays `/`. */
export function normalisePath(path: string): string {
  return path.replace(/\/+$/, "") || "/";
}

/** Every concrete base path of one workspace (each allowed param value). */
export function workspacePaths(ws: WorkspaceDef): string[] {
  let paths = [ws.path];
  for (const [name, values] of Object.entries(ws.params ?? {})) {
    paths = paths.flatMap((p) => values.map((v) => p.replace(`:${name}`, String(v))));
  }
  return paths;
}

function paramsFor(ws: WorkspaceDef, concrete: string): Record<string, string> {
  const pattern = ws.path.split("/");
  const parts = concrete.split("/");
  const out: Record<string, string> = {};
  pattern.forEach((seg, i) => { if (seg.startsWith(":")) out[seg.slice(1)] = parts[i]; });
  return out;
}

type Index = Map<string, PageMatch>;
const INDEX_CACHE = new WeakMap<RouteManifest, Index>();

function index(manifest: RouteManifest): Index {
  const hit = INDEX_CACHE.get(manifest);
  if (hit) return hit;
  const map: Index = new Map();
  for (const ws of manifest.workspaces) {
    for (const concrete of workspacePaths(ws)) {
      const base = normalisePath(concrete);
      const params = paramsFor(ws, base);
      map.set(base, { workspace: ws.id, params, tab: null, base });
      const prefix = base === "/" ? "" : base;
      for (const tab of ws.tabs ?? []) {
        map.set(`${prefix}/${tab}`, { workspace: ws.id, params, tab, base });
      }
    }
  }
  INDEX_CACHE.set(manifest, map);
  return map;
}

/** Every exact page path (normalised, no trailing slash). */
export function pagePaths(manifest: RouteManifest = MANIFEST): string[] {
  return [...index(manifest).keys()];
}

/** The workspace/params/tab a page path addresses, or null for non-pages. */
export function matchPage(path: string, manifest: RouteManifest = MANIFEST): PageMatch | null {
  if (!path) return null;
  const hit = index(manifest).get(normalisePath(path));
  return hit ? { ...hit, params: { ...hit.params } } : null;
}

/** True when `path` is served by the SPA shell (same rule as Flask). */
export function isPagePath(path: string, manifest: RouteManifest = MANIFEST): boolean {
  return matchPage(path, manifest) != null;
}

/** A slash, backslash, space or control character (U+0000–U+0020): the set
 *  `spa_routes._UNSAFE_LEADING` strips. */
const isUnsafeLeading = (ch: string) => ch === "/" || ch === "\\" || ch.charCodeAt(0) <= 0x20;

/** `rest` as a path on this host: `//evil.example` → `/evil.example`
 *  (`spa_routes._same_host_path`: every leading unsafe character collapses
 *  into one `/`). `//host` and `/\host` are protocol-relative to a browser,
 *  so this keeps the `/app/<rest>` redirect from ever leaving the host. */
function sameHostPath(rest: string): string {
  let i = 0;
  while (i < rest.length && isUnsafeLeading(rest[i])) i++;
  return `/${rest.slice(i)}`;
}

/** Where a legacy URL moves to (query preserved), or null. `search` may be
 *  given with or without the leading "?". The `/app/<rest>` target always
 *  stays on this host (see `sameHostPath`). */
export function redirectTarget(
  path: string,
  search = "",
  manifest: RouteManifest = MANIFEST,
): string | null {
  if (!path) return null;
  const normalised = normalisePath(path);
  let target: string | undefined = manifest.redirects?.[normalised];
  if (target == null && path.startsWith(`${APP_PREFIX}/`)) {
    target = sameHostPath(path.slice(APP_PREFIX.length));
  }
  if (target == null) return null;
  const q = search.startsWith("?") ? search.slice(1) : search;
  return q ? `${target}?${q}` : target;
}

/** The workspace definition by id (throws on an unknown id — a programming error). */
export function workspace(id: string, manifest: RouteManifest = MANIFEST): WorkspaceDef {
  const ws = manifest.workspaces.find((w) => w.id === id);
  if (!ws) throw new Error(`unknown workspace "${id}" (not in spa_routes.json)`);
  return ws;
}
