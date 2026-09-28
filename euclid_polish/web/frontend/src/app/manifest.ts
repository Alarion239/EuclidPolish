/* Route manifest (contract C1) — typed access + the page matcher.
 *
 * `euclid_polish/web/spa_routes.json` is the single source of truth for page
 * URLs. Flask reads it through `spa_routes.py`; the SPA router, the rail, the
 * command palette and the Vite dev proxy read it through this module. The
 * matcher below mirrors `spa_routes.py` exactly (trailing slash ignored,
 * params substituted from their allowed values, one optional tab segment;
 * redirects: the query-aware `redirectRules` first (first match wins), then
 * the exact-path `redirects` with the query preserved, then `/app/<rest>` →
 * `/<rest>` with every leading slash/backslash/control character collapsed so
 * the target never leaves the host). manifest.test.ts ports the cases of
 * tests/test_spa_routes.py, including the open-redirect ones, and runs the
 * shared `spa_redirect_cases.json` so both sides produce byte-identical
 * targets.
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

/** A query-aware legacy redirect (spa_routes.json `redirectRules`). */
export type RedirectRule = {
  /** Path pattern; `:name` binds one segment (e.g. "/ensemble/:mode/curves"). */
  from: string;
  /** The values each `:name` may take (every `:name` is constrained). */
  params?: Record<string, string[]>;
  /** Required keys: "*" (any value), a value, or a list of allowed values.
   *  A repeated key is judged by its last value. */
  query?: Record<string, string | string[]>;
  /** Target path; `:name` substituted. */
  to: string;
  /** The target query is the original pairs (order kept) after, in order: */
  drop?: string[];
  rename?: Record<string, string>;
  /** Value → value per (renamed) key; unmapped values are kept. */
  map?: Record<string, Record<string, string>>;
  /** Prepended to the value of a key (e.g. `{ run: "local:" }`). */
  prefix?: Record<string, string>;
  /** Replace every occurrence of the key, or append it; `:name` substituted. */
  set?: Record<string, string>;
};

export type RouteManifest = {
  version: number;
  workspaces: WorkspaceDef[];
  redirectRules?: RedirectRule[];
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

/** The bytes WHATWG application/x-www-form-urlencoded keeps as they are. */
const FORM_SAFE = /^[A-Za-z0-9*\-._]$/;
const UTF8 = new TextEncoder();

/** Form encoding byte for byte like `URLSearchParams` (and
 *  `spa_routes._form_encode`): safe bytes kept, space → "+", every other byte
 *  as upper-case %XX. */
export function formEncode(text: string): string {
  let out = "";
  for (const byte of UTF8.encode(text)) {
    const ch = String.fromCharCode(byte);
    if (byte < 0x80 && FORM_SAFE.test(ch)) out += ch;
    else if (byte === 0x20) out += "+";
    else out += `%${byte.toString(16).toUpperCase().padStart(2, "0")}`;
  }
  return out;
}

/** The `:name` bindings of `path` against the rule's `from`, or null. */
function matchRulePath(rule: RedirectRule, path: string): Record<string, string> | null {
  const want = normalisePath(rule.from).split("/");
  const got = normalisePath(path).split("/");
  if (want.length !== got.length) return null;
  const allowed = rule.params ?? {};
  const bound: Record<string, string> = {};
  for (let i = 0; i < want.length; i++) {
    const w = want[i];
    const g = got[i];
    if (w.startsWith(":")) {
      const name = w.slice(1);
      if (!g || (name in allowed && !allowed[name].includes(g))) return null;
      bound[name] = g;
    } else if (w !== g) {
      return null;
    }
  }
  return bound;
}

function queryMatches(cond: RedirectRule["query"], pairs: [string, string][]): boolean {
  if (!cond) return true;
  const have = new Map<string, string>();
  for (const [k, v] of pairs) have.set(k, v); // the last occurrence wins, like dict(pairs)
  for (const [key, want] of Object.entries(cond)) {
    const value = have.get(key);
    if (value === undefined) return false;
    if (want === "*") continue;
    if (!(Array.isArray(want) ? want : [want]).includes(value)) return false;
  }
  return true;
}

function substitute(text: string, bound: Record<string, string>): string {
  let out = text;
  for (const [name, value] of Object.entries(bound)) out = out.split(`:${name}`).join(value);
  return out;
}

function applyRule(rule: RedirectRule, bound: Record<string, string>, pairs: [string, string][]): string {
  const target = substitute(rule.to, bound);
  const drop = new Set(rule.drop ?? []);
  const rename = rule.rename ?? {};
  const mapping = rule.map ?? {};
  const prefix = rule.prefix ?? {};
  const own = (o: object, k: string) => Object.prototype.hasOwnProperty.call(o, k);
  let out: [string, string][] = pairs
    .filter(([k]) => !drop.has(k))
    .map(([k, v]) => [own(rename, k) ? rename[k] : k, v]);
  out = out.map(([k, v]) => [k, own(mapping, k) && own(mapping[k], v) ? mapping[k][v] : v]);
  out = out.map(([k, v]) => [k, own(prefix, k) ? `${prefix[k]}${v}` : v]);
  for (const [key, value] of Object.entries(rule.set ?? {})) {
    const text = substitute(String(value), bound);
    if (out.some(([k]) => k === key)) out = out.map(([k, v]) => [k, k === key ? text : v]);
    else out.push([key, text]);
  }
  if (!out.length) return target;
  return `${target}?${out.map(([k, v]) => `${formEncode(k)}=${formEncode(v)}`).join("&")}`;
}

/** Where a legacy URL moves to, or null. `search` may be given with or
 *  without the leading "?". A `redirectRules` match rewrites the query; an
 *  exact `redirects` entry (and `/app/<rest>`) keeps it as it was. The
 *  `/app/<rest>` target always stays on this host (see `sameHostPath`) and
 *  resolves `<rest>` once more, so an old path under `/app` lands in one hop. */
export function redirectTarget(
  path: string,
  search = "",
  manifest: RouteManifest = MANIFEST,
): string | null {
  if (!path) return null;
  const q = search.startsWith("?") ? search.slice(1) : search;
  const rules = manifest.redirectRules ?? [];
  if (rules.length) {
    const pairs = [...new URLSearchParams(q).entries()];
    for (const rule of rules) {
      const bound = matchRulePath(rule, path);
      if (bound && queryMatches(rule.query, pairs)) return applyRule(rule, bound, pairs);
    }
  }
  const normalised = normalisePath(path);
  let target: string | undefined = manifest.redirects?.[normalised];
  if (target == null && path.startsWith(`${APP_PREFIX}/`)) {
    const rest = sameHostPath(path.slice(APP_PREFIX.length));
    // Resolve the old path in one hop: /app/ensemble → /models/…, not /ensemble.
    const onward = redirectTarget(rest, search, manifest);
    if (onward != null) return onward;
    target = rest;
  }
  if (target == null) return null;
  return q ? `${target}?${q}` : target;
}

/** The workspace definition by id (throws on an unknown id — a programming error). */
export function workspace(id: string, manifest: RouteManifest = MANIFEST): WorkspaceDef {
  const ws = manifest.workspaces.find((w) => w.id === id);
  if (!ws) throw new Error(`unknown workspace "${id}" (not in spa_routes.json)`);
  return ws;
}
