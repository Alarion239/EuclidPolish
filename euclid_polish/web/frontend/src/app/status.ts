/* Shared status resources for the shell chrome, Home and System › Code:
 * the server version (C3) and the FASRC connection (C4). One TanStack cache
 * entry per URL, so the top bar, the rail and a page share one request. */
import { useResource } from "../api/query";

/** GET /api/version (contract C3). */
export type VersionInfo = {
  boot_commit: string | null;
  boot_short: string | null;
  head_commit: string | null;
  head_short: string | null;
  /** A backend .py file the server loaded changed on disk since it was
   *  loaded: restart to run the new code. (Commits alone never set it — a
   *  commit of the code the server already runs is not "older code"; a new
   *  SPA build never needs a restart.) */
  behind: boolean;
  /** The changed backend files, newest first (repo-relative; the first 8). */
  changed_files?: string[];
  /** How many backend files changed (all of them). */
  changed_count?: number;
  /** Names the whole changed set (order-free); null when nothing changed. */
  changed_digest?: string | null;
  dirty: boolean;
  started_at: string | null;
  pid: number | null;
  /** The served SPA build: index.html mtime + hash, and its entry script
   *  (the `<script type="module" src>`; its name hashes the whole build). */
  dist: { built_at: string | null; index_hash: string | null; entry?: string | null } | null;
};

/** GET /api/fasrc/status (contract C4). */
export type FasrcStatus = {
  ssh_connected: boolean;
  connected_at?: string | number | null;
  socket?: string | null;
  /** Startup auto-connect error or the last failed connect; null when fine. */
  last_error?: string | null;
};

export const VERSION_URL = "/api/version";
export const FASRC_STATUS_URL = "/api/fasrc/status";

export function useVersion() {
  return useResource<VersionInfo>(VERSION_URL, [], { ttl: 30_000, poll: 60_000 });
}

/** What a dismissal of the restart banner is tied to: this server process
 *  and the SET of changed files — a restart, or one more changed file, shows
 *  it again; re-saving an already-changed file does not (it only reorders
 *  the capped, newest-first `changed_files`). HEAD, `dirty` and the SPA
 *  build play no part. Keyed on the server's `changed_digest`; an older
 *  server without it falls back to the listed files + count. */
export function bannerKey(v: VersionInfo): string {
  const id = `${v.started_at ?? ""}|${v.pid ?? ""}`;
  if (v.changed_digest) return `${id}|set:${v.changed_digest}`;
  const files = [...(v.changed_files ?? [])].sort();
  return `${id}|${files.join(",")}|${v.changed_count ?? 0}`;
}

/** A new SPA build is served: `loaded` is the build this page started with,
 *  `served` the one Flask serves now (both `dist.index_hash`). */
export function distUpdated(loaded: string | null | undefined, served: string | null | undefined): boolean {
  return !!loaded && !!served && loaded !== served;
}

/** The entry script this document was loaded from (`<script type="module"
 *  src>`; in a production page the build's entry chunk). */
export function documentEntry(doc: Document | undefined = typeof document === "undefined" ? undefined : document): string | null {
  const el = doc?.querySelector<HTMLScriptElement>('script[type="module"][src]');
  const src = el?.getAttribute("src");
  return src || null;
}

/** Is the served build a different one from this page's? By the entry
 *  script when both are known (exact: the page's own script vs the one
 *  index.html now names — right even when the rebuild happened before the
 *  first /api/version answer); else by the first `index_hash` the page saw. */
export function buildChanged(
  pageEntry: string | null | undefined,
  served: VersionInfo["dist"] | undefined,
  firstSeenHash: string | null | undefined,
): boolean {
  if (pageEntry && served?.entry) return pageEntry !== served.entry;
  return distUpdated(firstSeenHash, served?.index_hash);
}

/** The first `dist.index_hash` this page saw (fallback for `buildChanged`). */
let loadedBuild: string | null = null;

/** Tests: forget the build this page loaded with. */
export function __resetLoadedBuild(): void { loadedBuild = null; }

/** Names the served build, for a per-build dismissal of the "newer build"
 *  message: its entry script (the name hashes the whole build), else the
 *  index.html hash. */
export function buildKey(served: VersionInfo["dist"] | undefined): string | null {
  return served?.entry || served?.index_hash || null;
}

/** Is a newer console build served than the one this page runs, and which
 *  (`key`, see `buildKey`)? Only a page served FROM the build can be out of
 *  date (`fromBuild`, default: a production bundle; the Vite dev server
 *  always serves the current source). Its lazy chunks may be gone, so the
 *  next new page could fail to load: reload to get the new build. Never
 *  reloads by itself. */
export function useConsoleBuild(
  fromBuild: boolean = import.meta.env.PROD,
  pageEntry: string | null = documentEntry(),
): { updated: boolean; key: string | null } {
  const dist = useVersion().data?.dist ?? undefined;
  const served = dist?.index_hash ?? null;
  if (served && loadedBuild == null) loadedBuild = served;
  const updated = fromBuild && buildChanged(pageEntry, dist, loadedBuild);
  return { updated, key: updated ? buildKey(dist) : null };
}

/** True when a newer console build is served (see `useConsoleBuild`). */
export function useConsoleUpdate(
  fromBuild: boolean = import.meta.env.PROD,
  pageEntry: string | null = documentEntry(),
): boolean {
  return useConsoleBuild(fromBuild, pageEntry).updated;
}

export function useFasrcStatus() {
  return useResource<FasrcStatus>(FASRC_STATUS_URL, [], { ttl: 10_000, poll: 30_000 });
}

/* ── Home health checks (GET /api/system/alerts, routes/system.py) ───────── */

export type CheckState = "ok" | "warn" | "bad" | "unknown";

/** A one-click fix a check offers (a local job endpoint; confirm first). */
export type HealthAction = {
  label: string;
  method: "POST";
  url: string;
  params?: Record<string, string>;
  confirm?: string;
};

export type HealthCheck = {
  id: string;
  label: string;
  state: CheckState;
  title: string;
  detail?: string | null;
  /** SPA path of the page that fixes / explains it. */
  to?: string | null;
  action?: HealthAction;
  facts?: Record<string, unknown>;
};

export type SystemAlerts = {
  computed_at: string;
  ttl_s: number;
  checks: HealthCheck[];
  /** The warn / bad checks, bad first. */
  alerts: HealthCheck[];
  counts: Record<CheckState, number>;
};

export const SYSTEM_ALERTS_URL = "/api/system/alerts";

/** The shared health-check resource (Home, the rail badge, the palette). */
export function useSystemAlerts() {
  return useResource<SystemAlerts>(SYSTEM_ALERTS_URL, [], { ttl: 30_000, poll: 120_000 });
}

/** The rail badge for Home: the number of warn/bad checks and the worst tone. */
export function alertBadge(data: SystemAlerts | null | undefined): { count: number; tone: "bad" | "warn"; label: string } | null {
  const alerts = data?.alerts ?? [];
  if (!alerts.length) return null;
  const tone = alerts.some((a) => a.state === "bad") ? "bad" : "warn";
  const n = alerts.length;
  return { count: n, tone, label: `${n} health alert${n === 1 ? "" : "s"}: ${alerts.map((a) => a.title).join("; ")}` };
}
