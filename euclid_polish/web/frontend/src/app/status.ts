/* Shared status resources for the shell chrome, Home and Settings › About:
 * the server version (C3) and the FASRC connection (C4). One TanStack cache
 * entry per URL, so the top bar, the rail and a page share one request. */
import { useResource } from "../api/query";

/** GET /api/version (contract C3). */
export type VersionInfo = {
  boot_commit: string | null;
  boot_short: string | null;
  head_commit: string | null;
  head_short: string | null;
  /** The server runs older code than the checkout's HEAD. */
  behind: boolean;
  dirty: boolean;
  started_at: string | null;
  pid: number | null;
  dist: { built_at: string | null; index_hash: string | null } | null;
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
