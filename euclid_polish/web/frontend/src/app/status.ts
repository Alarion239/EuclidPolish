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
