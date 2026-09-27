/* Data layer (contract C8): one TanStack Query client + `useResource`.
 *
 * `useResource(url, deps?, {ttl?, poll?})` is the compatibility-shaped GET
 * hook every page uses: `{data, loading, error, reload}` (+ `fetching`,
 * `staleError`, `updatedAt`). Under the hood it is a TanStack query keyed by
 * ["GET", url], so it gets:
 *   - in-flight dedupe + shared subscribers (N components → 1 request),
 *   - a shared cache with stale-while-revalidate (`ttl`, default 5 min),
 *   - visibility-aware polling (`poll` ms; paused while the tab is hidden,
 *     refreshed on return),
 *   - a retry for idempotent GETs on network/5xx failures (not 4xx, not the
 *     503 FASRC gate),
 *   - request abort when nothing is subscribed any more.
 * `deps` force a refetch when they change while the URL stays the same (the
 * old hook ignored them when the cache was fresh — the "PreviousRuns does not
 * refresh after submit" bug). They are compared BY VALUE (`sameDep`), so an
 * inline `[{...}]`, `[data?.x ?? []]` or `[asArray(y)]` rebuilt on every
 * render cannot start a refetch loop; functions are ignored; class instances
 * compare by identity (keep those stable). Invalidate related resources after
 * a mutation with `invalidate("/ensemble/")` (URL prefix).
 */
import { QueryClient, useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useSyncExternalStore, type DependencyList } from "react";
import { ApiError, apiGet } from "./client";

/** How long a cached GET is served without a background refetch. */
export const DEFAULT_TTL_MS = 5 * 60_000;

function shouldRetry(failureCount: number, error: unknown): boolean {
  if (failureCount >= 2) return false;
  if (!(error instanceof ApiError)) return false;
  return error.status === 0 || (error.status >= 500 && error.status !== 501 && error.status !== 503);
}

export const queryClient = new QueryClient({
  defaultOptions: {
    // Every request goes to the local Flask server, which keeps answering when
    // the browser reports `offline` (Wi-Fi drop = exactly when FASRC
    // disconnects). TanStack's default networkMode "online" would pause every
    // fetch, poll and retry until an `online` event — breaking offline-first.
    queries: {
      networkMode: "always",
      staleTime: DEFAULT_TTL_MS,
      gcTime: 30 * 60_000,
      refetchOnWindowFocus: false,
      retry: shouldRetry,
      retryDelay: (attempt) => Math.min(4000, 500 * 2 ** attempt),
    },
    mutations: { networkMode: "always" },
  },
});

/** The query key of a GET resource (use with `queryClient` directly). */
export const resourceKey = (url: string) => ["GET", url] as const;

export type Resource<T> = {
  data: T | null;
  /** True only while there is no data yet and a first fetch is running. */
  loading: boolean;
  /** True whenever a request is in flight (including background refreshes). */
  fetching: boolean;
  /** The failure, when there is NO data to show (compat: truthy on failure). */
  error: ApiError | null;
  /** A failed background refresh while (older) data is still shown. */
  staleError: ApiError | null;
  /** When `data` was fetched (ms epoch). */
  updatedAt: number | null;
  /** Force a fresh fetch (every subscriber of the URL sees the result). */
  reload: () => void;
};

export type ResourceOpts = {
  /** Freshness window in ms (default 5 min). */
  ttl?: number;
  /** Poll interval in ms while the tab is visible (off by default). */
  poll?: number;
};

/** Deeper than this, a dep is compared by identity (guards cyclic values). */
const MAX_DEP_DEPTH = 16;

const isPlainObject = (v: object): v is Record<string, unknown> => {
  const proto = Object.getPrototypeOf(v);
  return proto === Object.prototype || proto === null;
};

/** Value equality for refetch triggers. Primitives: `Object.is`. Arrays,
 *  plain objects and Map values: element-wise (recursively; Map keys and Set
 *  members by `has`, i.e. SameValueZero). Dates: by time.
 *  Functions: always equal (a callback carries no fetch-relevant value, and
 *  an inline one would otherwise refetch on every render). Anything else
 *  (class instances, typed arrays, …): by identity. */
function sameDep(a: unknown, b: unknown, depth = 0): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a === "function" && typeof b === "function") return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null) return false;
  if (depth >= MAX_DEP_DEPTH) return false;
  const d = depth + 1;
  if (Array.isArray(a) || Array.isArray(b)) {
    return Array.isArray(a) && Array.isArray(b) && a.length === b.length
      && a.every((x, i) => sameDep(x, b[i], d));
  }
  if (a instanceof Date || b instanceof Date) {
    return a instanceof Date && b instanceof Date && Object.is(a.getTime(), b.getTime());
  }
  if (a instanceof Map || b instanceof Map) {
    if (!(a instanceof Map && b instanceof Map) || a.size !== b.size) return false;
    for (const [k, v] of a) if (!b.has(k) || !sameDep(v, b.get(k), d)) return false;
    return true;
  }
  if (a instanceof Set || b instanceof Set) {
    if (!(a instanceof Set && b instanceof Set) || a.size !== b.size) return false;
    for (const v of a) if (!b.has(v)) return false;
    return true;
  }
  if (!isPlainObject(a) || !isPlainObject(b)) return false;
  const ka = Object.keys(a);
  if (ka.length !== Object.keys(b).length) return false;
  return ka.every((k) => Object.prototype.hasOwnProperty.call(b, k) && sameDep(a[k], b[k], d));
}

function depsChanged(a: DependencyList, b: DependencyList): boolean {
  if (a.length !== b.length) return true;
  for (let i = 0; i < a.length; i++) if (!sameDep(a[i], b[i])) return true;
  return false;
}

function asApiError(e: unknown): ApiError | null {
  if (e == null) return null;
  if (e instanceof ApiError) return e;
  return new ApiError({ status: 0, message: e instanceof Error ? e.message : String(e) });
}

/** The (never fetched) key an idle resource subscribes to — distinct from
 *  every URL key, so an idle hook cannot show another resource's state. */
const IDLE_KEY = ["GET:idle"] as const;

/** Fetch JSON from `url` through the shared query cache. A falsy URL (null or
 *  "") is idle: no request, `{data: null, loading: false, error: null}`. */
export function useResource<T = unknown>(
  url: string | null | undefined,
  deps: DependencyList = [],
  opts: ResourceOpts = {},
): Resource<T> {
  const ttl = opts.ttl ?? DEFAULT_TTL_MS;
  const poll = opts.poll && opts.poll > 0 ? opts.poll : undefined;
  const active = !!url;
  const q = useQuery<T, ApiError>({
    queryKey: active ? resourceKey(url) : IDLE_KEY,
    queryFn: ({ signal }) => apiGet<T>(url as string, { signal }),
    enabled: active,
    staleTime: poll ? Math.min(ttl, poll) : ttl,
    refetchInterval: poll ?? false,
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: poll ? true : false,
  }, queryClient);

  const { refetch } = q;
  const prev = useRef<{ url: string | null | undefined; deps: DependencyList }>({ url, deps });
  useEffect(() => {
    const before = prev.current;
    prev.current = { url, deps };
    if (active && before.url === url && depsChanged(before.deps, deps)) void refetch();
  });

  const reload = useCallback(() => { if (active) void refetch(); }, [active, refetch]);
  const hasData = active && q.data !== undefined;
  return {
    data: hasData ? (q.data as T) : null,
    loading: active && q.isPending,
    fetching: active && q.isFetching,
    error: active && !hasData ? asApiError(q.error) : null,
    staleError: hasData ? asApiError(q.error) : null,
    updatedAt: hasData && q.dataUpdatedAt ? q.dataUpdatedAt : null,
    reload,
  };
}

const isGet = (key: readonly unknown[]) => key[0] === "GET" && typeof key[1] === "string";

/** Mark every GET resource whose URL starts with `prefix` stale and refetch
 *  the mounted ones (all resources when omitted). */
export function invalidate(prefix = ""): Promise<void> {
  return queryClient.invalidateQueries({
    predicate: (query) => isGet(query.queryKey) && (query.queryKey[1] as string).startsWith(prefix),
  });
}

/** Like `invalidate` but matches a URL substring (the old hooks.ts API). */
export function invalidateMatching(substr = ""): Promise<void> {
  return queryClient.invalidateQueries({
    predicate: (query) => isGet(query.queryKey) && (query.queryKey[1] as string).includes(substr),
  });
}

/** Warm the cache for a URL (e.g. on hover) without subscribing. */
export function prefetchResource(url: string, ttl = DEFAULT_TTL_MS): Promise<void> {
  return queryClient.prefetchQuery({
    queryKey: resourceKey(url),
    queryFn: ({ signal }) => apiGet(url, { signal }),
    staleTime: ttl,
  });
}

/** Read / replace the cached body of a URL (optimistic updates). */
export function getResourceData<T>(url: string): T | undefined {
  return queryClient.getQueryData<T>(resourceKey(url));
}
export function setResourceData<T>(url: string, data: T): void {
  queryClient.setQueryData<T>(resourceKey(url), data);
}

/* ── Server health: is the local server answering? ─────────────────────────
 *
 * Every query outcome in the shared cache feeds one tracker. An answer from
 * the server — any success, or an HTTP error it chose to send (4xx, 501, the
 * 503 FASRC gate, and ANY 5xx carrying a JSON body: Flask's `{ok:false,
 * error}` envelope, e.g. the deliberate 502 when an SSH/FASRC call fails or
 * the JSON 500 of a crashed route) — means it is up. It is "down" after a
 * request got no response at all (status 0, after the retries), or when two
 * DIFFERENT resources failed with a body-less (non-JSON) 500/502/504 since
 * the last answer: Vite's dev proxy answers an empty 500 when Flask is gone,
 * while one route answering 500 is just that route's bug. Stale data stays on
 * screen (`staleError`); the shell marks it (`useServerHealth` → the top
 * bar). When the server answers again, every active resource whose refresh
 * failed is refetched. */

export type ServerHealth = {
  /** The local server is not answering. */
  down: boolean;
  /** When the server last answered (ms epoch). */
  lastOkAt: number | null;
  /** When the current outage was first seen (ms epoch); null while up. */
  downSince: number | null;
  /** The last failure's message while down. */
  lastError: string | null;
};

/** The error body is the app's own JSON (an object): the server answered. */
function fromApp(e: ApiError): boolean {
  return e.body != null && typeof e.body === "object";
}

/** A failure that says the request did not reach a working server: no
 *  response at all, or a 500/502/504 without the app's JSON body (a proxy's
 *  or gateway's answer). */
export function isServerDownError(e: unknown): boolean {
  if (!(e instanceof ApiError)) return false;
  if (e.status === 0) return true;
  return (e.status === 500 || e.status === 502 || e.status === 504) && !fromApp(e);
}

const UP: ServerHealth = { down: false, lastOkAt: null, downSince: null, lastError: null };

export class ServerHealthTracker {
  private state: ServerHealth = UP;
  private failing = new Set<string>();
  private listeners = new Set<() => void>();
  private readonly now: () => number;
  /** Different resources failing with a 5xx before the server counts as down. */
  private readonly distinct: number;

  constructor(opts: { now?: () => number; distinct?: number } = {}) {
    this.now = opts.now ?? Date.now;
    this.distinct = opts.distinct ?? 2;
  }

  getState = (): ServerHealth => this.state;

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };

  /** The server answered (anything but a no-response / proxy error). */
  answered(): void {
    this.failing.clear();
    this.set({ down: false, lastOkAt: this.now(), downSince: null, lastError: null });
  }

  /** `source` (a query key) failed with `error`. */
  failed(source: string, error: unknown): void {
    if (!isServerDownError(error)) return;
    this.failing.add(source);
    const noResponse = (error as ApiError).status === 0;
    if (!noResponse && this.failing.size < this.distinct) return;
    const s = this.state;
    this.set({
      down: true, lastOkAt: s.lastOkAt, downSince: s.down ? s.downSince : this.now(),
      lastError: s.down && s.lastError ? s.lastError : (error as ApiError).message,
    });
  }

  reset(): void {
    this.failing.clear();
    this.set(UP);
  }

  private set(next: ServerHealth): void {
    const s = this.state;
    // `lastOkAt` alone moving is not worth a re-render of every subscriber
    // while up; it is read when the state changes (or via getState()).
    if (s.down === next.down && s.downSince === next.downSince && s.lastError === next.lastError
      && (s.lastOkAt === null) === (next.lastOkAt === null)) {
      this.state = { ...s, lastOkAt: next.lastOkAt };
      return;
    }
    this.state = next;
    for (const l of [...this.listeners]) l();
  }
}

/** The one tracker of the shared query cache. */
export const serverHealth = new ServerHealthTracker();

queryClient.getQueryCache().subscribe((event) => {
  if (event.type === "removed") {
    // Nothing cached any more (e.g. `queryClient.clear()`): nothing is stale.
    if (queryClient.getQueryCache().getAll().length === 0) serverHealth.reset();
    return;
  }
  if (event.type !== "updated") return;
  const { action } = event;
  if (action.type === "success") {
    if (!action.manual) serverHealth.answered();   // a setQueryData is not an answer
  } else if (action.type === "error") {
    const err = action.error;
    if (isServerDownError(err)) serverHealth.failed(JSON.stringify(event.query.queryKey), err);
    else if (err instanceof ApiError) serverHealth.answered();
  }
});

let wasDown = false;
serverHealth.subscribe(() => {
  const { down } = serverHealth.getState();
  if (wasDown && !down) {
    // Back: refresh what failed while it was gone (its stale data is marked).
    void queryClient.refetchQueries({ type: "active", predicate: (q) => q.state.status === "error" });
  }
  wasDown = down;
});

/** The local server's health (re-renders when it goes down or comes back). */
export function useServerHealth(): ServerHealth {
  return useSyncExternalStore(serverHealth.subscribe, serverHealth.getState, serverHealth.getState);
}
