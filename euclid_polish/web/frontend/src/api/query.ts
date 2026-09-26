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
import { useCallback, useEffect, useRef, type DependencyList } from "react";
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
