import { useCallback, useEffect, useRef, useState } from "react";
import { getJSON } from "./api";

export type Resource<T> = {
  data: T | null;
  loading: boolean;
  error: boolean;
  reload: () => void;
};

/* ── shared response cache ─────────────────────────────────────────────────
   A process-wide Map keyed by request URL. Navigating between pages (or back
   to one) reuses the last response INSTANTLY instead of re-hitting the server
   — several endpoints do real work per request (status walks, viewer meta,
   listings), so this is the difference between snappy and sluggish. Freshness
   is preserved two ways: stale entries revalidate in the background (SWR), and
   `reload()` (the ↻ buttons + every post-job onDone) always force-fetches and
   overwrites the entry. */
type Entry = { data: unknown; ts: number };
const CACHE = new Map<string, Entry>();

/** How long a cached entry is served without a background refetch. Within this
    window navigation makes ZERO server calls; past it, the cached copy shows
    immediately while a fresh one loads. */
export const DEFAULT_TTL_MS = 5 * 60_000;

/** Drop cached entries whose URL contains `substr` (or the whole cache when
    omitted) — for the rare mutation that invalidates unrelated resources. */
export function invalidateCache(substr?: string): void {
  if (!substr) { CACHE.clear(); return; }
  for (const k of CACHE.keys()) if (k.includes(substr)) CACHE.delete(k);
}

/** Fetch JSON from `url` (null → skip), cache-first with stale-while-revalidate.
 *  The DRY data-loading primitive every page uses. `deps` refetch when they
 *  change (e.g. a subset/mode selector); `reload()` forces a fresh fetch. */
export function useResource<T = unknown>(
  url: string | null,
  deps: React.DependencyList = [],
  opts: { ttl?: number } = {},
): Resource<T> {
  const ttl = opts.ttl ?? DEFAULT_TTL_MS;
  const cached = url ? CACHE.get(url) : undefined;
  const [data, setData] = useState<T | null>(() => (cached ? (cached.data as T) : null));
  const [loading, setLoading] = useState<boolean>(!!url && !cached);
  const [error, setError] = useState(false);
  const [nonce, setNonce] = useState(0);
  const lastNonce = useRef(0);
  const reload = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    if (!url) { setData(null); setLoading(false); setError(false); return; }
    const force = nonce !== lastNonce.current;
    lastNonce.current = nonce;

    const entry = CACHE.get(url);
    if (entry) { setData(entry.data as T); setError(false); setLoading(false); }
    else {
      // A dependency/URL change must not leave the previous resource on screen.
      // This matters for mode-specific pages: if the new URL 404s, retaining the
      // old response permanently presents the other mode's artifacts as current.
      setData(null);
      setError(false);
      setLoading(true);
    }

    // Fresh-enough cache and not an explicit reload → serve it, no server hit.
    const fresh = entry != null && Date.now() - entry.ts < ttl;
    if (entry && fresh && !force) return;

    // Otherwise (re)validate. Cached copy (if any) stays on screen meanwhile.
    let live = true;
    getJSON<T>(url).then((d) => {
      if (!live) return;
      if (d == null) { if (!entry) setError(true); }
      else { CACHE.set(url, { data: d, ts: Date.now() }); setData(d); setError(false); }
      setLoading(false);
    });
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, nonce, ttl, ...deps]);

  return { data, loading, error, reload };
}

/** Run `fn` now and every `ms` while `active`. Cleans up on unmount / when
 *  `active` flips false. Returns nothing — `fn` owns its own state writes. */
export function usePolling(fn: () => void, ms: number, active = true): void {
  const saved = useRef(fn);
  saved.current = fn;
  useEffect(() => {
    if (!active) return;
    saved.current();
    const id = setInterval(() => saved.current(), ms);
    return () => clearInterval(id);
  }, [ms, active]);
}
