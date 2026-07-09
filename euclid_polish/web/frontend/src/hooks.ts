import { useCallback, useEffect, useRef, useState } from "react";
import { getJSON } from "./api";

export type Resource<T> = {
  data: T | null;
  loading: boolean;
  error: boolean;
  reload: () => void;
};

/** Fetch JSON from `url` (null → skip), tracking loading/error and exposing a
 *  manual `reload`. The DRY data-loading primitive every page uses. `deps`
 *  refetches when they change (e.g. a subset/mode selector). */
export function useResource<T = unknown>(
  url: string | null,
  deps: React.DependencyList = [],
): Resource<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(!!url);
  const [error, setError] = useState(false);
  const [nonce, setNonce] = useState(0);
  const reload = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    if (!url) { setData(null); setLoading(false); return; }
    let live = true;
    setLoading(true);
    setError(false);
    getJSON<T>(url).then((d) => {
      if (!live) return;
      if (d == null) setError(true);
      setData(d);
      setLoading(false);
    });
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, nonce, ...deps]);

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
