/* Shared React hooks. `useResource` lives in the data layer (api/query.ts)
   and is re-exported here for the pages that import it from "../hooks". */
import { useEffect, useRef } from "react";
import { invalidateMatching } from "../api/query";

export { DEFAULT_TTL_MS, invalidate, useResource } from "../api/query";
export type { Resource, ResourceOpts } from "../api/query";
export { useUrlState } from "./useUrlState";
export type { UrlStateOpts } from "./useUrlState";

/** Invalidate cached resources whose URL contains `substr` (all when
 *  omitted). Old name; prefer `invalidate(prefix)` from api/query. */
export function invalidateCache(substr?: string): Promise<void> {
  return invalidateMatching(substr ?? "");
}

/** Run `fn` now and every `ms` while `active` and the tab is visible; runs
 *  once more when the tab becomes visible again. `fn` owns its state writes.
 *  Prefer `useResource(url, [], {poll})` for plain GET polling. */
export function usePolling(fn: () => void, ms: number, active = true): void {
  const saved = useRef(fn);
  saved.current = fn;
  useEffect(() => {
    if (!active) return;
    const hidden = () => typeof document !== "undefined" && document.visibilityState === "hidden";
    let id: ReturnType<typeof setInterval> | undefined;
    const start = () => {
      if (id != null || hidden()) return;
      saved.current();
      id = setInterval(() => { if (!hidden()) saved.current(); }, ms);
    };
    const stop = () => { if (id != null) { clearInterval(id); id = undefined; } };
    const onVisibility = () => (hidden() ? stop() : start());
    start();
    document.addEventListener("visibilitychange", onVisibility);
    return () => { stop(); document.removeEventListener("visibilitychange", onVisibility); };
  }, [ms, active]);
}
