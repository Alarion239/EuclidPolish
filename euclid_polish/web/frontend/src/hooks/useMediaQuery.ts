/* `useMediaQuery("(max-width: 899px)")` → boolean, following changes. False
 * when matchMedia is unavailable (tests, old engines). */
import { useCallback, useSyncExternalStore } from "react";

function mql(query: string): MediaQueryList | null {
  try {
    return typeof window !== "undefined" && typeof window.matchMedia === "function" ? window.matchMedia(query) : null;
  } catch {
    return null;
  }
}

export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback((notify: () => void) => {
    const m = mql(query);
    if (!m) return () => {};
    if (typeof m.addEventListener === "function") {
      m.addEventListener("change", notify);
      return () => m.removeEventListener("change", notify);
    }
    const legacy = m as MediaQueryList & { addListener?: (f: () => void) => void; removeListener?: (f: () => void) => void };
    legacy.addListener?.(notify);
    return () => legacy.removeListener?.(notify);
  }, [query]);
  return useSyncExternalStore(subscribe, () => !!mql(query)?.matches, () => false);
}
