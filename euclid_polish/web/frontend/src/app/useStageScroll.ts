/* Scroll management of the shell's scrolling stage (the page content scrolls
 * inside `<main class="stage">`, not the window, so react-router's
 * <ScrollRestoration> — which drives window scrolling — does not apply):
 *   - a change of PATHNAME scrolls to the top, except back/forward (POP),
 *     which restores the position that history entry had;
 *   - a query-only change (useUrlState writes, `?inspect=`) keeps the scroll.
 * Positions are kept per history entry for the session. */
import { useEffect, useLayoutEffect, useRef } from "react";
import { useLocation, useNavigationType } from "react-router-dom";

export function useStageScroll<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const location = useLocation();
  const navType = useNavigationType();
  const positions = useRef(new Map<string, number>());
  const currentKey = useRef(location.key);
  const lastPath = useRef(location.pathname);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onScroll = () => { positions.current.set(currentKey.current, el.scrollTop); };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useLayoutEffect(() => {
    const el = ref.current;
    currentKey.current = location.key;
    if (lastPath.current === location.pathname) return;
    lastPath.current = location.pathname;
    if (!el) return;
    const saved = navType === "POP" ? positions.current.get(location.key) : undefined;
    el.scrollTop = saved ?? 0;
  }, [location.key, location.pathname, navType]);

  return ref;
}
