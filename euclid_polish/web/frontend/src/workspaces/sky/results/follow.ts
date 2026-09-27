/* Keep a nav-less image viewer on one object with the page's tiers.
 *
 * A comparison viewer (Experiments, the real-tile card) shows ONE tile of a
 * collection that walks every tile of the source; the page, not the viewer,
 * decides which (the Scope select, the inspected tile). `useFollowViewer`
 * moves the viewer there once its meta is in, and then puts back the tiers the
 * page asked for when the viewer dropped some on mount (it prunes the
 * initial tiers against the collection's FIRST object before it moves to
 * `initialId`, so a model output the first tile lacks would be lost). The
 * tiers are restored once per mount, so a tier the user toggles later stays
 * toggled. */
import { useCallback, useEffect, useRef } from "react";
import type { ViewerApi } from "../../../viewer";

export type FollowState = { id: string | null; tiers: readonly string[] | undefined };
export type FollowStep =
  | { kind: "wait" }
  | { kind: "go"; id: string }
  | { kind: "tiers"; tiers: string[] }
  | { kind: "done" };

/** The follower's next step (pure). */
export function followStep(
  state: FollowState | null, want: string, tiers: readonly string[] | null | undefined, tiersDone: boolean,
): FollowStep {
  if (!state || state.id == null) return { kind: "wait" };
  if (want && state.id !== want) return { kind: "go", id: want };
  const have = state.tiers ?? [];
  if (!tiersDone && tiers?.length && tiers.some((t) => !have.includes(t))) return { kind: "tiers", tiers: [...tiers] };
  return { kind: "done" };
}

/** Returns the viewer's `onReady` / `onState` handlers. */
export function useFollowViewer(want: string, tiers?: readonly string[] | null) {
  const api = useRef<ViewerApi | null>(null);
  const wantRef = useRef(want);
  wantRef.current = want;
  const tiersRef = useRef(tiers);
  tiersRef.current = tiers;
  const armed = useRef(true);
  const busy = useRef(false);
  const tiersDone = useRef(false);
  const tried = useRef<string | null>(null);         // the id last gone to (never loop on it)
  const ensure = useCallback(() => {
    const a = api.current;
    if (!a || !armed.current || busy.current) return;
    const s = a.getState();
    const step = followStep({ id: s.id ?? null, tiers: s.tiers }, wantRef.current, tiersRef.current, tiersDone.current);
    if (step.kind === "wait") return;                  // meta not in yet: onState re-tries
    if (step.kind === "done") { armed.current = false; return; }
    if (step.kind === "tiers") {
      tiersDone.current = true;                         // once: never fight the user's tier choice
      a.setTiers(step.tiers);
      armed.current = false;
      return;
    }
    if (tried.current === step.id) { armed.current = false; return; }   // went there and did not arrive
    tried.current = step.id;
    busy.current = true;
    void a.goToId(step.id).then(
      (ok) => {
        busy.current = false;
        if (!ok && wantRef.current === step.id) { armed.current = false; return; }   // unknown id: stop
        ensure();                                       // reached (tiers next), or the target moved on
      },
      () => { busy.current = false; armed.current = false; },
    );
  }, []);
  useEffect(() => { armed.current = true; tried.current = null; ensure(); }, [want, ensure]);
  const onReady = useCallback((a: ViewerApi | null) => {
    api.current = a;
    armed.current = true;
    busy.current = false;
    tiersDone.current = false;
    tried.current = null;
    ensure();
  }, [ensure]);
  const onState = useCallback(() => ensure(), [ensure]);
  return { onReady, onState };
}
