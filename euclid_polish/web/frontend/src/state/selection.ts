/* Cross-view selection: one ordered, de-duplicated id list per scope, e.g.
 * `member` (Ensemble members table ↔ disagreement viewer ↔ curves), `tile`
 * (Sky atlas selection ↔ Results table ↔ Experiments), `star`, `job`.
 * Scopes are free-form strings; ids are the same ids the inspector uses.
 * Session-only (not persisted): a selection is a working set, not a pref.
 * Lists are read-only (typed `readonly string[]` and frozen at runtime): copy
 * before editing (`[...ids]`) and write back through the actions.
 *
 *   const ids = useSelected("member");
 *   useSelection.getState().toggle("member", "member_196");
 */
import { create } from "zustand";

export type SelectionState = { sets: Record<string, readonly string[]> };

export type SelectionActions = {
  get: (scope: string) => readonly string[];
  has: (scope: string, id: string) => boolean;
  /** Replace a scope's selection. */
  select: (scope: string, ids: readonly string[]) => void;
  add: (scope: string, ids: readonly string[]) => void;
  remove: (scope: string, ids: readonly string[]) => void;
  toggle: (scope: string, id: string) => void;
  /** Empty one scope, or every scope when omitted. */
  clear: (scope?: string) => void;
};

export type SelectionStore = SelectionState & SelectionActions;

const EMPTY: readonly string[] = Object.freeze([]);

const uniq = (ids: readonly string[]) => [...new Set(ids)];
const sameList = (a: readonly string[], b: readonly string[]) =>
  a.length === b.length && a.every((v, i) => v === b[i]);

export const useSelection = create<SelectionStore>()((set, get) => {
  /** Write one scope; a no-op (same reference) when the list is unchanged. */
  const write = (scope: string, ids: string[]) => {
    const { sets } = get();
    const prev = sets[scope] ?? EMPTY;
    if (sameList(prev, ids)) return;
    const next = { ...sets };
    if (ids.length) next[scope] = Object.freeze(ids);
    else delete next[scope];
    set({ sets: next });
  };
  return {
    sets: {},
    get: (scope) => get().sets[scope] ?? EMPTY,
    has: (scope, id) => (get().sets[scope] ?? EMPTY).includes(id),
    select: (scope, ids) => write(scope, uniq(ids)),
    add: (scope, ids) => write(scope, uniq([...(get().sets[scope] ?? EMPTY), ...ids])),
    remove: (scope, ids) => {
      const drop = new Set(ids);
      write(scope, (get().sets[scope] ?? EMPTY).filter((id) => !drop.has(id)));
    },
    toggle: (scope, id) => {
      const cur = get().sets[scope] ?? EMPTY;
      write(scope, cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]);
    },
    clear: (scope) => {
      if (scope == null) { if (Object.keys(get().sets).length) set({ sets: {} }); return; }
      write(scope, []);
    },
  };
});

/** The selected ids of a scope (a stable, read-only empty array when none). */
export function useSelected(scope: string): readonly string[] {
  return useSelection((s) => s.sets[scope] ?? EMPTY);
}

/** Whether one id is selected in a scope. */
export function useIsSelected(scope: string, id: string): boolean {
  return useSelection((s) => (s.sets[scope] ?? EMPTY).includes(id));
}
