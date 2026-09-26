/* Inspector store — what the global right-hand Inspector panel shows.
 *
 * A target is `{kind, id}` (e.g. member:member_196, tile:nexus/123,
 * job:local/<id>, fits:<path>); it is mirrored in the URL as
 * `?inspect=kind:id` by the shell (app/inspector.ts, WP-F3), which also owns
 * the kind → component registry. This module is the state only:
 * current target, open flag, back/forward history, pins (persisted).
 * The panel width is a layout pref (`usePrefs().inspectorWidth`).
 */
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { safeJSONStorage } from "./storage";

export type InspectTarget = { kind: string; id: string };

export type InspectorState = {
  open: boolean;
  current: InspectTarget | null;
  /** Older entries, oldest first (the last one is "back"). */
  back: InspectTarget[];
  /** Entries undone by goBack, nearest first. */
  forward: InspectTarget[];
  pinned: InspectTarget[];
};

export type InspectorActions = {
  /** Open `target` (a new history entry unless it is already current). */
  show: (target: InspectTarget, opts?: { replace?: boolean }) => void;
  /** Close the panel, keeping the current target (reopen with setOpen). */
  hide: () => void;
  setOpen: (open: boolean) => void;
  /** Close and forget the current target and the history. */
  clear: () => void;
  goBack: () => void;
  goForward: () => void;
  /** Pin / unpin a target (the current one by default). */
  togglePin: (target?: InspectTarget) => void;
  isPinned: (target: InspectTarget | null) => boolean;
  reset: () => void;
};

export type InspectorStore = InspectorState & InspectorActions;

export const HISTORY_LIMIT = 50;
export const INSPECTOR_STORAGE_KEY = "ep-inspector";

const KIND_RE = /^[a-z][a-z0-9_-]*$/i;

/** `{kind, id}` → "kind:id" (the `?inspect=` value). */
export function formatInspectParam(t: InspectTarget): string {
  return `${t.kind}:${t.id}`;
}

/** "kind:id" → target (split at the FIRST colon; ids may contain colons).
 *  Null for anything malformed. */
export function parseInspectParam(raw: string | null | undefined): InspectTarget | null {
  if (!raw) return null;
  const i = raw.indexOf(":");
  if (i <= 0) return null;
  const kind = raw.slice(0, i);
  const id = raw.slice(i + 1);
  if (!KIND_RE.test(kind) || !id) return null;
  return { kind, id };
}

export function sameTarget(a: InspectTarget | null | undefined, b: InspectTarget | null | undefined): boolean {
  return !!a && !!b && a.kind === b.kind && a.id === b.id;
}

const EMPTY: InspectorState = { open: false, current: null, back: [], forward: [], pinned: [] };
const copy = (t: InspectTarget): InspectTarget => ({ kind: t.kind, id: t.id });

export const useInspector = create<InspectorStore>()(
  persist(
    (set, get) => ({
      ...EMPTY,
      show: (target, opts = {}) => {
        const { current, back } = get();
        const next = copy(target);
        if (sameTarget(current, next)) { set({ open: true }); return; }
        if (opts.replace || !current) {
          set({ open: true, current: next, forward: opts.replace ? get().forward : [] });
          return;
        }
        set({ open: true, current: next, back: [...back, current].slice(-HISTORY_LIMIT), forward: [] });
      },
      hide: () => set({ open: false }),
      setOpen: (open) => set({ open: open && get().current != null }),
      clear: () => set({ open: false, current: null, back: [], forward: [] }),
      goBack: () => {
        const { current, back, forward } = get();
        if (!back.length) return;
        const prev = back[back.length - 1];
        set({ current: prev, back: back.slice(0, -1), forward: current ? [current, ...forward] : forward, open: true });
      },
      goForward: () => {
        const { current, back, forward } = get();
        if (!forward.length) return;
        const [next, ...rest] = forward;
        set({ current: next, forward: rest, back: current ? [...back, current].slice(-HISTORY_LIMIT) : back, open: true });
      },
      togglePin: (target) => {
        const t = target ?? get().current;
        if (!t) return;
        const { pinned } = get();
        set({
          pinned: pinned.some((p) => sameTarget(p, t))
            ? pinned.filter((p) => !sameTarget(p, t))
            : [...pinned, copy(t)],
        });
      },
      isPinned: (target) => !!target && get().pinned.some((p) => sameTarget(p, target)),
      reset: () => set({ ...EMPTY }),
    }),
    {
      name: INSPECTOR_STORAGE_KEY,
      version: 1,
      storage: safeJSONStorage,
      partialize: (s) => ({ pinned: s.pinned }),
      merge: (persisted, current) => {
        const raw = (persisted as { pinned?: unknown } | undefined)?.pinned;
        const pinned = Array.isArray(raw)
          ? raw.flatMap((p) => {
            const t = p && typeof p === "object" ? p as Partial<InspectTarget> : {};
            return typeof t.kind === "string" && typeof t.id === "string" && KIND_RE.test(t.kind) && t.id
              ? [{ kind: t.kind, id: t.id }] : [];
          })
          : [];
        return { ...current, pinned };
      },
    },
  ),
);
