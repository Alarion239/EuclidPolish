/* Page actions for the command palette (contract C8).
 *
 *   usePageActions([
 *     { id: "evaluate", label: "Evaluate on test set", group: "Ensemble",
 *       keywords: ["psnr"], shortcut: "Shift+E", run: () => evalJob.run(...) },
 *   ]);
 *
 * A page registers its actions while it is mounted; the palette lists them
 * under their `group` (default "This page") next to the routes and the global
 * commands. `run` is read from a ref, so passing new closures on every render
 * costs nothing (the registration only changes when an id, label, group,
 * keyword list, shortcut or `disabled` changes). An action with a `shortcut`
 * (tinykeys syntax, see hooks/useShortcut.ts) is also bound while mounted and
 * listed in the ? sheet; a disabled action ignores its shortcut.
 */
import { useEffect, useId, useMemo, useRef } from "react";
import { create } from "zustand";
import { bindShortcut } from "../hooks/useShortcut";

export type PageAction = {
  id: string;
  label: string;
  group?: string;
  keywords?: string[];
  /** tinykeys combo, e.g. "Shift+E" or "$mod+Enter". */
  shortcut?: string;
  disabled?: boolean;
  run: () => void;
};

type Registry = {
  /** Registering hook instance → its actions (insertion order kept). */
  sources: Record<string, PageAction[]>;
  order: string[];
  set: (source: string, actions: PageAction[]) => void;
  remove: (source: string) => void;
  list: () => PageAction[];
  reset: () => void;
};

export const usePaletteRegistry = create<Registry>()((set, get) => ({
  sources: {},
  order: [],
  set: (source, actions) => {
    const { sources, order } = get();
    set({
      sources: { ...sources, [source]: actions },
      order: order.includes(source) ? order : [...order, source],
    });
  },
  remove: (source) => {
    const { sources, order } = get();
    if (!(source in sources)) return;
    const next = { ...sources };
    delete next[source];
    set({ sources: next, order: order.filter((s) => s !== source) });
  },
  list: () => {
    const { sources, order } = get();
    return order.flatMap((s) => sources[s] ?? []);
  },
  reset: () => set({ sources: {}, order: [] }),
}));

const signature = (actions: PageAction[]) => actions.map((a) => [
  a.id, a.label, a.group ?? "", (a.keywords ?? []).join(","), a.shortcut ?? "", a.disabled ? 1 : 0,
].join("\u0001")).join("\u0002");

/** Register the current page's palette actions (see the module comment). */
export function usePageActions(actions: PageAction[]): void {
  const source = useId();
  const latest = useRef(actions);
  useEffect(() => { latest.current = actions; });
  const sig = signature(actions);
  useEffect(() => {
    const call = (id: string) => () => {
      const a = latest.current.find((x) => x.id === id);
      if (a && !a.disabled) a.run();
    };
    const snapshot = latest.current.map((a) => ({ ...a, run: call(a.id) }));
    usePaletteRegistry.getState().set(source, snapshot);
    const unbind = snapshot
      .filter((a) => a.shortcut)
      .map((a) => bindShortcut(a.shortcut!, () => {
        if (a.disabled) return false;
        a.run();
        return true;
      }, { description: a.label, scope: a.group ?? "This page" }));
    return () => {
      unbind.forEach((off) => off());
      usePaletteRegistry.getState().remove(source);
    };
    // `sig` captures everything the registration depends on.
  }, [source, sig]);
}

/** Every registered page action (re-renders on change). */
export function usePaletteActions(): PageAction[] {
  const sources = usePaletteRegistry((s) => s.sources);
  const order = usePaletteRegistry((s) => s.order);
  return useMemo(() => order.flatMap((s) => sources[s] ?? []), [sources, order]);
}

/* ── query-driven suggestions (global commands that take the typed text) ── */

export type Suggestion =
  | { id: string; label: string; hint?: string; kind: "navigate"; to: string }
  | { id: string; label: string; hint?: string; kind: "inspect"; target: { kind: string; id: string } };

const fmt = (v: number) => String(Number(v.toFixed(5)));

/** What the palette offers for free text, in order:
 *  - "RA Dec" (degrees or sexagesimal) → the Sky atlas centred there
 *    (`/sky/atlas?ra=&dec=`);
 *  - "member 196" / "member_196" → the member inspector (`member:member_196`);
 *  - "nexus 12" / "tile 12" → the NEXUS tile inspector (`tile:nexus/12`);
 *  - a path containing ".fits" → the FITS inspector (`/inspect?fits=`);
 *  - any other text with a letter → "find on the sky" through the atlas's
 *    name resolver (`/sky/atlas?goto=<text>`). */
export function paletteSuggestions(query: string, parseCoord: (text: string) => { ra: number; dec: number } | null): Suggestion[] {
  const text = query.trim();
  if (!text) return [];
  const coord = parseCoord(text);
  if (coord) {
    const ra = fmt(coord.ra); const dec = fmt(coord.dec);
    return [{
      id: "sky-coord", kind: "navigate", label: `Go to RA ${ra}°, Dec ${dec}° on the sky`,
      hint: "Sky › Atlas", to: `/sky/atlas?ra=${ra}&dec=${dec}`,
    }];
  }
  const out: Suggestion[] = [];
  const member = /^member[\s_:#-]*([\w.-]+)$/i.exec(text);
  if (member) {
    const name = /^\d+$/.test(member[1]) ? `member_${member[1]}` : member[1].startsWith("member_") ? member[1] : `member_${member[1]}`;
    out.push({ id: "member", kind: "inspect", label: `Open ${name}`, hint: "inspector", target: { kind: "member", id: name } });
  }
  const tile = /^(?:nexus|tile)[\s_:#/-]*(\d+)$/i.exec(text);
  if (tile) {
    out.push({ id: "tile", kind: "inspect", label: `Open NEXUS tile ${tile[1]}`, hint: "inspector", target: { kind: "tile", id: `nexus/${tile[1]}` } });
  }
  if (/\.fits(\.gz|\.fz)?$/i.test(text) || /\.fits\b/i.test(text)) {
    out.push({ id: "fits", kind: "navigate", label: `Inspect ${text}`, hint: "Inspect", to: `/inspect?fits=${encodeURIComponent(text)}` });
  }
  if (!out.length && /[a-z]/i.test(text) && text.length >= 2) {
    out.push({
      id: "sky-name", kind: "navigate", label: `Find “${text}” on the sky`, hint: "Sky › Atlas · name resolver",
      to: `/sky/atlas?goto=${encodeURIComponent(text)}`,
    });
  }
  return out;
}
