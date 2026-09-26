/* Keyboard shortcuts (contract C8) on tinykeys, with a registry that feeds
 * the "?" cheat sheet.
 *
 *   useShortcut("$mod+k", openPalette, { description: "Command palette", allowInInputs: true });
 *   useShortcut("g s", () => navigate("/sky/atlas"), { description: "Go to Sky", scope: "Navigation" });
 *   useShortcut("ArrowRight", next, { description: "Next object", target: viewerRef, scope: "Viewer" });
 *   const off = bindShortcut("g h", goHome, { description: "Go to Home" });  // imperative
 *
 * Combos use tinykeys syntax: `$mod` is ⌘ on macOS and Ctrl elsewhere,
 * `Shift+?`, a space separates the presses of a sequence ("g s"), and
 * `[Shift]` marks an optional modifier.
 *
 * A shortcut is skipped when:
 *   - the key is typed into an input, textarea, select or contenteditable
 *     (unless `allowInInputs`);
 *   - focus is inside a modal dialog (`role=dialog|alertdialog`, unless
 *     `allowInInputs`), so "g s" cannot navigate behind an open dialog;
 *   - another handler already called `preventDefault()` on the event (the
 *     image viewer consumes its own keys on `document`, which runs before
 *     the window listeners here);
 *   - it is a key-repeat or IME composition.
 * A handler that fires calls `preventDefault()` unless it returns `false`.
 *
 * Page shortcuts register on mount and unregister on unmount. `target`
 * scopes a shortcut to an element (e.g. the focused viewer) instead of the
 * window. The handler is read from a ref, so a new closure on every render
 * never rebinds.
 */
import { useEffect, useRef, type RefObject } from "react";
import { tinykeys } from "tinykeys";
import { create } from "zustand";

export type ShortcutHandler = (event: KeyboardEvent) => void | boolean;

export type ShortcutOptions = {
  /** What it does (shown in the ? sheet). */
  description: string;
  /** Group heading in the ? sheet (default "Global"). */
  scope?: string;
  /** Fire even while typing in a field or inside a dialog (e.g. $mod+k). */
  allowInInputs?: boolean;
  /** Listen on this element (or ref) instead of the window. */
  target?: RefObject<HTMLElement | null> | HTMLElement | Window | null;
  /** Default true; false unbinds (and drops the sheet entry). */
  enabled?: boolean;
  /** Keep it out of the ? sheet (still bound). */
  hidden?: boolean;
};

export type ShortcutEntry = {
  id: number;
  combo: string;
  description: string;
  scope: string;
  hidden: boolean;
};

type Registry = {
  entries: ShortcutEntry[];
  add: (entry: ShortcutEntry) => void;
  remove: (id: number) => void;
  reset: () => void;
};

/** Every bound shortcut, in registration order (the ? sheet reads this). */
export const useShortcutRegistry = create<Registry>()((set, get) => ({
  entries: [],
  add: (entry) => set({ entries: [...get().entries, entry] }),
  remove: (id) => set({ entries: get().entries.filter((e) => e.id !== id) }),
  reset: () => set({ entries: [] }),
}));

let nextId = 1;

const EDITABLE = "input, textarea, select, [contenteditable]:not([contenteditable='false'])";
const MODAL = "[role='dialog'], [role='alertdialog'], [aria-modal='true']";

function isElement(t: EventTarget | null): t is Element {
  return !!t && typeof (t as Element).closest === "function";
}

function ignoreFor(allowInInputs: boolean) {
  return (event: KeyboardEvent): boolean => {
    if (event.repeat || event.isComposing) return true;
    if (allowInInputs) return false;
    const t = event.target;
    if (!isElement(t)) return false;
    if ((t as HTMLElement).isContentEditable || t.matches(EDITABLE)) return true;
    return t.closest(MODAL) != null;
  };
}

type Target = HTMLElement | Window;

function resolveTarget(target: ShortcutOptions["target"]): Target | null {
  if (target == null) return typeof window === "undefined" ? null : window;
  if (typeof window !== "undefined" && target === window) return window;
  if (target instanceof HTMLElement) return target;
  return (target as RefObject<HTMLElement | null>).current ?? null;
}

/** Bind `combo` now; returns the unbind function (also drops the sheet entry). */
export function bindShortcut(combo: string, handler: ShortcutHandler, opts: ShortcutOptions): () => void {
  if (opts.enabled === false) return () => {};
  const target = resolveTarget(opts.target);
  if (!target) return () => {};
  const unbind = tinykeys(target, {
    [combo]: (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      if (handler(event) !== false) event.preventDefault();
    },
  }, { ignore: ignoreFor(!!opts.allowInInputs) });
  const id = nextId++;
  useShortcutRegistry.getState().add({
    id, combo, description: opts.description, scope: opts.scope ?? "Global", hidden: !!opts.hidden,
  });
  let done = false;
  return () => {
    if (done) return;
    done = true;
    unbind();
    useShortcutRegistry.getState().remove(id);
  };
}

/** Bind a shortcut for the lifetime of the component. */
export function useShortcut(combo: string, handler: ShortcutHandler, opts: ShortcutOptions): void {
  const latest = useRef(handler);
  useEffect(() => { latest.current = handler; });
  const { description, scope, allowInInputs, target, enabled, hidden } = opts;
  useEffect(() => bindShortcut(combo, (e) => latest.current(e), {
    description, scope, allowInInputs, target, enabled, hidden,
  }), [combo, description, scope, allowInInputs, target, enabled, hidden]);
}

/** A tinykeys combo as display strings for `<Kbd keys>`, one per press:
 *  "$mod+k" → ["mod+k"], "g s" → ["g", "s"], "Shift+?" → ["?"]. */
export function comboParts(combo: string): string[] {
  return combo.trim().split(/\s+/).map((press) => {
    const parts = press.split(/(?<=\w|\])\+/);
    const key = parts.pop() ?? "";
    const symbol = key.length === 1 && !/[a-z0-9]/i.test(key);
    const mods = parts
      .filter((m) => !/^\[.*\]$/.test(m))
      .map((m) => (m === "$mod" ? "mod" : m.toLowerCase()))
      .filter((m) => !(symbol && m === "shift"));
    return [...mods, key].join("+");
  });
}
