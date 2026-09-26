/* User preferences store: theme (light | dark | system), accent, density,
 * rail collapsed, inspector width. Persisted to localStorage ("ep-prefs",
 * try/catch-guarded) and sanitised on load.
 *
 * Theme mechanics: the pre-paint script in index.html and
 * `bindPrefsToDocument()` (called once by main.tsx) resolve "system" through
 * `prefers-color-scheme` and write the RESOLVED theme to <html data-theme>
 * (plus data-theme-pref, data-accent, data-density), which is what
 * theme/tokens.css keys on. Components that must recompute on a theme flip
 * (canvas figures reading tokens) use `useResolvedTheme()`.
 *
 * The pre-rework key "ep-theme" ("light" | "dark") is migrated on first load.
 */
import { useSyncExternalStore } from "react";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { readStorage, safeJSONStorage } from "./storage";

export type ThemePref = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";
export type Accent = "blue" | "violet" | "teal" | "amber" | "rose";
export type Density = "comfortable" | "compact";

export type Prefs = {
  theme: ThemePref;
  /** Accent preset (tokens.css `[data-accent]`); "blue" is the default. */
  accent: Accent;
  density: Density;
  railCollapsed: boolean;
  /** Inspector panel width in px. */
  inspectorWidth: number;
};

export type PrefsActions = {
  set: (patch: Partial<Prefs>) => void;
  setTheme: (theme: ThemePref) => void;
  /** Flip the currently RESOLVED theme to the other one (an explicit choice). */
  toggleTheme: () => void;
  toggleRail: () => void;
  reset: () => void;
};

export type PrefsStore = Prefs & PrefsActions;

export const THEME_PREFS: readonly ThemePref[] = ["light", "dark", "system"];
export const ACCENTS: readonly Accent[] = ["blue", "violet", "teal", "amber", "rose"];
export const DENSITIES: readonly Density[] = ["comfortable", "compact"];
export const INSPECTOR_WIDTH_RANGE: [number, number] = [260, 1200];

export const DEFAULT_PREFS: Readonly<Prefs> = Object.freeze({
  theme: "light",
  accent: "blue",
  density: "comfortable",
  railCollapsed: false,
  inspectorWidth: 380,
});

export const PREFS_STORAGE_KEY = "ep-prefs";
/** Pre-rework theme key ("light" | "dark"), read once for migration. */
export const LEGACY_THEME_KEY = "ep-theme";

const oneOf = <T extends string>(v: unknown, options: readonly T[], fallback: T): T =>
  (typeof v === "string" && (options as readonly string[]).includes(v) ? v as T : fallback);

/** Coerce anything (persisted JSON, a patch) into valid prefs. An invalid
 *  field falls back to `fallback` (the defaults by default). */
export function sanitizePrefs(raw: unknown, fallback: Prefs = DEFAULT_PREFS): Prefs {
  const r = (raw && typeof raw === "object" ? raw : {}) as Partial<Record<keyof Prefs, unknown>>;
  const [wMin, wMax] = INSPECTOR_WIDTH_RANGE;
  const w = r.inspectorWidth;
  return {
    theme: oneOf(r.theme, THEME_PREFS, fallback.theme),
    accent: oneOf(r.accent, ACCENTS, fallback.accent),
    density: oneOf(r.density, DENSITIES, fallback.density),
    railCollapsed: typeof r.railCollapsed === "boolean" ? r.railCollapsed : fallback.railCollapsed,
    inspectorWidth: typeof w === "number" && Number.isFinite(w) && w > 0
      ? Math.round(Math.min(wMax, Math.max(wMin, w))) : fallback.inspectorWidth,
  };
}

/** The prefs to start from: the saved ones, else the migrated legacy theme. */
export function initialPrefs(): Prefs {
  const saved = readStorage(PREFS_STORAGE_KEY);
  if (saved) {
    try {
      const parsed = JSON.parse(saved) as { state?: unknown };
      return sanitizePrefs(parsed?.state);
    } catch { /* corrupt → defaults */ }
    return { ...DEFAULT_PREFS };
  }
  const legacy = readStorage(LEGACY_THEME_KEY);
  return { ...DEFAULT_PREFS, theme: legacy === "dark" || legacy === "light" ? legacy : DEFAULT_PREFS.theme };
}

/* ── system colour scheme ────────────────────────────────────────────────── */

const DARK_QUERY = "(prefers-color-scheme: dark)";

function darkQuery(): MediaQueryList | null {
  try {
    const mm = (globalThis as { matchMedia?: (q: string) => MediaQueryList }).matchMedia;
    return typeof mm === "function" ? mm.call(globalThis, DARK_QUERY) : null;
  } catch {
    return null;
  }
}

/** The OS colour scheme ("light" when unknown). */
export function systemTheme(): ResolvedTheme {
  return darkQuery()?.matches ? "dark" : "light";
}

/** light | dark for a preference ("system" → the OS scheme). */
export function resolveTheme(pref: ThemePref): ResolvedTheme {
  return pref === "system" ? systemTheme() : pref;
}

/** Subscribe to OS colour-scheme changes; returns the unsubscribe. */
function onSystemThemeChange(fn: () => void): () => void {
  const mql = darkQuery();
  if (!mql) return () => {};
  const legacy = mql as MediaQueryList & {
    addListener?: (f: () => void) => void; removeListener?: (f: () => void) => void;
  };
  if (typeof mql.addEventListener === "function") {
    mql.addEventListener("change", fn);
    return () => mql.removeEventListener("change", fn);
  }
  legacy.addListener?.(fn);
  return () => legacy.removeListener?.(fn);
}

/* ── store ───────────────────────────────────────────────────────────────── */

const prefsOf = (s: PrefsStore): Prefs => ({
  theme: s.theme, accent: s.accent, density: s.density, railCollapsed: s.railCollapsed,
  inspectorWidth: s.inspectorWidth,
});

export const usePrefs = create<PrefsStore>()(
  persist(
    (set, get) => ({
      ...initialPrefs(),
      set: (patch) => { const cur = prefsOf(get()); set(sanitizePrefs({ ...cur, ...patch }, cur)); },
      setTheme: (theme) => { const cur = prefsOf(get()); set(sanitizePrefs({ ...cur, theme }, cur)); },
      toggleTheme: () => set({ theme: resolveTheme(get().theme) === "dark" ? "light" : "dark" }),
      toggleRail: () => set({ railCollapsed: !get().railCollapsed }),
      reset: () => set({ ...DEFAULT_PREFS }),
    }),
    {
      name: PREFS_STORAGE_KEY,
      version: 1,
      storage: safeJSONStorage,
      partialize: (s) => prefsOf(s),
      // persist calls merge(undefined, current) when nothing is saved yet:
      // keep `current` then (it holds the migrated legacy "ep-theme").
      merge: (persisted, current) => (persisted == null
        ? current
        : { ...current, ...sanitizePrefs(persisted, prefsOf(current)) }),
    },
  ),
);

/** The resolved theme, re-rendering on a pref change or an OS scheme flip. */
export function useResolvedTheme(): ResolvedTheme {
  const pref = usePrefs((s) => s.theme);
  const system = useSyncExternalStore(onSystemThemeChange, systemTheme, () => "light" as const);
  return pref === "system" ? system : pref;
}

/** Apply the prefs to <html> now and keep them applied (store changes + OS
 *  scheme flips). Returns the unbind function. Called once by main.tsx. */
export function bindPrefsToDocument(root: HTMLElement = document.documentElement): () => void {
  const apply = () => {
    const p = usePrefs.getState();
    root.setAttribute("data-theme", resolveTheme(p.theme));
    root.setAttribute("data-theme-pref", p.theme);
    root.setAttribute("data-accent", p.accent);
    root.setAttribute("data-density", p.density);
  };
  apply();
  const unsubStore = usePrefs.subscribe(apply);
  const unsubSystem = onSystemThemeChange(apply);
  return () => { unsubStore(); unsubSystem(); };
}
