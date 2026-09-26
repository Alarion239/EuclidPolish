import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ACCENTS,
  DEFAULT_PREFS,
  LEGACY_THEME_KEY,
  PREFS_STORAGE_KEY,
  bindPrefsToDocument,
  initialPrefs,
  resolveTheme,
  sanitizePrefs,
  systemTheme,
  usePrefs,
  useResolvedTheme,
} from "./prefs";

/** A controllable `prefers-color-scheme: dark` media query. */
function mockSystemDark(initial: boolean) {
  const listeners = new Set<(e: { matches: boolean }) => void>();
  const mql = {
    matches: initial,
    media: "(prefers-color-scheme: dark)",
    addEventListener: (_: string, fn: (e: { matches: boolean }) => void) => listeners.add(fn),
    removeEventListener: (_: string, fn: (e: { matches: boolean }) => void) => listeners.delete(fn),
  };
  vi.stubGlobal("matchMedia", vi.fn(() => mql));
  return {
    set(dark: boolean) { mql.matches = dark; for (const fn of listeners) fn({ matches: dark }); },
    listeners,
  };
}

const html = () => document.documentElement;

beforeEach(() => {
  localStorage.clear();
  usePrefs.getState().reset();
  for (const a of ["data-theme", "data-theme-pref", "data-accent", "data-density"]) html().removeAttribute(a);
});
afterEach(() => { vi.unstubAllGlobals(); });

describe("prefs store", () => {
  it("defaults to the light theme, blue accent, comfortable density, open rail", () => {
    expect(DEFAULT_PREFS).toMatchObject({
      theme: "light", accent: "blue", density: "comfortable", railCollapsed: false,
    });
    expect(usePrefs.getState()).toMatchObject(DEFAULT_PREFS);
    expect(ACCENTS[0]).toBe("blue");
  });

  it("set / setTheme / toggleRail update and persist", () => {
    usePrefs.getState().set({ accent: "teal", density: "compact" });
    usePrefs.getState().setTheme("system");
    usePrefs.getState().toggleRail();
    const s = usePrefs.getState();
    expect(s).toMatchObject({ accent: "teal", density: "compact", theme: "system", railCollapsed: true });
    const saved = JSON.parse(localStorage.getItem(PREFS_STORAGE_KEY)!);
    expect(saved.state).toMatchObject({ accent: "teal", theme: "system", railCollapsed: true });
    expect(saved.state.set).toBeUndefined();
  });

  it("sanitises invalid values", () => {
    expect(sanitizePrefs({ theme: "neon", accent: "plaid", density: 3, railCollapsed: "yes", inspectorWidth: -4 }))
      .toEqual(DEFAULT_PREFS);
    expect(sanitizePrefs({ inspectorWidth: 10_000 }).inspectorWidth).toBeLessThanOrEqual(1200);
    usePrefs.getState().set({ theme: "bogus" as never });
    expect(usePrefs.getState().theme).toBe("light");
    usePrefs.getState().set({ theme: "dark", accent: "rose" });
    usePrefs.getState().set({ theme: "bogus" as never, accent: 1 as never, density: "compact" });
    expect(usePrefs.getState()).toMatchObject({ theme: "dark", accent: "rose", density: "compact" });
  });

  it("migrates the pre-rework `ep-theme` key when no prefs were saved", () => {
    localStorage.clear();                         // reset() in beforeEach saved prefs
    expect(initialPrefs().theme).toBe("light");
    localStorage.setItem(LEGACY_THEME_KEY, "dark");
    expect(initialPrefs().theme).toBe("dark");
    localStorage.setItem(PREFS_STORAGE_KEY, JSON.stringify({ state: { theme: "light" }, version: 1 }));
    expect(initialPrefs().theme).toBe("light");
  });

  it("the STORE starts from the migrated `ep-theme` (persist hydration keeps it)", async () => {
    localStorage.clear();
    localStorage.setItem(LEGACY_THEME_KEY, "dark");
    vi.resetModules();
    const fresh = await import("./prefs");
    expect(fresh.initialPrefs().theme).toBe("dark");
    expect(fresh.usePrefs.getState().theme).toBe("dark");
    const unbind = fresh.bindPrefsToDocument();
    expect(html().getAttribute("data-theme")).toBe("dark");    // no flash back to light
    unbind();
  });

  it("the STORE hydrates saved prefs over the legacy key", async () => {
    localStorage.clear();
    localStorage.setItem(LEGACY_THEME_KEY, "dark");
    localStorage.setItem(PREFS_STORAGE_KEY, JSON.stringify({ state: { theme: "system", accent: "teal" }, version: 1 }));
    vi.resetModules();
    const fresh = await import("./prefs");
    expect(fresh.usePrefs.getState()).toMatchObject({ theme: "system", accent: "teal", density: "comfortable" });
  });

  it("the STORE sanitises a hand-edited saved value field by field", async () => {
    localStorage.clear();
    localStorage.setItem(PREFS_STORAGE_KEY, JSON.stringify({ state: { theme: "neon", density: "compact" }, version: 1 }));
    vi.resetModules();
    const fresh = await import("./prefs");
    expect(fresh.usePrefs.getState()).toMatchObject({ theme: "light", density: "compact" });
  });

  it("keeps working when localStorage throws", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("quota"); });
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("denied"); });
    expect(() => usePrefs.getState().set({ accent: "rose" })).not.toThrow();
    expect(usePrefs.getState().accent).toBe("rose");
    expect(initialPrefs()).toMatchObject({ theme: "light" });
  });
});

describe("theme resolution", () => {
  it("resolves system through prefers-color-scheme", () => {
    const sys = mockSystemDark(true);
    expect(systemTheme()).toBe("dark");
    expect(resolveTheme("system")).toBe("dark");
    expect(resolveTheme("light")).toBe("light");
    sys.set(false);
    expect(resolveTheme("system")).toBe("light");
  });

  it("falls back to light without matchMedia", () => {
    vi.stubGlobal("matchMedia", undefined);
    expect(systemTheme()).toBe("light");
  });

  it("toggleTheme flips the RESOLVED theme to an explicit choice", () => {
    mockSystemDark(true);
    usePrefs.getState().setTheme("system");
    usePrefs.getState().toggleTheme();
    expect(usePrefs.getState().theme).toBe("light");
    usePrefs.getState().toggleTheme();
    expect(usePrefs.getState().theme).toBe("dark");
  });

  it("useResolvedTheme follows the pref and the OS scheme", () => {
    const sys = mockSystemDark(false);
    const { result } = renderHook(() => useResolvedTheme());
    expect(result.current).toBe("light");
    act(() => usePrefs.getState().setTheme("dark"));
    expect(result.current).toBe("dark");
    act(() => usePrefs.getState().setTheme("system"));
    expect(result.current).toBe("light");
    act(() => sys.set(true));
    expect(result.current).toBe("dark");
  });
});

describe("bindPrefsToDocument", () => {
  it("writes the resolved theme, pref, accent and density onto <html> and tracks changes", () => {
    const sys = mockSystemDark(false);
    const unbind = bindPrefsToDocument();
    expect(html().getAttribute("data-theme")).toBe("light");
    expect(html().getAttribute("data-theme-pref")).toBe("light");
    expect(html().getAttribute("data-accent")).toBe("blue");
    expect(html().getAttribute("data-density")).toBe("comfortable");

    usePrefs.getState().set({ theme: "system", accent: "amber", density: "compact" });
    expect(html().getAttribute("data-theme")).toBe("light");
    expect(html().getAttribute("data-theme-pref")).toBe("system");
    expect(html().getAttribute("data-accent")).toBe("amber");
    expect(html().getAttribute("data-density")).toBe("compact");

    sys.set(true);
    expect(html().getAttribute("data-theme")).toBe("dark");

    unbind();
    expect(sys.listeners.size).toBe(0);
    usePrefs.getState().setTheme("light");
    expect(html().getAttribute("data-theme")).toBe("dark");   // no longer bound
  });
});
