import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  COLOR_MODES,
  COLORMAPS,
  DEFAULT_DISPLAY,
  DEFAULT_TRANSFER,
  DISPLAY_STORAGE_KEY,
  STRETCHES,
  mergeDisplay,
  sanitizeDisplay,
  transferFor,
  useDisplay,
} from "./display";

beforeEach(() => { localStorage.clear(); useDisplay.getState().reset(); });
afterEach(() => { vi.restoreAllMocks(); });

describe("display store (C7)", () => {
  it("starts from the locked defaults", () => {
    const s = useDisplay.getState();
    expect(s.color).toBe("VIS");
    expect(s.rgb).toEqual(["H_E", "J_E", "VIS"]);
    expect(s.stretch).toBe("asinh-abs");
    expect(s.colormap).toBe("gray");
    expect(s.residualColormap).toBe("rdbu");
    expect(s.invert).toBe(false);
    expect(s.nanColor).toBe("#5b6475");
    expect(s.linked).toBe(true);
    expect(s.wheel).toBe("zoom-when-focused");
    expect(Object.keys(s.groups).sort()).toEqual(["default", "euclid", "jwst"]);
    for (const g of Object.values(s.groups)) expect(g).toEqual({ knee: 100, gain: 1, black: 0 });
    expect(DEFAULT_TRANSFER).toEqual({ knee: 100, gain: 1, black: 0 });
  });

  it("lists every option of the contract", () => {
    expect(COLOR_MODES).toEqual(["VIS", "Y_E", "J_E", "H_E", "lupton", "temp", "rgb", "native"]);
    expect(STRETCHES).toEqual(["asinh-abs", "linear", "log", "sqrt", "asinh-auto", "zscale"]);
    expect(COLORMAPS).toEqual(["gray", "viridis", "magma", "inferno", "cividis", "rdbu"]);
  });

  it("set(patch) merges top-level settings", () => {
    useDisplay.getState().set({ color: "lupton", invert: true });
    const s = useDisplay.getState();
    expect(s.color).toBe("lupton");
    expect(s.invert).toBe(true);
    expect(s.stretch).toBe("asinh-abs");
  });

  it("setGroup(name, patch) edits one transfer group (creating it if needed)", () => {
    useDisplay.getState().setGroup("euclid", { knee: 250 });
    useDisplay.getState().setGroup("hst", { gain: 2 });
    const { groups } = useDisplay.getState();
    expect(groups.euclid).toEqual({ knee: 250, gain: 1, black: 0 });
    expect(groups.jwst).toEqual({ knee: 100, gain: 1, black: 0 });
    expect(groups.hst).toEqual({ knee: 100, gain: 2, black: 0 });
  });

  it("rejects invalid values instead of storing them", () => {
    useDisplay.getState().setGroup("default", { knee: -5, gain: Number.NaN });
    useDisplay.getState().set({ color: "bogus" as never, stretch: "nope" as never });
    const s = useDisplay.getState();
    expect(s.groups.default).toEqual({ knee: 100, gain: 1, black: 0 });
    expect(s.color).toBe("VIS");
    expect(s.stretch).toBe("asinh-abs");
  });

  it("an invalid patch value keeps the current (not the default) value", () => {
    useDisplay.getState().set({ color: "lupton", colormap: "magma", nanColor: "#000" });
    useDisplay.getState().set({ color: "bogus" as never, colormap: 7 as never, nanColor: "  ", invert: true });
    const s = useDisplay.getState();
    expect(s.color).toBe("lupton");
    expect(s.colormap).toBe("magma");
    expect(s.nanColor).toBe("#000");
    expect(s.invert).toBe(true);
  });

  it("reset() restores the defaults", () => {
    useDisplay.getState().set({ colormap: "magma", linked: false });
    useDisplay.getState().setGroup("jwst", { knee: 3 });
    useDisplay.getState().reset();
    expect(sanitizeDisplay(useDisplay.getState())).toEqual(DEFAULT_DISPLAY);
  });

  it("persists to localStorage and rehydrates sanitised", async () => {
    useDisplay.getState().set({ colormap: "viridis" });
    const saved = JSON.parse(localStorage.getItem(DISPLAY_STORAGE_KEY)!);
    expect(saved.state.colormap).toBe("viridis");
    expect(saved.state.set).toBeUndefined();              // actions are not persisted

    localStorage.setItem(DISPLAY_STORAGE_KEY, JSON.stringify({
      state: { color: "temp", stretch: "wat", groups: { euclid: { knee: 42 } }, rgb: ["VIS"] },
      version: 1,
    }));
    await useDisplay.persist.rehydrate();
    const s = useDisplay.getState();
    expect(s.color).toBe("temp");
    expect(s.stretch).toBe("asinh-abs");
    expect(s.groups.euclid).toEqual({ knee: 42, gain: 1, black: 0 });
    expect(s.groups.default).toEqual(DEFAULT_TRANSFER);
    expect(s.rgb).toEqual(["H_E", "J_E", "VIS"]);
    expect(typeof s.set).toBe("function");
  });

  it("migrates the v1 magenta NaN default to the muted default but keeps a chosen colour", async () => {
    localStorage.setItem(DISPLAY_STORAGE_KEY, JSON.stringify({ state: { nanColor: "#FF00FF" }, version: 1 }));
    await useDisplay.persist.rehydrate();
    expect(useDisplay.getState().nanColor).toBe(DEFAULT_DISPLAY.nanColor);
    localStorage.setItem(DISPLAY_STORAGE_KEY, JSON.stringify({ state: { nanColor: "#00ff00" }, version: 1 }));
    await useDisplay.persist.rehydrate();
    expect(useDisplay.getState().nanColor).toBe("#00ff00");
  });

  it("the locked defaults are deeply frozen (a stray write cannot change reset())", () => {
    expect(Object.isFrozen(DEFAULT_DISPLAY)).toBe(true);
    expect(Object.isFrozen(DEFAULT_DISPLAY.groups)).toBe(true);
    for (const g of Object.values(DEFAULT_DISPLAY.groups)) expect(Object.isFrozen(g)).toBe(true);
    expect(Object.isFrozen(DEFAULT_DISPLAY.rgb)).toBe(true);
    expect(() => { (DEFAULT_DISPLAY.groups.default as { knee: number }).knee = 5; }).toThrow(TypeError);
    expect(() => { (DEFAULT_DISPLAY.rgb as string[])[0] = "X"; }).toThrow(TypeError);
    useDisplay.getState().reset();
    const s = useDisplay.getState();
    expect(s.groups.default.knee).toBe(100);
    expect(s.rgb[0]).toBe("H_E");
    expect(Object.isFrozen(s.groups.default)).toBe(false);  // the store holds fresh, editable copies
    expect(Object.isFrozen(s.rgb)).toBe(false);
  });

  it("hydrates a saved value on a fresh load and starts from the defaults when nothing is saved", async () => {
    localStorage.clear();
    vi.resetModules();
    const empty = await import("./display");
    expect(empty.sanitizeDisplay(empty.useDisplay.getState())).toEqual(DEFAULT_DISPLAY);
    localStorage.setItem(DISPLAY_STORAGE_KEY, JSON.stringify({
      state: { colormap: "magma", groups: { jwst: { knee: 7 } } }, version: 1,
    }));
    vi.resetModules();
    const saved = await import("./display");
    expect(saved.useDisplay.getState().colormap).toBe("magma");
    expect(saved.useDisplay.getState().groups.jwst).toEqual({ knee: 7, gain: 1, black: 0 });
    expect(saved.useDisplay.getState().stretch).toBe("asinh-abs");
  });

  it("keeps working when localStorage throws", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("QuotaExceededError"); });
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("SecurityError"); });
    expect(() => useDisplay.getState().set({ invert: true })).not.toThrow();
    expect(useDisplay.getState().invert).toBe(true);
  });
});

describe("display helpers", () => {
  it("transferFor falls back to the default group", () => {
    const s = { ...DEFAULT_DISPLAY, groups: { ...DEFAULT_DISPLAY.groups, jwst: { knee: 7, gain: 2, black: 1 } } };
    expect(transferFor(s, "jwst")).toEqual({ knee: 7, gain: 2, black: 1 });
    expect(transferFor(s, "unknown")).toEqual(DEFAULT_TRANSFER);
    expect(transferFor(s)).toEqual(DEFAULT_TRANSFER);
  });

  it("mergeDisplay applies a per-viewer override on top of the global settings", () => {
    const merged = mergeDisplay(DEFAULT_DISPLAY, { color: "temp", groups: { euclid: { knee: 50, gain: 1, black: 0 } } });
    expect(merged.color).toBe("temp");
    expect(merged.groups.euclid.knee).toBe(50);
    expect(merged.groups.jwst).toEqual(DEFAULT_TRANSFER);
    expect(DEFAULT_DISPLAY.color).toBe("VIS");            // inputs untouched
  });
});
