/* Display (colour) settings store — contract C7.
 *
 * The global Display panel edits this store; every image viewer reads it
 * (WP-V) unless it has a per-viewer override (`linked: false` in the viewer,
 * merged with `mergeDisplay`). Persisted to localStorage ("ep-display"),
 * sanitised on load so a stale/hand-edited value can never break a viewer.
 *
 * Locked default: absolute asinh ("asinh-abs") with knee 100 e⁻, gain 1,
 * black 0 (spec 2026-06-24 unified cutout viewer; parity with
 * visualization/color.py). New stretches/colormaps are opt-in.
 */
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { safeJSONStorage } from "./storage";

export type ColorMode = "VIS" | "Y_E" | "J_E" | "H_E" | "lupton" | "temp" | "rgb" | "native";
export type Stretch = "asinh-abs" | "linear" | "log" | "sqrt" | "asinh-auto" | "zscale";
export type Colormap = "gray" | "viridis" | "magma" | "inferno" | "cividis" | "rdbu";
export type TransferGroup = { knee: number; gain: number; black: number };
export type WheelMode = "zoom-when-focused" | "always-zoom" | "scroll";

export type DisplaySettings = {
  color: ColorMode;
  /** Band names for R, G, B in "rgb" mode. */
  rgb: [string, string, string];
  /** Default "asinh-abs" (locked default). */
  stretch: Stretch;
  /** "default" | "euclid" | "jwst" (+ any viewer-specific group); knee in e⁻. */
  groups: Record<string, TransferGroup>;
  colormap: Colormap;
  residualColormap: Colormap;
  invert: boolean;
  /** CSS colour painted for NaN pixels. */
  nanColor: string;
  /** true: every viewer follows these global settings. */
  linked: boolean;
  wheel: WheelMode;
};

export type DisplayActions = {
  set: (patch: Partial<DisplaySettings>) => void;
  setGroup: (name: string, patch: Partial<TransferGroup>) => void;
  reset: () => void;
};

export type DisplayStore = DisplaySettings & DisplayActions;

export const COLOR_MODES: readonly ColorMode[] = ["VIS", "Y_E", "J_E", "H_E", "lupton", "temp", "rgb", "native"];
export const STRETCHES: readonly Stretch[] = ["asinh-abs", "linear", "log", "sqrt", "asinh-auto", "zscale"];
export const COLORMAPS: readonly Colormap[] = ["gray", "viridis", "magma", "inferno", "cividis", "rdbu"];
export const WHEEL_MODES: readonly WheelMode[] = ["zoom-when-focused", "always-zoom", "scroll"];
export const TRANSFER_GROUPS = ["default", "euclid", "jwst"] as const;

/** Slider ranges of the old viewer (log sliders). */
export const KNEE_RANGE: [number, number] = [5, 5000];
export const GAIN_RANGE: [number, number] = [0.1, 10];

export const DEFAULT_TRANSFER: Readonly<TransferGroup> = Object.freeze({ knee: 100, gain: 1, black: 0 });

const DEFAULTS: DisplaySettings = {
  color: "VIS",
  rgb: Object.freeze(["H_E", "J_E", "VIS"]) as unknown as [string, string, string],
  stretch: "asinh-abs",
  groups: Object.freeze({
    default: DEFAULT_TRANSFER,
    euclid: DEFAULT_TRANSFER,
    jwst: DEFAULT_TRANSFER,
  }) as Record<string, TransferGroup>,
  colormap: "gray",
  residualColormap: "rdbu",
  invert: false,
  nanColor: "#ff00ff",
  linked: true,
  wheel: "zoom-when-focused",
};

/** The C7 defaults, DEEPLY frozen (groups, each group and the rgb tuple too),
 *  so a stray write cannot change them; `reset()` / `sanitizeDisplay()` hand
 *  out fresh, editable copies. */
export const DEFAULT_DISPLAY: Readonly<DisplaySettings> = Object.freeze(DEFAULTS);

export const DISPLAY_STORAGE_KEY = "ep-display";

const oneOf = <T extends string>(v: unknown, options: readonly T[], fallback: T): T =>
  (typeof v === "string" && (options as readonly string[]).includes(v) ? v as T : fallback);

const positive = (v: unknown, fallback: number) =>
  (typeof v === "number" && Number.isFinite(v) && v > 0 ? v : fallback);
const finite = (v: unknown, fallback: number) =>
  (typeof v === "number" && Number.isFinite(v) ? v : fallback);

function sanitizeGroup(raw: unknown, base: TransferGroup = DEFAULT_TRANSFER): TransferGroup {
  const g = (raw && typeof raw === "object" ? raw : {}) as Partial<TransferGroup>;
  return { knee: positive(g.knee, base.knee), gain: positive(g.gain, base.gain), black: finite(g.black, base.black) };
}

function freshGroups(): Record<string, TransferGroup> {
  return Object.fromEntries(TRANSFER_GROUPS.map((n) => [n, { ...DEFAULT_TRANSFER }]));
}

/** Coerce anything (persisted JSON, a URL payload) into valid settings.
 *  An invalid field falls back to `fallback` (the C7 defaults by default). */
export function sanitizeDisplay(raw: unknown, fallback: DisplaySettings = DEFAULT_DISPLAY): DisplaySettings {
  const r = (raw && typeof raw === "object" ? raw : {}) as Partial<Record<keyof DisplaySettings, unknown>>;
  const f = fallback === DEFAULT_DISPLAY ? DEFAULT_DISPLAY : sanitizeDisplay(fallback);
  const groups = freshGroups();
  for (const [name, g] of Object.entries(f.groups)) groups[name] = { ...g };
  if (r.groups && typeof r.groups === "object") {
    for (const [name, g] of Object.entries(r.groups as Record<string, unknown>)) {
      groups[name] = sanitizeGroup(g, groups[name] ?? DEFAULT_TRANSFER);
    }
  }
  const rgb = Array.isArray(r.rgb) && r.rgb.length === 3 && r.rgb.every((b) => typeof b === "string" && b)
    ? [...r.rgb] as [string, string, string] : [...f.rgb] as [string, string, string];
  return {
    color: oneOf(r.color, COLOR_MODES, f.color),
    rgb,
    stretch: oneOf(r.stretch, STRETCHES, f.stretch),
    groups,
    colormap: oneOf(r.colormap, COLORMAPS, f.colormap),
    residualColormap: oneOf(r.residualColormap, COLORMAPS, f.residualColormap),
    invert: typeof r.invert === "boolean" ? r.invert : f.invert,
    nanColor: typeof r.nanColor === "string" && r.nanColor.trim() ? r.nanColor : f.nanColor,
    linked: typeof r.linked === "boolean" ? r.linked : f.linked,
    wheel: oneOf(r.wheel, WHEEL_MODES, f.wheel),
  };
}

/** Settings with a (partial) override applied; groups merge per group and an
 *  invalid override field keeps the base value. */
export function mergeDisplay(base: DisplaySettings, patch: Partial<DisplaySettings> | null | undefined): DisplaySettings {
  const b = sanitizeDisplay(base);
  if (!patch) return b;
  return sanitizeDisplay({
    ...b,
    ...patch,
    groups: { ...b.groups, ...(patch.groups ?? {}) },
  }, b);
}

/** The transfer (knee/gain/black) of a group, falling back to "default". */
export function transferFor(settings: Pick<DisplaySettings, "groups">, group = "default"): TransferGroup {
  return settings.groups[group] ?? settings.groups.default ?? { ...DEFAULT_TRANSFER };
}

const settingsOf = (s: DisplayStore): DisplaySettings => ({
  color: s.color, rgb: s.rgb, stretch: s.stretch, groups: s.groups, colormap: s.colormap,
  residualColormap: s.residualColormap, invert: s.invert, nanColor: s.nanColor, linked: s.linked,
  wheel: s.wheel,
});

export const useDisplay = create<DisplayStore>()(
  persist(
    (set, get) => ({
      ...sanitizeDisplay(DEFAULT_DISPLAY),
      set: (patch) => set(mergeDisplay(settingsOf(get()), patch)),
      setGroup: (name, patch) => {
        const s = settingsOf(get());
        const current = s.groups[name] ?? s.groups.default ?? DEFAULT_TRANSFER;
        set({ groups: { ...s.groups, [name]: sanitizeGroup({ ...current, ...patch }, current) } });
      },
      reset: () => set(sanitizeDisplay(DEFAULT_DISPLAY)),
    }),
    {
      name: DISPLAY_STORAGE_KEY,
      version: 1,
      storage: safeJSONStorage,
      partialize: (s) => settingsOf(s),
      merge: (persisted, current) => (persisted == null
        ? current
        : { ...current, ...sanitizeDisplay(persisted, settingsOf(current)) }),
    },
  ),
);
