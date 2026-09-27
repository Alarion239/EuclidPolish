/* Atlas state that is NOT in the URL.
 *
 * `useAtlas` — live, session-only: the engine status, the current view (as
 * the engine reports it), the cursor + pixel readout, the hovered feature,
 * the region-selection mode, narrow-layout panel state. (The pixel overlays
 * and blink are URL state: pixelOverlays.ts, `img` / `blink`.)
 * `useSkyDisplay` — the Display panel's "Sky" section (per-background colour
 * settings, grid, marker scale, overlay stretch); persisted per viewer
 * ("ep-sky-display", try/catch storage), sanitised on load. */
import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { BaseColor } from "../../../sky/engine";
import { ALADIN_COLORMAPS, ALADIN_STRETCHES } from "../../../sky/surveys";
import { safeJSONStorage } from "../../../state/storage";
import type { SkyFeature } from "./layerModel";

export type EngineStatus = "idle" | "loading" | "ready" | "error";

export type AtlasState = {
  status: EngineStatus;
  error: string | null;
  errorCode: string | null;
  view: { ra: number; dec: number; fov: number } | null;
  size: { width: number; height: number };
  projection: string | null;
  cursor: { ra: number; dec: number; x: number; y: number } | null;
  pixel: unknown;
  hover: { feature: SkyFeature; x: number; y: number } | null;
  selecting: null | "rect" | "circle" | "poly";
  panelOpen: boolean;
  set: (patch: Partial<Omit<AtlasState, "set">>) => void;
};

export const useAtlas = create<AtlasState>()((set) => ({
  status: "idle",
  error: null,
  errorCode: null,
  view: null,
  size: { width: 0, height: 0 },
  projection: null,
  cursor: null,
  pixel: null,
  hover: null,
  selecting: null,
  panelOpen: false,
  set: (patch) => set(patch),
}));

/* ── the Display panel's Sky section ─────────────────────────────────── */

/** `follow`: take the colour from the Display panel (overlayColor.ts); off →
 *  the own colormap / stretch / cuts below. */
export type OverlayStretch = { follow: boolean; colormap: string; stretch: string; minCut: number | null; maxCut: number | null };

export type SkyDisplay = {
  byBase: Record<string, BaseColor>;
  grid: boolean;
  gridOpacity: number;
  /** Marker size multiplier (catalogue points, centroid markers). */
  markerScale: number;
  /** Colour of the FITS pixel overlays (LR / SR / JWST). */
  overlay: OverlayStretch;
};

export type SkyDisplayStore = SkyDisplay & {
  setBase: (base: string, patch: Partial<BaseColor>) => void;
  resetBase: (base: string) => void;
  set: (patch: Partial<SkyDisplay>) => void;
  reset: () => void;
};

export const DEFAULT_OVERLAY_STRETCH: OverlayStretch = { follow: true, colormap: "grayscale", stretch: "asinh", minCut: null, maxCut: null };

export const DEFAULT_SKY_DISPLAY: SkyDisplay = {
  byBase: {}, grid: false, gridOpacity: 0.6, markerScale: 1, overlay: { ...DEFAULT_OVERLAY_STRETCH },
};

const finite = (v: unknown): number | undefined => (typeof v === "number" && Number.isFinite(v) ? v : undefined);
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

export function sanitizeBaseColor(c: unknown): BaseColor {
  if (!c || typeof c !== "object") return {};
  const o = c as Record<string, unknown>;
  const out: BaseColor = {};
  if (typeof o.colormap === "string" && (ALADIN_COLORMAPS as readonly string[]).includes(o.colormap)) out.colormap = o.colormap;
  if (typeof o.stretch === "string" && (ALADIN_STRETCHES as readonly string[]).includes(o.stretch)) out.stretch = o.stretch;
  if (typeof o.reversed === "boolean") out.reversed = o.reversed;
  const lo = finite(o.minCut), hi = finite(o.maxCut);
  if (lo != null && hi != null && lo < hi) { out.minCut = lo; out.maxCut = hi; }
  const g = finite(o.gamma); if (g != null) out.gamma = clamp(g, 0.1, 10);
  const s = finite(o.saturation); if (s != null) out.saturation = clamp(s, -1, 1);
  const b = finite(o.brightness); if (b != null) out.brightness = clamp(b, -1, 1);
  const k = finite(o.contrast); if (k != null) out.contrast = clamp(k, -1, 1);
  return out;
}

export function sanitizeSkyDisplay(v: unknown): SkyDisplay {
  const o = (v && typeof v === "object" ? v : {}) as Record<string, unknown>;
  const byBase: Record<string, BaseColor> = {};
  if (o.byBase && typeof o.byBase === "object") {
    for (const [k, c] of Object.entries(o.byBase as Record<string, unknown>)) byBase[k] = sanitizeBaseColor(c);
  }
  const ov = (o.overlay && typeof o.overlay === "object" ? o.overlay : {}) as Record<string, unknown>;
  const lo = finite(ov.minCut) ?? null, hi = finite(ov.maxCut) ?? null;
  return {
    byBase,
    grid: typeof o.grid === "boolean" ? o.grid : DEFAULT_SKY_DISPLAY.grid,
    gridOpacity: clamp(finite(o.gridOpacity) ?? DEFAULT_SKY_DISPLAY.gridOpacity, 0.1, 1),
    markerScale: clamp(finite(o.markerScale) ?? 1, 0.5, 3),
    overlay: {
      follow: typeof ov.follow === "boolean" ? ov.follow : DEFAULT_OVERLAY_STRETCH.follow,
      colormap: typeof ov.colormap === "string" && (ALADIN_COLORMAPS as readonly string[]).includes(ov.colormap) ? ov.colormap : DEFAULT_OVERLAY_STRETCH.colormap,
      stretch: typeof ov.stretch === "string" && (ALADIN_STRETCHES as readonly string[]).includes(ov.stretch) ? ov.stretch : DEFAULT_OVERLAY_STRETCH.stretch,
      minCut: lo != null && hi != null && lo < hi ? lo : null,
      maxCut: lo != null && hi != null && lo < hi ? hi : null,
    },
  };
}

export const useSkyDisplay = create<SkyDisplayStore>()(
  persist(
    (set, get) => ({
      ...sanitizeSkyDisplay(DEFAULT_SKY_DISPLAY),
      setBase: (base, patch) => {
        const cur = get().byBase[base] ?? {};
        set({ byBase: { ...get().byBase, [base]: sanitizeBaseColor({ ...cur, ...patch }) } });
      },
      resetBase: (base) => {
        const next = { ...get().byBase };
        delete next[base];
        set({ byBase: next });
      },
      set: (patch) => set(sanitizeSkyDisplay({ ...pick(get()), ...patch })),
      reset: () => set(sanitizeSkyDisplay(DEFAULT_SKY_DISPLAY)),
    }),
    {
      name: "ep-sky-display",
      version: 1,
      storage: safeJSONStorage,
      partialize: (s) => pick(s),
      merge: (persisted, current) => ({ ...current, ...sanitizeSkyDisplay(persisted) }),
    },
  ),
);

function pick(s: SkyDisplay): SkyDisplay {
  return { byBase: s.byBase, grid: s.grid, gridOpacity: s.gridOpacity, markerScale: s.markerScale, overlay: s.overlay };
}
