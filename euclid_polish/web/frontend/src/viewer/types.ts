/* Public and shared types of the image viewer (src/viewer/README.md). */
import type { DisplaySettings } from "../state/display";
import type { ColorMeta } from "./color";
import type { CubeRec, Params } from "./cube";
import type { Residual, ResidualOp } from "./residual";
import type { ReceptiveField, Selection, WireSelection } from "./selection";
import type { Sky } from "./wcs";

export type TierMeta = { key: string; label: string; unit?: string; hidden?: boolean; disabled?: boolean };

export type ViewerObject = {
  id?: string;
  label?: string;
  ra?: number;
  dec?: number;
  tiers?: string[];
  jwst_bands?: string[];
  [k: string]: unknown;
};

/** GET /viewer/meta/<collection> (contract C6). */
export type ViewerMeta = {
  collection?: string;
  count: number;
  tiers: TierMeta[];
  default_tier?: string;
  band_names: string[];
  color: ColorMeta;
  objects?: ViewerObject[];
  receptive_fields?: ReceptiveField[];
  render_mode?: string;
  empty_label?: string;
  color_label?: string;
  missing_tier_labels?: Record<string, string>;
  transfer_groups?: string[];
  jwst_band_options?: { value: string; label: string }[];
  bhr_fwhm_control?: { param?: string; default_arcsec: number; min_arcsec: number; max_arcsec: number; step_arcsec: number };
  morph_base_tier?: string;
  pca_n?: number;
  pca_max?: number;
  pca_amps?: number[][];
  member_labels?: string[];
  index?: number;
  [k: string]: unknown;
};

export type Layout = "one-row" | "two-rows";
export type Tool = "pan" | "lens";
export type Compare = "off" | "blink" | "swipe";

export type FrameStatus =
  | { kind: "loading" }
  | { kind: "ready" }
  | { kind: "missing"; message: string }
  | { kind: "error"; message: string; hint?: string };

/** What a frame shows: a served cube, a client residual, or the movie. */
export type Shown =
  | { kind: "cube"; rec: CubeRec }
  | { kind: "residual"; rec: Residual & { key: string; pixscale: number; wcs: CubeRec["wcs"]; displayScale: 1; transferGroup: string }; op: ResidualOp; a: string; b: string };

export type ReadoutTier = {
  tier: string;
  label: string;
  /** Integer pixel on this tier's grid (null outside). */
  x: number | null;
  y: number | null;
  /** Continuous position on this tier's grid (crosshair). */
  fx: number;
  fy: number;
  bands: string[];
  values: number[] | null;
  unit: string;
};

export type Readout = {
  /** The tier under the pointer. */
  tier: string;
  sky: Sky | null;
  tiers: ReadoutTier[];
};

export type ProfileGeom =
  | { kind: "line"; tier: string; p0: { x: number; y: number }; p1: { x: number; y: number } }
  | { kind: "radial"; tier: string; c: { x: number; y: number } };

export type LensPlacement = { left: number; top: number; corner: string | null };

export type SaveStatus = { text: string; tone: "" | "busy" | "saved" | "error"; kind?: "save-blocker" };

/** The state the old engine exposed through getState()/onChange (unchanged
 *  keys), plus the new view state. */
export type ViewerState = {
  index: number;
  id: string | null;
  tier: string;
  tiers: string[];
  color: string;
  layout: Layout;
  knee: number;
  gain: number;
  transfers: Record<string, { knee: number; gain: number }>;
  params: Record<string, string>;
  selection: WireSelection | null;
  view: Selection | null;
  tool: Tool;
  compare: Compare;
};

export type ViewerApi = {
  goTo(i: number): void;
  /** Go to the object whose meta id is `id` (falls back to the server's ?id= lookup). */
  goToId(id: string): Promise<boolean>;
  setTiers(keys: string | string[]): void;
  /** Per-viewer colour / knee / gain (wins over the Display panel). */
  setView(patch: { color?: string; knee?: number; gain?: number }): void;
  /** Patch cube-query parameters and refresh the visible pixels in place. */
  setParams(patch: Record<string, string>): Promise<void>;
  setMorphMembers(csv: string | null): void;
  getIndex(): number;
  isReady(): boolean;
  getState(): ViewerState;
  exportFigure(): void;
  savePng(): void;
  saveCropToResults(): Promise<void>;
  reload(): Promise<void>;
  /** Centre the view on (ra, dec) with a field of view of `fovArcsec`. */
  zoomTo(ra: number, dec: number, fovArcsec?: number): boolean;
  resetView(): void;
  getReadout(): Readout | null;
  destroy(): void;
};

export type ToolbarMode = "full" | "compact" | "none";

export type ImageViewerProps = {
  collection: string;
  params?: Record<string, string>;
  /** Initial tiers (default: meta.default_tier). */
  tiers?: string[];
  initialIndex?: number;
  /** Initial object by meta id (wins over initialIndex). */
  initialId?: string;
  /** Instance id (keyboard arbitration, ARIA); default: the collection. */
  id?: string;
  /** URL-state prefix `v.<urlKey>.` (index/id, tiers, view, colour). Off when absent. */
  urlKey?: string;
  onState?: (s: ViewerState) => void;
  onReady?: (api: ViewerApi | null) => void;
  toolbar?: ToolbarMode;
  /** Show the navigation row (default true). */
  nav?: boolean;
  className?: string;
  /** Per-viewer display override (wins over the Display panel). */
  display?: Partial<DisplaySettings>;
};

export type { Params };
