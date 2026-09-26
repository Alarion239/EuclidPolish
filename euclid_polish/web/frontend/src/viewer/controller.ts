/* ViewerController — the engine behind one <ImageViewer>: a per-instance
 * zustand store plus the loading logic ported from the pre-rework
 * static/cutout_viewer.js closure (meta, index/tier loads with revision
 * guards, prefetch, in-place parameter refresh, the BHR convolution refresh,
 * the disagreement movie, lens selection, save-to-results) and the new
 * inspection features (pan/zoom view, pixel readout with WCS, residual
 * tiers, profiles). React components subscribe to slices of `store`; the
 * canvases draw imperatively through registered FrameHandles. */
import { createStore, type StoreApi } from "zustand/vanilla";
import { apiPost, isAbortError } from "../api/client";
import { useShortcutRegistry } from "../hooks/useShortcut";
import { mergeDisplay, useDisplay, type DisplaySettings, type TransferGroup } from "../state/display";
import { prepareCore, type Prepared } from "./color";
import { parseCssColor } from "./colormaps";
import { cubeKey, cubeUrl, fetchMeta, sharedCubeCache, ViewerError, type CubeRec, type Params } from "./cube";
import { exportStem, heatbarStops, publicationUnitLabel, type HeatbarInfo } from "./export";
import {
  MORPH_FRAMES, MOVIE_RADIUS, MovieStore, morphBaseTier, morphCoefficients, movieBytes, movieKey, movieLabel,
  pcaCount, slotAt, synthesizeMorphFrame,
} from "./movie";
import { magInfo, magLabel, pixelValues, sigmaMagnitude } from "./readout";
import { finiteSorted, percentile, robustStats } from "./stats";
import { frameAutoStats, prepareColor, renderPrepared, renderSigned, type DisplayParams } from "./render";
import { computeResidual, parseResidualKey, residualKey, residualLabel, residualMismatch, type ResidualOp } from "./residual";
import {
  LENS_ZOOM_STEP, clampSelectionToFrames, contentBoxOrigin, currentViewport, frameLayout, imageToFrame, lensSide, placeLensPopup,
  rectOf, resolveCrop, resolveLensOverlaps, selectionAt, serializeSelection, zoomSelection,
  type Crop, type FrameGeom, type FrameLayout, type LensPosition, type Selection,
} from "./selection";
import type {
  Compare, FrameStatus, ImageViewerProps, Layout, LensPlacement, ProfileGeom, Readout, ReadoutTier, SaveStatus,
  Shown, Tool, ViewerApi, ViewerMeta, ViewerState,
} from "./types";
import { pixToSky, skyToPix } from "./wcs";

export const PLAY_INTERVAL_MS = 1500;
export const RESULT_MAX_TIERS = 4;
export const RESULT_SAVEABLE_TIERS = new Set(["dirty", "lr", "real", "original", "original_stack", "sr", "hr", "jwst"]);
export const NATIVE_F200W_SAVE_REASON = "Choose native F200W first.";
export const COLOR_KEYS = ["q", "w", "e", "r", "t", "y"];
export const COLOR_MODES_EXTRA = [
  { key: "lupton", label: "Lupton", title: "4-band solar-balanced Lupton RGB" },
  { key: "temp", label: "Temp", title: "Per-pixel blackbody-T colour (Planckian locus)" },
];
export const VIEW_LAYOUT_STORAGE_KEY = "euclid-polish.cutout-viewer.layout";
/** Maximum view zoom (relative to the full frame). */
export const VIEW_MAX_ZOOM = 64;

function savedViewLayout(): Layout {
  try {
    return window.localStorage.getItem(VIEW_LAYOUT_STORAGE_KEY) === "two-rows" ? "two-rows" : "one-row";
  } catch {
    return "one-row";
  }
}
function saveViewLayout(value: Layout): void {
  try { window.localStorage.setItem(VIEW_LAYOUT_STORAGE_KEY, value); } catch { /* optional */ }
}

/** What a mounted <Frame> exposes to the engine. */
export type FrameHandle = {
  tier: string;
  /** Natural-resolution rendered image (h × w). */
  source: HTMLCanvasElement;
  /** The on-screen canvas. */
  visible: HTMLCanvasElement;
  element: HTMLElement;
  /** Frame CSS side (px). */
  size(): number;
  /** Re-blit source → visible (after a movie tick). */
  redraw(): void;
};

export type ViewerStoreState = {
  meta: ViewerMeta | null;
  metaError: string | null;
  params: Params;
  index: number;
  tiers: string[];
  residuals: string[];
  layout: Layout;
  override: Partial<DisplaySettings>;
  unlinked: boolean;
  view: Selection | null;
  tool: Tool;
  altLens: boolean;
  hover: Selection | null;
  frozen: Selection | null;
  lens: Record<string, LensPlacement>;
  shown: Record<string, Shown>;
  status: Record<string, FrameStatus>;
  overlay: Record<string, string>;
  readout: Readout | null;
  compare: Compare;
  blinkMs: number;
  blinkAt: number;
  swipe: number;
  histogram: boolean;
  profileOpen: boolean;
  profile: ProfileGeom | null;
  playing: boolean;
  playMs: number;
  morphAmp: number;
  morphSpeed: number;
  morphMembers: string | null;
  movieProgress: Record<string, number | null>;
  save: SaveStatus;
  saveInFlight: boolean;
  recording: boolean;
  hot: boolean;
  /** Bumped when a frame's pixels change (lens popups redraw). */
  drawn: number;
};

type MovieEntry = {
  frames: (Prepared | null)[];
  w: number; h: number; mode: string; bytes: number; color: string; amp: number; done: boolean;
  label: string; pixscale: number; rec: CubeRec;
};

/** The one viewer that answers document-level keys (most recently hovered,
 *  focused or frozen) — several viewers can be mounted on one page. */
const keyboard: { active: ViewerController | null } = { active: null };

const errMessage = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** The shell's own Shift+letters (FOUNDATION §10.3: pages must not rebind them). */
const SHELL_SHIFT_KEYS = new Set(["d", "j", "t"]);

/** Whether Shift+`letter` belongs to the shell or to a shortcut a page has
 *  bound (the useShortcut registry, e.g. an ensemble page's Shift+E). */
export function shiftComboClaimed(letter: string): boolean {
  const l = letter.toLowerCase();
  if (SHELL_SHIFT_KEYS.has(l)) return true;
  return useShortcutRegistry.getState().entries.some((en) => {
    const press = en.combo.trim().toLowerCase().replace(/[[\]]/g, "");
    return press === `shift+${l}`;
  });
}

export class ViewerController {
  readonly collection: string;
  readonly store: StoreApi<ViewerStoreState>;
  readonly id: string;
  private cache = sharedCubeCache;
  private prepCache = new Map<string, Prepared>();
  private residualScale = new Map<string, { knee: number; range: number }>();
  readonly frames = new Map<string, FrameHandle>();
  private showRevision = 0;
  private paramRefreshToken = 0;
  private tierRefreshRevisions = new Map<string, number>();
  private bhrRefreshTimer: ReturnType<typeof setTimeout> | null = null;
  private playTimer: ReturnType<typeof setInterval> | null = null;
  private blinkTimer: ReturnType<typeof setInterval> | null = null;
  private morphRaf: number | null = null;
  private buildToken = 0;
  private movies = new MovieStore<MovieEntry>();
  private saveController: AbortController | null = null;
  private life = new AbortController();
  private destroyed = false;
  private selectionRevision = 0;
  private initialId: string | null;
  private onState: ((s: ViewerState) => void) | undefined;
  private recorder: { stop: () => void } | null = null;
  readonly api: ViewerApi;

  constructor(props: Pick<ImageViewerProps, "collection" | "params" | "tiers" | "initialIndex" | "initialId" | "id" | "display">) {
    this.collection = props.collection;
    this.id = props.id ?? props.collection;
    this.initialId = props.initialId ?? null;
    this.store = createStore<ViewerStoreState>()(() => ({
      meta: null, metaError: null,
      params: { ...(props.params ?? {}) },
      index: props.initialIndex ?? 0,
      tiers: Array.isArray(props.tiers) ? props.tiers.map(String) : [],
      residuals: [],
      layout: savedViewLayout(),
      override: { ...(props.display ?? {}) },
      unlinked: false,
      view: null, tool: "pan", altLens: false, hover: null, frozen: null, lens: {},
      shown: {}, status: {}, overlay: {}, readout: null,
      compare: "off", blinkMs: 700, blinkAt: 0, swipe: 0.5,
      histogram: false, profileOpen: false, profile: null,
      playing: false, playMs: PLAY_INTERVAL_MS,
      morphAmp: 1.6, morphSpeed: 0.5, morphMembers: null, movieProgress: {},
      save: { text: "", tone: "" }, saveInFlight: false, recording: false, hot: false, drawn: 0,
    }));
    this.api = this.buildApi();
    document.addEventListener("keydown", this.onKey);
  }

  // ---- small accessors ----------------------------------------------------
  get s(): ViewerStoreState { return this.store.getState(); }
  private set(patch: Partial<ViewerStoreState>) { if (!this.destroyed) this.store.setState(patch); }

  setOnState(fn: ((s: ViewerState) => void) | undefined) { this.onState = fn; }

  /** Effective display settings: the Display panel merged with this viewer's override. */
  settings(): DisplaySettings {
    return mergeDisplay(useDisplay.getState(), this.s.override);
  }
  linked(): boolean { return useDisplay.getState().linked && !this.s.unlinked; }
  K0(): number { return (this.s.meta && this.s.meta.color && this.s.meta.color.default_asinh) || 100; }

  tierMeta(key: string) { return (this.s.meta?.tiers ?? []).find((t) => t.key === key); }
  tierLabel(key: string): string {
    const r = parseResidualKey(key);
    if (r) return residualLabel(r.op, this.tierLabel(r.a), this.tierLabel(r.b));
    return this.tierMeta(key)?.label ?? key;
  }
  private currentObject() { return this.s.meta?.objects?.[this.s.index]; }
  tierAvail(key: string): boolean {
    const r = parseResidualKey(key);
    if (r) return this.tierAvail(r.a) && this.tierAvail(r.b);
    const obj = this.currentObject();
    return !obj || !obj.tiers || obj.tiers.includes(key);
  }
  tierDisabled(key: string): boolean {
    const tm = this.tierMeta(key);
    return !!(tm && tm.disabled) || !this.tierAvail(key);
  }
  missingTierLabel(key: string): string {
    return this.s.meta?.missing_tier_labels?.[key] || `no ${key}`;
  }
  jwstBandAvailable(value: string): boolean {
    if (value === "colour") return true;
    const obj = this.currentObject();
    return !obj || !obj.jwst_bands || obj.jwst_bands.includes(value);
  }
  /** Selected tiers in the meta's canonical order, then the residual tiers. */
  frameKeys(): string[] {
    const s = this.s;
    const real = (s.meta?.tiers ?? []).map((t) => t.key).filter((k) => s.tiers.includes(k));
    return [...real, ...s.residuals];
  }
  /** Transfer groups with their own sliders: meta.transfer_groups ∩ {euclid, jwst}, else default. */
  activeGroups(): string[] {
    const g = (this.s.meta?.transfer_groups ?? []).filter((x) => x === "euclid" || x === "jwst");
    return g.length ? Array.from(new Set(g)) : ["default"];
  }
  groupOf(rec: { transferGroup?: string } | null | undefined): string {
    const groups = this.activeGroups();
    if (groups.length === 1 && groups[0] === "default") return "default";
    const g = rec?.transferGroup ?? "default";
    return groups.includes(g) ? g : groups[0];
  }
  transfer(group: string, settings = this.settings()): TransferGroup {
    return settings.groups[group] ?? settings.groups.default ?? { knee: 100, gain: 1, black: 0 };
  }
  displayParams(rec: { transferGroup?: string } | null, settings = this.settings()): DisplayParams {
    const t = this.transfer(this.groupOf(rec), settings);
    return {
      stretch: settings.stretch, knee: t.knee, gain: t.gain, black: t.black, K0: this.K0(),
      colormap: settings.colormap, invert: settings.invert, nanColor: parseCssColor(settings.nanColor),
    };
  }

  // ---- display edits (toolbar / keyboard / histogram) -----------------------
  /** A colour / transfer edit goes where the effective value comes from: this
   *  viewer's override when it has one (or is unlinked), else the Display panel. */
  setColor(color: string) {
    const o = this.s.override;
    if ("color" in o || !this.linked()) this.set({ override: { ...o, color: color as DisplaySettings["color"] } });
    else useDisplay.getState().set({ color: color as DisplaySettings["color"] });
    this.afterDisplayChange();
  }
  setTransfer(group: string, patch: Partial<TransferGroup>) {
    const o = this.s.override;
    if ((o.groups && group in o.groups) || !this.linked()) {
      const cur = this.transfer(group);
      this.set({ override: { ...o, groups: { ...(o.groups ?? {}), [group]: { ...cur, ...patch } } } });
    } else {
      useDisplay.getState().setGroup(group, patch);
    }
    this.afterDisplayChange();
  }
  setOverride(patch: Partial<DisplaySettings>) {
    this.set({ override: { ...this.s.override, ...patch } });
    this.afterDisplayChange();
  }
  /** Stop following the Display panel (a copy of it becomes this viewer's own). */
  setUnlinked(unlinked: boolean) {
    if (unlinked) {
      const cur = this.settings();
      this.set({ unlinked: true, override: { ...cur, groups: { ...cur.groups } } });
    } else {
      this.set({ unlinked: false, override: {} });
    }
    this.afterDisplayChange();
  }
  /** Colour changed (store or override): refresh overlays, the movie keeps playing. */
  afterDisplayChange() {
    this.refreshOverlays();
    this.notify();
  }

  // ---- state out --------------------------------------------------------------
  getState(): ViewerState {
    const s = this.s;
    const settings = this.settings();
    const transfers: Record<string, { knee: number; gain: number }> = {};
    for (const g of this.activeGroups()) { const t = this.transfer(g, settings); transfers[g] = { knee: t.knee, gain: t.gain }; }
    const first = transfers[this.activeGroups()[0]] ?? { knee: 100, gain: 1 };
    const obj = s.meta?.objects?.[s.index];
    return {
      index: s.index, id: typeof obj?.id === "string" ? obj.id : null,
      tier: s.tiers[0], tiers: s.tiers.slice(), color: settings.color, layout: s.layout,
      knee: first.knee, gain: first.gain, transfers, params: { ...s.params },
      selection: serializeSelection(s.frozen), view: s.view, tool: s.tool, compare: s.compare,
    };
  }
  notify() { if (!this.destroyed) this.onState?.(this.getState()); }

  // ---- meta + loading ---------------------------------------------------------
  async loadMeta(): Promise<void> {
    const s = this.s;
    let meta: ViewerMeta;
    try {
      meta = await fetchMeta<ViewerMeta>(this.collection, s.params, this.life.signal);
    } catch (e) {
      if (isAbortError(e)) return;
      this.set({ metaError: errMessage(e), meta: null });
      throw e;
    }
    if (this.destroyed) return;
    const params = { ...this.s.params };
    const bhr = meta.bhr_fwhm_control;
    if (bhr) {
      const param = bhr.param || "bhr_fwhm_arcsec";
      if (params[param] == null) params[param] = String(Number(bhr.default_arcsec));
    }
    const jwstOptions = meta.jwst_band_options || [];
    if (jwstOptions.length && !jwstOptions.some((o) => o.value === params.jwst_band)) params.jwst_band = jwstOptions[0].value;
    // Cubes of the previous meta may be stale (a regenerated SR, a new field).
    this.cache.deletePrefix(`${this.collection}|`);
    this.prepCache.clear();
    this.set({ meta, metaError: null, params });
    const tierKeys = (meta.tiers || []).map((t) => t.key);
    let tiers = this.s.tiers.filter((t) => tierKeys.includes(t) && !this.tierDisabled(t));
    if (!tiers.length) {
      const def = meta.default_tier as string;
      const firstEnabled = (meta.tiers || []).find((t) => !this.tierDisabled(t.key));
      tiers = tierKeys.includes(def) && !this.tierDisabled(def) ? [def]
        : firstEnabled ? [firstEnabled.key] : tierKeys.slice(0, 1);
    }
    const residuals = this.s.residuals.filter((k) => {
      const r = parseResidualKey(k);
      return !!r && tierKeys.includes(r.a) && tierKeys.includes(r.b);
    });
    this.set({ tiers, residuals });
    if (this.initialId) {
      const id = this.initialId;
      this.initialId = null;
      await this.resolveId(id);
    }
  }

  /** Index of an object id (meta objects, else the server's ?id= lookup). */
  async resolveId(id: string): Promise<boolean> {
    const objs = this.s.meta?.objects ?? [];
    const i = objs.findIndex((o) => o.id === id);
    if (i >= 0) { this.set({ index: i }); return true; }
    try {
      const m = await fetchMeta<ViewerMeta>(this.collection, { ...this.s.params, id }, this.life.signal);
      if (Number.isInteger(m.index)) { this.set({ index: m.index as number }); return true; }
    } catch { /* unknown id */ }
    return false;
  }

  private cacheKeyFor(tier: string, index: number, extra?: Params) {
    return cubeKey(this.collection, tier, index, this.s.params, extra);
  }

  fetchCube(tier: string, index: number, extra?: Params, signal?: AbortSignal): Promise<CubeRec> {
    const key = this.cacheKeyFor(tier, index, extra);
    return this.cache.load(key, cubeUrl(this.collection, index, tier, this.s.params, extra), signal ?? this.life.signal);
  }

  private setStatus(tier: string, st: FrameStatus) {
    this.set({ status: { ...this.s.status, [tier]: st } });
  }
  private setShown(tier: string, shown: Shown | null) {
    const next = { ...this.s.shown };
    if (shown) next[tier] = shown; else delete next[tier];
    this.set({ shown: next });
  }
  private failureStatus(tier: string, e: unknown): FrameStatus {
    if (!this.tierAvail(tier)) return { kind: "missing", message: this.missingTierLabel(tier) };
    const hint = this.s.meta?.missing_tier_labels?.[tier];
    return { kind: "error", message: errMessage(e), hint };
  }

  /** Load + show the current index across every frame. */
  async show({ preserveFrozen = false } = {}): Promise<void> {
    const revision = ++this.showRevision;
    if (!preserveFrozen) this.clearAllLenses();
    else this.hideHover();
    const meta = this.s.meta;
    if (!meta || meta.count === 0) {
      const msg = meta?.empty_label || "No cutouts available.";
      const status: Record<string, FrameStatus> = {};
      for (const k of this.frameKeys()) status[k] = { kind: "missing", message: msg };
      this.set({ status, shown: {} });
      this.notify();
      return;
    }
    const index = Math.max(0, Math.min(this.s.index, meta.count - 1));
    const params = { ...this.s.params };
    if (!this.jwstBandAvailable(params.jwst_band || "colour") && params.jwst_band) {
      params.jwst_band = "colour";
    }
    this.set({ index, params });
    this.buildToken++;
    this.stopMorph();
    const keys = this.frameKeys();
    this.set({ shown: {}, status: Object.fromEntries(keys.map((k) => [k, { kind: "loading" } as FrameStatus])), readout: null });
    this.updateSaveControls();
    await Promise.all(keys.map(async (tier) => {
      if (!this.tierAvail(tier)) { this.setStatus(tier, { kind: "missing", message: this.missingTierLabel(tier) }); return; }
      if (tier === "morph") { await this.startMorph(index); return; }
      const tierRevision = this.tierRefreshRevisions.get(tier) || 0;
      const stale = () => revision !== this.showRevision || this.s.index !== index || !this.frameKeys().includes(tier)
        || (this.tierRefreshRevisions.get(tier) || 0) !== tierRevision;
      try {
        const shown = await this.loadShown(tier, index);
        if (stale()) return;
        this.setShown(tier, shown);
        this.setStatus(tier, { kind: "ready" });
      } catch (e) {
        if (stale() || isAbortError(e)) return;
        this.setStatus(tier, this.failureStatus(tier, e));
      }
    }));
    if (revision !== this.showRevision || this.s.index !== index) return;
    this.refreshOverlays();
    this.prefetch(index);
    this.notify();
  }

  /** A served cube, or a residual computed from two (+ σ) served cubes. */
  private async loadShown(tier: string, index: number): Promise<Shown> {
    const r = parseResidualKey(tier);
    if (!r) return { kind: "cube", rec: await this.fetchCube(tier, index) };
    const [A0, B0] = await Promise.all([this.fetchCube(r.a, index), this.fetchCube(r.b, index)]);
    // Units: X-Cube-Unit, else the meta tier's (as the readout reads them).
    const A = { ...A0, unit: A0.unit || this.tierMeta(r.a)?.unit || "" };
    const B = { ...B0, unit: B0.unit || this.tierMeta(r.b)?.unit || "" };
    const why = residualMismatch(A, B);
    if (why) throw new ViewerError(0, `${this.tierLabel(r.a)} (${A.w}×${A.h}) and ${this.tierLabel(r.b)} (${B.w}×${B.h}) ${why}`);
    let S: CubeRec | null = null;
    if (r.op === "chi" && (this.s.meta?.tiers ?? []).some((t) => t.key === "std") && r.a !== "std" && r.b !== "std") {
      try { S = await this.fetchCube("std", index); } catch { S = null; }
    }
    const res = computeResidual(r.op, A, B, S);
    if (!res) throw new ViewerError(0, `${this.tierLabel(r.a)} and ${this.tierLabel(r.b)} cannot be combined`);
    const fine = res.h === A.h ? A : B;
    return {
      kind: "residual", op: r.op, a: r.a, b: r.b,
      rec: { ...res, key: `${A.key}~${B.key}~${r.op}`, pixscale: fine.pixscale, wcs: fine.wcs, displayScale: 1, transferGroup: A.transferGroup },
    };
  }

  /** Warm the cubes of the next indices (+1..+3, −1) for every frame. */
  prefetch(index: number) {
    const s = this.s;
    if (s.params.psf_warp === "1" || !s.meta) return;
    const subset = s.morphMembers;
    const extra = subset ? { members: subset } : undefined;
    const nPca = pcaCount(s.meta, subset);
    for (const di of [1, 2, 3, -1]) {
      const j = index + di;
      if (j < 0 || j >= s.meta.count) continue;
      for (const t of this.frameKeys()) {
        if (t === "morph") {
          this.fetchCube(morphBaseTier(s.meta), j, extra).catch(() => {});
          for (let k = 0; k < nPca; k++) this.fetchCube(`pca${k}`, j, extra).catch(() => {});
        } else {
          const r = parseResidualKey(t);
          for (const key of r ? [r.a, r.b] : [t]) if (this.tierAvail(key)) this.fetchCube(key, j).catch(() => {});
        }
      }
    }
  }

  /** Refresh only the visible cubes after a parameter change (frames stay mounted). */
  async refreshVisible(): Promise<void> {
    const meta = this.s.meta;
    if (!meta || meta.count === 0) return;
    const token = ++this.paramRefreshToken;
    const revision = ++this.showRevision;
    const index = this.s.index;
    this.buildToken++;
    this.stopMorph();
    this.cache.deletePrefix(`${this.collection}|`);
    this.prepCache.clear();
    this.updateSaveControls();
    const keys = this.frameKeys();
    await Promise.all(keys.map(async (tier) => {
      if (!this.tierAvail(tier)) return;
      if (tier === "morph") { await this.startMorph(index); return; }
      const stale = () => token !== this.paramRefreshToken || revision !== this.showRevision
        || this.s.index !== index || !this.frameKeys().includes(tier);
      this.setStatus(tier, { kind: "loading" });
      try {
        const shown = await this.loadShown(tier, index);
        if (stale()) return;
        this.setShown(tier, shown);
        this.setStatus(tier, { kind: "ready" });
      } catch (e) {
        if (stale() || isAbortError(e)) return;
        this.setStatus(tier, this.failureStatus(tier, e));
      }
    }));
    if (token === this.paramRefreshToken && revision === this.showRevision && this.s.index === index) {
      this.refreshOverlays();
      this.notify();
    }
  }

  /** Refresh one parameter-derived tier (the BHR convolution) in place. */
  async refreshTier(tier: string): Promise<void> {
    const meta = this.s.meta;
    if (!meta || meta.count === 0 || !this.s.tiers.includes(tier) || !this.tierAvail(tier)) return;
    const token = ++this.paramRefreshToken;
    const revision = this.showRevision;
    const tierRevision = (this.tierRefreshRevisions.get(tier) || 0) + 1;
    this.tierRefreshRevisions.set(tier, tierRevision);
    const index = this.s.index;
    const stale = () => token !== this.paramRefreshToken || revision !== this.showRevision
      || this.s.index !== index || this.tierRefreshRevisions.get(tier) !== tierRevision;
    this.setStatus(tier, { kind: "loading" });
    try {
      const rec = await this.fetchCube(tier, index);
      if (stale()) return;
      this.setShown(tier, { kind: "cube", rec });
      this.setStatus(tier, { kind: "ready" });
      this.refreshOverlays();
    } catch (e) {
      if (stale() || isAbortError(e)) return;
      this.setStatus(tier, this.failureStatus(tier, e));
    }
    if (!stale()) this.notify();
  }

  setParam(key: string, value: string, { reloadMeta = true } = {}) {
    const params = { ...this.s.params, [key]: value };
    this.set({ params });
    if (key === "jwst_band") {
      // The temperature band is a choice about THIS viewer's JWST frame: its
      // Temp colour is a per-viewer override (the Display panel, and so every
      // other linked viewer, keeps its colour); leaving it drops the override.
      if (value === "temperature") this.setOverride({ color: "temp" });
      else if (this.settings().color === "temp") {
        const { color: _drop, ...rest } = this.s.override;
        void _drop;
        if (this.s.unlinked || mergeDisplay(useDisplay.getState(), rest).color === "temp") {
          this.setOverride({ color: (this.s.meta?.band_names[0] ?? "VIS") as DisplaySettings["color"] });
        } else {
          this.set({ override: rest });
          this.afterDisplayChange();
        }
      }
    }
    if (reloadMeta) this.loadMeta().then(() => this.show(), () => this.show());
  }

  setBhrFwhm(arcsec: number) {
    const param = this.s.meta?.bhr_fwhm_control?.param || "bhr_fwhm_arcsec";
    this.set({ params: { ...this.s.params, [param]: String(Number(arcsec.toFixed(6))) } });
    if (this.bhrRefreshTimer) clearTimeout(this.bhrRefreshTimer);
    this.bhrRefreshTimer = setTimeout(() => { void this.refreshTier("bhr"); }, 90);
  }

  // ---- colour prepare + render ------------------------------------------------
  prepare(rec: CubeRec, color: string, scheme: string[] | undefined): Prepared {
    const key = `${rec.key}:${color}:${scheme ? scheme.join(",") : ""}`;
    if (!rec.noCache) {
      const hit = this.prepCache.get(key);
      if (hit) return hit;
    }
    const prepared = prepareCore(rec, (this.s.meta as ViewerMeta).color, color, scheme);
    if (!rec.noCache) {
      this.prepCache.set(key, prepared);
      if (this.prepCache.size > 48) this.prepCache.delete(this.prepCache.keys().next().value as string);
    }
    return prepared;
  }

  /** The ImageData of a frame's content with the current display settings. */
  renderShown(shown: Shown, settings = this.settings()): ImageData | null {
    if (!this.s.meta) return null;
    if (shown.kind === "cube") {
      const { color, scheme } = prepareColor(settings.color, settings.rgb);
      return renderPrepared(this.prepare(shown.rec, color, scheme), this.displayParams(shown.rec, settings));
    }
    const rec = shown.rec;
    const k = Math.max(0, rec.bands.indexOf(settings.color));
    const band = new Float32Array(rec.h * rec.w);
    for (let p = 0; p < band.length; p++) band[p] = rec.data[p * rec.c + k];
    const params = this.displayParams(rec, settings);
    if (shown.op !== "diff") {
      return renderSigned(band, rec.w, rec.h, { ...params, gain: 1, colormap: settings.residualColormap, scale: "linear", range: shown.op === "ratio" ? 2 : 5 });
    }
    const st = this.residualScale_(rec.key, k, band);
    // Its own scale: the image gain / black point do not apply to a residual.
    return renderSigned(band, rec.w, rec.h, {
      ...params, gain: 1, knee: st.knee, range: st.range, colormap: settings.residualColormap, scale: "asinh",
    });
  }

  /** A − B is scaled by its own noise: knee = σ(MAD), white at the 99.5th
   *  percentile of |A − B| (the image knee would flatten a few-e⁻ residual). */
  private residualScale_(key0: string, k: number, band: Float32Array): { knee: number; range: number } {
    const key = `${key0}:${k}`;
    let st = this.residualScale.get(key);
    if (!st) {
      const abs = Float32Array.from(band, (v) => Math.abs(v));
      const sigma = robustStats(band).sigma;
      const knee = sigma > 0 ? sigma : Math.max(percentile(finiteSorted(abs), 50), 1e-12);
      st = { knee, range: Math.max(percentile(finiteSorted(abs), 99.5), 3 * knee) };
      this.residualScale.set(key, st);
      if (this.residualScale.size > 64) this.residualScale.delete(this.residualScale.keys().next().value as string);
    }
    return st;
  }

  /** How a residual tier is displayed (the figure's signed heat bar). */
  residualDisplay(tier: string, settings = this.settings()): { scale: "asinh" | "linear"; knee: number; range: number; band: string } | null {
    const shown = this.s.shown[tier];
    if (!shown || shown.kind !== "residual") return null;
    const rec = shown.rec;
    const k = Math.max(0, rec.bands.indexOf(settings.color));
    const bandName = rec.bands[k] ?? "";
    if (shown.op !== "diff") return { scale: "linear", knee: 1, range: shown.op === "ratio" ? 2 : 5, band: bandName };
    const band = new Float32Array(rec.h * rec.w);
    for (let p = 0; p < band.length; p++) band[p] = rec.data[p * rec.c + k];
    return { scale: "asinh", ...this.residualScale_(rec.key, k, band), band: bandName };
  }

  /** The publication heat bar of one frame: its unit and display scale (the
   *  WP-B1b handoff) and the display it is rendered with — stretch, black
   *  point, the auto limits of asinh-auto / zscale, colormap and invert —
   *  so the exported plate is labelled with what its panels show. */
  heatbarInfo(tier: string, settings = this.settings()): HeatbarInfo | null {
    const shown = this.s.shown[tier];
    if (!shown || !this.s.meta) return null;
    const rec = shown.rec;
    const t = this.transfer(this.groupOf(rec), settings);
    const bandLabel = settings.color === "lupton" ? "Lupton RGB"
      : settings.color === "temp" ? "temperature composite"
        : settings.color === "rgb" ? `RGB ${settings.rgb.join("/")}`
          : rec.bands.length === 1 ? rec.bands[0] : settings.color;
    if (shown.kind === "residual") {
      const res = this.residualDisplay(tier, settings);
      if (!res) return null;
      return {
        band: res.band || bandLabel, knee: t.knee, gain: t.gain, log: false, unit: publicationUnitLabel(rec.unit), scale: 1,
        signed: { scale: res.scale, knee: res.knee, range: res.range, label: this.tierLabel(tier), stops: heatbarStops(settings.residualColormap, settings.invert, "gray", 9) },
      };
    }
    const cubeRec = shown.rec;
    const { color, scheme } = prepareColor(settings.color, settings.rgb);
    const prep = this.prepare(cubeRec, color, scheme);
    const info: HeatbarInfo = {
      band: bandLabel, knee: t.knee, gain: t.gain,
      log: prep.mode === "gray-log" || this.s.meta.color?.render_mode === "log",
      unit: publicationUnitLabel(cubeRec.unit || this.tierMeta(tier)?.unit),
      scale: cubeRec.displayScale > 0 ? cubeRec.displayScale : 1,
      stretch: settings.stretch, black: t.black,
      stops: heatbarStops(settings.colormap, settings.invert, prep.mode),
    };
    if (settings.stretch === "asinh-auto" || settings.stretch === "zscale") {
      const st = frameAutoStats(prep);
      const f = prep.factor > 0 ? prep.factor : 1;
      info.auto = settings.stretch === "zscale"
        ? { lo: st.z1 / f, hi: st.z2 / f }
        : { lo: st.lo / f, hi: st.hi / f, knee: st.knee / f };
    }
    return info;
  }

  /** Frame labels: tier label + magnitude (± σ on SR when a std tier exists). */
  refreshOverlays() {
    const s = this.s;
    if (!s.meta) return;
    const color = this.settings().color;
    const overlay: Record<string, string> = {};
    for (const [tier, shown] of Object.entries(s.shown)) {
      if (shown.kind === "residual") {
        const sig = shown.op === "chi" ? ` (σ: ${shown.rec.sigmaSource === "tier" ? "std tier" : "robust MAD"})` : "";
        overlay[tier] = `${this.tierLabel(tier)}${sig}`;
        continue;
      }
      overlay[tier] = shown.rec.label + magLabel(magInfo(shown.rec, s.meta.color, color));
    }
    this.set({ overlay: { ...s.overlay, ...overlay } });
    const sr = Object.keys(s.shown).find((t) => t.toLowerCase() === "sr");
    if (sr && (s.meta.tiers || []).some((t) => t.key === "std")) void this.addSigmaToSR(sr);
  }

  private async addSigmaToSR(tier: string) {
    const s = this.s;
    const shown = s.shown[tier];
    if (!shown || shown.kind !== "cube" || !s.meta) return;
    const color = this.settings().color;
    const mi = magInfo(shown.rec, s.meta.color, color);
    if (!mi || mi.mag == null) return;
    const idx = s.index;
    let std: CubeRec;
    try { std = await this.fetchCube("std", idx); } catch { return; }
    if (this.s.index !== idx || this.settings().color !== color || this.s.shown[tier] !== shown) return;
    const si = magInfo(std, s.meta.color, color);
    if (!si || !(si.tot > 0)) return;
    this.set({ overlay: { ...this.s.overlay, [tier]: shown.rec.label + magLabel(mi, sigmaMagnitude(mi, si)) } });
  }

  // ---- frames / geometry ----------------------------------------------------------
  registerFrame(h: FrameHandle): () => void {
    this.frames.set(h.tier, h);
    return () => { if (this.frames.get(h.tier) === h) this.frames.delete(h.tier); };
  }

  geomOf(tier: string): FrameGeom | null {
    const shown = this.s.shown[tier];
    const st = this.s.status[tier];
    if (!shown) return null;
    const rec = shown.rec;
    return { tier, width: rec.w, height: rec.h, pixscale: rec.pixscale > 0 ? rec.pixscale : null, ready: st?.kind === "ready" };
  }
  readyGeoms(): FrameGeom[] {
    return this.frameKeys().map((k) => this.geomOf(k)).filter((g): g is FrameGeom => !!g && g.ready);
  }
  /** `sel` as `tier` sees it. A selection (the pan/zoom view, the lens, the
   *  frozen crop) is made on one tier (`sourceTier`); its centre is that
   *  tier's point mapped through both tiers' WCS (as the readout and the
   *  profiles are), so frames whose footprints differ slightly still show the
   *  same sky. Without a WCS on either side (or when the source tier is not
   *  shown) the normalised (u, v) is shared as-is. The side is angular. */
  selectionOn(tier: string, sel: Selection): Selection {
    const src = sel.sourceTier;
    if (!src || src === tier) return sel;
    const a = this.s.shown[src]?.rec, b = this.s.shown[tier]?.rec;
    if (!a || !b || !a.wcs || !b.wcs) return sel;
    const p = this.mapPoint(src, tier, sel.u * a.w, sel.v * a.h);
    if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y)) return sel;
    return { ...sel, u: p.x / b.w, v: p.y / b.h };
  }
  /** The pixel crop of `sel` on `tier` (matched through the WCS). */
  cropOf(tier: string, sel: Selection | null): Crop | null {
    const g = this.geomOf(tier);
    return g && sel ? resolveCrop(g, this.selectionOn(tier, sel)) : null;
  }
  /** Which part of the image `tier`'s frame of side S draws for `view`. */
  layoutOf(tier: string, S?: number, view: Selection | null = this.s.view): FrameLayout | null {
    const g = this.geomOf(tier);
    const h = this.frames.get(tier);
    if (!g) return null;
    return frameLayout(g, view ? this.selectionOn(tier, view) : null, S ?? h?.size() ?? 1);
  }

  // ---- pan / zoom view ------------------------------------------------------------
  /** Zoom the shared view by `factor` about normalised (u, v) of `tier`. */
  zoomView(tier: string, factor: number, anchor?: { u: number; v: number }) {
    const g = this.geomOf(tier) ?? this.readyGeoms()[0];
    if (!g) return;
    const frames = this.readyGeoms();
    const cur = this.s.view;
    const crop = cur ? this.cropOf(g.tier, cur) : null;
    const extent = Math.min(g.width, g.height);
    const side = crop ? crop.side : extent;
    const nextSide = Math.max(extent / VIEW_MAX_ZOOM, Math.min(extent, side / factor));
    if (nextSide >= extent - 1e-9) { this.set({ view: null }); this.afterViewChange(); return; }
    const cu = cur ? (crop as NonNullable<typeof crop>).cx / g.width : 0.5;
    const cv = cur ? (crop as NonNullable<typeof crop>).cy / g.height : 0.5;
    const a = anchor ?? { u: cu, v: cv };
    // keep the anchor fixed on screen: new centre = a + (c − a)·(nextSide/side)
    const r = nextSide / side;
    const u = a.u + (cu - a.u) * r;
    const v = a.v + (cv - a.v) * r;
    const pixscale = g.pixscale;
    const view: Selection = {
      u, v, angularSideArcsec: pixscale ? nextSide * pixscale : null, relativeSide: nextSide / extent, sourceTier: g.tier,
    };
    this.set({ view: clampSelectionToFrames(view, frames) });
    this.afterViewChange();
  }
  /** Pan the view by a frame-pixel drag delta on `tier`. */
  panBy(tier: string, dX: number, dY: number, start: Selection) {
    const g = this.geomOf(tier);
    const L = this.layoutOf(tier);
    if (!g || !L || !this.s.view) return;
    const du = -(dX * (L.sw / L.dw)) / g.width;
    const dv = -(dY * (L.sh / L.dh)) / g.height;
    // The drag is measured on `tier`: move the view's centre in its coordinates.
    const base = this.selectionOn(tier, start);
    this.set({ view: clampSelectionToFrames({ ...base, sourceTier: tier, u: base.u + du, v: base.v + dv }, this.readyGeoms()) });
    this.afterViewChange();
  }
  resetView() { this.set({ view: null }); this.afterViewChange(); }
  setViewSelection(view: Selection | null) {
    this.set({ view: view ? clampSelectionToFrames(view, this.readyGeoms()) : null });
    this.afterViewChange();
  }
  private afterViewChange() {
    for (const h of this.frames.values()) h.redraw();
    if (this.s.hover || this.s.frozen) this.refreshLenses();
    this.notify();
  }
  zoomTo(ra: number, dec: number, fovArcsec = 5): boolean {
    for (const k of this.frameKeys()) {
      const shown = this.s.shown[k];
      if (!shown || !shown.rec.wcs) continue;
      const p = skyToPix(shown.rec.wcs, ra, dec);
      if (!p) continue;
      const rec = shown.rec;
      const u = (p.x + 0.5) / rec.w, v = (p.y + 0.5) / rec.h;
      if (u < 0 || u > 1 || v < 0 || v > 1) continue;
      this.setViewSelection({ u, v, angularSideArcsec: fovArcsec, relativeSide: rec.pixscale > 0 ? fovArcsec / rec.pixscale / Math.min(rec.w, rec.h) : 0.2, sourceTier: k });
      return true;
    }
    return false;
  }

  // ---- readout ------------------------------------------------------------------
  /** The pointer is over (fx, fy) — continuous image coordinates — of `tier`. */
  hoverAt(tier: string, fx: number, fy: number) {
    const s = this.s;
    const src = s.shown[tier];
    const srcRec = src?.rec ?? null;
    const srcW = srcRec?.w ?? this.geomOf(tier)?.width ?? 1;
    const srcH = srcRec?.h ?? this.geomOf(tier)?.height ?? 1;
    const sky = srcRec?.wcs ? pixToSky(srcRec.wcs, fx - 0.5, fy - 0.5) : null;
    const tiers: ReadoutTier[] = [];
    for (const k of this.frameKeys()) {
      const shown = s.shown[k];
      if (!shown) continue;
      const rec = shown.rec;
      let px: number, py: number;
      if (k === tier) { px = fx; py = fy; }
      else if (sky && rec.wcs) {
        const p = skyToPix(rec.wcs, sky.ra, sky.dec);
        if (!p) continue;
        px = p.x + 0.5; py = p.y + 0.5;
      } else {
        px = (fx / srcW) * rec.w; py = (fy / srcH) * rec.h;
      }
      const ix = Math.floor(px), iy = Math.floor(py);
      const inside = ix >= 0 && iy >= 0 && ix < rec.w && iy < rec.h;
      const bands = shown.kind === "cube"
        ? (rec.bands.length === rec.c ? rec.bands : (s.meta?.band_names ?? []).slice(0, rec.c))
        : rec.bands;
      tiers.push({
        tier: k, label: this.tierLabel(k), x: inside ? ix : null, y: inside ? iy : null, fx: px, fy: py,
        bands, values: inside ? pixelValues(rec, ix, iy) : null,
        unit: shown.kind === "cube" ? (rec.unit || this.tierMeta(k)?.unit || "") : rec.unit,
      });
    }
    this.set({ readout: { tier, sky, tiers } });
  }
  clearReadout() { if (this.s.readout) this.set({ readout: null }); }

  /** A continuous point of `from` on the grid of `to`: through the sky when
   *  both tiers have a WCS, else by normalised position. */
  mapPoint(from: string, to: string, fx: number, fy: number): { x: number; y: number } | null {
    if (from === to) return { x: fx, y: fy };
    const a = this.s.shown[from]?.rec, b = this.s.shown[to]?.rec;
    if (!a || !b) return null;
    if (a.wcs && b.wcs) {
      const sky = pixToSky(a.wcs, fx - 0.5, fy - 0.5);
      const p = sky ? skyToPix(b.wcs, sky.ra, sky.dec) : null;
      return p ? { x: p.x + 0.5, y: p.y + 0.5 } : null;
    }
    return { x: (fx / a.w) * b.w, y: (fy / a.h) * b.h };
  }

  /** The prepared colour mode a frame renders in (temp legend, histogram). */
  preparedMode(shown: Shown, settings = this.settings()): string {
    if (shown.kind !== "cube" || !this.s.meta) return "signed";
    const { color, scheme } = prepareColor(settings.color, settings.rgb);
    return this.prepare(shown.rec, color, scheme).mode;
  }

  /** A frame's pixels changed: lens popups redraw. */
  bumpDrawn() { if (this.s.hover || this.s.frozen) this.set({ drawn: this.s.drawn + 1 }); }

  setTool(tool: Tool) { this.set({ tool }); this.clearAllLenses(); this.notify(); }
  setAltLens(on: boolean) {
    if (on === this.s.altLens) return;
    this.set({ altLens: on });
    if (!on && !this.s.frozen) this.hideHover();
  }
  setPanels(patch: Partial<Pick<ViewerStoreState, "histogram" | "profileOpen" | "swipe" | "morphAmp" | "morphSpeed">>) { this.set(patch); }

  // ---- lens (magnifier) -----------------------------------------------------------
  lensActive(): boolean { return this.s.tool === "lens" || this.s.altLens || !!this.s.frozen; }

  lensHover(tier: string, u: number, v: number) {
    const s = this.s;
    if (s.frozen) return;
    const g = this.geomOf(tier);
    if (!g || !g.ready) return;
    const hover = selectionAt(g, u, v, s.hover, this.readyGeoms(), lensSide(currentViewport()));
    this.set({ hover });
    this.refreshLenses();
  }
  lensZoom(tier: string, u: number, v: number, deltaY: number) {
    const s = this.s;
    if (s.frozen) return;
    const g = this.geomOf(tier);
    if (!g || !g.ready) return;
    const side = lensSide(currentViewport());
    let hover = selectionAt(g, u, v, s.hover, this.readyGeoms(), side);
    if (!hover) return;
    hover = zoomSelection(g, hover, deltaY < 0 ? LENS_ZOOM_STEP : 1 / LENS_ZOOM_STEP, this.readyGeoms(), side);
    this.set({ hover });
    this.refreshLenses();
  }
  /** Horizontal wheel over a frame: brightness of that frame's group. */
  wheelGain(tier: string, deltaX: number) {
    const shown = this.s.shown[tier];
    const group = this.groupOf(shown?.rec ?? null);
    const t = this.transfer(group);
    this.setTransfer(group, { gain: Math.max(0.1, Math.min(10, t.gain * Math.exp(-deltaX * 0.002))) });
  }
  hideHover() {
    if (!this.s.hover && !Object.keys(this.s.lens).length) return;
    this.set({ hover: null, lens: this.s.frozen ? this.s.lens : {} });
    this.updateSaveControls();
  }
  clearAllLenses() {
    this.set({ hover: null, frozen: null, lens: {} });
    this.setSaveStatus({ text: "", tone: "" });
    if (!this.s.hot && keyboard.active === this) keyboard.active = null;
    this.updateSaveControls();
  }
  toggleFrozen(tier: string, u: number, v: number) {
    const s = this.s;
    if (s.frozen) {
      this.clearAllLenses();
      this.notify();
      return;
    }
    const g = this.geomOf(tier);
    if (!g) return;
    // Freeze what the user can already see (recomputing from the click could shift it).
    const selected = s.hover || selectionAt(g, u, v, null, this.readyGeoms(), lensSide(currentViewport()));
    if (!selected) return;
    const frozen: Selection = Object.freeze({ ...selected, revision: ++this.selectionRevision });
    keyboard.active = this;
    // Frozen popups keep the (document) position they had while hovering.
    const lens = Object.keys(s.lens).length ? s.lens : this.computeLensPlacements(frozen, true);
    this.set({ frozen, hover: null, lens });
    this.setSaveStatus({ text: "", tone: "" });
    this.set({ drawn: this.s.drawn + 1 });
    this.updateSaveControls();
    this.notify();
  }

  /** Popup positions (document coordinates) for every ready frame. */
  private computeLensPlacements(selection: Selection, frozen: boolean, vp = currentViewport()): Record<string, LensPlacement> {
    const side = lensSide(vp);
    const positions: (LensPosition & { tier: string })[] = [];
    for (const k of this.frameKeys()) {
      const h = this.frames.get(k);
      const g = this.geomOf(k);
      if (!h || !g || !g.ready) continue;
      const L = this.layoutOf(k);
      const crop = this.cropOf(k, selection);
      if (!L || !crop) continue;
      const rect = h.element.getBoundingClientRect();
      if (!(rect.width > 0 && rect.height > 0)) continue;
      // Frame CSS coordinates start at the padding box (inside the border).
      const o = contentBoxOrigin(h.element);
      const P = imageToFrame(L, crop.cx, crop.cy);
      const x = o.left + P.x, y = o.top + P.y;
      const init = placeLensPopup(side, x, y, vp, frozen);
      const docLeft = frozen ? init.left : init.left + vp.scrollX;
      const docTop = frozen ? init.top : init.top + vp.scrollY;
      const a = imageToFrame(L, crop.x, crop.y), b = imageToFrame(L, crop.x + crop.side, crop.y + crop.side);
      positions.push({
        tier: k, current: rectOf(docLeft, docTop, side, side),
        sourceRect: { left: o.left + a.x, top: o.top + a.y, width: b.x - a.x, height: b.y - a.y },
        corner: this.s.lens[k]?.corner ?? null,
      });
    }
    const placed = resolveLensOverlaps(positions, vp);
    const out: Record<string, LensPlacement> = {};
    positions.forEach((p, i) => { out[p.tier] = { left: placed[i].left, top: placed[i].top, corner: placed[i].corner }; });
    return out;
  }

  refreshLenses() {
    const s = this.s;
    if (s.frozen) {
      const frozen = Object.freeze({ ...clampSelectionToFrames(s.frozen, this.readyGeoms()), revision: s.frozen.revision });
      // Frozen popups stay where they were; frames added later get a placement.
      const missing = this.frameKeys().filter((k) => !s.lens[k]);
      const lens = missing.length ? { ...this.computeLensPlacements(frozen, true), ...s.lens } : s.lens;
      this.set({ frozen, lens });
    } else if (s.hover) {
      const hover = clampSelectionToFrames(s.hover, this.readyGeoms());
      this.set({ hover, lens: this.computeLensPlacements(hover, false) });
    }
    this.updateSaveControls();
  }

  // ---- profiles ---------------------------------------------------------------------
  setProfile(p: ProfileGeom | null) { this.set({ profile: p, profileOpen: p ? true : this.s.profileOpen }); }

  // ---- navigation ---------------------------------------------------------------------
  go(i: number, wrap = true) {
    const n = this.s.meta?.count ?? 0;
    if (n === 0) return;
    this.set({ index: wrap ? ((i % n) + n) % n : Math.max(0, Math.min(i, n - 1)) });
    void this.show();
  }
  togglePlay() {
    if (this.playTimer) {
      clearInterval(this.playTimer);
      this.playTimer = null;
      this.set({ playing: false });
    } else {
      this.playTimer = setInterval(() => this.go(this.s.index + 1), this.s.playMs);
      this.set({ playing: true });
    }
  }
  setPlaySpeed(ms: number) {
    this.set({ playMs: ms });
    if (this.playTimer) {
      clearInterval(this.playTimer);
      this.playTimer = setInterval(() => this.go(this.s.index + 1), ms);
    }
  }
  setLayout(layout: Layout) {
    this.set({ layout });
    saveViewLayout(layout);
    this.notify();
  }
  /** Chip click: multi-select, canonical order, at least one tier. */
  toggleTier(key: string) {
    const s = this.s;
    if (!s.tiers.includes(key) && this.tierDisabled(key)) return;
    const set = new Set(s.tiers);
    if (set.has(key)) { if (set.size > 1) set.delete(key); } else set.add(key);
    const tiers = (s.meta?.tiers || []).map((t) => t.key).filter((k) => set.has(k));
    this.set({ tiers });
    void this.show({ preserveFrozen: true });
  }
  setTiers(keys: string | string[]) {
    const s = this.s;
    if (!s.meta) return;
    const wanted = (Array.isArray(keys) ? keys : [keys]).filter((k) => !this.tierDisabled(k));
    const next = (s.meta.tiers || []).map((t) => t.key).filter((k) => wanted.includes(k));
    if (!next.length) return;
    this.set({ tiers: next });
    void this.show({ preserveFrozen: true });
  }
  addResidual(op: ResidualOp, a: string, b: string) {
    const key = residualKey(op, a, b);
    if (a === b || this.s.residuals.includes(key)) return;
    this.set({ residuals: [...this.s.residuals, key] });
    void this.show({ preserveFrozen: true });
  }
  removeResidual(key: string) {
    this.set({ residuals: this.s.residuals.filter((k) => k !== key) });
    void this.show({ preserveFrozen: true });
  }
  setCompare(compare: Compare) {
    this.set({ compare, blinkAt: 0 });
    if (this.blinkTimer) { clearInterval(this.blinkTimer); this.blinkTimer = null; }
    if (compare === "blink") {
      this.blinkTimer = setInterval(() => {
        const n = this.frameKeys().length;
        if (n > 1) this.set({ blinkAt: (this.s.blinkAt + 1) % n });
      }, this.s.blinkMs);
    }
    this.clearAllLenses();
    this.notify();
  }
  setBlinkMs(ms: number) {
    this.set({ blinkMs: ms });
    if (this.s.compare === "blink") this.setCompare("blink");
  }

  // ---- the disagreement movie --------------------------------------------------------
  setMorphMembers(csv: string | null) {
    const v = csv == null || csv === "" ? null : String(csv);
    if (v === this.s.morphMembers) return;
    this.set({ morphMembers: v });
    if (this.s.tiers.includes("morph")) void this.show({ preserveFrozen: true });
  }

  private stopMorph() {
    if (this.morphRaf != null) cancelAnimationFrame(this.morphRaf);
    this.morphRaf = null;
  }

  private movieFresh(e: MovieEntry | undefined, color: string): e is MovieEntry {
    return !!e && e.done && e.color === color && e.amp === this.s.morphAmp;
  }

  private movieColorKey(): string {
    const st = this.settings();
    const { color, scheme } = prepareColor(st.color, st.rgb);
    return `${color}${scheme ? `:${scheme.join(",")}` : ""}`;
  }

  private async movieCubes(index: number, subset: string | null) {
    const meta = this.s.meta as ViewerMeta;
    const extra = subset ? { members: subset } : undefined;
    const n = pcaCount(meta, subset);
    const sr = await this.fetchCube(morphBaseTier(meta), index, extra);
    const comps: CubeRec[] = [];
    for (let k = 0; k < n; k++) {
      try { comps.push(await this.fetchCube(`pca${k}`, index, extra)); } catch { /* fewer PCs */ }
    }
    const metaAmps = (meta.pca_amps && meta.pca_amps[index]) || [];
    const amps = comps.map((c, k) => (c.amp != null ? c.amp : (metaAmps[k] || 0)));
    return { sr, comps, amps, label: movieLabel(subset, comps) };
  }

  private async buildMovie(index: number, subset: string | null, token: number, onProgress?: (p: number) => void): Promise<MovieEntry | null> {
    const key = movieKey(index, subset);
    const colorKey = this.movieColorKey();
    const existing = this.movies.get(key);
    if (this.movieFresh(existing, colorKey)) return existing;
    const amp = this.s.morphAmp;
    let ing;
    try { ing = await this.movieCubes(index, subset); } catch { return null; }
    if (token !== this.buildToken) return null;
    const { sr, comps, amps, label } = ing;
    const data = new Float32Array(sr.data.length);
    const entry: MovieEntry = {
      frames: new Array(MORPH_FRAMES).fill(null), w: sr.w, h: sr.h, mode: "gray", bytes: 0, color: colorKey, amp,
      done: false, label, pixscale: sr.pixscale, rec: sr,
    };
    this.movies.set(key, entry);
    const st = this.settings();
    const { color, scheme } = prepareColor(st.color, st.rgb);
    const compData = comps.map((c) => c.data);
    for (let slot = 0; slot < MORPH_FRAMES; slot++) {
      if (token !== this.buildToken || this.destroyed) { this.movies.delete(key); return null; }
      synthesizeMorphFrame(sr.data, compData, morphCoefficients(amps, amp, slot / MORPH_FRAMES), data);
      entry.frames[slot] = this.prepare({ ...sr, key: `mv:${key}:${slot}`, data, noCache: true }, color, scheme);
      onProgress?.((slot + 1) / MORPH_FRAMES);
      await new Promise((r) => setTimeout(r, 0));
    }
    entry.mode = entry.frames[0] ? entry.frames[0].mode : "gray";
    entry.bytes = movieBytes(MORPH_FRAMES, entry.w, entry.h, entry.mode);
    entry.done = true;
    this.movies.evict();
    return entry;
  }

  private playMovie(index: number, entry: MovieEntry) {
    this.stopMorph();
    let phase = 0, last = performance.now();
    let lastAmp = this.s.morphAmp, lastColor = this.movieColorKey(), changedAt = 0;
    this.setShown("morph", { kind: "cube", rec: { ...entry.rec, label: entry.label, noCache: true } });
    this.setStatus("morph", { kind: "ready" });
    this.set({ overlay: { ...this.s.overlay, morph: entry.label } });
    const drawSlot = (slot: number) => {
      const prep = entry.frames[slot];
      const h = this.frames.get("morph");
      if (!prep || !h) return;
      const img = renderPrepared(prep, this.displayParams(entry.rec));
      if (h.source.width !== entry.w || h.source.height !== entry.h) { h.source.width = entry.w; h.source.height = entry.h; }
      h.source.getContext("2d")?.putImageData(img, 0, 0);
      h.redraw();
      if (this.s.hover || this.s.frozen) this.set({ drawn: this.s.drawn + 1 });
    };
    drawSlot(0);
    const tick = (now: number) => {
      // Colour / amplitude change the prepare → rebuild once the slider settles.
      const colorKey = this.movieColorKey();
      if (this.s.morphAmp !== lastAmp || colorKey !== lastColor) { lastAmp = this.s.morphAmp; lastColor = colorKey; changedAt = now; }
      if ((entry.amp !== this.s.morphAmp || entry.color !== colorKey) && now - changedAt > 250) { void this.startMorph(index); return; }
      phase += ((now - last) / 1000) * this.s.morphSpeed;
      last = now;
      drawSlot(slotAt(phase));
      this.morphRaf = requestAnimationFrame(tick);
    };
    this.morphRaf = requestAnimationFrame(tick);
  }

  private async prefetchMovies(index: number, subset: string | null, token: number) {
    const count = this.s.meta?.count ?? 0;
    for (let d = 1; d <= MOVIE_RADIUS; d++) {
      for (const j of [index + d, index - d]) {
        if (token !== this.buildToken || this.destroyed) return;
        if (j < 0 || j >= count) continue;
        if (this.movies.totalBytes() > this.movies.budget) return;
        if (this.movieFresh(this.movies.peek(movieKey(j, subset)), this.movieColorKey())) continue;
        await this.buildMovie(j, subset, token);
      }
    }
  }

  async startMorph(index: number) {
    this.stopMorph();
    const token = ++this.buildToken;
    const subset = this.s.morphMembers;
    const key = movieKey(index, subset);
    this.movies.playing = key;
    let entry = this.movies.get(key);
    if (!this.movieFresh(entry, this.movieColorKey())) {
      this.set({ movieProgress: { ...this.s.movieProgress, morph: 0 } });
      entry = (await this.buildMovie(index, subset, token, (p) => this.set({ movieProgress: { ...this.s.movieProgress, morph: p } }))) ?? undefined;
      this.set({ movieProgress: { ...this.s.movieProgress, morph: null } });
      if (token !== this.buildToken) return;
      if (!entry) { this.setStatus("morph", { kind: "error", message: "movie unavailable" }); return; }
    }
    this.playMovie(index, entry);
    void this.prefetchMovies(index, subset, token);
  }

  // ---- save crop to results -----------------------------------------------------------
  saveBlockReason(): string {
    const s = this.s;
    if (s.saveInFlight) return "A crop is already being saved.";
    if (!s.frozen) return "Click an image (lens tool) to freeze a matched crop first.";
    if (s.tiers.includes("morph")) return "Animated morph frames cannot be saved as science cubes.";
    const frames = this.frameKeys();
    if (!frames.length) return "No image frames are open.";
    if (frames.length > RESULT_MAX_TIERS) return `Select at most ${RESULT_MAX_TIERS} result tiers.`;
    const unsupported = frames.find((t) => !RESULT_SAVEABLE_TIERS.has(String(t).toLowerCase()));
    if (unsupported) return `${unsupported} is a display-only tier and cannot be saved.`;
    if (this.collection === "jwst-euclid" && frames.some((t) => String(t).toLowerCase() === "jwst")
      && String(s.params.jwst_band || "").toUpperCase() !== "F200W") return NATIVE_F200W_SAVE_REASON;
    if (frames.some((t) => s.status[t]?.kind !== "ready" || !s.shown[t])) return "Wait for every selected image cube to finish loading.";
    return "";
  }
  private setSaveStatus(save: SaveStatus) { this.set({ save }); }
  updateSaveControls() {
    const reason = this.saveBlockReason();
    if (reason === NATIVE_F200W_SAVE_REASON) this.setSaveStatus({ text: reason, tone: "error", kind: "save-blocker" });
    else if (this.s.save.kind === "save-blocker") this.setSaveStatus({ text: "", tone: "" });
  }
  async saveCropToResults(): Promise<void> {
    if (this.saveBlockReason()) { this.updateSaveControls(); return; }
    const s = this.s;
    const st = this.getState();
    const payload = {
      collection: this.collection,
      index: s.index,
      tiers: this.frameKeys(),
      params: { ...s.params },
      selection: serializeSelection(s.frozen),
      display: { color: st.color, layout: s.layout, knee: st.knee, gain: st.gain, transfers: st.transfers },
    };
    const controller = new AbortController();
    this.saveController = controller;
    this.set({ saveInFlight: true });
    this.setSaveStatus({ text: "Saving matched raw cubes…", tone: "busy" });
    try {
      const result = await apiPost<{ result_id?: string; id?: string }>("/viewer/results", payload, { json: true, signal: controller.signal });
      if (!this.destroyed) this.setSaveStatus({ text: `Saved ${result.result_id || result.id || "result"}`, tone: "saved" });
    } catch (e) {
      if (!this.destroyed && !isAbortError(e)) this.setSaveStatus({ text: `Save failed: ${errMessage(e)}`, tone: "error" });
    } finally {
      if (this.saveController === controller) { this.saveController = null; this.set({ saveInFlight: false }); }
      if (!this.destroyed) this.updateSaveControls();
    }
  }

  // ---- keyboard -------------------------------------------------------------------------
  activate() { this.set({ hot: true }); keyboard.active = this; }
  deactivate() {
    this.set({ hot: false });
    if (!this.s.frozen && keyboard.active === this) keyboard.active = null;
  }
  isActive() { return keyboard.active === this; }
  setRecorder(r: { stop: () => void } | null) { this.recorder = r; this.set({ recording: !!r }); }
  getRecorder() { return this.recorder; }

  /** The old engine's document-level keys (q–y colour, ← →, Space, S; like
   *  the old engine, Shift+letter = the letter) plus the new view keys
   *  (+ − 0, L lens, B blink, Esc). */
  private onKey = (e: KeyboardEvent) => {
    const t = e.target as HTMLElement | null;
    if (t && typeof t.matches === "function" && (t.matches("input, textarea, select, [contenteditable]:not([contenteditable='false'])") || t.closest?.("[role='dialog'], [role='alertdialog']"))) return;
    if (e.ctrlKey || e.metaKey || e.altKey || e.defaultPrevented) return;
    const key = e.key.toLowerCase();
    // A Shift+letter the shell (Shift+D/J/T) or the page binds is theirs:
    // this listener on `document` runs before theirs on `window`.
    if (e.shiftKey && /^[a-z]$/.test(key) && shiftComboClaimed(key)) return;
    // A frozen viewer keeps S even after the pointer left its tiles.
    if (key === "s") {
      if (keyboard.active === this && this.s.frozen) { void this.saveCropToResults(); e.preventDefault(); }
      return;
    }
    if (!this.s.hot || keyboard.active !== this) return;
    const meta = this.s.meta;
    const colorIndex = COLOR_KEYS.indexOf(key);
    if (colorIndex >= 0 && meta && meta.render_mode !== "log") {
      const colors = [...(meta.band_names || []), ...COLOR_MODES_EXTRA.map((m) => m.key)];
      const color = colors[colorIndex];
      if (color && color !== this.settings().color) this.setColor(color);
      if (color) e.preventDefault();
      return;
    }
    const first = this.frameKeys()[0];
    if (e.key === "ArrowLeft") { this.go(this.s.index - 1); e.preventDefault(); }
    else if (e.key === "ArrowRight") { this.go(this.s.index + 1); e.preventDefault(); }
    else if (e.key === " ") { this.togglePlay(); e.preventDefault(); }
    else if ((e.key === "+" || e.key === "=") && first) { this.zoomView(first, 1.5); e.preventDefault(); }
    else if ((e.key === "-" || e.key === "_") && first) { this.zoomView(first, 1 / 1.5); e.preventDefault(); }
    else if (e.key === "0") { this.resetView(); e.preventDefault(); }
    else if (key === "l") { this.set({ tool: this.s.tool === "lens" ? "pan" : "lens" }); this.clearAllLenses(); e.preventDefault(); }
    else if (key === "b" && this.frameKeys().length > 1) { this.setCompare(this.s.compare === "blink" ? "off" : "blink"); e.preventDefault(); }
    else if (e.key === "Escape") {
      if (this.s.frozen || this.s.profile) { this.clearAllLenses(); this.setProfile(null); e.preventDefault(); }
    }
  };

  // ---- lifecycle --------------------------------------------------------------------------
  async start(): Promise<void> {
    try {
      await this.loadMeta();
    } catch {
      return;
    }
    await this.show();
  }
  async reload(): Promise<void> {
    try { await this.loadMeta(); } catch { return; }
    await this.show();
  }

  destroy() {
    this.paramRefreshToken++;
    this.showRevision++;
    this.tierRefreshRevisions.clear();
    if (this.bhrRefreshTimer) clearTimeout(this.bhrRefreshTimer);
    this.destroyed = true;
    this.saveController?.abort();
    this.life.abort();
    document.removeEventListener("keydown", this.onKey);
    if (keyboard.active === this) keyboard.active = null;
    if (this.playTimer) clearInterval(this.playTimer);
    if (this.blinkTimer) clearInterval(this.blinkTimer);
    this.recorder?.stop();
    this.stopMorph();
    this.buildToken++;
    this.movies.clear();
    this.frames.clear();
  }

  private buildApi(): ViewerApi {
    return {
      goTo: (i) => this.go(i, false),
      goToId: async (id) => {
        const ok = await this.resolveId(id);
        if (ok) await this.show();
        return ok;
      },
      setTiers: (keys) => this.setTiers(keys),
      setView: (patch) => {
        const next = patch || {};
        const o: Partial<DisplaySettings> = { ...this.s.override };
        const meta = this.s.meta;
        if (typeof next.color === "string") {
          const allowed = !meta || meta.band_names.includes(next.color) || ["lupton", "temp"].includes(next.color);
          if (allowed) o.color = next.color as DisplaySettings["color"];
        }
        const knee = Number.isFinite(next.knee) && (next.knee as number) > 0 ? next.knee as number : null;
        const gain = Number.isFinite(next.gain) && (next.gain as number) > 0 ? next.gain as number : null;
        if (knee != null || gain != null) {
          const groups = { ...(o.groups ?? {}) };
          const cur = this.settings();
          for (const g of ["default", "euclid", "jwst"]) {
            const base = groups[g] ?? cur.groups[g] ?? { knee: 100, gain: 1, black: 0 };
            groups[g] = { ...base, ...(knee != null ? { knee } : {}), ...(gain != null ? { gain } : {}) };
          }
          o.groups = groups;
        }
        this.set({ override: o });
        this.afterDisplayChange();
      },
      setParams: (patch) => {
        this.set({ params: { ...this.s.params, ...(patch || {}) } });
        return this.refreshVisible();
      },
      setMorphMembers: (csv) => this.setMorphMembers(csv),
      getIndex: () => this.s.index,
      isReady: () => !!this.s.meta,
      getState: () => this.getState(),
      exportFigure: () => { this.onExportFigure?.(); },
      savePng: () => { this.onSavePng?.(); },
      saveCropToResults: () => this.saveCropToResults(),
      reload: () => this.reload(),
      zoomTo: (ra, dec, fov) => this.zoomTo(ra, dec, fov),
      resetView: () => this.resetView(),
      getReadout: () => this.s.readout,
      destroy: () => this.destroy(),
    };
  }

  /** Set by <ImageViewer> (the exports need the mounted canvases). */
  onExportFigure: (() => void) | null = null;
  onSavePng: (() => void) | null = null;

  /** Stem for downloaded files. */
  stem(): string {
    return exportStem(this.collection, this.s.index, this.frameKeys(), this.settings().color);
  }
}
