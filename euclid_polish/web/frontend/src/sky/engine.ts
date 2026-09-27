/* The Aladin Lite engine of the Sky atlas (spec §7.1, sky_tech §2).
 *
 *   const engine = await getSkyEngine({ view, theme, background });
 *   engine.attach(slotDiv);        // show it inside the atlas
 *   const off = engine.events.on("zoomChanged", (fov) => …);
 *   engine.detach(slotDiv);        // leaving the route: park it, keep it alive
 *
 * - `aladin-lite` is imported lazily (2.4 MB + a WASM compile at evaluation):
 *   nothing may import it statically.
 * - `A.init` is a Promise PROPERTY; it rejects (a string) without WebGL2, so
 *   WebGL2 is probed before the download and reported as `SkyEngineError`.
 * - ONE Aladin instance per page lifetime: there is no `destroy()` and every
 *   instance holds a WebGL2 context, an rAF loop and a ResizeObserver. The
 *   instance lives in a host div that moves between the atlas's slot and a
 *   hidden parking box on <body> (visibility:hidden at the last slot size, so
 *   Aladin never sees a zero-size resize). `getSkyEngine` is a module-level
 *   singleton, so React StrictMode's double effects share it.
 * - `al.on` keeps ONE callback per event and has no `off`: each event is
 *   registered once here and fanned out through `engine.events`.
 * - `log: false` (the default logs every page URL to CDS); the Aladin logo
 *   and credits are kept (terms of use).
 * - The image stack (background HiPS + overlay HiPS + FITS overlays) is
 *   owned here (`syncImageStack`): Aladin's base layer is simply the first
 *   stack entry, so removing the base ("none") would promote an overlay.
 */
import { Emitter } from "./emitter";
import type { HipsColor } from "./surveys";
import type {
  Aladin, AladinEventName, AladinImageLayer, AladinSelectionShape, AladinStatic, SkyXY,
} from "./types";

export type ClickEvent = { ra: number; dec: number; x: number; y: number; isDragging?: boolean };
/** Cursor position in ICRS degrees (NaN off the sky). */
export type MoveEvent = { ra: number; dec: number; x: number; y: number };
export type PositionEvent = { ra: number; dec: number; dragging?: boolean };
export type ContextEvent = { ra: number | null; dec: number | null; x: number; y: number; clientX: number; clientY: number };

export type SkyEventMap = {
  objectClicked: [obj: unknown, xy?: SkyXY];
  objectHovered: [obj: unknown, xy?: SkyXY];
  objectHoveredStop: [obj: unknown, xy?: SkyXY];
  footprintClicked: [obj: unknown, xy?: SkyXY];
  click: [e: ClickEvent];
  positionChanged: [e: PositionEvent];
  zoomChanged: [fov: number];
  mouseMove: [e: MoveEvent];
  projectionChanged: [projection: string];
  cooFrameChanged: [frame: unknown];
  resizeChanged: [width: number, height: number];
  /** Right-click on the sky (ours: Aladin's own menu is off). */
  contextMenu: [e: ContextEvent];
  /** The view changed: once per animation frame after position/zoom/resize. */
  viewChanged: [];
};

/** The Aladin events fanned out (registered exactly once per instance). */
export const FANNED_EVENTS: readonly AladinEventName[] = [
  "objectClicked", "objectHovered", "objectHoveredStop", "footprintClicked", "click",
  "positionChanged", "zoomChanged", "mouseMove", "projectionChanged", "cooFrameChanged", "resizeChanged",
];

export type SkyView = { ra: number; dec: number; fov: number };

export type EngineInit = {
  view: SkyView & { proj: string };
  theme: "light" | "dark";
  /** CSS colour behind the imagery (and the whole sky with no background HiPS). */
  background: string;
  /** Initial size of the parking box (the first slot's size). */
  size?: { width: number; height: number };
};

export type StackEntry = {
  /** Stable key (layer name in Aladin). */
  key: string;
  /** Rebuilds the layer when it changes (url, format, …). */
  signature: string;
  make: (A: AladinStatic) => AladinImageLayer;
  opacity?: number;
  /** Colour applied live (no rebuild) whenever it changes. */
  color?: BaseColor;
};

export type BaseColor = HipsColor & {
  gamma?: number;
  saturation?: number;
  brightness?: number;
  contrast?: number;
};

export class SkyEngineError extends Error {
  constructor(readonly code: "webgl2" | "load", message: string) {
    super(message);
    this.name = "SkyEngineError";
  }
}

/* ── lazy library load ───────────────────────────────────────────────── */

type Importer = () => Promise<{ default: unknown }>;

const hooks: { importer: Importer; hasWebGL2: () => boolean } = {
  importer: () => import("aladin-lite"),
  hasWebGL2: probeWebGL2,
};

function probeWebGL2(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return !!canvas.getContext("webgl2");
  } catch {
    return false;
  }
}

export function hasWebGL2(): boolean {
  return hooks.hasWebGL2();
}

let libPromise: Promise<AladinStatic> | null = null;

/** Import aladin-lite once and wait for its WASM (`await A.init`). */
export function loadAladin(): Promise<AladinStatic> {
  if (!libPromise) {
    libPromise = hooks.importer().then(async (mod) => {
      const A = mod.default as AladinStatic;
      try {
        await A.init;
      } catch (err) {
        const msg = typeof err === "string" ? err : err instanceof Error ? err.message : String(err);
        throw new SkyEngineError(/webgl/i.test(msg) ? "webgl2" : "load", msg);
      }
      return A;
    });
    libPromise.catch(() => { libPromise = null; }); // let a later visit retry
  }
  return libPromise;
}

/* ── the engine ──────────────────────────────────────────────────────── */

const BASE_KEY = "base";

type StackLayer = { entry: StackEntry; layer: AladinImageLayer; name: string; colorSig?: string };

const colorSig = (c: BaseColor | undefined) => (c ? JSON.stringify(c) : "");

/** Apply colour settings to an image layer (unknown setters are skipped). */
export function applyLayerColor(layer: AladinImageLayer, color: BaseColor): void {
  if (color.colormap || color.stretch || color.reversed != null) {
    layer.setColormap(color.colormap ?? "native", { stretch: color.stretch, reversed: color.reversed });
  }
  if (color.minCut != null && color.maxCut != null && Number.isFinite(color.minCut) && Number.isFinite(color.maxCut)) {
    layer.setCuts(color.minCut, color.maxCut);
  }
  if (color.gamma != null) layer.setGamma(color.gamma);
  if (color.saturation != null) layer.setSaturation(color.saturation);
  if (color.brightness != null) layer.setBrightness(color.brightness);
  if (color.contrast != null) layer.setContrast(color.contrast);
}

export class SkyEngine {
  readonly events = new Emitter<SkyEventMap>();
  private slot: HTMLElement | null = null;
  private frame = 0;
  private base: StackLayer | null = null;
  private overlays = new Map<string, StackLayer>();
  private baseColor: BaseColor = {};
  private selectJob: { cancelled: boolean; resolve: (shape: AladinSelectionShape | null) => void } | null = null;

  constructor(
    readonly A: AladinStatic,
    readonly al: Aladin,
    readonly host: HTMLDivElement,
    private readonly park: HTMLDivElement,
    private background: string,
  ) {
    for (const name of FANNED_EVENTS) {
      al.on(name, ((...args: unknown[]) => {
        // Aladin's mouseMove is in the VIEW frame (galactic after setFrame)
        // and null off the sky: always hand out ICRS (NaN off the sky).
        if (name === "mouseMove") args = [this.icrsMove(args[0] as { x: number; y: number })];
        (this.events.emit as (e: string, ...a: unknown[]) => void)(name, ...args);
        if (name === "positionChanged" || name === "zoomChanged" || name === "resizeChanged") this.scheduleView();
        // A resize clears the 2-D catalogue canvas without a repaint of the
        // overlays (3.8.2): ask for one.
        if (name === "resizeChanged") this.redraw();
      }) as (...args: never[]) => void);
    }
    host.addEventListener("contextmenu", this.onContextMenu, { capture: true });
  }

  /* placement */

  get attachedTo(): HTMLElement | null { return this.slot; }

  attach(slot: HTMLElement): void {
    if (this.slot === slot && this.host.parentElement === slot) return;
    slot.appendChild(this.host);
    this.slot = slot;
    this.scheduleView();
    this.redraw();
  }

  /** Park the host (hidden, same size). No-op when `slot` is not the current one. */
  detach(slot?: HTMLElement | null): void {
    if (slot && this.slot !== slot) return;
    const w = this.host.clientWidth, h = this.host.clientHeight;
    if (w > 0 && h > 0) {
      this.park.style.width = `${w}px`;
      this.park.style.height = `${h}px`;
    }
    this.park.appendChild(this.host);
    this.slot = null;
  }

  /** Repaint the overlays on the next frame (Aladin skips it after a
   *  resize, and after overlay edits made while the view is idle). */
  redraw(): void {
    try { this.al.view?.requestRedraw?.(); } catch { /* internal API moved */ }
  }

  size(): { width: number; height: number } {
    return { width: this.host.clientWidth, height: this.host.clientHeight };
  }

  /* view */

  getView(): SkyView {
    const [ra, dec] = this.al.getRaDec();
    const [fov] = this.al.getFov();
    return { ra, dec, fov };
  }

  setView(v: Partial<SkyView>): void {
    if (v.ra != null && v.dec != null && Number.isFinite(v.ra) && Number.isFinite(v.dec)) this.al.gotoRaDec(v.ra, v.dec);
    if (v.fov != null && Number.isFinite(v.fov) && v.fov > 0) this.al.setFoV(v.fov);
    this.scheduleView();
  }

  setProjection(p: string): void { this.al.setProjection(p); }

  setFrame(frame: "ICRS" | "Galactic"): void { this.al.setFrame(frame); }

  setTheme(theme: "light" | "dark"): void {
    try { this.al._applyTheme?.(theme); } catch { /* internal API moved: keep the old theme */ }
  }

  setGrid(enabled: boolean, color?: string): void {
    this.al.setCooGrid({ enabled, ...(color ? { color } : {}), opacity: 0.6, labelSize: 12 });
  }

  /** Screen pixel → ICRS (whatever the view frame). */
  pix2world(x: number, y: number): [number, number] | null {
    try {
      const p = this.al.pix2world(x, y, "icrs");
      return p && Number.isFinite(p[0]) && Number.isFinite(p[1]) ? [p[0], p[1]] : null;
    } catch {
      return null;
    }
  }

  /** ICRS → screen pixel (checked in 3.8.2: without a frame argument —
   *  which is broken there — `world2pix` reads ICRS in any view frame, while
   *  `pix2world` answers in the view frame unless asked for ICRS). */
  world2pix(ra: number, dec: number): [number, number] | null {
    try {
      const p = this.al.world2pix(ra, dec);
      return p && Number.isFinite(p[0]) && Number.isFinite(p[1]) ? [p[0], p[1]] : null;
    } catch {
      return null;
    }
  }

  /** Resolve a name with Sesame (Aladin's `gotoObject`) and centre on it. */
  gotoObject(name: string): Promise<[number, number]> {
    return new Promise((resolve, reject) => {
      this.al.gotoObject(name, {
        success: (raDec) => { this.scheduleView(); resolve(raDec); },
        error: (err) => reject(new Error(typeof err === "string" && err ? err : `“${name}” was not found by Sesame`)),
      });
    });
  }

  /** Start a region selection; resolves with the drawn shape (screen
   *  pixels), or null when cancelled (`cancelSelect`, a new `select`). */
  select(mode: "rect" | "circle" | "poly"): Promise<AladinSelectionShape | null> {
    this.cancelSelect();
    return new Promise((resolve) => {
      const job = { cancelled: false, resolve };
      this.selectJob = job;
      // 3.8.2's select is async (it awaits the reticle, then enters the
      // selection mode): a cancel that came first must undo it afterwards.
      const started = this.al.select(mode, (shape) => {
        if (job.cancelled) return;
        if (this.selectJob === job) this.selectJob = null;
        resolve(shape);
      });
      void Promise.resolve(started).then(() => { if (job.cancelled && !this.selectJob) this.exitSelectMode(); }, () => {});
    });
  }

  /** Leave Aladin's selection mode (back to pan) and settle the pending `select`. */
  cancelSelect(): void {
    const job = this.selectJob;
    if (!job) return;
    this.selectJob = null;
    job.cancelled = true;
    this.exitSelectMode();
    job.resolve(null);
  }

  private exitSelectMode(): void {
    try { this.al.view?.selector?.cancel?.(); } catch { /* internal API moved */ }
    try { this.al.fire?.("default"); } catch { /* internal API moved */ } // → pan mode
  }

  exportPng(): Promise<string> {
    return this.al.getViewDataURL({ format: "image/png" });
  }

  /** Value of the background HiPS under a screen pixel (FITS: a number; PNG/JPEG: RGB). */
  readPixel(x: number, y: number): unknown {
    try {
      return this.base?.layer.readPixel(x, y) ?? null;
    } catch {
      return null;
    }
  }

  /* image stack */

  /** Make the image stack exactly `base` + `overlays` (in order). A new base
   *  signature rebuilds the whole stack; overlays are diffed by key. */
  syncImageStack(base: StackEntry | null, overlays: readonly StackEntry[]): void {
    const baseChanged = (base?.signature ?? null) !== (this.base?.entry.signature ?? null);
    if (baseChanged) {
      for (const o of this.overlays.values()) this.removeLayer(o.name);
      this.overlays.clear();
      if (this.base) this.removeLayer(this.base.name);
      this.base = null;
      if (base) {
        const layer = base.make(this.A);
        this.al.setBaseImageLayer(layer);
        this.base = { entry: base, layer, name: (layer as { layer?: string }).layer ?? BASE_KEY };
        this.applyBaseColor(this.baseColor);
      }
      this.al.setBackgroundColor(this.background);
    }
    const want = new Set(overlays.map((o) => o.key));
    for (const [key, o] of this.overlays) {
      if (!want.has(key)) { this.removeLayer(o.name); this.overlays.delete(key); }
    }
    for (const entry of overlays) {
      const cur = this.overlays.get(entry.key);
      if (cur && cur.entry.signature === entry.signature) {
        if (entry.opacity != null && entry.opacity !== cur.entry.opacity) cur.layer.setOpacity(entry.opacity);
        const sig = colorSig(entry.color);
        if (entry.color && sig !== cur.colorSig) {
          try { applyLayerColor(cur.layer, entry.color); cur.colorSig = sig; } catch (err) {
            console.warn(`sky: could not recolour ${entry.key}`, err);
          }
        }
        cur.entry = entry;
        continue;
      }
      if (cur) this.removeLayer(cur.name);
      const layer = entry.make(this.A);
      const name = `ov:${entry.key}`;
      this.al.setOverlayImageLayer(layer, name);
      if (entry.opacity != null) layer.setOpacity(entry.opacity);
      // The colour went in as creation options; later changes are applied live.
      this.overlays.set(entry.key, { entry, layer, name, colorSig: colorSig(entry.color) });
    }
  }

  overlayLayer(key: string): AladinImageLayer | null {
    return this.overlays.get(key)?.layer ?? null;
  }

  hasBase(): boolean { return this.base != null; }

  applyBaseColor(color: BaseColor): void {
    this.baseColor = { ...color };
    const layer = this.base?.layer;
    if (!layer) return;
    try {
      applyLayerColor(layer, color);
    } catch (err) {
      console.warn("sky: could not apply the background colour settings", err);
    }
  }

  setBackground(color: string): void {
    this.background = color;
    this.al.setBackgroundColor(color);
  }

  /* internals */

  private icrsMove(e: { x: number; y: number } | null | undefined): MoveEvent {
    const x = Number(e?.x), y = Number(e?.y);
    const p = Number.isFinite(x) && Number.isFinite(y) ? this.pix2world(x, y) : null;
    return { ra: p ? p[0] : Number.NaN, dec: p ? p[1] : Number.NaN, x, y };
  }

  private removeLayer(name: string): void {
    try { this.al.removeImageLayer(name); } catch { /* already gone */ }
  }

  private scheduleView(): void {
    if (this.frame) return;
    const raf = typeof requestAnimationFrame === "function" ? requestAnimationFrame : (cb: FrameRequestCallback) => setTimeout(() => cb(0), 16) as unknown as number;
    this.frame = raf(() => {
      this.frame = 0;
      this.events.emit("viewChanged");
    });
  }

  private onContextMenu = (e: MouseEvent) => {
    // Captured on the host so neither the browser's menu nor Aladin's own
    // (3.8.2 opens its default menu from a catalogue-canvas listener even with
    // showContextMenu: false) appears: only the atlas's "Sky actions" menu.
    e.preventDefault();
    e.stopPropagation();
    const rect = this.host.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const p = this.pix2world(x, y);
    this.events.emit("contextMenu", { ra: p?.[0] ?? null, dec: p?.[1] ?? null, x, y, clientX: e.clientX, clientY: e.clientY });
  };
}

/* ── the singleton ───────────────────────────────────────────────────── */

let enginePromise: Promise<SkyEngine> | null = null;
let engineInstance: SkyEngine | null = null;

function makePark(size: { width: number; height: number }): HTMLDivElement {
  const park = document.createElement("div");
  park.className = "sky-engine-park";
  park.setAttribute("aria-hidden", "true");
  Object.assign(park.style, {
    position: "fixed", left: "0px", top: "0px", width: `${size.width}px`, height: `${size.height}px`,
    visibility: "hidden", pointerEvents: "none", overflow: "hidden", zIndex: "-1",
  });
  document.body.appendChild(park);
  return park;
}

async function createEngine(init: EngineInit): Promise<SkyEngine> {
  if (!hooks.hasWebGL2()) {
    throw new SkyEngineError("webgl2", "WebGL2 is not available in this browser, so the sky atlas cannot render.");
  }
  const A = await loadAladin();
  const size = init.size && init.size.width > 0 && init.size.height > 0 ? init.size : { width: 800, height: 600 };
  const park = makePark(size);
  const host = document.createElement("div");
  host.className = "sky-engine-host";
  Object.assign(host.style, { position: "absolute", inset: "0", width: "100%", height: "100%" });
  park.appendChild(host);
  const { view } = init;
  const al = A.aladin(host, {
    survey: [], // no background yet: the atlas installs its stack right after
    target: `${view.ra} ${view.dec}`, // numeric: a name here would trigger a Sesame lookup
    fov: view.fov,
    projection: view.proj,
    cooFrame: "ICRS",
    mode: init.theme,
    backgroundColor: init.background,
    hipsList: [],
    log: false,
    samp: false,
    showReticle: false,
    showCooGrid: false,
    showCooGridControl: false,
    showProjectionControl: false,
    showLayersControl: false,
    showFullscreenControl: false,
    showSettingsControl: false,
    showShareControl: false,
    showSimbadPointerControl: false,
    showColorPickerControl: false,
    showContextMenu: false,
    showFrame: false,
    showFov: false,
    showCooLocation: false,
    showZoomControl: true,
    showStatusBar: true,
  });
  // Aladin 3.8.2 opens its own default menu on right-click from two places (a
  // contextmenu listener and the mouseup after a right-button press), both
  // guarded by `aladin.contextMenu`, and showContextMenu: false covers neither.
  // Drop it so the atlas's "Sky actions" menu is the only one; right-drag
  // contrast still works (it does not use the menu).
  (al as { contextMenu?: unknown }).contextMenu = null;
  const engine = new SkyEngine(A, al, host, park, init.background);
  engineInstance = engine;
  // Dev-server debugging handle (never in a build).
  if (import.meta.env.DEV) (globalThis as { __skyEngine?: SkyEngine }).__skyEngine = engine;
  return engine;
}

/** The one engine (created on first call with `init`; later calls ignore it). */
export function getSkyEngine(init: EngineInit | (() => EngineInit)): Promise<SkyEngine> {
  if (!enginePromise) {
    const resolved = typeof init === "function" ? init() : init;
    enginePromise = createEngine(resolved);
    enginePromise.catch(() => { enginePromise = null; });
  }
  return enginePromise;
}

/** The engine if it exists already (never creates one). */
export function currentSkyEngine(): SkyEngine | null {
  return engineInstance;
}

/** Tests only: swap the library importer / WebGL2 probe, and forget the singleton. */
export function __setSkyEngineTestHooks(h: { importer?: Importer; hasWebGL2?: () => boolean }): void {
  if (h.importer) hooks.importer = h.importer;
  if (h.hasWebGL2) hooks.hasWebGL2 = h.hasWebGL2;
}

export function __resetSkyEngineForTests(): void {
  engineInstance?.host.remove();
  document.querySelectorAll(".sky-engine-park").forEach((n) => n.remove());
  enginePromise = null;
  engineInstance = null;
  libPromise = null;
  hooks.importer = () => import("aladin-lite");
  hooks.hasWebGL2 = probeWebGL2;
}
