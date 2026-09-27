/* A recording stand-in for aladin-lite (tests only; never imported by app
 * code). It implements the slice in `../types.ts`, records every call and
 * lets a test fire Aladin callbacks (`fire("zoomChanged", 12)`). */
import type {
  Aladin, AladinCatalog, AladinImageLayer, AladinMoc, AladinOverlay, AladinSelectionShape, AladinSource,
  AladinStatic,
} from "../types";

export type FakeLayer = AladinImageLayer & {
  url: string; opts: Record<string, unknown>; layer?: string; opacity?: number;
  colormap?: [string, unknown]; cuts?: [number, number]; kind: "hips" | "image";
};
export type FakeOverlay = AladinOverlay & { opts: Record<string, unknown>; items: unknown[]; visible: boolean };
export type FakeCatalog = AladinCatalog & { opts: Record<string, unknown>; sources: AladinSource[]; visible: boolean };
export type FakeMoc = AladinMoc & { url: string; opts: Record<string, unknown>; visible: boolean };
export type FakeShape = { type: "polygon" | "circle"; args: unknown[]; opts: Record<string, unknown> };

let uuid = 0;

export function makeFakeAladin(opts: { initError?: unknown } = {}) {
  const handlers: Record<string, ((...a: unknown[]) => void) | undefined> = {};
  const onCalls: string[] = [];
  const created: Record<string, unknown>[] = [];
  const view = { ra: 0, dec: 0, fov: 60, proj: "SIN", frame: "ICRS" };
  const stack: string[] = [];
  const layers = new Map<string, FakeLayer>();
  const overlays: FakeOverlay[] = [];
  const catalogs: FakeCatalog[] = [];
  const mocs: FakeMoc[] = [];
  const removed: unknown[] = [];
  const log: string[] = [];
  let background = "";
  let pendingSelect: ((shape: AladinSelectionShape, objects: unknown) => void) | null = null;
  let pendingGoto: { success?: (raDec: [number, number]) => void; error?: (e?: unknown) => void } | null = null;

  const makeLayer = (kind: "hips" | "image", url: string, o: Record<string, unknown> = {}): FakeLayer => ({
    kind, url, opts: o, name: String(o.name ?? url),
    setOpacity(v) { this.opacity = v; },
    setColormap(name, co) { this.colormap = [name, co]; },
    setCuts(a, b) { this.cuts = [a, b]; },
    setGamma() {}, setSaturation() {}, setBrightness() {}, setContrast() {},
    readPixel: (x, y) => x + y,
  });

  const al: Aladin & { hosts: HTMLElement[] } = {
    hosts: [],
    on(name, fn) { onCalls.push(name); handlers[name] = fn as ((...a: unknown[]) => void) | undefined; },
    gotoRaDec(ra, dec) { view.ra = ra; view.dec = dec; log.push(`goto ${ra} ${dec}`); },
    gotoObject(name, cb) { log.push(`gotoObject ${name}`); pendingGoto = cb ?? null; },
    setFoV(f) { view.fov = f; log.push(`fov ${f}`); },
    getFov: () => [view.fov, view.fov * 0.6],
    getRaDec: () => [view.ra, view.dec],
    pix2world: (x, y, frame) => { log.push(`pix2world ${frame ?? "view"}`); return [x / 10, y / 10]; },
    world2pix: (ra, dec) => [ra * 10, dec * 10],
    setProjection(p) { view.proj = p; log.push(`proj ${p}`); },
    setFrame(f) { view.frame = f; },
    setBaseImageLayer(l) {
      const layer = l as FakeLayer;
      layer.layer = stack[0] ?? `base-${++uuid}`;
      if (!stack.length) stack.push(layer.layer);
      layers.set(layer.layer, layer);
      return layer;
    },
    getBaseImageLayer: () => (stack[0] ? layers.get(stack[0]) ?? null : null),
    setOverlayImageLayer(l, name) {
      const layer = l as FakeLayer;
      layer.layer = name;
      if (!stack.includes(name)) stack.push(name);
      layers.set(name, layer);
      return layer;
    },
    getOverlayImageLayer: (name) => layers.get(name) ?? null,
    removeImageLayer(name) {
      layers.delete(name);
      const i = stack.indexOf(name);
      if (i >= 0) stack.splice(i, 1);
      log.push(`removeImageLayer ${name}`);
    },
    addMOC(m) { mocs.push(m as FakeMoc); },
    addOverlay(o) { overlays.push(o as FakeOverlay); },
    addCatalog(c) { catalogs.push(c as FakeCatalog); },
    removeOverlay(o) {
      removed.push(o);
      const drop = <T,>(arr: T[]) => { const i = arr.indexOf(o as T); if (i >= 0) arr.splice(i, 1); };
      drop(overlays); drop(catalogs); drop(mocs);
    },
    select(_mode, cb) { pendingSelect = cb; return undefined; },
    setCooGrid() {},
    setBackgroundColor(c) { background = c; },
    getViewDataURL: async () => "data:image/png;base64,AAAA",
    _applyTheme(t) { log.push(`theme ${t}`); },
  };

  const A: AladinStatic = {
    init: opts.initError ? Promise.reject(opts.initError) : Promise.resolve(),
    aladin(el, o) { created.push(o); al.hosts.push(el); return al; },
    HiPS: (url, o) => makeLayer("hips", url, o),
    image: (url, o) => makeLayer("image", url, o),
    MOCFromURL: (url, o = {}) => ({
      url, opts: o, name: String(o.name ?? "MOC"), opacity: Number(o.opacity ?? 1),
      color: String(o.color ?? ""), fillColor: String(o.fillColor ?? ""), visible: true,
      show() { this.visible = true; }, hide() { this.visible = false; },
    }) as FakeMoc,
    graphicOverlay: (o) => ({
      opts: o, name: String(o.name ?? "overlay"), items: [], visible: true,
      add(item) { this.items.push(item); },
      show() { this.visible = true; }, hide() { this.visible = false; },
      removeAll() { this.items = []; },
    }) as FakeOverlay,
    polygon: (v, o = {}) => ({ type: "polygon", args: [v], opts: o }) as FakeShape,
    circle: (ra, dec, r, o = {}) => ({ type: "circle", args: [ra, dec, r], opts: o }) as FakeShape,
    catalog: (o) => ({
      opts: o, name: String(o.name ?? "catalog"), sources: [], visible: true,
      addSources(s) { this.sources.push(...(Array.isArray(s) ? s : [s])); },
      show() { this.visible = true; }, hide() { this.visible = false; },
      removeAll() { this.sources = []; },
    }) as FakeCatalog,
    source: (ra, dec, data = {}) => ({ ra, dec, data }),
  };
  // Keep an unhandled rejection from failing the run when a test never awaits init.
  A.init.catch(() => {});

  return {
    A, al, view, stack, layers, overlays, catalogs, mocs, removed, created, onCalls, log,
    get background() { return background; },
    fire(name: string, ...args: unknown[]) { handlers[name]?.(...args); },
    finishSelect(shape: AladinSelectionShape) { const cb = pendingSelect; pendingSelect = null; cb?.(shape, []); },
    resolveGoto(raDec: [number, number] | null) {
      const cb = pendingGoto; pendingGoto = null;
      if (raDec) { view.ra = raDec[0]; view.dec = raDec[1]; cb?.success?.(raDec); } else cb?.error?.("not found");
    },
  };
}

export type FakeAladin = ReturnType<typeof makeFakeAladin>;
