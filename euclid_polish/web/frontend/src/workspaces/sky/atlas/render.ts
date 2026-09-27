/* Draws the atlas layers into Aladin (imperative; the pure decisions —
 * colours, LOD, what is visible — are made by the caller).
 *
 * One entry per layer: a coverage MOC (`A.MOCFromURL`, live opacity/colour
 * setters), a graphic overlay of polygons / circles (zoomed in), a catalogue
 * of centroid markers (zoomed out, or point layers), and for JWST rows their
 * observation footprints. An entry is rebuilt only when its signature (data
 * version, colour scale, opacity, LOD, marker size) changes; hidden layers
 * keep their primitives for an instant re-show. Every drawn object maps back
 * to its feature (WeakMap), so Aladin's click/hover callbacks resolve to
 * inspector targets. */
import type { Region } from "../../../sky/geometry";
import type { Lod } from "../../../sky/lod";
import type { Aladin, AladinCatalog, AladinMoc, AladinOverlay, AladinSource, AladinStatic } from "../../../sky/types";
import { colorOf, withAlpha, type ColorScale, type SkyPalette } from "./colorScale";
import type { LayerInfo, SkyFeature } from "./layerModel";

export type RenderSpec = {
  info: LayerInfo;
  features: readonly SkyFeature[];
  /** Changes whenever `features` changes (e.g. the payload's fetch time). */
  dataVersion: string | number;
  visible: boolean;
  opacity: number;
  scale: ColorScale;
  lod: Lod;
  markerScale: number;
  /** Coverage MOCs: fill the area (zoomed out), else only its outline, so the
   *  imagery inside the covered sky keeps a neutral colour (specs.ts). */
  fill?: boolean;
};

type Entry = {
  id: string;
  kind: LayerInfo["kind"];
  sig: string;
  visible: boolean;
  overlay?: AladinOverlay;
  catalog?: AladinCatalog;
  footprints?: AladinOverlay;
  moc?: AladinMoc;
};

const SHAPES: Record<string, string> = {
  circle: "circle", square: "square", diamond: "rhomb", rhomb: "rhomb", cross: "cross", plus: "plus", triangle: "triangle",
};

export const MIN_MARKER_PX = 5;
/** Polygon / circle fill alpha: result tiles vs. the large coverage shapes. */
export const SHAPE_FILL = 0.28;
export const COVERAGE_FILL = 0.1;

export function scaleSignature(s: ColorScale): string {
  return JSON.stringify(s);
}

export function specSignature(s: RenderSpec): string {
  return [s.dataVersion, scaleSignature(s.scale), s.opacity.toFixed(3), s.lod, s.markerScale.toFixed(2)].join("|");
}

export class LayerRenderer {
  private entries = new Map<string, Entry>();
  private byObject = new WeakMap<object, SkyFeature>();
  private region: AladinOverlay | null = null;
  private focus: AladinOverlay | null = null;

  constructor(
    private readonly engine: { A: AladinStatic; al: Aladin },
    private readonly palette: () => SkyPalette,
    private readonly onError?: (layer: LayerInfo, err: unknown) => void,
  ) {}

  /** Make the drawn layers match `specs` (layers absent from `specs` are removed). */
  sync(specs: readonly RenderSpec[]): void {
    const seen = new Set<string>();
    for (const s of specs) {
      seen.add(s.info.id);
      const e = this.entries.get(s.info.id);
      if (s.info.kind === "moc") { this.syncMoc(s, e); continue; }
      if (!s.visible) { if (e) this.setVisible(e, false); continue; }
      const sig = specSignature(s);
      if (e && e.sig === sig) { this.setVisible(e, true); continue; }
      if (e) this.dispose(e);
      try {
        this.entries.set(s.info.id, this.build(s, sig));
      } catch (err) {
        // One layer Aladin cannot draw must not take the atlas down.
        this.entries.delete(s.info.id);
        this.onError?.(s.info, err);
      }
    }
    for (const [id, e] of this.entries) {
      if (!seen.has(id)) { this.dispose(e); this.entries.delete(id); }
    }
    this.repaint();
  }

  /** The feature behind an Aladin object (a shape or a catalogue source). */
  featureOf(obj: unknown): SkyFeature | null {
    if (!obj || typeof obj !== "object") return null;
    const hit = this.byObject.get(obj as object);
    if (hit) return hit;
    const data = (obj as { data?: { __f?: SkyFeature } }).data;
    return data?.__f ?? null;
  }

  /** Outline the selection region (null clears it). */
  setRegion(region: Region | null): void {
    const { A, al } = this.engine;
    if (this.region) { al.removeOverlay(this.region); this.region = null; }
    if (!region) { this.repaint(); return; }
    const p = this.palette();
    const o = A.graphicOverlay({ name: "Selection", color: p.select, lineWidth: 2, lineDash: [6, 4] });
    al.addOverlay(o);
    if (region.type === "circle") o.add(A.circle(region.ra, region.dec, region.r, { color: p.select, lineWidth: 2 }));
    else o.add(A.polygon(region.points.map(([a, b]) => [a, b] as [number, number]), { color: p.select, lineWidth: 2 }));
    this.region = o;
    this.repaint();
  }

  /** Emphasise one feature (the inspected tile / source); null clears it. */
  setFocus(f: SkyFeature | null): void {
    const { A, al } = this.engine;
    if (this.focus) { al.removeOverlay(this.focus); this.focus = null; }
    if (!f) { this.repaint(); return; }
    const p = this.palette();
    const o = A.graphicOverlay({ name: "Inspected", color: p.select, lineWidth: 3 });
    al.addOverlay(o);
    if (f.polygon) o.add(A.polygon(f.polygon.map(([a, b]) => [a, b] as [number, number]), { color: p.select, lineWidth: 3 }));
    else if (f.radius) o.add(A.circle(f.ra, f.dec, f.radius, { color: p.select, lineWidth: 3 }));
    else o.add(A.circle(f.ra, f.dec, 2 / 3600, { color: p.select, lineWidth: 3 }));
    this.focus = o;
    this.repaint();
  }

  clear(): void {
    for (const e of this.entries.values()) this.dispose(e);
    this.entries.clear();
    this.setRegion(null);
    this.setFocus(null);
  }

  /* internals */

  /** Aladin repaints its overlay canvas lazily; ask for one after edits. */
  private repaint(): void {
    try { this.engine.al.view?.requestRedraw?.(); } catch { /* internal API moved */ }
  }

  private syncMoc(s: RenderSpec, e: Entry | undefined): void {
    const { A, al } = this.engine;
    const color = s.scale.type === "fixed" ? s.scale.color : this.palette().accent;
    if (!s.visible) { if (e?.moc) e.moc.hide(); if (e) e.visible = false; return; }
    if (!e?.moc) {
      if (!s.info.mocUrl) return;
      const moc = A.MOCFromURL(s.info.mocUrl, {
        name: s.info.label, color, fillColor: color, fill: s.fill !== false, perimeter: true, opacity: s.opacity, lineWidth: 1,
      });
      al.addMOC(moc);
      this.entries.set(s.info.id, { id: s.info.id, kind: "moc", sig: "", visible: true, moc });
      return;
    }
    try {
      if (e.moc.opacity !== s.opacity) e.moc.opacity = s.opacity;
      if (e.moc.color !== color) { e.moc.color = color; e.moc.fillColor = color; }
      const m = e.moc as AladinMoc & { fill?: boolean };
      const fill = s.fill !== false;
      if (m.fill !== fill) m.fill = fill;
    } catch { /* setters unavailable before the MOC is ready */ }
    e.moc.show();
    e.visible = true;
  }

  private setVisible(e: Entry, on: boolean): void {
    if (e.visible === on) return;
    for (const prim of [e.overlay, e.catalog, e.footprints, e.moc]) {
      if (!prim) continue;
      if (on) prim.show();
      else prim.hide();
    }
    e.visible = on;
  }

  private dispose(e: Entry): void {
    const { al } = this.engine;
    for (const prim of [e.overlay, e.catalog, e.footprints, e.moc]) {
      if (prim) { try { al.removeOverlay(prim); } catch { /* already gone */ } }
    }
  }

  private build(s: RenderSpec, sig: string): Entry {
    const e: Entry = { id: s.info.id, kind: s.info.kind, sig, visible: true };
    const shapes = s.lod === "shapes";
    if (s.info.kind === "points") {
      e.catalog = this.markers(s, s.info.style.shape, s.info.style.size ?? 6);
      if (shapes && s.features.some((f) => f.footprints?.length)) e.footprints = this.footprints(s);
    } else if (shapes) {
      e.overlay = this.shapes(s);
    } else {
      e.catalog = this.markers(s, "square", 6);
    }
    return e;
  }

  private markers(s: RenderSpec, shape: string | undefined, size: number): AladinCatalog {
    const { A, al } = this.engine;
    const p = this.palette();
    const cat = A.catalog({
      name: s.info.label, shape: SHAPES[shape ?? "square"] ?? "square",
      // Aladin draws its marker sprite with radius size/2 − 2: below 5 px it throws.
      sourceSize: Math.max(MIN_MARKER_PX, Math.round(size * s.markerScale)),
      color: (src: AladinSource) => String(src.data.__c ?? p.accent),
      hoverColor: p.hover, selectionColor: p.select, displayLabel: false,
    });
    al.addCatalog(cat);
    const sources: AladinSource[] = [];
    for (const f of s.features) {
      const src = A.source(f.ra, f.dec, { __f: f, __c: withAlpha(colorOf(s.scale, f.props), s.opacity) });
      this.byObject.set(src as unknown as object, f);
      sources.push(src);
    }
    cat.addSources(sources);
    return cat;
  }

  private shapes(s: RenderSpec): AladinOverlay {
    const { A, al } = this.engine;
    const p = this.palette();
    const o = A.graphicOverlay({ name: s.info.label, lineWidth: 1.5 });
    al.addOverlay(o);
    // Coverage shapes (Q1 tiles, deep fields) are large: a light tint keeps the imagery readable.
    const fillAlpha = s.info.group === "coverage" ? COVERAGE_FILL : SHAPE_FILL;
    for (const f of s.features) {
      const c = colorOf(s.scale, f.props);
      const opts = {
        color: c, fillColor: withAlpha(c, fillAlpha), fill: true, opacity: s.opacity, lineWidth: 1.5,
        hoverColor: p.hover, selectionColor: p.select,
      };
      const shape = f.polygon
        ? A.polygon(f.polygon.map(([a, b]) => [a, b] as [number, number]), opts)
        : A.circle(f.ra, f.dec, f.radius ?? 0, opts);
      if (shape && typeof shape === "object") this.byObject.set(shape as object, f);
      o.add(shape);
    }
    return o;
  }

  private footprints(s: RenderSpec): AladinOverlay {
    const { A, al } = this.engine;
    const p = this.palette();
    const o = A.graphicOverlay({ name: `${s.info.label} · footprints`, lineWidth: 1 });
    al.addOverlay(o);
    for (const f of s.features) {
      const c = colorOf(s.scale, f.props);
      for (const poly of f.footprints ?? []) {
        const shape = A.polygon(poly.map(([a, b]) => [a, b] as [number, number]), {
          color: c, fillColor: withAlpha(c, 0.15), fill: true, opacity: s.opacity, lineWidth: 1,
          hoverColor: p.hover, selectionColor: p.select,
        });
        if (shape && typeof shape === "object") this.byObject.set(shape as object, f);
        o.add(shape);
      }
    }
    return o;
  }
}
