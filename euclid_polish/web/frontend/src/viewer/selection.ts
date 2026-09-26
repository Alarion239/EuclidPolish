/* Selection geometry of the image viewer (pure; ported from the pre-rework
 * static/cutout_viewer.js lens code).
 *
 * A Selection is ONE shared square region of the sky: a normalised centre
 * (u, v ∈ [0, 1] of each frame) plus an angular side in arcsec (or a side
 * relative to the frame when a tier has no pixel scale). Resolving it on a
 * frame gives that frame's pixel crop, so LR at 0.1″, SR at 0.05″ and JWST
 * at 0.03″ crop the same sky. The magnifier lens, the frozen crop saved to
 * results / exported as a figure, and the pan/zoom view of the main frames
 * are all Selections.
 *
 * Also here: the frame layout (which part of the image a frame draws, and
 * pointer ↔ image mapping), receptive-field tags and the collision-free
 * layout of the lens popups. */

export const LENS_MIN_ZOOM = 1.0;
export const LENS_MAX_ZOOM = 16.0;
export const LENS_DEFAULT_ZOOM = 3.0;
export const LENS_ZOOM_STEP = 1.18;
export const LENS_SIDE = 280;
export const LENS_LAYOUT_GAP = 12;
/** The lens popup canvas is drawn at 2× its CSS side. */
export const LENS_CANVAS = 560;

/** What the geometry needs to know about one frame (its tier's image). */
export type FrameGeom = {
  tier: string;
  /** Image size in the tier's pixels. */
  width: number;
  height: number;
  /** arcsec per pixel, or null when unknown. */
  pixscale: number | null;
  /** Drawn and not showing a message. */
  ready: boolean;
};

export type Selection = {
  u: number;
  v: number;
  angularSideArcsec: number | null;
  relativeSide: number | null;
  /** The tier the pointer was over when it was made. */
  sourceTier?: string;
  /** Bumped every time a crop is frozen. */
  revision?: number;
};

export type Crop = {
  x: number;
  y: number;
  side: number;
  cx: number;
  cy: number;
  angularSideArcsec: number | null;
  relativeSide: number;
};

/** The `selection` object POSTed to /viewer/results. */
export type WireSelection = {
  u: number;
  v: number;
  angular_side_arcsec: number | null;
  relative_side: number | null;
  revision: number;
  relative_fallback_safe?: true;
};

export type Viewport = { width: number; height: number; scrollX: number; scrollY: number };

export function currentViewport(): Viewport {
  if (typeof window === "undefined") return { width: 1280, height: 800, scrollX: 0, scrollY: 0 };
  return { width: window.innerWidth, height: window.innerHeight, scrollX: window.scrollX || 0, scrollY: window.scrollY || 0 };
}

/** Lens popup side (CSS px): 280, shrunk on small windows (≥ 160). */
export function lensSide(vp: Pick<Viewport, "width" | "height">): number {
  return Math.max(160, Math.min(LENS_SIDE, vp.width - 24, vp.height - 24));
}

export function validPixscale(fr: Pick<FrameGeom, "pixscale">): number | null {
  return Number.isFinite(fr.pixscale) && (fr.pixscale as number) > 0 ? fr.pixscale as number : null;
}

function requestedSourceSide(fr: FrameGeom, selection: Selection | null): number | null {
  if (!fr || !selection || !(fr.width > 0 && fr.height > 0)) return null;
  const pixscale = validPixscale(fr);
  const extent = Math.min(fr.width, fr.height);
  const ang = selection.angularSideArcsec as number;
  const rel = selection.relativeSide as number;
  const requested = pixscale && ang > 0
    ? ang / pixscale
    : rel > 0
      ? rel * extent
      : extent / LENS_DEFAULT_ZOOM;
  return Math.min(extent, Math.max(1, requested));
}

/** Resolve the shared selection onto one frame without mutating either. */
export function resolveCrop(fr: FrameGeom, selection: Selection | null): Crop | null {
  const sourceSide = requestedSourceSide(fr, selection);
  if (!(sourceSide != null && sourceSide > 0) || !selection) return null;
  const u = Math.max(0, Math.min(1, Number(selection.u)));
  const v = Math.max(0, Math.min(1, Number(selection.v)));
  const cx = Math.max(sourceSide / 2, Math.min(fr.width - sourceSide / 2, u * fr.width));
  const cy = Math.max(sourceSide / 2, Math.min(fr.height - sourceSide / 2, v * fr.height));
  const pixscale = validPixscale(fr);
  return {
    x: cx - sourceSide / 2,
    y: cy - sourceSide / 2,
    side: sourceSide,
    cx,
    cy,
    angularSideArcsec: pixscale ? sourceSide * pixscale : null,
    relativeSide: sourceSide / Math.min(fr.width, fr.height),
  };
}

const readyOnly = (frames: FrameGeom[]) => frames.filter((f) => f.ready && f.width > 1 && f.height > 1);

/** Bound the side so the crop fits every ready frame and is at least one
 *  pixel of the coarsest one. */
export function normalizeSelectionScale(selection: Selection, frames: FrameGeom[]): Selection {
  const ready = readyOnly(frames);
  const originalAngular = Number(selection.angularSideArcsec);
  let angularSideArcsec: number | null = originalAngular > 0 ? originalAngular : null;
  let relativeSide: number | null = Number(selection.relativeSide);
  if (!(relativeSide > 0)) relativeSide = null;

  if (angularSideArcsec) {
    let maximum = Infinity;
    let minimum = 0;
    for (const fr of ready) {
      const pixscale = validPixscale(fr);
      if (!pixscale) continue;
      maximum = Math.min(maximum, Math.min(fr.width, fr.height) * pixscale);
      minimum = Math.max(minimum, pixscale);
    }
    if (Number.isFinite(maximum)) {
      angularSideArcsec = Math.min(angularSideArcsec, maximum);
      if (minimum <= maximum) angularSideArcsec = Math.max(angularSideArcsec, minimum);
    }
    if (relativeSide && originalAngular > 0) {
      relativeSide *= angularSideArcsec / originalAngular;
    }
  }

  if (relativeSide) {
    let minimum = 0;
    for (const fr of ready) minimum = Math.max(minimum, 1 / Math.min(fr.width, fr.height));
    relativeSide = Math.max(minimum, Math.min(1, relativeSide));
  }
  return { ...selection, angularSideArcsec, relativeSide };
}

/** Clamp the canonical centre once to the intersection valid in every ready frame. */
export function clampSelectionToFrames(selection: Selection, frames: FrameGeom[]): Selection {
  const normalized = normalizeSelectionScale(selection, frames);
  let uMin = 0, uMax = 1, vMin = 0, vMax = 1;
  for (const fr of readyOnly(frames)) {
    const sourceSide = requestedSourceSide(fr, normalized);
    if (!(sourceSide != null && sourceSide > 0)) continue;
    // The save endpoint rounds crop sides to source pixels. Ceil here keeps
    // the shared centre valid under that later integer conversion as well.
    const boundedSide = Math.min(Math.min(fr.width, fr.height), Math.ceil(sourceSide));
    const halfU = boundedSide / (2 * fr.width);
    const halfV = boundedSide / (2 * fr.height);
    uMin = Math.max(uMin, halfU);
    uMax = Math.min(uMax, 1 - halfU);
    vMin = Math.max(vMin, halfV);
    vMax = Math.min(vMax, 1 - halfV);
  }
  const u = Math.max(uMin, Math.min(uMax, Number(normalized.u)));
  const v = Math.max(vMin, Math.min(vMax, Number(normalized.v)));
  return { ...normalized, u, v };
}

/** The selection under the pointer at normalised (u, v) of frame `fr`,
 *  keeping the previous one's size (or starting at the default lens zoom). */
export function selectionAt(
  fr: FrameGeom, u: number, v: number, previous: Selection | null, frames: FrameGeom[], side: number,
): Selection | null {
  if (!(fr.width > 0 && fr.height > 0)) return previous;
  u = Math.max(0, Math.min(1, u));
  v = Math.max(0, Math.min(1, v));
  const extent = Math.min(fr.width, fr.height);
  const pixscale = validPixscale(fr);
  let angularSideArcsec = previous && (previous.angularSideArcsec as number) > 0 ? previous.angularSideArcsec : null;
  let relativeSide = previous && (previous.relativeSide as number) > 0 ? previous.relativeSide : null;
  if (!(relativeSide != null && relativeSide > 0)) {
    const sourceSide = Math.min(extent, Math.max(1, side / LENS_DEFAULT_ZOOM));
    relativeSide = sourceSide / extent;
    if (pixscale) angularSideArcsec = sourceSide * pixscale;
  } else if (!(angularSideArcsec != null && angularSideArcsec > 0) && pixscale) {
    angularSideArcsec = Math.min(extent, relativeSide * extent) * pixscale;
  }
  return clampSelectionToFrames({ u, v, angularSideArcsec, relativeSide, sourceTier: fr.tier }, frames);
}

/** Zoom the lens by `factor` (> 1 = magnify), between 1× and 16× of `side`. */
export function zoomSelection(fr: FrameGeom, selection: Selection, factor: number, frames: FrameGeom[], side: number): Selection {
  const crop = resolveCrop(fr, selection);
  if (!crop) return selection;
  const oldZoom = side / crop.side;
  const nextZoom = Math.max(LENS_MIN_ZOOM, Math.min(LENS_MAX_ZOOM, oldZoom * factor));
  const nextSide = Math.min(Math.min(fr.width, fr.height), Math.max(1, side / nextZoom));
  const ratio = nextSide / crop.side;
  const pixscale = validPixscale(fr);
  const angularSideArcsec = pixscale
    ? nextSide * pixscale
    : (selection.angularSideArcsec as number) > 0 ? (selection.angularSideArcsec as number) * ratio : null;
  return clampSelectionToFrames({
    ...selection,
    angularSideArcsec,
    relativeSide: nextSide / Math.min(fr.width, fr.height),
    sourceTier: fr.tier,
  }, frames);
}

export type ReceptiveField = { angular_side_arcsec: number; blocks?: number; label?: string; pixels?: number };

/** The WDSR receptive fields a crop side matches (within half a wheel step,
 *  log-symmetric), closest first. */
export function receptiveFieldLabels(angularSideArcsec: number | null | undefined, fields: ReceptiveField[] | undefined): string[] {
  const list = Array.isArray(fields) ? fields : [];
  if (!(angularSideArcsec != null && angularSideArcsec > 0) || !list.length) return [];
  const tolerance = Math.log(LENS_ZOOM_STEP) / 2 + 1e-9;
  return list
    .filter((field) => {
      const target = Number(field && field.angular_side_arcsec);
      return target > 0 && Math.abs(Math.log(angularSideArcsec / target)) <= tolerance;
    })
    .sort((a, b) => Math.abs(Math.log(angularSideArcsec / a.angular_side_arcsec))
      - Math.abs(Math.log(angularSideArcsec / b.angular_side_arcsec)))
    .map((field) => String(field.label || `${field.blocks}b`));
}

/** A frozen selection in the /viewer/results wire shape. */
export function serializeSelection(sel: Selection | null): WireSelection | null {
  if (!sel) return null;
  const out: WireSelection = {
    u: sel.u,
    v: sel.v,
    angular_side_arcsec: sel.angularSideArcsec,
    relative_side: sel.relativeSide,
    revision: sel.revision ?? 0,
  };
  if (!((sel.angularSideArcsec as number) > 0)) out.relative_fallback_safe = true;
  return out;
}

// ---------------------------------------------------------------------------
// Frame layout: which image rectangle a square frame of side S draws.
// ---------------------------------------------------------------------------

export type FrameLayout = {
  /** Source rectangle (image pixels). */
  sx: number; sy: number; sw: number; sh: number;
  /** Destination rectangle (frame CSS pixels). */
  dx: number; dy: number; dw: number; dh: number;
};

/** The full image, contained (never distorted), or the view's square crop
 *  filling the frame. */
export function frameLayout(fr: Pick<FrameGeom, "width" | "height" | "pixscale" | "tier" | "ready">, view: Selection | null, S: number): FrameLayout {
  const crop = view ? resolveCrop(fr as FrameGeom, view) : null;
  if (crop) return { sx: crop.x, sy: crop.y, sw: crop.side, sh: crop.side, dx: 0, dy: 0, dw: S, dh: S };
  const scale = Math.min(S / Math.max(1, fr.width), S / Math.max(1, fr.height));
  const dw = fr.width * scale, dh = fr.height * scale;
  return { sx: 0, sy: 0, sw: fr.width, sh: fr.height, dx: (S - dw) / 2, dy: (S - dh) / 2, dw, dh };
}

/** Frame CSS point → image pixel coordinates (continuous; null outside the image). */
export function frameToImage(L: FrameLayout, X: number, Y: number): { x: number; y: number } | null {
  if (X < L.dx || X > L.dx + L.dw || Y < L.dy || Y > L.dy + L.dh || !(L.dw > 0 && L.dh > 0)) return null;
  return { x: L.sx + ((X - L.dx) / L.dw) * L.sw, y: L.sy + ((Y - L.dy) / L.dh) * L.sh };
}

/** Like frameToImage, but a point outside the drawn image is first clamped
 *  onto its edge (a profile drag released outside the frame keeps its line). */
export function frameToImageClamped(L: FrameLayout, X: number, Y: number): { x: number; y: number } | null {
  if (!(L.dw > 0 && L.dh > 0)) return null;
  return frameToImage(L, Math.max(L.dx, Math.min(L.dx + L.dw, X)), Math.max(L.dy, Math.min(L.dy + L.dh, Y)));
}

/** The viewport position of an element's padding box — where its absolutely
 *  positioned canvas and SVG overlays are drawn and what `clientWidth`
 *  measures: the border box (getBoundingClientRect) moved in by the border
 *  widths. Pointer → frame CSS coordinates must subtract this, not the
 *  border-box corner (a `.cv-frame` has a 1 px border). Reads the DOM. */
export function contentBoxOrigin(el: HTMLElement): { left: number; top: number } {
  const r = el.getBoundingClientRect();
  return { left: r.left + (el.clientLeft || 0), top: r.top + (el.clientTop || 0) };
}

/** Image pixel coordinates → frame CSS point. */
export function imageToFrame(L: FrameLayout, x: number, y: number): { x: number; y: number } {
  return { x: L.dx + ((x - L.sx) / L.sw) * L.dw, y: L.dy + ((y - L.sy) / L.sh) * L.dh };
}

/** Move the view centre by (du, dv) (normalised), clamped to every frame. */
export function panSelection(view: Selection, du: number, dv: number, frames: FrameGeom[]): Selection {
  return clampSelectionToFrames({ ...view, u: view.u + du, v: view.v + dv }, frames);
}

// ---------------------------------------------------------------------------
// Lens popup layout. Rects are in DOCUMENT coordinates unless noted.
// ---------------------------------------------------------------------------

export type Rect = { left: number; top: number; right: number; bottom: number; width: number; height: number };
export type Candidate = Rect & { corner: string | null };
/** A source crop's rect in VIEWPORT coordinates (getBoundingClientRect). */
export type ViewportRect = { left: number; top: number; width: number; height: number };
export type LensPosition = { current: Rect; sourceRect: ViewportRect | null; corner: string | null };

/** Initial popup position beside the cursor (x, y viewport px). Hover popups
 *  are fixed (viewport coordinates, clamped to the window); frozen popups are
 *  document-positioned so they follow their source tile. */
export function placeLensPopup(side: number, x: number, y: number, vp: Viewport, frozen: boolean): { left: number; top: number } {
  const pad = 12;
  const gap = 18;
  let left = x + gap;
  let top = y + gap;
  if (left + side > vp.width - pad) left = x - side - gap;
  if (top + side > vp.height - pad) top = y - side - gap;
  if (frozen) return { left: left + vp.scrollX, top: top + vp.scrollY };
  left = Math.max(pad, Math.min(left, vp.width - side - pad));
  top = Math.max(pad, Math.min(top, vp.height - side - pad));
  return { left, top };
}

export function rectOf(left: number, top: number, width: number, height: number): Rect {
  return { left, top, right: left + width, bottom: top + height, width, height };
}

const overlaps = (a: Rect, b: Rect) => a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;

function sourceDocumentRect(position: LensPosition, vp: Viewport) {
  const s = position.sourceRect;
  if (!s) return null;
  return { left: s.left + vp.scrollX, top: s.top + vp.scrollY, right: s.left + s.width + vp.scrollX, bottom: s.top + s.height + vp.scrollY };
}

/** The four corner placements around a source crop (document coordinates). */
export function cornerCandidates(
  source: { left: number; top: number; right: number; bottom: number } | null, width: number, height: number,
): Candidate[] {
  if (!source) return [];
  const gap = LENS_LAYOUT_GAP;
  return [
    { corner: "top-left", left: source.left - width - gap, top: source.top - height - gap },
    { corner: "top-right", left: source.right + gap, top: source.top - height - gap },
    { corner: "bottom-left", left: source.left - width - gap, top: source.bottom + gap },
    { corner: "bottom-right", left: source.right + gap, top: source.bottom + gap },
  ].map((c) => ({ ...c, width, height, right: c.left + width, bottom: c.top + height }));
}

function candidateFitsViewport(c: Rect, vp: Viewport): boolean {
  const left = c.left - vp.scrollX, top = c.top - vp.scrollY, pad = 12;
  return left >= pad && top >= pad && c.right - vp.scrollX <= vp.width - pad && c.bottom - vp.scrollY <= vp.height - pad;
}

function candidateVisibleArea(c: Rect, vp: Viewport): number {
  const left = c.left - vp.scrollX, top = c.top - vp.scrollY;
  const right = c.right - vp.scrollX, bottom = c.bottom - vp.scrollY;
  return Math.max(0, Math.min(vp.width, right) - Math.max(0, left)) * Math.max(0, Math.min(vp.height, bottom) - Math.max(0, top));
}

function sourceIsVisible(position: LensPosition, vp: Viewport): boolean {
  const s = position.sourceRect;
  if (!s) return false;
  return s.left < vp.width && s.left + s.width > 0 && s.top < vp.height && s.top + s.height > 0;
}

/** Pick a free corner for one popup given those already placed. */
export function chooseLensPosition(current: Rect, placed: Rect[], position: LensPosition, vp: Viewport): Candidate | null {
  const candidates = cornerCandidates(sourceDocumentRect(position, vp), current.width, current.height);
  if (!candidates.length) candidates.push({ ...current, corner: null });
  const free = candidates.filter((c) => placed.every((other) => !overlaps(c, other)));
  if (!free.length) return null;
  // Preserve an existing corner even when it overlaps surrounding page chrome.
  if (position.corner) {
    const saved = free.find((c) => c.corner === position.corner);
    if (saved) return saved;
  }
  const preferred = (items: Candidate[]) => items.slice().sort((a, b) =>
    Math.hypot(a.left - current.left, a.top - current.top) - Math.hypot(b.left - current.left, b.top - current.top))[0];
  // While the source is visible, prefer a fully visible corner (prevents a
  // lens near the screen edge from oscillating off-screen).
  if (sourceIsVisible(position, vp)) {
    const fully = free.filter((c) => candidateFitsViewport(c, vp));
    if (fully.length) return preferred(fully);
    const partly = free.filter((c) => candidateVisibleArea(c, vp) > 0);
    if (partly.length) return partly.slice().sort((a, b) => candidateVisibleArea(b, vp) - candidateVisibleArea(a, vp))[0];
  }
  // Once the source has scrolled away, keep the corner so it follows the source.
  return preferred(free);
}

/** Lay every popup out in frame order without collisions (a source-attached
 *  fallback when every corner is taken). */
export function resolveLensOverlaps(positions: LensPosition[], vp: Viewport): Candidate[] {
  const placed: Candidate[] = [];
  for (const position of positions) {
    const chosen = chooseLensPosition(position.current, placed, position, vp);
    const fallback = chooseLensPosition(position.current, [], position, vp) || { ...position.current, corner: null };
    placed.push(chosen || fallback);
  }
  return placed;
}
