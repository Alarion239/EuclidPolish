/* The slice of the Aladin Lite v3.8.2 API the atlas uses (the package ships
 * no types). Checked against the 3.8.2 bundle: `A.init` is a Promise
 * property, `al.on` keeps ONE callback per event and has no `off`, there is
 * no `destroy()`, and `A.polygon` mutates (pops) a closed vertex list. */

export type SkyXY = { x: number; y: number };

export type AladinEventName =
  | "objectClicked" | "objectHovered" | "objectHoveredStop" | "footprintClicked" | "click"
  | "positionChanged" | "zoomChanged" | "mouseMove" | "projectionChanged" | "cooFrameChanged"
  | "resizeChanged" | "select" | "objectsSelected";

/** A catalogue source (`A.source`); our payload rides in `data`. */
export interface AladinSource {
  ra: number;
  dec: number;
  data: Record<string, unknown>;
}

export interface AladinCatalog {
  name: string;
  addSources(sources: AladinSource[] | AladinSource): void;
  show(): void;
  hide(): void;
  removeAll?(): void;
}

export interface AladinOverlay {
  name: string;
  add(item: unknown): void;
  show(): void;
  hide(): void;
  removeAll?(): void;
}

export interface AladinMoc {
  name: string;
  opacity: number;
  color: string;
  fillColor: string;
  show(): void;
  hide(): void;
  ready?: boolean;
  contains?(ra: number, dec: number): boolean;
}

export interface AladinImageLayer {
  name?: string;
  setOpacity(opacity: number): void;
  setColormap(name: string, opts?: { stretch?: string; reversed?: boolean }): void;
  setCuts(min: number, max: number): void;
  setGamma(g: number): void;
  setSaturation(s: number): void;
  setBrightness(b: number): void;
  setContrast(c: number): void;
  readPixel(x: number, y: number): unknown;
}

/** One selection shape handed to `al.select(mode, cb)` (screen pixels). */
export interface AladinSelectionShape {
  label?: "rect" | "circle" | "polygon" | string;
  x?: number; y?: number; w?: number; h?: number; r?: number;
  vertices?: SkyXY[];
  contains(p: SkyXY): boolean;
  bbox(): { x: number; y: number; w: number; h: number };
}

export interface Aladin {
  on(event: AladinEventName, fn: ((...args: never[]) => void) | undefined): void;
  gotoRaDec(ra: number, dec: number): void;
  gotoObject(name: string, cb?: { success?: (raDec: [number, number]) => void; error?: (err?: unknown) => void }): void;
  setFoV(fov: number): void;
  getFov(): [number, number];
  getRaDec(): [number, number];
  pix2world(x: number, y: number, frame?: string): [number, number] | undefined;
  world2pix(ra: number, dec: number, frame?: string): [number, number] | null | undefined;
  setProjection(p: string): void;
  getProjectionName?(): string;
  setFrame(frame: string): void;
  setBaseImageLayer(layer: AladinImageLayer | string): unknown;
  getBaseImageLayer(): AladinImageLayer | null | undefined;
  setOverlayImageLayer(layer: AladinImageLayer | string, name: string): unknown;
  getOverlayImageLayer(name: string): AladinImageLayer | null | undefined;
  removeImageLayer(name: string): void;
  addMOC(moc: AladinMoc): void;
  addOverlay(overlay: AladinOverlay): void;
  addCatalog(catalog: AladinCatalog): void;
  removeOverlay(overlay: unknown): void;
  select(mode: "rect" | "circle" | "poly", cb: (shape: AladinSelectionShape, objects: unknown) => void): unknown;
  setCooGrid(opts: { enabled?: boolean; color?: string; opacity?: number; labelSize?: number; thickness?: number }): void;
  setBackgroundColor(color: string): void;
  getViewDataURL(opts?: { format?: string }): Promise<string>;
  getSize?(): [number, number];
  /** 3.8.2: `fire("default")` puts the view back in pan mode (ends a selection). */
  fire?(event: string, payload?: unknown): void;
  /** Internal (3.8.2): re-themes the Aladin UI; the only way on an app theme flip. */
  _applyTheme?(theme: "light" | "dark"): void;
  /** Internal (3.8.2): `requestRedraw` flags a repaint of the overlay canvas. */
  view?: { overlayLayers?: string[]; requestRedraw?(): void; selector?: { cancel?(): void } };
}

export type AladinOptions = Record<string, unknown>;

export interface AladinStatic {
  init: Promise<void>;
  aladin(el: HTMLElement, opts: AladinOptions): Aladin;
  HiPS(url: string, opts?: Record<string, unknown>): AladinImageLayer;
  image(url: string, opts: Record<string, unknown>): AladinImageLayer;
  MOCFromURL(url: string, opts?: Record<string, unknown>, ok?: (moc: AladinMoc) => void, err?: (e: unknown) => void): AladinMoc;
  graphicOverlay(opts: Record<string, unknown>): AladinOverlay;
  polygon(vertices: [number, number][], opts?: Record<string, unknown>): unknown;
  circle(ra: number, dec: number, radiusDeg: number, opts?: Record<string, unknown>): unknown;
  catalog(opts: Record<string, unknown>): AladinCatalog;
  source(ra: number, dec: number, data?: Record<string, unknown>, opts?: Record<string, unknown>): AladinSource;
}
