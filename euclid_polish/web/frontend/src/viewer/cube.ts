/* Cube transport of the image viewer: /viewer/meta + /viewer/cube requests,
 * `X-Cube-*` header parsing (contract C6), JSON error surfacing and a shared
 * LRU cache bounded in bytes with in-flight dedupe and per-consumer abort.
 *
 * Wire format (routes/viewer.py): raw little-endian Float32 (H, W, C) with
 *   X-Cube-Shape "h,w,c"        X-Cube-Bands   channel names (CSV)
 *   X-Cube-Label                 X-Cube-Pixscale arcsec / px
 *   X-Cube-Transfer-Group        X-Cube-Display-Scale (display-only factor)
 *   X-Cube-Unit ("e-", "MJy/sr", "ADU/s", "arb")
 *   X-Cube-WCS (compact JSON of the tier's celestial WCS, FITS convention)
 *   X-Cube-Index (resolved position; also for ?id= requests)
 *   X-Cube-Amp / X-Cube-Var (PCA eigen-image amplitude / variance fraction)
 *   X-Cube-Direct-RGB "1" (a JWST colour composite).
 * Errors are JSON {error} with the status code; the message is shown verbatim. */
import type { CubeLike } from "./color";
import { parseWcs, type Wcs } from "./wcs";

export type CubeRec = CubeLike & {
  key: string;
  bands: string[];
  label: string;
  /** X-Cube-Asinh (informative; the transfer uses meta.color.default_asinh). */
  asinh: number;
  pixscale: number;
  transferGroup: string;
  displayScale: number;
  unit: string;
  directRgb: boolean;
  amp: number | null;
  varexp: number;
  wcs: Wcs | null;
  index: number | null;
  bytes: number;
  /** The movie's frames mutate every tick: never cache their colour prep. */
  noCache?: boolean;
  /** Whole-cube band sums (magnitude overlay), filled lazily. */
  sums?: Record<number, number>;
};

export class ViewerError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, message: string, body: unknown = null) {
    super(message);
    this.name = "ViewerError";
    this.status = status;
    this.body = body;
  }
}

export type Params = Record<string, string>;

const query = (entries: [string, string][]) => {
  const qs = new URLSearchParams(entries).toString();
  return qs ? `?${qs}` : "";
};
const paramEntries = (p: Params | undefined): [string, string][] =>
  Object.entries(p ?? {}).filter(([, v]) => v != null).map(([k, v]) => [k, String(v)]);

export function metaUrl(collection: string, params: Params): string {
  return `/viewer/meta/${encodeURIComponent(collection)}${query(paramEntries(params))}`;
}

/** `/viewer/cube/<c>/<index>?tier=…` or, for an object id, `/viewer/cube/<c>?tier=…&id=…`. */
export function cubeUrl(collection: string, at: number | { id: string }, tier: string, params: Params, extra?: Params): string {
  const entries: [string, string][] = [["tier", tier], ...paramEntries({ ...params, ...(extra ?? {}) })];
  if (typeof at === "number") return `/viewer/cube/${encodeURIComponent(collection)}/${at}${query(entries)}`;
  return `/viewer/cube/${encodeURIComponent(collection)}${query([...entries, ["id", at.id]])}`;
}

/** Cache key: collection + tier + index + every param (so a movie member
 *  subset or a PSF warp seed never collides with the plain cube). */
export function cubeKey(collection: string, tier: string, index: number, params: Params, extra?: Params): string {
  const suffix = new URLSearchParams(paramEntries({ ...params, ...(extra ?? {}) })).toString();
  return `${collection}|${tier}:${index}${suffix ? `:${suffix}` : ""}`;
}

/** The server's error text of a failed response (JSON {error}, else the body). */
export async function readViewerError(r: Response): Promise<ViewerError> {
  let text = "";
  try { text = await r.text(); } catch { /* no body */ }
  let body: unknown = null;
  try { body = text ? JSON.parse(text) : null; } catch { /* plain text */ }
  const b = body as { error?: unknown; message?: unknown } | null;
  const message = typeof b?.error === "string" ? b.error
    : b?.error && typeof (b.error as { message?: unknown }).message === "string" ? (b.error as { message: string }).message
      : typeof b?.message === "string" ? b.message
        : text.trim() && !text.trim().startsWith("<") ? text.trim()
          : `request failed (${r.status})`;
  return new ViewerError(r.status, message, body);
}

const num = (v: string | null): number => (v == null || v.trim() === "" ? NaN : Number(v));

/** Parse one cube response. Throws a ViewerError when the body does not
 *  match the declared shape (never silently misread). */
export function parseCube(key: string, headers: Headers, buffer: ArrayBuffer): CubeRec {
  const shape = (headers.get("X-Cube-Shape") || "").split(",").map(Number);
  const [h, w, c] = shape;
  if (!(h > 0 && w > 0 && c > 0) || buffer.byteLength !== h * w * c * 4) {
    throw new ViewerError(0, `cube shape ${headers.get("X-Cube-Shape") || "?"} does not match ${buffer.byteLength} bytes`);
  }
  const amp = num(headers.get("X-Cube-Amp"));
  const scale = num(headers.get("X-Cube-Display-Scale"));
  const index = num(headers.get("X-Cube-Index"));
  return {
    key, h, w, c,
    data: new Float32Array(buffer),
    label: headers.get("X-Cube-Label") || "",
    asinh: num(headers.get("X-Cube-Asinh")) || 100,
    pixscale: num(headers.get("X-Cube-Pixscale")) || 0,
    transferGroup: headers.get("X-Cube-Transfer-Group") || "default",
    displayScale: Number.isFinite(scale) && scale > 0 ? scale : 1,
    unit: headers.get("X-Cube-Unit") || "",
    bands: (headers.get("X-Cube-Bands") || "").split(",").filter(Boolean),
    directRgb: headers.get("X-Cube-Direct-RGB") === "1",
    amp: Number.isFinite(amp) ? amp : null,
    varexp: num(headers.get("X-Cube-Var")) || 0,
    wcs: parseWcs(headers.get("X-Cube-WCS")),
    index: Number.isInteger(index) ? index : null,
    bytes: buffer.byteLength,
  };
}

export async function fetchMeta<T = unknown>(collection: string, params: Params, signal?: AbortSignal): Promise<T> {
  const r = await fetch(metaUrl(collection, params), { signal, headers: { Accept: "application/json" } });
  if (!r.ok) throw await readViewerError(r);
  return await r.json() as T;
}

const abortError = () => new DOMException("The operation was aborted.", "AbortError");

type Pending = { promise: Promise<CubeRec>; controller: AbortController; consumers: number };

/** LRU cube cache bounded in BYTES, shared by every viewer (a remounted
 *  viewer finds its cubes). Concurrent loads of one key share one request;
 *  it is aborted only when every consumer has aborted. */
export class CubeCache {
  maxBytes: number;
  private entries = new Map<string, CubeRec>();
  private pending = new Map<string, Pending>();
  bytes = 0;

  constructor(maxBytes: number) {
    this.maxBytes = maxBytes;
  }

  get size(): number { return this.entries.size; }
  has(key: string): boolean { return this.entries.has(key); }

  get(key: string): CubeRec | undefined {
    const rec = this.entries.get(key);
    if (rec) { this.entries.delete(key); this.entries.set(key, rec); }
    return rec;
  }

  set(key: string, rec: CubeRec): void {
    const old = this.entries.get(key);
    if (old) { this.bytes -= old.bytes; this.entries.delete(key); }
    this.entries.set(key, rec);
    this.bytes += rec.bytes;
    for (const [k, r] of this.entries) {
      if (this.bytes <= this.maxBytes || k === key) break;
      this.entries.delete(k);
      this.bytes -= r.bytes;
    }
  }

  delete(key: string): void {
    const rec = this.entries.get(key);
    if (rec) { this.bytes -= rec.bytes; this.entries.delete(key); }
  }

  /** Drop every cube whose key starts with `prefix` (e.g. a collection). */
  deletePrefix(prefix: string): void {
    for (const key of [...this.entries.keys()]) if (key.startsWith(prefix)) this.delete(key);
  }

  clear(): void {
    this.entries.clear();
    this.bytes = 0;
  }

  /** The cached cube, or one GET of `url` shared by concurrent callers. */
  load(key: string, url: string, signal?: AbortSignal): Promise<CubeRec> {
    const hit = this.get(key);
    if (hit) return Promise.resolve(hit);
    if (signal?.aborted) return Promise.reject(abortError());
    let p = this.pending.get(key);
    if (!p) {
      const controller = new AbortController();
      const promise = (async () => {
        const r = await fetch(url, { signal: controller.signal });
        if (!r.ok) throw await readViewerError(r);
        const rec = parseCube(key, r.headers, await r.arrayBuffer());
        if (!controller.signal.aborted) this.set(key, rec);
        return rec;
      })();
      const entry: Pending = { promise, controller, consumers: 0 };
      p = entry;
      this.pending.set(key, entry);
      promise.then(() => undefined, () => undefined).finally(() => {
        if (this.pending.get(key) === entry) this.pending.delete(key);
      });
    }
    const shared = p;
    shared.consumers++;
    if (!signal) return shared.promise;
    return new Promise<CubeRec>((resolve, reject) => {
      const onAbort = () => {
        shared.consumers--;
        if (shared.consumers <= 0) shared.controller.abort();
        reject(abortError());
      };
      signal.addEventListener("abort", onAbort, { once: true });
      shared.promise.then(
        (rec) => { signal.removeEventListener("abort", onAbort); resolve(rec); },
        (err) => { signal.removeEventListener("abort", onAbort); reject(err); },
      );
    });
  }
}

/** The cache every viewer shares (≈ 96 full 510²×4 cubes). */
export const sharedCubeCache = new CubeCache(400 * 1024 * 1024);
