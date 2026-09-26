import { afterEach, describe, expect, it, vi } from "vitest";
import { CubeCache, ViewerError, cubeKey, cubeUrl, fetchMeta, metaUrl, parseCube, readViewerError } from "./cube";

const WCS = '{"CD1_1":-8.3333333333333e-06,"CD1_2":0.0,"CD2_1":0.0,"CD2_2":8.3333333333333e-06,"CRPIX1":-3885.5,"CRPIX2":12656.5,"CRVAL1":268.4625,"CRVAL2":65.199166666667,"CTYPE1":"RA---TAN","CTYPE2":"DEC--TAN"}';

function cubeResponse(h: number, w: number, c: number, headers: Record<string, string> = {}): Response {
  const data = new Float32Array(h * w * c).map((_, i) => i);
  return new Response(data.buffer, {
    status: 200,
    headers: { "X-Cube-Shape": `${h},${w},${c}`, "Content-Type": "application/octet-stream", ...headers },
  });
}

afterEach(() => { vi.unstubAllGlobals(); });

describe("URLs and keys", () => {
  it("builds the cube URL by index or by object id, params included", () => {
    expect(cubeUrl("nexus-field", 3, "jwst", { field: "f1" })).toBe("/viewer/cube/nexus-field/3?tier=jwst&field=f1");
    expect(cubeUrl("sky", { id: "test:4" }, "hr", { subset: "test" })).toBe("/viewer/cube/sky?tier=hr&subset=test&id=test%3A4");
    expect(metaUrl("ensemble", { mode: "starfull", members: "0,3" })).toBe("/viewer/meta/ensemble?mode=starfull&members=0%2C3");
    expect(metaUrl("cutouts", {})).toBe("/viewer/meta/cutouts");
  });
  it("keys include the collection and every param (a movie subset never collides)", () => {
    expect(cubeKey("ensemble", "sr", 2, { mode: "starfull" })).toBe("ensemble|sr:2:mode=starfull");
    expect(cubeKey("ensemble", "sr", 2, { mode: "starfull", members: "1,2" }))
      .not.toBe(cubeKey("ensemble", "sr", 2, { mode: "starfull" }));
  });
});

describe("parseCube reads every X-Cube-* header", () => {
  it("shape, bands, units, WCS, index, display scale, PCA amplitude", async () => {
    const r = cubeResponse(2, 3, 1, {
      "X-Cube-Bands": "F200W", "X-Cube-Label": "NEXUS native · F200W", "X-Cube-Pixscale": "0.03",
      "X-Cube-Transfer-Group": "jwst", "X-Cube-Display-Scale": "17743.0", "X-Cube-Unit": "MJy/sr",
      "X-Cube-WCS": WCS, "X-Cube-Index": "12", "X-Cube-Amp": "0.5", "X-Cube-Var": "0.25", "X-Cube-Asinh": "100.0",
    });
    const rec = parseCube("k", r.headers, await r.arrayBuffer());
    expect(rec).toMatchObject({
      key: "k", h: 2, w: 3, c: 1, bands: ["F200W"], label: "NEXUS native · F200W", pixscale: 0.03,
      transferGroup: "jwst", displayScale: 17743, unit: "MJy/sr", index: 12, amp: 0.5, varexp: 0.25, directRgb: false,
      bytes: 24,
    });
    expect(rec.wcs?.proj).toBe("TAN");
    expect(rec.data[5]).toBe(5);
  });
  it("defaults: group 'default', display scale 1, no unit, no WCS, no amplitude", async () => {
    const r = cubeResponse(1, 1, 4);
    const rec = parseCube("k", r.headers, await r.arrayBuffer());
    expect(rec).toMatchObject({ transferGroup: "default", displayScale: 1, unit: "", wcs: null, amp: null, index: null, bands: [] });
  });
  it("a body that does not match the declared shape is an error, not a misread", async () => {
    const r = cubeResponse(2, 2, 1, { "X-Cube-Shape": "3,3,1" });
    expect(() => parseCube("k", r.headers, new ArrayBuffer(16))).toThrow(ViewerError);
  });
});

describe("server errors are surfaced verbatim", () => {
  it("JSON {error}", async () => {
    const e = await readViewerError(new Response(JSON.stringify({ error: "no combiner for this regime" }), { status: 404 }));
    expect(e).toBeInstanceOf(ViewerError);
    expect(e.status).toBe(404);
    expect(e.message).toBe("no combiner for this regime");
  });
  it("plain text and empty bodies", async () => {
    expect((await readViewerError(new Response("boom", { status: 500 }))).message).toBe("boom");
    expect((await readViewerError(new Response("", { status: 502 }))).message).toBe("request failed (502)");
  });
  it("fetchMeta rejects with the server message", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ error: "no saved JWST × Euclid fields" }), { status: 404 })));
    await expect(fetchMeta("jwst-euclid", {})).rejects.toMatchObject({ status: 404, message: "no saved JWST × Euclid fields" });
  });
});

describe("CubeCache: LRU by bytes, in-flight dedupe, abort", () => {
  it("evicts least-recently-used cubes past the byte budget", async () => {
    const cache = new CubeCache(100);
    const put = async (k: string) => {
      const r = cubeResponse(1, 10, 1);   // 40 bytes
      cache.set(k, parseCube(k, r.headers, await r.arrayBuffer()));
    };
    await put("a"); await put("b");
    cache.get("a");                        // touch a → b is the LRU
    await put("c");
    expect(cache.has("a")).toBe(true);
    expect(cache.has("b")).toBe(false);
    expect(cache.has("c")).toBe(true);
    expect(cache.bytes).toBe(80);
    cache.deletePrefix("a");
    expect(cache.has("a")).toBe(false);
    expect(cache.bytes).toBe(40);
  });
  it("one request serves concurrent loads", async () => {
    const f = vi.fn(async () => cubeResponse(1, 2, 1));
    vi.stubGlobal("fetch", f);
    const cache = new CubeCache(1e6);
    const [a, b] = await Promise.all([cache.load("x", "/viewer/cube/sky/0?tier=hr"), cache.load("x", "/viewer/cube/sky/0?tier=hr")]);
    expect(f).toHaveBeenCalledTimes(1);
    expect(a).toBe(b);
    await cache.load("x", "/viewer/cube/sky/0?tier=hr");   // cached
    expect(f).toHaveBeenCalledTimes(1);
  });
  it("aborting one consumer keeps the shared request; aborting all cancels it", async () => {
    let seen: AbortSignal | undefined;
    let release!: () => void;
    const f = vi.fn((_url: string, init?: RequestInit) => {
      seen = init?.signal ?? undefined;
      return new Promise<Response>((resolve, reject) => {
        release = () => resolve(cubeResponse(1, 1, 1));
        init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      });
    });
    vi.stubGlobal("fetch", f);
    const cache = new CubeCache(1e6);
    const c1 = new AbortController(), c2 = new AbortController();
    const p1 = cache.load("y", "/u", c1.signal);
    const p2 = cache.load("y", "/u", c2.signal);
    c1.abort();
    await expect(p1).rejects.toMatchObject({ name: "AbortError" });
    expect(seen?.aborted).toBe(false);
    release();
    await expect(p2).resolves.toMatchObject({ h: 1 });

    const c3 = new AbortController();
    const p3 = cache.load("z", "/v", c3.signal);
    c3.abort();
    await expect(p3).rejects.toMatchObject({ name: "AbortError" });
    expect(seen?.aborted).toBe(true);
    expect(cache.has("z")).toBe(false);
  });
  it("an HTTP error rejects with a ViewerError and is not cached", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ error: "SR not generated" }), { status: 404 })));
    const cache = new CubeCache(1e6);
    await expect(cache.load("e", "/e")).rejects.toMatchObject({ status: 404, message: "SR not generated" });
    expect(cache.has("e")).toBe(false);
  });
});
