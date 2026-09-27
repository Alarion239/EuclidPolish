import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  FANNED_EVENTS, SkyEngineError, __resetSkyEngineForTests, __setSkyEngineTestHooks, currentSkyEngine,
  getSkyEngine, loadAladin, type EngineInit,
} from "./engine";
import { makeFakeAladin, type FakeAladin } from "./testing/fakeAladin";

const INIT: EngineInit = {
  view: { ra: 269.7, dec: 66, fov: 360, proj: "MOL" }, theme: "light", background: "black",
  size: { width: 640, height: 400 },
};

let fake: FakeAladin;
let imports = 0;

beforeEach(() => {
  fake = makeFakeAladin();
  imports = 0;
  __setSkyEngineTestHooks({
    importer: async () => { imports++; return { default: fake.A }; },
    hasWebGL2: () => true,
  });
});
afterEach(() => __resetSkyEngineForTests());

describe("sky engine", () => {
  it("imports the library once and creates ONE instance (StrictMode double effects share it)", async () => {
    const [a, b] = await Promise.all([getSkyEngine(INIT), getSkyEngine(INIT)]);
    const c = await getSkyEngine({ ...INIT, theme: "dark" });
    expect(a).toBe(b);
    expect(a).toBe(c);
    expect(imports).toBe(1);
    expect(fake.created).toHaveLength(1);
    expect(currentSkyEngine()).toBe(a);
  });

  it("creates Aladin with log:false, numeric target, the theme and no background yet", async () => {
    await getSkyEngine(INIT);
    const opts = fake.created[0];
    expect(opts.log).toBe(false);
    expect(opts.target).toBe("269.7 66");
    expect(opts.projection).toBe("MOL");
    expect(opts.fov).toBe(360);
    expect(opts.mode).toBe("light");
    expect(opts.survey).toEqual([]);
    expect(opts.showContextMenu).toBe(false);
  });

  it("registers each Aladin event exactly once and fans it out", async () => {
    const e = await getSkyEngine(INIT);
    expect([...fake.onCalls].sort()).toEqual([...FANNED_EVENTS].sort());
    const a = vi.fn(), b = vi.fn();
    const offA = e.events.on("zoomChanged", a);
    e.events.on("zoomChanged", b);
    fake.fire("zoomChanged", 12);
    expect(a).toHaveBeenCalledWith(12);
    expect(b).toHaveBeenCalledWith(12);
    offA();
    fake.fire("zoomChanged", 3);
    expect(a).toHaveBeenCalledTimes(1);
    expect(b).toHaveBeenCalledTimes(2);
    expect(fake.onCalls.length).toBe(FANNED_EVENTS.length); // still no re-registration
  });

  it("moves its host between a slot and the hidden park (never destroyed)", async () => {
    const e = await getSkyEngine(INIT);
    const park = document.querySelector(".sky-engine-park") as HTMLElement;
    expect(park).toBeTruthy();
    expect(park.style.visibility).toBe("hidden");
    expect(e.host.parentElement).toBe(park);
    const slot = document.createElement("div");
    document.body.appendChild(slot);
    e.attach(slot);
    expect(e.host.parentElement).toBe(slot);
    expect(e.attachedTo).toBe(slot);
    // A stale detach from another (unmounted) slot is ignored.
    e.detach(document.createElement("div"));
    expect(e.host.parentElement).toBe(slot);
    e.detach(slot);
    expect(e.host.parentElement).toBe(park);
    expect(e.attachedTo).toBeNull();
    expect(document.querySelectorAll(".sky-engine-park")).toHaveLength(1);
  });

  it("reports a missing WebGL2 before downloading the library", async () => {
    __setSkyEngineTestHooks({ hasWebGL2: () => false });
    await expect(getSkyEngine(INIT)).rejects.toBeInstanceOf(SkyEngineError);
    await expect(getSkyEngine(INIT)).rejects.toMatchObject({ code: "webgl2" });
    expect(imports).toBe(0);
  });

  it("turns an A.init rejection into a SkyEngineError and allows a retry", async () => {
    const broken = makeFakeAladin({ initError: "WebGL2 not supported by your browser" });
    __setSkyEngineTestHooks({ importer: async () => ({ default: broken.A }) });
    await expect(loadAladin()).rejects.toMatchObject({ code: "webgl2" });
    __setSkyEngineTestHooks({ importer: async () => ({ default: fake.A }) });
    await expect(loadAladin()).resolves.toBe(fake.A);
  });

  it("owns the image stack: base rebuilds everything, overlays are diffed", async () => {
    const e = await getSkyEngine(INIT);
    const base = (url: string) => ({ key: "base", signature: url, make: () => fake.A.HiPS(url, {}) });
    const ov = (key: string, opacity: number) => ({
      key, signature: key, opacity, make: () => fake.A.HiPS(`https://x/${key}`, {}),
    });
    e.syncImageStack(base("q1"), [ov("nircam", 0.5)]);
    expect(fake.stack).toHaveLength(2);
    expect(e.hasBase()).toBe(true);
    const nircam = e.overlayLayer("nircam")!;
    // Opacity only → same layer object, new opacity.
    e.syncImageStack(base("q1"), [ov("nircam", 0.3)]);
    expect(e.overlayLayer("nircam")).toBe(nircam);
    expect((nircam as unknown as { opacity: number }).opacity).toBe(0.3);
    // "None" background: the overlays stay, the base goes.
    e.syncImageStack(null, [ov("nircam", 0.3)]);
    expect(e.hasBase()).toBe(false);
    expect(fake.stack).toEqual(["ov:nircam"]);
    // A new base comes first again.
    e.syncImageStack(base("dss"), [ov("nircam", 0.3), ov("miri", 1)]);
    expect(fake.layers.get(fake.stack[0])!.url).toBe("dss");
    expect(fake.stack).toHaveLength(3);
    e.syncImageStack(base("dss"), []);
    expect(fake.stack).toHaveLength(1);
  });

  it("resolves names with Sesame and reports failures", async () => {
    const e = await getSkyEngine(INIT);
    const ok = e.gotoObject("M31");
    fake.resolveGoto([10.68, 41.27]);
    await expect(ok).resolves.toEqual([10.68, 41.27]);
    const bad = e.gotoObject("nothing-here");
    fake.resolveGoto(null);
    await expect(bad).rejects.toThrow();
  });

  it("emits its own context-menu event with sky coordinates", async () => {
    const e = await getSkyEngine(INIT);
    const got = vi.fn();
    e.events.on("contextMenu", got);
    const ev = new MouseEvent("contextmenu", { bubbles: true, cancelable: true, clientX: 50, clientY: 20 });
    e.host.dispatchEvent(ev);
    expect(ev.defaultPrevented).toBe(true);
    expect(got).toHaveBeenCalledTimes(1);
    const arg = got.mock.calls[0][0];
    expect(arg.ra).toBeCloseTo(arg.x / 10);
    expect(arg.clientX).toBe(50);
  });

  it("removes Aladin's own context menu (it also opens on a right-button mouseup)", async () => {
    (fake.al as { contextMenu?: unknown }).contextMenu = { _show: vi.fn() };
    await getSkyEngine(INIT);
    expect((fake.al as { contextMenu?: unknown }).contextMenu).toBeNull();
  });

  it("keeps a right-click away from Aladin's own context menu (only ours opens)", async () => {
    const e = await getSkyEngine(INIT);
    // Aladin 3.8.2 opens its default menu from a contextmenu listener on its
    // catalogue canvas, whatever showContextMenu says.
    const canvas = document.createElement("canvas");
    e.host.appendChild(canvas);
    const aladinMenu = vi.fn();
    canvas.addEventListener("contextmenu", aladinMenu);
    const ours = vi.fn();
    e.events.on("contextMenu", ours);
    canvas.dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true, clientX: 5, clientY: 5 }));
    expect(ours).toHaveBeenCalledTimes(1);
    expect(aladinMenu).not.toHaveBeenCalled();
    canvas.remove();
  });

  it("hands out ICRS whatever the view frame (Aladin reports the view frame)", async () => {
    const e = await getSkyEngine(INIT);
    const moves = vi.fn();
    e.events.on("mouseMove", moves);
    e.setFrame("Galactic");
    fake.fire("mouseMove", { ra: 94.4, dec: 29.8, x: 2684, y: 652, frame: "Galactic" });
    expect(moves).toHaveBeenCalledWith({ ra: 268.4, dec: 65.2, x: 2684, y: 652 });
    expect(fake.log).toContain("pix2world icrs");
    // world2pix reads ICRS in every frame (no conversion).
    expect(e.world2pix(268.4, 65.2)).toEqual([2684, 652]);
    // Off the sky: NaN, never a stale position.
    fake.al.pix2world = () => undefined;
    fake.fire("mouseMove", { ra: null, dec: null, x: 1, y: 1 });
    expect(Number.isNaN(moves.mock.calls.at(-1)![0].ra)).toBe(true);
  });

  it("asks Aladin to repaint the overlays after a resize and on attach", async () => {
    const e = await getSkyEngine(INIT);
    let redraws = 0;
    (fake.al as { view?: { requestRedraw?: () => void } }).view = { requestRedraw: () => { redraws++; } };
    fake.fire("resizeChanged", 300, 200);
    const slot = document.createElement("div");
    e.attach(slot);
    expect(redraws).toBe(2);
  });

  it("a region selection resolves with the shape, or null when cancelled", async () => {
    const e = await getSkyEngine(INIT);
    const exits: string[] = [];
    (fake.al as { view?: object }).view = { selector: { cancel: () => { exits.push("cancel"); } } };
    (fake.al as { fire?: (n: string) => void }).fire = (n) => { exits.push(`fire ${n}`); };
    const shape = { label: "rect", x: 1, y: 2, w: 3, h: 4, contains: () => true, bbox: () => ({ x: 1, y: 2, w: 3, h: 4 }) };
    const first = e.select("rect");
    fake.finishSelect(shape);
    await expect(first).resolves.toBe(shape);
    await Promise.resolve();
    expect(exits).toEqual([]);                 // a finished selection needs no exit
    const second = e.select("poly");
    e.cancelSelect();
    await expect(second).resolves.toBeNull();
    expect(exits).toContain("fire default");   // back to pan mode (also after the async start)
    // A new select supersedes a pending one, and the old one's late start never exits the new one.
    const a = e.select("circle");
    exits.length = 0;
    const b = e.select("rect");
    await expect(a).resolves.toBeNull();
    await Promise.resolve(); await Promise.resolve();
    const afterStart = exits.length;
    fake.finishSelect(shape);
    await expect(b).resolves.toBe(shape);
    await Promise.resolve();
    expect(exits.length).toBe(afterStart);
    e.cancelSelect(); // nothing pending: no-op
    expect(exits.length).toBe(afterStart);
  });
});
