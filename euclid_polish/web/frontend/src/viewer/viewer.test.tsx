/* The React engine against a mocked /viewer
 * backend (no network): loading, verbatim server errors, keyboard, ?id=
 * lookup, per-viewer override, readout through WCS, residual tiers, URL
 * state, and the JWST carousel's no-remount index follow. */
import { act, cleanup as cleanupRender, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { queryClient } from "../api/query";
import { useShortcutRegistry } from "../hooks/useShortcut";
import { useDisplay } from "../state/display";
import golden from "./__fixtures__/color_golden.json";
import type { ColorMeta } from "./color";
import { resetMetaNotes, sharedCubeCache } from "./cube";
import { ViewerController } from "./controller";
import { heatbarModel, heatbarStops } from "./export";
import { ViewerContext } from "./hooks";
import { ImageViewer, parseView, serializeView } from "./ImageViewer";
import { ProfilePanel } from "./ProfilePanel";
import type { ViewerApi, ViewerMeta } from "./types";

const COLOR = (golden as unknown as { color_meta: ColorMeta }).color_meta;
const WCS_LR = { CTYPE1: "RA---TAN", CTYPE2: "DEC--TAN", CRVAL1: 150, CRVAL2: 2, CRPIX1: 2.5, CRPIX2: 2.5, CD1_1: -0.1 / 3600, CD1_2: 0, CD2_1: 0, CD2_2: 0.1 / 3600 };
const WCS_SR = { ...WCS_LR, CRPIX1: 4.5, CRPIX2: 4.5, CD1_1: -0.05 / 3600, CD2_2: 0.05 / 3600 };

function meta(over: Partial<ViewerMeta> = {}): ViewerMeta {
  return {
    count: 3, band_names: ["VIS", "Y_E", "J_E", "H_E"], color: COLOR, default_tier: "lr",
    tiers: [{ key: "lr", label: "LR", unit: "e-" }, { key: "sr", label: "SR", unit: "e-" }, { key: "hr", label: "HR", unit: "e-" }],
    objects: [{ id: "a", label: "A", ra: 150, dec: 2 }, { id: "b", label: "B" }, { id: "c", label: "C" }],
    missing_tier_labels: { hr: "Generate HR" },
    ...over,
  };
}

function cube(h: number, w: number, c: number, fill: (i: number) => number, headers: Record<string, string> = {}): Response {
  const data = new Float32Array(h * w * c).map((_, i) => fill(i));
  return new Response(data.buffer, { status: 200, headers: { "X-Cube-Shape": `${h},${w},${c}`, "X-Cube-Bands": "VIS,Y_E,J_E,H_E", "X-Cube-Unit": "e-", ...headers } });
}

type Handler = (url: URL) => Response | Promise<Response>;
let calls: string[] = [];
function mockBackend(handler: Handler) {
  calls = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(String(input), "http://localhost");
    calls.push(url.pathname + url.search);
    return handler(url);
  }));
}

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

function defaultHandler(m = meta()): Handler {
  return (url) => {
    if (url.pathname.startsWith("/viewer/meta/")) {
      const id = url.searchParams.get("id");
      if (id) { const i = (m.objects ?? []).findIndex((o) => o.id === id); return i >= 0 ? json({ ...m, index: i }) : json({ error: `unknown object id: ${id}` }, 404); }
      return json(m);
    }
    const tier = url.searchParams.get("tier");
    const index = Number(url.pathname.split("/").pop());
    if (tier === "lr") return cube(4, 4, 4, (i) => 10 * (index + 1) + i, { "X-Cube-Label": `LR ${index}`, "X-Cube-Pixscale": "0.1", "X-Cube-WCS": JSON.stringify(WCS_LR), "X-Cube-Index": String(index) });
    if (tier === "sr") return cube(8, 8, 4, (i) => 2 * (index + 1) + i / 4, { "X-Cube-Label": `SR ${index}`, "X-Cube-Pixscale": "0.05", "X-Cube-WCS": JSON.stringify(WCS_SR) });
    if (tier === "hr") return json({ error: "HR records are not synced for this subset" }, 404);
    return json({ error: "no such tier" }, 404);
  };
}

beforeEach(() => {
  sharedCubeCache.clear();
  queryClient.clear();      // the meta is shared through the app's query cache
  resetMetaNotes();
  useDisplay.getState().reset();
});
afterEach(() => { vi.unstubAllGlobals(); });

describe("ViewerController", () => {
  it("matching surface brightness is an opt-in Display option: off, every frame renders exactly as before", async () => {
    mockBackend(defaultHandler());
    const c = new ViewerController({ collection: "test", tiers: ["lr", "sr"] });
    await c.start();
    const sr = c.s.shown.sr, lr = c.s.shown.lr;
    if (sr?.kind !== "cube" || lr?.kind !== "cube") throw new Error("not loaded");
    expect(c.areaRef()).toBe(0.1);
    // off (the default): the knee, K0 and the heat bar scale are untouched on every tier
    expect(useDisplay.getState().matchSurfaceBrightness).toBe(false);
    const knee = c.displayParams(lr.rec).knee;
    expect(c.displayParams(sr.rec)).toEqual(c.displayParams(lr.rec));
    expect(c.heatbarInfo("sr")?.scale).toBe(1);
    // on: SR (0.05″) is shown per LR (0.1″) pixel — values × 4 ≡ knee, K0 ÷ 4
    c.setDisplay({ matchSurfaceBrightness: true });
    expect(useDisplay.getState().matchSurfaceBrightness).toBe(true);    // a linked viewer edits the page-wide setting
    expect(c.displayParams(sr.rec).knee).toBeCloseTo(knee / 4);
    expect(c.displayParams(sr.rec).K0).toBeCloseTo(c.K0() / 4);
    expect(c.displayParams(lr.rec).knee).toBe(knee);                    // the reference tier is unchanged
    expect(c.heatbarInfo("sr")?.scale).toBeCloseTo(4);                  // the bar's ticks read native e⁻ per SR pixel
    expect(c.getState().knee).toBe(knee);                               // the knee the page sees is unchanged
    // the readout stays native e⁻ per pixel
    c.hoverAt("sr", 1.5, 1.5);
    expect(c.s.readout?.tiers.find((t) => t.tier === "sr")?.values?.[0]).toBe(2 + (1 * 8 + 1) * 4 / 4);
    c.setDisplay({ matchSurfaceBrightness: false });
    expect(c.displayParams(sr.rec).knee).toBe(knee);
    c.destroy();
  });

  it("loads meta and the default tier, and shows server errors verbatim", async () => {
    mockBackend(defaultHandler());
    const c = new ViewerController({ collection: "test", tiers: ["lr", "hr"] });
    await c.start();
    expect(c.s.status.lr).toEqual({ kind: "ready" });
    expect(c.s.status.hr).toEqual({ kind: "error", message: "HR records are not synced for this subset", hint: "Generate HR" });
    expect(c.s.overlay.lr).toMatch(/^LR 0 · VIS \d+\.\d\d AB$/);
    c.destroy();
  });

  it("a failed meta is the server's message", async () => {
    mockBackend(() => json({ error: "no saved JWST × Euclid fields" }, 404));
    const c = new ViewerController({ collection: "jwst-euclid" });
    await c.start();
    expect(c.s.metaError).toBe("no saved JWST × Euclid fields");
    c.destroy();
  });

  it("goToId uses meta ids (and the server's ?id= lookup) and wraps navigation", async () => {
    mockBackend(defaultHandler(meta({ objects: [{ id: "a" }] })));
    const c = new ViewerController({ collection: "test" });
    await c.start();
    expect(await c.api.goToId("a")).toBe(true);
    expect(await c.api.goToId("zzz")).toBe(false);
    c.go(-1);
    await waitFor(() => expect(c.s.index).toBe(2));
    c.api.goTo(99);
    await waitFor(() => expect(c.s.index).toBe(2));   // explicit jump clamps
    c.destroy();
  });

  it("maps the readout through the sky: LR pixel → SR pixel, RA/Dec from the tier WCS", async () => {
    mockBackend(defaultHandler());
    const c = new ViewerController({ collection: "test", tiers: ["lr", "sr"] });
    await c.start();
    c.hoverAt("lr", 2.5, 1.5);            // LR pixel (2, 1), 0-based centre
    const r = c.s.readout!;
    expect(r.tiers.map((t) => [t.tier, t.x, t.y])).toEqual([["lr", 2, 1], ["sr", 5, 3]]);
    expect(r.tiers[0].values).toEqual([10 + (1 * 4 + 2) * 4, 11 + (1 * 4 + 2) * 4, 12 + 24, 13 + 24]);
    expect(r.tiers[0].unit).toBe("e-");
    // pixel (2, 1) is FITS (3, 2): +0.5 px in x (RA decreases), −0.5 px in y from CRPIX (2.5, 2.5)
    expect(r.sky!.dec).toBeCloseTo(2 - 0.05 / 3600, 9);
    expect(r.sky!.ra).toBeCloseTo(150 - 0.05 / 3600 / Math.cos((2 * Math.PI) / 180), 9);
    c.destroy();
  });

  it("pan/zoom crops are matched across tiers through the WCS (normalised position only without one)", async () => {
    // SR's footprint is shifted by one SR pixel: the sky at LR (2, 2) is SR (5, 5), not (4, 4).
    const SHIFTED = { ...WCS_SR, CRPIX1: 5.5, CRPIX2: 5.5 };
    const base = defaultHandler();
    mockBackend((url) => {
      if (url.searchParams.get("tier") === "sr") return cube(8, 8, 4, (i) => i, { "X-Cube-Pixscale": "0.05", "X-Cube-WCS": JSON.stringify(SHIFTED) });
      return base(url);
    });
    const c = new ViewerController({ collection: "test", tiers: ["lr", "sr"] });
    await c.start();
    c.setViewSelection({ u: 0.5, v: 0.5, angularSideArcsec: 0.2, relativeSide: null, sourceTier: "lr" });
    const view = c.s.view!;
    const lr = c.cropOf("lr", view)!;
    expect([lr.cx, lr.cy, lr.side]).toEqual([2, 2, 2]);
    const sr = c.cropOf("sr", view)!;
    expect(sr.cx).toBeCloseTo(5, 9);
    expect(sr.cy).toBeCloseTo(5, 9);
    expect(sr.side).toBeCloseTo(4, 9);
    const L = c.layoutOf("sr", 240)!;
    expect([L.sx, L.sy, L.sw]).toEqual([expect.closeTo(3, 9), expect.closeTo(3, 9), expect.closeTo(4, 9)]);
    // a view made on SR maps back onto LR through the sky as well
    c.setViewSelection({ u: 5 / 8, v: 5 / 8, angularSideArcsec: 0.2, relativeSide: null, sourceTier: "sr" });
    const back = c.cropOf("lr", c.s.view!)!;
    expect(back.cx).toBeCloseTo(2, 9);
    expect(back.cy).toBeCloseTo(2, 9);
    c.destroy();
  });

  it("zoomTo centres the view on a sky position with a field of view", async () => {
    mockBackend(defaultHandler());
    const c = new ViewerController({ collection: "test", tiers: ["lr"] });
    await c.start();
    expect(c.api.zoomTo(150, 2, 0.2)).toBe(true);
    expect(c.s.view).toMatchObject({ angularSideArcsec: 0.2 });
    expect(c.s.view!.u).toBeCloseTo(0.5, 6);
    c.api.resetView();
    expect(c.s.view).toBeNull();
    c.destroy();
  });

  it("residual tiers are computed from two loaded tiers on the finer grid", async () => {
    mockBackend(defaultHandler());
    const c = new ViewerController({ collection: "test", tiers: ["lr", "sr"] });
    await c.start();
    c.addResidual("diff", "sr", "lr");
    await waitFor(() => expect(c.s.status["res:diff:sr:lr"]).toEqual({ kind: "ready" }));
    const shown = c.s.shown["res:diff:sr:lr"];
    expect(shown.kind).toBe("residual");
    expect(shown.rec.h).toBe(8);
    expect(c.s.overlay["res:diff:sr:lr"]).toBe("SR − LR");
    c.destroy();
  });

  it("a residual of tiers in different units is refused with the reason", async () => {
    mockBackend(mixedUnitHandler());
    const c = new ViewerController({ collection: "test", tiers: ["lr", "jw"] });
    await c.start();
    c.addResidual("diff", "jw", "lr");
    const key = "res:diff:jw:lr";
    await waitFor(() => expect(c.s.status[key]?.kind).toBe("error"));
    expect((c.s.status[key] as { message: string }).message).toBe("JW (8×8) and LR (4×4) are in different units (MJy/sr vs e⁻)");
    c.destroy();
  });

  it("setView is a per-viewer override that wins over the Display panel", async () => {
    mockBackend(defaultHandler());
    const c = new ViewerController({ collection: "test" });
    c.api.setView({ color: "lupton", knee: 42, gain: 2 });   // before meta: remembered
    await c.start();
    useDisplay.getState().set({ color: "H_E" });
    const st = c.api.getState();
    expect(st.color).toBe("lupton");
    expect(st.knee).toBe(42);
    expect(st.gain).toBe(2);
    // an edit where the override holds the value stays per-viewer
    c.setColor("temp");
    expect(useDisplay.getState().color).toBe("H_E");
    expect(c.api.getState().color).toBe("temp");
    c.destroy();
  });

  it("toolbar edits of a linked viewer go to the Display panel; unlinking copies it", async () => {
    mockBackend(defaultHandler());
    const c = new ViewerController({ collection: "test" });
    await c.start();
    c.setTransfer("default", { knee: 250 });
    expect(useDisplay.getState().groups.default.knee).toBe(250);
    c.setUnlinked(true);
    c.setTransfer("default", { knee: 30 });
    expect(useDisplay.getState().groups.default.knee).toBe(250);
    expect(c.transfer("default").knee).toBe(30);
    c.destroy();
  });

  it("setParams refreshes the visible cubes in place with the new params", async () => {
    mockBackend(defaultHandler());
    const c = new ViewerController({ collection: "test" });
    await c.start();
    await c.api.setParams({ psf_warp: "1", psf_warp_seed: "7" });
    expect(calls.some((u) => u.includes("tier=lr") && u.includes("psf_warp_seed=7"))).toBe(true);
    expect(c.s.status.lr).toEqual({ kind: "ready" });
    c.destroy();
  });

  it("keeps the old keys: Q–Y colour, ← →, S only for the frozen owner", async () => {
    mockBackend(defaultHandler());
    const c = new ViewerController({ collection: "test" });
    await c.start();
    c.activate();
    fireEvent.keyDown(document.body, { key: "e" });
    expect(c.settings().color).toBe("J_E");
    fireEvent.keyDown(document.body, { key: "t" });
    expect(c.settings().color).toBe("lupton");
    fireEvent.keyDown(document.body, { key: "ArrowRight" });
    await waitFor(() => expect(c.s.index).toBe(1));
    // Shift+T is the shell's theme toggle, not Lupton
    c.setColor("VIS");
    fireEvent.keyDown(document.body, { key: "T", shiftKey: true });
    expect(c.settings().color).toBe("VIS");
    // a viewer that is not hovered/focused ignores the keys
    c.deactivate();
    fireEvent.keyDown(document.body, { key: "w" });
    expect(c.settings().color).toBe("VIS");
    c.destroy();
  });

  it("Shift+letters act as the old engine's keys unless the shell or the page binds that combo", async () => {
    const base = defaultHandler();
    mockBackend(async (url) => (url.pathname === "/viewer/results" ? json({ id: "vr-2" }, 201) : base(url)));
    const c = new ViewerController({ collection: "test", tiers: ["lr"] });
    await c.start();
    c.activate();
    fireEvent.keyDown(document.body, { key: "E", shiftKey: true });
    expect(c.settings().color).toBe("J_E");                  // Shift+E = e (old engine)
    const reg = useShortcutRegistry.getState();
    reg.add({ id: 9001, combo: "Shift+E", description: "Evaluate", scope: "Page", hidden: false });
    c.setColor("VIS");
    const ev = new KeyboardEvent("keydown", { key: "E", shiftKey: true, bubbles: true, cancelable: true });
    document.body.dispatchEvent(ev);
    expect(c.settings().color).toBe("VIS");                  // the page's Shift+E wins
    expect(ev.defaultPrevented).toBe(false);
    reg.remove(9001);
    // Shift+S saves a frozen crop, like S
    c.toggleFrozen("lr", 0.5, 0.5);
    fireEvent.keyDown(document.body, { key: "S", shiftKey: true });
    await waitFor(() => expect(calls).toContain("/viewer/results"));
    c.destroy();
  });

  it("the JWST temperature chip recolours only its own viewer", async () => {
    mockBackend(defaultHandler(meta({ jwst_band_options: [{ value: "colour", label: "colour" }, { value: "temperature", label: "temperature" }] })));
    const c = new ViewerController({ collection: "test" });
    await c.start();
    try {
      c.setParam("jwst_band", "temperature");
      expect(c.settings().color).toBe("temp");
      expect(useDisplay.getState().color).toBe("VIS");        // the Display panel (other viewers) unchanged
      c.setParam("jwst_band", "colour");
      expect(c.settings().color).toBe("VIS");                  // back to following the Display panel
      expect("color" in c.s.override).toBe(false);
      await waitFor(() => expect(c.s.status.lr).toEqual({ kind: "ready" }));
    } finally {
      c.destroy();
    }
  });

  it("saves a frozen matched crop with the old /viewer/results payload", async () => {
    const base = defaultHandler();
    mockBackend(async (url) => {
      if (url.pathname === "/viewer/results") return json({ id: "vr-1", result_id: "vr-1" }, 201);
      return base(url);
    });
    const f = globalThis.fetch as unknown as ReturnType<typeof vi.fn>;
    const c = new ViewerController({ collection: "test", tiers: ["lr", "sr"] });
    await c.start();
    expect(c.saveBlockReason()).toMatch(/freeze a matched crop/);
    c.toggleFrozen("lr", 0.5, 0.5);
    expect(c.saveBlockReason()).toBe("");
    await c.api.saveCropToResults();
    const call = f.mock.calls.find(([u]) => String(u) === "/viewer/results")!;
    const posted = JSON.parse(String((call[1] as RequestInit).body)) as Record<string, unknown>;
    expect(posted).toMatchObject({
      collection: "test", index: 0, tiers: ["lr", "sr"], params: {},
      selection: { u: 0.5, v: 0.5, revision: 1, source_tier: "lr" },
      display: { color: "VIS", layout: "one-row", knee: 100, gain: 1, transfers: { default: { knee: 100, gain: 1 } } },
    });
    expect((posted.selection as { angular_side_arcsec: number }).angular_side_arcsec).toBeGreaterThan(0);
    expect(c.s.save).toEqual({ text: "Saved vr-1", tone: "saved" });
    // a residual (display-only) tier blocks saving
    c.addResidual("diff", "sr", "lr");
    expect(c.saveBlockReason()).toMatch(/display-only tier/);
    c.destroy();
  });

  it("the figure's heat bar carries the frame's stretch, black point, colormap and invert", async () => {
    mockBackend(defaultHandler());
    const c = new ViewerController({ collection: "test", tiers: ["lr"] });
    await c.start();
    const plain = c.heatbarInfo("lr")!;
    expect(heatbarModel(plain, c.K0()).parameterText).toBe("Band: VIS  ·  asinh knee: 100 e⁻");
    expect(plain.stops).toEqual(heatbarStops("gray", false, "gray"));
    c.setOverride({ stretch: "linear", colormap: "viridis" });
    const lin = c.heatbarInfo("lr")!;
    expect(lin).toMatchObject({ stretch: "linear", black: 0, unit: "e⁻", scale: 1 });
    expect(heatbarModel(lin, c.K0()).ticks.map((t) => t.label)).toEqual(["0.00", "750", "1.5k", "2.3k", "3.0k"]);
    expect(lin.stops).toEqual(heatbarStops("viridis", false, "gray"));
    c.setOverride({ invert: true });
    expect(c.heatbarInfo("lr")!.stops).toEqual(heatbarStops("viridis", true, "gray"));
    // zscale: the limits come from the frame's own values (LR 0: VIS = 10, 14, …, 70 e⁻)
    c.setOverride({ stretch: "zscale" });
    const z = c.heatbarInfo("lr")!;
    expect(z.auto!.lo).toBeGreaterThanOrEqual(10);
    expect(z.auto!.hi).toBeLessThanOrEqual(70);
    expect(z.auto!.hi).toBeGreaterThan(z.auto!.lo);
    // a colour composite keeps a luminance bar whatever the colormap
    c.setOverride({ color: "lupton", invert: false });
    expect(c.heatbarInfo("lr")!.stops).toEqual(heatbarStops("gray", false, "lupton"));
    c.destroy();
  });

  it("resolves the object FIRST, then filters the tiers by that object's own tiers (never settles on a disabled one)", async () => {
    // object "b" has LR only; the page asks for SR + HR
    const m = meta({ objects: [{ id: "a", tiers: ["lr", "sr", "hr"] }, { id: "b", tiers: ["lr"] }, { id: "c" }] });
    mockBackend(defaultHandler(m));
    const c = new ViewerController({ collection: "test", tiers: ["sr", "hr"], initialId: "b" });
    await c.start();
    expect(c.s.index).toBe(1);
    expect(c.s.tiers).toEqual(["lr"]);                     // the default tier, which "b" has
    expect(c.initialTiers).toEqual(["lr"]);
    c.destroy();
    // the same object requested by index: its tiers, not index 0's
    const d = new ViewerController({ collection: "test", tiers: ["lr", "sr"], initialIndex: 1 });
    await d.start();
    expect(d.s.tiers).toEqual(["lr"]);
    d.destroy();
  });

  it("prefetches each neighbour's own tiers only, and nothing without navigation", async () => {
    const m = meta({ objects: [{ id: "a", tiers: ["lr", "sr"] }, { id: "b", tiers: ["lr"] }, { id: "c", tiers: ["lr", "sr"] }] });
    mockBackend(defaultHandler(m));
    const c = new ViewerController({ collection: "test", tiers: ["lr", "sr"] });
    await c.start();
    await waitFor(() => expect(calls.some((u) => u.includes("/viewer/cube/test/2?"))).toBe(true));
    // neighbour 1 has no SR: it is not asked for it (no 404s for neighbour tiles)
    expect(calls.filter((u) => u.includes("/viewer/cube/test/1?") && u.includes("tier=sr"))).toEqual([]);
    expect(calls.some((u) => u.includes("/viewer/cube/test/1?") && u.includes("tier=lr"))).toBe(true);
    c.destroy();
    sharedCubeCache.clear();
    mockBackend(defaultHandler(m));
    const single = new ViewerController({ collection: "test", tiers: ["lr"], nav: false });
    await single.start();
    await new Promise((r) => setTimeout(r, 20));
    expect(calls.filter((u) => u.startsWith("/viewer/cube/")).map((u) => u.split("?")[0])).toEqual(["/viewer/cube/test/0"]);
    single.destroy();
  });

  it("a narrow viewer opens with at most two tiers (canonical order)", async () => {
    mockBackend(defaultHandler(meta({ tiers: [{ key: "lr", label: "LR" }, { key: "sr", label: "SR" }, { key: "hr", label: "HR" }], objects: [{ id: "a" }] })));
    const c = new ViewerController({ collection: "test", tiers: ["hr", "sr", "lr"], maxTiers: 2 });
    await c.start();
    expect(c.s.tiers).toEqual(["lr", "sr"]);
    expect(c.initialTiers).toEqual(["lr", "sr"]);
    c.setTiers(["lr", "sr", "hr"]);                        // the rest stay one click away
    expect(c.s.tiers).toEqual(["lr", "sr", "hr"]);
    c.destroy();
  });

  it("the whole image is drawn at a pixel-exact side, centred; the readout and markers follow that rectangle", async () => {
    mockBackend(defaultHandler());
    const c = new ViewerController({ collection: "test", tiers: ["lr", "sr"] });
    await c.start();
    // LR 4 px, SR 8 px in 30 css px frames at dpr 1: SR snaps to 3× (24 px), LR to 6× — one side for both
    const el = document.createElement("div");
    for (const tier of ["lr", "sr"]) c.registerFrame({ tier, source: document.createElement("canvas"), visible: document.createElement("canvas"), element: el, size: () => 30, redraw: () => {} });
    const L = c.layoutOf("lr")!, S = c.layoutOf("sr")!;
    expect([L.dx, L.dy, L.dw, L.dh]).toEqual([3, 3, 24, 24]);
    expect([S.dx, S.dw]).toEqual([3, 24]);
    expect(L.dw / L.sw).toBe(6);
    expect(S.dw / S.sw).toBe(3);
    // a snap that would keep less than 80 % of the frame keeps the whole frame (SR 1× = 8 of 15 px) …
    expect(c.layoutOf("sr", 15)!.dw).toBe(15);
    // … unless "Pixel-exact fit" is on
    c.setPixelExact(true);
    expect(c.layoutOf("sr", 15)!.dw).toBe(8);
    expect(c.layoutOf("sr", 15)!.dx).toBe(4);
    c.setPixelExact(false);
    c.destroy();
  });

  it("zoomBy steps through integer magnifications of the finest tier; setTool switches lens / profile / none", async () => {
    mockBackend(defaultHandler(meta({ tiers: [{ key: "lr", label: "LR" }, { key: "sr", label: "SR" }] })));
    const c = new ViewerController({ collection: "test", tiers: ["lr", "sr"] });
    await c.start();
    const el = document.createElement("div");
    for (const tier of ["lr", "sr"]) c.registerFrame({ tier, source: document.createElement("canvas"), visible: document.createElement("canvas"), element: el, size: () => 32, redraw: () => {} });
    // SR 8 px in a 32 px frame: fit 4× (8 px → 32); one step in → 6×, then 8×
    c.api.zoomBy(1.5);
    expect(32 / c.cropOf("sr", c.s.view)!.side).toBeCloseTo(6);
    c.api.zoomBy(1.5);
    expect(32 / c.cropOf("sr", c.s.view)!.side).toBeCloseTo(8);
    expect(32 / c.cropOf("lr", c.s.view)!.side).toBeCloseTo(16);   // LR (a 2× coarser grid): an integer too
    c.api.zoomBy(1 / 1.5);
    expect(32 / c.cropOf("sr", c.s.view)!.side).toBeCloseTo(6);
    c.api.zoomBy(1 / 1.5);
    expect(c.s.view).toBeNull();                                  // back to the fit
    // a fit already past the largest preset (8 px in a 700 px frame: 87.5×) still zooms in
    const big = new ViewerController({ collection: "test", tiers: ["sr"] });
    await big.start();
    big.registerFrame({ tier: "sr", source: document.createElement("canvas"), visible: document.createElement("canvas"), element: el, size: () => 700, redraw: () => {} });
    big.api.zoomBy(1.5);
    expect(big.s.view).not.toBeNull();
    const m = 700 / big.cropOf("sr", big.s.view)!.side;           // ≈ 131 (the crop rounds to whole px)
    expect(m).toBeGreaterThan(87.5 * 1.4);
    expect(m).toBeLessThan(87.5 * 1.6);
    big.destroy();
    c.api.setTool("lens");
    expect(c.getState().tool).toBe("lens");
    c.api.setTool("profile");
    expect(c.getState().tool).toBe("pan");
    expect(c.s.profileOpen).toBe(true);
    c.api.setTool("none");
    expect(c.s.profileOpen).toBe(false);
    expect(c.getState().tool).toBe("pan");
    c.destroy();
  });

  it("the disagreement movie centres on meta.morph_base_tier", async () => {
    const m = meta({ tiers: [...meta().tiers, { key: "mean", label: "Mean" }, { key: "morph", label: "movie" }], morph_base_tier: "mean", pca_n: 1 });
    mockBackend((url) => {
      if (url.pathname.startsWith("/viewer/meta/")) return json(m);
      const tier = url.searchParams.get("tier");
      if (tier === "mean" || tier === "pca0") return cube(2, 2, 4, () => 1, { "X-Cube-Amp": "0.5" });
      return defaultHandler(m)(url);
    });
    const c = new ViewerController({ collection: "ensemble", tiers: ["morph"] });
    await c.start();
    await waitFor(() => expect(c.s.status.morph).toEqual({ kind: "ready" }));
    const centre = calls.filter((u) => u.includes("/viewer/cube/ensemble/0?")).map((u) => new URLSearchParams(u.split("?")[1]).get("tier"));
    expect(centre).toContain("mean");
    expect(centre).not.toContain("sr");
    c.destroy();
  });
});

/** LR/SR (e⁻) plus a one-band JWST-like tier in MJy/sr. */
function mixedUnitHandler(): Handler {
  const m = meta({ tiers: [...meta().tiers, { key: "jw", label: "JW", unit: "MJy/sr" }] });
  const base = defaultHandler(m);
  return (url) => {
    if (url.searchParams.get("tier") === "jw") {
      return cube(8, 8, 1, (i) => 0.01 * i, { "X-Cube-Bands": "F200W", "X-Cube-Unit": "MJy/sr", "X-Cube-Label": "JW", "X-Cube-Pixscale": "0.05" });
    }
    return base(url);
  };
}

describe("<ProfilePanel>", () => {
  it("plots tiers of different units on separate axes, each labelled with its band and unit", async () => {
    mockBackend(mixedUnitHandler());
    const c = new ViewerController({ collection: "test", tiers: ["lr", "sr", "jw"] });
    await c.start();
    c.setProfile({ kind: "line", tier: "lr", p0: { x: 0.5, y: 0.5 }, p1: { x: 3.5, y: 0.5 } });
    render(<ViewerContext.Provider value={c}><ProfilePanel /></ViewerContext.Provider>);
    const figures = screen.getAllByRole("figure");
    expect(figures.map((f) => f.getAttribute("aria-label"))).toEqual(["Profile · e⁻", "Profile · MJy/sr"]);
    expect(document.body.textContent).toContain("y: VIS [e⁻]");
    expect(document.body.textContent).toContain("y: F200W [MJy/sr]");
    c.destroy();
  });

  it("one unit → one axis", async () => {
    mockBackend(defaultHandler());
    const c = new ViewerController({ collection: "test", tiers: ["lr", "sr"] });
    await c.start();
    c.setProfile({ kind: "radial", tier: "lr", c: { x: 2, y: 2 } });
    render(<ViewerContext.Provider value={c}><ProfilePanel /></ViewerContext.Provider>);
    expect(screen.getAllByRole("figure").map((f) => f.getAttribute("aria-label"))).toEqual(["Profile"]);
    expect(document.body.textContent).toContain("y: VIS [e⁻]");
    c.destroy();
  });
});

function Where() {
  const loc = useLocation();
  return <output data-testid="loc">{loc.search}</output>;
}

describe("<ImageViewer>", () => {
  it("renders frames, overlays and the error message of a missing tier", async () => {
    mockBackend(defaultHandler());
    const { container } = render(<MemoryRouter><ImageViewer collection="test" tiers={["lr", "hr"]} /></MemoryRouter>);
    expect(await screen.findByText(/^LR 0 · VIS/)).toBeTruthy();       // the label's hover detail
    expect(await screen.findByText("HR records are not synced for this subset")).toBeTruthy();
    // one light table: the bar, the frames, the readout line
    const table = container.querySelector(".cv-table")!;
    expect(table.querySelector(".cv-bar")).toBe(screen.getByRole("group", { name: "Image viewer controls" }));
    expect(table.querySelector(".cv-frames")).not.toBeNull();
    expect(table.querySelector(".cv-readout")).not.toBeNull();
    expect(container.querySelector(".cv-frame[data-tier='lr'] .cv-label__name")?.textContent).toBe("LR");
    // tier chips toggle; the navigation sits in the bar
    expect(screen.getByRole("button", { name: "LR" }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByRole("button", { name: "SR" }).getAttribute("aria-pressed")).toBe("false");
    expect(screen.getByRole("group", { name: "Navigation" })).toBeTruthy();
  });

  it("writes and restores the URL state (object id, tiers, view)", async () => {
    mockBackend(defaultHandler());
    let api: ViewerApi | null = null;
    render(<MemoryRouter initialEntries={["/x?v.t.id=b"]}>
      <ImageViewer collection="test" urlKey="t" onReady={(a) => { api = a; }} /><Where />
    </MemoryRouter>);
    await waitFor(() => expect(api?.getIndex()).toBe(1));
    act(() => { api!.setTiers(["lr", "sr"]); });
    await waitFor(() => expect(screen.getByTestId("loc").textContent).toContain("v.t.t=lr%2Csr"), { timeout: 2000 });
    expect(screen.getByTestId("loc").textContent).toContain("v.t.id=b");
  });

  it("leaves the URL alone for the default state (mount-time object and tiers)", async () => {
    mockBackend(defaultHandler());
    let api: ViewerApi | null = null;
    render(<MemoryRouter initialEntries={["/x?keep=1"]}>
      <ImageViewer collection="test" urlKey="t" tiers={["lr", "sr"]} onReady={(a) => { api = a; }} /><Where />
    </MemoryRouter>);
    await screen.findByText(/^LR 0/);
    await new Promise((r) => setTimeout(r, 450));            // past the 200 ms URL flush
    expect(screen.getByTestId("loc").textContent).toBe("?keep=1");
    // navigating writes the object id; the tiers stay implicit while unchanged
    act(() => { api!.goTo(2); });
    await waitFor(() => expect(screen.getByTestId("loc").textContent).toContain("v.t.id=c"), { timeout: 2000 });
    expect(screen.getByTestId("loc").textContent).not.toContain("v.t.t=");
    // back to the mount-time object: the id is dropped again
    act(() => { api!.goTo(0); });
    await waitFor(() => expect(screen.getByTestId("loc").textContent).toBe("?keep=1"), { timeout: 2000 });
  });

  it("never writes a tier set the object has none of to the URL (the requested tiers are unavailable)", async () => {
    mockBackend(defaultHandler(meta({ objects: [{ id: "a", tiers: ["lr"] }, { id: "b" }, { id: "c" }] })));
    render(<MemoryRouter initialEntries={["/x?keep=1"]}>
      <ImageViewer collection="test" urlKey="t" tiers={["sr"]} /><Where />
    </MemoryRouter>);
    await screen.findByText(/^LR 0/);
    await new Promise((r) => setTimeout(r, 450));
    expect(screen.getByTestId("loc").textContent).toBe("?keep=1");      // no v.t.t=sr (nor =lr)
  });

  it("a tier the object does not have is dimmed, not clickable, and says why", async () => {
    mockBackend(defaultHandler(meta({ objects: [{ id: "a", tiers: ["lr", "sr"] }, { id: "b" }, { id: "c" }] })));
    render(<MemoryRouter><ImageViewer collection="test" /></MemoryRouter>);
    await screen.findByText(/^LR 0/);
    const hr = screen.getByRole("button", { name: "HR: Generate HR" });             // the backend's reason
    expect(hr.getAttribute("aria-disabled")).toBe("true");
    expect(hr.getAttribute("aria-pressed")).toBe("false");
    fireEvent.click(hr);
    expect(hr.getAttribute("aria-pressed")).toBe("false");
    cleanupRender();
    queryClient.clear(); resetMetaNotes(); sharedCubeCache.clear();
    mockBackend(defaultHandler(meta({ missing_tier_labels: {}, objects: [{ id: "a", tiers: ["lr"] }, { id: "b" }, { id: "c" }] })));
    render(<MemoryRouter><ImageViewer collection="test" /></MemoryRouter>);
    await screen.findByText(/^LR 0/);
    expect(screen.getByRole("button", { name: "SR: Not available for this object" })).toBeTruthy();
  });

  it("a colour the page sets as the viewer loads is its default, not URL state", async () => {
    mockBackend(defaultHandler());
    let api: ViewerApi | null = null;
    render(<MemoryRouter initialEntries={["/x"]}>
      <ImageViewer collection="test" urlKey="t" onReady={(a) => { api = a; a?.setView({ color: "lupton" }); }} /><Where />
    </MemoryRouter>);
    await screen.findByText(/^LR 0/);
    await new Promise((r) => setTimeout(r, 450));
    expect(screen.getByTestId("loc").textContent).toBe("");
    act(() => { api!.setView({ color: "H_E" }); });
    await waitFor(() => expect(screen.getByTestId("loc").textContent).toBe("?v.t.c=H_E"), { timeout: 2000 });
  });

  it("a tier with no coverage shows a quiet caption; a sparse corner of data is painted with one at its foot", async () => {
    const base = defaultHandler();
    // SR: every pixel blank; LR: an 850 × 850-like sparse cube (1200 finite of 160 000 values)
    mockBackend((url) => {
      const tier = url.searchParams.get("tier");
      if (tier === "sr") return cube(8, 8, 4, () => NaN, { "X-Cube-Label": "SR 0", "X-Cube-Pixscale": "0.05" });
      if (tier === "lr") return cube(200, 200, 4, (i) => (i < 1200 ? 5 : NaN), { "X-Cube-Label": "LR 0", "X-Cube-Pixscale": "0.1" });
      return base(url);
    });
    const { container } = render(<MemoryRouter><ImageViewer collection="test" tiers={["lr", "sr"]} /></MemoryRouter>);
    expect(await screen.findByText("No SR data here")).toBeTruthy();
    expect(container.querySelector(".cv-frame[data-tier='sr'] .cv-msg--quiet")).not.toBeNull();
    expect(await screen.findByText("Little LR data here")).toBeTruthy();
    const foot = container.querySelector(".cv-frame[data-tier='lr'] .cv-msg--foot");
    expect(foot?.textContent).toContain("Only 1,200 of 160,000 values have data");
    expect(container.querySelector(".cv-frame[data-tier='lr'] .cv-msg--quiet:not(.cv-msg--foot)")).toBeNull();
  });

  it("round-trips the view in the URL codec", () => {
    const v = { u: 0.25, v: 0.75, angularSideArcsec: 3.2, relativeSide: null };
    expect(parseView(serializeView(v))).toEqual(v);
    expect(parseView("0.5,0.5,r0.25")).toEqual({ u: 0.5, v: 0.5, angularSideArcsec: null, relativeSide: 0.25 });
    expect(parseView("garbage")).toBeNull();
    expect(serializeView(null)).toBe("");
  });
});

describe("<ImageViewer> control bar, keys and focus mode", () => {
  const rootOf = (c: HTMLElement) => c.querySelector<HTMLElement>(".cv-root")!;

  it("full bar: tiers, bands, Display, compare, tools, zoom, navigation, layout, export, focus", async () => {
    mockBackend(defaultHandler());
    render(<MemoryRouter><ImageViewer collection="test" tiers={["lr", "sr"]} /></MemoryRouter>);
    await screen.findByText(/^LR 0/);
    const bar = screen.getByRole("group", { name: "Image viewer controls" });
    for (const name of ["Display settings for this viewer", "Tools: lens, profiles, residuals, playback", "Zoom in", "Previous", "Next",
      "Run through the objects", "Arrange the frames, focus mode, full screen", "Export: PNG, figure, video, save the crop", "Open large"]) {
      expect(bar.contains(screen.getByRole("button", { name }))).toBe(true);
    }
    // band chips use the short NISP names; Q–Y are their keys
    const bands = screen.getByRole("radiogroup", { name: "Band or colour" });
    expect(Array.from(bands.querySelectorAll("button")).map((b) => b.textContent)).toEqual(["VIS", "Y", "J", "H", "Lupton", "Temp"]);
    expect(screen.getByRole("radiogroup", { name: "Compare" })).toBeTruthy();
    // the counter counts from 1 over the object count; a typed position jumps there
    const pos = screen.getByRole("textbox", { name: "Object number (1 to 3)" }) as HTMLInputElement;
    expect(pos.value).toBe("1");
    expect(screen.getByRole("group", { name: "Navigation" }).textContent).toContain("/ 3");
    fireEvent.change(pos, { target: { value: "3" } });
    fireEvent.keyDown(pos, { key: "Enter" });
    await waitFor(() => expect(pos.value).toBe("3"));
  });

  it("export stays in a full bar without navigation", async () => {
    mockBackend(defaultHandler());
    render(<MemoryRouter><ImageViewer collection="test" nav={false} /></MemoryRouter>);
    await screen.findByText(/^LR 0/);
    expect(screen.queryByRole("group", { name: "Navigation" })).toBeNull();
    expect(screen.getByRole("button", { name: /^Export/ })).toBeTruthy();
  });

  it("the Display row sits under the bar (no dock beside the frames), keeps the keys to its controls, and Esc closes it", async () => {
    mockBackend(defaultHandler());
    let api: ViewerApi | null = null;
    const { container } = render(<MemoryRouter><ImageViewer collection="test" tiers={["lr", "sr"]} onReady={(a) => { api = a; }} /></MemoryRouter>);
    await screen.findByText(/^LR 0/);
    const toggle = screen.getByRole("button", { name: "Display settings for this viewer" });
    expect(toggle.getAttribute("aria-pressed")).toBe("false");
    fireEvent.click(toggle);
    const row = await screen.findByRole("group", { name: "Display settings for this viewer" });
    expect(toggle.getAttribute("aria-pressed")).toBe("true");
    // a row of the light table between the bar and the frames: it takes no width from them
    const table = container.querySelector(".cv-table")!;
    expect(row.parentElement).toBe(table);
    expect(row.previousElementSibling).toBe(table.querySelector(".cv-bar"));
    expect(row.nextElementSibling).toBe(table.querySelector(".cv-body"));
    expect(container.querySelector(".cv-body")!.children.length).toBe(1);   // the frames only
    // the three most-used controls in the row: knee, brightness, stretch
    const knee = screen.getByRole("slider", { name: "knee" });
    const stretch = screen.getByRole("combobox", { name: "Stretch" });
    expect(row.contains(knee) && row.contains(stretch) && row.contains(screen.getByRole("slider", { name: "brightness" }))).toBe(true);
    expect(knee.compareDocumentPosition(stretch) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    // the rest waits behind "More display settings" (a popover over the surround)
    expect(screen.queryByRole("combobox", { name: "Colormap" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "More display settings" }));
    const more = await screen.findByRole("dialog", { name: "More display settings" });
    expect(row.contains(more)).toBe(false);
    expect(screen.getByRole("combobox", { name: "Colormap" })).toBeTruthy();
    expect(screen.getByRole("textbox", { name: "black point (e⁻)" })).toBeTruthy();
    expect(screen.getByRole("switch", { name: "Use the page-wide display settings" })).toBeTruthy();
    expect(screen.getByRole("switch", { name: "Match surface brightness across pixel scales" })).toBeTruthy();
    // two columns (the popover is wide and short, so it covers little of the frames)
    expect(more.querySelectorAll(".cv-more-cols > .cv-more-col").length).toBe(2);
    // the histogram is a page of its own (it replaces the settings instead of growing the popover)
    expect(document.querySelector(".cv-hist")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Histogram and cuts" }));
    expect(document.querySelector(".cv-hist")).not.toBeNull();
    expect(screen.queryByRole("combobox", { name: "Colormap" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Back to the display settings" }));
    expect(document.querySelector(".cv-hist")).toBeNull();
    expect(screen.getByRole("combobox", { name: "Colormap" })).toBeTruthy();
    fireEvent.keyDown(more, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "More display settings" })).toBeNull());
    // an arrow on a row slider moves the slider, not the object
    fireEvent.mouseEnter(rootOf(container));
    const thumb = screen.getByRole("slider", { name: "knee" });
    thumb.focus();
    fireEvent.keyDown(thumb, { key: "ArrowRight" });
    expect(api!.getIndex()).toBe(0);
    fireEvent.keyDown(thumb, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("group", { name: "Display settings for this viewer" })).toBeNull());
  });

  it("matching surface brightness: the row names it and the knee reads per the coarsest pixel", async () => {
    mockBackend(defaultHandler());
    render(<MemoryRouter><ImageViewer collection="test" tiers={["lr", "sr"]} /></MemoryRouter>);
    await screen.findByText(/^SR 0/);
    fireEvent.click(screen.getByRole("button", { name: "Display settings for this viewer" }));
    expect(await screen.findByRole("textbox", { name: "knee (e⁻)" })).toBeTruthy();
    expect(screen.queryByText("Surface brightness matched")).toBeNull();
    act(() => { useDisplay.getState().set({ matchSurfaceBrightness: true }); });
    expect(await screen.findByText("Surface brightness matched")).toBeTruthy();
    expect(screen.getByRole("textbox", { name: "knee (e⁻ per 0.1″ px)" })).toBeTruthy();
  });

  it("compact bar: a basic Display row (knee, brightness), no tools or export menus; a lens toggle", async () => {
    mockBackend(defaultHandler());
    render(<MemoryRouter><ImageViewer collection="test" tiers={["lr", "sr"]} toolbar="compact" /></MemoryRouter>);
    await screen.findByText(/^LR 0/);
    fireEvent.click(screen.getByRole("button", { name: "Display settings for this viewer" }));
    expect(await screen.findByRole("textbox", { name: "knee (e⁻)" })).toBeTruthy();
    expect(screen.getByRole("slider", { name: "brightness" })).toBeTruthy();
    expect(screen.queryByRole("combobox", { name: "Stretch" })).toBeNull();     // in More display settings
    expect(screen.queryByRole("button", { name: "Histogram and cuts" })).toBeNull();
    expect(screen.queryByRole("button", { name: /^Tools/ })).toBeNull();
    expect(screen.queryByRole("button", { name: /^Export/ })).toBeNull();
    const lens = screen.getByRole("button", { name: "Magnifier lens" });
    fireEvent.click(lens);
    expect(lens.getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByRole("button", { name: "Next" })).toBeTruthy();
  });

  it("toolbar none: no bar, or the navigation alone with nav", async () => {
    mockBackend(defaultHandler());
    const { unmount } = render(<MemoryRouter><ImageViewer collection="test" toolbar="none" nav={false} /></MemoryRouter>);
    await screen.findByText(/^LR 0/);
    expect(screen.queryByRole("group", { name: "Image viewer controls" })).toBeNull();
    unmount();
    render(<MemoryRouter><ImageViewer collection="test" toolbar="none" /></MemoryRouter>);
    await screen.findByText(/^LR 0/);
    expect(screen.getByRole("group", { name: "Image viewer controls" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Next" })).toBeTruthy();
    expect(screen.queryByRole("group", { name: "Tiers" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Display settings for this viewer" })).toBeNull();
  });

  it("the old keys still answer in the hovered viewer: Q–Y, arrows, Space; the bar navigates", async () => {
    mockBackend(defaultHandler());
    let api: ViewerApi | null = null;
    const { container } = render(<MemoryRouter><ImageViewer collection="test" onReady={(a) => { api = a; }} /></MemoryRouter>);
    await screen.findByText(/^LR 0/);
    fireEvent.mouseEnter(rootOf(container));
    fireEvent.keyDown(document.body, { key: "w" });
    expect(useDisplay.getState().color).toBe("Y_E");
    expect(screen.getByRole("radio", { name: "Y" }).getAttribute("aria-checked")).toBe("true");
    fireEvent.keyDown(document.body, { key: "ArrowRight" });
    await waitFor(() => expect(api!.getIndex()).toBe(1));
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    await waitFor(() => expect(api!.getIndex()).toBe(2));
    fireEvent.keyDown(document.body, { key: " " });
    expect(api!.getState()).toBeTruthy();
    expect(screen.getByRole("button", { name: "Stop the run-through" }).getAttribute("aria-pressed")).toBe("true");
    fireEvent.keyDown(document.body, { key: " " });
    expect(screen.getByRole("button", { name: "Run through the objects" })).toBeTruthy();
    // after the pointer leaves, the keys belong to the page again
    fireEvent.mouseLeave(rootOf(container));
    fireEvent.keyDown(document.body, { key: "e" });
    expect(useDisplay.getState().color).toBe("Y_E");
  });

  it("L toggles the lens and reports it through onState (a page's shared lens button follows)", async () => {
    mockBackend(defaultHandler());
    const tools: string[] = [];
    const { container } = render(<MemoryRouter><ImageViewer collection="test" onState={(st) => { tools.push(st.tool); }} /></MemoryRouter>);
    await screen.findByText(/^LR 0/);
    fireEvent.mouseEnter(rootOf(container));
    fireEvent.keyDown(document.body, { key: "l" });
    expect(tools[tools.length - 1]).toBe("lens");
    fireEvent.keyDown(document.body, { key: "l" });
    expect(tools[tools.length - 1]).toBe("pan");
  });

  it("F enters focus mode and Esc (or the button) leaves it", async () => {
    mockBackend(defaultHandler());
    const { container } = render(<MemoryRouter><ImageViewer collection="test" /></MemoryRouter>);
    await screen.findByText(/^LR 0/);
    const root = rootOf(container);
    fireEvent.mouseEnter(root);
    fireEvent.keyDown(document.body, { key: "f" });
    await waitFor(() => expect(root.hasAttribute("data-focus")).toBe(true));
    expect(root.style.position === "" || root.style.top !== "").toBe(true);
    // the keys stay with the focused viewer even when the pointer leaves it
    fireEvent.mouseLeave(root);
    fireEvent.keyDown(document.body, { key: "Escape" });
    await waitFor(() => expect(root.hasAttribute("data-focus")).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: "Open large" }));
    await waitFor(() => expect(root.hasAttribute("data-focus")).toBe(true));
    fireEvent.click(screen.getByRole("button", { name: "Leave focus mode" }));
    await waitFor(() => expect(root.hasAttribute("data-focus")).toBe(false));
  });

  it("keys work for a viewer inside a dialog (the inspector sheet), and Esc there is the viewer's first", async () => {
    mockBackend(defaultHandler());
    const { container } = render(<MemoryRouter><div role="dialog" aria-label="Inspector"><ImageViewer collection="test" /></div>
      <div role="dialog" aria-label="Other"><button type="button">x</button></div></MemoryRouter>);
    await screen.findByText(/^LR 0/);
    const root = rootOf(container);
    fireEvent.mouseEnter(root);
    root.focus();
    fireEvent.keyDown(root, { key: "f" });
    await waitFor(() => expect(root.hasAttribute("data-focus")).toBe(true));
    const esc = new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true });
    root.dispatchEvent(esc);
    expect(esc.defaultPrevented).toBe(true);                      // a capture-phase dialog sees it taken
    await waitFor(() => expect(root.hasAttribute("data-focus")).toBe(false));
    // a key typed in ANOTHER dialog is not the viewer's
    fireEvent.keyDown(screen.getByRole("button", { name: "x" }), { key: "f" });
    expect(root.hasAttribute("data-focus")).toBe(false);
  });

  it("a plain g leaves the next key to the shell's g-sequence", async () => {
    mockBackend(defaultHandler());
    const { container } = render(<MemoryRouter><ImageViewer collection="test" /></MemoryRouter>);
    await screen.findByText(/^LR 0/);
    fireEvent.mouseEnter(rootOf(container));
    fireEvent.keyDown(document.body, { key: "g" });
    fireEvent.keyDown(document.body, { key: "e" });            // "g e" = go to Ensemble, not J
    expect(useDisplay.getState().color).toBe("VIS");
  });

  it("Display row: knees span 0.1–1e4 on a log slider, each transfer group in its own unit (one group at a time)", async () => {
    const m = meta({
      transfer_groups: ["euclid", "jwst"],
      tiers: [{ key: "lr", label: "LR", unit: "e-" }, { key: "jw", label: "JWST", unit: "MJy/sr" }],
    });
    const base = defaultHandler(m);
    mockBackend((url) => {
      const tier = url.searchParams.get("tier");
      if (tier === "jw") return cube(8, 8, 1, (i) => 0.01 * i, { "X-Cube-Bands": "F200W", "X-Cube-Unit": "MJy/sr", "X-Cube-Label": "JWST", "X-Cube-Transfer-Group": "jwst", "X-Cube-Display-Scale": "4000" });
      if (tier === "lr") return cube(4, 4, 4, (i) => i, { "X-Cube-Label": "LR 0", "X-Cube-Transfer-Group": "euclid" });
      return base(url);
    });
    render(<MemoryRouter><ImageViewer collection="test" tiers={["lr", "jw"]} /></MemoryRouter>);
    await screen.findByText(/^LR 0/);
    fireEvent.click(screen.getByRole("button", { name: "Display settings for this viewer" }));
    const euclid = await screen.findByRole("textbox", { name: "Euclid knee (e⁻)" });
    expect((euclid as HTMLInputElement).value).toBe("100");
    // the row edits one transfer group at a time: pick JWST, then back to Euclid
    fireEvent.click(screen.getByRole("radio", { name: "JWST" }));
    const jwst = await screen.findByRole("textbox", { name: "JWST knee (MJy/sr)" });
    expect((jwst as HTMLInputElement).value).toBe("0.025");          // 100 / display scale 4000
    fireEvent.click(screen.getByRole("radio", { name: "Euclid" }));
    await screen.findByRole("textbox", { name: "Euclid knee (e⁻)" });
    // typed values, and the slider's ends
    const euclid2 = screen.getByRole("textbox", { name: "Euclid knee (e⁻)" });
    fireEvent.change(euclid2, { target: { value: "3" } });
    fireEvent.keyDown(euclid2, { key: "Enter" });
    expect(useDisplay.getState().groups.euclid.knee).toBe(3);
    const thumb = screen.getByRole("slider", { name: "Euclid knee" });
    fireEvent.keyDown(thumb, { key: "Home" });
    expect(useDisplay.getState().groups.euclid.knee).toBeCloseTo(0.1, 6);
    fireEvent.keyDown(thumb, { key: "End" });
    expect(useDisplay.getState().groups.euclid.knee).toBeCloseTo(1e4, 3);
    // the default stays absolute asinh at 100 e⁻
    expect(useDisplay.getState().stretch).toBe("asinh-abs");
    expect(useDisplay.getState().groups.jwst.knee).toBe(100);
  });

  it("no band chips for single-plane tiers", async () => {
    const m = meta({ tiers: [{ key: "jw", label: "JWST", unit: "MJy/sr" }, { key: "lr", label: "LR", unit: "e-" }], default_tier: "jw" });
    const base = defaultHandler(m);
    mockBackend((url) => (url.searchParams.get("tier") === "jw"
      ? cube(8, 8, 1, (i) => i, { "X-Cube-Bands": "F200W", "X-Cube-Unit": "MJy/sr", "X-Cube-Label": "JWST 0" })
      : base(url)));
    render(<MemoryRouter><ImageViewer collection="test" /></MemoryRouter>);
    await screen.findByText(/^JWST 0/);
    expect(screen.queryByRole("radiogroup", { name: "Band or colour" })).toBeNull();
    // a four-band tier brings them back
    fireEvent.click(screen.getByRole("button", { name: "LR" }));
    await screen.findByText(/^LR 0/);
    expect(await screen.findByRole("radiogroup", { name: "Band or colour" })).toBeTruthy();
  });

  it("returning to a view reuses the cached cubes and the meta (no new requests)", async () => {
    mockBackend(defaultHandler());
    const first = render(<MemoryRouter><ImageViewer collection="test" tiers={["lr", "sr"]} /></MemoryRouter>);
    await screen.findByText(/^SR 0/);
    await waitFor(() => expect(calls.filter((u) => u.startsWith("/viewer/cube/")).length).toBeGreaterThan(2));
    await new Promise((r) => setTimeout(r, 50));             // the prefetch settles
    expect(calls.filter((u) => u.startsWith("/viewer/meta/"))).toHaveLength(1);   // one meta per mount
    first.unmount();
    const before = calls.length;
    render(<MemoryRouter><ImageViewer collection="test" tiers={["lr", "sr"]} /></MemoryRouter>);
    await screen.findByText(/^SR 0/);
    await new Promise((r) => setTimeout(r, 50));
    expect(calls.slice(before)).toEqual([]);
  });

  it("an explicit reload refetches the meta and the cubes", async () => {
    mockBackend(defaultHandler());
    let api: ViewerApi | null = null;
    render(<MemoryRouter><ImageViewer collection="test" onReady={(a) => { api = a; }} /></MemoryRouter>);
    await screen.findByText(/^LR 0/);
    await new Promise((r) => setTimeout(r, 50));
    const before = calls.length;
    await act(async () => { await api!.reload(); });
    const after = calls.slice(before);
    expect(after.some((u) => u.startsWith("/viewer/meta/"))).toBe(true);
    expect(after.some((u) => u.startsWith("/viewer/cube/test/0?") && u.includes("tier=lr"))).toBe(true);
  });
});

/* The DOM layout the pointer maths reads: every .cv-frame is SIDE css px of
   content inside a 1 px border, at (LEFT, TOP). The real frames have no
   border now; the pointer maths reads clientLeft/clientTop, so a border
   must still be measured from the padding box. */
const FRAME = { LEFT: 100, TOP: 50, SIDE: 240, BORDER: 1 };
function stubFrameLayout() {
  const { LEFT, TOP, SIDE, BORDER } = FRAME;
  const isFrame = (el: Element) => el.classList.contains("cv-frame") && !el.classList.contains("cv-frame--message");
  const props: [string, (el: HTMLElement) => number][] = [
    ["clientWidth", (el) => (isFrame(el) ? SIDE : 0)],
    ["clientHeight", (el) => (isFrame(el) ? SIDE : 0)],
    ["clientLeft", (el) => (isFrame(el) ? BORDER : 0)],
    ["clientTop", (el) => (isFrame(el) ? BORDER : 0)],
  ];
  for (const [name, get] of props) {
    Object.defineProperty(HTMLElement.prototype, name, { configurable: true, get() { return get(this as HTMLElement); } });
  }
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(function (this: Element) {
    if (isFrame(this)) return new DOMRect(LEFT, TOP, SIDE + 2 * BORDER, SIDE + 2 * BORDER);
    if (this.classList.contains("cv-canvas")) return new DOMRect(LEFT + BORDER, TOP + BORDER, SIDE, SIDE);
    return new DOMRect(0, 0, 0, 0);
  });
  return () => { for (const [name] of props) delete (HTMLElement.prototype as unknown as Record<string, unknown>)[name]; };
}

describe("<Frame> pointer → image", () => {
  let restore: () => void = () => {};
  afterEach(() => restore());

  it("draws markers on the listed tiers only, on their pixel centres, and picks them on click", async () => {
    restore = stubFrameLayout();
    mockBackend(defaultHandler());
    const onPick = vi.fn(), onHover = vi.fn();
    const { container } = render(<MemoryRouter><ImageViewer collection="test" tiers={["lr", "sr"]}
      markers={{ grid: { width: 8, height: 8 }, tiers: ["sr"], onPick, onHover,
        items: [{ key: "a", x: 3, y: 3, r: 1, kind: "galaxy", title: "source a" }] }} /></MemoryRouter>);
    await screen.findByText(/^SR 0/);
    await waitFor(() => expect(container.querySelector(".cv-frame[data-tier='sr'] .cv-mk")).not.toBeNull());
    expect(container.querySelector(".cv-frame[data-tier='lr'] .cv-mk")).toBeNull();
    const mk = container.querySelector<SVGGElement>(".cv-frame[data-tier='sr'] .cv-mk")!;
    // SR is 8×8 px in a 240 px frame: pixel 3's centre is 3.5 × 30 = 105 css px
    const circle = mk.querySelector("circle.cv-mk__shape")!;
    expect(Number(circle.getAttribute("cx"))).toBeCloseTo(105);
    expect(Number(circle.getAttribute("r"))).toBeCloseTo(30);
    expect(mk.querySelector("title")?.textContent).toBe("source a");
    fireEvent.pointerEnter(mk);
    expect(onHover).toHaveBeenCalledWith("a");
    fireEvent.click(mk);
    expect(onPick).toHaveBeenCalledWith("a");
  });

  it("measures the pointer from the content box, not the 1 px border", async () => {
    restore = stubFrameLayout();
    mockBackend(defaultHandler());
    let api: ViewerApi | null = null;
    const { container } = render(<MemoryRouter><ImageViewer collection="test" onReady={(a) => { api = a; }} /></MemoryRouter>);
    await screen.findByText(/^LR 0/);
    const frame = container.querySelector<HTMLElement>(".cv-frame[data-tier='lr']")!;
    // LR is 4×4 px in a 240 px frame: 60 css px per pixel. 179.5 px into the
    // content box is x = 2.992 (pixel 2); from the border box it would be 3.008 (pixel 3).
    const { LEFT, TOP, BORDER } = FRAME;
    fireEvent.pointerMove(frame, { clientX: LEFT + BORDER + 179.5, clientY: TOP + BORDER + 59.5, pointerId: 1 });
    await waitFor(() => expect(api!.getReadout()).not.toBeNull());
    const t = api!.getReadout()!.tiers[0];
    expect([t.x, t.y]).toEqual([2, 0]);
    expect(t.fx).toBeCloseTo(179.5 / 60, 9);
    expect(t.fy).toBeCloseTo(59.5 / 60, 9);
  });

  it("a shift-drag line profile released outside the frame ends on the image edge", async () => {
    restore = stubFrameLayout();
    mockBackend(defaultHandler());
    const states: string[] = [];
    const { container } = render(<MemoryRouter><ImageViewer collection="test" onState={(s) => states.push(s.tier)} /></MemoryRouter>);
    await screen.findByText(/^LR 0/);
    const frame = container.querySelector<HTMLElement>(".cv-frame[data-tier='lr']")!;
    const { LEFT, TOP, BORDER, SIDE } = FRAME;
    const at = (x: number, y: number) => ({ clientX: LEFT + BORDER + x, clientY: TOP + BORDER + y, pointerId: 1, shiftKey: true });
    fireEvent.pointerDown(frame, at(30, 30));
    fireEvent.pointerMove(frame, at(120, 60));
    fireEvent.pointerUp(frame, at(SIDE + 40, 90));     // released 40 px right of the frame
    const svg = await waitFor(() => {
      const line = container.querySelector(".cv-frame[data-tier='lr'] .cv-prof line");
      expect(line).not.toBeNull();
      return line!;
    });
    // p0 = LR (0.5, 0.5); p1 clamped to the right edge x = 4 (frame x = 240) at y = 1.5 (frame 90)
    expect(Number(svg.getAttribute("x1"))).toBeCloseTo(30, 6);
    expect(Number(svg.getAttribute("x2"))).toBeCloseTo(SIDE, 6);
    expect(Number(svg.getAttribute("y2"))).toBeCloseTo(90, 6);
  });

  it("places the lens popups around the crop in the content box", async () => {
    restore = stubFrameLayout();
    mockBackend(defaultHandler());
    const c = new ViewerController({ collection: "test", tiers: ["lr"] });
    await c.start();
    const el = document.createElement("div");
    el.className = "cv-frame";
    const vis = document.createElement("canvas");
    vis.className = "cv-canvas";
    c.registerFrame({ tier: "lr", source: document.createElement("canvas"), visible: vis, element: el, size: () => FRAME.SIDE, redraw: () => {} });
    c.setTool("lens");
    c.lensHover("lr", 0.5, 0.5);
    // The 4 px LR crop is the whole image: the content box (101, 51)–(341, 291).
    // The first corner that fits a 1024×768 window is bottom-right, 12 px off it.
    expect(window.innerWidth).toBe(1024);
    expect(c.s.lens.lr).toEqual({ left: 341 + 12, top: 291 + 12, corner: "bottom-right" });
    c.destroy();
  });
});
