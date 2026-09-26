/* The React engine and the legacy compat wrapper against a mocked /viewer
 * backend (no network): loading, verbatim server errors, keyboard, ?id=
 * lookup, per-viewer override, readout through WCS, residual tiers, URL
 * state, and the JWST carousel's no-remount index follow. */
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useShortcutRegistry } from "../hooks/useShortcut";
import { CutoutViewer } from "../legacy";
import { useDisplay } from "../state/display";
import golden from "./__fixtures__/color_golden.json";
import type { ColorMeta } from "./color";
import { sharedCubeCache } from "./cube";
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
  useDisplay.getState().reset();
});
afterEach(() => { vi.unstubAllGlobals(); });

describe("ViewerController", () => {
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
      selection: { u: 0.5, v: 0.5, revision: 1 },
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
    render(<MemoryRouter><ImageViewer collection="test" tiers={["lr", "hr"]} /></MemoryRouter>);
    expect(await screen.findByText(/^LR 0 · VIS/)).toBeTruthy();
    expect(await screen.findByText("HR records are not synced for this subset")).toBeTruthy();
    expect(screen.getByRole("toolbar", { name: "Viewer tools" })).toBeTruthy();
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

  it("round-trips the view in the URL codec", () => {
    const v = { u: 0.25, v: 0.75, angularSideArcsec: 3.2, relativeSide: null };
    expect(parseView(serializeView(v))).toEqual(v);
    expect(parseView("0.5,0.5,r0.25")).toEqual({ u: 0.5, v: 0.5, angularSideArcsec: null, relativeSide: 0.25 });
    expect(parseView("garbage")).toBeNull();
    expect(serializeView(null)).toBe("");
  });
});

/* The DOM layout the pointer maths reads: every .cv-frame is SIDE css px of
   content inside a 1 px border (the real `.cv-frame` style), at (LEFT, TOP). */
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

describe("legacy <CutoutViewer> compat", () => {
  it("follows a changing initialIndex without remounting (JWST carousel)", async () => {
    mockBackend(defaultHandler());
    const ready = vi.fn();
    const { rerender } = render(<MemoryRouter><CutoutViewer collection="test" urlKey="" initialIndex={0} onReady={ready} /></MemoryRouter>);
    await screen.findByText(/^LR 0 · VIS/);
    const metaCalls = () => calls.filter((u) => u.startsWith("/viewer/meta/")).length;
    const before = metaCalls();
    rerender(<MemoryRouter><CutoutViewer collection="test" urlKey="" initialIndex={2} onReady={ready} /></MemoryRouter>);
    expect(await screen.findByText(/^LR 2 · VIS/)).toBeTruthy();
    expect(metaCalls()).toBe(before);
    expect(ready.mock.calls.filter(([a]) => a === null)).toHaveLength(0);
  });

  it("hideToolbar keeps the navigation; compact hides both", async () => {
    mockBackend(defaultHandler());
    const { container, unmount } = render(<MemoryRouter><CutoutViewer collection="test" hideToolbar urlKey="" /></MemoryRouter>);
    await screen.findByText(/^LR 0/);
    expect(container.querySelector(".cv-toolbar")).toBeNull();
    expect(container.querySelector(".cv-nav")).not.toBeNull();
    unmount();
    const c2 = render(<MemoryRouter><CutoutViewer collection="test" compact urlKey="" /></MemoryRouter>);
    await screen.findByText(/^LR 0/);
    expect(c2.container.querySelector(".cv-nav")).toBeNull();
  });

  it("loadColorEngine resolves to the viewer's colour pipeline", async () => {
    const { loadColorEngine } = await import("../legacy");
    const fn = await loadColorEngine();
    const img = fn({ data: Float32Array.from([0, 3000, 3000, 3000]), h: 1, w: 1, c: 4 }, COLOR as never, { color: "VIS", knee: 100, gain: 1, K0: 100 });
    expect(Array.from(img.data)).toEqual([0, 0, 0, 255]);
  });
});
