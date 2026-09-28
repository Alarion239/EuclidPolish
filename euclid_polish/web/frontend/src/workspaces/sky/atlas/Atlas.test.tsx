/* The Atlas tab against a recording fake Aladin and a mocked backend:
 * lazy layer fetches, URL-driven layers and view, click → inspector,
 * keep-alive across unmounts. */
import { QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { queryClient } from "../../../api/query";
import { __resetSkyEngineForTests, __setSkyEngineTestHooks, currentSkyEngine } from "../../../sky/engine";
import { makeFakeAladin, type FakeAladin } from "../../../sky/testing/fakeAladin";
import { useInspector } from "../../../state/inspector";
import { resetConfirm } from "../../../ui";
import Atlas from "../tabs/Atlas";
import { LAST_TILE_KEY, rememberTile } from "./home";
import { useAtlas } from "./store";

const poly = (ra: number): [number, number][] => [[ra, 65.1], [ra + 0.007, 65.1], [ra + 0.007, 65.107], [ra, 65.107]];

const LAYERS = {
  groups: ["real", "targets", "inputs", "coverage"],
  layers: [
    {
      id: "nexus-tiles", label: "NEXUS × Euclid tiles", group: "real", kind: "polygons", count: 2, bbox: null,
      style: { color_by: "state", opacity: 0.5 }, ready: true, reason: null, fill_action: null, description: "", url: "/api/sky/layer/nexus-tiles",
      home: { path: "/sky/targets?set=nexus", label: "Sky › Targets" },
    },
    {
      id: "lens-candidates", label: "Q1 lens candidates", group: "targets", kind: "points", count: 1, bbox: null,
      style: { color_by: "grade", shape: "circle", size: 5 }, ready: true, reason: null, fill_action: null, description: "", url: "/api/sky/layer/lens-candidates",
    },
    {
      id: "pairs", label: "JWST × Euclid pairs", group: "real", kind: "polygons", count: 0, bbox: null,
      style: {}, ready: false, reason: "no local data yet", description: "", url: "/api/sky/layer/pairs",
      fill_action: { method: "POST", url: "/api/sky/jwst/pair", label: "Download a JWST × Euclid pair" },
      home: { path: "/sky/targets?set=pairs", label: "Sky › Targets" },
    },
  ],
};
const NEXUS = {
  id: "nexus-tiles", label: "NEXUS × Euclid tiles", group: "real", kind: "polygons", count: 2,
  features: [
    { id: "f200w-0000", polygon: poly(268.37), props: { state: "stale", label: "NEXUS F200W tile 0000" }, inspect: { kind: "realtile", id: "nexus/f200w-0000" } },
    { id: "f200w-0001", polygon: poly(268.39), props: { state: "current", label: "NEXUS F200W tile 0001" }, inspect: { kind: "realtile", id: "nexus/f200w-0001" } },
  ],
};
const LENSES = {
  id: "lens-candidates", label: "Q1 lens candidates", group: "targets", kind: "points", count: 1,
  columns: ["ra", "dec", "grade", "id"], rows: [[268.4, 65.2, "A", "L1"]],
  inspect: { kind: "source", prefix: "lens-candidates/", id_column: "id" },
};

let fake: FakeAladin;
let calls: string[];

function Probe() {
  const loc = useLocation();
  return <output data-testid="loc">{loc.pathname}{loc.search}</output>;
}

const show = (url: string) => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={[url]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes>
        <Route path="/sky/atlas" element={<><Atlas /><Probe /></>} />
        <Route path="/elsewhere" element={<Probe />} />
      </Routes>
    </MemoryRouter>
  </QueryClientProvider>,
);

const search = () => new URLSearchParams(screen.getByTestId("loc").textContent!.split("?")[1] ?? "");

beforeEach(() => {
  fake = makeFakeAladin();
  __setSkyEngineTestHooks({ importer: async () => ({ default: fake.A }), hasWebGL2: () => true });
  calls = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    calls.push(url);
    const body = url.startsWith("/api/sky/layers") ? LAYERS
      : url.startsWith("/api/sky/layer/nexus-tiles") ? NEXUS
        : url.startsWith("/api/sky/layer/lens-candidates") ? LENSES
          : null;
    return new Response(JSON.stringify(body ?? { error: `no route ${url}` }), { status: body ? 200 : 404 });
  }));
  queryClient.clear();
  useInspector.getState().clear();
  useAtlas.getState().set({ status: "idle", view: null, panelOpen: false });
  localStorage.removeItem(LAST_TILE_KEY);
});

afterEach(() => {
  act(() => resetConfirm());
  __resetSkyEngineForTests();
  queryClient.clear();
});

describe("Sky atlas", () => {
  it("creates the engine once, draws the URL's layers and fetches only enabled ones", async () => {
    show("/sky/atlas?ra=268.38&dec=65.1&fov=0.2&layers=nexus-tiles");
    await waitFor(() => expect(fake.created).toHaveLength(1));
    expect(fake.created[0].target).toBe("268.38 65.1");
    // Zoomed in: the tiles are polygons.
    await waitFor(() => expect(fake.overlays.some((o) => o.items.length === 2)).toBe(true));
    expect(calls.some((u) => u.includes("/api/sky/layer/nexus-tiles"))).toBe(true);
    expect(calls.some((u) => u.includes("/api/sky/layer/lens-candidates"))).toBe(false);
    // The background HiPS (Euclid Q1 colour) is the first stack entry.
    expect(fake.layers.get(fake.stack[0])?.url).toMatch(/CDS_P_Euclid_Q1_color$/);
    expect(await screen.findByText("NEXUS × Euclid tiles")).toBeTruthy();
  });

  it("turning a layer on writes the URL and lazily fetches its payload", async () => {
    show("/sky/atlas?ra=268.38&dec=65.1&fov=0.2&layers=nexus-tiles");
    const label = await screen.findByText("Q1 lens candidates");
    fireEvent.click(label);
    await waitFor(() => expect(search().get("layers")).toBe("nexus-tiles,lens-candidates"));
    await waitFor(() => expect(calls.some((u) => u.includes("/api/sky/layer/lens-candidates"))).toBe(true));
    await waitFor(() => expect(fake.catalogs.some((c) => c.sources.length === 1)).toBe(true));
  });

  it("a click on a drawn tile opens its atlas card in the inspector", async () => {
    show("/sky/atlas?ra=268.38&dec=65.1&fov=0.2&layers=nexus-tiles");
    await waitFor(() => expect(fake.overlays.some((o) => o.items.length === 2)).toBe(true));
    const overlay = fake.overlays.find((o) => o.items.length === 2)!;
    // An outline hit alone (Aladin fires no `click`, e.g. off the sky).
    act(() => fake.fire("objectClicked", overlay.items[1], { x: 1, y: 1 }));
    await waitFor(() => expect(useInspector.getState().current).toEqual({ kind: "tile", id: "nexus/f200w-0001" }));
  });

  it("a click INSIDE a tile opens it (Aladin only reports polygon outlines)", async () => {
    show("/sky/atlas?ra=268.38&dec=65.1&fov=0.2&layers=nexus-tiles");
    await waitFor(() => expect(fake.overlays.some((o) => o.items.length === 2)).toBe(true));
    act(() => fake.fire("click", { ra: 268.3735, dec: 65.1035, x: 10, y: 10, isDragging: false }));
    expect(useInspector.getState().current).toEqual({ kind: "tile", id: "nexus/f200w-0000" });
    // A drag (pan) that ends inside a tile opens nothing.
    useInspector.getState().clear();
    act(() => fake.fire("click", { ra: 268.3935, dec: 65.1035, x: 10, y: 10, isDragging: true }));
    expect(useInspector.getState().current).toBeNull();
    // Outline hit on tile 1 + a click position inside tile 0 → the smaller / inside one; both are tiles
    // of the same size, so Aladin's own hit is kept.
    const overlay = fake.overlays.find((o) => o.items.length === 2)!;
    act(() => {
      fake.fire("objectClicked", overlay.items[1], { x: 1, y: 1 });
      fake.fire("click", { ra: 268.3735, dec: 65.1035, x: 1, y: 1, isDragging: false });
    });
    expect(useInspector.getState().current).toEqual({ kind: "tile", id: "nexus/f200w-0001" });
  });

  it("clicks while drawing a region are vertices, not inspector opens", async () => {
    show("/sky/atlas?ra=268.38&dec=65.1&fov=0.2&layers=nexus-tiles");
    await waitFor(() => expect(fake.overlays.some((o) => o.items.length === 2)).toBe(true));
    act(() => useAtlas.getState().set({ selecting: "poly" }));
    act(() => fake.fire("click", { ra: 268.3735, dec: 65.1035, x: 10, y: 10, isDragging: false }));
    expect(useInspector.getState().current).toBeNull();
    act(() => useAtlas.getState().set({ selecting: null }));
  });

  it("hovering inside a tile shows its tooltip", async () => {
    show("/sky/atlas?ra=268.38&dec=65.1&fov=0.2&layers=nexus-tiles");
    await waitFor(() => expect(fake.overlays.some((o) => o.items.length === 2)).toBe(true));
    // The engine re-derives ICRS from the pixel (the fake maps x/10, y/10), whatever
    // frame Aladin reported the move in.
    act(() => fake.fire("mouseMove", { ra: 1, dec: 2, x: 2683.935, y: 651.035, frame: "Galactic" }));
    expect(await screen.findByRole("tooltip")).toHaveProperty("textContent", expect.stringContaining("NEXUS F200W tile 0001"));
    act(() => fake.fire("mouseMove", { ra: 200, dec: 10, x: 2000, y: 100 }));
    await waitFor(() => expect(screen.queryByRole("tooltip")).toBeNull());
  });

  it("a link that names only the inspected tile opens framed on it, not on the whole sky", async () => {
    act(() => { useInspector.getState().show({ kind: "tile", id: "nexus/f200w-0001" }); });
    show("/sky/atlas?layers=nexus-tiles");
    await waitFor(() => expect(search().get("ra")).not.toBeNull());
    expect(Number(search().get("ra"))).toBeCloseTo(268.3935, 3);
    expect(Number(search().get("fov"))).toBeLessThan(0.1);
  });

  it("a URL view change moves the engine; a user move is written back to the URL", async () => {
    show("/sky/atlas?ra=268.38&dec=65.1&fov=0.2&layers=-");
    await waitFor(() => expect(currentSkyEngine()?.attachedTo).toBeTruthy());
    // Simulate the user panning in Aladin.
    act(() => { fake.view.ra = 10; fake.view.dec = 20; fake.view.fov = 3; fake.fire("positionChanged", { ra: 10, dec: 20 }); });
    await waitFor(() => expect(search().get("ra")).toBe("10"), { timeout: 2000 });
    expect(search().get("fov")).toBe("3");
    expect(search().get("layers")).toBe("-");
  });

  it("pixel overlays live in the URL (`img`): drawn as FITS layers, edited from the Layers panel", async () => {
    show("/sky/atlas?ra=268.38&dec=65.1&fov=0.2&layers=-&img=nexus/f200w-0001|lr|VIS@0.5,!nexus/f200w-0000|m:rbf|VIS");
    await waitFor(() => expect([...fake.layers.values()].filter((l) => l.kind === "image")).toHaveLength(2));
    const images = [...fake.layers.values()].filter((l) => l.kind === "image");
    const lr = images.find((l) => l.url.includes("f200w-0001"))!;
    expect(lr.url).toBe("/api/real/nexus/f200w-0001/image.fits?tier=lr&band=VIS");
    expect(lr.opacity).toBe(0.5);
    expect(images.find((l) => l.url.includes("f200w-0000"))!.opacity).toBe(0); // hidden
    expect(await screen.findByText("Pixel overlays")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Remove f200w-0001 · LR · VIS" }));
    await waitFor(() => expect(search().get("img")).toBe("!nexus/f200w-0000|m:rbf|VIS"));
    await waitFor(() => expect([...fake.layers.values()].filter((l) => l.kind === "image")).toHaveLength(1));
  });

  it("`blink=1` shows one visible pixel overlay at a time", async () => {
    show("/sky/atlas?ra=268.38&dec=65.1&fov=0.2&layers=-&img=nexus/f200w-0001|lr|VIS,nexus/f200w-0001|m:rbf|VIS&blink=1");
    await waitFor(() => expect([...fake.layers.values()].filter((l) => l.kind === "image")).toHaveLength(2));
    const opacities = [...fake.layers.values()].filter((l) => l.kind === "image").map((l) => l.opacity).sort();
    expect(opacities).toEqual([0, 1]);
    expect((await screen.findByRole("switch", { name: "Blink between visible overlays" })).getAttribute("aria-checked")).toBe("true");
  });

  it("keeps the one engine alive across unmount / remount (parked, not destroyed)", async () => {
    const first = show("/sky/atlas?ra=268.38&dec=65.1&fov=0.2");
    await waitFor(() => expect(currentSkyEngine()?.attachedTo).toBeTruthy());
    const engine = currentSkyEngine()!;
    first.unmount();
    expect(engine.attachedTo).toBeNull();
    expect(engine.host.parentElement?.classList.contains("sky-engine-park")).toBe(true);
    show("/sky/atlas");
    await waitFor(() => expect(engine.attachedTo).toBeTruthy());
    expect(fake.created).toHaveLength(1);
  });

  it("opens on EDF-N without URL coordinates (not the all-sky view), or on the last inspected tile", async () => {
    const first = show("/sky/atlas?layers=-");
    await waitFor(() => expect(fake.created).toHaveLength(1));
    expect(fake.created[0].target).toBe("269.733 66.018");
    expect(fake.created[0].fov).toBe(14);
    first.unmount();
    __resetSkyEngineForTests();
    fake = makeFakeAladin();
    __setSkyEngineTestHooks({ importer: async () => ({ default: fake.A }), hasWebGL2: () => true });
    rememberTile("nexus/f200w-0040", 268.47, 65.14, 25.5 / 3600);
    show("/sky/atlas?layers=-");
    await waitFor(() => expect(fake.created).toHaveLength(1));
    expect(fake.created[0].target).toBe("268.47 65.14");
    expect(fake.created[0].fov as number).toBeLessThan(0.1);
  });

  it("picks the background from one Select: Euclid first, the all-sky surveys named as such", async () => {
    show("/sky/atlas?ra=268.38&dec=65.1&fov=0.2&layers=-");
    const survey = await screen.findByRole("combobox", { name: "Survey" }) as HTMLSelectElement;
    const labels = [...survey.options].map((o) => o.textContent);
    expect(labels.slice(0, 5)).toEqual(["Euclid Q1 colour", "Euclid Q1 VIS", "Euclid Q1 NISP Y", "Euclid Q1 NISP J", "Euclid Q1 NISP H"]);
    expect(labels).toContain("DSS2 colour (all-sky)");
    fireEvent.change(survey, { target: { value: "q1-vis" } });
    await waitFor(() => expect(search().get("base")).toBe("q1-vis"));
  });

  it("reads the pixel value only on a FITS background (an RGB survey has none)", async () => {
    const { unmount } = show("/sky/atlas?ra=268.38&dec=65.1&fov=0.2&layers=-");
    await screen.findByText("FoV");
    expect(screen.queryByText("Pixel")).toBeNull();                    // Euclid colour: PNG tiles
    unmount();
    show("/sky/atlas?ra=268.38&dec=65.1&fov=0.2&layers=-&base=q1-vis");
    expect(await screen.findByText("Pixel")).toBeTruthy();
  });

  it("groups the layers by what they are, links a shown layer to the tab that owns its data, and fills nothing from the panel", async () => {
    show("/sky/atlas?ra=268.38&dec=65.1&fov=0.2&layers=nexus-tiles");
    expect(await screen.findByText("Real tiles")).toBeTruthy();
    expect(screen.getByText("Targets")).toBeTruthy();
    const open = await screen.findByRole("link", { name: "NEXUS × Euclid tiles: open Sky › Targets" });
    expect(open.getAttribute("href")).toBe("/sky/targets?set=nexus");
    expect(screen.queryByRole("button", { name: /Cache|Download|Sync|Pull|Discover/ })).toBeNull();
  });

  it("says how to fill an empty layer: on the sky for positional data, else in its owning tab", async () => {
    show("/sky/atlas?ra=268.38&dec=65.1&fov=0.2&layers=pairs");
    expect(await screen.findByText("No local data yet: download one from the JWST menu, or right-click the sky.")).toBeTruthy();
  });

  it("shows the WebGL2 message instead of a sky when WebGL2 is missing", async () => {
    __setSkyEngineTestHooks({ hasWebGL2: () => false });
    show("/sky/atlas");
    expect(await screen.findByText("WebGL2 is not available")).toBeTruthy();
    expect(fake.created).toHaveLength(0);
  });
});
