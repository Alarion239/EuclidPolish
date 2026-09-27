/* The atlas inspector cards and their remote actions (mocked backend). */
import { QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobsStore } from "../../../../api/jobs";
import { queryClient } from "../../../../api/query";
import { useInspector } from "../../../../state/inspector";
import { useSelection } from "../../../../state/selection";
import { resetConfirm } from "../../../../ui";
import { cacheTileAt, runLayerFill, runNexusProduction } from "../actions";
import { JwstObservations } from "../JwstObservations";
import { IMG_CODEC, resolveOverlays } from "../pixelOverlays";
import { PointCard } from "./PointCard";
import SourceInspector from "./SourceInspector";
import { sourceViewerFor } from "./SourceViewer";
import TileInspector, { jwstFilters, outputOrigin, overlayTiers, tileFovDeg } from "./TileInspector";

vi.mock("../../../../viewer", () => ({
  ImageViewer: (p: { collection: string; initialId?: string; tiers?: string[]; params?: Record<string, string> }) => (
    <div data-testid="viewer" data-models={p.params?.models ?? ""}>{p.collection}:{p.initialId}:{(p.tiers ?? []).join(",")}</div>
  ),
}));

type Reply = { status?: number; body: unknown };
let routes: Record<string, (form: Record<string, string>) => Reply>;
let posts: { url: string; form: Record<string, string> }[];

const CARD = {
  source: "nexus", id: "f200w-0012", ref: "nexus/f200w-0012", label: "NEXUS F200W tile 0012",
  ra: 268.41079, dec: 65.11848, field: "EDF-N", shape: [255, 255], pixscale: 0.1, has_jwst: true, model_ready: true,
  production_state: "stale", runnable_models: ["production", "mean"],
  models: { rbf: { state: "current", legacy: true, label: "RBF", created: "2026-09-21T13:15:44Z" } },
  image_urls: { lr: "/x", jwst: "/y", "m:rbf": "/z" }, disk: { total_bytes: 4167360 },
  q1_tile: { tile: "102158584", levels_e: [27.66, 13.3, 14.1, 13.8], rejected: null },
};

const AT = (verdict: string) => ({
  ra: 268.78, dec: 65.4, field: "EDF-N", in_q1: verdict !== "outside", q1_observed: verdict === "observed", q1_verdict: verdict,
  q1_tiles: verdict === "outside" ? [] : [{ tile: "102158890", field: "EDF-N", levels_e: [27.5, 1, 1, 1], rejected: verdict === "unobserved" ? "no coverage" : null, margin_arcsec: 605 }],
  best_tile: verdict === "outside" ? null : "102158890", real_tiles: [{ ref: "nexus/f200w-0001", label: "NEXUS tile 1", state: "stale", inspect: { kind: "realtile", id: "nexus/f200w-0001" } }],
  nexus: [], pairs: [], jwst: [], jwst_discovered: false,
});

function Probe() {
  const loc = useLocation();
  return <output data-testid="loc">{loc.pathname}{loc.search}</output>;
}

const show = (el: ReactElement) => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={["/elsewhere"]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes><Route path="*" element={<>{el}<Probe /></>} /></Routes>
    </MemoryRouter>
  </QueryClientProvider>,
);

const formOf = (body: BodyInit | null | undefined): Record<string, string> => {
  const out: Record<string, string> = {};
  if (body instanceof FormData) body.forEach((v, k) => { out[k] = String(v); });
  return out;
};

beforeEach(() => {
  routes = {};
  posts = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = String(input);
    const method = init.method ?? "GET";
    const form = formOf(init.body);
    if (method === "POST") posts.push({ url, form });
    const key = Object.keys(routes).find((k) => `${method} ${url}`.startsWith(k));
    const r = key ? routes[key](form) : { status: 404, body: { ok: false, error: `no route ${url}` } };
    return new Response(JSON.stringify(r.body), { status: r.status ?? 200 });
  }));
  queryClient.clear();
  useJobsStore.getState().reset();
  useSelection.getState().clear();
});
afterEach(() => {
  act(() => resetConfirm());
  queryClient.clear();
});

const answer = async (title: RegExp, button: string) => {
  const dlg = await screen.findByRole("alertdialog", { name: title });
  fireEvent.click(within(dlg).getByRole("button", { name: button }));
  await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull());
};

describe("tile card", () => {
  it("pure helpers: overlay tiers and a framing field of view", () => {
    expect(overlayTiers(CARD as never)).toEqual(["lr", "m:rbf", "jwst"]);
    expect(tileFovDeg({ shape: [255, 255], pixscale: 0.1 })).toBeCloseTo((25.5 / 3600) * 2.5);
  });

  it("pure helpers: output provenance", () => {
    expect(outputOrigin({ legacy: true, origin: "nexus-field", member_labels: ["m1", "m2"], combiner_kind: "spatial_gate" }))
      .toBe("legacy nexus-field · 2 members · spatial gate");
    expect(outputOrigin({})).toBe("");
  });

  it("maps the palette's nexus/<n> to the real tile: the one real-tile card, image first", async () => {
    routes["GET /api/real/nexus/f200w-0012"] = () => ({ body: CARD });
    const { container } = show(<TileInspector id="nexus/12" />);
    expect(await screen.findByText("production stale")).toBeTruthy();
    const viewer = screen.getByTestId("viewer");
    expect(viewer.textContent).toBe("real:f200w-0012:lr,m:rbf");   // two large frames; JWST one chip away
    // the tier picker offers only this tile's own outputs (no source-wide spec list)
    expect(viewer.dataset.models).toBe("rbf");
    // the viewer is the first thing on the card
    expect(container.querySelector(".res-card")?.firstElementChild?.contains(viewer)).toBe(true);
    expect(screen.getByText(/102158584, VIS sky 27\.7 e⁻/)).toBeTruthy();
    expect(screen.getByText("rbf")).toBeTruthy();
    expect(screen.getByText(/Holes %/, { selector: "strong" })).toBeTruthy();          // defined on the card
  });

  it("server errors are shown verbatim", async () => {
    routes["GET /api/real/nexus/f200w-9999"] = () => ({ status: 404, body: { error: "unknown tile nexus/f200w-9999" } });
    show(<TileInspector id="nexus/f200w-9999" />);
    expect(await screen.findByText("unknown tile nexus/f200w-9999")).toBeTruthy();
  });

  it("overlay on the sky: writes the FITS overlay into the atlas URL and flies there", async () => {
    routes["GET /api/real/nexus/f200w-0012"] = () => ({ body: CARD });
    show(<TileInspector id="nexus/f200w-0012" />);
    fireEvent.click(await screen.findByRole("button", { name: "Overlay on the sky" }));   // folded off the atlas
    fireEvent.click(await screen.findByRole("button", { name: "Add to the sky" }));
    const loc = screen.getByTestId("loc").textContent!;
    expect(loc).toMatch(/^\/sky\/atlas\?ra=268\.41079&dec=65\.11848&fov=/);
    const img = IMG_CODEC.parse(new URLSearchParams(loc.split("?")[1]).get("img")!)!;
    expect(img).toEqual([{ ref: "nexus/f200w-0012", tier: "m:rbf", band: "VIS" }]);
    expect(resolveOverlays(img)[0].url).toBe("/api/real/nexus/f200w-0012/image.fits?tier=m%3Arbf&band=VIS");
  });

  it("compare models preselects exactly this tile (replacing an older pick) and opens Experiments", async () => {
    routes["GET /api/real/nexus/f200w-0012"] = () => ({ body: CARD });
    useSelection.getState().select("tile", ["archive/007"]);
    show(<TileInspector id="nexus/f200w-0012" />);
    fireEvent.click(await screen.findByRole("button", { name: "Compare models…" }));
    expect(useSelection.getState().get("tile")).toEqual(["nexus/f200w-0012"]);
    expect(screen.getByTestId("loc").textContent).toBe("/sky/experiments?tiles=nexus%2Ff200w-0012");
  });

  it("run models proposes the missing production + mean, asks first, then starts the experiment job", async () => {
    routes["GET /api/real/nexus/f200w-0012"] = () => ({ body: CARD });
    routes["POST /api/experiments"] = () => ({ body: { ok: true, job_id: "abc12345", experiment_id: "e1" } });
    routes["GET /api/jobs"] = () => ({ body: [] });
    routes["GET /api/models"] = () => ({ body: { regime: "starfull", models: [
      { spec: "production", kind: "production", label: "P", available: true, n_members: 2, reads: ["1·psnr", "2·psnr"] },
      { spec: "mean", kind: "mean", label: "M", available: true, n_members: 2, members: ["1·psnr", "2·psnr"] },
    ] } });
    show(<TileInspector id="nexus/f200w-0012" />);
    fireEvent.click(await screen.findByRole("button", { name: "Run models…" }));
    fireEvent.click(await screen.findByRole("button", { name: "Run 2" }));
    const dlg = await screen.findByRole("alertdialog", { name: /Run 2 models on 1 tile/ });
    // the confirm states the cost
    expect(dlg.textContent).toContain("2 outputs (2 models on 1 tile). Needs 2 member SRs per tile: at most 2 member inferences");
    fireEvent.click(within(dlg).getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts.map((p) => p.url)).toContain("/api/experiments"));
    expect(posts[0].form).toEqual({ tiles: "nexus/f200w-0012", models: "production,mean" });
    expect(useJobsStore.getState().keyed["sky:experiment"]).toBe("abc12345");
  });
});

describe("source and point cards", () => {
  it("a catalogue source comes from its layer payload", async () => {
    routes["GET /api/sky/layers"] = () => ({ body: { layers: [] } });
    routes["GET /api/sky/layer/lens-candidates"] = () => ({
      body: {
        id: "lens-candidates", label: "Q1 lens candidates", kind: "points", count: 1, columns: ["ra", "dec", "grade", "id"],
        rows: [[58.68, -51.28, "C", "L1"]], inspect: { kind: "source", prefix: "lens-candidates/", id_column: "id" },
      },
    });
    show(<SourceInspector id="lens-candidates/L1" />);
    expect(await screen.findByText("grade C")).toBeTruthy();
    expect(screen.getByRole("button", { name: "What covers this point" })).toBeTruthy();
    // No catalogue-eval reconstruction of L1 (the eval tile 404s): a note, no viewer.
    expect(await screen.findByText("No catalogue-eval reconstruction of this object yet.")).toBeTruthy();
    expect(screen.queryByTestId("viewer")).toBeNull();
  });

  it("a lens candidate with a reconstruction shows its LR / SR in the evaluation viewer", async () => {
    routes["GET /api/sky/layers"] = () => ({ body: { layers: [] } });
    routes["GET /api/sky/layer/lens-candidates"] = () => ({
      body: {
        id: "lens-candidates", label: "Q1 lens candidates", kind: "points", count: 1, columns: ["ra", "dec", "grade", "id"],
        rows: [[58.68, -51.28, "C", "L1"]], inspect: { kind: "source", prefix: "lens-candidates/", id_column: "id" },
      },
    });
    routes["GET /api/real/eval/L1"] = () => ({ body: { ...CARD, source: "eval", id: "L1", ref: "eval/L1" } });
    show(<SourceInspector id="lens-candidates/L1" />);
    expect((await screen.findByTestId("viewer")).textContent).toBe("evaluation:L1:LR,SR");
    expect(screen.getByRole("button", { name: "Evaluation tile card" })).toBeTruthy();
  });

  it("a PSF cluster shows its ePSF in the psfs viewer", async () => {
    routes["GET /api/sky/layers"] = () => ({ body: { layers: [] } });
    routes["GET /api/sky/layer/psf-clusters"] = () => ({
      body: {
        id: "psf-clusters", label: "PSF clusters", kind: "points", count: 1, columns: ["ra", "dec", "fwhm_arcsec", "cluster", "id"],
        rows: [[268.4, 65.2, 0.17, 1, "cluster-001"]], inspect: { kind: "source", prefix: "psf-clusters/", id_column: "id" },
      },
    });
    show(<SourceInspector id="psf-clusters/cluster-001" />);
    expect((await screen.findByTestId("viewer")).textContent).toBe("psfs:cluster-001:");
  });

  it("viewer collections per source layer", () => {
    expect(sourceViewerFor("psf-clusters", "cluster-012")).toEqual({ collection: "psfs", initialId: "cluster-012" });
    expect(sourceViewerFor("galaxies", "gal_1")).toMatchObject({ collection: "evaluation", evalRef: "eval/gal_1" });
    expect(sourceViewerFor("stars", "5")).toBeNull();
    expect(sourceViewerFor("jwst-mast", "jw01837-o001")).toBeNull();
    expect(sourceViewerFor("psf-clusters", "")).toBeNull();
  });

  it("a missing source says so", async () => {
    routes["GET /api/sky/layers"] = () => ({ body: { layers: [] } });
    routes["GET /api/sky/layer/stars"] = () => ({ body: { id: "stars", label: "Stars", kind: "points", count: 0, columns: ["ra", "dec"], rows: [] } });
    show(<SourceInspector id="stars/5" />);
    expect(await screen.findByText("Not found")).toBeTruthy();
  });

  it("the point card lists the Q1 verdict, tiles and real results", async () => {
    routes["GET /api/sky/at"] = () => ({ body: AT("observed") });
    show(<PointCard ra={268.78} dec={65.4} />);
    expect(await screen.findByText("Q1 observed")).toBeTruthy();
    expect(screen.getByText("102158890")).toBeTruthy();
    expect(screen.getByText("NEXUS tile 1")).toBeTruthy();
  });

  it("bad point ids are rejected", () => {
    show(<SourceInspector id="at/999,0" />);
    expect(screen.getByText("Bad position")).toBeTruthy();
  });
});

describe("cache a tile here", () => {
  it("outside Q1: explains and never posts", async () => {
    routes["GET /api/sky/at"] = () => ({ body: AT("outside") });
    show(<div />);
    await act(async () => { await cacheTileAt(150.1, 2.2); });
    expect(posts).toHaveLength(0);
    expect(screen.queryByRole("alertdialog")).toBeNull();
  });

  it("observed: confirms, then posts ra/dec", async () => {
    routes["GET /api/sky/at"] = () => ({ body: AT("observed") });
    routes["POST /api/real/tiles"] = () => ({ body: { ok: true, job_id: "j1", id: "ra268_dec65", ref: "tile/ra268_dec65" } });
    routes["GET /api/jobs"] = () => ({ body: [] });
    show(<div />);
    let p!: Promise<unknown>;
    act(() => { p = cacheTileAt(268.78, 65.4); });
    await answer(/Cache a 25\.6″ tile here/, "Cache tile");
    await act(async () => { await p; });
    expect(posts).toEqual([{ url: "/api/real/tiles", form: { ra: "268.78", dec: "65.4" } }]);
  });

  it("opens the new tile (LR + the models it ran) in the inspector when the job finishes", async () => {
    routes["GET /api/sky/at"] = () => ({ body: AT("observed") });
    routes["POST /api/real/tiles"] = () => ({ body: { ok: true, job_id: "j9", id: "ra268_dec65", ref: "tile/ra268_dec65" } });
    routes["GET /api/jobs"] = () => ({ body: [] });
    show(<div />);
    let p!: Promise<unknown>;
    act(() => { p = cacheTileAt(268.78, 65.4, { run: true }); });
    await answer(/Cache a 25\.6″ tile here/, "Cache tile");
    await act(async () => { await p; });
    expect(useInspector.getState().current).toBeNull();
    act(() => {
      useJobsStore.setState((st) => ({ jobs: { ...st.jobs, j9: { job_id: "j9", label: "cache tile", status: "done", result: { id: "ra268_dec65" } } as never } }));
    });
    await waitFor(() => expect(useInspector.getState().current).toEqual({ kind: "tile", id: "tile/ra268_dec65" }));
    act(() => { useInspector.getState().hide(); });
  });

  it("unobserved Q1 tile: a danger confirm, then force=1", async () => {
    routes["GET /api/sky/at"] = () => ({ body: AT("unobserved") });
    routes["POST /api/real/tiles"] = () => ({ body: { ok: true, job_id: "j2" } });
    routes["GET /api/jobs"] = () => ({ body: [] });
    show(<div />);
    let p!: Promise<unknown>;
    act(() => { p = cacheTileAt(268.78, 65.4, { run: true }); });
    await answer(/unobserved Q1 tile/, "Cache anyway");
    await act(async () => { await p; });
    expect(posts[0].form).toEqual({ ra: "268.78", dec: "65.4", run: "production,mean", force: "1" });
  });

  it("cancel posts nothing", async () => {
    routes["GET /api/sky/at"] = () => ({ body: AT("observed") });
    show(<div />);
    let p!: Promise<unknown>;
    act(() => { p = cacheTileAt(268.78, 65.4); });
    await answer(/Cache a 25\.6″ tile here/, "Cancel");
    await act(async () => { await p; });
    expect(posts).toHaveLength(0);
  });
});

describe("JWST × Euclid actions (absorbed from the legacy page)", () => {
  const NEXUS_FIELDS = { fields: [{ field_id: "nexus-qdr-ep05-f200w-euclid255", target_name: "NEXUS Deep Epoch 05 · F200W", count: 445, stale_sr_count: 445 }] };

  it("JWST filters of a tile: pair bands, the NEXUS filter, or the server default", () => {
    expect(jwstFilters({ has_jwst: true, extras: { jwst_bands: [{ filter: "F150W" }, { filter: "F444W" }, { filter: "F150W" }] } })).toEqual(["F150W", "F444W"]);
    expect(jwstFilters({ has_jwst: true, extras: { filter: "F200W" } })).toEqual(["F200W"]);
    expect(jwstFilters({ has_jwst: true, extras: {} })).toEqual([""]);
    expect(jwstFilters({ has_jwst: false, extras: { filter: "F200W" } })).toEqual([]);
  });

  it("run production on the stale NEXUS tiles: confirm with the count, then post the field id", async () => {
    routes["GET /api/jwst-euclid/nexus/fields"] = () => ({ body: NEXUS_FIELDS });
    routes["POST /api/jwst-euclid/nexus/infer"] = () => ({ body: { job_id: "n1", field_id: "nexus-qdr-ep05-f200w-euclid255" } });
    routes["GET /api/jobs"] = () => ({ body: [] });
    show(<div />);
    let p!: Promise<unknown>;
    act(() => { p = runNexusProduction(); });
    await answer(/Run production on 445 NEXUS tiles/, "Run production");
    await act(async () => { await p; });
    expect(posts).toEqual([{ url: "/api/jwst-euclid/nexus/infer", form: { field_id: "nexus-qdr-ep05-f200w-euclid255" } }]);
    expect(useJobsStore.getState().started).toContain("n1");
  });

  it("no cached NEXUS field: nothing is posted", async () => {
    routes["GET /api/jwst-euclid/nexus/fields"] = () => ({ body: { fields: [] } });
    show(<div />);
    await act(async () => { await runNexusProduction(); });
    expect(posts).toHaveLength(0);
    expect(screen.queryByRole("alertdialog")).toBeNull();
  });

  it("a VIS-only pair offers 'Build LR + run production' (the legacy infer endpoint)", async () => {
    routes["GET /api/real/pair/jw-ngc-1"] = () => ({
      body: { ...CARD, source: "pair", id: "jw-ngc-1", ref: "pair/jw-ngc-1", label: "NGC 1", model_ready: false, bands: ["VIS"], runnable_models: [] },
    });
    routes["POST /api/jwst-euclid/infer"] = () => ({ body: { job_id: "p1", field_id: "jw-ngc-1" } });
    routes["GET /api/jobs"] = () => ({ body: [] });
    show(<TileInspector id="pair/jw-ngc-1" />);
    fireEvent.click(await screen.findByRole("button", { name: "Build LR + run production" }));
    await answer(/Build the four-band LR/, "Build + run");
    await waitFor(() => expect(posts).toEqual([{ url: "/api/jwst-euclid/infer", form: { field_id: "jw-ngc-1" } }]));
  });

  it("the discovery fill action goes through the discovery confirm (all of Q1)", async () => {
    routes["POST /api/sky/jwst/discover"] = () => ({ body: { ok: true, job_id: "d1", tile_count: 352 } });
    routes["GET /api/jobs"] = () => ({ body: [] });
    show(<div />);
    let p!: Promise<unknown>;
    act(() => { p = runLayerFill({ url: "/api/sky/jwst/discover", label: "Discover JWST observations (MAST)" }); });
    await answer(/Discover JWST observations/, "Discover");
    await act(async () => { await p; });
    expect(posts).toEqual([{ url: "/api/sky/jwst/discover", form: {} }]);
  });

  it("discovered observations: a table with a per-row pair download", async () => {
    routes["GET /api/sky/layer/jwst-mast"] = () => ({
      body: {
        id: "jwst-mast", label: "JWST MAST footprints", kind: "points", count: 2,
        columns: ["ra", "dec", "obs_id", "instrument", "filters", "target", "polygons", "status"],
        rows: [
          [268.4, 65.2, "jw01837-o001", "NIRCAM/IMAGE", "F200W;F444W", "NEXUS", 3, "exact_intersection"],
          [53.1, -27.8, "jw01180-o002", "NIRCAM/IMAGE", "F115W", "GOODS-S", 1, "nearby"],
        ],
        inspect: { kind: "source", prefix: "jwst-mast/", id_column: "obs_id" },
      },
    });
    routes["POST /api/sky/jwst/pair"] = () => ({ body: { ok: true, job_id: "q1", pair_id: "x", ref: "pair/x", mode: "archive" } });
    routes["GET /api/jobs"] = () => ({ body: [] });
    const onShow = vi.fn();
    show(<JwstObservations open onOpenChange={() => {}} onShow={onShow} view={{ ra: 268.4, dec: 65.2, fov: 1 }} />);
    expect(await screen.findByText("GOODS-S")).toBeTruthy();
    expect(screen.getByText("exact intersection")).toBeTruthy();
    const [first] = screen.getAllByRole("button", { name: "Pair" });
    fireEvent.click(first);
    await answer(/Download a JWST × Euclid pair/, "Download pair");
    await waitFor(() => expect(posts[0]?.url).toBe("/api/sky/jwst/pair"));
    expect(["jw01837-o001", "jw01180-o002"]).toContain(posts[0].form.obs_id);
    expect(posts[0].form.size_arcsec).toBe("30");
  });

  it("no discovery yet: an empty state that offers one", async () => {
    routes["GET /api/sky/layer/jwst-mast"] = () => ({
      body: { id: "jwst-mast", label: "JWST MAST footprints", kind: "points", count: 0, columns: ["ra", "dec", "obs_id"], rows: [] },
    });
    show(<JwstObservations open onOpenChange={() => {}} onShow={() => {}} view={null} />);
    expect(await screen.findByText("No JWST discovery yet")).toBeTruthy();
  });
});

