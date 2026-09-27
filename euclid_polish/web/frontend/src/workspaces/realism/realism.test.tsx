/* The Realism workspace against a mocked Flask. Re-homes every UI behaviour
 * the pytest page-source checks used to pin (phase1 WP-B1a handoff): the ONE
 * Q1 MER + PHZ galaxy action and its notes, activation, the relations, the
 * three joint views, the figure export, include-training / "no training",
 * the stellar query / fit wording, the field statistics without calibration
 * lanes, the Synthetic–Real multipoint collection, the Noise tab and no
 * Round-trip entry — plus the overview checklist, the shared header, the
 * inspectors, atlas links and the synced viewer transfer. */
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import type { ComponentType, ReactElement } from "react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobsStore } from "../../api/jobs";
import { queryClient } from "../../api/query";
import { allPages } from "../../app/nav";
import { usePaletteRegistry } from "../../app/palette";
import { C, categorical } from "../../colors";
import { useInspector } from "../../state/inspector";
import { resetConfirm } from "../../ui";
import { jointColor } from "./chartKit";
import { PARAMETER_ORDER, USEFUL_SHAPE_KEYS, jointMapSeries, radiusEntries, radiusYLabel, xAxisOf, xDomainOf } from "./galaxies/model";
import { RealismHeader, resetTrainingPref } from "./header";
import { histogramSeries, relationPoints, visibleFrom } from "./pixels/model";
import {
  ARCHIVE_META, FIELDS, JOINT_MAPS, NOISE, NOISE_POSITION, OVERVIEW, PAIR, SKY_META, galaxyPayload, pixelsPayload, starPayload,
} from "./testFixtures";

type ViewerProps = { collection: string; params?: Record<string, string>; tiers?: string[]; urlKey?: string;
  toolbar?: string; nav?: boolean; onReady?: (api: unknown) => void; onState?: (s: unknown) => void };
const hoisted = vi.hoisted(() => ({ viewers: [] as { props: ViewerProps; api: { setView: ReturnType<typeof vi.fn>; resetView: ReturnType<typeof vi.fn> } }[] }));

vi.mock("../../viewer", async () => {
  const React = await import("react");
  return {
    ImageViewer: (props: ViewerProps) => {
      React.useEffect(() => {
        const api = { setView: vi.fn(), goTo: vi.fn(), resetView: vi.fn() };
        hoisted.viewers.push({ props, api });
        props.onReady?.(api);
        return () => props.onReady?.(null);
        // eslint-disable-next-line react-hooks/exhaustive-deps
      }, []);
      return React.createElement("div", { "data-testid": `viewer-${props.collection}`, "data-tiers": (props.tiers ?? []).join(",") });
    },
  };
});
vi.mock("../../fasrc", () => ({
  StepById: ({ stepId }: { stepId: string }) => <div data-testid={`step-${stepId}`} />,
}));

type Reply = { status?: number; body: unknown };
let routes: Record<string, (form: Record<string, string>) => Reply>;
let calls: { url: string; method: string; form: Record<string, string> }[];
const posts = (u: string) => calls.filter((c) => c.method === "POST" && c.url === u);
const gets = (prefix: string) => calls.filter((c) => c.method === "GET" && c.url.startsWith(prefix));

const formOf = (body: BodyInit | null | undefined): Record<string, string> => {
  const out: Record<string, string> = {};
  if (body instanceof FormData) body.forEach((v, k) => { out[k] = String(v); });
  else if (body instanceof URLSearchParams) body.forEach((v, k) => { out[k] = v; });
  else if (typeof body === "string") new URLSearchParams(body).forEach((v, k) => { out[k] = v; });
  return out;
};

const job = (id: string) => ({ job_id: id, label: id, status: "running", started: 1, finished: null, duration: 1, error: null,
  log: "", log_truncated: false, kind: null, cancellable: false, result: null, progress: null });

beforeEach(() => {
  calls = [];
  hoisted.viewers.length = 0;
  routes = {
    "GET /api/realism/overview": () => ({ body: OVERVIEW }),
    "GET /api/fasrc/status": () => ({ body: { ssh_connected: true, last_error: null } }),
    "GET /api/noise": () => ({ body: NOISE }),
    "GET /api/noise/positions/102021990": () => ({ body: NOISE_POSITION }),
    "GET /api/galaxy-distributions?include_training=0": () => ({ body: galaxyPayload() }),
    "GET /api/galaxy-distributions?include_training=1": () => ({ body: galaxyPayload({ training_included: true }) }),
    "GET /api/star-distribution?include_training=0": () => ({ body: starPayload() }),
    "GET /api/population-comparison?include_training=0": () => ({ body: pixelsPayload() }),
    "GET /viewer/meta/archive-fields": () => ({ body: ARCHIVE_META }),
    "GET /viewer/meta/sky?subset=test": () => ({ body: SKY_META }),
    "GET /viewer/meta/sky?subset=validate": () => ({ body: SKY_META }),
    "GET /api/jobs?summary=1": () => ({ body: [] }),
  };
  for (const [url, id] of [
    ["/api/galaxy-distributions/query-q1-counts", "gq"], ["/api/galaxy-distributions/activate", "ga"],
    ["/api/galaxy-distributions/build", "gb"], ["/api/star-distribution/query", "sq"], ["/api/star-distribution/fit", "sf"],
    ["/api/star-distribution/activate", "sa"], ["/api/population-comparison/build", "pb"], ["/api/archive-fields/sync", "as"],
    ["/api/population-comparison/sync-training-catalog", "ts"],
  ] as const) {
    routes[`POST ${url}`] = () => ({ body: { ok: true, job_id: id } });
    routes[`GET /api/jobs/${id}`] = () => ({ body: job(id) });
  }
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = String(input);
    const method = init.method ?? "GET";
    const form = formOf(init.body);
    calls.push({ url, method, form });
    const r = routes[`${method} ${url}`]?.(form) ?? { status: 404, body: { ok: false, error: `no route ${url}` } };
    return new Response(JSON.stringify(r.body), { status: r.status ?? 200 });
  }));
  queryClient.clear();
  useJobsStore.getState().reset();
  useInspector.getState().reset();
  resetTrainingPref();
  usePaletteRegistry.getState().reset();
});
afterEach(() => { act(() => resetConfirm()); queryClient.clear(); });

let lastLocation = "";
function Spy() { const l = useLocation(); lastLocation = l.pathname + l.search; return null; }
const params = () => new URLSearchParams(lastLocation.split("?")[1] ?? "");

const show = (el: ReactElement, url: string) => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={[url]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes><Route path="/realism/*" element={<>{el}<Spy /></>} /><Route path="*" element={<Spy />} /></Routes>
    </MemoryRouter>
  </QueryClientProvider>,
);

async function answer(title: RegExp | string, button: string) {
  const dlg = await screen.findByRole("alertdialog", { name: title });
  fireEvent.click(within(dlg).getByRole("button", { name: button }));
}

const paletteIds = () => usePaletteRegistry.getState().list().map((a) => a.id);
const runPalette = (id: string) => {
  const action = usePaletteRegistry.getState().list().find((a) => a.id === id);
  if (!action) throw new Error(`no palette action ${id}`);
  action.run();
};

const tab = async (name: string) => (await import(`./tabs/${name}.tsx`) as { default: ComponentType }).default;

/* ── overview ─────────────────────────────────────────────────────────── */

describe("overview", () => {
  it("shows the synthetic_generate gate and one checklist of every prior with its fix", async () => {
    const Overview = await tab("Overview");
    show(<Overview />, "/realism/overview");
    expect(await screen.findByText("Blocked")).toBeTruthy();
    expect(screen.getAllByText(/activate a valid Gaia\+Euclid stellar calibration/).length).toBeGreaterThan(0);
    for (const label of ["Galaxy joint model", "Stellar prior", "TNG radius manifest", "Noise model", "Field-statistics cache"]) {
      expect(screen.getByText(label)).toBeTruthy();
    }
    expect(screen.getByText("Stellar candidate needs a refit")).toBeTruthy();
    const open = screen.getAllByRole("link", { name: "Open" }).map((a) => a.getAttribute("href"));
    expect(open).toContain("/realism/galaxies");
    expect(open).toContain("/realism/pixels");
    fireEvent.click(screen.getByRole("button", { name: "Activate model" }));
    await answer(/Activate model/, "Activate model");
    await waitFor(() => expect(posts("/api/galaxy-distributions/activate")).toHaveLength(1));
  });

  it("disables FASRC fixes while offline and opens the readiness inspector", async () => {
    routes["GET /api/fasrc/status"] = () => ({ body: { ssh_connected: false, last_error: "timed out" } });
    const Overview = await tab("Overview");
    show(<Overview />, "/realism/overview");
    const validate = await screen.findByRole("button", { name: "Validate on FASRC" });
    await waitFor(() => expect((validate as HTMLButtonElement).disabled).toBe(true));
    fireEvent.click(screen.getByRole("button", { name: "Inspect Stellar prior" }));
    expect(useInspector.getState().current).toEqual({ kind: "readiness", id: "star-prior" });
  });

  it("registers the refresh and every open fix in the palette", async () => {
    const Overview = await tab("Overview");
    show(<Overview />, "/realism/overview");
    await screen.findByRole("button", { name: "Activate model" });
    await waitFor(() => expect(paletteIds()).toContain("realism-overview-galaxy-model"));
    expect(paletteIds()).toEqual(expect.arrayContaining(["realism-overview-refresh", "realism-overview-comparison-cache"]));
    expect(paletteIds()).not.toContain("realism-overview-noise-model");    // ok items have nothing to fix
    act(() => runPalette("realism-overview-galaxy-model"));
    await answer(/Activate model/, "Activate model");
    await waitFor(() => expect(posts("/api/galaxy-distributions/activate")).toHaveLength(1));
  });

  it("shows the server's error text when the overview fails", async () => {
    routes["GET /api/realism/overview"] = () => ({ status: 500, body: { ok: false, error: "population cache unreadable" } });
    const Overview = await tab("Overview");
    show(<Overview />, "/realism/overview");
    expect(await screen.findByText("population cache unreadable", {}, { timeout: 4000 })).toBeTruthy();
  });
});

/* ── the ONE shared header ────────────────────────────────────────────── */

describe("shared header", () => {
  it("offers include-training only once a training catalog exists, else a 'no training' state and ONE sync", async () => {
    const Galaxies = await tab("Galaxies");
    show(<><RealismHeader /><Galaxies /></>, "/realism/galaxies");
    const toggle = await screen.findByRole("switch", { name: "Include training catalog" });
    expect((toggle as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText("no training")).toBeTruthy();
    await waitFor(() => expect((screen.getByRole("button", { name: /Sync training catalog/ }) as HTMLButtonElement).disabled).toBe(false));
    const syncs = screen.getAllByRole("button", { name: /Sync training catalog/ });
    expect(syncs).toHaveLength(1);
    fireEvent.click(syncs[0]);
    await answer(/Sync training catalog/, "Sync training catalogue");
    await waitFor(() => expect(posts("/api/population-comparison/sync-training-catalog")).toHaveLength(1));
    expect(posts("/api/population-comparison/sync-training-catalog")[0].form).toEqual({ rebuild: "1" });
  });

  it("puts include_training in the URL and the requests", async () => {
    routes["GET /api/realism/overview"] = () => ({ body: { ...OVERVIEW, training: { ...OVERVIEW.training, available: true } } });
    const Galaxies = await tab("Galaxies");
    show(<><RealismHeader /><Galaxies /></>, "/realism/galaxies");
    const toggle = await screen.findByRole("switch", { name: "Include training catalog" });
    await waitFor(() => expect((toggle as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(toggle);
    await waitFor(() => expect(params().get("training")).toBe("1"));
    await waitFor(() => expect(gets("/api/galaxy-distributions?include_training=1").length).toBeGreaterThan(0));
    expect(await screen.findByText("train + test + val")).toBeTruthy();
  });

  it("keeps the self-connecting training sync enabled offline, saying it connects first", async () => {
    routes["GET /api/fasrc/status"] = () => ({ body: { ssh_connected: false, last_error: "timed out" } });
    show(<RealismHeader />, "/realism/galaxies");
    const sync = await screen.findByRole("button", { name: /Sync training catalog/ });
    await waitFor(() => expect(gets("/api/fasrc/status").length).toBeGreaterThan(0));
    await new Promise((r) => setTimeout(r, 20));
    expect((sync as HTMLButtonElement).disabled).toBe(false);
    // Narrow panes keep a short state badge next to the switch.
    expect(screen.getByText("no train")).toBeTruthy();
  });

  it("is hidden on the tabs without a training variant", async () => {
    show(<RealismHeader />, "/realism/noise");
    await new Promise((r) => setTimeout(r, 20));
    expect(screen.queryByRole("switch", { name: "Include training catalog" })).toBeNull();
  });
});

/* ── noise ────────────────────────────────────────────────────────────── */

describe("noise", () => {
  it("lists the measured positions, opens one in the inspector and links the atlas", async () => {
    const Noise = await tab("Noise");
    show(<Noise />, "/realism/noise");
    expect(await screen.findByText("Sky noise level per band")).toBeTruthy();
    expect(screen.getByText("Depth steps inside one field")).toBeTruthy();
    const table = screen.getByRole("grid", { name: "Noise positions" });
    fireEvent.click(within(table).getByText("102021990"));
    expect(useInspector.getState().current).toEqual({ kind: "noisepos", id: "102021990" });
    const sky = screen.getAllByRole("link", { name: /sky/i }).map((a) => a.getAttribute("href"));
    expect(sky.some((h) => h?.startsWith("/sky/atlas?") && h.includes("layers=q1-tiles:0.3,noise-positions"))).toBe(true);
    fireEvent.click(screen.getByRole("radio", { name: "VIS·Y" }));
    await waitFor(() => expect(params().get("pair")).toBe("VIS|Y_E"));
  });

  it("registers the jitter toggle and the atlas link in the palette", async () => {
    const Noise = await tab("Noise");
    show(<Noise />, "/realism/noise");
    await screen.findByText("102021990");
    act(() => runPalette("noise-jitter"));
    await waitFor(() => expect(params().get("jitter")).toBe("1"));
    act(() => runPalette("noise-sky"));
    await waitFor(() => expect(lastLocation).toBe("/sky/atlas?layers=q1-tiles:0.3,noise-positions"));
  });

  it("the noisepos inspector shows the band levels, the 4×4 sub-grids and the seam step", async () => {
    const { NoisePositionInspector } = await import("./inspectors");
    show(<NoisePositionInspector id="102021990" />, "/realism/noise");
    expect(await screen.findByText("EDF-S")).toBeTruthy();
    expect(screen.getAllByRole("table", { name: /sub-tile levels/ })).toHaveLength(4);
    expect(screen.getAllByText("seam ×1.20")).toHaveLength(3);
    const link = screen.getByRole("link", { name: "Open on sky" }).getAttribute("href") ?? "";
    expect(link).toContain("inspect=source:noise-positions/102021990");
    expect(link).toContain("ra=61.2");
  });
});

/* ── galaxies ─────────────────────────────────────────────────────────── */

describe("galaxies", () => {
  it("draws the marginals in PARAMETER_ORDER on a two-column grid with the shape panel", async () => {
    expect([...PARAMETER_ORDER]).toEqual(["magnitude", "radius", "color_vis_y", "color_y_j", "color_j_h"]);
    const Galaxies = await tab("Galaxies");
    const { container } = show(<Galaxies />, "/realism/galaxies");
    expect(await screen.findByLabelText("Apparent brightness")).toBeTruthy();
    const panels = [...container.querySelectorAll(".rl-plot-grid > .rl-panel")].map((p) => p.getAttribute("aria-label"));
    expect(panels).toEqual(["Apparent brightness", "Angular size", "VIS − Y colour", "Y − J colour", "J − H colour", "Normalized half-light shape"]);
    expect(screen.getByText("Q1 count turnover")).toBeTruthy();
    expect(screen.queryByText("Euclid · Kron")).toBeNull();            // only the useful brightness keys
    // No walls of text: the notes and curve definitions live in each panel's popover.
    expect(screen.queryByText("PHZ-weighted brackets")).toBeNull();
    expect(container.querySelectorAll(".rl-plot-grid .rl-note, .rl-plot-grid .rl-defs")).toHaveLength(0);
    expect(screen.queryByText(/fixed joins VIS/)).toBeNull();
    const about = async (title: string) => {
      fireEvent.click(screen.getByRole("button", { name: `About ${title}` }));
      return within(await screen.findByRole("dialog", { name: `About ${title}` }));
    };
    // The generation law is the three-segment bright bridge / main / flat law, with its joins and slopes.
    const brightness = await about("Apparent brightness");
    expect(brightness.getByText("PHZ-weighted brackets")).toBeTruthy();
    expect(brightness.getByText(/Generation law · VIS · three-segment bright bridge\/main\/flat law/)).toBeTruthy();
    expect(brightness.getByText(/fixed joins VIS 19\.50 \/ 20\.50 \/ 21\.50 · bridge slopes 0\.500 \/ 0\.420 \/ 0\.370/)).toBeTruthy();
    fireEvent.keyDown(document.activeElement ?? document.body, { key: "Escape" });
    // The shape panel: the Q1 magnitude mix plus the full faint extension.
    const shape = await about("Normalized half-light shape");
    expect(shape.getByText(/Q1 magnitude mix/)).toBeTruthy();
    expect(shape.getByText(/full faint extension/)).toBeTruthy();
  });

  it("picks curves with compact chips that double as the legend, kept in the URL", async () => {
    const Galaxies = await tab("Galaxies");
    const { container } = show(<Galaxies />, "/realism/galaxies");
    const picker = await screen.findByRole("group", { name: "Brightness curves" });
    const brightness = screen.getByRole("article", { name: "Apparent brightness" });
    expect(brightness.querySelector(".plot-legend")).toBeNull();       // the chips are the legend
    const chips = within(picker).getAllByRole("button", { pressed: true });
    expect(chips.length).toBeGreaterThan(0);
    expect(chips[0].getAttribute("title")).toMatch(/Selection:/);     // the definition is the chip's tooltip
    fireEvent.click(chips[0]);
    await waitFor(() => expect(params().get("mag")).toBeTruthy());
    const radius = screen.getByRole("group", { name: "Radius curves" });
    fireEvent.click(within(radius).getByRole("button", { name: "Hide every Half-light radius curve" }));
    await waitFor(() => expect(params().get("re")).toBe("synthetic_clean_half_light"));
    expect(container.querySelectorAll(".rl-chip").length).toBeGreaterThan(3);
  });

  it("registers the view switches, the rebuild and the cones link in the palette", async () => {
    const Galaxies = await tab("Galaxies");
    show(<Galaxies />, "/realism/galaxies");
    await screen.findByLabelText("Apparent brightness");
    expect(paletteIds()).toEqual(expect.arrayContaining(["galaxies-view-joint", "galaxies-query", "galaxies-rebuild", "galaxies-activate"]));
    act(() => runPalette("galaxies-view-relations"));
    await waitFor(() => expect(params().get("view")).toBe("relations"));
    act(() => runPalette("galaxies-rebuild"));
    await waitFor(() => expect(posts("/api/galaxy-distributions/build")).toHaveLength(1));
    act(() => runPalette("galaxies-query"));
    await answer(/Query MER \+ PHZ/, "Query");
    await waitFor(() => expect(posts("/api/galaxy-distributions/query-q1-counts")).toHaveLength(1));
    act(() => runPalette("galaxies-cones"));
    await waitFor(() => expect(lastLocation).toBe("/sky/atlas?layers=q1-tiles:0.2,population-cones"));
  });

  it("reads log10 parameters on physical log axes with their x_domain, and unit-integral radii as 'normalized probability / dex'", () => {
    const radius = galaxyPayload().parameters.radius;
    expect(xAxisOf(radius)).toMatchObject({ scale: "log", label: "radius (arcsec, log scale)" });
    const [lo, hi] = xDomainOf(radius, []);
    expect(lo).toBeCloseTo(10 ** -2.4);
    expect(hi).toBeCloseTo(10);
    const shapes = radiusEntries(radius, USEFUL_SHAPE_KEYS);
    expect(shapes.map(([, c]) => c.radius_type)).toEqual(["half_light_shape", "half_light_shape", "half_light_shape"]);
    expect(radiusYLabel(radius, shapes)).toBe("normalized probability / dex (log scale)");
    expect(xAxisOf(galaxyPayload().parameters.color_vis_y).scale).toBe("linear");
  });

  it("has exactly one Q1 MER + PHZ action, its galaxy-only notes and progress phases — and no retired actions", async () => {
    const Galaxies = await tab("Galaxies");
    show(<Galaxies />, "/realism/galaxies?view=model");
    const query = await screen.findByRole("button", { name: "Query MER + PHZ" });
    expect(screen.getAllByRole("button", { name: /Query MER \+ PHZ/ })).toHaveLength(1);
    for (const retired of [/PHZ.?recovery/i, /multi.?cone/i, /fit.?euclid/i, /fit.?q1.?counts/i]) {
      expect(screen.queryByRole("button", { name: retired })).toBeNull();
    }
    expect(screen.getByText("Rₑ brackets")).toBeTruthy();
    expect(screen.getByText("POINT_LIKE_FLAG IS NULL")).toBeTruthy();
    expect(screen.getByText(/never refreshes star caches/)).toBeTruthy();
    expect(screen.queryByText(/stellar bins|stellar colou?rs|Gaia/i)).toBeNull();
    const phases = within(screen.getByRole("list", { name: "Progressive magnitude-bin sampling phases" })).getAllByRole("listitem");
    expect(phases.map((p) => p.getAttribute("data-state"))).toEqual(["cached", "cached", "cached", "waiting", "waiting"]);
    // Activation goes through /api/galaxy-distributions/activate.
    fireEvent.click(screen.getByRole("button", { name: "Activate model" }));
    await answer(/Activate this galaxy model/, "Activate");
    await waitFor(() => expect(posts("/api/galaxy-distributions/activate")).toHaveLength(1));
    fireEvent.click(query);
    await answer(/Query MER \+ PHZ/, "Query");
    await waitFor(() => expect(posts("/api/galaxy-distributions/query-q1-counts")).toHaveLength(1));
    // While the query runs, activation waits for it.
    await waitFor(() => expect((screen.getByRole("button", { name: "Activate model" }) as HTMLButtonElement).disabled).toBe(true));
  });

  it("shows the brightness–radius, FWHM and colour relations of the fitted model", async () => {
    const Galaxies = await tab("Galaxies");
    show(<Galaxies />, "/realism/galaxies?view=relations");
    expect(await screen.findByText("Joint brightness–radius relation")).toBeTruthy();
    expect(screen.getByText(/one fitted straight truncated-Gaussian conditional law/)).toBeTruthy();
    expect(screen.getByText("VIS magnitude–MER aperture FWHM relation")).toBeTruthy();
    expect(screen.getByText(/Bars only where Q1 populates the bin; elsewhere the model uses the nearest populated bin/)).toBeTruthy();
    expect(screen.getByText("Empirical colour model")).toBeTruthy();
    expect(screen.getByText("model mean VIS − Y, Y − J, J − H")).toBeTruthy();
    expect(screen.getByText("observed (solid) vs reported noise (dashed)")).toBeTruthy();
  });

  it("renders the corner plot with both triangles; a cell opens that pair in the explorer, which can swap axes", async () => {
    routes["GET /api/galaxy-distributions/joint-pair?x=vis&y=log_re&r=25%3A41234%3A18%2C26"] = () => ({ body: PAIR });
    routes["GET /api/galaxy-distributions/joint-pair?x=vis&y=log_sfr&r=25%3A41234%3A18%2C26"] = () => ({ body: { ...PAIR, y: PAIR.x } });
    routes["GET /api/galaxy-distributions/joint-pair?x=log_sfr&y=vis&r=25%3A41234%3A18%2C26"] = () => ({ body: PAIR });
    const Galaxies = await tab("Galaxies");
    const { container } = show(<Galaxies />, "/realism/galaxies?view=joint");
    expect(await screen.findByText("Joint distributions")).toBeTruthy();
    const triangles = [...container.querySelectorAll(".rl-corner__cell")].map((c) => c.getAttribute("data-triangle"));
    expect(triangles.filter((t) => t === "lower")).toHaveLength(3);
    expect(triangles.filter((t) => t === "upper")).toHaveLength(3);
    expect(screen.getByText("Joint distribution explorer")).toBeTruthy();
    await waitFor(() => expect(gets("/api/galaxy-distributions/joint-pair?x=vis&y=log_re").length).toBe(1));
    fireEvent.click(screen.getByRole("button", { name: "Explore VIS 2FWHM vs log₁₀ SFR (Euclid Q1)" }));
    await waitFor(() => expect(params().get("py")).toBe("log_sfr"));
    await waitFor(() => expect(gets("/api/galaxy-distributions/joint-pair?x=vis&y=log_sfr").length).toBe(1));
    fireEvent.click(screen.getByRole("button", { name: "Swap axes" }));
    await waitFor(() => expect(params().get("px")).toBe("log_sfr"));
    expect(params().get("py")).toBe("vis");
    expect(screen.getByText("Magnitude × radius")).toBeTruthy();
  });

  it("draws the magnitude × radius maps: gray Q1, blue dashed generated, red solid model, labelled by enclosed mass", () => {
    const series = jointMapSeries(JOINT_MAPS);
    const q1 = series.filter((s) => s.key === "q1");
    const generated = series.filter((s) => s.key === "synthetic");
    const model = series.filter((s) => s.key === "model");
    expect(q1.every((s) => s.color === C.cross && !s.dash)).toBe(true);
    expect(generated.every((s) => s.color === categorical(0) && s.dash?.length)).toBe(true);
    expect(model.every((s) => s.color === categorical(6) && !s.dash)).toBe(true);
    expect(jointColor("q1", "maps")).toBe(C.cross);
    expect(q1.map((s) => s.label)).toEqual(["10%", "50%", "80%", "95%", "99%", "99.5%", "99.9%"]);
    expect(q1[0].y[0]).toBeCloseTo(10 ** (-0.5 + 0.2 * 0.4));    // radius drawn in arcsec (log axis)
  });

  it("exports the publication plate (SVG / PDF / PNG) and previews it", async () => {
    const Galaxies = await tab("Galaxies");
    show(<Galaxies />, "/realism/galaxies?view=figure");
    expect(await screen.findByText("Galaxy population diagnostics · 2 × 2")).toBeTruthy();
    for (const f of ["SVG", "PDF", "PNG"]) {
      const a = screen.getByRole("link", { name: f });
      expect(a.getAttribute("href")).toBe(`/view/galaxy-distribution-plate?include_training=0&format=${f.toLowerCase()}&dpi=300`);
      expect(a.getAttribute("download")).toBe(`euclidpolish_galaxy_distributions_2x2.${f.toLowerCase()}`);
    }
    const img = screen.getByRole("img", { name: /Four-panel figure/ });
    expect(img.getAttribute("src")).toContain("format=svg&dpi=300&inline=1&v=25");
  });
});

/* ── stars ────────────────────────────────────────────────────────────── */

describe("stars", () => {
  it("queries MER + PHZ + Gaia and fits from the cached colour sample, with no galaxy selection or link", async () => {
    const Stars = await tab("Stars");
    const { container } = show(<Stars />, "/realism/stars?view=prior");
    const query = await screen.findByRole("button", { name: "Query stars · MER + PHZ + Gaia" });
    expect(screen.getByText("No galaxy selection is used by this action.")).toBeTruthy();
    expect(screen.getByText(/keeps Q1 at 0\.1-mag resolution and bins the smaller Gaia shape sample at 0\.5 mag/)).toBeTruthy();
    expect(screen.getByText("5,963 Euclid candidates")).toBeTruthy();       // from color_sample, not availability
    expect(container.querySelector('a[href^="/realism/galaxies"]')).toBeNull();
    expect(screen.queryByText(/random cones?/i)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Fit stellar prior from cached data" }));
    await waitFor(() => expect(posts("/api/star-distribution/fit")).toHaveLength(1));
    fireEvent.click(query);
    await answer(/Query stars/, "Query");
    await waitFor(() => expect(posts("/api/star-distribution/query")).toHaveLength(1));
    const gaia = screen.getAllByRole("link", { name: "On sky" }).map((a) => a.getAttribute("href") ?? "");
    expect(gaia.some((h) => h.includes("layers=q1-tiles:0.2,gaia-fields") && h.includes("inspect=source:gaia-fields/EDF-N"))).toBe(true);
  });

  it("shows the density panels (Q1 at 0.1 mag vs the Gaia shape sample at 0.5 mag) and keeps the legend in the URL", async () => {
    const Stars = await tab("Stars");
    const { container } = show(<Stars />, "/realism/stars");
    expect(await screen.findByText("Stellar density in Euclid magnitude and colour")).toBeTruthy();
    expect(screen.getByText("Q1 at 0.1 mag · Gaia shape sample at 0.5 mag · guides mark the fitted regions")).toBeTruthy();
    expect(container.querySelectorAll(".rl-panel")).toHaveLength(7);
    fireEvent.click(screen.getByRole("button", { name: "native Gaia G_AB" }));
    await waitFor(() => expect(params().get("shide")).toBe("Gaia G_AB"));
    expect(container.querySelector('a[href^="/realism/galaxies"]')).toBeNull();
  });

  it("registers the query, fit and Gaia atlas link in the palette", async () => {
    const Stars = await tab("Stars");
    show(<Stars />, "/realism/stars");
    await screen.findByText("Stellar density in Euclid magnitude and colour");
    expect(paletteIds()).toEqual(expect.arrayContaining(["stars-query", "stars-fit", "stars-activate", "stars-gaia-sky"]));
    act(() => runPalette("stars-fit"));
    await waitFor(() => expect(posts("/api/star-distribution/fit")).toHaveLength(1));
    act(() => runPalette("stars-query"));
    await answer(/Query stars/, "Query");
    await waitFor(() => expect(posts("/api/star-distribution/query")).toHaveLength(1));
    act(() => runPalette("stars-gaia-sky"));
    await waitFor(() => expect(lastLocation).toMatch(/^\/sky\/atlas\?.*layers=q1-tiles:0\.2,gaia-fields/));
  });

  it("shows no made-up Q1 footprint before the stellar query ran", async () => {
    const base = starPayload();
    routes["GET /api/star-distribution?include_training=0"] = () => ({ body: { ...base, q1_counts: null } });
    const Stars = await tab("Stars");
    show(<Stars />, "/realism/stars?view=prior");
    await screen.findByRole("button", { name: "Query stars · MER + PHZ + Gaia" });
    expect(screen.queryByText(/63\.1 deg²/)).toBeNull();
  });

  it("points an empty distribution at the prior workflow", async () => {
    routes["GET /api/star-distribution?include_training=0"] = () => ({ body: starPayload({ distribution: null }) });
    const Stars = await tab("Stars");
    show(<Stars />, "/realism/stars?view=colours");
    fireEvent.click(await screen.findByRole("button", { name: "Open the prior workflow" }));
    await waitFor(() => expect(params().get("view")).toBe("prior"));
  });
});

/* ── pixels (field statistics) ────────────────────────────────────────── */

describe("pixels", () => {
  it("shows the pixel figures without any calibration lanes, random cones or Euclid population", async () => {
    const Pixels = await tab("Pixels");
    show(<Pixels />, "/realism/pixels");
    for (const title of ["Brightness distribution", "Pixel quantile profile", "Angular-scale power", "Mean brightness vs field variation",
      "Background vs robust noise", "Inter-band pixel correlation", "Scale-spectrum similarity", "Median field metrics"]) {
      expect(await screen.findByText(title)).toBeTruthy();
    }
    expect(screen.queryByText(/random cones?/i)).toBeNull();
    expect(screen.queryByText(/calibration/i)).toBeNull();
    expect(screen.queryByText(/VIS noise/i)).toBeNull();
    expect(screen.queryByText(/Euclid population|Euclid MER \+ PHZ population/i)).toBeNull();
    fireEvent.click(within(screen.getByRole("group", { name: "Bands and samples" })).getByRole("button", { name: "VIS" }));
    await waitFor(() => expect(params().get("hide")).toBe("VIS"));
  });

  it("shows the source detection the build measures", async () => {
    const Pixels = await tab("Pixels");
    show(<Pixels />, "/realism/pixels?view=detection");
    expect(await screen.findByText("VIS source detection per field")).toBeTruthy();
    const rows = screen.getAllByRole("row");
    const synthetic = rows.find((r) => r.textContent?.startsWith("synthetic LR"))!;
    expect(synthetic.textContent).toContain("81.5%");          // 53 matched of 65 truth galaxies
    expect(synthetic.textContent).toContain("8.1%");           // 6 negative islands per 74 detections
    expect(rows.find((r) => r.textContent?.startsWith("real Euclid LR"))!.textContent).toContain("no truth");
    expect(screen.getByText("Galaxy completeness per field")).toBeTruthy();
  });

  it("rebuilds only after a confirmation, and explains a missing cache", async () => {
    routes["GET /api/population-comparison?include_training=0"] = () => ({ body: pixelsPayload({ comparison: null }) });
    const Pixels = await tab("Pixels");
    show(<Pixels />, "/realism/pixels");
    expect(await screen.findByText("The field-statistics cache has not been built")).toBeTruthy();
    fireEvent.click(screen.getAllByRole("button", { name: "Measure fields" })[0]);
    await answer(/Rebuild the field statistics/, "Rebuild");
    await waitFor(() => expect(posts("/api/population-comparison/build")).toHaveLength(1));
  });

  it("registers the rebuild (behind its confirmation) and the band toggles in the palette", async () => {
    const Pixels = await tab("Pixels");
    show(<Pixels />, "/realism/pixels");
    expect(await screen.findByText("Brightness distribution")).toBeTruthy();
    expect(paletteIds()).toEqual(expect.arrayContaining(["pixels-build", "pixels-band-VIS", "pixels-band-H_E"]));
    act(() => runPalette("pixels-band-J_E"));
    await waitFor(() => expect(params().get("hide")).toBe("J_E"));
    act(() => runPalette("pixels-build"));
    await answer(/Rebuild the field statistics/, "Rebuild");
    await waitFor(() => expect(posts("/api/population-comparison/build")).toHaveLength(1));
  });

  it("gives every figure exact axis bounds one click away (a RangeSlider per axis, reset to the full range)", async () => {
    const Pixels = await tab("Pixels");
    show(<Pixels />, "/realism/pixels");
    expect(await screen.findByText("Brightness distribution")).toBeTruthy();
    fireEvent.click(screen.getAllByRole("button", { name: "bounds" })[0]);
    const dlg = within(await screen.findByRole("dialog", { name: "Brightness distribution: axis bounds" }));
    for (const axis of ["x", "y"]) {
      expect(dlg.getByRole("slider", { name: `Brightness distribution ${axis} minimum` })).toBeTruthy();
      expect(dlg.getByRole("slider", { name: `Brightness distribution ${axis} maximum` })).toBeTruthy();
    }
    expect((dlg.getByRole("button", { name: "Full range" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.keyDown(dlg.getByRole("slider", { name: "Brightness distribution x minimum" }), { key: "ArrowRight" });
    await waitFor(() => expect(screen.getAllByRole("button", { name: "custom bounds" })).toHaveLength(1));
    fireEvent.click(dlg.getByRole("button", { name: "Full range" }));
    await waitFor(() => expect(screen.queryByRole("button", { name: "custom bounds" })).toBeNull());
  });

  it("filters bands and samples, and maps real dots to their parent pointing", () => {
    const v = visibleFrom(["VIS", "real"]);
    const series = histogramSeries(FIELDS, v);
    expect(series.map((s) => s.key)).toEqual(["Y_E:synthetic", "J_E:synthetic", "H_E:synthetic"]);
    const points = relationPoints(FIELDS, "mean_std", visibleFrom([]));
    expect(points.filter((p) => p.sample === "real").map((p) => p.parent)).toContain("parent-3");
  });
});

/* ── visual (synthetic–real) ──────────────────────────────────────────── */

describe("visual", () => {
  it("compares the multipoint archive collection with synthetic dirty LR, on one Lupton transfer", async () => {
    const Visual = await tab("Visual");
    show(<Visual />, "/realism/visual");
    expect(await screen.findByTestId("viewer-archive-fields")).toBeTruthy();
    expect(await screen.findByTestId("viewer-sky")).toBeTruthy();
    const real = hoisted.viewers.find((v) => v.props.collection === "archive-fields")!;
    const syn = hoisted.viewers.find((v) => v.props.collection === "sky")!;
    expect(real.props.tiers).toEqual(["lr"]);
    expect(syn.props).toMatchObject({ params: { subset: "test" }, tiers: ["dirty"] });
    expect(hoisted.viewers.some((v) => v.props.collection === "real-field")).toBe(false);
    expect(gets("/viewer/meta/archive-fields").length).toBeGreaterThan(0);
    expect(calls.some((c) => c.url.includes("/api/inference/field.json"))).toBe(false);
    // Explicit defaults: the viewers hold the page's transfer, not the Display panel's.
    expect(real.api.setView).toHaveBeenCalledWith({ color: "lupton", knee: 100, gain: 1 });
    expect(syn.api.setView).toHaveBeenCalledWith({ color: "lupton", knee: 100, gain: 1 });
    expect(screen.getByTestId("step-archive_field_sample")).toBeTruthy();
  });

  it("applies a colour / knee edit made in one viewer's toolbar to the other", async () => {
    const Visual = await tab("Visual");
    show(<Visual />, "/realism/visual");
    await screen.findByTestId("viewer-sky");
    const real = hoisted.viewers.find((v) => v.props.collection === "archive-fields")!;
    const syn = hoisted.viewers.find((v) => v.props.collection === "sky")!;
    const state = (color: string, knee = 100, gain = 1) => ({ index: 0, id: "17", color, knee, gain, tiers: ["lr"] });
    act(() => { real.props.onState?.(state("lupton")); syn.props.onState?.(state("lupton")); });
    syn.api.setView.mockClear();
    act(() => { real.props.onState?.(state("J_E", 250)); });
    await waitFor(() => expect(syn.api.setView).toHaveBeenLastCalledWith({ color: "J_E", knee: 250, gain: 1 }));
    expect(params().get("c")).toBe("J_E");
    expect(params().get("k")).toBe("250");
    // Unlocked, an edit stays in its own viewer.
    fireEvent.click(screen.getByRole("switch", { name: "Same transfer" }));
    syn.api.setView.mockClear();
    act(() => { real.props.onState?.(state("VIS", 250)); });
    await new Promise((r) => setTimeout(r, 20));
    expect(syn.api.setView).not.toHaveBeenCalled();
  });

  it("puts both lanes side by side under ONE shared display row (the viewers carry navigation only)", async () => {
    const Visual = await tab("Visual");
    show(<Visual />, "/realism/visual");
    await screen.findByTestId("viewer-sky");
    const real = hoisted.viewers.find((v) => v.props.collection === "archive-fields")!;
    const syn = hoisted.viewers.find((v) => v.props.collection === "sky")!;
    for (const v of [real, syn]) expect(v.props).toMatchObject({ toolbar: "none", nav: true });
    const row = screen.getByRole("toolbar", { name: "Shared display of both lanes" });
    // The shared row is the colour control of both lanes.
    real.api.setView.mockClear(); syn.api.setView.mockClear();
    fireEvent.click(within(row).getByRole("radio", { name: "VIS" }));
    await waitFor(() => expect(params().get("c")).toBe("VIS"));
    await waitFor(() => expect(real.api.setView).toHaveBeenLastCalledWith({ color: "VIS", knee: 100, gain: 1 }));
    expect(syn.api.setView).toHaveBeenLastCalledWith({ color: "VIS", knee: 100, gain: 1 });
    // An exact knee (a training knee) is one typed entry away.
    const kneeBox = within(row).getByRole("textbox", { name: "Shared knee (e⁻)" });
    fireEvent.change(kneeBox, { target: { value: "3" } });
    fireEvent.keyDown(kneeBox, { key: "Enter" });
    await waitFor(() => expect(params().get("k")).toBe("3"));
    await waitFor(() => expect(syn.api.setView).toHaveBeenLastCalledWith({ color: "VIS", knee: 3, gain: 1 }));
    expect(real.api.setView).toHaveBeenLastCalledWith({ color: "VIS", knee: 3, gain: 1 });
    // The knee slider spans the research grid 0.1–10⁴ e⁻.
    const slider = within(row).getByRole("slider", { name: "Shared knee" });
    expect(slider.getAttribute("aria-valuetext")).toBe("3 e⁻");
    // One button fits both lanes (the viewers' own zoom stays on their keys, listed in ⓘ).
    fireEvent.click(within(row).getByRole("button", { name: "Fit both" }));
    expect(real.api.resetView).toHaveBeenCalledTimes(1);
    expect(syn.api.resetView).toHaveBeenCalledTimes(1);
    // Narrow blocks put knee and brightness in a Display menu (the same controls).
    fireEvent.click(within(row).getByRole("button", { name: /^Display/ }));
    const pop = await screen.findByRole("dialog", { name: "Knee and brightness of both lanes" });
    expect(within(pop).getByRole("slider", { name: "Shared brightness" })).toBeTruthy();
    fireEvent.keyDown(pop, { key: "Escape" });
    // The synthetic subset lives in the synthetic lane's caption.
    fireEvent.click(screen.getByRole("radio", { name: "Validate" }));
    await waitFor(() => expect(params().get("sub")).toBe("validate"));
  });

  it("resets both viewers to the default transfer, and the URL with them", async () => {
    const Visual = await tab("Visual");
    show(<Visual />, "/realism/visual");
    await screen.findByTestId("viewer-sky");
    const real = hoisted.viewers.find((v) => v.props.collection === "archive-fields")!;
    const syn = hoisted.viewers.find((v) => v.props.collection === "sky")!;
    const state = (knee: number, gain = 1) => ({ index: 0, id: "17", color: "lupton", knee, gain, tiers: ["lr"] });
    act(() => { real.props.onState?.(state(100)); syn.props.onState?.(state(100)); });
    act(() => { syn.props.onState?.(state(1256, 2)); });
    await waitFor(() => expect(params().get("k")).toBe("1256"));
    expect(params().get("g")).toBe("2");
    act(() => { real.props.onState?.(state(1256, 2)); });           // the real viewer follows
    real.api.setView.mockClear();
    syn.api.setView.mockClear();
    fireEvent.click(screen.getByRole("button", { name: "Reset the shared transfer (Lupton, default knee)" }));
    await waitFor(() => expect(real.api.setView).toHaveBeenLastCalledWith({ color: "lupton", knee: 100, gain: 1 }));
    expect(syn.api.setView).toHaveBeenLastCalledWith({ color: "lupton", knee: 100, gain: 1 });
    expect(params().get("k")).toBeNull();
    expect(params().get("g")).toBeNull();
    // The viewers' echo of the reset does not write the default back into the URL.
    act(() => { real.props.onState?.(state(100)); syn.props.onState?.(state(100)); });
    await new Promise((r) => setTimeout(r, 20));
    expect(params().get("k")).toBeNull();
    // No running transfer text (the controls show it); the Display menu's tooltip says it.
    expect(screen.getByRole("button", { name: /^Display/ }).getAttribute("title")).toBe("Knee 100 e⁻, brightness ×1");
  });

  it("registers its palette actions: lock, colour, reset and the archive sync", async () => {
    const Visual = await tab("Visual");
    show(<Visual />, "/realism/visual");
    await screen.findByTestId("viewer-sky");
    const ids = paletteIds();
    for (const id of ["visual-lock", "visual-lupton", "visual-vis", "visual-reset", "visual-sync"]) expect(ids).toContain(id);
    act(() => runPalette("visual-vis"));
    await waitFor(() => expect(params().get("c")).toBe("VIS"));
    act(() => runPalette("visual-lock"));
    await waitFor(() => expect(params().get("lock")).toBe("0"));
    act(() => runPalette("visual-sync"));
    await answer(/Sync the archive fields/, "Sync");
    await waitFor(() => expect(posts("/api/archive-fields/sync")).toHaveLength(1));
  });

  it("syncs the archive fields from FASRC only after a confirmation", async () => {
    const Visual = await tab("Visual");
    show(<Visual />, "/realism/visual");
    fireEvent.click(await screen.findByRole("button", { name: "Sync archive fields from FASRC" }));
    await answer(/Sync the archive fields/, "Sync");
    await waitFor(() => expect(posts("/api/archive-fields/sync")).toHaveLength(1));
  });
});

/* ── inspectors, navigation ───────────────────────────────────────────── */

describe("inspectors and navigation", () => {
  it("the archivefield inspector shows the sample with a compact viewer and links to Visual and the atlas", async () => {
    const { ArchiveFieldInspector } = await import("./inspectors");
    show(<ArchiveFieldInspector id="18" />, "/realism/pixels");
    expect(await screen.findByText("parent-4")).toBeTruthy();
    expect(screen.getByTestId("viewer-archive-fields")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Open in Visual" }).getAttribute("href")).toBe("/realism/visual?v.real.id=18");
    expect(screen.getByRole("link", { name: "Open on sky" }).getAttribute("href")).toContain("layers=archive-fields");
  });

  it("keeps the Noise tab in the navigation and has no Round-trip entry anywhere", async () => {
    const { TABS } = await import("./index");
    expect(Object.keys(TABS)).toEqual(["overview", "noise", "galaxies", "stars", "pixels", "visual"]);
    const pages = allPages();
    expect(pages.some((p) => p.path === "/realism/noise")).toBe(true);
    expect(pages.some((p) => /round.?trip/i.test(p.label))).toBe(false);
  });
});
