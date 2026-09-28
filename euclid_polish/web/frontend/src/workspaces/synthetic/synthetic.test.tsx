/* Synthetic workspace (console regrouping): Status, the shared training
 * header, Galaxies, Stars, Noise and Fields against a mocked backend
 * (/api/realism/overview, /api/noise, /api/galaxy-distributions,
 * /api/star-distribution, /api/population-comparison, the archive / sky
 * viewer metas). The image viewer and the FASRC step cards are mocked.
 * Records, PSF and the TNG templates are in recordsPsf.test.tsx. Every
 * page follows the statistics rule (no stat tiles, at most one summary
 * line) and starts nothing on a visit. */
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
import REALISM_CSS from "./realism.css?raw";
import { PARAMETER_ORDER, USEFUL_SHAPE_KEYS, jointMapSeries, radiusEntries, radiusYLabel, xAxisOf, xDomainOf } from "./galaxies/model";
import { SyntheticHeader, resetTrainingPref } from "./header";
import { histogramSeries, relationPoints, visibleFrom } from "./fields/model";
import {
  ARCHIVE_META, FIELDS, JOINT_MAPS, NOISE, NOISE_POSITION, OVERVIEW, PAIR, SKY_META, galaxyPayload, pixelsPayload, starPayload,
} from "./testFixtures";

type ViewerProps = { collection: string; params?: Record<string, string>; tiers?: string[]; urlKey?: string;
  toolbar?: string; nav?: boolean; onReady?: (api: unknown) => void; onState?: (s: unknown) => void };
const hoisted = vi.hoisted(() => ({ viewers: [] as { props: ViewerProps; api: {
  setView: ReturnType<typeof vi.fn>; resetView: ReturnType<typeof vi.fn>; zoomBy: ReturnType<typeof vi.fn>; setTool: ReturnType<typeof vi.fn>;
} }[] }));

vi.mock("../../viewer", async () => {
  const React = await import("react");
  return {
    ImageViewer: (props: ViewerProps) => {
      React.useEffect(() => {
        const api = { setView: vi.fn(), goTo: vi.fn(), resetView: vi.fn(), zoomBy: vi.fn(), setTool: vi.fn() };
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
  StepCard: ({ step }: { step: { step_id: string } }) => <div data-testid={`step-${step.step_id}`} />,
  useStepsStatus: () => ({ data: { ssh_connected: true, steps: [] } }),
}));

type Reply = { status?: number; body: unknown };
let routes: Record<string, (form: Record<string, string>) => Reply>;
let calls: { url: string; method: string; form: Record<string, string> }[];
const posts = (u: string) => calls.filter((c) => c.method === "POST" && c.url === u);
const allPosts = () => calls.filter((c) => c.method === "POST" && c.url !== "/api/jobs?summary=1");
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
    "GET /api/tng/properties": () => ({ body: { present: false, files: { properties: { present: false }, atlas: { present: false } }, rows: [], columns: [] } }),
    "GET /api/tng/results": () => ({ body: { grid: { present: false, pulled_at: null, size_bytes: null }, stack: { present: false, pulled_at: null, size_bytes: null }, pull_job: null } }),
  };
  for (const [url, id] of [
    ["/api/galaxy-distributions/query-q1-counts", "gq"], ["/api/galaxy-distributions/activate", "ga"], ["/api/galaxy-distributions/fit", "gf"],
    ["/api/galaxy-distributions/build", "gb"], ["/api/galaxy-distributions/refresh-population-cones", "gc"],
    ["/api/star-distribution/query", "sq"], ["/api/star-distribution/fit", "sf"],
    ["/api/star-distribution/activate", "sa"], ["/api/population-comparison/build", "pb"], ["/api/archive-fields/sync", "as"],
    ["/api/population-comparison/sync-training-catalog", "ts"], ["/api/tng/radii/refresh", "tr"],
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
      <Routes><Route path="/synthetic/*" element={<>{el}<Spy /></>} /><Route path="*" element={<Spy />} /></Routes>
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
const noTiles = (container: HTMLElement) => expect(container.querySelector(".ui-stat, .ui-kpi, .rl-stats, .rl-score")).toBeNull();
const factRows = (title: string) =>
  within(screen.getByRole("heading", { name: title }).closest(".ui-facts") as HTMLElement).getAllByRole("group").map((r) => r.textContent);
/** Nothing but the jobs feed was POSTed (a visit starts nothing). */
const noJobs = () => expect(allPosts()).toEqual([]);

/* ── status ───────────────────────────────────────────────────────────── */

describe("status", () => {
  it("leads with the generation gate, then the two groups of rows, each with ONE verdict number", async () => {
    const Status = await tab("Status");
    const { container } = show(<Status />, "/synthetic/status");
    expect(await screen.findByText("Blocked by 1")).toBeTruthy();
    expect(screen.getAllByText(/activate a valid Gaia\+Euclid stellar calibration/).length).toBeGreaterThan(0);
    const generate = screen.getByRole("button", { name: "Generate validate+test on FASRC" }) as HTMLButtonElement;
    expect(generate.disabled).toBe(true);
    const blocks = within(screen.getByRole("region", { name: "Blocks generation" }));
    expect(blocks.getAllByRole("listitem").map((li) => li.querySelector(".syn-row__name")?.textContent))
      .toEqual(["Galaxies", "Stars", "Noise", "PSF", "TNG radii", "Saturation rule", "Training catalogue"]);
    const diagnostic = within(screen.getByRole("region", { name: "Diagnostic caches" }));
    expect(diagnostic.getAllByRole("listitem").map((li) => li.querySelector(".syn-row__name")?.textContent)).toEqual(["Galaxy plots", "Field statistics"]);
    await waitFor(() => expect(screen.getByText("generated 5.03 vs prior 5.08 stars arcmin⁻²")).toBeTruthy());
    expect(screen.getByText("generated 151 vs prior 152 galaxies arcmin⁻²")).toBeTruthy();
    expect(screen.getByText(/^blackout 20% → 90% of cores from 5× to 20× the well$/)).toBeTruthy();
    expect(screen.getByText("records predate it")).toBeTruthy();
    expect(screen.getAllByText("records built with it").length).toBeGreaterThan(0);
    // The OK rows are one quiet line; fingerprints live in the row inspector.
    expect(screen.queryByText(/g{12}/)).toBeNull();
    noTiles(container);
    noJobs();
  });

  it("fixes a row behind its confirmation, links each row to its tab and opens the row inspector", async () => {
    const Status = await tab("Status");
    show(<Status />, "/synthetic/status");
    fireEvent.click(await screen.findByRole("button", { name: "Activate model" }));
    await answer(/Activate model/, "Activate model");
    await waitFor(() => expect(posts("/api/galaxy-distributions/activate")).toHaveLength(1));
    const open = screen.getAllByRole("link", { name: /^Open / }).map((a) => a.getAttribute("href"));
    expect(open).toEqual(expect.arrayContaining(["/synthetic/galaxies", "/synthetic/stars", "/synthetic/noise",
      "/synthetic/psf?view=epsf", "/synthetic/galaxies?view=templates", "/synthetic/fields?view=stats"]));
    fireEvent.click(screen.getByRole("button", { name: "Inspect Stars" }));
    expect(useInspector.getState().current).toEqual({ kind: "readiness", id: "star-prior" });
  });

  it("disables the FASRC fixes while offline (the single Validate TNG radii copy lives here)", async () => {
    routes["GET /api/fasrc/status"] = () => ({ body: { ssh_connected: false, last_error: "timed out" } });
    const Status = await tab("Status");
    show(<Status />, "/synthetic/status");
    const validate = await screen.findByRole("button", { name: "Validate TNG radii on FASRC" });
    await waitFor(() => expect((validate as HTMLButtonElement).disabled).toBe(true));
    expect(screen.getAllByRole("button", { name: "Validate TNG radii on FASRC" })).toHaveLength(1);
    expect(screen.getAllByRole("button", { name: "Rebuild field statistics" })).toHaveLength(1);
  });

  it("registers the refresh, the generation and every open fix in the palette", async () => {
    const Status = await tab("Status");
    show(<Status />, "/synthetic/status");
    await screen.findByRole("button", { name: "Activate model" });
    await waitFor(() => expect(paletteIds()).toContain("status-fix-galaxy-model"));
    expect(paletteIds()).toEqual(expect.arrayContaining(["status-refresh", "status-generate", "status-fix-comparison-cache"]));
    expect(paletteIds()).not.toContain("status-fix-noise-model");    // ok rows have nothing to fix
    act(() => runPalette("status-fix-galaxy-model"));
    await answer(/Activate model/, "Activate model");
    await waitFor(() => expect(posts("/api/galaxy-distributions/activate")).toHaveLength(1));
  });

  it("shows the server's error text when the status fails", async () => {
    routes["GET /api/realism/overview"] = () => ({ status: 500, body: { ok: false, error: "population cache unreadable" } });
    const Status = await tab("Status");
    show(<Status />, "/synthetic/status");
    expect(await screen.findByText("population cache unreadable", {}, { timeout: 4000 })).toBeTruthy();
  });
});

/* ── the ONE shared header ────────────────────────────────────────────── */

describe("shared header", () => {
  it("offers include-training only once a training catalog exists, else a 'no training' state and ONE sync", async () => {
    show(<SyntheticHeader />, "/synthetic/galaxies");
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
    show(<><SyntheticHeader /><Galaxies /></>, "/synthetic/galaxies");
    const toggle = await screen.findByRole("switch", { name: "Include training catalog" });
    await waitFor(() => expect((toggle as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(toggle);
    await waitFor(() => expect(params().get("training")).toBe("1"));
    await waitFor(() => expect(gets("/api/galaxy-distributions?include_training=1").length).toBeGreaterThan(0));
    expect(await screen.findByText("train + test + val")).toBeTruthy();
  });

  it("keeps the self-connecting training sync enabled offline", async () => {
    routes["GET /api/fasrc/status"] = () => ({ body: { ssh_connected: false, last_error: "timed out" } });
    show(<SyntheticHeader />, "/synthetic/galaxies");
    const sync = await screen.findByRole("button", { name: /Sync training catalog/ });
    await waitFor(() => expect(gets("/api/fasrc/status").length).toBeGreaterThan(0));
    await new Promise((r) => setTimeout(r, 20));
    expect((sync as HTMLButtonElement).disabled).toBe(false);
    expect(screen.getByText("no train")).toBeTruthy();
  });

  it("is hidden where the training split changes nothing (Status, Noise, PSF, Fields)", async () => {
    for (const path of ["/synthetic/status", "/synthetic/noise", "/synthetic/psf", "/synthetic/fields"]) {
      const { unmount } = show(<SyntheticHeader />, path);
      await new Promise((r) => setTimeout(r, 10));
      expect(screen.queryByRole("switch", { name: "Include training catalog" })).toBeNull();
      unmount();
    }
  });
});

/* ── noise ────────────────────────────────────────────────────────────── */

describe("noise", () => {
  it("starts with how a scene gets its noise, then the level histograms with the field legend on top", async () => {
    const Noise = await tab("Noise");
    const { container } = show(<Noise />, "/synthetic/noise");
    expect(await screen.findByRole("heading", { name: "How a scene gets its noise" })).toBeTruthy();
    const how = container.querySelector(".syn-howto__text")?.textContent ?? "";
    expect(how).toMatch(/^Each scene takes the four band levels of one of the 3 measured Q1 positions, picked uniformly\. Its depth is scaled by ×0\.99–1\.01/);
    expect(screen.getByText("3 Q1 positions in EDF-N/S")).toBeTruthy();
    const card = screen.getByText("Sky noise level per band").closest(".ui-card") as HTMLElement;
    // The legend comes before the first histogram, and carries the field sizes.
    const legend = card.querySelector(".plot-legend, .legend, [role=group]");
    const firstFig = card.querySelector("figure");
    expect(legend && firstFig && legend.compareDocumentPosition(firstFig) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(within(card).getByRole("button", { name: "EDF-N · 2" })).toBeTruthy();
    expect(screen.getAllByText(/^median .* · p5–p95 /)).toHaveLength(4);
    expect(screen.queryByText("Level quantiles")).toBeNull();
    expect(screen.getByText("NOISE_MODEL v5 · Q1_R1 · retrieved 2026-09-19 · mer_noise_levels.json").className).toContain("ui-caption");
    noTiles(container);
    noJobs();
  });

  it("compares the realised background σ, synthetic vs real, as the tab's one summary line", async () => {
    const Noise = await tab("Noise");
    const { container } = show(<Noise />, "/synthetic/noise");
    expect(await screen.findByText("Realised noise, synthetic vs real")).toBeTruthy();
    await waitFor(() => expect(container.querySelectorAll(".ui-summary")).toHaveLength(1));
    // robust σ medians: synthetic 1.5·(1+i), real 1.5·(1.1+i) → 0.91, 0.95, 0.97, 0.98.
    expect(container.querySelector(".ui-summary")?.textContent).toBe("Background σ, synthetic ÷ real: VIS 0.91 · Y 0.95 · J 0.97 · H 0.98");
    const table = within(screen.getByRole("table", { name: "Background σ per band, real and synthetic" }));
    expect(table.getAllByRole("row")[1].textContent).toBe("VIS1.651.500.91");
    expect(screen.getByText("Background vs robust noise")).toBeTruthy();
  });

  it("keeps the last result of a stale cache behind a badge, measured on Fields", async () => {
    routes["GET /api/population-comparison?include_training=0"] = () => ({
      body: pixelsPayload({ comparison: null, previous: pixelsPayload().comparison }) });
    const Noise = await tab("Noise");
    show(<Noise />, "/synthetic/noise");
    expect(await screen.findByText("last result")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Measure on Fields" }).getAttribute("href")).toBe("/synthetic/fields?view=stats");
    noJobs();
  });

  it("puts the pair switch in the band-pair card's header and the 4 × 4 table on demand", async () => {
    const Noise = await tab("Noise");
    show(<Noise />, "/synthetic/noise");
    const card = (await screen.findByText("How the bands move together")).closest(".ui-card") as HTMLElement;
    fireEvent.click(within(card).getByRole("radio", { name: "VIS·Y" }));
    await waitFor(() => expect(params().get("pair")).toBe("VIS|Y_E"));
    expect(within(card).getByText("r = 0.40")).toBeTruthy();
    expect(screen.queryByRole("table", { name: "Correlation of the log levels" })).toBeNull();
    fireEvent.click(within(card).getByRole("button", { name: /Correlation of the log levels/ }));
    expect(await screen.findByRole("table", { name: "Correlation of the log levels" })).toBeTruthy();
  });

  it("lists the measured positions on demand; a row opens its inspector, the atlas is their map", async () => {
    const Noise = await tab("Noise");
    show(<Noise />, "/synthetic/noise");
    await screen.findByText("Sky noise level per band");
    expect(screen.queryByRole("grid", { name: "Noise positions" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /Measured positions/ }));
    const table = await screen.findByRole("grid", { name: "Noise positions" });
    fireEvent.click(within(table).getByText("102021990"));
    expect(useInspector.getState().current).toEqual({ kind: "noisepos", id: "102021990" });
    const sky = screen.getAllByRole("link", { name: /sky/i }).map((a) => a.getAttribute("href"));
    expect(sky.some((h) => h?.startsWith("/sky/atlas?") && h.includes("layers=q1-tiles:0.3,noise-positions"))).toBe(true);
  });

  it("holds the jitter switch, the MER downloader and vis_noise_sample in the How-this-is-produced drawer", async () => {
    const Noise = await tab("Noise");
    show(<Noise />, "/synthetic/noise?how=1");
    const cmds = await screen.findByRole("list", { name: "MER noise downloader commands" });
    expect(within(cmds).getAllByRole("listitem").map((li) => li.querySelector("code")?.textContent)).toEqual([
      "python scripts/download_mer_noise_levels.py plan", "python scripts/download_mer_noise_levels.py acquire --max-minutes 55",
      "python scripts/download_mer_noise_levels.py finalize"]);
    fireEvent.click(screen.getByRole("switch", { name: /scene-scale jitter/ }));
    await waitFor(() => expect(params().get("jitter")).toBe("1"));
    fireEvent.click(screen.getByRole("button", { name: /Real VIS noise fields/ }));
    expect(await screen.findByTestId("step-vis_noise_sample")).toBeTruthy();
    noJobs();
  });

  it("registers the jitter toggle, the drawer and the atlas link in the palette", async () => {
    const Noise = await tab("Noise");
    show(<Noise />, "/synthetic/noise");
    await screen.findByText("Sky noise level per band");
    act(() => runPalette("noise-jitter"));
    await waitFor(() => expect(params().get("jitter")).toBe("1"));
    act(() => runPalette("noise-how"));
    await waitFor(() => expect(params().get("how")).toBe("1"));
    act(() => runPalette("noise-sky"));
    await waitFor(() => expect(lastLocation).toBe("/sky/atlas?layers=q1-tiles:0.3,noise-positions"));
  });

  it("the noisepos inspector shows the band levels, the 4×4 sub-grids and the seam step", async () => {
    const { NoisePositionInspector } = await import("./inspectors");
    show(<NoisePositionInspector id="102021990" />, "/synthetic/noise");
    expect(await screen.findByText("EDF-S")).toBeTruthy();
    expect(screen.getAllByRole("table", { name: /sub-tile levels/ })).toHaveLength(4);
    expect(screen.getAllByText("seam ×1.20")).toHaveLength(3);
    const link = screen.getByRole("link", { name: "Open on sky" }).getAttribute("href") ?? "";
    expect(link).toContain("inspect=source:noise-positions/102021990");
  });
});

/* ── galaxies ─────────────────────────────────────────────────────────── */

describe("galaxies", () => {
  it("distributions: trust boxes above the wide brightness panel, then size, colours and shape, each with a caption", async () => {
    expect([...PARAMETER_ORDER]).toEqual(["magnitude", "radius", "color_vis_y", "color_y_j", "color_j_h"]);
    const Galaxies = await tab("Galaxies");
    const { container } = show(<Galaxies />, "/synthetic/galaxies");
    expect(await screen.findByLabelText("Apparent brightness")).toBeTruthy();
    const panels = [...container.querySelectorAll(".rl-plot-grid > .rl-panel")].map((p) => p.getAttribute("aria-label"));
    expect(panels).toEqual(["Apparent brightness", "Angular size", "VIS − Y colour", "Y − J colour", "J − H colour", "Normalized half-light shape"]);
    const brightness = screen.getByRole("article", { name: "Apparent brightness" });
    const trust = brightness.querySelector(".rl-trust")!;
    expect(trust.compareDocumentPosition(brightness.querySelector(".plot")!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(within(brightness).getByText(/^Q1: PHZ-weighted MER brackets over the 63\.1 deg² deep fields · generated: 5,489 galaxies in 200 test \+ validate scenes \(36\.4 arcmin²\)/)).toBeTruthy();
    expect(within(screen.getByRole("article", { name: "Angular size" })).getByText(/4,802 half-light radii measured on their clean images/)).toBeTruthy();
    expect(screen.getByText(/^Q1: raw forced-photometry colours, measurement noise included/)).toBeTruthy();
    expect(within(screen.getByRole("article", { name: "Normalized half-light shape" })).getByText(/integrates to one over log radius/)).toBeTruthy();
    // The explanations stay in each panel's popover.
    expect(screen.queryByText(/fixed joins VIS/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "About Apparent brightness" }));
    const about = within(await screen.findByRole("dialog", { name: "About Apparent brightness" }));
    expect(about.getByText(/fixed joins VIS 19\.50 \/ 20\.50 \/ 21\.50 · bridge slopes 0\.500 \/ 0\.420 \/ 0\.370/)).toBeTruthy();
    noTiles(container);
    noJobs();
  });

  it("distributions: two trust boxes, the generator plateau folded into the turnover box with its unit", async () => {
    const Galaxies = await tab("Galaxies");
    const { container } = show(<Galaxies />, "/synthetic/galaxies");
    await screen.findByText("Q1 count turnover");
    const boxes = [...container.querySelectorAll(".rl-trust > div")];
    expect(boxes).toHaveLength(2);
    expect(boxes[0].textContent).toContain("generator plateau 61.2 arcmin⁻² mag⁻¹ = 2.0× Q1 peak");
    expect(screen.queryByText("Generation ceiling")).toBeNull();
  });

  it("picks curves with compact chips that double as the legend, kept in the URL", async () => {
    const Galaxies = await tab("Galaxies");
    show(<Galaxies />, "/synthetic/galaxies");
    const picker = await screen.findByRole("group", { name: "Brightness curves" });
    const chips = within(picker).getAllByRole("button", { pressed: true });
    expect(chips[0].getAttribute("title")).toMatch(/Selection:/);
    fireEvent.click(chips[0]);
    await waitFor(() => expect(params().get("mag")).toBeTruthy());
    const radius = screen.getByRole("group", { name: "Radius curves" });
    fireEvent.click(within(radius).getByRole("button", { name: "Hide every Half-light radius curve" }));
    await waitFor(() => expect(params().get("re")).toBe("synthetic_clean_half_light"));
  });

  it("reads log10 parameters on physical log axes, and unit-integral radii as 'normalized probability / dex'", () => {
    const radius = galaxyPayload().parameters.radius;
    expect(xAxisOf(radius)).toMatchObject({ scale: "log", label: "radius (arcsec, log scale)" });
    const [lo, hi] = xDomainOf(radius, []);
    expect(lo).toBeCloseTo(10 ** -2.4);
    expect(hi).toBeCloseTo(10);
    const shapes = radiusEntries(radius, USEFUL_SHAPE_KEYS);
    expect(radiusYLabel(radius, shapes)).toBe("normalized probability / dex (log scale)");
  });

  it("relations: the radius and FWHM laws with their slope and scatter at 2 significant figures; the colour forest moved to the Prior", async () => {
    const Galaxies = await tab("Galaxies");
    show(<Galaxies />, "/synthetic/galaxies?view=relations");
    expect(await screen.findByText("Joint brightness–radius relation")).toBeTruthy();
    expect(screen.getByText(/^Slope −0\.15 dex\/mag, scatter 0\.23 dex at fixed magnitude, fitted to [\d,]+ aggregate Q1 radii$/)).toBeTruthy();
    expect(screen.getByText("VIS magnitude–MER aperture FWHM relation")).toBeTruthy();
    expect(screen.queryByText("Colour-forest fit diagnostic")).toBeNull();
    expect(screen.queryByText("Empirical colour model")).toBeNull();
  });

  it("joint: the corner plot's sample sizes are read on hover, not printed in every cell", async () => {
    routes["GET /api/galaxy-distributions/joint-pair?x=vis&y=log_re&r=25%3A41234%3A18%2C26"] = () => ({ body: PAIR });
    routes["GET /api/galaxy-distributions/joint-pair?x=vis&y=log_sfr&r=25%3A41234%3A18%2C26"] = () => ({ body: { ...PAIR, y: PAIR.x } });
    routes["GET /api/galaxy-distributions/joint-pair?x=log_sfr&y=vis&r=25%3A41234%3A18%2C26"] = () => ({ body: PAIR });
    const Galaxies = await tab("Galaxies");
    const { container } = show(<Galaxies />, "/synthetic/galaxies?view=joint");
    expect(await screen.findByText("Joint distributions")).toBeTruthy();
    expect(container.querySelector(".rl-corner__count")).toBeNull();
    expect(screen.queryByText(/^n = /)).toBeNull();
    const titles = [...container.querySelectorAll(".rl-corner__cell title")].map((t) => t.textContent ?? "");
    expect(titles.some((t) => / · n = [\d,]+ Q1 rows$/.test(t))).toBe(true);
    expect(titles.some((t) => / · n = [\d,]+ model draws$/.test(t))).toBe(true);
    await waitFor(() => expect(gets("/api/galaxy-distributions/joint-pair?x=vis&y=log_re").length).toBe(1));
    fireEvent.click(screen.getByRole("button", { name: "Explore VIS 2FWHM vs log₁₀ SFR (Euclid Q1)" }));
    await waitFor(() => expect(params().get("py")).toBe("log_sfr"));
    fireEvent.click(screen.getByRole("button", { name: "Swap axes" }));
    await waitFor(() => expect(params().get("px")).toBe("log_sfr"));
    // The magnitude × radius map states its numbers in its caption.
    expect(screen.getByText(/^Q1 shading: .* objects arcmin⁻² inside the map · .* · contours enclose /)).toBeTruthy();
    // The model draws are forward noised, so their contour widths compare with the raw Q1 colours.
    expect(screen.getByText(/upper triangle · 6,000 model draws inside VIS 18\.00–26\.00, colours with Q1 measurement noise/)).toBeTruthy();
  });

  it("joint: a corner built before forward noising says its model colours are deconvolved", async () => {
    const base = galaxyPayload();
    routes["GET /api/galaxy-distributions?include_training=0"] = () => ({
      body: { ...base, corner: { ...base.corner, model_noise: undefined } } });
    const Galaxies = await tab("Galaxies");
    show(<Galaxies />, "/synthetic/galaxies?view=joint");
    expect(await screen.findByText(/model draws inside VIS .*, deconvolved colours \(an older build: rebuild the plots to add Q1 noise\)/)).toBeTruthy();
  });

  it("draws the magnitude × radius maps: gray Q1, blue dashed generated, red solid model", () => {
    const series = jointMapSeries(JOINT_MAPS);
    expect(series.filter((s) => s.key === "q1").every((s) => s.color === C.cross && !s.dash)).toBe(true);
    expect(series.filter((s) => s.key === "synthetic").every((s) => s.color === categorical(0) && s.dash?.length)).toBe(true);
    expect(series.filter((s) => s.key === "model").every((s) => s.color === categorical(6) && !s.dash)).toBe(true);
    expect(jointColor("q1", "maps")).toBe(C.cross);
  });

  it("Prior drawer: the density against the generated fields, the laws as facts, the colour-forest diagnostic, Fit and Activate", async () => {
    const Galaxies = await tab("Galaxies");
    const { container } = show(<Galaxies />, "/synthetic/galaxies");
    fireEvent.click(await screen.findByRole("button", { name: "Prior" }));
    await waitFor(() => expect(params().get("prior")).toBe("1"));
    const summary = await waitFor(() => {
      const el = container.querySelector(".syn-drawer .ui-summary");
      expect(el).toBeTruthy();
      return el!;
    });
    expect(summary.textContent).toBe("Prior 152 galaxies arcmin⁻² at scene depth (VIS 14–29); the generated fields hold 151 · not active");
    expect(factRows("Model laws")).toEqual(["Radius slope−0.15dex/mag", "Radius scatter0.23dex", "Colour forest83,583Q1 rows"]);
    expect(screen.getByText("SFR is known for 33% of the weight; Rₑ resolved for 97.5%.")).toBeTruthy();
    expect(screen.getByText("Brightness law and fingerprints").closest("details")?.open).toBe(false);
    expect(screen.getByText("Colour-forest fit diagnostic")).toBeTruthy();
    // The cached Q1 checkpoints are incomplete (400 of 560): the refit waits for the query.
    expect((screen.getByRole("button", { name: "Fit galaxy prior from cached data" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Activate galaxy prior" }));
    await answer(/Activate this galaxy model/, "Activate");
    await waitFor(() => expect(posts("/api/galaxy-distributions/activate")).toHaveLength(1));
    for (const gone of [/PHZ weight/, /bin width/, /trees/, /^integrated density$/, /2FWHM · 14–29/]) expect(screen.queryByText(gone)).toBeNull();
  });

  it("Prior drawer: refits from the cached data (no archive query) once every checkpoint is cached", async () => {
    const base = galaxyPayload();
    routes["GET /api/galaxy-distributions?include_training=0"] = () => ({
      body: { ...base, q1_counts: { ...base.q1_counts!, complete: true, completed_queries: 560, query_count: 560 } } });
    const Galaxies = await tab("Galaxies");
    show(<Galaxies />, "/synthetic/galaxies?prior=1");
    const fit = await screen.findByRole("button", { name: "Fit galaxy prior from cached data" });
    await waitFor(() => expect((fit as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(fit);
    await answer(/Fit the galaxy prior from the cached data/, "Fit");
    await waitFor(() => expect(posts("/api/galaxy-distributions/fit")).toHaveLength(1));
    expect(posts("/api/galaxy-distributions/query-q1-counts")).toHaveLength(0);
  });

  it("How drawer: ONE Q1 MER + PHZ query with its ledger and caption, the cones, the rebuild and the TNG steps", async () => {
    const Galaxies = await tab("Galaxies");
    show(<Galaxies />, "/synthetic/galaxies?how=1");
    const query = await screen.findByRole("button", { name: "Query MER + PHZ" });
    expect(screen.getAllByRole("button", { name: /Query MER \+ PHZ/ })).toHaveLength(1);
    const ledger = within(screen.getByRole("region", { name: "Galaxy distribution data layers" }));
    expect(ledger.getByText("Q1 query: 140,085 rows over 1,885 arcmin² of population cones, 132,147 with PHZ PDFs.")).toBeTruthy();
    expect(ledger.getByText("Generated test + validate: 5,489 galaxies over 36.4 arcmin² of scenes (200 fields), 4,802 radii measured on clean images.")).toBeTruthy();
    expect(screen.getByText("Q1 cache v7 · VIS 14–28").className).toContain("ui-caption");
    // Idle: no progress counters; the interrupted query is one status line.
    expect(screen.queryByRole("list", { name: "Progressive magnitude-bin sampling phases" })).toBeNull();
    expect(screen.getByText("The last Q1 query stopped at 400 of 560 checkpoints (3 of 5 passes); run it again to resume.")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Refresh properties" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Atlas download/ }));
    expect(await screen.findByTestId("step-download_tng_skirt")).toBeTruthy();
    noJobs();
    fireEvent.click(query);
    await answer(/Query MER \+ PHZ/, "Query");
    await waitFor(() => expect(posts("/api/galaxy-distributions/query-q1-counts")).toHaveLength(1));
    const phases = within(await screen.findByRole("list", { name: "Progressive magnitude-bin sampling phases" })).getAllByRole("listitem");
    expect(phases.map((p) => p.getAttribute("data-state"))).toEqual(["cached", "cached", "cached", "querying", "waiting"]);
    expect(screen.getByText("checkpoints 400 of 560 · Rₑ brackets 170 of 170")).toBeTruthy();
  });

  it("the old model view opens the Prior drawer; the figure view (now Figures › Plates) shows the distributions", async () => {
    const Galaxies = await tab("Galaxies");
    const { unmount } = show(<Galaxies />, "/synthetic/galaxies?view=model");
    await waitFor(() => expect(params().get("prior")).toBe("1"));
    expect(params().get("view")).toBeNull();
    unmount();
    show(<Galaxies />, "/synthetic/galaxies?view=figure");
    expect(await screen.findByLabelText("Apparent brightness")).toBeTruthy();
    await waitFor(() => expect(params().get("view")).toBeNull());
    expect(screen.queryByText("Galaxy population diagnostics · 2 × 2")).toBeNull();
  });

  it("registers the views, the drawers and the jobs in the palette; the plate stays one download away", async () => {
    const Galaxies = await tab("Galaxies");
    show(<Galaxies />, "/synthetic/galaxies");
    await screen.findByLabelText("Apparent brightness");
    expect(paletteIds()).toEqual(expect.arrayContaining(["galaxies-view-templates", "galaxies-prior", "galaxies-fit",
      "galaxies-activate", "galaxies-query", "galaxies-rebuild", "galaxies-cones"]));
    const views = screen.getByRole("radiogroup", { name: "Galaxy view" });
    expect(within(views).getAllByRole("radio").map((r) => r.textContent)).toEqual(["Distributions", "Relations", "Joint", "Templates"]);
    act(() => runPalette("galaxies-view-relations"));
    await waitFor(() => expect(params().get("view")).toBe("relations"));
    act(() => runPalette("galaxies-rebuild"));
    await waitFor(() => expect(posts("/api/galaxy-distributions/build")).toHaveLength(1));
    expect(screen.getByRole("button", { name: "Download the galaxy figure" })).toBeTruthy();
    act(() => runPalette("galaxies-cones"));
    await waitFor(() => expect(lastLocation).toBe("/sky/atlas?layers=q1-tiles:0.2,population-cones"));
  });
});

/* ── stars ────────────────────────────────────────────────────────────── */

describe("stars", () => {
  it("one view: the verdict, the legend with sample sizes, the wide VIS panel and six colour PDFs, then the caption", async () => {
    const Stars = await tab("Stars");
    const { container } = show(<Stars />, "/synthetic/stars");
    expect(await screen.findByText("Stellar density in VIS")).toBeTruthy();
    noTiles(container);
    const summaries = container.querySelectorAll(".ui-summary");
    expect(summaries).toHaveLength(1);
    // 6,040 stars / 1,201.5 arcmin² = 5.03 against the 5.084 prior: −1.1%, inside the 5% tolerance.
    expect(summaries[0].textContent).toBe("Generated 5.03 vs prior 5.08 stars arcmin⁻² (−1.1%), trusted window VIS 18.0–23.0");
    expect(summaries[0].querySelector("[data-tone=warn]")).toBeNull();
    for (const label of ["Q1 PHZ stars · ≈403k", "Q1 point sources · ≈520k", "Gaia-matched Q1 stars · 3,456",
      "generated stars (test + validate) · 6,040 in 1,201 arcmin²"]) {
      expect(screen.getByRole("button", { name: label })).toBeTruthy();
    }
    expect(container.querySelectorAll(".rl-panel")).toHaveLength(7);
    expect(screen.getByText(/^Q1 footprint 63\.1 deg² · colours from the Gaia-matched stars in 3 fixed Q1 fields/)).toBeTruthy();
    // The model's colour draws are forward noised with the Q1 flux errors.
    expect(screen.getByRole("button", { name: "model (VIS law · colour draws with Q1 noise)" })).toBeTruthy();
    expect(screen.getByText(/the model's colour draws and the generated stars are compared over the Q1 colour sample's VIS 17\.1–21\.3 with Q1 measurement noise \(flux errors borrowed from 3,456 Q1 stars at the same VIS\)$/)).toBeTruthy();
    expect(screen.getByText(/training toggle adds the train split to the generated stars only/)).toBeTruthy();
    for (const gone of [/4,156/, /5,963/, /403,069/, /519,611/, /Gaia field area/]) expect(screen.queryByText(gone)).toBeNull();
    noJobs();
  });

  it("warns when the generated density is more than 5% off the prior", async () => {
    const base = starPayload();
    const comparison = { ...base.distribution!.density_comparison!, synthetic_star_count: 5000 };
    routes["GET /api/star-distribution?include_training=0"] = () => ({
      body: { ...base, distribution: { ...base.distribution!, density_comparison: comparison } } });
    const Stars = await tab("Stars");
    const { container } = show(<Stars />, "/synthetic/stars");
    await screen.findByText("Stellar density in VIS");
    expect(container.querySelector(".ui-summary [data-tone=warn]")?.textContent).toBe("−18%");
  });

  it("the deleted colours / gaia views fall back to the one view; the old prior view opens the Prior drawer", async () => {
    const Stars = await tab("Stars");
    const { unmount } = show(<Stars />, "/synthetic/stars?view=colours");
    expect(await screen.findByText("Stellar density in VIS")).toBeTruthy();
    await waitFor(() => expect(params().get("view")).toBeNull());
    expect(screen.queryByText("Gaia colour versus fitted Euclid distributions")).toBeNull();
    expect(screen.queryByText("Gaia colour–magnitude diagram")).toBeNull();
    unmount();
    show(<Stars />, "/synthetic/stars?view=prior");
    await waitFor(() => expect(params().get("prior")).toBe("1"));
    expect(params().get("view")).toBeNull();
  });

  it("Prior drawer: the fit sample in one sentence, Fit and Activate behind confirmations, the Gaia fields collapsed", async () => {
    const Stars = await tab("Stars");
    const { container } = show(<Stars />, "/synthetic/stars?prior=1");
    fireEvent.click(await screen.findByRole("button", { name: "Activate stellar prior" }));
    await answer(/Activate this stellar prior/, "Activate");
    await waitFor(() => expect(posts("/api/star-distribution/activate")).toHaveLength(1));
    fireEvent.click(screen.getByRole("button", { name: "Fit stellar prior from cached data" }));
    await answer(/Fit the stellar prior/, "Fit");
    await waitFor(() => expect(posts("/api/star-distribution/fit")).toHaveLength(1));
    const drawer = container.querySelector("#syn-drawer-prior") as HTMLElement;
    expect(drawer.querySelector(".syn-sentence")?.textContent).toBe("Colours fitted on 2,398 stars with S/N ≥ 5 in all bands, of 3,456 matched");
    expect(container.querySelectorAll(".ui-summary")).toHaveLength(1);        // the page keeps ONE summary line
    const details = within(drawer).getByText("Inputs: the Gaia colour fields").closest("details")!;
    expect(details.open).toBe(false);
    expect(within(details).getAllByRole("row")).toHaveLength(4);             // header + the three fixed fields
    const onSky = within(details).getAllByRole("link", { name: "On sky" }).map((a) => a.getAttribute("href") ?? "");
    expect(onSky).toContain("/sky/atlas?ra=269.733&dec=66.018&fov=1&layers=q1-tiles:0.2");
    expect(container.querySelector('a[href*="gaia-fields"]')).toBeNull();
  });

  it("How drawer: the confirmed query with its result as facts (≈ counts, the footprint)", async () => {
    const Stars = await tab("Stars");
    show(<Stars />, "/synthetic/stars?how=1");
    await screen.findByRole("heading", { name: "Query result" });
    expect(factRows("Query result")).toEqual(["Objects selected≈537k", "Point sources≈520k", "PHZ stars≈403k", "Footprint63.1deg²"]);
    expect(screen.getByText("No galaxy selection is used by this action.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Query stars · MER + PHZ + Gaia" }));
    await answer(/Query stars/, "Query");
    await waitFor(() => expect(posts("/api/star-distribution/query")).toHaveLength(1));
  });

  it("registers the drawers and the jobs in the palette, with no Gaia views", async () => {
    const Stars = await tab("Stars");
    show(<Stars />, "/synthetic/stars");
    await screen.findByText("Stellar density in VIS");
    expect(paletteIds()).toEqual(expect.arrayContaining(["stars-prior", "stars-query", "stars-fit", "stars-activate"]));
    expect(paletteIds().filter((id) => /gaia|stars-view-/.test(id))).toEqual([]);
    act(() => runPalette("stars-prior"));
    await waitFor(() => expect(params().get("prior")).toBe("1"));
  });

  it("points an empty distribution at How this is produced", async () => {
    routes["GET /api/star-distribution?include_training=0"] = () => ({ body: starPayload({ distribution: null }) });
    const Stars = await tab("Stars");
    show(<Stars />, "/synthetic/stars");
    fireEvent.click(await screen.findByRole("button", { name: "How this is produced" }, { timeout: 3000 }));
    await waitFor(() => expect(params().get("how")).toBe("1"));
  });
});

/* ── fields ───────────────────────────────────────────────────────────── */

describe("fields", () => {
  it("titles the tab, states the VIS verdict and opens on the look view (two lanes, one Lupton transfer)", async () => {
    const Fields = await tab("Fields");
    const { container } = show(<Fields />, "/synthetic/fields");
    expect(await screen.findByRole("heading", { name: "Synthetic vs real LR fields" })).toBeTruthy();
    await waitFor(() => expect(container.querySelector(".ui-summary")?.textContent).toBe("VIS overlap 0.97 (0.87–1.07), power syn/real 1.10 (0.99–1.21)"));
    expect(container.querySelectorAll(".ui-summary")).toHaveLength(1);
    expect(await screen.findByTestId("viewer-archive-fields")).toBeTruthy();
    const real = hoisted.viewers.find((v) => v.props.collection === "archive-fields")!;
    const syn = hoisted.viewers.find((v) => v.props.collection === "sky")!;
    expect(real.props.tiers).toEqual(["lr"]);
    expect(syn.props).toMatchObject({ params: { subset: "test" }, tiers: ["dirty"] });
    expect(real.api.setView).toHaveBeenCalledWith({ color: "lupton", knee: 100, gain: 1 });
    // Each lane's count is its own sample.
    expect(screen.getByText("Real Euclid · 2 fields")).toBeTruthy();
    expect(screen.getByText("Synthetic · 3 test records")).toBeTruthy();
    // The synthetic split is test or validate (train stays on FASRC).
    expect(screen.queryByRole("radio", { name: "Train" })).toBeNull();
    fireEvent.click(screen.getByRole("radio", { name: "Validate" }));
    await waitFor(() => expect(params().get("sub")).toBe("validate"));
    // The archive sync lives once, in the Real reference drawer.
    expect(screen.queryByRole("button", { name: "Sync archive fields from FASRC" })).toBeNull();
    noTiles(container);
    noJobs();
  });

  it("look: the shared row drives both lanes; an edit in one viewer reaches the other while locked", async () => {
    const Fields = await tab("Fields");
    show(<Fields />, "/synthetic/fields");
    await screen.findByTestId("viewer-sky");
    const real = hoisted.viewers.find((v) => v.props.collection === "archive-fields")!;
    const syn = hoisted.viewers.find((v) => v.props.collection === "sky")!;
    for (const v of [real, syn]) expect(v.props).toMatchObject({ toolbar: "none", nav: true });
    const row = screen.getByRole("toolbar", { name: "Shared display of both lanes" });
    fireEvent.click(within(row).getByRole("radio", { name: "VIS" }));
    await waitFor(() => expect(params().get("c")).toBe("VIS"));
    await waitFor(() => expect(syn.api.setView).toHaveBeenLastCalledWith({ color: "VIS", knee: 100, gain: 1 }));
    const state = (color: string, knee = 100, gain = 1) => ({ index: 0, id: "17", color, knee, gain, tiers: ["lr"] });
    act(() => { real.props.onState?.(state("VIS")); syn.props.onState?.(state("VIS")); });
    syn.api.setView.mockClear();
    act(() => { real.props.onState?.(state("J_E", 250)); });
    await waitFor(() => expect(syn.api.setView).toHaveBeenLastCalledWith({ color: "J_E", knee: 250, gain: 1 }));
    fireEvent.click(within(row).getByRole("button", { name: "Zoom in both lanes" }));
    expect(real.api.zoomBy).toHaveBeenCalledTimes(1);
    expect(syn.api.zoomBy).toHaveBeenCalledTimes(1);
    expect(REALISM_CSS).toMatch(/@container rl-vis \(max-width: 720px\) \{\s*\.rl-vis__lock \.ui-switchrow > label \{[^}]*clip: rect\(0 0 0 0\)/);
  });

  it("stats: the sample chips carry their sizes, then the six figures, the median metrics and the corrected geometry", async () => {
    const Fields = await tab("Fields");
    const { container } = show(<Fields />, "/synthetic/fields?view=stats");
    for (const title of ["Brightness distribution", "Pixel quantile profile", "Angular-scale power", "Mean brightness vs field variation",
      "Inter-band pixel correlation", "Scale-spectrum similarity", "Median field metrics"]) {
      expect(await screen.findByText(title)).toBeTruthy();
    }
    // The background vs robust noise figure lives on Noise only, linked from a caption.
    expect(screen.queryByText("Background vs robust noise")).toBeNull();
    expect(screen.getByRole("link", { name: "Synthetic › Noise" }).getAttribute("href")).toBe("/synthetic/noise");
    const chips = within(screen.getByRole("group", { name: "Bands and samples" }));
    expect(chips.getByRole("button", { name: "synthetic LR · 200 fields" })).toBeTruthy();
    expect(chips.getByRole("button", { name: "real Euclid LR · 176 fields / 44 pointings" })).toBeTruthy();
    fireEvent.click(chips.getByRole("button", { name: "VIS" }));
    await waitFor(() => expect(params().get("hide")).toBe("VIS"));
    expect(screen.getByText("Each field: 255 × 255 px at 0.1″, 0.181 arcmin² (the centre of each 256-px tile)").className).toContain("ui-caption");
    const scores = within(screen.getByRole("table", { name: "Scale-spectrum similarity per NISP band" }));
    expect(scores.getAllByRole("row").map((r) => r.querySelector("td")?.textContent ?? "")).toEqual(["", "Y", "J", "H"]);
    expect(container.querySelectorAll(".ui-summary")).toHaveLength(1);
    noTiles(container);
    noJobs();
  });

  it("stats: every figure has exact axis bounds one click away", async () => {
    const Fields = await tab("Fields");
    show(<Fields />, "/synthetic/fields?view=stats");
    expect(await screen.findByText("Brightness distribution")).toBeTruthy();
    fireEvent.click(screen.getAllByRole("button", { name: "bounds" })[0]);
    const dlg = within(await screen.findByRole("dialog", { name: "Brightness distribution: axis bounds" }));
    expect(dlg.getByRole("slider", { name: "Brightness distribution x minimum" })).toBeTruthy();
    expect((dlg.getByRole("button", { name: "Full range" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("detection: detections and negative islands per field, the completeness in a caption (the real side has no truth)", async () => {
    const Fields = await tab("Fields");
    show(<Fields />, "/synthetic/fields?view=detection");
    expect(await screen.findByText("VIS source detection per field")).toBeTruthy();
    const rows = screen.getAllByRole("row");
    const synthetic = rows.find((r) => r.textContent?.startsWith("synthetic LR"))!;
    expect(synthetic.textContent).toContain("8.1%");           // 6 negative islands per 74 detections
    expect(rows.some((r) => /no truth/.test(r.textContent ?? ""))).toBe(false);
    expect(screen.queryByText("Galaxy completeness per field")).toBeNull();
    expect(screen.getByText(/^Galaxy completeness, synthetic only \(the real fields have no truth\): 81\.5% of the truth galaxies detected · per field /)).toBeTruthy();
    expect(screen.getByText("Detections per field")).toBeTruthy();
    expect(screen.getByText("Negative islands per field")).toBeTruthy();
  });

  it("a stale cache keeps its last result behind a badge with an explicit Measure (confirmed, never on a visit)", async () => {
    routes["GET /api/population-comparison?include_training=0"] = () => ({
      body: pixelsPayload({ comparison: null, previous: { ...pixelsPayload().comparison!, samples: { ...pixelsPayload().comparison!.samples,
        real: { fields: 220, area_arcmin2: 40, independent_parents: 44 } } },
      availability: { ...pixelsPayload().availability, real: { ...pixelsPayload().availability.real, fields: 220, compared_fields: 176 },
        comparison_cache: { present: true, schema_current: false, fresh: false, reason: "comparison cache uses an older schema" } } }) });
    const Fields = await tab("Fields");
    show(<Fields />, "/synthetic/fields?view=stats");
    expect(await screen.findByText("last result")).toBeTruthy();
    expect(await screen.findByText("Brightness distribution")).toBeTruthy();
    // The chips show what the last result measured; the caption says what a new measurement compares.
    expect(screen.getByRole("button", { name: "real Euclid LR · 220 fields / 44 pointings" })).toBeTruthy();
    expect(screen.getByText(/a new measurement compares 176, leaving out the 44 centre tiles, which were placed to avoid bright stars\./)).toBeTruthy();
    noJobs();
    fireEvent.click(screen.getByRole("button", { name: "Measure" }));
    await answer(/Measure the field statistics/, "Measure");
    await waitFor(() => expect(posts("/api/population-comparison/build")).toHaveLength(1));
  });

  it("explains a missing cache and measures from the empty state", async () => {
    routes["GET /api/population-comparison?include_training=0"] = () => ({ body: pixelsPayload({ comparison: null }) });
    const Fields = await tab("Fields");
    show(<Fields />, "/synthetic/fields?view=stats");
    expect(await screen.findByText("The field statistics have not been measured")).toBeTruthy();
    expect(screen.getByText("The field statistics have not been measured yet.")).toBeTruthy();
    fireEvent.click(screen.getAllByRole("button", { name: "Measure" })[0]);
    await answer(/Measure the field statistics/, "Measure");
    await waitFor(() => expect(posts("/api/population-comparison/build")).toHaveLength(1));
  });

  it("Real reference drawer: the archive fields and pointings, the confirmed sync and archive_field_sample", async () => {
    const Fields = await tab("Fields");
    show(<Fields />, "/synthetic/fields?ref=1");
    expect(await screen.findByText(/^44 independent parent pointings · 176 four-band samples \(44 star-avoiding centre tiles left out\) · Q1_R1\. Per field: EDF-F 36 · EDF-N 64 · EDF-S 76\.$/)).toBeTruthy();
    expect(screen.getByTestId("step-archive_field_sample")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Sync archive fields from FASRC" }));
    await answer(/Sync the archive fields/, "Sync");
    await waitFor(() => expect(posts("/api/archive-fields/sync")).toHaveLength(1));
  });

  it("the absorbed pages' views land on their new view, and the palette has the views, Measure and the drawer", async () => {
    const Fields = await tab("Fields");
    show(<Fields />, "/synthetic/fields?view=pixels");
    await waitFor(() => expect(params().get("view")).toBe("stats"));
    expect(await screen.findByText("Brightness distribution")).toBeTruthy();
    expect(paletteIds()).toEqual(expect.arrayContaining(["fields-view-look", "fields-view-detection", "fields-measure", "fields-reference", "fields-band-VIS"]));
    act(() => runPalette("fields-reference"));
    await waitFor(() => expect(params().get("ref")).toBe("1"));
  });

  it("filters bands and samples, and maps real dots to their parent pointing", () => {
    const series = histogramSeries(FIELDS, visibleFrom(["VIS", "real"]));
    expect(series.map((s) => s.key)).toEqual(["Y_E:synthetic", "J_E:synthetic", "H_E:synthetic"]);
    const points = relationPoints(FIELDS, "mean_std", visibleFrom([]));
    expect(points.filter((p) => p.sample === "real").map((p) => p.parent)).toContain("parent-3");
  });
});

/* ── inspectors, navigation ───────────────────────────────────────────── */

describe("inspectors and navigation", () => {
  it("the archivefield inspector shows the sample with a compact viewer and links to Fields and the atlas", async () => {
    const { ArchiveFieldInspector } = await import("./inspectors");
    show(<ArchiveFieldInspector id="18" />, "/synthetic/fields");
    expect(await screen.findByText("parent-4")).toBeTruthy();
    expect(screen.getByTestId("viewer-archive-fields")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Open in Fields" }).getAttribute("href")).toBe("/synthetic/fields?view=look&v.real.id=18");
    expect(screen.getByRole("link", { name: "Open on sky" }).getAttribute("href")).toContain("layers=archive-fields");
  });

  it("declares exactly the manifest's tabs, each a merged module; no Round-trip entry anywhere", async () => {
    const { TABS } = await import("./index");
    expect(Object.keys(TABS)).toEqual(["status", "records", "galaxies", "stars", "noise", "psf", "fields"]);
    const pages = allPages();
    expect(pages.some((p) => p.path === "/synthetic/noise")).toBe(true);
    expect(pages.some((p) => /round.?trip/i.test(p.label))).toBe(false);
  });
});
