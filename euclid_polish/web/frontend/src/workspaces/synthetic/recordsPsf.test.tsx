/* Synthetic › Records, PSF (catalogue → cutouts → ePSF) and the Galaxies
 * TNG templates, plus the star / truth / psf / tng inspector cards, against
 * a mocked backend (routes/{views,cutouts,psfs,tng}.py). The image viewer is
 * mocked: it reports an object through onState, records goTo / goToId and
 * renders the page's markers as buttons. Every page starts nothing on a
 * visit; every job is confirmed. */
import { QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobsStore } from "../../api/jobs";
import { queryClient } from "../../api/query";
import { useInspectorRegistry } from "../../app/inspector";
import { useInspector } from "../../state/inspector";
import { useSelection } from "../../state/selection";
import { resetConfirm } from "../../ui";
import { PsfInspector, StarInspector, TngInspector, TruthInspector } from "./dataInspectors";
import "./register";
import Galaxies from "./tabs/Galaxies";
import Psf from "./tabs/Psf";
import Records from "./tabs/Records";
import { galaxyPayload, starPayload } from "./testFixtures";

type MockMarkers = {
  items: { key: string; title?: string; x: number; y: number }[]; tiers?: string[];
  activeKey?: string | null; onPick?: (key: string) => void; onHover?: (key: string | null) => void;
} | null | undefined;
type MockViewerProps = {
  collection: string; urlKey?: string; params?: Record<string, string>; tiers?: string[];
  onReady?: (api: unknown) => void; onState?: (s: unknown) => void; markers?: MockMarkers; display?: unknown;
};
const viewer = vi.hoisted(() => ({
  goTo: [] as number[], goToId: [] as string[], mounts: 0, index: 1, id: "test:1" as string | null,
  display: undefined as unknown,
}));
vi.mock("../../viewer", async () => {
  const { useEffect } = await import("react");
  return {
    ImageViewer: (p: MockViewerProps) => {
      viewer.display = p.display;
      useEffect(() => {
        viewer.mounts += 1;
        p.onReady?.({
          goTo: (i: number) => { viewer.goTo.push(i); p.onState?.({ index: i, id: `test:${i}` }); },
          goToId: async (id: string) => { viewer.goToId.push(id); return true; },
          setParams: async () => undefined,
        });
        p.onState?.({ index: viewer.index, id: viewer.id });
        return () => p.onReady?.(null);
        // eslint-disable-next-line react-hooks/exhaustive-deps
      }, []);
      // markers render as buttons so a test can see what the page overlays and pick one
      return <div data-testid="viewer">{p.collection}|{p.params?.subset ?? ""}|{(p.tiers ?? []).join(",")}
        {p.markers && <div role="group" aria-label={`markers on ${p.markers.tiers?.join(",") ?? "every tier"}`}>
          {p.markers.items.map((m) => (
            <button key={m.key} type="button" aria-label={m.title} data-x={m.x} data-y={m.y}
              data-active={p.markers?.activeKey === m.key || undefined} onClick={() => p.markers?.onPick?.(m.key)} />
          ))}
        </div>}
      </div>;
    },
  };
});

type Reply = { status?: number; body: unknown };
let routes: Record<string, (form: Record<string, string>) => Reply>;
let posts: { url: string; form: Record<string, string> }[];

const formOf = (body: BodyInit | null | undefined): Record<string, string> => {
  const out: Record<string, string> = {};
  if (body instanceof FormData) body.forEach((v, k) => { out[k] = String(v); });
  return out;
};

function Probe() {
  const loc = useLocation();
  return <output data-testid="loc">{loc.pathname}{loc.search}</output>;
}
const loc = () => screen.getByTestId("loc").textContent ?? "";

const show = (el: ReactElement, url = "/synthetic/records") => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={[url]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes><Route path="*" element={<>{el}<Probe /></>} /></Routes>
    </MemoryRouter>
  </QueryClientProvider>,
);

const answer = async (title: RegExp | string, button: string) => {
  const dlg = await screen.findByRole("alertdialog", { name: title });
  fireEvent.click(within(dlg).getByRole("button", { name: button }));
  await waitFor(() => expect(screen.queryByRole("alertdialog", { name: title })).toBeNull());
};

const RUNNING_JOB = (id: string, label: string) => ({
  job_id: id, label, kind: null, status: "running", started: 1, finished: null, duration: 1, error: null,
  log: "", log_truncated: false, cancellable: true, cancel_requested: false, result: null,
  progress: { current: 0, total: 0, pct: 0, label: "" },
});

const file = (name: string, count?: number | null) => ({ name, size_bytes: 1_000_000, mtime: 1_790_000_000, ...(count !== undefined ? { count } : {}) });
const SR_STATUS = {
  records: true, checkpoint: true, can_generate: true, subsets: ["test"], sr: { test: 3, validate: 0, train: 0 },
  records_dir: "/cache/records",
  splits: {
    test: { files: { dirty: file("dirty_test.tfrecord", 3), hr: file("hr_test.tfrecord", 3), clean: file("clean_test.tfrecord", null),
      sources: file("sources_test.csv") }, count: 3, present: true,
      sr: { state: "stale", reasons: ["members changed (29 → 30 STARFULL members)"], count: 3, records_count: 3, manifest: null } },
    validate: { files: { dirty: null, hr: null, clean: null, sources: null }, count: 0, present: false,
      sr: { state: "missing", reasons: [], count: 0, records_count: null, manifest: null } },
    train: { files: { dirty: null, hr: null, clean: null, sources: null }, count: 0, present: false,
      sr: { state: "missing", reasons: [], count: 0, records_count: null, manifest: null } },
  },
  model: { member_labels: ["1·psnr"], combiner_kind: "spatial_gate", combiner_fingerprint: "x" },
  sync_job: null, generate_job: null,
};
const GEOMETRY = { hr: { height: 510, width: 510, pixscale: 0.05 }, lr: { height: 255, width: 255, pixscale: 0.1 } };
const src = (row: number, over: Record<string, unknown>) => ({
  row, type: "galaxy", render: "tng", x_pix: 10 + row, y_pix: 20, off_field: false, flux_vis_e: 1000, flux_y_e: null,
  flux_j_e: null, flux_h_e: null, mag_vis: 24, mag_y_e: null, mag_j_e: null, mag_h_e: null, target_vis_mag: null, z: null,
  re_arcsec: 0.2, theta_E_arcsec: null, orientation: null, temperature_k: null, subhalo_id: "658592", source_subhalo_id: null,
  sfr_class: null, ...over,
});
const SOURCES = {
  subset: "test", field_index: 1, present: true, geometry: GEOMETRY,
  counts: { galaxy: 1, star: 1, lens: 0, other: 0, off_field: 1 },
  sources: [src(0, {}), src(1, { type: "star", mag_vis: 18.5, off_field: true, subhalo_id: null })],
};
const CENSUS = { subset: "test", present: true, geometry: GEOMETRY, fields: [
  { field_index: 0, galaxy: 5, star: 0, lens: 0, other: 0, off_field: 0, n: 5, brightest_star_mag: null, brightest_galaxy_mag: 22.1, total_vis_e: 1e5 },
  { field_index: 1, galaxy: 1, star: 1, lens: 1, other: 0, off_field: 1, n: 2, brightest_star_mag: 18.5, brightest_galaxy_mag: 24, total_vis_e: 2e5 },
] };

const BITS = { valid: 1, corrupted: 2, failed: 4, size_shift: 3 };
const V = 1 | (1 << 4);
const STARS = {
  present: true, source: "fasrc-mirror", path: "/n/x/stars.csv", local_path: "/l/stars.csv", size_bytes: 10,
  mtime: Date.now() / 1000 - 3600, age_s: 3600, bands: ["VIS", "Y_E", "J_E", "H_E"], sizes: [255, 511], bits: BITS,
  columns: ["id", "ra", "dec", "mag", "flux_uJy", "fluxerr_uJy", "field", "b_VIS", "b_Y_E", "b_J_E", "b_H_E", "nav"],
  rows: [
    [1, 269.7, 66.0, 17.5, 275, 0.4, "EDF-N", V, V, V, V, 1],
    [2, 61.2, -48.4, 18.2, 180, 0.5, "EDF-S", V, 0, 0, 2, 0],
    [3, 10, 10, 18.9, 100, 0.6, "", 0, 0, 4, 0, 0],
  ],
  summary: { total: 3, valid: 2, corrupted: 0, failed: 1, pending: 0, valid_all4: 1, navigator: { size: 511, count: 1 }, mag_min: 17.5, mag_max: 18.9 },
  band_stats: [
    { band: "VIS", valid: 2, corrupted: 0, failed: 0, pending: 1, by_size: { 255: 0, 511: 2 } },
    { band: "Y_E", valid: 1, corrupted: 0, failed: 0, pending: 2, by_size: { 255: 0, 511: 1 } },
    { band: "J_E", valid: 1, corrupted: 0, failed: 1, pending: 1, by_size: { 255: 0, 511: 1 } },
    { band: "H_E", valid: 1, corrupted: 1, failed: 0, pending: 1, by_size: { 255: 0, 511: 1 } },
  ],
};
const EUCLID_QUERY = {
  step_id: "euclid_query", label: "Query Euclid catalog (brightest N)", needs_gpu: false,
  defaults: { partition: "shared", n_cpus: 1, n_gpus: 0, memory: "4G", time_limit: "30:00" },
  task_params: [], last_params: { num_stars: 10000, magnitude_min: 18, magnitude_limit: 19, snr_min: 50 }, outputs: [],
};
const STEPS = { ssh_connected: true, steps: [EUCLID_QUERY], artifacts: {}, remote_paths: {} };
const CONFIG = { config: { n_train: 6400, n_valid: 100, n_test: 100, hr_image_size: 512, galaxy_density_arcmin2: 151.5032458303819,
  star_density_arcmin2: 5.08, lens_density_arcmin2: 0.5, psf_warp_prob: 0.5, saturation_mask_prob: 0.2,
  psf_warp_alpha_max: 20, psf_warp_sigma: 3 } };

const isPost = (url: string) => posts.filter((p) => p.url === url);
const jobPosts = () => posts.filter((p) => p.url !== "/api/jobs?summary=1");

beforeEach(() => {
  posts = [];
  viewer.goTo = []; viewer.goToId = []; viewer.mounts = 0; viewer.index = 1; viewer.id = "test:1"; viewer.display = undefined;
  routes = {
    "GET /api/fasrc/status": () => ({ body: { ssh_connected: true } }),
    "GET /api/fasrc/steps/status": () => ({ body: STEPS }),
    "GET /api/sky/sr-status": () => ({ body: SR_STATUS }),
    "GET /api/sky/records/sources?subset=test&index=1": () => ({ body: SOURCES }),
    "GET /api/sky/records/sources?subset=test": () => ({ body: CENSUS }),
    "GET /api/catalog/stars": () => ({ body: STARS }),
    "GET /api/config": () => ({ body: CONFIG }),
    "GET /api/galaxy-distributions?include_training=0": () => ({ body: galaxyPayload() }),
    "GET /api/star-distribution?include_training=0": () => ({ body: starPayload() }),
    "GET /api/jobs?summary=1": () => ({ body: [] }),
  };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = String(input);
    const method = init.method ?? "GET";
    const form = formOf(init.body);
    if (method === "POST") posts.push({ url, form });
    const handler = routes[`${method} ${url}`];
    const r = handler ? handler(form) : { status: 404, body: { ok: false, error: `no route ${method} ${url}` } };
    return new Response(JSON.stringify(r.body), { status: r.status ?? 200 });
  }));
  queryClient.clear();
  useJobsStore.getState().reset();
  useInspector.getState().clear();
  useSelection.getState().clear();
});
afterEach(() => {
  act(() => resetConfirm());
  queryClient.clear();
  vi.unstubAllGlobals();
});

/* ── records ───────────────────────────────────────────────────────────── */

describe("Records", () => {
  it("one toolbar row: the split with its counts (train disabled), the truth sources, a badge only on a problem", async () => {
    show(<Records />);
    const bar = await screen.findByRole("toolbar", { name: "Records" });
    const split = within(bar).getByRole("radiogroup", { name: "Split" });
    await waitFor(() => expect(within(split).getAllByRole("radio").map((r) => r.textContent)).toEqual(["test 3", "validate 0", "train"]));
    const train = within(split).getByRole("radio", { name: "train" }) as HTMLButtonElement;
    expect(train.disabled).toBe(true);
    expect(train.getAttribute("title")).toMatch(/generated and read on FASRC/);
    // clean_test is truncated: ONE badge says so; the SR state and the noise check are not here
    expect(await within(bar).findByText("1 corrupt file")).toBeTruthy();
    expect(screen.queryByText("SR stale")).toBeNull();
    expect(screen.queryByText("Old noise model")).toBeNull();
    expect(screen.queryByRole("button", { name: "Generate SR" })).toBeNull();
    expect(screen.getByTestId("viewer").textContent).toBe("sky|test|dirty,hr");
    expect(viewer.display).toEqual({ matchSurfaceBrightness: true });          // HR at the LR's surface brightness
    expect(jobPosts()).toEqual([]);
  });

  it("puts the truth-source controls in the toolbar row and the viewer before every table", async () => {
    show(<Records />);
    const viewerEl = await screen.findByTestId("viewer");
    const caption = await screen.findByRole("group", { name: "Truth sources of record 1" });
    expect(screen.getByRole("toolbar", { name: "Records" }).contains(caption)).toBe(true);
    const census = await screen.findByRole("grid", { name: "Records in test" });
    const sources = await screen.findByRole("grid", { name: "Sources of record 1" });
    const before = (a: Element, b: Element) => !!(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING);
    expect(before(caption, viewerEl)).toBe(true);
    expect(before(viewerEl, sources)).toBe(true);
    expect(before(sources, census)).toBe(true);
    expect(within(caption).queryByRole("button", { name: /^lens/ })).toBeNull();
  });

  it("overlays the record's truth sources on HR; the type chips are the legend AND the filter", async () => {
    show(<Records />);
    const marks = await screen.findByRole("group", { name: "markers on hr" });
    expect(within(marks).getAllByRole("button")).toHaveLength(2);
    fireEvent.click(within(marks).getByRole("button", { name: /star at \(11\.0, 20\.0\) px, VIS 18\.50/ }));
    expect(useInspector.getState().current).toEqual({ kind: "truth", id: "test/1/1" });
    expect(useInspectorRegistry.getState().kinds.truth).toBeTruthy();
    const chip = screen.getByRole("button", { name: /^star 1$/ });
    expect(chip.getAttribute("aria-pressed")).toBe("true");
    expect(screen.queryByText(/ shown/)).toBeNull();                        // the chips carry the counts
    fireEvent.click(chip);
    await waitFor(() => expect(loc()).toContain("hide=star"));
    expect(await screen.findByText(/^1 of 2 shown · HR /)).toBeTruthy();    // only while a chip hides rows
    expect(within(screen.getByRole("group", { name: "markers on hr" })).getAllByRole("button")).toHaveLength(1);
    fireEvent.click(screen.getByRole("radio", { name: "All tiers" }));
    expect(await screen.findByRole("group", { name: "markers on every tier" })).toBeTruthy();
    fireEvent.click(screen.getByRole("radio", { name: "Off" }));
    await waitFor(() => expect(screen.queryByRole("group", { name: /^markers on/ })).toBeNull());
  });

  it("a crowded toolbar keeps the overlay as one menu button; the problem badge keeps its word", async () => {
    show(<Records />);
    await screen.findByRole("group", { name: "markers on hr" });
    expect(screen.getByRole("button", { name: "Truth sources on: HR" }).textContent).toBe("Sources: HR");
    const status = screen.getByRole("group", { name: "Problems of the test split" });
    const kept = [...status.querySelectorAll(".dt-tipbadge--keep")];
    expect(kept.map((b) => b.textContent)).toEqual(["1 corrupt file"]);
    expect(screen.getByRole("toolbar", { name: "Records" }).getAttribute("data-compact")).toBeNull();
  });

  it("the census: Σ VIS and brightest-star histograms, sources per arcmin² generated · prior · Q1, and a row that moves the viewer", async () => {
    show(<Records />);
    const census = await screen.findByRole("region", { name: "Census" });
    expect(within(census).getByRole("figure", { name: "Σ VIS per record histogram" })).toBeTruthy();
    expect(within(census).getByRole("figure", { name: "Brightest star per record histogram" })).toBeTruthy();
    const table = await within(census).findByRole("table", { name: "Sources per arcmin²: generated, prior and Q1" });
    const rows = within(table).getAllByRole("row").map((r) => r.querySelector("th")?.textContent ?? "");
    expect(rows).toEqual(["Kind", "Galaxies", "", "Stars", "", "Lenses"]);
    // one lens in 2 records of 0.18 arcmin² = 2.8 arcmin⁻², against the configured prior 0.5
    const lens = within(table).getAllByRole("row").at(-1)!;
    expect([...lens.querySelectorAll("td")].map((c) => c.textContent)).toEqual(["all · test split", "2.8", "0.5", "no Q1 count"]);
    // each count once per screen: the section sub carries the record count
    expect(census.querySelector(".ui-dt__count")).toBeNull();
    const heads = [...census.querySelectorAll(".rl-fig__head")].map((h) => h.textContent);
    expect(heads[0]).toBe("Σ VIS per recordclick a bar to open its record");
    expect(heads[1]).toMatch(/^Brightest star per record\d+ records · click a bar/);   // a different number: records with a star
    const grid = within(census).getByRole("grid", { name: "Records in test" });
    fireEvent.click(await within(grid).findByText(/^22\.1/));
    expect(viewer.goTo).toEqual([0]);
  });

  it("the lens densities share one decimal count (2.77 beside 0.02, not 2.8)", async () => {
    routes["GET /api/config"] = () => ({ body: { config: { ...CONFIG.config, lens_density_arcmin2: 0.02 } } });
    show(<Records />);
    const census = await screen.findByRole("region", { name: "Census" });
    const table = await within(census).findByRole("table", { name: "Sources per arcmin²: generated, prior and Q1" });
    const lens = within(table).getAllByRole("row").at(-1)!;
    await waitFor(() => expect([...lens.querySelectorAll("td")].map((c) => c.textContent).slice(1, 3)).toEqual(["2.77", "0.02"]));
  });

  it("the record's SR is one click away, in Models › Images", async () => {
    show(<Records />);
    const link = await screen.findByRole("link", { name: "Open its SR in Models › Images" });
    await waitFor(() => expect(link.getAttribute("href")).toBe("/models/images?set=records&split=test&id=test%3A1"));
  });

  it("links back to System › Config when a scene or lens knob differs from its default", async () => {
    routes["GET /api/config"] = () => ({ body: { ...CONFIG, defaults: { ...CONFIG.config, n_test: 200 } } });
    show(<Records />);
    const link = await screen.findByRole("link", { name: "1 knob changed · Edit" });
    expect(link.getAttribute("href")).toBe("/system/config?changed=1");
  });

  it("the old census links (?section=census, ?view=census) land on the census", async () => {
    show(<Records />, "/synthetic/records?section=census");
    expect(await screen.findByRole("region", { name: "Census" })).toBeTruthy();
    expect(document.getElementById("syn-census")).toBeTruthy();
  });

  it("the Generate and sync drawer: the step card, the sync (a confirmed job) and the knobs read-only with an edit link", async () => {
    routes["POST /api/sky/sync"] = () => ({ body: { ok: true, job_id: "s1" } });
    routes["GET /api/jobs/s1"] = () => ({ body: RUNNING_JOB("s1", "records: sync test+validate+train from FASRC") });
    show(<Records />);
    const bar = await screen.findByRole("toolbar", { name: "Records" });
    fireEvent.click(within(bar).getByRole("button", { name: "Generate and sync" }));
    await waitFor(() => expect(loc()).toContain("gen=1"));
    const knobs = within(await screen.findByRole("heading", { name: "Generation knobs" }).then((h) => h.closest(".ui-facts") as HTMLElement));
    const facts = knobs.getAllByRole("group").map((r) => r.textContent);
    expect(facts).toContain("Galaxy density152arcmin⁻²");                 // rounded, never 151.5032…
    expect(facts).toContain("Star densityset by the active stellar prior");
    expect(screen.getByRole("link", { name: "Edit in Config" }).getAttribute("href")).toBe("/system/config");
    fireEvent.click(screen.getByRole("button", { name: "Sync from FASRC…" }));
    fireEvent.click(await screen.findByRole("checkbox", { name: /train/ }));
    fireEvent.click(screen.getByRole("button", { name: "Start sync" }));
    await answer(/Sync test \+ validate \+ train from FASRC/, "Sync");
    await waitFor(() => expect(isPost("/api/sky/sync")).toHaveLength(1));
    expect(isPost("/api/sky/sync")[0].form).toEqual({ subsets: "test,validate,train", kinds: "dirty,hr,clean,sources" });
  });

  it("a resume never inherits the last run's --regenerate-splits from its extra flags", async () => {
    const gen = {
      step_id: "synthetic_generate", label: "Generate synthetic training pairs (CPU)", needs_gpu: false,
      defaults: { partition: "shared", n_cpus: 16, n_gpus: 0, memory: "64G", time_limit: "6:00:00" },
      task_params: [
        { name: "force", type: "bool", default: false, help: "Regenerate every split from scratch." },
        { name: "regenerate_splits", type: "str", default: null, help: "Comma list of splits to rebuild." },
        { name: "extra_flags", type: "str", default: null, help: "Extra run_pipeline.py flags." },
      ],
      last_params: { extra_flags: "--regenerate-splits=train --seed 2", force: false },
      outputs: [],
    };
    routes["GET /api/fasrc/steps/status"] = () => ({ body: { ...STEPS, steps: [gen] } });
    routes["GET /api/fasrc/steps/synthetic_generate/history"] = () => ({ body: { history: [] } });
    routes["POST /api/fasrc/steps/synthetic_generate/submit"] = () => ({ body: { jobid: "777" } });
    show(<Records />, "/synthetic/records?gen=1");
    expect(await screen.findByDisplayValue("--seed 2")).toBeTruthy();
    expect(screen.queryByDisplayValue(/regenerate/)).toBeNull();
    expect(screen.getByText("dropped --regenerate-splits=train")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /^(Submit|Queue)/ }));
    await answer(/Generate synthetic training pairs/, "Submit");
    await waitFor(() => expect(posts.find((p) => p.url.endsWith("/synthetic_generate/submit"))).toBeTruthy());
    const body = posts.find((p) => p.url.endsWith("/synthetic_generate/submit"))!.form;
    expect(body.extra_flags).toBe("--seed 2");
    expect(body.regenerate_splits).toBeUndefined();
  });

  it("offers Generate and sync when the split is not on this machine, and never mounts a viewer on it", async () => {
    show(<Records />, "/synthetic/records?split=validate");
    expect(await screen.findByText("No validate records on this machine")).toBeTruthy();
    expect(screen.queryByTestId("viewer")).toBeNull();
    expect(viewer.mounts).toBe(0);
  });

  it("the sync is disabled while FASRC is offline", async () => {
    routes["GET /api/fasrc/status"] = () => ({ body: { ssh_connected: false } });
    show(<Records />, "/synthetic/records?gen=1");
    await waitFor(() => expect((screen.getByRole("button", { name: "Sync from FASRC…" }) as HTMLButtonElement).disabled).toBe(true));
  });
});

/* ── PSF ───────────────────────────────────────────────────────────────── */

describe("PSF › catalogue", () => {
  it("opens with the stars and the usable ones, then the filters, the table and the magnitude histogram with its windows", async () => {
    const { container } = show(<Psf />, "/synthetic/psf?cut=nav");
    await waitFor(() => expect(container.querySelector(".ui-summary")?.textContent)
      .toBe("Real Euclid Q1 stars the empirical PSFs are built from: 3 stars, 1 usable (valid in all 4 bands at 511 px)"));
    const grid = await screen.findByRole("grid", { name: "Stars" });
    await waitFor(() => expect(within(grid).getAllByRole("row")).toHaveLength(2));   // header + star 1
    expect(within(grid).getByRole("img", { name: "VIS valid, Y valid, J valid, H valid" })).toBeTruthy();
    expect(screen.getByText("1 of 3 shown")).toBeTruthy();
    expect(screen.getByRole("figure", { name: "Magnitude distribution of the star catalogue" })).toBeTruthy();
    expect(screen.getByText(/windows' edges \(last run: brightest 10,000, VIS 18–19, S\/N ≥ 50\)\.$/)).toBeTruthy();
    // Deleted: 'valid in a band', the best-band counts, the VIS range, the per-band meter table.
    for (const gone of [/valid in a band/, /corrupted$/, /^16\.00–19\.01/]) expect(screen.queryByText(gone)).toBeNull();
    expect(screen.queryByRole("table", { name: "Cutout validity per band" })).toBeNull();
    expect(jobPosts()).toEqual([]);
  });

  it("handles a catalogue beyond the argument-spread limit (~120k stars)", async () => {
    const rows = Array.from({ length: 130_000 }, (_x, i) => [i + 1, 10, 10, 16 + (i % 300) / 100, 1, 0.1, "EDF-N", V, V, V, V, 1]);
    routes["GET /api/catalog/stars"] = () => ({ body: { ...STARS, rows, summary: { ...STARS.summary, total: rows.length } } });
    show(<Psf />, "/synthetic/psf");
    expect(await screen.findByRole("figure", { name: "Magnitude distribution of the star catalogue" }, { timeout: 8000 })).toBeTruthy();
  }, 20_000);

  it("a band-state filter narrows the table; the Best band filter uses a star's best band", async () => {
    const { unmount } = show(<Psf />, "/synthetic/psf?band=H_E&bst=corrupted");
    const grid = await screen.findByRole("grid", { name: "Stars" });
    await waitFor(() => expect(within(grid).getAllByRole("row")).toHaveLength(2));
    expect(within(grid).getByRole("img", { name: /H corrupted/ })).toBeTruthy();
    unmount();
    show(<Psf />, "/synthetic/psf?bst=corrupted");
    const all = await screen.findByRole("grid", { name: "Stars" });
    await waitFor(() => expect(within(all).queryByRole("img", { name: /H corrupted/ })).toBeNull());
  });

  it("opens a star in the inspector; switching views drops the view's own band key", async () => {
    show(<Psf />, "/synthetic/psf?band=H_E");
    const grid = await screen.findByRole("grid", { name: "Stars" });
    fireEvent.click(within(grid).getAllByText(/^1[78]\.\d0$/)[0]);
    expect(useInspector.getState().current?.kind).toBe("star");
    fireEvent.click(screen.getByRole("radio", { name: "Cutouts" }));
    await waitFor(() => expect(loc()).toBe("/synthetic/psf?view=cutouts"));
  });

  it("an unsynchronised mirror points at How this is produced, never a local copy", async () => {
    routes["GET /api/catalog/stars"] = () => ({ body: { ...STARS, present: false, rows: [], summary: null, band_stats: [] } });
    show(<Psf />, "/synthetic/psf");
    expect(await screen.findByText("The FASRC star catalogue is not synchronised")).toBeTruthy();
    fireEvent.click(screen.getAllByRole("button", { name: "How this is produced" }).at(-1)!);
    await waitFor(() => expect(loc()).toContain("how=1"));
  });

  it("shows the server's error text", async () => {
    routes["GET /api/catalog/stars"] = () => ({ status: 500, body: { error: "stars.csv: bad header" } });
    show(<Psf />, "/synthetic/psf");
    expect(await screen.findByText(/stars\.csv: bad header/, {}, { timeout: 5000 })).toBeTruthy();
  });
});

describe("PSF › cutouts", () => {
  beforeEach(() => {
    viewer.id = "1";
    routes["GET /api/star-cutouts/totals"] = () => ({ body: { count: 1, size: 511, cached: true,
      catalog: { present: true, path: "/n/x", mtime: Date.now() / 1000 - 60, age_s: 60 } } });
    const page = (band: string) => ({ body: {
      band, files: ["star_0001_255.fits", "star_0001_511.fits"], total: 2, page: 1, n_pages: 1, per_page: 96, output_dir: "/o",
      items: [
        { file: "star_0001_255.fits", id: 1, size: 255, ra: 269.7, dec: 66, mag: 17.5 },
        { file: "star_0001_511.fits", id: 1, size: 511, ra: 269.7, dec: 66, mag: 17.5 },
      ],
    } });
    for (const b of ["VIS", "Y_E", "J_E", "H_E"]) routes[`GET /api/cutouts/${b}/list.json?page=1&per_page=96`] = () => page(b);
  });

  it("marks the target star, keeps its catalogue magnitude apart from the whole cutout's, and opens gallery stars", async () => {
    show(<Psf />, "/synthetic/psf?view=cutouts");
    const marks = await screen.findByRole("group", { name: "markers on every tier" });
    const target = within(marks).getByRole("button", { name: "Star 1 (the catalogue target)" });
    expect([target.getAttribute("data-x"), target.getAttribute("data-y")]).toEqual(["255", "255"]);
    const current = await screen.findByRole("group", { name: "Star 1" });
    expect(within(current).getByText("star VIS 17.50 AB")).toBeTruthy();
    // the navigator's count is its pager's: no badge repeats it; the gallery counts stars, not files
    expect(screen.queryByText("1 stars")).toBeNull();
    expect(screen.getByText("1 star on this page")).toBeTruthy();
    const thumb = await screen.findByRole("button", { name: "Star 1, VIS 17.50: show in the viewer" });
    expect(thumb.getAttribute("aria-current")).toBe("true");
    expect(thumb.querySelector("img")?.getAttribute("src")).toMatch(/[?&]stretch=star$/);   // a per-star stretch
    fireEvent.click(thumb);
    await waitFor(() => expect(viewer.goToId).toEqual(["1"]));
    expect(viewer.display).toBeUndefined();          // the cutouts are served in e⁻: no page-side stretch
  });

  it("draws the per-band validity as one stacked bar per band", async () => {
    show(<Psf />, "/synthetic/psf?view=cutouts");
    const bars = await screen.findByRole("region", { name: "Cutout validity per band" });
    const imgs = within(bars).getAllByRole("img");
    expect(imgs.map((i) => i.getAttribute("aria-label"))).toEqual([
      "VIS: 2 valid, 1 pending", "Y: 1 valid, 2 pending", "J: 1 valid, 1 failed, 1 pending", "H: 1 valid, 1 corrupted, 1 pending"]);
    expect(within(bars).getAllByText(/% valid$/).map((e) => e.textContent)).toEqual(["67% valid", "33% valid", "33% valid", "33% valid"]);
  });

  it("the old /cutouts/<band> links pick the gallery band (?band=, and the interim ?gband=)", async () => {
    const { unmount } = show(<Psf />, "/synthetic/psf?view=cutouts&band=Y_E");
    expect((await screen.findByRole("radio", { name: "Y" })).getAttribute("aria-checked")).toBe("true");
    unmount();
    show(<Psf />, "/synthetic/psf?view=cutouts&gband=J_E");
    expect((await screen.findByRole("radio", { name: "J" })).getAttribute("aria-checked")).toBe("true");
  });
});

const INVENTORY = {
  bands: [
    { name: "VIS", fwhm: 0.16, oversampling: 3, epsf_pixel_scale: 0.033, state: "empirical", empirical: true,
      path: "data/_fasrc_cache/x/euclid_psf_VIS.fits", size_bytes: 7e7, synced_at: 1_790_000_000, n_psf: 2, shape: [99, 99],
      pixel_scale: 0.0333, measured_fwhm: 0.17, last_sync: { ok: true } },
    { name: "Y_E", fwhm: 0.49, oversampling: 3, epsf_pixel_scale: 0.1, state: "no_empirical", empirical: false,
      last_sync: { ok: false, error: "No such file", missing_remote: true } },
    { name: "J_E", fwhm: 0.49, oversampling: 3, epsf_pixel_scale: 0.1, state: "not_cached", empirical: false, error: null, last_sync: null },
    { name: "H_E", fwhm: 0.5, oversampling: 3, epsf_pixel_scale: 0.1, state: "not_cached", empirical: false, error: null, last_sync: null },
  ],
  clusters: [{ index: 1, id: "cluster-001", ra: 269.1, dec: 66.2, n_stars: 12, fwhm_by_band: { VIS: 0.171 } }],
  clusters_source: "metadata", clusters_meta: { present: true, synced_at: 1_790_000_000 }, last_sync: 1_790_000_000,
};

describe("PSF › ePSF", () => {
  beforeEach(() => {
    viewer.id = "cluster-001";
    routes["GET /api/euclid-psf/inventory"] = () => ({ body: INVENTORY });
  });

  it("leads with what generation uses, then the kernel viewer with the cluster's FWHM, ePSF vs Gaussian and the cluster map", async () => {
    const { container } = show(<Psf />, "/synthetic/psf?view=epsf");
    await waitFor(() => expect(container.querySelector(".syn-sentence")?.textContent).toBe(
      "Used by generation: Gaussian fallback in Y; empirical in VIS; J, H not synced here"));
    expect(container.querySelectorAll(".ui-summary")).toHaveLength(1);        // the PSF header's, alone
    expect(screen.getByTestId("viewer").textContent).toBe("psfs||");
    expect(screen.getByText("FWHM VIS 0.171″")).toBeTruthy();
    const table = within(screen.getByRole("table", { name: "ePSF and Gaussian FWHM per band" }));
    expect(table.getAllByRole("row").slice(1).map((r) => r.textContent)).toEqual([
      "VIS0.1700.160Empirical", "Y—0.490Gaussian fallback", "J—0.490Not synced here", "H—0.500Not synced here"]);
    expect(screen.getByText("Kernels and files").closest("details")?.open).toBe(false);
    // Each synced band's file opens in the FITS inspector.
    fireEvent.click(screen.getByRole("button", { name: "Open the VIS ePSF file in the FITS inspector" }));
    expect(useInspector.getState().current).toMatchObject({ kind: "fits", id: "data/_fasrc_cache/x/euclid_psf_VIS.fits" });
    expect(screen.getByRole("figure", { name: "PSF clusters on the sky coloured by their VIS FWHM" })).toBeTruthy();
    // the RA/Dec list became the map; the table is one click away
    expect(screen.queryByRole("grid", { name: "PSF clusters" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /Clusters as a table/ }));
    const clusters = await screen.findByRole("grid", { name: "PSF clusters" });
    fireEvent.click(within(clusters).getByText("12"));
    await waitFor(() => expect(viewer.goToId).toContain("cluster-001"));
    expect(jobPosts()).toEqual([]);
  });

  it("puts the warp parameters in the live-warps switch's tooltip", async () => {
    show(<Psf />, "/synthetic/psf?view=epsf");
    const sw = await screen.findByRole("switch", { name: "Live warps" });
    expect(screen.queryByText(/α up to 20/)).toBeNull();                    // no status text on the page
    fireEvent.focus(sw.closest("span")!);
    expect((await screen.findAllByText(/displacement α up to 20, smoothed over σ 3 px/)).length).toBeGreaterThan(0);
  });
});

describe("PSF › How this is produced", () => {
  it("holds the chain's steps and actions, each confirmed", async () => {
    routes["POST /api/status/refresh-catalog"] = () => ({ body: { ok: true, catalog: {} } });
    routes["POST /api/euclid-psf/sync"] = () => ({ body: { ok: true, job_id: "p1" } });
    routes["GET /api/jobs/p1"] = () => ({ body: RUNNING_JOB("p1", "PSFs: sync ePSFs from FASRC") });
    show(<Psf />, "/synthetic/psf?how=1");
    for (const h of ["1 · Star catalogue", "2 · Cutouts", "3 · ePSFs"]) expect(await screen.findByRole("heading", { name: h })).toBeTruthy();
    for (const name of [/Query the catalogue/, /Verify the photometry scale/, /Download the star cutouts/, /Extract the ePSFs/, /Pre-rotate the kernel pools/]) {
      expect(screen.getByRole("button", { name })).toBeTruthy();
    }
    fireEvent.click(screen.getByRole("button", { name: "Pull stars.csv" }));
    await answer(/Pull stars.csv from FASRC/, "Pull");
    await waitFor(() => expect(isPost("/api/status/refresh-catalog")).toHaveLength(1));
    const sync = screen.getByRole("button", { name: "Sync ePSFs" }) as HTMLButtonElement;
    await waitFor(() => expect(sync.disabled).toBe(false));
    fireEvent.click(sync);
    await answer(/Sync the ePSFs from FASRC/, "Sync");
    expect(await screen.findByText("PSFs: sync ePSFs from FASRC")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Sync PSF cluster metadata" })).toBeTruthy();
  });
});

/* ── Galaxies › templates (TNG) ────────────────────────────────────────── */

const TNG = {
  present: true, files: { properties: { present: true, name: "tng_properties.csv", rows: 3, mtime: Date.now() / 1000 - 3600 },
    atlas: { present: true, name: "tng_atlas_parameters.csv", rows: 3, mtime: 1_790_000_000 } },
  atlas_meta: null,
  columns: ["id", "sfr", "mass_stars", "m_halo", "reff", "re_kpc", "re_kpc_min", "re_kpc_max", "n_orient", "local"],
  rows: [[1, 0.5, 3e11, 2e12, 8.4, 6, 5, 7, 2, 0], [9, 2, 1e9, 1e11, 1.5, 1, 1, 1, 1, 4], [12, 0, 5e10, 1e12, 3, 2.5, 2, 3, 5, 0]],
  orientations: { 9: [[1, 10, 1.0]] },
  summary: { n: 3, n_quenched: 1, n_missing_sfr: 0, n_in_atlas: 3, n_local: 1 },
};

describe("Galaxies › templates", () => {
  beforeEach(() => {
    routes["GET /tng-auth/status"] = () => ({ body: { present: true, connected: true, chars: 32 } });
    routes["GET /api/tng/properties"] = () => ({ body: TNG });
    routes["GET /api/tng/results"] = () => ({ body: { grid: { present: false }, stack: { present: false }, pull_job: null } });
    routes["GET /api/tng/radii/status"] = () => ({ body: { valid: true, stale: true, connected: true, refresh_job: null,
      expected_count: 15, valid_count: 15, checked_at: Date.now() / 1000 - 2 * 86_400 } });
  });

  it("captions the atlas once, draws SFR = 0 on a floor strip and lists the galaxies", async () => {
    show(<Galaxies />, "/synthetic/galaxies?view=templates");
    expect(await screen.findByText("TNG50-1 atlas: 3 galaxies (3 with measured Rₑ)")).toBeTruthy();
    expect(await screen.findByRole("figure", { name: /SFR against Stellar mass, coloured by Measured VIS Rₑ/ })).toBeTruthy();
    expect(screen.getByRole("button", { name: "SFR = 0: 1" })).toBeTruthy();         // the quenched galaxy, not dropped
    expect(screen.getByRole("figure", { name: "Distribution of Stellar mass" })).toBeTruthy();   // the explorer's marginal
    const grid = screen.getByRole("grid", { name: "TNG galaxies" });
    fireEvent.click(within(grid).getByText("9"));
    expect(useInspector.getState().current).toEqual({ kind: "tng", id: "9" });
    // Freshness and the token only when they matter.
    expect(screen.queryByText(/token saved/)).toBeNull();
    expect(screen.queryByText(/^properties /)).toBeNull();
    expect(screen.getByText("No template grid pulled yet")).toBeTruthy();
  });

  it("states the radius manifest ONCE and never validates it here (the fix is on Status)", async () => {
    show(<Galaxies />, "/synthetic/galaxies?view=templates");
    expect(await screen.findByText("15 of 15 measured radii valid · checked 2 d ago")).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Validate/ })).toBeNull();
    expect(screen.queryByText(/out of date/)).toBeNull();
    await new Promise((r) => setTimeout(r, 30));
    expect(isPost("/api/tng/radii/refresh")).toHaveLength(0);
  });

  it("an invalid manifest links to its fix on Status", async () => {
    routes["GET /api/tng/radii/status"] = () => ({ body: { valid: false, stale: true, connected: true, refresh_job: null,
      expected_count: 15, valid_count: 0, reasons: ["manifest hash mismatch"] } });
    show(<Galaxies />, "/synthetic/galaxies?view=templates");
    expect(await screen.findByText("Radius manifest invalid: manifest hash mismatch")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Validate on Status" }).getAttribute("href")).toBe("/synthetic/status");
  });

  it("a missing API token is a badge; the TNG steps and Refresh properties sit in How this is produced", async () => {
    routes["GET /tng-auth/status"] = () => ({ body: { present: false, connected: true } });
    routes["POST /api/tng/properties/refresh"] = () => ({ body: { ok: true, job_id: "t1" } });
    routes["GET /api/jobs/t1"] = () => ({ body: RUNNING_JOB("t1", "TNG properties") });
    show(<Galaxies />, "/synthetic/galaxies?view=templates&how=1");
    expect(await screen.findByText("No TNG API token")).toBeTruthy();
    const refresh = await screen.findByRole("button", { name: "Refresh properties" }) as HTMLButtonElement;
    await waitFor(() => expect(refresh.disabled).toBe(false));
    fireEvent.click(refresh);
    await answer(/Refresh the galaxy properties from the TNG API/, "Refresh");
    await waitFor(() => expect(isPost("/api/tng/properties/refresh")).toHaveLength(1));
  });
});

/* ── inspectors ────────────────────────────────────────────────────────── */

describe("inspector cards", () => {
  it("star: position, per-band states and links", async () => {
    show(<StarInspector id="2" />, "/synthetic/psf");
    expect(await screen.findByText("Star 2")).toBeTruthy();
    expect(screen.getByText("corrupted")).toBeTruthy();
    expect(screen.getByRole("link", { name: "On sky" }).getAttribute("href")).toContain("/sky/atlas?ra=61.2&dec=-48.4");
    expect(screen.queryByRole("link", { name: "Cutouts" })).toBeNull();
  });

  it("truth: every column behind a disclosure, the record and TNG links", async () => {
    routes["GET /api/sky/records/source?subset=test&index=1&row=0"] = () => ({ body: {
      subset: "test", field_index: 1, row: 0, source: SOURCES.sources[0],
      values: { type: "galaxy", x_pix: 10, y_pix: 20, flux_vis_e: 1000, re_arcsec: 0.2, tng_render_trace: { a: 1 } },
    } });
    show(<TruthInspector id="test/1/0" />);
    expect(await screen.findByText("Record 1 of test, source 0")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Record" }).getAttribute("href")).toBe("/synthetic/records?split=test&v.rec.id=test%3A1");
    fireEvent.click(screen.getByRole("button", { name: "TNG 658592" }));
    expect(useInspector.getState().current).toEqual({ kind: "tng", id: "658592" });
  });

  it("psf: cluster facts", async () => {
    routes["GET /api/euclid-psf/inventory"] = () => ({ body: INVENTORY });
    show(<PsfInspector id="1" />);
    expect(await screen.findByText("cluster-001")).toBeTruthy();
    expect(screen.getByText("0.171″")).toBeTruthy();
  });

  it("tng: properties and the local VIS frames", async () => {
    routes["GET /api/tng/properties"] = () => ({ body: TNG });
    show(<TngInspector id="9" />);
    expect(await screen.findByText("TNG50 subhalo 9")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "VIS" }));
    expect(useInspector.getState().current).toEqual({ kind: "fits", id: "data/tng_skirt/9/TNG9_O1_Euclid_VIS.fits" });
  });
});
