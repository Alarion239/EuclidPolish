/* Data workspace tabs and inspector cards against a mocked backend
 * (routes/{views,cutouts,psfs,tng}.py). The image viewer is mocked: it
 * reports an object through onState and records goTo / goToId. The TNG radii
 * cases are re-homed from the deleted pages/contracts.test.tsx. */
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
import { PsfInspector, StarInspector, TngInspector, TruthInspector } from "./inspectors";
import "./register";
import Catalog from "./tabs/Catalog";
import Cutouts from "./tabs/Cutouts";
import Psfs from "./tabs/Psfs";
import Records from "./tabs/Records";
import Tng from "./tabs/Tng";

type MockViewerProps = {
  collection: string; urlKey?: string; params?: Record<string, string>; tiers?: string[];
  onReady?: (api: unknown) => void; onState?: (s: unknown) => void;
};
const viewer = vi.hoisted(() => ({ goTo: [] as number[], goToId: [] as string[], mounts: 0, index: 1, id: "test:1" as string | null }));
vi.mock("../../viewer", async () => {
  const { useEffect } = await import("react");
  return {
    ImageViewer: (p: MockViewerProps) => {
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
      return <div data-testid="viewer">{p.collection}|{p.params?.subset ?? ""}|{(p.tiers ?? []).join(",")}</div>;
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

const show = (el: ReactElement, url = "/data/records") => render(
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
  { field_index: 1, galaxy: 1, star: 1, lens: 0, other: 0, off_field: 1, n: 2, brightest_star_mag: 18.5, brightest_galaxy_mag: 24, total_vis_e: 2e5 },
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
  // exclusive states (valid > corrupted > failed > pending): each band row and the summary sum to the total
  summary: { total: 3, valid: 2, corrupted: 0, failed: 1, pending: 0, valid_all4: 1, navigator: { size: 511, count: 1 }, mag_min: 17.5, mag_max: 18.9 },
  band_stats: [
    { band: "VIS", valid: 2, corrupted: 0, failed: 0, pending: 1, by_size: { 255: 0, 511: 2 } },
    { band: "Y_E", valid: 1, corrupted: 0, failed: 0, pending: 2, by_size: { 255: 0, 511: 1 } },
    { band: "J_E", valid: 1, corrupted: 0, failed: 1, pending: 1, by_size: { 255: 0, 511: 1 } },
    { band: "H_E", valid: 1, corrupted: 1, failed: 0, pending: 1, by_size: { 255: 0, 511: 1 } },
  ],
};

const STEPS = { ssh_connected: true, steps: [], artifacts: {}, remote_paths: {} };

beforeEach(() => {
  posts = [];
  viewer.goTo = []; viewer.goToId = []; viewer.mounts = 0; viewer.index = 1; viewer.id = "test:1";
  routes = {
    "GET /api/fasrc/status": () => ({ body: { ssh_connected: true } }),
    "GET /api/fasrc/steps/status": () => ({ body: STEPS }),
    "GET /api/system/alerts": () => ({ body: { computed_at: "", ttl_s: 30, alerts: [], counts: {}, checks: [
      { id: "records-noise", label: "Records noise model", state: "bad", title: "Training records use an older noise model", detail: "dirty_test: v4" },
    ] } }),
    "GET /api/sky/sr-status": () => ({ body: SR_STATUS }),
    "GET /api/sky/records/sources?subset=test&index=1": () => ({ body: SOURCES }),
    "GET /api/sky/records/sources?subset=test": () => ({ body: CENSUS }),
    "GET /api/catalog/stars": () => ({ body: STARS }),
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
  it("shows the split, its files, the SR state and the noise check", async () => {
    show(<Records />);
    expect(await screen.findByText("SR stale")).toBeTruthy();
    expect(screen.getByText("Clean")).toBeTruthy();
    expect(screen.getByText("noise model")).toBeTruthy();
    expect(screen.getByTestId("viewer").textContent).toBe("sky|test|dirty,hr");
  });

  it("maps the current record's truth sources and opens a source in the inspector", async () => {
    show(<Records />);
    const map = await screen.findByRole("img", { name: /Truth sources of record 1: 2 shown/ });
    const star = within(map).getByRole("button", { name: /star · \(11\.0, 20\.0\) px · VIS 18\.50/ });
    fireEvent.click(star);
    expect(useInspector.getState().current).toEqual({ kind: "truth", id: "test/1/1" });
    expect(useInspectorRegistry.getState().kinds.truth).toBeTruthy();
    // the type chips filter the map and the table
    fireEvent.click(screen.getByRole("button", { name: /^star 1$/ }));
    await waitFor(() => expect(screen.getByTestId("loc").textContent).toContain("st=star"));
    expect(within(screen.getByRole("img", { name: /Truth sources/ })).getAllByRole("button")).toHaveLength(1);
  });

  it("the census row moves the viewer", async () => {
    show(<Records />);
    const table = await screen.findByRole("grid", { name: "Records in test" });
    fireEvent.click(await within(table).findByText(/^22\.1/));
    expect(viewer.goTo).toEqual([0]);
  });

  it("syncs the chosen splits as a job", async () => {
    routes["POST /api/sky/sync"] = () => ({ body: { ok: true, job_id: "s1" } });
    routes["GET /api/jobs/s1"] = () => ({ body: RUNNING_JOB("s1", "records: sync test+validate+train from FASRC") });
    show(<Records />);
    fireEvent.click(await screen.findByRole("button", { name: "Sync" }));
    fireEvent.click(await screen.findByRole("checkbox", { name: /train/ }));
    fireEvent.click(screen.getByRole("button", { name: "Start sync" }));
    await answer(/Sync test \+ validate \+ train from FASRC/, "Sync");
    await waitFor(() => expect(posts.map((p) => p.url)).toContain("/api/sky/sync"));
    expect(posts[0].form).toEqual({ subsets: "test,validate,train", kinds: "dirty,hr,clean,sources" });
    expect(await screen.findByText(/records: sync test\+validate\+train/)).toBeTruthy();
  });

  it("generates the SR with overwrite preselected for a stale split", async () => {
    routes["POST /api/sky/generate-sr"] = () => ({ body: { ok: true, job_id: "g1" } });
    routes["GET /api/jobs/g1"] = () => ({ body: RUNNING_JOB("g1", "records: generate SR (test, overwrite)") });
    show(<Records />);
    fireEvent.click(await screen.findByRole("button", { name: "Generate SR" }));
    expect((await screen.findByRole("switch", { name: "Overwrite existing SR" })).getAttribute("aria-checked")).toBe("true");
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));
    await answer(/Generate the production SR for test/, "Generate");
    await waitFor(() => expect(posts.find((p) => p.url === "/api/sky/generate-sr")?.form).toEqual({ subsets: "test", overwrite: "1" }));
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
    show(<Records />, "/data/records?gen=1");
    expect(await screen.findByDisplayValue("--seed 2")).toBeTruthy();
    expect(screen.queryByDisplayValue(/regenerate/)).toBeNull();
    expect(screen.getByText("resume")).toBeTruthy();
    expect(screen.getByText("dropped --regenerate-splits=train")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /^(Submit|Queue)/ }));
    await answer(/Generate synthetic training pairs/, "Submit");
    await waitFor(() => expect(posts.find((p) => p.url.endsWith("/synthetic_generate/submit"))).toBeTruthy());
    const body = posts.find((p) => p.url.endsWith("/synthetic_generate/submit"))!.form;
    expect(body.extra_flags).toBe("--seed 2");
    expect(body.regenerate_splits).toBeUndefined();
  });

  it("offers a sync when the split is not on this machine", async () => {
    show(<Records />, "/data/records?split=validate");
    expect(await screen.findByText("No validate records on this machine")).toBeTruthy();
    expect(screen.queryByTestId("viewer")).toBeNull();
  });

  it("the sync is disabled while FASRC is offline", async () => {
    routes["GET /api/fasrc/status"] = () => ({ body: { ssh_connected: false } });
    show(<Records />);
    await screen.findByText("SR stale");
    await waitFor(() => expect((screen.getByRole("button", { name: "Sync" }) as HTMLButtonElement).disabled).toBe(true));
  });
});

/* ── catalog ───────────────────────────────────────────────────────────── */

describe("Catalog", () => {
  it("lists the mirror's stars with their band states and filters them from the URL", async () => {
    show(<Catalog />, "/data/catalog?cut=nav");
    const grid = await screen.findByRole("grid", { name: "Stars" });
    await waitFor(() => expect(within(grid).getAllByRole("row")).toHaveLength(2));   // header + star 1
    expect(within(grid).getByRole("img", { name: "VIS valid, Y valid, J valid, H valid" })).toBeTruthy();
    expect(screen.getByText("Valid in all 4")).toBeTruthy();
  });

  it("a band-state filter narrows the table", async () => {
    show(<Catalog />, "/data/catalog?band=H_E&bst=corrupted");
    const grid = await screen.findByRole("grid", { name: "Stars" });
    await waitFor(() => expect(within(grid).getAllByRole("row")).toHaveLength(2));
    expect(within(grid).getByRole("img", { name: /H corrupted/ })).toBeTruthy();
  });

  it("the Overall band filter uses a star's best band, matching the KPI strip", async () => {
    show(<Catalog />, "/data/catalog?bst=corrupted");
    const grid = await screen.findByRole("grid", { name: "Stars" });
    // star 2 has a corrupted H but a valid VIS: overall valid, like summary.corrupted = 0
    await waitFor(() => expect(within(grid).queryAllByRole("img", { name: /^VIS / })).toHaveLength(0));
    expect(within(grid).queryByRole("img", { name: /H corrupted/ })).toBeNull();
  });

  it("opens a star in the inspector and pulls the mirror on demand", async () => {
    routes["POST /api/status/refresh-catalog"] = () => ({ body: { ok: true, catalog: {} } });
    show(<Catalog />, "/data/catalog");
    const grid = await screen.findByRole("grid", { name: "Stars" });
    fireEvent.click(within(grid).getByText("17.500"));
    expect(useInspector.getState().current).toEqual({ kind: "star", id: "1" });
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }));
    await waitFor(() => expect(posts.map((p) => p.url)).toContain("/api/status/refresh-catalog"));
  });

  it("an unsynchronised mirror shows the pull action, never a local copy", async () => {
    routes["GET /api/catalog/stars"] = () => ({ body: { ...STARS, present: false, rows: [], summary: null, band_stats: [] } });
    show(<Catalog />, "/data/catalog");
    expect(await screen.findByText("The FASRC star catalogue is not synchronised")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Pull from FASRC" })).toBeTruthy();
  });

  it("shows the server's error text", async () => {
    routes["GET /api/catalog/stars"] = () => ({ status: 500, body: { error: "stars.csv: bad header" } });
    show(<Catalog />, "/data/catalog");
    expect(await screen.findByText(/stars\.csv: bad header/, {}, { timeout: 5000 })).toBeTruthy();
  });
});

/* ── cutouts ───────────────────────────────────────────────────────────── */

describe("Cutouts", () => {
  beforeEach(() => {
    viewer.id = "1";
    routes["GET /api/star-cutouts/totals"] = () => ({ body: { count: 1, size: 511, cached: true,
      catalog: { present: true, path: "/n/x", mtime: Date.now() / 1000 - 60, age_s: 60 } } });
    routes["GET /api/cutouts/VIS/list.json?page=1&per_page=48"] = () => ({ body: {
      band: "VIS", files: ["star_0001_511.fits"], total: 1, page: 1, n_pages: 1, per_page: 48, output_dir: "/o",
      items: [{ file: "star_0001_511.fits", id: 1, size: 511, ra: 269.7, dec: 66, mag: 17.5 }],
    } });
  });

  it("labels the navigator's star and opens gallery stars in it", async () => {
    show(<Cutouts />, "/data/cutouts");
    expect(await screen.findByText("1 stars @ 511 px")).toBeTruthy();
    expect(await screen.findByText("star 1")).toBeTruthy();                     // the current star's facts
    fireEvent.click(await screen.findByRole("button", { name: /1 · 511px · 17\.50/ }));
    await waitFor(() => expect(viewer.goToId).toEqual(["1"]));
  });
});

/* ── PSFs ──────────────────────────────────────────────────────────────── */

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

describe("PSFs", () => {
  beforeEach(() => {
    routes["GET /api/euclid-psf/inventory"] = () => ({ body: INVENTORY });
    routes["GET /api/config"] = () => ({ body: { config: { psf_warp_alpha_max: 20, psf_warp_sigma: 3 } } });
  });

  it("distinguishes not-cached from no-empirical bands and lists the clusters", async () => {
    show(<Psfs />, "/data/psfs");
    expect(await screen.findByText("Y · no empirical PSF")).toBeTruthy();
    expect(screen.getByText("J · not cached")).toBeTruthy();
    expect(screen.getByText("Gaussian fallback in use")).toBeTruthy();
    const clusters = screen.getByRole("grid", { name: "PSF clusters" });
    fireEvent.click(within(clusters).getByRole("button", { name: "View" }));
    expect(viewer.goToId).toContain("cluster-001");
    expect(useInspector.getState().current).toEqual({ kind: "psf", id: "1" });
  });

  it("runs the ePSF sync as a job", async () => {
    routes["POST /api/euclid-psf/sync"] = () => ({ body: { ok: true, job_id: "p1" } });
    routes["GET /api/jobs/p1"] = () => ({ body: RUNNING_JOB("p1", "PSFs: sync ePSFs from FASRC") });
    show(<Psfs />, "/data/psfs");
    const button = await screen.findByRole("button", { name: "Sync ePSFs" }) as HTMLButtonElement;
    await waitFor(() => expect(button.disabled).toBe(false));
    fireEvent.click(button);
    await answer(/Sync the ePSFs from FASRC/, "Sync");
    expect(await screen.findByText("PSFs: sync ePSFs from FASRC")).toBeTruthy();
  });
});

/* ── TNG ───────────────────────────────────────────────────────────────── */

const TNG = {
  present: true, files: { properties: { present: true, name: "tng_properties.csv", rows: 2, mtime: 1_790_000_000 },
    atlas: { present: true, name: "tng_atlas_parameters.csv", rows: 2, mtime: 1_790_000_000 } },
  atlas_meta: null,
  columns: ["id", "sfr", "mass_stars", "m_halo", "reff", "re_kpc", "re_kpc_min", "re_kpc_max", "n_orient", "local"],
  rows: [[1, 0.5, 3e11, 2e12, 8.4, 6, 5, 7, 2, 0], [9, 2, 1e9, 1e11, 1.5, 1, 1, 1, 1, 4]],
  orientations: { 9: [[1, 10, 1.0]] },
  summary: { n: 2, n_quenched: 0, n_missing_sfr: 0, n_in_atlas: 2, n_local: 1 },
};

describe("TNG", () => {
  beforeEach(() => {
    routes["GET /tng-auth/status"] = () => ({ body: { present: true, connected: true, chars: 32 } });
    routes["GET /api/tng/properties"] = () => ({ body: TNG });
    routes["GET /api/tng/results"] = () => ({ body: { grid: { present: false }, stack: { present: false }, pull_job: null } });
  });

  it("starts the radii refresh job when the cache is stale and FASRC is connected, and shows it", async () => {
    routes["GET /api/tng/radii/status"] = () => ({ body: { valid: false, stale: true, connected: true, refresh_job: null, expected_count: 10, valid_count: 0 } });
    routes["POST /api/tng/radii/refresh"] = () => ({ body: { ok: true, job_id: "r1" } });
    routes["GET /api/jobs/r1"] = () => ({ body: RUNNING_JOB("r1", "TNG radii validation") });
    show(<Tng />, "/data/tng");
    await waitFor(() => expect(posts.filter((p) => p.url === "/api/tng/radii/refresh")).toHaveLength(1));
    expect(await screen.findByText("TNG radii validation")).toBeTruthy();
  });

  it("does not start it while offline or when a refresh already runs", async () => {
    routes["GET /api/tng/radii/status"] = () => ({ body: { valid: false, stale: true, connected: false, refresh_job: null } });
    const { unmount } = show(<Tng />, "/data/tng");
    expect(await screen.findByText(/connect to FASRC to re-check it/)).toBeTruthy();
    unmount();
    queryClient.clear();
    routes["GET /api/tng/radii/status"] = () => ({ body: { valid: false, stale: true, connected: true, refresh_job: "other" } });
    show(<Tng />, "/data/tng");
    await screen.findByText(/frames valid/);
    expect(posts.filter((p) => p.url === "/api/tng/radii/refresh")).toHaveLength(0);
  });

  it("explores the properties and lists the galaxies", async () => {
    routes["GET /api/tng/radii/status"] = () => ({ body: { valid: true, stale: false, connected: true, refresh_job: null, expected_count: 2, valid_count: 2 } });
    show(<Tng />, "/data/tng");
    expect(await screen.findByRole("figure", { name: /SFR against Stellar mass, coloured by Measured VIS Rₑ/ })).toBeTruthy();
    const grid = screen.getByRole("grid", { name: "TNG galaxies" });
    fireEvent.click(within(grid).getByText("9"));
    expect(useInspector.getState().current).toEqual({ kind: "tng", id: "9" });
    expect(screen.getByText("2 galaxies · 2 measured · 1 local")).toBeTruthy();
  });
});

/* ── inspectors ────────────────────────────────────────────────────────── */

describe("inspector cards", () => {
  it("star: position, per-band states and links", async () => {
    show(<StarInspector id="2" />, "/data/catalog");
    expect(await screen.findByText("Star 2")).toBeTruthy();
    expect(screen.getByText("corrupted")).toBeTruthy();
    expect(screen.getByRole("link", { name: "On sky" }).getAttribute("href")).toContain("/sky/atlas?ra=61.2&dec=-48.4");
    expect(screen.queryByRole("link", { name: "Cutouts" })).toBeNull();          // not in the navigator
  });

  it("truth: every column behind a disclosure, the record and TNG links", async () => {
    routes["GET /api/sky/records/source?subset=test&index=1&row=0"] = () => ({ body: {
      subset: "test", field_index: 1, row: 0, source: SOURCES.sources[0],
      values: { type: "galaxy", x_pix: 10, y_pix: 20, flux_vis_e: 1000, re_arcsec: 0.2, tng_render_trace: { a: 1 } },
    } });
    show(<TruthInspector id="test/1/0" />);
    expect(await screen.findByText("test · record 1 · #0")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Record" }).getAttribute("href")).toBe("/data/records?split=test&v.rec.id=test%3A1");
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
