/* Models › Spectra against a mocked Flask (fixtures in
 * ./testing/spectraFixture): opening the tab only GETs the summary (no
 * POST, no per-model read outside Direction); Power draws the truth and the
 * default pick (current spectra: the best ∫PSNR, the newest without the
 * isotropy rule, the newest) with the transfer, and names the models it
 * cannot draw with their reason; the URL picks the section, the band and the
 * models (a stale one drawn when picked), the picker adds one; Coherence
 * says where each falls to ½; Direction states which models show a grid,
 * reads one model's map on demand (another from its picker, a 404 said so),
 * draws the angular profiles by period band and marks the grid cells of the
 * axis / diagonal table; Compute spectra asks first, POSTs spectra=1 to the
 * score job, shows it and reads the spectra again when it ends; a score job
 * started elsewhere is followed; records that are not local, no spectra yet
 * and a partial payload are each said plainly. */
import { QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { JOBS_FEED_TIMING, useJobsStore, type Job } from "../../api/jobs";
import { queryClient } from "../../api/query";
import { resetConfirm } from "../../ui";
import Spectra from "./tabs/Spectra";
import { REASON, payload, record, row, SPECS } from "./testing/spectraFixture";

type Reply = { status?: number; body: unknown };
let routes: Record<string, (form: Record<string, string>) => Reply>;
let calls: { url: string; method: string; form: Record<string, string> }[];
const posts = () => calls.filter((c) => c.method === "POST");
const gets = (u: string) => calls.filter((c) => c.method === "GET" && c.url === u);
const SUMMARY = "/ensemble/universal/spectra.json";
const recordUrl = (n: number) => `/ensemble/universal/spectra/member_${n}.json`;

const formOf = (body: BodyInit | null | undefined): Record<string, string> => {
  const out: Record<string, string> = {};
  if (body instanceof FormData) body.forEach((v, k) => { out[k] = String(v); });
  else if (body instanceof URLSearchParams) body.forEach((v, k) => { out[k] = v; });
  else if (typeof body === "string") new URLSearchParams(body).forEach((v, k) => { out[k] = v; });
  return out;
};

const job = (id: string, status: string, label = "universal: score every universal model on the test fields + spectra"): Job => ({
  job_id: id, label, kind: "universal-score", status, duration: 3, error: null, log: "", log_truncated: false,
  progress: { current: 1, total: 4, pct: 25, label: "member_230: field 25/100" },
});

beforeEach(() => {
  calls = [];
  routes = {
    [`GET ${SUMMARY}`]: () => ({ body: payload() }),
    ...Object.fromEntries(SPECS.filter((s) => s.a != null).map((s) => [`GET ${recordUrl(s.n)}`, () => ({ body: record(s.n) })])),
    "GET /api/jobs?summary=1": () => ({ body: [] }),
    "POST /ensemble/universal/score": () => ({ body: { ok: true, job_id: "sp1", already_running: false } }),
    "GET /api/jobs/sp1": () => ({ body: job("sp1", "running") }),
  };
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
});
afterEach(() => { act(() => resetConfirm()); queryClient.clear(); });

let lastLocation = "";
function Spy() { const l = useLocation(); lastLocation = l.pathname + l.search; return null; }

const show = (url = "/models/spectra") => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={[url]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes><Route path="/models/*" element={<><Spectra /><Spy /></>} /></Routes>
    </MemoryRouter>
  </QueryClientProvider>,
);

const figure = (name: string) => screen.findByRole("figure", { name });
/** The Plot's visually hidden summary: "N series; x: …". */
const seriesCount = (fig: HTMLElement) => Number(/^(\d+) series/.exec(fig.querySelector(".sr-only")?.textContent ?? "")?.[1] ?? NaN);
const legendLabels = () => [...document.querySelectorAll(".plot-legend__label")].map((n) => n.textContent);
const summary = () => document.querySelector(".ui-summary")?.textContent ?? "";
const param = (name: string) => new URLSearchParams(lastLocation.split("?")[1] ?? "").get(name);
const radio = (group: string, name: string) => within(screen.getByRole("radiogroup", { name: group })).getByRole("radio", { name });

async function answer(title: RegExp | string, button: string) {
  const dlg = await screen.findByRole("alertdialog", { name: title });
  fireEvent.click(within(dlg).getByRole("button", { name: button }));
}

describe("spectra tab", () => {
  it("opens on Power: the truth and the default pick, the transfer, the models it cannot draw; only GETs", async () => {
    show();
    const power = await figure("Radial power by period, VIS");
    expect(seriesCount(power)).toBe(5);                                    // the truth + 218, 227, 233, 238 (235 is stale)
    const transfer = await figure("Transfer by period, VIS");
    expect(seriesCount(transfer)).toBe(4);
    expect(transfer.querySelector(".sr-only")?.textContent).toContain("; y: transfer √(P_out / P_truth), VIS");
    expect(legendLabels()).toEqual(["truth", "#218", "#227", "#233", "#238"]);
    expect(screen.getByRole("button", { name: "Models shown #218, #227, #233, #238" })).toBeTruthy();
    expect(screen.getByText("2 universal models have no spectra for the current test records: 230, 235.")).toBeTruthy();
    expect(screen.getByText(`No spectra for #239: ${REASON}.`)).toBeTruthy();
    expect(screen.queryByText(/model\(s\) without spectra/)).toBeNull();  // the note the callout already says
    expect(posts()).toHaveLength(0);
    expect(gets(SUMMARY)).toHaveLength(1);
    expect(calls.filter((c) => c.url.startsWith("/ensemble/universal/spectra/"))).toHaveLength(0);
  });

  it("follows the URL's section, band and models, a stale model drawn when picked", async () => {
    show("/models/spectra?s=coherence&band=H_E&m=member_238,member_235");
    const coh = await figure("Coherence by period, H");
    expect(seriesCount(coh)).toBe(2);
    expect(legendLabels()).toEqual(["#235 (stale)", "#238"]);
    // the number and its unit are held together by a no-break space
    expect(summary()).toMatch(/^In H, the coherence falls to ½ at 3\.9\d\spx for #235; #238 stays above ½ down to 2\.06 px: the finer that period/);
    expect(radio("Band", "H").getAttribute("aria-checked")).toBe("true");
    fireEvent.click(radio("Section", "Power"));
    expect(await figure("Radial power by period, H")).toBeTruthy();
    expect(lastLocation).toBe("/models/spectra?band=H_E&m=member_238,member_235");
  });

  it("adds a model from the picker", async () => {
    show();
    await figure("Radial power by period, VIS");
    fireEvent.click(screen.getByRole("button", { name: "Models shown #218, #227, #233, #238" }));
    fireEvent.click(await screen.findByText("#235 (stale)"));
    await waitFor(() => expect(param("m")).toBe("member_218,member_227,member_233,member_238,member_235"));
    await waitFor(() => expect(seriesCount(screen.getByRole("figure", { name: "Radial power by period, VIS" }))).toBe(6));
    expect(legendLabels()).toEqual(["truth", "#218", "#227", "#233", "#235 (stale)", "#238"]);
  });

  it("Direction: the grid verdict, one model's map on demand, the angular profiles by period band and the table", async () => {
    show("/models/spectra?s=direction");
    const map = await figure("Error map, #218, VIS");
    expect(seriesCount(map)).toBe(2);                                      // the 21 and 3 px rings
    expect(summary()).toBe("In VIS at 3–6 px, 2 of the 5 models measured show a pixel-axis grid (axis / diagonal ≥ 1.2): #227, #218.");
    expect(screen.getByText(/The dashed rings mark periods of 21 and 3 px \(period = 64 px \/ \|k\|\)/)).toBeTruthy();
    expect(screen.getByText(/\(3,600 windows\)/)).toBeTruthy();
    expect(screen.getByText("Click a cell to read it.")).toBeTruthy();
    expect(gets(recordUrl(218))).toHaveLength(1);
    expect(gets(recordUrl(238))).toHaveLength(0);

    const angular = await figure("Error power by direction, 3–6 px, VIS");
    expect(seriesCount(angular)).toBe(4);
    fireEvent.click(radio("Period band", "2–3"));
    expect(await figure("Error power by direction, 2–3 px, VIS")).toBeTruthy();
    expect(lastLocation).toContain("pb=2-3");
    expect(summary()).toBe("In VIS at 2–3 px, 2 of the 5 models measured show a pixel-axis grid (axis / diagonal ≥ 1.2): #227, #218.");

    const grid = screen.getByRole("grid", { name: "Axis over diagonal error power, VIS" });
    const rows = [...grid.querySelectorAll("tbody tr")];
    expect(rows.map((tr) => tr.querySelector("td")?.textContent)).toEqual(["#238", "#235stale", "#233", "#227", "#218"]);
    const r218 = rows[4];
    const marked = [...r218.querySelectorAll("[data-grid]")].map((c) => c.textContent);
    expect(marked).toEqual(["2.00", "1.50"]);
    expect(r218.querySelector("[data-grid]")?.getAttribute("title")).toBe("axis 2.000 · diagonal 1.000 (error power over round): a grid");
    expect(rows[0].querySelectorAll("[data-grid]")).toHaveLength(0);
    expect(rows[0].textContent).toContain("0.5");                         // its isotropy weight

    fireEvent.change(screen.getByRole("combobox", { name: "Error map of" }), { target: { value: "member_238" } });
    expect(await figure("Error map, #238, VIS")).toBeTruthy();
    expect(gets(recordUrl(238))).toHaveLength(1);
    expect(lastLocation).toContain("map=member_238");

    fireEvent.click(radio("Band", "H"));
    await waitFor(() => expect(summary()).toBe(
      "In H at 2–3 px, 3 of the 5 models measured show a pixel-axis grid (axis / diagonal ≥ 1.2); the 2 round ones: #235, #233."));
    expect(posts()).toHaveLength(0);
  });

  it("says when a model's map is not there yet (the server's 404)", async () => {
    routes[`GET ${recordUrl(218)}`] = () => ({ status: 404, body: { ok: false, error: "no spectra for member_218" } });
    show("/models/spectra?s=direction");
    expect(await screen.findByText("No error map for #218 yet")).toBeTruthy();
    expect(screen.getByText("no spectra for member_218")).toBeTruthy();
    expect(await figure("Error power by direction, 3–6 px, VIS")).toBeTruthy();
  });

  it("Compute spectra asks first, POSTs spectra=1 to the score job, shows it, and reads the spectra again when it ends", async () => {
    const saved = JOBS_FEED_TIMING.detailMinMs;
    JOBS_FEED_TIMING.detailMinMs = 10;
    try {
      let reads = 0;
      routes["GET /api/jobs/sp1"] = () => ({ body: job("sp1", ++reads < 3 ? "running" : "done") });
      show();
      await figure("Radial power by period, VIS");
      fireEvent.click(screen.getByRole("button", { name: "Compute spectra" }));
      const dlg = await screen.findByRole("alertdialog", { name: "Compute the spectra of 2 universal models?" });
      expect(dlg.textContent).toContain("Local job: 2 models, ~50 s of spectra. The universal score job loads 230, 235 on this machine");
      await answer("Compute the spectra of 2 universal models?", "Cancel");
      expect(posts()).toHaveLength(0);

      fireEvent.click(screen.getByRole("button", { name: "Compute spectra" }));
      await answer("Compute the spectra of 2 universal models?", "Compute");
      await waitFor(() => expect(posts()).toHaveLength(1));
      expect(posts()[0]).toMatchObject({ url: "/ensemble/universal/score", form: { spectra: "1" } });
      expect(Object.keys(posts()[0].form)).toEqual(["spectra"]);
      expect(await screen.findByText("universal: score every universal model on the test fields + spectra")).toBeTruthy();
      await waitFor(() => expect(gets(SUMMARY).length).toBeGreaterThan(1));
    } finally {
      JOBS_FEED_TIMING.detailMinMs = saved;
    }
  });

  it("follows a score job started elsewhere", async () => {
    routes["GET /api/jobs?summary=1"] = () => ({ body: [{ ...job("951d6d29", "running"), log: null }] });
    routes["GET /api/jobs/951d6d29"] = () => ({ body: job("951d6d29", "running") });
    show();
    expect(await screen.findByText("universal: score every universal model on the test fields + spectra")).toBeTruthy();
    expect(screen.getByText(/The score job is computing them: they appear here as each model finishes\./)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Compute spectra" }).getAttribute("aria-busy")).toBe("true");
    expect(posts()).toHaveLength(0);
  });

  it("says when the test records are not local, and does not offer the job", async () => {
    routes[`GET ${SUMMARY}`] = () => ({ body: payload({ records_local: false }) });
    show();
    expect(await screen.findByText("The synthetic test records are not on this machine, so no spectra can be computed here.")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Synthetic › Records" }).getAttribute("href")).toBe("/synthetic/records");
    expect((screen.getByRole("button", { name: "Compute spectra" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("explains a page with no spectra yet, with the reason when every model has one", async () => {
    const none = SPECS.map((s) => ({ ...row(s), status: "missing", radial: null, axis_diag: null, angular: null, reason: null }));
    routes[`GET ${SUMMARY}`] = () => ({ body: payload({ models: none, truth_power: null, missing: none.map((m) => m.name) }) });
    const view = show();
    expect(await screen.findByText("No spectra yet")).toBeTruthy();
    expect(screen.getByText(/The universal score job measures them on the 100 synthetic test fields, about 25 s per model/)).toBeTruthy();
    expect(screen.getByText(/7 universal models have no spectra for the current test records: 218, 227, 230, 233, 235, 238 and 1 more\./))
      .toBeTruthy();
    view.unmount();
    queryClient.clear();
    const small = SPECS.map((s) => ({ ...row(s), radial: null, axis_diag: null, angular: null, reason: REASON }));
    routes[`GET ${SUMMARY}`] = () => ({ body: payload({ models: small, truth_power: null, missing: [] }) });
    show();
    expect(await screen.findByText(`The spectra cannot be measured here: #218, #227, #230, #233, #235, #238, #239: ${REASON}.`)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Compute spectra" })).toBeNull();
  });

  it("explains a page with no universal model at all", async () => {
    routes[`GET ${SUMMARY}`] = () => ({ body: payload({ models: [], missing: [], notes: [] }) });
    show();
    expect(await screen.findByText("No universal models yet")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Universal models" }).getAttribute("href")).toBe("/models/universal");
  });

  it("draws a partial payload without breaking", async () => {
    routes[`GET ${SUMMARY}`] = () => ({ body: {
      models: [
        { name: "member_240", number: 240, status: "current", radial: { power_sr: [[1, null, "x"], null], transfer: "bad" },
          axis_diag: { "2-3": [[1.5, null], "x"], "3-6": [[1.05, 1]] }, angular: { "3-6": [[1.2, null, 0.9]] } },
        { status: "current" }, "nonsense",
      ],
      notes: ["the test records changed"],
    } });
    show();
    const power = await figure("Radial power by period, VIS");
    expect(seriesCount(power)).toBe(1);
    expect(screen.getByText("The test records changed.")).toBeTruthy();
    fireEvent.click(radio("Section", "Direction"));
    expect(await figure("Error power by direction, 3–6 px, VIS")).toBeTruthy();
    expect(summary()).toBe("In VIS at 3–6 px, no model shows a pixel-axis grid: the one model measured stays under 1.2 axis / diagonal.");
  });

  it("says when the server has no spectra endpoint", async () => {
    routes[`GET ${SUMMARY}`] = () => ({ status: 404, body: { ok: false, error: "not found" } });
    show();
    expect(await screen.findByText("Nothing here yet")).toBeTruthy();
  });
});
