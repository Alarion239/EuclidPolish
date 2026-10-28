/* Models › Knees against a mocked Flask (the knee-attribution API of spec
 * 2026-10-15, fixtures in ./testing/kneesFixture): opening the tab only
 * GETs (no POST, and no catalogue read while the run panel is closed); the
 * main chart draws one line per knee in each model's panel with the knee
 * legend; the class bars list the classes in order with their n, and the
 * split toggle changes the values; the local (3×3) split swaps the chart,
 * the bars, the cross-band tables and the explorer's dots (a version-2
 * record → a note and the receptive field's); the input band switches the
 * chart and the bars to that band's own knee split (disabled under the
 * ablation split; an older record → the per-electron warning and a note);
 * the cross-band tables; the field explorer draws a dot per sample of the
 * band on the record's image, with the field's noise σ; the run panel
 * preselects the newest universal models, asks with the estimate and POSTs
 * the right form; no runs → the explanation and an open run panel; a failed
 * run → its errors; a cancelled run → what it kept. The pixel types view
 * (?view=pixel-types) shows the newest analysis' figure and band facts,
 * follows its figure chips, band and grouping, the comparison's figures, and
 * starts an analysis or a comparison only after a confirm (different seeds
 * refused); the run panel's mixed sampling posts its samples per band and
 * turns the ablation off. */
import { QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobsStore } from "../../api/jobs";
import { queryClient } from "../../api/query";
import { resetConfirm } from "../../ui";
import { kneeLabel, sharePct } from "./knees";
import Knees from "./tabs/Knees";
import {
  KNEES_8, NEXUS_TILES, OLD_RUN_ID, POSTER_TILES, RUN_ID, SOURCES, byInput, catalogue, failedRecord, localSplit, peaked, record, runs,
  v1Record, v2Record,
} from "./testing/kneesFixture";

type Reply = { status?: number; body: unknown };
let routes: Record<string, (form: Record<string, string>) => Reply>;
let calls: { url: string; method: string; form: Record<string, string> }[];
const posts = () => calls.filter((c) => c.method === "POST");
const gets = (u: string) => calls.filter((c) => c.method === "GET" && c.url === u);

const formOf = (body: BodyInit | null | undefined): Record<string, string> => {
  const out: Record<string, string> = {};
  if (body instanceof FormData) body.forEach((v, k) => { out[k] = String(v); });
  else if (body instanceof URLSearchParams) body.forEach((v, k) => { out[k] = v; });
  else if (typeof body === "string") new URLSearchParams(body).forEach((v, k) => { out[k] = v; });
  return out;
};

beforeEach(() => {
  calls = [];
  routes = {
    "GET /api/knee-attribution": () => ({ body: runs() }),
    [`GET /api/knee-attribution/${RUN_ID}`]: () => ({ body: record() }),
    [`GET /api/knee-attribution/${OLD_RUN_ID}`]: () => ({ body: failedRecord() }),
    "GET /api/knee-attribution/models": () => ({ body: catalogue() }),
    "GET /api/real/sources": () => ({ body: SOURCES }),
    "GET /api/real/poster": () => ({ body: POSTER_TILES }),
    "GET /api/real/nexus": () => ({ body: NEXUS_TILES }),
    "GET /api/jobs?summary=1": () => ({ body: { jobs: [] } }),
    "POST /api/knee-attribution": () => ({ body: { ok: true, job_id: "ka1", run_id: "20261015-120000-ffffff" } }),
    "GET /api/jobs/ka1": () => ({ body: { job_id: "ka1", status: "running", label: "Knee attribution", progress: { current: 1, total: 12, pct: 8.3, label: "universal:member_223: test/0" }, log: "", log_truncated: false, duration: 3, error: null } }),
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

const show = (url = "/models/knees") => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={[url]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes><Route path="/models/*" element={<><Knees /><Spy /></>} /></Routes>
    </MemoryRouter>
  </QueryClientProvider>,
);

const panels = () => [...document.querySelectorAll<HTMLElement>(".kn-chart")];
/** The Plot's visually hidden summary: "N series; x: …". */
const seriesCount = (fig: HTMLElement) => Number(/^(\d+) series/.exec(fig.querySelector(".plot .sr-only")?.textContent ?? "")?.[1] ?? NaN);
const legendLabels = () => [...document.querySelectorAll(".plot-legend__label")].map((n) => n.textContent);
/** A galaxy-core class bar's aria-label for the given shares (#222's knees). */
const coreLabel = (shares: number[]) => `galaxy core: ${KNEES_8.map((q, k) => `${kneeLabel(q)} ${sharePct(shares[k])}`).join(", ")}`;
const splitRadio = (name: string) => within(screen.getByRole("radiogroup", { name: "Split" })).getByRole("radio", { name });

async function answer(title: RegExp | string, button: string) {
  const dlg = await screen.findByRole("alertdialog", { name: title });
  fireEvent.click(within(dlg).getByRole("button", { name: button }));
}

describe("knees tab", () => {
  it("draws the newest run: one line per knee in each model's panel, the knee legend, and only GETs", async () => {
    show();
    await waitFor(() => expect(panels()).toHaveLength(2));
    const [p222, p221] = panels();
    expect(p222.dataset.model).toBe("universal:member_222");
    expect(seriesCount(p222)).toBe(8);
    expect(seriesCount(p221)).toBe(6);
    expect(within(p222).getByText(/#222 · depth 16 · 8 knees · n = /)).toBeTruthy();
    expect(legendLabels()).toEqual(["10⁻³ e⁻", "10⁻² e⁻", "10⁻¹ e⁻", "1 e⁻", "10 e⁻", "10² e⁻", "10³ e⁻", "10⁴ e⁻"]);
    expect(screen.getByRole("figure", { name: "Knee share vs pixel brightness, #222, VIS, synthetic" })).toBeTruthy();
    // the one-line answer: the faint bins lean on a faint knee, the bright on the brightest
    const line = document.querySelector(".ui-summary")?.textContent ?? "";
    expect(line).toMatch(/^For #222 in VIS \(synthetic fields\), the faintest pixels .* lean on the 10⁻² e⁻ knee .* the brightest .* on the 10⁴ e⁻ knee/);
    expect(posts()).toHaveLength(0);
    expect(gets("/api/knee-attribution")).toHaveLength(1);
    expect(gets(`/api/knee-attribution/${RUN_ID}`)).toHaveLength(1);
    expect(gets("/api/knee-attribution/models")).toHaveLength(0);       // the run panel is closed
    expect(screen.getByRole("button", { name: "New run" }).getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByText(/Per-electron weights/)).toBeNull();          // a version-2 record
  });

  it("overlays only the models picked in the URL, and follows the band", async () => {
    show("/models/knees?km=universal:member_221&band=H_E");
    await waitFor(() => expect(panels()).toHaveLength(1));
    expect(panels()[0].dataset.model).toBe("universal:member_221");
    expect(seriesCount(panels()[0])).toBe(6);
    expect(legendLabels()).toEqual(["10⁻¹ e⁻", "1 e⁻", "10 e⁻", "10² e⁻", "10³ e⁻", "10⁴ e⁻"]);
    expect(screen.getByRole("figure", { name: "Knee share vs pixel brightness, #221, H, synthetic" })).toBeTruthy();
  });

  it("lists the classes in order with their n, synthetic and real apart; the ablation toggle changes the bars", async () => {
    show("/models/knees?km=universal:member_222");
    const syn = await screen.findByRole("region", { name: "Synthetic fields · truth classes" });
    const real = screen.getByRole("region", { name: "Real fields · brightness classes" });
    const classes = (root: HTMLElement) => [...root.querySelectorAll<HTMLElement>(".kn-class")].map((c) => c.dataset.class);
    expect(classes(syn)).toEqual(["star core", "star wing", "galaxy core", "galaxy outskirts", "faint", "sky"]);   // no saturated in VIS
    expect(classes(real)).toEqual(["sky", "faint", "mid", "bright", "core", "blank"]);
    const core = syn.querySelector<HTMLElement>(".kn-class[data-class='galaxy core']")!;
    expect(core.querySelector(".kn-barrow__n")?.textContent).toBe("30");
    const bar = () => within(core).getByRole("img").getAttribute("aria-label");
    const gradient = bar();
    expect(gradient).toMatch(/^galaxy core: 10⁻³ e⁻ \d+%, .*10² e⁻ 4\d%/);
    expect(core.querySelectorAll(".kn-stack__seg").length).toBeGreaterThan(0);
    fireEvent.click(within(screen.getByRole("radiogroup", { name: "Split" })).getByRole("radio", { name: "Ablation" }));
    await waitFor(() => expect(bar()).not.toBe(gradient));
    expect(bar()).toMatch(/10⁻¹ e⁻ 4\d%/);                                   // the reversed split leads with the mirrored knee
    expect(lastLocation).toContain("split=ablation");
    fireEvent.click(splitRadio("Gradient (receptive field)"));
    await waitFor(() => expect(bar()).toBe(gradient));
  });

  it("the input band draws that band's own knee split in the chart and the class bars; the agreement stays all-input", async () => {
    show("/models/knees?km=universal:member_222");
    const syn = await screen.findByRole("region", { name: "Synthetic fields · truth classes" });
    const core = syn.querySelector<HTMLElement>(".kn-class[data-class='galaxy core']")!;
    const bar = () => within(core).getByRole("img").getAttribute("aria-label");
    const agree = () => core.querySelector(".kn-barrow__agree")?.textContent;
    const summary = () => document.querySelector(".ui-summary")?.textContent ?? "";
    const label = coreLabel;
    const g = peaked(KNEES_8, 300);
    expect(bar()).toBe(label(g));
    const agreeAll = agree();
    const summaryAll = summary();
    expect(document.body.textContent).not.toContain("≈ 0.06σ in VIS");
    const picker = screen.getByRole("radiogroup", { name: "Input band" });
    expect(within(picker).getAllByRole("radio").map((r) => r.textContent)).toEqual(["All (σ-weighted)", "VIS", "Y", "J", "H"]);

    fireEvent.click(within(picker).getByRole("radio", { name: "VIS" }));
    await waitFor(() => expect(bar()).toBe(label(byInput(g)[0])));
    expect(bar()).not.toBe(label(g));
    expect(agree()).toBe(agreeAll);
    expect(screen.getByRole("figure", { name: "Knee share vs pixel brightness, #222, VIS, synthetic, VIS input" })).toBeTruthy();
    expect(summary()).toMatch(/^For #222 in VIS \(synthetic fields, VIS input\), the faintest pixels/);
    expect(summary()).not.toBe(summaryAll);
    expect(document.body.textContent).toContain("1 e⁻ ≈ 1σ in NISP but ≈ 0.06σ in VIS");
    expect(lastLocation).toContain("input=VIS");

    fireEvent.click(within(picker).getByRole("radio", { name: "Y" }));
    await waitFor(() => expect(bar()).toBe(label(byInput(g)[1])));
    expect(screen.getByRole("figure", { name: "Knee share vs pixel brightness, #222, VIS, synthetic, Y input" })).toBeTruthy();

    fireEvent.click(within(picker).getByRole("radio", { name: "All (σ-weighted)" }));
    await waitFor(() => expect(bar()).toBe(label(g)));
    expect(summary()).toBe(summaryAll);
    expect(lastLocation).not.toContain("input=");
  });

  it("the local split swaps every gradient view (chart, bars, cross-band, explorer dots); the agreement stays the receptive field's", async () => {
    show("/models/knees?km=universal:member_222");
    const syn = await screen.findByRole("region", { name: "Synthetic fields · truth classes" });
    const core = syn.querySelector<HTMLElement>(".kn-class[data-class='galaxy core']")!;
    const bar = () => within(core).getByRole("img").getAttribute("aria-label");
    const agree = () => core.querySelector(".kn-barrow__agree")?.textContent;
    const summary = () => document.querySelector(".ui-summary")?.textContent ?? "";
    const dots = () => [...screen.getByRole("group", { name: "4 sampled pixels in VIS" }).querySelectorAll<SVGCircleElement>("circle")]
      .map((d) => d.dataset.knee);
    const crossRow = (name: string) => [...screen.getByRole("table", { name }).querySelectorAll("tbody tr")][0];
    const cells = (tr: Element) => [...tr.querySelectorAll("th, td")].map((c) => c.textContent);
    expect(within(screen.getByRole("radiogroup", { name: "Split" })).getAllByRole("radio").map((r) => r.textContent))
      .toEqual(["Gradient (receptive field)", "Gradient (local 3×3)", "Ablation"]);
    await waitFor(() => expect(dots()).toEqual(["5", "1", "6", "2"]));
    const agreeRf = agree();
    expect(cells(crossRow("Cross-band shares, #222"))).toEqual(["VIS", "82%", "6%", "6%", "6%"]);
    expect(document.body.textContent).toContain("Summed over the whole receptive field, mostly the surrounding sky");

    fireEvent.click(splitRadio("Gradient (local 3×3)"));
    await waitFor(() => expect(bar()).toBe(coreLabel(localSplit(KNEES_8, 300))));
    expect(agree()).toBe(agreeRf);
    expect(lastLocation).toContain("split=local");
    expect(screen.getByRole("figure", { name: "Knee share vs pixel brightness, #222, VIS, synthetic, local 3×3" })).toBeTruthy();
    expect(summary()).toMatch(/^For #222 in VIS \(synthetic fields, local 3×3\), the faintest pixels .* lean on the 10⁻¹ e⁻ knee/);
    expect(cells(crossRow("Cross-band shares, #222, local 3×3"))).toEqual(["VIS", "93%", "3%", "2%", "2%"]);
    // galaxy core → 10³ e⁻, sky → 10⁻¹, star core → 10⁴, faint → 1 e⁻ (the fixture's local split leans on the knee nearest 10·b)
    expect(dots()).toEqual(["6", "2", "7", "3"]);
    expect(document.body.textContent).toContain("Summed only over the 3×3 LR pixels under the sample");

    // the input band stays enabled and draws the local per-input rows
    const picker = screen.getByRole("radiogroup", { name: "Input band" });
    expect((within(picker).getAllByRole("radio") as HTMLButtonElement[]).some((r) => r.disabled)).toBe(false);
    fireEvent.click(within(picker).getByRole("radio", { name: "Y" }));
    await waitFor(() => expect(bar()).toBe(coreLabel(byInput(localSplit(KNEES_8, 300))[1])));
    expect(screen.getByRole("figure", { name: "Knee share vs pixel brightness, #222, VIS, synthetic, local 3×3, Y input" })).toBeTruthy();
    expect(agree()).toBe(agreeRf);
  });

  it("notes a version-2 record predates the local split and draws the receptive field's", async () => {
    routes[`GET /api/knee-attribution/${RUN_ID}`] = () => ({ body: v2Record() });
    show("/models/knees?km=universal:member_222&split=local");
    expect(await screen.findByText(/^This run predates the local split; re-run to see it\./)).toBeTruthy();
    expect(lastLocation).toContain("split=local");
    expect(splitRadio("Gradient (receptive field)").getAttribute("aria-checked")).toBe("true");
    expect(screen.getByRole("figure", { name: "Knee share vs pixel brightness, #222, VIS, synthetic" })).toBeTruthy();
    expect(screen.getByRole("table", { name: "Cross-band shares, #222" })).toBeTruthy();
    const core = screen.getByRole("region", { name: "Synthetic fields · truth classes" }).querySelector<HTMLElement>(".kn-class[data-class='galaxy core']")!;
    expect(within(core).getByRole("img").getAttribute("aria-label")).toBe(coreLabel(peaked(KNEES_8, 300)));
    const dots = [...screen.getByRole("group", { name: "4 sampled pixels in VIS" }).querySelectorAll<SVGCircleElement>("circle")];
    expect(dots.map((d) => d.dataset.knee)).toEqual(["5", "1", "6", "2"]);
    expect(screen.queryByText(/Per-electron weights/)).toBeNull();          // version 2 is noise-weighted
    expect(screen.queryByText(/predates the per-input split/)).toBeNull();
    // the per-input split of version 2 still works
    fireEvent.click(within(screen.getByRole("radiogroup", { name: "Input band" })).getByRole("radio", { name: "Y" }));
    await waitFor(() => expect(within(core).getByRole("img").getAttribute("aria-label")).toBe(coreLabel(byInput(peaked(KNEES_8, 300))[1])));
  });

  it("disables the input band under the ablation split (it ablates every input band at once)", async () => {
    show("/models/knees?km=universal:member_222&input=Y&split=ablation");
    const picker = await screen.findByRole("radiogroup", { name: "Input band" });
    const radios = () => within(picker).getAllByRole("radio") as HTMLButtonElement[];
    expect(radios().every((r) => r.disabled)).toBe(true);
    expect(radios()[2].title).toBe("The ablation removes a knee in every input band at once");
    expect(screen.getByText("ablation: every input band at once")).toBeTruthy();
    expect(screen.getByRole("figure", { name: "Knee share vs pixel brightness, #222, VIS, synthetic" })).toBeTruthy();   // all inputs
    // back to the gradient: the URL's Y input returns
    fireEvent.click(splitRadio("Gradient (receptive field)"));
    expect(await screen.findByRole("figure", { name: "Knee share vs pixel brightness, #222, VIS, synthetic, Y input" })).toBeTruthy();
    expect(radios().some((r) => r.disabled)).toBe(false);
    expect(screen.queryByText("ablation: every input band at once")).toBeNull();
  });

  it("warns on a per-electron (version 1) record and notes it predates the per-input split, drawing all inputs", async () => {
    routes[`GET /api/knee-attribution/${RUN_ID}`] = () => ({ body: v1Record() });
    show("/models/knees?km=universal:member_222");
    expect(await screen.findByText(/^Per-electron weights \(record v1\): band and knee shares lean toward NISP; re-run for noise-weighted shares\.$/))
      .toBeTruthy();
    expect(screen.queryByText(/predates the per-input split/)).toBeNull();
    fireEvent.click(within(screen.getByRole("radiogroup", { name: "Input band" })).getByRole("radio", { name: "Y" }));
    expect(await screen.findByText(/This run predates the per-input split; re-run to see it/)).toBeTruthy();
    expect(lastLocation).toContain("input=Y");
    expect(screen.getByRole("figure", { name: "Knee share vs pixel brightness, #222, VIS, synthetic" })).toBeTruthy();
    expect(within(screen.getByRole("radiogroup", { name: "Input band" })).getByRole("radio", { name: "All (σ-weighted)" }).getAttribute("aria-checked"))
      .toBe("true");
    expect(document.querySelector(".kn-noise")).toBeNull();                  // no noise σ before version 2
  });

  it("shows a bar per model when several are overlaid, and the cross-band table per model", async () => {
    show();
    const syn = await screen.findByRole("region", { name: "Synthetic fields · truth classes" });
    const core = syn.querySelector<HTMLElement>(".kn-class[data-class='galaxy core']")!;
    expect(within(core).getAllByRole("img").map((b) => b.getAttribute("aria-label")?.split(":")[0])).toEqual(["galaxy core, #222", "galaxy core, #221"]);
    const t = screen.getByRole("table", { name: "Cross-band shares, #222" });
    const rows = [...t.querySelectorAll("tbody tr")].map((tr) => [...tr.querySelectorAll("th, td")].map((c) => c.textContent));
    expect(rows[0]).toEqual(["VIS", "82%", "6%", "6%", "6%"]);
    expect(rows[3]).toEqual(["H", "20%", "10%", "10%", "60%"]);
    expect(screen.getByRole("table", { name: "Cross-band shares, #221" })).toBeTruthy();
  });

  it("draws the field explorer's sampled pixels as dots in their dominant knee's colour on the record's image", async () => {
    show("/models/knees?km=universal:member_222");
    const svg = await screen.findByRole("group", { name: "4 sampled pixels in VIS" });
    const img = screen.getByRole("img", { name: "test/0, SR VIS" });
    expect(screen.getByText("noise σ: VIS 17.6 · Y 0.87 · J 0.98 · H 0.90 e⁻")).toBeTruthy();
    expect(img.getAttribute("src")).toBe(`/api/knee-attribution/${RUN_ID}/field/universal__member_222/test__0.png?band=VIS&layer=sr`);
    const dots = [...svg.querySelectorAll<SVGCircleElement>("circle")];
    expect(dots).toHaveLength(4);
    // galaxy core (300 e⁻) → 10² e⁻ (index 5), sky (0.02 e⁻) → 10⁻² e⁻, star core → 10³ e⁻, faint (0.3 e⁻) → 10⁻¹ e⁻
    expect(dots.map((d) => d.dataset.knee)).toEqual(["5", "1", "6", "2"]);
    expect(dots[0].querySelector("title")?.textContent).toMatch(/^galaxy core · 300 e⁻ · \(x 12, y 10\)/);
    expect(dots[0].getAttribute("cx")).toBe("12.5");
    fireEvent.mouseEnter(dots[0]);
    expect(await screen.findByText(/Leads: 10² e⁻/)).toBeTruthy();
    // the Y band: its two samples; the LR layer keeps the HR coordinate system
    fireEvent.click(within(screen.getByRole("radiogroup", { name: "Band" })).getByRole("radio", { name: "Y" }));
    expect(await screen.findByRole("group", { name: "2 sampled pixels in Y" })).toBeTruthy();
    fireEvent.click(within(screen.getByRole("radiogroup", { name: "Image layer" })).getByRole("radio", { name: "LR" }));
    await waitFor(() => expect(screen.getByRole("img", { name: "test/0, LR Y" }).getAttribute("src")).toMatch(/band=Y_E&layer=lr$/));
    expect(screen.getByRole("group", { name: "2 sampled pixels in Y" }).getAttribute("viewBox")).toBe("0 0 128 128");
  });

  it("builds the image URL from the slugs when the record has none, and switches to the real fields", async () => {
    show("/models/knees?km=universal:member_221&set=real");
    const img = await screen.findByRole("img", { name: "poster/target_181255, SR VIS" });
    expect(img.getAttribute("src")).toBe(`/api/knee-attribution/${RUN_ID}/field/universal__member_221/poster__target_181255.png?band=VIS&layer=sr`);
    expect(screen.getByRole("figure", { name: "Knee share vs pixel brightness, #221, VIS, real" })).toBeTruthy();
  });

  it("run panel: opening it only GETs; Run asks with the estimate and POSTs the picked models and fields", async () => {
    show();
    await waitFor(() => expect(panels()).toHaveLength(2));
    fireEvent.click(screen.getByRole("button", { name: "New run" }));
    const models = await screen.findByRole("group", { name: "Models to run" });
    const box = (name: RegExp) => within(models).getByRole("checkbox", { name }) as HTMLInputElement;
    await waitFor(() => expect(box(/#223/).checked).toBe(true));
    expect(box(/#222/).checked).toBe(true);
    expect(box(/#221/).checked).toBe(true);
    expect(box(/#211/).checked).toBe(false);
    expect(box(/#224/).disabled).toBe(true);
    expect(within(models).getByText("no checkpoint: pull it from FASRC")).toBeTruthy();
    expect(box(/#178/).disabled).toBe(true);
    fireEvent.click(box(/#221/));
    fireEvent.change(screen.getByRole("spinbutton", { name: "Synthetic test fields" }), { target: { value: "5" } });
    const tiles = await screen.findByRole("group", { name: "Real tiles" });
    fireEvent.change(screen.getByRole("textbox", { name: "Filter real tiles" }), { target: { value: "poster" } });
    await waitFor(() => expect(within(tiles).getAllByRole("checkbox")).toHaveLength(1));
    fireEvent.click(within(tiles).getByRole("checkbox", { name: /poster\/target_181255/ }));
    expect(screen.getByText("local job: 2 models × 6 fields, ~4 min")).toBeTruthy();
    expect(posts()).toHaveLength(0);
    expect(gets("/api/knee-attribution/models")).toHaveLength(1);
    expect(gets("/api/real/tile")).toHaveLength(0);                       // no cached tiles: never asked

    fireEvent.click(screen.getByRole("button", { name: "Run…" }));
    const dlg = await screen.findByRole("alertdialog", { name: "Run knee attribution?" });
    expect(dlg.textContent).toContain("Local job: 2 models × 6 fields, ~4 min.");
    await answer("Run knee attribution?", "Cancel");
    expect(posts()).toHaveLength(0);

    fireEvent.click(screen.getByRole("button", { name: "Run…" }));
    await answer("Run knee attribution?", "Run");
    await waitFor(() => expect(posts()).toHaveLength(1));
    expect(posts()[0]).toMatchObject({ url: "/api/knee-attribution", form: {
      models: "universal:member_223,universal:member_222", synthetic: "5", tiles: "poster/target_181255", per_class: "24", seed: "0",
    } });
    // the panel folds away and the job's progress shows; the runs list is read again
    await waitFor(() => expect(screen.getByRole("button", { name: "New run" }).getAttribute("aria-expanded")).toBe("false"));
    await waitFor(() => expect(gets("/api/knee-attribution").length).toBeGreaterThan(1));
    await waitFor(() => expect(document.querySelector(".job-panel__label")?.textContent).toBe("Knee attribution"));
    expect(lastLocation).toBe("/models/knees");                        // the newest run (the new one) is shown
  });

  it("refuses to run with nothing to attribute", async () => {
    routes["GET /api/real/sources"] = () => ({ body: { sources: [] } });
    show();
    fireEvent.click(await screen.findByRole("button", { name: "New run" }));
    await screen.findByRole("group", { name: "Models to run" });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Synthetic test fields" }), { target: { value: "0" } });
    expect(await screen.findByText("Pick synthetic test fields or real tiles")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Run…" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByRole("spinbutton", { name: "Synthetic test fields" }), { target: { value: "101" } });
    expect(await screen.findByText("A whole number from 0 to 100")).toBeTruthy();
    expect(posts()).toHaveLength(0);
  });

  it("explains an empty history and opens the run panel, still without a POST", async () => {
    routes["GET /api/knee-attribution"] = () => ({ body: { runs: [] } });
    show();
    expect(await screen.findByText("No knee attribution runs yet")).toBeTruthy();
    expect(screen.getByRole("button", { name: "New run" }).getAttribute("aria-expanded")).toBe("true");
    expect(await screen.findByRole("group", { name: "Models to run" })).toBeTruthy();
    expect(posts()).toHaveLength(0);
  });

  it("shows a failed run's errors", async () => {
    show(`/models/knees?run=${OLD_RUN_ID}`);
    const alert = await screen.findByText("This run failed");
    const callout = alert.closest(".ui-callout") as HTMLElement;
    expect(within(callout).getByText("#211")).toBeTruthy();
    expect(callout.textContent).toContain("OSError: no checkpoint for member_211");
    expect(panels()).toHaveLength(0);
    expect(gets(`/api/knee-attribution/${OLD_RUN_ID}`)).toHaveLength(1);
  });

  it("says what a cancelled run kept", async () => {
    routes["GET /api/knee-attribution"] = () => ({ body: { runs: [{ ...runs().runs[0], status: "cancelled" }] } });
    const rec = record({ status: "cancelled" });
    delete rec.models["universal:member_221"];
    routes[`GET /api/knee-attribution/${RUN_ID}`] = () => ({ body: rec });
    show();
    expect(await screen.findByText("This run was cancelled: it kept the 1 of 2 models it finished.")).toBeTruthy();
    await waitFor(() => expect(panels()).toHaveLength(1));
    expect(screen.queryByText("No model results in this run")).toBeNull();
  });

  it("names partial errors of a finished run above its figures", async () => {
    routes[`GET /api/knee-attribution/${RUN_ID}`] = () => ({ body: record({ errors: { "universal:member_222|test/9": "ValueError: empty field" } }) });
    show();
    expect(await screen.findByText("1 part of this run failed")).toBeTruthy();
    expect(screen.getByText("#222 › test/9")).toBeTruthy();
    await waitFor(() => expect(panels()).toHaveLength(2));
  });
});

/* ── pixel types ───────────────────────────────────────────────────────── */

const ANALYSED = "2026-10-25T07:40:00+00:00";
const FOLDER = `${RUN_ID}/universal__member_222`;
const COMPARE = "compare_member_221_vs_member_222";
const pixelReports = () => ({
  analyses: [{
    folder: FOLDER, run_id: RUN_ID, model: "universal:member_222", created: ANALYSED, kinds: ["synthetic", "real"],
    params: { synthetic: 2, tiles: ["poster/target_181255"], sampling: "mixed", per_band: 1024, seed: 0 },
    files: ["knee_map_synthetic.png", "knee_map_real.png", "groups_learned_synthetic_VIS.png", "groups_synthetic_VIS.png",
      "groups_learned_synthetic_H_E.png", "types_synthetic_VIS.png", "look_around_synthetic.png", "summary.json"],
  }],
  comparisons: [{
    folder: COMPARE, models: { member_221: "universal:member_221", member_222: "universal:member_222" },
    runs: { member_221: RUN_ID, member_222: RUN_ID }, created: ANALYSED, tag: "", kinds: ["synthetic"],
    files: ["compare_knee_synthetic.png", "compare_share_diff_synthetic.png", "compare_pairs_synthetic_VIS.png", "summary.json"],
  }],
});
const fileUrl = (folder: string, name: string) =>
  `/api/knee-attribution/pixel-types/file/${folder.split("/").map(encodeURIComponent).join("/")}/${name}?v=${encodeURIComponent(ANALYSED)}`;
const figureSrcs = () => [...document.querySelectorAll<HTMLImageElement>(".kn-pt__figure img")].map((i) => i.getAttribute("src"));
const chip = (group: string, name: string) => within(screen.getByRole("radiogroup", { name: group })).getByRole("radio", { name });

describe("knees tab › pixel types", () => {
  beforeEach(() => {
    routes["GET /api/knee-attribution/pixel-types"] = () => ({ body: pixelReports() });
    routes[`GET ${fileUrl(FOLDER, "summary.json")}`] = () => ({
      body: { synthetic: { VIS: { n: 36921, noise_sigma_e: 20.31, blank_fraction: 0, types: [{}, {}, {}, {}] } } },
    });
    routes["POST /api/knee-attribution/pixel-types/analyse"] = () => ({ body: { ok: true, job_id: "pt1", run_id: RUN_ID } });
    routes["POST /api/knee-attribution/pixel-types/compare"] = () => ({ body: { ok: true, job_id: "pt2" } });
    routes["GET /api/jobs/pt1"] = () => ({ body: { job_id: "pt1", status: "running", label: "Pixel types", progress: null, log: "", log_truncated: false, duration: 1, error: null } });
    routes["GET /api/jobs/pt2"] = () => ({ body: { job_id: "pt2", status: "running", label: "Pixel types: compare", progress: null, log: "", log_truncated: false, duration: 1, error: null } });
  });

  it("shows the newest analysis' groups and band facts, follows the chips, and only GETs", async () => {
    show("/models/knees?view=pixel-types");
    await waitFor(() => expect(figureSrcs()).toContain(fileUrl(FOLDER, "groups_learned_synthetic_VIS.png")));
    expect(figureSrcs()).toContain(fileUrl(COMPARE, "compare_knee_synthetic.png"));
    expect(await screen.findByText(/36,921/)).toBeTruthy();
    expect(document.querySelector(".ui-summary")?.textContent).toMatch(/#222, synthetic fields, VIS: 36,921 sampled pixels, noise σ = 20\.3 e⁻ per pixel; 4 knee types\./);
    // the grouping: by the knee itself
    fireEvent.click(chip("Group by", "Knee"));
    await waitFor(() => expect(figureSrcs()).toContain(fileUrl(FOLDER, "groups_synthetic_VIS.png")));
    expect(lastLocation).toContain("pg=raw");
    // a band without the figure is disabled; H has it
    expect((chip("Pixel types band", "Y") as HTMLButtonElement).disabled).toBe(true);
    // a whole-field figure: the band toggle is off
    fireEvent.click(chip("Pixel types figure", "Knee vs brightness"));
    await waitFor(() => expect(figureSrcs()).toContain(fileUrl(FOLDER, "knee_map_synthetic.png")));
    expect((chip("Pixel types figure", "Chain rule") as HTMLButtonElement).disabled).toBe(true);    // not drawn
    expect(posts()).toHaveLength(0);
    expect(gets("/api/knee-attribution/pixel-types")).toHaveLength(1);
    expect(gets(`/api/knee-attribution/${RUN_ID}`)).toHaveLength(0);    // no run record in this view
  });

  it("Analyse… asks with the estimate and POSTs the run", async () => {
    show("/models/knees?view=pixel-types");
    fireEvent.click(await screen.findByRole("button", { name: "Re-analyse…" }));      // #222 of the run is analysed
    const dlg = await screen.findByRole("alertdialog", { name: "Analyse the pixel types?" });
    expect(dlg.textContent).toContain("Local job: 2 models, ~4 min.");
    await answer("Analyse the pixel types?", "Analyse");
    await waitFor(() => expect(posts()).toHaveLength(1));
    expect(posts()[0]).toMatchObject({ url: "/api/knee-attribution/pixel-types/analyse", form: { run_id: RUN_ID } });
  });

  it("Compare… POSTs the two models; runs of different seeds are refused", async () => {
    routes["GET /api/knee-attribution"] = () => ({ body: { runs: [
      { ...runs().runs[0], seed: 0, sampling: "mixed", version: 5 },
      { id: "20261020-080000-aaaaaa", created: "2026-10-20T08:00:00Z", status: "done", models: ["universal:member_238"], synthetic: 30,
        tiles: [], per_class: 48, seed: 7, sampling: "mixed", version: 5 },
    ] } });
    show("/models/knees?view=pixel-types");
    const button = await screen.findByRole("button", { name: "Compare…" });
    fireEvent.change(screen.getByRole("combobox", { name: "Model B" }), { target: { value: "20261020-080000-aaaaaa|universal:member_238" } });
    expect((await screen.findByRole("alert")).textContent).toMatch(/different seeds \(0 and 7\)/);
    expect((button as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByRole("combobox", { name: "Model B" }), { target: { value: `${RUN_ID}|universal:member_222` } });
    await waitFor(() => expect((screen.getByRole("button", { name: "Compare…" }) as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: "Compare…" }));
    await answer("Compare #222 against #221?", "Compare");
    await waitFor(() => expect(posts()).toHaveLength(1));
    expect(posts()[0]).toMatchObject({ url: "/api/knee-attribution/pixel-types/compare",
      form: { a: `${RUN_ID}|universal:member_221`, b: `${RUN_ID}|universal:member_222` } });
  });

  it("explains an empty pixel-types folder and still offers to analyse", async () => {
    routes["GET /api/knee-attribution/pixel-types"] = () => ({ body: { analyses: [], comparisons: [] } });
    show("/models/knees?view=pixel-types");
    expect(await screen.findByText("No pixel-types analysis yet")).toBeTruthy();
    expect(screen.getByText("No comparison yet")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Analyse…" })).toBeTruthy();
    expect(posts()).toHaveLength(0);
  });

  it("run panel: mixed sampling posts its samples per band and turns the ablation off", async () => {
    show();
    await waitFor(() => expect(panels()).toHaveLength(2));
    fireEvent.click(screen.getByRole("button", { name: "New run" }));
    await screen.findByRole("group", { name: "Models to run" });
    fireEvent.click(chip("Sampling", "Mixed"));
    expect((screen.getByRole("checkbox", { name: "Ablation check" }) as HTMLInputElement).checked).toBe(false);
    expect(screen.getByRole("spinbutton", { name: "Samples per band" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Run…" }));
    await answer("Run knee attribution?", "Run");
    await waitFor(() => expect(posts()).toHaveLength(1));
    expect(posts()[0].form).toMatchObject({ sampling: "mixed", per_band: "1024", ablation: "0", per_class: "24" });
  });
});

/* ── pixel types, drawn natively ───────────────────────────────────────── */

const stamp = () => btoa(String.fromCharCode(...new Array(41 * 41).fill(7)));
const nativeView = () => ({
  kind: "synthetic", model: "universal:member_222", knees: [0.1, 10, 1000], edges_sigma: [0.001, 0.01, 0.1, 1, 10],
  centres_log: [-2.5, -1.5, -0.5, 0.5], bin_labels: ["< 0.01σ", "0.01–0.1σ", "0.1–1σ", "1–10σ"], radii: [0, 2, 4],
  type_hist_x: [-3, 1, 5], features: [{ key: "nearest_source_px", label: "distance to nearest >1σ pixel (px)", log: false }],
  groups: ["lower", "usual", "higher"],
  bands: { VIS: {
    sigma: 20.3, blank_fraction: 0, n: 3908, logk: [-2.3, -0.3, 1.7],
    heat: { x_edges: [-3, 1, 5], y_edges: [-2.5, 0, 2], z: [[3, 0], [1, 5]] },
    curve: [{ q20: -2.2, median: -2.1, q80: -2, mech: -2.4, n: 300 }, null, { q20: -1, median: -0.8, q80: -0.6, mech: -1.5, n: 50 }, null],
    features: { raw: {}, learned: { lower: { nearest_source_px: [29, 20, 11, 0] }, usual: { nearest_source_px: [29, 19, 6, 0] }, higher: { nearest_source_px: [29, 10, 2, 0] } } },
    types: [{ index: 0, prevalence: 0.8, n: 900, mean_share: [0.6, 0.3, 0.1], log_q: -2, bright_median: 0, dist: 29, classes: { sky: 0.9 }, hist: [1, 0] },
      { index: 1, prevalence: 0.2, n: 300, mean_share: [0.1, 0.5, 0.4], log_q: 0, bright_median: 30, dist: 0, classes: { "galaxy core": 0.7 }, hist: [0, 1] }],
    type_mix: [[1, 0], null, [0.5, 0.5], [0, 1]],
    look: { raw: { "2": [0.2, 0.4, 0.5] }, learned: { "2": [0.1, 0.3, 0.4] } },
    mech: { edges: [-2.5, 0, 2], z: [[1, 0], [0, 2]], gap: [0.2, 0.4, 0.6] },
  } },
});
const nativeStamps = () => {
  const cell = { sr: stamp(), ref: stamp(), occ: stamp(), radial: [1, 0.5, 0.2], n: 40, log_q: -2, learned: 0.4, dist: 12 };
  return { side: 41, radial_px: [0, 1, 2], groups: { learned: [{ bin: 2, label: "0.1–1σ", norm: [-0.3, 3, 0.3],
    cells: { lower: cell, usual: cell, higher: cell } }], raw: [] }, types: [cell, cell] };
};

describe("knees tab › pixel types, native", () => {
  beforeEach(() => {
    const reports = pixelReports();
    reports.analyses[0].files = [...reports.analyses[0].files, "view_synthetic.json", "stamps_synthetic_VIS.json"];
    routes["GET /api/knee-attribution/pixel-types"] = () => ({ body: reports });
    routes[`GET ${fileUrl(FOLDER, "summary.json")}`] = () => ({ body: {} });
    routes[`GET ${fileUrl(FOLDER, "view_synthetic.json")}`] = () => ({ body: nativeView() });
    routes[`GET ${fileUrl(FOLDER, "stamps_synthetic_VIS.json")}`] = () => ({ body: nativeStamps() });
  });

  it("draws the groups as canvas stamps with the radial occupancy, keeping the PNG one click away", async () => {
    show("/models/knees?view=pixel-types");
    const table = await screen.findByRole("table", { name: "Pixel groups" });
    expect(within(table).getAllByRole("img", { name: /0\.1–1σ, Lower knee: SR/ })).toHaveLength(1);
    expect(table.querySelectorAll(".kn-stamp canvas")).toHaveLength(9);
    expect(within(table).getAllByText(/n = 40 · q̄ = 0\.01σ \(\+0\.40 dex\) · d = 12 px/)).toHaveLength(3);
    expect(screen.getByRole("figure", { name: "Source occupancy vs distance, 0.1–1σ" })).toBeTruthy();
    expect(screen.getByRole("link", { name: "Open as PNG" }).getAttribute("href")).toBe(fileUrl(FOLDER, "groups_learned_synthetic_VIS.png"));
    expect(figureSrcs()).not.toContain(fileUrl(FOLDER, "groups_learned_synthetic_VIS.png"));   // not the PNG
    expect(posts()).toHaveLength(0);
  });

  it("draws the knee map, the types and the type mix from the view", async () => {
    show("/models/knees?view=pixel-types&pf=map");
    expect(await screen.findByRole("figure", { name: "Effective knee vs brightness, VIS" })).toBeTruthy();
    fireEvent.click(chip("Pixel types figure", "Pixel types"));
    const types = await screen.findByRole("list", { name: "Knee types, VIS" });
    expect(within(types).getAllByRole("listitem")).toHaveLength(2);
    expect(types.textContent).toMatch(/Type 0 · 80\.0% of pixels/);
    expect(types.textContent).toMatch(/empty sky/);
    fireEvent.click(chip("Pixel types figure", "Types by brightness"));
    const mix = await screen.findByRole("table", { name: "Knee-type mix per brightness decade, VIS" });
    expect(within(mix).getByText("no pixels")).toBeTruthy();
  });
});
