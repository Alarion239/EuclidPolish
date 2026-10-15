/* Models › Knees against a mocked Flask (the knee-attribution API of spec
 * 2026-10-15, fixtures in ./testing/kneesFixture): opening the tab only
 * GETs (no POST, and no catalogue read while the run panel is closed); the
 * main chart draws one line per knee in each model's panel with the knee
 * legend; the class bars list the classes in order with their n, and the
 * gradient / ablation toggle changes the values; the cross-band tables; the
 * field explorer draws a dot per sample of the band on the record's image;
 * the run panel preselects the newest universal models, asks with the
 * estimate and POSTs the right form; no runs → the explanation and an open
 * run panel; a failed run → its errors. */
import { QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobsStore } from "../../api/jobs";
import { queryClient } from "../../api/query";
import { resetConfirm } from "../../ui";
import Knees from "./tabs/Knees";
import {
  NEXUS_TILES, OLD_RUN_ID, POSTER_TILES, RUN_ID, SOURCES, catalogue, failedRecord, record, runs,
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
    fireEvent.click(within(screen.getByRole("radiogroup", { name: "Split" })).getByRole("radio", { name: "Gradient" }));
    await waitFor(() => expect(bar()).toBe(gradient));
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
    expect(screen.getByText("local job: 2 models × 6 fields, ~1 min")).toBeTruthy();
    expect(posts()).toHaveLength(0);
    expect(gets("/api/knee-attribution/models")).toHaveLength(1);
    expect(gets("/api/real/tile")).toHaveLength(0);                       // no cached tiles: never asked

    fireEvent.click(screen.getByRole("button", { name: "Run…" }));
    const dlg = await screen.findByRole("alertdialog", { name: "Run knee attribution?" });
    expect(dlg.textContent).toContain("Local job: 2 models × 6 fields, ~1 min.");
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

  it("names partial errors of a finished run above its figures", async () => {
    routes[`GET /api/knee-attribution/${RUN_ID}`] = () => ({ body: record({ errors: { "universal:member_222|test/9": "ValueError: empty field" } }) });
    show();
    expect(await screen.findByText("1 part of this run failed")).toBeTruthy();
    expect(screen.getByText("#222 › test/9")).toBeTruthy();
    await waitFor(() => expect(panels()).toHaveLength(2));
  });
});
