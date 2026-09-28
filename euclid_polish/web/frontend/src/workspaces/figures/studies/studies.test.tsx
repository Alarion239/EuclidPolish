/* Figures › Studies against a mocked backend: the pure shaping (CSV, the
 * chart / export URLs, recipe keys), the list (resume, confirmed delete),
 * the study view drawing its charts from the chart CSVs, export links that
 * carry the selection, and the field fetches (explicit, one at a time,
 * never on open). */
import { QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { JOBS_FEED_TIMING, useJobsStore } from "../../../api/jobs";
import { LOSS_COLOR, categorical } from "../../../colors";
import { queryClient } from "../../../api/query";
import { resetConfirm, toast } from "../../../ui";
import Studies from "../tabs/Studies";
import type { StudyDetail, StudySummary } from "./api";
import { makePalette } from "./StudyCharts";
import { chartQuery, csvUrl, deltaText, figureUrl, groupKey, kneeCurves, memberKey, parseCsv } from "./model";

vi.mock("../../../viewer", () => ({
  ImageViewer: (p: { collection: string; params: Record<string, string>; initialId?: string; tiers?: string[] }) => (
    <div data-testid="viewer" data-collection={p.collection} data-study={p.params.study} data-id={p.initialId} data-tiers={(p.tiers ?? []).join(",")} />
  ),
}));

type Reply = { status?: number; body: unknown; text?: string };
let routes: Record<string, (body: Record<string, string>) => Reply>;
let calls: { method: string; url: string; body: Record<string, string> }[];

const ID = "20260928-120000-loss-and-knee";
const L = ["170·psnr", "171·psnr", "203·psnr"];
const SUMMARY: StudySummary = {
  id: ID, name: "Loss and knee", created: "2026-09-28T12:00:00Z", completed: "2026-09-28T12:03:00Z", regime: "starfull", members: 3,
  gate: "spatial_gate_p20s1", fields: 2, field_ids: ["test-00000", "real-poster-181255"], note: "for §4",
  commit: { short: "9f25d8c", dirty: false }, state: "complete", reason: null, numbers_bytes: 3_000_000, fields_bytes: 200_000_000,
};
const INCOMPLETE: StudySummary = { ...SUMMARY, id: "20260927-090000-half", name: "Half done", state: "incomplete", reason: "1 field(s) not uploaded", fields: 1 };

const field = (fid: string, kind: string, patch: Partial<StudyDetail["fields"][number]> = {}): StudyDetail["fields"][number] => ({
  fid, kind, ref: fid, label: fid === "test-00000" ? "test · idx 0" : "Poster galaxy", state: "uploaded", bytes: 100_000_000, estimated_bytes: 150_000_000,
  core_bytes: 20_000_000, member_bytes: { member_170: 10_000_000, member_171: 10_000_000, member_203: 10_000_000 }, gate: null,
  fetched: false, cached_products: [], members_fetched: 0, products: ["lr", "hr", "mean", "gate", "member_170", "member_171", "member_203"],
  thumb_url: `/api/studies/${ID}/numbers/thumbs/${fid}.jpg`, viewer: { collection: "study", params: { study: ID }, id: fid }, ...patch,
});

const DETAIL: StudyDetail = {
  ok: true, study: SUMMARY, manifest_sha256: "ab".repeat(32), note: "for §4",
  manifest: {
    id: ID, name: "Loss and knee", created: SUMMARY.created ?? undefined, completed: SUMMARY.completed, regime: "starfull", complete: true,
    commit: { short: "9f25d8c", dirty: false },
    ensemble: { members: [
      { label: L[0], loss: "l1", asinh_knee: 10, seed: 1 }, { label: L[1], loss: "l1", asinh_knee: 100, seed: 2 },
      { label: L[2], loss: "l2", asinh_knees: [0.1, 1, 10], seed: 3 },
    ] },
    gate: { name: "spatial_gate_p20s1", reads: L, mix_space: "linear", fingerprint: "gatefp" },
    warnings: ["Combiner comparison: it compared 2 members, 3 are active now."],
  },
  selections: [{ name: "L1 only", members: [L[0], L[1]], group: "training_knee" }],
  fields: [field("test-00000", "test"), field("real-poster-181255", "real")],
  charts: ["knee", "integrated", "paired", "gate", "training", "real"],
  group_fields: ["loss", "training_knee", "asinh_knee", "output_knee", "knee_loss", "blocks", "bootstrap", "noise_aug", "icnr", "status", "op"],
  citation: `model study ${ID} “Loss and knee” (study.json sha256 abababababababab)`,
  numbers: {
    knee_psnr: { knees: [0.1, 1, 10], bands: ["VIS", "H_E"], fields: [0, 1, 2, 3] },
    gate: { diagnostic: { available: true, labels: L, bands: ["VIS", "H_E"], brightness_names: ["sky", "core"] }, compare_note: null },
    real: { experiments: [], note: "no Sky › Compare run used its membership before the freeze" },
  },
};

const KNEE_CSV = [
  "series,kind,n_members,band,knee_e,psnr,lo,hi",
  ...["170·psnr", "171·psnr", "mean", "gate"].flatMap((s, i) => ["VIS", "H_E"].flatMap((b) => [0.1, 1, 10].map((k) =>
    `${s},${s === "mean" || s === "gate" ? s : "member"},1,${b},${k},${50 + i + k / 10},,`))),
].join("\n");
const PAIRED_CSV = [
  "target,n_members,reference,band,mean_delta,lo,hi,n_fields,n_resamples,seed",
  "170·psnr,1,gate,VIS,0.12,0.05,0.19,100,2000,0", "170·psnr,1,gate,H_E,-0.3,-0.5,-0.1,100,2000,0",
  "171·psnr,1,gate,VIS,0.01,-0.04,0.06,100,2000,0", "171·psnr,1,gate,H_E,0.2,0.1,0.3,100,2000,0",
].join("\n");
const GATE_CSV = [
  "level,name,family,band,weight,uniform,source",
  "member,170·psnr,l1,VIS,0.5,0.333333,all", "member,171·psnr,l1,VIS,0.3,0.333333,all", "member,203·psnr,l2,VIS,0.2,0.333333,all",
  "family,l1,l1,VIS,0.8,0.666667,all", "family,l2,l2,VIS,0.2,0.333333,all",
].join("\n");
const INTEGRATED_CSV = [
  "series,kind,loss,training_knee,seed,group,band,integrated_psnr",
  "170·psnr,member,l1,10,1,,VIS,55.1", "171·psnr,member,l1,100,2,,VIS,55.3", "203·psnr,member,l2,0.1+1+10,3,,VIS,55.6",
  "mean,mean,,,,,VIS,55.4", "gate,gate,,,,,VIS,55.9",
].join("\n");
const TRAINING_CSV = ["member,group,metric,step,value", "170·psnr,170·psnr,psnr,1000,40.1", "170·psnr,170·psnr,psnr,2000,40.9"].join("\n");

/** A form body as a record; a JSON body stays its text under `json`. */
const bodyOf = (body: BodyInit | null | undefined): Record<string, string> => {
  const out: Record<string, string> = {};
  if (body instanceof FormData) body.forEach((v, k) => { out[k] = String(v); });
  else if (typeof body === "string") out.json = body;
  return out;
};

const fig = (chart: string, query = "") => `/api/studies/${ID}/figure/${chart}.csv${query}`;

beforeEach(() => {
  calls = [];
  routes = {
    "GET /api/studies": () => ({ body: { ok: true, studies: [SUMMARY, INCOMPLETE], root: "tracking/studies", max_fields: 10, freezing: null } }),
    [`GET /api/studies/${ID}`]: () => ({ body: DETAIL }),
    [`GET ${fig("knee")}`]: () => ({ body: null, text: KNEE_CSV }),
    [`GET ${fig("integrated")}`]: () => ({ body: null, text: INTEGRATED_CSV }),
    [`GET ${fig("paired", "?reference=gate")}`]: () => ({ body: null, text: PAIRED_CSV }),
    [`GET ${fig("paired")}`]: () => ({ body: null, text: PAIRED_CSV.replace(/,gate,/g, ",mean,") }),
    [`GET ${fig("gate")}`]: () => ({ body: null, text: GATE_CSV }),
    [`GET ${fig("training")}`]: () => ({ body: null, text: TRAINING_CSV }),
    "GET /api/jobs?summary=1": () => ({ body: [] }),
    [`POST /api/studies/${ID}/fields/test-00000/fetch`]: () => ({ body: { ok: true, job_id: "fet00001", study_id: ID, fid: "test-00000", products: ["lr"], bytes: 1 } }),
    "GET /api/jobs/fet00001": () => ({ body: { job_id: "fet00001", label: `study ${ID}: fetch test-00000 (core)`, kind: "study-fetch", status: "running", duration: 1, error: null, log: "", log_truncated: false, progress: null } }),
    [`POST /api/studies/${INCOMPLETE.id}/delete`]: () => ({ body: { ok: true, id: INCOMPLETE.id, remote: [] } }),
    [`POST /api/studies/${INCOMPLETE.id}/resume`]: () => ({ body: { ok: true, job_id: "frz00002", study_id: INCOMPLETE.id } }),
    "GET /api/jobs/frz00002": () => ({ body: { job_id: "frz00002", label: "study resume: Half done", kind: "study-freeze", status: "running", duration: 1, error: null, log: "", log_truncated: false, progress: null } }),
    [`POST /api/studies/${ID}/selections`]: () => ({ body: { ok: true } }),
  };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = String(input);
    const method = init.method ?? "GET";
    calls.push({ method, url, body: bodyOf(init.body) });
    // Any selection of a chart gets that chart's table (exact routes above win).
    const chart = /\/figure\/(\w+)\.csv/.exec(url)?.[1];
    const table = chart ? ({ knee: KNEE_CSV, integrated: INTEGRATED_CSV, paired: PAIRED_CSV, gate: GATE_CSV, training: TRAINING_CSV } as Record<string, string>)[chart] : undefined;
    const r = routes[`${method} ${url}`]?.(bodyOf(init.body))
      ?? (method === "GET" && table ? { body: null, text: table } : { status: 404, body: { ok: false, error: `no route ${method} ${url}` } });
    return new Response(r.text ?? JSON.stringify(r.body), { status: r.status ?? 200 });
  }));
  queryClient.clear();
  useJobsStore.getState().reset();
});
afterEach(() => {
  act(() => resetConfirm());
  queryClient.clear();
  vi.unstubAllGlobals();
});

function Probe() {
  const loc = useLocation();
  return <output data-testid="loc">{loc.pathname}{loc.search}</output>;
}
const show = (url: string) => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={[url]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Studies /><Probe />
    </MemoryRouter>
  </QueryClientProvider>,
);
const loc = () => decodeURIComponent(screen.getByTestId("loc").textContent ?? "");
const answer = async (title: string, button: string) => {
  const dlg = await screen.findByRole("alertdialog", { name: title });
  fireEvent.click(within(dlg).getByRole("button", { name: button }));
  await waitFor(() => expect(screen.queryByRole("alertdialog", { name: title })).toBeNull());
};
const posts = () => calls.filter((c) => c.method === "POST");

describe("model", () => {
  it("parses RFC 4180 CSV (quotes, escaped quotes, CRLF)", () => {
    expect(parseCsv('a,b\r\n"x, y","say ""hi"""\r\n1,\n')).toEqual([{ a: "x, y", b: 'say "hi"' }, { a: "1", b: "" }]);
    expect(parseCsv("")).toEqual([]);
  });

  it("builds chart and export URLs that carry the selection and only that chart's options", () => {
    const sel = { members: ["170·psnr", "171·psnr"], group: "loss", reference: "gate", source: "core", metric: "VIS" };
    expect(chartQuery("paired", sel).toString()).toBe("members=170%C2%B7psnr%2C171%C2%B7psnr&group=loss&reference=gate");
    expect(chartQuery("knee", sel).get("reference")).toBeNull();
    expect(chartQuery("gate", sel).get("source")).toBe("core");
    expect(csvUrl("S", "training", sel, true)).toBe("/api/studies/S/figure/training.csv?members=170%C2%B7psnr%2C171%C2%B7psnr&group=loss&metric=VIS&download=1");
    expect(figureUrl("S", "knee", "pdf", {}, 600)).toBe("/api/studies/S/figure/knee?format=pdf&dpi=600&download=1");
  });

  it("names recipe groups like the backend", () => {
    expect(groupKey([0.1, 1, 10])).toBe("0.1+1+10");
    expect(groupKey(100)).toBe("100");
    expect(groupKey(null)).toBe("—");
    expect(groupKey(true)).toBe("True");
    expect(memberKey({ asinh_knee: 10, asinh_knees: [] }, "training_knee")).toBe("10");
    expect(memberKey({ asinh_knee: 10, asinh_knees: [1, 10] }, "training_knee")).toBe("1+10");
  });

  it("rounds a difference before it picks its sign", () => {
    expect(deltaText({ mean: -0.004, lo: -0.3, hi: 0.2 })).toBe("0.00 [−0.30, +0.20]");
    expect(deltaText({ mean: 0.126, lo: null, hi: 0.199 })).toBe("+0.13 [—, +0.20]");
  });

  it("colours a loss with the console's loss colour, other fields in the export palette's order", () => {
    const pal = makePalette(DETAIL.manifest.ensemble!.members!);
    expect(pal("loss").of("l2").color).toBe(LOSS_COLOR.l2);
    expect(pal("loss").facet(L[0]).color).toBe(LOSS_COLOR.l1);
    const knee = pal("training_knee");
    expect(knee.items.map((f) => f.label)).toEqual(["Training knee 10", "Training knee 100", "Training knee 0.1+1+10"]);
    expect(knee.items.map((f) => f.color)).toEqual([categorical(0), categorical(2), categorical(1)]);   // blue, orange, green: render.py PALETTE
  });

  it("arranges the knee table into one curve per series per band", () => {
    const k = kneeCurves(parseCsv(KNEE_CSV));
    expect(k.bands).toEqual(["VIS", "H_E"]);
    expect(k.curves.VIS.map((c) => c.name)).toEqual(["170·psnr", "171·psnr", "mean", "gate"]);
    expect(k.curves.VIS[0]).toMatchObject({ x: [0.1, 1, 10], lo: null });
  });
});

describe("the list", () => {
  it("lists every study with its members, gate, fields and note; incomplete ones say so and offer Resume", async () => {
    show("/figures/studies");
    expect(await screen.findByRole("link", { name: "Loss and knee" })).toBeTruthy();
    expect(screen.getByRole("link", { name: "Loss and knee" }).getAttribute("href")).toBe(`/figures/studies?study=${ID}`);
    expect(screen.getAllByText("spatial_gate_p20s1")).toHaveLength(2);
    expect(screen.getAllByText("for §4").length).toBeGreaterThan(0);
    expect(screen.getByText("incomplete")).toBeTruthy();
    expect(screen.getAllByRole("button", { name: "Resume" })).toHaveLength(1);
    expect(posts()).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "Resume" }));
    await waitFor(() => expect(posts().map((p) => p.url)).toEqual([`/api/studies/${INCOMPLETE.id}/resume`]));
  });

  it("keeps the freeze job while the dialog opened from the list shows it, and lets go when it closes", async () => {
    const saved = JOBS_FEED_TIMING.detailMinMs;
    JOBS_FEED_TIMING.detailMinMs = 10;
    try {
      routes["GET /api/studies/candidates?mode=starfull"] = () => ({ body: {
        ok: true, regime: "starfull", fields: [], max_fields: 10, can_freeze: true, blocking: null, fasrc_connected: true, fields_note: null,
        ensemble: { members: L, n_members: 3, gate: { available: true, state: "current", name: "spatial_gate_p20s1" }, evaluated_at: null,
          blocks: [{ id: "members", title: "Members", state: "current", detail: "3 active starfull members." }], stale: [], numbers_bytes: 1000 },
      } });
      routes["POST /api/studies"] = () => ({ body: { ok: true, job_id: "frz00003", study_id: "20260928-130000-new", fields: [], upload_bytes: 0 } });
      let reads = 0;
      const job = { job_id: "frz00003", label: "study freeze: New", kind: "study-freeze", duration: 1, error: null, log: "", log_truncated: false, progress: null };
      routes["GET /api/jobs/frz00003"] = () => ({ body: { ...job, status: ++reads < 3 ? "running" : "done", result: { study_id: "20260928-130000-new" } } });
      show("/figures/studies");
      fireEvent.click(await screen.findByRole("button", { name: "Freeze study…" }));
      const dlg = await screen.findByRole("dialog", { name: "Freeze a study" });
      fireEvent.click(await within(dlg).findByRole("button", { name: "Freeze without fields" }));
      fireEvent.change(within(screen.getByRole("dialog")).getByRole("textbox", { name: "Name" }), { target: { value: "New" } });
      fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Freeze" }));
      expect(await screen.findByText("Study frozen")).toBeTruthy();
      await new Promise((r) => setTimeout(r, 50));
      expect(screen.getByText("Study frozen")).toBeTruthy();                    // the list did not take the job away
      expect(within(screen.getByRole("dialog")).getByRole("link", { name: "Open the study" }).getAttribute("href")).toBe("/figures/studies?study=20260928-130000-new");
      fireEvent.click(within(screen.getByRole("dialog")).getAllByRole("button", { name: "Close" }).at(-1)!);   // the footer's (the × is also "Close")
      await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
      expect(useJobsStore.getState().keyed["study:freeze"]).toBeUndefined();
    } finally {
      JOBS_FEED_TIMING.detailMinMs = saved;
    }
  });

  it("deletes only after the confirmation", async () => {
    show("/figures/studies");
    await screen.findByText("Half done");
    const row = screen.getByText("Half done").closest("tr") as HTMLElement;
    fireEvent.click(within(row).getByRole("button", { name: "Delete" }));
    const first = await screen.findByRole("alertdialog", { name: "Delete the study “Half done”?" });
    fireEvent.click(within(first).getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull());
    expect(posts()).toHaveLength(0);
    fireEvent.click(within(row).getByRole("button", { name: "Delete" }));
    const again = await screen.findByRole("alertdialog", { name: "Delete the study “Half done”?" });
    fireEvent.click(within(again).getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(posts()).toHaveLength(1));
    expect(posts()[0]).toMatchObject({ url: `/api/studies/${INCOMPLETE.id}/delete`, body: { confirm: "1" } });
  });
});

describe("the study view", () => {
  it("states what was frozen and draws the knee chart from the chart CSV", async () => {
    show(`/figures/studies?study=${ID}`);
    expect(await screen.findByRole("heading", { name: "Loss and knee" })).toBeTruthy();
    expect(screen.getByText("Frozen with 1 block not current")).toBeTruthy();
    expect(await screen.findByRole("figure", { name: "PSNR vs knee, VIS" })).toBeTruthy();
    expect(screen.getByRole("figure", { name: "PSNR vs knee, H" })).toBeTruthy();
    expect(calls.some((c) => c.url === fig("knee"))).toBe(true);
    expect(screen.getByText(/averaged over 4 test fields/)).toBeTruthy();
  });

  it("draws every other chart, and says plainly when the study has no real-tile data", async () => {
    show(`/figures/studies?study=${ID}`);
    const tab = async (name: string) => fireEvent.mouseDown(await screen.findByRole("tab", { name }), { button: 0 });
    await tab("Integrated");
    expect(await screen.findByRole("figure", { name: "Integrated PSNR by training knee, VIS" })).toBeTruthy();
    await tab("Paired Δ");
    expect(await screen.findByRole("table", { name: "Paired differences with 95% intervals" })).toBeTruthy();
    expect(screen.getByText("+0.12 [+0.05, +0.19]")).toBeTruthy();
    await tab("Gate weights");
    expect(await screen.findByRole("figure", { name: "Gate weight per member, VIS" })).toBeTruthy();
    expect(screen.getByText("80%")).toBeTruthy();
    await tab("Training");
    expect(await screen.findByRole("figure", { name: "Training curves" })).toBeTruthy();
    await tab("Real tiles");
    expect(await screen.findByText(/No real-tile data in this study/)).toBeTruthy();
  });

  it("gives every chart PDF / PNG / SVG / CSV links that carry the selection, and a notebook entry citing the hash", async () => {
    show(`/figures/studies?study=${ID}&chart=paired&members=${encodeURIComponent(`${L[0]},${L[1]}`)}&ref=gate&dpi=600`);
    const pdf = await screen.findByRole("link", { name: "PDF" });
    const href = decodeURIComponent(pdf.getAttribute("href") ?? "");
    expect(href).toContain(`/api/studies/${ID}/figure/paired?format=pdf&dpi=600`);
    expect(href).toContain(`members=${L[0]},${L[1]}`);
    expect(href).toContain("reference=gate");
    expect(href).toContain("download=1");
    const csv = decodeURIComponent(screen.getByRole("link", { name: "CSV" }).getAttribute("href") ?? "");
    expect(csv).toBe(`/api/studies/${ID}/figure/paired.csv?members=${L[0]},${L[1]}&reference=gate&download=1`);
    expect(screen.getByRole("link", { name: "SVG" }).getAttribute("href")).toContain("format=svg");
    expect(screen.getByRole("link", { name: "PNG" }).getAttribute("href")).toContain("format=png");
    expect(screen.getByRole("button", { name: "Log to notebook" })).toBeTruthy();
  });

  it("applies a saved selection and saves a new one only on Save", async () => {
    show(`/figures/studies?study=${ID}`);
    await screen.findByRole("figure", { name: "PSNR vs knee, VIS" });
    fireEvent.change(screen.getByRole("combobox", { name: "Apply a saved selection" }), { target: { value: "L1 only" } });
    await waitFor(() => expect(calls.some((c) => c.url.startsWith(fig("knee", "?members=")) && c.url.includes("group=training_knee"))).toBe(true));
    expect(posts()).toHaveLength(0);
  });

  it("fetches nothing on open; Fetch field is explicit and every fetch button waits while a fetch runs", async () => {
    show(`/figures/studies?study=${ID}`);
    await screen.findByRole("figure", { name: "PSNR vs knee, VIS" });
    await new Promise((r) => setTimeout(r, 30));
    expect(posts()).toHaveLength(0);
    expect(screen.queryByTestId("viewer")).toBeNull();
    const fetches = screen.getAllByRole("button", { name: "Fetch field" }) as HTMLButtonElement[];
    expect(fetches).toHaveLength(2);
    expect(fetches.every((b) => !b.disabled)).toBe(true);
    fireEvent.click(fetches[0]);
    await waitFor(() => expect(posts()).toHaveLength(1));
    expect(posts()[0]).toMatchObject({ url: `/api/studies/${ID}/fields/test-00000/fetch`, body: { products: "core" } });
    await waitFor(() => expect((screen.getAllByRole("button", { name: "Fetch field" }) as HTMLButtonElement[]).every((b) => b.disabled)).toBe(true));
    expect((screen.getAllByRole("button", { name: "Fetch every member that fits" })[1] as HTMLButtonElement).disabled).toBe(true);
  });

  it("disables fetching while another page's fetch runs", async () => {
    routes["GET /api/jobs?summary=1"] = () => ({ body: [{ job_id: "other001", label: "study X: fetch real-a (core)", kind: "study-fetch", status: "running", duration: 5, error: null, log: null, log_truncated: false, progress: null }] });
    show(`/figures/studies?study=${ID}`);
    await screen.findByRole("figure", { name: "PSNR vs knee, VIS" });
    await waitFor(() => expect((screen.getAllByRole("button", { name: "Fetch field" }) as HTMLButtonElement[]).every((b) => b.disabled)).toBe(true));
    const why = screen.getByText(/Another field is being fetched \(study X: fetch real-a \(core\)\); one fetch runs at a time/);
    expect(screen.getAllByRole("button", { name: "Fetch field" })[0].getAttribute("aria-describedby")).toBe(why.id);
  });

  it("waits for FASRC to fetch, and says so", async () => {
    routes["GET /api/fasrc/status"] = () => ({ body: { ssh_connected: false, connected_at: null, socket: null, last_error: "timed out" } });
    show(`/figures/studies?study=${ID}`);
    const why = await screen.findByText(/FASRC is not connected \(timed out\): fields are fetched from holylabs/);
    const fetches = screen.getAllByRole("button", { name: "Fetch field" }) as HTMLButtonElement[];
    expect(fetches.every((b) => b.disabled && b.getAttribute("aria-describedby") === why.id)).toBe(true);
    expect(posts()).toHaveLength(0);
  });

  it("opens a fetched field in the study viewer, with only the fetched members as tiers to add", async () => {
    routes[`GET /api/studies/${ID}`] = () => ({ body: { ...DETAIL, fields: [field("test-00000", "test", { fetched: true, cached_products: ["lr", "hr", "mean", "gate", "member_171"], members_fetched: 1 }), field("real-poster-181255", "real")] } });
    show(`/figures/studies?study=${ID}&field=test-00000`);
    const viewer = await screen.findByTestId("viewer");
    expect(viewer.dataset).toMatchObject({ collection: "study", study: ID, id: "test-00000", tiers: "lr,sr,hr" });
    const chips = screen.getByRole("group", { name: "Fetched member SRs" });
    expect(within(chips).getAllByRole("button").map((b) => b.textContent)).toEqual(["member 171"]);
    const select = screen.getByRole("combobox", { name: "Member to fetch for test · idx 0" }) as HTMLSelectElement;
    expect([...select.options].map((o) => o.value)).toEqual(["member_170", "member_203"]);
    expect(posts()).toHaveLength(0);
  });

  it("saves a named selection with the current members and grouping, only on Save", async () => {
    show(`/figures/studies?study=${ID}&members=${encodeURIComponent(`${L[0]},${L[2]}`)}&group=loss`);
    await screen.findByRole("figure", { name: "PSNR vs knee, VIS" });
    fireEvent.click(screen.getByRole("button", { name: "Save selection…" }));
    const input = await screen.findByRole("textbox", { name: "Name" });
    fireEvent.change(input, { target: { value: "L1 vs L2" } });
    expect(posts()).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(posts()).toHaveLength(1));
    expect(posts()[0].url).toBe(`/api/studies/${ID}/selections`);
    expect(JSON.parse(posts()[0].body.json)).toEqual({ selections: [
      { name: "L1 only", members: [L[0], L[1]], group: "training_knee" },
      { name: "L1 vs L2", members: [L[0], L[2]], group: "loss" },
    ] });
  });

  it("offers no export or notebook entry on a chart without data", async () => {
    routes[`GET /api/studies/${ID}`] = () => ({ body: { ...DETAIL, numbers: { ...DETAIL.numbers, gate: { diagnostic: { available: false } } } } });
    show(`/figures/studies?study=${ID}&chart=gate`);
    expect(await screen.findByText(/No gate weight diagnostic in this study/)).toBeTruthy();
    expect(screen.queryByRole("link", { name: "PDF" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Log to notebook" })).toBeNull();
    expect(calls.some((c) => c.url.includes("/figure/gate"))).toBe(false);
    fireEvent.mouseDown(screen.getByRole("tab", { name: "Real tiles" }), { button: 0 });
    expect(await screen.findByText(/No real-tile data in this study/)).toBeTruthy();
    expect(screen.queryByRole("link", { name: "CSV" })).toBeNull();
  });

  it("saves the note only when asked", async () => {
    routes[`POST /api/studies/${ID}/note`] = (b) => ({ body: { ok: true, id: ID, note: b.note } });
    show(`/figures/studies?study=${ID}`);
    const note = await screen.findByRole("textbox", { name: "Note" });
    fireEvent.change(note, { target: { value: "for §4 and §5" } });
    expect(posts()).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "Save note" }));
    await waitFor(() => expect(posts()).toHaveLength(1));
    expect(posts()[0]).toMatchObject({ url: `/api/studies/${ID}/note`, body: { note: "for §4 and §5" } });
  });

  it("sends a new paired reference to the CSV and drops a member reference the subset excludes", async () => {
    routes[`GET ${fig("paired", `?members=${encodeURIComponent(`${L[0]},${L[1]}`)}&reference=${encodeURIComponent(L[1])}`)}`] = () => ({ body: null, text: PAIRED_CSV });
    show(`/figures/studies?study=${ID}&chart=paired`);
    await screen.findByRole("table", { name: "Paired differences with 95% intervals" });
    fireEvent.change(screen.getByRole("combobox", { name: "Reference" }), { target: { value: "gate" } });
    await waitFor(() => expect(calls.some((c) => c.url === fig("paired", "?reference=gate"))).toBe(true));
    fireEvent.change(screen.getByRole("combobox", { name: "Reference" }), { target: { value: L[2] } });
    await waitFor(() => expect(loc()).toContain(`ref=${L[2]}`));
    // picking a subset without that member clears the reference
    fireEvent.click(screen.getByRole("button", { name: /Members/ }));
    const list = await screen.findByRole("listbox");
    fireEvent.click(within(list).getByRole("option", { name: /member 170/ }));
    await waitFor(() => expect(loc()).not.toContain("ref="));
  });

  it("drops a group reference whose group leaves the subset", async () => {
    show(`/figures/studies?study=${ID}&chart=paired&group=loss&ref=${encodeURIComponent("group:l2")}`);
    await screen.findByRole("table", { name: "Paired differences with 95% intervals" });
    expect(loc()).toContain("ref=group:l2");
    fireEvent.click(screen.getByRole("button", { name: /Members/ }));
    const list = await screen.findByRole("listbox");
    fireEvent.click(within(list).getByRole("option", { name: /member 170/ }));      // an l1 member only
    await waitFor(() => expect(loc()).not.toContain("ref="));
  });

  it("after a confirmed delete lands on the list; offline with fields it asks to delete the local copy only", async () => {
    let tries = 0;
    routes[`POST /api/studies/${ID}/delete`] = (b) => (++tries === 1 && !b.local_only
      ? { status: 409, body: { ok: false, error: "FASRC is not connected: this study's fields are on holylabs; delete with local_only=1 (the holylabs copy then stays)" } }
      : { body: { ok: true, id: ID, remote: [] } });
    show(`/figures/studies?study=${ID}`);
    fireEvent.click(await screen.findByRole("button", { name: "Delete study…" }));
    await answer("Delete the study “Loss and knee”?", "Delete");
    await answer("Delete only the local copy?", "Delete local copy");
    await waitFor(() => expect(posts()).toHaveLength(2));
    expect(posts()[0].body).toEqual({ confirm: "1" });
    expect(posts()[1].body).toEqual({ confirm: "1", local_only: "1" });
    await waitFor(() => expect(loc()).toBe("/figures/studies"));
  });

  it("words a resume refusal", async () => {
    const error = vi.spyOn(toast, "error").mockImplementation(() => "t");
    routes[`POST /api/studies/${INCOMPLETE.id}/resume`] = () => ({ status: 503, body: { ok: false, code: "fasrc_offline", error: "FASRC not connected" } });
    show("/figures/studies");
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(error).toHaveBeenCalledWith(expect.stringMatching(/^FASRC is not connected: /)));
    routes[`POST /api/studies/${INCOMPLETE.id}/resume`] = () => ({ status: 409, body: { ok: false, error: "the ensemble changed since this study started (labels); delete it and freeze again" } });
    fireEvent.click(screen.getByRole("button", { name: "Resume" }));
    await waitFor(() => expect(error).toHaveBeenCalledWith(expect.stringMatching(/the ensemble changed since this study started/)));
    error.mockRestore();
  });

  it("says why Resume waits while another study is being frozen", async () => {
    routes["GET /api/studies"] = () => ({ body: { ok: true, studies: [SUMMARY, INCOMPLETE], root: "r", max_fields: 10, freezing: { job_id: "frz00002", study_id: ID } } });
    show("/figures/studies");
    const resume = await screen.findByRole("button", { name: "Resume" }) as HTMLButtonElement;
    await waitFor(() => expect(resume.disabled).toBe(true));
    const why = document.getElementById(resume.getAttribute("aria-describedby") ?? "");
    expect(why?.textContent).toMatch(/A study is being frozen \(“Loss and knee”\): Resume waits/);
  });
});
