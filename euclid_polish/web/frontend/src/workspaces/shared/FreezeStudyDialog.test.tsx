/* The freeze-study dialog against a mocked backend: only GETs until
 * "Freeze", the three steps, the 10-field limit, disabled fields with their
 * reasons, and every refusal (409 stale / busy, 503 offline, 507 disk). */
import { QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { JOBS_FEED_TIMING, useJobsStore, type Job } from "../../api/jobs";
import { queryClient } from "../../api/query";
import { resetConfirm } from "../../ui";
import { FreezeStudyDialog } from "./FreezeStudyDialog";
import { FREEZE_JOB_KEY } from "./studyFreeze";
import { counterText, groupFields, refusalMessage, togglePick, type CandidateField, type Candidates } from "./studyFreeze";
import { ApiError } from "../../api/client";

type Reply = { status?: number; body: unknown };
let routes: Record<string, (body: Record<string, string>) => Reply>;
let calls: { method: string; url: string; body: Record<string, string> }[];

const MB = 1024 * 1024;
const field = (i: number, patch: Partial<CandidateField> = {}): CandidateField => ({
  fid: `test-${String(i).padStart(5, "0")}`, kind: "test", ref: String(i), label: `test · idx ${i}`, available: true, reason: null,
  bytes: 40 * MB, core_bytes: 4 * MB, largest_product_bytes: MB, bytes_upper_bound: true,
  thumb_url: `/api/studies/candidates/thumb/test-${String(i).padStart(5, "0")}.jpg?mode=starfull`, ...patch,
});

const CANDIDATES: Candidates = {
  ok: true, regime: "starfull",
  ensemble: {
    members: ["170·psnr", "171·psnr", "203·psnr"], n_members: 3,
    gate: { available: true, state: "current", name: "spatial_gate_p20s1", reads: ["170·psnr", "171·psnr", "203·psnr"], mix_space: "linear", fitted_at: "2026-09-27T10:00:00Z" },
    evaluated_at: "2026-09-28T08:00:00Z",
    blocks: [
      { id: "members", title: "Members", state: "current", detail: "3 active starfull members." },
      { id: "test_cubes", title: "Test-set curves", state: "current", detail: "Current." },
      { id: "compare", title: "Combiner comparison", state: "stale", detail: "It compared 2 members, 3 are active now." },
    ],
    stale: ["compare"], numbers_bytes: 3 * MB,
  },
  fields: [
    ...Array.from({ length: 12 }, (_, i) => field(i)),
    field(50, { available: false, reason: "no cached SR of member(s) 203 in this field" }),
    field(51, { available: false, reason: null }),
    field(60, { fid: "blackout-00000", kind: "blackout", label: "blackout · idx 0", available: false, reason: "the blackout cubes hold members that are no longer active" }),
    field(61, { fid: "blackout-00002", kind: "blackout", label: "blackout · idx 2", available: false, reason: "the blackout cubes hold members that are no longer active" }),
    field(70, { fid: "real-poster-181255", kind: "real", label: "Poster galaxy", bytes: 90 * MB }),
  ],
  max_fields: 10, can_freeze: true, blocking: null, fasrc_connected: true, fields_note: null,
};

const bodyOf = (body: BodyInit | null | undefined): Record<string, string> => {
  const out: Record<string, string> = {};
  if (body instanceof FormData) body.forEach((v, k) => { out[k] = String(v); });
  return out;
};

beforeEach(() => {
  calls = [];
  routes = {
    "GET /api/studies/candidates?mode=starfull": () => ({ body: CANDIDATES }),
    "POST /api/studies": () => ({ body: { ok: true, job_id: "frz00001", study_id: "20260928-120000-loss-and-knee", fields: [], upload_bytes: 0 } }),
    "GET /api/jobs/frz00001": () => ({ body: { job_id: "frz00001", label: "study freeze: Loss and knee", kind: "study-freeze", status: "running", duration: 3, error: null, log: "", log_truncated: false, cancellable: true,
      progress: { current: 0, total: 5, pct: 0, label: "numbers · knee curves per field" } } }),
    "GET /api/jobs?summary=1": () => ({ body: [] }),
  };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = String(input);
    const method = init.method ?? "GET";
    const body = bodyOf(init.body);
    calls.push({ method, url, body });
    const r = routes[`${method} ${url}`]?.(body) ?? { status: 404, body: { ok: false, error: `no route ${method} ${url}` } };
    return new Response(JSON.stringify(r.body), { status: r.status ?? 200 });
  }));
  queryClient.clear();
  useJobsStore.getState().reset();
});
afterEach(() => {
  act(() => resetConfirm());
  queryClient.clear();
  vi.unstubAllGlobals();
});

const open = () => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={["/models/leaderboard"]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <FreezeStudyDialog onClose={() => {}} />
    </MemoryRouter>
  </QueryClientProvider>,
);

const posts = () => calls.filter((c) => c.method === "POST");
const dialog = () => screen.getByRole("dialog");
const click = (name: string | RegExp) => fireEvent.click(within(dialog()).getByRole("button", { name }));

describe("pure rules", () => {
  it("groups fields by kind and says a reason every unavailable field shares once", () => {
    const g = groupFields(CANDIDATES.fields);
    expect(g.map((x) => [x.kind, x.fields.length, x.available])).toEqual([["test", 14, 12], ["blackout", 2, 0], ["real", 1, 1]]);
    expect(g[1].sharedReason).toBe("the blackout cubes hold members that are no longer active");
    expect(g[0].sharedReason).toBeNull();
  });

  it("never picks past the limit and counts the upper-bound upload", () => {
    let p: string[] = [];
    for (let i = 0; i < 12; i++) p = togglePick(p, `f${i}`, 10);
    expect(p).toHaveLength(10);
    expect(togglePick(p, "f0", 10)).toHaveLength(9);
    expect(counterText(3, 10, 480 * MB)).toBe("3 of 10 · ≤ 480 MiB to upload");
    expect(counterText(0, 10, 0)).toBe("0 of 10");
  });

  it("words every refusal", () => {
    expect(refusalMessage(new ApiError({ status: 503, code: "fasrc_offline", message: "x" })).title).toBe("FASRC is not connected");
    expect(refusalMessage(new ApiError({ status: 409, code: "busy", message: "busy: another study is being frozen" })).title).toBe("Another study is being frozen");
    const disk = refusalMessage(new ApiError({ status: 507, code: "insufficient_storage", message: "x", body: { needed_bytes: 2 * 1024 ** 3, free_bytes: 1024 ** 3 } }));
    expect(disk.text).toContain("It needs 2 GiB; 1 GiB is free.");
    expect(refusalMessage(new ApiError({ status: 409, message: "the production gate changed since the evaluation." })).text)
      .toBe("the production gate changed since the evaluation. Fix it (re-evaluate, refit), then check again.");
  });
});

describe("the dialog flow", () => {
  it("shows what will be frozen, with each block's state and where a stale one is fixed", async () => {
    open();
    expect(await screen.findByText("Combiner comparison")).toBeTruthy();
    expect(screen.getByText("It compared 2 members, 3 are active now.")).toBeTruthy();
    expect(within(dialog()).getByRole("button", { name: "Combiner" })).toBeTruthy();       // the stale block's fix
    expect(screen.getByText("1 of 3 blocks is not current")).toBeTruthy();
    expect(within(dialog()).getByRole("button", { name: "Choose fields…" })).toBeTruthy();
    expect(within(dialog()).getByRole("button", { name: "Freeze without fields" })).toBeTruthy();
    expect(within(dialog()).getByRole("button", { name: "Cancel" })).toBeTruthy();
    expect(posts()).toHaveLength(0);
  });

  it("lists fields with thumbnails, disables unavailable ones with their reason, and blocks an 11th pick", async () => {
    open();
    fireEvent.click(await within(dialog()).findByRole("button", { name: "Choose fields…" }));
    expect(screen.getByText("Synthetic test fields")).toBeTruthy();
    expect(screen.getByText("no cached SR of member(s) 203 in this field")).toBeTruthy();
    expect(screen.getByText("Unavailable: the blackout cubes hold members that are no longer active")).toBeTruthy();
    const off = screen.getByRole("checkbox", { name: "test · idx 50" }) as HTMLInputElement;
    expect(off.disabled).toBe(true);
    // the reason is the box's accessible description
    expect(document.getElementById(off.getAttribute("aria-describedby") ?? "")?.textContent).toBe("no cached SR of member(s) 203 in this field");
    // a field without any reason on the page points at nothing
    expect((screen.getByRole("checkbox", { name: "test · idx 51" }) as HTMLInputElement).hasAttribute("aria-describedby")).toBe(false);
    for (const box of screen.getAllByRole("checkbox")) {
      for (const id of (box.getAttribute("aria-describedby") ?? "").split(" ").filter(Boolean)) expect(document.getElementById(id)).not.toBeNull();
    }
    // a group with nothing to pick is its header and reason only
    expect(screen.queryByRole("checkbox", { name: "blackout · idx 0" })).toBeNull();
    expect(document.querySelector('img[src="/api/studies/candidates/thumb/test-00000.jpg?mode=starfull"]')).toBeTruthy();
    for (let i = 0; i < 10; i++) fireEvent.click(screen.getByRole("checkbox", { name: `test · idx ${i}` }));
    expect(screen.getByText("10 of 10 · ≤ 400 MiB to upload")).toBeTruthy();
    const eleventh = screen.getByRole("checkbox", { name: "test · idx 10" }) as HTMLInputElement;
    expect(eleventh.disabled).toBe(true);
    fireEvent.click(eleventh);
    expect(screen.getByText("10 of 10 · ≤ 400 MiB to upload")).toBeTruthy();
    expect(screen.getByText("At most 10 fields: unpick one to choose another.")).toBeTruthy();
    expect(document.getElementById(eleventh.getAttribute("aria-describedby") ?? "")?.textContent).toBe("At most 10 fields: unpick one to choose another.");
    expect(posts()).toHaveLength(0);
  });

  it("writes nothing until Freeze, which needs a name, then follows the job and links the study", async () => {
    open();
    fireEvent.click(await within(dialog()).findByRole("button", { name: "Choose fields…" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "test · idx 0" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Poster galaxy" }));
    expect(screen.getByText("2 of 10 · ≤ 130 MiB to upload")).toBeTruthy();
    click("Continue with 2 fields");
    const freeze = within(dialog()).getByRole("button", { name: "Freeze" }) as HTMLButtonElement;
    expect(freeze.disabled).toBe(true);
    expect(screen.getByText("Will be written")).toBeTruthy();
    expect(posts()).toHaveLength(0);
    fireEvent.change(within(dialog()).getByRole("textbox", { name: "Name" }), { target: { value: "Loss and knee" } });
    fireEvent.change(within(dialog()).getByRole("textbox", { name: "Note" }), { target: { value: "for §4" } });
    expect(posts()).toHaveLength(0);
    click("Freeze");
    await waitFor(() => expect(posts()).toHaveLength(1));
    expect(posts()[0]).toMatchObject({ url: "/api/studies", body: { name: "Loss and knee", note: "for §4", mode: "starfull", fields: "test-00000,real-poster-181255" } });
    expect(await screen.findByText("Freezing the study")).toBeTruthy();
    expect(await screen.findByText("numbers · knee curves per field")).toBeTruthy();
    // focus is on the step's body, never on the job's Cancel (a second Enter would press it)
    await waitFor(() => expect((document.activeElement as HTMLElement).classList.contains("sfz-step")).toBe(true));
    expect(within(dialog()).getByRole("button", { name: "Cancel" })).not.toBe(document.activeElement);
    const link = within(dialog()).getByRole("link", { name: "Open the study" });
    expect(link.getAttribute("href")).toBe("/figures/studies?study=20260928-120000-loss-and-knee");
  });

  it("freezes without fields straight from the first step", async () => {
    open();
    fireEvent.click(await within(dialog()).findByRole("button", { name: "Freeze without fields" }));
    fireEvent.change(within(dialog()).getByRole("textbox", { name: "Name" }), { target: { value: "Numbers only" } });
    click("Freeze");
    await waitFor(() => expect(posts()).toHaveLength(1));
    expect(posts()[0].body.fields).toBe("");
  });

  const refuse = async (reply: Reply) => {
    routes["POST /api/studies"] = () => reply;
    open();
    fireEvent.click(await within(dialog()).findByRole("button", { name: "Choose fields…" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "test · idx 0" }));
    click("Continue with 1 field");
    fireEvent.change(within(dialog()).getByRole("textbox", { name: "Name" }), { target: { value: "S" } });
    click("Freeze");
    await waitFor(() => expect(posts()).toHaveLength(1));
  };

  it("says FASRC is offline and offers to drop the fields (503)", async () => {
    await refuse({ status: 503, body: { ok: false, code: "fasrc_offline", error: "FASRC not connected" } });
    expect(await screen.findByText("FASRC is not connected")).toBeTruthy();
    fireEvent.click(within(dialog()).getByRole("button", { name: "Drop the fields" }));
    expect(screen.queryByText("FASRC is not connected")).toBeNull();
    click("Freeze");
    await waitFor(() => expect(posts()).toHaveLength(2));
    expect(posts()[1].body.fields).toBe("");
  });

  it("names another running freeze (409 busy)", async () => {
    await refuse({ status: 409, body: { ok: false, code: "busy", error: "busy: another study is being frozen (job abc); try again when it finishes", job_id: "abc" } });
    expect(await screen.findByText("Another study is being frozen")).toBeTruthy();
    expect(screen.getByText(/job abc/)).toBeTruthy();
  });

  it("names stale inputs and re-reads the candidates (409)", async () => {
    await refuse({ status: 409, body: { ok: false, error: "member 203's checkpoint changed since the evaluation" } });
    expect(await screen.findByText("The ensemble cannot be frozen as it is")).toBeTruthy();
    await waitFor(() => expect(calls.filter((c) => c.url.startsWith("/api/studies/candidates")).length).toBeGreaterThanOrEqual(2));
    fireEvent.click(within(dialog()).getByRole("button", { name: "Check again" }));
    expect(await screen.findByText("Combiner comparison")).toBeTruthy();
  });

  it("says the disk margin with the sizes (507)", async () => {
    await refuse({ status: 507, body: { ok: false, code: "insufficient_storage", error: "x", needed_bytes: 6 * 1024 ** 3, free_bytes: 5.5 * 1024 ** 3 } });
    expect(await screen.findByText("Not enough free disk")).toBeTruthy();
    expect(screen.getByText(/It needs 6 GiB; 5.5 GiB is free/)).toBeTruthy();
  });

  it("says why the ensemble cannot be frozen and keeps both freeze buttons off", async () => {
    routes["GET /api/studies/candidates?mode=starfull"] = () => ({ body: { ...CANDIDATES, can_freeze: false, blocking: "the test cubes predate members 203–205" } });
    open();
    expect(await screen.findByText("the test cubes predate members 203–205")).toBeTruthy();
    expect((within(dialog()).getByRole("button", { name: "Choose fields…" }) as HTMLButtonElement).disabled).toBe(true);
    expect((within(dialog()).getByRole("button", { name: "Freeze without fields" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("disables picking while FASRC is offline and says so", async () => {
    routes["GET /api/studies/candidates?mode=starfull"] = () => ({ body: { ...CANDIDATES, fasrc_connected: false, fields_note: "FASRC is not connected: fields are stored on holylabs." } });
    open();
    fireEvent.click(await within(dialog()).findByRole("button", { name: "Choose fields…" }));
    expect(screen.getByText(/fields are stored on holylabs\./)).toBeTruthy();
    const box = screen.getByRole("checkbox", { name: "test · idx 0" }) as HTMLInputElement;
    expect(box.disabled).toBe(true);
    expect(document.getElementById(box.getAttribute("aria-describedby") ?? "")?.textContent).toMatch(/You can still freeze without fields/);
  });

  it("moves focus with the steps: Enter on Choose fields… lands in the gallery, Continue on the Name input", async () => {
    open();
    const choose = await within(dialog()).findByRole("button", { name: "Choose fields…" });
    choose.focus();
    fireEvent.keyDown(choose, { key: "Enter" });
    fireEvent.click(choose);                     // a button's Enter is a click
    await waitFor(() => expect(document.activeElement).toBe(screen.getByRole("checkbox", { name: "test · idx 0" })));
    click("Continue without fields");
    await waitFor(() => expect(document.activeElement).toBe(within(dialog()).getByRole("textbox", { name: "Name" })));
    click("Back");
    await waitFor(() => expect(dialog().contains(document.activeElement)).toBe(true));
    expect(within(dialog()).getByRole("button", { name: "Choose fields…" })).toBeTruthy();
  });

  it("always freezes the starfull ensemble: no regime choice", async () => {
    open();
    await within(dialog()).findByText("Combiner comparison");
    expect(within(dialog()).queryByRole("radio", { name: "starless" })).toBeNull();
    expect(within(dialog()).queryByRole("radiogroup", { name: "Regime" })).toBeNull();
    expect(calls.every((c) => !c.url.includes("starless"))).toBe(true);
    fireEvent.click(within(dialog()).getByRole("button", { name: "Freeze without fields" }));
    fireEvent.change(within(dialog()).getByRole("textbox", { name: "Name" }), { target: { value: "Starfull" } });
    click("Freeze");
    await waitFor(() => expect(posts()).toHaveLength(1));
    expect(posts()[0].body).toMatchObject({ mode: "starfull", fields: "" });
  });

  it("re-attaches to a running freeze and keeps the study link after the job finishes", async () => {
    const saved = JOBS_FEED_TIMING.detailMinMs;
    JOBS_FEED_TIMING.detailMinMs = 10;
    try {
      const running: Job = { job_id: "frz00009", label: "study freeze: Earlier", kind: "study-freeze", status: "running", duration: 1, error: null,
        log: "", log_truncated: false, progress: { current: 1, total: 5, pct: 20, label: "numbers · members" } as Job["progress"] };
      let reads = 0;
      routes["GET /api/jobs/frz00009"] = () => ({ body: ++reads < 3 ? running : { ...running, status: "done", result: {} } });
      let listed = 0;
      routes["GET /api/studies"] = () => ({ body: { ok: true, studies: [], freezing: ++listed === 1 ? { job_id: "frz00009", study_id: "20260928-090000-earlier" } : null } });
      act(() => {
        useJobsStore.getState().upsert(running);
        useJobsStore.getState().register("frz00009", FREEZE_JOB_KEY);
      });
      open();
      expect(await screen.findByText("Freezing the study")).toBeTruthy();
      const link = await within(dialog()).findByRole("link", { name: "Open the study" });
      expect(link.getAttribute("href")).toBe("/figures/studies?study=20260928-090000-earlier");
      expect(await screen.findByText("Study frozen")).toBeTruthy();
      expect(within(dialog()).getByRole("link", { name: "Open the study" }).getAttribute("href")).toBe("/figures/studies?study=20260928-090000-earlier");
      expect(calls.filter((c) => c.url.startsWith("/api/studies/candidates"))).toHaveLength(0);
      expect(posts()).toHaveLength(0);
    } finally {
      JOBS_FEED_TIMING.detailMinMs = saved;
    }
  });
});
