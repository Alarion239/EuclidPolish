/* Notebook workspace against a mocked Flask: the Log (campaign bar, a
 * prefilled entry from a page's "Log to notebook", the notebook newest
 * first), Backups (filter chips with counts, ⏱ time travel on every kind,
 * the archived campaigns) and Sandboxes (confirmed removal). */
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import type { ReactElement } from "react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { queryClient } from "../../api/query";
import { resetConfirm } from "../../ui";
import Backups from "./tabs/Backups";
import Log from "./tabs/Log";
import Sandboxes from "./tabs/Sandboxes";

type Reply = { status?: number; body: unknown };
type Call = { url: string; method: string; form: Record<string, string> };
let routes: Record<string, (form: Call["form"]) => Reply>;
let calls: Call[];
let location = "";

const STATE = {
  active: { title: "gate-sweep", slug: "gate-sweep", created_at: "2026-09-01T00:00:00Z", created_commit: { short: "fff0000" } },
  archived: [{ title: "old run", slug: "old-run", _dir: "old-run-2", saved_commit: { short: "abc1234", hash: "abc1234ffff" }, models: [] }],
  backups: {
    models: [{ name: "member_196", kind: "model", size_bytes: 1024, commit: { short: "aaa1111", hash: "aaa1111" } }],
    fits: [{ name: "sr.fits", kind: "fits", size_bytes: 2048, commit: { short: "bbb2222", hash: "bbb2222" } },
      { name: "lr.fits", kind: "fits", size_bytes: 2048, commit: { short: "bbb2222", hash: "bbb2222" } }],
    images: [],
  },
  jobs_count: 0, unassigned_count: 0, log_md: "# gate-sweep\n\n## 2026-09-01T00:00:00Z\n\nA **result**.",
  ssh_connected: false, tracking_dir: "/Users/x/EuclidPolish/tracking",
  sandboxes: [{ short: "tt01", source: { kind: "campaign", slug: "old-run" }, source_label: "campaign old-run", running: true }],
};

beforeEach(() => {
  calls = [];
  routes = {
    "GET /api/tracking/state": () => ({ body: STATE }),
    "POST /api/tracking/timetravel/remove": () => ({ body: { ok: true } }),
    "POST /api/tracking/save": () => ({ body: { ok: true } }),
    "POST /api/tracking/new": () => ({ body: { ok: true } }),
    "POST /api/tracking/log": () => ({ body: { ok: true } }),
    "POST /api/tracking/timetravel/restore": () => ({ body: { ok: true, short: "abc1234", url: "http://127.0.0.1:8766/", warning: null } }),
  };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = new URL(String(input), "http://localhost");
    const method = init.method ?? "GET";
    const form: Record<string, string> = {};
    if (init.body instanceof FormData) init.body.forEach((v, k) => { form[k] = String(v); });
    calls.push({ url: `${url.pathname}${url.search}`, method, form });
    const r = routes[`${method} ${url.pathname}`]?.(form) ?? { status: 404, body: { ok: false, error: `no route ${url.pathname}` } };
    return new Response(JSON.stringify(r.body), { status: r.status ?? 200 });
  }));
  queryClient.clear();
});
afterEach(() => {
  act(() => resetConfirm());
  queryClient.clear();
  vi.unstubAllGlobals();
});

function LocationProbe() {
  const loc = useLocation();
  location = `${loc.pathname}${loc.search}`;
  return null;
}
const params = () => new URLSearchParams(location.split("?")[1] ?? "");
const show = (el: ReactElement, url = "/notebook/log") => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={[url]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>{el}<LocationProbe /></MemoryRouter>
  </QueryClientProvider>,
);
const posts = (path: string) => calls.filter((c) => c.method === "POST" && c.url === path);
const answer = async (title: RegExp | string, button: string) => {
  const dlg = await screen.findByRole("alertdialog", { name: title });
  fireEvent.click(within(dlg).getByRole("button", { name: button }));
  await waitFor(() => expect(screen.queryByRole("alertdialog", { name: title })).toBeNull());
  return dlg;
};

describe("Notebook › Log", () => {
  it("renders the notebook as markdown under the campaign bar", async () => {
    show(<Log />);
    expect((await screen.findByText("result")).tagName).toBe("STRONG");
    const bar = screen.getByRole("toolbar", { name: "Campaign" });
    expect(within(bar).getByText("gate-sweep")).toBeTruthy();
    expect(within(bar).getByText("fff0000")).toBeTruthy();
    for (const name of ["Back up…", "Push", "New campaign…"]) expect(within(bar).getByRole("button", { name })).toBeTruthy();
    expect(within(bar).queryByRole("button", { name: "Save snapshot" })).toBeNull();     // secondary: in the menu
  });

  it("opens the notebook newest first, with a jump-to-day menu", async () => {
    routes["GET /api/tracking/state"] = () => ({ body: { ...STATE,
      log_md: "# gate-sweep\n\n## 2026-07-02T02:37:41Z\n\nold one\n\n## 2026-09-21T01:36:49Z\n\nmiddle\n\n## 2026-09-21T14:42:29Z\n\nnewest",
    } });
    show(<Log />);
    await screen.findByText("newest");
    const doc = document.querySelector(".nb-notebook__doc") as HTMLElement;
    const order = [...doc.querySelectorAll("h3, h2")].map((h) => h.textContent).filter((t) => t?.startsWith("2026"));
    expect(order).toEqual(["2026-09-21T14:42:29Z", "2026-09-21T01:36:49Z", "2026-07-02T02:37:41Z"]);
    expect(screen.getByText("3 entries, 2026-07-02 to 2026-09-21")).toBeTruthy();
    const jump = screen.getByRole("combobox", { name: "Jump to a day" });
    expect([...jump.querySelectorAll("option")].map((o) => o.textContent)).toEqual(["Jump to a day…", "Sep 21, 2026", "Jul 2, 2026"]);
    fireEvent.click(screen.getByRole("radio", { name: "Oldest first" }));
    await waitFor(() => expect([...doc.querySelectorAll("h3, h2")].map((h) => h.textContent).filter((t) => t?.startsWith("2026"))[0])
      .toBe("2026-07-02T02:37:41Z"));
  });

  it("lands with a prefilled entry from a page's Log to notebook, adds it, then drops it from the URL", async () => {
    show(<Log />, `/notebook/log?${new URLSearchParams({ entry: "Gate v7: ∫PSNR 61.02 dB", from: "Models › Combiner" })}`);
    const box = await screen.findByRole("textbox", { name: "New notebook entry" }) as HTMLTextAreaElement;
    expect(box.value).toBe("Gate v7: ∫PSNR 61.02 dB");
    expect(screen.getByText("Prefilled from Models › Combiner: edit it, then add it")).toBeTruthy();
    fireEvent.change(box, { target: { value: "Gate v7: ∫PSNR 61.02 dB, promoted" } });
    fireEvent.click(screen.getByRole("button", { name: "Add entry" }));
    await waitFor(() => expect(posts("/api/tracking/log")[0]?.form).toEqual({ text: "Gate v7: ∫PSNR 61.02 dB, promoted", mode: "append" }));
    await waitFor(() => expect(params().get("entry")).toBeNull());
    expect(params().get("from")).toBeNull();
  });

  it("saves a snapshot from the bar's menu, after confirming", async () => {
    show(<Log />);
    const bar = await screen.findByRole("toolbar", { name: "Campaign" });
    await within(bar).findByText("gate-sweep");
    fireEvent.pointerDown(within(bar).getByRole("button", { name: "More campaign actions" }), { button: 0 });
    fireEvent.click(await screen.findByRole("menuitem", { name: "Save snapshot" }));
    await answer("Save “gate-sweep”?", "Save snapshot");
    await waitFor(() => expect(posts("/api/tracking/save")).toHaveLength(1));
  });

  it("starts a new campaign by saving the active one first", async () => {
    show(<Log />);
    fireEvent.click(await screen.findByRole("button", { name: "New campaign…" }));
    const dlg = await screen.findByRole("dialog", { name: "New campaign" });
    expect(dlg.textContent).toContain("Saves “gate-sweep” first");
    fireEvent.change(within(dlg).getByRole("textbox", { name: "Title" }), { target: { value: "next" } });
    fireEvent.click(within(dlg).getByRole("button", { name: "Save “gate-sweep” and start" }));
    await waitFor(() => expect(posts("/api/tracking/new")).toHaveLength(1));
    expect(calls.findIndex((c) => c.url === "/api/tracking/save")).toBeLessThan(calls.findIndex((c) => c.url === "/api/tracking/new"));
    expect(posts("/api/tracking/new")[0].form).toEqual({ title: "next", description: "" });
  });
});

describe("Notebook › Backups", () => {
  it("filters by kind with counts on the chips and time-travels a FITS backup to its commit", async () => {
    show(<Backups />, "/notebook/backups");
    const chips = await screen.findByRole("radiogroup", { name: "Show" });
    await waitFor(() => expect(within(chips).getByRole("radio", { name: "FITS · 2" })).toBeTruthy());
    expect(within(chips).getByRole("radio", { name: "Archived campaigns · 1" })).toBeTruthy();
    expect(await screen.findByText("member_196")).toBeTruthy();
    fireEvent.click(within(chips).getByRole("radio", { name: "FITS · 2" }));
    await waitFor(() => expect(params().get("show")).toBe("fits"));
    expect((await screen.findByRole("link", { name: "Open sr.fits in Files" })).getAttribute("href"))
      .toBe("/files?fits=tracking%2Fcurrent%2Ffits%2Fsr.fits");
    fireEvent.click(screen.getByRole("button", { name: "Time-travel to sr.fits" }));
    const dlg = await screen.findByRole("dialog");
    expect(dlg.textContent).toContain("bbb2222");
    fireEvent.click(within(dlg).getByRole("button", { name: "Start sandbox" }));
    await waitFor(() => expect(posts("/api/tracking/timetravel/restore")[0]?.form)
      .toEqual({ campaign: "current", backup: "sr.fits", kind: "fits", remote: "0" }));
  });

  it("lists the archived campaigns under ?show=campaigns and time-travels one by its dir", async () => {
    show(<Backups />, "/notebook/backups?show=campaigns");
    fireEvent.click(await screen.findByRole("button", { name: "Time-travel to old run" }));
    const dlg = await screen.findByRole("dialog");
    expect(dlg.textContent).toContain("abc1234");
    fireEvent.click(within(dlg).getByRole("button", { name: "Start sandbox" }));
    await waitFor(() => expect(posts("/api/tracking/timetravel/restore")[0]?.form).toEqual({ campaign: "old-run-2", remote: "0" }));
    expect(await screen.findByText("Sandbox abc1234 is running")).toBeTruthy();
  });

  it("reads the old ?bk= kind", async () => {
    show(<Backups />, "/notebook/backups?bk=fits");
    expect(await screen.findByText("sr.fits")).toBeTruthy();
  });
});

describe("Notebook › Sandboxes", () => {
  it("renders the sandbox source_label (source is an object) and confirms removal", async () => {
    show(<Sandboxes />, "/notebook/sandboxes");
    expect(await screen.findByText("campaign old-run")).toBeTruthy();
    expect(screen.getByText(/1 of 1 running/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Remove" }));
    await answer("Remove sandbox tt01?", "Cancel");
    expect(posts("/api/tracking/timetravel/remove")).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "Remove" }));
    await answer("Remove sandbox tt01?", "Remove");
    await waitFor(() => expect(posts("/api/tracking/timetravel/remove")[0]?.form).toEqual({ short: "tt01" }));
  });
});
