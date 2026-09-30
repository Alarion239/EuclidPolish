/* The "Recommended from N past runs" callout against a mocked advisor: the
 * debounced JSON POST (params + resources), the change list, Apply handing
 * back the recommended resources, and the quiet no-history / error states. */
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ResourceAdvice } from "./ResourceAdvice";
import type { Recommendation } from "./resourceAdviceModel";

type Call = { url: string; method: string; json: unknown; type: string | null };
let calls: Call[];
let reply: () => { status?: number; body: unknown };

const REC: Recommendation = {
  ok: true, step_id: "synthetic_generate", available: true, confidence: "medium",
  resources: { n_cpus: "20", n_gpus: "0", memory: "36G", time_limit: "3:15:00" },
  current: { n_cpus: "20", n_gpus: "0", memory: "32G", time_limit: "2:00:00" },
  changes: [
    { field: "memory", current: "32G", recommended: "36G", reason: "p90 peak 29.8 GB of 32 GB over 4 runs; 1 OOM at 30 GB" },
    { field: "time_limit", current: "2:00:00", recommended: "3:15:00", reason: "p90 2.4 s per image × 4,000 images × 1.2" },
  ],
  basis: { level: "similar", level_label: "same splits and image size", n_runs: 4, jobids: ["11", "12", "13", "14"],
    units: 4000, units_label: "images", rate_s_per_unit: 2.4 },
  notes: ["Time from 4,000 images at the p90 rate."], warnings: ["1 TIMEOUT run would still time out."],
};
const RES = { n_cpus: "20", n_gpus: "0", memory: "32G", time_limit: "2:00:00" };
const PARAMS = { regenerate_splits: "validate,test", n_valid: "2000", n_test: "2000" };

beforeEach(() => {
  calls = [];
  reply = () => ({ body: REC });
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = new URL(String(input), "http://localhost");
    const headers = new Headers(init.headers);
    calls.push({ url: url.pathname, method: init.method ?? "GET", type: headers.get("Content-Type"),
      json: typeof init.body === "string" ? JSON.parse(init.body) : null });
    const r = reply();
    return new Response(JSON.stringify(r.body), { status: r.status ?? 200 });
  }));
});
afterEach(() => { vi.unstubAllGlobals(); });

const show = (el: ReactElement) => render(
  <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>{el}</MemoryRouter>,
);

describe("ResourceAdvice", () => {
  it("asks the advisor with the params and resources as JSON, and lists what it would change", async () => {
    show(<ResourceAdvice stepId="synthetic_generate" params={PARAMS} resources={RES} onApply={() => {}} delay={0} />);
    expect(await screen.findByText("Recommended from 4 past runs (same splits and image size) · medium confidence")).toBeTruthy();
    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({ url: "/api/fasrc/resources/synthetic_generate/recommend", method: "POST", type: "application/json" });
    expect(calls[0].json).toEqual({ params: PARAMS, resources: RES });
    expect(screen.getByText("36G")).toBeTruthy();
    expect(screen.getByText("3:15:00")).toBeTruthy();
    expect(screen.getByText(/1 OOM at 30 GB/)).toBeTruthy();
    // notes and warnings are collapsed behind one disclosure
    expect(screen.getByText("1 warning · 1 note").closest("details")?.open).toBe(false);
    expect(screen.getByRole("link", { name: "Open usage" }).getAttribute("href")).toBe("/runs/resources?step=synthetic_generate");
  });

  it("Apply hands back the current resources with the recommended values in place", async () => {
    const onApply = vi.fn();
    show(<ResourceAdvice stepId="synthetic_generate" params={PARAMS} resources={RES} onApply={onApply} delay={0} />);
    fireEvent.click(await screen.findByRole("button", { name: "Apply" }));
    expect(onApply).toHaveBeenCalledWith({ n_cpus: "20", n_gpus: "0", memory: "36G", time_limit: "3:15:00" });
  });

  it("offers only the fields the host edits, labelled per array task", async () => {
    const onApply = vi.fn();
    show(<ResourceAdvice stepId="ensemble_train" params={{}} resources={RES} fields={["n_cpus", "memory"]} perTask="member"
      onApply={onApply} delay={0} />);
    expect(await screen.findByText("Memory / member")).toBeTruthy();
    expect(screen.queryByText("3:15:00")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Apply" }));
    expect(onApply).toHaveBeenCalledWith({ ...RES, memory: "36G" });
  });

  it("disables Apply when the form already asks for the recommendation", async () => {
    reply = () => ({ body: { ...REC, changes: [], warnings: [], notes: [] } });
    show(<ResourceAdvice stepId="synthetic_generate" params={PARAMS} resources={RES} onApply={() => {}} delay={0} />);
    expect(await screen.findByText("The form already asks for this.")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Apply" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("disables Apply while an edit is re-asked: the answer on screen is for the previous plan", async () => {
    const onApply = vi.fn();
    const at = (params: Record<string, string>) => (
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <ResourceAdvice stepId="synthetic_generate" params={params} resources={RES} onApply={onApply} delay={30} />
      </MemoryRouter>
    );
    const view = render(at(PARAMS));
    const apply = () => screen.getByRole("button", { name: "Apply" }) as HTMLButtonElement;
    await screen.findByRole("button", { name: "Apply" });
    expect(apply().disabled).toBe(false);
    // validate+test → train: until the new answer lands, Apply would write the old plan's time.
    view.rerender(at({ ...PARAMS, regenerate_splits: "train" }));
    expect(apply().disabled).toBe(true);
    fireEvent.click(apply());
    expect(onApply).not.toHaveBeenCalled();
    await waitFor(() => expect(calls).toHaveLength(2));
    await waitFor(() => expect(apply().disabled).toBe(false));
    expect((calls[1].json as { params: Record<string, string> }).params.regenerate_splits).toBe("train");
    fireEvent.click(apply());
    expect(onApply).toHaveBeenCalledTimes(1);
  });

  it("says quietly that there is no history (available: false, or a step the ledger never saw)", async () => {
    reply = () => ({ body: { ...REC, available: false, changes: [], basis: null } });
    const view = show(<ResourceAdvice stepId="tng_grid" params={{}} resources={RES} onApply={() => {}} delay={0} />);
    expect(await screen.findByText("No finished run of this step to recommend resources from yet.")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Apply" })).toBeNull();
    view.unmount();
    reply = () => ({ status: 404, body: { ok: false, error: "no runs of step 'new_step'" } });
    show(<ResourceAdvice stepId="new_step" params={{}} resources={RES} onApply={() => {}} delay={0} />);
    expect(await screen.findByText("No finished run of this step to recommend resources from yet.")).toBeTruthy();
  });

  it("keeps an error to one quiet line", async () => {
    reply = () => ({ status: 500, body: { ok: false, error: "ledger unreadable" } });
    show(<ResourceAdvice stepId="synthetic_generate" params={PARAMS} resources={RES} onApply={() => {}} delay={0} />);
    expect(await screen.findByText(/Past-run advice is unavailable/)).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("debounces: a burst of edits asks once, with the last values", async () => {
    const view = show(<ResourceAdvice stepId="synthetic_generate" params={PARAMS} resources={RES} onApply={() => {}} delay={40} />);
    for (const memory of ["33G", "34G", "35G"]) {
      view.rerender(
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <ResourceAdvice stepId="synthetic_generate" params={PARAMS} resources={{ ...RES, memory }} onApply={() => {}} delay={40} />
        </MemoryRouter>,
      );
    }
    await screen.findByText(/Recommended from 4 past runs/);
    await waitFor(() => expect(calls).toHaveLength(1));
    expect((calls[0].json as { resources: { memory: string } }).resources.memory).toBe("35G");
  });
});
