/* Sky › Compare against a mocked backend: it opens on the newest comparison
 * (sentence, viewer with the Δm footer, band strip, pivot), the scope moves
 * the viewer, the metric definitions live in one popover (`?defs=1`), Log to
 * notebook, the history with its headline result, and the New comparison
 * drawer (set chips, pasted refs, a confirmed Run). */
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Compare from "../tabs/Compare";
import { RECORD, answer, installBackend, show, teardownBackend, type Post, type Routes } from "../testing/backend";

type MockViewerProps = {
  collection: string; initialId?: string; tiers?: string[]; params?: Record<string, string>; nav?: boolean;
  onReady?: (api: unknown) => void; onState?: (s: unknown) => void;
};
const viewerMock = vi.hoisted(() => ({ id: null as string | null, goTo: [] as string[], mounts: 0 }));
vi.mock("../../../viewer", async () => {
  const { useEffect } = await import("react");
  return {
    ImageViewer: (p: MockViewerProps) => {
      useEffect(() => {
        viewerMock.mounts += 1;
        viewerMock.id = p.initialId ?? null;
        const api = {
          getState: () => ({ id: viewerMock.id, tiers: p.tiers }),
          goToId: async (id: string) => { viewerMock.goTo.push(id); viewerMock.id = id; p.onState?.({ id, tiers: p.tiers }); return true; },
          setTiers: () => undefined,
        };
        p.onReady?.(api);
        return () => p.onReady?.(null);
        // eslint-disable-next-line react-hooks/exhaustive-deps
      }, []);
      return (
        <div data-testid="viewer" data-nav={String(p.nav ?? true)}>
          {p.collection}|{p.initialId ?? ""}|{(p.tiers ?? []).join(",")}|{p.params?.models ?? ""}
        </div>
      );
    },
  };
});

let routes: Routes;
let posts: Post[];

beforeEach(() => {
  ({ routes, posts } = installBackend());
  Object.assign(viewerMock, { id: null, goTo: [], mounts: 0 });
});
afterEach(() => teardownBackend());

const sentence = () => screen.getByText(/leaves holes in/).closest("p")?.textContent;

describe("Sky › Compare", () => {
  it("opens on the newest comparison: one sentence, the viewer with its Δm footer, the pivot; history below", async () => {
    const older = { ...RECORD, id: "20260901-000000-000000", label: "older", created: "2026-09-01T00:00:00Z" };
    routes["GET /api/experiments"] = () => ({ body: { experiments: [older, RECORD] } });
    const { container } = show(<Compare />, "/sky/compare");
    await waitFor(() => expect(sentence()).toBe(
      "On 2 tiles, production leaves holes in 8.3 % of the bright pixels of its worst band (J), against 30.5 % for the member mean (VIS)."));
    expect(screen.getByText("Models on real tiles, no truth.")).toBeTruthy();
    const viewer = screen.getByTestId("viewer");
    expect(viewer.textContent).toBe("real|f200w-0001|lr,m:production,m:mean,m:member:member_1,jwst|production,mean,member:member_1");
    expect(viewer.dataset.nav).toBe("false");
    // Δm per model against the LR on the shown tile, warned beyond 0.1 mag
    const delta = container.querySelector(".cmp-delta") as HTMLElement;
    expect(delta.textContent).toBe("VIS, Δm vs LR on nexus/f200w-0001: production +0.04 · member mean +1.31");
    expect(delta.querySelector("[data-warn]")?.textContent).toContain("member mean");
    // models × bands for one metric, the best per band bold
    const pivot = screen.getByRole("grid", { name: "Hole % per model and band" });
    expect(within(pivot).getAllByRole("columnheader").map((h) => h.textContent)).toEqual(["Model", "VIS (%)", "Y (%)", "J (%)", "H (%)"]);
    expect(within(pivot).getByText("2.5").tagName).toBe("STRONG");            // member 1 has the fewest VIS holes
    // the history with its headline result, below the comparison
    const history = screen.getByRole("grid", { name: "Comparisons" });
    expect(within(history).getAllByText("production 8.3 % (J) · member mean 30.5 % (VIS)")).toHaveLength(2);
    expect(viewer.compareDocumentPosition(history) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    // the URL is untouched, the drawer folded, nothing posted
    expect(screen.getByTestId("loc").textContent).toBe("/sky/compare");
    expect(screen.getByRole("button", { name: /New comparison/, expanded: false })).toBeTruthy();
    expect(posts).toEqual([]);
  });

  it("keeps the viewer on the scope's tile; per tile, the sentence and pivot are that tile's", async () => {
    show(<Compare />, "/sky/compare?exp=20260926-101010-abcdef");
    await screen.findByTestId("viewer");
    fireEvent.change(screen.getByRole("combobox", { name: "Scope" }), { target: { value: "poster/p1" } });
    // another source: the viewer remounts on that tile (the collection is per source)
    await waitFor(() => expect(screen.getByTestId("viewer").textContent).toMatch(/^real\|p1\|lr,m:production/));
    expect(screen.getByTestId("loc").textContent).toContain("scope=poster%2Fp1");
    expect(sentence()).toBe("On poster/p1, production leaves holes in 7.0 % of the bright pixels of its worst band (VIS).");
    expect(screen.getByLabelText("Gate core weights")).toBeTruthy();
  });

  it("moves the viewer between tiles of one source without remounting it", async () => {
    const two = { ...RECORD, id: "x1", tiles: ["nexus/f200w-0001", "nexus/f200w-0002"], results: {} };
    routes["GET /api/experiments"] = () => ({ body: { experiments: [two] } });
    routes["GET /api/experiments/x1"] = () => ({ body: two });
    show(<Compare />, "/sky/compare?exp=x1");
    expect((await screen.findByTestId("viewer")).textContent).toMatch(/^real\|f200w-0001\|/);
    fireEvent.change(screen.getByRole("combobox", { name: "Scope" }), { target: { value: "nexus/f200w-0002" } });
    await waitFor(() => expect(viewerMock.goTo).toEqual(["f200w-0002"]));
    expect(viewerMock.mounts).toBe(1);
    fireEvent.change(screen.getByRole("combobox", { name: "Scope" }), { target: { value: "pooled" } });
    await waitFor(() => expect(viewerMock.goTo).toEqual(["f200w-0002", "f200w-0001"]));   // pooled shows the first tile
  });

  it("pivots another metric from the metric select", async () => {
    show(<Compare />, "/sky/compare?exp=20260926-101010-abcdef&metric=flux_ratio");
    const pivot = await screen.findByRole("grid", { name: "Flux SR/LR per model and band" });
    expect(within(pivot).getByText("0.9900")).toBeTruthy();
    fireEvent.change(screen.getByRole("combobox", { name: "Metric" }), { target: { value: "median_R" } });
    expect(await screen.findByRole("grid", { name: "Median R per model and band" })).toBeTruthy();
  });

  it("keeps the ONE metric-definitions text in its popover (?defs=1 opens it)", async () => {
    show(<Compare />, "/sky/compare?defs=1");
    expect(await screen.findByText(/SR pixels under the brightest 1 % of LR pixels/)).toBeTruthy();
    expect(screen.getByText(/Median enclosed-flux ratio over the peaks/)).toBeTruthy();
  });

  it("logs the comparison on Notebook › Log, appending nothing itself", async () => {
    show(<Compare />, "/sky/compare?exp=20260926-101010-abcdef");
    fireEvent.click(await screen.findByRole("button", { name: "Log to notebook" }));
    await waitFor(() => expect(screen.getByTestId("loc").textContent?.startsWith("/notebook/log?")).toBe(true));
    const q = new URLSearchParams((screen.getByTestId("loc").textContent ?? "").split("?")[1]);
    expect(q.get("from")).toBe("Sky › Compare");
    expect(q.get("entry")).toContain("**Real-data experiment `20260926-101010-abcdef`** — core check");
    expect(q.get("entry")).toContain("| `production` | 5.5");
    expect(posts).toHaveLength(0);
  });
});

describe("Sky › Compare › New comparison", () => {
  it("opens with handed-over tiles, states the cost and runs only after a confirm", async () => {
    routes["POST /api/experiments"] = () => ({ body: { ok: true, job_id: "j2", experiment_id: "20260926-121212-111111", skipped: {} } });
    routes["GET /api/experiments/20260926-121212-111111"] = () => ({ body: { ...RECORD, id: "20260926-121212-111111", status: "running" } });
    show(<Compare />, "/sky/compare?tiles=nexus%2Ff200w-0001%2Cposter%2Fp1");
    const list = await screen.findByLabelText("Comparison tiles");
    expect(within(list).getByText("nexus/f200w-0001")).toBeTruthy();
    expect(within(list).getByText("poster/p1")).toBeTruthy();
    await screen.findByRole("checkbox", { name: /production/ });
    expect(screen.getByText(/4 outputs \(2 models on 2 tiles\)\. Needs 2 member SRs per tile: at most 4 member inferences/)).toBeTruthy();
    // tiles handed over: the form is the point, no comparison opens by itself
    expect(screen.queryByTestId("viewer")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Run 2 models on 2 tiles…" }));
    const dlg = await answer(/Run 2 models on 2 tiles/, "Run");
    expect(dlg.textContent).toContain("4 outputs (2 models on 2 tiles).");
    await waitFor(() => expect(posts[0]).toEqual({
      url: "/api/experiments", form: { tiles: "nexus/f200w-0001,poster/p1", models: "production,mean" },
    }));
    await waitFor(() => expect(screen.getByTestId("loc").textContent).toContain("exp=20260926-121212-111111"));
  });

  it("adds a target set's tiles from its chip (poster, lens candidates, Q1 galaxies)", async () => {
    show(<Compare />, "/sky/compare?new=1");
    fireEvent.click(await screen.findByRole("button", { name: "Lens candidates 2" }));
    const list = await screen.findByLabelText("Comparison tiles");
    expect(within(list).getAllByText(/^eval\//).map((el) => el.textContent)).toEqual(["eval/lensA", "eval/lensB"]);
    fireEvent.click(screen.getByRole("button", { name: "Poster galaxy 1" }));
    expect(within(list).getByText("poster/p1")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Lens candidates 2" }));           // a second click takes them out
    await waitFor(() => expect(within(list).queryByText("eval/lensA")).toBeNull());
    expect(screen.getByTestId("loc").textContent).toContain("tiles=poster%2Fp1");
    // the NEXUS chip is the shared selection's NEXUS tiles: none selected, nothing to add
    expect((screen.getByRole("button", { name: "NEXUS selection" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("before any comparison: says so and opens the drawer", async () => {
    routes["GET /api/experiments"] = () => ({ body: { experiments: [] } });
    show(<Compare />, "/sky/compare");
    expect(await screen.findByText("No comparison yet")).toBeTruthy();
    expect(await screen.findByLabelText("Target sets")).toBeTruthy();
  });
});
