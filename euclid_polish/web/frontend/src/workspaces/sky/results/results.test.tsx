/* The Sky cards against a mocked backend (C9 + /api/evaluation): the
 * inspector registration, the model picker, the one real-tile card (in the
 * console-regrouping order) and the comparison card. The tabs have their own
 * suites (targets/targets.test.tsx, compare/compare.test.tsx). */
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { Suspense } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobsStore } from "../../../api/jobs";
import { registerInspector, useInspectorRegistry } from "../../../app/inspector";
import { useInspector } from "../../../state/inspector";
import { useSelection } from "../../../state/selection";
import { LAST_TILE_KEY } from "../atlas/home";
import { CARD, POSTER, answer, installBackend, show, teardownBackend, type Post, type Routes } from "../testing/backend";
import ExperimentInspector from "./ExperimentInspector";
import { ModelPicker } from "./ModelPicker";
import RealTileInspector from "./RealTileInspector";
import "./register";

type MockViewerProps = {
  collection: string; initialId?: string; tiers?: string[]; params?: Record<string, string>; nav?: boolean;
  onReady?: (api: unknown) => void; onState?: (s: unknown) => void;
};
/* The viewer engine is mocked: it mounts on initialId, reports its object id
 * through getState/onState and records every goToId and setTiers. */
const viewerMock = vi.hoisted(() => ({ id: null as string | null, goTo: [] as string[], tiers: [] as string[][], mounts: 0 }));
vi.mock("../../../viewer", async () => {
  const { useEffect } = await import("react");
  return {
    ImageViewer: (p: MockViewerProps) => {
      useEffect(() => {
        viewerMock.mounts += 1;
        viewerMock.id = p.initialId ?? null;
        const api = {
          getState: () => ({ id: viewerMock.id }),
          goToId: async (id: string) => { viewerMock.goTo.push(id); viewerMock.id = id; p.onState?.({ id }); return true; },
          setTiers: (t: string[]) => { viewerMock.tiers.push(t); },
          setFocus: () => undefined,
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
  routes["GET /api/real/nexus/f200w-0001"] = () => ({ body: CARD });
  Object.assign(viewerMock, { id: null, goTo: [], tiers: [], mounts: 0 });
  localStorage.removeItem(LAST_TILE_KEY);
});
afterEach(() => teardownBackend());

describe("inspector registration", () => {
  it("registers realtile and experiment on import", () => {
    const kinds = useInspectorRegistry.getState().kinds;
    expect(kinds.realtile?.title).toBeTypeOf("function");
    const title = kinds.experiment?.title;
    expect(typeof title === "function" ? title("x1") : title).toBe("Comparison x1");
  });

  it("turns a realtile: target into tile: in place (one card, one kind label)", async () => {
    const off = registerInspector("tile", () => <p>tile card</p>, { title: (id) => `Tile ${id}` });
    try {
      act(() => useInspector.getState().show({ kind: "realtile", id: "nexus/f200w-0040" }));
      const Alias = useInspectorRegistry.getState().kinds.realtile.Component;
      render(<Suspense fallback={null}><Alias id="nexus/f200w-0040" /></Suspense>);   // the inspector panel provides this boundary
      await waitFor(() => expect(useInspector.getState().current).toEqual({ kind: "tile", id: "nexus/f200w-0040" }));
      expect(useInspector.getState().back).toEqual([]);                  // replaced, not a history step
    } finally { off(); }
  });
});

describe("model picker", () => {
  it("disables unavailable specs with the reason and offers quick picks", async () => {
    const seen: string[][] = [];
    show(<ModelPicker value={[]} onChange={(v) => seen.push(v)} />);
    const old = await screen.findByRole("checkbox", { name: /gate:old/ });
    expect((old as HTMLInputElement).disabled).toBe(true);
    expect(screen.getByText("fitted for 20 archived members")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Production + mean" }));
    expect(seen.at(-1)).toEqual(["production", "mean"]);
    fireEvent.click(screen.getByRole("button", { name: "+ all members" }));
    expect(seen.at(-1)).toEqual(["member:member_1", "member:member_2"]);
    // a pruned gate says how many members it reads, not how many it was fitted on
    expect(screen.getByText(/6 of 20 members/)).toBeTruthy();
    expect(screen.getByText(/^Runs all 2 members/)).toBeTruthy();
  });

  it("keeps the legacy RBF combiner behind a toggle", async () => {
    show(<ModelPicker value={[]} onChange={() => undefined} />);
    await screen.findByRole("checkbox", { name: /gate:old/ });
    expect(screen.queryByRole("checkbox", { name: /rbf/ })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Legacy RBF" }));
    expect(screen.getByRole("group", { name: "Legacy RBF combiner" })).toBeTruthy();
    expect(screen.getByRole("checkbox", { name: /rbf/ })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Hide legacy RBF" }));
    expect(screen.queryByRole("checkbox", { name: /rbf/ })).toBeNull();
  });

  it("shows the legacy RBF while it is picked", async () => {
    show(<ModelPicker value={["rbf"]} onChange={() => undefined} />);
    expect(await screen.findByRole("checkbox", { name: /rbf/ })).toBeTruthy();
  });

  it("says how many members a pruned production runs", async () => {
    const reads = Array.from({ length: 20 }, (_, i) => `${170 + i}·psnr`);
    routes["GET /api/models"] = () => ({ body: { regime: "starfull", models: [
      { spec: "production", kind: "production", label: "Production · spatial gate", available: true,
        n_members: 20, n_fitted: 30, reads, details: { prune_threshold: 0.005, mix_space: "linear" } },
    ] } });
    show(<ModelPicker value={[]} onChange={() => undefined} />);
    expect(await screen.findByText("Runs 20 of 30 members: those with ≥ 0.5% of the gate's weight somewhere, linear mix")).toBeTruthy();
  });
});

describe("the real-tile card", () => {
  it("puts the viewer first with the shown model's Δm, then one status sentence and two headline numbers", async () => {
    const { container } = show(<RealTileInspector id="nexus/f200w-0001" />);
    const viewer = await screen.findByTestId("viewer");
    // two large frames: LR and the scored model it shows; the picker lists exactly this tile's outputs
    expect(viewer.textContent).toBe("real|f200w-0001|lr,m:rbf|rbf,member:member_1");
    expect(container.querySelector(".res-card")?.firstElementChild?.contains(viewer)).toBe(true);
    // footer: the VIS flux against the LR as Δm, warn-toned beyond 0.1 mag
    const delta = container.querySelector(".res-card__delta") as HTMLElement;
    expect(delta.textContent).toBe("VIS, LR vs RBF combiner: Δm +1.27 (flux ×0.31)");
    expect(delta.dataset.warn).toBe("true");
    expect(screen.getByRole("button", { name: "Open large" })).toBeTruthy();
    // ONE status sentence in the Targets vocabulary (no state pills)
    expect(screen.getByText("Production SR is stale, made by legacy RBF: only a legacy SR exists.")).toBeTruthy();
    // two headline numbers of the shown model
    const facts = screen.getByRole("heading", { name: "Measured on RBF combiner" }).closest("section") as HTMLElement;
    expect(within(facts).getByText("Holes, worst band (VIS)")).toBeTruthy();
    expect(within(facts).getByText("4.3")).toBeTruthy();
    expect(within(facts).getByText("0.970")).toBeTruthy();
    // then the actions, the danger zone alone at the foot
    expect(screen.getByRole("button", { name: "Compare models on this tile…" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Open in Files" })).toBeTruthy();
    const danger = screen.getByRole("region", { name: "Danger zone" });
    expect(container.querySelector(".res-card")?.lastElementChild).toBe(danger);
    // grid, Q1 tile and disk wait in the collapsed Details
    const details = container.querySelector("details.res-card__details") as HTMLDetailsElement;
    expect(details.open).toBe(false);
    expect(within(details).getByText(/VIS 27\.7 e⁻ \(the MER noise map's per-pixel RMS/)).toBeTruthy();
  });

  it("hides an all-empty R̃ column and says why the tile has no median R", async () => {
    routes["GET /api/real/nexus/f200w-0001"] = () => ({ body: { ...CARD, models: {
      rbf: { state: "current", legacy: true, label: "RBF", metrics: {
        per_band: { VIS: { hole_pct: 70, flux_ratio: 0.31 }, H_E: { hole_pct: 77.5, flux_ratio: 0.39 } },
        summary: { hole_pct_max: 77.5, median_R: null, n_peaks: 0 } } },
      "member:member_1": { state: "current", label: "Member 1", metrics: { per_band: { VIS: { hole_pct: 60 } }, summary: { hole_pct_max: 60 } } },
    } } });
    show(<RealTileInspector id="nexus/f200w-0001" />);
    await screen.findByTestId("viewer");
    const facts = screen.getByRole("heading", { name: "Measured on RBF combiner" }).closest("section") as HTMLElement;
    expect(within(facts).getByText("Flux SR/LR, lowest NISP band (H)")).toBeTruthy();
    expect(within(facts).getByText("0.39")).toBeTruthy();
    expect(screen.getByText(/No median R: the tile has no bright peak/)).toBeTruthy();
    const table = screen.getByRole("grid", { name: "Model outputs" });
    expect(within(table).getByRole("columnheader", { name: /Holes/ })).toBeTruthy();
    expect(within(table).queryByRole("columnheader", { name: /R̃|Median R/ })).toBeNull();
  });

  it("shows another model on a row click: its frames, Δm and headline", async () => {
    const { container } = show(<RealTileInspector id="nexus/f200w-0001" />);
    await screen.findByTestId("viewer");
    fireEvent.click(screen.getByText("member 1"));
    expect(viewerMock.tiers.at(-1)).toEqual(["lr", "m:member:member_1"]);
    expect(container.querySelector(".res-card__delta")?.textContent).toBe("VIS, LR vs member 1: Δm +0.02 (flux ×0.98)");
    expect(screen.getByRole("heading", { name: "Measured on member 1" })).toBeTruthy();
  });

  it("remembers the tile for the atlas's opening view", async () => {
    show(<RealTileInspector id="nexus/f200w-0001" />);
    await screen.findByTestId("viewer");
    const saved = JSON.parse(localStorage.getItem(LAST_TILE_KEY) ?? "{}");
    expect(saved).toMatchObject({ ra: 268.4, dec: 65.1, ref: "nexus/f200w-0001" });
  });

  it("opens the tile's FITS in Files", async () => {
    show(<RealTileInspector id="nexus/f200w-0001" />);
    fireEvent.pointerDown(await screen.findByRole("button", { name: "Open in Files" }), { button: 0 });
    fireEvent.click(await screen.findByRole("menuitem", { name: "member 1" }));
    expect(screen.getByTestId("loc").textContent).toBe("/files?fits=real_outputs%2Fnexus%2Ff200w-0001%2Fmember_1.fits");
  });

  it("deletes the model outputs only after the word is typed", async () => {
    routes["POST /api/real/nexus/f200w-0001/delete-outputs"] = () => ({ body: { ok: true, removed_count: 3, cache_bytes_freed: 2048 } });
    show(<RealTileInspector id="nexus/f200w-0001" />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete model outputs…" }));
    const dlg = await screen.findByRole("alertdialog", { name: /Delete the model outputs of 1 tile/ });
    expect((within(dlg).getByRole("button", { name: "Delete outputs" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(within(dlg).getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull());
    expect(posts).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "Delete model outputs…" }));
    await answer(/Delete the model outputs of 1 tile/, "Delete outputs", "delete");
    await waitFor(() => expect(posts.map((p) => p.url)).toEqual(["/api/real/nexus/f200w-0001/delete-outputs"]));
  });

  it("a tile without model outputs asks the viewer for no model tier at all", async () => {
    routes["GET /api/real/poster/p1"] = () => ({ body: { ...POSTER.tiles[0], model_ready: true, image_urls: { lr: "/a" } } });
    show(<RealTileInspector id="poster/p1" />);
    expect((await screen.findByTestId("viewer")).textContent).toBe("real|p1|lr|,");
    expect(screen.getByText("No production SR yet.")).toBeTruthy();
    expect(screen.getByText("LR only: no SR yet.")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Models" })).toBeNull();          // said once, by the sentence
    expect(screen.queryByRole("region", { name: "Danger zone" })).toBeNull();   // nothing to delete
  });

  it("reads a catalogue object's state from its evaluation record, its SR's flux and its LR / SR totals", async () => {
    routes["GET /api/real/eval/lensA"] = () => ({ body: {
      ...CARD, source: "eval", id: "lensA", ref: "eval/lensA", has_jwst: false, models: {}, image_urls: { lr: "/a" }, files: {},
      extras: { grade: "A", kind: "lens", flux_ratio_sr_over_lr: 0.67 },
    } });
    routes["GET /api/evaluation/objects/lensA"] = () => ({ body: {
      id: "lensA", ok: "True", lr_total_e: "123456", sr_total_e: "82716", state: "stale", state_reason: "membership changed: made by 22 member(s), the production gate is fitted for 30 now",
      members: { member_labels: Array(22).fill("x"), combiner_kind: null }, current: { n_members: 30, combiner_kind: "spatial_gate" },
      provenance: [{ id: "2941ea92", git: "e969277", dirty: true, created_at: "2026-07-26T16:04:34Z" }],
      downloads: { LR: "/eval-files/lensA/original_stack.fits", SR: "/eval-files/lensA/SR.fits" },
      viewer: { collection: "evaluation", id: "lensA" },
    } });
    const { container } = show(<RealTileInspector id="eval/lensA" />);
    // its SR lives in the catalogue evaluation: the card shows that collection's LR beside SR
    expect((await screen.findByTestId("viewer")).textContent).toBe("evaluation|lensA|LR,SR|");
    // the reason already names the model: said once
    expect(await screen.findByText(
      "Production SR is stale: membership changed: made by 22 members, the production gate is fitted for 30 now.")).toBeTruthy();
    // the footer names the SR by what made it, not "production" (it predates the gate)
    expect(container.querySelector(".res-card__delta")?.textContent).toBe("VIS, LR vs SR (22-member ensemble): Δm +0.43 (flux ×0.67)");
    // its two numbers: the LR and SR total VIS flux (it has no holes or R)
    expect(screen.getByText("Total VIS flux")).toBeTruthy();
    expect(screen.getByText("123k")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Catalogue evaluation" }));
    expect(await screen.findByText("30 starfull members · spatial gate")).toBeTruthy();
    expect(screen.getByRole("link", { name: "SR" }).getAttribute("href")).toBe("/eval-files/lensA/SR.fits");
  });

  it("explains an unknown tile with the server's error", async () => {
    routes["GET /api/real/nexus/nope"] = () => ({ status: 404, body: { ok: false, error: "unknown nexus tile 'nope'" } });
    show(<RealTileInspector id="nexus/nope" />);
    expect(await screen.findByText("unknown nexus tile 'nope'")).toBeTruthy();
  });

  it("hands the tile to Sky › Compare (exactly this tile)", async () => {
    useSelection.getState().select("tile", ["poster/p1"]);
    show(<RealTileInspector id="nexus/f200w-0001" />);
    fireEvent.click(await screen.findByRole("button", { name: "Compare models on this tile…" }));
    expect(useSelection.getState().get("tile")).toEqual(["nexus/f200w-0001"]);
    expect(screen.getByTestId("loc").textContent).toBe("/sky/compare?tiles=nexus%2Ff200w-0001");
  });

  it("runs models on the tile as an experiment after a confirm", async () => {
    routes["POST /api/experiments"] = () => ({ body: { ok: true, job_id: "j9", experiment_id: "20260926-111111-000000", tiles: ["nexus/f200w-0001"], models: ["production", "mean"], skipped: {} } });
    show(<RealTileInspector id="nexus/f200w-0001" />);
    fireEvent.click(await screen.findByRole("button", { name: "Run models…" }));
    fireEvent.click(await screen.findByRole("button", { name: "Run 2" }));
    await answer(/Run 2 models on 1 tile/, "Run");
    await waitFor(() => expect(posts[0]).toEqual({ url: "/api/experiments", form: { tiles: "nexus/f200w-0001", models: "production,mean" } }));
    expect(useJobsStore.getState().keyed["sky:experiment"]).toBe("j9");
  });
});

describe("the comparison card", () => {
  it("says the comparison in one sentence, pivots one metric and links to Compare", async () => {
    show(<ExperimentInspector id="20260926-101010-abcdef" />, "/sky/atlas");
    expect(await screen.findByText(/core check/)).toBeTruthy();
    expect(screen.queryByTestId("viewer")).toBeNull();
    expect(screen.getByText(/against/).closest("p")?.textContent).toBe(
      "On 2 tiles, production leaves holes in 8.3 % of the bright pixels of its worst band (J), against 30.5 % for the member mean (VIS).");
    const grid = screen.getByRole("grid", { name: "Hole % per model and band" });
    expect(within(grid).getByText("30.5")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Open in Compare" }));
    expect(screen.getByTestId("loc").textContent).toBe("/sky/compare?exp=20260926-101010-abcdef");
  });
});
