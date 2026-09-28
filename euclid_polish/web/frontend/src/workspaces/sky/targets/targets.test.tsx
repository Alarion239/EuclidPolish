/* Sky › Targets against a mocked backend: the set chips with their counts,
 * one sentence per set, the one state vocabulary, the table sorted by flux,
 * Holes / R̃ only for scored rows, "Run production on stale" (one confirm,
 * then the jobs of the plan) and the Sources, each confirmed. Opening the
 * page posts nothing. */
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useInspector } from "../../../state/inspector";
import { useSelection } from "../../../state/selection";
import Targets from "../tabs/Targets";
import { answer, installBackend, show, teardownBackend, type Post, type Routes } from "../testing/backend";

let routes: Routes;
let posts: Post[];

beforeEach(() => { ({ routes, posts } = installBackend()); });
afterEach(() => teardownBackend());

const grid = () => screen.getByRole("grid", { name: "Targets" });
const targetIds = () => within(grid()).getAllByRole("row").slice(1).map((r) => r.querySelector(".res-tilecell .mono")?.textContent);
const query = () => new URLSearchParams((screen.getByTestId("loc").textContent ?? "").split("?")[1] ?? "");
const gets = () => vi.mocked(fetch).mock.calls.map(([u, init]) => `${(init as RequestInit | undefined)?.method ?? "GET"} ${String(u)}`);

describe("Sky › Targets", () => {
  it("shows every science target by default: chips with real labels and counts, one sentence per set, rows by flux", async () => {
    show(<Targets />);
    expect(await screen.findByRole("button", { name: "NEXUS × JWST 2" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Poster galaxy 1" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Lens candidates 2" })).toBeTruthy();    // the failed cutout is not counted
    expect(screen.getByRole("button", { name: "Q1 galaxies 1" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Legacy field/ })).toBeNull();          // under More
    await waitFor(() => expect(targetIds()).toEqual(["f200w-0001", "lensA", "lensB", "gal1", "f200w-0002", "p1"]));
    const lead = (text: string) => screen.getByText((_, el) => el?.tagName === "P" && el.textContent === text);
    expect(lead("Lens candidates: all 2 reconstructions predate the current gate · median flux SR/LR 0.67")).toBeTruthy();
    expect(lead("NEXUS × JWST: 1 current, 1 stale · median flux SR/LR 0.74, median worst-band holes 2.9 %")).toBeTruthy();
    expect(lead("Poster galaxy: the tile has no production SR yet")).toBeTruthy();
    // the counts live on the state control, once
    expect(screen.getByRole("radio", { name: "Stale 4" })).toBeTruthy();
    expect(screen.getByRole("radio", { name: "Missing 1" })).toBeTruthy();
    // no legacy / pair lists, and nothing is posted by opening the page
    expect(gets()).not.toContain("GET /api/real/field");
    expect(gets()).not.toContain("GET /api/real/pair");
    expect(posts).toEqual([]);
  });

  it("gives a set its sentence only once its own list is in: a loading line, never 'none yet'", async () => {
    // The evaluation rows (lens candidates, Q1 galaxies) arrive late.
    const base = vi.mocked(fetch).getMockImplementation()!;
    let release: () => void = () => undefined;
    const late = new Promise<void>((r) => { release = r; });
    vi.mocked(fetch).mockImplementation(async (input, init) => {
      if (String(input) === "/api/evaluation/runs") await late;
      return base(input, init);
    });
    show(<Targets />);
    const lead = (text: string) => screen.queryByText((_, el) => el?.tagName === "P" && el.textContent === text);
    await waitFor(() => expect(lead("Poster galaxy: the tile has no production SR yet")).toBeTruthy());
    expect(lead("Lens candidates: loading…")).toBeTruthy();
    expect(lead("Q1 galaxies: loading…")).toBeTruthy();
    expect(lead("Lens candidates: none yet")).toBeNull();
    expect(lead("Q1 galaxies: none yet")).toBeNull();
    // no partial counts on the state control or the run button meanwhile
    expect(screen.getByRole("radio", { name: "Stale" })).toBeTruthy();
    expect((screen.getByRole("button", { name: /Run production on stale/ }) as HTMLButtonElement).disabled).toBe(true);
    release();
    expect(await screen.findByText((_, el) => el?.tagName === "P"
      && el.textContent === "Lens candidates: all 2 reconstructions predate the current gate · median flux SR/LR 0.67")).toBeTruthy();
    expect(lead("Lens candidates: loading…")).toBeNull();
    expect(screen.getByRole("radio", { name: "Stale 4" })).toBeTruthy();
  });

  it("keeps the Holes / R̃ columns out of a view that is not all scored, and offers the scored rows alone", async () => {
    show(<Targets />);
    await waitFor(() => expect(targetIds()).toHaveLength(6));
    expect(within(grid()).queryByRole("columnheader", { name: /Holes/ })).toBeNull();
    expect(within(grid()).queryByRole("columnheader", { name: /R̃|Median R/ })).toBeNull();
    expect(screen.getByText(/Holes and R̃ are measured on 2 scored tiles/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Show only them" }));
    await waitFor(() => expect(targetIds()).toEqual(["f200w-0001", "f200w-0002"]));
    expect(within(grid()).getByRole("columnheader", { name: /Holes/ })).toBeTruthy();
    expect(screen.getByTestId("loc").textContent).toContain("scored=1");
    expect(screen.getByRole("radio", { name: "All 2" })).toBeTruthy();
  });

  it("names the model that made each SR and shows Holes / R̃ only when every row is scored", async () => {
    show(<Targets />, "/sky/targets?set=lenses");
    await waitFor(() => expect(targetIds()).toEqual(["lensA", "lensB"]));
    expect(within(grid()).queryByRole("columnheader", { name: /Holes/ })).toBeNull();
    expect(within(grid()).getAllByText("22-member ensemble (combiner not recorded)")).toHaveLength(2);
    // the failed cutout: a caption, shown on request as "missing" with its error
    expect(screen.getByText(/1 catalogue cutout failed to download/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Show them" }));
    await waitFor(() => expect(targetIds()).toContain("bad"));
    expect(screen.getByTestId("loc").textContent).toContain("failed=1");
  });

  it("scored tiles bring the Holes and R̃ columns", async () => {
    show(<Targets />, "/sky/targets?set=nexus");
    await waitFor(() => expect(targetIds()).toEqual(["f200w-0001", "f200w-0002"]));
    expect(within(grid()).getByRole("columnheader", { name: /Holes/ })).toBeTruthy();
    expect(within(grid()).getByText("4.3")).toBeTruthy();
    expect(within(grid()).getByText("legacy minibatched convex all-asinh RBF")).toBeTruthy();
  });

  it("filters by the one state vocabulary through the URL", async () => {
    show(<Targets />);
    fireEvent.click(await screen.findByRole("radio", { name: "Current 1" }));
    await waitFor(() => expect(targetIds()).toEqual(["f200w-0002"]));
    expect(screen.getByTestId("loc").textContent).toContain("state=current");
  });

  it("reads an old link's store ids (?src=) and a lens grade (?g=)", async () => {
    routes["GET /api/real/tile"] = () => ({ body: { source: "tile", label: "Cached", count: 1, tiles: [
      { source: "tile", id: "t1", ref: "tile/t1", label: "Tile t1", ra: 1, dec: 2, production_state: "current",
        models: { production: { state: "current", label: "Production · spatial gate" } } },
    ] } });
    const { unmount } = show(<Targets />, "/sky/targets?src=tile");
    await waitFor(() => expect(targetIds()).toEqual(["t1"]));
    expect(screen.getByRole("button", { name: /Cached tiles/ }).getAttribute("aria-pressed")).toBe("true");
    unmount();
    show(<Targets />, "/sky/targets?set=lenses&g=B,gal");
    await waitFor(() => expect(targetIds()).toEqual(["lensB"]));
    expect(screen.getByRole("button", { name: "Grade B" })).toBeTruthy();
    // the failed grade-A cutout is not counted under a grade-B filter
    expect(screen.queryByText(/catalogue cutouts? failed to download/)).toBeNull();
  });

  it("keeps an old link's comma-list filters: store ids in ?set=, the old group list ?g=, the old ?st=", async () => {
    routes["GET /api/real/tile"] = () => ({ body: { source: "tile", label: "Cached", count: 1, tiles: [
      { source: "tile", id: "t1", ref: "tile/t1", label: "Tile t1", ra: 1, dec: 2, production_state: "current",
        models: { production: { state: "current", label: "Production · spatial gate" } } },
    ] } });
    // /sky/results?src=nexus,tile → ?set=nexus,tile (the redirect maps whole values only)
    let view = show(<Targets />, "/sky/targets?set=nexus,tile");
    await waitFor(() => expect(targetIds()).toEqual(["f200w-0001", "f200w-0002", "t1"]));
    expect(screen.getByRole("button", { name: /Cached tiles/ }).getAttribute("aria-pressed")).toBe("true");
    await waitFor(() => expect(query().get("set")).toBe("nexus,cached"));
    view.unmount();
    // /sky/catalog-eval?g=A,B → ?g=A,B&set=lenses,galaxies: lens A and B only, as the old page showed
    view = show(<Targets />, "/sky/targets?g=A,B&set=lenses,galaxies");
    await waitFor(() => expect(targetIds()).toEqual(["lensA", "lensB"]));
    await waitFor(() => expect(query().get("set")).toBe("lenses"));
    expect(query().get("g")).toBe("A,B");
    view.unmount();
    // /sky/catalog-eval?g=lensA,galaxies&st=unknown → lens A and the galaxies, stale
    view = show(<Targets />, "/sky/targets?g=lensA,galaxies&st=unknown&set=lenses,galaxies");
    await waitFor(() => expect(targetIds()).toEqual(["lensA", "gal1"]));
    expect(screen.getByRole("radio", { name: "Stale 2" }).getAttribute("aria-checked")).toBe("true");
    await waitFor(() => expect(query().get("state")).toBe("stale"));
    expect(query().get("st")).toBeNull();
    expect(query().get("g")).toBe("A");
    view.unmount();
  });

  it("opens the tile card from a row", async () => {
    show(<Targets />, "/sky/targets?set=lenses");
    await waitFor(() => expect(targetIds()).toEqual(["lensA", "lensB"]));
    fireEvent.click(within(grid()).getByText("lensA"));
    expect(useInspector.getState().current).toEqual({ kind: "tile", id: "eval/lensA" });
  });

  it("runs production on the stale targets after ONE confirm naming the plan", async () => {
    routes["POST /api/jwst-euclid/nexus/infer"] = () => ({ body: { ok: true, job_id: "n1" } });
    routes["POST /api/evaluation/run-grouped"] = () => ({ body: { ok: true, job_id: "g1" } });
    show(<Targets />);
    const run = await screen.findByRole("button", { name: "Run production on stale (4)…" });
    fireEvent.click(run);
    const dlg = await answer("Run production on 4 stale targets?", "Cancel");
    expect(dlg.textContent).toContain("1 NEXUS × JWST tile: NEXUS field inference");
    expect(dlg.textContent).toContain("3 lens candidates and Q1 galaxies: the grouped analysis");
    expect(posts).toEqual([]);
    fireEvent.click(screen.getByRole("button", { name: "Run production on stale (4)…" }));
    await answer("Run production on 4 stale targets?", "Run production");
    await waitFor(() => expect(posts).toEqual([
      { url: "/api/jwst-euclid/nexus/infer", form: { field_id: "nf", tiles: "f200w-0001", spec: "production" } },
      { url: "/api/evaluation/run-grouped", form: { n: "2", synthetic: "0" } },
    ]));
  });

  it("hands the selection to Sky › Compare", async () => {
    show(<Targets />, "/sky/targets?set=nexus");
    await waitFor(() => expect(targetIds()).toEqual(["f200w-0001", "f200w-0002"]));
    fireEvent.click(within(grid()).getAllByRole("checkbox")[1]);
    fireEvent.click(await screen.findByRole("button", { name: "Compare models (1)" }));
    expect(useSelection.getState().get("tile")).toEqual(["nexus/f200w-0001"]);
    expect(screen.getByTestId("loc").textContent).toBe("/sky/compare?tiles=nexus%2Ff200w-0001");
  });

  it("links the metric definitions to their one home in Sky › Compare", async () => {
    show(<Targets />);
    expect((await screen.findByRole("link", { name: "Metric definitions" })).getAttribute("href")).toBe("/sky/compare?defs=1");
  });
});

describe("Sky › Targets › Sources", () => {
  const openSources = async (item: RegExp) => {
    fireEvent.pointerDown(await screen.findByRole("button", { name: "Sources" }), { button: 0 });
    fireEvent.click(await screen.findByRole("menuitem", { name: item }));
  };

  it("runs the grouped analysis with the chosen size, after a confirm", async () => {
    routes["POST /api/evaluation/run-grouped"] = () => ({ body: { ok: true, job_id: "g1" } });
    show(<Targets />);
    await openSources(/Grouped analysis/);
    const dlg = await screen.findByRole("dialog", { name: "Grouped analysis" });
    fireEvent.change(within(dlg).getByRole("spinbutton"), { target: { value: "4" } });
    fireEvent.click(within(dlg).getByRole("button", { name: "Run…" }));
    await answer("Run the grouped analysis?", "Run");
    await waitFor(() => expect(posts).toEqual([{ url: "/api/evaluation/run-grouped", form: { n: "4", synthetic: "0" } }]));
  });

  it("needs the Euclid archive login to query galaxies (links to System › Connections)", async () => {
    routes["GET /auth/status"] = () => ({ body: { authenticated: false } });
    show(<Targets />);
    await openSources(/Query Q1 galaxies/);
    const dlg = await screen.findByRole("dialog", { name: "Query Q1 galaxies" });
    expect(await within(dlg).findByRole("link", { name: "System › Connections" })).toBeTruthy();
    expect((within(dlg).getByRole("button", { name: "Query…" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("syncs the evaluation results from FASRC only after the --delete-after is confirmed, with confirm=1", async () => {
    routes["POST /api/evaluation/sync"] = () => ({ body: { ok: true, n_ok: 913, n: 930 } });
    show(<Targets />);
    await openSources(/Sync the evaluation results/);
    await answer("Sync the evaluation results from FASRC?", "Cancel");
    expect(posts).toEqual([]);
    await openSources(/Sync the evaluation results/);
    await answer("Sync the evaluation results from FASRC?", "Sync and delete local-only");
    await waitFor(() => expect(posts).toEqual([{ url: "/api/evaluation/sync", form: { confirm: "1" } }]));
  });

  it("fetches the Q1 lens catalogue only after a confirm", async () => {
    routes["POST /api/evaluation/fetch-catalog"] = () => ({ body: { ok: true, rows: 500 } });
    show(<Targets />);
    await openSources(/Fetch the Q1 lens catalogue/);
    await answer("Fetch the Q1 strong-lens catalogue?", "Fetch");
    await waitFor(() => expect(posts.map((p) => p.url)).toEqual(["/api/evaluation/fetch-catalog"]));
  });

  it("drops the evaluation run's cached eye/solar PNGs only after a confirm", async () => {
    routes["POST /api/evaluation/rerender"] = () => ({ body: { ok: true, removed: 12 } });
    show(<Targets />);
    await openSources(/Drop the cached eye\/solar PNGs/);
    await answer("Drop the cached eye/solar PNGs?", "Cancel");
    expect(posts).toEqual([]);
    await openSources(/Drop the cached eye\/solar PNGs/);
    await answer("Drop the cached eye/solar PNGs?", "Drop cached PNGs");
    await waitFor(() => expect(posts.map((p) => p.url)).toEqual(["/api/evaluation/rerender"]));
  });
});
