import { act, cleanup, render, screen } from "@testing-library/react";
import { RouterProvider, createMemoryRouter, useLocation } from "react-router-dom";
import { Outlet } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { MANIFEST, pagePaths, redirectTarget } from "./manifest";
import { ROUTER_FUTURE, buildRoutes, redirectRoutePaths, workspaceComponents, type WorkspaceLoader } from "./routes";
import { Workspace, defineTabs } from "./workspace";

/* Fake workspaces: each renders <Workspace> with stub tabs, so the route
   table and the manifest validation are tested without the legacy pages. */
function Where() {
  const loc = useLocation();
  return <output data-testid="where">{`${loc.pathname}${loc.search}${loc.hash}`}</output>;
}

const fakes: Record<string, WorkspaceLoader> = Object.fromEntries(MANIFEST.workspaces.map((ws) => {
  const tabs = defineTabs(ws.id, Object.fromEntries(ws.tabs.map((t) => [t, {
    load: async () => ({ default: () => <p>{`${ws.id}:${t}`}</p> }),
  }])));
  const Component = () => (
    <>
      <Workspace id={ws.id} tabs={tabs}><p>{`${ws.id}:root`}</p></Workspace>
      <Where />
    </>
  );
  return [ws.id, async () => ({ default: Component })];
}));

function go(url: string) {
  const router = createMemoryRouter(buildRoutes({ components: fakes }), { initialEntries: [url], future: ROUTER_FUTURE });
  render(<RouterProvider router={router} future={{ v7_startTransition: true }} />);
  return router;
}

async function landed(text: string) {
  expect(await screen.findByText(text)).toBeTruthy();
}

describe("buildRoutes", () => {
  it("has a workspace component for every manifest workspace", () => {
    expect(Object.keys(workspaceComponents).sort()).toEqual(MANIFEST.workspaces.map((w) => w.id).sort());
    expect(() => buildRoutes({ components: {} })).toThrow(/no workspace component/);
  });

  it("renders every page path of the manifest", async () => {
    for (const path of pagePaths()) {
      const router = go(path);
      const m = path === "/" ? "home:root" : null;
      if (m) await landed(m);
      else await screen.findByTestId("where");
      // tabless pages keep their path; workspace roots redirect to the default tab
      expect(router.state.location.pathname.startsWith(path === "/" ? "/" : path)).toBe(true);
      cleanup();
    }
  });

  it("renders the right tab", async () => {
    go("/models/starless/combiner");
    await landed("models:combiner");
  });

  it("redirects a workspace root to its default tab, keeping query and hash", async () => {
    const router = go("/sky?ra=10#x");
    await landed("sky:atlas");
    expect(router.state.location).toMatchObject({ pathname: "/sky/atlas", search: "?ra=10", hash: "#x" });
  });

  it("follows legacy redirects with the query preserved", async () => {
    const router = go("/config?x=1");
    await landed("system:config");
    expect(router.state.location.pathname + router.state.location.search).toBe("/system/config?x=1");
  });

  it("follows the query-aware rules exactly as Flask does, keeping the hash", async () => {
    for (const [from, to] of [
      ["/ensemble/starless/curves?layout=time&x=2", "/runs/history?x=2&step=ensemble_train"],
      ["/ensemble/starfull/curves", "/models/starfull/members?view=curves"],
      ["/sky/results?source=tile&inspect=tile%3Aposter%2Fp1", "/sky/targets?set=cached&inspect=tile%3Aposter%2Fp1"],
      ["/realism/pixels?view=census", "/synthetic/records?section=census"],
      ["/cutouts/Y_E", "/synthetic/psf?view=cutouts&band=Y_E"],
      ["/ops/fasrc?view=logs&job=7&task=2", "/runs/history?run=7&task=2&logs=1"],
      ["/tracking?view=archive", "/notebook/backups?show=campaigns"],
    ] as const) {
      const router = go(`${from}#frag`);
      await screen.findByTestId("where");
      const loc = router.state.location;
      expect(`${loc.pathname}${loc.search}`, from).toBe(to);
      expect(loc.hash, from).toBe("#frag");
      cleanup();
    }
  });

  it("registers a client route for every rule pattern and exact entry", () => {
    const paths = redirectRoutePaths();
    for (const rule of MANIFEST.redirectRules ?? []) expect(paths, rule.from).toContain(rule.from);
    for (const from of Object.keys(MANIFEST.redirects)) expect(paths, from).toContain(from);
    expect(new Set(paths).size).toBe(paths.length);
  });

  it("redirects every manifest entry client-side", async () => {
    for (const [from, to] of Object.entries(MANIFEST.redirects)) {
      const router = go(`${from}?q=1`);
      await screen.findByTestId("where");
      const target = new URL(to, "http://x").pathname;
      expect(router.state.location.pathname.startsWith(target === "/" ? "/" : target), from).toBe(true);
      expect(router.state.location.search, from).toBe("?q=1");
      cleanup();
    }
  });

  it("redirects every rule source client-side to the same URL the matcher gives", async () => {
    for (const rule of MANIFEST.redirectRules ?? []) {
      let sources = [rule.from];
      for (const [name, values] of Object.entries(rule.params ?? {})) {
        sources = sources.flatMap((s) => values.map((v) => s.replace(`:${name}`, v)));
      }
      const q = new URLSearchParams(Object.entries(rule.query ?? {}).map(([k, want]) => [
        k, want === "*" ? "x" : Array.isArray(want) ? want[0] : want,
      ])).toString();
      for (const source of sources.slice(0, 1)) {
        const url = q ? `${source}?${q}` : source;
        const router = go(url);
        await screen.findByTestId("where");
        const loc = router.state.location;
        const [path, search = ""] = redirectTarget(source, q)!.split("?");
        // the workspace then adds its default tab to a bare workspace path
        expect(loc.pathname.startsWith(path), url).toBe(true);
        expect(loc.search, url).toBe(search ? `?${search}` : "");
        cleanup();
      }
    }
  }, 60_000);

  it("maps /app/<rest> to /<rest> and never off-host", async () => {
    const a = go("/app/sky/compare?y=1");
    await landed("sky:compare");
    expect(a.state.location.pathname + a.state.location.search).toBe("/sky/compare?y=1");
    cleanup();
    // an old page behind /app follows its own redirect next
    const c = go("/app/sky/results?y=1");
    await landed("sky:targets");
    expect(c.state.location.pathname + c.state.location.search).toBe("/sky/targets?y=1");
    cleanup();
    const b = go("/app//evil.example");
    await screen.findByText("No page here");
    expect(b.state.location.pathname).toBe("/evil.example");
  });

  it("shows Not found for unknown paths, tabs and params", async () => {
    for (const path of ["/nope", "/sky/unknown", "/models/foo", "/models/foo/train", "/models/starfull/train/x",
      "/ensemble/foo", "/ensemble/foo/knee", "/cutouts/K"]) {
      go(path);
      expect(await screen.findByText("No page here"), path).toBeTruthy();
      // exactly one (visually hidden) h1, in plain words
      const h1s = document.querySelectorAll("h1");
      expect(h1s.length, path).toBe(1);
      expect(h1s[0].textContent, path).toBe("Not found");
      cleanup();
    }
  });

  it("redirects a bare workspace path to `redirectTab` when given (a valid one only)", async () => {
    const tabs = defineTabs("models", Object.fromEntries(MANIFEST.workspaces.find((w) => w.id === "models")!.tabs
      .map((t) => [t, { load: async () => ({ default: () => <p>{`models:${t}`}</p> }) }])));
    for (const [redirectTab, landing] of [["combiner", "combiner"], ["nope", "leaderboard"]] as const) {
      const components = { ...fakes, models: async () => ({ default: () => <Workspace id="models" tabs={tabs} redirectTab={redirectTab} /> }) };
      const router = createMemoryRouter(buildRoutes({ components }), { initialEntries: ["/models/starless?x=1"], future: ROUTER_FUTURE });
      render(<RouterProvider router={router} future={{ v7_startTransition: true }} />);
      await landed(`models:${landing}`);
      expect(router.state.location.pathname + router.state.location.search).toBe(`/models/starless/${landing}?x=1`);
      cleanup();
    }
  });

  it("contains a workspace whose chunk fails to load: the shell stays, the next page works", async () => {
    // The console was rebuilt under an open page: the old chunk 404s.
    const quiet = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      const components = {
        ...fakes,
        sky: () => Promise.reject(new TypeError("Failed to fetch dynamically imported module: /static/dist/assets/sky-old.js")),
      };
      const Layout = () => <div><nav aria-label="Shell">rail</nav><Outlet /></div>;
      const router = createMemoryRouter(buildRoutes({ components, layout: Layout }), {
        initialEntries: ["/sky/atlas"], future: ROUTER_FUTURE,
      });
      render(<RouterProvider router={router} future={{ v7_startTransition: true }} />);
      expect(await screen.findByText("Sky hit an error")).toBeTruthy();
      expect(screen.getByText(/A newer console build is available/)).toBeTruthy();
      expect(screen.getByRole("button", { name: "Reload page" })).toBeTruthy();
      expect(screen.getByRole("navigation", { name: "Shell" })).toBeTruthy();   // not the root error page
      await act(() => router.navigate("/synthetic/records"));
      await landed("synthetic:records");
      expect(screen.queryByText("Sky hit an error")).toBeNull();
    } finally {
      quiet.mockRestore();
    }
  });

  it("keeps navigating inside the app", async () => {
    const router = go("/synthetic/records");
    await landed("synthetic:records");
    await act(() => router.navigate("/synthetic/psf"));
    await landed("synthetic:psf");
    // an in-app link to an old page follows the rule, like Flask's 308
    await act(() => router.navigate("/data/psfs"));
    await landed("synthetic:psf");
    expect(router.state.location.pathname + router.state.location.search).toBe("/synthetic/psf?view=epsf");
  });
});
