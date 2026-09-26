import { act, cleanup, render, screen } from "@testing-library/react";
import { RouterProvider, createMemoryRouter, useLocation } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { MANIFEST, pagePaths } from "./manifest";
import { ROUTER_FUTURE, buildRoutes, workspaceComponents, type WorkspaceLoader } from "./routes";
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
    go("/ensemble/starless/knee");
    await landed("ensemble:knee");
  });

  it("redirects a workspace root to its default tab, keeping query and hash", async () => {
    const router = go("/sky?ra=10#x");
    await landed("sky:atlas");
    expect(router.state.location).toMatchObject({ pathname: "/sky/atlas", search: "?ra=10", hash: "#x" });
  });

  it("follows legacy redirects with the query preserved", async () => {
    const router = go("/config?x=1");
    await landed("settings:config");
    expect(router.state.location.pathname + router.state.location.search).toBe("/settings/config?x=1");
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

  it("maps /app/<rest> to /<rest> and never off-host", async () => {
    const a = go("/app/sky/results?y=1");
    await landed("sky:results");
    expect(a.state.location.pathname + a.state.location.search).toBe("/sky/results?y=1");
    cleanup();
    const b = go("/app//evil.example");
    await screen.findByText("No page here");
    expect(b.state.location.pathname).toBe("/evil.example");
  });

  it("shows Not found for unknown paths, tabs and params", async () => {
    for (const path of ["/nope", "/sky/unknown", "/ensemble/foo", "/ensemble/foo/knee", "/ensemble/starfull/knee/x"]) {
      go(path);
      expect(await screen.findByText("No page here"), path).toBeTruthy();
      cleanup();
    }
  });

  it("redirects a bare workspace path to `redirectTab` when given (a valid one only)", async () => {
    const tabs = defineTabs("ensemble", Object.fromEntries(MANIFEST.workspaces.find((w) => w.id === "ensemble")!.tabs
      .map((t) => [t, { load: async () => ({ default: () => <p>{`ensemble:${t}`}</p> }) }])));
    for (const [redirectTab, landing] of [["knee", "knee"], ["nope", "overview"]] as const) {
      const components = { ...fakes, ensemble: async () => ({ default: () => <Workspace id="ensemble" tabs={tabs} redirectTab={redirectTab} /> }) };
      const router = createMemoryRouter(buildRoutes({ components }), { initialEntries: ["/ensemble/starless?x=1"], future: ROUTER_FUTURE });
      render(<RouterProvider router={router} future={{ v7_startTransition: true }} />);
      await landed(`ensemble:${landing}`);
      expect(router.state.location.pathname + router.state.location.search).toBe(`/ensemble/starless/${landing}?x=1`);
      cleanup();
    }
  });

  it("keeps navigating inside the app", async () => {
    const router = go("/data/records");
    await landed("data:records");
    await act(() => router.navigate("/data/psfs"));
    await landed("data:psfs");
  });
});
