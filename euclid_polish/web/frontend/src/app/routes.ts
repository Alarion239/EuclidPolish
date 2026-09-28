/* The data router's route table, built from the route manifest (C1).
 *
 *   createBrowserRouter(buildRoutes())
 *
 * One root layout route (the shell) holds:
 *   - every workspace: `<path>/*` (e.g. `/sky/*`, `/models/:mode/*`; the
 *     home page is exactly `/`). The workspace component is lazy
 *     (`workspaceComponents`); it validates the rest of the path against the
 *     manifest itself (<Workspace>), so the router and Flask agree on what a
 *     page is. It renders inside a contained error boundary (reset on
 *     navigation): a workspace chunk that fails to load — the console was
 *     rebuilt under an open page — shows "Reload page" in the stage while
 *     the rail, top bar and other workspaces keep working;
 *   - every legacy redirect — each `manifest.redirectRules` `from` pattern
 *     (`/ensemble/:mode/curves`; a static old path such as `/sky/results`
 *     outranks the `/sky/*` workspace route), each exact `manifest.redirects`
 *     path and `/app/<rest>` — as a client-side <Navigate replace> to
 *     `redirectTarget()` (the rules rewrite the query, the exact entries keep
 *     it) plus the hash; Flask answers the same URLs with the same 308;
 *   - a catch-all Not found.
 *
 * Adding a workspace = a manifest entry + nav.ts metadata + one line in
 * `workspaceComponents` + its folder (tests fail when one is missing).
 */
import { Suspense, createElement, lazy, type ComponentType, type LazyExoticComponent } from "react";
import { Navigate, useLocation, type RouteObject } from "react-router-dom";
import { ErrorBoundary, RouteError } from "./ErrorBoundary";
import { MANIFEST, normalisePath, redirectTarget, type RouteManifest } from "./manifest";
import { workspaceLabel } from "./nav";
import { NotFound } from "./NotFound";
import { TabSkeleton } from "./workspace";

export type WorkspaceModule = { default: ComponentType };
export type WorkspaceLoader = () => Promise<WorkspaceModule>;

/** Lazy import of every workspace's `index.tsx`, by manifest id. */
export const workspaceComponents: Record<string, WorkspaceLoader> = {
  home: () => import("../workspaces/home"),
  synthetic: () => import("../workspaces/synthetic"),
  models: () => import("../workspaces/models"),
  sky: () => import("../workspaces/sky"),
  figures: () => import("../workspaces/figures"),
  files: () => import("../workspaces/files"),
  runs: () => import("../workspaces/runs"),
  notebook: () => import("../workspaces/notebook"),
  system: () => import("../workspaces/system"),
};

/** Every client-side redirect route path: the rule patterns, then the exact
 *  entries (each once, in manifest order). */
export function redirectRoutePaths(manifest: RouteManifest = MANIFEST): string[] {
  const out: string[] = [];
  for (const rule of manifest.redirectRules ?? []) out.push(normalisePath(rule.from));
  for (const from of Object.keys(manifest.redirects ?? {})) out.push(normalisePath(from));
  return [...new Set(out)].filter((p) => p !== "/");
}

/** Client-side legacy redirect: the manifest target (query per the rule) + hash. */
export function RedirectRoute() {
  const location = useLocation();
  const target = redirectTarget(location.pathname, location.search);
  if (target == null) return createElement(NotFound);
  return createElement(Navigate, { to: `${target}${location.hash}`, replace: true });
}

const LAZY = new WeakMap<WorkspaceLoader, LazyExoticComponent<ComponentType>>();

function lazyWorkspace(load: WorkspaceLoader): LazyExoticComponent<ComponentType> {
  let hit = LAZY.get(load);
  if (!hit) {
    hit = lazy(load);
    LAZY.set(load, hit);
  }
  return hit;
}

/** One workspace's route element: its lazy component in a Suspense
 *  skeleton, inside an error boundary that resets when the path changes. */
function WorkspaceRoute({ Component, label }: { Component: ComponentType; label: string }) {
  const { pathname } = useLocation();
  const children = createElement(Suspense, { fallback: createElement(TabSkeleton) }, createElement(Component));
  return createElement(ErrorBoundary, { resetKey: pathname, label, children });
}

export type BuildRoutesOpts = {
  manifest?: RouteManifest;
  components?: Record<string, WorkspaceLoader>;
  /** The root layout (the shell); must render an <Outlet/>. */
  layout?: ComponentType;
};

export function buildRoutes(opts: BuildRoutesOpts = {}): RouteObject[] {
  const manifest = opts.manifest ?? MANIFEST;
  const components = opts.components ?? workspaceComponents;
  const children: RouteObject[] = [];

  for (const ws of manifest.workspaces) {
    const load = components[ws.id];
    if (!load) throw new Error(`no workspace component for "${ws.id}" (app/routes.ts workspaceComponents)`);
    const element = createElement(WorkspaceRoute, { Component: lazyWorkspace(load), label: workspaceLabel(ws.id) });
    children.push(ws.path === "/"
      ? { path: "/", element }
      : { path: `${ws.path}/*`, element });
  }
  for (const from of redirectRoutePaths(manifest)) {
    children.push({ path: from, Component: RedirectRoute });
  }
  children.push({ path: "/app/*", Component: RedirectRoute });
  children.push({ path: "*", Component: NotFound });

  const root: RouteObject = { id: "root", errorElement: createElement(RouteError), children };
  if (opts.layout) root.Component = opts.layout;
  return [root];
}

/** Future flags shared by the app router and the route tests. */
export const ROUTER_FUTURE = {
  v7_relativeSplatPath: true,
  v7_fetcherPersist: true,
  v7_normalizeFormMethod: true,
  v7_partialHydration: true,
  v7_skipActionErrorRevalidation: true,
} as const;
