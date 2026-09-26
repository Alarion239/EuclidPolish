/* The data router's route table, built from the route manifest (C1).
 *
 *   createBrowserRouter(buildRoutes())
 *
 * One root layout route (the shell) holds:
 *   - every workspace: `<path>/*` (e.g. `/sky/*`, `/ensemble/:mode/*`; the
 *     home page is exactly `/`). The workspace component is lazy
 *     (`workspaceComponents`); it validates the rest of the path against the
 *     manifest itself (<Workspace>), so the router and Flask agree on what a
 *     page is;
 *   - every legacy redirect (`manifest.redirects`, exact paths) and
 *     `/app/<rest>`, as a client-side <Navigate replace> that keeps the query
 *     and hash (Flask answers the same URLs with a 308);
 *   - a catch-all Not found.
 *
 * Adding a workspace = a manifest entry + nav.ts metadata + one line in
 * `workspaceComponents` + its folder (tests fail when one is missing).
 */
import { Suspense, createElement, lazy, type ComponentType, type LazyExoticComponent } from "react";
import { Navigate, useLocation, type RouteObject } from "react-router-dom";
import { RouteError } from "./ErrorBoundary";
import { MANIFEST, redirectTarget, type RouteManifest } from "./manifest";
import { NotFound } from "./NotFound";
import { TabSkeleton } from "./workspace";

export type WorkspaceModule = { default: ComponentType };
export type WorkspaceLoader = () => Promise<WorkspaceModule>;

/** Lazy import of every workspace's `index.tsx`, by manifest id. */
export const workspaceComponents: Record<string, WorkspaceLoader> = {
  home: () => import("../workspaces/home"),
  sky: () => import("../workspaces/sky"),
  ensemble: () => import("../workspaces/ensemble"),
  realism: () => import("../workspaces/realism"),
  data: () => import("../workspaces/data"),
  figures: () => import("../workspaces/figures"),
  inspect: () => import("../workspaces/inspect"),
  ops: () => import("../workspaces/ops"),
  settings: () => import("../workspaces/settings"),
};

/** Client-side legacy redirect: the manifest target with query + hash kept. */
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
    const Component = lazyWorkspace(load);
    const element = createElement(Suspense, { fallback: createElement(TabSkeleton) }, createElement(Component));
    children.push(ws.path === "/"
      ? { path: "/", element }
      : { path: `${ws.path}/*`, element });
  }
  for (const from of Object.keys(manifest.redirects ?? {})) {
    if (from === "/") continue;
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
