/* Workspace contract (C8).
 *
 * Every workspace is a folder `src/workspaces/<id>/` whose `index.tsx`
 * default-exports the workspace component. It declares its tabs once, as lazy
 * modules `./tabs/<Tab>.tsx`, and renders <Workspace>:
 *
 *   const TABS = defineTabs("realism", {
 *     overview: { load: () => import("./tabs/Overview") },
 *     noise:    { load: () => import("./tabs/Noise") },
 *     …                                   // exactly the manifest's tabs
 *   });
 *   export default function Realism() { return <Workspace id="realism" tabs={TABS} />; }
 *
 * <Workspace> validates the URL against the manifest (`/sky/unknown`,
 * `/ensemble/foo` → Not found), redirects a bare workspace path to its default
 * tab (or `redirectTab`; query and hash kept), renders the router-linked tab
 * strip (<WorkspaceTabs>) and the active tab inside a per-tab error boundary
 * and a Suspense skeleton. A tabless workspace (home, inspect) passes
 * `children`. Tab labels default to `app/nav.ts`; `aside` sits right of the
 * tab strip.
 *
 * A theme or accent flip re-renders the active tab (or `children`): the route
 * elements above a workspace are static, and the legacy pages read colour
 * tokens during render (`categorical()`, `C.muted`, `surfaceRgb()`), so
 * nothing else would repaint them in the new theme (`useTokenRerender`).
 */
import {
  Suspense, cloneElement, isValidElement, lazy, type ComponentType, type LazyExoticComponent, type ReactNode,
} from "react";
import { Link, Navigate, useLocation } from "react-router-dom";
import { useUrlState } from "../hooks/useUrlState";
import { usePrefs, useResolvedTheme } from "../state/prefs";
import { Button, EmptyState, Page, Skeleton, Tabs } from "../ui";
import type { IconName } from "../ui/icons";
import { ErrorBoundary } from "./ErrorBoundary";
import { matchPage, workspace } from "./manifest";
import { landingPath, pagePath, tabLabel, workspaceLabel, workspaceMeta } from "./nav";
import { NotFound } from "./NotFound";

export type TabModule = { default: ComponentType };

export type TabSpec = {
  /** Lazy tab module; its default export is the tab component. */
  load: () => Promise<TabModule>;
  /** Overrides the nav.ts label. */
  label?: string;
  /** A count or status shown after the label. */
  badge?: ReactNode;
};

export type DefinedTab = TabSpec & { Component: LazyExoticComponent<ComponentType> };
export type WorkspaceTabDefs = Readonly<Record<string, DefinedTab>>;

/** Declare a workspace's tabs once, at module scope (the lazy components must
 *  be created once). Warns about tabs the manifest does not list. */
export function defineTabs(workspaceId: string, specs: Record<string, TabSpec>): WorkspaceTabDefs {
  const known = new Set(workspace(workspaceId).tabs);
  const out: Record<string, DefinedTab> = {};
  for (const [tab, spec] of Object.entries(specs)) {
    if (!known.has(tab)) console.warn(`workspace "${workspaceId}": tab "${tab}" is not in spa_routes.json`);
    out[tab] = { ...spec, Component: lazy(spec.load) };
  }
  return Object.freeze(out);
}

export function TabSkeleton() {
  return (
    <div className="page ws-skeleton" aria-busy="true" aria-label="Loading">
      <Skeleton width={260} height={26} />
      <Skeleton lines={4} style={{ marginTop: "var(--s5)" }} />
      <Skeleton height={220} style={{ marginTop: "var(--s5)" }} />
    </div>
  );
}

/** The router-linked tab strip of a workspace (keeps `?inspect=`). */
export function WorkspaceTabs(
  { id, base, current, tabs, aside }: {
    id: string; base: string; current: string | null; tabs?: WorkspaceTabDefs; aside?: ReactNode;
  },
) {
  const [inspect] = useUrlState("inspect", "");
  const ws = workspace(id);
  const keep = inspect ? `?${new URLSearchParams({ inspect }).toString()}` : "";
  const items = ws.tabs.map((tab) => ({
    id: tab,
    label: tabs?.[tab]?.label ?? tabLabel(id, tab),
    badge: tabs?.[tab]?.badge,
    to: `${base === "/" ? "" : base}/${tab}${keep}`,
  }));
  return (
    <div className="ws__bar">
      <Tabs value={current ?? ""} tabs={items} variant="line" aria-label={`${ws.label} tabs`}
        className="ws__tabs" />
      {aside != null && <div className="ws__aside">{aside}</div>}
    </div>
  );
}

/** Re-render the caller when the resolved theme or the accent changes
 *  (<html data-theme / data-accent> is already updated when it runs:
 *  `bindPrefsToDocument` applies the prefs synchronously in the store
 *  listener, before React re-renders). Returns "<theme>/<accent>". */
export function useTokenRerender(): string {
  const theme = useResolvedTheme();
  const accent = usePrefs((s) => s.accent);
  return `${theme}/${accent}`;
}

export function Workspace(
  { id, tabs, aside, redirectTab, children }: {
    id: string;
    tabs?: WorkspaceTabDefs;
    /** Controls right of the tab strip (e.g. the ensemble regime switch). */
    aside?: ReactNode;
    /** Where a bare workspace path goes instead of the manifest default tab
     *  (ignored unless it is one of the workspace's tabs). */
    redirectTab?: string | null;
    /** Content of a workspace without tabs. */
    children?: ReactNode;
  },
) {
  useTokenRerender();
  const location = useLocation();
  const m = matchPage(location.pathname);
  if (!m || m.workspace !== id) return <NotFound />;
  const ws = workspace(id);
  if (ws.tabs.length && !m.tab) {
    const tab = redirectTab && ws.tabs.includes(redirectTab) ? redirectTab : null;
    const to = `${pagePath(id, { tab, params: m.params })}${location.search}${location.hash}`;
    return <Navigate replace to={to} />;
  }
  const tab = m.tab ? tabs?.[m.tab] : null;
  const label = m.tab ? `${workspaceLabel(id)} › ${tabLabel(id, m.tab)}` : workspaceLabel(id);
  let body: ReactNode;
  if (tab) body = <tab.Component />;
  else if (m.tab) body = <PendingTab workspace={id} tab={m.tab} />;
  // `children` was created by the (static) workspace component: clone it so
  // this render — e.g. a theme flip — reaches the page instead of bailing out.
  else body = isValidElement(children) ? cloneElement(children) : children;
  return (
    <div className={`ws ws--${id}`} data-workspace={id}>
      {ws.tabs.length > 0 && (
        <WorkspaceTabs id={id} base={m.base} current={m.tab} tabs={tabs} aside={aside} />
      )}
      <div className="ws__body">
        <ErrorBoundary resetKey={location.pathname} label={label}>
          <Suspense fallback={<TabSkeleton />}>{body}</Suspense>
        </ErrorBoundary>
      </div>
    </div>
  );
}

/** A tab whose redesigned page arrives in phase 3 (no legacy page to adapt). */
export function PendingTab(
  { workspace: id, tab, icon, children, links }: {
    workspace: string; tab: string; icon?: IconName; children?: ReactNode;
    /** Related pages that already exist: [label, path]. */
    links?: [string, string][];
  },
) {
  const meta = workspaceMeta(id);
  const description = meta.tabs[tab]?.description;
  return (
    <Page>
      <EmptyState icon={icon ?? meta.icon} title={`${tabLabel(id, tab)} arrives in phase 3`}
        action={links?.length ? (
          <div className="row" style={{ gap: "var(--s2)", justifyContent: "center" }}>
            {links.map(([text, to]) => (
              <Button key={to} asChild size="sm"><Link to={to}>{text}</Link></Button>
            ))}
          </div>
        ) : (
          <Button asChild size="sm" variant="ghost"><Link to={landingPath(id)}>{workspaceLabel(id)} home</Link></Button>
        )}>
        {description ? `${description}. ` : ""}
        {children ?? "This tab is part of the console rework and is not built yet."}
      </EmptyState>
    </Page>
  );
}
