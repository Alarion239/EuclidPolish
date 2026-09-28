/* Workspace contract (C8).
 *
 * Every workspace is a folder `src/workspaces/<id>/` whose `index.tsx`
 * default-exports the workspace component. It declares its tabs once, as lazy
 * modules `./tabs/<Tab>.tsx`, and renders <Workspace>:
 *
 *   const TABS = defineTabs("synthetic", {
 *     status: { load: () => import("./tabs/Status") },
 *     noise:  { load: () => import("./tabs/Noise") },
 *     …                                   // exactly the manifest's tabs
 *   });
 *   export default function Synthetic() { return <Workspace id="synthetic" tabs={TABS} />; }
 *
 * <Workspace> validates the URL against the manifest (`/sky/unknown`,
 * `/models/foo` → Not found), redirects a bare workspace path to its default
 * tab (or `redirectTab`; query and hash kept), renders the router-linked tab
 * strip (<WorkspaceTabs>) and the active tab inside a per-tab error boundary
 * and a Suspense skeleton, under a visually hidden h1 ("Members, Models
 * (starless)"; dropped by CSS when the page has its own h1, e.g. a PageHead).
 * The strip stays one line and never cuts a label: a fixed run of leading
 * tabs is shown whole, the active tab is always visible (in one reserved
 * slot when it is past the run, so tabs never trade places), the rest sit in
 * a "More" menu of router links inside the strip's <nav> (`tabFit.ts`). A tabless workspace (home, inspect) passes
 * `children`. Tab labels default to `app/nav.ts`; `aside` sits right of the
 * tab strip.
 *
 * A theme or accent flip re-renders the active tab (or `children`): the route
 * elements above a workspace are static, and the legacy pages read colour
 * tokens during render (`categorical()`, `C.muted`, `surfaceRgb()`), so
 * nothing else would repaint them in the new theme (`useTokenRerender`).
 */
import {
  Suspense, cloneElement, isValidElement, lazy, useEffect, useLayoutEffect, useRef, useState,
  type ComponentType, type LazyExoticComponent, type ReactNode,
} from "react";
import * as RMenu from "@radix-ui/react-dropdown-menu";
import { Link, Navigate, useLocation } from "react-router-dom";
import { useUrlState } from "../hooks/useUrlState";
import { usePrefs, useResolvedTheme } from "../state/prefs";
import { Button, EmptyState, Icon, Page, Skeleton } from "../ui";
import type { IconName } from "../ui/icons";
import { ErrorBoundary } from "./ErrorBoundary";
import { matchPage, workspace } from "./manifest";
import { landingPath, pageHeading, pagePath, tabLabel, workspaceLabel, workspaceMeta } from "./nav";
import { NotFound } from "./NotFound";
import { fitTabs, sameIndices } from "./tabFit";

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

/** Width reserved for the "More" button until it has been measured. */
const MORE_ESTIMATE_PX = 76;

/** Measure the strip's tabs and fit them (see `tabFit.ts`). `signature`
 *  changes with the labels/badges (→ re-measure every tab); the strip's own
 *  resizes only re-fit. Returns the visible indices (null until measured:
 *  every tab is rendered for that first, pre-paint measurement). */
function useTabFit(signature: string, count: number, active: number) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const widths = useRef<number[]>([]);
  const moreWidth = useRef(MORE_ESTIMATE_PX);
  const [visible, setVisible] = useState<number[] | null>(null);
  const [tick, setTick] = useState(0);
  const density = usePrefs((s) => s.density);
  const measured = useRef<string | null>(null);
  const key = `${signature}|${density}`;

  useLayoutEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    // New labels (or density, or the web fonts arrived): render every tab
    // once and measure them all, before the browser paints.
    if (measured.current !== key && visible !== null) { setVisible(null); return; }
    const tabEls = [...wrap.querySelectorAll<HTMLElement>(".ws__tabs .ui-tab")];
    const shown = visible ?? Array.from({ length: count }, (_, i) => i);
    if (visible === null) {
      if (tabEls.length !== count) return;
      widths.current = tabEls.map((el) => el.getBoundingClientRect().width);
      measured.current = key;
    } else {
      // Keep the visible tabs' widths current (fonts, badges); hidden ones keep theirs.
      tabEls.forEach((el, j) => { if (shown[j] != null) widths.current[shown[j]] = el.getBoundingClientRect().width; });
    }
    const more = wrap.querySelector<HTMLElement>(".ws__more");
    if (more) moreWidth.current = more.getBoundingClientRect().width;
    const next = fitTabs(widths.current, wrap.clientWidth, moreWidth.current, active);
    if (!sameIndices(next, visible)) setVisible(next);
  }, [key, visible, count, active, tick]);

  // Re-fit when the strip is resized; re-measure once the web fonts are in.
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return undefined;
    let raf = 0;
    const ro = typeof ResizeObserver === "function"
      ? new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => setTick((t) => t + 1)); })
      : null;
    ro?.observe(wrap);
    let alive = true;
    void document.fonts?.ready.then(() => { if (alive) { measured.current = null; setVisible(null); } });
    return () => { alive = false; ro?.disconnect(); cancelAnimationFrame(raf); };
  }, []);

  return { wrapRef, visible };
}

/** The router-linked tab strip of a workspace (keeps `?inspect=`). One line
 *  at every width: a fixed run of leading tabs, the active tab when it is
 *  past them, then "More" for the rest (never cut; `tabFit.ts`). The More
 *  button sits inside the strip's <nav> landmark and its items are router
 *  links, so a middle- or ⌘-click opens a tab in a new browser tab. */
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
  const active = current ? ws.tabs.indexOf(current) : -1;
  const signature = items.map((t) => `${t.label}#${typeof t.badge === "string" || typeof t.badge === "number" ? t.badge : t.badge != null ? "*" : ""}`).join("|");
  const { wrapRef, visible } = useTabFit(signature, items.length, active);
  const shown = visible ? visible.map((i) => items[i]) : items;
  const overflow = visible ? items.filter((_, i) => !visible.includes(i)) : [];
  return (
    <div className="ws__bar">
      <nav className="ws__tabs-wrap" ref={wrapRef} aria-label={`${ws.label} tabs`}>
        <div className="ui-tabs ui-tabs--line ws__tabs">
          {shown.map((t) => (
            <Link key={t.id} to={t.to} className="ui-tab" data-on={t.id === current}
              aria-current={t.id === current ? "page" : undefined}>
              {t.label}{t.badge != null && <span className="ui-tab__badge">{t.badge}</span>}
            </Link>
          ))}
        </div>
        {overflow.length > 0 && (
          <RMenu.Root modal={false}>
            <RMenu.Trigger asChild>
              <button type="button" className="ui-tab ws__more"
                aria-label={`More tabs: ${overflow.map((t) => t.label).join(", ")}`}>
                More<Icon name="chevronDown" size={14} />
              </button>
            </RMenu.Trigger>
            <RMenu.Portal>
              <RMenu.Content className="ui-menu" align="end" side="bottom" sideOffset={6}
                collisionPadding={8} aria-label={`More ${ws.label} tabs`}>
                {overflow.map((t) => (
                  <RMenu.Item key={t.id} asChild className="ui-menu__item">
                    <Link to={t.to}>
                      <span className="ui-menu__text">{t.label}</span>
                      {t.badge != null && <span className="ui-tab__badge">{t.badge}</span>}
                    </Link>
                  </RMenu.Item>
                ))}
              </RMenu.Content>
            </RMenu.Portal>
          </RMenu.Root>
        )}
      </nav>
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
      {/* The page's h1 for screen readers and the outline; hidden (CSS) when
          the page renders its own h1. */}
      <h1 className="sr-only ws__h1">{pageHeading(location.pathname)}</h1>
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
