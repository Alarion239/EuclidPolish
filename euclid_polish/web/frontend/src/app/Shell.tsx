/* The console shell: the data router's root layout route.
 *
 *   ┌ rail ┬ top bar: breadcrumbs, ⌘K, FASRC, jobs, display, theme, ?  ─────┐
 *   │      ├ stage (scrolls) ─────────────────────────╥ inspector (resizable) │
 *   │      │  restart banner (only when backend code  ║                       │
 *   │      │  changed; scrolls away with the page)    ║                       │
 *   │      │  page (<Outlet/>: the workspace; gets    ║                       │
 *   │      │  the height left under the banner)       ║                       │
 *   └──────┴──────────────────────────────────────────╨───────────────────────┘
 *
 * Only the one-line top bar stays put; nothing else is pinned over the
 * images.
 *
 * It mounts, exactly once: `UiProvider` (tooltips, toasts, confirm() host —
 * inside the router so their content can use <Link>, FOUNDATION §9.1), the
 * command palette (+ the global "Run a job" actions), the ? sheet, the
 * Display panel, the global shortcuts, the `?inspect=` sync, local and SLURM
 * job toasts, `document.title` and the stage scroll management. Below 900 px the rail becomes a drawer and the inspector a
 * sheet. The inspector width is persisted (prefs.inspectorWidth). While the
 * Display sheet is open (`data-display`) the body gives up the sheet's width
 * (from 640 px), so the images refit beside it rather than under it.
 */
import * as RDialog from "@radix-ui/react-dialog";
import { useEffect, useRef } from "react";
import { Group, Panel, Separator, type PanelImperativeHandle } from "react-resizable-panels";
import { Outlet, useLocation } from "react-router-dom";
import { useMediaQuery } from "../hooks/useMediaQuery";
import { useInspector } from "../state/inspector";
import { INSPECTOR_WIDTH_RANGE, usePrefs } from "../state/prefs";
import { UiProvider } from "../ui";
import { CommandPalette } from "./CommandPalette";
import { DisplayPanel } from "./DisplayPanel";
import { GlobalShortcuts } from "./GlobalShortcuts";
import { InspectorPanel } from "./InspectorPanel";
import { registerInspector, useInspectorUrlSync } from "./inspector";
import { JobInspector, jobTitle } from "./inspectors/JobInspector";
import { useJobToasts, useSlurmToasts } from "./JobTray";
import { pageTitle } from "./nav";
import { Rail } from "./Rail";
import { RunActions } from "./RunActions";
import { useShellUi } from "./shellStore";
import { ShortcutSheet } from "./ShortcutSheet";
import { TopBar, VersionBanner } from "./TopBar";
import { useStageScroll } from "./useStageScroll";
import { viewerTakesEscape } from "./viewerEscape";
import "./shell.css";
// Workspace inspector kinds register at app start (their components are
// lazy chunks), so a cold `?inspect=<kind>:<id>` link opens from any page.
import "../workspaces/data/register";
import "../workspaces/ensemble/register";
import "../workspaces/figures/register";
import "../workspaces/inspect/register";
import "../workspaces/ops/register";
import "../workspaces/realism/register";
import "../workspaces/sky/atlas/inspectors/register";
import "../workspaces/sky/results/register";

/* Built-in inspector kinds (workspaces register theirs when they load). */
registerInspector("job", JobInspector, { title: jobTitle });

export const NARROW_QUERY = "(max-width: 899px)";

function RailDrawer() {
  const open = useShellUi((s) => s.drawer);
  const close = () => useShellUi.getState().setOpen("drawer", false);
  return (
    <RDialog.Root open={open} onOpenChange={(v) => useShellUi.getState().setOpen("drawer", v)}>
      <RDialog.Portal>
        {/* The scrim sits UNDER the drawer (the kit's dialog overlay is above
            every drawer): a click on it closes, a click in the drawer navigates. */}
        <RDialog.Overlay className="shell__scrim" />
        <RDialog.Content className="rail-drawer" aria-describedby={undefined}>
          <RDialog.Title className="sr-only">Navigation</RDialog.Title>
          <Rail inDrawer onNavigate={close} />
        </RDialog.Content>
      </RDialog.Portal>
    </RDialog.Root>
  );
}

function InspectorSheet() {
  const open = useInspector((s) => s.open && s.current != null);
  return (
    <RDialog.Root open={open} onOpenChange={(v) => { if (!v) useInspector.getState().hide(); }}>
      <RDialog.Portal>
        {/* A light scrim under the sheet: its images stay bright and take the pointer. */}
        <RDialog.Overlay className="shell__scrim shell__scrim--light" />
        <RDialog.Content className="inspector-sheet" aria-describedby={undefined}
          onEscapeKeyDown={(e) => { if (viewerTakesEscape(document.querySelector(".inspector-sheet"))) e.preventDefault(); }}>
          <RDialog.Title className="sr-only">Inspector</RDialog.Title>
          <InspectorPanel />
        </RDialog.Content>
      </RDialog.Portal>
    </RDialog.Root>
  );
}

function ShellFrame() {
  useInspectorUrlSync();
  useJobToasts();
  useSlurmToasts();
  const location = useLocation();
  const narrow = useMediaQuery(NARROW_QUERY);
  const railCollapsed = usePrefs((s) => s.railCollapsed);
  const inspectorWidth = usePrefs((s) => s.inspectorWidth);
  const inspecting = useInspector((s) => s.open && s.current != null);
  const displayOpen = useShellUi((s) => s.display);
  const stageRef = useStageScroll<HTMLElement>();
  const inspectorRef = useRef<PanelImperativeHandle | null>(null);

  useEffect(() => { document.title = pageTitle(location.pathname); }, [location.pathname]);
  // Navigating closes the narrow-screen drawer.
  useEffect(() => { useShellUi.getState().setOpen("drawer", false); }, [location.pathname]);

  const docked = inspecting && !narrow;
  const [wMin, wMax] = INSPECTOR_WIDTH_RANGE;
  return (
    <div className="shell" data-rail={railCollapsed && !narrow ? "collapsed" : "expanded"}
      data-narrow={narrow || undefined} data-display={displayOpen || undefined}>
      <a className="skip-link" href="#main">Skip to content</a>
      {!narrow && <Rail collapsed={railCollapsed} />}
      <div className="shell__main">
        <TopBar narrow={narrow} />
        <Group orientation="horizontal" className="shell__body"
          onLayoutChanged={(_layout, meta) => {
            const size = inspectorRef.current?.getSize().inPixels;
            if (meta.isUserInteraction && size && size > 0) usePrefs.getState().set({ inspectorWidth: Math.round(size) });
          }}>
          <Panel id="stage" minSize={320} className="shell__stage-panel">
            <main id="main" className="stage" ref={stageRef} tabIndex={-1}>
              {/* in the scrolling stage, so it scrolls away with the page */}
              <VersionBanner />
              <div className="stage__page"><Outlet /></div>
            </main>
          </Panel>
          {docked && <Separator className="shell__sep" aria-label="Resize the inspector" />}
          {docked && (
            <Panel id="inspector" panelRef={inspectorRef} defaultSize={inspectorWidth}
              minSize={wMin} maxSize={wMax} groupResizeBehavior="preserve-pixel-size"
              className="shell__inspector-panel">
              <InspectorPanel />
            </Panel>
          )}
        </Group>
      </div>
      {narrow && <RailDrawer />}
      {narrow && <InspectorSheet />}
      <GlobalShortcuts />
      <RunActions />
      <CommandPalette />
      <ShortcutSheet />
      <DisplayPanel />
    </div>
  );
}

/** Root layout route: the kit hosts, then the frame. */
export function Shell() {
  return (
    <UiProvider>
      <ShellFrame />
    </UiProvider>
  );
}
