/* The console shell: the data router's root layout route.
 *
 *   ┌ rail ┬ top bar (breadcrumbs · ⌘K · FASRC · jobs · display · theme · ?) ┐
 *   │      ├ version banner (server behind HEAD)                              │
 *   │      ├ stage (<Outlet/>: the workspace) ║ inspector (resizable)         │
 *   └──────┴──────────────────────────────────────────────────────────────────┘
 *
 * It mounts, exactly once: `UiProvider` (tooltips, toasts, confirm() host —
 * inside the router so their content can use <Link>, FOUNDATION §9.1), the
 * command palette, the ? sheet, the Display panel, the global shortcuts, the
 * `?inspect=` sync, job toasts, `document.title` and the stage scroll
 * management. Below 900 px the rail becomes a drawer and the inspector a
 * sheet. The inspector width is persisted (prefs.inspectorWidth).
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
import { useJobToasts } from "./JobTray";
import { pageTitle } from "./nav";
import { Rail } from "./Rail";
import { useShellUi } from "./shellStore";
import { ShortcutSheet } from "./ShortcutSheet";
import { TopBar, VersionBanner } from "./TopBar";
import { useStageScroll } from "./useStageScroll";
import "./shell.css";

/* Built-in inspector kinds (workspaces register theirs when they load). */
registerInspector("job", JobInspector, { title: jobTitle });

export const NARROW_QUERY = "(max-width: 899px)";

function RailDrawer() {
  const open = useShellUi((s) => s.drawer);
  const close = () => useShellUi.getState().setOpen("drawer", false);
  return (
    <RDialog.Root open={open} onOpenChange={(v) => useShellUi.getState().setOpen("drawer", v)}>
      <RDialog.Portal>
        <RDialog.Overlay className="ui-dialog__overlay" />
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
        <RDialog.Overlay className="ui-dialog__overlay" />
        <RDialog.Content className="inspector-sheet" aria-describedby={undefined}>
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
  const location = useLocation();
  const narrow = useMediaQuery(NARROW_QUERY);
  const railCollapsed = usePrefs((s) => s.railCollapsed);
  const inspectorWidth = usePrefs((s) => s.inspectorWidth);
  const inspecting = useInspector((s) => s.open && s.current != null);
  const stageRef = useStageScroll<HTMLElement>();
  const inspectorRef = useRef<PanelImperativeHandle | null>(null);

  useEffect(() => { document.title = pageTitle(location.pathname); }, [location.pathname]);
  // Navigating closes the narrow-screen drawer.
  useEffect(() => { useShellUi.getState().setOpen("drawer", false); }, [location.pathname]);

  const docked = inspecting && !narrow;
  const [wMin, wMax] = INSPECTOR_WIDTH_RANGE;
  return (
    <div className="shell" data-rail={railCollapsed && !narrow ? "collapsed" : "expanded"}
      data-narrow={narrow || undefined}>
      <a className="skip-link" href="#main">Skip to content</a>
      {!narrow && <Rail collapsed={railCollapsed} />}
      <div className="shell__main">
        <TopBar narrow={narrow} />
        <VersionBanner />
        <Group orientation="horizontal" className="shell__body"
          onLayoutChanged={(_layout, meta) => {
            const size = inspectorRef.current?.getSize().inPixels;
            if (meta.isUserInteraction && size && size > 0) usePrefs.getState().set({ inspectorWidth: Math.round(size) });
          }}>
          <Panel id="stage" minSize={320} className="shell__stage-panel">
            <main id="main" className="stage" ref={stageRef} tabIndex={-1}>
              <Outlet />
            </main>
          </Panel>
          {docked && <Separator className="shell__sep" />}
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
