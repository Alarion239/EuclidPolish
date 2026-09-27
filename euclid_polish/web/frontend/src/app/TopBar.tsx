/* The top bar (spec §4): breadcrumbs (workspace › tab › inspected entity),
 * the ⌘K palette trigger, the FASRC connection badge (→ Settings ›
 * Connections), the job tray, the Display panel button, the theme toggle and
 * the shortcut sheet. `VersionBanner` is the "server behind HEAD — restart"
 * strip under it (from GET /api/version, C3). */
import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { useInspector } from "../state/inspector";
import { usePrefs, useResolvedTheme } from "../state/prefs";
import { Button, Callout, Icon, IconButton, Kbd, Tooltip } from "../ui";
import { inspectorTitle, useInspectorRegistry } from "./inspector";
import { JobTray } from "./JobTray";
import { describePath, landingPath, pagePath } from "./nav";
import { openDisplayPanel, openPalette, openShortcutSheet, useShellUi } from "./shellStore";
import { useFasrcStatus, useVersion } from "./status";

export function Breadcrumbs() {
  const { pathname } = useLocation();
  const info = describePath(pathname);
  const current = useInspector((s) => (s.open ? s.current : null));
  useInspectorRegistry((s) => s.kinds);
  if (!info) return <nav className="crumbs" aria-label="Breadcrumbs"><span className="crumbs__here">Not found</span></nav>;
  const { match } = info;
  const ws = info.paramLabels.length ? `${info.workspaceLabel} · ${info.paramLabels.join(" · ")}` : info.workspaceLabel;
  const wsTo = match.tab ? pagePath(match.workspace, { params: match.params }) : landingPath(match.workspace);
  const entity = current ? inspectorTitle(current) : null;
  // Every crumb truncates with an ellipsis (its full text is the tooltip),
  // so the top bar stays one line and never runs under its controls.
  return (
    <nav className="crumbs" aria-label="Breadcrumbs"
      title={[ws, info.tabLabel, entity].filter(Boolean).join(" › ")}>
      <ol>
        <li className="crumbs__ws">{info.tabLabel
          ? <Link to={wsTo} className="crumbs__text">{ws}</Link>
          : <span className="crumbs__here crumbs__text" aria-current="page">{ws}</span>}</li>
        {info.tabLabel && (
          <li className="crumbs__tab"><span className="crumbs__here crumbs__text" aria-current="page">{info.tabLabel}</span></li>
        )}
        {current && (
          <li className="crumbs__entity">
            <span className="crumbs__text" title={`${current.kind}:${current.id}`}>{entity}</span>
          </li>
        )}
      </ol>
    </nav>
  );
}

export function ConnectionBadge() {
  const s = useFasrcStatus();
  const ok = !!s.data?.ssh_connected;
  const state = s.loading ? "unknown" : ok ? "on" : "off";
  const tip = s.loading ? "Checking the FASRC connection…"
    : ok ? "FASRC connected"
    : s.data?.last_error ? `FASRC offline — ${s.data.last_error}` : "FASRC offline — local pages still work";
  return (
    <Tooltip content={tip}>
      <Link to="/settings/connections" className="topbar__conn" data-state={state}
        aria-label={`FASRC ${state === "on" ? "connected" : state === "off" ? "offline" : "status unknown"}`}>
        <span className="topbar__dot" aria-hidden="true" />
        <span className="topbar__conn-label">FASRC</span>
      </Link>
    </Tooltip>
  );
}

export function ThemeToggle() {
  const theme = useResolvedTheme();
  const toggle = usePrefs((s) => s.toggleTheme);
  const next = theme === "dark" ? "light" : "dark";
  return <IconButton icon={theme === "dark" ? "moon" : "sun"} label={`Switch to ${next} theme`} onClick={toggle} />;
}

export function TopBar({ narrow = false }: { narrow?: boolean }) {
  return (
    <header className="topbar">
      {narrow && (
        <IconButton icon="menu" label="Open navigation" onClick={() => useShellUi.getState().openOnly("drawer")} />
      )}
      <Breadcrumbs />
      <button type="button" className="topbar__search" onClick={openPalette} aria-label="Search pages and actions (⌘K)">
        <Icon name="search" size={14} />
        <span className="topbar__search-text">Search or jump to…</span>
        <Kbd keys="mod+k" />
      </button>
      <ConnectionBadge />
      <JobTray />
      <IconButton icon="contrast" label="Display settings" onClick={openDisplayPanel} />
      <ThemeToggle />
      <IconButton icon="keyboard" label="Keyboard shortcuts (?)" onClick={openShortcutSheet} />
    </header>
  );
}

/** "Server behind HEAD — restart" strip (dismissable per HEAD commit). */
export function VersionBanner() {
  const v = useVersion().data;
  const [dismissed, setDismissed] = useState<string | null>(null);
  if (!v?.behind || dismissed === v.head_commit) return null;
  return (
    <div className="shell__banner">
      <Callout tone="warn" title="The server is running older code"
        onDismiss={() => setDismissed(v.head_commit)}
        action={<Button asChild size="sm" variant="ghost"><Link to="/settings/about">Details</Link></Button>}>
        It started at <code className="mono">{v.boot_short ?? "?"}</code>; the checkout is at
        {" "}<code className="mono">{v.head_short ?? "?"}</code>. Restart the server to load the new code.
      </Callout>
    </div>
  );
}
