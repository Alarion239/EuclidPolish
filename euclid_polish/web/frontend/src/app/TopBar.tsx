/* The top bar (spec §4), one line at every width: breadcrumbs (workspace ›
 * tab › inspected entity), the ⌘K palette trigger, "Reload to update" when
 * the console was rebuilt under this page, the FASRC connection badge (→
 * Settings › Connections) — or, while the local server is not answering, the
 * calm "Server not responding — retrying" status in its place — the job tray,
 * the Display panel button, the theme toggle and the shortcut sheet.
 * `VersionBanner` is the "backend code changed — restart" strip at the top of
 * the stage (from GET /api/version, C3); it scrolls away with the page. */
import { useEffect, useRef, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { useServerHealth } from "../api/query";
import { formatDateTime, formatRelative } from "../format";
import { useInspector } from "../state/inspector";
import { usePrefs, useResolvedTheme } from "../state/prefs";
import { readStorage, writeStorage } from "../state/storage";
import { Button, Callout, Icon, IconButton, Kbd, Tooltip } from "../ui";
import { inspectorTitle, useInspectorRegistry } from "./inspector";
import { JobTray } from "./JobTray";
import { describePath, landingPath, pagePath } from "./nav";
import { openDisplayPanel, openPalette, openShortcutSheet, useShellUi } from "./shellStore";
import { bannerKey, useConsoleUpdate, useFasrcStatus, useVersion } from "./status";

/** While the server is not answering, ask it again this often (a GET of the
 *  version: no job, nothing heavy). */
export const SERVER_PROBE_MS = 5_000;

export function Breadcrumbs() {
  const { pathname } = useLocation();
  const info = describePath(pathname);
  const current = useInspector((s) => (s.open ? s.current : null));
  useInspectorRegistry((s) => s.kinds);
  if (!info) return <nav className="crumbs" aria-label="Breadcrumbs"><span className="crumbs__here">Not found</span></nav>;
  const { match } = info;
  // "Ensemble (starfull)": the regime is part of where you are.
  const ws = info.paramLabels.length ? `${info.workspaceLabel} (${info.paramLabels.join(", ")})` : info.workspaceLabel;
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

/** "Server not responding — retrying": shown in the FASRC badge's place
 *  (the cluster state is unknown while the local server is gone). Probes the
 *  server every SERVER_PROBE_MS while the tab is visible; a click retries now.
 *  The live region announces the outage and the recovery. */
export function ServerStatus() {
  const health = useServerHealth();
  const version = useVersion();
  const { reload } = version;
  const { down } = health;
  useEffect(() => {
    if (!down) return undefined;
    const id = setInterval(() => {
      if (typeof document === "undefined" || document.visibilityState !== "hidden") reload();
    }, SERVER_PROBE_MS);
    return () => clearInterval(id);
  }, [down, reload]);
  // Announce the recovery too (only after an outage this page saw).
  const prevDown = useRef(down);
  const [recovered, setRecovered] = useState(false);
  useEffect(() => {
    if (prevDown.current !== down) setRecovered(!down);
    prevDown.current = down;
  }, [down]);
  const since = health.lastOkAt
    ? `It last answered at ${formatDateTime(health.lastOkAt, { seconds: true }).slice(11)} (${formatRelative(health.lastOkAt)}).`
    : "It has not answered since this page opened.";
  const tip = `The local server is not responding. ${since} Pages keep showing the last data they received. `
    + `Retrying every ${SERVER_PROBE_MS / 1000} s — click to retry now.`;
  return (
    <>
      <span className="sr-only" role="status">
        {down ? "Server not responding — retrying" : recovered ? "The server is answering again" : ""}
      </span>
      {down && (
        <Tooltip content={tip}>
          <button type="button" className="topbar__server" onClick={reload}
            aria-label="Server not responding — retrying. Retry now" aria-busy={version.fetching || undefined}>
            <span className="topbar__dot" aria-hidden="true" />
            <span className="topbar__server-label">Server not responding</span>
            <span className="topbar__server-more" aria-hidden="true">— retrying</span>
          </button>
        </Tooltip>
      )}
    </>
  );
}

/** "Reload to update": the console was rebuilt since this page loaded (its
 *  lazy chunks may be gone, so the next new page could fail to load). Only
 *  for a page served from the build (`fromBuild`, see useConsoleUpdate). */
export function UpdateNotice({ fromBuild, pageEntry }: { fromBuild?: boolean; pageEntry?: string | null }) {
  const updated = useConsoleUpdate(fromBuild, pageEntry);
  if (!updated) return null;
  return (
    <Tooltip content="The console was rebuilt since this page opened. Reload to use the new version.">
      <button type="button" className="topbar__update" onClick={() => window.location.reload()}>
        <Icon name="reset" size={14} />
        <span className="topbar__update-label">Reload to update</span>
      </button>
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
  const down = useServerHealth().down;
  return (
    <header className="topbar" data-server={down ? "down" : undefined}>
      {narrow && (
        <IconButton icon="menu" label="Open navigation" onClick={() => useShellUi.getState().openOnly("drawer")} />
      )}
      <Breadcrumbs />
      <button type="button" className="topbar__search" onClick={openPalette} aria-label="Search pages and actions (⌘K)">
        <Icon name="search" size={14} />
        <span className="topbar__search-text">Search or jump to…</span>
        <Kbd keys="mod+k" />
      </button>
      <UpdateNotice />
      <ServerStatus />
      {!down && <ConnectionBadge />}
      <JobTray />
      <IconButton icon="contrast" label="Display settings" onClick={openDisplayPanel} />
      <ThemeToggle />
      <IconButton icon="keyboard" label="Keyboard shortcuts (?)" onClick={openShortcutSheet} />
    </header>
  );
}

/** Where the dismissed restart banner is remembered (this browser). */
export const BANNER_DISMISSED_KEY = "ep.restartBanner.dismissed";

/** The "restart" strip: a backend Python file the server loaded changed on
 *  disk (C3 `behind`). A dismissal is remembered in this browser until the
 *  server restarts or another file changes (`bannerKey`). */
export function VersionBanner() {
  const v = useVersion().data;
  const [dismissed, setDismissed] = useState<string | null>(() => readStorage(BANNER_DISMISSED_KEY));
  const [open, setOpen] = useState(false);
  const files = v?.changed_files ?? [];
  if (!v?.behind) return null;
  const key = bannerKey(v);
  if (dismissed === key) return null;
  const more = Math.max(0, (v.changed_count ?? files.length) - files.length);
  const dismiss = () => { writeStorage(BANNER_DISMISSED_KEY, key); setDismissed(key); };
  return (
    <div className="shell__banner">
      <Callout tone="warn" onDismiss={dismiss}
        title="Backend code changed since the server started — restart it to load the new code."
        action={(
          <Button size="sm" variant="ghost" aria-expanded={open} aria-controls="version-banner-files"
            onClick={() => setOpen((o) => !o)}>
            Details
          </Button>
        )}>
        {open && (
          <div id="version-banner-files" className="shell__banner-files">
            <p>{files.length ? "Changed on disk after they were loaded:" : "Changed files are not listed by this server."}</p>
            {files.length > 0 && (
              <ul>{files.map((f) => <li key={f}><code className="mono">{f}</code></li>)}</ul>
            )}
            {more > 0 && <p>and {more} more.</p>}
            <p>
              Server started {formatDateTime(v.started_at)}{v.pid ? `, process ${v.pid}` : ""}.
              {" "}<Link to="/settings/about">Server details</Link>
            </p>
          </div>
        )}
      </Callout>
    </div>
  );
}
