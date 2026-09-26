/* The workspace rail (spec §4): brand, the nine workspaces from the route
 * manifest with their icons, a running-jobs badge on Ops and a "server
 * behind HEAD" badge on Settings, and the collapse toggle (persisted in
 * prefs). Below 900 px the shell shows it in a drawer instead. */
import { Link, useLocation } from "react-router-dom";
import { useJobsFeed } from "../api/jobs";
import { usePrefs } from "../state/prefs";
import { Icon, IconButton, Tooltip } from "../ui";
import { MANIFEST, matchPage } from "./manifest";
import { WORKSPACE_META, landingPath } from "./nav";
import { useVersion } from "./status";

export function Rail({ collapsed = false, inDrawer = false, onNavigate }: {
  collapsed?: boolean; inDrawer?: boolean; onNavigate?: () => void;
}) {
  const { pathname } = useLocation();
  const current = matchPage(pathname)?.workspace ?? null;
  const running = useJobsFeed().runningCount;
  const behind = !!useVersion().data?.behind;
  const toggleRail = usePrefs((s) => s.toggleRail);
  const badges: Record<string, { text: string; label: string } | null> = {
    ops: running > 0 ? { text: String(running), label: `${running} running job${running === 1 ? "" : "s"}` } : null,
    settings: behind ? { text: "!", label: "server behind HEAD" } : null,
  };
  return (
    <nav className="rail" aria-label="Workspaces" data-collapsed={collapsed || undefined}>
      <Link to="/" className="rail__brand" onClick={onNavigate} aria-label="EuclidPolish home">
        <span className="rail__logo" aria-hidden="true">◎</span>
        <span className="rail__name">
          <span className="rail__title">EUCLID<span>POLISH</span></span>
          <span className="rail__tag">super-resolution console</span>
        </span>
      </Link>
      <ul className="rail__list">
        {MANIFEST.workspaces.map((ws) => {
          const meta = WORKSPACE_META[ws.id];
          const active = current === ws.id;
          const badge = badges[ws.id] ?? null;
          const link = (
            <Link to={landingPath(ws.id)} className="rail__item" data-active={active || undefined}
              aria-current={active ? "page" : undefined} onClick={onNavigate}>
              <Icon name={meta.icon} size={17} />
              <span className="rail__label">{ws.label}</span>
              {badge && (
                <span className="rail__badge" title={badge.label}>
                  <span aria-hidden="true">{badge.text}</span>
                  <span className="sr-only">{badge.label}</span>
                </span>
              )}
            </Link>
          );
          return (
            <li key={ws.id}>
              {collapsed ? <Tooltip content={ws.label} side="right">{link}</Tooltip> : link}
            </li>
          );
        })}
      </ul>
      {!inDrawer && (
        <div className="rail__foot">
          <IconButton icon="sidebar" size="sm" label={collapsed ? "Expand navigation" : "Collapse navigation"}
            tooltipSide="right" onClick={toggleRail} />
        </div>
      )}
    </nav>
  );
}
