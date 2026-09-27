/* The health checks of GET /api/system/alerts (routes/system.py): the list
 * Home shows (warn/bad first, the rest folded), each row opening its
 * inspector (`check:<id>`) and offering the check's page and one-click fix
 * (a local job, confirmed first). */
import { useState } from "react";
import { Link } from "react-router-dom";
import { openInspector } from "../../app/inspector";
import { startJob } from "../../app/RunActions";
import {
  useSystemAlerts, type CheckState, type HealthAction, type HealthCheck, type SystemAlerts,
} from "../../app/status";
import { formatBytes, formatDateTime, formatRelative } from "../../format";
import { Badge, Button, Callout, DefList, EmptyState, Icon, JsonTree, Skeleton, type Tone } from "../../ui";

export const STATE_TONE: Record<CheckState, Tone> = { ok: "good", warn: "warn", bad: "bad", unknown: "neutral" };
const STATE_ICON = { ok: "success", warn: "warn", bad: "error", unknown: "help" } as const;

export async function runCheckAction(check: HealthCheck, action: HealthAction): Promise<void> {
  await startJob({
    key: `check:${check.id}`, label: `${action.label} (${check.label})`, url: action.url,
    data: action.params ?? {},
    question: action.confirm ? { title: `${action.label}?`, message: action.confirm, confirmLabel: action.label } : undefined,
  });
}

function CheckRow({ check }: { check: HealthCheck }) {
  const [busy, setBusy] = useState(false);
  const act = async () => {
    if (!check.action) return;
    setBusy(true);
    try { await runCheckAction(check, check.action); } finally { setBusy(false); }
  };
  return (
    <li className="health__row" data-state={check.state}>
      <Icon name={STATE_ICON[check.state]} size={16} className="health__icon" />
      <div className="health__main">
        <button type="button" className="health__title" onClick={() => openInspector({ kind: "check", id: check.id })}
          title={check.detail ?? check.title}>
          {check.title}
        </button>
        {check.detail && <span className="health__detail">{check.detail}</span>}
      </div>
      <div className="health__side">
        <Badge tone={STATE_TONE[check.state]} size="sm">{check.label}</Badge>
        {check.action && (
          <Button size="sm" variant={check.state === "ok" ? "ghost" : "default"} loading={busy} onClick={act}>
            {check.action.label}
          </Button>
        )}
        {check.to && (
          <Button asChild size="sm" variant="ghost" aria-label={`Open ${check.label}`}>
            <Link to={check.to}><Icon name="chevronRight" size={14} /></Link>
          </Button>
        )}
      </div>
    </li>
  );
}

/** Warn/bad checks, then the rest behind a disclosure. */
export function HealthList({ data, loading, error }: {
  data: SystemAlerts | null; loading: boolean; error: { message: string } | null;
}) {
  const [showAll, setShowAll] = useState(false);
  if (loading && !data) return <Skeleton lines={4} />;
  if (error && !data) return <Callout tone="bad" title="Health checks unavailable">{error.message}</Callout>;
  if (!data) return null;
  const alerts = data.alerts;
  const rest = data.checks.filter((c) => c.state !== "warn" && c.state !== "bad");
  return (
    <div className="health">
      {alerts.length === 0 && (
        <EmptyState compact icon="success" title="Everything is current">
          {rest.length} checks pass.
        </EmptyState>
      )}
      {alerts.length > 0 && <ul className="health__list">{alerts.map((c) => <CheckRow key={c.id} check={c} />)}</ul>}
      {rest.length > 0 && (
        <>
          <button type="button" className="health__more" aria-expanded={showAll} onClick={() => setShowAll((v) => !v)}>
            <Icon name={showAll ? "chevronDown" : "chevronRight"} size={14} />
            {rest.filter((c) => c.state === "ok").length} OK
            {rest.some((c) => c.state === "unknown") ? ` · ${rest.filter((c) => c.state === "unknown").length} unverified` : ""}
          </button>
          {showAll && <ul className="health__list health__list--rest">{rest.map((c) => <CheckRow key={c.id} check={c} />)}</ul>}
        </>
      )}
    </div>
  );
}

const isScalar = (v: unknown) => v == null || ["string", "number", "boolean"].includes(typeof v);
/** A fact key as a sentence-case label ("ssh_connected" → "Ssh connected"). */
const factLabel = (key: string) => {
  const words = key.replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
};
const factValue = (key: string, v: unknown) =>
  typeof v === "number" && /bytes$/.test(key) ? formatBytes(v)
    : typeof v === "string" && /(_at|_entry)$/.test(key) ? `${formatDateTime(v)} (${formatRelative(v)})`
      : String(v ?? "—");

/** Inspector kind `check:<id>`: one health check with its facts. */
export function CheckInspector({ id }: { id: string }) {
  const alerts = useSystemAlerts();
  const check = alerts.data?.checks.find((c) => c.id === id) ?? null;
  const [busy, setBusy] = useState(false);
  if (alerts.loading && !alerts.data) return <Skeleton lines={4} />;
  if (!check) return <EmptyState compact icon="info" title="Unknown check">{id}</EmptyState>;
  const facts = check.facts ?? {};
  const scalars = Object.entries(facts).filter(([, v]) => isScalar(v));
  const nested = Object.entries(facts).filter(([, v]) => !isScalar(v));
  return (
    <div className="health-insp">
      <Callout tone={STATE_TONE[check.state] === "neutral" ? "info" : STATE_TONE[check.state]} title={check.title}>
        {check.detail}
      </Callout>
      {scalars.length > 0 && <DefList dense items={scalars.map(([k, v]) => [factLabel(k), factValue(k, v)])} />}
      {nested.map(([k, v]) => (
        <div key={k}>
          <div className="eyebrow">{factLabel(k)}</div>
          <JsonTree data={v} expandDepth={2} />
        </div>
      ))}
      <div className="row" style={{ gap: "var(--s2)", flexWrap: "wrap" }}>
        {check.action && (
          <Button variant="primary" size="sm" loading={busy} onClick={async () => {
            setBusy(true);
            try { await runCheckAction(check, check.action!); } finally { setBusy(false); }
          }}>{check.action.label}</Button>
        )}
        {check.to && <Button asChild size="sm"><Link to={check.to}>Open {check.label}</Link></Button>}
      </div>
      <p className="muted" style={{ margin: 0, fontSize: "var(--fs-xs)" }}>
        checked {formatRelative(alerts.data?.computed_at ?? null)}
      </p>
    </div>
  );
}
