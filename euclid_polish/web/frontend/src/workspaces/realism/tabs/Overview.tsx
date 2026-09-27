/* realism/overview (spec §8.3): can synthetic fields be generated right now,
   and what is stale? The synthetic_generate gate, then ONE checklist of every
   prior and cache (GET /api/realism/overview) with its fix: activate a
   candidate, rebuild a cache, sync from FASRC, or open the tab that owns it.
   Every row opens the `readiness` inspector. */
import { Link } from "react-router-dom";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { useFasrcStatus } from "../../../app/status";
import { formatRelative } from "../../../format";
import {
  Badge, Button, Card, CardBody, CardHead, EmptyState, IconButton, Page, Tooltip,
} from "../../../ui";
import { useOverview, type ItemAction, type OverviewItem, type OverviewPayload } from "../api";
import { BarGroup, BarSpacer, LoadState, RealismBar, SkyLink, StateDot, STATE_TONE } from "../common";
import { keyForUrl, offlinePolicy, runAction, useRealismJob, type JobKey } from "../jobs";

const STATE_LABEL = { ok: "ok", warn: "attention", bad: "blocking", unknown: "unknown" } as const;

function ItemActionButton({ item, action, offline }: { item: OverviewItem; action: ItemAction; offline: boolean }) {
  const job = useRealismJob(keyForUrl(action.url) as JobKey);
  const policy = offlinePolicy(action, offline);
  const button = (
    <Button size="sm" variant={item.state === "ok" ? "ghost" : "default"} loading={job.busy}
      disabled={policy.disabled} onClick={() => void runAction(action)}>
      {action.label}
    </Button>
  );
  if (!policy.hint) return button;
  return <Tooltip content={policy.hint}>{policy.disabled ? <span tabIndex={0}>{button}</span> : button}</Tooltip>;
}

function ChecklistRow({ item, offline }: { item: OverviewItem; offline: boolean }) {
  return (
    <li className="rl-check" data-state={item.state}>
      <StateDot state={item.state} label={STATE_LABEL[item.state]} />
      <div className="rl-check__main">
        <div className="rl-check__label">{item.label}</div>
        <div className="rl-check__title">{item.title}</div>
        {item.detail && <div className="rl-check__detail">{item.detail}</div>}
      </div>
      <div className="rl-check__actions">
        {item.action && <ItemActionButton item={item} action={item.action} offline={offline} />}
        {item.to && (
          <Button asChild size="sm" variant="ghost" iconRight="chevronRight">
            <Link to={item.to}>Open</Link>
          </Button>
        )}
        <IconButton size="sm" icon="panelRight" label={`Inspect ${item.label}`}
          onClick={() => openInspector({ kind: "readiness", id: item.id })} />
      </div>
    </li>
  );
}

function Gate({ data }: { data: OverviewPayload }) {
  const gate = data.gate;
  return (
    <Card className="rl-gate" aria-label="synthetic_generate gate">
      <CardBody>
        <div className="rl-gate__row">
          <div className="rl-gate__status" data-ready={gate.ready}>
            <span className="rl-gate__step rl-mono">{gate.step}</span>
            <span className="rl-gate__verdict">{gate.ready ? "Ready to generate" : "Blocked"}</span>
          </div>
          <div className="rl-gate__counts" aria-label="Checklist summary">
            {(["bad", "warn", "unknown", "ok"] as const).map((state) => data.counts[state] > 0 && (
              <Badge key={state} size="sm" tone={STATE_TONE[state]} dot>{data.counts[state]} {STATE_LABEL[state]}</Badge>
            ))}
          </div>
          <BarSpacer />
          <Button asChild size="sm" variant={gate.ready ? "primary" : "default"} iconRight="chevronRight">
            <Link to={gate.to}>Records</Link>
          </Button>
        </div>
        {!gate.ready && (
          <ul className="rl-gate__blockers">
            {gate.blockers.map((b) => <li key={b.id + b.message}><StateDot state="bad" /> {b.message}</li>)}
          </ul>
        )}
      </CardBody>
    </Card>
  );
}

export default function Overview() {
  const overview = useOverview();
  const data = overview.data;
  const fasrc = useFasrcStatus().data;
  const offline = fasrc ? !fasrc.ssh_connected : false;
  usePageActions([
    { id: "realism-overview-refresh", label: "Refresh readiness", group: "Realism", keywords: ["priors", "gate"],
      run: () => overview.reload() },
    ...(data?.items ?? []).filter((i) => i.action && i.state !== "ok").map((i) => ({
      id: `realism-overview-${i.id}`, label: `${i.action!.label} (${i.label})`, group: "Realism",
      run: () => { void runAction(i.action!); },
    })),
  ]);
  return (
    <Page className="rl-page">
      <RealismBar label="Overview controls">
        <BarGroup label="readiness">
          <span className="rl-faint">{data ? `checked ${formatRelative(data.computed_at)}` : "…"}</span>
        </BarGroup>
        <BarSpacer />
        <SkyLink layers={["q1-tiles", "noise-positions", "archive-fields"]}
          hint="Noise positions and archive fields on the sky atlas">On sky</SkyLink>
        <IconButton size="sm" icon="reset" label="Refresh readiness" loading={overview.fetching}
          onClick={() => overview.reload()} />
      </RealismBar>
      <LoadState loading={overview.loading && !data} error={overview.error} onRetry={overview.reload}>
        {data && (
          <div className="rl-stack">
            <Gate data={data} />
            <Card>
              <CardHead title="Priors and caches" sub="What synthetic generation and the realism checks read" />
              <CardBody>
                {data.items.length ? (
                  <ul className="rl-checks">
                    {data.items.map((item) => <ChecklistRow key={item.id} item={item} offline={offline} />)}
                  </ul>
                ) : <EmptyState icon="table" title="No readiness items" />}
              </CardBody>
            </Card>
          </div>
        )}
      </LoadState>
    </Page>
  );
}
