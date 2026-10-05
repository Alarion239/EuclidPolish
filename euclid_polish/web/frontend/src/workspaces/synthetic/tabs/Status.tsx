/* Synthetic › Status (landing tab): can synthetic fields be generated now,
   and is each ingredient right? First the synthetic_generate gate — "Ready to
   generate" or "Blocked by N", the ingredients the local records predate, and
   the confirmed "Generate validate+test on FASRC" (a dialog with the step
   card; the train split is one chip away). Then two groups of rows from GET
   /api/realism/overview: what generation reads (galaxies, stars, noise, PSF,
   TNG radii, the saturation rule, the training catalogue) and the diagnostic
   caches the realism checks read (galaxy plots; field statistics, which also
   owns a missing or changed real reference). Each row: a state dot, its
   name, ONE verdict number with its unit (statusModel.ts rowVerdict, from
   the galaxy / star / field statistics payloads the ingredient tabs share),
   whether the local records were built with it, its fix where one exists
   (Validate TNG radii, Rebuild field statistics, …, all confirmed), a link
   to its tab and the row inspector (fingerprints and facts). Opening the tab
   starts nothing. */
import { useState } from "react";
import { Link } from "react-router-dom";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { useFasrcStatus } from "../../../app/status";
import { formatDateTime, formatRelative } from "../../../format";
import {
  Badge, Button, Caption, Dialog, EmptyState, IconButton, Num, Page, SummaryLine, Tooltip,
} from "../../../ui";
import { useGalaxies, useOverview, usePixels, useStars, type ItemAction, type OverviewItem } from "../api";
import { LoadState, StateDot } from "../common";
import { GeneratePanel } from "../generate";
import { keyForUrl, offlinePolicy, runAction, useRealismJob, type JobKey } from "../jobs";
import {
  gateSummary, listText, recordsTickText, rowVerdict, statusGroups, type VerdictContext,
} from "../statusModel";
import "../synthetic.css";

const STATE_LABEL = { ok: "ok", warn: "needs attention", bad: "blocking", unknown: "unknown" } as const;

function FixButton({ item, action, offline }: { item: OverviewItem; action: ItemAction; offline: boolean }) {
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

function RecordsTick({ item }: { item: OverviewItem }) {
  const t = recordsTickText(item.records);
  if (!t) return <span className="syn-row__tick" aria-hidden="true" />;
  const when = item.records?.records_at ? ` Records ${formatDateTime(item.records.records_at)}` : "";
  const prior = item.records?.prior_at ? `, activated ${formatDateTime(item.records.prior_at)}.` : when ? "." : "";
  return (
    <Tooltip content={`${t.tip}${when}${prior}`}>
      <span tabIndex={0} className="syn-row__tick" data-tone={t.tone}>
        <span className="syn-row__tickmark" aria-hidden="true">{t.tone === "good" ? "✓" : t.tone === "warn" ? "!" : "?"}</span>
        {t.label}
      </span>
    </Tooltip>
  );
}

function StatusRow({ item, ctx, offline }: { item: OverviewItem; ctx: VerdictContext; offline: boolean }) {
  const verdict = rowVerdict(item, ctx);
  const problem = item.state !== "ok";
  return (
    <li className="syn-row" data-state={item.state}>
      <StateDot state={item.state} label={STATE_LABEL[item.state]} />
      <div className="syn-row__main">
        <div className="syn-row__line">
          <span className="syn-row__name">{item.label}</span>
          {verdict && <span className="syn-row__verdict" data-tone={verdict.tone}>{verdict.text}</span>}
        </div>
        {/* The OK state is the line above alone; a problem says what and why. */}
        {problem && (
          <div className="syn-row__problem">{item.title}{item.detail ? <span className="syn-row__detail"> · {item.detail}</span> : null}</div>
        )}
      </div>
      <RecordsTick item={item} />
      <div className="syn-row__actions">
        {item.action && <FixButton item={item} action={item.action} offline={offline} />}
        {item.to && (
          <Button asChild size="sm" variant="ghost" iconRight="chevronRight">
            <Link to={item.to} aria-label={`Open ${item.label}`}>Open</Link>
          </Button>
        )}
        <IconButton size="sm" icon="panelRight" label={`Inspect ${item.label}`}
          onClick={() => openInspector({ kind: "readiness", id: item.id })} />
      </div>
    </li>
  );
}

function Group({ title, sub, items, ctx, offline }: {
  title: string; sub: string; items: OverviewItem[]; ctx: VerdictContext; offline: boolean;
}) {
  if (!items.length) return null;
  return (
    <section className="syn-group" aria-label={title}>
      <header className="syn-group__head"><h2 className="syn-group__title">{title}</h2><span className="syn-group__sub">{sub}</span></header>
      <ul className="syn-rows">{items.map((i) => <StatusRow key={i.id} item={i} ctx={ctx} offline={offline} />)}</ul>
    </section>
  );
}

export default function Status() {
  const overview = useOverview();
  const data = overview.data;
  // The verdicts read the payloads the ingredient tabs share (cached GETs;
  // a row shows its prior's own number until they arrive).
  const galaxies = useGalaxies(false);
  const stars = useStars(false);
  const pixels = usePixels(false);
  const ctx: VerdictContext = { galaxies: galaxies.data, stars: stars.data, pixels: pixels.data };
  const fasrc = useFasrcStatus().data;
  const offline = fasrc ? !fasrc.ssh_connected : false;
  const [generate, setGenerate] = useState(false);
  usePageActions([
    { id: "status-refresh", label: "Refresh the synthetic status", group: "Synthetic", keywords: ["priors", "gate", "readiness"],
      run: () => overview.reload() },
    { id: "status-generate", label: "Generate validate + test records on FASRC…", group: "Synthetic",
      keywords: ["synthetic_generate", "regenerate"], disabled: !data?.gate.ready, run: () => setGenerate(true) },
    ...(data?.items ?? []).filter((i) => i.action && i.state !== "ok").map((i) => ({
      id: `status-fix-${i.id}`, label: `${i.action!.label} (${i.label})`, group: "Synthetic",
      run: () => { void runAction(i.action!); },
    })),
  ]);
  const gate = data ? gateSummary(data) : null;
  const groups = data ? statusGroups(data.items) : null;
  return (
    <Page className="rl-page syn-page">
      <LoadState loading={overview.loading && !data} error={overview.error} onRetry={overview.reload}>
        {data && gate && groups && (
          <>
            <section className="syn-gate" aria-label="Generation gate" data-ready={gate.ready}>
              <div className="syn-gate__line">
                <SummaryLine className="syn-gate__summary">
                  <Num tone={gate.ready ? "good" : "bad"}>{gate.headline}</Num>
                  {gate.ready && gate.predates.length > 0 && (
                    <> · the local records predate the <Num tone="warn">{listText(gate.predates)}</Num> ingredient{gate.predates.length > 1 ? "s" : ""}</>
                  )}
                </SummaryLine>
                <Tooltip content={gate.ready ? "Rebuild the validate and test splits with the current ingredients (the step card confirms)"
                  : "Fix the blockers first"}>
                  <span tabIndex={gate.ready ? -1 : 0}>
                    <Button variant={gate.ready ? "primary" : "default"} icon="server" disabled={!gate.ready}
                      onClick={() => setGenerate(true)}>Generate validate+test on FASRC</Button>
                  </span>
                </Tooltip>
                <IconButton size="sm" icon="reset" label="Refresh the status" loading={overview.fetching}
                  onClick={() => overview.reload()} />
              </div>
              {!gate.ready && (
                <ul className="syn-gate__blockers">
                  {gate.blockers.map((b) => <li key={b}><StateDot state="bad" /> {b}</li>)}
                </ul>
              )}
              <Caption>
                {data.records?.generated_at ? `Local test + validate records generated ${formatRelative(data.records.generated_at)}` : "No local records"}
                {` · checked ${formatRelative(data.computed_at)}`}
              </Caption>
            </section>
            <Group title="Blocks generation" sub="Galaxies and stars gate synthetic_generate; the others change what it draws"
              items={groups.generation} ctx={ctx} offline={offline} />
            <Group title="Diagnostic caches" sub="What the realism checks read" items={groups.diagnostic} ctx={ctx} offline={offline} />
            {!data.items.length && <EmptyState icon="table" title="No status rows" />}
          </>
        )}
      </LoadState>
      <Dialog open={generate} onOpenChange={setGenerate} size="lg" title="Generate synthetic records on FASRC"
        description="synthetic_generate draws every scene from the active ingredients above. Validate + test are preselected; add train for a full rebuild.">
        {generate && <GeneratePanel defaultSplits={["validate", "test"]} />}
        {data && !data.gate.ready && <Badge tone="bad">Blocked: {data.gate.message}</Badge>}
      </Dialog>
    </Page>
  );
}
