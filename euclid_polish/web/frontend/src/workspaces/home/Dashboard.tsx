/* Home (spec §8.1; statistics rule of the 2026-09-27 console spec): one
 * sentence with the production model's number against its references, a
 * "Running now" line while jobs run, notes for FASRC / the production
 * endpoint / the disk ONLY when they are broken (a changed backend is the
 * shell's notice strip), then the health checks, recent jobs, quick
 * actions and a small sky. Everything works offline.
 *
 * Numbers and their definitions are in homeModel.ts: the knee-integrated
 * PSNR of the production spatial gate (/ensemble/knee-psnr.json) with its
 * gain over the best member and the plain mean; without knee curves, its
 * test PSNR from eval_summary.json's spatial_gate_* keys
 * (/api/system/production); the active STARFULL member count from the
 * regime labels (/api/models). Health checks: /api/system/alerts
 * (routes/system.py); free space: /api/system. Quick actions › Log to
 * tracking opens the shared dialog pre-filled from the tracking check's
 * unlogged results (homeModel.ts). */
import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { useJobsFeed } from "../../api/jobs";
import { invalidate, useResource } from "../../api/query";
import { registerInspector } from "../../app/inspector";
import { JobList, SlurmRow } from "../../app/JobTray";
import { usePageActions } from "../../app/palette";
import { refreshHealth, useRunJobs } from "../../app/RunActions";
import { useFasrcStatus, useSystemAlerts } from "../../app/status";
import { formatBytes, formatNumber, formatRelative } from "../../format";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, IconButton, Num, Page, Skeleton, SummaryLine,
} from "../../ui";
import { LogToTrackingDialog } from "../shared/LogToTracking";
import { PageLead } from "../shared/PageLead";
import { CheckInspector, HealthList } from "./Health";
import {
  kneeHeadline, productionFromStatus, productionHeadline, productionModel, runningItems, starfullMembers, trackingCatchUpNote,
  unloggedItems, type EnsembleStatusSlice, type KneeHeadline, type KneePayload, type ModelsCatalog, type ProductionHeadline,
  type ProductionPayload,
} from "./homeModel";
import { SkyOverview } from "./SkyOverview";
import "./home.css";

registerInspector("check", CheckInspector, { title: (id) => `Health check · ${id}` });

type SystemInfo = {
  disk?: { free_bytes: number; total_bytes: number; used_fraction: number | null; level: "ok" | "warn" | "bad" | "unknown" };
};

const db = (v: number | null | undefined) => formatNumber(v, { digits: 2 });

const KNEE_HINT = "Mean over VIS, Y, J, H of each model's PSNR integrated over log asinh knee 0.1–10⁴ e⁻ "
  + "(test cubes). The knee-independent way to compare models.";
const TEST_HINT = "Asinh-stretched test PSNR from the last STARFULL evaluation (eval_summary.json spatial_gate_* "
  + "keys; the RBF combiner is never the headline).";
const MEMBERS_HINT = "Registry-active members of the STARFULL regime (the default). Starless members are an opt-in "
  + "regime and never enter the production gate.";

/** "+1.02 dB over the best member (#196)"; a loss reads "0.30 dB under …" in warn.
 *  A non-breaking space keeps each number with its unit. */
function gain(v: number | null | undefined, what: string): ReactNode {
  if (v == null || !Number.isFinite(v)) return null;
  return v >= 0
    ? <><Num>{formatNumber(v, { digits: 2, signed: true })}</Num>{"\u00a0"}dB over {what}</>
    : <><Num tone="warn">{formatNumber(-v, { digits: 2 })}</Num>{"\u00a0"}dB under {what}</>;
}

/** "a, b and c" over the non-empty parts. */
function joinClauses(parts: ReactNode[]): ReactNode[] {
  const ok = parts.filter((p) => p != null);
  return ok.flatMap((p, i) => [i === 0 ? null : i === ok.length - 1 ? " and " : ", ", <span key={i}>{p}</span>]);
}

/** The production model's one sentence: its knee-integrated PSNR against the
 *  best member and the plain mean (or, without knee curves, its test PSNR),
 *  then the member count and the evaluation time. Null without a number. */
function ProductionSummary({ knee, prod, members, evaluatedAt }: {
  knee: KneeHeadline | null; prod: ProductionHeadline | null; members: number | null; evaluatedAt: string | null;
}) {
  let head: ReactNode;
  let stale = !!prod?.stale;
  const kneeValue = knee?.gate ?? knee?.mean ?? null;
  if (knee && kneeValue != null) {
    const best = knee.best ? `the best member (#${knee.best.name.replace(/^member_/, "")})` : "the best member";
    head = <>
      {knee.gate != null ? "Production gate" : "Plain mean"} <Num>{db(kneeValue)}</Num>{"\u00a0"}dB{" "}
      <Link to="/ensemble/starfull/knee" title={KNEE_HINT}>integrated PSNR</Link>
      {(knee.vsBest != null || knee.vsMean != null) && ", "}
      {joinClauses([gain(knee.vsBest, best), knee.gate != null ? gain(knee.vsMean, "the plain mean") : null])}
    </>;
    stale = stale || knee.stale;
  } else if (prod) {
    const clauses = prod.kind === "gate"
      ? [gain(prod.vsBest, "the best member"), gain(prod.vsMean, "the plain mean")]
      : [gain(prod.vsMeanMember, "the mean member")];
    head = <>
      {prod.kind === "gate" ? "Production gate" : "Plain mean"} <Num>{db(prod.psnr)}</Num>{"\u00a0"}dB{" "}
      <Link to="/ensemble/starfull/overview" title={TEST_HINT}>test PSNR</Link>
      {clauses.some((c) => c != null) && ", "}{joinClauses(clauses)}
    </>;
  } else {
    return null;
  }
  const tail = [
    members != null ? <Link key="m" to="/ensemble/starfull/members" title={MEMBERS_HINT}>{members} members</Link> : null,
    evaluatedAt ? <span key="e">evaluated {formatRelative(evaluatedAt).replace(/ /g, "\u00a0")}</span> : null,
  ].filter(Boolean);
  return (
    <SummaryLine className="home__summary">
      {head}
      {tail.length > 0 && <> · {tail.flatMap((t, i) => (i ? [", ", t] : [t]))}</>}
      {stale && <> (<Num tone="warn">stale</Num>)</>}
    </SummaryLine>
  );
}

export default function Dashboard() {
  const fasrc = useFasrcStatus();
  const feed = useJobsFeed();
  const alerts = useSystemAlerts();
  const system = useResource<SystemInfo>("/api/system", [], { ttl: 60_000 });
  const prodRes = useResource<ProductionPayload>("/api/system/production", [], { ttl: 60_000 });
  // A server started before /api/system/production existed answers 404: fall
  // back to the (slow, ~8 s) ensemble status so the tile still has a number.
  const legacy = useResource<EnsembleStatusSlice>(
    prodRes.error?.status === 404 ? "/ensemble/status.json?mode=starfull" : null, [], { ttl: 60_000 });
  const viaLegacy = !prodRes.data && prodRes.error?.status === 404;
  const ens = viaLegacy
    ? { data: productionFromStatus(legacy.data), loading: legacy.loading, error: legacy.error }
    : { data: prodRes.data, loading: prodRes.loading, error: prodRes.error };
  const models = useResource<ModelsCatalog>("/api/models", [], { ttl: 60_000 });
  const knee = useResource<KneePayload>("/ensemble/knee-psnr.json?mode=starfull", [], { ttl: 60_000 });
  const run = useRunJobs();
  const [logOpen, setLogOpen] = useState(false);

  const refreshAll = () => {
    void refreshHealth();
    for (const prefix of ["/api/system", "/ensemble/", "/api/models", "/api/version", "/api/fasrc/status", "/api/sky/layer/"]) void invalidate(prefix);
  };
  usePageActions([
    { id: "home:refresh", label: "Refresh Home", group: "Home", keywords: ["reload", "health", "summary"], run: refreshAll },
    { id: "home:log", label: "Log the unlogged results to tracking…", group: "Home", keywords: ["notebook", "tracking"], run: () => setLogOpen(true) },
  ]);

  const prod = productionHeadline(ens.data?.eval_summary ?? null, !!ens.data?.stale);
  const kh = kneeHeadline(knee.data);
  const members = starfullMembers(models.data, ens.data);
  const production = productionModel(models.data);
  const disk = system.data?.disk;
  const alertCount = alerts.data?.alerts.length ?? 0;
  const trackingCheck = alerts.data?.checks.find((c) => c.id === "tracking");
  const unlogged = unloggedItems(trackingCheck).length;
  const running = runningItems(feed.jobs, feed.slurm);
  const summaryLoading = (knee.loading && !knee.data) || (ens.loading && !ens.data);

  // Problems only (spec: "a badge appears only on a problem"): FASRC, the
  // server and the disk are silent while healthy, and nothing is said twice
  // on one screen: a changed backend or a new console build is the shell's
  // notice strip (every page, dismissible), and the disk note is skipped when
  // the health list below already carries the disk alert.
  const notes: { id: string; tone: "warn" | "bad"; body: ReactNode; action?: ReactNode }[] = [];
  if (fasrc.data && !fasrc.data.ssh_connected) {
    notes.push({ id: "fasrc", tone: "warn",
      body: <>FASRC is not connected — local pages still work.{fasrc.data.last_error ? <> <span className="home__note-detail">{fasrc.data.last_error}</span></> : null}</>,
      action: <Button asChild size="sm" variant="ghost"><Link to="/settings/connections">Connections</Link></Button> });
  }
  if (viaLegacy && legacy.data) {
    notes.push({ id: "legacy", tone: "warn",
      body: "This server predates /api/system/production: the numbers come from the slower ensemble status. Restart the server for the fast endpoint." });
  } else if (viaLegacy && legacy.error) {
    notes.push({ id: "legacy", tone: "warn", body: "This server does not serve the production numbers — restart it." });
  } else if (!viaLegacy && ens.error && !ens.data) {
    notes.push({ id: "production", tone: "warn", body: `Could not read the production numbers: ${ens.error.message}` });
  }
  const diskAlerted = alerts.data?.alerts.some((c) => c.id === "disk") ?? false;
  if (disk && (disk.level === "warn" || disk.level === "bad") && !diskAlerted) {
    notes.push({ id: "disk", tone: disk.level === "bad" ? "bad" : "warn",
      body: <>{disk.level === "bad" ? "Disk critically low" : "Low disk"}: {formatBytes(disk.free_bytes)} free on the{" "}
        <Link to="/settings/about">data disk</Link>
        {disk.used_fraction != null ? ` (${Math.round(disk.used_fraction * 100)}% of ${formatBytes(disk.total_bytes)} used)` : ""}</> });
  }

  return (
    <Page className="home">
      <PageLead right={(
        <>
          {alertCount > 0 && alerts.data && (
            <Badge tone={alerts.data.counts.bad ? "bad" : "warn"} dot>
              {`${alertCount} alert${alertCount === 1 ? "" : "s"}`}
            </Badge>
          )}
          <IconButton icon="reset" label="Refresh Home" onClick={refreshAll} />
        </>
      )}>
        The production model, health checks and running work.
      </PageLead>

      <div className="home__lead">
        <ProductionSummary knee={kh} prod={prod} members={members?.count ?? null} evaluatedAt={ens.data?.evaluated_at ?? null} />
        {summaryLoading && !kh && !prod && <Skeleton lines={1} width="60%" />}
        {running.length > 0 && (
          <p className="home__running">
            <span className="home__running-lead">Running now:</span> {running.map((r) => r.text).join("; ")}.{" "}
            <Link to="/ops/jobs">All jobs</Link>
          </p>
        )}
        {notes.length > 0 && (
          <div className="home__notes">
            {notes.map((n) => <Callout key={n.id} tone={n.tone} dense action={n.action}>{n.body}</Callout>)}
          </div>
        )}
      </div>

      <div className="home__grid">
        <div className="home__col">
          <Card>
            <CardHead title="Health" sub={alerts.data ? `checked ${formatRelative(alerts.data.computed_at)}` : undefined}
              right={<IconButton icon="reset" size="sm" label="Re-run the health checks" onClick={() => { void refreshHealth(); }} />} />
            <CardBody>
              <HealthList data={alerts.data} loading={alerts.loading} error={alerts.error} />
            </CardBody>
          </Card>
          <Card>
            <CardHead title="Recent jobs"
              right={<Button asChild size="sm" variant="ghost"><Link to="/ops/jobs">All jobs</Link></Button>} />
            <CardBody>
              <JobList jobs={feed.jobs} limit={5} empty="No local jobs since the server started." />
              {feed.slurm.length > 0 && (
                <>
                  <div className="eyebrow home__jobs-sub">SLURM</div>
                  <ul className="joblist">{feed.slurm.slice(0, 4).map((j) => <SlurmRow key={j.jobid} job={j} />)}</ul>
                </>
              )}
            </CardBody>
          </Card>
        </div>
        <div className="home__col">
          <Card>
            <CardHead title="Quick actions" />
            <CardBody>
              <div className="home__actions">
                <Button variant="primary" size="sm" icon="activity" loading={run.evaluate.busy}
                  onClick={() => { void run.evaluate.run(); }}>Evaluate</Button>
                <Button size="sm" loading={run.knee.busy} onClick={() => { void run.knee.run(); }}>PSNR vs knee</Button>
                <Button asChild size="sm"><Link to="/ensemble/starfull/combiners">Fit a gate…</Link></Button>
                <Button asChild size="sm"><Link to="/sky/experiments">Real-tile experiments</Link></Button>
                <Button asChild size="sm"><Link to="/ensemble/starfull/train">Train members</Link></Button>
                <Button size="sm" variant="ghost" icon="pin" onClick={() => setLogOpen(true)}
                  title={unlogged ? `${unlogged} result${unlogged === 1 ? "" : "s"} since the last notebook entry` : "Append a note to the tracking notebook"}>
                  Log to tracking{unlogged ? ` (${unlogged})` : ""}
                </Button>
              </div>
            </CardBody>
          </Card>
          <Card>
            <CardHead title="Sky" sub="Q1 deep fields · NEXUS"
              right={<Button asChild size="sm" variant="ghost"><Link to="/sky/atlas">Open atlas</Link></Button>} />
            <CardBody><SkyOverview /></CardBody>
          </Card>
        </div>
      </div>
      <LogToTrackingDialog open={logOpen} onOpenChange={setLogOpen}
        note={() => trackingCatchUpNote(trackingCheck, { knee: kh, prod, members: members?.count ?? null, production })} />
    </Page>
  );
}
