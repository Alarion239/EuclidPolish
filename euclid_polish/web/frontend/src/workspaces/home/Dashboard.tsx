/* Home (spec §8.1): the production model's numbers, the health checks, the
 * running work, quick actions and a small sky. Everything works offline.
 *
 * Numbers and their definitions are in homeModel.ts: the knee-integrated
 * PSNR of the production spatial gate (/ensemble/knee-psnr.json), its test
 * PSNR from eval_summary.json's spatial_gate_* keys (/ensemble/status.json),
 * the active STARFULL member count from the regime labels (/api/models).
 * Health checks: /api/system/alerts (routes/system.py); free space:
 * /api/system. */
import { Link } from "react-router-dom";
import { useJobsFeed } from "../../api/jobs";
import { invalidate, useResource } from "../../api/query";
import { registerInspector } from "../../app/inspector";
import { JobList, SlurmRow } from "../../app/JobTray";
import { usePageActions } from "../../app/palette";
import { refreshHealth, useRunJobs } from "../../app/RunActions";
import { useFasrcStatus, useSystemAlerts, useVersion } from "../../app/status";
import { formatBytes, formatCount, formatNumber, formatRelative } from "../../format";
import { Badge, Button, Card, CardBody, CardHead, IconButton, Kpi, Page, PageHead, type Tone } from "../../ui";
import { CheckInspector, HealthList } from "./Health";
import {
  kneeHeadline, productionFromStatus, productionHeadline, productionModel, starfullMembers,
  type EnsembleStatusSlice, type KneePayload, type ModelsCatalog, type ProductionPayload,
} from "./homeModel";
import { SkyOverview } from "./SkyOverview";
import "./home.css";

registerInspector("check", CheckInspector, { title: (id) => `Health check · ${id}` });

type SystemInfo = {
  disk?: { free_bytes: number; total_bytes: number; used_fraction: number | null; level: "ok" | "warn" | "bad" | "unknown" };
};

const db = (v: number | null | undefined) => formatNumber(v, { digits: 2 });
const dbSigned = (v: number | null | undefined) => formatNumber(v, { digits: 2, signed: true });
const deltaTone = (v: number | null | undefined): Tone => (v != null && v >= 0 ? "good" : "warn");
const LEVEL_TONE: Record<string, Tone> = { ok: "neutral", warn: "warn", bad: "bad", unknown: "neutral" };

const KNEE_HINT = "Mean over VIS, Y, J, H of each model's PSNR integrated over log asinh knee 0.1–10⁴ e⁻ "
  + "(test cubes). The knee-independent way to compare models.";
const TEST_HINT = "Asinh-stretched test PSNR of the production spatial gate from the last STARFULL evaluation "
  + "(eval_summary.json spatial_gate_* keys; the RBF combiner is never the headline).";
const MEMBERS_HINT = "Registry-active members of the STARFULL regime (the default). Starless members are an opt-in "
  + "regime and never enter the production gate.";

export default function Dashboard() {
  const fasrc = useFasrcStatus();
  const version = useVersion();
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

  const refreshAll = () => {
    void refreshHealth();
    for (const prefix of ["/api/system", "/ensemble/", "/api/models", "/api/version", "/api/fasrc/status", "/api/sky/layer/"]) void invalidate(prefix);
  };
  usePageActions([
    { id: "home:refresh", label: "Refresh Home", group: "Home", keywords: ["reload", "health", "kpi"], run: refreshAll },
  ]);

  const v = version.data;
  const prod = productionHeadline(ens.data?.eval_summary ?? null, !!ens.data?.stale);
  const kh = kneeHeadline(knee.data);
  const members = starfullMembers(models.data, ens.data);
  const production = productionModel(models.data);
  const disk = system.data?.disk;
  const alertCount = alerts.data?.alerts.length ?? 0;

  const kneeValue = kh?.gate ?? kh?.mean ?? null;
  const kneeDelta = kh?.best && kh.vsBest != null ? `${dbSigned(kh.vsBest)} dB vs ${kh.best.name.replace("member_", "member ")}` : undefined;
  const kneeFooter = kh
    ? [kh.gate == null ? "plain mean (no gate in the cubes)" : kh.vsMean != null ? `${dbSigned(kh.vsMean)} dB vs plain mean` : null,
      kh.nFields != null ? `${kh.nFields} fields` : null, kh.stale ? "stale" : null].filter(Boolean).join(" · ")
    : knee.error ? "no knee curves" : knee.data && !knee.data.available ? "not computed yet" : undefined;

  const testLabel = prod?.kind === "mean" ? "Test PSNR · plain mean" : "Test PSNR · production gate";
  const testDelta = prod?.kind === "gate"
    ? (prod.vsBest != null ? `${dbSigned(prod.vsBest)} dB vs best member` : undefined)
    : prod?.kind === "mean" && prod.vsMeanMember != null ? `${dbSigned(prod.vsMeanMember)} dB vs mean member` : undefined;
  const testFooter = prod
    ? [prod.kind === "gate" && prod.vsMean != null ? `${dbSigned(prod.vsMean)} dB vs plain mean${prod.meanPsnr != null ? ` ${db(prod.meanPsnr)}` : ""}` : null,
      prod.kind === "mean" ? "no production-gate score yet" : null, prod.stale ? "summary stale" : null,
      viaLegacy ? "via ensemble status (restart the server for the fast endpoint)" : null].filter(Boolean).join(" · ")
    : ens.error ? (viaLegacy ? "not served by this server — restart it" : ens.error.message)
      : ens.data ? "not evaluated yet" : undefined;

  return (
    <Page className="home">
      <PageHead eyebrow="console" title="Home" sub="The production model, health checks and running work."
        right={(
          <div className="home__head-right">
            {alerts.data && (
              <Badge tone={alertCount ? (alerts.data.counts.bad ? "bad" : "warn") : "good"} dot>
                {alertCount ? `${alertCount} alert${alertCount === 1 ? "" : "s"}` : "all clear"}
              </Badge>
            )}
            <IconButton icon="reset" label="Refresh Home" onClick={refreshAll} />
          </div>
        )} />

      <div className="home__kpis" role="group" aria-label="Production model">
        <Kpi label={kh?.gate == null && kh ? "∫PSNR · plain mean" : "∫PSNR · production gate"} icon="layers"
          to="/ensemble/starfull/knee" loading={knee.loading && !knee.data} hint={KNEE_HINT}
          value={db(kneeValue)} unit={kneeValue != null ? "dB" : undefined}
          delta={kneeDelta} deltaTone={deltaTone(kh?.vsBest)} footer={kneeFooter} />
        <Kpi label={testLabel} icon="activity" to="/ensemble/starfull/overview" loading={ens.loading && !ens.data}
          hint={prod?.stale && ens.data?.stale_reason ? `${TEST_HINT} Stale: ${ens.data.stale_reason}.` : TEST_HINT}
          value={db(prod?.psnr)} unit={prod ? "dB" : undefined}
          delta={testDelta}
          deltaTone={deltaTone(prod?.kind === "gate" ? prod.vsBest : prod?.kind === "mean" ? prod.vsMeanMember : null)}
          footer={testFooter} tone={prod?.stale ? "warn" : undefined} />
        <Kpi label="STARFULL members" icon="database" to="/ensemble/starfull/members" hint={MEMBERS_HINT}
          loading={models.loading && ens.loading && !members} value={formatCount(members?.count ?? null)}
          delta={production ? (production.available ? `${production.label} fits them` : `${production.label} out of date`) : undefined}
          deltaTone={production?.available ? "good" : "warn"}
          footer={[
            production?.mix ? `${production.mix} mix` : null,
            production?.fittedAt ? `fitted ${formatRelative(production.fittedAt)}` : null,
            members?.starless != null ? `${members.starless} starless (opt-in)` : null,
          ].filter(Boolean).join(" · ") || undefined} />
      </div>
      <div className="home__kpis home__kpis--system" role="group" aria-label="System">
        <Kpi label="Running jobs" icon="activity" to="/ops/jobs" loading={feed.loading}
          value={formatCount(feed.runningCount)}
          footer={feed.fasrcOffline ? "local only · FASRC offline" : `${feed.running.length} local · ${feed.slurm.length} SLURM`} />
        <Kpi label="FASRC" icon="server" to="/settings/connections" loading={fasrc.loading && !fasrc.data}
          value={fasrc.data?.ssh_connected ? "connected" : "offline"}
          tone={fasrc.data?.ssh_connected ? "good" : "neutral"}
          footer={fasrc.data?.ssh_connected ? "SSH ControlMaster up" : (fasrc.data?.last_error ?? "local pages still work")} />
        <Kpi label="Server" icon="info" to="/settings/about" loading={version.loading && !v}
          value={v?.boot_short ?? "—"} tone={v?.behind ? "warn" : "neutral"}
          delta={v ? (v.behind ? `HEAD ${v.head_short} — restart` : "at HEAD") : undefined}
          deltaTone={v?.behind ? "warn" : "good"}
          footer={v?.started_at ? `started ${formatRelative(v.started_at)}${v.dirty ? " · dirty tree" : ""}` : undefined} />
        <Kpi label="Free disk" icon="database" to="/settings/about" loading={system.loading && !disk}
          value={disk ? formatBytes(disk.free_bytes) : "—"} tone={disk ? LEVEL_TONE[disk.level] : undefined}
          delta={disk && disk.level !== "ok" ? (disk.level === "bad" ? "critically low" : "low") : undefined}
          deltaTone={disk?.level === "bad" ? "bad" : "warn"}
          footer={disk?.used_fraction != null ? `${Math.round(disk.used_fraction * 100)} % of ${formatBytes(disk.total_bytes)} used` : undefined} />
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
                <Button asChild size="sm" variant="ghost"><Link to="/ops/tracking">Log to tracking</Link></Button>
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
    </Page>
  );
}
