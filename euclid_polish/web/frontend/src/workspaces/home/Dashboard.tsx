/* The Home dashboard: FASRC connection, server version, running jobs, the
   STARFULL ensemble headline and the local data, from existing endpoints
   (/api/fasrc/status, /api/version, the jobs feed,
   /ensemble/status.json?mode=starfull, /api/status); recent jobs; quick
   links. Everything here works offline.

   The ensemble tile says which number it shows: the production spatial gate
   (`spatial_gate_combiner_psnr`, written to eval_summary.json when that
   evaluation scored the gate) or, failing that, the plain mean of the
   members (`ensemble_psnr`). The gain "vs mean member" is derived here from
   `ensemble_psnr − mean_member_psnr`: the summary's `ensemble_gain_db` means
   "vs mean member" or "vs best member" depending on which job wrote it. */
import { Link } from "react-router-dom";
import { useJobsFeed } from "../../api/jobs";
import { useResource } from "../../api/query";
import { useFasrcStatus, useVersion } from "../../app/status";
import { JobList } from "../../app/JobTray";
import { formatCount, formatNumber, formatRelative } from "../../format";
import { Button, Card, CardBody, CardHead, Kpi, Page, PageHead } from "../../ui";
import "./home.css";

/** The production combiner kind (`eval/combiner.py` ACTIVE_COMBINER_KINDS[0]);
 *  eval_summary.json carries `<kind>_combiner_psnr` / `_vs_mean_db` (vs the
 *  plain mean) / `_vs_best_member_db` for every combiner it scored. The bare
 *  `combiner_psnr` is the RBF combiner, not production. */
const PRODUCTION_GATE = "spatial_gate";

type EvalSummary = {
  ensemble_psnr?: number | null;
  mean_member_psnr?: number | null;
} & Partial<Record<`${typeof PRODUCTION_GATE}_combiner_${"psnr" | "vs_mean_db" | "vs_best_member_db"}`, number | null>>;

type EnsembleStatus = {
  n_members?: number;
  eval_summary?: EvalSummary | null;
  eval_summary_stale?: boolean;
  test_present?: boolean;
};

const finite = (v: number | null | undefined): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const db = (v: number) => formatNumber(v, { digits: 2 });
const dbSigned = (v: number) => formatNumber(v, { digits: 2, signed: true });

type Headline = { label: string; value: number | null; delta?: number | null; deltaText?: string; footer: string };

/** What the ensemble tile shows, with explicit definitions (spec §8.1). */
function ensembleHeadline(status: EnsembleStatus | null, failed: boolean): Headline {
  if (failed) return { label: "Ensemble", value: null, footer: "no ensemble status" };
  if (!status) return { label: "Ensemble", value: null, footer: "" };
  const s = status.eval_summary ?? null;
  const n = `${formatCount(status.n_members)} starfull members`;
  const stale = status.eval_summary_stale ? " · summary stale" : "";
  const gate = finite(s?.[`${PRODUCTION_GATE}_combiner_psnr`]);
  const mean = finite(s?.ensemble_psnr);
  if (gate != null) {
    const vsMean = finite(s?.[`${PRODUCTION_GATE}_combiner_vs_mean_db`]);
    const vsBest = finite(s?.[`${PRODUCTION_GATE}_combiner_vs_best_member_db`]);
    const parts = ["spatial gate", vsBest != null ? `${dbSigned(vsBest)} dB vs best member` : null,
      mean != null ? `plain mean ${db(mean)} dB` : null, n];
    return {
      label: "Production gate", value: gate, delta: vsMean, deltaText: vsMean != null ? `${dbSigned(vsMean)} dB vs plain mean` : undefined,
      footer: parts.filter(Boolean).join(" · ") + stale,
    };
  }
  const meanMember = finite(s?.mean_member_psnr);
  const vsMember = mean != null && meanMember != null ? mean - meanMember : null;
  return {
    label: "Ensemble mean", value: mean, delta: vsMember,
    deltaText: vsMember != null ? `${dbSigned(vsMember)} dB vs mean member` : undefined,
    footer: s ? `plain mean of ${n} · no production-gate score in the eval summary${stale}` : `${n} · not evaluated yet`,
  };
}

type LocalStatus = {
  catalog?: { present?: boolean; cached?: boolean } | null;
  psfs?: { bands?: { name: string; empirical?: boolean }[] } | null;
  tfrecords?: { dir?: string | null; files?: unknown[] } | null;
  checkpoints?: { dir?: string | null; files?: { member?: string }[] } | null;
};

const QUICK: [string, string, string][] = [
  ["Sky atlas", "/sky/atlas", "Coverage and real results"],
  ["Ensemble", "/ensemble/starfull/overview", "Members, evaluation, combiners"],
  ["Train members", "/ensemble/starfull/train", "FASRC training"],
  ["Realism", "/realism/noise", "Synthetic vs real"],
  ["Training records", "/data/records", "TFRecords viewer"],
  ["FASRC", "/ops/fasrc", "Cluster console"],
];

export default function Dashboard() {
  const fasrc = useFasrcStatus();
  const version = useVersion();
  const feed = useJobsFeed();
  const ens = useResource<EnsembleStatus>("/ensemble/status.json?mode=starfull", [], { ttl: 60_000 });
  const local = useResource<LocalStatus>("/api/status", [], { ttl: 60_000 });

  const v = version.data;
  const head = ensembleHeadline(ens.data, !!ens.error);
  const members = new Set((local.data?.checkpoints?.files ?? []).map((f) => f.member).filter(Boolean));
  const empirical = (local.data?.psfs?.bands ?? []).filter((b) => b.empirical).length;
  const bands = (local.data?.psfs?.bands ?? []).length;
  return (
    <Page className="home">
      <PageHead eyebrow="console" title="Home"
        sub="The console at a glance. Every tile opens its workspace; everything here works with FASRC offline." />

      <div className="home__kpis">
        <Kpi label="FASRC" icon="server" to="/settings/connections" loading={fasrc.loading}
          value={fasrc.data?.ssh_connected ? "connected" : "offline"}
          tone={fasrc.data?.ssh_connected ? "good" : "neutral"}
          footer={fasrc.data?.ssh_connected ? "SSH ControlMaster up" : (fasrc.data?.last_error ?? "local pages still work")} />
        <Kpi label="Server" icon="info" to="/settings/about" loading={version.loading}
          value={v?.boot_short ?? "—"} tone={v?.behind ? "warn" : "neutral"}
          delta={v ? (v.behind ? `HEAD ${v.head_short} — restart` : "at HEAD") : undefined}
          deltaTone={v?.behind ? "warn" : "good"}
          footer={v?.started_at ? `started ${formatRelative(v.started_at)}${v.dirty ? " · dirty tree" : ""}` : undefined} />
        <Kpi label="Running jobs" icon="activity" to="/ops/jobs" loading={feed.loading}
          value={formatCount(feed.runningCount)}
          footer={feed.fasrcOffline ? "local only · FASRC offline" : `${feed.slurm.length} SLURM live`} />
        <Kpi label={head.label} icon="layers" to="/ensemble/starfull/overview" loading={ens.loading}
          value={head.value != null ? db(head.value) : "—"} unit={head.value != null ? "dB" : undefined}
          delta={head.deltaText} deltaTone={head.delta != null && head.delta >= 0 ? "good" : "warn"}
          footer={head.footer || undefined} />
        <Kpi label="Local data" icon="database" to="/data/records" loading={local.loading}
          value={formatCount(local.data?.tfrecords?.files?.length ?? null)} unit="TFRecords"
          footer={`${members.size} member checkpoints · ePSF ${empirical}/${bands} · catalogue ${local.data?.catalog?.present ? "cached" : "missing"}`} />
      </div>

      <div className="home__grid">
        <Card>
          <CardHead title="Recent jobs" sub="local background jobs"
            right={<Button asChild size="sm" variant="ghost"><Link to="/ops/jobs">All jobs</Link></Button>} />
          <CardBody><JobList jobs={feed.jobs} limit={6} empty="No local jobs since the server started." /></CardBody>
        </Card>
        <Card>
          <CardHead title="Go to" />
          <CardBody>
            <ul className="home__links">
              {QUICK.map(([label, to, hint]) => (
                <li key={to}><Link to={to}><b>{label}</b><span>{hint}</span></Link></li>
              ))}
            </ul>
          </CardBody>
        </Card>
      </div>
    </Page>
  );
}
