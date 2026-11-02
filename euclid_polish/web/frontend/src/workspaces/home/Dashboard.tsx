/* Home (console regrouping; statistics rule of the 2026-09-27 spec): what
 * changed since you last looked, what is running and what needs you now.
 * Top to bottom:
 *  1. one verdict sentence for the production model ("Universal 257",
 *     /api/models `production`): its own ∫PSNR (GET /ensemble/universal.json
 *     `production`), then its worst-band real holes from the newest Sky ›
 *     Compare run of THIS production model, or "no real benchmark for" it —
 *     and one caption (when it was scored);
 *  2. the Loop strip: Priors, Records, Models, Production, Real SR, Figures,
 *     each with a state dot and one reason, linking to the tab whose
 *     confirmed button fixes it — the verdicts of the backend staleness
 *     service (GET /api/system/loop, shared with System › Lineage;
 *     "checking" until it answers, "not checked" and a note if it fails);
 *     then, only when broken, the disk, FASRC and notebook warnings;
 *  3. "Running now" (local + SLURM);
 *  4. up to six cached thumbnails (the newest cached production SRs of real
 *     tiles, saved real crops, the newest plates).
 * No KPI tiles and no job launchers: opening Home only reads. Everything
 * works offline. Numbers and their definitions: homeModel.ts. */
import { type ReactNode } from "react";
import { Link } from "react-router-dom";
import { apiGet } from "../../api/client";
import { useJobsFeed } from "../../api/jobs";
import { invalidate, useResource } from "../../api/query";
import { usePageActions } from "../../app/palette";
import { refreshHealth } from "../../app/RunActions";
import { useFasrcStatus, useSystemAlerts } from "../../app/status";
import { formatDate, formatNumber, formatRelative } from "../../format";
import { Callout, Caption, IconButton, Num, Page, Skeleton, SummaryLine, Tooltip } from "../../ui";
import { useLogToNotebook } from "../shared/LogToNotebook";
import { PageLead } from "../shared/PageLead";
import {
  compareRunPath, homeCaption, productionModel, productionRealHoles, trackingCatchUpNote, universalScore, type ExperimentDetail,
  type ExperimentSummary, type ModelsCatalog, type ProductionModel, type RealHoles, type UniversalScore, type UniversalSlice,
} from "./homeModel";
import {
  homeThumbs, loopStages, runningLine, stripAlerts, type PlatesSlice, type RunningLine, type Stage,
  type LoopPayload, type RealSrSlice, type StripAlert, type Thumb,
} from "./loop";
import "./home.css";

const TTL = { ttl: 60_000 };
/** The backend staleness service (helpers/system_alerts.py). */
const LOOP_URL = "/api/system/loop";
/** The production model's score (Models › Universal's URL, so they share its cache). */
const UNIVERSAL_URL = "/ensemble/universal.json?mode=starfull";
const db = (v: number | null | undefined) => formatNumber(v, { digits: 2 });

const UNIVERSAL_HINT = "Mean over VIS, Y, J, H of the model's PSNR integrated over log asinh knee 0.1–10⁴ e⁻ "
  + "on the test fields: its score on Models › Universal.";
const HOLES_HINT = "Hole %: the share of bright LR pixels the SR blanks on real tiles (no truth), per band; the worst band is named.";
const STATE_TEXT: Record<Stage["state"], string> = { current: "current", stale: "stale", blocked: "blocked", unknown: "unknown", loading: "checking" };

/** Sky › Compare with the new-comparison drawer open. */
const NEW_COMPARE = "/sky/compare?new=1";

/** The real-data half of the verdict: production's worst-band holes, or why
 *  there is none. `subject`: what the current production is ("Universal 257"). */
function realClause(h: RealHoles, subject: string): ReactNode {
  if (h.state === "loading") return null;
  if (h.state === "none") {
    if (h.why !== "earlier") return <>; <Link to={NEW_COMPARE} title="Score production on real tiles in Sky › Compare">no real benchmark yet</Link></>;
    // The words name what the link does: score this model. The older run gets its own link, by date.
    const when = h.created ? formatDate(h.created, { fallback: "" }) : "";
    return <>
      ; <Link to={NEW_COMPARE} title={`Score ${subject} on real tiles in Sky › Compare`}>no real benchmark for {subject}</Link>
      {h.expId && (when
        ? <> (the last run, <Link to={compareRunPath(h.expId)}>{when}</Link>, scored an earlier one)</>
        : <> (the <Link to={compareRunPath(h.expId)}>last run</Link> scored an earlier one)</>)}
    </>;
  }
  if (!h.worst) return <>; real holes <Link to={compareRunPath(h.expId)}>on {h.label ?? "Sky › Compare"}</Link> have no per-band values</>;
  const when = h.created ? formatDate(h.created, { fallback: "" }) : "";
  return <>
    ; real holes <Num>{formatNumber(h.worst.pct, { digits: 0 })}%</Num> in {h.worst.band}, its worst band{" "}
    (<Link to={compareRunPath(h.expId)} title={HOLES_HINT}>Sky › Compare{when ? `, ${when}` : ""}</Link>)
  </>;
}

/** The production model's sentence: its own ∫PSNR, then its real holes.
 *  `score` undefined: not read (yet), so no score is claimed either way. */
function UniversalVerdict({ production, score, holes }: { production: ProductionModel; score: UniversalScore | null | undefined; holes: RealHoles }) {
  const head = score === undefined ? <>Production is {production.label}</>
    : score?.mean != null
      ? <>Production is {production.label}: <Num>{db(score.mean)}</Num>{"\u00a0"}dB{" "}
        <Link to="/models/universal" title={UNIVERSAL_HINT}>integrated PSNR</Link></>
      : <>Production is {production.label}, <Link to="/models/universal">not scored on the test fields yet</Link></>;
  return (
    <SummaryLine className="home__summary">
      {head}{realClause(holes, production.label)}{score?.stale && <> (<Num tone="warn">stale</Num>)</>}.
    </SummaryLine>
  );
}

function StageChip({ stage }: { stage: Stage }) {
  const label = `${stage.label}: ${STATE_TEXT[stage.state]} — ${stage.reason}`;
  return (
    <li className="home-loop__item">
      <Tooltip content={stage.detail ? `${stage.reason}. ${stage.detail}` : stage.reason}>
        <Link to={stage.to} className="home-loop__chip" data-state={stage.state} aria-label={label}>
          <span className="home-loop__top">
            <span className="home-loop__dot" data-state={stage.state} aria-hidden />
            <span className="home-loop__label">{stage.label}</span>
          </span>
          <span className="home-loop__reason">{stage.reason}</span>
        </Link>
      </Tooltip>
    </li>
  );
}

function AlertChip({ alert, onLog }: { alert: StripAlert; onLog: () => void }) {
  const body = <><span className="home-loop__dot" data-state={alert.tone === "bad" ? "blocked" : "stale"} aria-hidden />{alert.text}</>;
  const tip = alert.detail ?? alert.text;
  return (
    <li className="home-alerts__item">
      <Tooltip content={tip}>
        {alert.log
          ? <button type="button" className="home-alerts__chip" data-tone={alert.tone} onClick={onLog}>{body}</button>
          : <Link to={alert.to} className="home-alerts__chip" data-tone={alert.tone}>{body}</Link>}
      </Tooltip>
      {alert.log && <Link to={alert.to} className="home-alerts__more">Notebook › Log</Link>}
    </li>
  );
}

function Running({ line }: { line: RunningLine }) {
  return (
    <p className="home__running">
      {line.items.length > 0 ? (
        <><span className="home__running-lead">Running now:</span> {line.items.join("; ")}.{" "}</>
      ) : <><span className="home__running-lead">Nothing running.</span>{" "}</>}
      <Link to="/runs/live">Runs › Live</Link>
    </p>
  );
}

function Thumbs({ thumbs }: { thumbs: Thumb[] }) {
  if (!thumbs.length) return null;
  return (
    <section className="home-thumbs" aria-labelledby="home-thumbs-title">
      <h2 id="home-thumbs-title" className="home-thumbs__title">Latest real SR and plates</h2>
      <ul className="home-thumbs__list">
        {thumbs.map((t) => (
          <li key={t.key}>
            <Link to={t.to} className="home-thumbs__card" title={t.kind === "plate" ? `Open the ${t.label} plate` : `${t.label}: open its tile card`}>
              <img src={t.src} alt="" loading="lazy" decoding="async" className="home-thumbs__img" data-kind={t.kind} />
              <span className="home-thumbs__label">{t.label}</span>
              <span className="home-thumbs__sub">{[t.kind === "plate" ? null : t.sub, t.at ? formatRelative(t.at) : null].filter(Boolean).join(" · ")}</span>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}

export default function Dashboard() {
  const fasrc = useFasrcStatus();
  const feed = useJobsFeed();
  const alerts = useSystemAlerts();
  const models = useResource<ModelsCatalog>("/api/models", [], TTL);
  const production = productionModel(models.data);
  const universal = useResource<UniversalSlice>(UNIVERSAL_URL, [], TTL);
  const exps = useResource<{ experiments: ExperimentSummary[] }>("/api/experiments", [], TTL);
  const benchRun = (exps.data?.experiments ?? [])
    .filter((e) => !!e.summary && "production" in e.summary)
    .sort((a, b) => String(b.created ?? "").localeCompare(String(a.created ?? "")))[0] ?? null;
  const benchDetail = useResource<ExperimentDetail>(benchRun ? `/api/experiments/${encodeURIComponent(benchRun.id)}` : null, [benchRun?.id], { ttl: 5 * 60_000 });
  const loop = useResource<LoopPayload>(LOOP_URL, [], TTL);
  const plates = useResource<PlatesSlice>("/api/figures/nexus-plates", [], TTL);
  const realSr = useResource<RealSrSlice>("/api/figures/real-sr", [], TTL);
  const saved = useResource<{ results?: Parameters<typeof homeThumbs>[0]["results"] }>("/viewer/results", [], TTL);
  const poster = useResource<{ png?: { mtime?: number; pulled_at?: string } | null }>("/poster/result/status", [], TTL);
  const logToNotebook = useLogToNotebook("Home");

  const refreshAll = () => {
    void refreshHealth();
    void apiGet(`${LOOP_URL}?fresh=1`).then(() => invalidate(LOOP_URL), () => undefined);
    for (const prefix of ["/api/system", "/ensemble/", "/api/models", "/api/experiments", "/api/figures/",
      "/viewer/results", "/poster/result/", "/api/version", "/api/fasrc/status"]) void invalidate(prefix);
  };
  // undefined while the payload is not in (or failed): the verdict claims no score then
  const score = universal.data ? universalScore(universal.data) : undefined;
  const holes = productionRealHoles(exps.error ? [] : exps.data?.experiments, benchDetail.data, models.data);
  const caption = homeCaption({ production, score });
  const loopFailed = !!loop.error && !loop.data;
  const stages = loopStages(loop.data, loopFailed);
  const warnings = stripAlerts({ alerts: alerts.data, fasrc: fasrc.data });
  const running = runningLine({ local: feed.jobs, slurm: feed.slurm });
  const thumbs = homeThumbs({ realSr: realSr.data, results: saved.data?.results, plates: plates.data, poster: poster.data });
  const trackingCheck = alerts.data?.checks.find((c) => c.id === "tracking");
  // The "no notebook entry since …" alert and its palette action: Notebook ›
  // Log, prefilled with what was written after the last entry.
  const logCatchUp = () => logToNotebook(trackingCatchUpNote(trackingCheck, { production, score }));
  usePageActions([
    { id: "home:refresh", label: "Refresh Home", group: "Home", keywords: ["reload", "loop", "summary"], run: refreshAll },
    { id: "home:log", label: "Log the unlogged results to the notebook…", group: "Home", keywords: ["notebook", "tracking"], run: logCatchUp },
  ]);
  // No verdict before the catalogue says what production is.
  const modelsPending = models.loading && !models.data;
  const summaryLoading = modelsPending || (universal.loading && !universal.data);

  // The catalogue or the Loop failing: the only note above the strip.
  let note: ReactNode = null;
  if (models.error && !models.data) note = `Could not read the model catalogue: ${models.error.message}`;
  else if (loopFailed) note = `Could not read the Loop (GET ${LOOP_URL}): ${loop.error?.message ?? "no answer"}. Restart the server if it predates the staleness service.`;

  return (
    <Page className="home">
      <PageLead right={<IconButton icon="reset" label="Refresh Home" onClick={refreshAll} />}>
        What changed, what is running and what needs you now.
      </PageLead>

      <section className="home__lead" aria-label="Production model">
        {!summaryLoading && production && <UniversalVerdict production={production} score={score} holes={holes} />}
        {summaryLoading && <Skeleton lines={1} width="60%" />}
        {!summaryLoading && production && (
          <Caption className="home__caption">
            <Link to="/models/universal">one universal model</Link>{caption.map((part) => <span key={part}> · {part}</span>)}
          </Caption>
        )}
        {note && <Callout tone="warn" dense>{note}</Callout>}
      </section>

      <nav className="home-loop" aria-label="The loop">
        <ol className="home-loop__list">{stages.map((s) => <StageChip key={s.id} stage={s} />)}</ol>
        {warnings.length > 0 && (
          <ul className="home-alerts" aria-label="Needs attention">
            {warnings.map((a) => <AlertChip key={a.id} alert={a} onLog={logCatchUp} />)}
          </ul>
        )}
      </nav>

      {running && <Running line={running} />}

      <Thumbs thumbs={thumbs} />

    </Page>
  );
}
