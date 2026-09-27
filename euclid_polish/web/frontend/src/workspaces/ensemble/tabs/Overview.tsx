/* ensemble/overview (spec §8.2; statistics rule of the 2026-09-27 console
   spec): one status line (all current, or each failing staleness check with
   its confirmed fix), one sentence with the production gate against its
   references, a comparison table gate / plain mean / best member on the SAME
   metric (knee-integrated PSNR, per band) so "best member" is tied to it, a
   caption (members, gate fit, the integration, evaluation time), one alert
   line for TIMEOUT members, and the run actions — evaluate (optionally
   forced), refresh member PSNR, PSNR vs knee, and a FASRC pull with a member
   picker (probe first, then pull what you pick). The numbers come from
   knee-psnr.json (the overview headline when it cannot be read). "Log to
   tracking" opens the Evaluate summary as an editable notebook entry
   (../notes.ts evaluationNote). */
import { useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { useJob, type Job } from "../../../api/jobs";
import { pagePath } from "../../../app/nav";
import { usePageActions } from "../../../app/palette";
import { startJob } from "../../../app/RunActions";
import { useFasrcStatus } from "../../../app/status";
import { formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Button, Callout, Caption, Card, CardBody, CardHead, Checkbox, Dialog, EmptyState, Input, JobProgress, Num,
  NumberField, Page, SummaryLine, Table, Tooltip, type Column,
} from "../../../ui";
import {
  BAND_SHORT, BANDS, useKnee, useMembers, useMode, useOverview, type Check, type MemberRow, type Mode, type Overview as OverviewData,
} from "../api";
import { BarGroup, EnsBar, LoadState } from "../common";
import { JOB, useOnJobEnd } from "../jobs";
import { db, dbDelta, memberNumber, overviewComparison, parseMemberList, type Comparison, type ComparisonRow } from "../model";
import { evaluationNote } from "../notes";
import { LogToTrackingButton } from "../../shared/LogToTracking";
import "../ensemble.css";

const tabPath = (mode: Mode, tab: string) => pagePath("ensemble", { tab, params: { mode } });

function evaluate(mode: Mode, n: number, force: boolean) {
  return startJob({
    key: JOB.evaluate, url: "/ensemble/evaluate",
    label: `Evaluate the ${mode} ensemble${force ? " (forced)" : ""}`,
    data: { mode, num_images: String(n), force: force ? "1" : "0" },
    question: {
      title: `Evaluate the ${mode} ensemble on ${n} test fields?`,
      message: force
        ? "Forced: every member is re-run (TensorFlow) even if an identical evaluation is cached. Several minutes."
        : "Loads every active member (TensorFlow) unless an identical evaluation is cached. Several minutes.",
      confirmLabel: "Evaluate",
    },
  });
}

const knee = (mode: Mode) => startJob({
  key: JOB.knee, url: "/ensemble/knee-psnr", label: `PSNR vs knee (${mode})`, data: { mode },
  question: { title: "Recompute PSNR vs knee?", message: "Scores every member, the mean and each combiner at every knee from the cached test cubes.", confirmLabel: "Compute" },
});

const memberPsnr = () => startJob({
  key: JOB.memberPsnr, url: "/ensemble/member-psnr", label: "Re-score the members' test PSNR",
  question: { title: "Re-score the members' test PSNR?", message: "Only changed or unscored members are evaluated (TensorFlow).", confirmLabel: "Re-score" },
});

/** A knee in e⁻ as prose reads it: 0.1, 100, 10⁴. */
const SUP: Record<string, string> = { 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
function kneeE(v: number): string {
  const p = Math.log10(v);
  return v >= 1000 && Number.isInteger(p) ? `10${String(p).split("").map((d) => SUP[d]).join("")}` : String(v);
}

/** Non-breaking spaces: "2 d ago" and "10⁴ e⁻" never wrap inside. */
const nb = (t: string) => t.replace(/ /g, "\u00a0");

/** "+1.02 dB over …"; a loss reads "0.30 dB under …" in warn. A non-breaking
 *  space keeps each number with its unit. */
function gain(v: number | null, what: string): ReactNode {
  if (v == null || !Number.isFinite(v)) return null;
  return v >= 0 ? <><Num>{dbDelta(v)}</Num>{"\u00a0"}dB over {what}</> : <><Num tone="warn">{db(-v)}</Num>{"\u00a0"}dB under {what}</>;
}

/** The production gate (else the plain mean) against the best member and the
 *  plain mean, on the comparison table's own numbers. */
function Summary({ cmp, mode }: { cmp: Comparison; mode: Mode }) {
  const row = (id: ComparisonRow["id"]) => cmp.rows.find((r) => r.id === id)?.integrated ?? null;
  const gate = row("gate"), mean = row("mean"), best = row("best");
  const head = gate ?? mean;
  if (head == null) return null;
  const bestWhat = cmp.bestNumber ? `the best member (#${cmp.bestNumber})` : "the best member";
  const clauses = [
    best != null ? gain(head - best, bestWhat) : null,
    gate != null && mean != null ? gain(gate - mean, "the plain mean") : null,
  ].filter((c) => c != null);
  return (
    <SummaryLine>
      {gate != null ? "Production gate" : "Plain mean"} <Num>{db(head)}</Num>{"\u00a0"}dB{" "}
      <Link to={tabPath(mode, "knee")}>integrated PSNR</Link>
      {clauses.length > 0 && ", "}
      {clauses.map((c, i) => <span key={i}>{i > 0 && " and "}{c}</span>)}
    </SummaryLine>
  );
}

function ComparisonTable({ cmp, bands }: { cmp: Comparison; bands: readonly string[] }) {
  const columns: Column<ComparisonRow>[] = [
    { header: "", cell: (r) => (r.id === "gate" ? <strong>{r.label}</strong> : r.label) },
    { header: "∫PSNR [dB]", align: "right", cell: (r) => <span className="ens-tnum">{db(r.integrated)}</span> },
    ...bands.map((b, i): Column<ComparisonRow> => ({
      header: `∫${BAND_SHORT[b] ?? b} [dB]`, align: "right", cell: (r) => <span className="ens-tnum">{db(r.bands[i])}</span>,
    })),
  ];
  return <Table className="ens-compare" aria-label="Production vs references" columns={columns} rows={cmp.rows} rowKey={(r) => r.id} />;
}

/** One quiet line when every check passes; else each failing check once,
 *  with its fix (each fix offered once, on the first check that needs it). */
function StatusLine({ o, mode, onEvaluate, onKnee }: {
  o: OverviewData; mode: Mode; onEvaluate: () => void; onKnee: () => void;
}) {
  const failing = o.checks.filter((c) => !c.ok);
  if (!failing.length) {
    return <p className="ens-status" data-tone="good"><span className="ens-check__dot" aria-hidden />All current</p>;
  }
  const offered = new Set<string>();
  const fix = (c: Check) => {
    if (!c.action || offered.has(c.action)) return <span />;
    offered.add(c.action);
    if (c.action === "evaluate") {
      return (
        <Tooltip content={o.test_present ? "Evaluate the members, the mean and the combiners on the test records" : "No local test records — sync them in Data › Records"}>
          <span><Button size="sm" onClick={onEvaluate} disabled={!o.test_present}>Evaluate</Button></span>
        </Tooltip>
      );
    }
    if (c.action === "knee") return <Button size="sm" onClick={onKnee}>Compute</Button>;
    if (c.action === "combiners") return <Button size="sm" asChild><Link to={tabPath(mode, "combiners")}>Combiners</Link></Button>;
    return <span />;
  };
  return (
    <ul className="ens-checks ens-status-list" aria-label="Staleness">
      {failing.map((c) => (
        <li key={c.id} className="ens-check" data-tone={c.tone}>
          <span className="ens-check__dot" aria-hidden />
          <span>
            <span className="ens-check__title">{c.title}</span>{" "}
            <span className="ens-check__detail">{c.detail}</span>
          </span>
          {fix(c)}
        </li>
      ))}
    </ul>
  );
}

/** "178, 179, 182" (at most `max`, then "…"). */
const numberList = (rows: readonly MemberRow[], max = 8) => {
  const nums = rows.map((m) => memberNumber(m.name) ?? m.name);
  return nums.length > max ? `${nums.slice(0, max).join(", ")}, …` : nums.join(", ");
};

/** Probe FASRC for changed members, then pull the ones picked. */
function PullDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  const fasrc = useFasrcStatus().data;
  const offline = fasrc ? !fasrc.ssh_connected : false;
  const probe = useJob(JOB.pullCheck);
  const pull = useJob(JOB.pull);
  const [text, setText] = useState("");
  const [picked, setPicked] = useState<string[]>([]);
  const changed = useMemo(() => {
    const r = probe.job?.status === "done" ? (probe.job.result as { changed?: string[] } | null) : null;
    return r?.changed ?? null;
  }, [probe.job]);
  const parsed = parseMemberList(text);
  const wanted = [...new Set([...picked, ...parsed.names])];
  const check = () => probe.run("/ensemble/pull", { dry_run: "1" }, { onDone: (j: Job) => {
    const r = j.result as { changed?: string[] } | null;
    setPicked(r?.changed ?? []);
  } });
  const doPull = (members: string[]) => pull.run("/ensemble/pull", members.length ? { members: members.join(",") } : {});
  useOnJobEnd(pull.job);
  return (
    <Dialog open={open} onOpenChange={onOpenChange} size="lg" title="Pull members from FASRC"
      description="Check what changed on FASRC, pick members, pull. Unchanged members are never downloaded."
      footer={<>
        <Button variant="ghost" onClick={() => onOpenChange(false)}>Close</Button>
        <Button disabled={offline || pull.busy} onClick={() => doPull([])}>Pull every changed member</Button>
        <Button variant="primary" disabled={offline || pull.busy || !wanted.length} loading={pull.busy}
          onClick={() => doPull(wanted)}>Pull {wanted.length || ""} selected</Button>
      </>}>
      <div className="ens-stack">
        {offline && <EmptyState compact icon="warn" title="FASRC not connected">
          <span className="ens-mono">{fasrc?.last_error ?? "connect in Settings › Connections"}</span>
        </EmptyState>}
        <div className="ens-row">
          <Button icon="search" disabled={offline || probe.busy} loading={probe.busy} onClick={check}>Check FASRC</Button>
          <Input value={text} onChange={setText} placeholder="or type members: 195 196 199-202" aria-label="Members to pull"
            style={{ flex: "1 1 220px" }} />
        </div>
        {parsed.bad.length > 0 && <span className="ens-warn">Not member names: {parsed.bad.join(", ")}</span>}
        {changed && (changed.length === 0
          ? <span className="ens-muted">Every member is up to date on FASRC.</span>
          : (
            <div className="ens-picker" role="group" aria-label="Changed members">
              {changed.map((name) => (
                <label key={name} className="ens-pick" data-on={picked.includes(name)}>
                  <span className="ens-pick__top">
                    <span>#{memberNumber(name)}</span>
                    <Checkbox checked={picked.includes(name)} aria-label={`Pull ${name}`}
                      onChange={(on) => setPicked((p) => (on ? [...p, name] : p.filter((x) => x !== name)))} />
                  </span>
                  <span className="ens-pick__meta">changed on FASRC</span>
                </label>
              ))}
            </div>
          ))}
        <JobProgress job={probe.job} error={probe.error} />
        <JobProgress job={pull.job} error={pull.error} />
      </div>
    </Dialog>
  );
}

export default function Overview() {
  const mode = useMode();
  const ov = useOverview(mode);
  const members = useMembers(mode);
  const kneeRes = useKnee(mode);
  const [n, setN] = useUrlState("n", "100");
  const [force, setForce] = useUrlState("force", false);
  const [pullOpen, setPullOpen] = useState(false);
  const evalJob = useJob(JOB.evaluate);
  const kneeJob = useJob(JOB.knee);
  const psnrJob = useJob(JOB.memberPsnr);
  useOnJobEnd(evalJob.job);
  useOnJobEnd(kneeJob.job);
  useOnJobEnd(psnrJob.job);
  const fasrc = useFasrcStatus().data;
  const nFields = Math.max(1, Math.min(2000, Math.round(Number(n) || 100)));

  const run = {
    evaluate: () => void evaluate(mode, nFields, force),
    knee: () => void knee(mode),
    psnr: () => void memberPsnr(),
  };
  usePageActions([
    { id: "ens-evaluate", label: `Evaluate the ${mode} ensemble`, group: "Ensemble", keywords: ["test", "psnr"], shortcut: "Shift+E", run: run.evaluate },
    { id: "ens-evaluate-force", label: `Evaluate the ${mode} ensemble (force re-inference)`, group: "Ensemble", run: () => void evaluate(mode, nFields, true) },
    { id: "ens-knee", label: "Compute PSNR vs knee", group: "Ensemble", keywords: ["integrated"], run: run.knee },
    { id: "ens-member-psnr", label: "Refresh member PSNR", group: "Ensemble", run: run.psnr },
    { id: "ens-pull", label: "Pull members from FASRC…", group: "Ensemble", keywords: ["download", "rsync"], run: () => setPullOpen(true) },
  ]);

  const o = ov.data;
  const kneeData = kneeRes.data;
  const cmp = useMemo(() => overviewComparison(kneeData, o?.headline.knee), [kneeData, o]);
  const bands = kneeData?.available && kneeData.bands?.length ? kneeData.bands : [...BANDS];
  const timeouts = (members.data?.members ?? []).filter((m) => m.timeout);
  const integration = kneeData?.integration ?? o?.headline.knee.integration ?? null;
  const nFieldsKnee = kneeData?.n_fields ?? o?.headline.knee.n_fields ?? null;
  const caption = o ? [
    `${o.n_members} member${o.n_members === 1 ? "" : "s"}`,
    o.production_gate.available && o.production_gate.fitted_at
      ? `production gate fitted ${nb(formatRelative(o.production_gate.fitted_at))}${o.production_gate.mix_space ? ` (${o.production_gate.mix_space} mix)` : ""}` : null,
    cmp.rows.length ? `∫ = PSNR averaged over log knee ${nb(`${kneeE(integration?.from_e ?? 0.1)}–${kneeE(integration?.to_e ?? 1e4)} e⁻`)}${nFieldsKnee ? ` on ${nFieldsKnee} test fields` : ""}` : null,
    o.evaluated_at ? `evaluated ${nb(formatRelative(o.evaluated_at))}` : "not evaluated yet",
  ].filter(Boolean).join(" · ") : "";
  return (
    <Page>
      <EnsBar label="Run actions">
        <BarGroup label="Evaluate">
          <NumberField size="sm" aria-label="Test fields" value={n} onChange={setN} min={1} max={2000} unit="fields" />
          <Tooltip content="Re-run every member even when an identical evaluation is cached">
            <span><Checkbox checked={force} onChange={setForce}>force</Checkbox></span>
          </Tooltip>
          <Button size="sm" variant="primary" icon="activity" loading={evalJob.busy} disabled={o ? !o.test_present : false}
            onClick={run.evaluate}>Evaluate</Button>
        </BarGroup>
        <span className="ens-bar__sep" aria-hidden />
        <Button size="sm" loading={psnrJob.busy} onClick={run.psnr}>Member PSNR</Button>
        <Button size="sm" loading={kneeJob.busy} onClick={run.knee}>Knee PSNR</Button>
        <span className="ens-bar__spacer" />
        <LogToTrackingButton disabled={!o} note={() => (o ? evaluationNote(o, mode) : "")}
          title="Append the Evaluate summary to the tracking notebook (you edit it first)" />
        <Tooltip content={fasrc && !fasrc.ssh_connected ? `FASRC offline: ${fasrc.last_error ?? "not connected"}` : "Check FASRC and pull changed members"}>
          <span><Button size="sm" icon="download" onClick={() => setPullOpen(true)}>Pull…</Button></span>
        </Tooltip>
      </EnsBar>
      <LoadState loading={ov.loading} error={ov.error} onRetry={ov.reload}>
        {o && (
          <div className="ens-stack ens-overview">
            <StatusLine o={o} mode={mode} onEvaluate={run.evaluate} onKnee={run.knee} />
            {timeouts.length > 0 && (
              <Callout tone="warn" dense action={(
                <Button size="sm" asChild>
                  <Link to={`${tabPath(mode, "train")}?mode=continue&members=${timeouts.map((m) => m.name).join(",")}`}>
                    {timeouts.length === 1 ? "Continue it…" : "Continue them…"}
                  </Link>
                </Button>
              )}>
                {`${timeouts.length} member${timeouts.length === 1 ? "" : "s"} stopped at the time limit: ${numberList(timeouts)}.`}
              </Callout>
            )}
            {members.error && !members.data && (
              <p className="ens-muted ens-overview__note">Could not read the members: {members.error.message}</p>
            )}
            {cmp.rows.length > 0 && (
              <section className="ens-overview__result" aria-label="Production model">
                <Summary cmp={cmp} mode={mode} />
                <ComparisonTable cmp={cmp} bands={bands} />
                <Caption className="ens-overview__caption">{caption}</Caption>
              </section>
            )}
            {!cmp.rows.length && <Caption className="ens-overview__caption">{caption}</Caption>}
            {(evalJob.job || kneeJob.job || psnrJob.job || evalJob.error) && (
              <Card>
                <CardHead title="Jobs" />
                <CardBody>
                  <JobProgress job={evalJob.job} error={evalJob.error} />
                  <JobProgress job={kneeJob.job} error={kneeJob.error} />
                  <JobProgress job={psnrJob.job} error={psnrJob.error} />
                </CardBody>
              </Card>
            )}
          </div>
        )}
      </LoadState>
      {pullOpen && <PullDialog open={pullOpen} onOpenChange={setPullOpen} />}
    </Page>
  );
}
