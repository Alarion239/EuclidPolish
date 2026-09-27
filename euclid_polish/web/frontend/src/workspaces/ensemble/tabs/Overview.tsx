/* ensemble/overview (spec §8.2): the headline numbers, each with its exact
   definition (production spatial gate, plain mean, best member, knee-
   integrated), the staleness checks with their fix, and the run actions —
   evaluate (optionally forced), refresh member PSNR, PSNR vs knee, and a
   FASRC pull with a member picker (probe first, then pull what you pick). */
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useJob, type Job } from "../../../api/jobs";
import { pagePath } from "../../../app/nav";
import { usePageActions } from "../../../app/palette";
import { startJob } from "../../../app/RunActions";
import { useFasrcStatus } from "../../../app/status";
import { formatDateTime, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, Checkbox, Dialog, EmptyState, Input, JobProgress, Kpi,
  NumberField, Page, Tooltip,
} from "../../../ui";
import { useMembers, useMode, useOverview, type Check, type Mode, type Overview as OverviewData } from "../api";
import { BarGroup, EnsBar, LoadState } from "../common";
import { JOB, useOnJobEnd } from "../jobs";
import { db, dbDelta, deltaTone, memberNumber, parseMemberList } from "../model";
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

function Headline({ o, mode }: { o: OverviewData; mode: Mode }) {
  const h = o.headline;
  const fields = h.n_scored ? `${h.n_scored} test fields` : "the test fields";
  const metric = `VIS asinh PSNR (knee ${h.knee_e ?? 100} e⁻) over ${fields}`;
  const k = h.knee;
  const kneeRange = k.integration ? `${k.integration.from_e}–${k.integration.to_e} e⁻` : "0.1–10⁴ e⁻";
  const kneeDelta = k.production != null && k.best_member != null ? k.production - k.best_member : null;
  const kneeVsMean = k.production != null && k.mean != null ? k.production - k.mean : null;
  const best = h.best_member.label ? `member ${memberNumber(h.best_member.label) ?? h.best_member.label}` : "best member";
  return (
    <div className="ens-kpis">
      <Kpi label="∫PSNR · production gate" value={db(k.production)} unit=" dB" loading={false}
        to={tabPath(mode, "knee")}
        delta={kneeDelta != null ? `${dbDelta(kneeDelta)} dB vs ${k.best_member_label ? `member ${memberNumber(k.best_member_label)}` : "best member"}` : undefined}
        deltaTone={deltaTone(kneeDelta)}
        footer={kneeVsMean != null ? `${dbDelta(kneeVsMean)} dB vs plain mean${k.stale ? " · stale" : ""}` : k.available ? undefined : "not computed"}
        tone={k.stale ? "warn" : undefined}
        hint={`Knee-integrated PSNR: the production spatial gate's PSNR averaged uniformly in log(knee) over ${kneeRange}, then over the four bands (${k.n_fields ?? "?"} test fields). The knee-independent metric.`} />
      <Kpi label="Test PSNR · production gate" value={db(h.production.psnr)} unit=" dB"
        delta={h.production.vs_best_member_db != null ? `${dbDelta(h.production.vs_best_member_db)} dB vs best member` : undefined}
        deltaTone={deltaTone(h.production.vs_best_member_db)}
        footer={h.production.vs_mean_db != null ? `${dbDelta(h.production.vs_mean_db)} dB vs plain mean` : "not evaluated"}
        hint={`${metric} of the production spatial gate (eval summary spatial_gate_combiner_psnr).`} />
      <Kpi label="Plain mean" value={db(h.mean.psnr)} unit=" dB"
        delta={h.mean.vs_mean_member_db != null ? `${dbDelta(h.mean.vs_mean_member_db)} dB vs mean member` : undefined}
        deltaTone={deltaTone(h.mean.vs_mean_member_db)}
        hint={`${metric} of the unweighted mean of the members (ensemble_psnr); the gain is over the average member (ensemble_gain_db).`} />
      <Kpi label="Best member" value={db(h.best_member.psnr)} unit=" dB" footer={h.best_member.label ? best : undefined}
        delta={h.best_member.mean_member_psnr != null ? `mean member ${db(h.best_member.mean_member_psnr)}` : undefined}
        to={tabPath(mode, "members")}
        hint={`${metric} of the single best member on the same fields (best_member_psnr).`} />
      <Kpi label="Members" value={String(o.n_members)} to={tabPath(mode, "members")}
        footer={o.production_gate.available ? `gate fitted for ${o.production_gate.n_members}${o.production_gate.mix_space ? ` · ${o.production_gate.mix_space} mix` : ""}` : "no production gate"}
        tone={o.production_gate.available && o.production_gate.n_members !== o.n_members ? "warn" : undefined}
        hint={`Active ${mode} members (registry). The production gate is fitted for a fixed member list.`} />
      <Kpi label="Evaluated" value={o.evaluated_at ? formatRelative(o.evaluated_at) : "never"}
        footer={o.evaluated_at ? formatDateTime(o.evaluated_at) : o.test_present ? "test records present" : "no test records"}
        hint={`When eval_summary.json was last written (${o.eval_subset ?? "test"} subset).`} />
    </div>
  );
}

function Checks({ checks, mode, onEvaluate, onKnee }: {
  checks: Check[]; mode: Mode; onEvaluate: () => void; onKnee: () => void;
}) {
  const bad = checks.filter((c) => !c.ok);
  return (
    <Card>
      <CardHead title="Staleness" sub={bad.length ? `${bad.length} to act on` : "everything current"} />
      <CardBody>
        <ul className="ens-checks">
          {checks.map((c) => (
            <li key={c.id} className="ens-check" data-tone={c.tone}>
              <span className="ens-check__dot" aria-hidden />
              <span>
                <span className="ens-check__title">{c.title}</span>{" "}
                <span className="ens-check__detail">{c.detail}</span>
              </span>
              {!c.ok && c.action === "evaluate" && <Button size="sm" onClick={onEvaluate}>Evaluate</Button>}
              {!c.ok && c.action === "knee" && <Button size="sm" onClick={onKnee}>Compute</Button>}
              {!c.ok && c.action === "combiners" && (
                <Button size="sm" asChild><Link to={tabPath(mode, "combiners")}>Combiners</Link></Button>
              )}
              {c.ok && <span />}
            </li>
          ))}
        </ul>
      </CardBody>
    </Card>
  );
}

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
  const timeouts = (members.data?.members ?? []).filter((m) => m.timeout);
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
        <Tooltip content={fasrc && !fasrc.ssh_connected ? `FASRC offline: ${fasrc.last_error ?? "not connected"}` : "Check FASRC and pull changed members"}>
          <span><Button size="sm" icon="download" onClick={() => setPullOpen(true)}>Pull…</Button></span>
        </Tooltip>
      </EnsBar>
      <LoadState loading={ov.loading} error={ov.error} onRetry={ov.reload}>
        {o && (
          <div className="ens-stack">
            {o.summary == null && (
              <EmptyState icon="activity" title={`No ${mode} evaluation yet`}
                action={<Button variant="primary" onClick={run.evaluate} disabled={!o.test_present}>Evaluate</Button>}>
                {o.test_present ? "Score the members, the mean and the combiners on the test records." : "No local test records — sync them in Data › Records."}
              </EmptyState>
            )}
            {o.checks.some((c) => !c.ok && c.tone !== "info") && (
              <Callout tone={o.checks.some((c) => !c.ok && c.tone === "bad") ? "bad" : "warn"}
                title={o.checks.filter((c) => !c.ok && c.tone !== "info").map((c) => c.title).join(" · ")}>
                {o.checks.find((c) => !c.ok && c.tone !== "info")?.detail}
              </Callout>
            )}
            <Headline o={o} mode={mode} />
            <div className="ens-grid">
              <Checks checks={o.checks} mode={mode} onEvaluate={run.evaluate} onKnee={run.knee} />
              <Card>
                <CardHead title="Training status" sub={`${members.data?.members.length ?? "…"} members`}
                  right={<Button size="sm" variant="ghost" asChild><Link to={tabPath(mode, "members")}>Members</Link></Button>} />
                <CardBody>
                  <LoadState loading={members.loading && !members.data} error={members.data ? null : members.error} onRetry={members.reload} lines={2}>
                  {!members.data ? null : timeouts.length === 0
                    ? <span className="ens-muted">{members.data.members.length ? "Every member reached its target steps." : "No members in this regime."}</span>
                    : (
                      <div className="ens-stack" style={{ gap: "var(--s2)" }}>
                        <span><Badge tone="warn">TIMEOUT</Badge> {timeouts.length} stopped short of their target:</span>
                        <div className="ens-row">
                          {timeouts.map((m) => (
                            <Badge key={m.name} tone="warn">#{memberNumber(m.name)} {Math.round((m.step ?? 0) / 1000)}k/{Math.round((m.target_steps ?? 0) / 1000)}k</Badge>
                          ))}
                        </div>
                        <Button size="sm" asChild>
                          <Link to={`${tabPath(mode, "train")}?mode=continue&members=${timeouts.map((m) => m.name).join(",")}`}>Continue them…</Link>
                        </Button>
                      </div>
                    )}
                  </LoadState>
                </CardBody>
              </Card>
            </div>
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
