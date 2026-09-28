/* "Pull from FASRC…" (Models › Members; merges the old Overview Pull dialog
   and System › Storage's "Pull checkpoints"): check what changed on FASRC,
   pick members (or type them), pull just those or every changed one;
   unchanged members are never downloaded. The whole-ensemble mirror
   (rsync --delete-after, typed confirmation) sits apart as the last resort.
   Nothing runs until a button is pressed. */
import { useMemo, useState } from "react";
import { useJob, type Job } from "../../../api/jobs";
import { invalidate, useResource } from "../../../api/query";
import { useFasrcStatus } from "../../../app/status";
import { formatRelative } from "../../../format";
import { Button, Caption, Checkbox, Dialog, EmptyState, Input, JobProgress, confirm, toast } from "../../../ui";
import { url } from "../api";
import { JOB, useOnJobEnd } from "../jobs";
import { memberNumber, parseMemberList } from "../model";

type MirrorStatus = { local_dir?: string | null; last_run_at?: string | null; last_rc?: number | null; last_error?: string | null };

export function PullDialog({ open, onOpenChange, waiting }: {
  open: boolean; onOpenChange: (v: boolean) => void; waiting: readonly string[];
}) {
  const fasrc = useFasrcStatus().data;
  const offline = fasrc ? !fasrc.ssh_connected : false;
  const mirror = useResource<MirrorStatus>(open ? url.mirrorStatus() : null, [open], { ttl: 10_000 });
  const probe = useJob(JOB.pullCheck);
  const pull = useJob(JOB.pull);
  const mirrorJob = useJob(JOB.mirror);
  const [text, setText] = useState("");
  const [picked, setPicked] = useState<string[]>(() => [...waiting]);
  const changed = useMemo(() => {
    const r = probe.job?.status === "done" ? (probe.job.result as { changed?: string[] } | null) : null;
    return r?.changed ?? null;
  }, [probe.job]);
  const choices = changed ?? (waiting.length ? [...waiting] : null);
  const parsed = parseMemberList(text);
  const wanted = [...new Set([...picked, ...parsed.names])];
  const check = () => probe.run("/ensemble/pull", { dry_run: "1" }, { onDone: (j: Job) => {
    const r = j.result as { changed?: string[] } | null;
    setPicked(r?.changed ?? []);
  } });
  const doPull = (members: string[]) => pull.run("/ensemble/pull", members.length ? { members: members.join(",") } : {});
  useOnJobEnd(pull.job);
  async function mirrorAll() {
    if (!(await confirm({ title: "Mirror the whole ensemble from FASRC?",
      message: `rsync --delete-after into ${mirror.data?.local_dir || "the local ensemble dir"}: local files the FASRC copy lacks are DELETED.`,
      tone: "danger", confirmLabel: "Mirror and delete", requireText: "pull" }))) return;
    await mirrorJob.run("/api/fasrc/mirror/trigger", { confirm: "1" }, {
      onDone: (j) => {
        void invalidate("/api/fasrc/mirror/status");
        void invalidate("/ensemble/");
        if (j.status === "done") toast.success("Ensemble mirrored");
      },
    });
  }
  const last = mirror.data?.last_run_at;
  return (
    <Dialog open={open} onOpenChange={onOpenChange} size="lg" title="Pull from FASRC"
      description="Check what changed on FASRC, pick members, pull. Unchanged members are never downloaded."
      footer={<>
        <Button variant="ghost" onClick={() => onOpenChange(false)}>Close</Button>
        <Button disabled={offline || pull.busy} onClick={() => doPull([])}>Pull every changed member</Button>
        <Button variant="primary" disabled={offline || pull.busy || !wanted.length} loading={pull.busy}
          onClick={() => doPull(wanted)}>Pull {wanted.length || ""} selected</Button>
      </>}>
      <div className="mdl-stack">
        {offline && <EmptyState compact icon="warn" title="FASRC not connected">
          <span className="mdl-mono">{fasrc?.last_error ?? "connect in System › Connections"}</span>
        </EmptyState>}
        <div className="mdl-row">
          <Button icon="search" disabled={offline || probe.busy} loading={probe.busy} onClick={check}>Check FASRC</Button>
          <Input value={text} onChange={setText} placeholder="or type members: 195 196 199-202" aria-label="Members to pull"
            className="mdl-grow" />
        </div>
        {parsed.bad.length > 0 && <span className="mdl-warn">Not member names: {parsed.bad.join(", ")}</span>}
        {choices && (choices.length === 0
          ? <span className="mdl-muted">Every member is up to date on FASRC.</span>
          : (
            <div className="mdl-picker" role="group" aria-label="Changed members">
              {choices.map((name) => (
                <label key={name} className="mdl-pick" data-on={picked.includes(name)}>
                  <span className="mdl-pick__top">
                    <span>#{memberNumber(name)}</span>
                    <Checkbox checked={picked.includes(name)} aria-label={`Pull ${name}`}
                      onChange={(on) => setPicked((p) => (on ? [...p, name] : p.filter((x) => x !== name)))} />
                  </span>
                  <span className="mdl-pick__meta">{changed ? "changed on FASRC" : "finished on FASRC"}</span>
                </label>
              ))}
            </div>
          ))}
        <JobProgress job={probe.job} error={probe.error} />
        <JobProgress job={pull.job} error={pull.error} />
        <div className="mdl-apart">
          <div className="mdl-row">
            <Button size="sm" variant="danger" disabled={offline} loading={mirrorJob.busy} onClick={() => void mirrorAll()}>Mirror everything…</Button>
            <span className="mdl-muted">rsync of the whole ensemble; deletes local-only files.</span>
          </div>
          {last && <Caption>Last mirror {formatRelative(last)}{mirror.data?.last_rc ? ` · failed (rc ${mirror.data.last_rc})` : ""}{mirror.data?.last_error ? ` · ${mirror.data.last_error}` : ""}</Caption>}
          <JobProgress job={mirrorJob.job} error={mirrorJob.error} />
        </div>
      </div>
    </Dialog>
  );
}
