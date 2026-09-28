/* Time travel: re-run a backup's exact code in an isolated sandbox (a git
 * worktree at its commit + a second console; optionally a FASRC worktree
 * and netscratch sandbox so jobs stay isolated too). One dialog for a whole
 * campaign, a model backup (its checkpoint seeds the sandbox) or a FITS /
 * image backup (the code at the commit it was saved at). */
import { useState } from "react";
import { apiPost } from "../../api/client";
import { invalidate } from "../../api/query";
import { Button, Callout, Dialog, DefList, Switch, toast } from "../../ui";
import { TRACKING_STATE_URL, type Commit, type RestoreResp } from "./api";
import { commitText } from "./model";

export type TimeTravelTarget = {
  campaign: string;          // "current" or an archived campaign's dir
  model?: string | null;     // a model backup's name (dir or .zip)
  /** A FITS / image backup: the code at its commit (nothing to seed). */
  backup?: { name: string; kind: "fits" | "image" } | null;
  title: string;
  commit?: Commit;
};

export function TimeTravelDialog({ target, onClose, fasrcConnected }: {
  target: TimeTravelTarget | null; onClose: () => void; fasrcConnected: boolean;
}) {
  const [remote, setRemote] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<RestoreResp | null>(null);
  const open = target != null;
  const zip = !!target?.model?.endsWith(".zip");

  async function start() {
    if (!target) return;
    setBusy(true); setResult(null);
    try {
      const r = await apiPost<RestoreResp>("/api/tracking/timetravel/restore", {
        campaign: target.campaign, model: target.model || undefined,
        backup: target.backup?.name, kind: target.backup?.kind, remote: remote ? "1" : "0",
      });
      setResult(r);
      if (r.ok) toast.success(`Sandbox ${r.short} is up`);
      else toast.error(r.error || "the sandbox server did not start");
    } catch (e) {
      setResult({ ok: false, error: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
      void invalidate(TRACKING_STATE_URL);
    }
  }
  const close = () => { setResult(null); setRemote(false); onClose(); };
  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) close(); }} title={`Time-travel to “${target?.title ?? ""}”`}
      description="Checks out the recorded commit in a sandbox worktree and starts a second console on it. Inputs are symlinked; outputs stay in the sandbox."
      footer={result?.ok ? <Button onClick={close}>Close</Button> : <>
        <Button variant="ghost" onClick={close}>Cancel</Button>
        <Button variant="primary" loading={busy} onClick={start} disabled={!target?.commit}>Start sandbox</Button>
      </>}>
      {target && (
        <div className="nb-stack">
          <DefList dense items={[
            ["commit", <code className="mono">{commitText(target.commit)}{typeof target.commit === "object" && target.commit?.branch ? ` (${target.commit.branch})` : ""}</code>],
            target.model ? ["model", <code className="mono">{target.model}</code>]
              : target.backup ? [target.backup.kind === "fits" ? "FITS" : "image", <code className="mono">{target.backup.name}</code>]
              : ["scope", "whole campaign"],
          ]} />
          {typeof target.commit === "object" && target.commit?.dirty && (
            <Callout tone="warn" title="Recorded from a dirty working tree">Uncommitted changes of that time are not reproduced.</Callout>
          )}
          {!target.commit && <Callout tone="bad">No git commit was recorded: the exact code cannot be restored.</Callout>}
          {zip && <Callout tone="info">A retired-model zip restores the code only; no checkpoint is seeded.</Callout>}
          <Switch checked={remote} onChange={setRemote} disabled={!fasrcConnected || busy}>
            Also prepare a FASRC sandbox{fasrcConnected ? "" : " (FASRC offline)"}
          </Switch>
          {result && !result.ok && <Callout tone="bad" title="Time travel failed"><span className="nb-pre">{result.error}</span></Callout>}
          {result?.ok && (
            <Callout tone={result.warning ? "warn" : "good"} title={`Sandbox ${result.short} is running`}
              action={result.url ? <Button asChild size="sm" variant="primary"><a href={result.url} target="_blank" rel="noreferrer noopener">Open console</a></Button> : undefined}>
              {result.warning || result.url}
              {result.remote && !result.remote.ok && <div className="nb-bad nb-small">FASRC half: {result.remote.error}</div>}
            </Callout>
          )}
        </div>
      )}
    </Dialog>
  );
}
