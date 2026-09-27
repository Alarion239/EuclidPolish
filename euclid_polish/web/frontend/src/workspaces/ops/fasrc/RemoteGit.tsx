/* Ops › FASRC › Git: the FASRC checkout vs this laptop's HEAD, `git pull`
 * on FASRC, and the conda-env update as a local job whose output streams
 * into its log (offered automatically when a pull touched environment.yml). */
import { useState } from "react";
import { Link } from "react-router-dom";
import { apiPost } from "../../../api/client";
import { useJob } from "../../../api/jobs";
import { invalidate, useResource } from "../../../api/query";
import { usePageActions } from "../../../app/palette";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, DefList, IconButton, JobProgress, LogTail, LogView, Skeleton,
  Tooltip, confirm, toast,
} from "../../../ui";
import type { GitPullResp, RemoteGit } from "../api";
import { relationText } from "../model";

export function RemoteGitPanel({ fasrcConnected }: { fasrcConnected: boolean }) {
  const git = useResource<RemoteGit>(fasrcConnected ? "/api/fasrc/git-status" : null, [fasrcConnected], { ttl: 60_000 });
  const env = useJob("fasrc:env-update");
  const [pulling, setPulling] = useState(false);
  const [pullOut, setPullOut] = useState<GitPullResp | null>(null);
  const d = git.data;
  const rel = relationText(d?.relation as never);

  async function pull() {
    if (!(await confirm({ title: `git pull on FASRC (${d?.branch ?? "branch"})?`,
      message: "Fast-forward only. Running jobs keep their already-loaded code.", confirmLabel: "Pull" }))) return;
    setPulling(true); setPullOut(null);
    try {
      const r = await apiPost<GitPullResp>("/api/fasrc/git-pull");
      setPullOut(r);
      if (r.ok) toast.success(r.changed_files?.length ? `Pulled ${r.changed_files.length} changed files` : "Already up to date");
      else toast.error(r.error || "git pull failed");
    } catch (e) {
      setPullOut({ ok: false, error: e instanceof Error ? e.message : String(e) });
    } finally { setPulling(false); void invalidate("/api/fasrc/git-status"); }
  }
  async function updateEnv() {
    if (!(await confirm({ title: "Update the conda environment on FASRC?",
      message: "Runs `yes | mamba env update -f environment.yml` (minutes). A cancel can leave the env half-updated; re-run to finish.",
      confirmLabel: "Update env" }))) return;
    await env.run("/api/fasrc/env-update", {}, {
      onDone: (j) => { if (j.status === "done") toast.success("Conda environment updated"); },
    });
  }
  usePageActions([
    { id: "fasrc-git-pull", label: "git pull on FASRC", group: "FASRC", disabled: !fasrcConnected, run: () => void pull() },
    { id: "fasrc-env-update", label: "Update the FASRC conda environment", group: "FASRC", keywords: ["mamba", "environment.yml"],
      disabled: !fasrcConnected || env.busy, run: () => void updateEnv() },
  ]);

  if (!fasrcConnected) return <Callout tone="warn" title="FASRC offline">Connect to read the FASRC checkout.</Callout>;
  return (
    <div className="ops-stack">
      <Card>
        <CardHead title="FASRC checkout" sub={d?.repo ? <code className="mono">{d.repo}</code> : undefined} right={
          <div className="ops-row">
            <Button size="sm" variant="primary" loading={pulling} onClick={pull}>git pull</Button>
            <Button size="sm" loading={env.busy} onClick={updateEnv}>Update env</Button>
            <IconButton size="sm" icon="reset" label="Refresh (fetches on FASRC)" onClick={() => git.reload()} />
          </div>} />
        <CardBody>
          {git.loading && !d ? <Skeleton lines={4} /> : git.error && !d ? <Callout tone="bad">{git.error.message}</Callout> : d && (
            <>
              <div className="ops-row ops-gitrel">
                <Tooltip content={rel.hint}><span tabIndex={0}><Badge tone={rel.tone}>{rel.label}</Badge></span></Tooltip>
                {(d.ahead ?? 0) > 0 && <Badge size="sm" tone="warn">↑ {d.ahead} vs upstream</Badge>}
                {(d.behind ?? 0) > 0 && <Badge size="sm" tone="warn">↓ {d.behind} vs upstream</Badge>}
                {d.dirty && <Badge size="sm" tone="warn">{d.dirty_files?.length} dirty</Badge>}
              </div>
              <DefList dense items={[
                ["branch", <code className="mono">{d.branch}</code>],
                ["FASRC HEAD", <code className="mono">{d.head?.slice(0, 10) || "—"}</code>],
                ["local HEAD", <span><code className="mono">{d.local_head?.slice(0, 10) || "—"}</code> <Link to="/ops/git" className="ops-small">local git</Link></span>],
                ["last commit", d.last?.hash ? <span><code className="mono">{d.last.hash}</code> {d.last.subject} <span className="ops-dim">· {d.last.relative}</span></span> : "—"],
              ]} />
              {d.dirty && d.dirty_files && <LogTail text={d.dirty_files.join("\n")} style={{ maxHeight: 160 }} />}
            </>
          )}
          {pullOut && (
            <Callout tone={pullOut.ok ? (pullOut.env_update_needed ? "warn" : "good") : "bad"}
              title={pullOut.ok ? (pullOut.env_update_needed ? "environment.yml changed" : "Pulled") : "git pull failed"}
              action={pullOut.env_update_needed ? <Button size="sm" onClick={updateEnv}>Update env</Button> : undefined}
              onDismiss={() => setPullOut(null)}>
              <span className="ops-pre">{pullOut.ok ? (pullOut.stdout || "Already up to date.") : pullOut.error}</span>
            </Callout>
          )}
        </CardBody>
      </Card>
      {(env.job || env.error) && (
        <Card>
          <CardHead title="Conda environment update" sub="mamba env update · streamed from FASRC" />
          <CardBody>
            <JobProgress job={env.job ? { ...env.job, log: null } : null} error={env.error} />
            {env.job && <LogView text={env.job.log} title="mamba" exportName="fasrc-env-update" maxHeight={420} />}
          </CardBody>
        </Card>
      )}
    </div>
  );
}
