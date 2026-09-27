/* ensemble/members (spec §8.2): ONE table joining status + origin.json +
   training job + knee-integrated PSNR per band + production-gate usage +
   coherence (GET /ensemble/members.json). Filter / sort / columns / CSV are
   the DataTable's (in the URL as m.q / m.sort); quick filters write the same
   m.q. Multi-select → continue / fork / archive (confirm) / show in the
   disagreement movie / compare curves. A row opens the member inspector.
   Archived members (tombstones) are listed below with restore-from-zip. */
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { apiGet, apiPost } from "../../../api/client";
import { invalidate } from "../../../api/query";
import { useJob, type Job } from "../../../api/jobs";
import { pagePath } from "../../../app/nav";
import { usePageActions } from "../../../app/palette";
import { formatDate, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { useSelected, useSelection } from "../../../state/selection";
import {
  Badge, Button, Card, CardBody, CardHead, Chip, DataTable, JobProgress, Page, Tooltip, confirm, toast,
  type DataColumn,
} from "../../../ui";
import { BAND_SHORT, BANDS, useMembers, useMode, type MemberRow, type Mode, type Tombstone } from "../api";
import { LoadState, useFacetColors } from "../common";
import { JOB, useOnJobEnd } from "../jobs";
import { GATE_BANDS, db, gateUsage, kneeText, memberNumber, stepsText, type GateUsage } from "../model";
import "../ensemble.css";

const tabPath = (mode: Mode, tab: string) => pagePath("ensemble", { tab, params: { mode } });
const pct = (v: number | null | undefined) => (v == null || !Number.isFinite(v) ? "—" : `${(100 * v).toFixed(1)}%`);
const num = (v: number | null | undefined, d = 2) => (v == null || !Number.isFinite(v) ? null : Number(v.toFixed(d)));

function Progress({ m }: { m: MemberRow }) {
  const f = m.fraction ?? (m.step && m.target_steps ? m.step / m.target_steps : null);
  const title = m.status === "timeout"
    ? `Stopped at ${m.step?.toLocaleString()} of ${m.target_steps?.toLocaleString()} steps after its job ended (${m.job?.state ?? "no job record"}${m.job?.req_time_limit ? `, limit ${m.job.req_time_limit}` : ""})`
    : m.status === "running" ? `Job ${m.job?.jobid} is ${m.job?.state?.toLowerCase()}` : `${m.step?.toLocaleString() ?? "?"} steps`;
  return (
    <Tooltip content={title}>
      <span className="ens-progress" data-status={m.status} tabIndex={0}>
        <span className="ens-progress__track"><span className="ens-progress__fill" style={{ width: `${Math.round(100 * (f ?? 0))}%` }} /></span>
        <span className="ens-num">{stepsText(m.step, m.target_steps)}</span>
      </span>
    </Tooltip>
  );
}

function StatusBadge({ m }: { m: MemberRow }) {
  if (m.status === "timeout") return <Badge tone="warn">TIMEOUT</Badge>;
  if (m.status === "running") return <Badge tone="info" dot>running</Badge>;
  if (m.status === "complete") return <Badge tone="good">done</Badge>;
  return <Badge>?</Badge>;
}

/** Gate use: the mean share of the production gate's weight over the bands,
 *  with one small bar per band (VIS Y J H) — the VIS weight alone hid
 *  members the gate leans on for NISP. `max` scales the bars (the table's
 *  largest band share). */
function GateUse({ usage, max }: { usage: GateUsage; max: number }) {
  if (usage.mean == null) return <span className="ui-dt__nil">—</span>;
  const text = usage.bands.map((b) => `${b.short} ${pct(b.v)}`).join(" · ");
  return (
    <Tooltip content={`Share of the production gate's weight: ${text} (mean ${pct(usage.mean)})`}>
      <span className="ens-gate" tabIndex={0} aria-label={`Gate use ${pct(usage.mean)} (mean over bands): ${text}`}>
        <span className="ens-num">{pct(usage.mean)}</span>
        <span className="ens-gate__bars" aria-hidden>
          {usage.bands.map((b) => (
            <i key={b.band} data-band={b.short} style={{ height: `${b.v == null ? 0 : Math.max(4, Math.min(100, (100 * b.v) / Math.max(max, 1e-9)))}%` }} />
          ))}
        </span>
      </span>
    </Tooltip>
  );
}

/** Archive members one after another (each job rewrites the registry). */
async function archiveAll(names: string[], onStep: (i: number) => void): Promise<string[]> {
  const failed: string[] = [];
  for (let i = 0; i < names.length; i++) {
    onStep(i);
    try {
      const r = await apiPost<{ job_id?: string; error?: string }>("/ensemble/archive-member", { member: names[i] });
      if (!r.job_id) throw new Error(r.error ?? "not started");
      for (;;) {
        await new Promise((res) => setTimeout(res, 800));
        const j = await apiGet<Job>(`/api/jobs/${r.job_id}`);
        if (j.status !== "running") { if (j.status !== "done") failed.push(names[i]); break; }
      }
    } catch {
      failed.push(names[i]);
    }
  }
  return failed;
}

/** A column header with its metric definition on hover. */
function DefHead({ def, children }: { def: string; children: string }) {
  return <Tooltip content={def}><span className="ens-defhead">{children}</span></Tooltip>;
}

export default function Members() {
  const mode = useMode();
  const navigate = useNavigate();
  const res = useMembers(mode);
  const rows = useMemo(() => res.data?.members ?? [], [res.data]);
  const [, setQuery] = useUrlState("m.q", "");
  const selected = useSelected("member");
  const setSelected = (keys: string[]) => useSelection.getState().select("member", keys);
  const [archiving, setArchiving] = useState<string | null>(null);
  const restore = useJob(JOB.restore);
  useOnJobEnd(restore.job);
  const colors = useFacetColors(rows, "loss");
  const usage = useMemo(() => new Map(rows.map((r) => [r.name, gateUsage(r.gate_usage)])), [rows]);
  const maxUsage = useMemo(() => [...usage.values()].reduce((m, u) => Math.max(m, u.max ?? 0), 0), [usage]);
  const sel = selected.filter((k) => rows.some((r) => r.name === k));

  const psnrFields = res.data?.psnr_fields;
  const visMeta = res.data?.vis_psnr;
  const columns = useMemo<DataColumn<MemberRow>[]>(() => [
    { id: "name", header: "Member", headerText: "member", accessor: (m) => m.name, width: 104,
      sortFn: (a, b) => Number(memberNumber(a.name)) - Number(memberNumber(b.name)),
      cell: (m) => <span className="ens-member"><span className="ens-swatch" style={{ ["--sw" as string]: colors.of(m) }} />#{memberNumber(m.name)}</span> },
    { id: "status", header: "Status", accessor: (m) => m.status, cell: (m) => <StatusBadge m={m} />, width: 92 },
    { id: "steps", header: "Steps", accessor: (m) => m.fraction ?? null, cell: (m) => <Progress m={m} />, width: 150, csv: (m) => m.step ?? "" },
    { id: "loss", header: "Loss", accessor: (m) => m.loss, cell: (m) => <Badge>{m.loss.toUpperCase()}</Badge>, width: 64 },
    { id: "knee", header: "Knee", accessor: (m) => kneeText(m).text, sortFn: (a, b) => kneeText(a).sort - kneeText(b).sort,
      cell: (m) => { const k = kneeText(m); return <Tooltip content={k.title}><span className="ens-mono" tabIndex={0}>{k.kind === "default" ? <span className="ens-muted">{k.text}</span> : k.text}</span></Tooltip>; }, width: 128 },
    { id: "knee_mean", header: "∫PSNR", headerText: "integrated psnr", numeric: true, width: 72, accessor: (m) => num(m.knee_integrated?.mean, 3),
      cell: (m) => <span className="ens-num"><b>{db(m.knee_integrated?.mean)}</b></span> },
    ...BANDS.map((b): DataColumn<MemberRow> => ({
      id: `knee_${b}`, header: `∫${BAND_SHORT[b]}`, headerText: `integrated ${BAND_SHORT[b]}`, numeric: true,
      accessor: (m) => num(m.knee_integrated?.[b], 3), cell: (m) => db(m.knee_integrated?.[b]), hidden: b !== "VIS", width: 64,
    })),
    { id: "knee_rank", header: "∫ rank", numeric: true, accessor: (m) => m.knee_rank ?? null, width: 64, hidden: true },
    { id: "vis_psnr", header: <DefHead def={`Test PSNR, VIS asinh${visMeta?.knee_e ? ` at the ${visMeta.knee_e} e⁻ knee` : ""}${visMeta?.n_scored ? ` over ${visMeta.n_scored} fields` : ""}, from the last evaluation: the metric of the Overview's Best member tile.`}>Test VIS</DefHead>,
      headerText: "test VIS psnr", numeric: true, width: 76, accessor: (m) => num(m.vis_psnr, 3), cell: (m) => db(m.vis_psnr, 3) },
    { id: "psnr", header: <DefHead def={`Test PSNR, joint 4-band asinh (the training psnr_stretched) over ${psnrFields ?? "—"} test fields, from the per-member PSNR cache (Refresh member PSNR). Higher than Test VIS by construction; not comparable with it.`}>Test 4b</DefHead>,
      headerText: "test 4-band psnr", numeric: true, width: 72, accessor: (m) => num(m.psnr, 3), cell: (m) => db(m.psnr, 3) },
    { id: "psnr_rank", header: "rank", numeric: true, accessor: (m) => m.psnr_rank ?? null, hidden: true },
    { id: "gate", header: <DefHead def="Share of the production gate's weight given to this member, averaged over VIS, Y, J and H; the bars show each band (hover for the numbers). Per-band columns are in the column menu.">Gate use</DefHead>,
      headerText: "gate use (mean over bands)", accessor: (m) => num(usage.get(m.name)?.mean, 5),
      cell: (m) => <GateUse usage={usage.get(m.name) ?? gateUsage(null)} max={maxUsage} />, width: 124 },
    ...GATE_BANDS.map(({ band, short }): DataColumn<MemberRow> => ({
      id: `gate_${short}`, header: `Gate ${short}`, headerText: `gate use ${short}`, numeric: true, hidden: true, width: 72,
      accessor: (m) => num(m.gate_usage?.[band], 5), cell: (m) => pct(m.gate_usage?.[band]),
    })),
    { id: "gate_src", header: "Gate use, sources", headerText: "gate use on source pixels (mean over bands)", hidden: true,
      accessor: (m) => num(gateUsage(m.gate_usage_source).mean, 5), cell: (m) => pct(gateUsage(m.gate_usage_source).mean) },
    { id: "coh_sr", header: "Coh SR", headerText: "coherence SR", numeric: true, width: 64, accessor: (m) => num(m.coherence?.sr, 3) },
    { id: "coh_all", header: "Coh all", headerText: "coherence overall", numeric: true, accessor: (m) => num(m.coherence?.overall, 3), hidden: true },
    { id: "depth", header: "Depth", numeric: true, accessor: (m) => m.blocks ?? null, hidden: true },
    { id: "output_knee", header: "Out knee", numeric: true, accessor: (m) => m.output_knee ?? null, hidden: true },
    { id: "knee_loss", header: "Knee loss", accessor: (m) => m.knee_loss ?? null, hidden: true },
    { id: "noise_aug", header: "Noise aug", numeric: true, accessor: (m) => m.noise_aug ?? null, hidden: true },
    { id: "bootstrap", header: "Bootstrap", numeric: true, accessor: (m) => m.bootstrap ?? null, hidden: true },
    { id: "icnr", header: "ICNR", accessor: (m) => (m.icnr ? "yes" : "no"), hidden: true },
    { id: "seed", header: "Seed", numeric: true, accessor: (m) => m.seed ?? null, hidden: true },
    { id: "commit", header: "Commit", accessor: (m) => m.commit ?? null, cell: (m) => <code>{m.commit ?? "—"}</code>, hidden: true },
    { id: "forked_from", header: "Forked from", accessor: (m) => m.forked_from ?? null, hidden: true },
    { id: "created", header: "Created", accessor: (m) => m.created_at ?? null, cell: (m) => (m.created_at ? formatDate(m.created_at) : "—"), hidden: true },
    { id: "job", header: "Job", accessor: (m) => m.job?.jobid ?? null, hidden: true },
    { id: "gpu", header: "GPU util", numeric: true, accessor: (m) => m.job?.gpu_util_mean ?? null,
      cell: (m) => (m.job?.gpu_util_mean != null ? `${Math.round(m.job.gpu_util_mean)}%` : "—"), hidden: true },
    { id: "size", header: "Size", numeric: true, accessor: (m) => m.size_mb ?? null, cell: (m) => (m.size_mb != null ? `${m.size_mb} MB` : "—"), hidden: true },
  ], [colors, usage, maxUsage, psnrFields, visMeta]);

  async function archive() {
    const names = sel;
    if (!names.length) return;
    const ok = await confirm({
      title: `Archive ${names.length === 1 ? names[0] : `${names.length} members`}?`,
      message: "Each checkpoint is zipped to the tracking campaign, tombstoned and deleted (also on FASRC when connected). The evaluation rebuilds from cached cubes on the next Evaluate. A member can be restored from its zip.",
      tone: "danger", confirmLabel: "Archive",
      ...(names.length > 3 ? { requireText: "archive" } : {}),
    });
    if (!ok) return;
    const failed = await archiveAll(names, (i) => setArchiving(`${i + 1}/${names.length} ${names[i]}`));
    setArchiving(null);
    setSelected([]);
    void invalidate("/ensemble/");
    if (failed.length) toast.error(`Archive failed for ${failed.join(", ")}`, { description: "Open the job log in the tray." });
    else toast.success(`Archived ${names.length} member${names.length > 1 ? "s" : ""}`);
  }

  const go = (tab: string, query: string) => navigate(`${tabPath(mode, tab)}?${query}`);
  const numbers = sel.map((n) => memberNumber(n)).join(",");
  const actions = {
    continue: () => go("train", `mode=continue&members=${sel.join(",")}`),
    fork: () => go("train", `mode=fork&member=${sel[0]}`),
    disagreement: () => go("disagreement", `sel=${numbers}`),
    curves: () => go("curves", `sel=${numbers}`),
  };
  usePageActions([
    { id: "mem-continue", label: "Continue the selected members", group: "Members", disabled: !sel.length, run: actions.continue },
    { id: "mem-fork", label: "Fork the selected member", group: "Members", disabled: sel.length !== 1, run: actions.fork },
    { id: "mem-archive", label: "Archive the selected members…", group: "Members", disabled: !sel.length, run: () => void archive() },
    { id: "mem-disagree", label: "Show the selected members in the disagreement movie", group: "Members", disabled: sel.length < 2, run: actions.disagreement },
    { id: "mem-curves", label: "Compare the selected members' curves", group: "Members", disabled: !sel.length, run: actions.curves },
    { id: "mem-timeout", label: "Show TIMEOUT members", group: "Members", run: () => setQuery("status:timeout") },
    { id: "mem-refresh", label: "Refresh the members table", group: "Members", run: () => void res.reload() },
  ]);

  const toolbar = (
    <div className="ens-row">
      <Chip onClick={() => setQuery("status:timeout")} title="Members that stopped before their target steps">TIMEOUT</Chip>
      <Chip onClick={() => setQuery("knee:multi")} title="Members trained at several knees at once">multi-knee</Chip>
      <Chip onClick={() => setQuery("")}>all</Chip>
      {sel.length > 0 && <>
        <span className="ens-bar__sep" aria-hidden />
        <Badge tone="accent">{sel.length} selected</Badge>
        <Button size="sm" onClick={actions.continue}>Continue</Button>
        <Button size="sm" disabled={sel.length !== 1} onClick={actions.fork}>Fork</Button>
        <Button size="sm" disabled={sel.length < 2} onClick={actions.disagreement}>Disagreement</Button>
        <Button size="sm" onClick={actions.curves}>Curves</Button>
        <Button size="sm" variant="danger" loading={archiving != null} onClick={() => void archive()}>Archive</Button>
      </>}
    </div>
  );

  const data = res.data;
  return (
    <Page>
      <LoadState loading={res.loading} error={res.error} onRetry={res.reload}>
        {data && (
          <div className="ens-stack">
            {(data.knee.stale || data.gate.stale) && (
              <div className="ens-row ens-faint">
                {data.knee.stale && <Badge tone="warn">knee PSNR stale</Badge>}
                {data.gate.stale && <Badge tone="warn">gate usage from a stale gate</Badge>}
              </div>
            )}
            <DataTable rows={rows} columns={columns} rowKey={(m) => m.name} aria-label={`${mode} members`}
              selectable selected={sel} onSelectedChange={(keys) => setSelected(keys)}
              inspect={(m) => ({ kind: "member", id: m.name })} exportName={`ensemble-members-${mode}`}
              urlKey="m" height={560} toolbar={toolbar} filterPlaceholder="Filter: loss:l2  status:timeout  knee_mean>59"
              defaultSort={[{ id: "knee_mean", desc: true }]}
              empty={`No ${mode} members${data.other_regime_members ? ` (${data.other_regime_members} in the other regime)` : ""} — train some, then pull them.`} />
            {archiving && <span className="ens-muted">Archiving {archiving}…</span>}
            <Archived rows={data.archived} onRestore={(name) => restore.run("/ensemble/restore-member", { member: name })}
              busy={restore.busy} />
            <JobProgress job={restore.job} error={restore.error} />
          </div>
        )}
      </LoadState>
    </Page>
  );
}

const ARCHIVED_COLUMNS = (onRestore: (name: string) => void, busy: boolean): DataColumn<Tombstone>[] => [
  { id: "name", header: "Member", sortFn: (a, b) => Number(memberNumber(a.name)) - Number(memberNumber(b.name)),
    cell: (t) => <span className="ens-mono">#{memberNumber(t.name)}</span>, width: 90 },
  { id: "archived_at", header: "Archived", cell: (t) => (t.archived_at ? <Tooltip content={t.archived_at}><span tabIndex={0}>{formatRelative(t.archived_at)}</span></Tooltip> : "—") },
  { id: "commit", header: "Commit", cell: (t) => <code>{t.commit ?? "—"}</code> },
  { id: "zip", header: "Zip", accessor: (t) => t.zip ?? null,
    cell: (t) => (t.zip_found ? <span className="ens-mono">{t.zip?.replace(/^models\//, "")} <span className="ens-faint">· {t.campaign}</span></span>
      : <span className="ens-faint">not found</span>) },
  { id: "size", header: "Size", numeric: true, accessor: (t) => t.size_bytes ?? null,
    cell: (t) => (t.size_bytes != null ? `${(t.size_bytes / 1e6).toFixed(1)} MB` : "—") },
  { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 96,
    cell: (t) => (
      <Button size="sm" variant="ghost" disabled={!t.zip_found || busy}
        onClick={async () => {
          if (await confirm({ title: `Restore ${t.name}?`, message: "Unzips the archive back into the ensemble and makes the member active again. The evaluation and the production gate then read stale until re-run.", confirmLabel: "Restore" })) onRestore(t.name);
        }}>Restore</Button>
    ) },
];

function Archived({ rows, onRestore, busy }: { rows: Tombstone[]; onRestore: (name: string) => void; busy: boolean }) {
  const columns = useMemo(() => ARCHIVED_COLUMNS(onRestore, busy), [onRestore, busy]);
  return (
    <Card>
      <CardHead title="Archived members" sub={`${rows.length} tombstones · names are never reused`} />
      <CardBody>
        <DataTable rows={rows} columns={columns} rowKey={(t) => t.name} aria-label="Archived members"
          inspect={(t) => ({ kind: "member", id: t.name })} urlKey="a" height={320} dense
          exportName="ensemble-archived" empty="No archived members." />
      </CardBody>
    </Card>
  );
}
