/* Models › Members, roster view: ONE table of the regime's active members
   (members.json: status + origin.json + training job + knee-integrated PSNR
   + production-gate share + coherence). Default columns: Member, Status,
   Steps, Recipe (loss · knee), ∫PSNR and the Gate share bar (the peak share
   when the payload has it: what decides pruning); per-band, Test VIS, Test
   4b and the telemetry are in the Columns menu. The TIMEOUT / multi-knee chips
   write the DataTable filter (m.q). The selection toolbar is always visible:
   Continue, Fork, Curves, Images, and Archive set apart (confirmed; typed for
   more than 3). The selection is the shared "member" set (Combiner's "Open in
   Members with this selection" fills it). A row opens the member inspector. */
import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { apiGet, apiPost } from "../../../api/client";
import { invalidate } from "../../../api/query";
import type { Job } from "../../../api/jobs";
import { pagePath } from "../../../app/nav";
import { usePageActions } from "../../../app/palette";
import { formatDate } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { useSelected, useSelection } from "../../../state/selection";
import { Badge, Button, Caption, Chip, DataTable, Tooltip, confirm, toast, type DataColumn } from "../../../ui";
import { BAND_SHORT, BANDS, type MemberRow, type MembersPayload, type Mode } from "../api";
import { ShareBar, useFacetColors } from "../common";
import { GATE_BANDS, db, gatePeak, gateShare, kneeText, memberNumber, share as sharePct, stepsText, usedByGate } from "../model";

const tabPath = (mode: Mode, tab: string) => pagePath("models", { tab, params: { mode } });
const num = (v: number | null | undefined, d = 2) => (v == null || !Number.isFinite(v) ? null : Number(v.toFixed(d)));

function Progress({ m }: { m: MemberRow }) {
  const f = m.fraction ?? (m.step && m.target_steps ? m.step / m.target_steps : null);
  const title = m.status === "timeout"
    ? `Stopped at ${m.step?.toLocaleString()} of ${m.target_steps?.toLocaleString()} steps after its job ended (${m.job?.state ?? "no job record"}${m.job?.req_time_limit ? `, limit ${m.job.req_time_limit}` : ""})`
    : m.status === "running" ? `Job ${m.job?.jobid} is ${m.job?.state?.toLowerCase()}` : `${m.step?.toLocaleString() ?? "?"} steps`;
  return (
    <Tooltip content={title}>
      <span className="mdl-progress" data-status={m.status} tabIndex={0}>
        <span className="mdl-progress__track"><span className="mdl-progress__fill" style={{ width: `${Math.round(100 * (f ?? 0))}%` }} /></span>
        <span className="mdl-num">{stepsText(m.step, m.target_steps)}</span>
      </span>
    </Tooltip>
  );
}

/** A status badge only on a problem or a running job; done is quiet text. */
function Status({ m }: { m: MemberRow }) {
  if (m.status === "timeout") return <Badge tone="warn">TIMEOUT</Badge>;
  if (m.status === "running") return <Badge tone="info" dot>running</Badge>;
  if (m.status === "complete") return <span className="mdl-muted">done</span>;
  return <span className="mdl-muted">unknown</span>;
}

/** A column header with its metric definition on hover. */
function DefHead({ def, children }: { def: string; children: string }) {
  return <Tooltip content={def}><span className="mdl-defhead">{children}</span></Tooltip>;
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

export function Roster({ data, mode }: { data: MembersPayload; mode: Mode }) {
  const navigate = useNavigate();
  const rows = data.members;
  const [query, setQuery] = useUrlState("m.q", "");
  const selected = useSelected("member");
  const setSelected = (keys: string[]) => useSelection.getState().select("member", keys);
  const [archiving, setArchiving] = useState<string | null>(null);
  const colors = useFacetColors(rows, "loss");
  const share = useMemo(() => new Map(rows.map((r) => [r.name, gateShare(r)])), [rows]);
  const maxShare = useMemo(() => [...share.values()].reduce((m, s) => Math.max(m, s.value ?? 0), 0), [share]);
  const sel = selected.filter((k) => rows.some((r) => r.name === k));
  const nTimeout = rows.filter((m) => m.timeout).length;
  const nMulti = rows.filter((m) => kneeText(m).kind === "multi").length;

  const psnrFields = data.psnr_fields;
  const visMeta = data.vis_psnr;
  const columns = useMemo<DataColumn<MemberRow>[]>(() => [
    { id: "name", header: "Member", headerText: "member", accessor: (m) => m.name, width: 96,
      sortFn: (a, b) => Number(memberNumber(a.name)) - Number(memberNumber(b.name)),
      cell: (m) => <span className="mdl-member"><span className="mdl-swatch" style={{ ["--sw" as string]: colors.of(m) }} />#{memberNumber(m.name)}</span> },
    { id: "status", header: "Status", accessor: (m) => m.status, cell: (m) => <Status m={m} />, width: 92, priority: 1 },
    { id: "steps", header: "Steps", accessor: (m) => m.fraction ?? null, cell: (m) => <Progress m={m} />, width: 150, csv: (m) => m.step ?? "", priority: 3 },
    { id: "recipe", header: "Recipe", headerText: "recipe (loss · knee)", width: 168, priority: 2,
      accessor: (m) => `${m.loss.toUpperCase()} · ${kneeText(m).text}`,
      sortFn: (a, b) => a.loss.localeCompare(b.loss) || kneeText(a).sort - kneeText(b).sort,
      cell: (m) => { const k = kneeText(m); return (
        <Tooltip content={k.title}><span className="mdl-recipe" tabIndex={0}>{m.loss.toUpperCase()} · <span className={k.kind === "default" ? "mdl-muted" : undefined}>{k.text}</span></span></Tooltip>
      ); } },
    { id: "knee_mean", header: "∫PSNR", headerText: "integrated psnr [dB]", numeric: true, width: 72, accessor: (m) => num(m.knee_integrated?.mean, 3),
      cell: (m) => <span className="mdl-num"><b>{db(m.knee_integrated?.mean)}</b></span> },
    { id: "gate", header: <DefHead def="The member's share of the production gate's weight: the peak (the largest share in any band and brightness bin, what decides whether the gate reads it) beside the all-pixel mean over VIS, Y, J and H. Faint bars: members the gate does not read. Per-band shares are in the Columns menu.">Gate share</DefHead>,
      headerText: "gate share (peak, else mean over bands)", accessor: (m) => num(share.get(m.name)?.value, 5), width: 232,
      csv: (m) => share.get(m.name)?.text ?? "",
      cell: (m) => { const s = share.get(m.name); return s?.value == null ? <ShareBar value={null} max={maxShare} text="—" /> : (
        <Tooltip content={`${s.text}${usedByGate(m) === false ? " · not read by the gate" : ""}`}>
          <span tabIndex={0}><ShareBar value={s.value} max={maxShare} text={s.short} read={usedByGate(m)} /></span>
        </Tooltip>
      ); } },
    ...BANDS.map((b): DataColumn<MemberRow> => ({
      id: `knee_${b}`, header: `∫${BAND_SHORT[b]}`, headerText: `integrated ${BAND_SHORT[b]}`, numeric: true,
      accessor: (m) => num(m.knee_integrated?.[b], 3), cell: (m) => db(m.knee_integrated?.[b]), hidden: true, width: 64,
    })),
    { id: "knee_rank", header: "∫ rank", numeric: true, accessor: (m) => m.knee_rank ?? null, width: 64, hidden: true },
    { id: "vis_psnr", header: <DefHead def={`Test PSNR, VIS asinh${visMeta?.knee_e ? ` at the ${visMeta.knee_e} e⁻ knee` : ""}${visMeta?.n_scored ? ` over ${visMeta.n_scored} fields` : ""}, from the last evaluation.`}>Test VIS</DefHead>,
      headerText: "test VIS psnr", numeric: true, width: 76, hidden: true, accessor: (m) => num(m.vis_psnr, 3), cell: (m) => db(m.vis_psnr, 3) },
    { id: "psnr", header: <DefHead def={`Test PSNR, joint 4-band asinh (the training psnr_stretched) over ${psnrFields ?? "—"} test fields, from the per-member PSNR cache (Member PSNR). Higher than Test VIS by construction; not comparable with it.`}>Test 4b</DefHead>,
      headerText: "test 4-band psnr", numeric: true, width: 72, hidden: true, accessor: (m) => num(m.psnr, 3), cell: (m) => db(m.psnr, 3) },
    { id: "gate_read", header: "Read by gate", headerText: "read by the production gate", width: 96, hidden: true,
      accessor: (m) => { const u = usedByGate(m); return u == null ? null : u ? "read" : "not read"; } },
    { id: "gate_peak", header: "Gate peak", headerText: "gate share peak (max over bands and brightness bins)", numeric: true, hidden: true, width: 80,
      accessor: (m) => num(gatePeak(m).v, 5), cell: (m) => sharePct(gatePeak(m).v) },
    ...GATE_BANDS.map(({ band, short }): DataColumn<MemberRow> => ({
      id: `gate_${short}`, header: `Gate ${short}`, headerText: `gate share ${short}`, numeric: true, hidden: true, width: 72,
      accessor: (m) => num(m.gate_usage?.[band], 5), cell: (m) => sharePct(m.gate_usage?.[band]),
    })),
    { id: "coh_sr", header: "Coh SR", headerText: "coherence SR", numeric: true, width: 64, hidden: true, accessor: (m) => num(m.coherence?.sr, 3) },
    { id: "coh_all", header: "Coh all", headerText: "coherence overall", numeric: true, accessor: (m) => num(m.coherence?.overall, 3), hidden: true },
    { id: "depth", header: "Depth", numeric: true, accessor: (m) => m.blocks ?? null, hidden: true },
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
  ], [colors, share, maxShare, psnrFields, visMeta]);

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

  const go = (tab: string, q: string) => navigate(`${tabPath(mode, tab)}?${q}`);
  const numbers = sel.map((n) => memberNumber(n)).join(",");
  const actions = {
    continue: () => go("train", `mode=continue&members=${sel.join(",")}`),
    fork: () => go("train", `mode=fork&member=${sel[0]}`),
    images: () => go("images", `sel=${numbers}`),
    curves: () => go("members", `view=curves&sel=${numbers}`),
  };
  usePageActions([
    { id: "mem-continue", label: "Continue the selected members", group: "Members", disabled: !sel.length, run: actions.continue },
    { id: "mem-fork", label: "Fork the selected member", group: "Members", disabled: sel.length !== 1, run: actions.fork },
    { id: "mem-archive", label: "Archive the selected members…", group: "Members", disabled: !sel.length, run: () => void archive() },
    { id: "mem-images", label: "Show the selected members in Images", group: "Members", disabled: !sel.length, run: actions.images },
    { id: "mem-curves", label: "Compare the selected members' curves", group: "Members", disabled: !sel.length, run: actions.curves },
    { id: "mem-timeout", label: "Show TIMEOUT members", group: "Members", run: () => setQuery("status:timeout") },
  ]);

  const none = !sel.length;
  const hint = none ? "Select members in the table first" : undefined;
  const toolbar = (
    <div className="mdl-row mdl-seltools" role="group" aria-label="Selection">
      <Chip on={query === "status:timeout"} onClick={() => setQuery(query === "status:timeout" ? "" : "status:timeout")} disabled={!nTimeout}
        title="Members that stopped before their target steps">TIMEOUT {nTimeout}</Chip>
      <Chip on={query === "recipe:multi"} onClick={() => setQuery(query === "recipe:multi" ? "" : "recipe:multi")} disabled={!nMulti}
        title="Members trained at several knees at once">multi-knee {nMulti}</Chip>
      <span className="mdl-seltools__sep" aria-hidden />
      <span className="mdl-muted mdl-seltools__count" aria-live="polite">{none ? "None selected" : `${sel.length} selected`}</span>
      <Button size="sm" disabled={none} title={hint} onClick={actions.continue}>Continue</Button>
      <Button size="sm" disabled={sel.length !== 1} title={sel.length > 1 ? "Fork one member at a time" : hint} onClick={actions.fork}>Fork</Button>
      <Button size="sm" disabled={none} title={hint} onClick={actions.curves}>Curves</Button>
      <Button size="sm" disabled={none} title={hint} onClick={actions.images}>Images</Button>
      <span className="mdl-seltools__apart" />
      <Button size="sm" variant="danger" disabled={none} title={hint} loading={archiving != null} onClick={() => void archive()}>Archive</Button>
    </div>
  );
  return (
    <div className="mdl-stack">
      {(data.knee.stale || data.gate.stale) && (
        <p className="mdl-note">
          {data.knee.stale && <Badge tone="warn">∫PSNR stale</Badge>}{" "}
          {data.gate.stale && <Badge tone="warn">gate share from a stale gate</Badge>}
        </p>
      )}
      <DataTable rows={rows} columns={columns} rowKey={(m) => m.name} aria-label={`${mode} members`}
        selectable selected={sel} onSelectedChange={(keys) => setSelected(keys)}
        inspect={(m) => ({ kind: "member", id: m.name })} exportName={`ensemble-members-${mode}`}
        urlKey="m" height={560} toolbar={toolbar} filterPlaceholder="Filter: recipe:l2  status:timeout  knee_mean>59"
        defaultSort={[{ id: "knee_mean", desc: true }]}
        empty={`No ${mode} members${data.other_regime_members ? ` (${data.other_regime_members} in the other regime)` : ""}: train some, then pull them.`} />
      {archiving && <span className="mdl-muted">Archiving {archiving}…</span>}
      <Caption>
        Sorted by ∫PSNR over {data.knee.n_fields ?? "the"} test fields; the ranking with the plain mean and the production gate
        is the <Link to={tabPath(mode, "leaderboard")}>Leaderboard</Link>.
      </Caption>
    </div>
  );
}
