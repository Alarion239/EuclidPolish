/* Models › Combiner (`/models/combiner`; absorbs the old Ensemble
   Combiners tab). From the top:
   - the action bar: Fit variant…, Compare…, the compare-report select (only
     when reports exist), "Real holes from" (ONE Sky › Compare run, ?bench=,
     default the newest that scored production; a link opens it), the history
     chip (variants of earlier memberships and promotion backups, ?history=1)
     and Log to notebook;
   - the fit / compare / promote jobs while they run;
   - the variants table: production and the variants fitted for the current
     membership (model.ts variantScope) — Variant, Members, Mix, Held-out,
     ∫PSNR, Real holes VIS · Y · J · H % (the header links to the run they
     came from), Fitted, the gate-share link and a row menu (Inspect, Promote
     confirmed, Compare with production, Log to notebook);
   - the gate share per member of the production gate (members.json) or of a
     variant in the compare report (?share=), sortable, selectable, with
     "Open in Members with this selection" for pruning;
   - the held-out curves, the loss drawn only for fits on production's loss
     scale, each curve its own style (model.ts heldOutCurves);
   - the compare report, only when one exists (its blackout group only when
     the report scored blackout fields).
   The legacy RBF is never listed or offered. Opening the page reads caches
   only; every job asks first. */
import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Plot, { useLegend, type Series } from "../../../charts/Plot";
import { C, categorical } from "../../../colors";
import { useJob } from "../../../api/jobs";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { pagePath } from "../../../app/nav";
import { usePageActions } from "../../../app/palette";
import { formatDateTime, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { useSelected, useSelection } from "../../../state/selection";
import {
  Badge, Button, Caption, Checkbox, Chip, DataTable, Dialog, EmptyState, Field, IconButton, Input, JobProgress, Menu, MultiSelect,
  NumberField, Page, Segmented, Select, Skeleton, Toolbar, ToolbarGroup, ToolbarSeparator, ToolbarSpacer, Tooltip, confirm, type DataColumn,
} from "../../../ui";
import {
  BAND_SHORT, REGIME, url, useCombiners, useExperiment, useExperiments, useMembers, useModelCatalog, type CombinersPayload, type CompareReport, type Variant,
} from "../api";
import { LoadState, ShareBar } from "../common";
import { JOB, useOnJobEnd } from "../jobs";
import {
  benchFreshness, benchmarkChoices, benchmarkExperiment, compareRunPath, db, dbDelta, heldOutComparable, heldOutCurves, holesText, memberNumber,
  productionRunsText, productionShares, pruneThreshold, readsText, reportShares, variantLabel, variantScope,
  type Bench, type HeldOutMetric, type ShareRow,
} from "../model";
import { compareNote, compareRows, holesLine, promoteNote, utcText, variantNote } from "../notes";
import { LogToNotebookButton, useLogToNotebook } from "../../shared/LogToNotebook";
import "../models.css";

const tabPath = (tab: string) => pagePath("models", { tab });
const method = (v: Variant) => `gate:${v.name}`;
const meanOf = (vs: (number | null | undefined)[] | null | undefined) => {
  const f = (vs ?? []).filter((v): v is number => v != null && Number.isFinite(v));
  return f.length ? f.reduce((a, b) => a + b, 0) / f.length : null;
};

type Row = Variant & {
  testVis: number | null; kneeMean: number | null; blackoutVis: number | null;
  /** The run's real holes of THIS fit (its fingerprint matches the catalogue's), else null. */
  bench: Bench | null;
  /** The run scored an earlier fit of this spec: its numbers, shown only as stale. */
  staleBench: Bench | null;
  /** The run's score is being checked against the current fingerprints. */
  benchPending: boolean;
  lossComparable: boolean;
};

/* ── fit dialog ─────────────────────────────────────────────────────────── */
type FitKnobs = {
  out_name: string; mix_space: string; loss_knees: string; use_lr: boolean; width: string; steps: string;
  learning_rate: string; crop: string; batch_size: string; eval_every: string; holdout: string;
  blackout_fields: string; seed: string; num_images: string; members: string[]; compare_after: boolean; overwrite: boolean;
};
const DEFAULT_FIT: FitKnobs = {
  out_name: "", mix_space: "linear", loss_knees: "all", use_lr: false, width: "32", steps: "2000",
  learning_rate: "0.002", crop: "192", batch_size: "8", eval_every: "250", holdout: "15",
  blackout_fields: "40", seed: "0", num_images: "100", members: [], compare_after: true, overwrite: false,
};

function FitDialog({ open, onOpenChange, data, onStart }: {
  open: boolean; onOpenChange: (v: boolean) => void; data: CombinersPayload;
  onStart: (body: Record<string, string>) => void;
}) {
  const [k, setK] = useState<FitKnobs>({ ...DEFAULT_FIT, out_name: `trial_${new Date().toISOString().slice(5, 10).replace("-", "")}` });
  const set = <K extends keyof FitKnobs>(key: K, v: FitKnobs[K]) => setK((p) => ({ ...p, [key]: v }));
  const tableSel = useSelected("member");
  const name = k.out_name.trim().replace(/^spatial_gate_/, "");
  const exists = data.variants.some((v) => v.name === `spatial_gate_${name}`);
  const bad = !/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(name) ? "letters, digits, . _ - only"
    : name === "combiner" ? "that is the production gate" : name.startsWith("backup_") ? "reserved for promotion backups"
      : name.startsWith("comparison") || /(\.json|_evals)$/.test(name) ? "reserved for compare reports / eval sidecars"
      : exists && !k.overwrite ? "exists — allow overwrite or rename" : null;
  const options = data.active_members.map((l) => ({ value: l, label: `#${memberNumber(l) ?? l}` }));
  const submit = async () => {
    const ok = await confirm({
      title: `Fit gate variant spatial_gate_${name}?`,
      message: `A local TensorFlow job: ${Number(k.steps).toLocaleString()} steps on the validate cubes${k.compare_after ? ", then a compare on the test cubes" : ""}. It writes spatial_gate_${name}; production is never touched.`,
      confirmLabel: "Fit",
    });
    if (!ok) return;
    const body: Record<string, string> = { mode: REGIME };
    for (const [key, v] of Object.entries(k)) {
      if (key === "members") { if ((v as string[]).length) body.members = (v as string[]).map((l) => memberNumber(l)).join(","); continue; }
      body[key] = typeof v === "boolean" ? (v ? "1" : "0") : String(v);
    }
    body.out_name = name;
    onStart(body);
    onOpenChange(false);
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange} size="lg" title="Fit a spatial-gate variant"
      description="Fits on the cached validate member cubes (+ blackout copies) and saves a NAMED variant."
      footer={<>
        <Button variant="ghost" onClick={() => onOpenChange(false)}>Cancel</Button>
        <Button variant="primary" disabled={!!bad} onClick={() => void submit()}>Fit variant</Button>
      </>}>
      <div className="mdl-form">
        <Field label="Variant name" error={bad} className="mdl-form__wide">
          <Input value={k.out_name} onChange={(v) => set("out_name", v)} placeholder="trial_0926" />
        </Field>
        <Field label="Mix space" hint="linear: average the members in electrons (knee-free, flux-conserving, production). asinh: in band-knee asinh space.">
          <Select value={k.mix_space} onChange={(v) => set("mix_space", v)} options={[{ value: "linear", label: "linear (e⁻)" }, { value: "asinh", label: "asinh" }]} />
        </Field>
        <Field label="Loss knees" hint="all: the loss at 11 knees 0.1–10⁴ e⁻ (the knee-integrated PSNR, production). band: the band knee only. Or a comma list of knees in e⁻.">
          <Input value={k.loss_knees} onChange={(v) => set("loss_knees", v)} placeholder="all | band | 1,10,100" />
        </Field>
        <NumberField label="Width" value={k.width} onChange={(v) => set("width", v)} min={4} max={256} />
        <NumberField label="Steps" value={k.steps} onChange={(v) => set("steps", v)} min={1} max={200000} step={100} />
        <NumberField label="Learning rate" value={k.learning_rate} onChange={(v) => set("learning_rate", v)} min={1e-7} max={1} step="any" />
        <NumberField label="Crop" value={k.crop} onChange={(v) => set("crop", v)} min={32} max={1024} step={16} unit="px" />
        <NumberField label="Batch" value={k.batch_size} onChange={(v) => set("batch_size", v)} min={1} max={64} />
        <NumberField label="Eval every" value={k.eval_every} onChange={(v) => set("eval_every", v)} min={1} unit="steps" />
        <NumberField label="Held-out fields" value={k.holdout} onChange={(v) => set("holdout", v)} min={1} />
        <NumberField label="Blackout fields" value={k.blackout_fields} onChange={(v) => set("blackout_fields", v)} min={0} max={400} />
        <NumberField label="Seed" value={k.seed} onChange={(v) => set("seed", v)} min={0} />
        <NumberField label="Validate fields" value={k.num_images} onChange={(v) => set("num_images", v)} min={2} max={2000} />
        <Field label="Members (pruned gate)" className="mdl-form__wide"
          hint="Empty: every active member. Pick a subset for a pruned gate that reads only those members.">
          <div className="mdl-row">
            <MultiSelect value={k.members} onChange={(v) => set("members", v)} options={options} placeholder="all active members" maxSummary={6} />
            <Button size="sm" variant="ghost" disabled={!tableSel.length}
              onClick={() => set("members", data.active_members.filter((l) => tableSel.includes(`member_${memberNumber(l)}`)))}>
              Use Members selection ({tableSel.length})
            </Button>
          </div>
        </Field>
        <div className="mdl-row mdl-form__wide">
          <Checkbox checked={k.use_lr} onChange={(v) => set("use_lr", v)}>LR input</Checkbox>
          <Checkbox checked={k.compare_after} onChange={(v) => set("compare_after", v)}>compare with production afterwards</Checkbox>
          <Checkbox checked={k.overwrite} onChange={(v) => set("overwrite", v)}>overwrite an existing variant</Checkbox>
        </div>
      </div>
    </Dialog>
  );
}

function CompareDialog({ open, onOpenChange, data, onStart }: {
  open: boolean; onOpenChange: (v: boolean) => void; data: CombinersPayload; onStart: (body: Record<string, string>) => void;
}) {
  const gates = data.variants.filter((v) => v.kind === "gate" && v.applies_to_test_cubes);
  const [picked, setPicked] = useState<string[]>(() => variantScope(gates, false).shown.map((v) => v.name));
  const [blackout, setBlackout] = useState("40");
  const [knee, setKnee] = useState(true);
  const submit = () => {
    onStart({ gates: picked.join(","), blackout_fields: blackout, knee: knee ? "1" : "0" });
    onOpenChange(false);
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange} size="lg" title="Compare gate variants"
      description="Scores each variant, the plain mean and every member on the cached test cubes and blackout copies."
      footer={<>
        <Button variant="ghost" onClick={() => onOpenChange(false)}>Cancel</Button>
        <Button variant="primary" disabled={!picked.length} onClick={submit}>Compare {picked.length}</Button>
      </>}>
      <div className="mdl-stack">
        <div className="mdl-picker" role="group" aria-label="Variants to compare">
          {gates.map((v) => (
            <label key={v.name} className="mdl-pick" data-on={picked.includes(v.name)}>
              <span className="mdl-pick__top">
                <span>{variantLabel(v.name)}</span>
                <Checkbox checked={picked.includes(v.name)} aria-label={`Compare ${v.name}`}
                  onChange={(on) => setPicked((p) => (on ? [...p, v.name] : p.filter((x) => x !== v.name)))} />
              </span>
              <span className="mdl-pick__meta">{readsText(v)} · {v.mix_space}{v.production ? " · production" : v.backup ? " · backup" : ""}</span>
            </label>
          ))}
        </div>
        <div className="mdl-form">
          <NumberField label="Blackout fields" value={blackout} onChange={setBlackout} min={0} max={400}
            hint="0 = natural fields only (no member inference)" />
          <Checkbox checked={knee} onChange={setKnee}>PSNR-vs-knee curves</Checkbox>
        </div>
      </div>
    </Dialog>
  );
}

/* ── compare report table ───────────────────────────────────────────────── */
/** A squared error ÷ the best member's, or "—" (no score, or no reference). */
function ratio(v: number | null | undefined, ref: number | null | undefined): string {
  const x = v != null && ref ? v / Math.max(ref, 1e-30) : null;
  return x == null || !Number.isFinite(x) ? "—" : x.toFixed(3);
}

function ReportTable({ report }: { report: CompareReport }) {
  const [asked, setGroup] = useUrlState("group", "natural");
  // A report scored without blackout fields carries an empty blackout group
  // (null PSNRs, zero errors): it is never shown, whatever ?group= says.
  const hasBlackout = (report.n_fields?.blackout ?? 0) > 0 && !!report.groups.blackout;
  const group = report.groups[asked] && (asked !== "blackout" || hasBlackout) ? asked : "natural";
  const block = report.groups[group];
  if (!block) return null;
  const rows = compareRows(report, group);
  const best = rows.find((r) => r.startsWith("member:")) ?? null;
  const ref = best ? block[best] : null;
  const knee = report.knee?.methods ?? {};
  const vis = (r: string) => block[r].band_psnr[0] ?? -Infinity;
  return (
    <div className="mdl-stack mdl-stack--tight">
      <div className="mdl-row">
        <Segmented size="sm" aria-label="Field group" value={group} onChange={setGroup}
          options={[{ value: "natural", label: `natural ${report.n_fields?.natural ?? ""}`.trim() },
            { value: "blackout", label: `blackout ${report.n_fields?.blackout ?? 0}`, disabled: !hasBlackout,
              title: hasBlackout ? undefined : "This report scored no blackout fields" }]} />
        <span className="mdl-faint">bins, halo, holes: VIS squared error ÷ the best member's (lower is better)</span>
      </div>
      <div className="mdl-scroll-x">
        <table className="mdl-table-mini">
          <thead><tr>
            <th>Method</th>{report.bands.map((b) => <th key={b}>{BAND_SHORT[b] ?? b} [dB]</th>)}
            {group === "natural" && report.knee && <th>∫ mean [dB]</th>}
            {report.brightness_names.map((n) => <th key={n}>{n}</th>)}<th>halo</th>{group === "blackout" && <th>holes</th>}
          </tr></thead>
          <tbody>
            {rows.map((r) => {
              const s = block[r];
              const top = s.band_psnr[0] != null && rows.every((x) => vis(x) <= vis(r));
              return (
                <tr key={r} data-best={top}>
                  <td>{r.startsWith("member:") ? `best member #${memberNumber(r.slice(7))}` : variantLabel(r)}</td>
                  {s.band_psnr.map((v, i) => <td key={i}>{db(v, 3)}</td>)}
                  {group === "natural" && report.knee && <td>{db(meanOf(knee[r]?.integrated), 3)}</td>}
                  {s.bin_mse.map((v, i) => <td key={i}>{ratio(v, ref?.bin_mse[i])}</td>)}
                  <td>{ratio(s.halo_mse[0], ref?.halo_mse[0])}</td>
                  {group === "blackout" && <td>{ratio(meanOf(s.hole_mse), meanOf(ref?.hole_mse))}</td>}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {report.created && <Caption>Report {report.id} · {formatDateTime(report.created)} · {report.members.length} cube members</Caption>}
    </div>
  );
}

/* ── gate share per member ──────────────────────────────────────────────── */
function GateShare({ rows, loading, variant, onPick }: {
  rows: ShareRow[] | null; loading: boolean; variant: Variant | null; onPick: (name: string) => void;
}) {
  const navigate = useNavigate();
  const [sel, setSel] = useState<string[]>([]);
  const list = rows ?? [];
  const max = list.reduce((m, r) => Math.max(m, r.share ?? 0), 0);
  const unread = list.filter((r) => r.read === false).map((r) => r.name);
  const columns = useMemo<DataColumn<ShareRow>[]>(() => [
    { id: "num", header: "Member", width: 88, sortFn: (a, b) => Number(a.num) - Number(b.num),
      cell: (r) => <span className="mdl-mono">#{r.num}</span> },
    { id: "share", header: "Gate share", headerText: "gate share (peak, else the mean over bands)", width: 260,
      accessor: (r) => r.share, cell: (r) => <ShareBar value={r.share} max={max} text={r.text} read={r.read} /> },
    { id: "read", header: "Read", headerText: "read by the gate", width: 84, hidden: true, accessor: (r) => (r.read == null ? null : r.read ? "read" : "not read") },
  ], [max]);
  const open = () => {
    useSelection.getState().select("member", sel);
    navigate(tabPath("members"));
  };
  const toolbar = (
    <div className="mdl-row" role="group" aria-label="Gate share selection">
      {unread.length > 0 && <Button size="sm" variant="ghost" onClick={() => setSel(unread)}>Select the {unread.length} not read</Button>}
      <Button size="sm" disabled={!sel.length} onClick={open}
        title={sel.length ? "Select these members on the Members roster (to archive or continue them)" : "Select members in this list first"}>
        Open in Members with this selection{sel.length ? ` (${sel.length})` : ""}
      </Button>
    </div>
  );
  return (
    <section id="mdl-share" className="mdl-stack mdl-stack--tight" aria-labelledby="mdl-share-title">
      <div className="mdl-row">
        <h2 id="mdl-share-title" className="mdl-h2">Gate share per member</h2>
        {variant && <span className="mdl-muted">{variant.production ? "the production gate" : `variant ${variantLabel(variant.name)}`}</span>}
        {variant && !variant.production && (
          <Button size="sm" variant="ghost" onClick={() => onPick("")}>Show production</Button>
        )}
      </div>
      {rows == null
        ? loading
          ? <div aria-busy="true" aria-label="Reading the gate share"><Skeleton lines={4} /></div>
          : <p className="mdl-note">{variant?.production
            ? "The production gate's share per member is not readable (members.json)."
            : "No gate share recorded for this variant: compare it with production to measure one."}</p>
        : (
          <DataTable rows={list} columns={columns} rowKey={(r) => r.name} aria-label="Gate share per member" dense
            selectable selected={sel} onSelectedChange={(keys) => setSel(keys)} toolbar={toolbar} urlKey="gs"
            defaultSort={[{ id: "share", desc: true }]} height={360} exportName="gate-share"
            inspect={(r) => ({ kind: "member", id: r.name })} />
        )}
      <Caption>
        The member's share of the gate's weight: the peak (the largest share in any band and brightness bin, what decides
        whether the gate reads it) when recorded, else the all-pixel mean over VIS, Y, J and H. Faint bars: members the gate does not read.
      </Caption>
    </section>
  );
}

/* ── the tab ────────────────────────────────────────────────────────────── */
export default function Combiner() {
  const res = useCombiners();
  const members = useMembers();
  const exps = useExperiments();
  const [reportId, setReportId] = useUrlState("report", "");
  const [historyMetric, setHistoryMetric] = useUrlState<HeldOutMetric>("hist", "loss");
  const [history, setHistory] = useUrlState("history", false);
  const [benchId, setBenchId] = useUrlState("bench", "");
  // The old Combiners page kept "show backups" in ?backups=1: read it once as History.
  const [legacyBackups, setLegacyBackups] = useUrlState("backups", false);
  const showHistory = history || legacyBackups;
  const toggleHistory = () => { setHistory(!showHistory); setLegacyBackups(false); };
  const [shareOf, setShareOf] = useUrlState("share", "");
  const logToNotebook = useLogToNotebook("Models › Combiner");
  const data = res.data;
  const hasReports = !!data?.compare || !!data?.reports.length;
  const report = useResource<CompareReport>(hasReports ? url.report(reportId || null) : null, [reportId, data?.compare?.id]);
  const fit = useJob(JOB.fit);
  const compare = useJob(JOB.compare);
  const promote = useJob(JOB.promote);
  useOnJobEnd(fit.job);
  useOnJobEnd(compare.job);
  useOnJobEnd(promote.job);
  const [fitOpen, setFitOpen] = useState(false);
  const [cmpOpen, setCmpOpen] = useState(false);
  const lg = useLegend();
  const rep = hasReports ? report.data ?? null : null;

  const benchmark = useMemo(() => benchmarkExperiment(exps.data?.experiments, benchId), [exps.data, benchId]);
  // A run's score counts only for the fit it scored: check each spec's fingerprint
  // against the catalogue's, so an earlier production's holes are never shown as current.
  const benchDetail = useExperiment(benchmark?.expId);
  const catalog = useModelCatalog();
  const freshness = useMemo(() => benchFreshness(benchmark?.expId, benchDetail.data, catalog.data), [benchmark, benchDetail.data, catalog.data]);
  const benchOptions = useMemo(() => benchmarkChoices(exps.data?.experiments).map((c) => ({
    value: c.value, label: c.label, hint: c.production ? "scored production" : "no production run",
  })), [exps.data]);
  const scope = useMemo(() => variantScope(data?.variants ?? [], showHistory), [data, showHistory]);
  const production = data?.variants.find((v) => v.production) ?? null;
  const rows = useMemo<Row[]>(() => scope.shown.map((v) => {
    const nat = rep?.groups?.natural?.[method(v)];
    const blk = rep?.groups?.blackout?.[method(v)];
    const scored = benchmark?.bySpec.get(v.spec) ?? null;
    const fresh = scored ? freshness(v.spec) : null;
    return {
      ...v,
      testVis: nat?.band_psnr?.[0] ?? v.test?.band_psnr?.[0] ?? (v.production ? v.eval?.psnr ?? null : null),
      kneeMean: meanOf(rep?.knee?.methods?.[method(v)]?.integrated ?? v.knee?.integrated),
      blackoutVis: blk?.band_psnr?.[0] ?? v.test?.blackout_band_psnr?.[0] ?? null,
      bench: fresh === "current" ? scored : null,
      staleBench: fresh === "earlier" ? scored : null,
      benchPending: fresh === "loading",
      lossComparable: heldOutComparable(v, production),
    };
  }), [scope, rep, benchmark, freshness, production]);
  const prod = rows.find((r) => r.production) ?? null;
  const prodScores = prod ? { testVis: prod.testVis, kneeMean: prod.kneeMean } : null;
  const noteFor = (v: Row) => variantNote(v, { prod: prodScores, benchmark });
  const tableNote = () => [
    `**Combiner variants** — ${rows.length} variants${rep ? ` · compare report \`${rep.id ?? "latest"}\`` : ""}${benchmark ? ` · real holes on ${benchmark.tileSet} (experiment \`${benchmark.expId}\`)` : ""}`, "",
    "| Variant | Members | Mix | Held-out | ∫PSNR | Real holes VIS · Y · J · H | Fitted |",
    "| --- | --- | --- | ---: | ---: | --- | --- |",
    ...rows.map((v) => `| ${variantLabel(v.name)}${v.production ? " (production)" : ""} | ${readsText(v)} | ${v.mix_space ?? "—"} | ${v.lossComparable && v.selected?.loss != null ? v.selected.loss.toFixed(4) : "—"} | ${db(v.kneeMean)} | ${v.bench ? holesText(v.bench) : "—"} | ${v.fitted_at ? utcText(v.fitted_at) : "—"} |`),
  ].join("\n");
  const fitResult = fit.job?.status === "done" ? (fit.job.result as { variant?: string } | null) : null;
  const fitted = fitResult?.variant ? rows.find((r) => r.name === fitResult.variant || r.name === `spatial_gate_${fitResult.variant}`) ?? null : null;
  const promoted = promote.job?.status === "done" ? (promote.job.result as { promoted?: string; backup?: string; test_rescored?: boolean } | null) : null;

  /* gate share: production from members.json; a variant from the compare report */
  const shareVariant = (shareOf && data?.variants.find((v) => v.name === shareOf)) || production;
  const shareRows = useMemo<ShareRow[] | null>(() => {
    if (!shareVariant) return null;
    if (shareVariant.production) return members.data ? productionShares(members.data.members) : null;
    const usage = rep?.usage?.[method(shareVariant)] ?? rep?.usage?.[shareVariant.name];
    return reportShares(usage);
  }, [shareVariant, members.data, rep]);
  const shareLoading = !!shareVariant && (shareVariant.production ? members.loading && !members.data : report.loading && !report.data);
  const hasShare = (v: Variant) => v.production || !!(rep?.usage?.[method(v)] ?? rep?.usage?.[v.name]);
  const showShare = (v: Variant) => {
    setShareOf(v.production ? "" : v.name);
    requestAnimationFrame(() => document.getElementById("mdl-share")?.scrollIntoView({ block: "start" }));
  };

  async function doPromote(v: Variant) {
    const mismatch = !v.membership.current;
    const blocked = v.promotion && !v.promotion.ok ? ` ${v.promotion.reason ?? ""}` : "";
    const ok = await confirm({
      title: `Promote ${v.name} to production?`,
      message: `The current production gate is first backed up to spatial_gate_backup_<UTC time> (promote that to roll back). Then the gate is re-applied to the cached test cubes and the summary + knee curves rebuilt.${mismatch ? ` WARNING: fitted for ${v.n_members} members; the active ensemble has ${data?.active_members.length}. Production would read unavailable until refit.` : ""}${blocked}`,
      tone: mismatch ? "danger" : "default", confirmLabel: "Promote",
      ...(mismatch ? { requireText: "promote" } : {}),
    });
    if (ok) await promote.run("/ensemble/combiners/promote", { mode: REGIME, variant: v.name, ...(mismatch ? { force: "1" } : {}) });
  }
  const startCompare = (body: Record<string, string>) => void (async () => {
    if (await confirm({ title: "Run the compare?", message: "Scores the picked variants on the test cubes (blackout fields may need member inference the first time).", confirmLabel: "Compare" })) {
      await compare.run("/ensemble/combiners/compare", { mode: REGIME, ...body });
    }
  })();

  usePageActions([
    { id: "comb-fit", label: "Fit a spatial-gate variant…", group: "Combiner", keywords: ["gate", "train"], run: () => setFitOpen(true) },
    { id: "comb-compare", label: "Compare gate variants…", group: "Combiner", run: () => setCmpOpen(true) },
    { id: "comb-history", label: showHistory ? "Combiner: current membership only" : "Combiner: show every variant (history)", group: "Combiner", run: toggleHistory },
  ]);

  const benchHeader = benchmark ? (
    <Tooltip content={`Real-data hole % per band (VIS · Y · J · H) from ONE Sky › Compare run on ${benchmark.tileSet}${benchmark.created ? ` (${formatRelative(benchmark.created)})` : ""}; a variant it did not run is blank, and one whose fit changed since the run shows "earlier fit".`}>
      <Link className="mdl-defhead" to={compareRunPath(benchmark.expId)}>Real holes [%]</Link>
    </Tooltip>
  ) : "Real holes [%]";
  const columns = useMemo<DataColumn<Row>[]>(() => [
    { id: "name", header: "Variant", accessor: (v) => v.name, width: 190,
      cell: (v) => (
        <span className="mdl-variant-name">
          <code>{v.name.replace(/^spatial_gate_/, "")}</code>
          {v.production && <Badge tone="accent">production</Badge>}
          {v.backup && <Badge>backup</Badge>}
        </span>
      ) },
    { id: "members", header: "Members", accessor: (v) => v.n_reads, numeric: true, width: 96,
      cell: (v) => (
        <Tooltip content={`${readsText(v)}${v.pruned ? " (a pruned gate)" : ""}${v.membership.extra.length ? ` · ${v.membership.extra.length} joined after the fit` : ""}${v.membership.missing_reads?.length ? ` · reads members no longer active: ${v.membership.missing_reads.map((l) => memberNumber(l)).join(", ")}` : ""}`}>
          <span tabIndex={0} className={v.membership.current ? undefined : "mdl-warn"}>{v.pruned ? `${v.n_reads} of ${v.n_members}` : v.n_reads}</span>
        </Tooltip>
      ) },
    { id: "mix", header: "Mix", accessor: (v) => v.mix_space ?? null, width: 70, priority: 4 },
    { id: "loss", header: "Held-out", headerText: "held-out loss (production's definition only)", numeric: true, width: 92, priority: 1,
      accessor: (v) => (v.lossComparable ? v.selected?.loss ?? null : null),
      cell: (v) => (v.selected?.loss == null ? "—" : v.lossComparable ? v.selected.loss.toFixed(4) : (
        <Tooltip content={`Not comparable: this fit's loss is “${String(v.fit.loss ?? "not recorded")}”, production's is “${String(prod?.fit.loss ?? "—")}”. Compare the variants by ∫PSNR or the compare report.`}>
          <span tabIndex={0} className="mdl-faint">other scale</span>
        </Tooltip>
      )) },
    { id: "kneeMean", header: "∫PSNR [dB]", headerText: "knee-integrated PSNR", numeric: true, width: 96, accessor: (v) => v.kneeMean,
      cell: (v) => <span><b>{db(v.kneeMean)}</b>{prod && !v.production && v.kneeMean != null && prod.kneeMean != null
        ? <span className={v.kneeMean > prod.kneeMean ? "mdl-good" : "mdl-faint"}> {dbDelta(v.kneeMean - prod.kneeMean)}</span> : null}</span> },
    { id: "holes", header: benchHeader,
      headerText: `real-data hole % VIS/Y/J/H${benchmark ? ` on ${benchmark.tileSet} (experiment ${benchmark.expId})` : ""}`, numeric: true, width: 136,
      accessor: (v) => v.bench?.worst?.pct ?? v.bench?.holeMean ?? null,
      csv: (v) => (v.bench ? holesText(v.bench) : ""),
      cell: (v) => (v.bench ? (
        <Tooltip content={`${holesLine(v.bench, benchmark!).replace(/^- /, "")} · max ${v.bench.holeMax?.toFixed(1) ?? "—"} % · median R ${v.bench.medianR?.toFixed(3) ?? "—"} · R<0.8 ${v.bench.rLt08?.toFixed(1) ?? "—"} %`}>
          <span tabIndex={0} className="mdl-holes">
            {v.bench.bands.length ? v.bench.bands.map((b, i) => (
              <span key={b.band}>{i > 0 && <span className="mdl-faint"> · </span>}
                <span data-worst={b === v.bench!.worst || undefined} title={`${b.short} ${b.pct?.toFixed(1) ?? "—"} %`}>{b.pct == null ? "—" : b.pct.toFixed(0)}</span></span>
            )) : holesText(v.bench)}
          </span>
        </Tooltip>
      ) : v.staleBench ? (
        <Tooltip content={`This run${benchmark?.created ? ` (${formatDateTime(benchmark.created)})` : ""} scored an earlier fit of ${v.production ? "production" : variantLabel(v.name)} (${holesText(v.staleBench)} % VIS · Y · J · H), not this one. Score it again in Sky › Compare.`}>
          <span tabIndex={0} className="mdl-faint">earlier fit</span>
        </Tooltip>
      ) : v.benchPending ? <span className="mdl-faint">checking…</span> : <span className="mdl-faint">—</span>) },
    { id: "fitted", header: "Fitted", width: 96, priority: 3, accessor: (v) => v.fitted_at ?? null, cell: (v) => (v.fitted_at ? formatRelative(v.fitted_at) : "—") },
    { id: "share", header: "Gate share", sortable: false, filterable: false, csv: false, width: 96, priority: 2,
      cell: (v) => (hasShare(v)
        ? <Button size="sm" variant="ghost" onClick={() => showShare(v)} aria-label={`Gate share of ${v.name}`}>Per member</Button>
        : <Tooltip content="Compare it with production to measure its gate share"><span tabIndex={0} className="mdl-faint">not measured</span></Tooltip>) },
    { id: "testVis", header: "Test VIS [dB]", headerText: "test PSNR VIS", numeric: true, hidden: true, accessor: (v) => v.testVis, cell: (v) => db(v.testVis, 3) },
    { id: "blackoutVis", header: "Blackout VIS [dB]", numeric: true, hidden: true, accessor: (v) => v.blackoutVis, cell: (v) => db(v.blackoutVis, 3) },
    { id: "lr", header: "LR input", accessor: (v) => (v.use_lr ? "yes" : "no"), hidden: true },
    { id: "width", header: "Width", numeric: true, accessor: (v) => v.width ?? null, hidden: true },
    { id: "steps", header: "Steps", numeric: true, accessor: (v) => (v.fit.steps_run as number | undefined) ?? (v.fit.steps as number | undefined) ?? null, hidden: true },
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 48,
      cell: (v) => (
        <Menu label={`${v.name} actions`} trigger={<IconButton size="sm" icon="more" label={`${v.name} actions`} />} items={[
          { label: "Inspect", onSelect: () => openInspector({ kind: "combiner", id: v.name }) },
          ...(!v.production ? [{ label: "Promote to production…", onSelect: () => void doPromote(v) }] : []),
          ...(v.applies_to_test_cubes && !v.production ? [{ label: "Compare with production", onSelect: () => startCompare({ gates: [data?.production ?? "spatial_gate_combiner", v.name].filter((x, i, a) => a.indexOf(x) === i).join(","), blackout_fields: "40" }) }] : []),
          { type: "separator" as const },
          { label: "Log to notebook…", onSelect: () => logToNotebook(noteFor(v)) },
        ]} />
      ) },
  ], [prod, data, benchmark, rep]); // eslint-disable-line react-hooks/exhaustive-deps

  const held = useMemo(() => {
    // The categorical colours without the orange (--cat-2), the combiner's own hue in
    // both themes (production's line), so no two curves look alike.
    const palette = [0, 1, 3, 4, 5, 6, 7].map((i) => categorical(i));
    const h = heldOutCurves(rows, historyMetric, palette.length);
    const series: Series[] = h.curves.map((c) => ({
      x: c.x, y: c.y, color: c.slot < 0 ? C.comb : palette[c.slot], dash: c.dash ? [6, 3] : undefined,
      width: c.production ? 2.6 : 1.5, dots: true, name: c.production ? "production gate" : variantLabel(c.name),
    }));
    const ys = series.flatMap((s) => s.y.filter((y): y is number => y != null));
    const lo = ys.length ? Math.min(...ys) : 0, hi = ys.length ? Math.max(...ys) : 1;
    const pad = (hi - lo) * 0.08 || 0.05;
    const maxStep = Math.max(1, ...h.curves.flatMap((c) => c.x));
    return { series, offScale: h.offScale, xDomain: [0, maxStep] as [number, number], yDomain: [lo - pad, hi + pad] as [number, number] };
  }, [rows, historyMetric]);

  const jobs = (fit.job || compare.job || promote.job || fit.error || compare.error || promote.error) && (
    <div className="mdl-jobs">
      <JobProgress job={fit.job} error={fit.error} />
      {fitted && (
        <div className="mdl-row">
          <span className="mdl-muted">Fitted {variantLabel(fitted.name)}.</span>
          <LogToNotebookButton from="Models › Combiner" label="Log this fit to the notebook" note={() => noteFor(fitted)} />
        </div>
      )}
      <JobProgress job={compare.job} error={compare.error} />
      <JobProgress job={promote.job} error={promote.error} />
      {promoted?.promoted && (
        <div className="mdl-row">
          <span className="mdl-muted">Promoted {variantLabel(promoted.promoted)} to production.</span>
          <LogToNotebookButton from="Models › Combiner" label="Log the promotion to the notebook" note={() => promoteNote(promoted, prodScores)} />
        </div>
      )}
    </div>
  );

  return (
    <Page className="mdl-page">
      <Toolbar label="Combiner actions">
        <Button size="sm" variant="primary" icon="plus" loading={fit.busy} disabled={!data} onClick={() => setFitOpen(true)}>Fit variant…</Button>
        <Button size="sm" loading={compare.busy} disabled={!data} onClick={() => setCmpOpen(true)}>Compare…</Button>
        <ToolbarSeparator />
        {hasReports && (
          <ToolbarGroup label="Report">
            <Select size="sm" aria-label="Compare report" value={reportId} onChange={setReportId} placeholder="latest"
              options={[{ value: "", label: "latest" }, ...(data?.reports ?? []).map((r) => ({ value: r.id, label: r.id }))]} />
          </ToolbarGroup>
        )}
        <ToolbarGroup label="Real holes from">
          <Select size="sm" className="mdl-bench-select" aria-label="Real holes from (Sky › Compare run)" value={benchmark?.expId ?? ""}
            disabled={!benchOptions.length} onChange={setBenchId} placeholder={exps.loading ? "loading…" : "no Sky › Compare run yet"} options={benchOptions} />
          {benchmark && (
            <Tooltip content="Open this run in Sky › Compare">
              <Button size="sm" variant="ghost" icon="external" asChild aria-label="Open the run in Sky › Compare">
                <Link to={compareRunPath(benchmark.expId)}>Open</Link>
              </Button>
            </Tooltip>
          )}
        </ToolbarGroup>
        <ToolbarSpacer />
        <Chip on={showHistory} onClick={toggleHistory}
          title="Variants fitted for earlier memberships, and the promotion backups">History{scope.hidden ? ` ${scope.hidden}` : ""}</Chip>
        <LogToNotebookButton from="Models › Combiner" disabled={!rows.length} note={tableNote} title="Open Notebook › Log with the variant table; you edit it there first" />
      </Toolbar>
      <LoadState loading={res.loading} error={res.error} onRetry={res.reload}
        empty={data && !scope.shown.length && !scope.hidden && (
          <EmptyState icon="layers" title="No combiner yet" action={<Button variant="primary" onClick={() => setFitOpen(true)}>Fit a variant</Button>}>
            Fit a spatial gate on the validate member cubes, compare it, then promote it.
          </EmptyState>
        )}>
        {data && (
          <div className="mdl-stack">
            {jobs}
            {prod && (
              <p className="mdl-status">
                {productionRunsText(prod.n_reads, prod.n_members, pruneThreshold(prod))}
                {prod.fitted_at ? `; fitted ${formatRelative(prod.fitted_at)}` : ""}.
              </p>
            )}
            <DataTable rows={rows} columns={columns} rowKey={(v) => v.name} aria-label="Combiner variants"
              inspect={(v) => ({ kind: "combiner", id: v.name })} urlKey="v" height="auto"
              defaultSort={[{ id: "kneeMean", desc: true }]} exportName="combiner-variants"
              empty={showHistory ? "No variants." : "No variant is fitted for the current membership: fit one, or open History."} />
            <Caption>
              ∫PSNR: PSNR averaged over the scoring knees 0.1–10⁴ e⁻ on the test fields, the metric the{" "}
              <Link to={tabPath("leaderboard")}>Leaderboard</Link> ranks by.{" "}
              {benchmark
                ? <>Real holes: the % of bright LR pixels the SR blanks, per band (the worst in colour), from the{" "}
                    <Link to={compareRunPath(benchmark.expId)}>Sky › Compare run</Link> on {benchmark.tileSet}
                    {benchmark.created ? ` (${formatDateTime(benchmark.created)})` : ""}.
                    {prod?.staleBench ? " That run scored an earlier production gate, so production's holes are not shown; score this one again in Sky › Compare." : ""}</>
                : <>Real holes come from a <Link to="/sky/compare">Sky › Compare</Link> run; there is none yet.</>}
              {!showHistory && scope.hidden > 0 ? ` ${scope.hidden} more variant${scope.hidden === 1 ? "" : "s"} (earlier memberships, backups) under History.` : ""}
            </Caption>
            <GateShare rows={shareRows} loading={shareLoading} variant={shareVariant ?? null} onPick={setShareOf} />
            <section className="mdl-stack mdl-stack--tight" aria-labelledby="mdl-held-title">
              <div className="mdl-row">
                <h2 id="mdl-held-title" className="mdl-h2">Held-out curves</h2>
                <span className="mdl-muted">checkpoint selection during each fit</span>
                <span className="mdl-grow" />
                <Segmented<HeldOutMetric> size="sm" aria-label="Held-out metric" value={historyMetric} onChange={setHistoryMetric}
                  options={[{ value: "loss", label: "Loss" }, { value: "vis", label: "VIS PSNR" }, { value: "int", label: "∫PSNR" }]} />
              </div>
              {held.series.length
                ? <Plot {...lg.plotProps} xDomain={held.xDomain} yDomain={held.yDomain} xLabel="fit step"
                    yLabel={historyMetric === "loss" ? "held-out loss (1 = best member)" : "PSNR [dB]"}
                    series={held.series} legend="auto" aspect={0.42} exportName="gate-history" aria-label="Held-out fit curves" />
                : <EmptyState compact icon="activity" title="No fit history" />}
              {held.offScale.length > 0 && (
                <Caption>
                  Not drawn: {held.offScale.map((n) => variantLabel(n)).join(", ")}, whose loss is on another scale than production's.
                  Compare {held.offScale.length === 1 ? "it" : "them"} by PSNR.
                </Caption>
              )}
            </section>
            {rep && (
              <section className="mdl-stack mdl-stack--tight" aria-labelledby="mdl-report-title">
                <div className="mdl-row">
                  <h2 id="mdl-report-title" className="mdl-h2">Compare report</h2>
                  <span className="mdl-grow" />
                  <LogToNotebookButton from="Models › Combiner" note={() => compareNote(rep)} title="Open Notebook › Log with this compare report; you edit it there first" />
                </div>
                <ReportTable report={rep} />
              </section>
            )}
          </div>
        )}
      </LoadState>
      {data && fitOpen && <FitDialog open={fitOpen} onOpenChange={setFitOpen} data={data}
        onStart={(body) => void fit.run("/ensemble/combiners/fit", body)} />}
      {data && cmpOpen && <CompareDialog open={cmpOpen} onOpenChange={setCmpOpen} data={data} onStart={startCompare} />}
    </Page>
  );
}
