/* ensemble/combiners (spec §8.2): the variant registry — every
   spatial_gate_* directory (production, named variants, promotion backups)
   and the RBF — with fit summary, membership, held-out loss curves overlaid,
   test and knee-integrated PSNR per variant (latest compare report), the
   real-data benchmark (Sky › Experiments), the COMPARE job, the FIT job with
   every knob (writes a NAMED variant, never production) and PROMOTE (backs
   the production gate up first; confirm). */
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import Plot, { useLegend, type Series } from "../../../charts/Plot";
import { C, categorical } from "../../../colors";
import { useJob } from "../../../api/jobs";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { formatDateTime, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { useSelected } from "../../../state/selection";
import {
  Badge, Button, Card, CardBody, CardHead, Checkbox, DataTable, Dialog, EmptyState, Field, Input,
  JobProgress, Menu, MultiSelect, NumberField, Page, Segmented, Select, Tooltip, confirm, type DataColumn,
} from "../../../ui";
import {
  BAND_SHORT, url, useCombiners, useMode, type CombinersPayload, type CompareReport, type ExperimentSummary,
  type Mode, type Variant,
} from "../api";
import { BarGroup, EnsBar, LoadState } from "../common";
import { JOB, useOnJobEnd } from "../jobs";
import { db, dbDelta, memberNumber, variantLabel } from "../model";
import "../ensemble.css";

const method = (v: Variant) => (v.kind === "rbf" ? "rbf" : `gate:${v.name}`);
const meanOf = (vs: (number | null | undefined)[] | null | undefined) => {
  const f = (vs ?? []).filter((v): v is number => v != null && Number.isFinite(v));
  return f.length ? f.reduce((a, b) => a + b, 0) / f.length : null;
};

/* ── real-data benchmark (newest experiment that ran each spec) ────────── */
type Bench = { expId: string; created?: string; nTiles?: number; holeMean?: number | null; holeMax?: number | null; medianR?: number | null; rLt08?: number | null };
function benchBySpec(exps: ExperimentSummary[] | undefined): Map<string, Bench> {
  const out = new Map<string, Bench>();
  const sorted = [...(exps ?? [])].sort((a, b) => String(b.created ?? "").localeCompare(String(a.created ?? "")));
  for (const e of sorted) {
    for (const [spec, agg] of Object.entries(e.summary ?? {})) {
      if (out.has(spec)) continue;
      const a = agg as { n_tiles?: number; summary?: Record<string, number | null> };
      out.set(spec, { expId: e.id, created: e.created, nTiles: a.n_tiles, holeMean: a.summary?.hole_pct_mean,
        holeMax: a.summary?.hole_pct_max, medianR: a.summary?.median_R, rLt08: a.summary?.pct_R_lt_0p8 });
    }
  }
  return out;
}

type Row = Variant & { testVis: number | null; testMean: number | null; kneeMean: number | null; blackoutVis: number | null; bench: Bench | null };

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

function FitDialog({ open, onOpenChange, data, mode, onStart }: {
  open: boolean; onOpenChange: (v: boolean) => void; data: CombinersPayload; mode: Mode;
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
    const body: Record<string, string> = { mode };
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
      <div className="ens-form">
        <Field label="Variant name" error={bad} className="ens-form__wide">
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
        <Field label="Members (pruned gate)" className="ens-form__wide"
          hint="Empty: every active member. Pick a subset for a pruned gate that reads only those members.">
          <div className="ens-row">
            <MultiSelect value={k.members} onChange={(v) => set("members", v)} options={options} placeholder="all active members" maxSummary={6} />
            <Button size="sm" variant="ghost" disabled={!tableSel.length}
              onClick={() => set("members", data.active_members.filter((l) => tableSel.includes(`member_${memberNumber(l)}`)))}>
              Use Members selection ({tableSel.length})
            </Button>
          </div>
        </Field>
        <div className="ens-row ens-form__wide">
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
  const [picked, setPicked] = useState<string[]>(() => gates.filter((v) => !v.backup).map((v) => v.name));
  const [blackout, setBlackout] = useState("40");
  const [rbf, setRbf] = useState(true);
  const [knee, setKnee] = useState(true);
  const submit = () => {
    onStart({ gates: picked.join(","), blackout_fields: blackout, include_rbf: rbf ? "1" : "0", knee: knee ? "1" : "0" });
    onOpenChange(false);
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange} size="lg" title="Compare gate variants"
      description="Scores each variant, the plain mean, the RBF and every member on the cached test cubes and blackout copies."
      footer={<>
        <Button variant="ghost" onClick={() => onOpenChange(false)}>Cancel</Button>
        <Button variant="primary" disabled={!picked.length} onClick={submit}>Compare {picked.length}</Button>
      </>}>
      <div className="ens-stack">
        <div className="ens-picker" role="group" aria-label="Variants to compare">
          {gates.map((v) => (
            <label key={v.name} className="ens-pick" data-on={picked.includes(v.name)}>
              <span className="ens-pick__top">
                <span>{variantLabel(v.name)}</span>
                <Checkbox checked={picked.includes(v.name)} aria-label={`Compare ${v.name}`}
                  onChange={(on) => setPicked((p) => (on ? [...p, v.name] : p.filter((x) => x !== v.name)))} />
              </span>
              <span className="ens-pick__meta">{v.n_reads} members · {v.mix_space}{v.production ? " · production" : v.backup ? " · backup" : ""}</span>
            </label>
          ))}
        </div>
        <div className="ens-form">
          <NumberField label="Blackout fields" value={blackout} onChange={setBlackout} min={0} max={400}
            hint="0 = natural fields only (no member inference)" />
          <Checkbox checked={rbf} onChange={setRbf}>include the RBF</Checkbox>
          <Checkbox checked={knee} onChange={setKnee}>PSNR-vs-knee curves</Checkbox>
        </div>
      </div>
    </Dialog>
  );
}

/* ── compare report table ───────────────────────────────────────────────── */
function ReportTable({ report }: { report: CompareReport }) {
  const [group, setGroup] = useUrlState("group", "natural");
  const block = report.groups[group] ?? report.groups.natural;
  if (!block) return null;
  const memberKeys = Object.keys(block).filter((k) => k.startsWith("member:"));
  const best = memberKeys.reduce<string | null>((b, k) => (b == null || block[k].band_psnr[0] > block[b].band_psnr[0] ? k : b), null);
  const rows = ["mean", ...(best ? [best] : []), ...report.methods.filter((m) => m !== "mean")];
  const ref = best ? block[best] : null;
  const rel = (v: number, r: number | undefined) => (r ? v / Math.max(r, 1e-30) : NaN);
  const knee = report.knee?.methods ?? {};
  return (
    <div className="ens-stack">
      <div className="ens-row">
        <Segmented size="sm" aria-label="Field group" value={group} onChange={setGroup}
          options={[{ value: "natural", label: `natural (${report.n_fields?.natural ?? "?"})` }, { value: "blackout", label: `blackout (${report.n_fields?.blackout ?? 0})` }]} />
        <span className="ens-faint">bins, halo, holes: VIS squared error ÷ the best member's (lower is better)</span>
      </div>
      <div className="ens-scroll-x">
        <table className="ens-table-mini">
          <thead><tr>
            <th>method</th>{report.bands.map((b) => <th key={b}>{BAND_SHORT[b] ?? b}</th>)}
            {group === "natural" && report.knee && <th>∫ mean</th>}
            {report.brightness_names.map((n) => <th key={n}>{n}</th>)}<th>halo</th>{group === "blackout" && <th>holes</th>}
          </tr></thead>
          <tbody>
            {rows.filter((r) => block[r]).map((r) => {
              const s = block[r];
              const vs = s.band_psnr;
              const top = rows.filter((x) => block[x]).every((x) => block[x].band_psnr[0] <= vs[0]);
              return (
                <tr key={r} data-best={top}>
                  <td>{r.startsWith("member:") ? `best member #${memberNumber(r.slice(7))}` : variantLabel(r)}</td>
                  {vs.map((v, i) => <td key={i}>{db(v, 3)}</td>)}
                  {group === "natural" && report.knee && <td>{db(meanOf(knee[r]?.integrated), 3)}</td>}
                  {s.bin_mse.map((v, i) => <td key={i}>{rel(v, ref?.bin_mse[i]).toFixed(3)}</td>)}
                  <td>{rel(s.halo_mse[0], ref?.halo_mse[0]).toFixed(3)}</td>
                  {group === "blackout" && <td>{rel(meanOf(s.hole_mse) ?? NaN, meanOf(ref?.hole_mse) ?? undefined).toFixed(3)}</td>}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {report.created && <span className="ens-faint">Report {report.id} · {formatDateTime(report.created)} · {report.members.length} cube members</span>}
    </div>
  );
}

/* ── the tab ────────────────────────────────────────────────────────────── */
export default function Combiners() {
  const mode = useMode();
  const res = useCombiners(mode);
  const [reportId, setReportId] = useUrlState("report", "");
  const [historyMetric, setHistoryMetric] = useUrlState("hist", "loss");
  const [showBackups, setShowBackups] = useUrlState("backups", false);
  const report = useResource<CompareReport>(res.data?.compare || reportId ? url.report(mode, reportId || null) : null,
    [mode, reportId, res.data?.compare?.id]);
  const exps = useResource<{ experiments: ExperimentSummary[] }>(url.experiments(), [], { ttl: 60_000 });
  const fit = useJob(JOB.fit);
  const compare = useJob(JOB.compare);
  const promote = useJob(JOB.promote);
  useOnJobEnd(fit.job);
  useOnJobEnd(compare.job);
  useOnJobEnd(promote.job);
  const [fitOpen, setFitOpen] = useState(false);
  const [cmpOpen, setCmpOpen] = useState(false);
  const lg = useLegend();
  const data = res.data;
  const rep = report.data;

  const rows = useMemo<Row[]>(() => {
    const bench = benchBySpec(exps.data?.experiments);
    return (data?.variants ?? []).filter((v) => showBackups || !v.backup).map((v) => {
      const nat = rep?.groups?.natural?.[method(v)];
      const blk = rep?.groups?.blackout?.[method(v)];
      return {
        ...v,
        testVis: nat?.band_psnr?.[0] ?? v.test?.band_psnr?.[0] ?? (v.production ? v.eval?.psnr ?? null : null),
        testMean: meanOf(nat?.band_psnr ?? v.test?.band_psnr),
        kneeMean: meanOf(rep?.knee?.methods?.[method(v)]?.integrated ?? v.knee?.integrated),
        blackoutVis: blk?.band_psnr?.[0] ?? v.test?.blackout_band_psnr?.[0] ?? null,
        bench: bench.get(v.spec) ?? null,
      };
    });
  }, [data, rep, exps.data, showBackups]);
  const prod = rows.find((r) => r.production);

  async function doPromote(v: Variant) {
    const mismatch = !v.membership.current;
    const ok = await confirm({
      title: `Promote ${v.name} to production?`,
      message: `The current production gate is first backed up to spatial_gate_backup_<UTC time> (promote that to roll back). Then the gate is re-applied to the cached test cubes and the summary + knee curves rebuilt.${mismatch ? ` WARNING: fitted for ${v.n_members} members; the active ensemble has ${data?.active_members.length}. Production would read unavailable until refit.` : ""}`,
      tone: mismatch ? "danger" : "default", confirmLabel: "Promote",
      ...(mismatch ? { requireText: "promote" } : {}),
    });
    if (ok) await promote.run("/ensemble/combiners/promote", { mode, variant: v.name, ...(mismatch ? { force: "1" } : {}) });
  }
  const startCompare = (body: Record<string, string>) => void (async () => {
    if (await confirm({ title: "Run the compare?", message: "Scores the picked variants on the test cubes (blackout fields may need member inference the first time).", confirmLabel: "Compare" })) {
      await compare.run("/ensemble/combiners/compare", { mode, ...body });
    }
  })();

  usePageActions([
    { id: "comb-fit", label: "Fit a spatial-gate variant…", group: "Combiners", keywords: ["gate", "train"], run: () => setFitOpen(true) },
    { id: "comb-compare", label: "Compare gate variants…", group: "Combiners", run: () => setCmpOpen(true) },
    { id: "comb-backups", label: showBackups ? "Hide promotion backups" : "Show promotion backups", group: "Combiners", run: () => setShowBackups(!showBackups) },
  ]);

  const columns = useMemo<DataColumn<Row>[]>(() => [
    { id: "name", header: "Variant", accessor: (v) => v.name, width: 200,
      cell: (v) => (
        <span className="ens-variant-name">
          <code>{v.kind === "rbf" ? "RBF" : v.name.replace(/^spatial_gate_/, "")}</code>
          {v.production && <Badge tone="accent">production</Badge>}
          {v.backup && <Badge>backup</Badge>}
          {v.kind === "rbf" && <Badge tone="warn">stale kind</Badge>}
        </span>
      ) },
    { id: "members", header: "Members", accessor: (v) => v.n_reads, width: 92,
      cell: (v) => (
        <Tooltip content={v.membership.current ? "Fitted for the active members" : `Missing now: ${v.membership.missing.map((l) => memberNumber(l)).join(", ") || "none"} · not read: ${v.membership.extra.length}`}>
          <span tabIndex={0} className={v.membership.current ? "ens-good" : "ens-warn"}>
            {v.n_reads}{v.pruned ? ` of ${v.n_members}` : ""}{v.membership.current ? " ✓" : ""}
          </span>
        </Tooltip>
      ) },
    { id: "mix", header: "Mix", accessor: (v) => v.mix_space ?? null, width: 64 },
    { id: "lr", header: "LR", accessor: (v) => (v.use_lr ? "yes" : "no"), width: 48, hidden: true },
    { id: "width", header: "Width", numeric: true, accessor: (v) => v.width ?? null, hidden: true },
    { id: "steps", header: "Steps", numeric: true, accessor: (v) => (v.fit.steps_run as number | undefined) ?? (v.fit.steps as number | undefined) ?? null, hidden: true },
    { id: "loss", header: "Held-out", headerText: "held-out loss", numeric: true, width: 84, accessor: (v) => v.selected?.loss ?? null, cell: (v) => (v.selected?.loss != null ? v.selected.loss.toFixed(4) : "—") },
    { id: "testVis", header: "Test VIS", headerText: "test PSNR VIS", numeric: true, width: 112, accessor: (v) => v.testVis,
      cell: (v) => <span>{db(v.testVis, 3)}{prod && !v.production && v.testVis != null && prod.testVis != null
        ? <span className={v.testVis > prod.testVis ? "ens-good" : "ens-faint"}> {dbDelta(v.testVis - prod.testVis)}</span> : null}</span> },
    { id: "kneeMean", header: "∫PSNR", headerText: "knee integrated mean", numeric: true, width: 76, accessor: (v) => v.kneeMean,
      cell: (v) => <b>{db(v.kneeMean, 3)}</b> },
    { id: "blackoutVis", header: "Blackout VIS", numeric: true, accessor: (v) => v.blackoutVis, cell: (v) => db(v.blackoutVis, 3), hidden: true },
    { id: "holes", header: "Real holes", headerText: "real-data hole % (mean over bands)", numeric: true, width: 88, accessor: (v) => v.bench?.holeMean ?? null,
      cell: (v) => (v.bench ? <Tooltip content={`Experiment ${v.bench.expId}: ${v.bench.nTiles} real tiles · max ${v.bench.holeMax?.toFixed(1)}% · median R ${v.bench.medianR?.toFixed(3)} · R<0.8 ${v.bench.rLt08?.toFixed(1)}%`}>
        <span tabIndex={0}>{v.bench.holeMean != null ? `${v.bench.holeMean.toFixed(1)}%` : "—"}</span></Tooltip> : <span className="ens-faint">—</span>) },
    { id: "fitted", header: "Fitted", width: 84, accessor: (v) => v.fitted_at ?? null, cell: (v) => (v.fitted_at ? formatRelative(v.fitted_at) : "—") },
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 48,
      cell: (v) => (
        <Menu label={`${v.name} actions`} trigger={<Button size="sm" variant="ghost" icon="more" aria-label={`${v.name} actions`} />} items={[
          { label: "Inspect", onSelect: () => openCombiner(mode, v.name) },
          ...(v.kind === "gate" && !v.production ? [{ label: "Promote to production…", onSelect: () => void doPromote(v) }] : []),
          ...(v.kind === "gate" && v.applies_to_test_cubes ? [{ label: "Compare with production", onSelect: () => startCompare({ gates: [data?.production ?? "spatial_gate_combiner", v.name].filter((x, i, a) => a.indexOf(x) === i).join(","), blackout_fields: "40" }) }] : []),
        ]} />
      ) },
  ], [prod, mode, data]); // eslint-disable-line react-hooks/exhaustive-deps

  const history = useMemo(() => {
    const series: Series[] = [];
    let maxStep = 1;
    rows.filter((v) => v.kind === "gate" && v.history.length).forEach((v, i) => {
      const pts = v.history.map((h) => [h.step, historyMetric === "loss" ? h.loss : historyMetric === "vis" ? h.vis_psnr : meanOf(h.integrated_psnr)] as const)
        .filter(([, y]) => y != null && Number.isFinite(y));
      if (!pts.length) return;
      maxStep = Math.max(maxStep, ...pts.map(([s]) => s));
      series.push({ x: pts.map(([s]) => s), y: pts.map(([, y]) => y as number), color: v.production ? C.comb : categorical(i),
        width: v.production ? 2.6 : 1.4, dots: true, name: variantLabel(v.name) });
    });
    const ys = series.flatMap((s) => s.y.filter((y): y is number => y != null));
    const lo = Math.min(...ys), hi = Math.max(...ys);
    const pad = (hi - lo) * 0.08 || 0.05;
    return { series, xDomain: [0, maxStep] as [number, number], yDomain: [lo - pad, hi + pad] as [number, number] };
  }, [rows, historyMetric]);

  return (
    <Page>
      <EnsBar label="Combiner actions">
        <Button size="sm" variant="primary" icon="plus" loading={fit.busy} onClick={() => setFitOpen(true)}>Fit variant…</Button>
        <Button size="sm" loading={compare.busy} onClick={() => setCmpOpen(true)}>Compare…</Button>
        <span className="ens-bar__sep" aria-hidden />
        <BarGroup label="Report">
          <Select size="sm" aria-label="Compare report" value={reportId}
            onChange={setReportId} placeholder="latest"
            options={[{ value: "", label: "latest" }, ...(data?.reports ?? []).map((r) => ({ value: r.id, label: r.id }))]} />
        </BarGroup>
        <span className="ens-bar__spacer" />
        <Checkbox checked={showBackups} onChange={setShowBackups}>backups</Checkbox>
      </EnsBar>
      <LoadState loading={res.loading} error={res.error} onRetry={res.reload}
        empty={data && !data.variants.length && (
          <EmptyState icon="layers" title={`No ${mode} combiner yet`} action={<Button variant="primary" onClick={() => setFitOpen(true)}>Fit a variant</Button>}>
            Fit a spatial gate on the validate member cubes, compare it, then promote it.
          </EmptyState>
        )}>
        {data && (
          <div className="ens-stack">
            <JobProgress job={fit.job} error={fit.error} />
            <JobProgress job={compare.job} error={compare.error} />
            <JobProgress job={promote.job} error={promote.error} />
            <DataTable rows={rows} columns={columns} rowKey={(v) => v.name} aria-label="Combiner variants"
              inspect={(v) => ({ kind: "combiner", id: `${mode}/${v.name}` })} urlKey="v" height="auto"
              defaultSort={[{ id: "kneeMean", desc: true }]} exportName={`combiner-variants-${mode}`} />
            <div className="ens-grid">
              <Card>
                <CardHead title="Held-out curves" sub="checkpoint selection during each fit"
                  right={<Segmented size="sm" aria-label="History metric" value={historyMetric} onChange={setHistoryMetric}
                    options={[{ value: "loss", label: "loss" }, { value: "vis", label: "VIS PSNR" }, { value: "int", label: "∫PSNR" }]} />} />
                <CardBody>
                  {history.series.length
                    ? <Plot {...lg.plotProps} xDomain={history.xDomain} yDomain={history.yDomain} xLabel="fit step"
                        yLabel={historyMetric === "loss" ? "held-out loss (1 = best member)" : "PSNR [dB]"}
                        series={history.series} legend="auto" aspect={0.55} exportName={`gate-history-${mode}`} aria-label="Held-out fit curves" />
                    : <EmptyState compact icon="activity" title="No fit history" />}
                </CardBody>
              </Card>
              <Card>
                <CardHead title="Compare report" sub={rep ? `${rep.methods.length} methods` : undefined} />
                <CardBody>
                  {report.loading ? null : rep ? <ReportTable report={rep} />
                    : <EmptyState compact icon="table" title="No compare report yet"
                        action={<Button size="sm" onClick={() => setCmpOpen(true)}>Compare…</Button>}>Score the variants on the test cubes.</EmptyState>}
                </CardBody>
              </Card>
            </div>
            <span className="ens-faint">
              Real-data hole % comes from <Link to="/sky/experiments">Sky › Experiments</Link> (newest experiment per model).
            </span>
          </div>
        )}
      </LoadState>
      {data && fitOpen && <FitDialog open={fitOpen} onOpenChange={setFitOpen} data={data} mode={mode}
        onStart={(body) => void fit.run("/ensemble/combiners/fit", body)} />}
      {data && cmpOpen && <CompareDialog open={cmpOpen} onOpenChange={setCmpOpen} data={data} onStart={startCompare} />}
    </Page>
  );
}

function openCombiner(mode: Mode, name: string) {
  openInspector({ kind: "combiner", id: `${mode}/${name}` });
}
