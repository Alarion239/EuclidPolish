/* Data › Records (spec §8.4): the synthetic training TFRecords, image first
 * (docs/superpowers/specs/2026-09-27-image-first-viewer-design.md).
 *
 * One toolbar row: the split (test / validate / train, with record counts),
 * the split's state as three badges — its local files (one badge, the files
 * in its tip), the SR tier against the production model, the records'
 * noise-model check — and the actions: sync from FASRC and generate the
 * production SR (background jobs), more in the ⋯ menu (generate new pairs,
 * regenerate this split's SR, refresh). Then one caption row for the image:
 * the record and its truth sources (sources_<split>.csv) — drawn on the HR
 * image, on every tier or not at all, filtered by type. Then the viewer
 * (LR · HR · BHR · Clean (starless) · SR), sized so its first frame row is in
 * view (ViewerStage). Below it: a running job, the record's sources as a table
 * (marker / row → the `truth` inspector), the split's per-record census
 * (row → that record) and the synthetic_generate step (collapsed). The split,
 * viewer object/tiers/view, overlay mode, hidden source types and open sections
 * live in the URL. */
import { useMemo, useRef, useState, type ComponentProps, type ReactNode } from "react";
import { useJob } from "../../../api/jobs";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { useSystemAlerts } from "../../../app/status";
import { StepById, StepCard, useStepsStatus } from "../../../fasrc";
import { formatCount, formatNumber } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { useInspector } from "../../../state/inspector";
import {
  Badge, Button, Callout, Checkbox, Chip, DataTable, EmptyState, IconButton, Menu, Page,
  Popover, Section, Segmented, Switch, Tooltip, type DataColumn,
} from "../../../ui";
import { ImageViewer, type ViewerApi, type ViewerMarker, type ViewerMarkers, type ViewerState } from "../../../viewer";
import type { LayoutMode } from "../../../viewer/fit";
import {
  SPLITS, SYNC_KINDS, URLS, useSrStatus, type FieldCensus, type RecordSources, type SourcesCensus, type Split,
  type SrStatus, type TruthSource,
} from "../api";
import { BarActions, DataBar, JobStrip, OFFLINE_HINT, Spacer, startDataJob, useFasrcOnline } from "../common";
import {
  SR_STATE_LABEL, SR_STATE_TONE, formatCompact, noiseBadge, recordFiles, recordObjectId, resumeSafeStep, shownTypes,
  sourceMarker, sourceTypeChips, toggleHiddenType, truthId, type SourceType,
} from "../model";
import { ViewerStage } from "../ViewerStage";
import "../register";
import "../data.css";

const isSplit = (v: string): v is Split => (SPLITS as readonly string[]).includes(v);

/** Where the truth sources are drawn: nowhere, on the HR image, or on every tier. */
type Overlay = "off" | "hr" | "all";
const OVERLAYS: readonly Overlay[] = ["off", "hr", "all"];
const isOverlay = (v: string): v is Overlay => (OVERLAYS as readonly string[]).includes(v);

/** The tiers the viewer opens with (its URL `v.rec.t` may name others). */
const DEFAULT_TIERS = ["dirty", "hr"];

/* ── toolbar pieces ────────────────────────────────────────────────────── */

/** A status badge with its detail in a tooltip (focusable, so the tip is reachable by keyboard). */
function TipBadge({ tip, tone, children }: { tip: ReactNode; tone: ComponentProps<typeof Badge>["tone"]; children: ReactNode }) {
  return (
    <Tooltip content={tip}>
      <span tabIndex={0} className="dt-tipbadge"><Badge size="sm" tone={tone} dot>{children}</Badge></span>
    </Tooltip>
  );
}

function FilesBadge({ status, split }: { status: SrStatus; split: Split }) {
  const files = status.splits[split]?.files;
  if (!files) return null;
  const { label, tone, rows } = recordFiles(files, split);
  return (
    <TipBadge tone={tone} tip={(
      <dl className="dt-tipdl">
        {rows.map((r) => (
          <div key={r.key} data-state={r.state}><dt>{r.label}</dt><dd>{r.detail}</dd></div>
        ))}
      </dl>
    )}>{label}</TipBadge>
  );
}

function SrBadge({ status, split }: { status: SrStatus; split: Split }) {
  const sr = status.splits[split]?.sr;
  if (!sr) return null;
  const who = sr.manifest?.model_label ? ` by ${sr.manifest.model_label}` : "";
  const tip = [
    `${sr.count}${sr.records_count != null ? ` of ${sr.records_count}` : ""} SR cubes${who}`,
    ...sr.reasons,
    sr.manifest?.generated_at ? `generated ${sr.manifest.generated_at}` : "",
  ].filter(Boolean).join("\n");
  return (
    <TipBadge tone={SR_STATE_TONE[sr.state]} tip={<span className="dt-pre">{tip}</span>}>
      {SR_STATE_LABEL[sr.state]}{sr.state === "partial" && sr.records_count ? ` ${sr.count}/${sr.records_count}` : ""}
    </TipBadge>
  );
}

function NoiseBadge() {
  const check = useSystemAlerts().data?.checks.find((c) => c.id === "records-noise");
  if (!check) return null;
  const { label, tone } = noiseBadge(check.state);
  return (
    <TipBadge tone={tone} tip={<span className="dt-pre">{[check.title, check.detail].filter(Boolean).join("\n")}</span>}>
      {label}
    </TipBadge>
  );
}

function SyncPopover({ open, onOpenChange, onStart, busy, online }: {
  open: boolean; onOpenChange: (v: boolean) => void; busy: boolean; online: boolean;
  onStart: (subsets: Split[], kinds: string[]) => void;
}) {
  const [subsets, setSubsets] = useState<Split[]>(["test", "validate"]);
  const [kinds, setKinds] = useState<string[]>([...SYNC_KINDS]);
  const toggle = <T extends string>(list: T[], v: T) => (list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);
  if (!online) {
    return (
      <Tooltip content={OFFLINE_HINT}>
        <span tabIndex={0}><Button size="sm" icon="download" loading={busy} disabled aria-label="Sync">Sync</Button></span>
      </Tooltip>
    );
  }
  return (
    <Popover open={open} onOpenChange={onOpenChange} label="Sync records from FASRC" width={300}
      trigger={<Button size="sm" icon="download" loading={busy} aria-label="Sync" title="Sync records from FASRC">Sync</Button>}>
      <div className="dt-pop">
        <div className="dt-pop__title">Sync from FASRC</div>
        <fieldset className="dt-pop__set">
          <legend>Splits</legend>
          {SPLITS.map((s) => (
            <Checkbox key={s} checked={subsets.includes(s)} onChange={() => setSubsets((c) => toggle(c, s))}>
              {s}{s === "train" ? " (large)" : ""}
            </Checkbox>
          ))}
        </fieldset>
        <fieldset className="dt-pop__set">
          <legend>Files</legend>
          {SYNC_KINDS.map((k) => (
            <Checkbox key={k} checked={kinds.includes(k)} onChange={() => setKinds((c) => toggle(c, k))}>{k}</Checkbox>
          ))}
        </fieldset>
        <Button variant="primary" size="sm" disabled={!subsets.length || !kinds.length}
          onClick={() => { onOpenChange(false); onStart(subsets, kinds); }}>
          Start sync
        </Button>
      </div>
    </Popover>
  );
}

function GeneratePopover({ status, open, onOpenChange, onStart, busy }: {
  status: SrStatus; open: boolean; onOpenChange: (v: boolean) => void; busy: boolean;
  onStart: (subsets: Split[], overwrite: boolean) => void;
}) {
  const present = status.subsets;
  const [subsets, setSubsets] = useState<Split[]>(() => present.filter((s) => s !== "train"));
  const anyExisting = subsets.some((s) => (status.sr[s] ?? 0) > 0);
  const anyStale = subsets.some((s) => ["stale", "unknown", "partial"].includes(status.splits[s]?.sr.state));
  const [overwrite, setOverwrite] = useState(anyStale);
  const reason = !status.records ? "Sync the records first." : !status.checkpoint ? "No active STARFULL members." : null;
  if (reason || !status.can_generate) {
    return (
      <Tooltip content={reason ?? "Nothing to generate"}>
        <span tabIndex={0}><Button size="sm" icon="wave" loading={busy} disabled aria-label="Generate SR">Generate SR</Button></span>
      </Tooltip>
    );
  }
  return (
    <Popover open={open} onOpenChange={onOpenChange} label="Generate SR" width={300}
      trigger={<Button size="sm" icon="wave" loading={busy} aria-label="Generate SR" title="Generate the production SR">Generate SR</Button>}>
      <div className="dt-pop">
        <div className="dt-pop__title">Production SR over the local records</div>
        <fieldset className="dt-pop__set">
          <legend>Splits</legend>
          {present.map((s) => (
            <Checkbox key={s} checked={subsets.includes(s)}
              onChange={(on) => setSubsets((c) => (on ? [...c, s] : c.filter((x) => x !== s)))}>
              {s} <span className="muted">({status.splits[s]?.count ?? 0} records, {status.sr[s] ?? 0} SR)</span>
            </Checkbox>
          ))}
        </fieldset>
        <Switch checked={overwrite} onChange={setOverwrite}>Overwrite existing SR</Switch>
        {anyExisting && !overwrite && <p className="dt-note">Splits that already have SR are skipped.</p>}
        <Button variant="primary" size="sm" disabled={!subsets.length}
          onClick={() => { onOpenChange(false); onStart(subsets, overwrite); }}>
          Generate
        </Button>
      </div>
    </Popover>
  );
}

/* ── truth sources ─────────────────────────────────────────────────────── */

const SOURCE_COLUMNS: DataColumn<TruthSource>[] = [
  { id: "row", header: "#", width: 44, numeric: true },
  { id: "type", header: "Type", width: 80,
    cell: (s) => <Badge size="sm" tone={s.type === "star" ? "warn" : s.type === "lens" ? "accent" : "info"}>{s.type}</Badge> },
  { id: "x_pix", header: "x", width: 64, numeric: true, cell: (s) => formatNumber(s.x_pix, { digits: 1 }) },
  { id: "y_pix", header: "y", width: 64, numeric: true, cell: (s) => formatNumber(s.y_pix, { digits: 1 }) },
  { id: "mag_vis", header: "mag", width: 64, headerText: "VIS mag", numeric: true, cell: (s) => formatNumber(s.mag_vis, { digits: 2 }) },
  { id: "flux_vis_e", header: "e⁻", width: 64, headerText: "VIS e-", numeric: true, cell: (s) => (s.flux_vis_e == null ? "—" : formatCompact(s.flux_vis_e)) },
  { id: "size", header: "size ″", width: 76, headerText: "Re or thetaE arcsec", numeric: true,
    accessor: (s) => (s.type === "lens" ? s.theta_E_arcsec : s.re_arcsec),
    cell: (s) => formatNumber(s.type === "lens" ? s.theta_E_arcsec : s.re_arcsec, { digits: 3 }) },
  { id: "z", header: "z", width: 56, numeric: true, hidden: true, cell: (s) => formatNumber(s.z, { digits: 2 }) },
  { id: "subhalo_id", header: "TNG", width: 80, headerText: "TNG subhalo", accessor: (s) => s.subhalo_id ?? "" },
  { id: "sfr_class", header: "SFR class", width: 96, hidden: true, accessor: (s) => s.sfr_class ?? "" },
  { id: "off_field", header: "off", width: 60, headerText: "Off-field", accessor: (s) => (s.off_field ? "yes" : ""),
    cell: (s) => (s.off_field ? <Badge size="sm">off</Badge> : "") },
];

/** The current record's truth sources, the type filter (the URL keeps the
 *  HIDDEN types: every type starts shown, a chip toggles its type) and the
 *  hovered / inspected source, shared by the viewer overlay and the table. */
function useTruthSources(split: Split, index: number | null) {
  const [hidden, setHidden] = useUrlState<string[]>("hide", []);
  const [hover, setHover] = useState<number | null>(null);
  const inspected = useInspector((s) => (s.current?.kind === "truth" ? s.current.id : null));
  const res = useResource<RecordSources>(index != null ? URLS.sources(split, index) : null, [split, index], { ttl: 5 * 60_000 });
  const data = res.data;
  const rows = useMemo(() => {
    const show = shownTypes(hidden);
    return (data?.sources ?? []).filter((s) => show(s.type));
  }, [data, hidden]);
  const activeRow = hover ?? (inspected && index != null && inspected.startsWith(`${split}/${index}/`)
    ? Number(inspected.split("/")[2]) : null);
  const toggle = (t: SourceType) => setHidden(toggleHiddenType(hidden, t));
  const pick = (row: number) => { if (index != null) openInspector({ kind: "truth", id: truthId(split, index, row) }); };
  return { res, data, rows, hidden, toggle, activeRow, setHover, pick };
}
type Truth = ReturnType<typeof useTruthSources>;

/** Where and which of the record's truth sources are drawn (the keys match
 *  the markers' colours) — a group in the tab's toolbar row, so the viewer
 *  starts one row higher (the record itself is the viewer's position and its
 *  readout label). */
function TruthControls({ truth, overlay, onOverlay, index, split }: {
  truth: Truth; overlay: Overlay; onOverlay: (v: Overlay) => void; index: number | null; split: Split;
}) {
  const { data, res, hidden, toggle } = truth;
  const counts = data?.present ? data.counts : null;
  return (
    <div className="dt-caption" role="group" aria-label={index != null ? `Truth sources of record ${index}` : "Truth sources overlay"}>
      <span className="dt-caption__label" aria-hidden="true">Sources on</span>
      <Segmented size="sm" className="dt-seg-text" value={overlay} onChange={onOverlay} aria-label="Draw the truth sources on"
        options={[{ value: "off", label: "Off" }, { value: "hr", label: "HR" }, { value: "all", label: "All tiers" }]} />
      {/* one toggle per type the record has: on = drawn and listed in the table */}
      {counts && sourceTypeChips(counts, hidden).map((c) => (
        <Chip key={c.type} on={c.shown} onClick={() => toggle(c.type)}
          title={`${c.shown ? "Hide" : "Show"} the ${c.type} sources (markers and table rows)`}>
          <span className="dt-key" data-kind={c.type} aria-hidden="true" />{c.type} <span className="muted">{c.count}</span>
        </Chip>
      ))}
      {counts && counts.off_field > 0 && (
        <Tooltip content="Centred outside the frame, their light spills in (dashed)">
          <span tabIndex={0} className="dt-caption__note">{counts.off_field} off-field</span>
        </Tooltip>
      )}
      {index != null && res.loading && !data && <span className="dt-caption__note">Loading sources…</span>}
      {index != null && data && !data.present && (
        <Tooltip content={`Sync the ${split} split with its sources file to see the truth catalogue`}>
          <span tabIndex={0}><Badge size="sm">No sources_{split}.csv</Badge></span>
        </Tooltip>
      )}
      {res.error && <Badge size="sm" tone="bad">Sources: {res.error.message}</Badge>}
    </div>
  );
}

/** The same sources as a table, full width under the viewer. */
function SourcesTable({ truth, split, index }: { truth: Truth; split: Split; index: number | null }) {
  const [open, setOpen] = useUrlState("srct", true);
  const { data, rows, activeRow } = truth;
  if (index == null || !data?.present) return null;
  return (
    <Section title={`Truth sources of record ${index}`} collapsible open={open} onOpenChange={setOpen}
      sub={data.geometry.hr ? `${rows.length} shown on the ${data.geometry.hr.width} px HR grid (${data.geometry.hr.pixscale}″ per px)` : undefined}>
      <DataTable rows={rows} columns={SOURCE_COLUMNS} rowKey={(s) => String(s.row)} dense height={300}
        aria-label={`Sources of record ${index}`} exportName={`sources_${split}_${index}`}
        activeKey={activeRow != null ? String(activeRow) : null}
        inspect={(s) => ({ kind: "truth", id: truthId(split, index, s.row) })}
        empty={data.sources.length ? "No source of the selected types" : "This record has no sources"} />
    </Section>
  );
}

const CENSUS_COLUMNS: DataColumn<FieldCensus>[] = [
  // short headers so the table fits a ~720 px pane; headerText is the full name (sort tooltip, CSV, column menu)
  { id: "field_index", header: "Rec", headerText: "Record", width: 64, numeric: true },
  { id: "n", header: "Src", headerText: "Sources", width: 64, numeric: true },
  { id: "galaxy", header: "Gal", headerText: "Galaxies", width: 64, numeric: true },
  { id: "star", header: "★", headerText: "Stars", width: 56, numeric: true },
  { id: "lens", header: "Lens", headerText: "Lenses", width: 72, numeric: true },
  { id: "off_field", header: "Off", headerText: "Off-field", width: 64, numeric: true, hidden: true },
  { id: "brightest_star_mag", header: "★ mag", width: 92, headerText: "Brightest star mag", numeric: true,
    cell: (f) => formatNumber(f.brightest_star_mag, { digits: 2 }) },
  { id: "brightest_galaxy_mag", header: "Gal mag", width: 104, headerText: "Brightest galaxy mag", numeric: true,
    cell: (f) => formatNumber(f.brightest_galaxy_mag, { digits: 2 }) },
  { id: "total_vis_e", header: "Σ VIS", headerText: "Σ VIS e⁻", width: 92, numeric: true, cell: (f) => formatCompact(f.total_vis_e) },
];

function CensusSection({ split, index, onGo }: { split: Split; index: number | null; onGo: (i: number) => void }) {
  const [open, setOpen] = useUrlState("census", true);
  const res = useResource<SourcesCensus>(open ? URLS.sources(split) : null, [split], { ttl: 5 * 60_000 });
  return (
    <Section title={`Records in ${split}`} collapsible open={open} onOpenChange={setOpen}
      sub={res.data?.present ? `${res.data.fields.length} records; a row opens it in the viewer` : undefined}>
      {res.error ? <Callout tone="bad" title="Census did not load"><span className="dt-pre">{res.error.message}</span></Callout>
        : res.data && !res.data.present ? <EmptyState compact icon="table" title={`No sources_${split}.csv`} />
          : (
            <DataTable rows={res.data?.fields ?? []} columns={CENSUS_COLUMNS} rowKey={(f) => String(f.field_index)}
              loading={res.loading && !res.data} dense height={320} urlKey="cn" aria-label={`Records in ${split}`}
              exportName={`records_${split}`} activeKey={index != null ? String(index) : null}
              onRowClick={(f) => onGo(f.field_index)} />
          )}
    </Section>
  );
}

function GenerationSection() {
  const [open, setOpen] = useUrlState("gen", false);
  const [splits, setSplits] = useState<Split[]>([]);
  const steps = useStepsStatus();
  const toggle = (s: Split) => setSplits((c) => (c.includes(s) ? c.filter((x) => x !== s) : [...c, s]));
  const found = steps.data?.steps?.find((st) => st.step_id === "synthetic_generate");
  // The card prefills from the last run: a --regenerate-splits/--force left in its
  // extra flags would silently turn this "resume" into a rebuild, so drop it.
  const safe = useMemo(() => (found ? resumeSafeStep(found) : null), [found]);
  const extraParams = splits.length ? { regenerate_splits: splits.join(",") } : undefined;
  const hideParams = splits.length ? ["force", "regenerate_splits"] : ["regenerate_splits"];
  return (
    <Section id="dt-generate" title="Generate synthetic training pairs" collapsible open={open} onOpenChange={setOpen}
      sub={splits.length ? <Badge size="sm" tone="warn">rebuild {splits.join(" + ")}</Badge> : <Badge size="sm">resume</Badge>}>
      <div className="dt-gen">
        <div className="dt-gen__splits" role="group" aria-label="Splits to rebuild">
          <span className="dt-label">Rebuild only</span>
          {SPLITS.map((s) => <Chip key={s} on={splits.includes(s)} onClick={() => toggle(s)}>{s}</Chip>)}
          <Button size="sm" variant="ghost" onClick={() => setSplits(["validate", "test"])}>validate + test</Button>
          {splits.length > 0 && <Button size="sm" variant="ghost" onClick={() => setSplits([])}>clear</Button>}
          <Tooltip content={splits.length
            ? "The selected splits are deleted at job start and rebuilt from zero; the others are left untouched."
            : "Complete splits are reused and incomplete ones resume."}>
            <span className="dt-help" tabIndex={0} aria-label="About the rebuild">?</span>
          </Tooltip>
          {safe && safe.dropped.length > 0 && (
            <Tooltip content={`The last run's extra flags carried ${safe.dropped.join(" ")}; it was left out of this form so a resume stays a resume. Pick the splits above to rebuild.`}>
              <span tabIndex={0}><Badge size="sm" tone="info">dropped {safe.dropped.join(" ")}</Badge></span>
            </Tooltip>
          )}
        </div>
        {safe ? (
          <div className="ops-step-inline">
            <StepCard key={safe.step.step_id} step={safe.step} sshConnected={!!steps.data?.ssh_connected} embedded
              extraParams={extraParams} hideParams={hideParams} />
          </div>
        ) : (
          <StepById stepId="synthetic_generate" embedded extraParams={extraParams} hideParams={hideParams} />
        )}
      </div>
    </Section>
  );
}

/* ── the tab ───────────────────────────────────────────────────────────── */

export default function Records() {
  const [rawSplit, setSplit] = useUrlState("split", "test");
  const split: Split = isSplit(rawSplit) ? rawSplit : "test";
  const sync = useJob("data:sky-sync");
  const generate = useJob("data:generate-sr");
  const running = sync.busy || generate.busy;
  const status = useSrStatus(running ? 3_000 : undefined);
  const s = status.data;
  const { online } = useFasrcOnline();
  const [viewerKey, setViewerKey] = useState(0);
  // Primitives from the viewer's state (a pan does not re-render the page).
  const [index, setIndex] = useState<number | null>(null);
  const [layout, setLayout] = useState<LayoutMode>("auto");
  const [tierCount, setTierCount] = useState(DEFAULT_TIERS.length);
  const [syncOpen, setSyncOpen] = useState(false);
  const [genOpen, setGenOpen] = useState(false);
  const [, setGenSection] = useUrlState("gen", false);
  const api = useRef<ViewerApi | null>(null);
  const params = useMemo(() => ({ subset: split }), [split]);
  const bump = () => setViewerKey((k) => k + 1);
  const [rawOverlay, setOverlay] = useUrlState("ov", "hr");
  const overlay: Overlay = isOverlay(rawOverlay) ? rawOverlay : "hr";
  const truth = useTruthSources(split, index);
  const { data: truthData, rows: truthRows, activeRow, setHover, pick } = truth;
  const markers = useMemo<ViewerMarkers | null>(() => {
    const grid = truthData?.geometry.hr;
    if (overlay === "off" || !truthData?.present || !grid) return null;
    const items: ViewerMarker[] = [];
    for (const src of truthRows) {
      const m = sourceMarker(src, grid);
      if (m) items.push({ key: String(m.row), x: m.cx, y: m.cy, r: m.r, kind: m.kind, title: m.title, dim: m.off });
    }
    return {
      grid: { width: grid.width, height: grid.height }, items,
      tiers: overlay === "hr" ? ["hr"] : undefined,
      activeKey: activeRow != null ? String(activeRow) : null,
      onHover: (key) => setHover(key == null ? null : Number(key)),
      onPick: (key) => pick(Number(key)),
    };
    // setHover / pick are stable enough for a marker overlay; the inputs drive it
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overlay, truthData, truthRows, activeRow]);

  const runSync = (subsets: Split[], kinds: string[]) => void startDataJob(sync, URLS.sync,
    { subsets: subsets.join(","), kinds: kinds.join(",") }, {
      label: "Records sync",
      question: {
        title: `Sync ${subsets.join(" + ")} from FASRC?`,
        message: `Pulls ${kinds.join(", ")} for ${subsets.join(", ")} (TFRecords are hundreds of MB each${subsets.includes("train") ? "; train is large" : ""}).`,
        confirmLabel: "Sync",
      },
      onDone: bump,
    });
  const runGenerate = (subsets: Split[], overwrite: boolean) => void startDataJob(generate, URLS.generateSr,
    { subsets: subsets.join(","), overwrite: overwrite ? 1 : 0 }, {
      label: "Generate SR",
      question: {
        title: `Generate the production SR for ${subsets.join(" + ")}?`,
        message: `Loads every active STARFULL member (TensorFlow) and the production gate${overwrite ? "; the existing SR of these splits is deleted first" : ""}.`,
        confirmLabel: "Generate", tone: overwrite ? "danger" : "default",
      },
      onDone: bump,
    });
  const openGeneration = () => {
    setGenSection(true);
    requestAnimationFrame(() => document.getElementById("dt-generate")?.scrollIntoView({ behavior: "smooth", block: "start" }));
  };

  usePageActions([
    ...SPLITS.map((sp) => ({ id: `records-split-${sp}`, label: `Records: show the ${sp} split`, group: "Records", run: () => setSplit(sp) })),
    { id: "records-sync", label: "Sync the records from FASRC…", group: "Records", disabled: !online,
      keywords: ["tfrecord", "rsync", "pull"], run: () => setSyncOpen(true) },
    { id: "records-generate-sr", label: "Generate the production SR over the local records…", group: "Records",
      disabled: !s?.can_generate, run: () => setGenOpen(true) },
    { id: "records-generate-pairs", label: "Generate synthetic training pairs (FASRC)…", group: "Records",
      keywords: ["synthetic_generate", "regenerate", "splits"], run: openGeneration },
    { id: "records-refresh", label: "Refresh the records status", group: "Records", run: () => { void status.reload(); bump(); } },
  ]);

  const splitInfo = s?.splits[split];
  // The viewer mounts once the (fast, local) status says the split is here —
  // or the status failed: a viewer opened on an absent split would settle on
  // its lone disabled tier and write that into the URL (v.rec.t) on a visit.
  const absent = !!s && !splitInfo?.present;
  const viewable = s ? !absent : !!status.error;
  const onViewerState = (st: ViewerState) => {
    setIndex(st.index);
    setLayout(st.layout);
    setTierCount(st.tiers?.length || DEFAULT_TIERS.length);
  };
  return (
    <Page className="dt-page dt-page--image">
      <DataBar label="Records">
        <Segmented size="sm" className="dt-seg-text" value={split} onChange={(v) => setSplit(v)} aria-label="Split"
          options={SPLITS.map((sp) => ({
            value: sp, label: <>{sp} <span className="muted">{s ? formatCount(s.splits[sp]?.count ?? 0) : ""}</span></>,
          }))} />
        {viewable && !absent && <TruthControls truth={truth} overlay={overlay} onOverlay={setOverlay} index={index} split={split} />}
        <Spacer />
        <div className="dt-bar__status" role="group" aria-label={`State of the ${split} split`}>
          {s && <FilesBadge status={s} split={split} />}
          {s && <SrBadge status={s} split={split} />}
          <NoiseBadge />
        </div>
        <BarActions>
          {s && <SyncPopover open={syncOpen} onOpenChange={setSyncOpen} busy={sync.busy} online={online} onStart={runSync} />}
          {s && <GeneratePopover key={split} status={s} open={genOpen} onOpenChange={setGenOpen} busy={generate.busy} onStart={runGenerate} />}
          <Menu label="More record actions" trigger={<IconButton icon="more" size="sm" label="More record actions" />} items={[
            { label: "Generate synthetic training pairs…", onSelect: openGeneration },
            { label: "Regenerate the SR of this split (overwrite)…", disabled: !s?.can_generate || !s?.subsets.includes(split),
              onSelect: () => runGenerate([split], true) },
            { type: "separator" },
            { label: "Refresh the status", onSelect: () => { void status.reload(); bump(); } },
          ]} />
        </BarActions>
      </DataBar>
      {status.error && !s && (
        <Callout tone="bad" title="Records status did not load" action={<Button size="sm" onClick={() => void status.reload()}>Retry</Button>}>
          <span className="dt-pre">{status.error.message}</span>
        </Callout>
      )}
      {absent ? (
        <EmptyState icon="database" title={`No ${split} records on this machine`}
          action={<Button variant="primary" icon="download" disabled={!online} onClick={() => setSyncOpen(true)}>Sync from FASRC</Button>}>
          {online ? "Pull the split from FASRC (a background job)." : OFFLINE_HINT}
        </EmptyState>
      ) : viewable && (
        <div className="dt-figure">
          <ViewerStage layout={layout} frames={tierCount}>
            <ImageViewer key={`${split}-${viewerKey}`} collection="sky" params={params} urlKey="rec"
              tiers={DEFAULT_TIERS} initialId={index != null ? recordObjectId(split, index) : undefined}
              markers={markers}
              onReady={(a) => { api.current = a; }}
              onState={onViewerState} />
          </ViewerStage>
        </div>
      )}
      <JobStrip job={sync} />
      <JobStrip job={generate} />
      {viewable && <SourcesTable truth={truth} split={split} index={index} />}
      {s && !absent && <CensusSection split={split} index={index} onGo={(i) => api.current?.goTo(i)} />}
      <GenerationSection />
    </Page>
  );
}
