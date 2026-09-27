/* Data › Records (spec §8.4): the synthetic training TFRecords.
 *
 * Toolbar: split (test / validate / train, with record counts), the local
 * files, the SR tier's state against the production model, the records'
 * noise-model health check, and the actions — sync from FASRC (a background
 * job with progress), generate the production SR (with overwrite), generate
 * new pairs (the synthetic_generate step). Body: the viewer (LR · HR · BHR ·
 * Clean (starless) · SR) beside the current record's truth sources (a source
 * map + table from sources_<split>.csv; row / marker → the `truth`
 * inspector), then the split's per-record census (row → that record). The
 * split, viewer object/tiers/view, source-type filter and open sections live
 * in the URL. */
import { useMemo, useRef, useState } from "react";
import { useJob } from "../../../api/jobs";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { useSystemAlerts } from "../../../app/status";
import { StepById, StepCard, useStepsStatus } from "../../../fasrc";
import { formatBytes, formatCount, formatNumber } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { useInspector } from "../../../state/inspector";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, Checkbox, Chip, DataTable, EmptyState, IconButton, Menu, Page,
  Popover, Section, Segmented, Switch, Tooltip, type DataColumn,
} from "../../../ui";
import { ImageViewer, type ViewerApi, type ViewerState } from "../../../viewer";
import {
  SPLITS, SYNC_KINDS, URLS, useSrStatus, type FieldCensus, type RecordSources, type SourcesCensus, type Split,
  type SrStatus, type TruthSource,
} from "../api";
import { BarGroup, DataBar, JobStrip, OFFLINE_HINT, Spacer, startDataJob, useFasrcOnline } from "../common";
import {
  SR_STATE_LABEL, SR_STATE_TONE, formatCompact, mapViewBox, recordObjectId, resumeSafeStep, sourceMarker, truthId,
  type Marker,
} from "../model";
import "../register";
import "../data.css";

const TYPES = ["galaxy", "star", "lens", "other"] as const;
const isSplit = (v: string): v is Split => (SPLITS as readonly string[]).includes(v);

/* ── toolbar pieces ────────────────────────────────────────────────────── */

function FileBadges({ status, split }: { status: SrStatus; split: Split }) {
  const files = status.splits[split]?.files;
  if (!files) return null;
  const items: [string, string][] = [["dirty", "LR"], ["hr", "HR"], ["clean", "Clean"], ["sources", "Sources"]];
  return (
    <BarGroup label="Local files">
      {items.map(([key, label]) => {
        const f = files[key as keyof typeof files];
        const count = f && "count" in f ? f.count : undefined;
        const tip = f
          ? `${f.name} · ${formatBytes(f.size_bytes)}${count === null ? " · truncated or corrupt" : count != null ? ` · ${count} records` : ""}`
          : `${key}_${split} is not synced`;
        return (
          <Tooltip key={key} content={tip}>
            <span tabIndex={0}>
              <Badge size="sm" dot tone={!f ? "neutral" : count === null ? "bad" : "good"}>{label}</Badge>
            </span>
          </Tooltip>
        );
      })}
    </BarGroup>
  );
}

function SrBadge({ status, split }: { status: SrStatus; split: Split }) {
  const sr = status.splits[split]?.sr;
  if (!sr) return null;
  const who = sr.manifest?.model_label ? ` · ${sr.manifest.model_label}` : "";
  const tip = [
    `${sr.count}${sr.records_count != null ? ` / ${sr.records_count}` : ""} SR cubes${who}`,
    ...sr.reasons,
    sr.manifest?.generated_at ? `generated ${sr.manifest.generated_at}` : "",
  ].filter(Boolean).join("\n");
  return (
    <Tooltip content={<span className="dt-pre">{tip}</span>}>
      <span tabIndex={0}>
        <Badge size="sm" tone={SR_STATE_TONE[sr.state]} dot>
          {SR_STATE_LABEL[sr.state]}{sr.state === "partial" && sr.records_count ? ` ${sr.count}/${sr.records_count}` : ""}
        </Badge>
      </span>
    </Tooltip>
  );
}

function NoiseBadge() {
  const check = useSystemAlerts().data?.checks.find((c) => c.id === "records-noise");
  if (!check || check.state === "ok") {
    return check ? <Tooltip content={check.title}><span tabIndex={0}><Badge size="sm" tone="good">noise ✓</Badge></span></Tooltip> : null;
  }
  const tone = check.state === "bad" ? "bad" : check.state === "warn" ? "warn" : "neutral";
  return (
    <Tooltip content={<span className="dt-pre">{[check.title, check.detail].filter(Boolean).join("\n")}</span>}>
      <span tabIndex={0}><Badge size="sm" tone={tone} dot>{check.state === "unknown" ? "noise model ?" : "noise model"}</Badge></span>
    </Tooltip>
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
        <span tabIndex={0}><Button size="sm" icon="download" loading={busy} disabled>Sync</Button></span>
      </Tooltip>
    );
  }
  return (
    <Popover open={open} onOpenChange={onOpenChange} label="Sync records from FASRC" width={300}
      trigger={<Button size="sm" icon="download" loading={busy}>Sync</Button>}>
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
        <span tabIndex={0}><Button size="sm" icon="wave" loading={busy} disabled>Generate SR</Button></span>
      </Tooltip>
    );
  }
  return (
    <Popover open={open} onOpenChange={onOpenChange} label="Generate SR" width={300}
      trigger={<Button size="sm" icon="wave" loading={busy}>Generate SR</Button>}>
      <div className="dt-pop">
        <div className="dt-pop__title">Production SR over the local records</div>
        <fieldset className="dt-pop__set">
          <legend>Splits</legend>
          {present.map((s) => (
            <Checkbox key={s} checked={subsets.includes(s)}
              onChange={(on) => setSubsets((c) => (on ? [...c, s] : c.filter((x) => x !== s)))}>
              {s} <span className="muted">· {status.splits[s]?.count ?? 0} records · {status.sr[s] ?? 0} SR</span>
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

function SourceMap({ data, types, hover, onHover, onPick }: {
  data: RecordSources; types: string[]; hover: number | null; onHover: (row: number | null) => void;
  onPick: (row: number) => void;
}) {
  const grid = data.geometry.hr;
  const width = grid?.width ?? 512, height = grid?.height ?? 512;
  const markers = useMemo(() => data.sources
    .filter((s) => !types.length || types.includes(TYPES.includes(s.type as (typeof TYPES)[number]) ? s.type : "other"))
    .map((s) => sourceMarker(s, grid)).filter((m): m is Marker => m != null), [data, types, grid]);
  const [x, y, w, h] = mapViewBox(markers, width, height);
  return (
    <svg className="dt-map" viewBox={`${x} ${y} ${w} ${h}`} role="img" preserveAspectRatio="xMidYMid meet"
      aria-label={`Truth sources of record ${data.field_index}: ${markers.length} shown`}>
      <rect className="dt-map__frame" x={0} y={0} width={width} height={height} />
      {markers.map((m) => (
        <g key={m.row} className="dt-map__mk" data-kind={m.kind} data-off={m.off || undefined}
          data-hover={hover === m.row || undefined} tabIndex={0} role="button" aria-label={m.title}
          onMouseEnter={() => onHover(m.row)} onMouseLeave={() => onHover(null)}
          onFocus={() => onHover(m.row)} onBlur={() => onHover(null)}
          onClick={() => onPick(m.row)} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onPick(m.row); } }}>
          <title>{m.title}</title>
          {m.kind === "star" ? (
            <path d={`M${m.cx - m.r} ${m.cy}H${m.cx + m.r}M${m.cx} ${m.cy - m.r}V${m.cy + m.r}`} />
          ) : (
            <circle cx={m.cx} cy={m.cy} r={m.r} />
          )}
        </g>
      ))}
    </svg>
  );
}

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

function SourcesCard({ split, index }: { split: Split; index: number | null }) {
  const [types, setTypes] = useUrlState<string[]>("st", []);
  const [hover, setHover] = useState<number | null>(null);
  const inspected = useInspector((s) => (s.current?.kind === "truth" ? s.current.id : null));
  const url = index != null ? URLS.sources(split, index) : null;
  const res = useResource<RecordSources>(url, [split, index], { ttl: 5 * 60_000 });
  const data = res.data;
  const rows = useMemo(() => (data?.sources ?? []).filter((s) => !types.length
    || types.includes(TYPES.includes(s.type as (typeof TYPES)[number]) ? s.type : "other")), [data, types]);
  const pick = (row: number) => { if (index != null) openInspector({ kind: "truth", id: truthId(split, index, row) }); };
  const activeRow = hover ?? (inspected && index != null && inspected.startsWith(`${split}/${index}/`)
    ? Number(inspected.split("/")[2]) : null);
  const counts = data?.counts;
  const toggle = (t: string) => setTypes(types.includes(t) ? types.filter((x) => x !== t) : [...types, t]);
  return (
    <Card className="dt-sources">
      <CardHead title={index != null ? `Truth sources · record ${index}` : "Truth sources"}
        sub={data?.geometry.hr ? `HR ${data.geometry.hr.width}² px · ${data.geometry.hr.pixscale}″/px` : undefined}
        right={counts && (
          <div className="dt-chips" role="group" aria-label="Source types">
            {TYPES.filter((t) => t !== "other" || counts.other > 0).map((t) => (
              <Chip key={t} on={!types.length || types.includes(t)} onClick={() => toggle(t)}>
                {t} <span className="muted">{counts[t]}</span>
              </Chip>
            ))}
          </div>
        )} />
      <CardBody>
        {index == null ? <EmptyState compact icon="image" title="Pick a record in the viewer" />
          : res.loading && !data ? <div className="dt-map dt-map--empty" aria-busy="true" />
            : res.error ? <Callout tone="bad" title="Sources did not load"><span className="dt-pre">{res.error.message}</span></Callout>
              : !data?.present ? (
                <EmptyState compact icon="table" title={`No sources_${split}.csv`}>
                  Sync the split with its sources file to see the truth catalogue.
                </EmptyState>
              ) : (
                <div className="dt-sources__body">
                  <SourceMap data={data} types={types} hover={activeRow} onHover={setHover} onPick={pick} />
                  <DataTable rows={rows} columns={SOURCE_COLUMNS} rowKey={(s) => String(s.row)} dense height={300}
                    aria-label={`Sources of record ${index}`} exportName={`sources_${split}_${index}`}
                    activeKey={activeRow != null ? String(activeRow) : null}
                    inspect={(s) => ({ kind: "truth", id: truthId(split, index, s.row) })}
                    empty={data.sources.length ? "No source of the selected types" : "This record has no sources"}
                    toolbar={counts && counts.off_field > 0
                      ? <Tooltip content="Centred outside the frame, their light spills in"><span tabIndex={0} className="muted">{counts.off_field} off-field</span></Tooltip>
                      : undefined} />
                </div>
              )}
      </CardBody>
    </Card>
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
      sub={res.data?.present ? `${res.data.fields.length} records` : undefined}>
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
          <span className="eyebrow">Rebuild only</span>
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
  const [index, setIndex] = useState<number | null>(null);
  const [syncOpen, setSyncOpen] = useState(false);
  const [genOpen, setGenOpen] = useState(false);
  const [, setGenSection] = useUrlState("gen", false);
  const api = useRef<ViewerApi | null>(null);
  const params = useMemo(() => ({ subset: split }), [split]);
  const bump = () => setViewerKey((k) => k + 1);

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
  return (
    <Page className="dt-page">
      <DataBar label="Records">
        <Segmented size="sm" value={split} onChange={(v) => setSplit(v)} aria-label="Split"
          options={SPLITS.map((sp) => ({
            value: sp, label: <>{sp} <span className="muted">{s ? formatCount(s.splits[sp]?.count ?? 0) : ""}</span></>,
          }))} />
        {s && <FileBadges status={s} split={split} />}
        {s && <SrBadge status={s} split={split} />}
        <NoiseBadge />
        <Spacer />
        {s && <SyncPopover open={syncOpen} onOpenChange={setSyncOpen} busy={sync.busy} online={online} onStart={runSync} />}
        {s && <GeneratePopover key={split} status={s} open={genOpen} onOpenChange={setGenOpen} busy={generate.busy} onStart={runGenerate} />}
        <Menu label="More record actions" trigger={<IconButton icon="more" size="sm" label="More record actions" />} items={[
          { label: "Generate synthetic training pairs…", onSelect: openGeneration },
          { label: "Regenerate the SR of this split (overwrite)…", disabled: !s?.can_generate || !s?.subsets.includes(split),
            onSelect: () => runGenerate([split], true) },
          { type: "separator" },
          { label: "Refresh the status", onSelect: () => { void status.reload(); bump(); } },
        ]} />
      </DataBar>
      <JobStrip job={sync} />
      <JobStrip job={generate} />
      {status.error && !s && (
        <Callout tone="bad" title="Records status did not load" action={<Button size="sm" onClick={() => void status.reload()}>Retry</Button>}>
          <span className="dt-pre">{status.error.message}</span>
        </Callout>
      )}
      {s && !splitInfo?.present ? (
        <EmptyState icon="database" title={`No ${split} records on this machine`}
          action={<Button variant="primary" icon="download" disabled={!online} onClick={() => setSyncOpen(true)}>Sync from FASRC</Button>}>
          {online ? "Pull the split from FASRC (a background job)." : OFFLINE_HINT}
        </EmptyState>
      ) : (
        <div className="dt-split">
          <Card className="dt-viewer-card">
            <CardBody>
              <ImageViewer key={`${split}-${viewerKey}`} collection="sky" params={params} urlKey="rec"
                tiers={["dirty", "hr"]} initialId={index != null ? recordObjectId(split, index) : undefined}
                onReady={(a) => { api.current = a; }}
                onState={(st: ViewerState) => setIndex(st.index)} />
            </CardBody>
          </Card>
          <SourcesCard split={split} index={index} />
        </div>
      )}
      {s && splitInfo?.present && <CensusSection split={split} index={index} onGo={(i) => api.current?.goTo(i)} />}
      <GenerationSection />
    </Page>
  );
}
