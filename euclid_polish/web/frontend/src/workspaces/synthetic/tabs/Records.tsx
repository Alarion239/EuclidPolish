/* Synthetic › Records: what comes out of the generator — the local
   synthetic records against their truth sources, then the census.

   One toolbar row: the split (test / validate with their record counts;
   train stays on FASRC, disabled with the reason), the truth sources (drawn
   on Off / HR / All tiers, one chip per type with its count: legend and
   filter at once), a badge only when a local file is corrupt or missing, and
   the "Generate and sync" drawer. Then the viewer (LR · HR · Blurred HR ·
   Clean; HR at the LR's surface brightness, so both tiers read alike; the
   SR tier stays viewable, its generation moved to Models › Images). Below:
   the current record's truth sources, then the census (`?section=census`
   scrolls to it): Σ VIS and brightest-star histograms (a click opens the
   nearest record), the sources per arcmin² generated · prior · Q1 and the
   per-record table. The drawer holds the synthetic_generate step, the sync
   from FASRC and the generation knobs read-only, edited in System › Config.
   The split, viewer object/tiers/view, overlay, hidden types and the
   drawer are in the URL. */
import { useMemo, useRef, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { useJob } from "../../../api/jobs";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import Plot from "../../../charts/Plot";
import { C } from "../../../colors";
import { formatCount, formatNumber } from "../../../format";
import { useArrivalScroll } from "../../../hooks/arrival";
import { useUrlState } from "../../../hooks/useUrlState";
import { linearTicks } from "../../../ticks";
import { useInspector } from "../../../state/inspector";
import {
  Badge, Button, Callout, Caption, Checkbox, Chip, DataTable, EmptyState, FactsList, Menu, Page, Popover, Section,
  Segmented, Tooltip, type DataColumn, type MenuItem, type Tone,
} from "../../../ui";
import { ImageViewer, type ViewerApi, type ViewerMarker, type ViewerMarkers, type ViewerState } from "../../../viewer";
import { useGalaxies, useStars } from "../api";
import { Drawer, DrawerButton, Info } from "../common";
import {
  SPLITS, SYNC_KINDS, URLS, useSrStatus, type FieldCensus, type Geometry, type RecordSources, type SourcesCensus, type Split,
  type SrStatus, type TruthSource,
} from "../dataApi";
import { BarActions, DataBar, JobStrip, OFFLINE_HINT, Spacer, startDataJob, useFasrcOnline } from "../dataCommon";
import {
  formatCompact, recordFiles, recordObjectId, shownTypes, sourceMarker, sourceTypeChips, toggleHiddenType, truthId,
  type SourceType,
} from "../dataModel";
import { ConfigKnobsLink } from "../../shared/ConfigKnobsLink";
import type { GroupId } from "../../system/configFields";
import { EditInConfig, GeneratePanel } from "../generate";
import { useIncludeTraining } from "../header";
import { censusRows, generatedSample, type CensusRow } from "../fields/model";
import { censusHistogram, lensDensity, nearestRecord, recordArea, recordSrLink } from "../recordsModel";
import "../register";
import "../data.css";
import "../synthetic.css";

/** Splits a visitor can open here (train is generated and read on FASRC). */
const LOCAL_SPLITS = ["test", "validate"] as const;
const isSplit = (v: string): v is Split => (LOCAL_SPLITS as readonly string[]).includes(v);

/** Where the truth sources are drawn: nowhere, on the HR image, or on every tier. */
type Overlay = "off" | "hr" | "all";
const OVERLAYS: readonly Overlay[] = ["off", "hr", "all"];
const isOverlay = (v: string): v is Overlay => (OVERLAYS as readonly string[]).includes(v);
const OVERLAY_LABEL: Record<Overlay, string> = { off: "Off", hr: "HR", all: "All tiers" };

/** The tiers the viewer opens with (its URL `v.rec.t` may name others). */
const DEFAULT_TIERS = ["dirty", "hr"];
/** HR at the LR's surface brightness (each e⁻ tier × (0.1″ / its pixel)²). */
const MATCHED = { matchSurfaceBrightness: true };

/* ── toolbar pieces ────────────────────────────────────────────────────── */

/** The split's local files as ONE badge, shown only on a problem (a corrupt
 *  or truncated file, or a split partly synced). */
function FilesProblem({ status, split }: { status: SrStatus; split: Split }) {
  const files = status.splits[split]?.files;
  if (!files) return null;
  const { label, tone, rows } = recordFiles(files, split);
  if (tone === "good") return null;
  return (
    <Tooltip content={(
      <dl className="dt-tipdl">
        {rows.map((r) => <div key={r.key} data-state={r.state}><dt>{r.label}</dt><dd>{r.detail}</dd></div>)}
      </dl>
    )}>
      <span tabIndex={0} className="dt-tipbadge dt-tipbadge--keep">
        <Badge size="sm" tone={tone === "neutral" ? "warn" : tone as Tone} dot><span className="dt-tipbadge__text">{label}</span></Badge>
      </span>
    </Tooltip>
  );
}

function SyncPopover({ onStart, busy, online }: {
  busy: boolean; online: boolean; onStart: (subsets: Split[], kinds: string[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const [subsets, setSubsets] = useState<Split[]>(["test", "validate"]);
  const [kinds, setKinds] = useState<string[]>([...SYNC_KINDS]);
  const toggle = <T extends string>(list: T[], v: T) => (list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);
  if (!online) {
    return (
      <Tooltip content={OFFLINE_HINT}>
        <span tabIndex={0}><Button size="sm" icon="download" loading={busy} disabled>Sync from FASRC…</Button></span>
      </Tooltip>
    );
  }
  return (
    <Popover open={open} onOpenChange={setOpen} label="Sync records from FASRC" width={300}
      trigger={<Button size="sm" icon="download" loading={busy}>Sync from FASRC…</Button>}>
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
          onClick={() => { setOpen(false); onStart(subsets, kinds); }}>
          Start sync
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
  { id: "mag_vis", header: "VIS mag", width: 76, numeric: true, cell: (s) => formatNumber(s.mag_vis, { digits: 2 }) },
  { id: "flux_vis_e", header: "VIS e⁻", width: 70, numeric: true, cell: (s) => (s.flux_vis_e == null ? "—" : formatCompact(s.flux_vis_e)) },
  { id: "size", header: "Size (″)", width: 76, headerText: "Re or thetaE arcsec", numeric: true,
    accessor: (s) => (s.type === "lens" ? s.theta_E_arcsec : s.re_arcsec),
    cell: (s) => formatNumber(s.type === "lens" ? s.theta_E_arcsec : s.re_arcsec, { digits: 3 }) },
  { id: "z", header: "z", width: 56, numeric: true, hidden: true, cell: (s) => formatNumber(s.z, { digits: 2 }) },
  { id: "subhalo_id", header: "TNG", width: 80, headerText: "TNG subhalo", accessor: (s) => s.subhalo_id ?? "" },
  { id: "sfr_class", header: "SFR class", width: 96, hidden: true, accessor: (s) => s.sfr_class ?? "" },
  { id: "off_field", header: "Off", width: 60, headerText: "Off-field", accessor: (s) => (s.off_field ? "yes" : ""),
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

/** Where and which truth sources are drawn: the overlay tiers and one chip
 *  per type with its count (the chips are the markers' legend AND the
 *  filter). In a crowded toolbar (compact level 3) the tiers become one menu. */
function TruthControls({ truth, overlay, onOverlay, index, split }: {
  truth: Truth; overlay: Overlay; onOverlay: (v: Overlay) => void; index: number | null; split: Split;
}) {
  const { data, res, hidden, toggle } = truth;
  const counts = data?.present ? data.counts : null;
  return (
    <div className="dt-caption" role="group" aria-label={index != null ? `Truth sources of record ${index}` : "Truth sources overlay"}>
      <span className="dt-caption__label dt-ov--wide" aria-hidden="true">Sources on</span>
      <Segmented size="sm" className="dt-ov--wide" value={overlay} onChange={onOverlay} aria-label="Draw the truth sources on"
        options={OVERLAYS.map((v) => ({ value: v, label: OVERLAY_LABEL[v] }))} />
      <span className="dt-ov--menu">
        <Menu label="Draw the truth sources on" items={[
          { type: "label", label: "Draw the truth sources on" },
          ...OVERLAYS.map((v): MenuItem => ({
            type: "checkbox", label: OVERLAY_LABEL[v], checked: overlay === v, keepOpen: false,
            onCheckedChange: () => onOverlay(v),
          })),
        ]} trigger={<Button size="sm" variant="ghost" iconRight="chevronDown" aria-label={`Truth sources on: ${OVERLAY_LABEL[overlay]}`}>Sources: {OVERLAY_LABEL[overlay]}</Button>} />
      </span>
      {counts && sourceTypeChips(counts, hidden).map((c) => (
        <Chip key={c.type} on={c.shown} onClick={() => toggle(c.type)}
          title={`${c.shown ? "Hide" : "Show"} the ${c.type} sources (markers and table rows)`}>
          <span className="dt-key" data-kind={c.type} aria-hidden="true" />{c.type} <span className="muted">{c.count}</span>
        </Chip>
      ))}
      {counts && counts.off_field > 0 && (
        <Tooltip content="Centred outside the frame, their light spills in (dashed)">
          <span tabIndex={0} className="dt-caption__note dt-ov--wide">{counts.off_field} off-field</span>
        </Tooltip>
      )}
      {index != null && res.loading && !data && <span className="dt-caption__note">Loading sources…</span>}
      {index != null && data && !data.present && (
        <Tooltip content={`Sync the ${split} split with its sources file to see the truth catalogue`}>
          <span tabIndex={0}><Badge size="sm" tone="warn">No sources_{split}.csv</Badge></span>
        </Tooltip>
      )}
      {res.error && <Badge size="sm" tone="bad">Sources: {res.error.message}</Badge>}
    </div>
  );
}

/** The sources table's sub line: the HR geometry, plus "N of M shown" only
 *  while the type chips hide some rows (the chips already carry the counts). */
function sourcesSub(shown: number, total: number, hr: Geometry["hr"]): string | undefined {
  const parts = [
    shown < total ? `${formatCount(shown)} of ${formatCount(total)} shown` : null,
    hr ? `HR ${hr.width} px at ${hr.pixscale}″` : null,
  ].filter(Boolean);
  return parts.length ? parts.join(" · ") : undefined;
}

function SourcesTable({ truth, split, index }: { truth: Truth; split: Split; index: number | null }) {
  const [open, setOpen] = useUrlState("srct", true);
  const { data, rows, activeRow } = truth;
  if (index == null || !data?.present) return null;
  return (
    <Section title={`Truth sources of record ${index}`} collapsible open={open} onOpenChange={setOpen}
      sub={sourcesSub(rows.length, data.sources.length, data.geometry.hr)}>
      <DataTable rows={rows} columns={SOURCE_COLUMNS} rowKey={(s) => String(s.row)} dense height={300} countText={null}
        aria-label={`Sources of record ${index}`} exportName={`sources_${split}_${index}`}
        activeKey={activeRow != null ? String(activeRow) : null}
        inspect={(s) => ({ kind: "truth", id: truthId(split, index, s.row) })}
        empty={data.sources.length ? "No source of the selected types" : "This record has no sources"} />
    </Section>
  );
}

/* ── census ────────────────────────────────────────────────────────────── */

const CENSUS_COLUMNS: DataColumn<FieldCensus>[] = [
  { id: "field_index", header: "Record", width: 72, numeric: true },
  { id: "n", header: "Sources", width: 76, numeric: true },
  { id: "galaxy", header: "Galaxies", width: 80, numeric: true },
  { id: "star", header: "Stars", width: 64, numeric: true },
  { id: "lens", header: "Lenses", width: 68, numeric: true },
  { id: "off_field", header: "Off-field", width: 80, numeric: true, hidden: true },
  { id: "brightest_star_mag", header: "Brightest star", headerText: "Brightest star VIS mag", width: 110, numeric: true,
    cell: (f) => formatNumber(f.brightest_star_mag, { digits: 2 }) },
  { id: "brightest_galaxy_mag", header: "Brightest galaxy", headerText: "Brightest galaxy VIS mag", width: 120, numeric: true,
    priority: 1, cell: (f) => formatNumber(f.brightest_galaxy_mag, { digits: 2 }) },
  { id: "total_vis_e", header: "Σ VIS (e⁻)", headerText: "Σ VIS e⁻", width: 92, numeric: true, cell: (f) => formatCompact(f.total_vis_e) },
];

const density3 = (v: number | null) => formatNumber(v, { sig: 3 });
/** One decimal count for values shown side by side: 2 significant figures
 *  of the largest, and at least 1 of the smallest (0.055 · 0.020; 2.8 · 0.5). */
function sharedDecimals(values: readonly (number | null | undefined)[]): number {
  const positive = values.filter((v): v is number => v != null && Number.isFinite(v) && v > 0);
  if (!positive.length) return 2;
  const order = (v: number) => Math.floor(Math.log10(v));
  return Math.max(0, 1 - order(Math.max(...positive)), -order(Math.min(...positive)));
}
const mag2 = (v: number) => v.toFixed(2);
const WINDOW_LABEL: Record<CensusRow["kind"], string> = { galaxies: "to the Q1 5σ limit", stars: "Q1 trusted window" };
const KIND_LABEL: Record<CensusRow["kind"], string> = { galaxies: "Galaxies", stars: "Stars" };

/** Generated vs prior vs Q1 sources per arcmin² (galaxies and stars from the
 *  shared galaxy / star payloads, over one VIS window per row; lenses from
 *  this split's census and the configured prior). Never waits for the pixel
 *  cache; the training toggle adds the train split's catalogue. */
function DensityTable({ census, lensPrior }: { census: SourcesCensus | null | undefined; lensPrior: number | null }) {
  const [training] = useIncludeTraining();
  const galaxies = useGalaxies(training);
  const stars = useStars(training);
  const rows = censusRows(galaxies.data, stars.data);
  const lens = census?.present ? lensDensity(census) : null;
  // Generated and prior lens densities to the same decimals ("0.055" beside
  // "0.020", not "0.02").
  const lensDigits = sharedDecimals([lens?.density, lensPrior]);
  const caption = [
    generatedSample(galaxies.data?.sources.synthetic?.available ? galaxies.data.sources.synthetic : undefined,
      stars.data?.distribution?.density_comparison),
    "Q1 is compared only over the magnitudes where it is complete; lenses are also counted among the galaxies",
  ].filter(Boolean).join(" · ");
  if (!rows.length && !lens) {
    return galaxies.loading || stars.loading ? <Caption>Loading the galaxy and star densities…</Caption>
      : <EmptyState compact icon="table" title="No galaxy or stellar prior has been fitted yet" />;
  }
  return (
    <div className="syn-scroll">
      <table className="syn-table" aria-label="Sources per arcmin²: generated, prior and Q1">
        <thead>
          <tr><th scope="col">Kind</th><th scope="col">VIS range</th><th scope="col" data-num="">Generated</th>
            <th scope="col" data-num="">Prior</th><th scope="col" data-num="">Q1</th></tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={`${r.kind}:${r.window}`}>
              <th scope="row">{i === 0 || rows[i - 1].kind !== r.kind ? KIND_LABEL[r.kind] : ""}</th>
              <td>{`${mag2(r.range[0])}–${mag2(r.range[1])} · ${r.window === "q1" ? WINDOW_LABEL[r.kind] : "full prior"}`}</td>
              <td data-num="" data-tone={r.generated != null && r.prior != null && r.prior > 0 && Math.abs(r.generated / r.prior - 1) > 0.05 ? "warn" : undefined}>
                {density3(r.generated)}</td>
              <td data-num="">{density3(r.prior)}</td>
              <td data-num="">{r.window === "q1" ? density3(r.q1) : <span className="rl-faint">incomplete</span>}</td>
            </tr>
          ))}
          {lens && (
            <tr>
              <th scope="row">Lenses</th>
              <td>{`all · ${census?.subset ?? ""} split`}</td>
              <td data-num="">
                <Tooltip content={`${formatCount(lens.count)} lens${lens.count === 1 ? "" : "es"} in ${formatNumber(lens.area)} arcmin²`}>
                  <span tabIndex={0}>{formatNumber(lens.density, { digits: lensDigits })}</span>
                </Tooltip>
              </td>
              <td data-num="">{lensPrior != null ? formatNumber(lensPrior, { digits: lensDigits }) : "—"}</td>
              <td data-num=""><span className="rl-faint">no Q1 count</span></td>
            </tr>
          )}
        </tbody>
      </table>
      <Caption>{`arcmin⁻² · ${caption}`}</Caption>
    </div>
  );
}

/** One census histogram (Σ VIS in log e⁻, or the brightest star's VIS mag);
 *  a click opens the record whose value is nearest. */
function CensusHistogram({ fields, which, onGo }: {
  fields: FieldCensus[]; which: "total" | "star"; onGo: (index: number) => void;
}) {
  const h = censusHistogram(fields, which);
  if (!h) return <EmptyState compact icon="activity" title={which === "star" ? "No record has a star" : "No records"} />;
  const top = Math.max(1, ...h.counts);
  const title = which === "total" ? "Σ VIS per record" : "Brightest star per record";
  return (
    <figure className="rl-fig">
      <figcaption className="rl-fig__head"><strong>{title}</strong><small>{which === "star" ? `${formatCount(h.n)} records · click a bar to open its record` : "click a bar to open its record"}</small></figcaption>
      <Plot xDomain={h.domain} yDomain={[0, top * 1.1]} yTicks={linearTicks([0, top * 1.1], { count: 4 })}
        xScale={which === "total" ? "log" : "linear"}
        xLabel={which === "total" ? "Σ VIS (e⁻, log)" : "brightest star VIS (AB mag)"} yLabel="records"
        series={[{ x: h.centers, y: h.counts, mode: "histogram", color: which === "total" ? C.mean : C.comb, fillAlpha: 0.3, name: title }]}
        aspect={0.5} exportName={`records-${which}`} aria-label={`${title} histogram`}
        xFormat={which === "total" ? (v) => formatCompact(v) : (v) => v.toFixed(1)}
        onPlotClick={(p) => { const i = nearestRecord(fields, which, p.x); if (i != null) onGo(i); }} />
    </figure>
  );
}

function Census({ split, index, onGo, lensPrior }: {
  split: Split; index: number | null; onGo: (i: number) => void; lensPrior: number | null;
}) {
  const res = useResource<SourcesCensus>(URLS.sources(split), [split], { ttl: 5 * 60_000 });
  const [section] = useUrlState("section", "");
  const [legacyView] = useUrlState("view", "");
  const ref = useRef<HTMLElement>(null);
  const wanted = section === "census" || legacyView === "census";
  useArrivalScroll(ref, wanted, !!res.data);
  const fields = res.data?.fields ?? [];
  const area = res.data ? recordArea(res.data) : null;
  return (
    <section ref={ref} id="syn-census" className="syn-census" aria-label="Census">
      <Section title={`Census of the ${split} split`}
        sub={res.data?.present ? `${formatCount(fields.length)} records${area ? ` · ${formatNumber(area, { sig: 3 })} arcmin² each` : ""}` : undefined}>
        {res.error ? <Callout tone="bad" title="Census did not load"><span className="dt-pre">{res.error.message}</span></Callout>
          : res.data && !res.data.present ? <EmptyState compact icon="table" title={`No sources_${split}.csv`} />
            : (
              <div className="rl-stack">
                <div className="rl-grid rl-grid--2">
                  <CensusHistogram fields={fields} which="total" onGo={onGo} />
                  <CensusHistogram fields={fields} which="star" onGo={onGo} />
                </div>
                <h3 className="syn-subtitle">Sources per arcmin²</h3>
                <DensityTable census={res.data} lensPrior={lensPrior} />
                <DataTable rows={fields} columns={CENSUS_COLUMNS} rowKey={(f) => String(f.field_index)}
                  loading={res.loading && !res.data} dense height={320} urlKey="cn" aria-label={`Records in ${split}`}
                  exportName={`records_${split}`} activeKey={index != null ? String(index) : null} countText={null}
                  onRowClick={(f) => onGo(f.field_index)} />
              </div>
            )}
      </Section>
    </section>
  );
}

/* ── generation knobs (read-only here; System › Config edits them) ────── */

/** The System › Config groups whose effect this tab judges ("N knobs changed · Edit"). */
const RECORDS_CONFIG_GROUPS: readonly GroupId[] = ["scenes", "lenses"];

type ConfigPayload = { config?: Record<string, number | string> };

function GenerationKnobs({ config }: { config: ConfigPayload | null | undefined }) {
  const c = config?.config ?? {};
  const n = (k: string) => (typeof c[k] === "number" ? (c[k] as number) : null);
  const facts = [
    n("n_train") != null && { label: "Train scenes", value: formatCount(n("n_train")) },
    n("n_valid") != null && { label: "Validate scenes", value: formatCount(n("n_valid")) },
    n("n_test") != null && { label: "Test scenes", value: formatCount(n("n_test")) },
    n("hr_image_size") != null && { label: "HR scene side", value: formatCount(n("hr_image_size")), unit: "px" },
    n("galaxy_density_arcmin2") != null && { label: "Galaxy density", value: formatNumber(n("galaxy_density_arcmin2"), { sig: 3 }),
      unit: "arcmin⁻²", hint: "Set by the active galaxy model when it is activated" },
    n("star_density_arcmin2") != null && { label: "Star density", value: "set by the active stellar prior",
      hint: "synthetic_generate overrides the configured star density with the active stellar prior's" },
    n("lens_density_arcmin2") != null && { label: "Lens density", value: formatNumber(n("lens_density_arcmin2"), { sig: 2 }), unit: "arcmin⁻²" },
    n("psf_warp_prob") != null && { label: "PSF warp probability", value: formatNumber(n("psf_warp_prob"), { sig: 2 }) },
    n("saturation_mask_prob") != null && { label: "Dark-core probability", value: formatNumber(n("saturation_mask_prob"), { sig: 2 }),
      hint: "Base chance an above-well core is blacked out; it ramps up with peak ÷ well" },
  ];
  return (
    <div className="rl-stack">
      <FactsList title="Generation knobs" facts={facts} />
      <div className="rl-row"><EditInConfig /></div>
    </div>
  );
}

/* ── the tab ───────────────────────────────────────────────────────────── */

export default function Records() {
  const [rawSplit, setSplit] = useUrlState("split", "test");
  const split: Split = isSplit(rawSplit) ? rawSplit : "test";
  const sync = useJob("data:sky-sync");
  const status = useSrStatus(sync.busy ? 3_000 : undefined);
  const s = status.data;
  const config = useResource<ConfigPayload>(URLS.config, [], { ttl: 5 * 60_000 });
  const { online } = useFasrcOnline();
  const [viewerKey, setViewerKey] = useState(0);
  // Primitives from the viewer's state (a pan does not re-render the page).
  const [index, setIndex] = useState<number | null>(null);
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
  const [, setGen] = useUrlState("gen", false);
  const openGeneration = () => {
    setGen(true);
    requestAnimationFrame(() => document.getElementById("syn-drawer-gen")?.scrollIntoView({ block: "start" }));
  };

  usePageActions([
    ...LOCAL_SPLITS.map((sp) => ({ id: `records-split-${sp}`, label: `Records: show the ${sp} split`, group: "Records", run: () => setSplit(sp) })),
    { id: "records-generate", label: "Generate and sync the synthetic records…", group: "Records",
      keywords: ["synthetic_generate", "regenerate", "splits", "tfrecord", "rsync", "pull"], run: openGeneration },
    { id: "records-refresh", label: "Refresh the records status", group: "Records", run: () => { void status.reload(); bump(); } },
  ]);

  const splitInfo = s?.splits[split];
  // The viewer mounts once the (fast, local) status says the split is here —
  // or the status failed: a viewer opened on an absent split would settle on
  // its lone disabled tier and write that into the URL (v.rec.t) on a visit.
  const absent = !!s && !splitInfo?.present;
  const viewable = s ? !absent : !!status.error;
  const onViewerState = (st: ViewerState) => { setIndex(st.index); };
  const trainCount = s?.splits.train?.count ?? 0;
  const lensPrior = typeof config.data?.config?.lens_density_arcmin2 === "number" ? config.data.config.lens_density_arcmin2 as number : null;
  const splitOptions: { value: Split; label: ReactNode; disabled?: boolean; title?: string }[] = [
    ...LOCAL_SPLITS.map((sp) => ({
      value: sp as Split, label: <>{sp} <span className="muted">{s ? formatCount(s.splits[sp]?.count ?? 0) : ""}</span></>,
    })),
    { value: "train", label: "train", disabled: true,
      title: trainCount ? `${formatCount(trainCount)} train records here; browse test or validate` : "The train split is generated and read on FASRC; it is not synced here" },
  ];
  return (
    <Page className="dt-page dt-page--image">
      <DataBar label="Records" compactable={3}>
        <Segmented size="sm" value={split} onChange={(v) => { if (isSplit(v)) setSplit(v); }} aria-label="Split" options={splitOptions} />
        {viewable && !absent && <TruthControls truth={truth} overlay={overlay} onOverlay={setOverlay} index={index} split={split} />}
        <Spacer />
        {s && <div className="dt-bar__status" role="group" aria-label={`Problems of the ${split} split`}><FilesProblem status={s} split={split} /></div>}
        <BarActions>
          <Info label="About the tiers">
            <p><b>LR</b>: the dirty synthetic Euclid image the model receives (0.1″, detector noise, warped PSFs).</p>
            <p><b>HR</b>: the truth at 0.05″, drawn at the LR's surface brightness so both tiers read alike.</p>
            <p><b>Blurred HR</b>: the HR truth convolved to the LR PSF (what a perfect LR would show).</p>
            <p><b>Clean</b>: the starless target; <b>SR</b>: the production model's output (generated on Models › Images).</p>
          </Info>
          <DrawerButton flag="gen" icon="server" hint="The synthetic_generate step, the sync from FASRC and the generation knobs">Generate and sync</DrawerButton>
        </BarActions>
      </DataBar>
      {status.error && !s && (
        <Callout tone="bad" title="Records status did not load" action={<Button size="sm" onClick={() => void status.reload()}>Retry</Button>}>
          <span className="dt-pre">{status.error.message}</span>
        </Callout>
      )}
      {absent ? (
        <EmptyState icon="database" title={`No ${split} records on this machine`}
          action={<Button variant="primary" icon="download" onClick={openGeneration}>Generate and sync</Button>}>
          {online ? "Sync the split from FASRC (a background job)." : OFFLINE_HINT}
        </EmptyState>
      ) : viewable && (
        <div className="dt-figure">
          <ImageViewer key={`${split}-${viewerKey}`} collection="sky" params={params} urlKey="rec"
            tiers={DEFAULT_TIERS} initialId={index != null ? recordObjectId(split, index) : undefined}
            markers={markers} display={MATCHED}
            onReady={(a) => { api.current = a; }}
            onState={onViewerState} />
          {s && (() => {
            const sr = recordSrLink(split, index, s.sr?.[split] ?? 0);
            // Under the viewer (not in the bar, which stays one row): where the
            // record's SR is, and the generation knobs that differ from their defaults.
            return (
              <Caption className="dt-caption">
                <Link to={sr.to}>{sr.label}</Link>
                <ConfigKnobsLink groups={RECORDS_CONFIG_GROUPS} className="syn-config-link" />
              </Caption>
            );
          })()}
        </div>
      )}
      <JobStrip job={sync} />
      {viewable && <SourcesTable truth={truth} split={split} index={index} />}
      {s && !absent && <Census split={split} index={index} lensPrior={lensPrior} onGo={(i) => api.current?.goTo(i)} />}
      <Drawer flag="gen" title="Generate and sync" sub="synthetic_generate on FASRC, the local copy, the knobs">
        <GeneratePanel />
        <div className="rl-row">
          <SyncPopover busy={sync.busy} online={online} onStart={runSync} />
          <span className="rl-faint">Pulls the chosen splits' records and truth catalogues to this machine.</span>
        </div>
        <GenerationKnobs config={config.data} />
      </Drawer>
    </Page>
  );
}
