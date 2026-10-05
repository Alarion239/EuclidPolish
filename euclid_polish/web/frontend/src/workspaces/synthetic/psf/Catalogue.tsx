/* Synthetic › PSF › catalogue: the real Euclid Q1 stars the empirical PSFs
   are built from, over the FASRC-mirror stars.csv (43k stars; never the
   stale 200-row local copy; works offline).

   The filter row (deep field with its counts, cutout coverage, a band's
   status, the VIS range; "N of M shown" only while filtered), the star
   table (search, sort, CSV; a row opens the `star` inspector; the selection
   is the shared `star` selection; each row links to the atlas), then the
   magnitude histogram (all / the usable ones / the filtered ones) with the
   caption that explains its edges (the euclid_query windows). The catalogue
   mirror's freshness shows only when it is stale. Every filter is in the
   URL. */
import { useMemo, useState } from "react";
import { usePageActions } from "../../../app/palette";
import Plot from "../../../charts/Plot";
import { C } from "../../../colors";
import { useStepsStatus } from "../../../fasrc";
import { formatCount, formatDeg, formatNumber } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { useSelected, useSelection } from "../../../state/selection";
import { extent, linearTicks, magnitudeTicks } from "../../../ticks";
import {
  Badge, Button, Caption, DataTable, EmptyState, RangeSlider, Select, Switch, Toolbar, ToolbarGroup, ToolbarSpacer,
  ToolbarText, type DataColumn,
} from "../../../ui";
import type { StarsPayload } from "../dataApi";
import { Freshness, SkyButton } from "../dataCommon";
import {
  BANDS, bandShort, bandState, decodeStars, fieldCounts, filterStars, histogram, magBins, parseRange, serializeRange,
  type BandState, type CutoutFilter, type Star, type StarFilter,
} from "../dataModel";
import { magnitudeWindowCaption } from "./psfModel";

const FIELDS = ["all", "EDF-N", "EDF-S", "EDF-F", "none"] as const;
const CUTOUTS: { value: CutoutFilter; label: string }[] = [
  { value: "any", label: "Any cutouts" },
  { value: "nav", label: "Usable (all 4 bands)" },
  { value: "all4", label: "Valid in all 4 bands" },
  { value: "some", label: "Valid in 1–3 bands" },
  { value: "none", label: "No valid cutout" },
];
const STATES: (BandState | "any")[] = ["any", "valid", "corrupted", "failed", "pending"];
const STALE_S = 3 * 24 * 3600;

function BandDots({ star }: { star: Star }) {
  return (
    <span className="dt-dots" role="img"
      aria-label={BANDS.map((b) => `${bandShort(b)} ${bandState(star.bands[b])}`).join(", ")}>
      {BANDS.map((b) => {
        const band = star.bands[b];
        const st = bandState(band);
        return <span key={b} className="dt-dot" data-state={st}
          title={`${bandShort(b)}: ${st}${band?.sizes.length ? ` (${band.sizes.join(", ")} px)` : ""}`} />;
      })}
    </span>
  );
}

/* Widths sum to ~698 px with the select column: the table fits a ~720 px pane
   without clipping the coordinates or the Sky button. */
const COLUMNS: DataColumn<Star>[] = [
  { id: "id", header: "Star", numeric: true, width: 68 },
  { id: "field", header: "Field", width: 64, accessor: (s) => s.field || "—" },
  { id: "ra", header: "RA", numeric: true, width: 100, cell: (s) => <span className="mono">{formatDeg(s.ra, 5)}</span> },
  { id: "dec", header: "Dec", numeric: true, width: 100, cell: (s) => <span className="mono">{formatDeg(s.dec, 5, { signed: true })}</span> },
  { id: "mag", header: "VIS mag", numeric: true, width: 86, cell: (s) => formatNumber(s.mag, { digits: 2 }) },
  { id: "flux", header: "Flux (µJy)", numeric: true, width: 90, hidden: true, cell: (s) => formatNumber(s.flux, { digits: 1 }) },
  { id: "fluxErr", header: "σ (µJy)", numeric: true, width: 76, hidden: true, cell: (s) => formatNumber(s.fluxErr, { digits: 3 }) },
  { id: "bands", header: "Bands", headerText: "Bands (VIS Y J H)", width: 88, accessor: (s) => s.nValid,
    filterText: (s) => BANDS.map((b) => `${bandShort(b)}:${bandState(s.bands[b])}`).join(" "),
    csv: (s) => BANDS.map((b) => `${b}=${bandState(s.bands[b])}`).join(" "), cell: (s) => <BandDots star={s} /> },
  { id: "nav", header: "Usable", headerText: "Usable (valid in all 4 bands at one size)", width: 84, accessor: (s) => (s.nav ? "yes" : ""),
    cell: (s) => (s.nav ? <Badge size="sm" tone="good">yes</Badge> : "") },
  { id: "sky", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 72,
    cell: (s) => <SkyButton ra={s.ra} dec={s.dec} fov={0.02} layers={["stars", "q1-tiles:0.3"]} label="Sky" /> },
];

function MagHistogram({ all, shown, usable, filtered, caption }: {
  all: Star[]; shown: Star[]; usable: Star[]; filtered: boolean; caption: string;
}) {
  const [logY, setLogY] = useUrlState("hlog", false);
  // extent(), never Math.min(...mags): an argument spread overflows the stack past ~120k stars.
  const range = extent(all.map((s) => s.mag));
  if (!range) return <EmptyState compact icon="activity" title="No magnitudes" />;
  const { lo, hi, bins } = magBins(range[0], range[1]);
  const hAll = histogram(all.map((s) => s.mag), lo, hi, bins);
  const hShown = histogram(shown.map((s) => s.mag), lo, hi, bins);
  const hUsable = histogram(usable.map((s) => s.mag), lo, hi, bins);
  const top = Math.max(1, ...hAll.counts);
  const floor = logY ? 0.8 : 0;
  const gap = (c: number) => (logY && c === 0 ? null : c);
  const series = [
    { x: hAll.centers, y: hAll.counts.map(gap), color: C.muted, mode: "histogram" as const, name: `all stars · ${formatCount(all.length)}`, fillAlpha: 0.25 },
    { x: hUsable.centers, y: hUsable.counts.map(gap), color: C.comb, mode: "histogram" as const, name: `usable · ${formatCount(usable.length)}`, fillAlpha: 0.3 },
    ...(filtered ? [{ x: hShown.centers, y: hShown.counts.map(gap), color: C.mean, mode: "histogram" as const, name: `filtered · ${formatCount(shown.length)}`, fillAlpha: 0.35 }] : []),
  ];
  return (
    <figure className="rl-fig">
      <figcaption className="rl-fig__head">
        <strong>Magnitude distribution</strong>
        <Switch size="sm" checked={logY} onChange={setLogY}>Log counts</Switch>
      </figcaption>
      <Plot xDomain={[lo, hi]} yDomain={[floor, top * (logY ? 1.6 : 1.08)]} yScale={logY ? "log" : "linear"}
        xTicks={magnitudeTicks([lo, hi])} yTicks={logY ? undefined : linearTicks([0, top])}
        xLabel="VIS magnitude (AB)" yLabel="stars per 0.05 mag" series={series} aspect={0.34}
        legend="auto" exportName="star-magnitudes" aria-label="Magnitude distribution of the star catalogue"
        xFormat={(v) => v.toFixed(2)} />
      <Caption>{caption}</Caption>
    </figure>
  );
}

export function Catalogue({ data }: { data: StarsPayload }) {
  const all = useMemo(() => decodeStars(data), [data]);
  const summary = data.summary;
  const steps = useStepsStatus();
  const last = steps.data?.steps?.find((s) => s.step_id === "euclid_query")?.last_params ?? null;
  const [field, setField] = useUrlState("field", "all");
  const [cutouts, setCutouts] = useUrlState("cut", "any");
  const [band, setBand] = useUrlState("band", "any");
  const [bstate, setBstate] = useUrlState("bst", "any");
  const [magRaw, setMagRaw] = useUrlState("mag", "");
  const mag = parseRange(magRaw);
  const filter: StarFilter = {
    field, cutouts: (CUTOUTS.some((c) => c.value === cutouts) ? cutouts : "any") as CutoutFilter,
    band: band === "any" || (BANDS as readonly string[]).includes(band) ? band : "any",
    bandState: (STATES.includes(bstate as BandState) ? bstate : "any") as BandState | "any", mag,
  };
  const shown = useMemo(() => filterStars(all, filter),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [all, field, cutouts, band, bstate, magRaw]);
  const usable = useMemo(() => all.filter((s) => s.nav), [all]);
  const counts = useMemo(() => fieldCounts(all), [all]);
  const selectedAll = useSelected("star");
  const shownKeys = useMemo(() => new Set(shown.map((s) => String(s.id))), [shown]);
  const selected = useMemo(() => selectedAll.filter((k) => shownKeys.has(k)), [selectedAll, shownKeys]);
  const magLo = summary?.mag_min ?? 16, magHi = summary?.mag_max ?? 19;
  const [magDraft, setMagDraft] = useState<[number, number] | null>(null);
  const magValue: [number, number] = magDraft ?? mag ?? [Math.floor(magLo * 10) / 10, Math.ceil(magHi * 10) / 10];
  const filtered = shown.length !== all.length;
  const stale = data.mtime != null && Date.now() / 1000 - data.mtime > STALE_S;
  const clear = () => { setField("all"); setCutouts("any"); setBand("any"); setBstate("any"); setMagRaw(""); setMagDraft(null); };

  usePageActions([
    { id: "catalog-usable", label: "PSF catalogue: the usable stars (valid in all 4 bands)", group: "PSF", run: () => setCutouts("nav") },
    { id: "catalog-none", label: "PSF catalogue: stars without any valid cutout", group: "PSF", run: () => setCutouts("none") },
    { id: "catalog-clear", label: "PSF catalogue: clear the filters", group: "PSF", disabled: !filtered, run: clear },
  ]);

  return (
    <div className="rl-stack">
      <Toolbar label="Catalogue filters">
        <Select size="sm" value={FIELDS.includes(field as (typeof FIELDS)[number]) ? field : "all"} onChange={setField} aria-label="Deep field"
          options={FIELDS.map((f) => ({
            value: f,
            label: f === "all" ? `All fields (${formatCount(all.length)})` : `${f === "none" ? "Outside the deep fields" : f} (${formatCount(counts[f] ?? 0)})`,
            disabled: f !== "all" && !counts[f],
          }))} />
        <Select size="sm" value={filter.cutouts} onChange={setCutouts} aria-label="Cutout coverage" options={CUTOUTS} />
        <ToolbarGroup label="Band status" hideLabel>
          <Select size="sm" value={filter.band} onChange={setBand} aria-label="Band"
            options={[{ value: "any", label: "Best band" }, ...BANDS.map((b) => ({ value: b, label: `${bandShort(b)} band` }))]} />
          <Select size="sm" value={filter.bandState} onChange={setBstate} aria-label="Band status"
            options={STATES.map((s) => ({ value: s, label: s === "any" ? "Any status" : s[0].toUpperCase() + s.slice(1) }))} />
        </ToolbarGroup>
        <ToolbarGroup label="VIS mag">
          <RangeSlider value={magValue} min={Math.floor(magLo * 10) / 10} max={Math.ceil(magHi * 10) / 10} step={0.05}
            showValue format={(v) => v.toFixed(2)} aria-label="VIS magnitude"
            onChange={setMagDraft} onCommit={(v) => { setMagDraft(null); setMagRaw(serializeRange(v)); }} />
        </ToolbarGroup>
        {filtered && <ToolbarText>{`${formatCount(shown.length)} of ${formatCount(all.length)} shown`}</ToolbarText>}
        {filtered && <Button size="sm" variant="ghost" icon="reset" onClick={clear}>Clear</Button>}
        <ToolbarSpacer />
        {stale && <Freshness at={data.mtime} label="catalogue synced" stale={STALE_S} />}
        <SkyButton layers={["stars", "q1-tiles:0.3"]} label="Stars on sky" />
      </Toolbar>
      <DataTable rows={shown} columns={COLUMNS} rowKey={(s) => String(s.id)} urlKey="cat" height={480}
        aria-label="Stars" exportName="stars" filterPlaceholder="Filter stars (e.g. mag<17 field:EDF-S)"
        inspect={(s) => ({ kind: "star", id: String(s.id) })}
        selectable selected={selected}
        onSelectedChange={(keys) => {
          const hidden = selectedAll.filter((k) => !shownKeys.has(k));
          useSelection.getState().select("star", [...hidden, ...keys]);
        }}
        toolbar={selected.length > 0 && (
          <Button size="sm" variant="ghost" onClick={() => useSelection.getState().clear("star")}>
            Clear {selected.length}
          </Button>
        )}
        empty={all.length ? "No star matches the filters" : "The catalogue is empty"} />
      <MagHistogram all={all} shown={shown} usable={usable} filtered={filtered} caption={magnitudeWindowCaption(last)} />
    </div>
  );
}
