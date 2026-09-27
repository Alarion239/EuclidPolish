/* Data › Catalog (spec §8.4): the star catalogue explorer over the FASRC-mirror
 * stars.csv (43k stars; never the stale 200-row local copy).
 *
 * Toolbar filters (deep field, cutout coverage, a band's status, magnitude
 * range) + the mirror's freshness and an explicit "refresh from FASRC". Then
 * one summary line (the headline counts in words, the ones that need it
 * explained in their tips; the navigator count links to Cutouts), the
 * DataTable of the filtered stars (search, sort, CSV; row → the `star`
 * inspector; multi-select → the shared `star` selection; each row links to
 * the atlas), then the magnitude distribution (all / filtered / navigator,
 * Plot v2) and the per-band cutout validity with the states explained. The
 * euclid_query / verify-photometry FASRC steps sit at the bottom (collapsed).
 * Every filter is in the URL. */
import { useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { apiPost, isFasrcOffline } from "../../../api/client";
import { invalidate } from "../../../api/query";
import { usePageActions } from "../../../app/palette";
import Plot from "../../../charts/Plot";
import { C } from "../../../colors";
import { StepById } from "../../../fasrc";
import { formatCount, formatDeg, formatNumber, formatPercent } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { useSelected, useSelection } from "../../../state/selection";
import { extent, linearTicks, magnitudeTicks } from "../../../ticks";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, DataTable, EmptyState, Page, Popover, RangeSlider,
  Section, Select, Switch, Tooltip, toast, type DataColumn,
} from "../../../ui";
import { URLS, useStars, type StarsPayload } from "../api";
import { BarGroup, DataBar, Freshness, LoadState, OFFLINE_HINT, SkyButton, Spacer, useFasrcOnline } from "../common";
import {
  BAND_STATE_HELP, BANDS, bandShort, bandState, decodeStars, fieldCounts, filterStars, histogram, magBins,
  parseRange, serializeRange, type BandState, type CutoutFilter, type Star, type StarFilter,
} from "../model";
import "../register";
import "../data.css";

const FIELDS = ["all", "EDF-N", "EDF-S", "EDF-F", "none"] as const;
const CUTOUTS: { value: CutoutFilter; label: string }[] = [
  { value: "any", label: "Any cutouts" },
  { value: "nav", label: "In the navigator" },
  { value: "all4", label: "Valid in all 4 bands" },
  { value: "some", label: "Valid in 1–3 bands" },
  { value: "none", label: "No valid cutout" },
];
const STATES: (BandState | "any")[] = ["any", "valid", "corrupted", "failed", "pending"];

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

/* Widths sum to ~714 px with the select column: the table fits a ~720 px pane
   without clipping the coordinates or the Sky button. */
const COLUMNS: DataColumn<Star>[] = [
  { id: "id", header: "Star", numeric: true, width: 68 },
  { id: "field", header: "Field", width: 64, accessor: (s) => s.field || "—" },
  { id: "ra", header: "RA", numeric: true, width: 100, cell: (s) => <span className="mono">{formatDeg(s.ra, 5)}</span> },
  { id: "dec", header: "Dec", numeric: true, width: 100, cell: (s) => <span className="mono">{formatDeg(s.dec, 5, { signed: true })}</span> },
  { id: "mag", header: "VIS mag", numeric: true, width: 86, cell: (s) => formatNumber(s.mag, { digits: 3 }) },
  { id: "flux", header: "Flux µJy", numeric: true, width: 84, hidden: true, cell: (s) => formatNumber(s.flux, { digits: 1 }) },
  { id: "fluxErr", header: "σ µJy", numeric: true, width: 70, hidden: true, cell: (s) => formatNumber(s.fluxErr, { digits: 3 }) },
  { id: "bands", header: "Bands", headerText: "Bands (VIS Y J H)", width: 88, accessor: (s) => s.nValid,
    filterText: (s) => BANDS.map((b) => `${bandShort(b)}:${bandState(s.bands[b])}`).join(" "),
    csv: (s) => BANDS.map((b) => `${b}=${bandState(s.bands[b])}`).join(" "), cell: (s) => <BandDots star={s} /> },
  { id: "nav", header: "Navigator", width: 100, accessor: (s) => (s.nav ? "yes" : ""),
    cell: (s) => (s.nav ? <Badge size="sm" tone="good">yes</Badge> : "") },
  { id: "sky", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 72,
    cell: (s) => <SkyButton ra={s.ra} dec={s.dec} fov={0.02} layers={["stars", "q1-tiles:0.3"]} label="Sky" /> },
];

function BandTable({ data, total }: { data: StarsPayload; total: number }) {
  const meter = (n: number, tone: string) => (
    <span className="dt-meter">
      {formatCount(n)}
      <span className="dt-meter__bar" aria-hidden="true">
        <span className="dt-meter__fill" data-tone={tone} style={{ width: `${total ? (100 * n) / total : 0}%` }} />
      </span>
    </span>
  );
  return (
    <div className="dt-bands-wrap">
      <table className="dt-bands" aria-label="Cutout validity per band">
        <thead>
          <tr>
            <th scope="col">Band</th>
            {(["valid", "corrupted", "failed", "pending"] as BandState[]).map((st) => (
              <th key={st} scope="col"><Tooltip content={BAND_STATE_HELP[st]}><span tabIndex={0}>{st}</span></Tooltip></th>
            ))}
            {data.sizes.map((sz) => <th key={sz} scope="col">valid @ {sz} px</th>)}
          </tr>
        </thead>
        <tbody>
          {data.band_stats.map((b) => (
            <tr key={b.band}>
              <th scope="row">{bandShort(b.band)}</th>
              <td>{meter(b.valid, "good")}</td>
              <td>{meter(b.corrupted, "warn")}</td>
              <td>{meter(b.failed, "bad")}</td>
              <td>{meter(b.pending, "neutral")}</td>
              {data.sizes.map((sz) => <td key={sz}>{formatCount(b.by_size[String(sz)] ?? 0)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** One summary figure: the value in tabular figures, then what it counts;
 *  a definition, when it needs one, in its tip. */
function Fig({ value, children, hint, tone }: { value: ReactNode; children: ReactNode; hint?: string; tone?: "good" | "warn" | "bad" }) {
  const body = <><b className="dt-summary__v" data-tone={tone}>{value}</b> {children}</>;
  return hint
    ? <Tooltip content={hint}><span tabIndex={0} className="dt-summary__item dt-summary__item--hint">{body}</span></Tooltip>
    : <span className="dt-summary__item">{body}</span>;
}

type Summary = NonNullable<StarsPayload["summary"]>;

/** The headline counts as one quiet line of words (it wraps in a narrow pane). */
function SummaryLine({ summary, shown, filtered }: { summary: Summary; shown: number; filtered: boolean }) {
  const nav = summary.navigator;
  return (
    <p className="dt-summary" aria-label="Catalogue summary">
      <Fig value={formatCount(summary.total)}>stars{filtered ? <>, <b className="dt-summary__v">{formatCount(shown)}</b> shown</> : null}</Fig>
      <Fig value={formatCount(summary.valid)} tone="good" hint={BAND_STATE_HELP.valid}>valid in a band</Fig>
      <Fig value={formatCount(summary.valid_all4)}>valid in all 4 ({formatPercent(summary.valid_all4 / Math.max(1, summary.total))})</Fig>
      <Tooltip content="Valid in all four bands at one common cutout size (the size with the most such stars): what Data › Cutouts browses.">
        <Link to="/data/cutouts" className="dt-summary__item dt-summary__link">
          <b className="dt-summary__v">{formatCount(nav.count)}</b> in the navigator{nav.size ? ` at ${nav.size} px` : ""}
        </Link>
      </Tooltip>
      <Fig value={formatCount(summary.corrupted)} tone={summary.corrupted ? "warn" : undefined}
        hint="No valid band; at least one band's cutout was downloaded but rejected.">corrupted</Fig>
      <Fig value={formatCount(summary.failed)} tone={summary.failed ? "bad" : undefined}
        hint="No valid or rejected band; at least one band's download failed.">failed</Fig>
      <Fig value={formatCount(summary.pending)} hint="No band ever attempted.">pending</Fig>
      <Fig value={`${formatNumber(summary.mag_min, { digits: 2 })}–${formatNumber(summary.mag_max, { digits: 2 })}`}>VIS mag</Fig>
    </p>
  );
}

function MagHistogram({ all, shown, nav }: { all: Star[]; shown: Star[]; nav: Star[] }) {
  const [logY, setLogY] = useUrlState("hlog", false);
  // extent(), never Math.min(...mags): an argument spread overflows the stack past ~120k stars.
  const range = extent(all.map((s) => s.mag));
  if (!range) return <EmptyState compact icon="activity" title="No magnitudes" />;
  const { lo, hi, bins } = magBins(range[0], range[1]);
  const hAll = histogram(all.map((s) => s.mag), lo, hi, bins);
  const hShown = histogram(shown.map((s) => s.mag), lo, hi, bins);
  const hNav = histogram(nav.map((s) => s.mag), lo, hi, bins);
  const top = Math.max(1, ...hAll.counts);
  const floor = logY ? 0.8 : 0;
  const series = [
    { x: hAll.centers, y: hAll.counts.map((c) => (logY && c === 0 ? null : c)), color: C.muted, mode: "histogram" as const, name: "all stars", fillAlpha: 0.25 },
    { x: hNav.centers, y: hNav.counts.map((c) => (logY && c === 0 ? null : c)), color: C.comb, mode: "histogram" as const, name: "navigator", fillAlpha: 0.3 },
    { x: hShown.centers, y: hShown.counts.map((c) => (logY && c === 0 ? null : c)), color: C.mean, mode: "histogram" as const, name: "filtered", fillAlpha: 0.35 },
  ];
  return (
    <>
      <Plot xDomain={[lo, hi]} yDomain={[floor, top * (logY ? 1.6 : 1.08)]} yScale={logY ? "log" : "linear"}
        xTicks={magnitudeTicks([lo, hi])} yTicks={logY ? undefined : linearTicks([0, top])}
        xLabel="VIS magnitude (AB)" yLabel="stars per 0.05 mag" series={series} height={230}
        legend="auto" exportName="star-magnitudes" aria-label="Magnitude distribution of the star catalogue"
        xFormat={(v) => v.toFixed(2)} />
      <Switch size="sm" checked={logY} onChange={setLogY}>Log counts</Switch>
    </>
  );
}

export default function Catalog() {
  const stars = useStars();
  const all = useMemo(() => decodeStars(stars.data), [stars.data]);
  const summary = stars.data?.summary ?? null;
  const [field, setField] = useUrlState("field", "all");
  const [cutouts, setCutouts] = useUrlState("cut", "any");
  const [band, setBand] = useUrlState("band", "any");
  const [bstate, setBstate] = useUrlState("bst", "any");
  const [magRaw, setMagRaw] = useUrlState("mag", "");
  const [showSteps, setShowSteps] = useUrlState("steps", false);
  const mag = parseRange(magRaw);
  const filter: StarFilter = {
    field, cutouts: (CUTOUTS.some((c) => c.value === cutouts) ? cutouts : "any") as CutoutFilter,
    band, bandState: (STATES.includes(bstate as BandState) ? bstate : "any") as BandState | "any", mag,
  };
  const shown = useMemo(() => filterStars(all, filter),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [all, field, cutouts, band, bstate, magRaw]);
  const nav = useMemo(() => all.filter((s) => s.nav), [all]);
  const counts = useMemo(() => fieldCounts(all), [all]);
  const { online } = useFasrcOnline();
  const [refreshing, setRefreshing] = useState(false);
  const selectedAll = useSelected("star");
  const shownKeys = useMemo(() => new Set(shown.map((s) => String(s.id))), [shown]);
  const selected = useMemo(() => selectedAll.filter((k) => shownKeys.has(k)), [selectedAll, shownKeys]);
  const magLo = summary?.mag_min ?? 16, magHi = summary?.mag_max ?? 19;
  const [magDraft, setMagDraft] = useState<[number, number] | null>(null);
  const magValue: [number, number] = magDraft ?? mag ?? [Math.floor(magLo * 10) / 10, Math.ceil(magHi * 10) / 10];
  const filtered = shown.length !== all.length;

  async function refresh() {
    setRefreshing(true);
    try {
      await apiPost(URLS.refreshCatalog, {});
      await Promise.all([invalidate("/api/catalog/"), invalidate("/api/star-cutouts/"), invalidate("/api/status")]);
      toast.success("Catalogue pulled from FASRC");
    } catch (e) {
      toast.error(isFasrcOffline(e) ? "FASRC not connected" : "Catalogue pull failed",
        { description: e instanceof Error ? e.message : String(e) });
    } finally { setRefreshing(false); }
  }
  const clear = () => { setField("all"); setCutouts("any"); setBand("any"); setBstate("any"); setMagRaw(""); setMagDraft(null); };

  usePageActions([
    { id: "catalog-refresh", label: "Pull the star catalogue from FASRC", group: "Catalog", disabled: !online || refreshing,
      keywords: ["stars.csv", "rsync"], run: () => { void refresh(); } },
    { id: "catalog-nav", label: "Catalog: stars in the cutouts navigator", group: "Catalog", run: () => setCutouts("nav") },
    { id: "catalog-none", label: "Catalog: stars without any valid cutout", group: "Catalog", run: () => setCutouts("none") },
    { id: "catalog-clear", label: "Catalog: clear the filters", group: "Catalog", disabled: !filtered, run: clear },
    { id: "catalog-build", label: "Build the catalogue (euclid_query)…", group: "Catalog", keywords: ["query", "brightest"],
      run: () => setShowSteps(true) },
  ]);

  return (
    <Page className="dt-page">
      <DataBar label="Catalogue filters">
        <Select size="sm" value={FIELDS.includes(field as (typeof FIELDS)[number]) ? field : "all"} onChange={setField} aria-label="Deep field"
          options={FIELDS.map((f) => ({
            value: f,
            label: f === "all" ? "All fields" : `${f === "none" ? "Outside the deep fields" : f} (${formatCount(counts[f] ?? 0)})`,
            disabled: f !== "all" && !counts[f],
          }))} />
        <Select size="sm" value={filter.cutouts} onChange={setCutouts} aria-label="Cutout coverage" options={CUTOUTS} />
        <BarGroup label="Band status">
          <Select size="sm" value={band} onChange={setBand} aria-label="Band"
            options={[{ value: "any", label: "Best band" }, ...BANDS.map((b) => ({ value: b, label: `${bandShort(b)} band` }))]} />
          <Select size="sm" value={filter.bandState} onChange={setBstate} aria-label="Band status"
            options={STATES.map((s) => ({ value: s, label: s === "any" ? "Any status" : s[0].toUpperCase() + s.slice(1) }))} />
        </BarGroup>
        <BarGroup label="Magnitude">
          <span className="dt-bar__label">VIS mag</span>
          <RangeSlider value={magValue} min={Math.floor(magLo * 10) / 10} max={Math.ceil(magHi * 10) / 10} step={0.05}
            showValue format={(v) => v.toFixed(2)} aria-label="VIS magnitude"
            onChange={setMagDraft} onCommit={(v) => { setMagDraft(null); setMagRaw(serializeRange(v)); }} />
        </BarGroup>
        {filtered && <Button size="sm" variant="ghost" icon="reset" onClick={clear}>Clear</Button>}
        <Spacer />
        {stars.data?.present && <Freshness at={stars.data.mtime} label="mirror synced" stale={3 * 24 * 3600} />}
        <Tooltip content={online ? "Re-pull stars.csv from FASRC (netscratch)" : OFFLINE_HINT}>
          <span>
            <Button size="sm" icon="download" loading={refreshing} disabled={!online} onClick={() => void refresh()}>Refresh</Button>
          </span>
        </Tooltip>
        <SkyButton layers={["stars", "q1-tiles:0.3"]} label="Stars on sky" />
      </DataBar>

      <LoadState loading={stars.loading && !stars.data} error={stars.error} onRetry={() => void stars.reload()} lines={6}>
        {!stars.data?.present ? (
          <EmptyState icon="database" title="The FASRC star catalogue is not synchronised"
            action={<Button variant="primary" icon="download" disabled={!online} onClick={() => void refresh()}>Pull from FASRC</Button>}>
            {online ? "Pull stars.csv, or build it with the euclid_query step below." : OFFLINE_HINT}
          </EmptyState>
        ) : (
          <>
            {summary && <SummaryLine summary={summary} shown={shown.length} filtered={filtered} />}
            <DataTable rows={shown} columns={COLUMNS} rowKey={(s) => String(s.id)} urlKey="cat" height={520}
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
            <div className="dt-two">
              <Card>
                <CardHead title="Magnitude distribution" sub="All stars, the navigator's and the filtered ones" />
                <CardBody><MagHistogram all={all} shown={shown} nav={nav} /></CardBody>
              </Card>
              <Card>
                <CardHead title="Cutout validity per band"
                  right={(
                    <Popover label="What the states mean" width={340}
                      trigger={<Button size="sm" variant="ghost" icon="help">States</Button>}>
                      <div className="dt-explain">
                        {(["valid", "corrupted", "failed", "pending"] as BandState[]).map((st) => (
                          <div key={st}><Badge size="sm" tone={st === "valid" ? "good" : st === "corrupted" ? "warn" : st === "failed" ? "bad" : "neutral"}>{st}</Badge> {BAND_STATE_HELP[st]}</div>
                        ))}
                        <div className="muted">Each row sums to the star count: a band counts once, under its best outcome (valid › corrupted › failed › pending). The summary and the “Best band” filter use a star’s best band.</div>
                      </div>
                    </Popover>
                  )} />
                <CardBody>{stars.data && <BandTable data={stars.data} total={summary?.total ?? 0} />}</CardBody>
              </Card>
            </div>
          </>
        )}
      </LoadState>

      <Section title="Build and verify the catalogue (FASRC)" collapsible open={showSteps} onOpenChange={setShowSteps}>
        <div className="dt-steps">
          <StepById stepId="euclid_query" />
          <StepById stepId="euclid_verify_photometry" />
        </div>
        {!online && <Callout tone="info" title="FASRC offline">{OFFLINE_HINT}</Callout>}
      </Section>
    </Page>
  );
}
