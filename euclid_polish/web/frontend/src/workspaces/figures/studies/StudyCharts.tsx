/* The six charts of a study, each drawn from the backend's chart table
 * (`…/figure/<chart>.csv`, the rows its publication figure draws) and each
 * with its exports (PDF · PNG · SVG at the chosen dpi, CSV) and "Log to
 * notebook" citing the study id and the manifest hash. A chart whose data
 * the study lacks says what is missing (the backend's words). */
import { useMemo, type ReactNode } from "react";
import Plot, { Legend, useLegend, type Guide, type LegendItem, type Series } from "../../../charts/Plot";
import { C, LOSS_COLOR, bandColor, categorical } from "../../../colors";
import { formatCount, formatDate, formatPercent } from "../../../format";
import { logTicks, paddedDomain } from "../../../ticks";
import {
  Button, Callout, Caption, Segmented, Select, Skeleton, Table, Toolbar, ToolbarGroup, ToolbarSpacer, type Column,
} from "../../../ui";
import { LogToNotebookButton } from "../../shared/LogToNotebook";
import { useStudyCsv, type StudyDetail, type StudyMember } from "./api";
import {
  CHART_TITLE, GROUP_LABEL, csvUrl, deltaText, figureUrl, groupNames, intervalVerdict, kneeCurves, kneeTick, memberKey, memberName, memberNumber,
  minus, num, pairedRows, selectionText, sortKneeKeys, sparseTicks, type Chart, type ChartSelection, type CsvRow, type PairedRow,
} from "./model";

/* ── colours ──────────────────────────────────────────────────────────── */

export type Facet = { key: string; label: string; color: string };

/** One recipe field's colours: the loss keeps the console's loss colour
 *  (as Leaderboard and Members); any other field's groups take the export
 *  palette's hues (SLOT_ORDER) in first-seen order over the selected members,
 *  the order the backend export colours them in. `facet(label)` is a
 *  member's group. */
export type FieldColours = { of: (group: string) => Facet; facet: (label: string) => Facet; items: Facet[] };
export type Palette = (field: string) => FieldColours;

/** The backend export's PALETTE (studies/render.py) as theme tokens, in its
 *  order: blue, orange, green, purple, magenta, brown, teal; the export's
 *  grey / amber / indigo have no distinct token, so red closes the cycle. */
const SLOT_ORDER = [0, 2, 1, 3, 4, 7, 5, 6];

/** The ONE group → colour map every chart of a study view reads. Build it
 *  once per selection (StudyView) and on a theme flip (the tokens change). */
export function makePalette(members: readonly StudyMember[]): Palette {
  const cache = new Map<string, FieldColours>();
  return (field: string) => {
    const hit = cache.get(field);
    if (hit) return hit;
    const names = groupNames(members, field);
    const facets = new Map(names.map((n, i) => [n, { key: `${field}:${n}`, label: `${GROUP_LABEL[field] ?? field} ${n}`, color: field === "loss" ? LOSS_COLOR[n.toLowerCase()] : categorical(SLOT_ORDER[i % SLOT_ORDER.length]) }]));
    const fallback = (n: string): Facet => ({ key: `${field}:${n}`, label: n, color: C.muted });
    const byLabel = new Map(members.map((m) => [m.label, memberKey(m, field)]));
    const out: FieldColours = {
      of: (group) => facets.get(group) ?? fallback(group),
      facet: (label) => { const g = byLabel.get(label); return g != null ? facets.get(g) ?? fallback(g) : fallback(label); },
      items: [...facets.values()],
    };
    cache.set(field, out);
    return out;
  };
}

/* ── the frame every chart shares ─────────────────────────────────────── */

export type ChartProps = {
  detail: StudyDetail;
  chart: Chart;
  sel: ChartSelection;
  /** Members in the selection (all when none is picked). */
  members: StudyMember[];
  colourBy: string;
  palette: Palette;
  dpi: number;
  setDpi: (v: number) => void;
  set: (patch: Partial<ChartSelection>) => void;
  view: "absolute" | "relative";
  setView: (v: "absolute" | "relative") => void;
};

const DPIS = [150, 300, 600];

function Exports({ id, chart, sel, dpi, setDpi }: { id: string; chart: Chart; sel: ChartSelection; dpi: number; setDpi: (v: number) => void }) {
  return (
    <ToolbarGroup label="Export">
      <Select size="sm" aria-label="Resolution" value={String(dpi)} onChange={(v) => setDpi(Number(v))}
        options={DPIS.map((d) => ({ value: String(d), label: `${d} dpi` }))} />
      {(["pdf", "png", "svg"] as const).map((f) => (
        <Button key={f} size="sm" href={figureUrl(id, chart, f, sel, dpi)} download>{f.toUpperCase()}</Button>
      ))}
      <Button size="sm" href={csvUrl(id, chart, sel, true)} download>CSV</Button>
    </ToolbarGroup>
  );
}

function noteFor(detail: StudyDetail, chart: Chart, sel: ChartSelection, members: number, extra: string[] = []): string {
  const total = detail.manifest.ensemble?.members?.length ?? members;
  const id = detail.study.id;
  return [
    `### Study “${detail.study.name}” — ${CHART_TITLE[chart]}`,
    "",
    `- Source: ${detail.citation}`,
    `- Manifest sha256: \`${detail.manifest_sha256}\``,
    `- Selection: ${selectionText(members, total)}${sel.group ? `, grouped by ${GROUP_LABEL[sel.group] ?? sel.group}` : ""}`
      + (chart === "paired" ? `, reference ${sel.reference || "mean"}` : ""),
    `- Figure: ${figureUrl(id, chart, "pdf", sel)}`,
    `- Numbers: ${csvUrl(id, chart, sel)}`,
    ...extra,
  ].join("\n");
}

/** A chart: its title, its controls, and — only once its table has rows —
 *  the exports and "Log to notebook"; then loading / the backend's refusal
 *  (which names what the study lacks) / the drawing. `url` null = the study
 *  has no data for it: `empty` says so and nothing is offered. */
function ChartFrame({ p, url, controls, empty, children, note }: {
  p: ChartProps; url: string | null; controls?: ReactNode; empty?: ReactNode;
  children: (rows: CsvRow[]) => ReactNode; note?: (rows: CsvRow[]) => string[];
}) {
  const t = useStudyCsv(url);
  const rows = t.rows && t.rows.length ? t.rows : null;
  let body: ReactNode;
  if (!url) body = <p className="stu-empty">{empty}</p>;
  else if (t.error) body = <Callout tone="neutral" title="Nothing to draw" action={<Button size="sm" onClick={t.reload}>Retry</Button>}>{t.error.message}</Callout>;
  else if (!t.rows) body = <Skeleton height={260} />;
  else if (!rows) body = <p className="stu-empty">The table for this selection is empty.</p>;
  else body = children(rows);
  return (
    <section className="stu-chart" aria-label={CHART_TITLE[p.chart]}>
      <h3 className="stu-chart__title">{CHART_TITLE[p.chart]}</h3>
      {(controls || rows) && (
        <Toolbar label={`${CHART_TITLE[p.chart]} controls`}>
          {controls}
          <ToolbarSpacer />
          {rows && (
            <>
              <Exports id={p.detail.study.id} chart={p.chart} sel={p.sel} dpi={p.dpi} setDpi={p.setDpi} />
              <LogToNotebookButton from="Figures › Studies" note={() => noteFor(p.detail, p.chart, p.sel, p.members.length, note?.(rows))}
                title="Open Notebook › Log with this chart's citation (study id, manifest hash) and its export links" />
            </>
          )}
        </Toolbar>
      )}
      {body}
    </section>
  );
}

const bandShort = (b: string) => b.replace(/_E$/, "");

/* ── 1. PSNR vs knee ──────────────────────────────────────────────────── */

function KneePlots({ rows, p }: { rows: CsvRow[]; p: ChartProps }) {
  const facets = p.palette(p.colourBy);
  const groupColours = p.palette(p.sel.group ?? p.colourBy);
  const lg = useLegend();
  const { bands, curves } = useMemo(() => kneeCurves(rows), [rows]);
  const relative = p.view === "relative";
  const groupOrder = useMemo(() => [...new Set(rows.filter((r) => r.kind === "group").map((r) => r.series))], [rows]);
  const panels = useMemo(() => bands.map((band) => {
    const list = curves[band];
    const ref = relative ? list.find((c) => c.kind === "mean")?.y : undefined;
    const series: Series[] = list.map((c) => {
      const y = minus(c.y, ref);
      if (c.kind === "mean" || c.kind === "gate") {
        return { x: c.x, y, color: c.kind === "mean" ? C.mean : C.comb, width: 2.6, dots: true, dash: c.kind === "mean" ? [6, 3] : undefined,
          name: c.kind === "mean" ? "plain mean" : "production gate", key: c.kind };
      }
      if (c.kind === "group") {
        const color = groupColours.of(c.name).color;
        return { x: c.x, y, low: c.lo ? minus(c.lo, ref) : undefined, high: c.hi ? minus(c.hi, ref) : undefined, fillAlpha: 0.16,
          color, width: 2.2, name: `${c.name} (median of ${c.n})`, key: `group:${c.name}` };
      }
      const f = facets.facet(c.name);
      return { x: c.x, y, color: f.color, width: 1.2, alpha: 0.8, name: `${memberName(c.name)} · ${f.label}`, key: f.key };
    });
    const all = series.flatMap((s) => [...s.y, ...(s.low ?? []), ...(s.high ?? [])]);
    const xs = list.flatMap((c) => c.x).filter((v) => v > 0);
    const xDomain: [number, number] = xs.length ? [Math.min(...xs), Math.max(...xs)] : [0.1, 1e4];
    return { band, series, xDomain, yDomain: paddedDomain(all, { pad: 0.06, minSpan: 0.2 }) };
  }), [bands, curves, relative, facets, groupColours]);

  const legend = useMemo<LegendItem[]>(() => {
    const items: LegendItem[] = [
      { label: "production gate", key: "gate", color: C.comb, line: true },
      { label: "plain mean", key: "mean", color: C.mean, line: true, dash: true },
    ];
    if (groupOrder.length) {
      for (const g of groupOrder) items.push({ label: g, key: `group:${g}`, color: groupColours.of(g).color });
    } else {
      const seen = new Set(rows.filter((r) => r.kind === "member").map((r) => facets.facet(r.series).key));
      for (const f of facets.items) if (seen.has(f.key)) items.push({ label: f.label, key: f.key, color: f.color });
    }
    return items;
  }, [groupOrder, rows, facets, groupColours]);

  return (
    <>
      <Legend items={legend} {...lg.legendProps} />
      <div className="stu-charts">
        {panels.map((panel) => (
          <div key={panel.band} className="stu-panel">
            <h4 className="stu-panel__title">{bandShort(panel.band)}</h4>
            <Plot {...lg.plotProps} xScale="log" xDomain={panel.xDomain} xTicks={logTicks(panel.xDomain)} yDomain={panel.yDomain}
              xLabel="asinh knee [e⁻]" yLabel={relative ? "PSNR − plain mean [dB]" : "PSNR [dB]"} series={panel.series}
              guides={relative ? [{ axis: "y", v: 0, color: C.guide }] : undefined}
              aspect={0.62} syncKey="stu-knee" xFormat={(v) => `${+v.toPrecision(3)} e⁻`} yFormat={(v) => v.toFixed(2)}
              aria-label={`PSNR vs knee, ${bandShort(panel.band)}`} />
          </div>
        ))}
      </div>
    </>
  );
}

export function KneeChart(p: ChartProps) {
  const nFields = p.detail.numbers?.knee_psnr?.fields?.length ?? null;
  return (
    <ChartFrame p={p} url={csvUrl(p.detail.study.id, "knee", p.sel)} controls={(
      <ToolbarGroup label="View">
        <Segmented<"absolute" | "relative"> size="sm" aria-label="Curve view" value={p.view} onChange={p.setView}
          options={[{ value: "absolute", label: "absolute" }, { value: "relative", label: "vs mean" }]} />
      </ToolbarGroup>
    )}>{(rows) => (
      <>
        <KneePlots rows={rows} p={p} />
        <Caption>
          PSNR at each asinh knee, averaged over {nFields != null ? `${formatCount(nFields)} test fields` : "the test fields"}
          {p.sel.group ? "; a group line is the median of its members with their p16–p84 band" : ""}.
          {p.view === "relative" ? " Drawn minus the plain mean; the export, the CSV and the notebook entry keep absolute PSNR." : ""}
        </Caption>
      </>
    )}</ChartFrame>
  );
}

/* ── 2. integrated PSNR by loss × training knee ───────────────────────── */

function IntegratedPlots({ rows, p }: { rows: CsvRow[]; p: ChartProps }) {
  const facets = p.palette(p.colourBy);
  const lg = useLegend();
  const grouped = !!p.sel.group;
  const groupColours = p.palette(p.sel.group ?? p.colourBy);
  const members = rows.filter((r) => r.kind === "member");
  const bands = [...new Set(rows.map((r) => r.band))];
  const cats = sortKneeKeys(members.map((r) => r.training_knee));
  const colourOf = (r: CsvRow): Facet => (grouped ? groupColours.of(r.group) : facets.facet(r.series));
  const panels = bands.map((band) => {
    const at = members.filter((r) => r.band === band);
    const series: Series[] = [];
    for (const [ci, cat] of cats.entries()) {
      const here = at.filter((r) => r.training_knee === cat);
      here.forEach((r, j) => {
        const offset = here.length > 1 ? -0.25 + (0.5 * j) / (here.length - 1) : 0;
        const f = colourOf(r);
        series.push({ x: [ci + offset], y: [num(r.integrated_psnr)], mode: "scatter", color: f.color, width: 2.4,
          name: `${memberName(r.series)}${r.seed ? ` · seed ${r.seed}` : ""} · ${f.label}`, key: f.key });
      });
    }
    const refs = rows.filter((r) => r.band === band && (r.kind === "mean" || r.kind === "gate"));
    const guides: Guide[] = refs.map((r) => ({ axis: "y", v: num(r.integrated_psnr) ?? NaN, color: r.kind === "mean" ? C.mean : C.comb,
      dash: r.kind === "mean" ? [6, 3] : undefined, width: 1.6, label: r.kind === "mean" ? "plain mean" : "gate" }));
    const yDomain = paddedDomain([...at.map((r) => num(r.integrated_psnr)), ...guides.map((g) => g.v)], { pad: 0.08, minSpan: 0.2 });
    return { band, series, guides, yDomain };
  });
  const legendItems = useMemo<LegendItem[]>(() => {
    const seen = new Map<string, Facet>();
    for (const r of members) { const f = colourOf(r); if (!seen.has(f.key)) seen.set(f.key, f); }
    return [...seen.values()].map((f) => ({ label: f.label, key: f.key, color: f.color, marker: "filled" }));
  }, [rows, facets, groupColours, grouped]); // eslint-disable-line react-hooks/exhaustive-deps
  const xTicks = cats.map((c, i) => ({ v: i, label: kneeTick(c) }));
  return (
    <>
      <Legend items={legendItems} {...lg.legendProps} />
      {/* one panel per row: every training-knee category keeps its label */}
      <div className="stu-charts stu-charts--one">
        {panels.map((panel) => (
          <div key={panel.band} className="stu-panel">
            <h4 className="stu-panel__title">{bandShort(panel.band)}</h4>
            <Plot {...lg.plotProps} xDomain={[-0.6, Math.max(0.6, cats.length - 0.4)]} xTicks={xTicks} yDomain={panel.yDomain}
              xLabel="training knee [e⁻]" yLabel="integrated PSNR [dB]" series={panel.series} guides={panel.guides}
              aspect={0.34} zoomAxes="y" yFormat={(v) => v.toFixed(2)} aria-label={`Integrated PSNR by training knee, ${bandShort(panel.band)}`} />
          </div>
        ))}
      </div>
    </>
  );
}

export function IntegratedChart(p: ChartProps) {
  return (
    <ChartFrame p={p} url={csvUrl(p.detail.study.id, "integrated", p.sel)}>{(rows) => (
      <>
        <IntegratedPlots rows={rows} p={p} />
        <Caption>
          One dot per member (seeds side by side), coloured by {(GROUP_LABEL[p.sel.group ?? p.colourBy] ?? p.sel.group ?? p.colourBy).toLowerCase()};
          PSNR averaged over log knee, then over the test fields. The lines are the plain mean and the production gate.
        </Caption>
      </>
    )}</ChartFrame>
  );
}

/* ── 3. paired differences ────────────────────────────────────────────── */

function PairedPlots({ rows, p }: { rows: PairedRow[]; p: ChartProps }) {
  const grouped = !!p.sel.group;
  const targets = [...new Set(rows.map((r) => r.target))];
  const bands = [...new Set(rows.map((r) => r.band))];
  // Tick labels are mono: a loss reads "L1" there, never "l1" (≈ "11").
  const label = (t: string) => (grouped ? (p.sel.group === "loss" ? t.toUpperCase() : t) : memberNumber(t));
  const ticks = sparseTicks(targets.length).map((i) => ({ v: i, label: label(targets[i]) }));
  const tone = (r: PairedRow) => { const v = intervalVerdict(r); return v === "better" ? "stu-good" : v === "worse" ? "stu-bad" : undefined; };
  const columns: Column<string>[] = [
    { header: grouped ? GROUP_LABEL[p.sel.group ?? ""] ?? "Group" : "Member", cell: (t) => (grouped ? `${t} · ${rows.find((r) => r.target === t)?.n ?? "?"} members` : memberName(t)) },
    ...bands.map((b): Column<string> => ({
      header: `Δ ${bandShort(b)} [dB]`, align: "right",
      cell: (t) => { const r = rows.find((x) => x.target === t && x.band === b); return r ? <span className={`stu-tnum ${tone(r) ?? ""}`}>{deltaText(r)}</span> : "—"; },
    })),
  ];
  return (
    <>
      <div className="stu-charts">
        {bands.map((band) => {
          const at = targets.map((t) => rows.find((r) => r.target === t && r.band === band));
          const series: Series[] = at.map((r, i) => ({
            x: [i], y: [r?.mean ?? null], errorLow: [r?.lo ?? null], errorHigh: [r?.hi ?? null], mode: "scatter",
            color: r && intervalVerdict(r) === "unresolved" ? C.muted : bandColor(band), width: 2.2,
            name: r ? `${grouped ? r.target : memberName(r.target)}: ${deltaText(r)} dB` : "",
          }));
          const yDomain = paddedDomain([0, ...at.flatMap((r) => [r?.lo ?? null, r?.hi ?? null])], { pad: 0.1, minSpan: 0.05 });
          return (
            <div key={band} className="stu-panel">
              <h4 className="stu-panel__title">{bandShort(band)}</h4>
              <Plot xDomain={[-0.6, Math.max(0.6, targets.length - 0.4)]} xTicks={ticks} yDomain={yDomain} series={series}
                guides={[{ axis: "y", v: 0, color: C.guide, width: 1.4 }]} xLabel={grouped ? (GROUP_LABEL[p.sel.group ?? ""] ?? "group").toLowerCase() : "member"}
                yLabel={`Δ integrated PSNR vs ${rows[0]?.reference ?? "reference"} [dB]`} aspect={0.62} zoomAxes="y"
                yFormat={(v) => v.toFixed(2)} aria-label={`Paired differences, ${bandShort(band)}`} />
            </div>
          );
        })}
      </div>
      <Table className="stu-table" aria-label="Paired differences with 95% intervals" columns={columns} rows={targets} rowKey={(t) => t} />
    </>
  );
}

export function PairedChart(p: ChartProps) {
  const refOptions = useMemo(() => {
    const opts = [
      { value: "mean", label: "plain mean" }, { value: "gate", label: "production gate" }, { value: "best", label: "best member" },
    ];
    if (p.sel.group) for (const g of groupNames(p.members, p.sel.group)) opts.push({ value: `group:${g}`, label: `group ${g}` });
    for (const m of p.members) opts.push({ value: m.label, label: memberName(m.label) });
    return opts;
  }, [p.members, p.sel.group]);
  const ref = p.sel.reference || "mean";
  return (
    <ChartFrame p={p} url={csvUrl(p.detail.study.id, "paired", p.sel)} controls={(
      <ToolbarGroup label="Against">
        <Select size="sm" aria-label="Reference" value={ref} onChange={(v) => p.set({ reference: v === "mean" ? null : v })} options={refOptions} />
      </ToolbarGroup>
    )} note={(raw) => pairedRows(raw).slice(0, 24).map((r) => `- ${r.target} ${bandShort(r.band)}: ${deltaText(r)} dB vs ${r.reference}`)}>{(raw) => {
      const rows = pairedRows(raw);
      const first = rows[0];
      return (
        <>
          <PairedPlots rows={rows} p={p} />
          <Caption>
            Mean over {first?.nFields != null ? formatCount(first.nFields) : "the"} test fields of the per-field difference in knee-integrated PSNR
            against {first?.reference ?? ref}, with its 95 % paired bootstrap interval ({first?.resamples != null ? formatCount(first.resamples) : "2,000"} resamples,
            seed {first?.seed ?? 0}). In the plot a point is in its band's colour when the interval excludes zero and grey when it does not;
            in the table green is above zero, red below.
          </Caption>
        </>
      );
    }}</ChartFrame>
  );
}

/* ── 4. gate weights ──────────────────────────────────────────────────── */

function GatePlots({ rows, p }: { rows: CsvRow[]; p: ChartProps }) {
  const lg = useLegend();
  const members = rows.filter((r) => r.level === "member");
  const families = [...new Set(members.map((r) => r.family))];
  const bands = [...new Set(members.map((r) => r.band))];
  const familyColours = p.palette(p.sel.group ?? "loss");
  const colour = (f: string) => familyColours.of(f).color;
  const fam = rows.filter((r) => r.level === "family");
  const famColumns: Column<string>[] = [
    { header: GROUP_LABEL[p.sel.group ?? "loss"] ?? "Family", cell: (f) => <span className="stu-swatch-row"><span className="stu-swatch" style={{ ["--sw" as string]: colour(f) }} />{f}</span> },
    ...bands.map((b): Column<string> => ({
      header: `${bandShort(b)} share`, align: "right",
      cell: (f) => {
        const r = fam.find((x) => x.name === f && x.band === b);
        return r ? <span className="stu-tnum">{formatPercent(num(r.weight), 0)} <span className="muted">(uniform {formatPercent(num(r.uniform), 0)})</span></span> : "—";
      },
    })),
  ];
  return (
    <>
      <Legend items={families.map((f) => ({ label: f, key: `fam:${f}`, color: colour(f), marker: "filled" }))} {...lg.legendProps} />
      <div className="stu-charts">
        {bands.map((band) => {
          const at = members.filter((r) => r.band === band)
            .sort((a, b) => families.indexOf(a.family) - families.indexOf(b.family) || a.name.localeCompare(b.name, undefined, { numeric: true }));
          const uniform = num(at[0]?.uniform);
          const series: Series[] = at.map((r, i) => ({ x: [i], y: [num(r.weight)], mode: "scatter", color: colour(r.family), width: 2.4,
            name: `${memberName(r.name)} · ${r.family}: ${formatPercent(num(r.weight), 1)}`, key: `fam:${r.family}` }));
          return (
            <div key={band} className="stu-panel">
              <h4 className="stu-panel__title">{bandShort(band)}</h4>
              <Plot {...lg.plotProps} xDomain={[-0.6, Math.max(0.6, at.length - 0.4)]}
                xTicks={sparseTicks(at.length).map((i) => ({ v: i, label: memberNumber(at[i].name) }))}
                yDomain={paddedDomain([0, ...at.map((r) => num(r.weight)), uniform], { pad: 0.08 })} series={series}
                guides={uniform != null ? [{ axis: "y", v: uniform, color: C.guide, dash: [3, 3], label: "uniform" }] : undefined}
                xLabel="member" yLabel="mean gate weight" aspect={0.62} zoomAxes="y" yFormat={(v) => formatPercent(v, 0)}
                aria-label={`Gate weight per member, ${bandShort(band)}`} />
            </div>
          );
        })}
      </div>
      <Table className="stu-table" aria-label="Gate weight per family" columns={famColumns} rows={families} rowKey={(f) => f} />
    </>
  );
}

export function GateChart(p: ChartProps) {
  const diag = p.detail.numbers?.gate?.diagnostic;
  const sources = [{ value: "all", label: "all pixels" }, { value: "sources", label: "source pixels" },
    ...(diag?.brightness_names ?? []).map((n) => ({ value: n, label: `${n} pixels` }))];
  return (
    <ChartFrame p={p} url={diag?.available ? csvUrl(p.detail.study.id, "gate", p.sel) : null}
      empty="No gate weight diagnostic in this study: the production gate had no diagnostic payload when it was frozen."
      controls={diag?.available ? (
        <ToolbarGroup label="Pixels">
          <Select size="sm" aria-label="Which pixels" value={p.sel.source || "all"} onChange={(v) => p.set({ source: v === "all" ? null : v })} options={sources} />
        </ToolbarGroup>
      ) : undefined}>{(rows) => (
      <>
        <GatePlots rows={rows} p={p} />
        <Caption>
          The production gate's held-out weight per member, and summed per {GROUP_LABEL[p.sel.group ?? "loss"]?.toLowerCase() ?? "family"} against the share a uniform
          gate would give it.{p.detail.numbers?.gate?.compare_note ? ` ${p.detail.numbers.gate.compare_note}` : ""}
        </Caption>
      </>
    )}</ChartFrame>
  );
}

/* ── 5. training curves ───────────────────────────────────────────────── */

const METRICS = [
  { value: "psnr", label: "PSNR" }, { value: "VIS", label: "VIS" }, { value: "Y_E", label: "Y" },
  { value: "J_E", label: "J" }, { value: "H_E", label: "H" }, { value: "loss", label: "loss" },
];

function TrainingPlot({ rows, p }: { rows: CsvRow[]; p: ChartProps }) {
  const facets = p.palette(p.colourBy);
  const groupColours = p.palette(p.sel.group ?? p.colourBy);
  const lg = useLegend();
  const grouped = !!p.sel.group;
  const groups = [...new Set(rows.map((r) => r.group))];
  const byMember = new Map<string, CsvRow[]>();
  for (const r of rows) { const l = byMember.get(r.member) ?? []; l.push(r); byMember.set(r.member, l); }
  const series: Series[] = [...byMember.entries()].map(([m, rs]) => {
    const f: Facet = grouped ? groupColours.of(rs[0].group) : facets.facet(m);
    return { x: rs.map((r) => num(r.step) ?? NaN), y: rs.map((r) => num(r.value)), color: f.color, width: 1.4, alpha: 0.85,
      name: `${memberName(m)} · ${f.label}`, key: f.key };
  });
  const legend: LegendItem[] = grouped
    ? groups.map((g) => { const f = groupColours.of(g); return { label: f.label, key: f.key, color: f.color }; })
    : facets.items.filter((f) => series.some((s) => s.key === f.key)).map((f) => ({ label: f.label, key: f.key, color: f.color }));
  const metric = p.sel.metric || "psnr";
  return (
    <>
      <Legend items={legend} {...lg.legendProps} />
      <Plot {...lg.plotProps} xDomain={paddedDomain(rows.map((r) => num(r.step)), { pad: 0.01 })}
        yDomain={paddedDomain(rows.map((r) => num(r.value)), { pad: 0.05 })} series={series}
        xLabel="step" yLabel={metric === "loss" ? "combined loss" : `validation PSNR${metric === "psnr" ? " (joint)" : ` ${bandShort(metric)}`} [dB]`}
        aspect={0.42} xFormat={(v) => formatCount(Math.round(v))} yFormat={(v) => v.toFixed(metric === "loss" ? 4 : 2)} aria-label="Training curves" />
    </>
  );
}

export function TrainingChart(p: ChartProps) {
  return (
    <ChartFrame p={p} url={csvUrl(p.detail.study.id, "training", p.sel)} controls={(
      <ToolbarGroup label="Metric">
        <Segmented size="sm" aria-label="Training metric" value={p.sel.metric || "psnr"} onChange={(v) => p.set({ metric: v === "psnr" ? null : v })} options={METRICS} />
      </ToolbarGroup>
    )}>{(rows) => (
      <>
        <TrainingPlot rows={rows} p={p} />
        <Caption>Validation curves from each member's training log, as pulled when the study was frozen.</Caption>
      </>
    )}</ChartFrame>
  );
}

/* ── 6. real-tile metrics ─────────────────────────────────────────────── */

const REAL_COLUMNS = [
  { id: "hole_pct", header: "Holes [%]", digits: 1 }, { id: "pct_R_lt_0p8", header: "Peaks R < 0.8 [%]", digits: 1 },
  { id: "median_R", header: "Median R", digits: 2 }, { id: "flux_ratio", header: "ΣSR/ΣLR", digits: 3 },
];

const modelName = (m: string) => (/^\d/.test(m) ? memberName(m) : m);

/** Holes % and median R per model as grouped bars (one colour per band):
 *  every band's series spans every slot, so each bar is one slot wide. */
function RealBars({ rows }: { rows: CsvRow[] }) {
  const models = [...new Set(rows.map((r) => r.model))];
  const bands = [...new Set(rows.map((r) => r.band))];
  const width = bands.length + 1;
  const slots = models.length * width - 1;
  const xs = Array.from({ length: slots }, (_, i) => i);
  const bars = (metric: string): Series[] => bands.map((b, j) => ({
    x: xs, mode: "histogram", color: bandColor(b), fillAlpha: 0.7, width: 1, name: bandShort(b), key: b,
    y: xs.map((i) => {
      if (i % width !== j) return null;
      const r = rows.find((x) => x.model === models[Math.floor(i / width)] && x.band === b);
      return r ? num(r[metric]) : null;
    }),
  }));
  const ticks = models.map((m, k) => ({ v: k * width + (bands.length - 1) / 2, label: /^\d/.test(m) ? memberNumber(m) : m }));
  const panel = (metric: string, label: string, guide?: number) => {
    const series = bars(metric);
    return (
      <div className="stu-panel">
        <h4 className="stu-panel__title">{label}</h4>
        <Plot xDomain={[-0.6, slots - 0.4]} xTicks={ticks} yDomain={paddedDomain([0, ...series.flatMap((x) => x.y), guide], { pad: 0.08 })}
          series={series} guides={guide != null ? [{ axis: "y", v: guide, color: C.guide, dash: [3, 3] }] : undefined}
          xLabel="model" yLabel={label} aspect={0.5} zoom={false} yFormat={(v) => v.toFixed(metric === "median_R" ? 2 : 0)}
          aria-label={`${label} per model and band`} />
      </div>
    );
  };
  return (
    <>
      <Legend items={bands.map((b) => ({ label: bandShort(b), color: bandColor(b), histogram: true }))} />
      <div className="stu-charts">
        {panel("hole_pct", "Holes [%]")}
        {panel("median_R", "Median R", 1)}
      </div>
    </>
  );
}

export function RealChart(p: ChartProps) {
  const exps = p.detail.numbers?.real?.experiments ?? [];
  const current = p.sel.experiment || exps[0]?.id;
  return (
    <ChartFrame p={p} url={exps.length ? csvUrl(p.detail.study.id, "real", p.sel) : null}
      empty={`No real-tile data in this study${p.detail.numbers?.real?.note ? `: ${p.detail.numbers.real.note}` : "."}`}
      controls={exps.length > 1 ? (
        <ToolbarGroup label="Run">
          <Select size="sm" aria-label="Sky › Compare run" value={current} onChange={(v) => p.set({ experiment: v === exps[0].id ? null : v })}
            options={exps.map((e) => ({ value: e.id, label: `${e.label || e.id}${e.created ? ` · ${formatDate(e.created)}` : ""}` }))} />
        </ToolbarGroup>
      ) : undefined}>{(rows) => {
      const columns: Column<CsvRow>[] = [
        { header: "Model", cell: (r) => modelName(r.model) },
        { header: "Band", cell: (r) => bandShort(r.band) },
        ...REAL_COLUMNS.map((c): Column<CsvRow> => ({ header: c.header, align: "right", cell: (r) => <span className="stu-tnum">{num(r[c.id])?.toFixed(c.digits) ?? "—"}</span> })),
      ];
      return (
        <>
          <RealBars rows={rows} />
          <Table className="stu-table" aria-label="Real-tile metrics" columns={columns} rows={rows} rowKey={(r) => `${r.spec}:${r.band}`} />
          <Caption>Sky › Compare run {rows[0]?.experiment} on {rows[0]?.n_tiles || "?"} real tiles (no truth): holes = % of bright LR pixels the SR blanks; R = SR/LR enclosed peak flux (1 = flux kept).</Caption>
        </>
      );
    }}</ChartFrame>
  );
}

export const CHART_COMPONENT: Record<Chart, (p: ChartProps) => ReactNode> = {
  knee: KneeChart, integrated: IntegratedChart, paired: PairedChart, gate: GateChart, training: TrainingChart, real: RealChart,
};

