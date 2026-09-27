/* Galaxies › marginals: VIS 2FWHM brightness (Q1 aggregate, generated
   galaxies, the generation law, with the Q1 support overlays), the
   surface-density radii, the three NISP/VIS colours and the normalized
   half-light shape — a two-column grid of panels on the chart kit. Every
   panel keeps its explanations (the server's note, the curve definitions and
   disclosures) in ONE header popover; the curve pickers are compact chips
   whose swatch is the line style, so they double as the legend. */
import type { ReactNode } from "react";
import Plot, { type Series } from "../../../charts/Plot";
import { useUrlState } from "../../../hooks/useUrlState";
import { extent, linearTicks } from "../../../ticks";
import { Badge, Button, Chip, EmptyState } from "../../../ui";
import type { BrightnessCurve, Parameter, RadiusCurve } from "../api";
import { SOURCE_META, surveyColor, ticksFor } from "../chartKit";
import { Info, Swatch } from "../common";
import {
  MARGINAL_ORDER, SURVEY_GROUP, USEFUL_RADIUS_KEYS, USEFUL_SHAPE_KEYS, brightnessDisclosure, brightnessEntries,
  brightnessOverlays, brightnessSeries, densityDomain, densitySeries, normalizationOf, radiusEntries, radiusSeries,
  radiusYLabel, toggleRadius, xAxisOf, xDomainOf,
} from "./model";

const NONE = "-";

/** A URL-kept selection of curve keys; absent = the curves' defaults. */
function useSelection(urlKey: string, defaults: string[]): [string[], (keys: string[]) => void] {
  const [raw, setRaw] = useUrlState<string[]>(urlKey, []);
  const selected = raw.length === 0 ? defaults : raw.filter((k) => k !== NONE);
  return [selected, (keys) => setRaw(keys.length ? keys : [NONE])];
}

/** A figure panel: title, at most one short subtitle, and the long text in
 *  the header's info popover. */
export function MarginalPanel(
  { title, sub, about, wide = false, children }: {
    title: string; sub?: string; about?: ReactNode; wide?: boolean; children: ReactNode;
  },
) {
  return (
    <article className={`rl-panel${wide ? " rl-wide" : ""}`} aria-label={title}>
      <header className="rl-panel__head">
        <strong>{title}</strong>
        {sub && <small>{sub}</small>}
        {about && <span className="rl-panel__tools"><Info label={`About ${title}`}>{about}</Info></span>}
      </header>
      {children}
    </article>
  );
}

type Def = { key: string; color: string; dash?: boolean; label: string; text: string };

/** The per-curve definitions, inside a panel's info popover. */
function Defs({ items, title = "Curves" }: { items: Def[]; title?: string }) {
  if (!items.length) return null;
  return (
    <>
      <p className="rl-subhead">{title}</p>
      <ul className="rl-info__defs">
        {items.map((d) => (
          <li key={d.key}><Swatch color={d.color} dash={d.dash} /><span><b>{d.label}</b> {d.text}</span></li>
        ))}
      </ul>
    </>
  );
}

type PickItem = { key: string; label: string; hint: string; color: string; dash: boolean };
type PickGroup = { id: string; title: string; hint: string; items: PickItem[]; missing?: string[] };

/** Compact curve chips by group; each group has one all / none toggle. */
function CurvePicker(
  { groups, selected, onToggle, onGroup, label }: {
    groups: PickGroup[]; selected: string[]; label: string;
    onToggle: (key: string) => void; onGroup: (group: PickGroup, on: boolean) => void;
  },
) {
  return (
    <div className="rl-curves" role="group" aria-label={label}>
      {groups.map((g) => {
        const all = g.items.every((it) => selected.includes(it.key));
        return (
          <div key={g.id} className="rl-curves__group" role="group" aria-label={g.title}>
            <span className="rl-curves__title" title={g.hint}>{g.title}</span>
            {g.items.map((it) => (
              <Chip key={it.key} className="rl-chip" on={selected.includes(it.key)} onClick={() => onToggle(it.key)} title={it.hint}>
                <Swatch color={it.color} dash={it.dash} />
                <span className="rl-chip__label">{it.label}</span>
              </Chip>
            ))}
            {g.items.length > 1 && (
              <Button size="sm" variant="ghost" aria-label={`${all ? "Hide" : "Show"} every ${g.title} curve`}
                onClick={() => onGroup(g, !all)}>{all ? "none" : "all"}</Button>
            )}
            {!!g.missing?.length && <Badge size="sm" tone="warn" title={g.missing.join("\n")}>{g.missing.length} missing</Badge>}
          </div>
        );
      })}
    </div>
  );
}

const styleOf = (series: Series[]) => {
  const byKey = new Map(series.map((s) => [s.key ?? s.name ?? "", s]));
  return (key: string) => ({ color: byKey.get(key)?.color ?? surveyColor("euclid"), dash: !!byKey.get(key)?.dash?.length });
};

/* ─── colours (three-source marginal) ───────────────────────────────────── */

export function DensityPanel({ parameter, name }: { parameter: Parameter; name: string }) {
  const series = densitySeries(parameter);
  const defs: Def[] = MARGINAL_ORDER.flatMap((key) => parameter.series[key]
    ? [{ key, color: surveyColor(key), label: SOURCE_META[key].label, text: parameter.series[key]!.definition }] : []);
  const about = <><p>{parameter.note}</p><Defs items={defs} /></>;
  let body: ReactNode = <EmptyState compact icon="activity" title="No source has produced this marginal yet" />;
  if (series.length && series.some((s) => s.y.some((v) => v != null))) {
    const axis = xAxisOf(parameter);
    const xDomain = xDomainOf(parameter, series.flatMap((s) => s.x).map((v) => (axis.scale === "log" ? Math.log10(v) : v)));
    const yDomain = densityDomain(series);
    body = (
      <Plot xDomain={xDomain} yDomain={yDomain} xScale={axis.scale} yScale="log"
        xTicks={ticksFor(xDomain, axis.scale)} yTicks={ticksFor(yDomain, "log")}
        xLabel={axis.label} yLabel={`${parameter.density_unit} (log scale)`} series={series}
        legend="auto" aspect={0.62} exportName={`galaxy-${name}`} aria-label={`${parameter.label} marginal`} />
    );
  }
  return <MarginalPanel title={parameter.label} about={about}>{body}</MarginalPanel>;
}

/* ─── brightness ────────────────────────────────────────────────────────── */

const SURVEYS: BrightnessCurve["survey"][] = ["euclid", "synthetic", "cosmos", "fit", "generation"];

export function BrightnessPanel({ parameter }: { parameter: Parameter }) {
  const entries = brightnessEntries(parameter);
  const defaults = entries.filter(([, c]) => c.default_on).map(([k]) => k);
  const [selected, setSelected] = useSelection("mag", defaults);
  const visible = entries.filter(([k]) => selected.includes(k));
  const series = brightnessSeries(visible);
  const style = styleOf(brightnessSeries(entries));
  const xs = visible.flatMap(([, c]) => c.x);
  const toggle = (key: string) => setSelected(selected.includes(key) ? selected.filter((k) => k !== key) : [...selected, key]);
  const groups: PickGroup[] = SURVEYS.flatMap((survey) => {
    const group = entries.filter(([, c]) => c.survey === survey);
    if (!group.length) return [];
    return [{
      id: survey, title: SURVEY_GROUP[survey].title, hint: SURVEY_GROUP[survey].sub,
      missing: survey === "cosmos" ? parameter.photometry_missing : undefined,
      items: group.map(([key, curve]) => ({
        key, label: curve.label, ...style(key),
        hint: `${curve.band}; ${curve.estimator}. Selection: ${curve.selection}. ${brightnessDisclosure(curve)}`,
      })),
    }];
  });
  let plot: ReactNode = <EmptyState compact icon="activity" title="Select at least one catalogue measurement" />;
  let summary: ReactNode = null;
  let trustAbout: ReactNode = null;
  if (series.length && xs.length) {
    const xDomain: [number, number] = extent(xs) ?? [0, 1];
    const yDomain = densityDomain(series);
    const o = brightnessOverlays(entries, xDomain, yDomain);
    plot = (
      <Plot xDomain={xDomain} yDomain={yDomain} yScale="log" xTicks={linearTicks(xDomain, { count: 7 })}
        yTicks={ticksFor(yDomain, "log")} xLabel={parameter.x_label} yLabel={`${parameter.density_unit} (log scale)`}
        bands={o.bands} guides={o.guides} series={series} aspect={0.4}
        exportName="galaxy-brightness" aria-label="VIS 2FWHM brightness marginal" />
    );
    if (o.trust && o.peak != null && o.peakMag != null) {
      summary = (
        <div className="rl-trust" aria-label="Euclid magnitude support and five-sigma boundary">
          <div><small>Q1 count turnover</small><b>VIS {o.peakMag.toFixed(2)}</b>
            <span>peak {o.peak.toFixed(1)} arcmin⁻² mag⁻¹{o.cumulativeToBoundary != null ? ` · ${o.cumulativeToBoundary.toFixed(1)} arcmin⁻² to 5σ` : ""}</span></div>
          <div><small>MER {o.trust.snr}σ limit</small><b>VIS {o.trust.magnitude.toFixed(2)}</b>
            <span>{o.trust.lower_magnitude.toFixed(2)}–{o.trust.upper_magnitude.toFixed(2)} (16–84%) · {Math.round(o.trust.sample_size).toLocaleString("en")} rows</span></div>
          <div><small>generation ceiling</small><b>{o.generationCap?.toFixed(1) ?? "—"}</b>
            <span>{o.generationCap != null && o.generationCap > o.peak ? `${(o.generationCap / o.peak).toFixed(1)}× the Q1 peak` : "matches the Q1 peak"}</span></div>
        </div>
      );
      trustAbout = (
        <>
          <p className="rl-subhead">Brightness support</p>
          <p>{o.trust.estimator}. {o.trust.caveat}</p>
          <p>Selection: {o.trust.selection}. The trust overlay does not change the sampler; the plateau beyond the MER {o.trust.snr}σ range is an explicit extrapolation.</p>
        </>
      );
    }
  }
  const about = (
    <>
      <p>{parameter.note}</p>
      <p>One fitted brightness coordinate: VIS 2FWHM (Q1 aggregate, the generated galaxies, the active generation law).</p>
      {trustAbout}
      {groups.map((g) => (
        <Defs key={g.id} title={`${g.title} · ${g.hint}`}
          items={entries.filter(([, c]) => c.survey === g.id).map(([key, curve]) => ({
            key, ...style(key), label: curve.label, text: brightnessDisclosure(curve),
          }))} />
      ))}
      {parameter.photometry_missing?.map((m) => <p key={m} className="rl-faint">{m}</p>)}
    </>
  );
  return (
    <MarginalPanel title={parameter.label} about={about} wide>
      <div className="rl-marginal">
        {summary}
        <CurvePicker label="Brightness curves" groups={groups} selected={selected} onToggle={toggle}
          onGroup={(g, on) => setSelected(on
            ? Array.from(new Set([...selected, ...g.items.map((it) => it.key)]))
            : selected.filter((k) => !g.items.some((it) => it.key === k)))} />
        {plot}
      </div>
    </MarginalPanel>
  );
}

/* ─── radius ────────────────────────────────────────────────────────────── */

const RADIUS_GROUP: Record<"half_light" | "rendered_half_light", { title: string; hint: string }> = {
  half_light: { title: "Half-light radius", hint: "catalogue and requested geometry" },
  rendered_half_light: { title: "Rendered image half-light", hint: "measured on clean generated images" },
};

export function RadiusPanel({ parameter }: { parameter: Parameter }) {
  const entries = radiusEntries(parameter, USEFUL_RADIUS_KEYS);
  const byKey = Object.fromEntries(entries) as Record<string, RadiusCurve>;
  const [selected, setSelected] = useSelection("re", entries.map(([k]) => k));
  const visible = entries.filter(([k]) => selected.includes(k));
  const series: Series[] = radiusSeries(parameter, visible);
  const style = styleOf(radiusSeries(parameter, entries));
  const groups: PickGroup[] = (["half_light", "rendered_half_light"] as const).flatMap((type) => {
    const group = entries.filter(([, c]) => c.radius_type === type);
    if (!group.length) return [];
    return [{
      id: type, ...RADIUS_GROUP[type], missing: type === "half_light" ? parameter.radius_missing : undefined,
      items: group.map(([key, curve]) => ({ key, label: curve.label, ...style(key), hint: `${curve.units}; ${curve.definition}` })),
    }];
  });
  const selectGroup = (g: PickGroup, on: boolean) => {
    if (!on) { setSelected(selected.filter((k) => !g.items.some((it) => it.key === k))); return; }
    const norm = normalizationOf(byKey[g.items[0]?.key]);
    setSelected(Array.from(new Set([...selected.filter((k) => normalizationOf(byKey[k]) === norm), ...g.items.map((it) => it.key)])));
  };
  const axis = xAxisOf(parameter);
  let plot: ReactNode = <EmptyState compact icon="activity" title="Select at least one radius observable" />;
  if (series.length) {
    const xDomain = xDomainOf(parameter, visible.flatMap(([, c]) => c.x));
    const yDomain = densityDomain(series);
    plot = (
      <Plot xDomain={xDomain} yDomain={yDomain} xScale={axis.scale} yScale="log"
        xTicks={ticksFor(xDomain, axis.scale)} yTicks={ticksFor(yDomain, "log")}
        xLabel={axis.label} yLabel={radiusYLabel(parameter, visible)} series={series} aspect={0.42}
        exportName="galaxy-radius" aria-label="Circularized half-light radius marginal" />
    );
  }
  const about = (
    <>
      <p>{parameter.note}</p>
      <p>Surface-density radii only: Q1 circularized Sérsic Rₑ, requested generated Rₑ, clean-image half-light, and the model marginal.</p>
      {groups.map((g) => (
        <Defs key={g.id} title={`${g.title} · ${g.hint}`}
          items={entries.filter(([, c]) => c.radius_type === g.id).map(([key, curve]) => ({
            key, ...style(key), label: curve.label, text: `${curve.units}; ${curve.definition}`,
          }))} />
      ))}
      {parameter.radius_missing?.map((m) => <p key={m} className="rl-faint">{m}</p>)}
    </>
  );
  return (
    <MarginalPanel title={parameter.label} about={about} wide>
      <div className="rl-marginal">
        <CurvePicker label="Radius curves" groups={groups} selected={selected}
          onToggle={(key) => setSelected(toggleRadius(selected, key, byKey))} onGroup={selectGroup} />
        {plot}
      </div>
    </MarginalPanel>
  );
}

/* ─── normalized half-light shape ───────────────────────────────────────── */

export function RadiusShapePanel({ parameter }: { parameter: Parameter }) {
  const entries = radiusEntries(parameter, USEFUL_SHAPE_KEYS);
  const series = radiusSeries(parameter, entries);
  const about = (
    <p>Each curve integrates to one over log-radius: solid model = the Q1 magnitude mix, dashed = the full faint extension used for generation.</p>
  );
  let body: ReactNode = <EmptyState compact icon="activity" title="Build the Q1 radius aggregate and candidate model first" />;
  if (series.length && series.some((s) => s.y.some((v) => v != null))) {
    const axis = xAxisOf(parameter);
    const xDomain = xDomainOf(parameter, entries.flatMap(([, c]) => c.x));
    const yDomain = densityDomain(series);
    body = (
      <Plot xDomain={xDomain} yDomain={yDomain} xScale={axis.scale} yScale="log"
        xTicks={ticksFor(xDomain, axis.scale)} yTicks={ticksFor(yDomain, "log")}
        xLabel={axis.label} yLabel="normalized probability / dex (log scale)" series={series} legend="auto" aspect={0.42}
        exportName="galaxy-radius-shape" aria-label="Normalized half-light shape" />
    );
  }
  return (
    <MarginalPanel title="Normalized half-light shape" sub="unit-integral Q1 and model radius densities" about={about}>
      {body}
    </MarginalPanel>
  );
}
