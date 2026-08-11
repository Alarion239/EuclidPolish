import { useEffect, useMemo, useState } from "react";
import { NavLink } from "react-router-dom";
import Plot, { type Series, type Tick } from "../charts/Plot";
import { useResource } from "../hooks";
import { JobProgressView, useJob } from "../jobs";
import {
  Badge, Button, Card, CardBody, CardHead, Chip, Empty,
  Page, PageHead, Segmented, Spinner, Stat,
} from "../ui";
import "./galaxy-distributions.css";

type Curve = {
  x: number[];
  density: number[];
  weighted_count: number;
  definition: string;
};
type BrightnessCurve = Curve & {
  label: string;
  survey: "euclid" | "cosmos" | "fit";
  band: string;
  estimator: string;
  selection: string;
  default_on?: boolean;
  fit_interval?: [number, number];
  sampling_interval?: [number, number];
  extrapolated_interval?: [number, number];
};
type RadiusCurve = Curve & {
  label: string;
  source: SourceKey;
  radius_type: "detection" | "kron" | "half_light";
  units: string;
  default_on?: boolean;
};
type Parameter = {
  label: string;
  x_label: string;
  x_domain?: [number, number];
  density_unit: string;
  note: string;
  series: Partial<Record<SourceKey, Curve>>;
  photometry_series?: Record<string, BrightnessCurve>;
  photometry_missing?: string[];
  radius_series?: Record<string, RadiusCurve>;
  radius_missing?: string[];
};
type Source = {
  available?: boolean;
  detail?: string;
  rows?: number;
  area_arcmin2?: number;
  schema_version?: number;
  phz_pdf_rows?: number;
  physical_rows?: number;
  phz_pdf_source?: string;
  fingerprint?: string;
  active_fingerprint?: string;
  is_active?: boolean;
  validated?: boolean;
  version?: number;
  aperture_scatter?: ApertureScatter;
};
type FluxKey = "f1" | "f2" | "f3" | "f4";
type GrowthKey = "g1" | "g2" | "g3";
type ApertureScatter = {
  count: number;
  magnitudes: Record<FluxKey, number[]>;
  growth: Record<GrowthKey, number[]>;
  definitions: Record<GrowthKey, string>;
  selection: string;
};
type Payload = {
  stale: boolean;
  authenticated?: boolean;
  sources: Partial<Record<SourceKey, Source>>;
  q1_counts?: Q1Counts | null;
  parameters: Record<string, Parameter>;
};
type Q1Counts = {
  footprint_area_deg2: number;
  bright: number;
  faint: number;
  bin_width: number;
  query_count: number;
  completed_queries?: number;
  total_queries?: number;
  complete?: boolean;
  phases_completed?: number;
  phase_count?: number;
  fit_available?: boolean;
  fit_ready?: boolean;
  selection: string;
  apertures: Record<FluxKey, {
    label: string;
    selected_galaxies: number;
    expected_galaxies?: number;
    queried_bins?: number;
  }>;
};
type SourceKey = "euclid" | "cosmos" | "fit";

const SOURCE: Record<SourceKey, { label: string; kicker: string; color: string }> = {
  euclid: { label: "Euclid MER + PHZ", kicker: "observed layer", color: "#2478d4" },
  cosmos: { label: "COSMOS2025", kicker: "diagnostic only", color: "#00a078" },
  fit: { label: "Euclid joint fit", kicker: "generator candidate", color: "#e25543" },
};
const ORDER: SourceKey[] = ["euclid", "cosmos", "fit"];
const PARAMETER_ORDER = ["redshift", "magnitude", "radius", "stellar_mass", "specific_sfr"];
const GROWTH: Record<GrowthKey, { label: string; color: string }> = {
  g1: { label: "g₁ = m₁ − m₄", color: "#31a7d8" },
  g2: { label: "g₂ = m₂ − m₄", color: "#d39b32" },
  g3: { label: "g₃ = m₃ − m₄", color: "#dc6658" },
};
const BRIGHTNESS_COLORS = {
  euclid: ["#2478d4", "#31a7d8", "#33c4c9", "#786fd4", "#ad6bd8", "#e268a7"],
  cosmos: ["#00a078", "#35b45f", "#83b93d", "#c2a82f", "#dd843c", "#df5f51", "#c95285", "#9e68c7", "#697fd0", "#3c9eb9", "#4b8c70", "#8c8751", "#ad6d69", "#88738e", "#5f8290"],
  fit: ["#e25543", "#ef754b", "#c94d68", "#9f518d"],
};
const RADIUS_COLORS: Record<string, string> = {
  euclid_detection: "#2478d4",
  euclid_kron: "#786fd4",
  euclid_sersic_re: "#31a7d8",
  cosmos_re: "#00a078",
  fit_re: "#e25543",
};
const RADIUS_TYPE_LABEL: Record<RadiusCurve["radius_type"], string> = {
  detection: "Detection / deblender",
  kron: "Kron photometry",
  half_light: "Half-light radius",
};

const compact = (value?: number) => value == null ? "—" : new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
const ticks = ([a, b]: [number, number], count = 5): Tick[] => Array.from({ length: count }, (_, i) => {
  const v = a + (b - a) * i / (count - 1);
  return { v, label: Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(Math.abs(b - a) < 2 ? 2 : 1) };
});

const physicalTickLabel = (value: number): string => {
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  if (magnitude >= 100_000 || magnitude < 0.0001) {
    return value.toExponential(1).replace(".0e", "e").replace("e+", "e");
  }
  const decimals = magnitude >= 100 ? 0
    : magnitude >= 10 ? 1
      : magnitude >= 0.1 ? 2
        : magnitude >= 0.01 ? 3
          : 4;
  const fixed = value.toFixed(decimals);
  return decimals ? fixed.replace(/\.?0+$/, "") : fixed;
};

// Curves stay linear in their already-transformed log10 coordinates. Only the
// tick text is inverted, so readers see physical values without changing plot
// geometry or interpolation.
const physicalLogTicks = (domain: [number, number], count = 5): Tick[] => ticks(domain, count)
  .map(({ v }) => ({ v, label: physicalTickLabel(10 ** v) }));

const physicalLogAxisLabel = (label: string): string => {
  const physical = label.replace(/^log₁₀\s*/, "");
  return physical.endsWith(")")
    ? `${physical.slice(0, -1)}, log scale)`
    : `${physical} (log scale)`;
};

const paddedDomain = (values: number[], minimumSpan: number): [number, number] => {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return [0, minimumSpan];
  const lo = Math.min(...finite), hi = Math.max(...finite);
  const span = Math.max(hi - lo, minimumSpan);
  const pad = 0.045 * span;
  return [lo - pad, hi + pad];
};

function SourceLedger({ sources }: { sources: Payload["sources"] }) {
  return <section className="galaxy-ledger" aria-label="Galaxy distribution data layers">
    {ORDER.map((key, index) => {
      const source = sources[key] ?? {};
      const fitted = key === "fit";
      return <article className="galaxy-ledger__source" key={key} style={{ "--source": SOURCE[key].color } as React.CSSProperties}>
        <div className="galaxy-ledger__number">0{index + 1}</div>
        <div>
          <div className="galaxy-ledger__kicker">{SOURCE[key].kicker}</div>
          <h2>{SOURCE[key].label}</h2>
          <p>{source.detail ?? "No cached product yet."}</p>
        </div>
        <div className="galaxy-ledger__metrics">
          {source.available ? <Badge tone={fitted && !source.validated ? "warn" : "good"}>{fitted ? (source.is_active ? "active" : "candidate") : "cached"}</Badge> : <Badge tone="warn">missing</Badge>}
          {!fitted && <span><b>{compact(source.rows)}</b> objects</span>}
          {!fitted && <span><b>{source.area_arcmin2?.toFixed(1) ?? "—"}</b> arcmin²</span>}
          {key === "euclid" && <span><b>{compact(source.phz_pdf_rows)}</b> {source.phz_pdf_source === "summary_reconstruction" ? "reconstructed PDFs" : "PHZ PDFs"}</span>}
          {fitted && source.fingerprint && <span title={source.fingerprint}><b>{source.fingerprint.slice(0, 10)}</b> fingerprint</span>}
        </div>
      </article>;
    })}
  </section>;
}

function DensityPlot({ parameter }: { parameter: Parameter }) {
  const curves = ORDER.flatMap((key) => parameter.series[key] ? [[key, parameter.series[key]!] as const] : []);
  const logValues = curves.flatMap(([, curve]) => curve.density.filter((v) => v > 0).map(Math.log10));
  if (!curves.length || !logValues.length) return <Empty>No source has produced this marginal yet.</Empty>;
  const xValues = curves.flatMap(([, curve]) => curve.x);
  const xDomain: [number, number] = [Math.min(...xValues), Math.max(...xValues)];
  const lo = Math.floor(Math.min(...logValues) * 2) / 2;
  const hi = Math.ceil(Math.max(...logValues) * 2) / 2;
  const yDomain: [number, number] = [lo, hi <= lo ? lo + 1 : hi];
  const logarithmicX = parameter.x_label.startsWith("log₁₀");
  const series: Series[] = curves.map(([key, curve]) => ({
    x: curve.x,
    y: curve.density.map((v) => v > 0 ? Math.log10(v) : null),
    color: SOURCE[key].color,
    width: key === "fit" ? 2.7 : 1.8,
    dash: key === "cosmos" ? [7, 4] : undefined,
    marker: key === "euclid" ? "ring" : undefined,
    dots: key === "euclid",
    markerEvery: Math.max(1, Math.ceil(curve.x.length / 18)),
  }));
  return <>
    <Plot
      xDomain={xDomain} yDomain={yDomain}
      xTicks={logarithmicX ? physicalLogTicks(xDomain) : ticks(xDomain)}
      yTicks={physicalLogTicks(yDomain)}
      xLabel={logarithmicX ? physicalLogAxisLabel(parameter.x_label) : parameter.x_label}
      yLabel={`${parameter.density_unit} (log scale)`}
      series={series} aspect={0.62}
    />
    <div className="galaxy-plot__definitions">
      {curves.map(([key, curve]) => <span key={key}><i style={{ background: SOURCE[key].color }} />{SOURCE[key].label}: {curve.definition}</span>)}
    </div>
  </>;
}

function ApparentBrightnessPlot({ parameter }: { parameter: Parameter }) {
  const entries = Object.entries(parameter.photometry_series ?? {});
  const [selected, setSelected] = useState<string[]>(() => entries
    .filter(([, curve]) => curve.default_on)
    .map(([key]) => key));
  const colorByKey = useMemo(() => {
    const indices = { euclid: 0, cosmos: 0, fit: 0 };
    return Object.fromEntries(entries.map(([key, curve]) => {
      const palette = BRIGHTNESS_COLORS[curve.survey];
      const color = palette[indices[curve.survey] % palette.length];
      indices[curve.survey] += 1;
      return [key, color];
    }));
  }, [parameter.photometry_series]);
  const visible = entries.filter(([key]) => selected.includes(key));
  const logValues = visible.flatMap(([, curve]) => curve.density
    .filter((value) => value > 0).map(Math.log10));
  const xValues = visible.flatMap(([, curve]) => curve.x);
  const toggle = (key: string) => setSelected((current) => (
    current.includes(key) ? current.filter((item) => item !== key) : [...current, key]
  ));
  const surveyEntries = (survey: BrightnessCurve["survey"]) => entries
    .filter(([, curve]) => curve.survey === survey);

  let plot = <Empty>Select at least one catalogue measurement to draw.</Empty>;
  if (visible.length && logValues.length && xValues.length) {
    const xDomain: [number, number] = [Math.min(...xValues), Math.max(...xValues)];
    const lo = Math.floor(Math.min(...logValues) * 2) / 2;
    const hi = Math.ceil(Math.max(...logValues) * 2) / 2;
    const yDomain: [number, number] = [lo, hi <= lo ? lo + 1 : hi];
    const guides = visible.flatMap(([, curve]) => [
      ...(curve.fit_interval ?? []).map((value) => ({
        axis: "x" as const, v: value, color: "#2478d4",
        dash: [3, 4], width: 1, alpha: 0.65,
      })),
      ...(curve.extrapolated_interval ? [{
        axis: "x" as const, v: curve.extrapolated_interval[0],
        color: "#e25543", dash: [7, 4], width: 1.2, alpha: 0.75,
      }] : []),
    ]);
    plot = <>
      <Plot
        xDomain={xDomain} yDomain={yDomain}
        xTicks={ticks(xDomain, 6)} yTicks={physicalLogTicks(yDomain, 6)}
        xLabel={parameter.x_label}
        yLabel={`${parameter.density_unit} (log scale)`}
        guides={guides}
        series={visible.map(([key, curve]) => ({
          x: curve.x,
          y: curve.density.map((value) => value > 0 ? Math.log10(value) : null),
          color: colorByKey[key],
          width: curve.survey === "fit" ? 2.7 : 2.0,
          dash: curve.survey === "cosmos" ? [7, 4]
            : curve.survey === "fit" ? [3, 3] : undefined,
        }))}
        aspect={0.54}
      />
      <div className="brightness-disclosure">
        {visible.map(([key, curve]) => <div key={key}>
          <i style={{ background: colorByKey[key] }} />
          <span><b>{curve.label}</b><small>{curve.band} · {curve.estimator}</small></span>
          <em>{curve.fit_interval
            ? `fit ${curve.fit_interval[0].toFixed(2)}–${curve.fit_interval[1].toFixed(2)} · sample ${curve.sampling_interval?.[0].toFixed(0)}–${curve.sampling_interval?.[1].toFixed(0)}`
            : `${compact(curve.weighted_count)} weighted objects`}</em>
        </div>)}
      </div>
    </>;
  }

  return <div className="brightness-comparison">
    <div className="brightness-warning">
      <strong>Same AB convention, different measurements.</strong>
      <span>The default is the Q1 MER + PHZ VIS 2FWHM raw count and its straight log-density law over 14–29; 28–29 is explicit extrapolation. Optional VIS/F814W diagnostics retain their native estimators.</span>
    </div>
    <div className="brightness-controls">
      {(["euclid", "cosmos", "fit"] as const).map((survey) => {
        const group = surveyEntries(survey);
        if (!group.length) return null;
        return <section key={survey}>
          <header>
            <div><span>{survey === "euclid" ? "Euclid MER" : survey === "cosmos" ? "COSMOS2025" : "Q1 curve fits"}</span><small>{survey === "euclid" ? "VIS · solid measurements" : survey === "cosmos" ? "HST/ACS F814W · long dashes" : "VIS · short-dashed local fits"}</small></div>
            <div>
              <Button size="sm" variant="ghost" onClick={() => setSelected((current) => Array.from(new Set([...current, ...group.map(([key]) => key)])))}>all</Button>
              <Button size="sm" variant="ghost" onClick={() => setSelected((current) => current.filter((key) => !group.some(([candidate]) => candidate === key)))}>none</Button>
            </div>
          </header>
          <div className="brightness-controls__chips">{group.map(([key, curve]) => <Chip
            key={key} on={selected.includes(key)} onClick={() => toggle(key)}
            dot={colorByKey[key]}
            title={`${curve.band}; ${curve.estimator}. Selection: ${curve.selection}`}
          >{curve.label}</Chip>)}</div>
          {survey === "cosmos" && parameter.photometry_missing?.map((message) =>
            <p className="brightness-controls__missing" key={message}>{message}</p>)}
        </section>;
      })}
    </div>
    <div className="brightness-plot">{plot}</div>
  </div>;
}

function RadiusPlot({ parameter }: { parameter: Parameter }) {
  const entries = Object.entries(parameter.radius_series ?? {});
  const [selected, setSelected] = useState<string[]>(() => entries
    .filter(([, curve]) => curve.default_on)
    .map(([key]) => key));
  const visible = entries.filter(([key]) => selected.includes(key));
  const toggle = (key: string) => setSelected((current) => (
    current.includes(key) ? current.filter((item) => item !== key) : [...current, key]
  ));
  const grouped = (["detection", "kron", "half_light"] as RadiusCurve["radius_type"][])
    .map((radiusType) => [radiusType, entries.filter(([, curve]) => curve.radius_type === radiusType)] as const)
    .filter(([, group]) => group.length);
  const logValues = visible.flatMap(([, curve]) => curve.density
    .filter((value) => value > 0).map(Math.log10));
  const xValues = visible.flatMap(([, curve]) => curve.x);

  let plot = <Empty>Select at least one radius observable to draw.</Empty>;
  if (visible.length && logValues.length && xValues.length) {
    const xDomain: [number, number] = parameter.x_domain
      ? [parameter.x_domain[0], parameter.x_domain[1]]
      : [Math.min(...xValues), Math.max(...xValues)];
    const lo = Math.floor(Math.min(...logValues) * 2) / 2;
    const hi = Math.ceil(Math.max(...logValues) * 2) / 2;
    const yDomain: [number, number] = [lo, hi <= lo ? lo + 1 : hi];
    plot = <>
      <Plot
        xDomain={xDomain} yDomain={yDomain}
        xTicks={physicalLogTicks(xDomain, 6)} yTicks={physicalLogTicks(yDomain, 6)}
        xLabel={physicalLogAxisLabel(parameter.x_label)}
        yLabel={`${parameter.density_unit} (log scale)`}
        series={visible.map(([key, curve]) => ({
          x: curve.x,
          y: curve.density.map((value) => value > 0 ? Math.log10(value) : null),
          color: RADIUS_COLORS[key] ?? SOURCE[curve.source].color,
          width: curve.source === "fit" ? 2.7 : 1.9,
          dash: curve.source === "cosmos" ? [7, 4] : undefined,
          marker: curve.source === "euclid" ? "ring" : undefined,
          dots: curve.source === "euclid",
          markerEvery: Math.max(1, Math.ceil(curve.x.length / 18)),
        }))}
        aspect={0.58}
      />
      <div className="radius-disclosure">
        {visible.map(([key, curve]) => <div key={key}>
          <i style={{ background: RADIUS_COLORS[key] ?? SOURCE[curve.source].color }} />
          <span><b>{curve.label}</b><small>{curve.definition}</small></span>
          <em>{compact(curve.weighted_count)} weighted objects</em>
        </div>)}
      </div>
    </>;
  }

  return <div className="radius-comparison">
    <div className="radius-warning">
      <strong>Catalogue size concepts, kept separate.</strong>
      <span>The generator fits only PHZ/MER VIS Sérsic Rₑ jointly with VIS 2FWHM brightness. Detection, Kron, and COSMOS curves are labeled diagnostics and do not affect sampling.</span>
    </div>
    <div className="radius-controls">
      {grouped.map(([radiusType, group]) => <section key={radiusType}>
        <header>
          <div><span>{RADIUS_TYPE_LABEL[radiusType]}</span><small>{radiusType === "detection" ? "diagnostic only" : radiusType === "kron" ? "diagnostic only" : "Euclid data + Euclid fit; COSMOS diagnostic"}</small></div>
          <div>
            <Button size="sm" variant="ghost" onClick={() => setSelected((current) => Array.from(new Set([...current, ...group.map(([key]) => key)])))}>all</Button>
            <Button size="sm" variant="ghost" onClick={() => setSelected((current) => current.filter((key) => !group.some(([candidate]) => candidate === key)))}>none</Button>
          </div>
        </header>
        <div className="radius-controls__chips">{group.map(([key, curve]) => <Chip
          key={key} on={selected.includes(key)} onClick={() => toggle(key)}
          dot={RADIUS_COLORS[key] ?? SOURCE[curve.source].color}
          title={`${curve.units}; ${curve.definition}`}
        >{curve.label}</Chip>)}</div>
        {radiusType === "half_light" && parameter.radius_missing?.map((message) =>
          <p className="brightness-controls__missing" key={message}>{message}</p>)}
      </section>)}
    </div>
    <div className="radius-plot">{plot}</div>
  </div>;
}

function ApertureLadder({ data }: { data?: ApertureScatter }) {
  const [flux, setFlux] = useState<FluxKey>("f4");
  const [growth, setGrowth] = useState<GrowthKey[]>(["g1", "g2", "g3"]);
  const selected = (["g1", "g2", "g3"] as GrowthKey[]).filter((key) => growth.includes(key));
  const x = data?.magnitudes[flux] ?? [];
  const xDomain = paddedDomain(x, 1);
  const yValues = selected.flatMap((key) => data?.growth[key] ?? []);
  const yDomain = paddedDomain(yValues, 0.1);
  const toggle = (key: GrowthKey) => setGrowth((current) => (
    current.includes(key) ? current.filter((item) => item !== key) : [...current, key]
  ));
  const series: Series[] = selected.map((key) => ({
    x,
    y: data?.growth[key] ?? [],
    color: GROWTH[key].color,
    mode: "scatter",
    marker: "filled",
    width: 0.7,
    alpha: 0.24,
  }));

  return <section className="aperture-ladder" aria-labelledby="aperture-ladder-title">
    <header className="aperture-ladder__head">
      <div>
        <div className="eyebrow">VIS aperture ladder · catalogue galaxies</div>
        <h2 id="aperture-ladder-title">How the enclosed-light curve changes with brightness</h2>
        <p>Choose the aperture magnitude on the horizontal axis, then overlay any combination of flux-growth ratios.</p>
      </div>
      <Badge tone={data?.count ? "good" : "warn"}>{compact(data?.count)} plotted galaxies</Badge>
    </header>
    <div className="aperture-ladder__controls">
      <div className="aperture-ladder__control">
        <span>magnitude axis</span>
        <Segmented<FluxKey> value={flux} onChange={setFlux} options={([
          "f1", "f2", "f3", "f4",
        ] as FluxKey[]).map((value) => ({ value, label: value.toUpperCase() }))} />
      </div>
      <div className="aperture-ladder__control aperture-ladder__growth-controls">
        <span>growth ratios</span>
        <div>
          {(["g1", "g2", "g3"] as GrowthKey[]).map((key) => <Chip
            key={key} on={growth.includes(key)} onClick={() => toggle(key)}
            dot={GROWTH[key].color} title={data?.definitions[key] ?? GROWTH[key].label}
          >{GROWTH[key].label}</Chip>)}
        </div>
      </div>
    </div>
    {!data?.count ? <Empty>Build the density plots after refreshing the Euclid catalogue with F1–F4 aperture fluxes.</Empty>
      : !selected.length ? <Empty>Select at least one growth ratio to draw.</Empty>
      : <div className="aperture-ladder__plot">
        <Plot
          xDomain={xDomain} yDomain={yDomain}
          xTicks={ticks(xDomain, 6)} yTicks={ticks(yDomain, 6)}
          xLabel={`VIS ${flux.toUpperCase()} aperture magnitude (AB)`}
          yLabel="aperture AB magnitude difference to F₄"
          series={series} aspect={0.48}
          guides={[{ axis: "y", v: 0, color: "#7e8da5", dash: [5, 5], alpha: 0.75 }]}
        />
      </div>}
    <footer className="aperture-ladder__foot">
      <span>More positive means more light lies outside that aperture.</span>
      {data?.selection && <span>{data.selection}</span>}
    </footer>
  </section>;
}

export default function GalaxyDistributionsPage() {
  const resource = useResource<Payload>("/api/galaxy-distributions", [], { ttl: 10_000 });
  const q1Query = useJob();
  const fit = useJob();
  const build = useJob();
  const api = resource.data;
  const refresh = (job: { status: string }) => { if (job.status !== "failed") resource.reload(); };
  const rebuildAfter = (job: { status: string }) => {
    if (job.status !== "failed") {
      build.run("/api/galaxy-distributions/build", {}, { onDone: refresh });
    }
  };
  const rebuildAfterCounts = (job: { status: string }) => {
    resource.reload();
    rebuildAfter(job);
  };
  useEffect(() => {
    if (!q1Query.busy) return;
    resource.reload();
    const timer = window.setInterval(resource.reload, 1500);
    return () => window.clearInterval(timer);
  }, [q1Query.busy, resource.reload]);
  const parameters = useMemo(() => api ? PARAMETER_ORDER.flatMap((key) => (
    api.parameters[key] ? [[key, api.parameters[key]] as const] : []
  )) : [], [api]);
  const q1FitReady = Boolean(api?.q1_counts && (
    api.q1_counts.fit_ready
    || Object.values(api.q1_counts.apertures).some((aperture) => (
      (aperture.queried_bins ?? 0) >= 4
    ))
  ));

  if (resource.loading && !api) return <Page><Empty><Spinner /> reading galaxy distributions…</Empty></Page>;
  if (!api) return <Page><Empty>Galaxy-distribution status is unavailable.</Empty></Page>;

  return <Page>
    <PageHead
      eyebrow="population laboratory · observed → latent → generated"
      title="Galaxy distributions"
      sub="Query the catalogues, rebuild the analytical fit, and compare surface density against the same physical and observational coordinates."
      right={<Badge tone={api.stale ? "warn" : "good"}>{api.stale ? "plots need rebuild" : "plots current"}</Badge>}
    />

    <SourceLedger sources={api.sources} />

    <Card className="galaxy-q1-counts galaxy-actions">
      <CardHead
        title="Q1 bright galaxies · data and plot controls"
        sub="One run checkpoints progressive 0.1-mag MER + PHZ aperture counts and rebuilds these plots. No cone sampling is required."
      />
      <CardBody>
        <div className="galaxy-q1-counts__content">
          <div className="galaxy-q1-counts__stats">
            <Stat k="Q1 footprint" v={`${(api.q1_counts?.footprint_area_deg2 ?? 63.1).toFixed(1)} deg²`} />
            <Stat k="VIS range" v={api.q1_counts ? `${api.q1_counts.bright.toFixed(1)}–${api.q1_counts.faint.toFixed(1)}` : "14.0–28.0"} />
            <Stat k="bin width" v={`${(api.q1_counts?.bin_width ?? 0.1).toFixed(1)} mag`} />
            <Stat k="checkpoints" v={`${api.q1_counts?.completed_queries ?? api.q1_counts?.query_count ?? 0}/${api.q1_counts?.total_queries ?? 560}`} />
            <Stat k="passes" v={`${api.q1_counts?.phases_completed ?? 0}/${api.q1_counts?.phase_count ?? 5}`} />
            <Stat k="F₁ PHZ weight" v={compact(api.q1_counts?.apertures.f1.expected_galaxies)} />
            <Stat k="F₄ PHZ weight" v={compact(api.q1_counts?.apertures.f4.expected_galaxies)} />
          </div>
        </div>
        <div className="galaxy-q1-phases" aria-label="Progressive magnitude-bin sampling phases">
          {Array.from({ length: api.q1_counts?.phase_count ?? 5 }, (_, index) => {
            const completed = index < (api.q1_counts?.phases_completed ?? 0);
            const active = q1Query.busy && index === (api.q1_counts?.phases_completed ?? 0);
            return <div
              className={`galaxy-q1-phases__step${completed ? " is-complete" : ""}${active ? " is-active" : ""}`}
              key={index}
            >
              <span>phase {index + 1}</span>
              <b>+{(index / 10).toFixed(1)} mag</b>
              <small>{completed ? "cached" : active ? "querying" : "waiting"}</small>
            </div>;
          })}
        </div>
        <div className="galaxy-actions__row">
          <div className="galaxy-actions__buttons">
            <Button
              variant="primary"
              disabled={!api.authenticated || q1Query.busy || build.busy}
              onClick={() => q1Query.run(
                "/api/galaxy-distributions/query-q1-counts",
                {},
                { onDone: rebuildAfterCounts },
              )}
            >
              {q1Query.busy ? "Querying progressive bins…" : build.busy ? "Building plots…" : "Query MER + PHZ"}
            </Button>
            <Button disabled={fit.busy || build.busy || q1Query.busy || !q1FitReady} onClick={() => fit.run(
              "/api/galaxy-distributions/fit-q1-counts", {}, { onDone: rebuildAfter },
            )}>{fit.busy ? "Fitting aperture curves…" : "Fit cached aperture curves"}</Button>
            <Button disabled={build.busy} onClick={() => build.run(
              "/api/galaxy-distributions/build", {}, { onDone: refresh },
            )}>{build.busy ? "Building plots…" : "Build density plots"}</Button>
            <Button variant="ghost" onClick={resource.reload}>Refresh view</Button>
            {!api.authenticated && <NavLink className="ui-btn" to="/catalog">Log in to Euclid archive</NavLink>}
          </div>
        </div>
        <p className="galaxy-q1-counts__note">
          <strong>Progressive cache:</strong> exact 0.1-mag bins are queried at 0.5-mag spacing first,
          then revisited at offsets of 0.1, 0.2, 0.3, and 0.4 mag. Each F₁–F₄ result
          is stored immediately and is skipped on later runs. Sources must have POINT_LIKE_FLAG
          unset and PHZ_GAL_PROB ≥ 0.5; density remains probability-weighted.
        </p>
        <JobProgressView job={q1Query.job} error={q1Query.error} />
        <JobProgressView job={fit.job} error={fit.error} />
        <JobProgressView job={build.job} error={build.error} />
      </CardBody>
    </Card>

    <ApertureLadder data={api.sources.euclid?.aperture_scatter} />

    <section className="galaxy-density-section">
      <header className="galaxy-density-section__head">
        <div>
          <div className="eyebrow">surface-density marginals</div>
          <h2>Where the three population layers agree—and where they do not</h2>
          <p>Every vertical axis is a sky density. Log scaling keeps faint and bright populations legible on the same panel.</p>
        </div>
        <div className="galaxy-key">
          {ORDER.map((key) => <span key={key}><i style={{ background: SOURCE[key].color }} />{SOURCE[key].label}</span>)}
        </div>
      </header>
      <div className="galaxy-plot-grid">
        {parameters.map(([key, parameter]) => <article className={`galaxy-plot${key === "magnitude" ? " galaxy-plot--brightness" : ""}`} key={key}>
          <header><span>{parameter.label}</span><small>{parameter.note}</small></header>
          {key === "magnitude"
            ? <ApparentBrightnessPlot
                key={Object.keys(parameter.photometry_series ?? {}).join("|")}
                parameter={parameter}
              />
            : key === "radius"
              ? <RadiusPlot
                  key={Object.keys(parameter.radius_series ?? {}).join("|")}
                  parameter={parameter}
                />
            : <DensityPlot parameter={parameter} />}
        </article>)}
      </div>
    </section>
  </Page>;
}
