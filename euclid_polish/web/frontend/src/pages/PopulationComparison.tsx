import { useEffect, useMemo, useState } from "react";
import { NavLink } from "react-router-dom";
import Plot, { Legend, type Series, type Tick } from "../charts/Plot";
import { C, categorical } from "../colors";
import { useResource } from "../hooks";
import { JobProgressView, useJob } from "../jobs";
import {
  Badge, Button, Card, CardBody, CardHead, Empty, Field, Input, Page,
  PageHead, Spinner, Stat,
} from "../ui";
import "./population-comparison.css";

type Band = "VIS" | "Y_E" | "J_E" | "H_E";
type Curve = { p16: (number | null)[]; median: (number | null)[]; p84: (number | null)[] };
type FieldComparison = {
  bands: Band[];
  histograms: Record<Band, {
    x: number[]; synthetic: number[]; real: number[];
    x_label: string; y_label: string;
  }>;
  power: Record<Band, {
    k: number[]; synthetic: Curve; real: Curve;
    x_label: string; y_label: string;
  }>;
  mean_cross_correlation: Record<Band, {
    k: number[]; r: (number | null)[]; x_label: string; y_label: string;
  }>;
  summary: Record<"synthetic" | "real", Record<Band, {
    mean: number; median: number; p16: number; p84: number;
  }>>;
};
type Histogram = {
  x: number[]; density: number[]; count: number; range?: [number, number];
};
type Parameter = {
  label: string;
  unit: string;
  series: Record<string, Histogram>;
};
type Population = {
  objects: number;
  counts: Record<string, number>;
  density_arcmin2: Record<string, number | null>;
  area_arcmin2: number;
  parameters: Record<string, Parameter>;
};
type Comparison = {
  version: number;
  geometry: {
    tile_size: number; analysis_size: number;
    pixel_scale_arcsec: number; field_area_arcmin2: number;
  };
  samples: {
    synthetic: { fields: number; area_arcmin2: number; splits: string[] };
    real: {
      fields: number; area_arcmin2: number;
      inference_fields: number; jwst_overlap_fields: number;
    };
  };
  fields: FieldComparison;
  population: {
    synthetic: Population;
    synthetic_field_count: number;
    euclid: Population | null;
    euclid_meta: {
      ra: number; dec: number; radius_arcmin: number; area_arcmin2: number;
      rows: number; limit: number; limit_reached: boolean; classification: string;
    } | null;
  };
};
type Availability = {
  synthetic: {
    fields: number; area_arcmin2: number; record_files: number;
    source_catalogs: number; train_source_catalog: boolean;
    population_fields: number; population_area_arcmin2: number;
  };
  real: {
    fields: number; area_arcmin2: number;
    inference_fields: number; jwst_overlap_fields: number;
  };
  euclid_catalog: { cached: boolean; meta: Comparison["population"]["euclid_meta"] };
  field_area_arcmin2: number;
  default_cone: { ra: number; dec: number; radius_arcmin: number; area_arcmin2: number };
};
type ApiPayload = {
  comparison: Comparison | null;
  availability: Availability;
  authenticated: boolean;
};

const BANDS: Band[] = ["VIS", "Y_E", "J_E", "H_E"];
const TYPE_ORDER = ["galaxy", "star", "unknown"];
const BAND_COLOR = (band: Band) => categorical(BANDS.indexOf(band));
const bandLabel = (band: Band) => band.replace("_E", "");

function finite(values: readonly (number | null)[]): number[] {
  return values.filter((value): value is number => value != null && Number.isFinite(value));
}

function domain(values: readonly (number | null)[], includeZero = false): [number, number] {
  const good = finite(values);
  if (!good.length) return [0, 1];
  let lo = Math.min(...good), hi = Math.max(...good);
  if (includeZero) lo = Math.min(0, lo);
  if (lo === hi) return [lo - 0.5, hi + 0.5];
  const pad = (hi - lo) * 0.06;
  return [includeZero ? 0 : lo - pad, hi + pad];
}

function ticks([lo, hi]: [number, number], count = 5): Tick[] {
  return Array.from({ length: count }, (_, index) => {
    const value = lo + (hi - lo) * index / (count - 1);
    const abs = Math.abs(value);
    const label = abs >= 1000 || (abs > 0 && abs < 0.01)
      ? value.toExponential(1) : Number(value.toPrecision(3)).toString();
    return { v: value, label };
  });
}

function log10(values: readonly (number | null)[]): (number | null)[] {
  return values.map((value) =>
    value != null && value > 0 && Number.isFinite(value) ? Math.log10(value) : null);
}

function DatasetRuler({ availability, comparison }: {
  availability: Availability; comparison: Comparison | null;
}) {
  const synthetic = comparison?.samples.synthetic.fields ?? availability.synthetic.fields;
  const real = comparison?.samples.real.fields ?? availability.real.fields;
  const max = Math.max(synthetic, real, 1);
  const inference = comparison?.samples.real.inference_fields ?? availability.real.inference_fields;
  const overlap = comparison?.samples.real.jwst_overlap_fields ?? availability.real.jwst_overlap_fields;
  return (
    <section className="survey-ruler" aria-label="Compared field census">
      <div className="survey-ruler__axis">
        <span>same footprint</span>
        <strong>{comparison?.geometry.tile_size ?? 256} px</strong>
        <span>× {comparison?.geometry.pixel_scale_arcsec ?? 0.1}″ px⁻¹</span>
        <i />
        <span>{availability.field_area_arcmin2.toFixed(3)} arcmin² / field</span>
      </div>
      <div className="survey-ruler__row survey-ruler__row--synthetic">
        <div className="survey-ruler__label">
          <strong>Synthetic LR</strong>
          <span>test + validate</span>
        </div>
        <div className="survey-ruler__track">
          <span style={{ width: `${100 * synthetic / max}%` }} />
        </div>
        <div className="survey-ruler__value">
          <strong>{synthetic}</strong><span>fields</span>
        </div>
      </div>
      <div className="survey-ruler__row survey-ruler__row--real">
        <div className="survey-ruler__label">
          <strong>Real Euclid LR</strong>
          <span>{inference} inference + {overlap} JWST-overlap</span>
        </div>
        <div className="survey-ruler__track">
          <span style={{ width: `${100 * real / max}%` }} />
        </div>
        <div className="survey-ruler__value">
          <strong>{real}</strong><span>fields</span>
        </div>
      </div>
      <div className="survey-ruler__scale mono">
        <span>0</span><span>{Math.round(max / 2)}</span><span>{max} fields</span>
      </div>
    </section>
  );
}

function FieldPlots({ comparison }: { comparison: Comparison }) {
  const histograms = BANDS.map((band) => [band, comparison.fields.histograms[band]] as const);
  const powers = BANDS.map((band) => [band, comparison.fields.power[band]] as const);
  const crosses = BANDS.map((band) => [band, comparison.fields.mean_cross_correlation[band]] as const);
  const histogramX = domain(histograms.flatMap(([, histogram]) => histogram.x));
  const histogramY = domain(histograms.flatMap(([, histogram]) => [
    ...histogram.synthetic, ...histogram.real,
  ]), true);
  const powerY = domain(powers.flatMap(([, power]) => [
    ...log10(power.synthetic.median), ...log10(power.real.median),
  ]));
  const allPowerK = powers.flatMap(([, power]) => power.k);
  const powerX: [number, number] = [Math.min(...allPowerK), Math.max(...allPowerK)];
  const crossY: [number, number] = [-1, 1];
  const histogramSeries: Series[] = histograms.flatMap(([band, histogram]) => [
    {
      x: histogram.x, y: histogram.synthetic, color: BAND_COLOR(band),
      mode: "histogram", width: 1.25, alpha: 0.72, fillAlpha: 0.13,
    },
    {
      x: histogram.x, y: histogram.real, color: BAND_COLOR(band),
      mode: "histogram", width: 1.4, dash: [4, 3], alpha: 0.95, fillAlpha: 0.025,
    },
  ]);
  const powerSeries: Series[] = powers.flatMap(([band, power]) => [
    { x: power.k, y: log10(power.synthetic.median), color: BAND_COLOR(band), width: 2.4 },
    { x: power.k, y: log10(power.real.median), color: BAND_COLOR(band), width: 2.2, dash: [6, 4] },
  ]);
  const crossSeries: Series[] = crosses.map(([band, cross]) => ({
    x: cross.k, y: cross.r, color: BAND_COLOR(band), width: 2.4,
  }));
  const bandLegend = BANDS.map((band) => ({
    color: BAND_COLOR(band), label: bandLabel(band),
  }));
  const sourceLegend = [
    { color: C.cross, label: "synthetic LR", dash: false },
    { color: C.cross, label: "real Euclid LR", dash: true },
  ];
  const histogramBandLegend = BANDS.map((band) => ({
    color: BAND_COLOR(band), label: bandLabel(band), histogram: true,
  }));
  const histogramSourceLegend = [
    {
      color: C.cross, label: "synthetic LR",
      histogram: true, filled: true, dash: false,
    },
    {
      color: C.cross, label: "real Euclid LR",
      histogram: true, filled: false, dash: true,
    },
  ];

  return (
    <section className="comparison-section">
      <header className="comparison-section__head">
        <div>
          <div className="eyebrow">image domain · one field at a time</div>
          <h2>Pixel statistics</h2>
          <p>VIS, Y, J and H share each plot; every histogram and curve uses the same 255×255 center crop and native 0.1″ pixel scale.</p>
        </div>
      </header>
      <div className="field-plot-grid">
        <Card className="comparison-plot">
          <CardHead title="Brightness distribution"
            sub="Normalized histogram over the full shared range; no pixels are clipped from the bright or negative tails." />
          <CardBody>
            <Plot xDomain={histogramX} yDomain={histogramY}
              xTicks={ticks(histogramX)} yTicks={ticks(histogramY)}
              xLabel="pixel brightness (e⁻ / stack)"
              yLabel="fraction of sampled pixels / bin"
              series={histogramSeries} />
            <Legend items={histogramBandLegend} />
            <Legend items={histogramSourceLegend} />
          </CardBody>
        </Card>

        <Card className="comparison-plot">
          <CardHead title="Angular power"
            sub="Median mean-subtracted field power; solid is synthetic and dashed is real Euclid." />
          <CardBody>
            <Plot xDomain={powerX} yDomain={powerY} xScale="log"
              xTicks={powers[0][1].k.filter((_, index) => index % 5 === 0)
                .map((value) => ({ v: value, label: value.toPrecision(2) }))}
              yTicks={ticks(powerY).map((tick) => ({ ...tick, label: `10^${tick.label}` }))}
              xLabel="angular frequency (cycles / arcsec)"
              yLabel="log₁₀ mean-subtracted power (e⁻²)"
              series={powerSeries} />
            <Legend items={bandLegend} />
            <Legend items={sourceLegend} />
          </CardBody>
        </Card>

        <Card className="comparison-plot comparison-plot--cross">
          <CardHead title="Population-mean cross-correlation"
            sub="Fourier coherence between the synthetic mean LR field and the real mean LR field; no HR reference." />
          <CardBody>
            <Plot xDomain={[Math.min(...crosses[0][1].k), Math.max(...crosses[0][1].k)]}
              yDomain={crossY} xScale="log"
              xTicks={crosses[0][1].k.filter((_, index) => index % 5 === 0)
                .map((value) => ({ v: value, label: value.toPrecision(2) }))}
              yTicks={[-1, -0.5, 0, 0.5, 1].map((v) => ({ v, label: String(v) }))}
              guides={[{ axis: "y", v: 0, dash: [4, 4] }]}
              xLabel="angular frequency (cycles / arcsec)"
              yLabel="mean-field coherence r(k)"
              series={crossSeries} />
            <Legend items={bandLegend} />
          </CardBody>
        </Card>
      </div>

      <div className="band-ledger">
        <div className="band-ledger__head">field-mean brightness · e⁻ / pixel</div>
        {BANDS.map((item) => {
          const synthetic = comparison.fields.summary.synthetic[item];
          const real = comparison.fields.summary.real[item];
          return (
            <div className="band-ledger__row" key={item}>
              <strong>{item}</strong>
              <span><i className="dot dot--synthetic" />synthetic {synthetic.median.toPrecision(4)}</span>
              <span><i className="dot dot--real" />real {real.median.toPrecision(4)}</span>
              <span className="mono">real / synthetic {synthetic.median !== 0
                ? (real.median / synthetic.median).toFixed(3) : "—"}</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function PopulationSummary({ title, eyebrow, population, tone }: {
  title: string; eyebrow: string; population: Population; tone: "synthetic" | "real";
}) {
  return (
    <Card className={`population-summary population-summary--${tone}`}>
      <CardHead title={title} sub={`${population.area_arcmin2.toFixed(2)} arcmin² sampled`} />
      <CardBody>
        <div className="population-summary__eyebrow eyebrow">{eyebrow}</div>
        <div className="population-summary__stats">
          <Stat k="catalog objects" v={population.objects.toLocaleString()} />
          {TYPE_ORDER.filter((kind) => kind in population.counts).map((kind) => (
            <Stat key={kind} k={`${kind} / arcmin²`}
              v={(population.density_arcmin2[kind] ?? 0).toFixed(2)} />
          ))}
        </div>
      </CardBody>
    </Card>
  );
}

function ParameterPlot({ parameter, source }: { parameter: Parameter; source: string }) {
  const entries = TYPE_ORDER
    .filter((kind) => parameter.series[kind]?.x.length)
    .map((kind) => [kind, parameter.series[kind]] as const);
  const xs = entries.flatMap(([, histogram]) => histogram.x);
  const ys = entries.flatMap(([, histogram]) => histogram.density);
  const xDomain = domain(xs);
  const yDomain = domain(ys, true);
  const series: Series[] = entries.map(([kind, histogram], index) => ({
    x: histogram.x,
    y: histogram.density,
    color: categorical(index + (source === "Euclid catalog" ? 4 : 0)),
    mode: "histogram",
    width: 1.3,
    alpha: 0.86,
    fillAlpha: 0.22,
  }));
  return (
    <Card className="parameter-card">
      <CardHead title={parameter.label}
        sub={`${source} · ${entries.map(([kind, histogram]) =>
          `${kind} n=${histogram.count.toLocaleString()}`).join(" · ")}`} />
      <CardBody>
        <Plot xDomain={xDomain} yDomain={yDomain}
          xTicks={ticks(xDomain, 4)} yTicks={ticks(yDomain, 4)}
          xLabel={parameter.unit} yLabel="fraction / bin"
          series={series} aspect={0.62} />
        <Legend items={entries.map(([kind], index) => ({
          color: categorical(index + (source === "Euclid catalog" ? 4 : 0)),
          label: kind,
          histogram: true,
          filled: true,
        }))} />
      </CardBody>
    </Card>
  );
}

function ParameterAtlas({ population, source }: { population: Population; source: string }) {
  return (
    <div className="parameter-atlas">
      {Object.entries(population.parameters).map(([key, parameter]) => (
        <ParameterPlot key={`${source}-${key}`} parameter={parameter} source={source} />
      ))}
    </div>
  );
}

function ConeQuery({ api, onQueried }: { api: ApiPayload; onQueried: () => void }) {
  const defaults = api.availability.default_cone;
  const [ra, setRa] = useState(String(defaults.ra));
  const [dec, setDec] = useState(String(defaults.dec));
  const [radius, setRadius] = useState(defaults.radius_arcmin.toFixed(3));
  const query = useJob();
  const radiusNumber = Number(radius);
  const area = Number.isFinite(radiusNumber) ? Math.PI * radiusNumber ** 2 : 0;
  return (
    <Card className="cone-query">
      <CardHead title="Euclid population cone"
        sub="Query clean MER sources, classify point-like rows as stars, and normalize every count by sky area."
        right={<Badge tone={api.authenticated ? "good" : "warn"}>
          {api.authenticated ? "archive session ready" : "archive login required"}
        </Badge>} />
      <CardBody>
        <div className="cone-query__form">
          <Field label="RA · deg"><Input value={ra} type="number" onChange={setRa} step={0.0001} /></Field>
          <Field label="Dec · deg"><Input value={dec} type="number" onChange={setDec} step={0.0001} /></Field>
          <Field label="Radius · arcmin"><Input value={radius} type="number"
            onChange={setRadius} min={0.01} max={30} step={0.01} /></Field>
          <div className="cone-query__area">
            <span className="eyebrow">cone area</span>
            <strong>{area.toFixed(2)} arcmin²</strong>
            <small>{Math.abs(area - defaults.area_arcmin2) < 0.1
              ? "matches the 200-field synthetic footprint"
              : `${(area / defaults.area_arcmin2).toFixed(2)}× synthetic footprint`}</small>
          </div>
          <Button variant="primary" disabled={!api.authenticated || query.busy}
            onClick={() => query.run("/api/population-comparison/query-euclid", {
              ra, dec, radius_arcmin: radius,
            }, { onDone: (job) => { if (job.status !== "failed") onQueried(); } })}>
            {query.busy ? "Querying…" : "Query Euclid cone"}
          </Button>
        </div>
        {!api.authenticated && (
          <p className="cone-query__login">
            Open <NavLink to="/catalog">Catalog</NavLink> and log in to the Euclid archive first.
          </p>
        )}
        <JobProgressView job={query.job} error={query.error} />
      </CardBody>
    </Card>
  );
}

export default function PopulationComparisonPage() {
  const resource = useResource<ApiPayload>("/api/population-comparison", [], { ttl: 10_000 });
  const build = useJob();
  const trainingCatalog = useJob();
  const api = resource.data;
  const comparison = api?.comparison ?? null;
  const stale = !!comparison && !!api && (
    comparison.samples.synthetic.fields !== api.availability.synthetic.fields
    || comparison.samples.real.fields !== api.availability.real.fields
    || comparison.population.synthetic_field_count !== api.availability.synthetic.population_fields
    || (!!comparison.population.euclid) !== api.availability.euclid_catalog.cached
  );

  const rebuild = () => build.run("/api/population-comparison/build", {}, {
    onDone: (job) => { if (job.status !== "failed") resource.reload(); },
  });
  const syncTrainingCatalog = () => trainingCatalog.run(
    "/api/population-comparison/sync-training-catalog",
    {},
    { onDone: (job) => {
      if (job.status !== "failed") {
        resource.reload();
        rebuild();
      }
    } },
  );
  const sections = useMemo(() => {
    if (!comparison) return [];
    const value: { title: string; source: string; population: Population }[] = [
      { title: "Synthetic truth parameters", source: "Synthetic source truth", population: comparison.population.synthetic },
    ];
    if (comparison.population.euclid) value.push({
      title: "Euclid catalog parameters",
      source: "Euclid catalog",
      population: comparison.population.euclid,
    });
    return value;
  }, [comparison]);

  useEffect(() => {
    if (api?.availability.euclid_catalog.cached && comparison && !comparison.population.euclid && !build.busy) {
      // A cone query may have completed in another tab. Keep this page explicit:
      // mark it stale, but do not launch a surprise image-statistics pass.
    }
  }, [api, comparison, build.busy]);

  if (resource.loading && !api) {
    return <Page><Empty><Spinner /> reading local field census…</Empty></Page>;
  }
  if (!api) {
    return <Page><Empty>Population-comparison status is unavailable.</Empty></Page>;
  }

  return (
    <Page>
      <PageHead eyebrow="survey validation · pixels + sources"
        title="Field statistics"
        sub="Compare the image domain and the source populations that train the model against cached Euclid sky."
        right={<div className="comparison-actions">
          {stale && <Badge tone="warn">cache is behind local data</Badge>}
          <Button variant={comparison ? "default" : "primary"} disabled={build.busy}
            onClick={rebuild}>
            {build.busy ? "Measuring…" : comparison ? "Rebuild statistics" : "Measure local fields"}
          </Button>
        </div>} />

      <DatasetRuler availability={api.availability} comparison={comparison} />
      <JobProgressView job={build.job} error={build.error} />

      {!comparison ? (
        <Empty>
          <div className="comparison-empty">
            <strong>The local fields are ready; the compact statistics cache has not been built.</strong>
            <span>The pass streams {api.availability.synthetic.fields + api.availability.real.fields} fields one at a time and leaves the source FITS/TFRecords unchanged.</span>
            <Button variant="primary" onClick={rebuild} disabled={build.busy}>Measure local fields</Button>
          </div>
        </Empty>
      ) : (
        <>
          <FieldPlots comparison={comparison} />

          <section className="comparison-section comparison-section--population">
            <header className="comparison-section__head">
              <div>
                <div className="eyebrow">source domain · area-normalized</div>
                <h2>Population census</h2>
                <p>Counts use the true sky footprint. Histograms below include every available scientific parameter.</p>
              </div>
              <div className="comparison-actions">
                <Badge tone={api.availability.synthetic.train_source_catalog ? "good" : "warn"}>
                  {api.availability.synthetic.train_source_catalog
                    ? `${comparison.population.synthetic_field_count.toLocaleString()} synthetic fields`
                    : "test + validate only"}
                </Badge>
                {!api.availability.synthetic.train_source_catalog && (
                  <Button size="sm" disabled={trainingCatalog.busy}
                    onClick={syncTrainingCatalog}>
                    {trainingCatalog.busy ? "Syncing…" : "Sync 6,400-field source catalog"}
                  </Button>
                )}
              </div>
            </header>
            <JobProgressView job={trainingCatalog.job} error={trainingCatalog.error} />
            <div className="population-summary-grid">
              <PopulationSummary title="Synthetic source truth" eyebrow="test + validate CSV sidecars"
                population={comparison.population.synthetic} tone="synthetic" />
              {comparison.population.euclid ? (
                <PopulationSummary title="Euclid MER catalog" eyebrow="clean cone-query sources"
                  population={comparison.population.euclid} tone="real" />
              ) : (
                <Card className="population-summary population-summary--empty">
                  <CardHead title="Euclid MER catalog" sub="No population cone is cached yet." />
                  <CardBody><Empty>Choose a cone below to add the real-source population.</Empty></CardBody>
                </Card>
              )}
            </div>

            <ConeQuery api={api} onQueried={rebuild} />

            {sections.map((section) => (
              <section className="parameter-section" key={section.title}>
                <header>
                  <div className="eyebrow">parameter atlas</div>
                  <h3>{section.title}</h3>
                  <span>{Object.keys(section.population.parameters).length} distributions</span>
                </header>
                <ParameterAtlas population={section.population} source={section.source} />
              </section>
            ))}
          </section>
        </>
      )}
    </Page>
  );
}
