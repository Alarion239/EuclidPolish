/* Galaxies › relations: the three conditional laws of the fitted model —
   brightness → radius (Q1 bracket means vs the one straight
   truncated-Gaussian law), brightness → MER aperture FWHM (the empirical
   histogram model with 16th–84th bars) and the empirical colour model (the
   forest's median colours and the variance split behind its noise
   deconvolution). On the Realism chart kit; radii on a physical log axis. */
import Plot, { type Series } from "../../../charts/Plot";
import { formatNumber } from "../../../format";
import { linearTicks } from "../../../ticks";
import { Card, CardBody, CardHead, EmptyState } from "../../../ui";
import type { GalaxyCandidate } from "../api";
import { domainOf, finite, logDomain, radiusBand, surveyColor, ticksFor } from "../chartKit";
import { Info, Swatch } from "../common";
import { conditionalFwhmInterval } from "../galaxyFwhm";
import { colorTrendSeries, colorVarianceSeries, COLOR_TREND } from "./model";

const MAG_LABEL = "VIS 2FWHM AB magnitude";
const toArcsec = (values: readonly (number | null)[]) => values.map((v) => (v == null || !Number.isFinite(v) ? null : 10 ** v));

export function RadiusRelation({ candidate }: { candidate: GalaxyCandidate }) {
  const relation = candidate.plots?.conditional_radius;
  if (!relation) return null;
  const band = radiusBand(relation);
  const observed = toArcsec(relation.observed_mean_log10_arcsec);
  const model = toArcsec(relation.model_mean_log10_arcsec);
  const low = toArcsec(band.low);
  const high = toArcsec(band.high);
  const q1 = surveyColor("euclid");
  const fit = surveyColor("fit");
  const series: Series[] = [
    ...(band.kind !== "none" ? [{ x: relation.magnitude, y: model, low, high, color: fit, fillAlpha: 0.12, alpha: 0, width: 0,
      name: band.kind === "core" ? "model core interval" : "model interval", key: "band" }] : []),
    { x: relation.magnitude, y: observed, color: q1, mode: "scatter", marker: "ring", width: 1.7, name: "Q1 bracket mean", key: "q1" },
    { x: relation.magnitude, y: model, color: fit, width: 2.6, name: "model conditional mean", key: "model" },
  ];
  const xDomain = domainOf(relation.magnitude, 1);
  const yDomain = logDomain([...finite(observed), ...finite(low), ...finite(high)]);
  return (
    <Card>
      <CardHead title="Joint brightness–radius relation"
        sub="Q1 Sérsic-Rₑ bracket means vs one fitted straight truncated-Gaussian conditional law"
        right={<Info label="About the brightness–radius relation">
          <p>Rings are the aggregate Q1 circularized Sérsic-Rₑ moments in each VIS 2FWHM bracket. The line and
            band are the single straight conditional mean and its {band.kind === "core" ? "one-scatter core" : "low/high"} interval.</p>
          <p>No magnitude break or broad radius tail; COSMOS and object-level samples are absent from this fit.</p>
        </Info>} />
      <CardBody>
        <Plot xDomain={xDomain} yDomain={yDomain} yScale="log" xTicks={linearTicks(xDomain, { count: 7 })}
          yTicks={ticksFor(yDomain, "log")} xLabel={MAG_LABEL} yLabel="Circularized Sérsic Rₑ (arcsec, log scale)"
          series={series} legend="auto" aspect={0.4} exportName="galaxy-radius-relation"
          yFormat={(v) => `${formatNumber(v, { sig: 3 })}″`} aria-label="Brightness–radius relation" />
        <p className="rl-faint">
          slope {candidate.radius_law.slope_log10_arcsec_per_mag.toFixed(4)} dex/mag · scatter {candidate.radius_law.scatter_dex.toFixed(4)} dex
          {" "}· {candidate.radius_law.fitted_rows.toLocaleString("en")} aggregate radii
        </p>
      </CardBody>
    </Card>
  );
}

export function FwhmRelation({ candidate }: { candidate: GalaxyCandidate }) {
  const relation = candidate.plots?.conditional_aperture_fwhm;
  if (!relation) return null;
  const distribution = candidate.aperture_fwhm_distribution;
  const interval = conditionalFwhmInterval(
    relation.observed_mean_arcsec, distribution?.probability ?? [], distribution?.fwhm_edges_arcsec ?? [],
  );
  const q1 = surveyColor("euclid");
  const fit = surveyColor("fit");
  const series: Series[] = [
    { x: relation.magnitude, y: relation.observed_mean_arcsec, errorLow: interval.low, errorHigh: interval.high,
      color: q1, mode: "scatter", marker: "ring", width: 1.7, name: "Q1 mean · 16th–84th", key: "q1" },
    { x: relation.magnitude, y: relation.model_mean_arcsec, color: fit, width: 2.6, name: "empirical model mean", key: "model" },
  ];
  const xDomain = domainOf(relation.magnitude, 1);
  const yDomain = domainOf([...relation.observed_mean_arcsec, ...interval.low, ...interval.high, ...relation.model_mean_arcsec], 0.25);
  return (
    <Card>
      <CardHead title="VIS magnitude–MER aperture FWHM relation"
        sub="Q1 catalogue-FWHM means per magnitude bin vs the active empirical model (synthetic aperture photometry)"
        right={<Info label="About the FWHM relation">
          <p>Rings are weighted mean MER catalogue FWHM values in populated Q1 magnitude bins; the bars span the
            weighted 16th–84th percentiles. The line is the mean of the active empirical FWHM | VIS 2FWHM model.</p>
          <p>Not a separate regression: the model keeps the measured magnitude-bin histograms and uses the nearest
            populated bin where direct Q1 support is absent ({relation.out_of_support_policy}).</p>
        </Info>} />
      <CardBody>
        <Plot xDomain={xDomain} yDomain={yDomain} xTicks={linearTicks(xDomain, { count: 7 })}
          yTicks={linearTicks(yDomain, { count: 6 })} xLabel={MAG_LABEL} yLabel="MER catalogue FWHM (arcsec)"
          series={series} legend="auto" aspect={0.4} exportName="galaxy-fwhm-relation"
          aria-label="VIS magnitude–MER aperture FWHM relation" />
        <p className="rl-faint">Bars only where Q1 populates the bin; elsewhere the model uses the nearest populated bin.</p>
      </CardBody>
    </Card>
  );
}

export function ColorModel({ candidate }: { candidate: GalaxyCandidate }) {
  const colors = candidate.plots?.conditional_colors;
  if (!colors) return null;
  const trend = colorTrendSeries(colors);
  const variance = colorVarianceSeries(colors);
  const trendX = domainOf(colors.magnitude, 1);
  const trendY = domainOf(COLOR_TREND.flatMap(({ key }) => colors[key]), 0.5);
  const varianceX = domainOf(variance[0]?.x ?? [], 1);
  const varianceY = logDomain(variance.flatMap((s) => s.y));
  const floor = candidate.color_sfr_model?.vis_snr_floor ?? 5;
  return (
    <Card>
      <CardHead title="Empirical colour model"
        sub="Conditional colours of the forest and the observed / noise ratio variance by magnitude"
        right={<Info label="About the colour model">
          <p>Generated colours resample real Q1 rows from the query point's forest leaves; the drawn row's ratio is
            then sampled from its Gaussian posterior with intrinsic variance = observed − noise (floored at zero).</p>
          <p>Where the dashed noise curve approaches the solid observed curve, the catalogue constrains only the median
            colour relation. Colour rows need VIS 2FWHM S/N ≥ {floor}; fainter magnitudes reuse the deepest leaves.</p>
        </Info>} />
      <CardBody>
        <div className="rl-grid rl-grid--2">
          <figure className="rl-fig">
            <figcaption className="rl-fig__head"><strong>Median colour</strong><small>model mean VIS − Y, Y − J, J − H</small></figcaption>
            <Plot xDomain={trendX} yDomain={trendY} xTicks={linearTicks(trendX, { count: 6 })}
              yTicks={linearTicks(trendY, { count: 6 })} xLabel={MAG_LABEL} yLabel="median colour (AB mag)"
              series={trend} legend="auto" aspect={0.62} exportName="galaxy-color-trend" syncKey="galaxy-colors"
              aria-label="Median colour trend" />
          </figure>
          <figure className="rl-fig">
            <figcaption className="rl-fig__head"><strong>Flux-ratio variance</strong><small>observed (solid) vs reported noise (dashed)</small></figcaption>
            {variance.length ? (
              <Plot xDomain={varianceX} yDomain={varianceY} yScale="log" xTicks={linearTicks(varianceX, { count: 6 })}
                yTicks={ticksFor(varianceY, "log")} xLabel={MAG_LABEL} yLabel="flux-ratio variance (log scale)"
                series={variance} legend="auto" aspect={0.62} exportName="galaxy-color-variance" syncKey="galaxy-colors"
                aria-label="Observed and noise ratio variance by magnitude" />
            ) : <EmptyState compact icon="activity" title="No per-magnitude variance in this candidate" />}
          </figure>
        </div>
        <div className="rl-defs">
          {COLOR_TREND.map(({ key, label, band }) => (
            <span key={key}><Swatch color={trend.find((s) => s.name === label)?.color ?? ""} />{label} · redder band {band}</span>
          ))}
        </div>
      </CardBody>
    </Card>
  );
}

export function Relations({ candidate }: { candidate: GalaxyCandidate | null }) {
  if (!candidate?.plots || !(candidate.plots.conditional_radius || candidate.plots.conditional_aperture_fwhm || candidate.plots.conditional_colors)) {
    return <EmptyState icon="activity" title="No fitted relations yet">Query MER + PHZ (Model view) to fit the galaxy model.</EmptyState>;
  }
  return (
    <div className="rl-stack">
      <RadiusRelation candidate={candidate} />
      <FwhmRelation candidate={candidate} />
      <ColorModel candidate={candidate} />
    </div>
  );
}
