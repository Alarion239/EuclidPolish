/* Galaxies › model view: the three data layers (Q1, generated, fitted) as
   one sentence each, the ONE Q1 MER + PHZ acquisition workflow (query → fit →
   rebuild; its progressive phases and checkpoints show only while it runs)
   and the population model card (the density as a summary line, the laws as
   facts; activate). */
import { Link } from "react-router-dom";
import { formatCount, formatNumber } from "../../../format";
import {
  Badge, Button, Callout, Caption, Card, CardBody, CardHead, Details, FactsList, JobProgress, Num, SummaryLine, Tooltip,
} from "../../../ui";
import type { GalaxyPayload } from "../api";
import { SOURCE_META, surveyColor, type SourceKey } from "../chartKit";
import { Info, SkyLink, Swatch } from "../common";
import { JOB, useRealismJob } from "../jobs";
import { activateGalaxyModel, queryGalaxies, rebuildGalaxyPlots, requeryCones } from "./actions";
import { MARGINAL_ORDER, queryPhases } from "./model";

/** An area in arcmin²: whole numbers from 1,000, else 3 significant figures ("1,885", "36.4"). */
const area = (v?: number | null) => formatNumber(v);

/** "Q1 query: 140,085 rows over 1,885 arcmin² (24 cones), 132,147 with PHZ PDFs." — each data layer as
 *  one sentence naming what its area covers (never a made-up cone count). */

/** Magnitudes at one shared precision: integers stay "14–28", otherwise every value gets the fewest
 *  decimals (≤ 2) the least round one needs ("14.0–28.5", never "14–28.5"). */
function magRange(...values: number[]): string {
  const need = (v: number) => (Math.abs(v - Math.round(v)) < 1e-6 ? 0 : Math.abs(v * 10 - Math.round(v * 10)) < 1e-6 ? 1 : 2);
  const digits = Math.max(...values.map(need));
  return values.map((v) => v.toFixed(digits)).join("–");
}
function ledgerSentence(api: GalaxyPayload, key: SourceKey): string {
  const s = api.sources[key] ?? {};
  if (key === "euclid") {
    if (!s.available) return "Q1 query: not cached yet.";
    const cones = s.cone_count ? ` (${formatCount(s.cone_count)}\u00a0cones)` : "";
    const pdfs = s.phz_pdf_rows
      ? `, ${formatCount(s.phz_pdf_rows)} with ${s.phz_pdf_source === "summary_reconstruction" ? "reconstructed PHZ PDFs" : "PHZ PDFs"}` : "";
    return `Q1 query: ${formatCount(s.rows)}\u00a0rows over ${area(s.area_arcmin2)}\u00a0arcmin²${cones}${pdfs}.`;
  }
  if (key === "synthetic") {
    const label = api.training_included ? "Generated train + test + validation" : "Generated test + validation";
    if (!s.available) return `${label}: no source catalogues yet.`;
    const fields = s.fields ? ` (${formatCount(s.fields)}\u00a0fields)` : "";
    const radii = s.measured_radius_rows ? `, ${formatCount(s.measured_radius_rows)}\u00a0radii measured on clean images` : "";
    return `${label}: ${formatCount(s.rows)}\u00a0galaxies over ${area(s.area_arcmin2)}\u00a0arcmin²${fields}${radii}.`;
  }
  if (!s.available) return "Fitted model: not fitted yet.";
  const footprint = api.q1_counts?.footprint_area_deg2;
  return footprint != null
    ? `Fitted model: the Q1 counts and radii over the ${formatNumber(footprint, { digits: 1 })}\u00a0deg² footprint.`
    : "Fitted model: from the Q1 counts and radii.";
}

export function SourceLedger({ api }: { api: GalaxyPayload }) {
  return (
    <section className="rl-ledger" aria-label="Galaxy distribution data layers">
      {MARGINAL_ORDER.map((key: SourceKey) => {
        const source = api.sources[key] ?? {};
        // A badge only on a problem: a missing layer, or a fit that failed validation.
        const problem = !source.available ? "missing" : key === "fit" && !source.validated ? "not validated" : null;
        return (
          <article className="rl-ledger__item" key={key}>
            <div className="rl-ledger__kicker">
              <Swatch color={surveyColor(key)} /> {SOURCE_META[key].kicker}
              {problem && <Badge size="sm" tone="warn">{problem}</Badge>}
            </div>
            <p className="rl-ledger__text">{ledgerSentence(api, key)}</p>
          </article>
        );
      })}
    </section>
  );
}

/** "Q1 cache v7 · VIS 14–28": the cache version and queried range. The rows, area and cones are in the
 *  SourceLedger sentence on the same screen, so each count appears once. */
function queryCaption(api: GalaxyPayload): string | null {
  const e = api.sources.euclid;
  const q1 = api.q1_counts;
  if (e?.schema_version == null && !q1) return null;
  return [
    `Q1 cache${e?.schema_version != null ? ` v${e.schema_version}` : ""}`,
    q1 ? `VIS ${magRange(q1.bright, q1.faint)}` : null,
  ].filter(Boolean).join(" · ");
}

/** "checkpoints 400 of 560 · Rₑ brackets 170 of 170" (only the counters the payload knows). */
function progressText(api: GalaxyPayload): string {
  const q1 = api.q1_counts;
  const done = q1?.completed_queries ?? q1?.query_count;
  const r = api.q1_radius;
  return [
    done != null && q1?.total_queries != null ? `checkpoints ${formatCount(done)} of ${formatCount(q1.total_queries)}` : null,
    r ? `Rₑ brackets ${formatCount(r.completed_queries)} of ${formatCount(r.total_queries)}` : null,
  ].filter(Boolean).join(" · ");
}

/** The interrupted-query line (a problem, so it is shown), or null when the cache is complete. */
function incompleteText(api: GalaxyPayload): string | null {
  const q1 = api.q1_counts;
  if (!q1) return null;
  const done = q1.completed_queries ?? q1.query_count;
  const total = q1.total_queries;
  const incomplete = q1.complete === false || (done != null && total != null && done < total);
  if (!incomplete) return null;
  const passes = q1.phases_completed != null && q1.phase_count != null ? ` (${q1.phases_completed} of ${q1.phase_count} passes)` : "";
  const where = done != null && total != null ? ` at ${formatCount(done)} of ${formatCount(total)} checkpoints` : "";
  return `The last Q1 query stopped${where}${passes}; run it again to resume.`;
}

export function QueryWorkflow({ api }: { api: GalaxyPayload }) {
  const query = useRealismJob(JOB.galaxyQuery);
  const cones = useRealismJob(JOB.galaxyCones);
  const build = useRealismJob(JOB.galaxyBuild);
  const q1 = api.q1_counts;
  const phases = queryPhases(q1?.phases_completed ?? 0, q1?.phase_count ?? 5, query.busy);
  const login = !api.authenticated;
  const caption = queryCaption(api);
  const coneCount = api.sources.euclid?.cone_count;
  const incomplete = query.busy ? null : incompleteText(api);
  const progress = progressText(api);
  return (
    <Card>
      <CardHead title="Q1 MER + PHZ galaxy workflow" sub="one run: brightness + Sérsic-radius brackets → fit → plots"
        right={<Info label="About the Q1 galaxy query">
          <p>Exact 0.1-mag bins are queried at 0.5-mag spacing first, then revisited at offsets of
            0.1–0.4 mag. Each F₁–F₄ result and each aggregate Sérsic-Rₑ result is stored immediately and
            skipped on later runs. The galaxy density fit uses aggregate brackets, not a downloaded object
            catalogue.</p>
          <p>Stellar counts and Gaia–Euclid colours have their own query on the Stars tab.</p>
        </Info>} />
      <CardBody>
        {caption && <Caption>{caption}</Caption>}
        <p className="rl-note">
          Selection: <code>POINT_LIKE_FLAG IS NULL</code> and <code>PHZ_GAL_PROB ≥ 0.5</code> · galaxies only,
          never refreshes star caches.
        </p>
        <div className="rl-row">
          <Tooltip content={login ? "Log in to the Euclid archive (Settings › Connections) first" : "Queries the Euclid archive; long"}>
            <span tabIndex={login ? 0 : -1}>
              <Button variant="primary" icon="activity" loading={query.busy} disabled={login}
                onClick={() => void queryGalaxies()}>
                {query.busy ? "Querying + fitting…" : "Query MER + PHZ"}
              </Button>
            </span>
          </Tooltip>
          <Button variant="ghost" loading={cones.busy} disabled={login || query.busy}
            onClick={() => void requeryCones()}>
            Re-query population cones
          </Button>
          <SkyLink layers={["q1-tiles:0.2", "population-cones"]}
            hint={`The ${coneCount ? `${formatCount(coneCount)} ` : ""}population cones on the sky atlas`}>Cones on sky</SkyLink>
          <Button variant="ghost" icon="reset" loading={build.busy} disabled={query.busy}
            onClick={() => void rebuildGalaxyPlots()}>
            Rebuild cached plots
          </Button>
          {login && <Button asChild size="sm" variant="ghost"><Link to="/settings/connections">Log in to Euclid archive</Link></Button>}
        </div>
        {incomplete && <Callout tone="warn" dense>{incomplete}</Callout>}
        {/* Progress counters exist only while the query runs, beside its job. */}
        {query.busy && (
          <div className="rl-progress">
            <ol className="rl-phases" aria-label="Progressive magnitude-bin sampling phases">
              {phases.map((p) => (
                <li key={p.index} className="rl-phase" data-state={p.state}>
                  <span>phase {p.index + 1}</span><b>+{p.offset.toFixed(1)} mag</b><small>{p.state}</small>
                </li>
              ))}
            </ol>
            {progress && <Caption>{progress}</Caption>}
          </div>
        )}
        <JobProgress job={query.job} error={query.error} />
        <JobProgress job={cones.job} error={cones.error} />
        <JobProgress job={build.job} error={build.error} />
      </CardBody>
    </Card>
  );
}

export function ModelCard({ api }: { api: GalaxyPayload }) {
  const activate = useRealismJob(JOB.galaxyActivate);
  const query = useRealismJob(JOB.galaxyQuery);
  const cal = api.calibration;
  const c = cal.candidate;
  // Weight fractions at 3 significant figures ("33%", "97.5%").
  const pct = (v: number) => `${formatNumber(100 * v, { sig: 3 })}%`;
  const sfr = c?.provenance?.color_sfr_valid_weight_fraction;
  const resolved = c?.provenance?.color_resolved_radius_weight_fraction;
  return (
    <Card>
      <CardHead title="Galaxy population model" sub="Q1 counts + size law; colours and SFR resampled from real rows"
        right={cal.is_active ? undefined : (  // the OK state is the bar's "model active" alone
          <Badge tone={c?.valid ? "warn" : undefined}>{c?.valid ? "candidate ready" : "not fitted"}</Badge>
        )} />
      <CardBody>
        {c && (
          <SummaryLine>
            <Num unit="galaxies arcmin⁻²">{formatNumber(c.generation.surface_density_arcmin2, { sig: 3 })}</Num> at scene depth
            (VIS {magRange(c.generation.vis_magnitude_min, c.generation.vis_magnitude_max)})
          </SummaryLine>
        )}
        <FactsList title="Model" facts={c ? [
          { label: "Radius slope", value: formatNumber(c.radius_law.slope_log10_arcsec_per_mag, { sig: 2 }), unit: "dex/mag",
            hint: "d log₁₀ Rₑ / d VIS of the straight truncated-Gaussian size law" },
          { label: "Radius scatter", value: formatNumber(c.radius_law.scatter_dex, { sig: 2 }), unit: "dex",
            hint: "Intrinsic scatter of log₁₀ Rₑ at fixed magnitude" },
          c.color_sfr_model && { label: "Colour forest", value: formatCount(c.color_sfr_model.row_count), unit: "rows",
            hint: "Q1 rows the conditional colour + SFR forest resamples" },
          sfr != null && { label: "SFR known for", value: pct(sfr), unit: "of the weight" },
          resolved != null && { label: "Rₑ resolved for", value: pct(resolved), unit: "of the weight" },
        ] : []} />
        {c && (
          <p className="rl-note" aria-label="Model laws">
            Counts: three bright-bridge slopes joined at fixed VIS {c.magnitude_law.bright_join_magnitudes.map((v) => v.toFixed(2)).join(" / ")}
            {" "}(slopes {c.magnitude_law.bright_slopes.map((v) => formatNumber(v, { sig: 2 })).join(" / ")}), the main Q1 line,
            then a flat faint plateau through VIS {magRange(c.generation.vis_magnitude_max)}.
            {" "}Size: one straight truncated-Gaussian log-radius law from {formatCount(c.radius_law.fitted_rows)} aggregate radii.
          </p>
        )}
        {c && (
          <Details summary="Fingerprints" className="rl-details">
            <p className="rl-mono">model {c.fingerprint}</p>
            {c.color_sfr_model && <p className="rl-mono">colour model {c.color_sfr_model.calibration_fingerprint}</p>}
          </Details>
        )}
        <div className="rl-row">
          <Button variant="primary" loading={activate.busy} disabled={!c?.valid || query.busy}
            onClick={() => void activateGalaxyModel(cal.is_active)}>
            {cal.is_active ? "Re-activate model" : "Activate model"}
          </Button>
          {cal.is_active && <Button asChild variant="ghost" iconRight="chevronRight"><Link to="/data/records">Open Records jobs</Link></Button>}
        </div>
        <JobProgress job={activate.job} error={activate.error} />
      </CardBody>
    </Card>
  );
}
