/* Galaxies › the two drawers' content.
   - PriorPanel (the Prior drawer, `?prior=1`): the prior's integrated
     density as the summary line (against the generated fields), the model
     laws as visible facts (2 significant figures), the SFR / Rₑ coverage as a
     sentence, the brightness law and fingerprints collapsed (provenance), the
     colour-forest fit diagnostic, then Fit (from the cached Q1 data) and
     Activate, both confirmed.
   - SourceLedger + QueryPanel (the How-this-is-produced drawer, `?how=1`):
     the three data layers as one sentence each, then the ONE Q1 MER + PHZ
     acquisition (query → fit → rebuild; its phases and checkpoints show only
     while it runs), the re-query of the population cones and the plot
     rebuild. Login is required for the archive queries. */
import { Link } from "react-router-dom";
import { formatCount, formatNumber } from "../../../format";
import {
  Badge, Button, Callout, Caption, Details, FactsList, JobProgress, Num, SummaryLine, Tooltip,
} from "../../../ui";
import type { GalaxyPayload } from "../api";
import { SOURCE_META, formatSigned, surveyColor, type SourceKey } from "../chartKit";
import { Info, SkyLink, Swatch } from "../common";
import { JOB, useRealismJob } from "../jobs";
import { activateGalaxyModel, fitGalaxies, queryGalaxies, rebuildGalaxyPlots, requeryCones } from "./actions";
import { galaxyFitReady, magRange } from "./galaxyText";
import { MARGINAL_ORDER, queryPhases } from "./model";
import { ColorModel } from "./Relations";

/** An area in arcmin²: whole numbers from 1,000, else 3 significant figures ("1,885", "36.4"). */
const area = (v?: number | null) => formatNumber(v);

/** Each data layer as one sentence naming what its area covers. */
function ledgerSentence(api: GalaxyPayload, key: SourceKey): string {
  const s = api.sources[key] ?? {};
  if (key === "euclid") {
    if (!s.available) return "Q1 query: not cached yet.";
    const cones = s.cone_count ? ` (${formatCount(s.cone_count)}\u00a0cones)` : "";
    const pdfs = s.phz_pdf_rows
      ? `, ${formatCount(s.phz_pdf_rows)} with ${s.phz_pdf_source === "summary_reconstruction" ? "reconstructed PHZ PDFs" : "PHZ PDFs"}` : "";
    return `Q1 query: ${formatCount(s.rows)}\u00a0rows over ${area(s.area_arcmin2)}\u00a0arcmin² of population cones${cones}${pdfs}.`;
  }
  if (key === "synthetic") {
    const label = api.training_included ? "Generated train + test + validate" : "Generated test + validate";
    if (!s.available) return `${label}: no source catalogues yet.`;
    const fields = s.fields ? ` (${formatCount(s.fields)}\u00a0fields)` : "";
    const radii = s.measured_radius_rows ? `, ${formatCount(s.measured_radius_rows)}\u00a0radii measured on clean images` : "";
    return `${label}: ${formatCount(s.rows)}\u00a0galaxies over ${area(s.area_arcmin2)}\u00a0arcmin² of scenes${fields}${radii}.`;
  }
  if (!s.available) return "Fitted model: not fitted yet.";
  const footprint = api.q1_counts?.footprint_area_deg2;
  return footprint != null
    ? `Fitted model: the Q1 brackets over the whole ${formatNumber(footprint, { digits: 1 })}\u00a0deg² deep-field footprint.`
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

/** "Q1 cache v7 · VIS 14–28": the cache version and queried range (the rows,
 *  area and cones are in the ledger sentence beside it, so each count appears once). */
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

export function QueryPanel({ api }: { api: GalaxyPayload }) {
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
    <div className="rl-stack">
      <div className="rl-row">
        <h3 className="syn-subtitle">Q1 MER + PHZ (Euclid archive)</h3>
        <Info label="About the Q1 galaxy query">
          <p>Exact 0.1-mag bins are queried at 0.5-mag spacing first, then revisited at offsets of
            0.1–0.4 mag. Each F₁–F₄ result and each aggregate Sérsic-Rₑ result is stored immediately and
            skipped on later runs. The galaxy density fit uses aggregate brackets, not a downloaded object
            catalogue.</p>
          <p>Selection: <code>POINT_LIKE_FLAG IS NULL</code> and <code>PHZ_GAL_PROB ≥ 0.5</code>; galaxies only, never the
            star caches (Stars has its own query).</p>
        </Info>
      </div>
      <SourceLedger api={api} />
      {caption && <Caption>{caption}</Caption>}
      <div className="rl-row">
        <Tooltip content={login ? "Log in to the Euclid archive (System › Connections) first" : "Queries the Euclid archive, then refits and rebuilds (confirmed; long)"}>
          <span tabIndex={login ? 0 : -1}>
            <Button variant="primary" icon="activity" loading={query.busy} disabled={login}
              onClick={() => void queryGalaxies()}>
              {query.busy ? "Querying + fitting…" : "Query MER + PHZ"}
            </Button>
          </span>
        </Tooltip>
        <Tooltip content={login ? "Log in to the Euclid archive first" : "Needed after a catalogue schema change (confirmed)"}>
          <span tabIndex={login ? 0 : -1}>
            <Button variant="ghost" loading={cones.busy} disabled={login || query.busy} onClick={() => void requeryCones()}>
              Re-query population cones
            </Button>
          </span>
        </Tooltip>
        <SkyLink layers={["q1-tiles:0.2", "population-cones"]}
          hint={`The ${coneCount ? `${formatCount(coneCount)} ` : ""}population cones on the sky atlas`}>Cones on sky</SkyLink>
        <Button variant="ghost" icon="reset" loading={build.busy} disabled={query.busy} onClick={() => void rebuildGalaxyPlots()}>
          Rebuild the plots
        </Button>
        {login && <Button asChild size="sm" variant="ghost"><Link to="/system/connections">Log in to the Euclid archive</Link></Button>}
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
    </div>
  );
}

/** Weight fractions at 3 significant figures ("33%", "97.5%"). */
const pct3 = (v: number) => `${formatNumber(100 * v, { sig: 3 })}%`;

/** "SFR is known for 33% of the weight; Rₑ resolved for 97.5%." (null without either). */
export function coverageSentence(sfr?: number | null, resolved?: number | null): string | null {
  const parts = [
    sfr != null ? `SFR is known for ${pct3(sfr)} of the weight` : null,
    resolved != null ? `Rₑ resolved for ${pct3(resolved)}` : null,
  ].filter(Boolean);
  return parts.length ? `${parts.join("; ")}.` : null;
}

export function PriorPanel({ api }: { api: GalaxyPayload }) {
  const activate = useRealismJob(JOB.galaxyActivate);
  const fit = useRealismJob(JOB.galaxyFit);
  const query = useRealismJob(JOB.galaxyQuery);
  const cal = api.calibration;
  const c = cal.candidate;
  const synthetic = api.sources.synthetic;
  const generated = synthetic?.available && synthetic.rows != null && synthetic.area_arcmin2
    ? synthetic.rows / synthetic.area_arcmin2 : null;
  const coverage = coverageSentence(c?.provenance?.color_sfr_valid_weight_fraction, c?.provenance?.color_resolved_radius_weight_fraction);
  const fitReady = galaxyFitReady(api);
  return (
    <div className="rl-stack">
      {c ? (
        <SummaryLine>
          Prior <Num unit="galaxies arcmin⁻²">{formatNumber(c.generation.surface_density_arcmin2, { sig: 3 })}</Num> at scene
          depth (VIS {magRange(c.generation.vis_magnitude_min, c.generation.vis_magnitude_max)})
          {generated != null && <>; the generated fields hold <Num tone={Math.abs(generated / c.generation.surface_density_arcmin2 - 1) > 0.05 ? "warn" : undefined}>
            {formatNumber(generated, { sig: 3 })}</Num></>}
          {!cal.is_active && <> · <Num tone="warn">{c.valid ? "not active" : "failed validation"}</Num></>}
        </SummaryLine>
      ) : <p className="rl-note">No galaxy prior has been fitted yet: run the Q1 MER + PHZ query (How this is produced).</p>}
      {c && (
        <FactsList title="Model laws" facts={[
          { label: "Radius slope", value: formatSigned(c.radius_law.slope_log10_arcsec_per_mag, { sig: 2 }), unit: "dex/mag",
            hint: "d log₁₀ Rₑ / d VIS of the straight truncated-Gaussian size law" },
          { label: "Radius scatter", value: formatNumber(c.radius_law.scatter_dex, { sig: 2 }), unit: "dex",
            hint: "Intrinsic scatter of log₁₀ Rₑ at fixed magnitude" },
          c.color_sfr_model && { label: "Colour forest", value: formatCount(c.color_sfr_model.row_count), unit: "Q1 rows",
            hint: "Q1 rows the conditional colour + SFR forest resamples" },
        ]} />
      )}
      {coverage && <p className="rl-note">{coverage}</p>}
      {c && (
        <Details summary="Brightness law and fingerprints" className="rl-details">
          <p className="rl-note">
            Counts: three bright-bridge slopes joined at fixed VIS {c.magnitude_law.bright_join_magnitudes.map((v) => v.toFixed(2)).join(" / ")}
            {" "}(slopes {c.magnitude_law.bright_slopes.map((v) => formatSigned(v, { sig: 2 })).join(" / ")}), the main Q1 line,
            then a flat faint plateau through VIS {magRange(c.generation.vis_magnitude_max)}.
          </p>
          <p className="rl-mono">model {c.fingerprint}</p>
          {c.color_sfr_model && <p className="rl-mono">colour model {c.color_sfr_model.calibration_fingerprint}</p>}
        </Details>
      )}
      {c && <ColorModel candidate={c} />}
      <div className="rl-row">
        <Tooltip content={fitReady ? "Refit the laws and the colour forest from the cached Q1 brackets (confirmed; no archive query)"
          : "The Q1 brackets are not all cached: run the query (How this is produced) first"}>
          <span tabIndex={fitReady ? -1 : 0}>
            <Button loading={fit.busy} disabled={!fitReady || query.busy} onClick={() => void fitGalaxies()}>
              Fit galaxy prior from cached data
            </Button>
          </span>
        </Tooltip>
        <Button variant="primary" loading={activate.busy} disabled={!c?.valid || query.busy || fit.busy}
          onClick={() => void activateGalaxyModel(cal.is_active)}>
          {cal.is_active ? "Re-activate galaxy prior" : "Activate galaxy prior"}
        </Button>
        {cal.is_active && <Button asChild variant="ghost" iconRight="chevronRight"><Link to="/synthetic/status">Status: regenerate</Link></Button>}
      </div>
      <JobProgress job={fit.job} error={fit.error} />
      <JobProgress job={activate.job} error={activate.error} />
    </div>
  );
}
