/* Galaxies › model view: the three data layers (Q1, generated, fitted), the
   ONE Q1 MER + PHZ acquisition workflow (query → fit → rebuild, with its
   progressive phases) and the population model card (activate). */
import { Link } from "react-router-dom";
import { formatCount, formatNumber } from "../../../format";
import { Badge, Button, Card, CardBody, CardHead, JobProgress, Stat, Tooltip } from "../../../ui";
import type { GalaxyPayload } from "../api";
import { SOURCE_META, surveyColor, type SourceKey } from "../chartKit";
import { Info, SkyLink, StatStrip, Swatch } from "../common";
import { JOB, useRealismJob } from "../jobs";
import { activateGalaxyModel, queryGalaxies, rebuildGalaxyPlots, requeryCones } from "./actions";
import { MARGINAL_ORDER, queryPhases } from "./model";

const NA = "—";
/** "done/total", or — while either is unknown (never a made-up total). */
const ratio = (done?: number | null, total?: number | null) => (done != null && total != null ? `${done}/${total}` : NA);
const compact = (v?: number | null) =>
  v == null ? "—" : new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(v);

export function SourceLedger({ api }: { api: GalaxyPayload }) {
  return (
    <section className="rl-ledger" aria-label="Galaxy distribution data layers">
      {MARGINAL_ORDER.map((key: SourceKey) => {
        const source = api.sources[key] ?? {};
        const fitted = key === "fit";
        const label = key === "synthetic"
          ? api.training_included ? "Generated train + test + validation" : "Generated test + validation"
          : SOURCE_META[key].label;
        return (
          <article className="rl-ledger__item" key={key}>
            <div className="rl-ledger__kicker"><Swatch color={surveyColor(key)} /> {SOURCE_META[key].kicker}</div>
            <div className="rl-ledger__title">{label}</div>
            <div className="rl-ledger__metrics">
              {source.available
                ? <Badge size="sm" tone={fitted && !source.validated ? "warn" : "good"}>{fitted ? (source.is_active ? "active" : "candidate") : "cached"}</Badge>
                : <Badge size="sm" tone="warn">missing</Badge>}
              {!fitted && <span><b>{compact(source.rows)}</b> objects</span>}
              {!fitted && <span><b>{source.area_arcmin2?.toFixed(1) ?? "—"}</b> arcmin²</span>}
              {key === "synthetic" && <span><b>{compact(source.measured_radius_rows)}</b> image radii</span>}
              {key === "euclid" && <span><b>{compact(source.phz_pdf_rows)}</b> {source.phz_pdf_source === "summary_reconstruction" ? "reconstructed PDFs" : "PHZ PDFs"}</span>}
              {fitted && source.fingerprint && <span className="rl-mono" title={source.fingerprint}>{source.fingerprint.slice(0, 10)}</span>}
            </div>
            {source.detail && <p className="rl-ledger__detail">{source.detail}</p>}
          </article>
        );
      })}
    </section>
  );
}

export function QueryWorkflow({ api }: { api: GalaxyPayload }) {
  const query = useRealismJob(JOB.galaxyQuery);
  const cones = useRealismJob(JOB.galaxyCones);
  const build = useRealismJob(JOB.galaxyBuild);
  const q1 = api.q1_counts;
  const phases = queryPhases(q1?.phases_completed ?? 0, q1?.phase_count ?? 5, query.busy);
  const login = !api.authenticated;
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
        <StatStrip label="Q1 query state">
          <Stat k="Q1 footprint" v={q1 ? `${q1.footprint_area_deg2.toFixed(1)} deg²` : NA} />
          <Stat k="VIS range" v={q1 ? `${q1.bright.toFixed(1)}–${q1.faint.toFixed(1)}` : NA} />
          <Stat k="bin width" v={q1 ? `${q1.bin_width.toFixed(1)} mag` : NA} />
          <Stat k="checkpoints" v={ratio(q1?.completed_queries ?? q1?.query_count, q1?.total_queries)} />
          <Stat k="Rₑ brackets" v={ratio(api.q1_radius?.completed_queries, api.q1_radius?.total_queries)} />
          <Stat k="passes" v={ratio(q1?.phases_completed, q1?.phase_count)} />
          <Stat k="F₁ PHZ weight" v={compact(q1?.apertures.f1.expected_galaxies)} />
          <Stat k="F₄ PHZ weight" v={compact(q1?.apertures.f4.expected_galaxies)} />
        </StatStrip>
        <ol className="rl-phases" aria-label="Progressive magnitude-bin sampling phases">
          {phases.map((p) => (
            <li key={p.index} className="rl-phase" data-state={p.state}>
              <span>phase {p.index + 1}</span><b>+{p.offset.toFixed(1)} mag</b><small>{p.state}</small>
            </li>
          ))}
        </ol>
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
          <SkyLink layers={["q1-tiles:0.2", "population-cones"]} hint="The 24 population cones on the sky atlas">Cones on sky</SkyLink>
          <Button variant="ghost" icon="reset" loading={build.busy} disabled={query.busy}
            onClick={() => void rebuildGalaxyPlots()}>
            Rebuild cached plots
          </Button>
          {login && <Button asChild size="sm" variant="ghost"><Link to="/settings/connections">Log in to Euclid archive</Link></Button>}
        </div>
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
  const pct = (v?: number) => (v == null ? "—" : `${(100 * v).toFixed(1)}%`);
  return (
    <Card>
      <CardHead title="Galaxy population model" sub="Q1 counts + size law; colours and SFR resampled from real rows"
        right={<Badge tone={cal.is_active ? "good" : c?.valid ? "warn" : undefined}>
          {cal.is_active ? "active for generation" : c?.valid ? "candidate ready" : "not fitted"}
        </Badge>} />
      <CardBody>
        <StatStrip label="Galaxy model">
          <Stat k="brightness" v="Q1 VIS 2FWHM · 14–29" />
          <Stat k="integrated density" v={c ? `${c.generation.surface_density_arcmin2.toFixed(0)} arcmin⁻²` : "—"} />
          <Stat k="faint plateau" v={c ? `${c.generation.differential_density_cap_arcmin2_mag.toFixed(0)} arcmin⁻² mag⁻¹` : "—"}
            sub={c ? `from VIS ${c.generation.break_magnitude.toFixed(2)}` : undefined} />
          <Stat k="radius slope" v={c ? `${c.radius_law.slope_log10_arcsec_per_mag.toFixed(4)} dex/mag` : "—"} />
          <Stat k="radius scatter" v={c ? `${c.radius_law.scatter_dex.toFixed(4)} dex` : "—"} />
          <Stat k="colour rows" v={c?.color_sfr_model ? formatCount(c.color_sfr_model.row_count) : "—"} />
          <Stat k="forest" v={c?.color_sfr_model ? `${c.color_sfr_model.tree_count} trees` : "—"} />
          <Stat k="SFR-valid weight" v={pct(c?.provenance?.color_sfr_valid_weight_fraction)} />
          <Stat k="resolved-Rₑ weight" v={pct(c?.provenance?.color_resolved_radius_weight_fraction)} />
        </StatStrip>
        {c && (
          <p className="rl-note" aria-label="Model laws">
            Counts: three bright-bridge slopes joined at fixed VIS {c.magnitude_law.bright_join_magnitudes.map((v) => v.toFixed(2)).join(" / ")}
            {" "}(slopes {c.magnitude_law.bright_slopes.map((v) => formatNumber(v, { digits: 3 })).join(" / ")}), the main Q1 line,
            then a flat faint plateau through VIS {c.generation.vis_magnitude_max.toFixed(0)}.
            {" "}Size: one straight truncated-Gaussian log-radius law from {formatCount(c.radius_law.fitted_rows)} aggregate radii.
            {" "}Fingerprint <code>{c.fingerprint.slice(0, 12)}…</code>
            {c.color_sfr_model && <> · colour model <code>{c.color_sfr_model.calibration_fingerprint.slice(0, 12)}…</code></>}
          </p>
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
