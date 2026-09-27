/* realism/galaxies (spec §8.3): the galaxy population — Q1 aggregates, the
   generated galaxies and the active VIS 2FWHM × circularized-size model.
   Views (?view=): distributions (marginals), relations (the conditional
   laws), joint (corner + pair explorer + magnitude × radius maps, one chart
   kit), model (data layers, the ONE Q1 MER + PHZ workflow, activation) and
   figure (the publication plate). The include-training toggle is the shared
   header's. GET /api/galaxy-distributions is read-only; every action is a
   keyed local job (./galaxies/actions.ts). */
import { useNavigate } from "react-router-dom";
import { usePageActions } from "../../../app/palette";
import { formatDateTime } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Card, CardBody, CardHead, EmptyState, IconButton, Menu, Page, Segmented, Tooltip,
} from "../../../ui";
import { useGalaxies, type GalaxyPayload, type Parameter } from "../api";
import { BarGroup, BarSpacer, Info, LoadState, RealismBar, SkyLink, atlasHref, downloadUrl } from "../common";
import { useIncludeTraining } from "../header";
import { JOB, useRealismJob } from "../jobs";
import {
  activateGalaxyModel, plateUrl, queryGalaxies, rebuildGalaxyPlots, type PlateFormat,
} from "../galaxies/actions";
import { CornerPlot, JointHelp, JointMapsView, PairExplorer, usePairKeys } from "../galaxies/Joint";
import { BrightnessPanel, DensityPanel, RadiusPanel, RadiusShapePanel } from "../galaxies/Marginals";
import { PARAMETER_ORDER } from "../galaxies/model";
import { Relations } from "../galaxies/Relations";
import { ModelCard, QueryWorkflow, SourceLedger } from "../galaxies/Workflow";

export const GALAXY_VIEWS = [
  { value: "distributions", label: "Distributions" },
  { value: "relations", label: "Relations" },
  { value: "joint", label: "Joint" },
  { value: "model", label: "Model" },
  { value: "figure", label: "Figure" },
] as const;
type View = (typeof GALAXY_VIEWS)[number]["value"];
const VIEW_SET = new Set<string>(GALAXY_VIEWS.map((v) => v.value));

function MarginalCard({ name, parameter }: { name: string; parameter: Parameter }) {
  return name === "magnitude" ? <BrightnessPanel parameter={parameter} />
    : name === "radius" ? <RadiusPanel parameter={parameter} />
      : <DensityPanel parameter={parameter} name={name} />;
}

function Distributions({ api }: { api: GalaxyPayload }) {
  const parameters = PARAMETER_ORDER.flatMap((key) => (api.parameters[key] ? [[key, api.parameters[key]] as const] : []));
  if (!parameters.length) {
    return <EmptyState icon="activity" title="No galaxy marginals cached yet">Rebuild the plots or query MER + PHZ (Model view).</EmptyState>;
  }
  return (
    <div className="rl-stack">
      {api.training_included && (
        <p className="rl-faint">Training adds <code>sources_train.csv</code> rows only; VIS 2FWHM and clean-image radii keep their test + validation area.</p>
      )}
      <div className="rl-plot-grid">
        {parameters.map(([key, parameter]) => <MarginalCard key={key} name={key} parameter={parameter} />)}
        {api.parameters.radius && <RadiusShapePanel parameter={api.parameters.radius} />}
      </div>
    </div>
  );
}

function Joint({ api }: { api: GalaxyPayload }) {
  const pair = usePairKeys();
  const corner = api.corner;
  const revision = `${api.version}:${corner?.q1_rows ?? 0}:${corner?.vis_range?.join(",") ?? ""}`;
  return (
    <div className="rl-stack">
      <Card>
        <CardHead title="Joint distributions" sub="Euclid Q1 rows below the diagonal, fitted-model draws above" right={<JointHelp />} />
        <CardBody><CornerPlot data={corner} active={pair} onPick={pair.set} /></CardBody>
      </Card>
      <Card>
        <CardHead title="Joint distribution explorer" sub="any two of the six variables, enlarged" />
        <CardBody><PairExplorer variables={corner?.available ? corner.variables : undefined} revision={revision} /></CardBody>
      </Card>
      <Card>
        <CardHead title="Magnitude × radius" sub="Q1, generated and model contours by enclosed mass" />
        <CardBody><JointMapsView data={api.joint_maps} /></CardBody>
      </Card>
    </div>
  );
}

function Model({ api }: { api: GalaxyPayload }) {
  return (
    <div className="rl-stack">
      <SourceLedger api={api} />
      <QueryWorkflow api={api} />
      <ModelCard api={api} />
    </div>
  );
}

const FORMATS: PlateFormat[] = ["svg", "pdf", "png"];

function Figure({ api, training }: { api: GalaxyPayload; training: boolean }) {
  return (
    <Card>
      <CardHead title="Galaxy population diagnostics · 2 × 2" sub="publication resolution, from the cached arrays"
        right={<div className="rl-row">
          {FORMATS.map((f) => (
            <Button key={f} size="sm" icon="download" href={plateUrl(training, f)}
              download={`euclidpolish_galaxy_distributions_2x2.${f}`}>{f.toUpperCase()}</Button>
          ))}
        </div>} />
      <CardBody>
        <div className="rl-plate">
          <img src={plateUrl(training, "svg", `&inline=1&v=${api.version}`)} loading="lazy"
            alt="Four-panel figure comparing Q1, the generated galaxies and the active galaxy population model" />
        </div>
      </CardBody>
    </Card>
  );
}

function StatusBadges({ api }: { api: GalaxyPayload }) {
  const cal = api.calibration;
  return (
    <>
      <Badge size="sm" tone={cal.is_active ? "good" : cal.candidate?.valid ? "warn" : "bad"} dot>
        {cal.is_active ? "model active" : cal.candidate?.valid ? "candidate not active" : "not fitted"}
      </Badge>
      {api.stale && <Badge size="sm" tone="warn">plots need rebuild</Badge>}
    </>
  );
}

export default function GalaxiesTab() {
  const [training] = useIncludeTraining();
  const [rawView, setView] = useUrlState("view", "distributions");
  const view: View = (VIEW_SET.has(rawView) ? rawView : "distributions") as View;
  const query = useRealismJob(JOB.galaxyQuery);
  const build = useRealismJob(JOB.galaxyBuild);
  // While the query runs, re-read its checkpoints (progressive phases).
  const resource = useGalaxies(training, query.busy ? 1500 : undefined);
  const api = resource.data;
  const navigate = useNavigate();
  const activeOk = !!api?.calibration.candidate?.valid;
  usePageActions([
    ...GALAXY_VIEWS.map((v) => ({ id: `galaxies-view-${v.value}`, label: `Galaxies: ${v.label}`, group: "Galaxies", run: () => setView(v.value) })),
    { id: "galaxies-query", label: "Query Q1 galaxies (MER + PHZ) and refit", group: "Galaxies", keywords: ["euclid", "archive", "fit"],
      disabled: !api?.authenticated, run: () => { void queryGalaxies(); } },
    { id: "galaxies-rebuild", label: "Rebuild the galaxy plots", group: "Galaxies", run: () => { void rebuildGalaxyPlots(); } },
    { id: "galaxies-activate", label: "Activate the galaxy model", group: "Galaxies", disabled: !activeOk,
      run: () => { void activateGalaxyModel(!!api?.calibration.is_active); } },
    { id: "galaxies-cones", label: "Show the population cones on the sky", group: "Galaxies", keywords: ["atlas"],
      run: () => navigate(atlasHref({ layers: ["q1-tiles:0.2", "population-cones"] })) },
  ]);
  return (
    <Page className="rl-page">
      <RealismBar label="Galaxy controls">
        <BarGroup>
          <Segmented size="sm" aria-label="Galaxy view" value={view} onChange={setView} options={[...GALAXY_VIEWS]} />
        </BarGroup>
        {api && <BarGroup><StatusBadges api={api} /></BarGroup>}
        <BarSpacer />
        {view === "joint" && <SkyLink layers={["q1-tiles:0.2", "population-cones"]} hint="The population cones on the sky atlas">Cones</SkyLink>}
        <Menu label="Download the galaxy figure" trigger={<IconButton size="sm" icon="download" label="Download the galaxy figure" />}
          items={FORMATS.map((f) => ({ label: `Figure · ${f.toUpperCase()}`, onSelect: () => downloadUrl(plateUrl(training, f), `euclidpolish_galaxy_distributions_2x2.${f}`) }))} />
        <Tooltip content={api?.stale ? "The inputs changed: rebuild the cached plots" : "Rebuild the cached plots"}>
          <IconButton size="sm" icon="reset" label="Rebuild galaxy plots" tooltip={false} loading={build.busy}
            disabled={query.busy} onClick={() => void rebuildGalaxyPlots()} />
        </Tooltip>
        <Info label="About the galaxy distributions">
          <p>Q1 aggregates (PHZ-weighted MER brackets), galaxies in the {training ? "training + test + validation source catalogues" : "current test + validation fields"}
            {" "}and the active VIS 2FWHM × circularized-size model.</p>
          {api?.calibration.candidate && <p>Candidate {api.calibration.candidate.fingerprint.slice(0, 12)}…{api.stale ? " · the cached plots are behind their inputs" : ""}.</p>}
          {resource.updatedAt ? <p>Read {formatDateTime(resource.updatedAt)}.</p> : null}
        </Info>
      </RealismBar>
      <LoadState loading={resource.loading && !api} error={resource.error} onRetry={resource.reload}>
        {api && (
          view === "distributions" ? <Distributions api={api} />
            : view === "relations" ? <Relations candidate={api.calibration.candidate} />
              : view === "joint" ? <Joint api={api} />
                : view === "model" ? <Model api={api} />
                  : <Figure api={api} training={training} />
        )}
      </LoadState>
    </Page>
  );
}
