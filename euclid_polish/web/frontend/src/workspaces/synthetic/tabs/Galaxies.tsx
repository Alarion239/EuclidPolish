/* Synthetic › Galaxies: does the galaxy prior match Euclid Q1, and what do
   the generated galaxies look like against it? GET /api/galaxy-distributions
   is read-only; every action is a keyed, confirmed job (../galaxies/actions.ts).

   Toolbar: the view (?view=) — distributions (default: the two trust boxes
   above the wide brightness panel, then size, the three colours and the
   radius shape, each with a caption), relations (brightness → radius and →
   MER aperture FWHM), joint (the corner plot with n on hover, the pair
   explorer, the magnitude × radius map) and templates (the TNG50-1 atlas the
   morphologies come from) — a badge only on a problem, then the drawers and
   the plate download. Drawers at the foot: Prior (`?prior=1`: the integrated
   density against the generated fields, the model laws, the colour-forest
   fit diagnostic, Fit / Activate) and How this is produced (`?how=1`: the
   Q1 MER + PHZ query, the cones, the plot rebuild and the TNG steps). The
   old model view opens the Prior drawer; the figure view is Figures ›
   Plates. The shared header's training toggle adds the train split's
   generated galaxies. */
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { usePageActions } from "../../../app/palette";
import { useUrlState } from "../../../hooks/useUrlState";
import { Badge, Caption, EmptyState, IconButton, Menu, Page, Segmented, Toolbar, ToolbarGroup, ToolbarSpacer, Tooltip } from "../../../ui";
import { useGalaxies, type GalaxyPayload } from "../api";
import { Drawer, DrawerButton, LoadState, SkyLink, atlasHref, downloadUrl, useDrawer } from "../common";
import { useIncludeTraining } from "../header";
import { JOB, useRealismJob } from "../jobs";
import {
  activateGalaxyModel, fitGalaxies, plateUrl, queryGalaxies, rebuildGalaxyPlots, type PlateFormat,
} from "../galaxies/actions";
import { galaxyFitReady, marginalCaptions } from "../galaxies/galaxyText";
import { CornerPlot, JointHelp, JointMapsView, PairExplorer, usePairKeys } from "../galaxies/Joint";
import { BrightnessPanel, DensityPanel, RadiusPanel, RadiusShapePanel } from "../galaxies/Marginals";
import { Relations } from "../galaxies/Relations";
import { Templates, TngSteps } from "../galaxies/Templates";
import { PriorPanel, QueryPanel } from "../galaxies/Workflow";
import "../synthetic.css";

export const GALAXY_VIEWS = [
  { value: "distributions", label: "Distributions" },
  { value: "relations", label: "Relations" },
  { value: "joint", label: "Joint" },
  { value: "templates", label: "Templates" },
] as const;
type View = (typeof GALAXY_VIEWS)[number]["value"];
const VIEW_SET = new Set<string>(GALAXY_VIEWS.map((v) => v.value));
const COLOUR_KEYS = ["color_vis_y", "color_y_j", "color_j_h"] as const;
const FORMATS: PlateFormat[] = ["png", "pdf", "svg"];

function Distributions({ api }: { api: GalaxyPayload }) {
  const p = api.parameters;
  const captions = marginalCaptions(api);
  const colours = COLOUR_KEYS.filter((k) => p[k]);
  if (!p.magnitude && !p.radius && !colours.length) {
    return <EmptyState icon="activity" title="No galaxy distributions cached yet">Rebuild the plots or run the Q1 query (How this is produced).</EmptyState>;
  }
  return (
    <div className="rl-plot-grid">
      {p.magnitude && <BrightnessPanel parameter={p.magnitude} caption={captions.brightness} />}
      {p.radius && <RadiusPanel parameter={p.radius} caption={captions.radius} />}
      {colours.map((k) => <DensityPanel key={k} name={k} parameter={p[k]} />)}
      {p.radius && <RadiusShapePanel parameter={p.radius} caption={captions.shape} />}
      {colours.length > 0 && <Caption className="rl-wide">{captions.colours}</Caption>}
      {api.training_included && (
        <Caption className="rl-wide">Training adds the sources_train.csv galaxies only; the VIS 2FWHM and clean-image radii keep their test + validate area.</Caption>
      )}
    </div>
  );
}

function Joint({ api }: { api: GalaxyPayload }) {
  const pair = usePairKeys();
  const corner = api.corner;
  const revision = `${api.version}:${corner?.q1_rows ?? 0}:${corner?.vis_range?.join(",") ?? ""}`;
  return (
    <div className="rl-stack">
      <section className="rl-panel" aria-label="Joint distributions">
        <header className="rl-panel__head">
          <strong>Joint distributions</strong><small>Euclid Q1 rows below the diagonal, fitted-model draws above</small>
          <span className="rl-panel__tools"><JointHelp /></span>
        </header>
        <CornerPlot data={corner} active={pair} onPick={pair.set} />
      </section>
      <section className="rl-panel" aria-label="Joint distribution explorer">
        <header className="rl-panel__head"><strong>Pair explorer</strong><small>any two of the six variables, enlarged</small></header>
        <PairExplorer variables={corner?.available ? corner.variables : undefined} revision={revision} />
      </section>
      <section className="rl-panel" aria-label="Magnitude × radius">
        <header className="rl-panel__head">
          <strong>Magnitude × radius</strong><small>Q1, generated and model contours by enclosed mass</small>
          <span className="rl-panel__tools"><SkyLink layers={["q1-tiles:0.2", "population-cones"]} hint="The population cones on the sky atlas">Cones</SkyLink></span>
        </header>
        <JointMapsView data={api.joint_maps} />
      </section>
    </div>
  );
}

/** A badge only on a problem: the prior not active, or the plots behind their inputs. */
function Problems({ api }: { api: GalaxyPayload }) {
  const cal = api.calibration;
  return (
    <>
      {!cal.is_active && (
        <Badge size="sm" tone={cal.candidate?.valid ? "warn" : "bad"} dot>{cal.candidate?.valid ? "candidate not active" : "prior not fitted"}</Badge>
      )}
      {api.stale && (
        <Tooltip content="The inputs changed since the plots were built: rebuild them (How this is produced)">
          <span tabIndex={0}><Badge size="sm" tone="warn" dot>plots need a rebuild</Badge></span>
        </Tooltip>
      )}
    </>
  );
}

export default function GalaxiesTab() {
  const [training] = useIncludeTraining();
  const [rawView, setView] = useUrlState("view", "distributions");
  const view: View = (VIEW_SET.has(rawView) ? rawView : "distributions") as View;
  const query = useRealismJob(JOB.galaxyQuery);
  // While the query runs, re-read its checkpoints (progressive phases).
  const resource = useGalaxies(training, query.busy ? 1500 : undefined);
  const api = resource.data;
  const navigate = useNavigate();
  const prior = useDrawer("prior");
  const how = useDrawer("how");
  const openPrior = prior.setOpen;
  // The absorbed page's views: the model view is the Prior drawer now, the
  // figure view Figures › Plates (an old link lands here on distributions).
  useEffect(() => {
    if (rawView === "model") { openPrior(true); setView("distributions"); }
    else if (rawView === "figure") setView("distributions");
  }, [rawView, openPrior, setView]);
  const candidateOk = !!api?.calibration.candidate?.valid;
  usePageActions([
    ...GALAXY_VIEWS.map((v) => ({ id: `galaxies-view-${v.value}`, label: `Galaxies: ${v.label}`, group: "Galaxies", run: () => setView(v.value) })),
    { id: "galaxies-prior", label: "Galaxies: open the prior (fit and activate)", group: "Galaxies", run: prior.reveal },
    { id: "galaxies-fit", label: "Fit the galaxy prior from cached data…", group: "Galaxies", disabled: !galaxyFitReady(api),
      run: () => { void fitGalaxies(); } },
    { id: "galaxies-activate", label: "Activate the galaxy prior…", group: "Galaxies", disabled: !candidateOk,
      run: () => { void activateGalaxyModel(!!api?.calibration.is_active); } },
    { id: "galaxies-query", label: "Query Q1 galaxies (MER + PHZ) and refit…", group: "Galaxies", keywords: ["euclid", "archive", "fit"],
      disabled: !api?.authenticated, run: () => { how.reveal(); void queryGalaxies(); } },
    { id: "galaxies-rebuild", label: "Rebuild the galaxy plots", group: "Galaxies", run: () => { void rebuildGalaxyPlots(); } },
    { id: "galaxies-cones", label: "Show the population cones on the sky", group: "Galaxies", keywords: ["atlas"],
      run: () => navigate(atlasHref({ layers: ["q1-tiles:0.2", "population-cones"] })) },
  ]);
  return (
    <Page className="rl-page syn-page">
      <Toolbar label="Galaxy controls">
        <ToolbarGroup label="View" hideLabel>
          <Segmented size="sm" aria-label="Galaxy view" value={view} onChange={setView} options={[...GALAXY_VIEWS]} />
        </ToolbarGroup>
        {api && <ToolbarGroup label="Problems" hideLabel><Problems api={api} /></ToolbarGroup>}
        <ToolbarSpacer />
        <DrawerButton flag="prior" icon="activity" hint="The galaxy prior: its laws, the fit diagnostic, fit and activate">Prior</DrawerButton>
        <DrawerButton flag="how" icon="database" hint="The Q1 MER + PHZ query and the TNG templates behind the prior">How this is produced</DrawerButton>
        <Menu label="Download the galaxy figure" trigger={<IconButton size="sm" icon="download" label="Download the galaxy figure" />}
          items={FORMATS.map((f) => ({ label: `Galaxy distributions plate · ${f.toUpperCase()}`,
            onSelect: () => downloadUrl(plateUrl(training, f), `euclidpolish_galaxy_distributions_2x2.${f}`) }))} />
        <IconButton size="sm" icon="reset" label="Refresh" loading={resource.fetching} onClick={() => resource.reload()} />
      </Toolbar>
      <LoadState loading={resource.loading && !api} error={resource.error} onRetry={resource.reload}>
        {api && (
          <>
            {view === "distributions" ? <Distributions api={api} />
              : view === "relations" ? <Relations candidate={api.calibration.candidate} />
                : view === "joint" ? <Joint api={api} />
                  : <Templates />}
            <Drawer flag="prior" title="Prior" sub="Q1 counts and radii set the laws; colours and SFR are resampled from real Q1 rows">
              <PriorPanel api={api} />
            </Drawer>
            <Drawer flag="how" title="How this is produced" sub="the Q1 MER + PHZ brackets and the TNG50-1 templates">
              <QueryPanel api={api} />
              <TngSteps />
            </Drawer>
          </>
        )}
      </LoadState>
    </Page>
  );
}
