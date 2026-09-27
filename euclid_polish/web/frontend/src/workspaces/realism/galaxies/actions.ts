/* The Galaxies tab's jobs, shared by its buttons and its palette actions (one
   spec each, so both start the same keyed job behind the same confirm). */
import { ENDPOINT } from "../api";
import { runJob } from "../jobs";

export const GALAXY_URL = {
  query: "/api/galaxy-distributions/query-q1-counts",
  cones: "/api/galaxy-distributions/refresh-population-cones",
  build: "/api/galaxy-distributions/build",
  activate: "/api/galaxy-distributions/activate",
} as const;

export const queryGalaxies = () => runJob({
  url: GALAXY_URL.query, label: "Query MER + PHZ",
  question: {
    title: "Query MER + PHZ?", confirmLabel: "Query",
    message: "Runs the aperture-count and Sérsic-radius bracket queries against the Euclid archive "
      + "(cached checkpoints are skipped), fits the galaxy model and rebuilds the plots.",
  },
});

export const requeryCones = () => runJob({
  url: GALAXY_URL.cones, label: "Re-query population cones",
  question: {
    title: "Re-query the 24 population cones?", confirmLabel: "Re-query",
    message: "Needed after a catalogue schema change: the colour+SFR fit refuses a stale cache. Queries the Euclid archive.",
  },
});

export const rebuildGalaxyPlots = () => runJob({ url: GALAXY_URL.build, label: "Rebuild galaxy plots" });

export const activateGalaxyModel = (isActive: boolean) => runJob({
  url: GALAXY_URL.activate, label: isActive ? "Re-activate galaxy model" : "Activate galaxy model",
  question: {
    title: "Activate this galaxy model?", confirmLabel: "Activate",
    message: "synthetic_generate reads the active galaxy model; this replaces it (and sets the generation density).",
  },
});

export type PlateFormat = "svg" | "pdf" | "png";

/** The publication plate (server-rendered from the cached arrays). */
export const plateUrl = (training: boolean, format: PlateFormat, extra = "") =>
  `${ENDPOINT.plate(training)}&format=${format}&dpi=300${extra}`;
