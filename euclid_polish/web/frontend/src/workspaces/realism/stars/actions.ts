/* The Stars tab's jobs (buttons and palette share one spec each). */
import { runJob } from "../jobs";

export const STAR_URL = {
  query: "/api/star-distribution/query",
  fit: "/api/star-distribution/fit",
  activate: "/api/star-distribution/activate",
  figure: "/view/star-population-calibration",
} as const;

export const queryStars = () => runJob({
  url: STAR_URL.query, label: "Query stars · MER + PHZ + Gaia",
  question: {
    title: "Query stars (MER + PHZ + Gaia)?", confirmLabel: "Query",
    message: "Queries the Q1 stellar count brackets and the fixed-field Gaia–Euclid colour sample from the "
      + "Euclid archive. No galaxy selection is used.",
  },
});

export const fitStars = () => runJob({ url: STAR_URL.fit, label: "Fit stellar prior from cached data" });

export const activateStars = (isActive: boolean) => runJob({
  url: STAR_URL.activate, label: isActive ? "Re-activate stellar prior" : "Activate stellar prior",
  question: {
    title: "Activate this stellar prior?", confirmLabel: "Activate",
    message: "synthetic_generate reads the active stellar prior; this replaces it.",
  },
});

export const starFigureUrl = (format: "png" | "pdf" | "svg") => `${STAR_URL.figure}?format=${format}&dpi=300`;
