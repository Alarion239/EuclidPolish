/* The one caption line of every plate on Figures › Plates: what it is made
 * with, whether that is current, and when it was made —
 * "made with galaxy model v15 · current · activated 13 d ago". Pure;
 * plateStatus.test.ts.
 *
 * Sources (all read-only GETs the page already makes):
 *  - the three calibration plates render on request from the reviewed
 *    calibration (routes/galaxy_distributions.py, routes/star_distribution.py),
 *    so their date is when that calibration was activated (the Synthetic ›
 *    Status rows of GET /api/realism/overview: `galaxy-model`, `star-prior`
 *    and, for the galaxy 2×2, the `galaxy-plots` cache it reads);
 *  - a NEXUS render is current when its recorded model fingerprint is the
 *    catalogue's (GET /api/models) and no tile output predates the model;
 *  - the synthetic poster scene is the last pulled poster_cutout result. */
import { formatRelative } from "../../../format";
import type { PlateRender, PlateRun, PosterStatus } from "../api";

export type OverviewItem = {
  id: string; state?: string; title?: string; detail?: string | null;
  facts?: Record<string, unknown> | null; records?: { prior_at?: string | null } | null;
};
/** The slice of GET /api/realism/overview read here. */
export type OverviewSlice = { items?: OverviewItem[] | null };
export type CatalogSlice = { models?: readonly { spec: string; fingerprint?: string | null }[] | null };

export type PlateState = "current" | "stale" | "not-active";
export type PlateCaption = {
  /** "galaxy model v15"; null when there is nothing to draw (see `empty`). */
  madeWith: string | null;
  state: PlateState | null;
  verb: "activated" | "built" | "rendered" | "pulled" | null;
  at: string | null;
  /** An extra clause ("a newer candidate is not active"). */
  note?: string | null;
  /** Why it is stale / not active, and where to fix it (a tooltip). */
  reason?: string | null;
  /** What the caption says when there is nothing to make the plate from. */
  empty?: string | null;
  /** Replaces "made with <madeWith>" (the poster scene is not made with a model). */
  phrase?: string | null;
  tone: "neutral" | "warn";
};

const STATE_TEXT: Record<PlateState, string> = { current: "current", stale: "stale", "not-active": "not active" };
const NONE: PlateCaption = { madeWith: null, state: null, verb: null, at: null, tone: "neutral" };

const item = (o: OverviewSlice | null | undefined, id: string) => o?.items?.find((i) => i.id === id) ?? null;
const str = (v: unknown) => (typeof v === "string" && v ? v : null);
const facts = (i: OverviewItem | null) => (i?.facts ?? {}) as Record<string, unknown>;

/** The caption's parts, joined with " · " by the page. */
export function captionParts(c: PlateCaption, now: number = Date.now()): string[] {
  if (!c.madeWith) return c.empty ? [c.empty] : [];
  return [
    c.phrase ?? `made with ${c.madeWith}`,
    c.state ? STATE_TEXT[c.state] : null,
    c.verb && c.at ? `${c.verb} ${formatRelative(c.at, now)}` : null,
    c.note ?? null,
  ].filter((p): p is string => !!p);
}

function galaxyModel(o: OverviewSlice | null | undefined): { text: string | null; current: boolean; version: string | null; at: string | null; known: boolean } {
  const g = item(o, "galaxy-model");
  if (!g) return { text: null, current: false, version: null, at: null, known: false };
  const f = facts(g);
  const version = f.version != null ? String(f.version) : null;
  const cand = str(f.candidate_fingerprint);
  const active = str(f.active_fingerprint);
  const current = !!f.is_active && !!active && (!cand || cand === active);
  return { text: version ? `galaxy model v${version}` : "the galaxy model", current, version, at: g.records?.prior_at ?? null, known: !!(cand || active) };
}

/** The caption of a calibration plate (`population`, `stars`, `galaxies`). */
export function staticPlateCaption(id: string, o: OverviewSlice | null | undefined): PlateCaption {
  if (!o) return NONE;
  if (id === "population") {
    // /view/population-atlas draws the joint galaxy CANDIDATE.
    const g = galaxyModel(o);
    if (!g.known) return { ...NONE, empty: "no galaxy fit yet" };
    if (g.current) return { madeWith: g.text, state: "current", verb: "activated", at: g.at, tone: "neutral" };
    return {
      madeWith: g.version ? `the galaxy candidate v${g.version}` : "the galaxy candidate", state: "not-active", verb: null, at: null, tone: "warn",
      reason: "Generation uses the active galaxy model, not this candidate. Activate it on Synthetic › Galaxies (Prior).",
    };
  }
  if (id === "stars") {
    // /view/star-population-calibration draws the ACTIVE prior, else the candidate.
    const s = item(o, "star-prior");
    if (!s) return NONE;
    const f = facts(s);
    const active = str(f.active_fingerprint);
    const cand = str(f.candidate_fingerprint);
    if (active) {
      const newer = !!cand && cand !== active && !!f.candidate_valid;
      return {
        madeWith: "the active stellar prior", state: "current", verb: "activated", at: s.records?.prior_at ?? null, tone: "neutral",
        note: newer ? "a newer candidate is not active" : null,
        reason: newer ? "A newer stellar candidate is fitted but not active; the plate shows the active prior. Activate it on Synthetic › Stars (Prior)." : null,
      };
    }
    if (cand) {
      return { madeWith: "the stellar candidate", state: "not-active", verb: null, at: null, tone: "warn",
        reason: "No stellar prior is active: the plate shows the candidate. Activate it on Synthetic › Stars (Prior)." };
    }
    return { ...NONE, empty: "no stellar fit yet" };
  }
  if (id === "galaxies") {
    // /view/galaxy-distribution-plate reads the galaxy plots cache (Q1 vs the active law).
    const g = galaxyModel(o);
    const p = item(o, "galaxy-plots");
    const pf = facts(p);
    if (!p || !pf.present) return { ...NONE, empty: "the galaxy plots are not built yet (Synthetic › Status)" };
    const stale = !!pf.stale;
    return {
      madeWith: g.text ?? "the active galaxy model", state: stale ? "stale" : "current", verb: "built", at: str(pf.built_at),
      tone: stale ? "warn" : "neutral",
      reason: stale ? `${str(pf.reason) ?? "the plots are out of date"} — rebuild the plots on Synthetic › Status` : null,
    };
  }
  return NONE;
}

/** The caption of one NEXUS comparison render. */
export function nexusCaption(render: PlateRender, run: PlateRun | undefined, catalog: CatalogSlice | null | undefined): PlateCaption {
  const at = render.created ?? run?.updated ?? null;
  const label = render.model_label ?? render.model_short ?? render.model ?? "an unknown model";
  if (render.legacy || !render.model) {
    return { madeWith: `${label} (legacy SR)`, state: "stale", verb: "rendered", at, tone: "warn",
      reason: "Drawn from a legacy SR output, not the production gate. Render the plates again with production." };
  }
  const staleTiles = render.tiles.filter((t) => t.model_state && t.model_state !== "current").length;
  const current = catalog?.models?.find((m) => m.spec === render.model)?.fingerprint ?? null;
  const earlier = !!render.model_fingerprint && !!current && render.model_fingerprint !== current;
  if (earlier || staleTiles) {
    return { madeWith: label, state: "stale", verb: "rendered", at, tone: "warn",
      reason: earlier ? `Rendered with an earlier fit of ${render.model}; render again for the current one.`
        : `${staleTiles} tile output${staleTiles === 1 ? "" : "s"} predate${staleTiles === 1 ? "s" : ""} the current ${render.model}.` };
  }
  const known = !!render.model_fingerprint && !!current;
  return { madeWith: label, state: known ? "current" : null, verb: "rendered", at, tone: "neutral" };
}

/** The caption of the synthetic poster scene (the last pulled poster_cutout). */
export function posterCaption(status: PosterStatus | null | undefined): PlateCaption {
  if (!status) return NONE;
  // Nothing pulled: the plate's empty image says what to do.
  if (!status.png) return NONE;
  return { madeWith: "poster_cutout", phrase: "synthetic scene from the poster_cutout step", state: null, verb: "pulled",
    at: status.png.pulled_at, tone: "neutral" };
}
