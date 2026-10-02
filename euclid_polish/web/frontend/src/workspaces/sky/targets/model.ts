/* Sky › Targets, pure logic (unit-tested in model.test.ts): the target sets,
 * the eval-store adapter that maps real tiles (`production_state`) and the
 * catalogue evaluation's objects (`current / stale / unknown / failed`) onto
 * ONE vocabulary — current / stale / missing, plus "made by <model>" in
 * words — the per-set sentence, the flux SR/LR strip and the "Run production
 * on stale" plan. No React, no fetch. */
import { formatCount, formatNumber } from "../../../format";
import { extent } from "../../../ticks";
import {
  splitRef, type EvalObjectCard, type EvalRow, type SourceId, type SourcesPayload, type TileCard, type TileModelRow, type TileRow,
} from "../results/api";
import { fluxDeltaMag, deltaMagWarn, headlineSpec, num, sortSpecs, specWords } from "../results/model";

/* ── the sets ──────────────────────────────────────────────────────────── */

export type TargetSetId = "nexus" | "poster" | "lenses" | "galaxies" | "cached" | "legacy" | "pairs";
export type TargetState = "current" | "stale" | "missing";

export type TargetSet = {
  id: TargetSetId;
  label: string;
  /** Singular and plural noun of one row. */
  noun: [string, string];
  /** The C9 real-tile store the set lists (tile sets). */
  store?: SourceId;
  /** Catalogue-evaluation grades the set lists (catalogue sets). */
  grades?: readonly string[];
  /** Listed under "more" (not a science target of the paper). */
  more?: boolean;
  /** What the set is (chip tooltip) and what an empty set needs. */
  about: string;
  empty: string;
};

export const TARGET_SETS: readonly TargetSet[] = [
  { id: "nexus", label: "NEXUS × JWST", noun: ["tile", "tiles"], store: "nexus",
    about: "Euclid tiles over the NEXUS F200W mosaic, each with its JWST image.",
    empty: "No NEXUS tiles yet: cache the NEXUS mosaic from the atlas's JWST menu." },
  { id: "poster", label: "Poster galaxy", noun: ["tile", "tiles"], store: "poster",
    about: "The poster galaxy's 102.4″ LR (one row; its SRs are the model outputs).",
    empty: "No poster target file in poster/." },
  { id: "lenses", label: "Lens candidates", noun: ["reconstruction", "reconstructions"], grades: ["A", "B", "C"],
    about: "Euclid Q1 strong-lens candidates (grades A–C) reconstructed by the grouped analysis.",
    empty: "No lens candidates yet: run the grouped analysis (Sources)." },
  { id: "galaxies", label: "Q1 galaxies", noun: ["reconstruction", "reconstructions"], grades: ["gal"],
    about: "Q1 galaxies drawn from the lens fields, reconstructed by the grouped analysis.",
    empty: "No Q1 galaxies yet: query galaxies, then run the grouped analysis (Sources)." },
  { id: "cached", label: "Cached tiles", noun: ["tile", "tiles"], store: "tile",
    about: "Four-band 25.6″ tiles cached anywhere in Q1.",
    empty: "No cached tiles yet: right-click the sky in the atlas, or Sources › Cache a tile." },
  { id: "legacy", label: "Legacy field", noun: ["tile", "tiles"], store: "field", more: true,
    about: "The 256″ legacy real field cut into 100 tiles.",
    empty: "No legacy real field is cached." },
  { id: "pairs", label: "JWST pairs", noun: ["tile", "tiles"], store: "pair", more: true,
    about: "Downloaded JWST × Euclid comparison pairs.",
    empty: "No JWST × Euclid pair yet: download one from the atlas's JWST menu." },
];
export const SET_BY_ID: Record<TargetSetId, TargetSet> =
  Object.fromEntries(TARGET_SETS.map((s) => [s.id, s])) as Record<TargetSetId, TargetSet>;
const SET_IDS = new Set<string>(TARGET_SETS.map((s) => s.id));
/** The sets a plain visit shows (every science target; "more" stays off). */
export const DEFAULT_SETS: readonly TargetSetId[] = TARGET_SETS.filter((s) => !s.more).map((s) => s.id);

/** The old real-tile store ids (`?src=` of the interim handover) → sets. */
const STORE_SETS: Record<string, TargetSetId[]> = {
  nexus: ["nexus"], poster: ["poster"], tile: ["cached"], field: ["legacy"], pair: ["pairs"],
  eval: ["lenses", "galaxies"],
};

const inOrder = (ids: Iterable<string>): TargetSetId[] => {
  const want = new Set(ids);
  return TARGET_SETS.filter((s) => want.has(s.id)).map((s) => s.id);
};

const splitList = (raw: string) => raw.split(",").map((s) => s.trim()).filter(Boolean);
/** One id of `?set=`: a set id, or an old store id that a redirect carried
 *  over inside a comma list (`/sky/results?src=nexus,tile` → `?set=nexus,tile`;
 *  the redirect maps only whole values). */
const setIds = (id: string): string[] => (SET_IDS.has(id) ? [id] : STORE_SETS[id] ?? []);

/** The chosen sets from `?set=` (v2, old store ids mapped), else the interim
 *  `?src=` store ids; unknown ids are dropped, the order is the chips'.
 *  Empty = the default. */
export function parseSets(set: string, src: string): TargetSetId[] {
  if (set.trim()) return inOrder(splitList(set).flatMap(setIds));
  return inOrder(splitList(src).flatMap((s) => STORE_SETS[s] ?? []));
}

/** Old Catalog-eval group ids of `?g=` → a lens grade, or "gal". */
const OLD_GROUP: Record<string, string> = { lensA: "A", lensB: "B", lensC: "C", galaxies: "gal" };

export type TargetsQuery = { set: string; src: string; g: readonly string[]; st: string; state: string };
export type TargetsPatch = { set?: string; src?: string; g?: string[]; st?: string; state?: string };

/** The one-shot rewrite of an old Catalog-eval / Real-results link that
 *  landed on Targets (null when there is nothing to rewrite):
 *  - `?g=` was the old page's group list (`A,B`, `lensA`, `gal`, `galaxies`):
 *    it chose the catalogue sets too, so when the sets are the catalogue's
 *    (or the default) they keep lenses only when a lens grade is listed and
 *    galaxies only when `gal` is (never adding a set); `?g=` keeps only the
 *    lens grades.
 *  - `?st=` was the old state filter (`unknown` is stale now): it becomes
 *    `?state=` unless one is already set.
 *  - old store ids inside `?set=` are written as set ids. */
export function legacyTargetsPatch(q: TargetsQuery): TargetsPatch | null {
  const patch: TargetsPatch = {};
  const sets = parseSets(q.set, q.src);
  const groups = q.g.map((g) => OLD_GROUP[g] ?? g);
  const grades = lensGrades(groups);
  const wantsLenses = grades.length > 0;
  const wantsGalaxies = groups.includes("gal");
  const catalogueOnly = !sets.length || sets.every((s) => s === "lenses" || s === "galaxies");
  let next = sets;
  if (q.g.length && catalogueOnly && (wantsLenses || wantsGalaxies)) {
    const pool: readonly TargetSetId[] = sets.length ? sets : ["lenses", "galaxies"];
    const narrowed = pool.filter((s) => (s === "lenses" ? wantsLenses : wantsGalaxies));
    if (narrowed.length) next = narrowed;
  }
  const setText = next.join(",");
  if (q.set.trim() && setText !== splitList(q.set).join(",")) patch.set = setText;
  else if (!q.set.trim() && next !== sets) { patch.set = setText; if (q.src) patch.src = ""; }
  if (q.g.length && grades.join(",") !== q.g.join(",")) patch.g = grades;
  if (q.st) {
    patch.st = "";
    const st = q.st === "unknown" ? "stale" : q.st;
    if ((!q.state || q.state === "all") && (st === "current" || st === "stale" || st === "missing")) patch.state = st;
  }
  return Object.keys(patch).length ? patch : null;
}

const isLensGrade = (g: string) => g === "A" || g === "B" || g === "C";
const catalogueSet = (grade: string): TargetSetId | null => (isLensGrade(grade) ? "lenses" : grade === "gal" ? "galaxies" : null);
const evalOk = (r: Pick<EvalRow, "ok">) => String(r.ok ?? "").toLowerCase() === "true";

/** The lens grades of a `?g=` list (A, B, C, or the old `lensA`…; anything
 *  else — a galaxy group from an old link — is not a grade), each once. */
export function lensGrades(g: readonly string[]): string[] {
  return [...new Set(g.map((x) => OLD_GROUP[x] ?? x).filter(isLensGrade))];
}

/** Rows per set for the chips: tile stores from `/api/real/sources`, the
 *  catalogue sets from the evaluation manifest's reconstructions (a failed
 *  cutout has no LR to run on: it is counted by `failedCutouts`). Unknown
 *  counts are left out. */
export function setCounts(
  sources: SourcesPayload | null | undefined, evalRows: readonly EvalRow[] | null | undefined,
): Partial<Record<TargetSetId, number>> {
  const out: Partial<Record<TargetSetId, number>> = {};
  const bySource = new Map((sources?.sources ?? []).map((s) => [s.id, s.count]));
  for (const s of TARGET_SETS) {
    if (s.store && bySource.has(s.store)) out[s.id] = bySource.get(s.store);
  }
  if (evalRows) {
    out.lenses = 0;
    out.galaxies = 0;
    for (const r of evalRows) {
      const set = catalogueSet(String(r.grade ?? "").trim());
      if (set && evalOk(r)) out[set] = (out[set] ?? 0) + 1;
    }
  }
  return out;
}

/** Catalogue cutouts that failed to download, per catalogue set. With
 *  `grades` (the ?g= lens-grade filter), only the lens candidates of those
 *  grades count, as in the table. */
export function failedCutouts(
  evalRows: readonly EvalRow[] | null | undefined, grades: readonly string[] = [],
): { lenses: number; galaxies: number } {
  const out = { lenses: 0, galaxies: 0 };
  for (const r of evalRows ?? []) {
    const grade = String(r.grade ?? "").trim();
    const set = catalogueSet(grade);
    if (set !== "lenses" && set !== "galaxies") continue;
    if (set === "lenses" && grades.length && !grades.includes(grade)) continue;
    if (!evalOk(r)) out[set] += 1;
  }
  return out;
}

/* ── one row per target ────────────────────────────────────────────────── */

export type TargetRow = {
  /** Unique over every set: the real-tile ref, `eval/<object>` for a failed cutout too. */
  key: string;
  set: TargetSetId;
  id: string;
  label: string;
  ra: number | null;
  dec: number | null;
  field: string | null;
  state: TargetState;
  /** Why stale or missing, in words. */
  reason: string | null;
  /** The model that made the production SR, in words ("spatial gate over 30 members"). */
  madeBy: string | null;
  /** Total SR/LR flux in VIS of the headline output (the catalogue objects' own flux column). */
  flux: number | null;
  /** Worst-band hole % and median R of a scored output (real-tile metrics). */
  holes: number | null;
  medR: number | null;
  scored: boolean;
  grade: string | null;
  hasJwst: boolean;
  /** The real-tile ref the card opens (`tile:<ref>`); null when there is nothing to open. */
  ref: string | null;
  /** The NEXUS field a NEXUS tile belongs to (its production run). */
  nexusField: string | null;
};

export { specWords };

/** A recorded combiner kind in words (`spatial_gate` → "spatial gate", any RBF → "RBF combiner"). */
function combinerWords(kind: string): string {
  return /rbf/i.test(kind) ? "RBF combiner" : kind.replace(/_/g, " ");
}

/** A legacy SR's label in words: "Poster SR · raw_incremental_minmeanmax_rbf (10
 *  members)" → "RBF combiner (10 members)"; a label already in words stays. */
function legacyWords(spec: string, label: string | null | undefined): string {
  const text = String(label ?? "").replace(/^[^·]*\bSR · /, "").trim();
  const members = /\((\d+ members?)\)\s*$/.exec(text)?.[1];
  const head = text.replace(/\s*\(\d+ members?\)\s*$/, "");
  if (!head || /_/.test(head)) return `${specWords(spec)}${members ? ` (${members})` : ""}`;
  return text;
}

/** "Made by" of a real tile: its production output, else the legacy SR that
 *  stands in for it; null when neither exists. */
export function madeByTile(models: Record<string, Pick<TileModelRow, "label" | "legacy"> | undefined> | undefined): string | null {
  const ms = models ?? {};
  const prod = ms.production;
  if (prod) {
    const label = String(prod.label ?? "").replace(/^Production\s*·\s*/, "").trim();
    return label || "production";
  }
  const legacy = sortSpecs(Object.keys(ms)).find((s) => ms[s]?.legacy);
  return legacy ? `legacy ${legacyWords(legacy, ms[legacy]?.label)}` : null;
}

/** "Made by" of a catalogue object from its recorded model. With no
 *  combiner kind, a current SR is the member mean (the production model is
 *  the mean then); a stale one may predate the combiner record, so the words
 *  say the combiner is not recorded instead of guessing. */
export function madeByEval(r: Pick<EvalRow, "n_members" | "combiner_kind"> & { state?: string | null }): string | null {
  const n = typeof r.n_members === "number" ? r.n_members : null;
  if (n == null) return null;
  const members = `${n} member${n === 1 ? "" : "s"}`;
  if (r.combiner_kind) return `${combinerWords(String(r.combiner_kind))} over ${members}`;
  if (r.state === "current") return n === 1 ? "a single member" : `mean of ${members}`;
  return `${n}-member ensemble (combiner not recorded)`;
}

const TILE_STATE: Record<string, TargetState> = { current: "current", stale: "stale" };

type StateWords = { state: TargetState; reason: string | null; madeBy: string | null };

/** A real tile's production SR in the one vocabulary (the store's
 *  `production_state`: current / stale / anything else = missing). */
export function tileState(t: Pick<TileRow, "production_state" | "models">): StateWords {
  const state = TILE_STATE[String(t.production_state)] ?? "missing";
  const models = t.models ?? {};
  const reason = state === "missing" ? "no production SR yet"
    : state === "stale" ? (models.production ? "made before the current production model" : "only a legacy SR exists")
      : null;
  return { state, reason, madeBy: madeByTile(models) };
}

/** A catalogue object's production SR in the one vocabulary (its evaluation
 *  record: current / stale / unknown → stale; a failed cutout → missing). */
export function evalState(r: Pick<EvalRow, "ok" | "state" | "state_reason" | "error" | "n_members" | "combiner_kind">): StateWords {
  if (!evalOk(r)) return { state: "missing", reason: `the cutout failed${r.error ? `: ${r.error}` : ""}`, madeBy: null };
  const state: TargetState = r.state === "current" ? "current" : "stale";
  return { state, reason: state === "current" ? null : plainReason(r.state_reason), madeBy: madeByEval(r) };
}

/** The evaluation's stale reason in plain words: "22 member(s)" → "22
 *  members", "STARFULL" → "starfull". */
export function plainReason(reason: string | null | undefined): string | null {
  if (!reason) return null;
  return reason
    .replace(/\b(\d+) member\(s\)/g, (_, n: string) => `${n} member${n === "1" ? "" : "s"}`)
    .replace(/\bSTAR(FULL|LESS)\b/g, (m) => m.toLowerCase());
}

/** Real tiles of one set as target rows. */
export function tileTargets(set: TargetSetId, tiles: readonly TileRow[]): TargetRow[] {
  return tiles.map((t) => {
    const { state, reason, madeBy } = tileState(t);
    const models = t.models ?? {};
    const spec = models.production?.summary ? "production" : headlineSpec(t);
    const out = spec ? models[spec] : undefined;
    const holes = num(out?.summary?.hole_pct_max);
    const medR = num(out?.summary?.median_R);
    const field = typeof t.extras?.field_id === "string" ? t.extras.field_id : null;
    return {
      key: t.ref, set, id: t.id, label: t.label, ra: num(t.ra), dec: num(t.dec), field: t.field ?? null,
      state, reason, madeBy, flux: num(out?.flux_ratio?.VIS),
      holes, medR, scored: holes != null || medR != null, grade: typeof t.extras?.grade === "string" && t.extras.grade ? t.extras.grade : null,
      hasJwst: !!t.has_jwst, ref: t.ref, nexusField: set === "nexus" ? field : null,
    };
  });
}

/** The catalogue evaluation's real objects (lens candidates, Q1 galaxies)
 *  as target rows; the synthetic stamps are Models' (left out). A failed
 *  cutout is a row only with `failed` (it reads "missing", with its error). */
export function evalTargets(rows: readonly EvalRow[], opts: { failed?: boolean } = {}): TargetRow[] {
  const out: TargetRow[] = [];
  for (const r of rows) {
    const grade = String(r.grade ?? "").trim();
    const set = catalogueSet(grade);
    if (!set) continue;
    const ok = evalOk(r);
    if (!ok && !opts.failed) continue;
    const sub = String(r.out_subdir || r.id || "");
    const { state, reason, madeBy } = evalState(r);
    out.push({
      key: `eval/${sub}`, set, id: String(r.id ?? sub), label: String(r.id ?? sub), ra: num(r.ra), dec: num(r.dec),
      field: r.field ?? null, state, reason, madeBy, flux: ok ? num(r.flux_ratio_sr_over_lr) : null,
      holes: null, medR: null, scored: false, grade: grade || null, hasJwst: false,
      ref: ok && r.realtile ? String(r.realtile) : null, nexusField: null,
    });
  }
  return out;
}

/* ── the tile card's status sentence ───────────────────────────────────── */

export type CardStatus = StateWords & { text: string };

/** One sentence for the tile card: the production SR's state, who made it
 *  and why it is stale. A catalogue object (`eval/`) reads its evaluation
 *  record (null while that loads), like its row in Targets. */
export function cardStatus(
  card: Pick<TileCard, "source" | "production_state" | "models">,
  evalCard?: Pick<EvalObjectCard, "ok" | "state" | "state_reason" | "error" | "members"> | null,
): CardStatus | null {
  let words: StateWords;
  if (card.source === "eval") {
    if (!evalCard) return null;
    const members = evalCard.members;
    words = evalState({
      ok: evalCard.ok, state: evalCard.state, state_reason: evalCard.state_reason, error: evalCard.error,
      n_members: members?.member_labels ? members.member_labels.length : null, combiner_kind: members?.combiner_kind ?? null,
    });
  } else {
    words = tileState({ production_state: card.production_state, models: card.models });
  }
  // The evaluation's own reason may already name the model ("membership
  // changed: made by 22 member(s), …"): then it is not said twice.
  const by = words.madeBy && !/\bmade by\b/i.test(words.reason ?? "") ? `, made by ${words.madeBy}` : "";
  const text = words.state === "missing"
    ? `No production SR yet${words.reason && words.reason !== "no production SR yet" ? `: ${words.reason}` : ""}.`
    : `Production SR is ${words.state}${by}${words.reason ? `: ${words.reason}` : ""}.`;
  return { ...words, text };
}

/* ── states, sentence ──────────────────────────────────────────────────── */

export type StateCounts = { all: number; current: number; stale: number; missing: number };

export function stateCounts(rows: readonly Pick<TargetRow, "state">[]): StateCounts {
  const out: StateCounts = { all: rows.length, current: 0, stale: 0, missing: 0 };
  for (const r of rows) out[r.state] += 1;
  return out;
}

export function withState<T extends Pick<TargetRow, "state">>(rows: readonly T[], state: string): T[] {
  return !state || state === "all" ? [...rows] : rows.filter((r) => r.state === state);
}

function median(values: readonly number[]): number | null {
  if (!values.length) return null;
  const v = [...values].sort((a, b) => a - b);
  const mid = Math.floor(v.length / 2);
  return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
}

export type SetSentence = {
  set: TargetSetId; label: string; noun: string; total: number;
  current: number; stale: number; missing: number;
  medianFlux: number | null; fluxN: number;
  /** The median flux is more than 0.1 mag off the LR (±10 %). */
  fluxWarn: boolean;
  /** Median worst-band holes % over the scored rows (real tiles only), and how many are scored. */
  medianHoles: number | null; scoredN: number;
};

/** One set's facts for its sentence (the rows are that set's). */
export function setSentence(set: TargetSetId, rows: readonly TargetRow[]): SetSentence {
  const def = SET_BY_ID[set];
  const c = stateCounts(rows);
  const fluxes = rows.map((r) => r.flux).filter((f): f is number => f != null);
  const medianFlux = median(fluxes);
  const holes = rows.filter((r) => r.scored && r.holes != null).map((r) => r.holes as number);
  return {
    set, label: def.label, noun: def.noun[1], total: rows.length,
    current: c.current, stale: c.stale, missing: c.missing,
    medianFlux, fluxN: fluxes.length, fluxWarn: medianFlux != null && deltaMagWarn(fluxDeltaMag(medianFlux)),
    medianHoles: median(holes), scoredN: rows.filter((r) => r.scored).length,
  };
}

export type SentencePiece = { text: string; num?: boolean; warn?: boolean };

/** The set's sentence as pieces (numbers marked, the out-of-tolerance ones
 *  warned), so the page can bold the numbers and tests read the text. */
export function sentencePieces(s: SetSentence, gate: string): SentencePiece[] {
  const def = SET_BY_ID[s.set];
  const [one, many] = def.noun;
  const n = (v: number, warn = false): SentencePiece => ({ text: String(v), num: true, warn });
  const t = (text: string): SentencePiece => ({ text });
  const out: SentencePiece[] = [t(`${s.label}: `)];
  if (!s.total) return [...out, t("none yet")];
  const only = s.current === s.total ? "current" : s.stale === s.total ? "stale" : s.missing === s.total ? "missing" : null;
  if (only === "stale") {
    out.push(...(s.total === 1 ? [t(`the ${one} predates the current ${gate}`)]
      : [t("all "), n(s.total, true), t(` ${many} predate the current ${gate}`)]));
  } else if (only === "current") {
    out.push(...(s.total === 1 ? [t(`the ${one} is current`)] : [t("all "), n(s.total), t(` ${many} are current`)]));
  } else if (only === "missing") {
    out.push(...(s.total === 1 ? [t(`the ${one} has no production SR yet`)]
      : [t("none of the "), n(s.total), t(` ${many} has a production SR yet`)]));
  } else {
    const parts: SentencePiece[][] = [];
    if (s.current) parts.push([n(s.current), t(" current")]);
    if (s.stale) parts.push([n(s.stale, true), t(" stale")]);
    if (s.missing) parts.push([n(s.missing), t(" without a production SR")]);
    parts.forEach((p, i) => { if (i) out.push(t(", ")); out.push(...p); });
  }
  // Holes are measured on scored real tiles only; the table shows them as
  // columns only when every row in view is scored, so the sentence carries them.
  const holes: SentencePiece[] = s.medianHoles != null
    ? [t(s.scoredN === 1 ? ", worst-band holes " : ", median worst-band holes "),
      { text: `${formatNumber(s.medianHoles, { digits: 1 })} %`, num: true }]
    : [];
  if (s.medianFlux != null && s.fluxN) {
    const f: SentencePiece = { text: formatNumber(s.medianFlux, { digits: 2 }), num: true, warn: s.fluxWarn };
    const scored = def.store ? "scored " : "";
    const over = Math.max(s.fluxN, s.scoredN);
    if (over === 1) {
      out.push(t(" · flux SR/LR "), f, ...holes);
      if (s.total > 1) out.push(t(` (1 ${scored}${one})`));
    } else {
      out.push(t(" · median flux SR/LR "), f, ...holes);
      if (over < s.total) out.push(t(" over "), n(over), t(` ${scored}${many}`));
    }
  } else if (holes.length) {
    out.push(t(" · "), { ...holes[0], text: holes[0].text.replace(/^, /, "") }, holes[1]);
    if (s.scoredN < s.total) out.push(t(" over "), n(s.scoredN), t(` scored ${s.scoredN === 1 ? one : many}`));
  }
  return out;
}

export function sentenceText(s: SetSentence, gate: string): string {
  return sentencePieces(s, gate).map((p) => p.text).join("");
}

/* ── the flux SR/LR strip ──────────────────────────────────────────────── */

export type StripPoint = { key: string; set: TargetSetId; x: number; y: number; ref: string | null; label: string };
/** `n` targets of the set have a flux (a dot each) out of its `total` rows in view. */
export type StripRow = { set: TargetSetId; label: string; n: number; total: number; median: number | null };

/** A stable jitter in [-0.28, 0.28] from the row key (the dots never move). */
function jitter(key: string): number {
  let h = 2166136261;
  for (let i = 0; i < key.length; i++) h = Math.imul(h ^ key.charCodeAt(i), 16777619);
  return (((h >>> 0) % 1000) / 999 - 0.5) * 0.56;
}

/** One strip row per chosen set (y = its index), a dot per target with a
 *  flux, and the set's median; the x domain always includes 1 (flux kept). */
export function fluxStrip(sets: readonly TargetSetId[], rows: readonly TargetRow[]): {
  rows: StripRow[]; points: StripPoint[]; domain: [number, number];
} {
  const points: StripPoint[] = [];
  const out: StripRow[] = sets.map((set, i) => {
    const xs: number[] = [];
    let total = 0;
    for (const r of rows) {
      if (r.set !== set) continue;
      total += 1;
      if (r.flux == null) continue;
      xs.push(r.flux);
      points.push({ key: r.key, set, x: r.flux, y: i + jitter(r.key), ref: r.ref, label: r.label });
    }
    return { set, label: SET_BY_ID[set].label, n: xs.length, total, median: median(xs) };
  });
  const span = extent([...points.map((p) => p.x), 1]) ?? [0.5, 1.5];
  const pad = Math.max(0.02, (span[1] - span[0]) * 0.06);
  // Floored at 0 unless a target's SR flux went negative (background-dominated): it stays in view.
  const lo = span[0] < 0 ? span[0] - pad : Math.max(0, span[0] - pad);
  return { rows: out, points, domain: [lo, span[1] + pad] };
}

/** The strip's coverage in words, for the sets where fewer targets have a
 *  flux than are listed ("Plotted: 8 of 445 in NEXUS × JWST; the other
 *  targets have no flux yet"); null when every listed target has a dot. */
export function stripCoverage(rows: readonly StripRow[]): string | null {
  const partial = rows.filter((r) => r.total > 0 && r.n < r.total);
  if (!partial.length) return null;
  const parts = partial.map((r) => `${formatCount(r.n)} of ${formatCount(r.total)} in ${r.label}`);
  return `Plotted: ${parts.join(", ")}; the other targets have no flux yet (no scored SR).`;
}

/* ── Run production on stale ───────────────────────────────────────────── */

export type ProductionPlan = {
  /** Every stale target the plan covers. */
  stale: number;
  /** NEXUS field inference over the stale NEXUS tiles (the per-tile SR the NEXUS viewer and plates read). */
  nexus: { field: string; tiles: string[] } | null;
  /** Other stale real tiles: one production experiment. */
  tiles: string[];
  /** Stale lens candidates / Q1 galaxies: the grouped analysis re-makes them
   *  (`n` per lens grade, 3n galaxies; current objects are reused). */
  catalogue: { stale: number; n: number } | null;
};

/** What "Run production on stale" starts for the given rows. `gradeCounts`
 *  is the whole evaluation store's objects per grade (A, B, C, gal), so the
 *  grouped run's `n` reaches every object. */
export function productionPlan(rows: readonly TargetRow[], gradeCounts: Readonly<Record<string, number>>): ProductionPlan {
  const stale = rows.filter((r) => r.state === "stale");
  const nexusRows = stale.filter((r) => r.set === "nexus" && r.nexusField);
  const field = nexusRows[0]?.nexusField ?? null;
  const onField = field ? nexusRows.filter((r) => r.nexusField === field) : [];
  const onFieldKeys = new Set(onField.map((r) => r.key));
  const tiles = stale
    .filter((r) => r.set !== "lenses" && r.set !== "galaxies" && !onFieldKeys.has(r.key) && r.ref)
    .map((r) => r.ref as string);
  const cat = stale.filter((r) => r.set === "lenses" || r.set === "galaxies");
  const n = groupedN(gradeCounts);
  return {
    stale: onField.length + tiles.length + cat.length,
    nexus: field && onField.length ? { field, tiles: onField.map((r) => splitRef(r.key)[1]) } : null,
    tiles,
    catalogue: cat.length ? { stale: cat.length, n } : null,
  };
}

/** The grouped run's `n` (objects per lens grade; it draws 3n galaxies)
 *  that reaches every object of the store. */
export function groupedN(gradeCounts: Readonly<Record<string, number>>): number {
  const lensMax = Math.max(0, ...["A", "B", "C"].map((g) => gradeCounts[g] ?? 0));
  return Math.max(1, lensMax, Math.ceil((gradeCounts.gal ?? 0) / 3));
}

/** Objects per grade of the evaluation store (the grouped run's size). */
export function gradeCounts(rows: readonly Pick<EvalRow, "grade">[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const r of rows) {
    const g = String(r.grade ?? "").trim();
    if (g) out[g] = (out[g] ?? 0) + 1;
  }
  return out;
}
