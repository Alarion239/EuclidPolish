/* Pure logic of the Models workspace (unit-tested in model.test.ts and
   loop.test.ts): model names, knee descriptions, the shared number formats,
   the one-experiment real-data benchmark (Sky › Compare), which training
   jobs trained universal models, the restore wording, Δm vs LR, and the
   Train tab's helpers (the running batch, the submit label, the forward
   model and LR schedule in words). No React, no DOM. */
import type { SlurmJob } from "../../api/jobs";
import { formatCount } from "../../format";
import type { ExperimentSummary, KneeInfo, Tombstone, TrainingJob } from "./api";

/* ── member names ──────────────────────────────────────────────────────── */

const MEMBER_RE = /^(?:member_)?(\d{1,6})(?:·psnr)?$/;

/** "196·psnr" / "member_196" / "196" → "196" (zero-padded to 2 digits like
 *  the member directories), or null. */
export function memberNumber(raw: string | null | undefined): string | null {
  const m = MEMBER_RE.exec(String(raw ?? "").trim());
  if (!m) return null;
  return String(Number(m[1])).padStart(2, "0");
}

export const memberName = (raw: string): string | null => {
  const n = memberNumber(raw);
  return n == null ? null : `member_${n}`;
};

/* ── knees ─────────────────────────────────────────────────────────────── */

export const DEFAULT_KNEE_E = 100;

const knum = (v: number) => (Math.abs(v) >= 1000 ? `${+(v / 1000).toPrecision(3)}k` : `${+v.toPrecision(3)}`);

export type KneeKind = "multi" | "single" | "default";

/** How a member was trained with respect to the asinh knee:
 *  multi-knee members read "multi ×6 → 10" (one output image scored at every
 *  knee, stretched at 10 e⁻), "multi ×6 → learned" (one image whose knee is
 *  learned per band) or "multi ×6 heads" (one image per knee) — never
 *  the "100e" the old table showed; single-knee members their knee; members
 *  without a knee the per-band default. */
export function kneeText(m: KneeInfo): { text: string; kind: KneeKind; title: string; sort: number } {
  const knees = Array.isArray(m.asinh_knees) ? m.asinh_knees.filter((v) => Number.isFinite(v)) : [];
  if (knees.length > 1) {
    const span = `${knum(Math.min(...knees))}–${knum(Math.max(...knees))} e⁻`;
    const loss = m.knee_loss && m.knee_loss !== "plain" ? `, ${m.knee_loss} channels` : "";
    if (m.output_knee != null) {
      return { text: `multi ×${knees.length} → ${knum(m.output_knee)}`, kind: "multi", sort: 1e6 + m.output_knee,
        title: `Trained at ${knees.length} knees (${span}) at once${loss}; one output image stretched at ${knum(m.output_knee)} e⁻` };
    }
    if (m.learned_output_knee) {
      return { text: `multi ×${knees.length} → learned`, kind: "multi", sort: 1.5e6,
        title: `Trained at ${knees.length} knees (${span}) at once${loss}; one output image, its knee learned per band` };
    }
    return { text: `multi ×${knees.length} heads`, kind: "multi", sort: 2e6,
      title: `Trained at ${knees.length} knees (${span})${loss}; one output image per knee` };
  }
  if (m.asinh_knee != null && Number.isFinite(m.asinh_knee)) {
    return { text: `${knum(m.asinh_knee)} e⁻`, kind: "single", sort: m.asinh_knee, title: `Asinh knee ${m.asinh_knee} e⁻` };
  }
  return { text: `${DEFAULT_KNEE_E} e⁻`, kind: "default", sort: DEFAULT_KNEE_E, title: "The per-band default knee (100 e⁻)" };
}

/** A universal model: trained at several knees at once with ONE output image
 *  (a fixed `output_knee` or one learned per band) — the rule of
 *  `ensemble_registry.is_universal_origin`. Multi-head members (one image per
 *  knee) are not. */
export function isUniversal(m: KneeInfo): boolean {
  const knees = Array.isArray(m.asinh_knees) ? m.asinh_knees.filter((v) => Number.isFinite(v)) : [];
  return knees.length > 0 && (m.output_knee != null || !!m.learned_output_knee);
}

/** A knee in e⁻ as the axes read it: 0.1, 100, 1k, 10k. */
export const kneeNum = knum;

/* ── formatting ────────────────────────────────────────────────────────── */

export function db(v: number | null | undefined, digits = 2): string {
  return v == null || !Number.isFinite(v) ? "—" : v.toFixed(digits);
}

export function dbDelta(v: number | null | undefined, digits = 2): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const s = v.toFixed(digits);
  return v > 0 ? `+${s}` : s.replace("-", "−");
}

/** "52k / 70k" style step progress. */
export function stepsText(step: number | null | undefined, target: number | null | undefined): string {
  const k = (v: number) => (v >= 1000 ? `${+(v / 1000).toFixed(1)}k` : String(v));
  if (step == null) return "—";
  return target ? `${k(step)} / ${k(target)}` : k(step);
}

/** Series `[[step, v]…]` → Plot-ready x/y arrays. */
export function xy(series: readonly (readonly [number, number])[] | null | undefined): { x: number[]; y: number[] } {
  const x: number[] = [];
  const y: number[] = [];
  for (const [a, b] of series ?? []) { x.push(a); y.push(b); }
  return { x, y };
}

/* ── real-data benchmark (Sky › Compare) ────────────────────────────────── */

/** The Euclid bands in canonical order and their short names (as the chips
 *  read). Kept here so this module stays free of the hooks in ./api. */
export const GATE_BANDS: readonly { band: string; short: string }[] = [
  { band: "VIS", short: "VIS" }, { band: "Y_E", short: "Y" }, { band: "J_E", short: "J" }, { band: "H_E", short: "H" },
];

/** A gate share as a percentage, ONE rule everywhere: two significant
 *  figures ("48%", "4.6%", "0.48%"), "<0.1%" below a tenth of a percent. */
export const share = (v: number | null | undefined) => {
  if (v == null || !Number.isFinite(v)) return "—";
  const p = 100 * v;
  if (p === 0) return "0%";
  if (Math.abs(p) < 0.1) return "<0.1%";
  return `${Number(p.toPrecision(2))}%`;
};

/** How many members production SR runs: "Runs 20 of 30 members: those with
 *  ≥ 0.5% of the gate's weight somewhere" for a pruned gate (the rule when
 *  the payload records its share threshold), else "Runs all 30 members". */
export function productionRunsText(reads: number, total: number, threshold?: number | null): string {
  const noun = (n: number) => `member${n === 1 ? "" : "s"}`;
  if (reads >= total) return total === 1 ? "Runs its 1 member" : `Runs all ${total} ${noun(total)}`;
  const why = threshold != null && Number.isFinite(threshold)
    ? `those with ≥ ${Number((100 * threshold).toPrecision(3))}% of the gate's weight somewhere`
    : "the ones the gate reads";
  return `Runs ${reads} of ${total} ${noun(total)}: ${why}`;
}

export function shareThreshold(rec: Record<string, unknown> | null | undefined): number | null {
  const t = rec?.prune_threshold ?? rec?.used_threshold;
  return typeof t === "number" && Number.isFinite(t) && t > 0 && t < 1 ? t : null;
}

/** One band's real-data hole % (the share of bright LR pixels the SR blanks). */
export type BandHole = { band: string; short: string; pct: number | null };
export type Bench = {
  holeMean: number | null; holeMax: number | null; medianR: number | null; rLt08: number | null; nTiles: number | null;
  /** Per band (VIS Y J H) when the experiment recorded `per_band`, else empty. */
  bands: BandHole[];
  /** The band with the most holes (null without per-band values). */
  worst: BandHole | null;
};
export type Benchmark = {
  expId: string; label?: string; created?: string; nTiles: number | null; tiles: string[];
  /** What the tiles are, for headers: the experiment label, else "9 real tiles". */
  tileSet: string;
  bySpec: Map<string, Bench>;
};

const byNewest = (exps: readonly ExperimentSummary[] | null | undefined) =>
  [...(exps ?? [])].sort((a, b) => String(b.created ?? "").localeCompare(String(a.created ?? "")));
const hasSummary = (e: ExperimentSummary) => !!e.summary && Object.keys(e.summary).length > 0;
const tilesText = (n: number | null) => (n == null ? "real tiles" : `${n} real tile${n === 1 ? "" : "s"}`);
const expTiles = (e: ExperimentSummary): number | null => {
  let n: number | null = null;
  for (const agg of Object.values(e.summary ?? {})) {
    const t = (agg as { n_tiles?: number }).n_tiles;
    if (typeof t === "number") n = Math.max(n ?? 0, t);
  }
  return n ?? (e.tiles?.length || null);
};

/** The real-data benchmark of ONE experiment, so every model is scored on
 *  the same tiles: the one the user picked (`chosen`), else the newest
 *  experiment that ran production (else the newest one). Models it did not
 *  run stay blank. */
export function benchmarkExperiment(exps: readonly ExperimentSummary[] | null | undefined, chosen?: string | null): Benchmark | null {
  const sorted = byNewest(exps);
  const e = (chosen ? sorted.find((x) => x.id === chosen && hasSummary(x)) : undefined)
    ?? sorted.find((x) => x.summary && "production" in x.summary) ?? sorted[0];
  if (!e) return null;
  const bySpec = new Map<string, Bench>();
  for (const [spec, agg] of Object.entries(e.summary ?? {})) {
    const a = agg as { n_tiles?: number; summary?: Record<string, number | null>; per_band?: Record<string, { hole_pct?: number | null }> };
    const bands: BandHole[] = a.per_band
      ? GATE_BANDS.filter(({ band }) => a.per_band?.[band]).map(({ band, short }) => {
        const v = a.per_band?.[band]?.hole_pct;
        return { band, short, pct: v != null && Number.isFinite(v) ? v : null };
      })
      : [];
    const worst = bands.reduce<BandHole | null>((w, b) => (b.pct != null && (w?.pct == null || b.pct > w.pct) ? b : w), null);
    bySpec.set(spec, { holeMean: a.summary?.hole_pct_mean ?? null, holeMax: a.summary?.hole_pct_max ?? null,
      medianR: a.summary?.median_R ?? null, rLt08: a.summary?.pct_R_lt_0p8 ?? null, nTiles: a.n_tiles ?? null, bands, worst });
  }
  const nTiles = expTiles(e);
  return { expId: e.id, label: e.label, created: e.created, nTiles, tiles: e.tiles ?? [], tileSet: e.label || tilesText(nTiles), bySpec };
}

/** A Sky › Compare run's page. */
export const compareRunPath = (expId: string) => `/sky/compare?${new URLSearchParams({ exp: expId }).toString()}`;

/** The newest Sky › Compare run that scored production, or null. */
export function productionRun(exps: readonly ExperimentSummary[] | null | undefined): ExperimentSummary | null {
  return byNewest(exps).find((e) => hasSummary(e) && "production" in (e.summary ?? {})) ?? null;
}

const TRUTHY = ["1", "true", "yes", "on"];

const truthy = (v: unknown) => (typeof v === "boolean" ? v : typeof v === "number" ? v !== 0
  : typeof v === "string" ? TRUTHY.includes(v.trim().toLowerCase()) : !!v && typeof v === "object");
const hasKnees = (v: unknown): boolean => {
  if (Array.isArray(v)) return v.some((k) => Number.isFinite(Number(k)) && String(k).trim() !== "");
  return typeof v === "string" && v.split(",").some((k) => k.trim() !== "" && Number.isFinite(Number(k)));
};
const hasValue = (v: unknown) => v != null && String(v).trim() !== "" && Number.isFinite(Number(v));

/** The models a fork job forked (`params.fork_from`, any spelling → member_NNN). */
export function forkSource(job: Pick<TrainingJob, "params"> & { mode?: string | null }): string | null {
  const p = job.params ?? {};
  if (job.mode !== "fork" && p.mode !== "fork") return null;
  return memberName(String(p.fork_from ?? ""));
}

/** The members a training job trained as universal models (isUniversal on
 *  the job's recipe: the run-wide params, overridden per member by
 *  `member_spec[i]`; the ledger spells the learned knee `learn_output_knee`).
 *  A fork inherits its source's knees, so a fork of a name in
 *  `knownUniversal` is universal whatever its own recipe says. */
export function universalJobMembers(job: Pick<TrainingJob, "member_names" | "params"> & { mode?: string | null },
  knownUniversal: ReadonlySet<string> = new Set()): Set<string> {
  const p = job.params ?? {};
  const source = forkSource(job);
  if (source && knownUniversal.has(source)) return new Set(job.member_names);
  let spec: unknown;
  try { spec = typeof p.member_spec === "string" ? JSON.parse(p.member_spec) : p.member_spec; } catch { spec = null; }
  const specs = Array.isArray(spec) ? spec : [];
  const out = new Set<string>();
  job.member_names.forEach((name, i) => {
    const own = specs[i] && typeof specs[i] === "object" ? specs[i] as Record<string, unknown> : {};
    const o = { ...p, ...own };
    const learned = truthy(o.learn_output_knee) || truthy(o.learned_output_knee);
    if (hasKnees(o.asinh_knees) && (learned || hasValue(o.output_knee))) out.add(name);
  });
  return out;
}

/** Every name known to be a universal model: the registered ones
 *  (members.json `universal`), the active members that are (isUniversal on
 *  their row), and every job's universal members (universalJobMembers),
 *  repeated until no fork adds one, so a fork of a fork of 211 counts. */
export function knownUniversal(jobs: readonly (Pick<TrainingJob, "member_names" | "params"> & { mode?: string | null })[],
  members: readonly (KneeInfo & { name: string })[] = [], registered: readonly string[] | null = []): Set<string> {
  const known = new Set<string>(registered ?? []);
  for (const m of members) if (isUniversal(m)) known.add(m.name);
  for (let grew = true; grew;) {
    grew = false;
    for (const j of jobs) {
      if (j.mode === "continue") continue;
      for (const n of universalJobMembers(j, known)) if (!known.has(n)) { known.add(n); grew = true; }
    }
  }
  return known;
}

/** The Restore confirm of an archived name, by its tombstone's class: an
 *  ensemble member comes back active (evaluations read stale); a universal
 *  model comes back standalone, and nothing goes stale. */
export function restoreText(t: Pick<Tombstone, "class"> | null | undefined): string {
  return t?.class === "universal"
    ? "Unzips the archive and registers it as a universal model again."
    : "Unzips the archive back into the ensemble and makes the member active again.";
}

/** SR's total-flux change against LR as a magnitude: "Δm +0.11 (flux ×0.90)",
 *  `warn` beyond 0.1 mag. Null without both fluxes. */
export function deltaMagText(lrE: number | null | undefined, srE: number | null | undefined): { text: string; warn: boolean } | null {
  if (lrE == null || srE == null || !(lrE > 0) || !(srE > 0)) return null;
  const ratio = srE / lrE;
  const dm = -2.5 * Math.log10(ratio);
  const s = dm.toFixed(2);
  const signed = dm > 0 ? `+${s}` : dm < 0 ? s.replace("-", "−") : s;
  return { text: `Δm ${signed} (flux ×${ratio.toFixed(2)})`, warn: Math.abs(dm) > 0.1 };
}

/** The band a cube plane is summed in: the viewer's band, or VIS (plane 0)
 *  for a colour composite. */
export function bandIndex(bands: readonly string[] | null | undefined, color: string): number {
  const i = (bands ?? []).indexOf(color);
  return i >= 0 ? i : 0;
}

/** A band name as the page shows it (Y_E → Y). */
export const bandLabel = (band: string) => band.replace(/_E$/, "");

/* ── train ─────────────────────────────────────────────────────────────── */

/** "Submit 4 members to SLURM" — the count repeated on the button, so a
 *  wrong batch size is caught before the confirm. */
export function submitLabel(mode: "add" | "continue" | "fork", count: number): string {
  if (mode === "continue") return count > 0 ? `Continue ${count} member${count === 1 ? "" : "s"} on SLURM` : "Continue members on SLURM";
  const noun = mode === "fork" ? "fork" : "member";
  return `Submit ${count} ${noun}${count === 1 ? "" : "s"} to SLURM`;
}

/** "members 199–202" for a contiguous run, else the numbers. */
function membersText(names: readonly string[]): string {
  const nums = names.map((n) => Number(memberNumber(n))).filter(Number.isFinite).sort((a, b) => a - b);
  if (!nums.length) return "";
  const contiguous = nums.every((v, i) => i === 0 || v === nums[i - 1] + 1);
  const pad = (v: number) => String(v).padStart(2, "0");
  if (nums.length > 2 && contiguous) return `members ${pad(nums[0])}–${pad(nums[nums.length - 1])}`;
  return `member${nums.length === 1 ? "" : "s"} ${nums.map(pad).join(", ")}`;
}

/** The Train tab's "Running batch" strip: the live SLURM ensemble_train jobs
 *  (the feed Runs › Live reads), with their members from the job log. A
 *  batch's step is its members' share of their steps done, in one member's
 *  steps (the server folds it from the tasks' event streams). */
export function runningBatches(slurm: readonly SlurmJob[], jobs: readonly Pick<TrainingJob, "jobid" | "member_names" | "mode">[]):
  { jobid: string; state: string; members: string[]; text: string }[] {
  const byId = new Map(jobs.map((j) => [String(j.jobid), j]));
  return slurm.filter((j) => j.step_id === "ensemble_train").map((j) => {
    const state = String(j.state ?? "").toUpperCase();
    const members = byId.get(String(j.jobid))?.member_names ?? [];
    const parts = [`Job ${j.jobid}`, membersText(members)];
    if (state === "PENDING") parts.push(`pending${j.reason ? ` (${j.reason})` : ""}`);
    else {
      parts.push(state.toLowerCase() || "running");
      const step = j.progress_step, total = j.progress_total;
      if (step != null && total) parts.push(`step ${formatCount(step)} / ${formatCount(total)} (${Math.round((100 * step) / total)}%)`);
      if (j.time) parts.push(j.time_limit ? `${j.time} of ${j.time_limit}` : String(j.time));
    }
    return { jobid: String(j.jobid), state, members: [...members], text: parts.filter(Boolean).join(" · ") };
  });
}

/** The forward-model values a training batch takes from System › Config
 *  (job_config FASRC_STEP_PARAMS for ensemble_train; the Train form never
 *  sends them, so the preview and the submit use Config's). */
export const CONFIG_FORWARD_KEYS: readonly { key: string; label: string; unit?: string }[] = [
  { key: "psf_warp_prob", label: "PSF warp probability" },
  { key: "psf_warp_alpha_max", label: "PSF warp α max" },
  { key: "psf_warp_sigma", label: "PSF warp σ", unit: "HR px" },
  { key: "saturation_mask_prob", label: "Saturation mask probability" },
];

export function forwardModelFacts(config: Record<string, unknown> | null | undefined): { label: string; value: string; unit?: string }[] {
  return CONFIG_FORWARD_KEYS.filter(({ key }) => config?.[key] != null && config[key] !== "")
    .map(({ key, label, unit }) => ({ label, value: String(Number(Number(config![key]).toPrecision(4))), ...(unit ? { unit } : {}) }));
}

/** The System › Config knobs a FASRC step reads (`used_by`) whose value
 *  differs from its default — the "N knobs changed · Edit" back-link of a
 *  tab judged by a Config group. `only` narrows to a group's keys; `except`
 *  leaves a group's keys out. */
export function changedConfigKnobs(
  payload: { config?: Record<string, unknown>; defaults?: Record<string, unknown>; used_by?: Record<string, string[]> } | null | undefined,
  step: string, group: { only?: readonly string[]; except?: readonly string[] } = {},
): string[] {
  const cfg = payload?.config, defs = payload?.defaults, used = payload?.used_by;
  if (!cfg || !defs || !used) return [];
  const same = (a: unknown, b: unknown) => {
    const x = Number(a), y = Number(b);
    if (a !== "" && b !== "" && a != null && b != null && Number.isFinite(x) && Number.isFinite(y)) return Math.abs(x - y) <= 1e-12 * Math.max(1, Math.abs(y));
    return String(a ?? "") === String(b ?? "");
  };
  return Object.keys(used).filter((k) => used[k]?.includes(step)
    && (!group.only || group.only.includes(k)) && !group.except?.includes(k)
    && k in defs && !same(cfg[k], defs[k])).sort();
}

/** "2 knobs changed" (or null when every knob is at its default). */
export function knobsChangedText(keys: readonly string[]): string | null {
  return keys.length ? `${keys.length} knob${keys.length === 1 ? "" : "s"} changed` : null;
}

const fnum = (v: unknown): number | null => {
  const n = typeof v === "number" ? v : typeof v === "string" && v.trim() !== "" ? Number(v) : NaN;
  return Number.isFinite(n) ? n : null;
};
const sci = (v: number) => (Math.abs(v) < 0.01 && v !== 0 ? v.toExponential().replace(/\.?0+e/, "e") : String(Number(v.toPrecision(3))));

/** The LR schedule the batch takes from System › Config, in one line. */
export function lrScheduleText(config: Record<string, unknown> | null | undefined): string | null {
  const peak = fnum(config?.lr_peak), final = fnum(config?.lr_final), warm = fnum(config?.lr_warmup_steps);
  if (peak == null) return null;
  const guard = config?.plateau_lr_enabled;
  const on = guard === true || TRUTHY.includes(String(guard).toLowerCase());
  return [`LR ${warm ? `warmup ${formatCount(warm)} steps to ` : ""}${sci(peak)}${final != null ? `, then cosine to ${sci(final)}` : ""}`,
    `plateau guard ${on ? "on" : "off"}`].join("; ");
}
