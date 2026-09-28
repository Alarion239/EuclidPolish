/* The Train tab's form model (pure; tested in model.test.ts): per-member rows →
   the positional `member_spec` JSON the ensemble_train step consumes, the
   run's scheduling and trainer forward-model knobs, the regime, the FASRC
   form body, presets and "clone a past job". The forward-model values System
   › Config owns (PSF warp, saturation mask; job_config FASRC_STEP_PARAMS) are
   never sent: the step fills them from Config, so Config is their one
   source. */
import type { Mode, TrainingJob } from "./api";

export type TrainMode = "add" | "continue" | "fork";
export const LOSSES = ["l1", "l2", "l3", "mse"] as const;
export const KNEE_LOSSES = ["plain", "balanced"] as const;
export const DEFAULT_MULTI_KNEES = "0.1,1,10,100,1000,10000";

/** One NEW member's knobs. Blank fields fall back to the run-wide defaults.
 *  The star regime is NOT a row knob: it is the workspace's (buildParams). */
export type SpecRow = {
  loss: string;
  blocks: string;
  kneeMode: "single" | "multi";
  knee: string;          // single knee (e⁻); "" or 100 = the per-band default
  knees: string;         // multi: comma list (e⁻)
  outputKnee: string;    // multi: "" = one image per knee (option 1)
  kneeLoss: "plain" | "balanced";
  noise: string;
  boot: string;
  icnr: boolean;
  seed: string;
};

/** The per-batch run knobs: `batch_size` (scheduling) and the trainer's own
 *  forward-model knobs (the ones Config does not own). */
export type Geometry = {
  batch_size: string; hr_crop_size: string; crops_per_field: string; forward_onthefly: boolean;
  psf_subset: string; target_psf_fwhm_arcsec: string;
};

export type TrainForm = {
  mode: TrainMode;
  /** The regime new members (add, fork) train in; continue keeps each member's. */
  regime: Mode;
  rows: SpecRow[];
  steps: string;
  forkFrom: string;
  forkTrack: "psnr" | "loss";
  members: string[];
  continueBasis: "extra" | "target";
  extraSteps: string;
  targetSteps: string;
  baseSeed: string;
  evaluateEvery: string;
  arrayMaxParallel: string;
  geometry: Geometry;
};

export const newRow = (from?: Partial<SpecRow>): SpecRow => ({
  loss: "l2", blocks: "32", kneeMode: "single", knee: "100", knees: DEFAULT_MULTI_KNEES,
  outputKnee: "10", kneeLoss: "balanced", noise: "0", boot: "0.7", icnr: true,
  ...from,
  seed: "",               // never cloned — two members with one seed are one model
});

/** The recipe of every batch since 2026-09-20 (memory: L2, bootstrap 0.7,
 *  32 blocks, ICNR, 70k steps, live forward; saturation 0.5 and PSF warp α 5
 *  are System › Config's). */
export const RECIPE_GEOMETRY: Geometry = {
  batch_size: "4", hr_crop_size: "256", crops_per_field: "8", forward_onthefly: true,
  psf_subset: "64", target_psf_fwhm_arcsec: "0.066",
};

export const defaultForm = (regime: Mode = "starfull"): TrainForm => ({
  mode: "add", regime, rows: [newRow()], steps: "70000", forkFrom: "", forkTrack: "psnr",
  members: [], continueBasis: "extra", extraSteps: "20000", targetSteps: "70000",
  baseSeed: "", evaluateEvery: "", arrayMaxParallel: "4",
  geometry: { ...RECIPE_GEOMETRY },
});

const num = (s: string) => (s.trim() === "" ? NaN : Number(s));

/** SLURM resources per model of the recipe (the 2026-09-24 batch 48107719:
 *  a 70k-step member takes 2.5–3 h; 16 CPUs keep the GPU fed — the step's
 *  own default of 4 CPUs / 48 h starves it and books far too long). */
export type Resources = { n_cpus: string; memory: string; time_limit: string };
export const RECIPE_RESOURCES: Resources = { n_cpus: "16", memory: "32G", time_limit: "3:00:00" };

/** Jobs of the same kind as a form mode: a continue job runs members a few
 *  thousand steps further; add and fork train whole new members. */
const sameKind = (jobMode: string | undefined, mode: TrainMode) =>
  mode === "continue" ? jobMode === "continue" : jobMode !== "continue";

/** The resources a fresh Train form starts from: the newest COMPLETED
 *  training job OF THE SAME KIND as the form's mode (what worked last time;
 *  `from` names it), else the recipe. A short continue job's hour never
 *  becomes the time limit of the next new 70k-step batch. */
export function defaultResources(jobs: readonly TrainingJob[], mode: TrainMode = "add"): Resources & { from: string | null } {
  const done = jobs.find((j) => j.state === "COMPLETED" && sameKind(j.mode, mode) && (j.req_cpus || j.req_time_limit));
  if (!done) return { ...RECIPE_RESOURCES, from: null };
  return {
    n_cpus: done.req_cpus ? String(done.req_cpus) : RECIPE_RESOURCES.n_cpus,
    memory: done.req_memory || RECIPE_RESOURCES.memory,
    time_limit: done.req_time_limit || RECIPE_RESOURCES.time_limit,
    from: done.jobid,
  };
}

/** "Continue them…" (TIMEOUT members): run the picked members up to their
 *  recorded target (the largest among them), not a fixed +N steps that
 *  leaves one short and runs another past it. Null when none has a target. */
export function continueTarget(rows: readonly { name: string; target_steps?: number | null }[], picked: readonly string[]): number | null {
  const set = new Set(picked);
  const ts = rows.filter((r) => set.has(r.name) && r.target_steps != null && r.target_steps > 0).map((r) => r.target_steps as number);
  return ts.length ? Math.max(...ts) : null;
}

export function parseKnees(text: string): number[] | null {
  const vs = text.split(/[\s,;]+/).filter(Boolean).map(Number);
  return vs.length && vs.every((v) => Number.isFinite(v) && v > 0) ? vs : null;
}

/** Rows → the positional per-member override list (only the keys that
 *  differ from the step defaults; `add` also sets depth + ICNR, fork inherits
 *  both from its source). */
export function buildSpec(rows: readonly SpecRow[], mode: TrainMode): Record<string, unknown>[] {
  return rows.map((r) => {
    const o: Record<string, unknown> = { loss: r.loss };
    const noise = num(r.noise); if (noise > 0) o.noise_aug = noise;
    const boot = num(r.boot); if (boot > 0 && boot < 1) o.bootstrap = boot;
    if (r.kneeMode === "multi") {
      const knees = parseKnees(r.knees);
      if (knees) o.asinh_knees = knees;
      const out = num(r.outputKnee); if (out > 0) o.output_knee = out;
      if (r.kneeLoss !== "plain") o.knee_loss = r.kneeLoss;
    } else {
      const knee = num(r.knee); if (knee > 0 && knee !== 100) o.asinh_knee = knee;
    }
    if (r.seed.trim() !== "" && Number.isInteger(num(r.seed))) o.seed = Math.trunc(num(r.seed));
    if (mode === "add") {
      const b = Math.trunc(num(r.blocks)); if (b > 0) o.num_res_blocks = b;
      if (r.icnr) o.icnr = true;
    }
    return o;
  });
}

/** member_spec objects (from a past job) → rows. */
export function rowsFromSpec(spec: unknown): SpecRow[] {
  if (!Array.isArray(spec)) return [];
  return spec.filter((o): o is Record<string, unknown> => !!o && typeof o === "object").map((o) => {
    const knees = Array.isArray(o.asinh_knees) ? (o.asinh_knees as unknown[]).map(Number).filter(Number.isFinite) : [];
    return newRow({
      loss: typeof o.loss === "string" ? o.loss : "l1",
      blocks: o.num_res_blocks != null ? String(o.num_res_blocks) : "32",
      kneeMode: knees.length > 1 ? "multi" : "single",
      knee: o.asinh_knee != null ? String(o.asinh_knee) : "100",
      knees: knees.length ? knees.join(",") : DEFAULT_MULTI_KNEES,
      outputKnee: o.output_knee != null ? String(o.output_knee) : "",
      kneeLoss: o.knee_loss === "balanced" ? "balanced" : "plain",
      noise: o.noise_aug != null ? String(o.noise_aug) : "0",
      boot: o.bootstrap != null ? String(o.bootstrap) : "",
      icnr: o.icnr === true,
    });
  });
}

type StringGeometryKey = Exclude<keyof Geometry, "forward_onthefly">;
const GEOMETRY_KEYS: StringGeometryKey[] = [
  "batch_size", "hr_crop_size", "crops_per_field", "psf_subset", "target_psf_fwhm_arcsec",
];

const TRUTHY = ["1", "true", "yes", "on"];

/** The form → the POST body of `/ensemble/train/preview` and
 *  `/api/fasrc/steps/ensemble_train/submit` (task params only). `regime` is
 *  the form's own field (default: the workspace's): add/fork send the
 *  run-wide `starless` flag for a starless batch (a fork's source regime still
 *  wins in the trainer); continue sends none, each member keeps its recorded
 *  regime. */
export function buildParams(f: TrainForm, regime: Mode = f.regime ?? "starfull"): Record<string, string> {
  const g: Record<string, string> = { forward_onthefly: f.geometry.forward_onthefly ? "1" : "0" };
  for (const k of GEOMETRY_KEYS) {
    const v = String(f.geometry[k] ?? "").trim();
    if (v !== "") g[k] = v;
  }
  const common: Record<string, string> = { ...g, array_max_parallel: f.arrayMaxParallel };
  if (f.baseSeed.trim()) common.base_seed = f.baseSeed.trim();
  if (f.evaluateEvery.trim()) common.evaluate_every = f.evaluateEvery.trim();
  if (f.mode === "continue") {
    return {
      mode: "continue", members: f.members.join(","), continue_basis: f.continueBasis,
      ...(f.continueBasis === "target" ? { target_steps: f.targetSteps } : { extra_steps: f.extraSteps }),
      ...common,
    };
  }
  const out: Record<string, string> = {
    mode: f.mode, count: String(f.rows.length), steps: f.steps,
    member_spec: JSON.stringify(buildSpec(f.rows, f.mode)), ...common,
  };
  if (f.mode === "fork") { out.fork_from = f.forkFrom.trim(); out.fork_track = f.forkTrack; }
  if (regime === "starless") out.starless = "1";
  return out;
}

/** The regime a past job trained in (run-wide flag or any per-member one). */
export function jobRegime(job: TrainingJob): Mode {
  const p = job.params ?? {};
  if (TRUTHY.includes(String(p.starless ?? "").toLowerCase())) return "starless";
  let spec: unknown;
  try { spec = typeof p.member_spec === "string" ? JSON.parse(p.member_spec) : p.member_spec; } catch { spec = null; }
  return Array.isArray(spec) && spec.some((o) => !!o && typeof o === "object" && (o as Record<string, unknown>).starless === true)
    ? "starless" : "starfull";
}

/** Everything the submit would refuse, as short messages (empty = valid). */
export function validate(f: TrainForm): string[] {
  const errs: string[] = [];
  const posInt = (s: string) => Number.isInteger(num(s)) && num(s) > 0;
  const g = f.geometry;
  if (!posInt(g.batch_size)) errs.push("batch size must be a positive whole number");
  if (!posInt(g.hr_crop_size) || num(g.hr_crop_size) % 2) errs.push("HR example side must be a positive even number");
  if (!posInt(g.crops_per_field)) errs.push("examples per field must be positive");
  if (!(num(g.target_psf_fwhm_arcsec) >= 0)) errs.push("target PSF FWHM must be ≥ 0");
  if (!posInt(f.arrayMaxParallel)) errs.push("models at once must be positive");
  if (f.baseSeed.trim() && !Number.isInteger(num(f.baseSeed))) errs.push("base seed must be a whole number");
  if (f.evaluateEvery.trim() && !posInt(f.evaluateEvery)) errs.push("evaluate every must be a positive whole number");
  if (f.mode === "continue") {
    if (!f.members.length) errs.push("pick at least one member to continue");
    if (!posInt(f.continueBasis === "target" ? f.targetSteps : f.extraSteps)) errs.push("steps must be a positive whole number");
    return errs;
  }
  if (!posInt(f.steps)) errs.push("steps must be a positive whole number");
  if (f.mode === "fork" && !f.forkFrom.trim()) errs.push("pick the member to fork from");
  if (!f.rows.length) errs.push("add at least one member");
  f.rows.forEach((r, i) => {
    if (r.kneeMode === "multi") {
      if (!parseKnees(r.knees)) errs.push(`member ${i + 1}: knees must be positive e⁻, e.g. ${DEFAULT_MULTI_KNEES}`);
      if (r.outputKnee.trim() && !(num(r.outputKnee) > 0)) errs.push(`member ${i + 1}: output knee must be positive`);
    } else if (r.knee.trim() && !(num(r.knee) > 0)) errs.push(`member ${i + 1}: knee must be positive`);
    const b = num(r.boot); if (r.boot.trim() && !(b >= 0 && b < 1)) errs.push(`member ${i + 1}: bootstrap must be in [0, 1)`);
    if (f.mode === "add" && !posInt(r.blocks)) errs.push(`member ${i + 1}: depth must be a positive whole number`);
  });
  return errs;
}

const str = (v: unknown, fallback = "") => (v == null || v === "" ? fallback : String(v));

/** A past ensemble_train job → a form (clone / "repeat last batch"). Seeds
 *  and member names are never cloned (they are allocated at submit). */
export function formFromJob(job: TrainingJob): TrainForm {
  const p = job.params ?? {};
  const base = defaultForm();
  const mode = (["add", "continue", "fork"].includes(job.mode) ? job.mode : "add") as TrainMode;
  let spec: unknown;
  try { spec = typeof p.member_spec === "string" ? JSON.parse(p.member_spec) : p.member_spec ?? []; } catch { spec = []; }
  const rows = rowsFromSpec(spec);
  const count = Number(p.count ?? rows.length) || rows.length || 1;
  const geometry = { ...base.geometry };
  for (const k of GEOMETRY_KEYS) if (p[k] != null && p[k] !== "") geometry[k] = String(p[k]);
  if (p.forward_onthefly != null) geometry.forward_onthefly = TRUTHY.includes(String(p.forward_onthefly).toLowerCase());
  return {
    ...base,
    mode,
    regime: jobRegime(job),
    rows: mode === "continue" ? base.rows : (rows.length ? rows : Array.from({ length: count }, () => newRow())),
    steps: str(p.steps, base.steps),
    forkFrom: str(p.fork_from),
    forkTrack: p.fork_track === "loss" ? "loss" : "psnr",
    members: mode === "continue" ? job.member_names : [],
    continueBasis: p.continue_basis === "target" ? "target" : "extra",
    extraSteps: str(p.extra_steps, base.extraSteps),
    targetSteps: str(p.target_steps, base.targetSteps),
    evaluateEvery: str(p.evaluate_every),
    arrayMaxParallel: str(p.array_max_parallel, base.arrayMaxParallel),
    geometry,
  };
}

/** The newest ensemble_train job that ADDED members (the "last batch"). */
export function lastBatch(jobs: readonly TrainingJob[]): TrainingJob | null {
  return jobs.find((j) => j.mode === "add" && j.member_names.length > 0) ?? null;
}

/** "4 × L2 · multi ×6 → 10 · 70k" — a one-line summary of a job's recipe. */
export function recipeSummary(job: TrainingJob): string {
  const f = formFromJob(job);
  if (f.mode === "continue") {
    return `continue ${job.member_names.length} · ${f.continueBasis === "target" ? `to ${f.targetSteps}` : `+${f.extraSteps}`}`;
  }
  const losses = [...new Set(f.rows.map((r) => r.loss.toUpperCase()))].join("/");
  const multi = f.rows.filter((r) => r.kneeMode === "multi").length;
  const knees = [...new Set(f.rows.filter((r) => r.kneeMode === "single").map((r) => r.knee))];
  const kneePart = [multi ? `${multi} multi-knee` : "", knees.length ? `knee ${knees.join("/")}` : ""].filter(Boolean).join(", ");
  const steps = Number(f.steps) >= 1000 ? `${Number(f.steps) / 1000}k` : f.steps;
  return `${f.mode} ${f.rows.length} × ${losses}${kneePart ? ` · ${kneePart}` : ""} · ${steps} steps`;
}
