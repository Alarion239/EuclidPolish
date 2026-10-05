/* Tracking-notebook notes of the Ensemble loop (pure; notes.test.ts): the
   markdown the "Log to notebook" buttons (workspaces/shared/LogToNotebook)
   open with — the Evaluate summary, the knee leaderboard over its
   integration range, a gate variant's fit, a compare report and a promote.
   Each states the facts on the page with their definitions; the user edits
   before appending. */
import { utcText } from "../shared/noteText";
import { BAND_SHORT, type CompareReport, type Overview, type Variant } from "./api";
import {
  db, dbDelta, kneeModelName, kneeNum, kneeText, memberNumber, readsText, variantLabel, type Bench, type Benchmark, type LeaderRow,
} from "./model";

export { utcText };

const range = (from: number, to: number) => `${kneeNum(from)}–${kneeNum(to)} e⁻`;
const shortBand = (b: string) => BAND_SHORT[b] ?? b;
const bestName = (label: string | null | undefined) => (label ? `#${memberNumber(label) ?? label}` : "the best member");
const mean = (vs: readonly (number | null | undefined)[] | null | undefined): number | null => {
  const f = (vs ?? []).filter((v): v is number => v != null && Number.isFinite(v));
  return f.length ? f.reduce((a, b) => a + b, 0) / f.length : null;
};
const row = (cells: readonly string[]) => `| ${cells.join(" | ")} |`;

/* ── Leaderboard: the Evaluate summary ─────────────────────────────────── */

export function evaluationNote(o: Overview): string {
  const h = o.headline;
  const k = h.knee;
  const head = o.evaluated_at
    ? `**Ensemble evaluation** — ${utcText(o.evaluated_at)} · ${o.n_members} members · ${h.n_scored ?? "?"} test fields`
    : `**Ensemble evaluation** — not evaluated yet · ${o.n_members} members`;
  const lines = [head, ""];
  lines.push(`- Metric: VIS asinh PSNR at knee ${h.knee_e ?? 100} e⁻ on the ${o.eval_subset ?? "test"} fields (eval_summary.json)`);
  const p = h.production;
  const vs = [p.vs_best_member_db != null ? `${dbDelta(p.vs_best_member_db)} dB vs best member` : null,
    p.vs_mean_db != null ? `${dbDelta(p.vs_mean_db)} dB vs plain mean` : null].filter(Boolean).join(", ");
  lines.push(p.psnr != null ? `- Production gate: ${db(p.psnr)} dB${vs ? ` (${vs})` : ""}` : "- Production gate: no score");
  if (h.mean.psnr != null) {
    lines.push(`- Plain mean: ${db(h.mean.psnr)} dB${h.mean.vs_mean_member_db != null ? ` (${dbDelta(h.mean.vs_mean_member_db)} dB vs mean member)` : ""}`);
  }
  if (h.best_member.psnr != null) lines.push(`- Best member: ${bestName(h.best_member.label)} ${db(h.best_member.psnr)} dB`);
  if (k.available && k.production != null) {
    const from = k.integration?.from_e ?? 0.1, to = k.integration?.to_e ?? 1e4;
    const deltas = [k.mean != null ? `${dbDelta(k.production - k.mean)} dB vs plain mean` : null,
      k.best_member != null ? `${dbDelta(k.production - k.best_member)} dB vs ${bestName(k.best_member_label)}` : null].filter(Boolean).join(", ");
    const bands = (k.production_bands ?? []).map((v, i) => `${["VIS", "Y", "J", "H"][i] ?? i} ${db(v)}`).join(" · ");
    lines.push(`- ∫PSNR over ${range(from, to)} (${k.n_fields ?? "?"} fields): production gate ${db(k.production)} dB${deltas ? ` (${deltas})` : ""}${bands ? ` · ${bands}` : ""}${k.stale ? " · stale" : ""}`);
  }
  const g = o.production_gate;
  if (g.available) {
    lines.push(`- Production gate fitted for ${g.n_members} members${g.mix_space ? `, ${g.mix_space} mix` : ""}${g.fitted_at ? `, ${utcText(g.fitted_at)}` : ""}${g.promoted_from ? ` (promoted from ${variantLabel(g.promoted_from)})` : ""}`);
  }
  const open = o.checks.filter((c) => !c.ok && c.tone !== "info");
  if (open.length) lines.push(`- Open: ${open.map((c) => (c.detail ? `${c.title} — ${c.detail}` : c.title)).join("; ")}`);
  return lines.join("\n");
}

/* ── Leaderboard: the knee ranking over its integration range ──────────── */

export function kneeNote(board: readonly LeaderRow[], opts: {
  range: [number, number]; full: [number, number]; band: string; bands: readonly string[];
  nFields?: number | null; top?: number;
}): string {
  const top = opts.top ?? 10;
  const isFull = opts.range[0] <= opts.full[0] && opts.range[1] >= opts.full[1];
  const where = isFull ? `${range(opts.full[0], opts.full[1])} (the full range, uniform in log knee)`
    : `${range(opts.range[0], opts.range[1])} (a sub-range of ${range(opts.full[0], opts.full[1])}, uniform in log knee)`;
  const bandText = opts.band === "all" ? "all bands" : shortBand(opts.band);
  const ranked = [...board].sort((a, b) => (a.rank ?? 1e9) - (b.rank ?? 1e9));
  const members = ranked.filter((r) => r.kind === "member");
  const keep = new Set(members.slice(0, top).map((r) => r.id));
  const rows = ranked.filter((r) => r.kind !== "member" || keep.has(r.id));
  const lines = [
    `**Knee-integrated PSNR** — ∫ over ${where} · ${bandText} · ${opts.nFields ?? "?"} test fields`, "",
    row(["#", "Model", "Trained", ...opts.bands.map((b) => `∫${shortBand(b)}`), opts.band === "all" ? "∫ mean" : `∫ ${shortBand(opts.band)}`, "vs mean"]),
    row(["---:", "---", "---", ...opts.bands.map(() => "---:"), "---:", "---:"]),
    ...rows.map((r) => row([
      r.rank == null ? "—" : String(r.rank), kneeModelName(r.model), r.kind === "member" ? kneeText(r.model).text : r.kind,
      ...opts.bands.map((_, i) => db(r.bands[i])), db(r.mean), dbDelta(r.vsMean),
    ])),
  ];
  if (members.length > top) lines.push("", `Top ${top} of ${members.length} members, plus the plain mean and the combiners.`);
  return lines.join("\n");
}

/* ── Combiner: a gate variant's fit, a compare report, a promote ───────── */

type Scores = { testVis: number | null; kneeMean: number | null };

const kneesText = (v: unknown): string | null => {
  if (Array.isArray(v)) return `${v.map((x) => kneeNum(Number(x))).join(", ")} e⁻`;
  return typeof v === "string" && v.trim() ? v.trim() : null;
};

export function holesLine(bench: Bench, benchmark: Benchmark): string {
  const where = `on ${benchmark.tileSet} (experiment \`${benchmark.expId}\`)`;
  if (bench.bands.length) {
    const bands = bench.bands.map((b) => `${b.short} ${b.pct == null ? "—" : b.pct.toFixed(0)}`).join(" · ");
    return `- Real holes ${where}: ${bands} %${bench.worst ? ` (worst: ${bench.worst.short})` : ""}`;
  }
  return `- Real holes ${where}: mean ${bench.holeMean != null ? bench.holeMean.toFixed(1) : "—"} %`;
}

export function variantNote(v: Variant & Scores & { bench: Bench | null }, opts: { prod: Scores | null; benchmark: Benchmark | null }): string {
  const name = v.kind === "rbf" ? "RBF" : variantLabel(v.name);
  const lines = [`**${v.production ? "Production gate" : "Gate variant"} \`${name}\`** — fitted ${utcText(v.fitted_at)}`, ""];
  const missing = v.membership.missing.map((l) => `#${memberNumber(l) ?? l}`).join(", ");
  lines.push(`- Reads ${readsText(v)} (${v.membership.current ? "the active set" : `not the active set${missing ? `: missing ${missing}` : ""}`})`);
  const steps = v.fit.steps_run ?? v.fit.steps;
  const fit = [v.mix_space ? `${v.mix_space} mix` : null, v.width != null ? `width ${v.width}` : null,
    steps != null ? `${String(steps)} steps` : null, kneesText(v.fit.loss_knees_e) ? `loss knees ${kneesText(v.fit.loss_knees_e)}` : null,
    v.use_lr ? "LR input" : "no LR input"].filter(Boolean).join(" · ");
  lines.push(`- Fit: ${fit}`);
  const s = v.selected;
  if (s && (s.loss != null || s.vis_psnr != null)) {
    lines.push(`- Held-out (step ${s.step}): ${[s.loss != null ? `loss ${s.loss.toFixed(4)}` : null, s.vis_psnr != null ? `VIS ${db(s.vis_psnr)} dB` : null].filter(Boolean).join(" · ")}`);
  }
  const vsProd = (a: number | null, b: number | null | undefined) =>
    (!v.production && a != null && b != null ? ` (${dbDelta(a - b)} vs production)` : "");
  const test = [v.testVis != null ? `VIS ${db(v.testVis)} dB${vsProd(v.testVis, opts.prod?.testVis)}` : null,
    v.kneeMean != null ? `∫PSNR ${db(v.kneeMean)} dB${vsProd(v.kneeMean, opts.prod?.kneeMean)}` : null].filter(Boolean).join(" · ");
  if (test) lines.push(`- Test: ${test}`);
  if (v.bench && opts.benchmark) lines.push(holesLine(v.bench, opts.benchmark));
  return lines.join("\n");
}

/** The compare table's rows for a field group: the plain mean, the best
 *  member (by VIS), then every other method in report order. */
export function compareRows(report: CompareReport, group: string): string[] {
  const block = report.groups[group];
  if (!block) return [];
  const vis = (k: string) => block[k].band_psnr[0] ?? -Infinity;
  const best = Object.keys(block).filter((k) => k.startsWith("member:"))
    .reduce<string | null>((b, k) => (b == null || vis(k) > vis(b) ? k : b), null);
  return ["mean", ...(best ? [best] : []), ...report.methods.filter((m) => m !== "mean")].filter((r) => block[r]);
}

const methodName = (r: string) => (r.startsWith("member:") ? `best member #${memberNumber(r.slice(7)) ?? r.slice(7)}` : variantLabel(r));

export function compareNote(report: CompareReport): string {
  const block = report.groups.natural ?? {};
  const n = report.n_fields ?? {};
  const fields = [n.natural != null ? `${n.natural} natural` : null, n.blackout ? `${n.blackout} blackout` : null].filter(Boolean).join(" + ");
  const lines = [
    `**Gate compare** — report \`${report.id ?? "latest"}\`${report.created ? `, ${utcText(report.created)}` : ""} · ${fields || "?"} test fields · ${report.members.length} cube members`, "",
    row(["Method", ...report.bands.map(shortBand), "∫ mean"]),
    row(["---", ...report.bands.map(() => "---:"), "---:"]),
    ...compareRows(report, "natural").map((r) => row([
      methodName(r), ...block[r].band_psnr.map((v) => db(v, 3)), db(mean(report.knee?.methods?.[r]?.integrated), 3),
    ])),
    "", "Natural test fields; ∫ mean = knee-integrated PSNR over the report's knee grid, mean over bands.",
  ];
  return lines.join("\n");
}

export function promoteNote(result: { promoted?: string | null; backup?: string | null; test_rescored?: boolean | null },
  after: Scores | null): string {
  const lines = [`**Promoted \`${variantLabel(result.promoted ?? "?")}\` to production**`, ""];
  if (result.backup) lines.push(`- The previous production gate is backed up as \`${result.backup}\` (promote it to roll back)`);
  if (result.test_rescored) {
    const s = [after?.testVis != null ? `VIS ${db(after.testVis)} dB` : null, after?.kneeMean != null ? `∫PSNR ${db(after.kneeMean)} dB` : null].filter(Boolean).join(" · ");
    lines.push(`- Test cubes re-scored${s ? `: production now ${s}` : ""}`);
  } else lines.push("- Test cubes not re-scored (the variant does not read the cached members)");
  return lines.join("\n");
}
