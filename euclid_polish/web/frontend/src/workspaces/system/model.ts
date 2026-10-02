/* Pure helpers of the System workspace (no React): git relations and
 * porcelain states in words, the one Code sentence (laptop / server / FASRC
 * commits), du sizes, the disk captions, the Config groups' home tabs, the
 * Appearance "Images" line and the Lineage callout. Unit-tested in
 * model.test.ts. */
import { CMAP_LABEL, COLOR_LABEL, STRETCH_LABEL } from "../../app/DisplayPanel";
import { pagePath } from "../../app/nav";
import { formatBytes } from "../../format";
import { transferFor, type DisplaySettings } from "../../state/display";
import type { Tone } from "../../ui";

/* ── git ──────────────────────────────────────────────────────────────────── */

export type Relation = "same" | "remote_behind" | "remote_ahead" | "diverged" | "unknown";

export function relationText(rel: { relation: Relation | string; ahead?: number | null; behind?: number | null } | null | undefined):
  { label: string; tone: Tone; hint: string } {
  const r = rel?.relation ?? "unknown";
  const n = (k: number | null | undefined) => `${k ?? "?"} commit${k === 1 ? "" : "s"}`;
  switch (r) {
    case "same": return { label: "in sync with this laptop", tone: "good", hint: "FASRC runs the same commit as the local checkout." };
    case "remote_behind": return { label: `FASRC ${n(rel?.ahead)} behind`, tone: "warn", hint: "Push the local commits, then git pull on FASRC." };
    case "remote_ahead": return { label: `FASRC ${n(rel?.behind)} ahead`, tone: "info", hint: "FASRC has commits this laptop has not pulled." };
    case "diverged": return { label: "diverged", tone: "bad", hint: `Local ${n(rel?.ahead)} ahead, FASRC ${n(rel?.behind)} ahead.` };
    default: return { label: "unknown", tone: "neutral", hint: "The FASRC commit is not in the local repo (fetch first)." };
  }
}

/** Porcelain XY → a short human status. */
export function gitStatusText(xy: string): string {
  if (xy === "??") return "untracked";
  if (xy.includes("U") || xy === "AA" || xy === "DD") return "conflict";
  const pick = (c: string) => ({ M: "modified", A: "added", D: "deleted", R: "renamed", C: "copied", T: "type" }[c]);
  return pick(xy[0]) ?? pick(xy[1]) ?? xy.trim();
}

/** Append one history page to the commits already loaded, dropping repeats
 *  (a new commit shifts the pages by one while the history is open). */
export function mergeCommitPages<C extends { hash: string; full?: string }>(older: readonly C[], page: readonly C[]): C[] {
  const seen = new Set(older.map((c) => c.full ?? c.hash));
  return [...older, ...page.filter((c) => !seen.has(c.full ?? c.hash))];
}

/* ── paths ────────────────────────────────────────────────────────────────── */

export const basename = (path: string): string => path.replace(/\/+$/, "").split("/").pop() || path;

/* ── Code: one question ──────────────────────────────────────────────────── */

type FasrcSide = { head?: string | null; relation?: string; ahead?: number | null; behind?: number | null } | null | "loading";

/** "Laptop, server and FASRC are on 3e8b270." — or which one differs. The
 *  laptop is the checkout's HEAD, the server the commit it booted from, FASRC
 *  its checkout's HEAD (null: not connected; "loading": not read yet). */
export function codeSentence({ laptop, server, fasrc }: { laptop: string | null; server: string | null; fasrc: FasrcSide }):
  { text: string; tone: Tone } {
  if (!laptop && !server) return { text: "Reading the commits…", tone: "neutral" };
  const f = fasrc && fasrc !== "loading" && fasrc.head ? fasrc.head.slice(0, 7) : null;
  const rel = fasrc && fasrc !== "loading" ? fasrc.relation : undefined;
  const fasrcSame = rel === "same" || (!!f && !!laptop && f === laptop.slice(0, 7));
  const serverSame = !!laptop && !!server && laptop === server;
  if (fasrc === "loading" || !fasrc || !f) {
    const tail = fasrc === "loading" ? "reading the FASRC checkout…" : "FASRC is not connected.";
    if (serverSame) return { text: `Laptop and server are on ${laptop}; ${tail}`, tone: fasrc === "loading" ? "neutral" : "warn" };
    return { text: `This laptop is on ${laptop ?? "—"}, the server started at ${server ?? "—"}; ${tail}`, tone: "warn" };
  }
  if (serverSame && fasrcSame) return { text: `Laptop, server and FASRC are on ${laptop}.`, tone: "good" };
  if (fasrcSame) return { text: `Laptop and FASRC are on ${laptop}; the server started at ${server ?? "—"}.`, tone: "warn" };
  const how = relationText({ relation: rel ?? "unknown", ahead: fasrc.ahead, behind: fasrc.behind });
  if (serverSame) {
    const words = rel === "remote_behind" || rel === "remote_ahead" ? how.label.replace(/^FASRC /, "") : `on ${f}`;
    const text = words.startsWith("on ") ? `Laptop and server are on ${laptop}; FASRC is ${words}.`
      : `Laptop and server are on ${laptop}; FASRC is ${words} (${f}).`;
    return { text, tone: "warn" };
  }
  return { text: `This laptop is on ${laptop ?? "—"}, the server started at ${server ?? "—"}, FASRC is on ${f} (${how.label}).`, tone: "warn" };
}

/* ── Storage ─────────────────────────────────────────────────────────────── */

const DU_SCALE: Record<string, number> = { "": 1, K: 1024, M: 1024 ** 2, G: 1024 ** 3, T: 1024 ** 4, P: 1024 ** 5 };

/** `du -h` sizes ("12G", "1.5T", "512K") in bytes, for sorting. */
export function duBytes(size: string): number | null {
  const m = /^\s*([\d.]+)\s*([KMGTP]?)i?B?\s*$/i.exec(size);
  if (!m) return null;
  return Number(m[1]) * DU_SCALE[m[2].toUpperCase()];
}

/** "68 GiB of 427 GiB used is ours": the data roots' share of the used disk. */
export function diskCaption(disk: { used_bytes: number }, ours: number | null): string {
  if (ours == null) return `${formatBytes(disk.used_bytes)} used; measure the data roots to see how much is ours`;
  return `${formatBytes(ours)} of ${formatBytes(disk.used_bytes)} used is ours`;
}

/** The ONE rule (routes/system.py disk_level) that tones the bar here and raises the Home alert. */
export function diskThresholdText(d: { warn_below_bytes: number; bad_below_bytes: number; warn_used_fraction: number }): string {
  return `Warns below ${formatBytes(d.warn_below_bytes)} free or at ${Math.round(d.warn_used_fraction * 100)}% used (here and on Home); `
    + `experiments stop below ${formatBytes(d.bad_below_bytes)}.`;
}

/** The experiments' share of the disk as one line: the member-SR cache and
 *  the outputs only when they hold something (a 0 B row says nothing), then
 *  the space the experiments keep free. */
export function experimentsLine(e: { cache_budget_bytes: number; min_free_bytes: number; cache_bytes: number | null; outputs_bytes: number | null }): string {
  const parts: string[] = [];
  if (e.cache_bytes) parts.push(`member-SR cache ${formatBytes(e.cache_bytes)} of its ${formatBytes(e.cache_budget_bytes)} budget`);
  if (e.outputs_bytes) parts.push(`outputs ${formatBytes(e.outputs_bytes)}`);
  return `Experiments: ${parts.length ? parts.join(", ") : "nothing cached"}; ${formatBytes(e.min_free_bytes)} kept free.`;
}

/* ── Config ──────────────────────────────────────────────────────────────── */

/** The tab where a Config group's effect is judged (its header links there). */
export function groupHome(group: string): { label: string; to: string } | null {
  const synthetic = (tab: string, label: string) => ({ label: `Synthetic › ${label}`, to: pagePath("synthetic", { tab }) });
  switch (group) {
    case "cutouts": case "psf": return synthetic("psf", "PSF");
    case "scenes": case "lenses": return synthetic("records", "Records");
    case "stars": return synthetic("stars", "Stars");
    case "lr": case "plateau": return { label: "Models › Train", to: pagePath("models", { tab: "train" }) };
    default: return null;
  }
}

/* ── Appearance ──────────────────────────────────────────────────────────── */

const STRETCH_WORDS: Record<string, string> = { "asinh-abs": "absolute asinh", "asinh-auto": "auto asinh" };
const sig3 = (v: number) => String(Number(v.toPrecision(3)));

/** "VIS, absolute asinh, knee 100 e⁻" (+ what else differs from the default). */
export function imagesLine(d: DisplaySettings): string {
  const t = transferFor(d);
  const stretch = STRETCH_WORDS[d.stretch] ?? (STRETCH_LABEL[d.stretch] ?? d.stretch).toLowerCase();
  const parts = [
    COLOR_LABEL[d.color] ?? d.color,
    stretch,
    `knee ${sig3(t.knee)} e⁻${t.gain !== 1 ? ` ×${sig3(t.gain)}` : ""}`,
    d.colormap !== "gray" ? (CMAP_LABEL[d.colormap] ?? d.colormap) : "",
    d.invert ? "inverted" : "",
  ];
  return parts.filter(Boolean).join(", ");
}

/* ── Lineage ─────────────────────────────────────────────────────────────── */

/** The one callout while most records carry no model id: it qualifies only the
 *  per-record model check, since each record then takes its Loop stage's verdict. */
export function lineageCallout(s: { total: number; counts: { verdicts: Partial<Record<"current" | "stale" | "unknown", number>> } } | null | undefined): string | null {
  if (!s || s.total <= 0) return null;
  const v = s.counts.verdicts;
  const unknown = v.unknown ?? 0;
  if (unknown <= (v.current ?? 0) + (v.stale ?? 0)) return null;
  return `${Math.round((100 * unknown) / s.total)}% of records carry no model id, so each record takes its Loop stage's verdict.`;
}

/** One stage of the staleness service (GET /api/system/loop), the slice read here. */
export type LoopStageSlice = { id: string; label: string; state: string; reason: string; detail?: string | null; to: string };
export type LineageVerdict = "current" | "stale" | "blocked" | "unknown";
export const LINEAGE_VERDICTS: readonly LineageVerdict[] = ["current", "stale", "blocked", "unknown"];

/** The Loop stage each record kind belongs to: a generation run makes
 *  records, a checkpoint is a member, and the SR runs and cutouts of real
 *  sky are Real SR. A kind not listed has no verdict. */
const KIND_STAGE: Record<string, string> = {
  generationrun: "records",
  checkpointartifact: "members",
  inferencerun: "real-sr",
  srcutoutartifact: "real-sr",
};

/** The Loop stage a record kind takes its verdict from (null: none, or the service has not answered). */
export function stageOfKind(kind: string, stages: readonly LoopStageSlice[] | null | undefined): LoopStageSlice | null {
  const id = KIND_STAGE[kind];
  return (id && stages?.find((s) => s.id === id)) || null;
}

const isVerdict = (s: string): s is LineageVerdict => (LINEAGE_VERDICTS as readonly string[]).includes(s);

/** Records per Loop verdict (from the per-kind counts of the index), and the
 *  kinds each verdict selects. A stage still checking gives no verdict. */
export function lineageVerdicts(kinds: Record<string, number>, stages: readonly LoopStageSlice[] | null | undefined): {
  counts: Record<LineageVerdict, number>; kinds: Record<LineageVerdict, string[]>;
} {
  const counts: Record<LineageVerdict, number> = { current: 0, stale: 0, blocked: 0, unknown: 0 };
  const out: Record<LineageVerdict, string[]> = { current: [], stale: [], blocked: [], unknown: [] };
  for (const [kind, n] of Object.entries(kinds)) {
    const state = stageOfKind(kind, stages)?.state ?? "";
    if (!isVerdict(state)) continue;
    counts[state] += n;
    out[state].push(kind);
  }
  for (const v of LINEAGE_VERDICTS) out[v].sort();
  return { counts, kinds: out };
}

/** The server's `kind` filter for a chosen kind and verdict: the verdict's
 *  kinds (comma list), intersected with the chosen kind; null when no record
 *  can match (the table is then empty without a request). */
export function verdictKindFilter(kind: string, verdict: LineageVerdict | "", v: { kinds: Record<LineageVerdict, string[]> }): string | null {
  if (!verdict) return kind;
  const allowed = v.kinds[verdict];
  const pick = kind ? allowed.filter((k) => k === kind) : allowed;
  return pick.length ? pick.join(",") : null;
}
