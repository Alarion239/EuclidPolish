/* The Home Loop: one chip per stage of the loop you run after every batch —
 * Priors, Records, Members, Evaluation, Gate, Real SR, Figures — each current,
 * stale, blocked or unknown with ONE reason and the tab whose confirmed
 * button fixes it; the problem-only strip warnings (disk, FASRC, unlogged
 * results); the "Running now" line with the TIMEOUT alert; and the strip of
 * cached thumbnails. Pure; loop.test.ts.
 *
 * The verdicts come only from the backend staleness service, GET
 * /api/system/loop (helpers/system_alerts.py, which System › Lineage reads
 * too; its rules are pinned by tests/test_system_alerts.py). Until it
 * answers, each chip reads "checking"; if it fails, "not checked" — never
 * "current". */
import type { HealthCheck, SystemAlerts } from "../../app/status";
import { memberRange } from "./homeModel";

export type StageId = "priors" | "records" | "members" | "evaluation" | "gate" | "real-sr" | "figures";
export type StageState = "current" | "stale" | "blocked" | "unknown" | "loading";
export type Stage = {
  id: StageId; label: string; state: StageState;
  /** One short reason ("3 new on FASRC", "predate the stellar prior"). */
  reason: string;
  /** The longer explanation (the chip's tooltip). */
  detail?: string | null;
  /** The tab that shows it, or whose confirmed button fixes it. */
  to: string;
};

/** The slice of GET /ensemble/members.json read here. */
export type MembersSlice = {
  members?: { name: string; status?: string; timeout?: boolean; step?: number | null }[] | null;
  archived?: { name: string }[] | null;
};
type PlateRenderSlice = {
  band: string; model: string | null; legacy?: boolean; model_label?: string | null; model_fingerprint?: string | null;
  created?: string | null; sheet: string | null; tiles: { index: number; file: string; ref?: string | null; model_state?: string | null }[];
};
/** The slice of GET /api/figures/nexus-plates read here. */
export type PlatesSlice = { runs?: { tag: string; updated: string | null; renders: PlateRenderSlice[] }[] | null };

/** GET /api/system/loop (helpers/system_alerts.py loop_payload). */
export type LoopPayload = {
  computed_at: string; ttl_s: number; stages: Stage[];
  counts: Record<string, number>; errors: Record<string, string>;
};

/** The seven stages in loop order, with the tab each one opens: the
 *  placeholders shown while the service has not answered (or failed). */
const STAGES: readonly { id: StageId; label: string; to: string }[] = [
  { id: "priors", label: "Priors", to: "/synthetic/status" },
  { id: "records", label: "Records", to: "/synthetic/records" },
  { id: "members", label: "Members", to: "/models/starfull/members" },
  { id: "evaluation", label: "Evaluation", to: "/models/starfull/leaderboard" },
  { id: "gate", label: "Gate", to: "/models/starfull/combiner" },
  { id: "real-sr", label: "Real SR", to: "/sky/targets" },
  { id: "figures", label: "Figures", to: "/figures/plates" },
];

/** The chips to show: the service's stages, or placeholders ("checking"
 *  while it loads, "not checked" when it failed). */
export function loopStages(payload: LoopPayload | null | undefined, failed = false): Stage[] {
  if (payload?.stages?.length) return payload.stages;
  return STAGES.map((s) => failed
    ? { ...s, state: "unknown" as const, reason: "not checked" }
    : { ...s, state: "loading" as const, reason: "checking" });
}

/* ── strip warnings ─────────────────────────────────────────────────────── */

const checkOf = (a: SystemAlerts | null | undefined, id: string): HealthCheck | null => a?.checks.find((c) => c.id === id) ?? null;
const bad = (c: HealthCheck | null) => c?.state === "warn" || c?.state === "bad";
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

export type StripAlert = {
  id: string; tone: "warn" | "bad"; text: string; detail?: string | null; to: string;
  /** The Log dialog opens (pre-filled with the unlogged results). */
  log?: boolean;
};

const shortDay = (iso: unknown) => (typeof iso === "string" && /^\d{4}-\d{2}-\d{2}/.test(iso) ? iso.slice(5, 10) : null);

/** Disk, FASRC and unlogged results: only when broken; nothing otherwise. */
export function stripAlerts({ alerts, fasrc }: {
  alerts: SystemAlerts | null | undefined; fasrc: { ssh_connected: boolean; last_error?: string | null } | null | undefined;
}): StripAlert[] {
  const out: StripAlert[] = [];
  const disk = checkOf(alerts, "disk");
  if (bad(disk)) {
    const free = /^([\d.]+\s*\w+) free/.exec(disk!.title)?.[1];
    out.push({ id: "disk", tone: disk!.state === "bad" ? "bad" : "warn", text: free ? `Disk: ${free} free` : "Low disk", detail: disk!.title, to: "/system/storage" });
  }
  if (fasrc && !fasrc.ssh_connected) {
    out.push({ id: "fasrc", tone: "warn", text: "FASRC not connected", detail: fasrc.last_error ?? null, to: "/system/connections" });
  }
  const tracking = checkOf(alerts, "tracking");
  if (bad(tracking)) {
    const unlogged = Array.isArray(tracking!.facts?.unlogged) ? (tracking!.facts!.unlogged as { label?: string }[]) : [];
    const day = shortDay(tracking!.facts?.last_entry);
    const n = unlogged.length;
    out.push({ id: "tracking", tone: "warn",
      text: `${day ? `No notebook entry since ${day}` : "Unlogged results"}${n ? ` · ${n} result${n === 1 ? "" : "s"}` : ""}`,
      detail: unlogged.map((u) => u.label).filter(Boolean).join(", ") || tracking!.detail || null, to: "/notebook/log", log: true });
  }
  return out;
}

/* ── running now ────────────────────────────────────────────────────────── */

type LocalJobSlice = { job_id: string; label: string; status: string; progress?: { current?: number; total?: number } | null };
type SlurmJobSlice = {
  jobid: string; state: string; label?: string | null; step_id?: string | null; params_json?: unknown;
  progress_step?: number | null; progress_total?: number | null; gpu_util_mean?: unknown; [key: string]: unknown;
};

const pct = (current: number, total: number) => (total > 0 ? ` · ${Math.round((100 * current) / total)}%` : "");

function slurmMembers(job: SlurmJobSlice): string | null {
  let params: Record<string, unknown>;
  try { params = JSON.parse(String(job.params_json ?? "{}")) as Record<string, unknown>; } catch { return null; }
  if (!params || typeof params !== "object") return null;
  const raw = params.mode === "continue" ? params.members : params.member_names ?? params.members;
  return typeof raw === "string" ? memberRange(raw.split(",")) : null;
}

export type RunningLine = {
  /** One phrase per running / queued job, SLURM first. */
  items: string[];
  /** Active STARFULL members that stopped short of their target (TIMEOUT). */
  timeouts: string[];
  continueTo?: string;
};

/** The "Running now" line, or null when nothing runs and nothing timed out. */
export function runningLine({ local, slurm, members }: {
  local: readonly LocalJobSlice[]; slurm: readonly SlurmJobSlice[]; members: MembersSlice | null | undefined;
}): RunningLine | null {
  const remote = slurm.filter((j) => j.state === "RUNNING" || j.state === "PENDING").map((j) => {
    const what = slurmMembers(j) ?? j.label ?? j.step_id ?? `job ${j.jobid}`;
    if (j.state === "PENDING") return `${what} on FASRC · queued`;
    const gpu = num(j.gpu_util_mean);
    return `${what} on FASRC${pct(Number(j.progress_step ?? 0), Number(j.progress_total ?? 0))}${gpu != null ? ` · GPU ${Math.round(gpu)}%` : ""}`;
  });
  const here = local.filter((j) => j.status === "running")
    .map((j) => `${j.label} on this laptop${pct(j.progress?.current ?? 0, j.progress?.total ?? 0)}`);
  const timeouts = (members?.members ?? []).filter((m) => m.timeout).map((m) => m.name);
  const items = [...remote, ...here];
  if (!items.length && !timeouts.length) return null;
  const line: RunningLine = { items, timeouts };
  if (timeouts.length) {
    line.continueTo = `/models/starfull/train?${new URLSearchParams({ mode: "continue", members: timeouts.join(",") }).toString()}`;
  }
  return line;
}

/* ── thumbnails ─────────────────────────────────────────────────────────── */

export type Thumb = { key: string; kind: "tile" | "crop" | "plate"; label: string; sub?: string | null; src: string; to: string; at?: string | null };

/** One cached production SR of a real tile (GET /api/figures/real-sr). */
export type RealSrItem = {
  ref: string; source: string; id: string; source_label?: string | null; label?: string | null;
  created?: string | null; state?: string | null; thumb: string;
};
export type RealSrSlice = { total?: number; items?: RealSrItem[] } | null;

type SavedSlice = {
  id: string; label?: string; regime?: string | null; created_utc?: string | null;
  source?: { collection?: string; params?: Record<string, string>; object?: { id?: string; ref?: string } | null } | null;
};
type PosterSlice = { png?: { mtime?: number; pulled_at?: string; size?: number } | null; [key: string]: unknown } | null;

/** Short source names for the thumbnail sub-line (the full label is too long for a card). */
const SOURCE_SHORT: Record<string, string> = {
  nexus: "NEXUS", tile: "Cached tile", field: "Legacy field", archive: "Archive field", eval: "Evaluation object",
  poster: "Poster target", pair: "JWST pair",
};

const tileCardOf = (ref: string) => `/sky/targets?${new URLSearchParams({ inspect: `realtile:${ref}` }).toString()}`;

/** The Sky targets tile card of a saved real crop (figures/model.ts
 *  viewerLink's real-tile cases), or null. */
function tileCard(r: SavedSlice): string | null {
  const src = r.source;
  const obj = src?.object ?? {};
  let ref: string | null = null;
  if (src?.collection === "real") ref = obj.ref ?? (obj.id && src.params?.source ? `${src.params.source}/${obj.id}` : null);
  else if (src?.collection === "jwst-euclid" && obj.id) ref = `pair/${obj.id}`;
  return ref ? tileCardOf(ref) : null;
}

const MAX_THUMBS = 6;
const MAX_PLATES = 2;

/** Up to six cached images: the newest production SRs of real tiles (each
 *  opens its tile card; crops saved from real SR tiles fill any room left,
 *  e.g. on a server that predates /api/figures/real-sr), then the newest plates
 *  (the NEXUS contact sheet, the pulled poster scene), each opening its
 *  plate. Everything is a file that already exists: nothing runs a model. */
export function homeThumbs({ realSr, results, plates, poster }: {
  realSr?: RealSrSlice | undefined;
  results: readonly SavedSlice[] | null | undefined; plates: PlatesSlice | null | undefined; poster: PosterSlice | undefined;
}): Thumb[] {
  const plateThumbs: Thumb[] = [];
  const run = plates?.runs?.[0];
  const sheet = run?.renders.find((r) => r.sheet);
  if (run && sheet?.sheet) {
    plateThumbs.push({ key: `plate:${run.tag}`, kind: "plate", label: "NEXUS comparison", sub: run.tag, at: sheet.created ?? run.updated,
      src: `/api/figures/nexus-plates/${encodeURIComponent(run.tag)}/${encodeURIComponent(sheet.sheet)}?thumb=320`,
      to: `/figures/plates?${new URLSearchParams({ plate: "nexus", run: run.tag }).toString()}` });
  }
  if (poster?.png) {
    plateThumbs.push({ key: "plate:poster", kind: "plate", label: "Synthetic poster scene", sub: null, at: poster.png.pulled_at ?? null,
      src: `/poster/result/cutout.png${poster.png.mtime ? `?v=${Math.round(poster.png.mtime)}` : ""}`, to: "/figures/plates?plate=poster" });
  }
  const shownPlates = plateThumbs.slice(0, MAX_PLATES);
  const room = MAX_THUMBS - shownPlates.length;
  const tiles: Thumb[] = (realSr?.items ?? []).slice(0, room).map((t) => ({
    key: `tile:${t.ref}`, kind: "tile", label: t.id, at: t.created ?? null, to: tileCardOf(t.ref),
    sub: [SOURCE_SHORT[t.source] ?? t.source_label ?? t.source, t.state === "stale" ? "stale" : null].filter(Boolean).join(" · "),
    src: `${t.thumb}${t.created ? `?v=${encodeURIComponent(t.created)}` : ""}`,
  }));
  const crops = (results ?? [])
    .map((r) => ({ r, to: tileCard(r) }))
    .filter((x): x is { r: SavedSlice; to: string } => !!x.to && x.r.regime !== "synthetic")
    .sort((a, b) => String(b.r.created_utc ?? "").localeCompare(String(a.r.created_utc ?? "")))
    .slice(0, room - tiles.length)
    .map(({ r, to }) => ({ key: r.id, kind: "crop" as const, label: r.label || r.id, sub: r.source?.params?.source ?? null, at: r.created_utc ?? null,
      src: `/viewer/results/${encodeURIComponent(r.id)}/panel.png?size=240`, to }));
  return [...tiles, ...crops, ...shownPlates];
}
