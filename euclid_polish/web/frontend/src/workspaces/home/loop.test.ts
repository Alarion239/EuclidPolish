// @vitest-environment node
import { describe, expect, it } from "vitest";
import {
  homeThumbs, loopStages, runningLine, stripAlerts, type LoopPayload, type MembersSlice, type PlatesSlice,
} from "./loop";
import type { SystemAlerts } from "../../app/status";

const NOW = Date.parse("2026-09-27T12:00:00Z");
const DAY = 86_400_000;
const iso = (daysAgo: number) => new Date(NOW - daysAgo * DAY).toISOString();

type Check = SystemAlerts["checks"][number];
const check = (id: string, state: Check["state"], title: string, patch: Partial<Check> = {}): Check =>
  ({ id, label: id, state, title, detail: null, to: null, ...patch } as Check);

const ALERTS = (patch: Record<string, Partial<Check>> = {}): SystemAlerts => {
  const checks = [
    check("disk", "ok", "400 GiB free on the data disk", { to: "/system/storage" }),
    check("real-sr", "ok", "All 12 real production SRs are current", { facts: { current: 12, stale: 0, missing: 3 } }),
    check("combiner", "ok", "Production gate fitted for the current 30 members", { facts: { members: 30 } }),
    check("evaluation", "ok", "Evaluation current (30 members, test records)", { facts: { evaluated_at: iso(0.1), n_scored: 100 } }),
    check("knee", "ok", "PSNR-vs-knee curves current"),
    check("records-noise", "unknown", "Noise model of the local records unverified"),
    check("tracking", "ok", "Tracking log up to date (last entry 2026-09-27)", { facts: { last_entry: iso(0.2), unlogged: [] } }),
  ].map((c) => ({ ...c, ...(patch[c.id] ?? {}) }));
  return { computed_at: iso(0), ttl_s: 30, counts: { bad: 0, warn: 0, ok: 0, unknown: 0 }, checks, alerts: checks.filter((c) => c.state === "warn" || c.state === "bad") } as SystemAlerts;
};

const MEMBERS: MembersSlice = {
  members: Array.from({ length: 30 }, (_, i) => ({ name: `member_${169 + i}`, status: "complete", timeout: false, step: 70000 })),
  archived: [{ name: "member_168" }],
};
const PLATES: PlatesSlice = { runs: [{ tag: "prod-0926", updated: iso(1), renders: [
  { band: "VIS", model: "production", model_label: "Production · spatial gate", model_fingerprint: "p1", created: iso(1), sheet: "s.png", tiles: [] },
] }] };

describe("the Loop strip", () => {
  it("shows the staleness service's stages as they are", () => {
    const stages = [{ id: "priors" as const, label: "Priors", state: "current" as const, reason: "galaxies v15", to: "/synthetic/status" }];
    const payload: LoopPayload = { computed_at: iso(0), ttl_s: 60, stages, counts: {}, errors: {} };
    expect(loopStages(payload)).toBe(stages);
  });

  it("reads 'checking' while the service loads and 'not checked' when it failed, never 'current'", () => {
    const loading = loopStages(null);
    expect(loading.map((s) => s.label)).toEqual(["Priors", "Records", "Members", "Evaluation", "Gate", "Real SR", "Figures"]);
    expect(loading.every((s) => s.state === "loading" && s.reason === "checking")).toBe(true);
    const failed = loopStages(undefined, true);
    expect(failed.every((s) => s.state === "unknown" && s.reason === "not checked")).toBe(true);
    expect(failed.find((s) => s.id === "real-sr")?.to).toBe("/sky/targets");
    expect(loopStages({ computed_at: iso(0), ttl_s: 60, stages: [], counts: {}, errors: {} }).every((s) => s.state === "loading")).toBe(true);
  });
});

describe("strip warnings (only when broken)", () => {
  it("says nothing while healthy", () => {
    expect(stripAlerts({ alerts: ALERTS(), fasrc: { ssh_connected: true } })).toEqual([]);
  });

  it("warns about the disk, FASRC and unlogged results", () => {
    const alerts = ALERTS({
      disk: { state: "bad", title: "8.0 GiB free on the data disk (98 % used)" },
      tracking: { state: "warn", title: "Results since the last tracking entry (2026-09-21)", facts: { last_entry: "2026-09-21T10:00:00+00:00", unlogged: [{ label: "evaluation", at: iso(1) }, { label: "PSNR vs knee", at: iso(1) }] } },
    });
    expect(stripAlerts({ alerts, fasrc: { ssh_connected: false, last_error: "timed out" } })).toEqual([
      { id: "disk", tone: "bad", text: "Disk: 8.0 GiB free", detail: "8.0 GiB free on the data disk (98 % used)", to: "/system/storage" },
      { id: "fasrc", tone: "warn", text: "FASRC not connected", detail: "timed out", to: "/system/connections" },
      { id: "tracking", tone: "warn", text: "No notebook entry since 09-21 · 2 results", detail: "evaluation, PSNR vs knee", to: "/notebook/log", log: true },
    ]);
  });
});

describe("running now", () => {
  it("names SLURM batches by their members with progress and GPU, then local jobs", () => {
    const line = runningLine({
      local: [{ job_id: "a", label: "evaluate", status: "running", progress: { current: 2, total: 5 } }],
      slurm: [{ jobid: "9", state: "RUNNING", label: "ensemble_train", params_json: JSON.stringify({ member_names: "member_199,member_200,member_201,member_202" }),
        progress_step: 10650, progress_total: 70000, gpu_util_mean: 78.4 }],
      members: MEMBERS,
    });
    expect(line?.items).toEqual(["members 199–202 on FASRC · 15% · GPU 78%", "evaluate on this laptop · 40%"]);
    expect(line?.timeouts).toEqual([]);
  });

  it("raises the members that stopped short (TIMEOUT) even when nothing runs", () => {
    const members = { ...MEMBERS, members: MEMBERS.members!.map((m, i) => (i < 2 ? { ...m, timeout: true, status: "timeout" } : m)) };
    const line = runningLine({ local: [], slurm: [], members });
    expect(line).toEqual({ items: [], timeouts: ["member_169", "member_170"], continueTo: "/models/starfull/train?mode=continue&members=member_169%2Cmember_170" });
    expect(runningLine({ local: [], slurm: [], members: MEMBERS })).toBeNull();
  });
});

describe("the thumbnail strip", () => {
  it("shows the newest real SR crops, then the newest plates, at most six, all cached", () => {
    const crops = Array.from({ length: 6 }, (_, i) => ({
      id: `vr-${i}`, label: `crop ${i}`, regime: "real", created_utc: iso(i), recipes: ["sr:VIS_H"], logical_tiers: ["sr"],
      source: { collection: "real", object: { ref: `nexus/f200w-00${40 + i}` } },
    }));
    const synthetic = { id: "vr-s", label: "syn", regime: "synthetic", created_utc: iso(0), recipes: [], logical_tiers: [], source: { collection: "sky" } };
    const thumbs = homeThumbs({ results: [synthetic, ...crops], plates: PLATES, poster: { ok: true, available: true, png: { size: 1, mtime: 5, pulled_at: iso(4) }, fits: null } });
    expect(thumbs.map((t) => t.kind)).toEqual(["crop", "crop", "crop", "crop", "plate", "plate"]);
    expect(thumbs[0]).toMatchObject({ label: "crop 0", src: "/viewer/results/vr-0/panel.png?size=240", to: "/sky/targets?inspect=realtile%3Anexus%2Ff200w-0040" });
    expect(thumbs[4]).toMatchObject({ label: "NEXUS comparison", src: "/api/figures/nexus-plates/prod-0926/s.png?thumb=320", to: "/figures/plates?plate=nexus&run=prod-0926" });
    expect(thumbs[5]).toMatchObject({ label: "Synthetic poster scene", src: "/poster/result/cutout.png?v=5", to: "/figures/plates?plate=poster" });
    expect(homeThumbs({ results: [], plates: { runs: [] }, poster: null })).toEqual([]);
  });

  it("puts the newest cached production SRs of real tiles first; saved crops fill the room left", () => {
    const tile = (id: string, state: string, daysAgo: number) => ({
      ref: `nexus/${id}`, source: "nexus", id, source_label: "NEXUS × Euclid tiles", state, created: iso(daysAgo),
      thumb: `/api/figures/real-sr/nexus/${id}.jpg`,
    });
    const crop = { id: "vr-0", label: "crop 0", regime: "real", created_utc: iso(0), source: { collection: "real", object: { ref: "poster/g1" } } };
    const realSr = { total: 2, items: [tile("f200w-0400", "stale", 1), tile("f200w-0350", "current", 2)] };
    const thumbs = homeThumbs({ realSr, results: [crop], plates: PLATES, poster: null });
    expect(thumbs.map((t) => t.kind)).toEqual(["tile", "tile", "crop", "plate"]);
    expect(thumbs[0]).toMatchObject({
      label: "f200w-0400", sub: "NEXUS · stale", to: "/sky/targets?inspect=realtile%3Anexus%2Ff200w-0400",
      src: `/api/figures/real-sr/nexus/f200w-0400.jpg?v=${encodeURIComponent(iso(1))}`,
    });
    expect(thumbs[1].sub).toBe("NEXUS");
    const many = { total: 9, items: Array.from({ length: 9 }, (_, i) => tile(`f200w-0${100 + i}`, "current", i)) };
    expect(homeThumbs({ realSr: many, results: [crop], plates: PLATES, poster: null }).map((t) => t.kind))
      .toEqual(["tile", "tile", "tile", "tile", "tile", "plate"]);
  });
});
