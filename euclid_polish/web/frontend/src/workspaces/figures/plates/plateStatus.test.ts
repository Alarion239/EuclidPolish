// @vitest-environment node
import { describe, expect, it } from "vitest";
import type { PlateRender, PlateRun } from "../api";
import { captionParts, nexusCaption, posterCaption, staticPlateCaption, type OverviewSlice } from "./plateStatus";

const NOW = Date.parse("2026-09-27T12:00:00Z");
const DAY = 86_400_000;
const iso = (daysAgo: number) => new Date(NOW - daysAgo * DAY).toISOString();

const overview = (patch: { galaxy?: Record<string, unknown>; star?: Record<string, unknown>; plots?: Record<string, unknown> } = {}): OverviewSlice => ({
  items: [
    { id: "galaxy-model", state: "ok", facts: { version: 15, is_active: true, active_fingerprint: "g1", candidate_fingerprint: "g1", candidate_valid: true, ...patch.galaxy },
      records: { prior_at: iso(13) } },
    { id: "star-prior", state: "ok", facts: { is_active: true, active_fingerprint: "s1", candidate_fingerprint: "s1", candidate_valid: true, ...patch.star },
      records: { prior_at: iso(46) } },
    { id: "galaxy-plots", state: "ok", facts: { present: true, stale: false, reason: null, built_at: iso(0.5), ...patch.plots } },
  ],
});

describe("static plate captions", () => {
  it("names the galaxy model the population plate is drawn from, and when it was activated", () => {
    const c = staticPlateCaption("population", overview());
    expect(captionParts(c, NOW)).toEqual(["made with galaxy model v15", "current", "activated 13 d ago"]);
    expect(c.tone).toBe("neutral");
  });

  it("says when the plate shows a galaxy candidate that is not active", () => {
    const c = staticPlateCaption("population", overview({ galaxy: { version: 16, candidate_fingerprint: "g2" } }));
    expect(captionParts(c, NOW)).toEqual(["made with the galaxy candidate v16", "not active"]);
    expect(c.tone).toBe("warn");
    expect(c.reason).toMatch(/Activate it on Synthetic › Galaxies/);
  });

  it("says there is nothing to draw without a fit", () => {
    const c = staticPlateCaption("population", overview({ galaxy: { candidate_fingerprint: null, active_fingerprint: null, is_active: false } }));
    expect(captionParts(c, NOW)).toEqual(["no galaxy fit yet"]);
  });

  it("draws the stellar plate from the active prior, noting a newer candidate", () => {
    expect(captionParts(staticPlateCaption("stars", overview()), NOW))
      .toEqual(["made with the active stellar prior", "current", "activated 46 d ago"]);
    const newer = staticPlateCaption("stars", overview({ star: { candidate_fingerprint: "s2" } }));
    expect(captionParts(newer, NOW)).toEqual(["made with the active stellar prior", "current", "activated 46 d ago", "a newer candidate is not active"]);
    const only = staticPlateCaption("stars", overview({ star: { active_fingerprint: null, is_active: false, candidate_fingerprint: "s2" } }));
    expect(captionParts(only, NOW)).toEqual(["made with the stellar candidate", "not active"]);
    expect(only.tone).toBe("warn");
  });

  it("dates the galaxy 2×2 by its plot cache and flags a stale one with the cache's reason", () => {
    expect(captionParts(staticPlateCaption("galaxies", overview()), NOW))
      .toEqual(["made with galaxy model v15", "current", "built 12 h ago"]);
    const stale = staticPlateCaption("galaxies", overview({ plots: { stale: true, reason: "the plot schema changed since the last build", built_at: iso(2) } }));
    expect(captionParts(stale, NOW)).toEqual(["made with galaxy model v15", "stale", "built 2 d ago"]);
    expect(stale.reason).toBe("the plot schema changed since the last build — rebuild the plots on Synthetic › Status");
    expect(stale.tone).toBe("warn");
  });

  it("says nothing it cannot know while the overview loads", () => {
    expect(captionParts(staticPlateCaption("population", null), NOW)).toEqual([]);
  });
});

describe("NEXUS and poster captions", () => {
  const render = (patch: Partial<PlateRender> = {}): PlateRender =>
    ({ band: "VIS", model: "production", model_label: "Production · spatial gate", model_fingerprint: "p1", sheet: null, tiles: [], created: iso(3), ...patch });
  const run = (r: PlateRender): PlateRun => ({ tag: "t", updated: iso(1), renders: [r], files: [] });
  const catalog = { models: [{ spec: "production", fingerprint: "p1" }, { spec: "rbf", fingerprint: "r1" }] };

  it("is current when the render's model fingerprint is today's", () => {
    const r = render();
    expect(captionParts(nexusCaption(r, run(r), catalog), NOW)).toEqual(["made with Production · spatial gate", "current", "rendered 3 d ago"]);
  });

  it("is stale for an earlier fit, for a stale tile output and for a legacy SR", () => {
    const earlier = render({ model_fingerprint: "p0" });
    expect(captionParts(nexusCaption(earlier, run(earlier), catalog), NOW)[1]).toBe("stale");
    const tile = render({ tiles: [{ index: 40, file: "a.png", model_state: "stale" }] });
    expect(nexusCaption(tile, run(tile), catalog).reason).toMatch(/1 tile output predates/);
    const legacy = render({ model: null, legacy: true, model_label: "minibatched RBF", model_fingerprint: null, created: null });
    const c = nexusCaption(legacy, run(legacy), catalog);
    expect(captionParts(c, NOW)).toEqual(["made with minibatched RBF (legacy SR)", "stale", "rendered 1 d ago"]);
    expect(c.reason).toMatch(/not the production gate/);
  });

  it("names the poster's pull time", () => {
    expect(captionParts(posterCaption({ ok: true, available: true, png: { size: 1, mtime: 0, pulled_at: iso(5) }, fits: null }), NOW))
      .toEqual(["synthetic scene from the poster_cutout step", "pulled 5 d ago"]);
    expect(captionParts(posterCaption({ ok: true, available: false, png: null, fits: null }), NOW)).toEqual([]);
    expect(captionParts(posterCaption(null), NOW)).toEqual([]);
  });
});
