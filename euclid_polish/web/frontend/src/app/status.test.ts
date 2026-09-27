import { describe, expect, it } from "vitest";
import { bannerKey, buildChanged, distUpdated, documentEntry, type VersionInfo } from "./status";

const V: VersionInfo = {
  boot_commit: "a", boot_short: "a", head_commit: "b", head_short: "b", behind: true, dirty: false,
  changed_files: ["euclid_polish/web/app.py"], changed_count: 1,
  started_at: "2026-09-27T01:00:00Z", pid: 7, dist: { built_at: null, index_hash: "h1" },
};

describe("distUpdated (a new SPA build is served)", () => {
  it("is true only when the hash this page loaded with and the served one differ", () => {
    expect(distUpdated("h1", "h2")).toBe(true);
    expect(distUpdated("h1", "h1")).toBe(false);
  });

  it("is false while either hash is unknown (no build, or not seen yet)", () => {
    expect(distUpdated(null, "h2")).toBe(false);
    expect(distUpdated("h1", null)).toBe(false);
    expect(distUpdated(undefined, undefined)).toBe(false);
  });
});

describe("bannerKey (what a dismissal of the restart banner is tied to)", () => {
  it("changes when the server restarts or another file changes", () => {
    const k = bannerKey(V);
    expect(bannerKey({ ...V })).toBe(k);
    expect(bannerKey({ ...V, started_at: "2026-09-27T02:00:00Z" })).not.toBe(k);
    expect(bannerKey({ ...V, changed_files: ["euclid_polish/web/app.py", "euclid_polish/jobs.py"], changed_count: 2 })).not.toBe(k);
    expect(bannerKey({ ...V, changed_count: 9 })).not.toBe(k);
  });

  it("does not change when an already-changed file is edited again (it only moves up the list)", () => {
    const two = { ...V, changed_files: ["a.py", "b.py"], changed_count: 2 };
    expect(bannerKey({ ...two, changed_files: ["b.py", "a.py"] })).toBe(bannerKey(two));
  });

  it("does not change with HEAD, dirty or the SPA build", () => {
    const k = bannerKey(V);
    expect(bannerKey({ ...V, head_commit: "c", head_short: "c", dirty: true, dist: { built_at: "x", index_hash: "h9" } })).toBe(k);
  });

  it("with the server's changed_digest: stable while the changed SET is the same, whatever subset is listed", () => {
    const a = { ...V, changed_files: Array.from({ length: 8 }, (_, i) => `f${i}.py`), changed_count: 12, changed_digest: "d1" };
    // f10 was re-saved: it moved into the listed newest 8, nothing new needs a restart.
    const b = { ...a, changed_files: ["f10.py", ...a.changed_files.slice(0, 7)] };
    expect(bannerKey(b)).toBe(bannerKey(a));
    expect(bannerKey({ ...a, changed_count: 13, changed_digest: "d2" })).not.toBe(bannerKey(a));
    expect(bannerKey({ ...a, pid: 8 })).not.toBe(bannerKey(a));
  });

  it("works for a server that does not list files", () => {
    const { changed_files: _f, changed_count: _c, ...old } = V;
    expect(bannerKey(old)).toBe("2026-09-27T01:00:00Z|7||0");
  });
});

describe("buildChanged (is the served console build this page's?)", () => {
  const dist = (entry: string | null, index_hash = "h") => ({ built_at: null, index_hash, entry });

  it("compares the page's own entry script with the one index.html now names", () => {
    expect(buildChanged("/static/dist/assets/index-A.js", dist("/static/dist/assets/index-A.js"), null)).toBe(false);
    expect(buildChanged("/static/dist/assets/index-A.js", dist("/static/dist/assets/index-B.js"), null)).toBe(true);
  });

  it("catches a rebuild that happened before the first version answer (first-seen hash is the new build)", () => {
    expect(buildChanged("/static/dist/assets/index-A.js", dist("/static/dist/assets/index-B.js", "hB"), "hB")).toBe(true);
  });

  it("falls back to the first index_hash seen when an entry is unknown", () => {
    expect(buildChanged(null, dist(null, "h2"), "h1")).toBe(true);
    expect(buildChanged("/x.js", { built_at: null, index_hash: "h1" }, "h1")).toBe(false);
    expect(buildChanged(null, undefined, "h1")).toBe(false);
  });
});

describe("documentEntry", () => {
  it("reads the module entry script of the document", () => {
    const doc = document.implementation.createHTMLDocument("t");
    doc.head.innerHTML = '<script>1</script><script type="module" crossorigin src="/static/dist/assets/index-Q.js"></script>';
    expect(documentEntry(doc)).toBe("/static/dist/assets/index-Q.js");
    expect(documentEntry(document.implementation.createHTMLDocument("none"))).toBeNull();
  });
});
