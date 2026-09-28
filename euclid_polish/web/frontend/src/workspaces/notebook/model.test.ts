import { describe, expect, it } from "vitest";
import { backupCounts, commitText, notebookEntryUrl, notebookDays, notebookOrder, parseShow, storedPath } from "./model";

describe("tracking notebook", () => {
  const LOG = [
    "# single-model retirement", "preamble",
    "## 2026-07-02T02:37:41Z", "first", "## 2026-07-02T19:58:51Z", "second",
    "## 2026-09-21T01:36:49Z", "third", "## Decisions", "not dated", "## 2026-09-21T14:42:29Z", "fourth",
  ].join("\n");
  it("reverses the entries for newest first, keeping the preamble on top", () => {
    const shown = notebookOrder(LOG, true);
    expect(shown.split("\n").filter((l) => l.startsWith("## "))).toEqual([
      "## 2026-09-21T14:42:29Z", "## Decisions", "## 2026-09-21T01:36:49Z", "## 2026-07-02T19:58:51Z", "## 2026-07-02T02:37:41Z"]);
    expect(shown.startsWith("# single-model retirement\npreamble")).toBe(true);
    expect(notebookOrder(LOG, false)).toBe(LOG);
  });
  it("groups the dated entries by UTC day in the displayed order, jumping to the day's first shown entry", () => {
    const heads = [
      { level: 2, text: "2026-09-21T14:42:29Z", id: "a" }, { level: 2, text: "Decisions", id: "d" },
      { level: 2, text: "2026-09-21T01:36:49Z", id: "b" }, { level: 2, text: "2026-07-02T19:58:51Z", id: "c" },
      { level: 2, text: "2026-07-02T02:37:41Z", id: "e" }, { level: 1, text: "2026-01-01T00:00:00Z", id: "h1" },
    ];
    expect(notebookDays(heads)).toEqual([
      { day: "2026-09-21", label: "Sep 21", month: "September 2026", count: 2, id: "a" },
      { day: "2026-07-02", label: "Jul 2", month: "July 2026", count: 2, id: "c" },
    ]);
    expect(notebookDays([])).toEqual([]);
  });
});

describe("backups filter", () => {
  const b = { models: [{ name: "m" }], fits: [{ name: "a" }, { name: "b" }], images: [] };
  it("counts each kind and the archived campaigns", () => {
    expect(backupCounts(b, 3)).toEqual({ models: 1, fits: 2, images: 0, campaigns: 3 });
    expect(backupCounts(null, 0)).toEqual({ models: 0, fits: 0, images: 0, campaigns: 0 });
  });
  it("reads ?show= (and the old ?bk=) as a filter", () => {
    expect(parseShow("campaigns", "")).toBe("campaigns");
    expect(parseShow("", "fits")).toBe("fits");
    expect(parseShow("", "")).toBe("models");
    expect(parseShow("bogus", "images")).toBe("images");
  });
  it("names the stored copy Files opens, and a commit in short", () => {
    expect(storedPath("/Users/x/EuclidPolish/tracking", "current", "fits", "sr.fits")).toBe("tracking/current/fits/sr.fits");
    expect(storedPath(undefined, "old-run-2", "images", "a.png")).toBe("tracking/archive/old-run-2/images/a.png");
    expect(commitText({ short: "abc1234" })).toBe("abc1234");
    expect(commitText({ hash: "abcdef123456" })).toBe("abcdef1");
    expect(commitText(null)).toBe("—");
  });
});

describe("notebookEntryUrl (every 'Log to notebook' button)", () => {
  it("lands on Notebook › Log with the entry and its page", () => {
    expect(notebookEntryUrl("## Gate fit\n- +0.14 dB", "Models › Combiner"))
      .toBe("/notebook/log?entry=%23%23+Gate+fit%0A-+%2B0.14+dB&from=Models+%E2%80%BA+Combiner");
    expect(notebookEntryUrl("x")).toBe("/notebook/log?entry=x");
  });
});
