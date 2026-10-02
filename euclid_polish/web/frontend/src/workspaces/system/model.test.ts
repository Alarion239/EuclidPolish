import { describe, expect, it } from "vitest";
import { DEFAULT_DISPLAY } from "../../state/display";
import {
  basename, codeSentence, diskCaption, diskThresholdText, duBytes, experimentsLine, gitStatusText, groupHome, imagesLine, lineageCallout, lineageVerdicts, stageOfKind, verdictKindFilter,
  mergeCommitPages, relationText,
} from "./model";

describe("git", () => {
  it("describes the FASRC-vs-local relation", () => {
    expect(relationText({ relation: "same" }).tone).toBe("good");
    expect(relationText({ relation: "remote_behind", ahead: 1 }).label).toBe("FASRC 1 commit behind");
    expect(relationText({ relation: "remote_ahead", behind: 3 }).label).toBe("FASRC 3 commits ahead");
    expect(relationText(null).label).toBe("unknown");
  });
  it("names porcelain states in words", () => {
    expect(gitStatusText("??")).toBe("untracked");
    expect(gitStatusText(" M")).toBe("modified");
    expect(gitStatusText("R ")).toBe("renamed");
    expect(gitStatusText("UU")).toBe("conflict");
  });
  it("appends a history page and drops commits already loaded", () => {
    const c = (h: string) => ({ hash: h, full: h + "full", author: "a", relative: "now", subject: h });
    expect(mergeCommitPages([c("a"), c("b")], [c("b"), c("c")]).map((x) => x.hash)).toEqual(["a", "b", "c"]);
  });
  it("splits paths", () => {
    expect(basename("/n/data/psf.fits")).toBe("psf.fits");
    expect(basename("/n/data/")).toBe("data");
  });
});

describe("one question: are laptop, server and FASRC on the same commit?", () => {
  it("says so in one sentence when they are", () => {
    expect(codeSentence({ laptop: "3e8b270", server: "3e8b270", fasrc: { head: "3e8b270aaaa", relation: "same" } }))
      .toEqual({ text: "Laptop, server and FASRC are on 3e8b270.", tone: "good" });
  });
  it("names the one that differs", () => {
    expect(codeSentence({ laptop: "2222222", server: "1111111", fasrc: { head: "2222222ffff", relation: "same" } }))
      .toEqual({ text: "Laptop and FASRC are on 2222222; the server started at 1111111.", tone: "warn" });
    expect(codeSentence({ laptop: "2222222", server: "2222222", fasrc: { head: "1111111ffff", relation: "remote_behind", ahead: 3 } }))
      .toEqual({ text: "Laptop and server are on 2222222; FASRC is 3 commits behind (1111111).", tone: "warn" });
    expect(codeSentence({ laptop: "2222222", server: "1111111", fasrc: { head: "3333333ffff", relation: "diverged", ahead: 1, behind: 2 } }).text)
      .toBe("This laptop is on 2222222, the server started at 1111111, FASRC is on 3333333 (diverged).");
  });
  it("says when FASRC is not connected or not read yet", () => {
    expect(codeSentence({ laptop: "2222222", server: "2222222", fasrc: null }).text)
      .toBe("Laptop and server are on 2222222; FASRC is not connected.");
    expect(codeSentence({ laptop: "2222222", server: "2222222", fasrc: "loading" }).text)
      .toBe("Laptop and server are on 2222222; reading the FASRC checkout…");
    expect(codeSentence({ laptop: null, server: null, fasrc: null }).text).toBe("Reading the commits…");
  });
});

describe("storage", () => {
  it("parses du sizes for sorting", () => {
    expect(duBytes("12G")).toBe(12 * 1024 ** 3);
    expect(duBytes("1.5T")).toBe(1.5 * 1024 ** 4);
    expect(duBytes("512K")).toBe(512 * 1024);
    expect(duBytes("0")).toBe(0);
    expect(duBytes("?")).toBeNull();
  });
  it("says how much of the used disk is the console's", () => {
    const GiB = 1024 ** 3;
    expect(diskCaption({ used_bytes: 427 * GiB }, 68 * GiB)).toBe("68 GiB of 427 GiB used is ours");
    expect(diskCaption({ used_bytes: 427 * GiB }, null)).toBe("427 GiB used; measure the data roots to see how much is ours");
    expect(diskThresholdText({ warn_below_bytes: 25 * GiB, bad_below_bytes: 10 * GiB, warn_used_fraction: 0.95 }))
      .toBe("Warns below 25 GiB free or at 95% used (here and on Home); experiments stop below 10 GiB.");
  });
  it("experiments line: nothing cached reads as words, not 0 B rows", () => {
    const GiB = 1024 ** 3;
    const e = { cache_budget_bytes: 4 * GiB, min_free_bytes: 5 * GiB, cache_bytes: 0, outputs_bytes: 0 };
    expect(experimentsLine(e)).toBe("Experiments: nothing cached; 5 GiB kept free.");
    expect(experimentsLine({ ...e, cache_bytes: null, outputs_bytes: null })).toBe("Experiments: nothing cached; 5 GiB kept free.");
    expect(experimentsLine({ ...e, cache_bytes: 1.5 * GiB, outputs_bytes: 0 }))
      .toBe("Experiments: member-SR cache 1.5 GiB of its 4 GiB budget; 5 GiB kept free.");
    expect(experimentsLine({ ...e, cache_bytes: 0, outputs_bytes: 2 * GiB }))
      .toBe("Experiments: outputs 2 GiB; 5 GiB kept free.");
  });
});

describe("config groups link to the tab that judges them", () => {
  it("maps each group to its tab", () => {
    expect(groupHome("scenes")).toEqual({ label: "Synthetic › Records", to: "/synthetic/records" });
    expect(groupHome("psf")?.to).toBe("/synthetic/psf");
    expect(groupHome("cutouts")?.to).toBe("/synthetic/psf");
    expect(groupHome("stars")?.to).toBe("/synthetic/stars");
    expect(groupHome("lr")).toEqual({ label: "Models › Train", to: "/models/train" });
    expect(groupHome("plateau")?.to).toBe("/models/train");
    expect(groupHome("other")).toBeNull();
  });
});

describe("appearance", () => {
  it("sums the live image settings up in one line", () => {
    expect(imagesLine(DEFAULT_DISPLAY)).toBe("VIS, absolute asinh, knee 100 e⁻");
    expect(imagesLine({ ...DEFAULT_DISPLAY, color: "lupton", invert: true,
      groups: { ...DEFAULT_DISPLAY.groups, default: { knee: 250, gain: 2, black: 0 } } }))
      .toMatch(/, knee 250 e⁻ ×2, inverted$/);
  });
});

describe("lineage", () => {
  it("says once that most records carry no model id, so each takes its stage's verdict", () => {
    expect(lineageCallout({ total: 11345, counts: { verdicts: { current: 0, stale: 0, unknown: 11094 } } }))
      .toBe("98% of records carry no model id, so each record takes its Loop stage's verdict.");
    expect(lineageCallout({ total: 10, counts: { verdicts: { current: 8, stale: 1, unknown: 1 } } })).toBeNull();
    expect(lineageCallout(null)).toBeNull();
  });
  const stages = [
    { id: "records", label: "Records", state: "stale", reason: "predate the stellar prior", to: "/synthetic/records" },
    { id: "members", label: "Members", state: "current", reason: "30 active", to: "/models/members" },
    { id: "real-sr", label: "Real SR", state: "stale", reason: "449 stale", to: "/sky/targets" },
  ] as const;
  const kinds = { checkpointartifact: 42, generationrun: 209, inferencerun: 6430, srcutoutartifact: 4664, oddkind: 3 };
  it("each record kind takes the verdict of its Loop stage (the staleness service Home reads)", () => {
    expect(stageOfKind("generationrun", stages)?.id).toBe("records");
    expect(stageOfKind("checkpointartifact", stages)?.state).toBe("current");
    expect(stageOfKind("srcutoutartifact", stages)?.label).toBe("Real SR");
    expect(stageOfKind("oddkind", stages)).toBeNull();
    expect(stageOfKind("generationrun", null)).toBeNull();
  });
  it("counts records per Loop verdict and names the kinds a verdict filter selects", () => {
    const v = lineageVerdicts(kinds, stages);
    expect(v.counts).toEqual({ current: 42, stale: 209 + 6430 + 4664, blocked: 0, unknown: 0 });
    expect(v.kinds.stale).toEqual(["generationrun", "inferencerun", "srcutoutartifact"]);
    expect(v.kinds.current).toEqual(["checkpointartifact"]);
    expect(v.kinds.blocked).toEqual([]);
    // A stage still checking ("loading") gives no verdict yet.
    const loading = lineageVerdicts(kinds, [{ ...stages[0], state: "loading" }]);
    expect(loading.counts.stale).toBe(0);
    expect(lineageVerdicts(kinds, null).counts).toEqual({ current: 0, stale: 0, blocked: 0, unknown: 0 });
  });
  it("the kind filter a verdict narrows to: the verdict's kinds, intersected with a chosen kind", () => {
    const v = lineageVerdicts(kinds, stages);
    expect(verdictKindFilter("", "", v)).toBe("");
    expect(verdictKindFilter("generationrun", "", v)).toBe("generationrun");
    expect(verdictKindFilter("", "stale", v)).toBe("generationrun,inferencerun,srcutoutartifact");
    expect(verdictKindFilter("inferencerun", "stale", v)).toBe("inferencerun");
    // Nothing has that verdict (or the chosen kind lacks it): a filter that matches no record.
    expect(verdictKindFilter("", "blocked", v)).toBeNull();
    expect(verdictKindFilter("checkpointartifact", "stale", v)).toBeNull();
  });
});
