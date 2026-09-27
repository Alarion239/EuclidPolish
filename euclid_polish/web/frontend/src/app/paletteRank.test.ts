import { describe, expect, it } from "vitest";
import { rankPalette, scoreEntry, type RankGroup } from "./paletteRank";

type E = { key: string; label: string; hint?: string; keywords?: string[] };

const PAGES: E[] = [
  { key: "page:/ops/git", label: "Ops › Git", keywords: ["ops", "git", "/ops/git"] },
  { key: "page:/realism/noise", label: "Realism › Noise", hint: "Sky-noise levels", keywords: ["realism", "noise"] },
  { key: "page:/ensemble/starfull/members", label: "Ensemble (starfull) › Members", keywords: ["ensemble", "members"] },
  { key: "page:/ensemble/starfull/knee", label: "Ensemble (starfull) › Knee PSNR", keywords: ["ensemble", "knee"] },
  { key: "page:/ensemble/starfull/overview", label: "Ensemble (starfull) › Overview", keywords: ["ensemble", "overview"] },
];
const COMMANDS: E[] = [
  { key: "cmd:theme", label: "Toggle light / dark theme" },
  { key: "cmd:theme-light", label: "Theme: light", keywords: ["appearance"] },
  { key: "cmd:display", label: "Open the Display panel", keywords: ["colour", "stretch", "knee"] },
];
const PAGE_ACTIONS: E[] = [
  { key: "action:continue", label: "Continue the selected members" },
  { key: "action:archive", label: "Archive the selected members…" },
];
const RUN: E[] = [
  { key: "run:fit-gate", label: "Fit a spatial-gate variant…", keywords: ["run", "job", "combiner", "gate", "fit"] },
  { key: "run:disk", label: "Measure disk usage per data root", keywords: ["run", "job", "disk", "storage"] },
  { key: "run:train", label: "Train or continue members on FASRC…", keywords: ["run", "job", "train", "members", "slurm"] },
];

// As the palette passes them: the page's own actions, then pages and
// commands (favoured on a tie), then the global "Run a job" group.
const groups = (): RankGroup<E>[] => [
  { heading: "Members", items: PAGE_ACTIONS },
  { heading: "Pages", items: PAGES, bias: 2 },
  { heading: "Commands", items: COMMANDS, bias: 1 },
  { heading: "Run a job", items: RUN, bias: -1 },
];
const first = (q: string) => rankPalette(q, groups())[0]?.items[0]?.key;
const order = (q: string) => rankPalette(q, groups()).flatMap((g) => g.items.map((i) => i.key));

describe("scoreEntry", () => {
  it("ranks exact > prefix > word start > substring > keyword > hint > fuzzy, and 0 for no match", () => {
    const s = (q: string, e: Partial<E>) => scoreEntry(q, { label: "", ...e });
    const exact = s("git", { label: "Git" });
    const prefix = s("git", { label: "Gitlab" });
    const word = s("git", { label: "Ops › Git" });
    const sub = s("git", { label: "Legitimate" });
    const kw = s("git", { label: "Version control", keywords: ["git"] });
    const hint = s("git", { label: "Version control", hint: "uses git" });
    const fuzzy = s("git", { label: "Fit a spatial-gate variant" });
    expect(exact).toBeGreaterThan(prefix);
    expect(prefix).toBeGreaterThan(word);
    expect(word).toBeGreaterThan(sub);
    expect(sub).toBeGreaterThan(hint);
    expect(kw).toBeGreaterThan(hint);
    expect(hint).toBeGreaterThan(fuzzy);
    expect(fuzzy).toBeGreaterThan(0);
    expect(s("git", { label: "Home" })).toBe(0);
  });

  it("is case- and space-insensitive and matches every word of a multi-word query", () => {
    expect(scoreEntry("  KNEE   psnr ", { label: "Ensemble › Knee PSNR" })).toBeGreaterThan(0);
    expect(scoreEntry("psnr knee", { label: "Ensemble › Knee PSNR" })).toBeGreaterThan(0);
    expect(scoreEntry("psnr zebra", { label: "Ensemble › Knee PSNR" })).toBe(0);
  });

  it("does not fuzzy-match short queries or scattered letters", () => {
    expect(scoreEntry("gt", { label: "Fit a spatial-gate variant" })).toBe(0);
    expect(scoreEntry("noise", { label: "Measure disk usage per data root" })).toBe(0);
  });
});

describe("rankPalette", () => {
  it("keeps the given groups and order when nothing is typed", () => {
    expect(rankPalette("", groups()).map((g) => g.heading)).toEqual(["Members", "Pages", "Commands", "Run a job"]);
    expect(rankPalette("  ", groups())[1].items).toEqual(PAGES);
  });

  it("puts the page the user named first, not a job that happens to contain the letters", () => {
    expect(first("git")).toBe("page:/ops/git");
    expect(first("noise")).toBe("page:/realism/noise");
    expect(order("git")).not.toContain("run:disk");
  });

  it("puts the theme commands above page actions for 'theme'", () => {
    expect(first("theme")).toBe("cmd:theme-light");
    const o = order("theme");
    expect(o.indexOf("cmd:theme")).toBeLessThan(o.indexOf("action:continue") === -1 ? Infinity : o.indexOf("action:continue"));
  });

  it("orders groups by their best match and items by score within a group", () => {
    const r = rankPalette("members", groups());
    expect(r[0].heading).toBe("Pages");
    expect(r[0].items[0].key).toBe("page:/ensemble/starfull/members");
    expect(r.map((g) => g.heading)).toContain("Members");
  });

  it("drops groups with no match", () => {
    const r = rankPalette("overview", groups());
    expect(r.map((g) => g.heading)).toEqual(["Pages"]);
    expect(r[0].items.map((i) => i.key)).toEqual(["page:/ensemble/starfull/overview"]);
  });

  it("returns nothing when nothing matches", () => {
    expect(rankPalette("zzqx", groups())).toEqual([]);
  });

  it("breaks ties by the given group order", () => {
    const tie: RankGroup<E>[] = [
      { heading: "A", items: [{ key: "a", label: "Knee" }] },
      { heading: "B", items: [{ key: "b", label: "Knee" }] },
    ];
    expect(rankPalette("knee", tie).map((g) => g.heading)).toEqual(["A", "B"]);
  });
});
