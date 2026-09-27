/* Guard (spec 2026-09-27, "Statistics: readable, informative, nothing
 * useless"): no page presents statistics as cards. The Stat, StatStrip and
 * Kpi tiles and their styles are gone; pages state numbers with SummaryLine,
 * FactsList, Caption, Details (ui/facts.tsx) or a table.
 *
 * Test files are not scanned: they are not pages, and several assert that a
 * page renders none of the old tile classes (`querySelector(".ui-kpi")` must
 * stay null), which names the class without using it. A test that rendered a
 * tile would not compile, since the components no longer exist. */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { expect, it } from "vitest";

const SRC = join(__dirname, "..");

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    return statSync(p).isDirectory() ? files(p) : /\.(tsx?|css)$/.test(n) ? [p] : [];
  });
}

// A tile element, or one of the tile class families (`ui-stat` but not `ui-status`).
const CARD = /<(Stat|StatStrip|Kpi)[\s/>]|\b(ui-stat|ui-kpi|rl-stats|ens-kpis|ops-kpis)(?![a-z])/;

it("no page presents statistics as cards", () => {
  const offenders = files(SRC)
    .filter((p) => !/\.test\.tsx?$/.test(p))
    .filter((p) => CARD.test(readFileSync(p, "utf8")))
    .map((p) => p.slice(SRC.length + 1));
  expect(offenders).toEqual([]);
});

it("the kit no longer exports the tiles", async () => {
  const kit: Record<string, unknown> = await import("./index");
  expect(kit.Stat).toBeUndefined();
  expect(kit.Kpi).toBeUndefined();
  expect(kit.FactsList).toBeTypeOf("function");
  expect(kit.SummaryLine).toBeTypeOf("function");
});
