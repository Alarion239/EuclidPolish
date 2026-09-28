/* The Sky › Targets columns: with the tile card open at 1024×768 the table is
 * about 340 px wide, and the key columns (select, Target, State, Flux SR/LR)
 * must fit it without a sideways scroll; everything else drops by priority,
 * RA / Dec first. Holes and R̃ exist only when a row in view is scored. */
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { estimateWidths, fitColumns } from "../../../ui/tableModel";
import { targetColumns } from "./columns";
import type { TargetRow } from "./model";

/** The table's client width with the tile card open at 1024×768, and the checkbox column. */
const INSPECTOR_TABLE_PX = 340;
const SELECT_COL_PX = 36;

const ROW: TargetRow = {
  key: "eval/102018666_NEG579859279508531437", set: "lenses", id: "102018666_NEG579859279508531437", label: "102018666_NEG579859279508531437 · A",
  ra: 57.985928, dec: -50.853144, field: "EDF-S", state: "stale", reason: "membership changed", madeBy: "22 members",
  flux: 0.67, holes: null, medR: null, scored: false, grade: "A", hasJwst: false, ref: "eval/102018666_NEG579859279508531437", nexusField: null,
};
const KEY = ["id", "state", "flux"];

describe("Targets columns", () => {
  it("fit the key columns into the table the tile card leaves (about 340 px)", () => {
    const cols = targetColumns({ scored: true, manySets: true, lenses: true });
    const w = estimateWidths([ROW], cols);
    expect(SELECT_COL_PX + KEY.reduce((n, id) => n + w[id], 0)).toBeLessThanOrEqual(INSPECTOR_TABLE_PX);
    for (const id of KEY) expect(cols.find((c) => c.id === id)!.minWidth).toBeUndefined();
  });

  it("drop every other column at that width, RA / Dec first", () => {
    const cols = targetColumns({ scored: true, manySets: true, lenses: true });
    const shown = cols.filter((c) => !c.hidden);
    const w = estimateWidths([ROW], cols);
    const fit = [{ id: "\u0000select", width: SELECT_COL_PX }, ...shown.map((c) => ({ id: c.id, width: w[c.id], priority: c.priority }))];
    const dropped = fitColumns(fit, INSPECTOR_TABLE_PX);
    expect(dropped[0]).toBe("ra");
    expect(shown.map((c) => c.id).filter((id) => !dropped.includes(id))).toEqual(KEY);
  });

  it("show Holes and R̃ only when a scored row is in view; the set column only over several sets", () => {
    const ids = (o: Parameters<typeof targetColumns>[0]) => targetColumns(o).filter((c) => !c.hidden).map((c) => c.id);
    expect(ids({ scored: false, manySets: false, lenses: false })).toEqual(["id", "state", "madeBy", "flux", "field", "ra"]);
    expect(ids({ scored: true, manySets: true, lenses: true })).toEqual(["id", "set", "state", "madeBy", "flux", "holes", "medR", "grade", "field", "ra"]);
  });

  it("words the state (a badge only on a problem) and warns a flux more than 0.1 mag off", () => {
    const cols = targetColumns({ scored: false, manySets: false, lenses: false });
    const cell = (id: string, row: TargetRow) => cols.find((c) => c.id === id)!.cell!(row, 0);
    const { container } = render(<table><tbody><tr>
      <td>{cell("state", ROW)}</td><td>{cell("flux", ROW)}</td><td>{cell("state", { ...ROW, state: "current", reason: null })}</td>
      <td>{cell("flux", { ...ROW, flux: 0.97 })}</td>
    </tr></tbody></table>);
    const tds = container.querySelectorAll("td");
    expect(tds[0].textContent).toBe("stale");
    expect(tds[0].querySelector(".ui-badge")).toBeTruthy();
    expect(tds[1].querySelector(".tg-warn")?.textContent).toBe("0.67");
    expect(tds[2].textContent).toBe("current");
    expect(tds[2].querySelector(".ui-badge")).toBeNull();
    expect(tds[3].querySelector(".tg-warn")).toBeNull();
  });
});
