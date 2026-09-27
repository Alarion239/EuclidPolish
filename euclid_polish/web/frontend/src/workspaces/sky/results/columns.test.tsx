/* The Real results columns: with the inspector open at 1024×768 the table is
 * about 340 px wide, and the key columns (select, Tile, RA / Dec, Production)
 * must fit it without a sideways scroll; everything else drops by priority. */
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { estimateWidths, fitColumns } from "../../../ui/tableModel";
import type { TileRow } from "./api";
import { INSPECTOR_TABLE_PX, REAL_TILE_COLUMNS, SELECT_COL_PX } from "./columns";

const ROW = {
  ref: "nexus/f200w-0214", id: "f200w-0214", label: "NEXUS f200w-0214", source: "nexus",
  ra: 268.377215, dec: 65.098512, has_jwst: true, production_state: "current", models: {},
} as unknown as TileRow;

const col = (id: string) => REAL_TILE_COLUMNS.find((c) => c.id === id)!;

describe("Real results columns", () => {
  it("fit the key columns into the inspector-open table (about 340 px)", () => {
    const w = estimateWidths([ROW], REAL_TILE_COLUMNS);
    const key = SELECT_COL_PX + w.id + w.ra + w.production;
    expect(key).toBeLessThanOrEqual(INSPECTOR_TABLE_PX);
    // none of the three may declare a CSS floor that could widen it again
    for (const id of ["id", "ra", "production"]) expect(col(id).minWidth).toBeUndefined();
  });

  it("drop every other column at that width, Source first", () => {
    const shown = REAL_TILE_COLUMNS.filter((c) => !c.hidden);
    const w = estimateWidths([ROW], REAL_TILE_COLUMNS);
    const fit = [{ id: "\u0000select", width: SELECT_COL_PX }, ...shown.map((c) => ({ id: c.id, width: w[c.id], priority: c.priority }))];
    const dropped = fitColumns(fit, INSPECTOR_TABLE_PX);
    expect(dropped[0]).toBe("source");
    expect(shown.map((c) => c.id).filter((id) => !dropped.includes(id))).toEqual(["id", "ra", "production"]);
    expect(["id", "ra", "production"].map((id) => col(id).priority)).toEqual([undefined, undefined, undefined]);
  });

  it("stack RA over Dec (the whole value, two tabular lines) and the JWST badge under the tile id", () => {
    const { container } = render(<table><tbody><tr>
      <td>{col("id").cell!(ROW, 0)}</td><td>{col("ra").cell!(ROW, 0)}</td>
    </tr></tbody></table>);
    const lines = [...container.querySelectorAll(".res-radec > span")].map((s) => s.textContent);
    expect(lines).toEqual(["268.3772°", "+65.0985°"]);
    const tile = container.querySelector(".res-tilecell")!;
    expect(tile.children[0].textContent).toBe("f200w-0214");
    expect(tile.children[1].textContent).toBe("JWST");
  });
});
