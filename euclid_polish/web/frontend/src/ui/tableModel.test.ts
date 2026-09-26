import { describe, expect, it } from "vitest";
import {
  cellText, compareValues, csvCell, estimateWidths, filterRows, nextSort, parseFilter, parseSort,
  rangeKeys, serializeSort, sortRows, toCSV, type DataColumn,
} from "./tableModel";

type Row = { name: string; psnr: number | null; loss: string; steps?: number; ok?: boolean };
const ROWS: Row[] = [
  { name: "member_10", psnr: 43.2, loss: "l1", steps: 200_000, ok: true },
  { name: "member_2", psnr: 44.1, loss: "l2", steps: 150_000, ok: false },
  { name: "member_196", psnr: null, loss: "l1", steps: 90_000 },
  { name: "member_3", psnr: 42.0, loss: "l3", steps: 200_000, ok: true },
];
const COLS: DataColumn<Row>[] = [
  { id: "name", header: "Member" },
  { id: "psnr", header: "PSNR", numeric: true },
  { id: "loss", header: "Loss" },
  { id: "steps", header: "Steps", accessor: (r) => r.steps ?? null },
];
const names = (rows: Row[]) => rows.map((r) => r.name);

describe("compareValues", () => {
  it("orders numbers numerically and strings naturally", () => {
    expect(compareValues(2, 10)).toBeLessThan(0);
    expect(compareValues("member_2", "member_10")).toBeLessThan(0);
    expect(compareValues("B", "a")).toBeGreaterThan(0);          // case-insensitive
    expect(compareValues(false, true)).toBeLessThan(0);
    expect(compareValues(new Date(0), new Date(5))).toBeLessThan(0);
  });
  it("puts numbers before strings of mixed columns", () => {
    expect(compareValues(5, "x")).toBeLessThan(0);
  });
});

describe("sortRows", () => {
  it("sorts ascending and descending, keeping empty values last both ways", () => {
    expect(names(sortRows(ROWS, COLS, [{ id: "psnr", desc: false }])))
      .toEqual(["member_3", "member_10", "member_2", "member_196"]);
    expect(names(sortRows(ROWS, COLS, [{ id: "psnr", desc: true }])))
      .toEqual(["member_2", "member_10", "member_3", "member_196"]);
  });
  it("sorts names naturally (member_2 before member_10)", () => {
    expect(names(sortRows(ROWS, COLS, [{ id: "name", desc: false }])))
      .toEqual(["member_2", "member_3", "member_10", "member_196"]);
  });
  it("multi-sorts by the second key within ties of the first, and is stable", () => {
    const out = sortRows(ROWS, COLS, [{ id: "steps", desc: true }, { id: "name", desc: false }]);
    expect(names(out)).toEqual(["member_3", "member_10", "member_2", "member_196"]);
    // no sort → original order, new array
    const same = sortRows(ROWS, COLS, []);
    expect(same).toEqual(ROWS);
    expect(same).not.toBe(ROWS);
  });
  it("uses a column's sortFn when given and ignores unknown ids", () => {
    const cols: DataColumn<Row>[] = [...COLS, { id: "len", header: "len", sortFn: (a, b) => a.name.length - b.name.length }];
    expect(names(sortRows(ROWS, cols, [{ id: "len", desc: false }]))[3]).toBe("member_196");
    expect(names(sortRows(ROWS, cols, [{ id: "nope", desc: false }]))).toEqual(names(ROWS));
  });
});

describe("nextSort", () => {
  it("cycles a single column asc → desc → none", () => {
    let s = nextSort([], "psnr", false);
    expect(s).toEqual([{ id: "psnr", desc: false }]);
    s = nextSort(s, "psnr", false);
    expect(s).toEqual([{ id: "psnr", desc: true }]);
    expect(nextSort(s, "psnr", false)).toEqual([]);
  });
  it("replaces the sort on a plain click and appends on a multi click", () => {
    const s = nextSort([{ id: "psnr", desc: true }], "name", false);
    expect(s).toEqual([{ id: "name", desc: false }]);
    const m = nextSort([{ id: "psnr", desc: true }], "name", true);
    expect(m).toEqual([{ id: "psnr", desc: true }, { id: "name", desc: false }]);
    expect(nextSort(m, "name", true)).toEqual([{ id: "psnr", desc: true }, { id: "name", desc: true }]);
    expect(nextSort(nextSort(m, "name", true), "name", true)).toEqual([{ id: "psnr", desc: true }]);
  });
  it("round-trips through the URL form", () => {
    const s = [{ id: "psnr", desc: true }, { id: "name", desc: false }];
    expect(serializeSort(s)).toBe("-psnr,name");
    expect(parseSort("-psnr,name")).toEqual(s);
    expect(parseSort("")).toEqual([]);
    expect(parseSort(" , -x ,")).toEqual([{ id: "x", desc: true }]);
  });
});

describe("parseFilter / filterRows", () => {
  it("parses plain, quoted, negated, column and comparison tokens", () => {
    expect(parseFilter('l1 "member 1" -bad psnr>43 loss:l2 steps<=1e5')).toEqual([
      { value: "l1", negate: false },
      { value: "member 1", negate: false },
      { value: "bad", negate: true },
      { column: "psnr", op: ">", value: "43", negate: false },
      { column: "loss", op: ":", value: "l2", negate: false },
      { column: "steps", op: "<=", value: "1e5", negate: false },
    ]);
    expect(parseFilter("   ")).toEqual([]);
  });
  it("matches plain text case-insensitively across columns (AND of tokens)", () => {
    expect(names(filterRows(ROWS, COLS, "L1"))).toEqual(["member_10", "member_196"]);
    expect(names(filterRows(ROWS, COLS, "l1 196"))).toEqual(["member_196"]);
    expect(filterRows(ROWS, COLS, "")).toEqual(ROWS);
  });
  it("scopes col:value to a column (by id or header) and negates with -", () => {
    expect(names(filterRows(ROWS, COLS, "loss:l2"))).toEqual(["member_2"]);
    expect(names(filterRows(ROWS, COLS, "Member:196"))).toEqual(["member_196"]);
    expect(names(filterRows(ROWS, COLS, "-loss:l1"))).toEqual(["member_2", "member_3"]);
  });
  it("compares numerically, excluding rows without a number", () => {
    expect(names(filterRows(ROWS, COLS, "psnr>43"))).toEqual(["member_10", "member_2"]);
    expect(names(filterRows(ROWS, COLS, "psnr<=42"))).toEqual(["member_3"]);
    expect(names(filterRows(ROWS, COLS, "steps=2e5"))).toEqual(["member_10", "member_3"]);
    expect(names(filterRows(ROWS, COLS, "psnr>abc"))).toEqual([]);
  });
  it("reads a leading - before a number as a negative value, not a negation (! still negates)", () => {
    expect(parseFilter("-0.3 -5 -1e3 -.5 -5abc")).toEqual([
      { value: "-0.3", negate: false },
      { value: "-5", negate: false },
      { value: "-1e3", negate: false },
      { value: "-.5", negate: false },
      { value: "5abc", negate: true },
    ]);
    const rows = [{ d: -0.3 }, { d: 0.3 }, { d: 1 }];
    const cols: DataColumn<{ d: number }>[] = [{ id: "d", header: "Δ" }];
    expect(filterRows(rows, cols, "-0.3")).toEqual([{ d: -0.3 }]);
    expect(filterRows(rows, cols, "!0.3")).toEqual([{ d: 1 }]);
    expect(filterRows(rows, cols, "d<-0.1")).toEqual([{ d: -0.3 }]);
  });
  it("treats an unknown column prefix as plain text", () => {
    const rows = [{ name: "a:b", psnr: 1, loss: "x" }, { name: "c", psnr: 2, loss: "y" }];
    expect(filterRows(rows, COLS, "a:b").map((r) => r.name)).toEqual(["a:b"]);
  });
  it("skips columns marked filterable: false and uses filterText when given", () => {
    const cols: DataColumn<Row>[] = [
      { id: "name", header: "Member", filterable: false },
      { id: "loss", header: "Loss", filterText: (r) => `loss-${r.loss}` },
    ];
    expect(filterRows(ROWS, cols, "member")).toEqual([]);
    expect(names(filterRows(ROWS, cols, "loss-l3"))).toEqual(["member_3"]);
  });
});

describe("cellText", () => {
  it("renders null as empty, numbers raw, booleans as yes/no", () => {
    expect(cellText(COLS[1], ROWS[2])).toBe("");
    expect(cellText(COLS[1], ROWS[0])).toBe("43.2");
    expect(cellText({ id: "ok", header: "ok" }, ROWS[1])).toBe("no");
  });
});

describe("toCSV", () => {
  it("writes a header of plain-text column names and RFC-4180 quoted cells", () => {
    const rows = [{ name: 'a,"b"', psnr: 1.5, loss: "x\ny" }];
    const csv = toCSV(rows, COLS.slice(0, 3));
    expect(csv).toBe('Member,PSNR,Loss\r\n"a,""b""",1.5,"x\ny"\r\n');
  });
  it("uses headerText for non-string headers, csv() overrides and drops csv:false columns", () => {
    const cols: DataColumn<Row>[] = [
      { id: "name", header: { toString: () => "x" } as never, headerText: "Name" },
      { id: "psnr", header: "PSNR", csv: (r) => (r.psnr == null ? "" : r.psnr.toFixed(2)) },
      { id: "loss", header: "Loss", csv: false },
    ];
    expect(toCSV(ROWS.slice(0, 1), cols)).toBe("Name,PSNR\r\nmember_10,43.20\r\n");
  });
  it("defuses spreadsheet formulas in text cells but keeps negative numbers", () => {
    const rows = [{ name: "=HYPERLINK(1)", psnr: -3, loss: "@x" }];
    expect(toCSV(rows, COLS.slice(0, 3))).toBe("Member,PSNR,Loss\r\n'=HYPERLINK(1),-3,'@x\r\n");
  });
  it("exports the shared csvCell encoder (also used by the Plot CSV export)", () => {
    expect(csvCell(null)).toBe("");
    expect(csvCell(Number.NaN)).toBe("");
    expect(csvCell(-2.5)).toBe("-2.5");
    expect(csvCell("-2.5")).toBe("-2.5");
    expect(csvCell("+cmd")).toBe("'+cmd");
    expect(csvCell('a "b"')).toBe('"a ""b"""');
  });
});

describe("rangeKeys", () => {
  const order = ["a", "b", "c", "d", "e"];
  it("returns the inclusive span between anchor and target in either direction", () => {
    expect(rangeKeys(order, "b", "d")).toEqual(["b", "c", "d"]);
    expect(rangeKeys(order, "d", "b")).toEqual(["b", "c", "d"]);
    expect(rangeKeys(order, "c", "c")).toEqual(["c"]);
  });
  it("falls back to the target alone when the anchor is not in view", () => {
    expect(rangeKeys(order, "zz", "c")).toEqual(["c"]);
    expect(rangeKeys(order, null, "c")).toEqual(["c"]);
  });
});

describe("estimateWidths", () => {
  it("sizes from header and content length, clamped, and honours explicit widths", () => {
    const w = estimateWidths(ROWS, [...COLS, { id: "fixed", header: "F", width: 222 }]);
    expect(w.name).toBeGreaterThan(w.loss);
    expect(w.fixed).toBe(222);
    for (const v of Object.values(w)) {
      expect(v).toBeGreaterThanOrEqual(56);
      expect(v).toBeLessThanOrEqual(420);
    }
  });
});
