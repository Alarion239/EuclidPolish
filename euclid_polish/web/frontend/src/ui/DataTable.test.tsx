import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useInspector } from "../state/inspector";
import { DataTable } from "./DataTable";
import type { DataColumn } from "./tableModel";

const downloads = vi.hoisted(() => ({ calls: [] as { name: string; text: string }[] }));
vi.mock("./download", async (orig) => ({
  ...(await orig<typeof import("./download")>()),
  downloadText: (name: string, text: string) => { downloads.calls.push({ name, text }); },
}));

type Row = { name: string; psnr: number | null; loss: string };
const ROWS: Row[] = [
  { name: "member_10", psnr: 43.2, loss: "l1" },
  { name: "member_2", psnr: 44.1, loss: "l2" },
  { name: "member_196", psnr: null, loss: "l1" },
  { name: "member_3", psnr: 42.0, loss: "l3" },
];
const COLS: DataColumn<Row>[] = [
  { id: "name", header: "Member" },
  { id: "psnr", header: "PSNR", numeric: true, cell: (r) => (r.psnr == null ? "—" : r.psnr.toFixed(1)) },
  { id: "loss", header: "Loss" },
];
const key = (r: Row) => r.name;

/** Body row names in DOM order. */
function bodyNames(): string[] {
  const grid = screen.getByRole("grid");
  return within(grid).queryAllByRole("row").filter((r) => r.getAttribute("data-key"))
    .map((r) => r.getAttribute("data-key")!);
}
const header = (name: string) => screen.getByRole("columnheader", { name: new RegExp(name) });
const sortButton = (name: string) => within(header(name)).getByRole("button");

afterEach(() => {
  downloads.calls = [];
  useInspector.getState().reset();
});

describe("DataTable sorting", () => {
  it("sorts on header click asc → desc → original, with aria-sort", () => {
    render(<DataTable rows={ROWS} columns={COLS} rowKey={key} aria-label="members" />);
    expect(bodyNames()).toEqual(["member_10", "member_2", "member_196", "member_3"]);
    fireEvent.click(sortButton("PSNR"));
    expect(bodyNames()).toEqual(["member_3", "member_10", "member_2", "member_196"]);
    expect(header("PSNR").getAttribute("aria-sort")).toBe("ascending");
    fireEvent.click(sortButton("PSNR"));
    expect(bodyNames()).toEqual(["member_2", "member_10", "member_3", "member_196"]);
    expect(header("PSNR").getAttribute("aria-sort")).toBe("descending");
    fireEvent.click(sortButton("PSNR"));
    expect(bodyNames()).toEqual(["member_10", "member_2", "member_196", "member_3"]);
    expect(header("PSNR").getAttribute("aria-sort")).toBe("none");
  });

  it("adds a secondary key on shift-click and reports sort changes", () => {
    const onSortChange = vi.fn();
    render(<DataTable rows={ROWS} columns={COLS} rowKey={key} onSortChange={onSortChange} />);
    fireEvent.click(sortButton("Loss"));
    fireEvent.click(sortButton("PSNR"), { shiftKey: true });
    expect(onSortChange).toHaveBeenLastCalledWith([{ id: "loss", desc: false }, { id: "psnr", desc: false }]);
    expect(bodyNames()).toEqual(["member_10", "member_196", "member_2", "member_3"]);
  });

  it("does not offer sorting on sortable:false columns", () => {
    const cols = COLS.map((c) => (c.id === "loss" ? { ...c, sortable: false } : c));
    render(<DataTable rows={ROWS} columns={cols} rowKey={key} />);
    expect(within(header("Loss")).queryByRole("button")).toBeNull();
  });
});

describe("DataTable filtering", () => {
  it("filters by the search box and shows the visible count", () => {
    render(<DataTable rows={ROWS} columns={COLS} rowKey={key} />);
    const box = screen.getByRole("searchbox", { name: /filter/i });
    fireEvent.change(box, { target: { value: "l1" } });
    expect(bodyNames()).toEqual(["member_10", "member_196"]);
    expect(screen.getByText("2 of 4 rows")).toBeTruthy();
    fireEvent.change(box, { target: { value: "psnr>43" } });
    expect(bodyNames()).toEqual(["member_10", "member_2"]);
  });

  it("shows an empty state when nothing matches and when there are no rows", () => {
    const { rerender } = render(<DataTable rows={ROWS} columns={COLS} rowKey={key} />);
    fireEvent.change(screen.getByRole("searchbox", { name: /filter/i }), { target: { value: "zzz" } });
    expect(screen.getByText(/No rows match/)).toBeTruthy();
    rerender(<DataTable rows={[]} columns={COLS} rowKey={key} empty="no members yet" />);
    expect(screen.getByText("no members yet")).toBeTruthy();
  });

  it("supports a controlled filter", () => {
    const onFilterChange = vi.fn();
    render(<DataTable rows={ROWS} columns={COLS} rowKey={key} filter="l3" onFilterChange={onFilterChange} />);
    expect(bodyNames()).toEqual(["member_3"]);
    fireEvent.change(screen.getByRole("searchbox", { name: /filter/i }), { target: { value: "l2" } });
    expect(onFilterChange).toHaveBeenCalledWith("l2");
  });
});

describe("DataTable selection", () => {
  function setup() {
    const onSelectedChange = vi.fn();
    render(<DataTable rows={ROWS} columns={COLS} rowKey={key} selectable onSelectedChange={onSelectedChange} />);
    const box = (name: string) => screen.getByRole("checkbox", { name: `Select ${name}` });
    return { onSelectedChange, box };
  }

  it("toggles single rows and reports keys and rows", () => {
    const { onSelectedChange, box } = setup();
    fireEvent.click(box("member_2"));
    expect(onSelectedChange).toHaveBeenLastCalledWith(["member_2"], [ROWS[1]]);
    expect((box("member_2") as HTMLInputElement).checked).toBe(true);
    fireEvent.click(box("member_2"));
    expect(onSelectedChange).toHaveBeenLastCalledWith([], []);
  });

  it("selects a range with shift-click in the current (sorted) order", () => {
    const { onSelectedChange, box } = setup();
    fireEvent.click(sortButton("PSNR"));  // member_3, member_10, member_2, member_196
    fireEvent.click(box("member_3"));
    fireEvent.click(box("member_2"), { shiftKey: true });
    expect(onSelectedChange).toHaveBeenLastCalledWith(
      ["member_3", "member_10", "member_2"], [ROWS[3], ROWS[0], ROWS[1]]);
  });

  it("select-all covers the filtered rows only and shows mixed state", () => {
    const { onSelectedChange, box } = setup();
    fireEvent.change(screen.getByRole("searchbox", { name: /filter/i }), { target: { value: "l1" } });
    const all = screen.getByRole("checkbox", { name: /select all/i }) as HTMLInputElement;
    fireEvent.click(all);
    expect(onSelectedChange).toHaveBeenLastCalledWith(["member_10", "member_196"], [ROWS[0], ROWS[2]]);
    fireEvent.click(box("member_10"));
    expect(all.indeterminate).toBe(true);
    expect(screen.getByText(/1 selected/)).toBeTruthy();
  });

  it("supports controlled selection", () => {
    const { rerender } = render(
      <DataTable rows={ROWS} columns={COLS} rowKey={key} selectable selected={["member_3"]} onSelectedChange={() => {}} />);
    expect((screen.getByRole("checkbox", { name: "Select member_3" }) as HTMLInputElement).checked).toBe(true);
    rerender(<DataTable rows={ROWS} columns={COLS} rowKey={key} selectable selected={[]} onSelectedChange={() => {}} />);
    expect((screen.getByRole("checkbox", { name: "Select member_3" }) as HTMLInputElement).checked).toBe(false);
  });
});

describe("DataTable keyboard and row activation", () => {
  it("moves a cursor with the arrow keys, toggles with Space and activates with Enter", () => {
    const onRowClick = vi.fn();
    const onSelectedChange = vi.fn();
    render(<DataTable rows={ROWS} columns={COLS} rowKey={key} selectable onRowClick={onRowClick}
      onSelectedChange={onSelectedChange} />);
    const grid = screen.getByRole("grid");
    grid.focus();
    fireEvent.keyDown(grid, { key: "ArrowDown" });
    const first = grid.getAttribute("aria-activedescendant")!;
    expect(document.getElementById(first)?.getAttribute("data-key")).toBe("member_10");
    fireEvent.keyDown(grid, { key: "ArrowDown" });
    expect(document.getElementById(grid.getAttribute("aria-activedescendant")!)?.getAttribute("data-key")).toBe("member_2");
    fireEvent.keyDown(grid, { key: " " });
    expect(onSelectedChange).toHaveBeenLastCalledWith(["member_2"], [ROWS[1]]);
    fireEvent.keyDown(grid, { key: "ArrowDown", shiftKey: true });
    expect(onSelectedChange).toHaveBeenLastCalledWith(["member_2", "member_196"], [ROWS[1], ROWS[2]]);
    fireEvent.keyDown(grid, { key: "Enter" });
    expect(onRowClick).toHaveBeenLastCalledWith(ROWS[2], 2);
    fireEvent.keyDown(grid, { key: "End" });
    expect(document.getElementById(grid.getAttribute("aria-activedescendant")!)?.getAttribute("data-key")).toBe("member_3");
    fireEvent.keyDown(grid, { key: "Escape" });
    expect(onSelectedChange).toHaveBeenLastCalledWith([], []);
  });

  it("rebuilds the shift range from the anchor, so moving back shrinks it", () => {
    const onSelectedChange = vi.fn();
    render(<DataTable rows={ROWS} columns={COLS} rowKey={key} selectable onSelectedChange={onSelectedChange} />);
    const grid = screen.getByRole("grid");
    grid.focus();
    fireEvent.keyDown(grid, { key: "ArrowDown" });                    // cursor member_10
    fireEvent.keyDown(grid, { key: "ArrowDown", shiftKey: true });    // + member_2
    fireEvent.keyDown(grid, { key: "ArrowDown", shiftKey: true });    // + member_196
    expect(onSelectedChange).toHaveBeenLastCalledWith(["member_10", "member_2", "member_196"], [ROWS[0], ROWS[1], ROWS[2]]);
    fireEvent.keyDown(grid, { key: "ArrowUp", shiftKey: true });      // back: member_196 leaves
    expect(onSelectedChange).toHaveBeenLastCalledWith(["member_10", "member_2"], [ROWS[0], ROWS[1]]);
    fireEvent.keyDown(grid, { key: "Home", shiftKey: true });         // anchor row alone
    expect(onSelectedChange).toHaveBeenLastCalledWith(["member_10"], [ROWS[0]]);
  });

  it("keeps a selection made before the shift range and extends from the new anchor", () => {
    const onSelectedChange = vi.fn();
    render(<DataTable rows={ROWS} columns={COLS} rowKey={key} selectable onSelectedChange={onSelectedChange} />);
    const grid = screen.getByRole("grid");
    fireEvent.click(screen.getByRole("checkbox", { name: "Select member_3" }));   // earlier pick
    fireEvent.click(screen.getByRole("checkbox", { name: "Select member_10" }));  // anchor
    fireEvent.keyDown(grid, { key: "ArrowDown", shiftKey: true });                // cursor member_2
    fireEvent.keyDown(grid, { key: "ArrowDown", shiftKey: true });                // member_196
    fireEvent.keyDown(grid, { key: "ArrowUp", shiftKey: true });                  // back to member_2
    expect(onSelectedChange).toHaveBeenLastCalledWith(["member_10", "member_2", "member_3"], [ROWS[0], ROWS[1], ROWS[3]]);
  });

  describe("a plain click or Enter moves the anchor (rows → inspector / onRowClick)", () => {
    type R = { id: string };
    const TEN: R[] = Array.from({ length: 10 }, (_, i) => ({ id: `r${i}` }));
    const C10: DataColumn<R>[] = [{ id: "id", header: "Id" }];
    const cell = (id: string) => screen.getByText(id);
    const variants = {
      inspect: { inspect: (r: R) => ({ kind: "tile", id: r.id }) },
      onRowClick: { onRowClick: () => {} },
    };
    for (const [name, extra] of Object.entries(variants)) {
      it(`keeps the earlier range when a new one starts from a plain click (${name})`, () => {
        const onSelectedChange = vi.fn();
        render(<DataTable rows={TEN} columns={C10} rowKey={(r) => r.id} selectable {...extra}
          onSelectedChange={onSelectedChange} />);
        fireEvent.click(cell("r1"));
        fireEvent.click(cell("r3"), { shiftKey: true });
        expect(onSelectedChange.mock.lastCall![0]).toEqual(["r1", "r2", "r3"]);
        fireEvent.click(cell("r6"));                         // plain: activates, moves the anchor
        fireEvent.click(cell("r8"), { shiftKey: true });
        expect(onSelectedChange.mock.lastCall![0]).toEqual(["r1", "r2", "r3", "r6", "r7", "r8"]);
        fireEvent.click(cell("r7"), { shiftKey: true });     // shrinks the new range only
        expect(onSelectedChange.mock.lastCall![0]).toEqual(["r1", "r2", "r3", "r6", "r7"]);
      });
    }

    it("Enter on the cursor row is a plain click: the next Shift range starts there", () => {
      const onSelectedChange = vi.fn();
      render(<DataTable rows={TEN} columns={C10} rowKey={(r) => r.id} selectable {...variants.inspect}
        onSelectedChange={onSelectedChange} />);
      const grid = screen.getByRole("grid");
      fireEvent.click(cell("r1"));
      fireEvent.click(cell("r3"), { shiftKey: true });                    // cursor r3
      grid.focus();
      for (let i = 0; i < 3; i++) fireEvent.keyDown(grid, { key: "ArrowDown" });  // cursor r6
      fireEvent.keyDown(grid, { key: "Enter" });
      expect(useInspector.getState().current).toEqual({ kind: "tile", id: "r6" });
      fireEvent.keyDown(grid, { key: "ArrowDown", shiftKey: true });
      fireEvent.keyDown(grid, { key: "ArrowDown", shiftKey: true });
      expect(onSelectedChange.mock.lastCall![0]).toEqual(["r1", "r2", "r3", "r6", "r7", "r8"]);
    });
  });

  it("drops selected keys whose rows disappeared (count, keys and rows stay consistent)", () => {
    const onSelectedChange = vi.fn();
    const { rerender } = render(<DataTable rows={ROWS} columns={COLS} rowKey={key} selectable
      defaultSelected={["member_2", "member_3"]} onSelectedChange={onSelectedChange} />);
    expect(screen.getByText(/2 selected/)).toBeTruthy();
    rerender(<DataTable rows={ROWS.slice(0, 3)} columns={COLS} rowKey={key} selectable
      defaultSelected={["member_2", "member_3"]} onSelectedChange={onSelectedChange} />);
    expect(screen.getByText(/1 selected/)).toBeTruthy();
    fireEvent.click(screen.getByRole("checkbox", { name: "Select member_10" }));
    expect(onSelectedChange).toHaveBeenLastCalledWith(["member_10", "member_2"], [ROWS[0], ROWS[1]]);
  });

  it("selects all filtered rows with mod+A", () => {
    const onSelectedChange = vi.fn();
    render(<DataTable rows={ROWS} columns={COLS} rowKey={key} selectable onSelectedChange={onSelectedChange} />);
    const grid = screen.getByRole("grid");
    fireEvent.keyDown(grid, { key: "a", ctrlKey: true });
    expect(onSelectedChange).toHaveBeenLastCalledWith(ROWS.map(key), ROWS);
  });

  it("opens the inspector on row click and highlights the inspected row", () => {
    render(<DataTable rows={ROWS} columns={COLS} rowKey={key} inspect={(r) => ({ kind: "member", id: r.name })} />);
    fireEvent.click(screen.getByText("member_3"));
    expect(useInspector.getState().current).toEqual({ kind: "member", id: "member_3" });
    const row = screen.getByText("member_3").closest("tr")!;
    expect(row.getAttribute("data-active")).toBe("true");
  });

  it("ctrl/shift-click on a row selects instead of activating", () => {
    const onRowClick = vi.fn();
    const onSelectedChange = vi.fn();
    render(<DataTable rows={ROWS} columns={COLS} rowKey={key} selectable onRowClick={onRowClick}
      onSelectedChange={onSelectedChange} />);
    fireEvent.click(screen.getByText("member_10"), { metaKey: true });
    fireEvent.click(screen.getByText("member_3"), { shiftKey: true });
    expect(onRowClick).not.toHaveBeenCalled();
    expect(onSelectedChange).toHaveBeenLastCalledWith(ROWS.map(key), ROWS);
  });
});

describe("DataTable columns and export", () => {
  it("hides columns by default or by controlled visibility", () => {
    const cols = COLS.map((c) => (c.id === "loss" ? { ...c, hidden: true } : c));
    const { rerender } = render(<DataTable rows={ROWS} columns={cols} rowKey={key} />);
    expect(screen.queryByRole("columnheader", { name: /Loss/ })).toBeNull();
    rerender(<DataTable rows={ROWS} columns={cols} rowKey={key} columnVisibility={{ loss: true, psnr: false }} />);
    expect(screen.getByRole("columnheader", { name: /Loss/ })).toBeTruthy();
    expect(screen.queryByRole("columnheader", { name: /PSNR/ })).toBeNull();
  });

  it("exports the visible, filtered, sorted rows as CSV (or only the selection)", () => {
    render(<DataTable rows={ROWS} columns={COLS} rowKey={key} selectable exportName="Members table" />);
    fireEvent.change(screen.getByRole("searchbox", { name: /filter/i }), { target: { value: "l1" } });
    fireEvent.click(sortButton("Member"));
    fireEvent.click(screen.getByRole("button", { name: /csv/i }));
    expect(downloads.calls[0]).toEqual({
      name: "members-table.csv",
      text: "Member,PSNR,Loss\r\nmember_10,43.2,l1\r\nmember_196,,l1\r\n",
    });
    fireEvent.click(screen.getByRole("checkbox", { name: "Select member_196" }));
    fireEvent.click(screen.getByRole("button", { name: /csv/i }));
    expect(downloads.calls[1].text).toBe("Member,PSNR,Loss\r\nmember_196,,l1\r\n");
  });

  it("exports every selected row, also those the filter hides, and labels the button with that count", () => {
    render(<DataTable rows={ROWS} columns={COLS} rowKey={key} selectable exportName="members" />);
    fireEvent.click(sortButton("PSNR"));                          // member_3, member_10, member_2, member_196
    fireEvent.click(screen.getByRole("checkbox", { name: "Select member_2" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Select member_3" }));
    fireEvent.change(screen.getByRole("searchbox", { name: /filter/i }), { target: { value: "l2" } });
    expect(bodyNames()).toEqual(["member_2"]);
    const csv = screen.getByRole("button", { name: /csv/i });
    expect(csv.textContent).toBe("CSV (2)");
    expect(csv.getAttribute("title")).toBe("Export the 2 selected rows (1 hidden by the filter)");
    fireEvent.click(csv);
    // the whole selection, in the current sort order
    expect(downloads.calls[0].text).toBe("Member,PSNR,Loss\r\nmember_3,42,l3\r\nmember_2,44.1,l2\r\n");
  });
});

describe("DataTable scale", () => {
  it("virtualises thousands of rows: only a window is in the DOM", () => {
    // happy-dom has no layout: give the scroll viewport a real height.
    vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockImplementation(function (this: HTMLElement) {
      return this.classList.contains("ui-dt__scroll") ? 400 : 0;
    });
    vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockImplementation(() => 800);
    const many = Array.from({ length: 5000 }, (_, i) => ({ name: `m${i}`, psnr: i, loss: "l1" }));
    render(<DataTable rows={many} columns={COLS} rowKey={key} height={400} />);
    const grid = screen.getByRole("grid");
    expect(grid.getAttribute("aria-rowcount")).toBe("5001");
    const rendered = bodyNames().length;
    expect(rendered).toBeGreaterThan(0);
    expect(rendered).toBeLessThan(200);
  });
});

describe("DataTable URL state", () => {
  function Loc() { return <div data-testid="loc">{useLocation().search}</div>; }
  it("keeps sort and filter in the URL under urlKey", () => {
    render(
      <MemoryRouter initialEntries={["/x?m.sort=-psnr"]}>
        <DataTable rows={ROWS} columns={COLS} rowKey={key} urlKey="m" />
        <Loc />
      </MemoryRouter>,
    );
    expect(bodyNames()).toEqual(["member_2", "member_10", "member_3", "member_196"]);
    act(() => { fireEvent.change(screen.getByRole("searchbox", { name: /filter/i }), { target: { value: "l1" } }); });
    expect(screen.getByTestId("loc").textContent).toContain("m.q=l1");
    expect(bodyNames()).toEqual(["member_10", "member_196"]);
  });
});
