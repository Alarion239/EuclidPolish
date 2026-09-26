import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { createRef, useState } from "react";
import { MemoryRouter, Link, useLocation } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { queryClient } from "../api/query";
import {
  Button, Checkbox, CopyButton, Dialog, Field, IconButton, Input, JobProgress, JsonTree, Kbd, Kpi, LogView, Menu,
  MultiSelect, NumberField, Popover, Section, Segmented, Select, Slider, Switch, Tabs, Toaster, UiProvider, comboKeys, findMatches,
  jsonPath, toast,
} from "./index";
import type { Job } from "../api/jobs";

/* Radix menus open on pointerdown (mouse, primary button). */
function pointerOpen(el: Element) {
  fireEvent.pointerDown(el, { button: 0, ctrlKey: false, pointerType: "mouse" });
}

/** The text of the elements an element's aria-describedby points at. */
function description(el: Element): string {
  return (el.getAttribute("aria-describedby") ?? "").split(/\s+/).filter(Boolean)
    .map((id) => document.getElementById(id)?.textContent ?? `<missing #${id}>`).join(" ");
}

describe("Button", () => {
  it("is type=button, guards clicks while loading and renders links", () => {
    const onClick = vi.fn();
    const { rerender } = render(<Button onClick={onClick}>Run</Button>);
    const b = screen.getByRole("button", { name: "Run" });
    expect(b.getAttribute("type")).toBe("button");
    fireEvent.click(b);
    expect(onClick).toHaveBeenCalledTimes(1);
    rerender(<Button onClick={onClick} loading>Run</Button>);
    expect(b.getAttribute("aria-busy")).toBe("true");
    fireEvent.click(b);
    expect(onClick).toHaveBeenCalledTimes(1);
    rerender(<Button href="/api/x.csv" download variant="primary" size="sm">Get</Button>);
    const a = screen.getByRole("link", { name: "Get" });
    expect(a.getAttribute("href")).toBe("/api/x.csv");
    expect(a.className).toContain("ui-btn--primary");
    expect(a.className).toContain("ui-btn--sm");
  });

  it("renders its child with button styles via asChild", () => {
    render(<MemoryRouter><Button asChild variant="ghost"><Link to="/sky">Sky</Link></Button></MemoryRouter>);
    const a = screen.getByRole("link", { name: "Sky" });
    expect(a.className).toContain("ui-btn");
    expect(a.className).toContain("ui-btn--ghost");
  });

  it("forwards the ref and the loading click guard in asChild mode", () => {
    const onClick = vi.fn();
    const ref = createRef<HTMLButtonElement>();
    function Where() { return <span data-testid="where">{useLocation().pathname}</span>; }
    const { rerender } = render(
      <MemoryRouter><Button ref={ref} asChild loading onClick={onClick}><Link to="/sky">Sky</Link></Button><Where /></MemoryRouter>);
    const a = screen.getByRole("link", { name: "Sky" });
    expect(ref.current).toBe(a);
    expect(a.getAttribute("aria-busy")).toBe("true");
    fireEvent.click(a);
    expect(onClick).not.toHaveBeenCalled();
    expect(screen.getByTestId("where").textContent).toBe("/");     // navigation blocked too
    rerender(<MemoryRouter><Button ref={ref} asChild onClick={onClick}><Link to="/sky">Sky</Link></Button><Where /></MemoryRouter>);
    fireEvent.click(screen.getByRole("link", { name: "Sky" }));
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("forwards the ref and the remaining props in href mode", () => {
    const ref = createRef<HTMLButtonElement>();
    render(<Button ref={ref} href="/api/x.csv" data-testid="dl" aria-describedby="why" name="ignored">Get</Button>);
    const a = screen.getByRole("link", { name: "Get" });
    expect(ref.current as unknown).toBe(a);
    expect(a.getAttribute("data-testid")).toBe("dl");
    expect(a.getAttribute("aria-describedby")).toBe("why");
    expect(a.hasAttribute("name")).toBe(false);                    // button-only attributes are dropped
  });

  it("IconButton has an accessible name and works without a tooltip provider", () => {
    const onClick = vi.fn();
    render(<IconButton icon="reset" label="Reset zoom" onClick={onClick} pressed={false} />);
    const b = screen.getByRole("button", { name: "Reset zoom" });
    expect(b.getAttribute("aria-pressed")).toBe("false");
    fireEvent.click(b);
    expect(onClick).toHaveBeenCalled();
  });
});

describe("form controls", () => {
  it("Field associates its label with any child and the hint does not toggle the control", async () => {
    function Demo() {
      const [on, setOn] = useState(false);
      return (
        <Field label="Include training" hint="Adds the training catalogue.">
          <input type="checkbox" checked={on} onChange={(e) => setOn(e.target.checked)} />
        </Field>
      );
    }
    render(<Demo />);
    const box = screen.getByRole("checkbox", { name: /Include training/ }) as HTMLInputElement;
    fireEvent.click(screen.getByText("Include training"));
    expect(box.checked).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "About Include training" }));
    expect(box.checked).toBe(true);
    expect(await screen.findByText("Adds the training catalogue.")).toBeTruthy();
  });

  it("Field: the error describes the control (aria-describedby + aria-invalid) and is not part of its name", () => {
    function Demo({ error }: { error?: string }) {
      const [v, setV] = useState("");
      return (
        <>
          <Field label="Knee" hint="Asinh knee." error={error}><Input value={v} onChange={setV} /></Field>
          <Field label="Raw" error={error}><input defaultValue="" /></Field>
          <Field label="Depth" error={error}>
            <Input value="" onChange={() => {}} aria-describedby="depth-unit" />
          </Field>
          <span id="depth-unit">residual blocks</span>
        </>
      );
    }
    const { rerender } = render(<Demo />);
    const knee = screen.getByRole("textbox", { name: "Knee" });
    expect(knee.getAttribute("aria-invalid")).toBeNull();
    expect(knee.getAttribute("aria-describedby")).toBeNull();
    act(() => { knee.focus(); });
    rerender(<Demo error="must be > 0" />);
    // the same element, still focused: an error appearing while typing does not remount the control
    expect(screen.getByRole("textbox", { name: "Knee" })).toBe(knee);
    expect(document.activeElement).toBe(knee);
    expect(knee.getAttribute("aria-invalid")).toBe("true");
    expect(description(knee)).toBe("must be > 0");
    const raw = screen.getByRole("textbox", { name: "Raw" });   // a raw child is wired too
    expect(raw.getAttribute("aria-invalid")).toBe("true");
    expect(description(raw)).toBe("must be > 0");
    // a caller's own description is kept, the error is added after it
    expect(description(screen.getByRole("textbox", { name: "Depth" }))).toBe("residual blocks must be > 0");
    expect(screen.getAllByRole("alert")).toHaveLength(3);
    rerender(<Demo />);
    expect(knee.getAttribute("aria-invalid")).toBeNull();
    expect(knee.getAttribute("aria-describedby")).toBeNull();
  });

  it("Field description / NumberField hint is the control's description, not part of its name", () => {
    render(
      <>
        <NumberField label="parallel workers" hint="per band" value="4" onChange={() => {}} />
        <Field label="Tiles" description="comma-separated"><input defaultValue="" /></Field>
      </>,
    );
    const workers = screen.getByRole("spinbutton", { name: "parallel workers" });
    expect(description(workers)).toBe("per band");
    expect(workers.getAttribute("aria-invalid")).toBeNull();
    expect(description(screen.getByRole("textbox", { name: "Tiles" }))).toBe("comma-separated");
  });

  it("a Field's error does not leak into controls inside a popover or dialog opened from it", async () => {
    const overlays = {
      popover: (
        <Popover open trigger={<Button>Filters</Button>} label="filter panel">
          <Input value="" onChange={() => {}} aria-label="inner" />
        </Popover>
      ),
      dialog: <Dialog open title="More"><Input value="" onChange={() => {}} aria-label="inner" /></Dialog>,
    };
    for (const [kind, overlay] of Object.entries(overlays)) {
      const { unmount } = render(<Field label="Filters" error="bad filter">{overlay}</Field>);
      const inner = await screen.findByRole("textbox", { name: "inner" });
      expect(inner.getAttribute("aria-describedby"), kind).toBeNull();
      expect(inner.getAttribute("aria-invalid"), kind).toBeNull();
      unmount();
    }
  });

  it("a searchable Select / MultiSelect is named by its label AND its current value", () => {
    render(
      <>
        <Field label="Member">
          <Select searchable value="a" onChange={() => {}} options={[{ value: "a", label: "A" }, { value: "b", label: "B" }]} />
        </Field>
        <MultiSelect value={["v", "y"]} onChange={() => {}} aria-label="bands"
          options={[{ value: "v", label: "VIS" }, { value: "y", label: "Y" }]} />
        <Select searchable value={"zz" as string} onChange={() => {}} options={[{ value: "a", label: "A" }]} />
      </>,
    );
    expect(screen.getByRole("button", { name: "Member A" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "bands VIS, Y" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Select…" })).toBeTruthy();   // no label: its text
  });

  it("Checkbox reflects the indeterminate state", () => {
    render(<Checkbox checked={false} indeterminate onChange={() => {}}>some</Checkbox>);
    const box = screen.getByRole("checkbox") as HTMLInputElement;
    // the native `indeterminate` property is what AT reads as "mixed"
    expect(box.indeterminate).toBe(true);
  });

  it("Switch toggles with its label", () => {
    const onChange = vi.fn();
    render(<Switch checked={false} onChange={onChange}>Linked</Switch>);
    fireEvent.click(screen.getByRole("switch", { name: "Linked" }));
    expect(onChange).toHaveBeenCalledWith(true);
  });

  it("Segmented selects an option and ignores deselecting the active one", () => {
    const onChange = vi.fn();
    render(<Segmented value="a" onChange={onChange} aria-label="mode"
      options={[{ value: "a", label: "A" }, { value: "b", label: "B" }]} />);
    fireEvent.click(screen.getByRole("radio", { name: "B" }));
    expect(onChange).toHaveBeenCalledWith("b");
    fireEvent.click(screen.getByRole("radio", { name: "A" }));
    expect(onChange).toHaveBeenCalledTimes(1);
  });

  it("native Select keeps an unknown current value visible", () => {
    render(<Select value={"x" as string} onChange={() => {}} aria-label="knee"
      options={[{ value: "a", label: "A" }]} placeholder="pick one" />);
    const sel = screen.getByRole("combobox", { name: "knee" }) as HTMLSelectElement;
    expect(sel.value).toBe("x");
    expect(within(sel).getByText("pick one")).toBeTruthy();
  });

  it("searchable Select filters and picks", async () => {
    const onChange = vi.fn();
    render(<Select searchable value="m1" onChange={onChange} aria-label="member"
      options={["m1", "m2", "m10", "other"].map((v) => ({ value: v, label: `member ${v}` }))} />);
    fireEvent.click(screen.getByRole("button", { name: "member member m1" }));
    const search = await screen.findByPlaceholderText("Search…");
    fireEvent.change(search, { target: { value: "oth" } });
    await waitFor(() => expect(screen.queryByText("member m2")).toBeNull());
    fireEvent.click(screen.getByText("member other"));
    expect(onChange).toHaveBeenCalledWith("other");
  });

  it("MultiSelect toggles values and select-all", async () => {
    function Demo() {
      const [v, setV] = useState<string[]>(["a"]);
      return <><MultiSelect value={v} onChange={setV} aria-label="bands"
        options={[{ value: "a", label: "VIS" }, { value: "b", label: "Y" }, { value: "c", label: "J" }]} />
        <output>{v.join(",")}</output></>;
    }
    render(<Demo />);
    const trigger = screen.getByRole("button", { name: "bands VIS" });
    expect(trigger.textContent).toContain("VIS");
    fireEvent.click(trigger);
    fireEvent.click(await screen.findByText("Y"));
    expect(screen.getByRole("status").textContent).toBe("a,b");
    fireEvent.click(screen.getByRole("button", { name: "Select all" }));
    expect(screen.getByRole("status").textContent).toBe("a,b,c");
  });

  it("Slider maps a log track and labels the thumb with the formatted value", () => {
    const onChange = vi.fn();
    render(<Slider value={100} onChange={onChange} min={0.1} max={1e4} scale="log" aria-label="knee"
      format={(v) => `${v} e-`} />);
    const thumb = screen.getByRole("slider", { name: "knee" });
    expect(thumb.getAttribute("aria-valuetext")).toBe("100 e-");
    expect(thumb.getAttribute("aria-valuenow")).toBe("600");
    fireEvent.keyDown(thumb, { key: "End" });
    expect(onChange).toHaveBeenLastCalledWith(1e4);
  });
});

describe("Slider on an invalid log range", () => {
  it("falls back to a linear track instead of throwing", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const onChange = vi.fn();
    render(<Slider value={5} onChange={onChange} min={0} max={10} scale="log" aria-label="bad" />);
    const thumb = screen.getByRole("slider", { name: "bad" });
    expect(thumb.getAttribute("aria-valuenow")).toBe("5");
    fireEvent.keyDown(thumb, { key: "End" });
    expect(onChange).toHaveBeenLastCalledWith(10);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});

describe("Tabs", () => {
  it("are ARIA tabs with onChange", () => {
    const onChange = vi.fn();
    render(<Tabs value="a" onChange={onChange} tabs={[{ id: "a", label: "Alpha" }, { id: "b", label: "Beta", badge: 3 }]} />);
    const beta = screen.getByRole("tab", { name: /Beta/ });
    expect(screen.getByRole("tab", { name: "Alpha" }).getAttribute("aria-selected")).toBe("true");
    fireEvent.mouseDown(beta, { button: 0 });
    fireEvent.focus(beta);
    expect(onChange).toHaveBeenCalledWith("b");
  });

  it("do not point aria-controls at panels that are not rendered", () => {
    const { rerender } = render(<Tabs value="a" onChange={() => {}} tabs={[{ id: "a", label: "Alpha" }, { id: "b", label: "Beta" }]} />);
    for (const tab of screen.getAllByRole("tab")) expect(tab.hasAttribute("aria-controls")).toBe(false);
    rerender(<Tabs value="a" onChange={() => {}} tabs={[{ id: "a", label: "Alpha" }, { id: "b", label: "Beta" }]}>panel A</Tabs>);
    const alpha = screen.getByRole("tab", { name: "Alpha" });
    const panel = document.getElementById(alpha.getAttribute("aria-controls")!);
    expect(panel?.textContent).toBe("panel A");
  });

  it("become router links (aria-current) when tabs carry `to`", () => {
    render(
      <MemoryRouter>
        <Tabs value="noise" aria-label="Realism" tabs={[
          { id: "overview", label: "Overview", to: "/realism/overview" },
          { id: "noise", label: "Noise", to: "/realism/noise" },
        ]} />
      </MemoryRouter>,
    );
    const nav = screen.getByRole("navigation", { name: "Realism" });
    const noise = within(nav).getByRole("link", { name: "Noise" });
    expect(noise.getAttribute("href")).toBe("/realism/noise");
    expect(noise.getAttribute("aria-current")).toBe("page");
    expect(within(nav).getByRole("link", { name: "Overview" }).getAttribute("aria-current")).toBeNull();
  });
});

describe("Menu", () => {
  it("opens from its trigger and runs an item", async () => {
    const onSelect = vi.fn();
    render(<Menu label="Actions" trigger={<Button>More</Button>}
      items={[{ label: "Fork", onSelect }, { type: "separator" }, { label: "Archive", tone: "danger", onSelect: () => {} }]} />);
    pointerOpen(screen.getByRole("button", { name: "More" }));
    const item = await screen.findByRole("menuitem", { name: "Fork" });
    fireEvent.click(item);
    expect(onSelect).toHaveBeenCalled();
  });
});

describe("LogView", () => {
  it("finds case-insensitive matches (pure)", () => {
    expect(findMatches(["Error one", "fine", "error two error"], "ERROR"))
      .toEqual([{ line: 0, start: 0, end: 5 }, { line: 2, start: 0, end: 5 }, { line: 2, start: 10, end: 15 }]);
    expect(findMatches(["x"], "")).toEqual([]);
  });

  it("highlights matches, counts them and steps with Enter", () => {
    render(<LogView text={"step 1\nERROR a\nok\nerror b"} title="job log" />);
    const box = screen.getByRole("searchbox", { name: "Search log" });
    fireEvent.change(box, { target: { value: "error" } });
    expect(screen.getByText("1/2")).toBeTruthy();
    expect(document.querySelectorAll("mark.ui-log__hit")).toHaveLength(2);
    fireEvent.keyDown(box, { key: "Enter" });
    expect(screen.getByText("2/2")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Only matching lines" }));
    expect(screen.getByLabelText("job log").textContent).not.toContain("step 1");
  });
});

describe("JsonTree", () => {
  it("expands, collapses and pages long arrays", () => {
    render(<JsonTree data={{ run: { id: "r1", ok: true }, xs: Array.from({ length: 120 }, (_, i) => i) }} expandDepth={1} pageSize={50} />);
    expect(screen.queryByText("\"r1\"")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /^run\b/, expanded: false }));
    expect(screen.getByText("\"r1\"")).toBeTruthy();
    expect(screen.getByRole("button", { name: /^run\b/ }).getAttribute("aria-expanded")).toBe("true");
    fireEvent.click(screen.getByRole("button", { name: /^xs\b/, expanded: false }));
    expect(screen.getByText(/show 50 more of 70/)).toBeTruthy();
    fireEvent.click(screen.getByText(/show 50 more of 70/));
    expect(screen.getByText(/show 20 more of 20/)).toBeTruthy();
  });

  it("is plain lists of disclosure buttons (no tree roles without a tree keyboard model)", () => {
    render(<JsonTree data={{ run: { id: "r1" }, n: 1 }} expandDepth={1} />);
    expect(document.querySelector("[role='tree'], [role='treeitem'], [role='group']")).toBeNull();
    const root = screen.getByRole("button", { name: /^root\b/, expanded: true });
    const body = document.getElementById(root.getAttribute("aria-controls")!);
    expect(body?.tagName).toBe("UL");
    const run = screen.getByRole("button", { name: /^run\b/, expanded: false });
    expect(run.hasAttribute("aria-controls")).toBe(false);          // collapsed: no dangling reference
  });

  it("builds JS access paths", () => {
    expect(jsonPath(["a", 3, "odd key", "b"])).toBe('a[3]["odd key"].b');
  });
});

describe("Section", () => {
  it("only references its body with aria-controls while the body is rendered", () => {
    render(<Section title="Calibration" collapsible defaultOpen={false}>body text</Section>);
    const toggle = screen.getByRole("button", { name: "Calibration" });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(toggle.hasAttribute("aria-controls")).toBe(false);
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(document.getElementById(toggle.getAttribute("aria-controls")!)?.textContent).toBe("body text");
  });
});

describe("CopyButton / Kbd", () => {
  it("copies through the clipboard and confirms", async () => {
    const writeText = vi.fn(() => Promise.resolve());
    vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
    render(<CopyButton value={() => "12.5 -30.1"} label="Copy coordinates" />);
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Copy coordinates" })); });
    expect(writeText).toHaveBeenCalledWith("12.5 -30.1");
    expect(screen.getByText("Copied")).toBeTruthy();
  });

  it("formats key combos per platform", () => {
    expect(comboKeys("mod+k", true)).toEqual(["⌘", "K"]);
    expect(comboKeys("mod+shift+p", false)).toEqual(["Ctrl", "Shift", "P"]);
    expect(comboKeys("shift+?", true)).toEqual(["⇧", "?"]);
    render(<Kbd keys="mod+k" />);
    expect(document.querySelectorAll("kbd")).toHaveLength(2);
  });
});

describe("JobProgress", () => {
  const job = (extra: Partial<Job> = {}): Job => ({
    job_id: "j9", label: "fit gate", status: "running", duration: 12, error: null, log: null,
    log_truncated: false, cancellable: true,
    progress: { current: 3, total: 10, pct: 30, label: "fitting" }, ...extra,
  });

  it("shows determinate progress and cancels through POST /api/jobs/<id>/cancel", async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ ok: true }), { status: 200, headers: { "Content-Type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);
    render(<JobProgress job={job()} />);
    expect(screen.getByRole("progressbar").getAttribute("aria-valuenow")).toBe("30");
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Cancel" })); });
    expect(fetchMock).toHaveBeenCalled();
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/jobs/j9/cancel");
    expect(init.method).toBe("POST");
    queryClient.clear();
  });

  it("hides Cancel for non-cancellable or finished jobs", () => {
    const { rerender } = render(<JobProgress job={job({ cancellable: false })} />);
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
    rerender(<JobProgress job={job({ status: "done" })} />);
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
  });
});

describe("Kpi", () => {
  it("navigates inside the SPA with `to` (a router link); `href` stays a plain link", () => {
    function Where() { return <output>{useLocation().pathname}</output>; }
    render(
      <MemoryRouter initialEntries={["/home"]}>
        <Kpi label="Members" value="26" to="/ensemble/starfull/members" />
        <Kpi label="Report" value="pdf" href="/api/report.pdf" />
        <Where />
      </MemoryRouter>,
    );
    const tile = screen.getByRole("link", { name: /Members/ });
    expect(tile.getAttribute("href")).toBe("/ensemble/starfull/members");
    fireEvent.click(tile);
    expect(screen.getByRole("status").textContent).toBe("/ensemble/starfull/members");
    expect(screen.getByRole("link", { name: /Report/ }).getAttribute("href")).toBe("/api/report.pdf");
  });

  it("an action tile keeps its hint out of the tab order and uses it as its description", () => {
    const onClick = vi.fn();
    const { rerender } = render(<Kpi label="PSNR" value="44" hint="Knee-integrated" onClick={onClick} />);
    const tile = screen.getByRole("button", { name: /PSNR/ });
    expect(tile.querySelector("[tabindex]")).toBeNull();          // no focusable inside a button
    expect(description(tile)).toBe("Knee-integrated");
    fireEvent.click(tile);
    expect(onClick).toHaveBeenCalledTimes(1);
    rerender(<Kpi label="PSNR" value="44" hint="Knee-integrated" />);   // a static tile: a focusable hint button
    const about = screen.getByRole("button", { name: "About this figure" });
    expect(about.tagName).toBe("BUTTON");
    expect(about.getAttribute("type")).toBe("button");
  });
});

describe("toast / UiProvider", () => {
  it("shows toasts in the Toaster and tooltips inside the shared provider", async () => {
    render(<UiProvider><IconButton icon="info" label="Details" /></UiProvider>);
    act(() => { toast.success("Saved config"); });
    expect(await screen.findByText("Saved config")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Details" })).toBeTruthy();
  });

  it("a standalone Toaster renders", () => {
    render(<Toaster />);
    expect(document.querySelector("section[aria-label]")).toBeTruthy();
  });
});
