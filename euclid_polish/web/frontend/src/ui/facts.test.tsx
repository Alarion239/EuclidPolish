import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Caption, Details, FactsList, Num, SummaryLine } from "./facts";

describe("statistics components", () => {
  it("SummaryLine is one sentence with bold numbers", () => {
    render(<SummaryLine>Generated <Num>5.03</Num> vs prior <Num>5.08</Num> arcmin⁻² (<Num tone="warn">−1%</Num>)</SummaryLine>);
    const p = screen.getByText(/Generated/);
    expect(p.tagName).toBe("P");
    expect(p.querySelectorAll("strong")).toHaveLength(3);
    expect(p.querySelector("[data-tone=warn]")?.textContent).toBe("−1%");
  });
  it("FactsList keeps label, value and unit on one row and skips empty facts", () => {
    render(<FactsList title="Sample" facts={[
      { label: "Q1 area", value: "63.1", unit: "deg²" },
      null,
      { label: "Matched stars", value: "3,456", hint: "Gaia × Euclid in 3 fields" },
    ]} />);
    expect(screen.getByRole("heading", { name: "Sample" })).toBeTruthy();
    const rows = screen.getAllByRole("group");
    expect(rows).toHaveLength(2);
    expect(rows[0].textContent).toBe("Q1 area63.1deg²");
    expect(screen.getByText("Matched stars").closest("[data-hint]")).toBeTruthy();
  });
  it("FactsList renders nothing without facts, tones a row and makes a hinted label focusable", () => {
    const { container } = render(<FactsList title="Empty" facts={[null, false, undefined]} />);
    expect(container.innerHTML).toBe("");
    render(<FactsList facts={[
      { label: "Holes", value: "8%", tone: "warn", hint: "Pixels below −100σ" },
      { label: "Plain", value: "1", tone: "neutral" },
    ]} />);
    expect(screen.queryByRole("heading")).toBeNull();
    const [warn, plain] = screen.getAllByRole("group");
    expect(warn.dataset.tone).toBe("warn");
    expect(plain.dataset.tone).toBeUndefined();
    expect(screen.getByText("Holes").tabIndex).toBe(0);
    expect(screen.getByText("Plain").hasAttribute("tabindex")).toBe(false);
  });
  it("Caption and Details render quietly", () => {
    render(<><Caption>NOISE_MODEL v5 · Q1_R1</Caption><Details summary="Provenance">sha 1a2b</Details></>);
    expect(screen.getByText("NOISE_MODEL v5 · Q1_R1").className).toContain("ui-caption");
    const d = screen.getByText("Provenance").closest("details")!;
    expect(d.open).toBe(false);
  });
});

describe("Num with a unit", () => {
  it("keeps the unit on the number's line", () => {
    render(<SummaryLine>Prior <Num unit="arcmin⁻²">5.08</Num></SummaryLine>);
    const wrap = screen.getByText("5.08").parentElement!;
    expect(wrap.className).toBe("ui-num-unit");
    expect(wrap.textContent).toBe("5.08\u00a0arcmin⁻²");
  });
});
