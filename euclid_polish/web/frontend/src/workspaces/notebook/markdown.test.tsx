import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Markdown, outline, parseMarkdown, safeHref, slugify } from "./markdown";

describe("parseMarkdown", () => {
  it("parses the notebook's blocks", () => {
    const blocks = parseMarkdown([
      "# single-model retirement", "", "> Campaign created at `35b992c`.", "", "---", "",
      "## 2026-07-02T02:37:41Z", "", "Archived `member_09` →", "`models/x.zip`.",
      "", "- one", "- two", "  continued", "", "1. first", "2. second",
      "", "```py", "print('x')", "```", "", "| a | b |", "|---|---|", "| 1 | 2 |",
    ].join("\n"));
    expect(blocks.map((b) => b.type)).toEqual(
      ["heading", "quote", "rule", "heading", "paragraph", "list", "list", "code", "table"]);
    expect(blocks[4]).toEqual({ type: "paragraph", text: "Archived `member_09` → `models/x.zip`." });
    expect(blocks[5]).toEqual({ type: "list", ordered: false, items: ["one", "two continued"] });
    expect(blocks[6]).toMatchObject({ ordered: true, items: ["first", "second"] });
    expect(blocks[7]).toEqual({ type: "code", lang: "py", text: "print('x')" });
    expect(blocks[8]).toEqual({ type: "table", head: ["a", "b"], rows: [["1", "2"]] });
  });
  it("gives headings unique ids", () => {
    const hs = outline("## Run\n\ntext\n\n## Run\n\n### 2026-07-02T02:37:41Z");
    expect(hs.map((h) => h.id)).toEqual(["run", "run-2", "2026-07-02t02-37-41z"]);
    expect(slugify("**!!**")).toBe("section");
  });
});

describe("safe rendering", () => {
  it("renders inline code, bold, italics and safe links", () => {
    const { container } = render(<Markdown text={"A **bold** *it* `code` [doc](https://x.org) [rel](/ops/git)"} />);
    expect(container.querySelector("strong")?.textContent).toBe("bold");
    expect(container.querySelector("em")?.textContent).toBe("it");
    expect(container.querySelector("code")?.textContent).toBe("code");
    const links = [...container.querySelectorAll("a")];
    expect(links.map((a) => a.getAttribute("href"))).toEqual(["https://x.org", "/ops/git"]);
    expect(links[0].getAttribute("rel")).toContain("noopener");
  });
  it("never emits script, event handlers or javascript: links", () => {
    const evil = '<script>alert(1)</script> <img src=x onerror=alert(1)> [x](javascript:alert(1))';
    const { container } = render(<Markdown text={evil} />);
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("a")).toBeNull();
    expect(container.textContent).toContain("<script>alert(1)</script>");
    expect(container.textContent).toContain("[x](javascript:alert(1))");
  });
  it("classifies hrefs", () => {
    expect(safeHref("https://a.b")).toBe("https://a.b");
    expect(safeHref("#top")).toBe("#top");
    expect(safeHref("//evil.example")).toBeNull();
    expect(safeHref("data:text/html,x")).toBeNull();
    expect(safeHref("vbscript:x")).toBeNull();
  });
  it("renders headings one level down (the page owns h1)", () => {
    const { container } = render(<Markdown text={"# Title\n\n## Entry"} />);
    expect(container.querySelector("h2")?.textContent).toBe("Title");
    expect(container.querySelector("h3")?.id).toBe("entry");
  });
});
