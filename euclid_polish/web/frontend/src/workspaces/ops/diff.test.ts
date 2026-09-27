import { describe, expect, it } from "vitest";
import { diffStat, parseDiff } from "./diff";

const DIFF = `diff --git a/README.md b/README.md
index 1111111..2222222 100644
--- a/README.md
+++ b/README.md
@@ -1,3 +1,3 @@
 seed
-old line
+new line
 tail
\\ No newline at end of file
diff --git a/sp ace.txt b/sp ace.txt
new file mode 100644
--- /dev/null
+++ b/sp ace.txt
@@ -0,0 +1,2 @@
+a
+b
diff --git a/img.png b/img.png
Binary files a/img.png and b/img.png differ
`;

describe("parseDiff", () => {
  it("splits files and numbers lines", () => {
    const files = parseDiff(DIFF);
    expect(files.map((f) => f.path)).toEqual(["README.md", "sp ace.txt", "img.png"]);
    const readme = files[0];
    expect(readme.added).toBe(1);
    expect(readme.removed).toBe(1);
    const body = readme.lines.filter((l) => l.kind !== "meta");
    expect(body.map((l) => [l.kind, l.text, l.old, l.new])).toEqual([
      ["hunk", "@@ -1,3 +1,3 @@", undefined, undefined],
      ["ctx", "seed", 1, 1],
      ["del", "old line", 2, undefined],
      ["add", "new line", undefined, 2],
      ["ctx", "tail", 3, 3],
      ["note", "\\ No newline at end of file", undefined, undefined],
    ]);
    expect(files[1].added).toBe(2);
    expect(files[1].oldPath).toBeUndefined();
    expect(files[2].binary).toBe(true);
    expect(diffStat(files)).toEqual({ added: 3, removed: 1 });
  });
  it("records a rename's source", () => {
    const [f] = parseDiff("diff --git a/old.py b/new.py\nsimilarity index 90%\nrename from old.py\nrename to new.py\n");
    expect(f.path).toBe("new.py");
    expect(f.oldPath).toBe("old.py");
  });
  it("tolerates empty output and truncation notes", () => {
    expect(parseDiff("")).toEqual([]);
    const files = parseDiff("diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n\n\n[…truncated, 10 more chars]");
    expect(files[0].lines.some((l) => l.kind === "note" && l.text.startsWith("[…truncated"))).toBe(true);
  });
});
