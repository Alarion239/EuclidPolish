/* A unified-diff parser for System › Code's diff viewer (git diff / git show
 * output): files, hunks and numbered lines. Pure; DiffView.tsx renders it. */

export type DiffLineKind = "meta" | "hunk" | "add" | "del" | "ctx" | "note";
export type DiffLine = { kind: DiffLineKind; text: string; old?: number; new?: number };
export type DiffFile = {
  path: string;
  oldPath?: string;
  lines: DiffLine[];
  added: number;
  removed: number;
  binary: boolean;
};

const HUNK = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/;

/** `a/path` / `b/path` / `"a/p q"` → the path (quotes and prefix dropped). */
function cleanPath(raw: string): string {
  let p = raw.trim();
  if (p.startsWith('"') && p.endsWith('"')) p = p.slice(1, -1);
  return p.replace(/^[ab]\//, "");
}

export function parseDiff(text: string): DiffFile[] {
  const files: DiffFile[] = [];
  let cur: DiffFile | null = null;
  let oldNo = 0;
  let newNo = 0;
  let inHunk = false;
  const start = (path: string) => {
    cur = { path, lines: [], added: 0, removed: 0, binary: false };
    files.push(cur);
    inHunk = false;
    return cur;
  };
  for (const line of text.replace(/\r\n?/g, "\n").split("\n")) {
    if (line.startsWith("diff --git ")) {
      const m = line.match(/^diff --git (\S+|"[^"]+") (\S+|"[^"]+")$/);
      const f = start(m ? cleanPath(m[2]) : line.slice(11));
      f.lines.push({ kind: "meta", text: line });
      continue;
    }
    if (!cur) {
      if (!line.trim()) continue;
      // Output that does not start with a file header (e.g. a truncation note).
      cur = start("");
    }
    const f: DiffFile = cur;
    const hunk = line.match(HUNK);
    if (hunk) {
      oldNo = Number(hunk[1]);
      newNo = Number(hunk[2]);
      inHunk = true;
      f.lines.push({ kind: "hunk", text: line });
      continue;
    }
    if (!inHunk) {
      if (line.startsWith("rename from ")) f.oldPath = line.slice(12);
      if (line.startsWith("+++ ") && line.slice(4) !== "/dev/null") f.path = cleanPath(line.slice(4));
      if (line.startsWith("--- ") && line.slice(4) !== "/dev/null" && !f.oldPath) {
        const old = cleanPath(line.slice(4));
        if (old !== f.path) f.oldPath = old;
      }
      if (/^Binary files .* differ$/.test(line)) f.binary = true;
      if (line || f.lines.length) f.lines.push({ kind: "meta", text: line });
      continue;
    }
    if (line.startsWith("+")) { f.lines.push({ kind: "add", text: line.slice(1), new: newNo++ }); f.added += 1; }
    else if (line.startsWith("-")) { f.lines.push({ kind: "del", text: line.slice(1), old: oldNo++ }); f.removed += 1; }
    else if (line.startsWith("\\")) f.lines.push({ kind: "note", text: line });
    else if (line.startsWith("[…truncated")) f.lines.push({ kind: "note", text: line });
    else if (line === "" ) { /* trailing newline of the output */ }
    else f.lines.push({ kind: "ctx", text: line.slice(1), old: oldNo++, new: newNo++ });
  }
  for (const f of files) if (f.oldPath === f.path) delete f.oldPath;
  return files;
}

/** `+12 −3` for a file or a whole diff. */
export function diffStat(files: readonly DiffFile[]): { added: number; removed: number } {
  return files.reduce((acc, f) => ({ added: acc.added + f.added, removed: acc.removed + f.removed }),
    { added: 0, removed: 0 });
}
