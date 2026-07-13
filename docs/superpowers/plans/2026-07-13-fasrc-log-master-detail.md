# FASRC Log Master/Detail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the FASRC Past runs table in place with the selected run's paginated raw log, opening `.out` first and returning to the same list page with one Back action.

**Architecture:** Keep run selection and list pagination in `LogsTab`, and render a focused `RunLogDetail` component while a run is selected. Extract the file-selection and URL-building rules into a small DOM-free TypeScript module tested with Node's built-in test runner; reuse the existing Flask run/log endpoints and the current UI tokens.

**Tech Stack:** React 18, TypeScript, Vite, Node built-in test runner, existing Flask JSON APIs.

## Global Constraints

- Do not change the backend API contract.
- Open `.out` by default and fall back to `.err` only when `.out` is unavailable.
- Preserve both run-list pagination and log-window pagination.
- Keep log scrolling inside the detail card and preserve the current runs-list page on Back.
- Reuse existing card, button, badge, mono-log, focus, spacing, light-theme, and dark-theme tokens.
- Rows without `.out` or `.err` remain visible and non-actionable.
- Do not modify or stage unrelated user work already present in the checkout.

---

### Task 1: Build the FASRC run-log master/detail interaction

**Files:**
- Create: `euclid_polish/web/frontend/src/pages/fasrcLogs.ts`
- Create: `euclid_polish/web/frontend/test/fasrcLogs.test.mjs`
- Modify: `euclid_polish/web/frontend/package.json`
- Modify: `euclid_polish/web/frontend/src/pages/Fasrc.tsx:13-180`
- Modify: `euclid_polish/web/frontend/src/ui/index.tsx:123-151`
- Modify: `euclid_polish/web/frontend/src/ui/ui.css:120-160`

**Interfaces:**
- Consumes: `GET /api/fasrc/runs?page=<n>` returning `RunRow.out_path` and `RunRow.err_path`; `GET /api/fasrc/runs/log?path=<path>&page=<n>&page_size=1000` returning a paginated log window.
- Produces: `LogKind`, `preferredLogKind(files)`, `logPath(files, kind)`, and `buildLogPageUrl(path, page, pageSize)` in `fasrcLogs.ts`; optional row activation in `Table`; in-place `RunLogDetail` rendering from `LogsTab`.

- [x] **Step 1: Write the failing navigation-rule tests**

Create `euclid_polish/web/frontend/test/fasrcLogs.test.mjs`:

```js
import assert from "node:assert/strict";
import test from "node:test";

const helpers = await import("../src/pages/fasrcLogs.ts").catch(() => ({}));
const preferredLogKind = helpers.preferredLogKind ?? (() => undefined);
const logPath = helpers.logPath ?? (() => undefined);
const buildLogPageUrl = helpers.buildLogPageUrl ?? (() => undefined);

test("prefers stdout and falls back to stderr", () => {
  assert.equal(preferredLogKind({ out_path: "/logs/a.out", err_path: "/logs/a.err" }), "out");
  assert.equal(preferredLogKind({ err_path: "/logs/a.err" }), "err");
  assert.equal(preferredLogKind({}), null);
});

test("selects the requested file without crossing streams", () => {
  const files = { out_path: "/logs/a.out", err_path: "/logs/a.err" };
  assert.equal(logPath(files, "out"), "/logs/a.out");
  assert.equal(logPath(files, "err"), "/logs/a.err");
  assert.equal(logPath({ out_path: "/logs/a.out" }, "err"), null);
});

test("builds an encoded paginated log URL", () => {
  assert.equal(
    buildLogPageUrl("/repo/logs/jobs/a b.out", 2, 1000),
    "/api/fasrc/runs/log?path=%2Frepo%2Flogs%2Fjobs%2Fa+b.out&page=2&page_size=1000",
  );
});
```

Add the test command to `euclid_polish/web/frontend/package.json`:

```json
"scripts": {
  "dev": "vite",
  "test": "node --experimental-strip-types --test test/*.test.mjs",
  "build": "vite build",
  "preview": "vite preview"
}
```

- [x] **Step 2: Run the tests and confirm the feature contract fails**

Run:

```bash
cd euclid_polish/web/frontend
npm test
```

Expected: 3 assertion failures whose actual value is `undefined`, proving the
missing selection and URL rules are what make the tests red.

- [x] **Step 3: Implement the tested log-selection rules**

Create `euclid_polish/web/frontend/src/pages/fasrcLogs.ts`:

```ts
export type LogKind = "out" | "err";

export type RunLogFiles = {
  out_path?: string | null;
  err_path?: string | null;
};

export function preferredLogKind(files: RunLogFiles): LogKind | null {
  if (files.out_path) return "out";
  if (files.err_path) return "err";
  return null;
}

export function logPath(files: RunLogFiles, kind: LogKind): string | null {
  return kind === "out" ? files.out_path ?? null : files.err_path ?? null;
}

export function buildLogPageUrl(path: string, page: number, pageSize: number): string {
  const query = new URLSearchParams({
    path,
    page: String(page),
    page_size: String(pageSize),
  });
  return `/api/fasrc/runs/log?${query.toString()}`;
}
```

- [x] **Step 4: Run the navigation-rule tests and confirm they pass**

Run:

```bash
cd euclid_polish/web/frontend
npm test
```

Expected: 3 tests pass, 0 fail.

- [x] **Step 5: Add optional whole-row pointer activation to the shared table**

Extend `Table` in `src/ui/index.tsx` with typed optional props and apply the action only to eligible rows:

```tsx
export function Table<T>(
  { columns, rows, empty, rowKey, onRowClick, isRowClickable }: {
    columns: Column<T>[]; rows: T[]; empty?: ReactNode;
    rowKey?: (row: T, i: number) => string | number;
    onRowClick?: (row: T, i: number) => void;
    isRowClickable?: (row: T, i: number) => boolean;
  },
) {
  if (!rows.length) return <Empty>{empty ?? "nothing here yet"}</Empty>;
  return (
    <div className="ui-table-wrap">
      <table className="ui-table">
        <thead>
          <tr>{columns.map((c, i) => (
            <th key={i} style={{ textAlign: c.align, width: c.width }}>{c.header}</th>
          ))}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const clickable = !!onRowClick && (isRowClickable?.(r, i) ?? true);
            return (
              <tr key={rowKey ? rowKey(r, i) : i}
                className={clickable ? "ui-table__row--action" : undefined}
                onClick={clickable ? () => onRowClick(r, i) : undefined}>
                {columns.map((c, j) => (
                  <td key={j} style={{ textAlign: c.align }}>{c.cell(r, i)}</td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
```

Add the pointer affordance to `src/ui/ui.css` while retaining the table's existing hover surface:

```css
.ui-table__row--action { cursor: pointer; }
.ui-table__row--action:focus-within { outline: none; box-shadow: inset 3px 0 0 var(--accent); }
```

The FASRC action cell added in Step 6 contains a native button, so keyboard users activate the same bubbling row action with Enter or Space.

- [x] **Step 6: Replace the appended status monitor with the raw-log detail card**

In `src/pages/Fasrc.tsx`, remove the `SlurmMonitor` import, add `useEffect` and `useRef`, import the tested helpers, extend `RunRow` with `out_path` / `err_path`, and define the log response:

```tsx
import { useEffect, useRef, useState } from "react";
import { ConnectionBar, CurrentSubmission } from "../fasrc";
import {
  buildLogPageUrl, logPath, preferredLogKind, type LogKind,
} from "./fasrcLogs";

type RunRow = {
  name: string; jobid?: string | null; label?: string | null; state?: string | null;
  submitted_at?: number; out_size?: number; err_size?: number; missing?: boolean;
  out_path?: string | null; err_path?: string | null;
};
type LogResp = {
  ok: boolean; path: string; page: number; page_size: number;
  total_lines: number; start_line: number; end_line: number;
  has_older: boolean; has_newer: boolean; content: string;
};
```

Add the detail component before `LogsTab`:

```tsx
const LOG_PAGE_SIZE = 1000;

function RunLogDetail({ run, onBack }: { run: RunRow; onBack: () => void }) {
  const [which, setWhich] = useState<LogKind>(() => preferredLogKind(run) ?? "out");
  const [page, setPage] = useState(0);
  const contentRef = useRef<HTMLPreElement>(null);
  const path = logPath(run, which);
  const url = path ? buildLogPageUrl(path, page, LOG_PAGE_SIZE) : null;
  const log = useResource<LogResp>(url, [path, page], { ttl: 0 });
  const data = log.data?.path === path ? log.data : null;

  useEffect(() => {
    const pre = contentRef.current;
    if (!pre || !data) return;
    pre.scrollTop = page === 0 ? pre.scrollHeight : 0;
  }, [data, page]);

  function show(kind: LogKind) {
    if (!logPath(run, kind)) return;
    setWhich(kind);
    setPage(0);
  }

  return (
    <Card>
      <CardHead title="Run logs"
        sub={<><code className="mono">{run.name}</code>{run.jobid ? ` · job ${run.jobid}` : ""}</>}
        right={<div className="fasrc-log__toolbar">
          <Button size="sm" variant="ghost" onClick={onBack}>← Back to past runs</Button>
          <Button size="sm" variant={which === "out" ? "primary" : "default"}
            disabled={!run.out_path} onClick={() => show("out")}>.out</Button>
          <Button size="sm" variant={which === "err" ? "primary" : "default"}
            disabled={!run.err_path} onClick={() => show("err")}>.err</Button>
        </div>} />
      <CardBody>
        {log.loading && !data ? <Empty><Spinner /> loading {which}…</Empty>
          : log.error && !data ? <Empty>couldn't load this log</Empty>
          : <pre ref={contentRef} className="ui-logtail fasrc-log__content">{data?.content || "(empty)"}</pre>}
        {data && <div className="fasrc-log__pager">
          <Button size="sm" variant="ghost" disabled={!data.has_older}
            onClick={() => setPage((p) => p + 1)}>← older</Button>
          <Button size="sm" variant="ghost" disabled={!data.has_newer}
            onClick={() => setPage((p) => Math.max(0, p - 1))}>newer →</Button>
          <Button size="sm" variant="ghost" disabled={!data.has_newer}
            onClick={() => setPage(0)}>newest ↓</Button>
          <span className="muted mono">
            {data.total_lines ? `lines ${data.start_line}–${data.end_line} of ${data.total_lines}` : "empty file"}
          </span>
        </div>}
      </CardBody>
    </Card>
  );
}
```

Update `LogsTab` so list selection swaps the card rather than appending `SlurmMonitor`:

```tsx
function LogsTab({ connected }: { connected: boolean }) {
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<RunRow | null>(null);
  const runs = useResource<RunsResp>(connected ? `/api/fasrc/runs?page=${page}` : null, [connected, page]);

  function openRun(run: RunRow) {
    if (preferredLogKind(run)) setSelected(run);
  }

  const cols: Column<RunRow>[] = [
    { header: "job", cell: (r) => <code className="mono">{r.jobid ?? "—"}</code> },
    { header: "label", cell: (r) => r.label ?? r.name },
    { header: "state", cell: (r) => r.state ? <Badge tone={stateTone(r.state)}>{r.state}</Badge> : <span className="muted">—</span> },
    { header: "", align: "right", cell: (r) => preferredLogKind(r)
      ? <Button size="sm">open logs →</Button>
      : <span className="muted">no files</span> },
  ];

  if (!connected) return <Card><CardBody><Empty>connect to FASRC to browse past runs</Empty></CardBody></Card>;
  if (selected) return <RunLogDetail run={selected} onBack={() => setSelected(null)} />;
  const d = runs.data;
  const hasOlder = d ? (d.page + 1) * d.page_size < d.total_runs : false;
  return (
    <Card>
      <CardHead title="Past runs" sub={d ? `${d.total_runs} total` : undefined}
        right={<div className="row" style={{ gap: 8 }}>
          <Button size="sm" variant="ghost" disabled={page === 0} onClick={() => setPage((p) => Math.max(0, p - 1))}>← newer</Button>
          <span className="muted mono" style={{ fontSize: 12 }}>pg {page + 1}</span>
          <Button size="sm" variant="ghost" disabled={!hasOlder} onClick={() => setPage((p) => p + 1)}>older →</Button>
          <Button size="sm" variant="ghost" onClick={() => runs.reload()}>↻</Button>
        </div>} />
      <CardBody>
        <Table columns={cols} rows={d?.runs ?? []} rowKey={(r) => r.name}
          onRowClick={openRun} isRowClickable={(r) => preferredLogKind(r) != null}
          empty={runs.loading ? "loading…" : "no past runs found"} />
      </CardBody>
    </Card>
  );
}
```

Add the detail layout rules to `src/ui/ui.css`:

```css
.fasrc-log__toolbar { display: flex; align-items: center; flex-wrap: wrap; gap: var(--s2); }
.fasrc-log__content { min-height: 280px; max-height: min(68vh, 760px); }
.fasrc-log__pager {
  display: flex; align-items: center; flex-wrap: wrap; gap: var(--s2);
  margin-top: var(--s3); font-size: 12px;
}
```

- [x] **Step 7: Run focused tests and build the production SPA**

Run:

```bash
cd euclid_polish/web/frontend
npm test
npm run build
```

Expected: 3 Node tests pass; Vite exits 0 and writes the production bundle to `euclid_polish/web/static/dist`.

- [x] **Step 8: Verify the interaction in the running app**

Open `http://127.0.0.1:8765/app/fasrc`, choose Logs, and verify:

1. Clicking an actionable row or its `open logs` button replaces the table in place.
2. `.out` is selected first; `.err` switches content and resets to the newest page.
3. The Back action restores the same runs-list page.
4. Log scrolling remains inside the card.
5. Run and log pagination buttons enable and disable according to the API response.
6. A run without either file remains visible and does not look clickable.

Expected: all six checks pass in the current theme and at a narrow viewport.

- [x] **Step 9: Commit the implementation without unrelated user work**

Run:

```bash
git add docs/superpowers/plans/2026-07-13-fasrc-log-master-detail.md \
  euclid_polish/web/frontend/package.json \
  euclid_polish/web/frontend/src/pages/fasrcLogs.ts \
  euclid_polish/web/frontend/test/fasrcLogs.test.mjs \
  euclid_polish/web/frontend/src/pages/Fasrc.tsx \
  euclid_polish/web/frontend/src/ui/index.tsx \
  euclid_polish/web/frontend/src/ui/ui.css \
  euclid_polish/web/static/dist
git commit -m "feat: open fasrc run logs in place"
```

Expected: commit succeeds and every unrelated pre-existing modification remains unstaged.
