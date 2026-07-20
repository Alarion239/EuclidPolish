import assert from "node:assert/strict";
import test from "node:test";

const helpers = await import("../src/pages/fasrcLogs.ts").catch(() => ({}));
const preferredLogKind = helpers.preferredLogKind ?? (() => undefined);
const logPath = helpers.logPath ?? (() => undefined);
const hasRunLogs = helpers.hasRunLogs ?? (() => undefined);
const buildLogPageUrl = helpers.buildLogPageUrl ?? (() => undefined);

test("prefers stdout and falls back to stderr", () => {
  assert.equal(
    preferredLogKind({ out_path: "/logs/a.out", err_path: "/logs/a.err" }),
    "out",
  );
  assert.equal(preferredLogKind({ err_path: "/logs/a.err" }), "err");
  assert.equal(preferredLogKind({}), null);
  assert.equal(
    preferredLogKind({ out_path: "/logs/late.out", missing: true }),
    "out",
    "a DB path remains openable even when the directory scan missed it",
  );
});

test("selects the requested file without crossing streams", () => {
  const files = { out_path: "/logs/a.out", err_path: "/logs/a.err" };
  assert.equal(logPath(files, "out"), "/logs/a.out");
  assert.equal(logPath(files, "err"), "/logs/a.err");
  assert.equal(logPath({ out_path: "/logs/a.out" }, "err"), null);
});

test("recognizes logs nested under an array parent", () => {
  assert.equal(hasRunLogs({ out_path: "/logs/single.out" }), true);
  assert.equal(hasRunLogs({ tasks: [
    { out_path: "/logs/array-1_0.out" },
    { err_path: "/logs/array-1_1.err" },
  ] }), true);
  assert.equal(hasRunLogs({ tasks: [] }), false);
});

test("builds an encoded paginated log URL", () => {
  assert.equal(
    buildLogPageUrl("/repo/logs/jobs/a b.out", 2, 1000),
    "/api/fasrc/runs/log?path=%2Frepo%2Flogs%2Fjobs%2Fa+b.out&page=2&page_size=1000",
  );
});
