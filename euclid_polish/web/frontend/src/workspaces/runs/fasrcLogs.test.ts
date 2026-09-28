// @vitest-environment node
/* FASRC run-log helpers (moved from pages/fasrcLogs with the Ops workspace). */
import assert from "node:assert/strict";
import { test } from "vitest";

import {
  buildLogPageUrl,
  expandArrayPath,
  hasRunLogs,
  logPath,
  logTargetFromRow,
  preferredLogKind,
} from "./fasrcLogs";

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

test("expands array tokens", () => {
  assert.equal(expandArrayPath("/l/train-%A_%a.out", "123", 4), "/l/train-123_4.out");
  assert.equal(expandArrayPath(null, "1", 0), null);
});

test("builds log targets from a ledger row", () => {
  const single = logTargetFromRow({ jobid: "9", log_path: "/l/q-1.out", err_path: "/l/q-1.err", state: "COMPLETED" });
  assert.deepEqual([single.name, single.out_path, single.err_path, single.tasks], ["q-1", "/l/q-1.out", "/l/q-1.err", undefined]);
  const arr = logTargetFromRow({
    jobid: "77", log_path: "/l/t-%A_%a.out", err_path: "/l/t-%A_%a.err",
    params_json: JSON.stringify({ array_count: 2, mode: "continue", members: "member_01, member_02" }),
  });
  assert.equal(arr.tasks?.length, 2);
  assert.deepEqual(arr.tasks?.[1], { index: 1, member: "member_02", jobid: "77_1", name: "t-77_1",
    out_path: "/l/t-77_1.out", err_path: "/l/t-77_1.err" });
  assert.equal(hasRunLogs(arr), true);
});
