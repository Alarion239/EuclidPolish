import assert from "node:assert/strict";
import test from "node:test";

import { asArray } from "../src/data.ts";
import { curveRecords } from "../src/fasrcCurves.ts";

test("normalizes missing and malformed collection fields", () => {
  assert.deepEqual(asArray(undefined), []);
  assert.deepEqual(asArray({ length: 9 }), []);
  assert.deepEqual(asArray(["member_00"]), ["member_00"]);
});

test("uses only finite-step Reporter metrics as training curve records", () => {
  const valid = { step: 500, psnr_stretched: 42.1, member: 1 };
  assert.deepEqual(
    curveRecords([null, {}, { step: "500" }, { step: Number.NaN }, valid]),
    [valid],
  );
  assert.deepEqual(curveRecords(undefined), []);
});
