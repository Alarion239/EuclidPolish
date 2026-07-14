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

test("makes older member-local lens-isolation metrics cumulative", () => {
  const oldEvents = [
    { member: 1, step: 49500, total: 50000, psnr_stretched: 51 },
    { member: 1, step: 50000, total: 50000, psnr_stretched: 52 },
    { member: 2, step: 500, total: 50000, psnr_stretched: 40 },
    { member: 2, step: 16500, total: 50000, psnr_stretched: 53 },
    { member: 3, step: 500, total: 50000, psnr_stretched: 41 },
  ];
  assert.deepEqual(
    curveRecords(oldEvents).map((r) => r.step),
    [49500, 50000, 50500, 66500, 100500],
  );

  const cumulative = [
    { member: 1, step: 50000, total: 150000 },
    { member: 2, step: 50500, total: 150000 },
  ];
  assert.deepEqual(curveRecords(cumulative).map((r) => r.step), [50000, 50500]);
});
