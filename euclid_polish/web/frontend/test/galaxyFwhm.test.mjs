import assert from "node:assert/strict";
import test from "node:test";

import { conditionalFwhmInterval } from "../src/pages/galaxyFwhm.ts";

test("interpolates the 16th and 84th percentiles inside FWHM histogram bins", () => {
  const interval = conditionalFwhmInterval(
    [1, 1],
    [[0.5, 0.5], [0, 1]],
    [0, 1, 2],
  );

  assert.deepEqual(interval.low, [0.32, 1.16]);
  assert.deepEqual(interval.high, [1.68, 1.8399999999999999]);
});

test("shows intervals only for populated observed magnitude bins", () => {
  const interval = conditionalFwhmInterval(
    [null, 1, 1],
    [[0.5, 0.5], [0, 0], [1]],
    [0, 1, 2],
  );

  assert.deepEqual(interval.low, [null, null, null]);
  assert.deepEqual(interval.high, [null, null, null]);
});
