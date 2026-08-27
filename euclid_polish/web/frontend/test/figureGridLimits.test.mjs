import assert from "node:assert/strict";
import test from "node:test";

import {
  canAddGridResult,
  canAddGridRow,
  MAX_GRID_RESULTS,
  MAX_GRID_ROWS,
  selectionForPreset,
  toggleGridResult,
} from "../src/pages/figure-grid/limits.ts";

test("enforces the saved-result column limit while still allowing removal", () => {
  const selected = Array.from({ length: MAX_GRID_RESULTS }, (_, index) => `result-${index}`);

  assert.equal(canAddGridResult(selected.length), false);
  assert.strictEqual(toggleGridResult(selected, "result-overflow"), selected);
  assert.deepEqual(toggleGridResult(selected, "result-5"), selected.filter((id) => id !== "result-5"));
  assert.deepEqual(toggleGridResult(["one", "two"], "three", 2), ["one", "two"]);
});

test("keeps an intentional empty selection when changing presets", () => {
  assert.deepEqual(selectionForPreset([], ["compatible-a", "compatible-b"]), []);
});

test("preserves compatible preset columns and falls back after a regime change", () => {
  assert.deepEqual(
    selectionForPreset(["keep-b", "drop", "keep-a"], ["keep-a", "keep-b"]),
    ["keep-b", "keep-a"],
  );
  assert.deepEqual(
    selectionForPreset(["old-regime"], Array.from({ length: 8 }, (_, index) => `new-${index}`)),
    ["new-0", "new-1", "new-2", "new-3", "new-4"],
  );
  assert.deepEqual(
    selectionForPreset(["keep-a", "keep-b", "keep-c"], ["keep-a", "keep-b", "keep-c"], 2),
    ["keep-a", "keep-b"],
  );
});

test("reports recipe-row capacity at the backend limit", () => {
  assert.equal(canAddGridRow(MAX_GRID_ROWS - 1), true);
  assert.equal(canAddGridRow(MAX_GRID_ROWS), false);
  assert.equal(canAddGridRow(4, 4), false);
});
