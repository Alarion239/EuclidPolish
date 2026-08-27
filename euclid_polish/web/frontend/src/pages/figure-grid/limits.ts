export const MAX_GRID_RESULTS = 12;
export const MAX_GRID_ROWS = 16;
export const DEFAULT_PRESET_RESULT_COUNT = 5;

export function canAddGridResult(
  selectedCount: number,
  maximum = MAX_GRID_RESULTS,
): boolean {
  return selectedCount < maximum;
}

export function canAddGridRow(rowCount: number, maximum = MAX_GRID_ROWS): boolean {
  return rowCount < maximum;
}

export function toggleGridResult(
  current: string[],
  resultId: string,
  maximum = MAX_GRID_RESULTS,
): string[] {
  if (current.includes(resultId)) {
    return current.filter((item) => item !== resultId);
  }
  return canAddGridResult(current.length, maximum) ? [...current, resultId] : current;
}

export function selectionForPreset(
  current: string[],
  compatibleResultIds: string[],
  maximum = MAX_GRID_RESULTS,
): string[] {
  // An empty selection is meaningful: changing the recipe must not silently
  // repopulate columns after someone deliberately cleared the contact sheet.
  if (current.length === 0) return current;

  const compatible = new Set(compatibleResultIds);
  const preserved = current
    .filter((resultId) => compatible.has(resultId))
    .slice(0, maximum);
  if (preserved.length > 0) return preserved;
  return compatibleResultIds.slice(
    0,
    Math.min(DEFAULT_PRESET_RESULT_COUNT, maximum),
  );
}
