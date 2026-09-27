/* Which workspace tabs fit on one line (the "More" overflow of the tab strip).
 *
 * The strip never scrolls, wraps or cuts a label, and tabs never trade
 * places: a fixed run of leading tabs is shown whole, in order, followed by
 * ONE reserved slot and the "More" menu. The run is the longest prefix that
 * still leaves room for the widest tab after it, so it does not depend on
 * which tab is active; the slot shows the active tab when it is past the run
 * (and stays empty otherwise). Choosing a tab from More therefore only ever
 * changes what is in the slot. Pure so the maths is unit-tested;
 * `WorkspaceTabs` measures the tabs. */

/** Sub-pixel slack for widths read from `getBoundingClientRect`. */
const EPS = 0.5;

/**
 * Indices of the tabs to show, in strip order.
 *
 * @param widths    each tab's width in px, in strip order
 * @param available the strip's width in px
 * @param moreWidth the "More" button's width, reserved only when a tab overflows
 * @param active    index of the active tab, or -1
 */
export function fitTabs(widths: readonly number[], available: number, moreWidth: number, active: number): number[] {
  const n = widths.length;
  const total = widths.reduce((s, w) => s + w, 0);
  if (total <= available + EPS) return widths.map((_, i) => i);

  const budget = available - moreWidth + EPS;
  // widest[k] = the widest tab at index ≥ k (the slot must hold any of them).
  const widest = new Array<number>(n + 1).fill(0);
  for (let i = n - 1; i >= 0; i--) widest[i] = Math.max(widths[i], widest[i + 1]);
  let lead = 0;
  let used = 0;
  for (let k = 1; k < n; k++) {
    const sum = used + widths[k - 1];
    if (sum + widest[k] > budget) break;
    lead = k;
    used = sum;
  }
  const shown = Array.from({ length: lead }, (_, i) => i);
  if (active >= lead && active < n) {
    // The slot fits any tab past the run; only a strip too narrow for even
    // one tab + the slot gives up leading tabs to keep the active one.
    while (shown.length && used + widths[active] > budget) used -= widths[shown.pop()!];
    shown.push(active);
  }
  return shown;
}

/** Index lists equal by value (`null` = not measured yet). */
export function sameIndices(a: readonly number[] | null, b: readonly number[] | null): boolean {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  return a.every((v, i) => v === b[i]);
}
