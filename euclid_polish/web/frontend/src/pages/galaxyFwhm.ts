export type ConditionalFwhmInterval = {
  low: Array<number | null>;
  high: Array<number | null>;
};

function binnedUniformQuantile(
  probability: number[],
  edges: number[],
  quantile: number,
): number | null {
  if (!(quantile >= 0 && quantile <= 1) || edges.length !== probability.length + 1) {
    return null;
  }
  for (let index = 0; index < edges.length; index++) {
    if (!Number.isFinite(edges[index]) || (index > 0 && edges[index] <= edges[index - 1])) {
      return null;
    }
  }
  const total = probability.reduce(
    (sum, value) => Number.isFinite(value) && value >= 0 ? sum + value : Number.NaN,
    0,
  );
  if (!(total > 0) || !Number.isFinite(total)) return null;

  const target = quantile * total;
  let cumulative = 0;
  for (let index = 0; index < probability.length; index++) {
    const mass = probability[index];
    const next = cumulative + mass;
    if (target <= next && mass > 0) {
      const fraction = Math.min(1, Math.max(0, (target - cumulative) / mass));
      return edges[index] + fraction * (edges[index + 1] - edges[index]);
    }
    cumulative = next;
  }
  return edges[edges.length - 1];
}

/**
 * Return the central conditional FWHM interval represented by the empirical
 * histogram. Null observed means deliberately mask nearest-bin continuation:
 * only directly populated Q1 magnitude bins receive blue error bars.
 */
export function conditionalFwhmInterval(
  observedMean: Array<number | null>,
  probability: number[][],
  edges: number[],
  lowerQuantile = 0.16,
  upperQuantile = 0.84,
): ConditionalFwhmInterval {
  const low: Array<number | null> = [];
  const high: Array<number | null> = [];
  for (let index = 0; index < observedMean.length; index++) {
    const observed = observedMean[index];
    const row = probability[index];
    if (observed == null || !Number.isFinite(observed) || row == null) {
      low.push(null);
      high.push(null);
      continue;
    }
    low.push(binnedUniformQuantile(row, edges, lowerQuantile));
    high.push(binnedUniformQuantile(row, edges, upperQuantile));
  }
  return { low, high };
}
