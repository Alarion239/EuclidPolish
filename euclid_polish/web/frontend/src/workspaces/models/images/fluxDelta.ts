/* SR's flux change against LR for the object a viewer shows (Models ›
 * Images footer): the two cubes come from the viewer's shared cube cache
 * (same keys as the viewer's own requests, so a shown tier is never fetched
 * twice), summed in the viewer's band (model.ts cubeDeltaMag). Reads only
 * cached-cube GETs; never starts a job. */
import { useEffect, useState } from "react";
import { cubeKey, cubeUrl, sharedCubeCache } from "../../../viewer/cube";
import { cubeDeltaMag } from "../model";

export type FluxDelta = { band: string; text: string; warn: boolean };

export function useFluxDelta(collection: string, params: Record<string, string>, index: number | null,
  tiers: readonly [string, string] | null, color: string): FluxDelta | null {
  const [out, setOut] = useState<FluxDelta | null>(null);
  const paramKey = JSON.stringify(params);
  const tierKey = tiers ? tiers.join(",") : "";
  useEffect(() => {
    if (index == null || index < 0 || !tiers) { setOut(null); return undefined; }
    const ctl = new AbortController();
    const load = (tier: string) => sharedCubeCache.load(cubeKey(collection, tier, index, params), cubeUrl(collection, index, tier, params), ctl.signal);
    Promise.all([load(tiers[0]), load(tiers[1])])
      .then(([lr, sr]) => { if (!ctl.signal.aborted) setOut(cubeDeltaMag(lr, sr, color)); })
      .catch(() => { if (!ctl.signal.aborted) setOut(null); });
    return () => ctl.abort();
  }, [collection, paramKey, index, tierKey, color]); // eslint-disable-line react-hooks/exhaustive-deps
  return out;
}
