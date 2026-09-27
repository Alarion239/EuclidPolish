/* Lazy layer payloads: `GET /api/sky/layer/<id>` is fetched the first time a
 * layer is switched on (shared TanStack cache — the source inspector reads
 * the same entries) and normalised once per fetch. */
import { useQueries, type UseQueryResult } from "@tanstack/react-query";
import { useMemo } from "react";
import { apiGet, ApiError } from "../../../api/client";
import { queryClient, resourceKey } from "../../../api/query";
import { typicalSizeDeg } from "../../../sky/lod";
import { CLIENT_LAYERS, normalisePayload, stubLayerInfo, type LayerInfo, type LayerPayload, type SkyFeature } from "./layerModel";

const CLIENT_IDS = new Set(CLIENT_LAYERS.map((l) => l.id));

export type LayerData = {
  features: SkyFeature[];
  /** Changes on every new payload (the fetch time). */
  version: number;
  /** Median feature diameter (deg), for the level of detail. */
  typicalSize: number;
  loading: boolean;
  fetching: boolean;
  error: ApiError | null;
  payload: LayerPayload | null;
};

const EMPTY: SkyFeature[] = [];
const cache = new Map<string, { at: number; features: SkyFeature[]; typical: number }>();

function normalised(url: string, at: number, payload: LayerPayload): { features: SkyFeature[]; typical: number } {
  const hit = cache.get(url);
  if (hit && hit.at === at) return hit;
  const features = normalisePayload(payload);
  const typical = typicalSizeDeg(features.map((f) => f.sizeDeg));
  const entry = { at, features, typical };
  cache.set(url, entry);
  return entry;
}

export const layerUrl = (info: Pick<LayerInfo, "id" | "url">) => info.url ?? `/api/sky/layer/${encodeURIComponent(info.id)}`;

type Slim = { data: LayerPayload | undefined; at: number; pending: boolean; fetching: boolean; error: unknown };

/* Stable (module-level) so TanStack keeps the combined array referentially
   stable while no query result changes. */
const combine = (results: UseQueryResult<LayerPayload>[]): Slim[] => results.map((r) => ({
  data: r.data, at: r.dataUpdatedAt, pending: r.isPending, fetching: r.isFetching, error: r.error,
}));

/** Data of the enabled server layers (client MOC layers need none). With
 *  `eager` (the catalogue has not arrived yet), an enabled id the catalogue
 *  does not list is fetched at its default URL anyway (`stubLayerInfo`). */
export function useLayerData(layers: readonly LayerInfo[], enabled: readonly string[], eager = false): Record<string, LayerData> {
  const targets = useMemo(
    () => enabled
      .map((id) => layers.find((l) => l.id === id) ?? (eager && !CLIENT_IDS.has(id) ? stubLayerInfo(id) : undefined))
      .filter((l): l is LayerInfo => !!l && !l.client),
    [layers, enabled, eager],
  );
  const results = useQueries({
    queries: targets.map((l) => ({
      queryKey: resourceKey(layerUrl(l)),
      queryFn: ({ signal }: { signal: AbortSignal }) => apiGet<LayerPayload>(layerUrl(l), { signal }),
      staleTime: 60_000,
    })),
    combine,
  }, queryClient);
  return useMemo(() => {
    const out: Record<string, LayerData> = {};
    targets.forEach((l, i) => {
      const r = results[i];
      if (!r) return;
      const payload = r.data ?? null;
      const n = payload ? normalised(layerUrl(l), r.at, payload) : null;
      const err = r.error;
      out[l.id] = {
        features: n?.features ?? EMPTY,
        version: r.at,
        typicalSize: n?.typical ?? 0,
        loading: r.pending,
        fetching: r.fetching,
        error: payload || !err ? null : err instanceof ApiError ? err : new ApiError({ status: 0, message: String(err) }),
        payload,
      };
    });
    return out;
  }, [targets, results]);
}
