export type ArchiveAvailability = {
  available: boolean;
  valid: boolean;
  ready: boolean;
  complete: boolean;
  current: boolean;
  reasons: string[];
  sample_count: number;
  planned_sample_count: number;
  parent_count: number;
  fields: Record<string, number>;
  comparison_sample_count?: number;
  comparison_fields?: Record<string, number>;
  comparison_excluded_positions?: string[];
  bands: string[];
  tile_size: number;
  manifest_fingerprint: string | null;
  collection_fingerprint: string | null;
  source_release: string | null;
  source_plan_fingerprint: string | null;
  source_manifest_sha256: string | null;
};

export type ArchiveObject = {
  /** Viewer object id (meta `objects[i].id`, the sample id as a string). */
  id?: string;
  label: string;
  tiers: string[];
  sample_id: number;
  source_sample_id: number;
  parent_id: string;
  field: string;
  ra: number;
  dec: number;
  position_name: string;
};

export type ArchiveCollectionMeta = {
  count: number;
  archive?: ArchiveAvailability;
  objects?: ArchiveObject[];
};

export function archiveOverview(status?: ArchiveAvailability): string {
  if (!status?.ready) {
    return status?.reasons?.[0] ?? "Multipoint archive samples are not synchronized.";
  }
  const release = status.source_release ? ` · ${status.source_release}` : "";
  const compared = status.comparison_sample_count ?? status.sample_count;
  const excluded = status.sample_count - compared;
  const note = excluded > 0
    ? ` (${excluded.toLocaleString()} star-avoiding centre tiles left out)`
    : "";
  return `${status.parent_count.toLocaleString()} independent parent pointings · `
    + `${compared.toLocaleString()} four-band samples${note}${release}`;
}

export function archiveFieldBreakdown(status?: ArchiveAvailability): string {
  if (!status?.ready) return "";
  return Object.entries(status.comparison_fields ?? status.fields)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([field, count]) => `${field} ${count}`)
    .join(" · ");
}

/** A sample's identity (1-based archive ids). The position in the
 *  collection is the viewer's own navigation readout, not repeated here. */
export function archiveSampleProvenance(sample: ArchiveObject | undefined, count: number): string {
  if (!sample) return `${count} samples`;
  return `archive sample ${sample.sample_id + 1} · source pointing ${sample.source_sample_id + 1}`
    + ` · ${sample.field} · ${sample.position_name}`;
}

export function shortArchiveFingerprint(value: string | null | undefined): string {
  return value ? `${value.slice(0, 12)}…` : "unknown";
}
