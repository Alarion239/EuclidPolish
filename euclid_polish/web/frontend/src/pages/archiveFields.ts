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
  bands: string[];
  tile_size: number;
  manifest_fingerprint: string | null;
  collection_fingerprint: string | null;
  source_release: string | null;
  source_plan_fingerprint: string | null;
  source_manifest_sha256: string | null;
};

export type ArchiveObject = {
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
  return `${status.parent_count.toLocaleString()} independent parent pointings · `
    + `${status.sample_count.toLocaleString()} four-band samples${release}`;
}

export function archiveFieldBreakdown(status?: ArchiveAvailability): string {
  if (!status?.ready) return "";
  return Object.entries(status.fields)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([field, count]) => `${field} ${count}`)
    .join(" · ");
}

export function archiveSampleProvenance(
  sample: ArchiveObject | undefined,
  index: number,
  count: number,
): string {
  if (!sample) return `sample ${Math.min(index + 1, Math.max(count, 1))} / ${count}`;
  return `sample ${sample.sample_id + 1} / ${count} · source pointing `
    + `${sample.source_sample_id + 1} · ${sample.field} · ${sample.position_name}`;
}

export function shortArchiveFingerprint(value: string | null | undefined): string {
  return value ? `${value.slice(0, 12)}…` : "unknown";
}
