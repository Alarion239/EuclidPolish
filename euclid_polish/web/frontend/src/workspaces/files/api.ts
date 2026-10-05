/* Files workspace — typed endpoints (euclid_polish/web/API.md, "Files workspace": the /api/inspect URLs keep their name). */

export type InspectRoot = { id: string; label: string; path: string; rel: string; exists: boolean };

export type BrowseEntry = {
  name: string;
  rel: string;
  kind: "dir" | "fits" | "root";
  size: number | null;
  mtime: number | null;
  root_id?: string;
  exists?: boolean;
};

export type Crumb = { name: string; rel: string };

export type BrowseResponse = {
  dir: string | null;
  root: InspectRoot | null;
  crumbs: Crumb[];
  query: string;
  roots: InspectRoot[];
  entries: BrowseEntry[];
  other?: number;
  truncated: boolean;
  visited?: number;
};

export type WcsSummary = {
  ctype: string[];
  ra: number;
  dec: number;
  pixscale_arcsec: number;
  width_arcsec: number;
  height_arcsec: number;
  fov_deg: number;
  corners: [number, number][];
  constructed: boolean;
};

export type ColumnKind = "numeric" | "text" | "bool" | "array";

export type TableColumn = {
  name: string;
  format: string;
  unit: string | null;
  dim: string | null;
  null: unknown;
  kind?: ColumnKind;
};

export type HduType = "image" | "vector" | "table" | "empty" | "other";

export type HduSummary = {
  index: number;
  hdu_index: number;
  name: string;
  ver?: number;
  kind: string;
  type: HduType;
  shape: number[] | null;
  dtype: string | null;
  ndim?: number;
  planes?: number;
  plane_axes?: number[];
  bunit?: string;
  wcs?: WcsSummary | null;
  bands?: string[] | null;
  bands_assumed?: boolean;
  band?: string | null;
  compressed?: boolean;
  size_bytes?: number;
  viewable: boolean;
  reason?: string | null;
  scaling?: { bscale: number; bzero: number };
  columns?: TableColumn[];
  nrows?: number;
  ncols?: number;
  cards?: [string, string, string][];
};

export type BandGroup = {
  id: string;
  prefix: string;
  label: string;
  hdus: number[];
  bands: string[];
  shape: number[];
  wcs: WcsSummary | null;
  bunit: string;
};

export type Stamp = {
  id: string;
  produced_by: string | null;
  parents: string[];
  schema_version: number;
  subset?: string | null;
};

export type FileInfo = {
  abspath: string;
  basename: string;
  size: number;
  size_kb: number;
  mtime: number;
  compressed: boolean;
};

export type InspectResponse = {
  file: FileInfo;
  hdus: HduSummary[];
  band_groups: BandGroup[];
  scan_truncated: boolean;
  stamp: Stamp | null;
  rel: string;
  root: InspectRoot | null;
  allowed_roots: string[];
  roots: InspectRoot[];
};

export type Histogram = { edges: number[]; counts: number[]; below: number; above: number };

export type ArrayStats = {
  n: number;
  n_finite: number;
  n_nan: number;
  n_posinf: number;
  n_neginf: number;
  n_zero: number;
  n_negative: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  std: number | null;
  median: number | null;
  mad_std: number | null;
  sum: number | null;
  percentiles: Record<string, number>;
  histogram: Histogram | null;
};

export type ImageStats = ArrayStats & {
  hdu: number;
  plane?: number;
  index?: number[];
  shape?: number[];
  sampled?: number | null;
  /** 1-D HDUs: the (strided) values. */
  series?: { x: number[]; y: (number | null)[] };
  step?: number;
  n_points?: number;
};

export type TablePage = {
  hdu: number;
  total: number;
  offset: number;
  limit: number;
  sort: string | null;
  desc: boolean;
  columns: TableColumn[];
  rows: unknown[][];
  row_index: number[];
};

export type ColumnStats = {
  name: string;
  unit?: string | null;
  kind: ColumnKind;
  n: number;
  n_finite?: number;
  n_nan?: number;
  n_null?: number;
  min?: number | null;
  max?: number | null;
  mean?: number | null;
  std?: number | null;
  median?: number | null;
  mad_std?: number | null;
  percentiles?: Record<string, number>;
  histogram?: Histogram | null;
  n_unique?: number;
  n_true?: number;
  n_false?: number;
  n_empty?: number;
  top?: [string, number][];
  unique_sampled?: boolean;
  shape?: number[] | null;
};

export type TableStats = { hdu: number; total: number; sampled: number | null; columns: ColumnStats[] };

export type ProvRecord = Record<string, unknown>;

export type Sidecar = { file: string; id: string; kind: string; current: boolean; record: ProvRecord };

export type RelatedRecord = {
  role: "produced_by" | "parent";
  id: string;
  file: string | null;
  kind: string | null;
  record: ProvRecord | null;
  checkpoint?: string;
};

export type Provenance = {
  stamp: Stamp | null;
  sidecars: Sidecar[];
  related: RelatedRecord[];
  stale_sidecars: number;
};

const enc = encodeURIComponent;

function qs(params: Record<string, string | number | boolean | null | undefined>): string {
  const parts: string[] = [];
  for (const [k, v] of Object.entries(params)) {
    if (v == null || v === "" || v === false) continue;
    parts.push(`${enc(k)}=${enc(v === true ? "1" : String(v))}`);
  }
  return parts.length ? `?${parts.join("&")}` : "";
}

export const inspectUrl = (fits: string) => `/api/inspect${qs({ fits })}`;
export const browseUrl = (dir: string, q = "") => `/api/inspect/browse${qs({ dir, q: q.trim() })}`;
export const imageStatsUrl = (fits: string, hdu: number, plane = 0) =>
  `/api/inspect/image/stats${qs({ fits, hdu, plane })}`;
export const tableUrl = (fits: string, hdu: number, o: { offset?: number; limit?: number; sort?: string | null; desc?: boolean } = {}) =>
  `/api/inspect/table${qs({ fits, hdu, offset: o.offset || null, limit: o.limit ?? null, sort: o.sort ?? null, desc: o.desc ?? false })}`;
export const tableStatsUrl = (fits: string, hdu: number) => `/api/inspect/table/stats${qs({ fits, hdu })}`;
export const provenanceUrl = (fits: string) => `/api/inspect/provenance${qs({ fits })}`;
export const previewUrl = (fits: string, o: { hdu?: number | null; plane?: number; size?: number } = {}) =>
  `/inspect/preview.png${qs({ fits, hdu: o.hdu ?? null, plane: o.plane || null, size: o.size ?? 256 })}`;
export const downloadUrl = (fits: string) => `/inspect/download${qs({ fits })}`;

/** The Files page URL for a file (and optionally an HDU key). */
export function inspectPageHref(fits: string, hdu?: string | null): string {
  return `/files${qs({ fits, hdu: hdu ?? null })}`;
}
