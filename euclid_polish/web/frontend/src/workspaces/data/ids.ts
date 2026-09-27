/* Ids of the Data workspace's viewer objects and inspector kinds (pure, no
 * imports): register.ts is loaded eagerly by the shell, so it reads these
 * from here rather than from the whole of model.ts (which re-exports them). */

/** `sky` viewer object id of a record: `"<split>:<index>"`. */
export const recordObjectId = (split: string, index: number) => `${split}:${index}`;

/** `truth:<split>/<index>/<row>` — one synthetic truth source. */
export const truthId = (split: string, index: number, row: number) => `${split}/${index}/${row}`;
export function parseTruthId(id: string): { split: string; index: number; row: number } | null {
  const m = /^(test|validate|train)\/(\d+)\/(\d+)$/.exec(id);
  return m ? { split: m[1], index: Number(m[2]), row: Number(m[3]) } : null;
}

/** `psf:<cluster index>` (1-based, as `cluster-NNN`). */
export const clusterObjectId = (index: number) => `cluster-${String(index).padStart(3, "0")}`;
export function parseClusterId(id: string): number | null {
  const m = /^(?:cluster-)?0*(\d+)$/.exec(id.trim());
  return m ? Number(m[1]) : null;
}
