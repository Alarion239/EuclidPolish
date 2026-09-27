/* Shared pieces of the atlas inspector cards. */
import type { ReactNode } from "react";
import { formatDec, formatDeg, formatRA } from "../../../../format";
import { CopyButton, IconButton, Menu, copyText, toast, type MenuItem } from "../../../../ui";
import { coordText, esaskyUrl, simbadUrl } from "../actions";

export function PositionValue({ ra, dec }: { ra: number; dec: number }) {
  return (
    <span className="sky-card__pos">
      <span className="mono">{formatRA(ra)} {formatDec(dec)}</span>
      <span className="mono muted">{formatDeg(ra, 5)} {formatDeg(dec, 5, { signed: true })}</span>
      <CopyButton value={() => coordText(ra, dec)} label="Copy coordinates (degrees)" />
    </span>
  );
}

const copy = async (text: string, what: string) => { if (await copyText(text)) toast.success(`Copied ${what}`); };

/** External + copy entries for a position (the card's "more" menu). */
export function positionMenuItems(ra: number, dec: number, fovDeg = 0.05): MenuItem[] {
  return [
    { label: "Open in ESASky", onSelect: () => window.open(esaskyUrl(ra, dec, fovDeg), "_blank", "noopener") },
    { label: "Open in SIMBAD", onSelect: () => window.open(simbadUrl(ra, dec), "_blank", "noopener") },
    { type: "separator" },
    { label: "Copy coordinates (degrees)", onSelect: () => { void copy(coordText(ra, dec), "coordinates"); } },
    { label: "Copy coordinates (sexagesimal)", onSelect: () => { void copy(coordText(ra, dec, "sex"), "coordinates"); } },
  ];
}

export function MoreMenu({ items, label = "More actions" }: { items: MenuItem[]; label?: string }) {
  return <Menu label={label} items={items} align="end" trigger={<IconButton icon="more" size="sm" label={label} />} />;
}

export function CardActions({ children }: { children: ReactNode }) {
  return <div className="sky-card__actions">{children}</div>;
}

/** A props value for a DefList. */
export function propValue(v: unknown): ReactNode {
  if (v == null || v === "") return "—";
  if (typeof v === "number") return <span className="mono">{Number.isInteger(v) ? v : Number(v.toPrecision(5))}</span>;
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (Array.isArray(v)) {
    if (v.every((x) => typeof x === "number")) return <span className="mono">{v.map((x) => Number((x as number).toPrecision(4))).join(", ")}</span>;
    return v.length <= 6 && v.every((x) => typeof x === "string") ? v.join(", ") : `${v.length} items`;
  }
  if (typeof v === "object") return <code className="mono">{JSON.stringify(v).slice(0, 80)}</code>;
  return String(v);
}

export const splitRef = (ref: string): [string, string] => {
  const i = ref.indexOf("/");
  return i < 0 ? [ref, ""] : [ref.slice(0, i), ref.slice(i + 1)];
};
