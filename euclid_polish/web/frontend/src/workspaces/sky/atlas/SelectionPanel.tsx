/* The region selection: every visible feature inside the drawn region, as a
 * DataTable (row → inspector), with bulk actions on its real tiles: Compare
 * models and Run production (both visible, each confirmed before anything
 * runs), overlay LR / SR on the sky and copy (the ⋯ menu). */
import { useMemo, useState } from "react";
import { formatCount, formatDec, formatRA } from "../../../format";
import {
  Badge, Button, DataTable, IconButton, Menu, copyText, toast, type DataColumn, type MenuItem,
} from "../../../ui";
import { coordText, runModels } from "./actions";
import { useCompareModels, useShowOnSky } from "./engineHooks";
import { tileRefOf, type LayerInfo, type SkyFeature } from "./layerModel";
import { MAX_PIXEL_OVERLAYS, tierLabel } from "./pixelOverlays";
import { tileRefsOf, type SelectionGroup } from "./selection";

type Row = { f: SkyFeature; layer: string; key: string };

export function SelectionPanel({ groups, layers, onClear }: {
  groups: readonly SelectionGroup[];
  layers: readonly LayerInfo[];
  onClear: () => void;
}) {
  const showOnSky = useShowOnSky();
  const [open, setOpen] = useState(true);
  const label = useMemo(() => new Map(layers.map((l) => [l.id, l.label])), [layers]);
  const rows = useMemo<Row[]>(() => groups.flatMap((g) => g.features.map((f) => ({ f, layer: g.layer, key: `${g.layer}/${f.key}` }))), [groups]);
  const total = groups.reduce((n, g) => n + g.total, 0);
  const refs = useMemo(() => tileRefsOf(groups), [groups]);
  const modelSpecs = useMemo(() => {
    const s = new Set<string>();
    for (const r of rows) if (tileRefOf(r.f) && Array.isArray(r.f.props.models)) for (const m of r.f.props.models as unknown[]) s.add(String(m));
    return [...s].sort();
  }, [rows]);

  const oneLayer = groups.length <= 1;
  const columns = useMemo<DataColumn<Row>[]>(() => [
    { id: "label", header: "Feature", accessor: (r) => r.f.label },
    { id: "state", header: "State", accessor: (r) => String(r.f.props.state ?? r.f.props.grade ?? ""), width: 72 },
    { id: "layer", header: "Layer", accessor: (r) => label.get(r.layer) ?? r.layer, width: 120, hidden: oneLayer },
    {
      id: "pos", header: "Position", accessor: (r) => r.f.ra, numeric: true, width: 150,
      cell: (r) => <span className="mono">{formatRA(r.f.ra, { digits: 0 })} {formatDec(r.f.dec, { digits: 0 })}</span>,
      csv: (r) => `${r.f.ra} ${r.f.dec}`,
    },
  ], [label, oneLayer]);

  const compareModels = useCompareModels();
  const compare = () => compareModels(refs);
  const overlay = (tier: string) => {
    const withTier = rows.filter((r) => {
      const ref = tileRefOf(r.f);
      if (!ref) return false;
      if (tier === "lr") return true;
      if (tier === "jwst") return r.f.props.has_jwst === true;
      return Array.isArray(r.f.props.models) && (r.f.props.models as unknown[]).map(String).includes(tier.slice(2));
    }).slice(0, MAX_PIXEL_OVERLAYS);
    if (!withTier.length) { toast.warning(`No selected tile has ${tierLabel(tier)}`); return; }
    const band = tier === "jwst" ? "" : "VIS";
    showOnSky({ overlays: withTier.map((r) => ({ ref: tileRefOf(r.f)!, tier, band })) });
    toast.success(`Overlaid ${withTier.length} tile${withTier.length === 1 ? "" : "s"} (${tierLabel(tier)} · VIS)`);
  };
  const items: MenuItem[] = [
    {
      type: "sub", label: "Overlay on the sky", disabled: !refs.length, items: [
        { label: "LR (VIS)", onSelect: () => overlay("lr") },
        { label: "JWST", onSelect: () => overlay("jwst") },
        ...modelSpecs.map((m) => ({ label: `SR · ${m} (VIS)`, onSelect: () => overlay(`m:${m}`) })),
      ],
    },
    { type: "separator" },
    { label: "Copy tile refs", disabled: !refs.length, onSelect: () => { void copyText(refs.join("\n")).then((ok) => ok && toast.success(`Copied ${refs.length} refs`)); } },
    { label: "Copy coordinates", onSelect: () => { void copyText(rows.map((r) => coordText(r.f.ra, r.f.dec)).join("\n")).then((ok) => ok && toast.success(`Copied ${rows.length} positions`)); } },
    { type: "separator" },
    { label: "Clear selection", onSelect: onClear },
  ];

  return (
    <section className="sky-selection" data-open={open} aria-label="Region selection">
      <header className="sky-selection__head">
        <IconButton icon={open ? "chevronDown" : "chevronUp"} size="sm" label={open ? "Collapse the selection" : "Expand the selection"}
          onClick={() => setOpen(!open)} />
        <strong>{formatCount(total)} in selection</strong>
        {refs.length > 0 && <Badge size="sm" tone="accent">{refs.length} real tile{refs.length === 1 ? "" : "s"}</Badge>}
        <span className="sky-toolbar__spacer" />
        <Button size="sm" variant="primary" disabled={!refs.length} onClick={compare}>Compare models…</Button>
        <Button size="sm" disabled={!refs.length} onClick={() => { void runModels(refs, ["production"]); }}>Run production…</Button>
        <Menu label="Selection actions" items={items} trigger={<IconButton icon="more" size="sm" label="Selection actions" />} />
        <IconButton icon="close" size="sm" label="Clear selection" onClick={onClear} />
      </header>
      {open && (
        total === 0 ? <p className="muted sky-selection__empty">Nothing visible inside the region. Turn on more layers or draw a larger region.</p> : (
          <DataTable rows={rows} columns={columns} rowKey={(r) => r.key} aria-label="Selected features" dense
            height={180} exportName="sky-selection" inspect={(r) => r.f.inspect} />
        )
      )}
    </section>
  );
}
