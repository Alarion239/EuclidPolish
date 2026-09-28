/* The atlas toolbar (never sticky; wraps at narrow widths): layers
 * toggle, go-to (a name via Sesame, or RA Dec), quick jumps, region
 * selection, the JWST menu (discover, cache the NEXUS mosaic, pair
 * downloads — each confirmed), PNG export and the all-sky view. Running
 * production on the tiles lives in Sky › Targets. */
import { useState } from "react";
import { currentSkyEngine } from "../../../sky/engine";
import { QUICK_JUMPS, type QuickJump } from "../../../sky/surveys";
import { Button, Chip, IconButton, Input, Menu, type MenuItem } from "../../../ui";
import { cacheNexusMosaic, discoverJwst, downloadAllPairs, viewRegion } from "./actions";
import { useAtlas } from "./store";
import type { AtlasUrl } from "./useAtlasUrl";
import { parseGoto } from "./urlState";

export const Q1_FIELD_NAMES = ["EDF-N", "EDF-S", "EDF-F", "LDN1641"] as const;

export function AtlasToolbar({ url, panelOpen, onTogglePanel, onSelect, onExport, onJump, onObservations }: {
  url: AtlasUrl;
  panelOpen: boolean;
  onTogglePanel: () => void;
  onSelect: (mode: "rect" | "circle" | "poly") => void;
  onExport: () => void;
  onJump: (j: QuickJump) => void;
  onObservations: () => void;
}) {
  const [text, setText] = useState("");
  const selecting = useAtlas((s) => s.selecting);
  const view = useAtlas((s) => s.view);
  const go = () => {
    const t = parseGoto(text);
    if (!t) return;
    if (t.kind === "coord") url.setView({ ra: t.ra, dec: t.dec, fov: view && view.fov < 10 ? view.fov : 0.5 });
    else url.setGoto(t.name);
  };
  const selectItems: MenuItem[] = [
    ...(selecting ? [{ label: "Cancel drawing", onSelect: () => currentSkyEngine()?.cancelSelect() }, { type: "separator" as const }] : []),
    { label: "Rectangle", onSelect: () => onSelect("rect") },
    { label: "Circle", onSelect: () => onSelect("circle") },
    { label: "Polygon", onSelect: () => onSelect("poly") },
    { type: "separator" },
    { label: "Clear selection", disabled: !url.sel, onSelect: () => url.setSel(null) },
  ];
  const jwstItems: MenuItem[] = [
    { label: "Discovered observations…", onSelect: onObservations },
    {
      label: "Discover in this view…", disabled: !view,
      onSelect: () => { if (view) void discoverJwst({ region: viewRegion(view), label: "the current view" }); },
    },
    {
      type: "sub", label: "Discover in a Q1 field", items: [
        ...Q1_FIELD_NAMES.map((f) => ({ label: f, onSelect: () => { void discoverJwst({ fields: f, label: f }); } })),
        { type: "separator" as const },
        { label: "All of Q1", onSelect: () => { void discoverJwst({ label: "all of Euclid Q1" }); } },
      ],
    },
    { type: "separator" },
    {
      type: "sub", label: "Cache the NEXUS mosaic", items: [
        { label: "F200W · 30 mas · ≈ 1 GB", onSelect: () => { void cacheNexusMosaic("F200W"); } },
        { label: "F444W · 60 mas · ≈ 250 MB", onSelect: () => { void cacheNexusMosaic("F444W"); } },
      ],
    },
    { label: "Download every discovered pair…", onSelect: () => { void downloadAllPairs(); } },
  ];
  return (
    <div className="sky-toolbar" role="toolbar" aria-label="Atlas tools">
      <IconButton icon="layers" label={panelOpen ? "Hide the layers panel" : "Show the layers panel"}
        pressed={panelOpen} onClick={onTogglePanel} />
      <div className="sky-toolbar__goto">
        <Input value={text} onChange={setText} onEnter={go} icon="search" clearable size="sm"
          placeholder="Object name or RA Dec" aria-label="Go to an object name or coordinates" />
      </div>
      <div className="sky-toolbar__jumps" role="group" aria-label="Quick jumps">
        {QUICK_JUMPS.map((j) => (
          <Chip key={j.id} onClick={() => onJump(j)} title={`Fly to ${j.label}`}>{j.label}</Chip>
        ))}
      </div>
      <span className="sky-toolbar__spacer" />
      <Menu label="Region selection" items={selectItems}
        trigger={<Button size="sm" variant={selecting || url.sel ? "subtle" : "default"} icon="filter">
          {selecting ? "Drawing…" : "Select"}
        </Button>} />
      <Menu label="JWST tools" items={jwstItems} trigger={<Button size="sm" icon="globe">JWST</Button>} />
      <IconButton icon="download" label="Export the view as PNG" onClick={onExport} />
      <IconButton icon="reset" label="All-sky view" onClick={() => url.setView({ ra: 165, dec: 0, fov: 360 })} />
    </div>
  );
}
