/* Sky › Targets › Sources: where the targets come from — the grouped
 * analysis (lens candidates + Q1 galaxies), the Q1 lens catalogue, the
 * real-galaxy query (the one Euclid archive login of System › Connections),
 * the FASRC sync of the evaluation results, dropping the evaluation
 * run's cached eye/solar PNGs and caching a 25.6″ tile anywhere in Q1.
 * Each opens its form, and every run is confirmed before anything starts. */
import { useState } from "react";
import { Link } from "react-router-dom";
import { useResource } from "../../../api/query";
import { formatDec, formatRA, parseSkyCoord } from "../../../format";
import {
  Button, Callout, Checkbox, Dialog, Field, Input, Menu, NumberField, type MenuItem,
} from "../../../ui";
import { cacheTile } from "../results/actions";
import { URLS } from "../results/api";
import { dropCachedPngs, fetchLensCatalogue, queryGalaxies, runGrouped, syncEvaluation } from "./actions";

type AuthStatus = { authenticated?: boolean; user?: string | null };
type Open = "grouped" | "galaxies" | "tile" | null;

const int = (raw: string, lo: number, hi: number): number | null => {
  const n = Number(raw);
  return Number.isInteger(n) && n >= lo && n <= hi ? n : null;
};

function GroupedDialog({ open, onOpenChange, defaultN }: { open: boolean; onOpenChange: (o: boolean) => void; defaultN: number }) {
  const [n, setN] = useState(String(defaultN));
  const [synthetic, setSynthetic] = useState(false);
  const [busy, setBusy] = useState(false);
  const value = int(n, 1, 500);
  const run = async () => {
    if (value == null) return;
    setBusy(true);
    try { if (await runGrouped(value, synthetic)) onOpenChange(false); } finally { setBusy(false); }
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange} title="Grouped analysis"
      description="Reconstructs the lens candidates and Q1 galaxies with the production model; current ones are reused."
      footer={<>
        <Button variant="ghost" onClick={() => onOpenChange(false)}>Cancel</Button>
        <Button variant="primary" loading={busy} disabled={value == null} onClick={() => { void run(); }}>Run…</Button>
      </>}>
      <div className="res-form">
        <NumberField label="Lens candidates per grade" value={n} onChange={setN} min={1} max={500}
          hint={value != null ? `and ${3 * value} Q1 galaxies` : "1–500"} />
        <Checkbox checked={synthetic} onChange={setSynthetic}>Also the synthetic stamps (Models › Images)</Checkbox>
      </div>
    </Dialog>
  );
}

function GalaxiesDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const auth = useResource<AuthStatus>(open ? URLS.authStatus : null, [], { ttl: 30_000 });
  const [n, setN] = useState("50");
  const [regen, setRegen] = useState(false);
  const [busy, setBusy] = useState(false);
  const loggedIn = !!auth.data?.authenticated;
  const value = int(n, 1, 2000);
  const run = async () => {
    if (value == null) return;
    setBusy(true);
    try { if (await queryGalaxies(value, regen)) onOpenChange(false); } finally { setBusy(false); }
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange} title="Query Q1 galaxies"
      description="Cone queries on the Euclid archive around the lens fields; the grouped analysis reconstructs what they return."
      footer={<>
        <Button variant="ghost" onClick={() => onOpenChange(false)}>Cancel</Button>
        <Button variant="primary" loading={busy} disabled={!loggedIn || value == null} onClick={() => { void run(); }}>Query…</Button>
      </>}>
      <div className="res-form">
        {auth.loading ? null : loggedIn ? <p className="res-note">Euclid archive: {auth.data?.user ?? "logged in"}</p> : (
          <Callout tone="warn" title="Not logged in to the Euclid archive">
            Log in once in <Link to="/system/connections" onClick={() => onOpenChange(false)}>System › Connections</Link>.
          </Callout>
        )}
        <NumberField label="Galaxies" value={n} onChange={setN} min={1} max={2000} />
        <Checkbox checked={regen} onChange={setRegen}>Discard the cached draw and query again</Checkbox>
      </div>
    </Dialog>
  );
}

function CacheTileDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const [text, setText] = useState("");
  const [run, setRun] = useState(true);
  const [busy, setBusy] = useState(false);
  const coord = parseSkyCoord(text);
  const bad = text.trim() !== "" && (!coord || coord.ra < 0 || coord.ra >= 360 || coord.dec < -90 || coord.dec > 90);
  const submit = async () => {
    if (!coord || bad) return;
    setBusy(true);
    try { const r = await cacheTile(coord.ra, coord.dec, { run }); if (r?.ok) onOpenChange(false); } finally { setBusy(false); }
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange} title="Cache a 25.6″ tile"
      description="Downloads VIS + NISP Y/J/H at a position in Euclid Q1 (checked against the Q1 tiles first)."
      footer={<>
        <Button variant="ghost" onClick={() => onOpenChange(false)}>Cancel</Button>
        <Button variant="primary" loading={busy} disabled={!coord || bad} onClick={() => { void submit(); }}>Cache tile…</Button>
      </>}>
      <div className="res-form">
        <Field label="Position" description={coord && !bad ? `${formatRA(coord.ra)} ${formatDec(coord.dec)}` : "RA Dec in degrees or hh:mm:ss ±dd:mm:ss"}
          error={bad ? "Not a sky position (RA 0–360°, Dec ±90°)" : undefined}>
          <Input value={text} onChange={setText} onEnter={() => { void submit(); }} placeholder="273.2309 68.3637" autoFocus />
        </Field>
        <Checkbox checked={run} onChange={setRun}>Then run production and the mean</Checkbox>
      </div>
    </Dialog>
  );
}

/** The Sources menu and its forms. `defaultN` sizes the grouped run to the
 *  store (every object is reached). */
export function SourcesMenu({ defaultN, busy }: { defaultN: number; busy?: boolean }) {
  const [open, setOpen] = useState<Open>(null);
  const close = (o: boolean) => { if (!o) setOpen(null); };
  const items: MenuItem[] = [
    { label: "Grouped analysis…", onSelect: () => setOpen("grouped") },
    { label: "Query Q1 galaxies…", onSelect: () => setOpen("galaxies") },
    { label: "Fetch the Q1 lens catalogue…", onSelect: () => { void fetchLensCatalogue(); } },
    { label: "Sync the evaluation results from FASRC…", onSelect: () => { void syncEvaluation(); } },
    { label: "Drop the cached eye/solar PNGs…", onSelect: () => { void dropCachedPngs(); } },
    { type: "separator" },
    { label: "Cache a 25.6″ tile…", onSelect: () => setOpen("tile") },
  ];
  return (
    <>
      <Menu label="Sources" items={items}
        trigger={<Button size="sm" iconRight="chevronDown" loading={busy}>Sources</Button>} />
      {open === "grouped" && <GroupedDialog open onOpenChange={close} defaultN={defaultN} />}
      {open === "galaxies" && <GalaxiesDialog open onOpenChange={close} />}
      {open === "tile" && <CacheTileDialog open onOpenChange={close} />}
    </>
  );
}
