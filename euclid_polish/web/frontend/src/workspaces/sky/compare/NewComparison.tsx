/* Sky › Compare › New comparison (the drawer at the foot of the page,
 * `?new=1`; tiles handed over with `?tiles=` open it). Target-set chips add
 * their tiles — the poster galaxy, the NEXUS tiles of the shared selection
 * (Sky › Targets, the atlas), every lens candidate or Q1 galaxy — or paste
 * refs; the model picker below; Run states the cost and asks first
 * (results/actions.ts runModels). Opening it starts nothing: the set lists
 * load only while the drawer is open. */
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useJob } from "../../../api/jobs";
import { useResource } from "../../../api/query";
import { formatCount } from "../../../format";
import { useSelected } from "../../../state/selection";
import { Button, Chip, Input, JobProgress, Tooltip } from "../../../ui";
import { runModels } from "../results/actions";
import { URLS, type TileList } from "../results/api";
import { ModelPicker, useModels } from "../results/ModelPicker";
import { experimentCost, experimentCostText, parseRefs } from "../results/model";
import { setRefs, type CompareSetId } from "./model";

const TTL = { ttl: 60_000 };
const plural = (n: number, word: string) => `${formatCount(n)} ${word}${n === 1 ? "" : "s"}`;

const SETS: { id: CompareSetId; label: string; hint: string }[] = [
  { id: "poster", label: "Poster galaxy", hint: "The poster galaxy's 102.4″ LR (one file; every poster file holds the same LR)." },
  { id: "nexus", label: "NEXUS selection", hint: "The NEXUS × JWST tiles selected in Sky › Targets or on the atlas." },
  { id: "lenses", label: "Lens candidates", hint: "Every Q1 lens candidate with a reconstruction (grades A–C)." },
  { id: "galaxies", label: "Q1 galaxies", hint: "Every Q1 galaxy of the catalogue evaluation." },
];

export function NewComparison({ tiles, setTiles, models, setModels, onStarted }: {
  tiles: string[]; setTiles: (t: string[]) => void; models: string[]; setModels: (m: string[]) => void;
  onStarted: (id: string | null) => void;
}) {
  const selection = useSelected("tile");
  const posters = useResource<TileList>(URLS.list("poster"), [], TTL);
  const evals = useResource<TileList>(URLS.list("eval"), [], TTL);
  const catalogue = useModels();
  const job = useJob("sky:experiment");
  const [paste, setPaste] = useState("");
  const [label, setLabel] = useState("");
  const [busy, setBusy] = useState(false);
  const from = useMemo(() => ({
    selection, evals: evals.data?.tiles ?? [], posters: posters.data?.tiles ?? [],
  }), [selection, evals.data, posters.data]);
  const sets = useMemo(() => SETS.map((s) => ({ ...s, refs: setRefs(s.id, from) })), [from]);
  const others = selection.filter((r) => !r.startsWith("nexus/") && !tiles.includes(r));
  const toggle = (refs: readonly string[]) => {
    const all = refs.every((r) => tiles.includes(r));
    setTiles(all ? tiles.filter((t) => !refs.includes(t)) : [...tiles, ...refs.filter((r) => !tiles.includes(r))]);
  };
  const addPasted = () => {
    const refs = parseRefs(paste);
    if (refs.length) setTiles([...tiles, ...refs.filter((r) => !tiles.includes(r))]);
    setPaste("");
  };
  const cost = catalogue.data && tiles.length && models.length
    ? experimentCostText(experimentCost(models, catalogue.data.models, tiles.length)) : "";
  const run = async () => {
    setBusy(true);
    try {
      const r = await runModels(tiles, models, { label });
      if (r) { onStarted(r.experimentId); setLabel(""); }
    } finally { setBusy(false); }
  };
  const loading = (id: CompareSetId) => (id === "poster" ? posters.loading : id === "lenses" || id === "galaxies" ? evals.loading : false);
  return (
    <div className="res-new cmp-new">
      <div className="res-new__tiles">
        <div className="cmp-sets" role="group" aria-label="Target sets">
          {sets.map((s) => {
            const on = s.refs.length > 0 && s.refs.every((r) => tiles.includes(r));
            return (
              <Tooltip key={s.id} content={s.refs.length || loading(s.id) ? s.hint : `${s.hint} None yet.`}>
                <span>
                  <Chip on={on} disabled={!s.refs.length} onClick={() => toggle(s.refs)}>
                    {s.label}{s.refs.length ? <>{" "}<span className="cmp-count">{formatCount(s.refs.length)}</span></> : null}
                  </Chip>
                </span>
              </Tooltip>
            );
          })}
          {!!others.length && (
            <Chip onClick={() => setTiles([...tiles, ...others])}>
              Other selected tiles{" "}<span className="cmp-count">{formatCount(others.length)}</span>
            </Chip>
          )}
        </div>
        <Input size="sm" value={paste} onChange={setPaste} onEnter={addPasted} icon="search"
          placeholder="Paste refs: nexus/f200w-0040 poster/… (Enter)" aria-label="Add tiles by ref" />
        <div className="res-bar res-bar--inline">
          <strong>{tiles.length ? plural(tiles.length, "tile") : "No tiles yet"}</strong>
          <span className="res-bar__spacer" />
          <Button size="sm" variant="ghost" asChild><Link to="/sky/targets">Pick in Targets</Link></Button>
          {!!tiles.length && <Button size="sm" variant="ghost" onClick={() => setTiles([])}>Clear</Button>}
        </div>
        {tiles.length ? (
          <div className="res-tilelist" aria-label="Comparison tiles">
            {tiles.map((t) => <Chip key={t} onRemove={() => setTiles(tiles.filter((x) => x !== t))}><span className="mono">{t}</span></Chip>)}
          </div>
        ) : <p className="muted res-note">Pick a target set, select tiles in Sky › Targets or on the atlas, or paste refs.</p>}
      </div>
      <div>
        <ModelPicker value={models} onChange={setModels} />
      </div>
      <div className="res-new__foot">
        <Input size="sm" value={label} onChange={setLabel} placeholder="Label (optional)" aria-label="Comparison label" />
        <Button variant="primary" icon="activity" loading={busy || job.busy} disabled={!tiles.length || !models.length}
          onClick={() => { void run(); }}>
          Run {plural(models.length, "model")} on {plural(tiles.length, "tile")}…
        </Button>
        {cost && <span className="res-note res-new__cost">{cost}</span>}
        <JobProgress job={job.job} error={job.error} />
      </div>
    </div>
  );
}
