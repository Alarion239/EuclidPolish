/* Models › Images (`/models/:mode/images`; absorbs the old Ensemble
   Disagreement tab, the synthetic groups of Sky › Catalog eval and the
   "Generate SR over local records" action of Data › Records). What SR looks
   like on synthetic truth:
   - the bar: the set — test fields (default; the regime's evaluated test
     cubes, the `ensemble` viewer collection), source-centred stamps
     (?set=stamps&g=syn-lens|syn-gal, the `evaluation` collection) or the
     local synthetic records with the production SR generated over them
     (?set=records&split=test&id=test:12, the `sky` collection, the SR tier
     Synthetic › Records shows) with their counts — and "Generate SR over
     local records…" (confirmed);
   - the viewer, opening on LR | SR | HR (SR = the production gate); tiers
     mean, member stills, the disagreement movie and BHR are in its tier menu;
   - under it the footer, SR's Δm vs LR in the shown band (images/fluxDelta),
     and the per-set caption (PSNR vs HR);
   - the side panel: on test fields the ONE member picker (?sel=196,195,
     ?sort=, ?loss=; Members › "Images" fills ?sel=), on stamps the group's
     stamps (a row shows it, ?id=).
   Opening the page reads caches only; every job asks first. */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useJob } from "../../../api/jobs";
import { useResource } from "../../../api/query";
import { pagePath } from "../../../app/nav";
import { usePageActions } from "../../../app/palette";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Button, Caption, Checkbox, Chip, DataTable, EmptyState, IconButton, Page, Popover, Segmented, Switch, Toolbar, ToolbarGroup,
  ToolbarSpacer, Tooltip, type DataColumn,
} from "../../../ui";
import { ImageViewer, type ViewerApi } from "../../../viewer";
import { url, useEvalRuns, useMembers, useMode, useOverview, useSrStatus, type Mode, type SrSplit, type SrStatus } from "../api";
import { useFacetColors } from "../common";
import { JOB, generateSr, useOnJobEnd } from "../jobs";
import {
  db, dbDelta, kneeText, memberMatches, memberNumber, movieStatus, recordSrSplits, stampCaption, stampSets, type StampPoint, type StampSet,
} from "../model";
import { MemberPanel, type Pick, type Sort } from "../images/MemberPanel";
import { useFluxDelta, type FluxDelta } from "../images/fluxDelta";
import "../models.css";

type SetKey = "fields" | "stamps" | "records";
const SETS: readonly SetKey[] = ["fields", "stamps", "records"];
const RECORD_TIERS = ["dirty", "sr", "hr"];
const FIELD_TIERS = ["lr", "sr", "hr"];
const STAMP_TIERS = ["LR", "SR", "HR"];
type Meta = { member_labels?: string[]; count?: number };

const tabPath = (mode: Mode, tab: string) => pagePath("models", { tab, params: { mode } });

/** The footer under the viewer: SR's total-flux change against LR. */
function FluxFooter({ delta, what }: { delta: FluxDelta | null; what: string }) {
  if (!delta) return null;
  return (
    <p className="mdl-footer" data-tone={delta.warn ? "warn" : undefined}>
      {what} vs LR in {delta.band}: <strong className="mdl-tnum">{delta.text}</strong>
      {delta.warn && <span className="mdl-muted"> · beyond 0.1 mag</span>}
    </p>
  );
}

/* ── Generate SR over the local records ────────────────────────────────── */
function GenerateSr({ mode }: { mode: Mode }) {
  const job = useJob(JOB.generateSr);
  const status = useSrStatus(job.busy ? 3_000 : undefined);
  useOnJobEnd(job.job);
  const [open, setOpen] = useState(false);
  const s = status.data;
  const reason = mode === "starless" ? "Generates the STARFULL production SR (switch to starfull)"
    : !s ? (status.error ? `Cannot read the records: ${status.error.message}` : "Reading the local records…")
      : !s.records ? "Sync the records first (Synthetic › Records)." : !s.checkpoint ? "No active starfull members." : !s.can_generate ? "Nothing to generate" : null;
  if (reason) {
    return (
      <Tooltip content={reason}>
        <span tabIndex={0}><Button size="sm" icon="wave" loading={job.busy} disabled>Generate SR over local records…</Button></span>
      </Tooltip>
    );
  }
  return (
    <Popover open={open} onOpenChange={setOpen} label="Generate SR over local records" width={320}
      trigger={<Button size="sm" icon="wave" loading={job.busy}>Generate SR over local records…</Button>}>
      {s && <GenerateForm status={s} onStart={(subsets, overwrite) => { setOpen(false); void generateSr(subsets, overwrite); }} />}
    </Popover>
  );
}

function GenerateForm({ status, onStart }: { status: SrStatus; onStart: (subsets: SrSplit[], overwrite: boolean) => void }) {
  const present = status.subsets;
  const [subsets, setSubsets] = useState<SrSplit[]>(() => present.filter((s) => s !== "train"));
  const anyExisting = subsets.some((s) => (status.sr[s] ?? 0) > 0);
  const anyStale = subsets.some((s) => ["stale", "unknown", "partial"].includes(status.splits[s]?.sr.state ?? ""));
  const [overwrite, setOverwrite] = useState(anyStale);
  return (
    <div className="mdl-stack mdl-stack--tight">
      <p className="mdl-note">The production SR over the local synthetic records: the SR tier of Synthetic › Records and the angular power spectrum.</p>
      <fieldset className="mdl-fieldset">
        <legend>Splits</legend>
        {present.map((s) => (
          <Checkbox key={s} checked={subsets.includes(s)} onChange={(on) => setSubsets((c) => (on ? [...c, s] : c.filter((x) => x !== s)))}>
            {s} <span className="mdl-muted">({status.splits[s]?.count ?? 0} records, {status.sr[s] ?? 0} SR)</span>
          </Checkbox>
        ))}
      </fieldset>
      <Switch checked={overwrite} onChange={setOverwrite}>Overwrite existing SR</Switch>
      {anyExisting && !overwrite && <p className="mdl-note">Splits that already have SR are skipped.</p>}
      <Button variant="primary" size="sm" disabled={!subsets.length} onClick={() => onStart(subsets, overwrite)}>Generate…</Button>
    </div>
  );
}

/* ── test fields ───────────────────────────────────────────────────────── */
function FieldsView({ mode }: { mode: Mode }) {
  const meta = useResource<Meta>(url.viewerMeta(mode), [mode], { ttl: 0 });
  const members = useMembers(mode);
  const ov = useOverview(mode);
  const [selRaw, setSelRaw] = useUrlState("sel", "");
  const [sort, setSort] = useUrlState<Sort>("sort", "knee");
  const [loss, setLoss] = useUrlState("loss", "");
  const [find, setFind] = useState("");
  const [index, setIndex] = useState<number | null>(null);
  const [color, setColor] = useState("VIS");
  const api = useRef<ViewerApi | null>(null);
  const ready = useRef(false);
  const params = useMemo(() => ({ mode }), [mode]);
  const delta = useFluxDelta("ensemble", params, index, ["lr", "sr"], color);

  const byNum = useMemo(() => new Map((members.data?.members ?? []).map((m) => [memberNumber(m.name) ?? "", m])), [members.data]);
  const picks = useMemo<Pick[]>(() => (meta.data?.member_labels ?? []).map((label, i) => {
    const num = memberNumber(label) ?? String(i);
    return { i, num, label, row: byNum.get(num) ?? null };
  }), [meta.data, byNum]);
  const colors = useFacetColors(picks.map((p) => p.row ?? { loss: "l1" }), "loss");
  const losses = useMemo(() => [...new Set(picks.map((p) => p.row?.loss ?? "l1"))].sort(), [picks]);
  const sel = useMemo(() => selRaw.split(",").map((s) => memberNumber(s)).filter((n): n is string => !!n), [selRaw]);
  const selSet = useMemo(() => new Set(sel), [sel]);
  const shown = useMemo(() => {
    const f = picks.filter((p) => (!loss || (p.row?.loss ?? "l1") === loss)
      && memberMatches({ num: p.num, loss: p.row?.loss, knee: p.row ? kneeText(p.row).text : null, label: p.label }, find));
    const cmp: Record<Sort, (a: Pick, b: Pick) => number> = {
      index: (a, b) => a.i - b.i,
      knee: (a, b) => (b.row?.knee_integrated?.mean ?? -1e9) - (a.row?.knee_integrated?.mean ?? -1e9),
      psnr: (a, b) => (b.row?.psnr ?? -1e9) - (a.row?.psnr ?? -1e9),
      loss: (a, b) => (a.row?.loss ?? "").localeCompare(b.row?.loss ?? "") || a.i - b.i,
    };
    return [...f].sort(cmp[sort] ?? cmp.knee);
  }, [picks, loss, sort, find]);

  /** Map the selection onto the viewer, keeping the user's base tiers. */
  const apply = useCallback((nums: string[]) => {
    const v = api.current;
    if (!v) return;
    const idx = nums.map((n) => picks.find((p) => p.num === n)?.i).filter((i): i is number => i != null).sort((a, b) => a - b);
    const base = v.getState().tiers.filter((t) => !/^member\d+$/.test(t) && t !== "morph");
    if (idx.length >= 2) {
      v.setMorphMembers(idx.join(","));
      v.setTiers([...base, "morph"]);
    } else if (idx.length === 1) {
      v.setMorphMembers(null);
      v.setTiers([...base, `member${idx[0]}`]);
    } else {
      v.setMorphMembers(null);
      v.setTiers(base.length ? base : ["sr"]);
    }
  }, [picks]);
  useEffect(() => { if (ready.current) apply(sel); }, [apply, sel.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggle = (num: string) => setSelRaw((selSet.has(num) ? sel.filter((n) => n !== num) : [...sel, num]).join(","));
  const ranked = picks.some((p) => p.row?.knee_integrated?.mean != null);
  const top = (k: number) => setSelRaw([...picks].filter((p) => p.row?.knee_integrated?.mean != null)
    .sort((a, b) => (b.row!.knee_integrated!.mean as number) - (a.row!.knee_integrated!.mean as number)).slice(0, k).map((p) => p.num).join(","));
  const reload = () => { void meta.reload(); void api.current?.reload(); };
  usePageActions([
    { id: "img-top5", label: "Images: disagreement movie over the top 5 members (∫PSNR)", group: "Images", disabled: !ranked, run: () => top(5) },
    { id: "img-clear", label: "Images: clear the member selection", group: "Images", disabled: !sel.length, run: () => setSelRaw("") },
    { id: "img-reload", label: "Images: reload the test-cube cache", group: "Images", run: reload },
  ]);

  const count = meta.data?.count ?? 0;
  if (meta.error) {
    return (
      <EmptyState icon="warn" title="The test-cube cache is not readable" action={<Button size="sm" icon="reset" onClick={reload}>Retry</Button>}>
        <span className="mdl-mono">{meta.error.message}</span>
      </EmptyState>
    );
  }
  if (!meta.loading && count === 0) {
    return (
      <EmptyState icon="image" title={`No ${mode} test fields cached`}>
        Evaluate the ensemble (Leaderboard) to cache the test fields for the viewer.
      </EmptyState>
    );
  }
  const h = ov.data?.headline;
  const target = mode === "starless" ? "the clean target" : "HR";
  const caption = h ? [
    `${h.n_scored ?? count} ${mode} test fields`,
    h.production.psnr != null || h.mean.psnr != null
      ? `PSNR vs ${target} (VIS, knee ${h.knee_e ?? 100} e⁻): ${[h.production.psnr != null ? `production ${db(h.production.psnr)} dB` : null,
        h.mean.psnr != null ? `plain mean ${db(h.mean.psnr)} dB` : null,
        h.best_member.psnr != null ? `best member ${db(h.best_member.psnr)} dB` : null].filter(Boolean).join(", ")}` : null,
    h.knee.production != null ? `production ∫PSNR ${db(h.knee.production)} dB` : null,
  ].filter(Boolean).join(" · ") : null;
  return (
    <div className="mdl-images">
      <div className="mdl-images__main">
        <ImageViewer key={mode} collection="ensemble" params={params} urlKey="ens" toolbar="full" tiers={FIELD_TIERS}
          onReady={(v) => { api.current = v; if (!v) ready.current = false; }}
          onState={(s) => {
            setIndex(s.index);
            setColor(s.color);
            if (!ready.current && s.tiers.length && picks.length) { ready.current = true; apply(sel); }
          }} />
        <FluxFooter delta={delta} what={mode === "starless" ? "SR (starless)" : "SR"} />
        {caption && <Caption>{caption}. The ranking is the <Link to={tabPath(mode, "leaderboard")}>Leaderboard</Link>.</Caption>}
      </div>
      <MemberPanel shown={shown} picks={picks} sel={selSet} status={movieStatus(sel)} find={find} setFind={setFind}
        loss={loss} setLoss={setLoss} losses={losses} sort={sort} setSort={setSort} colorOf={(row) => colors.of(row)}
        toggle={toggle} top={top} clear={() => setSelRaw("")} ranked={ranked} loading={!members.data && !members.error}
        unreadable={!members.data && !!members.error}
        leaderboard={tabPath(mode, "leaderboard")} />
    </div>
  );
}

/* ── source-centred stamps ─────────────────────────────────────────────── */
type StampRow = StampPoint & { gain: number };

function StampsView({ sets, set, loading, error, onRetry }: {
  sets: StampSet[]; set: StampSet | null; loading: boolean; error: Error | null; onRetry: () => void;
}) {
  const [id, setId] = useUrlState("id", "");
  const [index, setIndex] = useState<number | null>(null);
  const [color, setColor] = useState("VIS");
  const [current, setCurrent] = useState<string | null>(null);
  const api = useRef<ViewerApi | null>(null);
  const params = useMemo(() => ({}), []);
  const delta = useFluxDelta("evaluation", params, index, ["LR", "SR"], color);
  const rows = useMemo<StampRow[]>(() => (set?.points ?? []).map((p) => ({ ...p, gain: p.y - p.x })), [set]);
  const first = rows.length ? [...rows].sort((a, b) => b.gain - a.gain)[0].id : "";
  const start = id && rows.some((r) => r.id === id) ? id : first;
  const go = (next: string) => { setId(next); api.current?.goToId(next); };
  const columns = useMemo<DataColumn<StampRow>[]>(() => [
    { id: "id", header: "Stamp", width: 120, cell: (r) => <span className="mdl-mono">{r.id.replace(/^syn-(lens|gal)_/, "")}</span> },
    { id: "x", header: "LR [dB]", headerText: "PSNR LR vs HR", numeric: true, width: 72, cell: (r) => db(r.x), hidden: true },
    { id: "y", header: "SR [dB]", headerText: "PSNR SR vs HR", numeric: true, width: 72, cell: (r) => db(r.y), hidden: true },
    { id: "gain", header: "Gain [dB]", headerText: "PSNR gain SR − LR", numeric: true, width: 92, cell: (r) => dbDelta(r.gain) },
  ], []);
  if (error) {
    return <EmptyState icon="warn" title="The catalogue evaluation is not readable" action={<Button size="sm" onClick={onRetry}>Retry</Button>}>
      <span className="mdl-mono">{error.message}</span></EmptyState>;
  }
  if (loading && !sets.length) return <EmptyState compact icon="image" title="Reading the synthetic stamps…" />;
  if (!set) {
    return (
      <EmptyState icon="image" title="No synthetic stamps evaluated yet">
        The source-centred synthetic lenses and galaxies come from the grouped analysis (Sky › Targets › Sources).
      </EmptyState>
    );
  }
  return (
    <div className="mdl-images">
      <div className="mdl-images__main">
        {start && (
          <ImageViewer key={set.grade} collection="evaluation" urlKey="stamp" toolbar="full" nav={false}
            tiers={STAMP_TIERS} initialId={start}
            onReady={(v) => { api.current = v; }}
            onState={(s) => { setIndex(s.index); setColor(s.color); setCurrent(s.id); }} />
        )}
        <FluxFooter delta={delta} what="SR" />
        <Caption>
          {stampCaption(set)}.
          {set.stale > 0 && <span className="mdl-warn"> {set.stale === set.n ? "All" : `${set.stale} of them`} predate the current model.</span>}
          {set.madeBy != null && ` Made by ${set.madeBy} member${set.madeBy === 1 ? "" : "s"}.`}
        </Caption>
      </div>
      <aside className="mdl-members" aria-labelledby="mdl-img-stamps">
        <header className="mdl-members__head">
          <h2 id="mdl-img-stamps" className="mdl-members__title">{set.label} stamps</h2>
          <span className="mdl-members__status">sorted by the SR gain; a row shows it</span>
        </header>
        <DataTable rows={rows} columns={columns} rowKey={(r) => r.id} aria-label={`${set.label} stamps`} dense
          defaultSort={[{ id: "gain", desc: true }]} height={420} activeKey={current ?? start}
          onRowClick={(r) => go(r.id)} exportName={`stamps-${set.grade}`} filterPlaceholder="Filter stamps" />
      </aside>
    </div>
  );
}

/* ── the production SR over the local records ──────────────────────────── */
const SR_STATE_TEXT: Record<string, string> = {
  stale: "predates the current production model", partial: "covers only some of the records", missing: "not generated",
};

function RecordsView({ mode }: { mode: Mode }) {
  const status = useSrStatus();
  const splits = useMemo(() => recordSrSplits(status.data), [status.data]);
  const [splitRaw, setSplit] = useUrlState("split", "");
  const [id] = useUrlState("id", "");
  const [index, setIndex] = useState<number | null>(null);
  const [color, setColor] = useState("VIS");
  const cur = splits.find((x) => x.split === splitRaw) ?? splits[0] ?? null;
  const params = useMemo(() => ({ subset: cur?.split ?? "test" }), [cur?.split]);
  const delta = useFluxDelta("sky", params, cur ? index : null, ["dirty", "sr"], color);
  const initialId = cur && id.startsWith(`${cur.split}:`) ? id : undefined;
  if (status.error && !status.data) {
    return <EmptyState icon="warn" title="The local records are not readable" action={<Button size="sm" onClick={() => void status.reload()}>Retry</Button>}>
      <span className="mdl-mono">{status.error.message}</span></EmptyState>;
  }
  if (!status.data) return <EmptyState compact icon="image" title="Reading the local records…" />;
  if (!cur) {
    return (
      <EmptyState icon="image" title="No SR over the local records yet">
        {status.data.records
          ? "Generate the production SR over the synced splits (Generate SR over local records…, above); it then shows here and in Synthetic › Records."
          : <>Sync the records first in <Link to={pagePath("synthetic", { tab: "records" })}>Synthetic › Records</Link>, then generate their SR.</>}
      </EmptyState>
    );
  }
  return (
    <div className="mdl-stack mdl-stack--tight">
      {splits.length > 1 && (
        <Toolbar label="Records split">
          <ToolbarGroup label="Split">
            <Segmented<SrSplit> size="sm" aria-label="Records split" value={cur.split} onChange={setSplit}
              options={splits.map((x) => ({ value: x.split, label: `${x.split} ${x.n}` }))} />
          </ToolbarGroup>
        </Toolbar>
      )}
      <ImageViewer key={cur.split} collection="sky" params={params} urlKey="rsr" toolbar="full" tiers={RECORD_TIERS} initialId={initialId}
        onState={(st) => { setIndex(st.index); setColor(st.color); }} />
      <FluxFooter delta={delta} what="SR" />
      <Caption>
        {cur.n} of {cur.records} {cur.split} records carry the production SR (STARFULL gate), shown beside their LR input and HR truth
        {cur.state !== "current" && <span className="mdl-warn">; it {SR_STATE_TEXT[cur.state] ?? cur.state}{cur.reasons.length ? ` (${cur.reasons.join("; ")})` : ""}</span>}.
        {mode === "starless" && " The records' SR is always the starfull production model's."}{" "}
        The records themselves, their truth sources and census are in <Link to={pagePath("synthetic", { tab: "records" })}>Synthetic › Records</Link>.
      </Caption>
    </div>
  );
}

/* ── the tab ───────────────────────────────────────────────────────────── */
export default function Images() {
  const mode = useMode();
  const [set, setSet] = useUrlState<SetKey>("set", "fields", { parse: (r) => (SETS.includes(r as SetKey) ? r as SetKey : undefined) });
  const [group, setGroup] = useUrlState("g", "");
  const runs = useEvalRuns();
  const sets = useMemo(() => stampSets(runs.data?.rows ?? []), [runs.data]);
  const stamps = sets.find((s) => s.grade === group) ?? sets[0] ?? null;
  const nStamps = sets.reduce((n, s) => n + s.n, 0);
  const srStatus = useSrStatus();
  const nRecords = recordSrSplits(srStatus.data).reduce((n, s) => n + s.n, 0);
  usePageActions([
    { id: "img-fields", label: "Images: the test fields", group: "Images", run: () => setSet("fields") },
    { id: "img-stamps", label: "Images: the synthetic stamps", group: "Images", run: () => setSet("stamps") },
    { id: "img-records", label: "Images: the production SR over the local records", group: "Images", run: () => setSet("records") },
  ]);
  return (
    <Page className="mdl-page mdl-images-page">
      <Toolbar label="Images controls">
        <ToolbarGroup label="Set" hideLabel>
          <Segmented<SetKey> size="sm" aria-label="Image set" value={set} onChange={setSet} options={[
            { value: "fields", label: "Test fields", title: "The regime's evaluated synthetic test fields" },
            { value: "stamps", label: nStamps ? `Stamps ${nStamps}` : "Stamps", title: "Source-centred synthetic lenses and galaxies (with HR truth)" },
            { value: "records", label: nRecords ? `Records ${nRecords}` : "Records", title: "The local synthetic records with the production SR generated over them" },
          ]} />
        </ToolbarGroup>
        {set === "stamps" && sets.length > 0 && (
          <ToolbarGroup label="Group" hideLabel>
            {sets.map((s) => (
              <Chip key={s.grade} on={stamps?.grade === s.grade} onClick={() => setGroup(s.grade)}>{s.label} {s.n}</Chip>
            ))}
          </ToolbarGroup>
        )}
        <ToolbarSpacer />
        {set === "stamps" && <IconButton size="sm" icon="reset" label="Reload the stamps" onClick={() => void runs.reload()} />}
        <GenerateSr mode={mode} />
      </Toolbar>
      {set === "fields"
        ? <FieldsView mode={mode} />
        : set === "records" ? <RecordsView mode={mode} />
        : <>
            {mode === "starless" && <p className="mdl-note">The synthetic stamps are made by the STARFULL production model.</p>}
            <StampsView sets={sets} set={stamps} loading={runs.loading} error={runs.error ?? null} onRetry={() => void runs.reload()} />
          </>}
    </Page>
  );
}
