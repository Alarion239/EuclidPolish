/* Sky › Catalog eval (spec §7.4): the catalogue evaluation over
 * data/eval_results — the reconstruction browser (viewer `evaluation`) beside
 * a DataTable of every manifest row with its SR's model state against the
 * model an evaluation would load now (STARFULL members + production
 * combiner), position → atlas, real objects → the realtile inspector; the
 * grouped analysis, real-galaxy query (the one Euclid session of Settings),
 * lens catalogue fetch, FASRC sync (confirmed: rsync --delete-after) and the
 * summary figures, rendered only on request. */
import { useCallback, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { apiGet, apiPost } from "../../../api/client";
import { useJob } from "../../../api/jobs";
import { invalidate, useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { formatCount, formatDeg, formatNumber, formatSI } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, Card, CardBody, Checkbox, Chip, DataTable, IconButton, JobProgress, Menu,
  NumberField, Page, Popover, Section, Segmented, Skeleton, Tooltip, confirm, toast, type DataColumn,
  type MenuItem,
} from "../../../ui";
import { ImageViewer, type ViewerApi } from "../../../viewer";
import { errorText } from "../results/actions";
import { atlasHref, URLS, type EvalRow, type EvalRuns } from "../results/api";
import { StateBadge } from "../results/common";
import { EVAL_GROUPS, filterEvalRows, num } from "../results/model";
import "../results/register";
import "../results/results.css";


type AuthStatus = { authenticated?: boolean; user?: string | null };

const STATES = ["all", "current", "stale", "unknown"] as const;

function columns(goAtlas: (r: EvalRow) => void): DataColumn<EvalRow>[] {
  return [
    { id: "id", header: "Object", cell: (r) => <span className="mono res-ellipsis" title={String(r.id ?? "")}>{r.id}</span>, width: 220 },
    { id: "grade", header: "Group", accessor: (r) => r.grade ?? "", cell: (r) => (r.grade ? <Badge size="sm">{r.grade}</Badge> : "—"), width: 76 },
    { id: "state", header: "SR model", accessor: (r) => (String(r.ok).toLowerCase() === "true" ? r.state ?? "unknown" : "failed"),
      cell: (r) => (String(r.ok).toLowerCase() === "true"
        ? <StateBadge state={r.state ?? "unknown"} title={r.state_reason} />
        : <Tooltip content={String(r.error || "failed")}><span tabIndex={0}><Badge size="sm" tone="bad">failed</Badge></span></Tooltip>),
      width: 100 },
    { id: "field", header: "Field", accessor: (r) => r.field ?? "", width: 70 },
    { id: "ra", header: "RA °", numeric: true, accessor: (r) => num(r.ra), cell: (r) => formatDeg(num(r.ra), 4), width: 86 },
    { id: "dec", header: "Dec °", numeric: true, accessor: (r) => num(r.dec), cell: (r) => formatDeg(num(r.dec), 4, { signed: true }), width: 86 },
    { id: "flux", header: "Flux SR/LR", headerText: "flux_ratio_sr_over_lr", numeric: true, accessor: (r) => num(r.flux_ratio_sr_over_lr),
      cell: (r) => formatNumber(num(r.flux_ratio_sr_over_lr), { digits: 3 }), width: 88 },
    { id: "psnr_sr_hr", header: "PSNR SR", numeric: true, accessor: (r) => num(r.psnr_sr_hr), cell: (r) => formatNumber(num(r.psnr_sr_hr), { digits: 2 }), width: 80 },
    { id: "psnr_lr_hr", header: "PSNR LR", numeric: true, accessor: (r) => num(r.psnr_lr_hr), cell: (r) => formatNumber(num(r.psnr_lr_hr), { digits: 2 }), hidden: true },
    { id: "lr_total_e", header: "LR flux", numeric: true, accessor: (r) => num(r.lr_total_e), cell: (r) => formatSI(num(r.lr_total_e), { unit: "e⁻" }), hidden: true },
    { id: "sr_total_e", header: "SR flux", numeric: true, accessor: (r) => num(r.sr_total_e), cell: (r) => formatSI(num(r.sr_total_e), { unit: "e⁻" }), hidden: true },
    { id: "n_members", header: "Members", numeric: true, accessor: (r) => r.n_members ?? null, hidden: true },
    { id: "combiner_kind", header: "Combiner", accessor: (r) => r.combiner_kind ?? "", hidden: true },
    { id: "kind", header: "Kind", accessor: (r) => r.kind ?? "", hidden: true },
    { id: "error", header: "Error", accessor: (r) => r.error ?? "", hidden: true },
    { id: "out_subdir", header: "Directory", hidden: true },
    { id: "go", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 64,
      cell: (r) => (
        <span className="res-chips res-chips--tight">
          {r.realtile && <IconButton icon="panelRight" size="sm" label="Inspect the real tile" onClick={() => openInspector({ kind: "realtile", id: String(r.realtile) })} />}
          {num(r.ra) != null && num(r.dec) != null && <IconButton icon="globe" size="sm" label="Show on the sky" onClick={() => goAtlas(r)} />}
        </span>
      ) },
  ];
}

function EvalFigure({ title, url, sub }: { title: string; url: string; sub: string }) {
  const [nonce, setNonce] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const src = nonce ? `${url}?fresh=1&t=${nonce}` : url;
  const onError = () => {
    apiGet(url).then(() => setError("The figure could not be displayed."), (e) => setError(errorText(e)));
  };
  return (
    <div className="res-fig">
      <div className="res-bar res-bar--inline">
        <strong>{title}</strong>
        <span className="muted res-note">{sub}</span>
        <span className="res-bar__spacer" />
        <Button size="sm" variant="ghost" icon="reset" onClick={() => { setError(null); setNonce(Date.now()); }}>Re-render</Button>
      </div>
      {error ? <Callout tone="warn" title="Not available">{error}</Callout>
        : <img className="res-fig__img" src={src} alt={title} loading="lazy" onError={onError} />}
    </div>
  );
}

export default function CatalogEval() {
  const navigate = useNavigate();
  const runs = useResource<EvalRuns>(URLS.evalRuns, [], { ttl: 30_000 });
  const auth = useResource<AuthStatus>(URLS.authStatus, [], { ttl: 30_000 });
  const grouped = useJob("catalog-eval:grouped");
  const galaxies = useJob("catalog-eval:galaxies");
  const [groups, setGroups] = useUrlState<string[]>("g", []);
  const [state, setState] = useUrlState("st", "all");
  const [failed, setFailed] = useUrlState("failed", false);
  const [figs, setFigs] = useUrlState("figs", false);
  const [groupedOpen, setGroupedOpen] = useState(false);
  const [galOpen, setGalOpen] = useState(false);
  const [nPer, setNPer] = useState("8");
  const [synthetic, setSynthetic] = useState(true);
  const [nGal, setNGal] = useState("50");
  const [regen, setRegen] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const viewer = useRef<ViewerApi | null>(null);
  const [active, setActive] = useState<string | null>(null);

  const data = runs.data;
  const rows = useMemo(() => filterEvalRows(data?.rows ?? [], groups, state, !failed), [data, groups, state, failed]);
  const counts = data?.counts ?? { current: 0, stale: 0, unknown: 0 };
  const cur = data?.current;
  const goAtlas = useCallback((r: EvalRow) => {
    const ra = num(r.ra), dec = num(r.dec);
    if (ra != null && dec != null) navigate(atlasHref(ra, dec, r.realtile ?? undefined));
  }, [navigate]);
  const cols = useMemo(() => columns(goAtlas), [goAtlas]);
  const reload = () => { void runs.reload(); void invalidate("/viewer/meta/evaluation"); };
  const onDone = () => { reload(); void invalidate("/api/real/eval"); };

  const runGrouped = () => {
    setGroupedOpen(false);
    void grouped.run("/api/evaluation/run-grouped", { n: nPer, synthetic: synthetic ? 1 : 0 }, { onDone });
  };
  const queryGalaxies = () => {
    setGalOpen(false);
    void galaxies.run("/api/evaluation/query-galaxies", { n_galaxies: nGal, regenerate: regen ? 1 : 0 }, { onDone });
  };
  const post = async (key: string, url: string, body: Record<string, string>, ok: (r: Record<string, unknown>) => string) => {
    setBusy(key);
    try {
      const r = (await apiPost<Record<string, unknown>>(url, body)) ?? {};
      if (r.ok === false || r.error) throw new Error(String(r.error ?? "refused"));
      toast.success(ok(r));
      reload();
    } catch (e) {
      toast.error(`${key}: failed`, { description: errorText(e) });
    } finally { setBusy(null); }
  };
  const fetchCatalog = async () => {
    const yes = await confirm({
      title: "Fetch the Q1 strong-lens catalogue?",
      message: "Downloads the Euclid Q1 discovery-engine lens catalogue (≈ 0.4 MB, Zenodo) and rewrites lens_catalog/lenses.csv.",
      confirmLabel: "Fetch",
    });
    if (yes) await post("Lens catalogue", "/api/evaluation/fetch-catalog", {}, (r) => `Lens catalogue: ${formatCount(num(r.rows))} rows`);
  };
  // The sync runs `rsync --delete-after`: the server refuses it without
  // confirm=1 (400 confirm_required), so ask first.
  const sync = async () => {
    const yes = await confirm({
      title: "Sync evaluation results from FASRC?",
      message: "rsync --delete-after will delete local-only results in data/eval_results that FASRC does not have.",
      tone: "danger", confirmLabel: "Sync and delete local-only",
    });
    if (yes) await post("FASRC sync", "/api/evaluation/sync", { confirm: "1" }, (r) => `Synced: ${formatCount(num(r.n_ok))}/${formatCount(num(r.n))} objects ok`);
  };
  const dropPngs = () => post("Cached PNGs", "/api/evaluation/rerender", {}, (r) => `Dropped ${formatCount(num(r.removed))} cached PNGs`);

  usePageActions([
    { id: "ce-grouped", label: "Run the grouped catalogue analysis…", group: "Catalog eval", disabled: grouped.busy, run: () => setGroupedOpen(true) },
    { id: "ce-galaxies", label: "Query real galaxies (Euclid archive)…", group: "Catalog eval", disabled: galaxies.busy, run: () => setGalOpen(true) },
    { id: "ce-catalog", label: "Fetch the Q1 lens catalogue", group: "Catalog eval", run: () => { void fetchCatalog(); } },
    { id: "ce-sync", label: "Sync evaluation results from FASRC…", group: "Catalog eval", run: () => { void sync(); } },
    { id: "ce-stale", label: "Show stale reconstructions", group: "Catalog eval", run: () => setState("stale") },
    { id: "ce-figs", label: "Show the evaluation figures", group: "Catalog eval", run: () => setFigs(true) },
    { id: "ce-refresh", label: "Refresh the catalogue evaluation", group: "Catalog eval", run: reload },
  ]);

  const loggedIn = !!auth.data?.authenticated;
  const more: MenuItem[] = [
    { label: "Fetch the Q1 lens catalogue…", onSelect: () => { void fetchCatalog(); } },
    { label: "Sync results from FASRC…", onSelect: () => { void sync(); } },
    { type: "separator" },
    { label: "Drop cached eye/solar PNGs", onSelect: () => { void dropPngs(); } },
  ];
  const toggleGroup = (g: string) => setGroups(groups.includes(g) ? groups.filter((x) => x !== g) : [...groups, g]);

  return (
    <Page className="res-page">
      <div className="res-bar" role="toolbar" aria-label="Catalog eval">
        <div className="res-bar__group" role="group" aria-label="Groups">
          <Chip on={!groups.length} onClick={() => setGroups([])}>All</Chip>
          {EVAL_GROUPS.map((g) => (
            <Chip key={g.id} on={groups.includes(g.id)} onClick={() => toggleGroup(g.id)} disabled={!data?.groups?.[g.id] && !groups.includes(g.id)}>
              {g.label} <span className="muted">{formatCount(data?.groups?.[g.id] ?? 0)}</span>
            </Chip>
          ))}
          <Chip on={failed} onClick={() => setFailed(!failed)}>+ failed</Chip>
        </div>
        <Segmented size="sm" value={state} onChange={setState} aria-label="SR model state"
          options={STATES.map((s) => ({ value: s, label: s === "all" ? "All" : `${s} ${counts[s]}` }))} />
        <span className="res-bar__spacer" />
        <Popover open={groupedOpen} onOpenChange={setGroupedOpen} label="Grouped analysis" width={300} align="end"
          trigger={<Button size="sm" variant="primary" icon="activity" loading={grouped.busy}>Grouped analysis…</Button>}>
          <div className="res-form">
            <NumberField label="Objects per group" value={nPer} onChange={setNPer} min={1} max={200}
              hint="N lenses per grade; 3N galaxies, syn-lens and syn-gal." />
            <Checkbox checked={synthetic} onChange={setSynthetic}>Synthetic groups (HR truth)</Checkbox>
            <div className="res-form__foot"><Button size="sm" variant="primary" onClick={runGrouped}>Run</Button></div>
          </div>
        </Popover>
        <Popover open={galOpen} onOpenChange={setGalOpen} label="Query real galaxies" width={320} align="end"
          trigger={<Button size="sm" loading={galaxies.busy}>Query galaxies…</Button>}>
          <div className="res-form">
            {loggedIn ? <span className="res-note">Euclid archive: {auth.data?.user ?? "logged in"}</span> : (
              <Callout tone="warn" title="Not logged in">
                Log in once in <Link to="/settings/connections">Settings › Connections</Link>.
              </Callout>
            )}
            <NumberField label="Galaxies" value={nGal} onChange={setNGal} min={1} max={2000} />
            <Checkbox checked={regen} onChange={setRegen}>Discard the cache and re-query</Checkbox>
            <div className="res-form__foot"><Button size="sm" variant="primary" disabled={!loggedIn} onClick={queryGalaxies}>Query</Button></div>
          </div>
        </Popover>
        <Menu label="More catalogue actions" items={more}
          trigger={<IconButton icon="more" label="More catalogue actions" size="sm" loading={busy != null} />} />
        <IconButton icon="reset" label="Refresh" size="sm" onClick={reload} />
      </div>

      {(grouped.job || grouped.error) && <JobProgress job={grouped.job} error={grouped.error} />}
      {(galaxies.job || galaxies.error) && <JobProgress job={galaxies.job} error={galaxies.error} />}

      {runs.loading ? <Skeleton lines={6} /> : !data ? (
        <Callout tone="bad" title="Could not load the catalogue evaluation" action={<Button size="sm" onClick={reload}>Retry</Button>}>
          {runs.error?.message ?? "No data."}
        </Callout>
      ) : (
        <>
          <div className="res-ident" aria-label="Current model">
            <span>Now: <strong>{cur?.n_members ?? 0}</strong> STARFULL members · {cur?.combiner_kind ? <Badge size="sm" tone="accent">{cur.combiner_kind.replace(/_/g, " ")}</Badge> : <Badge size="sm" tone="warn">member mean</Badge>}</span>
            <span>{formatCount(data.n_ok)} / {formatCount(data.n)} ok</span>
            <StateBadge state="current" prefix={String(counts.current)} />
            <StateBadge state="stale" prefix={String(counts.stale)} />
            <StateBadge state="unknown" prefix={String(counts.unknown)} />
          </div>
          {counts.stale + counts.unknown > 0 && (
            <Callout tone="warn" title={`${formatCount(counts.stale + counts.unknown)} reconstructions predate the current model`}
              action={<Button size="sm" onClick={() => setGroupedOpen(true)}>Grouped analysis…</Button>}>
              The grouped analysis regenerates them with the STARFULL members and the production combiner.
            </Callout>
          )}
          <div className="res-eval">
            <DataTable rows={rows} columns={cols} rowKey={(r) => String(r.viewer_id || r.out_subdir || r.id)} aria-label="Evaluation objects"
              activeKey={active} onRowClick={(r) => { if (r.viewer_id) void viewer.current?.goToId(String(r.viewer_id)); }}
              exportName="eval-results" urlKey="ce" dense height="max(420px, calc(100vh - 330px))"
              filterPlaceholder="Filter: grade:A  state:stale  field:EDF-S  flux<0.8"
              empty={data.n ? "No object matches these filters." : "No evaluation results yet — run the grouped analysis."} />
            <div className="res-eval__viewer">
              {data.n_ok ? (
                <ImageViewer collection="evaluation" urlKey="cev" id="catalog-eval" toolbar="full"
                  onReady={(api) => { viewer.current = api; }}
                  onState={(s) => { if (s.id !== active) setActive(s.id ?? null); }} />
              ) : <p className="muted res-note">No reconstructions to browse yet.</p>}
            </div>
          </div>
        </>
      )}

      <Card><CardBody>
        <Section title="Figures" sub="rendered on request" collapsible open={figs} onOpenChange={setFigs}>
          <div className="res-figs">
            <EvalFigure title="SR → HR recovery" sub="synthetic objects" url="/api/evaluation/transformation" />
            <EvalFigure title="Angular power spectrum" sub="validation fields, HR vs SR" url="/api/evaluation/angular-power-spectrum" />
          </div>
        </Section>
      </CardBody></Card>
    </Page>
  );
}
