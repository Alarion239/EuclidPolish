/* Synthetic › Fields: synthetic vs real LR fields — the LR pixels the model
   is trained on (synthetic test + validate dirty records) against the
   multipoint Euclid archive fields.

   The title and the verdict first: the per-band scale-spectrum similarity
   of VIS as one line ("VIS overlap 0.98 (0.97–0.98), power syn/real 0.83
   (0.68–1.20)"); a stale cache keeps its last result behind a badge with an
   explicit Measure button (confirmed; never run on a visit). Then the view
   (?view=): look (default: the two lanes on one locked transfer), stats (the
   sample chips carrying their sizes, the band chips, the six figures, a
   caption linking the background-noise figure on Noise, the median field
   metrics, the geometry caption) and detection (detections
   and negative islands per field, the completeness caption). The Real
   reference drawer (`?ref=1`) holds the multipoint archive collection (its
   fields and pointings, the sync, archive_field_sample). */
import { useEffect, useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { useFasrcStatus } from "../../../app/status";
import { bandColor } from "../../../colors";
import { StepById } from "../../../fasrc";
import { formatCount, formatDateTime } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Caption, Chip, EmptyState, JobProgress, Num, Page, Segmented, SummaryLine, Toolbar, ToolbarGroup, ToolbarSpacer, Tooltip, toast,
} from "../../../ui";
import { useArchiveMeta, usePixels, type FieldComparison } from "../api";
import { archiveFieldBreakdown, archiveOverview, shortArchiveFingerprint } from "../archiveFields";
import { Drawer, DrawerButton, LoadState, SkyLink, useDrawer } from "../common";
import { Look, syncArchiveFields } from "../fields/Look";
import { BANDS, SAMPLES, bandLabel, sampleChipLabel } from "../fields/model";
import { BandLedger, Detection, GeometryCaption, PixelFigures, buildStatistics, useToggles } from "../fields/Stats";
import { JOB, offlinePolicy, useRealismJob } from "../jobs";
import { lastComparison, listText, powerOff, scaleVerdict } from "../statusModel";
import "../synthetic.css";

export const FIELD_VIEWS = [
  { value: "look", label: "Look" },
  { value: "stats", label: "Statistics" },
  { value: "detection", label: "Detection" },
] as const;
type View = (typeof FIELD_VIEWS)[number]["value"];
/** The absorbed pages' own values for the view key. */
const ALIASES: Record<string, View> = { pixels: "stats", census: "stats", inputs: "stats" };

const interval2 = (i: { median: number; p16: number; p84: number }) => `(${i.p16.toFixed(2)}–${i.p84.toFixed(2)})`;

/** The verdict: how closely the synthetic fields place their power across
 *  scales, VIS in full, then every band whose power ratio is off (warn). */
function Verdict({ fields }: { fields: FieldComparison }) {
  const s = fields.scale_similarity.VIS;
  const v = scaleVerdict(fields);
  if (!s || !v) return <SummaryLine>This field-statistics cache has no VIS scale-spectrum score.</SummaryLine>;
  const nisp = v.off.filter((b) => b.band !== "VIS");
  const powers = nisp.map((b) => b.power);
  const span = powers.length ? [Math.min(...powers), Math.max(...powers)].map((p) => p.toFixed(2)) : [];
  return (
    <SummaryLine>
      VIS overlap <Num>{s.overlap.median.toFixed(2)}</Num> {interval2(s.overlap)}, power syn/real{" "}
      <Num tone={powerOff(s.variance_ratio.median) ? "warn" : undefined}>{s.variance_ratio.median.toFixed(2)}</Num> {interval2(s.variance_ratio)}
      {nisp.length > 0 && <>; {listText(nisp.map((b) => bandLabel(b.band)))} power syn/real{" "}
        <Num tone="warn">{span[0] === span[1] ? span[0] : `${span[0]}–${span[1]}`}</Num></>}
    </SummaryLine>
  );
}

function ReferenceDrawer() {
  const archive = useArchiveMeta();
  const sync = useRealismJob(JOB.archiveSync);
  const info = archive.data?.archive;
  const fasrc = useFasrcStatus().data;
  // The archive sync is a self-connecting job: enabled offline, and says so.
  const syncHint = offlinePolicy({ self_connects: true }, fasrc ? !fasrc.ssh_connected : false).hint ?? undefined;
  const stale = !!info?.valid && !!info.complete && !info.current;
  return (
    <Drawer flag="ref" title="Real reference" sub="the multipoint Euclid archive fields the synthetic fields are compared with"
      right={stale ? <Badge tone="warn" dot>source changed</Badge> : info && !info.ready ? <Badge tone="warn" dot>not synchronised</Badge> : undefined}>
      <p className="rl-note">
        {info?.ready ? `${archiveOverview(info)}. Per field: ${archiveFieldBreakdown(info)}.` : archiveOverview(info)}
      </p>
      <div className="rl-row">
        <Tooltip content={syncHint ?? "Pulls the multipoint manifest and its four-band FITS bundles (confirmed)"}>
          <Button variant="primary" icon="download" loading={sync.busy} onClick={() => void syncArchiveFields()}>
            Sync archive fields from FASRC
          </Button>
        </Tooltip>
        <SkyLink layers={["q1-tiles:0.2", "archive-fields"]} hint="Every archive field on the sky atlas">All fields on sky</SkyLink>
      </div>
      <JobProgress job={sync.job} error={sync.error} />
      {info?.source_plan_fingerprint && <Caption>{`Sample plan ${shortArchiveFingerprint(info.source_plan_fingerprint)}`}</Caption>}
      <StepById stepId="archive_field_sample" embedded />
    </Drawer>
  );
}

export default function FieldsTab() {
  const [rawView, setView] = useUrlState("view", "look");
  const view: View = (FIELD_VIEWS.some((v) => v.value === rawView) ? rawView : ALIASES[rawView] ?? "look") as View;
  const t = useToggles();
  // The statistics read no training variant (test + validate records only).
  const resource = usePixels(false);
  const payload = resource.data;
  const build = useRealismJob(JOB.pixelsBuild);
  const archive = useArchiveMeta();
  const ref = useDrawer("ref");
  const navigate = useNavigate();
  const last = lastComparison(payload);
  const comparison = last.comparison;
  const cache = payload?.availability.comparison_cache;
  const realReady = !!payload?.availability.real.ready;
  const built = comparison?.provenance?.generated_at;
  // An absorbed page's value (Pixels › census, …) settles on its new view.
  useEffect(() => { if (ALIASES[rawView]) setView(ALIASES[rawView]); }, [rawView, setView]);
  const byParent = useMemo(() => {
    const map = new Map<string, string>();
    for (const o of archive.data?.objects ?? []) if (!map.has(o.parent_id)) map.set(o.parent_id, o.id ?? String(o.sample_id));
    return map;
  }, [archive.data]);
  const onRealField = (parent: string) => {
    const id = byParent.get(parent);
    if (id) openInspector({ kind: "archivefield", id });
    else toast.info(`Real field from pointing ${parent}`, { description: "Its archive sample is not in the local collection." });
  };
  const measure = () => void buildStatistics(payload);
  usePageActions([
    ...FIELD_VIEWS.map((v) => ({ id: `fields-view-${v.value}`, label: `Fields: ${v.label}`, group: "Fields", run: () => setView(v.value) })),
    { id: "fields-measure", label: "Measure the field statistics…", group: "Fields", keywords: ["pixels", "detection", "cache", "rebuild"],
      disabled: !realReady, run: measure },
    { id: "fields-reference", label: "Fields: the real reference (archive fields)", group: "Fields", keywords: ["archive", "multipoint", "sync"],
      run: ref.reveal },
    { id: "fields-census", label: "Open the records census", group: "Fields", run: () => navigate("/synthetic/records?section=census") },
    ...BANDS.map((b) => ({ id: `fields-band-${b}`, label: `${t.hidden.includes(b) ? "Show" : "Hide"} ${bandLabel(b)} in the field statistics`,
      group: "Fields", run: () => t.toggle(b) })),
  ]);
  const cacheTip = [cache?.reason, built ? `measured ${formatDateTime(built)}` : null].filter(Boolean).join(" · ") || undefined;
  return (
    <Page className="rl-page syn-page">
      <div className="syn-head">
        <h2 className="syn-title">Synthetic vs real LR fields</h2>
        <LoadState loading={resource.loading && !payload} error={resource.error} onRetry={resource.reload} lines={1}>
          <div className="syn-head__verdict">
            {comparison ? <Verdict fields={comparison.fields} />
              : <SummaryLine>The field statistics have not been measured yet.</SummaryLine>}
            {last.stale && (
              <Tooltip content={cacheTip ?? "The cache predates the current schema or inputs"}>
                <span tabIndex={0}><Badge size="sm" tone="warn" dot>last result</Badge></span>
              </Tooltip>
            )}
            {!comparison || last.stale ? (
              <Tooltip content={realReady ? "Streams every synthetic and real field through the statistics (confirmed; several minutes)"
                : payload?.availability.real.unavailable_reason ?? "The real reference is not ready (Real reference)"}>
                <span tabIndex={realReady ? -1 : 0}>
                  <Button size="sm" variant="primary" icon="reset" loading={build.busy} disabled={!realReady} onClick={measure}>Measure</Button>
                </span>
              </Tooltip>
            ) : cacheTip ? <Caption>{cacheTip}</Caption> : null}
          </div>
        </LoadState>
      </div>
      <JobProgress job={build.job} error={build.error} />
      <Toolbar label="Fields controls">
        <ToolbarGroup label="View" hideLabel>
          <Segmented size="sm" aria-label="Fields view" value={view} onChange={setView} options={[...FIELD_VIEWS]} />
        </ToolbarGroup>
        {view !== "look" && (
          <ToolbarGroup label="Show" hideLabel>
            <div className="rl-chips" role="group" aria-label="Bands and samples">
              {view === "stats" && BANDS.map((b) => (
                <Chip key={b} on={!t.hidden.includes(b)} dot={bandColor(b)} onClick={() => t.toggle(b)}>{bandLabel(b)}</Chip>
              ))}
              {SAMPLES.map((s) => (
                <Chip key={s} on={!t.hidden.includes(s)} onClick={() => t.toggle(s)}>
                  {sampleChipLabel(s, comparison, payload?.availability)}
                </Chip>
              ))}
            </div>
          </ToolbarGroup>
        )}
        <ToolbarSpacer />
        <DrawerButton flag="ref" icon="database" hint="The multipoint archive fields, their sync and archive_field_sample">Real reference</DrawerButton>
      </Toolbar>
      {view === "look" ? <Look onReference={ref.reveal} statsRealFields={last.stale ? comparison?.samples.real.fields : null} />
        : payload && !comparison ? (
          <EmptyState icon="table"
            title={realReady ? "The field statistics have not been measured" : "The multipoint Euclid reference is not ready"}
            action={realReady
              ? <Button variant="primary" loading={build.busy} onClick={measure}>Measure</Button>
              : <Button onClick={ref.reveal}>Real reference</Button>}>
            {realReady ? cache?.reason ?? "Streams every field once; the sources stay unchanged."
              : payload.availability.real.unavailable_reason ?? "Generate and synchronise the four-band archive fields first."}
          </EmptyState>
        ) : comparison && (
          view === "stats" ? (
            <div className="rl-stack">
              <PixelFigures comparison={comparison} t={t} onRealField={onRealField} />
              <Caption>The realised background σ per band, and the background vs robust noise per field, are on{" "}
                <Link to="/synthetic/noise">Synthetic › Noise</Link>.</Caption>
              <BandLedger fields={comparison.fields} />
              <GeometryCaption comparison={comparison} />
            </div>
          ) : (
            <div className="rl-stack">
              <Detection detection={comparison.fields.source_detection} t={t} />
              <GeometryCaption comparison={comparison} />
            </div>
          )
        )}
      {payload && view !== "look" && comparison && last.stale && payload.availability.real.compared_fields != null
        && payload.availability.real.compared_fields !== comparison.samples.real.fields && (
        <Caption>
          {`This result compared ${formatCount(comparison.samples.real.fields)} real fields; a new measurement compares `
            + `${formatCount(payload.availability.real.compared_fields)}${comparison.samples.real.fields > payload.availability.real.compared_fields
              ? `, leaving out the ${formatCount(comparison.samples.real.fields - payload.availability.real.compared_fields)} centre tiles, which were placed to avoid bright stars`
              : ""}.`}
        </Caption>
      )}
      <ReferenceDrawer />
    </Page>
  );
}
