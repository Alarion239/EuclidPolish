/* Synthetic › PSF: the chain real Euclid Q1 stars → their cutouts → the
   empirical PSFs synthetic scenes are convolved with.

   The header sentence ("Real Euclid Q1 stars the empirical PSFs are built
   from: 43,401 stars, 17,917 usable (valid in all 4 bands at 511 px)"), then
   the view (?view=): catalogue (default: filters, the star table, the
   magnitude histogram), cutouts (the navigator with the target marked, the
   cached gallery, the per-band validity bars) and epsf (what generation
   uses, the kernel viewer with FWHM and the live warps, ePSF vs Gaussian
   FWHM, the cluster FWHM map). Switching views drops the other view's
   `?band=` (the catalogue's band filter, the gallery's band). The
   How-this-is-produced drawer (`?how=1`) holds, in chain order, the
   catalogue steps (euclid_query, euclid_verify_photometry) and "Pull
   stars.csv", the cutout download (download_euclid_cutouts), the ePSF steps
   (extract_euclid_psf, psf_rotation_pool) and the two ePSF syncs; every
   action is confirmed. Nothing starts on a visit. */
import { useState } from "react";
import { Link } from "react-router-dom";
import { apiPost, isFasrcOffline } from "../../../api/client";
import { useJob } from "../../../api/jobs";
import { invalidate } from "../../../api/query";
import { usePageActions } from "../../../app/palette";
import { StepById } from "../../../fasrc";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Button, EmptyState, Num, Page, Section, Segmented, SummaryLine, Toolbar, ToolbarGroup, ToolbarSpacer, Tooltip, confirm, toast,
} from "../../../ui";
import { ConfigKnobsLink } from "../../shared/ConfigKnobsLink";
import type { GroupId } from "../../system/configFields";
import { Drawer, DrawerButton, useDrawer } from "../common";
import { URLS, usePsfInventory, useStars } from "../dataApi";
import { JobStrip, LoadState, OFFLINE_HINT, startDataJob, useFasrcOnline } from "../dataCommon";
import { Catalogue } from "../psf/Catalogue";
import { Cutouts } from "../psf/Cutouts";
import { Epsf } from "../psf/Epsf";
import { catalogueHeadline } from "../psf/psfModel";
import "../register";
import "../data.css";
import "../synthetic.css";

/** The System › Config groups whose effect this tab judges ("N knobs changed · Edit"). */
const PSF_CONFIG_GROUPS: readonly GroupId[] = ["cutouts", "psf"];

export const PSF_VIEWS = [
  { value: "catalogue", label: "Catalogue" },
  { value: "cutouts", label: "Cutouts" },
  { value: "epsf", label: "ePSF" },
] as const;
type View = (typeof PSF_VIEWS)[number]["value"];
const VIEW_SET = new Set<string>(PSF_VIEWS.map((v) => v.value));

function HowDrawer() {
  const { online } = useFasrcOnline();
  const sync = useJob("data:psf-sync");
  const [pulling, setPulling] = useState(false);
  const pullCatalogue = async () => {
    if (!(await confirm({ title: "Pull stars.csv from FASRC?", confirmLabel: "Pull",
      message: "Re-copies the star catalogue from netscratch over the local mirror (a few MB)." }))) return;
    setPulling(true);
    try {
      await apiPost(URLS.refreshCatalog, {});
      await Promise.all([invalidate("/api/catalog/"), invalidate("/api/star-cutouts/"), invalidate("/api/status")]);
      toast.success("Star catalogue pulled from FASRC");
    } catch (e) {
      toast.error(isFasrcOffline(e) ? "FASRC not connected" : "Catalogue pull failed",
        { description: e instanceof Error ? e.message : String(e) });
    } finally { setPulling(false); }
  };
  const run = (url: string, label: string, message: string) => void startDataJob(sync, url, {}, {
    label, question: { title: `${label}?`, message, confirmLabel: "Sync" },
  });
  const syncAll = () => run(URLS.psfSync, "Sync the ePSFs from FASRC",
    "Force-pulls the four band ePSFs (the VIS stack is tens to hundreds of MB) and the cluster metadata.");
  const syncMeta = () => run(URLS.psfSyncMeta, "Sync the PSF cluster metadata",
    "Dumps the cluster centroids, star counts and FWHMs from the ePSF headers on FASRC (seconds) and pulls the small JSON.");
  const gated = (label: string, hint: string, button: JSX.Element) => (
    <Tooltip content={online ? hint : OFFLINE_HINT}><span>{button}</span></Tooltip>
  );
  return (
    <Drawer flag="how" title="How this is produced" sub="the star catalogue, its cutouts and the ePSFs, on FASRC">
      <div className="rl-stack">
        <h3 className="syn-subtitle">1 · Star catalogue</h3>
        <div className="rl-row">
          {gated("Pull stars.csv", "Re-pull stars.csv from FASRC (confirmed)",
            <Button size="sm" icon="download" loading={pulling} disabled={!online} onClick={() => void pullCatalogue()}>Pull stars.csv</Button>)}
        </div>
        <Section title="Query the catalogue" sub="euclid_query" collapsible defaultOpen={false}>
          <StepById stepId="euclid_query" embedded />
        </Section>
        <Section title="Verify the photometry scale" sub="euclid_verify_photometry" collapsible defaultOpen={false}>
          <StepById stepId="euclid_verify_photometry" embedded />
        </Section>
      </div>
      <div className="rl-stack">
        <div className="rl-row">
          <h3 className="syn-subtitle">2 · Cutouts</h3>
          <Button asChild size="sm" variant="ghost"><Link to="/system/connections">Euclid archive login</Link></Button>
        </div>
        <Section title="Download the star cutouts" sub="download_euclid_cutouts" collapsible defaultOpen={false}>
          <StepById stepId="download_euclid_cutouts" embedded />
        </Section>
      </div>
      <div className="rl-stack">
        <h3 className="syn-subtitle">3 · ePSFs</h3>
        <div className="rl-row">
          {gated("Sync ePSFs", "Force-pull the four band ePSFs and the cluster metadata (confirmed)",
            <Button size="sm" icon="download" loading={sync.busy} disabled={!online} onClick={syncAll}>Sync ePSFs</Button>)}
          {gated("Sync PSF cluster metadata", "The cluster centroids, star counts and FWHMs only, kilobytes (confirmed)",
            <Button size="sm" variant="ghost" icon="database" disabled={!online || sync.busy} onClick={syncMeta}>Sync PSF cluster metadata</Button>)}
        </div>
        <JobStrip job={sync} />
        <Section title="Extract the ePSFs" sub="extract_euclid_psf" collapsible defaultOpen={false}>
          <StepById stepId="extract_euclid_psf" embedded />
        </Section>
        <Section title="Pre-rotate the kernel pools" sub="psf_rotation_pool" collapsible defaultOpen={false}>
          <StepById stepId="psf_rotation_pool" embedded />
        </Section>
      </div>
    </Drawer>
  );
}

export default function PsfTab() {
  const [rawView, setRawView] = useUrlState("view", "catalogue");
  const [, setBand] = useUrlState("band", "");
  const [, setGallery] = useUrlState("gpage", 1);
  const view: View = (VIEW_SET.has(rawView) ? rawView : "catalogue") as View;
  // The views share one URL: the catalogue's `band` filter and the gallery's
  // `band` / page belong to their own view.
  const setView = (v: string) => { setRawView(v); setBand(""); setGallery(1); };
  const stars = useStars();
  const inv = usePsfInventory();
  const how = useDrawer("how");
  const head = catalogueHeadline(stars.data?.summary);
  usePageActions([
    ...PSF_VIEWS.map((v) => ({ id: `psf-view-${v.value}`, label: `PSF: ${v.label}`, group: "PSF", run: () => setView(v.value) })),
    { id: "psf-how", label: "PSF: how this is produced (catalogue, cutouts, ePSF steps)", group: "PSF",
      keywords: ["euclid_query", "download_euclid_cutouts", "extract_euclid_psf", "psf_rotation_pool", "sync"], run: how.reveal },
  ]);
  return (
    <Page className="dt-page rl-page syn-page">
      {head ? (
        <SummaryLine>
          Real Euclid Q1 stars the empirical PSFs are built from: <Num>{head.total}</Num> stars, <Num>{head.usable}</Num> usable
          {head.size ? <> (valid in all 4 bands at <Num unit="px">{head.size}</Num>)</> : " (valid in all 4 bands)"}
        </SummaryLine>
      ) : null}
      <Toolbar label="PSF controls">
        <ToolbarGroup label="View" hideLabel>
          <Segmented size="sm" aria-label="PSF view" value={view} onChange={setView} options={[...PSF_VIEWS]} />
        </ToolbarGroup>
        <ToolbarSpacer />
        <ConfigKnobsLink groups={PSF_CONFIG_GROUPS} className="syn-config-link" />
        <DrawerButton flag="how" icon="database" hint="The catalogue query, the cutout download, the ePSF extraction and the syncs">How this is produced</DrawerButton>
      </Toolbar>
      {view === "epsf" ? (
        <LoadState loading={inv.loading && !inv.data} error={inv.error} onRetry={() => void inv.reload()}>
          {inv.data && <Epsf data={inv.data} />}
        </LoadState>
      ) : (
        <LoadState loading={stars.loading && !stars.data} error={stars.error} onRetry={() => void stars.reload()} lines={6}>
          {stars.data && !stars.data.present ? (
            <EmptyState icon="database" title="The FASRC star catalogue is not synchronised"
              action={<Button variant="primary" onClick={how.reveal}>How this is produced</Button>}>
              Pull stars.csv, or build it with the euclid_query step.
            </EmptyState>
          ) : stars.data && (view === "cutouts" ? <Cutouts data={stars.data} /> : <Catalogue data={stars.data} />)}
        </LoadState>
      )}
      <HowDrawer />
    </Page>
  );
}
