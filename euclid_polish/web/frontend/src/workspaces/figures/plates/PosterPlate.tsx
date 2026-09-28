/* The synthetic poster scene: the last pulled `poster_cutout` result (a
 * synthetic scene, served locally, works offline), an explicit confirmed
 * pull from FASRC, and the step that makes a new one on the cluster in the
 * "How this is produced" drawer (`?how=1`). Not the real poster galaxy (Sky ›
 * Targets, set Poster galaxy). */
import { useState } from "react";
import { Link } from "react-router-dom";
import { isFasrcOffline } from "../../../api/client";
import { useResource } from "../../../api/query";
import { usePageActions } from "../../../app/palette";
import { useFasrcStatus } from "../../../app/status";
import { StepById } from "../../../fasrc";
import { useUrlState } from "../../../hooks/useUrlState";
import { Button, Callout, Section, Select, Tooltip, confirm, toast } from "../../../ui";
import { URLS, posterExportUrl, pullPoster, type PlateFormat, type PosterStatus } from "../api";
import { ServerImage } from "../common";
import { PlateHead } from "./PlateHead";
import { posterCaption } from "./plateStatus";
import { DPI_OPTIONS } from "./StaticPlate";

export function PosterPlate() {
  const status = useResource<PosterStatus>(URLS.poster, [], { ttl: 10_000 });
  const [pulling, setPulling] = useState(false);
  const [how, setHow] = useUrlState<boolean>("how", false);
  const [dpiRaw, setDpi] = useUrlState<string>("dpi", "300");
  const dpi = Number(dpiRaw) || 300;
  const fasrc = useFasrcStatus().data;
  /* unknown (still loading) counts as online: the pull reports the error itself */
  const offline = fasrc ? !fasrc.ssh_connected : false;
  const st = status.data;
  const pull = async () => {
    if (offline || pulling) return;
    const ok = await confirm({
      title: "Pull the latest poster cutout from FASRC?",
      message: "Copies the cluster's poster_cutout PNG and FITS over the local copies; a changed PNG is also archived to data/vis/poster.",
      confirmLabel: "Pull",
    });
    if (!ok) return;
    setPulling(true);
    try {
      const r = await pullPoster();
      toast.success(r.archived ? `Pulled · archived to ${r.archived}` : "Pulled (unchanged)");
    } catch (e) {
      toast.error(isFasrcOffline(e) ? "FASRC is not connected — connect in System › Connections" : e instanceof Error ? e.message : String(e));
    } finally {
      setPulling(false);
      void status.reload();
    }
  };
  usePageActions([{ id: "fig-poster-pull", label: "Pull the latest poster cutout from FASRC…", group: "Plates", disabled: offline || pulling, run: () => void pull() }]);
  const version = st?.png?.mtime ? `?v=${Math.round(st.png.mtime)}` : "";
  return (
    <section className="fig-plate" aria-labelledby="fig-plate-poster">
      <PlateHead id="poster" title="Synthetic poster scene" caption={posterCaption(st)}
        sub="A synthetic scene rendered for the poster; the real poster galaxy is a Sky › Targets set."
        tools={<>
          {offline ? (
            <Tooltip content={`FASRC is offline${fasrc?.last_error ? `: ${fasrc.last_error}` : ""} — connect in System › Connections`}>
              <span tabIndex={0} className="fig-badge-wrap"><Button size="sm" icon="download" disabled>Pull latest</Button></span>
            </Tooltip>
          ) : (
            <Button size="sm" icon="download" loading={pulling} onClick={() => void pull()}>Pull latest</Button>
          )}
          <Tooltip content="Print resolution: it sets the printed size of the node's render (the pixels are not resampled)">
            <span><Select size="sm" value={String(dpi)} onChange={setDpi} options={DPI_OPTIONS} aria-label="Download resolution" /></span>
          </Tooltip>
          {(["png", "pdf", "svg"] as PlateFormat[]).map((f) => (
            <Button key={f} size="sm" icon={f === "png" ? "download" : undefined} disabled={!st?.png}
              href={st?.png ? posterExportUrl(f, dpi) : undefined} download>{f.toUpperCase()}</Button>
          ))}
          <Button size="sm" disabled={!st?.fits} href={st?.fits ? "/poster/result/cutout.fits" : undefined} download>FITS</Button>
          <Button asChild size="sm" variant="ghost" iconRight="chevronRight"><Link to="/runs/steps?step=poster_cutout">Steps</Link></Button>
        </>} />
      {status.error && <Callout tone="bad" title="Poster status did not load">{status.error.message}</Callout>}
      <ServerImage src={st?.png ? `/poster/result/cutout.png${version}` : null} alt="Synthetic poster scene" minHeight={st?.png ? 320 : 120}>
        {st && !st.png && <p className="fig-note muted fig-image__empty">No scene pulled yet: generate one on FASRC (How this is produced), then Pull latest.</p>}
      </ServerImage>
      <Section id="fig-drawer-how" className="fig-drawer" title="How this is produced" sub="the poster_cutout step on FASRC"
        collapsible open={how} onOpenChange={setHow}>
        <StepById stepId="poster_cutout" embedded />
      </Section>
    </section>
  );
}
