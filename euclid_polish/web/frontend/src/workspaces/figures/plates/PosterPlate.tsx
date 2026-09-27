/* The synthetic poster cutout: the last pulled `poster_cutout` result (a
 * synthetic scene, served locally, works offline), an explicit pull from
 * FASRC, and the step card that generates a new one on the cluster. Not the
 * real "Poster target" tile (Sky › Real results / the atlas chip). */
import { useState } from "react";
import { isFasrcOffline } from "../../../api/client";
import { useResource } from "../../../api/query";
import { usePageActions } from "../../../app/palette";
import { useFasrcStatus } from "../../../app/status";
import { StepById } from "../../../fasrc";
import { formatBytes, formatDateTime } from "../../../format";
import { Button, Callout, Section, Tooltip, confirm, toast } from "../../../ui";
import { URLS, pullPoster, type PosterStatus } from "../api";
import { ServerImage } from "../common";

export function PosterPlate() {
  const status = useResource<PosterStatus>(URLS.poster, [], { ttl: 10_000 });
  const [pulling, setPulling] = useState(false);
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
      toast.error(isFasrcOffline(e) ? "FASRC is not connected — connect in Settings › Connections" : e instanceof Error ? e.message : String(e));
    } finally {
      setPulling(false);
      void status.reload();
    }
  };
  usePageActions([{ id: "fig-poster-pull", label: "Pull the latest poster cutout from FASRC…", group: "Plates", disabled: offline || pulling, run: () => void pull() }]);
  const version = st?.png?.mtime ? `?v=${Math.round(st.png.mtime)}` : "";
  return (
    <section className="fig-plate" aria-labelledby="fig-plate-poster">
      <header className="fig-plate__head">
        <div className="fig-plate__heading">
          <h2 id="fig-plate-poster">Synthetic poster cutout</h2>
          <p className="muted">{st?.png ? `Pulled ${formatDateTime(st.png.pulled_at)} · PNG ${formatBytes(st.png.size)}${st.fits ? ` · FITS ${formatBytes(st.fits.size)}` : ""}` : "The latest poster_cutout job result (a synthetic scene; the real Poster target is in Sky › Real results)"}</p>
        </div>
        <div className="fig-plate__tools">
          {offline ? (
            <Tooltip content={`FASRC is offline${fasrc?.last_error ? `: ${fasrc.last_error}` : ""} — connect in Settings › Connections`}>
              <span tabIndex={0} className="fig-badge-wrap"><Button size="sm" icon="download" disabled>Pull latest</Button></span>
            </Tooltip>
          ) : (
            <Button size="sm" icon="download" loading={pulling} onClick={() => void pull()}>Pull latest</Button>
          )}
          <Button size="sm" disabled={!st?.png} href={st?.png ? "/poster/result/cutout.png" : undefined} download>PNG</Button>
          <Button size="sm" disabled={!st?.fits} href={st?.fits ? "/poster/result/cutout.fits" : undefined} download>FITS</Button>
        </div>
      </header>
      {status.error && <Callout tone="bad" title="Poster status did not load">{status.error.message}</Callout>}
      <ServerImage src={st?.png ? `/poster/result/cutout.png${version}` : null} alt="Synthetic poster cutout" minHeight={st?.png ? 320 : 120}>
        {st && !st.png && <p className="fig-note muted fig-image__empty">No cutout pulled yet — generate one on FASRC (below), then Pull latest.</p>}
      </ServerImage>
      <Section title="Generate on FASRC" collapsible defaultOpen={false}>
        <StepById stepId="poster_cutout" embedded />
      </Section>
    </section>
  );
}
