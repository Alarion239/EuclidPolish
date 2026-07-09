/* Visualization — the classic Flask "Visualization" page, ported.

   The classic page rendered a gallery of PNGs under data/vis/ from a server
   render context. There is NO JSON listing endpoint for those PNGs (files.py
   only exposes the raw byte server at /vis/<relpath>), so — per the port brief
   — this page instead surfaces the three standard diagnostic figures that DO
   have live PNG endpoints (/view/*), plus a FITS-inspector helper that links
   into the universal /inspect page. */
import { useState } from "react";
import {
  Button, Card, CardBody, CardHead, Field, Input, Page, PageHead, PngFigure,
} from "../ui";

export default function VisualizationPage() {
  // Cache-buster for the training-log "regenerate" chip: bumping this forces a
  // fresh ?force=1 render (the endpoint re-plots the CSV when force is set).
  const [logNonce, setLogNonce] = useState<number | null>(null);
  const trainingLogSrc = () =>
    logNonce == null
      ? "/view/training-log"
      : `/view/training-log?force=1&t=${logNonce}`;

  // FITS inspector helper: a project-relative path opens the universal
  // inspector in a new tab; a live preview renders inline once a path exists.
  const [fits, setFits] = useState("");
  const path = fits.trim();
  const openInspector = () => {
    if (!path) return;
    window.open(`/inspect?fits=${encodeURIComponent(path)}`, "_blank");
  };

  return (
    <Page>
      <PageHead
        eyebrow="pipeline · visualization"
        title="Visualization"
        sub="Standard diagnostic figures across the pipeline, plus a jump into the universal FITS inspector."
      />

      <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "var(--s4)" }}>
        <Card>
          <CardHead
            title="Training curve"
            sub="First active ensemble member's training_log.csv"
            right={
              <button
                className="ui-chip"
                onClick={() => setLogNonce(Date.now())}
                title="Force a fresh render of the training-log PNG"
              >
                regenerate
              </button>
            }
          />
          <CardBody>
            <PngFigure srcFor={trainingLogSrc} alt="training log" minHeight={280} />
          </CardBody>
        </Card>

        <Card>
          <CardHead title="ePSFs" sub="All bands (/view/psfs?band=all)" />
          <CardBody>
            <PngFigure
              srcFor={() => "/view/psfs?band=all"}
              alt="ePSF panel"
              minHeight={280}
            />
          </CardBody>
        </Card>

        <Card>
          <CardHead title="Catalog positions" sub="/view/catalog?view=positions" />
          <CardBody>
            <PngFigure
              srcFor={() => "/view/catalog?view=positions"}
              alt="catalog positions"
              minHeight={280}
            />
          </CardBody>
        </Card>

        <Card>
          <CardHead
            title="FITS inspector"
            sub="Open any project-relative FITS in the universal inspector"
          />
          <CardBody>
            <Field label="Project-relative FITS path">
              <Input
                value={fits}
                onChange={setFits}
                placeholder="e.g. data/euclid_inference/…/cutout.fits"
                onEnter={openInspector}
              />
            </Field>
            <div className="row" style={{ marginTop: "var(--s3)", gap: "var(--s2)" }}>
              <Button variant="primary" disabled={!path} onClick={openInspector}>
                Open in inspector
              </Button>
            </div>
            {path && (
              <div style={{ marginTop: "var(--s3)" }}>
                <PngFigure
                  srcFor={() =>
                    `/inspect/preview.png?fits=${encodeURIComponent(path)}&size=512`}
                  alt={`preview of ${path}`}
                  minHeight={280}
                />
              </div>
            )}
          </CardBody>
        </Card>
      </div>
    </Page>
  );
}
