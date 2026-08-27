/* Presentation-figure index plus the existing operational render gallery. */
import { useState } from "react";
import { NavLink } from "react-router-dom";
import { asArray } from "../data";
import { useResource } from "../hooks";
import { StepById } from "../fasrc";
import {
  Button, Card, CardBody, CardHead, Empty, Field, Gallery, Input, Page,
  PageHead, PngFigure, Spinner,
} from "../ui";
import FigureGridBuilder from "./figure-grid/FigureGridBuilder";
import "./presentation-figures.css";

type VisPng = { rel: string; mtime: number; size_kb: number; inspect_fits: string | null };

const FIGURE_BUILDERS = [
  { label: "Synthetic comparison", path: "/sky", tiers: ["Euclid Image", "Super-resolved Image", "High-resolution truth"],
    note: "Use the held-out synthetic records for the LR–SR–HR plate." },
  { label: "Evaluation fields", path: "/evaluation", tiers: ["Euclid Image", "Super-resolved Image", "High-resolution truth"],
    note: "Browse selected evaluation fields and ensemble products." },
  { label: "Real Euclid fields", path: "/inference", tiers: ["Euclid Image", "Super-resolved Image"],
    note: "Export cached real-field reconstructions without implying an unavailable truth image." },
  { label: "Euclid × JWST", path: "/jwst-euclid", tiers: ["Euclid Image", "Super-resolved Image", "JWST reference"],
    note: "Compare registered native-band Euclid and JWST views." },
  { label: "Selected ePSFs", path: "/psfs", tiers: ["VIS", "Y_E", "J_E", "H_E"],
    note: "Choose the empirical PSF and place the magnified region with the pointer." },
] as const;

export default function VisualizationPage() {
  const gallery = useResource<{ pngs: VisPng[] }>("/api/vis/list.json");
  const pngs = asArray<VisPng>(gallery.data?.pngs);
  const items = pngs.slice(0, 60).map((p) => ({
    src: `/vis/${p.rel}`,
    href: p.inspect_fits ? `/inspect?fits=${encodeURIComponent(p.inspect_fits)}` : `/vis/${p.rel}`,
    label: p.rel.split("/").pop(),
  }));
  // Cache-buster for the training-log "regenerate" chip: bumping this forces a
  // fresh ?force=1 render (the endpoint re-plots the CSV when force is set).
  const [logNonce, setLogNonce] = useState<number | null>(null);
  const [catalogView, setCatalogView] = useState("positions");
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
        eyebrow="presentation · publication exports"
        title="Presentation figures"
        sub="One index for calibrated population plots, catalog and PSF plates, and every cursor-driven image comparison."
      />

      <FigureGridBuilder />

      <section className="presentation-hero" aria-label="Figure export workflow">
        <div>
          <span className="eyebrow">viewer export</span>
          <h2>Select the feature. Export matched crops.</h2>
          <p>Click to freeze a matched region, then press S or use Save crop to results. Figure exports the same crop from every selected tier.</p>
        </div>
        <div className="presentation-hero__sequence" aria-label="Pointer to figure workflow">
          <span>select tiers</span><b>→</b><span>click region</span><b>→</b><span>⬇ Figure</span>
        </div>
      </section>

      <div className="presentation-section-head">
        <div><span className="eyebrow">calibration plates</span><h2>Population constraints</h2></div>
        <p>Cached reviewed artifacts only; opening this page never fits or activates a calibration.</p>
      </div>

      <div className="presentation-plate-stack">
        <Card className="presentation-plate">
          <CardHead title="Galaxy population calibration"
            sub="Q1 VIS 2FWHM continuous three-segment bright bridge/main/flat counts × one straight truncated-Gaussian circularized VIS Sérsic Rₑ law"
            right={<div className="presentation-plate__formats">
              <a href="/view/population-atlas?format=pdf" download>PDF</a>
              <a href="/view/population-atlas?format=svg" download>SVG</a>
            </div>} />
          <CardBody>
            <PngFigure
              srcFor={() => "/view/population-atlas?format=png&dpi=150&inline=1"}
              downloadSrc={() => "/view/population-atlas?format=png&dpi=300"}
              alt="Galaxy population calibration"
              minHeight={420}
            />
          </CardBody>
        </Card>

        <Card className="presentation-plate">
          <CardHead title="Stellar population calibration"
            sub="Q1 VIS × Gaia G_AB shared-slope straight counts · fitted and noise-tested colours"
            right={<div className="presentation-plate__formats">
              <a href="/view/star-population-calibration?format=pdf" download>PDF</a>
              <a href="/view/star-population-calibration?format=svg" download>SVG</a>
            </div>} />
          <CardBody>
            <PngFigure
              srcFor={() => "/view/star-population-calibration?format=png&dpi=150&inline=1"}
              downloadSrc={() => "/view/star-population-calibration?format=png&dpi=300"}
              alt="Stellar population calibration"
              minHeight={480}
            />
          </CardBody>
        </Card>
      </div>

      <div className="presentation-section-head">
        <div><span className="eyebrow">catalog and optics</span><h2>Observed inputs</h2></div>
        <p>Large-label plots with explicit coordinates, units, selection state, and comparable colour scales.</p>
      </div>

      <div className="presentation-observed-grid">
        <Card className="presentation-plate presentation-plate--wide">
          <CardHead title="Stellar catalog"
            sub="ICRS positions, VIS magnitude distribution, or four-band validity" />
          <CardBody>
            <PngFigure
              srcFor={(view) => `/view/catalog?view=${view ?? "positions"}&dpi=150`}
              toolbar={[
                { key: "positions", label: "positions" },
                { key: "magnitudes", label: "magnitudes" },
                { key: "saturation", label: "validity" },
              ]}
              active={catalogView}
              onActive={setCatalogView}
              downloadSrc={(view) => `/view/catalog?view=${view ?? "positions"}&dpi=300`}
              alt="Stellar catalog presentation figure"
              minHeight={410}
            />
          </CardBody>
        </Card>

        <Card className="presentation-plate">
          <CardHead title="PSF extraction clusters"
            sub="ICRS cluster membership and angular diameter by disjoint sky region" />
          <CardBody>
            <PngFigure
              srcFor={() => "/view/psf-clusters?dpi=150"}
              downloadSrc={() => "/view/psf-clusters?dpi=300"}
              alt="PSF extraction clusters"
              minHeight={380}
            />
          </CardBody>
        </Card>

        <Card className="presentation-plate">
          <CardHead title="Four-band empirical PSFs"
            sub="VIS, Y_E, J_E, and H_E on one shared logarithmic intensity scale" />
          <CardBody>
            <PngFigure
              srcFor={() => "/view/psfs?band=all&dpi=150"}
              downloadSrc={() => "/view/psfs?band=all&dpi=300"}
              alt="Four-band empirical PSFs"
              minHeight={320}
            />
          </CardBody>
        </Card>
      </div>

      <div className="presentation-section-head">
        <div><span className="eyebrow">interactive builders</span><h2>Image comparisons</h2></div>
        <p>Select the listed tiers and click a region for matched crop-only export; leave it unselected for full images.</p>
      </div>

      <div className="figure-builder-grid">
        {FIGURE_BUILDERS.map((builder) => (
          <article className="figure-builder" key={builder.path}>
            <div className="figure-builder__head">
              <h3>{builder.label}</h3>
              <NavLink className="ui-btn ui-btn--sm" to={builder.path}>Open viewer</NavLink>
            </div>
            <p>{builder.note}</p>
            <div className="figure-builder__tiers">
              {builder.tiers.map((tier) => <span key={tier}>{tier}</span>)}
            </div>
          </article>
        ))}
      </div>

      <div className="presentation-section-head presentation-section-head--diagnostics">
        <div><span className="eyebrow">working material</span><h2>Pipeline diagnostics</h2></div>
        <p>Operational renders remain available below, separate from the presentation set.</p>
      </div>

      <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "var(--s4)" }}>
        <Card>
          <CardHead title="Rendered figures" sub={`data/vis/ · ${pngs.length} PNG(s), newest first`}
            right={<Button size="sm" variant="ghost" onClick={() => gallery.reload()}>↻</Button>} />
          <CardBody>
            {gallery.loading ? <Empty><Spinner /> loading…</Empty>
              : <Gallery items={items} thumb={150} empty="no PNGs under data/vis/ yet" />}
          </CardBody>
        </Card>

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

        <StepById stepId="poster_cutout" />
      </div>
    </Page>
  );
}
