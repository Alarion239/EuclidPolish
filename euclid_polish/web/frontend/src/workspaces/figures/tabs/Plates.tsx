/* Figures › Plates (spec §8.5): the presentation plates — galaxy population
 * calibration, stellar calibration, the galaxy-distribution 2×2 — plus the
 * NEXUS × Euclid comparison plates (a local render job over any model spec)
 * and the poster cutout. One plate at a time (`?plate=`); the export dpi and
 * the training toggle are in the URL too. Catalogue, PSF and training-curve
 * figures live on their own pages (Data › Catalog / PSFs, Ensemble › Curves). */
import { usePageActions } from "../../../app/palette";
import { useUrlState } from "../../../hooks/useUrlState";
import { Page, Section, Segmented, Select, Gallery, Spinner, type SelectOption } from "../../../ui";
import { useResource } from "../../../api/query";
import { useFigBarHeight } from "../common";
import { NexusPlates } from "../plates/NexusPlates";
import { PosterPlate } from "../plates/PosterPlate";
import { StaticPlate, type StaticPlateDef } from "../plates/StaticPlate";
import "../figures.css";
import "../register";

const plateUrl = (path: string) => ({ format, dpi, inline, training }: { format: string; dpi: number; inline?: boolean; training?: boolean }) => {
  const q = new URLSearchParams({ format, dpi: String(dpi) });
  if (inline) q.set("inline", "1");
  if (training) q.set("include_training", "1");
  return `${path}?${q.toString()}`;
};

const STATIC_PLATES: StaticPlateDef[] = [
  { id: "population", title: "Galaxy population calibration", minHeight: 420,
    sub: "Q1 VIS counts (bright bridge / main / flat) × the truncated-Gaussian VIS Sérsic Rₑ law",
    url: plateUrl("/view/population-atlas"), source: { to: "/realism/galaxies", label: "Galaxies" } },
  { id: "stars", title: "Stellar population calibration", minHeight: 460,
    sub: "Q1 VIS × Gaia G_AB shared-slope counts · fitted and noise-tested colours",
    url: plateUrl("/view/star-population-calibration"), source: { to: "/realism/stars", label: "Stars" } },
  { id: "galaxies", title: "Galaxy distributions", minHeight: 460, training: true,
    sub: "Q1 aggregates vs generated sources under the active law (2 × 2)",
    url: plateUrl("/view/galaxy-distribution-plate"), source: { to: "/realism/galaxies", label: "Galaxies" } },
];

const PLATES = [
  ...STATIC_PLATES.map((p) => ({ value: p.id, label: p.id === "population" ? "Galaxy law" : p.id === "stars" ? "Stars" : "Galaxy 2×2" })),
  { value: "nexus", label: "NEXUS" },
  { value: "poster", label: "Poster" },
];
const DPI_OPTIONS: SelectOption[] = [150, 300, 600].map((d) => ({ value: String(d), label: `${d} dpi` }));

type VisPng = { rel: string; mtime: number; size_kb: number; inspect_fits: string | null };

/** The data/vis PNG gallery (pipeline renders, archived poster previews). */
function RenderedPngs() {
  const gallery = useResource<{ pngs: VisPng[] }>("/api/vis/list.json", [], { ttl: 60_000 });
  const pngs = gallery.data?.pngs ?? [];
  if (gallery.loading && !gallery.data) return <Spinner label="Loading renders" />;
  return (
    <Gallery thumb={140} empty={gallery.error ? gallery.error.message : "No PNGs under data/vis/ yet"}
      items={pngs.slice(0, 60).map((p) => ({
        src: `/vis/${p.rel}`,
        href: p.inspect_fits ? `/inspect?fits=${encodeURIComponent(p.inspect_fits)}` : `/vis/${p.rel}`,
        label: p.rel.split("/").pop(),
      }))} />
  );
}

export default function Plates() {
  const barRef = useFigBarHeight();
  const [plate, setPlate] = useUrlState<string>("plate", "population");
  const [dpi, setDpi] = useUrlState<string>("dpi", "300");
  const [training, setTraining] = useUrlState<boolean>("training", false);
  const active = PLATES.some((p) => p.value === plate) ? plate : "population";
  const staticPlate = STATIC_PLATES.find((p) => p.id === active);

  usePageActions(PLATES.map((p) => ({
    id: `fig-plate-${p.value}`, label: `Show the ${p.label} plate`, group: "Plates", keywords: ["figure", "publication"],
    run: () => setPlate(p.value),
  })));

  return (
    <Page className="fig-page">
      <div ref={barRef} className="fig-bar" role="toolbar" aria-label="Plates">
        <Segmented size="sm" value={active} onChange={setPlate} aria-label="Plate" options={PLATES} />
        <span className="fig-bar__spacer" />
        {staticPlate && <Select size="sm" value={dpi} onChange={setDpi} options={DPI_OPTIONS} aria-label="Download resolution" />}
      </div>
      {staticPlate && <StaticPlate plate={staticPlate} dpi={Number(dpi) || 300} training={training} onTraining={setTraining} />}
      {active === "nexus" && <NexusPlates />}
      {active === "poster" && <PosterPlate />}
      <Section title="Rendered PNGs" sub="data/vis" collapsible defaultOpen={false}>
        <RenderedPngs />
      </Section>
    </Page>
  );
}
