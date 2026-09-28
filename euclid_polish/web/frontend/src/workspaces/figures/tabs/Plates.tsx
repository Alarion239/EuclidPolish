/* Figures › Plates (console regrouping): the figures for the paper and the
 * poster, one at a time (`?plate=`), each named after its title — Galaxy
 * population calibration, Galaxy distributions, Stellar population
 * calibration, NEXUS comparison and Synthetic poster scene. Every plate
 * carries one caption line: what it is made with, whether that is current
 * and when it was made (plates/plateStatus.ts). The calibration plates render
 * on request from the reviewed calibration (resolution in `?dpi=`, the galaxy
 * plate's training population in `?training=`); NEXUS plates are a confirmed
 * local render job; the poster scene is pulled from FASRC (confirmed).
 * Opening the page runs nothing. */
import { usePageActions } from "../../../app/palette";
import { useResource } from "../../../api/query";
import { useUrlState } from "../../../hooks/useUrlState";
import { Gallery, Page, Section, Segmented, Spinner } from "../../../ui";
import { NexusPlates } from "../plates/NexusPlates";
import { PosterPlate } from "../plates/PosterPlate";
import { StaticPlate, type StaticPlateDef } from "../plates/StaticPlate";
import { staticPlateCaption, type OverviewSlice } from "../plates/plateStatus";
import "../figures.css";
import "../register";

/** Synthetic › Status readiness (read-only): which calibration each plate is drawn from. */
const OVERVIEW_URL = "/api/realism/overview";

const plateUrl = (path: string) => ({ format, dpi, inline, training }: { format: string; dpi: number; inline?: boolean; training?: boolean }) => {
  const q = new URLSearchParams({ format, dpi: String(dpi) });
  if (inline) q.set("inline", "1");
  if (training) q.set("include_training", "1");
  return `${path}?${q.toString()}`;
};

const STATIC_PLATES: StaticPlateDef[] = [
  { id: "population", title: "Galaxy population calibration", minHeight: 420,
    sub: "Q1 VIS counts (bright bridge / main / flat) × the truncated-Gaussian VIS Sérsic Rₑ law",
    url: plateUrl("/view/population-atlas"), source: { to: "/synthetic/galaxies?prior=1", label: "Galaxies" } },
  { id: "galaxies", title: "Galaxy distributions", minHeight: 460, training: true,
    sub: "Q1 aggregates vs generated sources under the active law (2 × 2)",
    url: plateUrl("/view/galaxy-distribution-plate"), source: { to: "/synthetic/galaxies", label: "Galaxies" } },
  { id: "stars", title: "Stellar population calibration", minHeight: 460,
    sub: "Q1 PHZ VIS counts and the fitted law · fitted and noise-tested colours",
    url: plateUrl("/view/star-population-calibration"), source: { to: "/synthetic/stars", label: "Stars" } },
];

/** The plate picker, named after the plate titles (the spec's order). */
export const PLATES = [
  { value: "population", label: "Galaxy population calibration" },
  { value: "galaxies", label: "Galaxy distributions" },
  { value: "stars", label: "Stellar population calibration" },
  { value: "nexus", label: "NEXUS comparison" },
  { value: "poster", label: "Synthetic poster scene" },
];

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
        href: p.inspect_fits ? `/files?fits=${encodeURIComponent(p.inspect_fits)}` : `/vis/${p.rel}`,
        label: p.rel.split("/").pop(),
      }))} />
  );
}

export default function Plates() {
  const [plate, setPlate] = useUrlState<string>("plate", "population");
  const [dpi, setDpi] = useUrlState<string>("dpi", "300");
  const [training, setTraining] = useUrlState<boolean>("training", false);
  const active = PLATES.some((p) => p.value === plate) ? plate : "population";
  const staticPlate = STATIC_PLATES.find((p) => p.id === active);
  const overview = useResource<OverviewSlice>(staticPlate ? OVERVIEW_URL : null, [staticPlate != null], { ttl: 60_000 });

  usePageActions(PLATES.map((p) => ({
    id: `fig-plate-${p.value}`, label: `Show the ${p.label} plate`, group: "Plates", keywords: ["figure", "publication"],
    run: () => setPlate(p.value),
  })));

  return (
    <Page className="fig-page">
      <div className="fig-bar" role="toolbar" aria-label="Plates">
        <Segmented size="sm" value={active} onChange={setPlate} aria-label="Plate" options={PLATES} className="fig-plates-seg" />
      </div>
      {staticPlate && (
        <StaticPlate plate={staticPlate} caption={staticPlateCaption(staticPlate.id, overview.data)}
          dpi={Number(dpi) || 300} onDpi={setDpi} training={training} onTraining={setTraining} />
      )}
      {active === "nexus" && <NexusPlates />}
      {active === "poster" && <PosterPlate />}
      <Section title="Rendered PNGs" sub="data/vis" collapsible defaultOpen={false}>
        <RenderedPngs />
      </Section>
    </Page>
  );
}
