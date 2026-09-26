/* sky/experiments — no legacy page: model comparison on real tiles arrives
   with W-SkyResults in phase 3 (spec §7.3). */
import { PendingTab } from "../../../app/workspace";

export default function Experiments() {
  return (
    <PendingTab workspace="sky" tab="experiments"
      links={[["Real results", "/sky/results"], ["Catalog eval", "/sky/catalog-eval"]]}>
      Pick real tiles and models, run every (tile, model) SR once and compare hole %, enclosed-flux
      R and flux ratios per band.
    </PendingTab>
  );
}
