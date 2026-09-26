/* figures/results — no legacy page: saved viewer results arrive with
   W-Figures in phase 3 (spec §8.5). */
import { PendingTab } from "../../../app/workspace";

export default function Results() {
  return (
    <PendingTab workspace="figures" tab="results"
      links={[["Figure grid", "/figures/grid"], ["Plates", "/figures/plates"]]}>
      Every crop saved from a viewer (S / Save crop to results), with its WCS, rename and delete.
    </PendingTab>
  );
}
