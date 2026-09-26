/* realism/overview — no legacy page: the prior-readiness overview arrives
   with W-Realism in phase 3 (spec §8.3). */
import { PendingTab } from "../../../app/workspace";

export default function Overview() {
  return (
    <PendingTab workspace="realism" tab="overview"
      links={[
        ["Noise", "/realism/noise"], ["Galaxies", "/realism/galaxies"], ["Stars", "/realism/stars"],
        ["Pixels", "/realism/pixels"], ["Visual", "/realism/visual"],
      ]}>
      One place for the galaxy joint model, the star prior, the TNG radius manifest, the noise
      model version and the comparison caches.
    </PendingTab>
  );
}
