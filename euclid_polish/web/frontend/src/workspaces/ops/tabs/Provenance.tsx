/* ops/provenance — no legacy page: the lineage browser arrives with W-Ops in
   phase 3 (spec §8.7). */
import { PendingTab } from "../../../app/workspace";

export default function Provenance() {
  return (
    <PendingTab workspace="ops" tab="provenance" links={[["Tracking", "/ops/tracking"]]}>
      Search products, walk ancestors and descendants, and check staleness over data/_prov and the
      per-object sidecars.
    </PendingTab>
  );
}
