/* The mini viewer of a `source:` card, for the catalogue layers whose objects
 * have pixels in a viewer collection (C6):
 *   psf-clusters/<cluster-NNN>          → `psfs` (the FASRC-cache ePSFs)
 *   lens-candidates/<id>, galaxies/<id> → `evaluation` (LR / SR of the
 *                                          catalogue-eval reconstruction, when
 *                                          one exists: real tile `eval/<id>`)
 * Real tiles (eval objects, archive / real fields, pairs, NEXUS) open the
 * tile card, which has its own `real` viewer. */
import { openInspector } from "../../../../app/inspector";
import { useResource } from "../../../../api/query";
import { Button, Skeleton } from "../../../../ui";
import { ImageViewer } from "../../../../viewer";

export type SourceViewerSpec = {
  collection: string;
  initialId: string;
  tiers?: string[];
  /** The real tile that must exist for the collection to hold this object. */
  evalRef?: string;
};

/** Which viewer collection (if any) holds a source feature's pixels. */
export function sourceViewerFor(layer: string, fid: string): SourceViewerSpec | null {
  if (!fid) return null;
  if (layer === "psf-clusters" && /^cluster-\d+$/.test(fid)) return { collection: "psfs", initialId: fid };
  if (layer === "lens-candidates" || layer === "galaxies") {
    return { collection: "evaluation", initialId: fid, tiers: ["LR", "SR"], evalRef: `eval/${fid}` };
  }
  return null;
}

function Viewer({ spec, layer }: { spec: SourceViewerSpec; layer: string }) {
  return (
    <div className="sky-card__viewer">
      <ImageViewer collection={spec.collection} initialId={spec.initialId} tiers={spec.tiers}
        id={`sky-${layer}-${spec.initialId}`} toolbar="compact" nav={false} />
    </div>
  );
}

function EvalViewer({ spec, layer }: { spec: SourceViewerSpec & { evalRef: string }; layer: string }) {
  const id = spec.evalRef.slice(spec.evalRef.indexOf("/") + 1);
  const card = useResource<{ ref: string }>(`/api/real/eval/${encodeURIComponent(id)}`, [], { ttl: 60_000 });
  if (card.loading) return <Skeleton lines={3} />;
  if (!card.data) {
    return (
      <p className="muted sky-card__note">
        {card.error?.status === 404 ? "No catalogue-eval reconstruction of this object yet." : card.error?.message ?? "No reconstruction."}
      </p>
    );
  }
  return (
    <>
      <Viewer spec={spec} layer={layer} />
      <div className="sky-card__actions">
        <Button size="sm" variant="ghost" onClick={() => openInspector({ kind: "tile", id: spec.evalRef })}>Evaluation tile card</Button>
      </div>
    </>
  );
}

export function SourceViewer({ layer, fid }: { layer: string; fid: string }) {
  const spec = sourceViewerFor(layer, fid);
  if (!spec) return null;
  if (spec.evalRef) return <EvalViewer spec={{ ...spec, evalRef: spec.evalRef }} layer={layer} />;
  return <Viewer spec={spec} layer={layer} />;
}
