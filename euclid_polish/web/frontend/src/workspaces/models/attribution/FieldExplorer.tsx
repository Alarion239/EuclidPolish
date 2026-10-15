/* Models › Knees › Field explorer: one field of the run as its SR (or LR)
   asinh PNG with the sampled pixels of the chosen band drawn on it as dots,
   each in the colour of its dominant knee (the largest gradient share). An
   SVG overlay in the field's own HR coordinates (shape [H, W]) is stretched
   over the image, so the dots land on their pixels at any size and for the
   half-size LR image too. Hovering a dot names its class, brightness and
   knee shares (also its `<title>`); one line gives the field's noise σ per
   input band (the shares' weights). Model, field and layer are URL state
   (`fm`, `field`, `layer`). */
import { useMemo, useState } from "react";
import { useUrlState } from "../../../hooks/useUrlState";
import { formatCount } from "../../../format";
import { Caption, EmptyState, Segmented, Select, Toolbar, ToolbarGroup } from "../../../ui";
import {
  dotTitle, electronText, fieldDots, fieldImageUrl, kneeLabel, modelShort, noiseText, parseLayer, sharePct, BAND_LABEL,
  type FieldDot, type FieldSet, type ImageLayer, type KneeBand,
} from "../knees";
import { ShareStack, type DrawnModel } from "./ClassBars";

export function FieldExplorer({ runId, models, band, set, color }: {
  runId: string; models: readonly DrawnModel[]; band: KneeBand; set: FieldSet; color: (q: number) => string;
}) {
  const [fmRaw, setFm] = useUrlState("fm", "");
  const [fieldRaw, setField] = useUrlState("field", "");
  const [layer, setLayer] = useUrlState<ImageLayer>("layer", "sr", { parse: parseLayer });
  const model = models.find((m) => m.spec === fmRaw) ?? models[0] ?? null;
  const fields = useMemo(() => {
    const all = model?.result.fields ?? [];
    const mine = all.filter((f) => f.kind === set);
    return mine.length ? mine : all;
  }, [model, set]);
  const field = fields.find((f) => f.ref === fieldRaw) ?? fields[0] ?? null;
  const dots = useMemo(() => fieldDots(field, band), [field, band]);
  const [hover, setHover] = useState<FieldDot | null>(null);
  const [broken, setBroken] = useState<string | null>(null);

  if (!model || !field) {
    return <EmptyState compact icon="image" title="No fields in this run">The run recorded no field for these models.</EmptyState>;
  }
  const knees = model.result.knees;
  const [H, W] = field.shape;
  const src = fieldImageUrl(runId, model.spec, field, band, layer);
  const r = Math.max(W, H) / 85;
  const shown = hover && dots.includes(hover) ? hover : null;
  const noise = noiseText(field.noise_sigma_e);
  return (
    <div className="kn-explorer">
      <Toolbar label="Field explorer" plain>
        {models.length > 1 && (
          <ToolbarGroup label="Model">
            <Select size="sm" aria-label="Explorer model" value={model.spec} onChange={(v) => { setFm(v); setHover(null); }}
              options={models.map((m) => ({ value: m.spec, label: modelShort(m.spec) }))} />
          </ToolbarGroup>
        )}
        <ToolbarGroup label="Field">
          <Select searchable size="sm" aria-label="Field" value={field.ref} onChange={(v) => { setField(v); setHover(null); }}
            options={fields.map((f) => ({ value: f.ref, label: f.ref, hint: `${f.kind} · ${formatCount(f.samples.band.length)} samples` }))} />
        </ToolbarGroup>
        <ToolbarGroup label="Image">
          <Segmented<ImageLayer> size="sm" aria-label="Image layer" value={layer} onChange={setLayer}
            options={[{ value: "sr", label: "SR" }, { value: "lr", label: "LR" }]} />
        </ToolbarGroup>
      </Toolbar>
      <div className="kn-explorer__main">
        <div className="kn-field" style={{ aspectRatio: W > 0 && H > 0 ? `${W} / ${H}` : "1 / 1" }}>
          {broken === src
            ? <span className="kn-field__none">The field image could not be loaded.</span>
            : <img className="kn-field__img" src={src} alt={`${field.ref}, ${layer.toUpperCase()} ${BAND_LABEL[band]}`} onError={() => setBroken(src)} />}
          {W > 0 && H > 0 && (
            <svg className="kn-field__dots" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="group"
              aria-label={`${dots.length} sampled pixels in ${BAND_LABEL[band]}`}>
              {dots.map((d) => (
                <circle key={d.i} cx={d.x + 0.5} cy={d.y + 0.5} r={r} className="kn-dot" data-on={shown === d || undefined}
                  data-knee={d.dominant >= 0 ? d.dominant : undefined}
                  style={d.dominant >= 0 ? { fill: color(knees[d.dominant] ?? NaN) } : undefined}
                  onMouseEnter={() => setHover(d)}>
                  <title>{dotTitle(d, knees)}</title>
                </circle>
              ))}
            </svg>
          )}
        </div>
        <div className="kn-explorer__side">
          {noise && <p className="mdl-note kn-noise" title="The field's noise σ per input band: the weights of its shares">{noise}</p>}
          <div className="kn-explorer__live" aria-live="polite">
            {shown ? (
              <>
                <p className="mdl-note"><strong>{shown.cls}</strong> · {electronText(shown.value)} · pixel ({shown.x}, {shown.y})</p>
                <ShareStack shares={shown.shares} knees={knees} color={color} label={`Pixel (${shown.x}, ${shown.y})`} />
                <p className="mdl-note">
                  Leads: {shown.dominant >= 0 ? `${kneeLabel(knees[shown.dominant])} (${sharePct(shown.shares[shown.dominant])})` : "—"}
                </p>
              </>
            ) : <p className="mdl-note">{formatCount(dots.length)} sampled pixels in {BAND_LABEL[band]}. Hover a dot for its class and knee shares.</p>}
          </div>
        </div>
      </div>
      <Caption>
        {field.kind === "synthetic" ? "A synthetic test field" : "A real tile"}, {layer === "sr" ? "the model's SR" : "the LR input"} in{" "}
        {BAND_LABEL[band]} (asinh-scaled). Each dot is a sampled output pixel in its dominant knee's colour (the largest gradient share,
        σ-weighted over all input bands).
      </Caption>
    </div>
  );
}
