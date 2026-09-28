/* The Layers panel: the background survey (a Select), overlay HiPS
 * (opacity), pixel overlays of real tiles (opacity, blink), and every data
 * layer by group — Real tiles, Targets, Scene inputs, Coverage — with its
 * visibility, opacity, colour, muted count, legend and zoom-to. A layer's
 * data is filled in the tab that owns it (a shown layer, its options and an
 * empty layer's note link there) or, for positional data, from the sky
 * itself; the panel starts no job. */
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { formatCount } from "../../../format";
import { BASE_SURVEYS, OVERLAY_SURVEYS } from "../../../sky/surveys";
import {
  Button, Callout, Checkbox, Field, Icon, IconButton, Popover, Section, Select, Skeleton, Slider, Spinner,
  Switch, Tooltip,
} from "../../../ui";
import { colorOptions, legendFor } from "./colorScale";
import type { LayerData } from "./layerData";
import { groupLayers, type LayerInfo } from "./layerModel";
import { SkyLegend } from "./Legend";
import type { RenderSpec } from "./render";
import { patchPixelOverlay, resolveOverlays, withoutOverlay } from "./pixelOverlays";
import type { AtlasUrl } from "./useAtlasUrl";
import { toggleLayer, updateLayer } from "./urlState";

const pct = (v: number) => `${Math.round(v * 100)}%`;

/** Opacity slider that writes on release (the URL and the sky redraw once). */
function OpacitySlider({ value, onCommit, label }: { value: number; onCommit: (v: number) => void; label: string }) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  return (
    <Slider value={draft} min={0} max={1} step={0.05} onChange={setDraft} onCommit={onCommit}
      showValue format={pct} aria-label={label} className="sky-opacity" />
  );
}

/** The background surveys for the Select: Euclid first, then the all-sky
 *  surveys (named as such), then none. */
export const BACKGROUND_OPTIONS = BASE_SURVEYS.map((b) => ({
  value: b.id,
  label: b.group === "All-sky" ? `${b.label} (all-sky)` : b.label,
}));

function BackgroundSection({ url }: { url: AtlasUrl }) {
  const current = BASE_SURVEYS.find((b) => b.id === url.base) ?? BASE_SURVEYS[0];
  return (
    <Section title="Background" collapsible defaultOpen>
      <Field label="Survey">
        <Select value={current.id} onChange={url.setBase} options={BACKGROUND_OPTIONS} />
      </Field>
      {current.credit && <p className="sky-credit">{current.credit}</p>}
    </Section>
  );
}

function OverlayHipsSection({ url }: { url: AtlasUrl }) {
  const on = new Map(url.ov.map((o) => [o.id, o]));
  return (
    <Section title="JWST imagery" sub={on.size ? `${on.size} on` : "HiPS overlays"} collapsible defaultOpen={on.size > 0}>
      <ul className="sky-list">
        {OVERLAY_SURVEYS.map((o) => {
          const s = on.get(o.id);
          return (
            <li key={o.id} className="sky-row" data-on={!!s}>
              <div className="sky-row__main">
                <Checkbox checked={!!s} onChange={(v) => url.setOv(v ? [...url.ov, { id: o.id }] : url.ov.filter((x) => x.id !== o.id))}>
                  {o.label}
                </Checkbox>
                <span className="sky-row__meta">{o.group.replace("JWST · ", "")}</span>
              </div>
              {s && (
                <OpacitySlider value={s.opacity ?? o.defaultOpacity} label={`${o.label} opacity`}
                  onCommit={(v) => url.setOv(updateLayer(url.ov, o.id, { opacity: v }))} />
              )}
            </li>
          );
        })}
      </ul>
    </Section>
  );
}

function PixelOverlaysSection({ url }: { url: AtlasUrl }) {
  const overlays = useMemo(() => resolveOverlays(url.img), [url.img]);
  if (!overlays.length) return null;
  const clear = () => { url.setImg([]); url.setBlink(false); };
  return (
    <Section title="Pixel overlays" sub={`${overlays.length} FITS`} collapsible defaultOpen
      right={<Button size="sm" variant="ghost" onClick={clear}>Clear</Button>}>
      <ul className="sky-list">
        {overlays.map((o) => (
          <li key={o.key} className="sky-row" data-on={o.visible}>
            <div className="sky-row__main">
              <Checkbox checked={o.visible} onChange={(v) => url.setImg((cur) => patchPixelOverlay(cur, o.key, { hidden: !v }))}>{o.label}</Checkbox>
              <IconButton icon="close" size="sm" label={`Remove ${o.label}`} onClick={() => url.setImg((cur) => withoutOverlay(cur, o.key))} />
            </div>
            {o.visible && (
              <OpacitySlider value={o.opacity} label={`${o.label} opacity`}
                onCommit={(v) => url.setImg((cur) => patchPixelOverlay(cur, o.key, { opacity: v }))} />
            )}
          </li>
        ))}
      </ul>
      <Switch size="sm" checked={url.blink} disabled={overlays.filter((o) => o.visible).length < 2}
        onChange={(v) => url.setBlink(v)}>Blink between visible overlays</Switch>
    </Section>
  );
}

/** Where a layer's data comes from: its owning tab. */
function HomeLink({ info, children, label }: { info: LayerInfo; children?: ReactNode; label?: string }) {
  if (!info.home) return null;
  return <Link className="sky-layer__home" to={info.home.path} aria-label={label}>{children ?? info.home.label}</Link>;
}

const sentence = (text: string) => (text ? text[0].toUpperCase() + text.slice(1) : text);

/** Layers filled from the sky itself (a position, the JWST menu), not from
 *  their owning tab: how to fill each. */
const FILLED_ON_THE_SKY: Record<string, string> = {
  "/api/sky/jwst/pair": "download one from the JWST menu, or right-click the sky",
  "/api/real/tiles": "right-click the sky to cache one",
  "/api/sky/jwst/discover": "JWST menu › Discover",
  "/api/jwst-euclid/nexus/download-field": "JWST menu › Cache the NEXUS mosaic",
  "/api/experiments": "compare models on tiles in Sky › Compare",
};

function hint(info: LayerInfo, extra?: string) {
  const text = [info.description, extra].filter(Boolean).join(" ");
  return text ? text : null;
}

function LayerRow({ info, spec, data, url, onZoomTo }: {
  info: LayerInfo; spec?: RenderSpec; data?: LayerData; url: AtlasUrl; onZoomTo: (id: string) => void;
}) {
  const setting = url.layers.find((l) => l.id === info.id);
  const on = !!setting;
  const count = data?.features.length || info.count;
  const legend = spec ? legendFor(spec.scale) : null;
  const swatch = legend?.items?.[0]?.color ?? legend?.gradient?.stops[2];
  const choices = colorOptions(info, data?.features ?? []);
  const about = hint(info);
  return (
    <li className="sky-row sky-layer" data-on={on} data-ready={info.ready}>
      <div className="sky-row__main">
        <Checkbox checked={on} onChange={() => url.setLayers(toggleLayer(url.layers, info.id))}>
          <span className="sky-layer__label" title={info.label}>
            {swatch && <span className="sky-layer__swatch" style={{ background: swatch }} aria-hidden="true" />}
            <span className="sky-layer__name">{info.label}</span>
          </span>
        </Checkbox>
        <span className="sky-row__tail">
          {data?.fetching && <Spinner size="sm" label={`Loading ${info.label}`} />}
          {data?.error && (
            <Tooltip content={data.error.message}>
              <span className="sky-layer__err" tabIndex={0} aria-label={`Failed: ${data.error.message}`}><Icon name="warn" size={14} /></span>
            </Tooltip>
          )}
          {info.kind !== "moc" && <span className="sky-row__count">{formatCount(count)}</span>}
          <IconButton icon="zoomIn" size="sm" label={`Zoom to ${info.label}`} onClick={() => onZoomTo(info.id)}
            disabled={!info.bbox && !data?.features.length && info.kind !== "moc"} />
          <Popover label={`${info.label} options`} width={280}
            trigger={<IconButton icon="more" size="sm" label={`${info.label} options`} />}>
            <div className="sky-layer__pop">
              {about && <p className="muted">{about}</p>}
              {info.kind !== "moc" && (
                <Field label="Colour">
                  <Select value={setting?.color ?? ""} options={choices}
                    onChange={(v) => {
                      const next = on ? url.layers : [...url.layers, { id: info.id }];
                      url.setLayers(updateLayer(next, info.id, { color: v || undefined }));
                    }} />
                </Field>
              )}
              {!info.ready && info.reason && <p className="muted">{sentence(info.reason)}</p>}
              {info.home && <p className="sky-layer__from">Its data: <HomeLink info={info} /></p>}
            </div>
          </Popover>
        </span>
      </div>
      {on && (
        <div className="sky-layer__detail">
          <OpacitySlider value={spec?.opacity ?? 0.7} label={`${info.label} ${spec?.fill != null ? "fill" : "opacity"}`}
            onCommit={(v) => url.setLayers(updateLayer(url.layers, info.id, { opacity: v }))} />
          {/* zoomed in, a coverage layer is only its outline (neutral imagery) until a fill is set */}
          {spec?.fill === false && <p className="sky-layer__note">Outline only while zoomed in; move the slider to fill it.</p>}
          {legend && spec?.scale.type !== "fixed" && info.kind !== "moc" && <SkyLegend legend={legend} compact />}
          {info.home && info.ready && (
            <p className="sky-layer__note">Its data: <HomeLink info={info} label={`${info.label}: open ${info.home.label}`} /></p>
          )}
        </div>
      )}
      {!info.ready && on && info.reason && (
        <div className="sky-layer__reason muted">
          {sentence(info.reason)}
          {info.fill_action && FILLED_ON_THE_SKY[info.fill_action.url] ? `: ${FILLED_ON_THE_SKY[info.fill_action.url]}.`
            : info.home ? <>; fill it in <HomeLink info={info} />.</> : "."}
        </div>
      )}
    </li>
  );
}

export function LayersPanel({ url, layers, data, specs, loading, error, onRetry, onZoomTo, footer }: {
  url: AtlasUrl;
  layers: readonly LayerInfo[];
  data: Readonly<Record<string, LayerData>>;
  specs: readonly RenderSpec[];
  loading: boolean;
  error: string | null;
  onRetry: () => void;
  onZoomTo: (id: string) => void;
  footer?: ReactNode;
}) {
  const groups = groupLayers(layers);
  const specById = new Map(specs.map((s) => [s.info.id, s]));
  return (
    <div className="sky-panel__scroll">
      <BackgroundSection url={url} />
      <OverlayHipsSection url={url} />
      <PixelOverlaysSection url={url} />
      {error && (
        <Callout tone="bad" title="Layer catalogue unavailable" action={<Button size="sm" onClick={onRetry}>Retry</Button>}>
          {error}
        </Callout>
      )}
      {loading && !layers.some((l) => !l.client) && <Skeleton lines={6} />}
      {groups.map((g) => {
        const onCount = g.layers.filter((l) => url.layers.some((s) => s.id === l.id)).length;
        return (
          <Section key={g.group} title={g.label} sub={`${onCount}/${g.layers.length}`} collapsible defaultOpen>
            <ul className="sky-list">
              {g.layers.map((l) => (
                <LayerRow key={l.id} info={l} spec={specById.get(l.id)} data={data[l.id]} url={url} onZoomTo={onZoomTo} />
              ))}
            </ul>
          </Section>
        );
      })}
      {footer}
    </div>
  );
}
