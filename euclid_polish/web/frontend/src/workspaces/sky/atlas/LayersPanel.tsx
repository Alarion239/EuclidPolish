/* The Layers panel: background HiPS (radio), overlay HiPS (opacity), pixel
 * overlays of real tiles (opacity, blink), and every data layer by group
 * (visibility, opacity, colour, count, legend, zoom-to, fill action). */
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useFasrcStatus } from "../../../app/status";
import { formatCount } from "../../../format";
import { BASE_SURVEYS, OVERLAY_SURVEYS } from "../../../sky/surveys";
import {
  Badge, Button, Callout, Checkbox, Field, Icon, IconButton, Popover, Section, Select, Skeleton, Slider, Spinner,
  Switch, Tooltip,
} from "../../../ui";
import { runLayerFill } from "./actions";
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

function BackgroundSection({ url }: { url: AtlasUrl }) {
  const current = BASE_SURVEYS.find((b) => b.id === url.base) ?? BASE_SURVEYS[0];
  return (
    <Section title="Background" sub={current.label} collapsible defaultOpen>
      <div className="sky-radios" role="radiogroup" aria-label="Background survey">
        {BASE_SURVEYS.map((b) => (
          <label key={b.id} className="sky-radio" data-on={b.id === current.id} title={b.credit ?? b.label}>
            <input type="radio" name="sky-base" value={b.id} checked={b.id === current.id}
              onChange={() => url.setBase(b.id)} />
            <span>{b.label}</span>
            {b.format === "fits" && <span className="sky-radio__tag">FITS</span>}
          </label>
        ))}
      </div>
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

function FillButton({ info }: { info: LayerInfo }) {
  const fasrc = useFasrcStatus().data;
  const action = info.fill_action;
  if (!action || action.method !== "POST") return null;
  // Actions needing parameters are driven from the sky itself.
  if (["/api/experiments", "/api/real/tiles", "/api/sky/jwst/pair"].includes(action.url)) return null;
  const offline = !!action.requires_fasrc && !fasrc?.ssh_connected;
  return (
    <Tooltip content={offline ? "Needs the FASRC connection (Settings › Connections)" : action.label}>
      <span>
        <Button size="sm" variant="subtle" disabled={offline} onClick={() => { void runLayerFill(action); }}>
          {action.label}
        </Button>
      </span>
    </Tooltip>
  );
}

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
          <span className="sky-layer__label">
            {swatch && <span className="sky-layer__swatch" style={{ background: swatch }} aria-hidden="true" />}
            {info.label}
          </span>
        </Checkbox>
        <span className="sky-row__tail">
          {data?.fetching && <Spinner size="sm" label={`Loading ${info.label}`} />}
          {data?.error && (
            <Tooltip content={data.error.message}>
              <span className="sky-layer__err" tabIndex={0} aria-label={`Failed: ${data.error.message}`}><Icon name="warn" size={14} /></span>
            </Tooltip>
          )}
          {info.kind !== "moc" && <Badge size="sm" tone={info.ready ? "neutral" : "warn"}>{formatCount(count)}</Badge>}
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
              {!info.ready && info.reason && <Callout tone="warn">{info.reason}</Callout>}
              <FillButton info={info} />
            </div>
          </Popover>
        </span>
      </div>
      {on && (
        <div className="sky-layer__detail">
          <OpacitySlider value={spec?.opacity ?? 0.7} label={`${info.label} opacity`}
            onCommit={(v) => url.setLayers(updateLayer(url.layers, info.id, { opacity: v }))} />
          {legend && spec?.scale.type !== "fixed" && info.kind !== "moc" && <SkyLegend legend={legend} compact />}
        </div>
      )}
      {!info.ready && on && info.reason && <div className="sky-layer__reason muted">{info.reason}</div>}
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
