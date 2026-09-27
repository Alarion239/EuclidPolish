/* A layer legend: swatches (categories) and/or a gradient bar (ramps). */
import type { Legend as LegendSpec } from "./colorScale";

export function SkyLegend({ legend, compact = false }: { legend: LegendSpec; compact?: boolean }) {
  const { items, gradient } = legend;
  return (
    <div className="sky-legend" data-compact={compact || undefined} aria-label={`Legend: ${legend.title}`}>
      {gradient && (
        <div className="sky-legend__ramp">
          <span className="sky-legend__bar" aria-hidden="true"
            style={{ background: `linear-gradient(to right, ${gradient.stops.join(", ")})` }} />
          <span className="sky-legend__ends mono">
            <span>{gradient.min}</span>
            {gradient.mid != null && <span>{gradient.mid}</span>}
            <span>{gradient.max}</span>
          </span>
          {!compact && <span className="sky-legend__title">{legend.title}</span>}
        </div>
      )}
      {items && items.length > 0 && (
        <ul className="sky-legend__items">
          {items.map((it) => (
            <li key={it.label}>
              <span className="sky-legend__dot" style={{ background: it.color }} aria-hidden="true" />
              {it.label}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
