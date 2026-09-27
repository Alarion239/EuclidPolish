/* Statistics presentation (spec 2026-09-27, "Statistics: readable,
   informative, nothing useless"). A page states its answer in one
   SummaryLine, lists the supporting numbers it needs in a FactsList (label,
   value and unit on one readable row), qualifies figures with a Caption, and
   collapses provenance-only numbers in Details. No tiles. */
import type { ReactNode } from "react";
import type { Tone } from "./display";
import { Tooltip } from "./overlays";
import { cx } from "./slot";
import "./facts.css";

const toneAttr = (tone?: Tone) => (tone && tone !== "neutral" ? tone : undefined);

/** A number inside a SummaryLine: bold, tabular, optionally toned. */
export function Num({ children, tone, unit }: { children: ReactNode; tone?: Tone; unit?: ReactNode }) {
  const value = <strong className="ui-num" data-tone={toneAttr(tone)}>{children}</strong>;
  // A unit never wraps away from its number: both sit in one nowrap span.
  return unit == null ? value : <span className="ui-num-unit">{value}{"\u00a0"}{unit}</span>;
}

/** The page's answer as one body-size sentence (at most one per page). */
export function SummaryLine({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={cx("ui-summary", className)}>{children}</p>;
}

export type Fact = { label: ReactNode; value: ReactNode; unit?: ReactNode; hint?: ReactNode; tone?: Tone };

/** Necessary supporting numbers, one row each: label left, value + unit right.
 *  Falsy entries are skipped, so a page can write `cond && {…}` inline; with
 *  no rows left the list renders nothing. */
export function FactsList({ title, facts, className }: {
  title?: ReactNode; facts: (Fact | null | false | undefined)[]; className?: string;
}) {
  const rows = facts.filter((f): f is Fact => !!f);
  if (!rows.length) return null;
  return (
    <section className={cx("ui-facts", className)}>
      {title != null && <h3 className="ui-facts__title">{title}</h3>}
      <dl className="ui-facts__list">
        {rows.map((f, i) => {
          const hinted = f.hint != null;
          const label = (
            <dt className="ui-facts__label" data-hint={hinted || undefined} tabIndex={hinted ? 0 : undefined}>
              {f.label}
            </dt>
          );
          return (
            <div key={i} role="group" className="ui-facts__row" data-tone={toneAttr(f.tone)}>
              {hinted ? <Tooltip content={f.hint}>{label}</Tooltip> : label}
              <dd className="ui-facts__value">
                <span className="ui-facts__v">{f.value}</span>
                {f.unit != null && <span className="ui-facts__u">{f.unit}</span>}
              </dd>
            </div>
          );
        })}
      </dl>
    </section>
  );
}

/** One muted line under the figure it qualifies (area, release, version). */
export function Caption({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={cx("ui-caption", className)}>{children}</p>;
}

/** Provenance-level numbers (fingerprints, bins, bytes), collapsed. */
export function Details({ summary, children, className }: { summary: ReactNode; children: ReactNode; className?: string }) {
  return (
    <details className={cx("ui-details", className)}>
      <summary>{summary}</summary>
      <div className="ui-details__body">{children}</div>
    </details>
  );
}
