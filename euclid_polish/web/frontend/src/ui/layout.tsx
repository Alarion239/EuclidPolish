/* Layout primitives: Card (+CardHead/CardBody), Section, and the compat
   Page/PageHead used by the pre-rework pages (workspace tabs use the shell's
   header instead; WP-F3). */
import { useId, useState, type CSSProperties, type ReactNode } from "react";
import { Icon } from "./icons";
import { cx } from "./slot";

type Box = { children?: ReactNode; className?: string; style?: CSSProperties; id?: string };

export function Card({ children, className, style, id, "aria-label": ariaLabel }: Box & { "aria-label"?: string }) {
  return <section className={cx("ui-card", className)} style={style} id={id} aria-label={ariaLabel}>{children}</section>;
}

export function CardHead(
  { title, sub, right, eyebrow, className }: {
    title: ReactNode; sub?: ReactNode; right?: ReactNode; eyebrow?: ReactNode; className?: string;
  },
) {
  return (
    <header className={cx("ui-card__head", className)}>
      <div className="ui-card__heading">
        {eyebrow != null && <div className="eyebrow">{eyebrow}</div>}
        <div className="ui-card__title">{title}</div>
        {sub && <div className="ui-card__sub">{sub}</div>}
      </div>
      <div className="ui-card__spacer" />
      {right}
    </header>
  );
}

export function CardBody({ children, className, style }: Box) {
  return <div className={cx("ui-card__body", className)} style={style}>{children}</div>;
}

/** A titled region inside a card or page. `collapsible` turns the title into
 *  a disclosure button (aria-expanded; aria-controls only while the body is
 *  mounted — a closed body is not rendered, so its children do no work). */
export function Section(
  { title, sub, right, children, collapsible = false, defaultOpen = true, open: openProp,
    onOpenChange, className, id }: {
    title: ReactNode; sub?: ReactNode; right?: ReactNode; children?: ReactNode;
    collapsible?: boolean; defaultOpen?: boolean; open?: boolean; onOpenChange?: (open: boolean) => void;
    className?: string; id?: string;
  },
) {
  const [inner, setInner] = useState(defaultOpen);
  const open = openProp ?? inner;
  const bodyId = useId();
  const setOpen = (v: boolean) => { setInner(v); onOpenChange?.(v); };
  return (
    <section className={cx("ui-section", className)} id={id} data-open={open}>
      <header className="ui-section__head">
        {collapsible ? (
          <button type="button" className="ui-section__toggle" aria-expanded={open}
            aria-controls={open ? bodyId : undefined}
            onClick={() => setOpen(!open)}>
            <Icon name={open ? "chevronDown" : "chevronRight"} size={14} />
            <span className="ui-section__title">{title}</span>
          </button>
        ) : <h3 className="ui-section__title">{title}</h3>}
        {sub != null && <span className="ui-section__sub">{sub}</span>}
        <span className="ui-section__spacer" />
        {right}
      </header>
      {(!collapsible || open) && <div className="ui-section__body" id={bodyId}>{children}</div>}
    </section>
  );
}

/* ─── compat page scaffold ────────────────────────────────────────────────── */

export function Page({ children, className, style }: Box) {
  return <div className={cx("page", className)} style={style}>{children}</div>;
}

export function PageHead(
  { eyebrow, title, sub, right }:
  { eyebrow?: ReactNode; title: ReactNode; sub?: ReactNode; right?: ReactNode },
) {
  return (
    <header className="page__head">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1 className="page__title">{title}</h1>
        {sub && <div className="page__sub">{sub}</div>}
      </div>
      {right && <><div className="page__spacer" />{right}</>}
    </header>
  );
}
