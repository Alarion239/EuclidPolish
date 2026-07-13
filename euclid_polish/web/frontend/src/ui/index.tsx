/* The reusable component kit. Small, composable, token-driven — the DRY
   backbone every page builds from. Keep primitives dumb; pages own data. */
import type { CSSProperties, ReactNode } from "react";
import "./ui.css";

type Div = { children?: ReactNode; className?: string; style?: CSSProperties };

export function Card({ children, className = "", style }: Div) {
  return <section className={`ui-card ${className}`} style={style}>{children}</section>;
}
export function CardHead(
  { title, sub, right }: { title: ReactNode; sub?: ReactNode; right?: ReactNode },
) {
  return (
    <header className="ui-card__head">
      <div>
        <div className="ui-card__title">{title}</div>
        {sub && <div className="ui-card__sub">{sub}</div>}
      </div>
      <div className="ui-card__spacer" />
      {right}
    </header>
  );
}
export function CardBody({ children, className = "", style }: Div) {
  return <div className={`ui-card__body ${className}`} style={style}>{children}</div>;
}

export function Button(
  { children, variant = "default", size, onClick, disabled, title, type = "button" }: {
    children: ReactNode; variant?: "default" | "primary" | "ghost"; size?: "sm";
    onClick?: () => void; disabled?: boolean; title?: string;
    type?: "button" | "submit";
  },
) {
  const cls = ["ui-btn", variant !== "default" && `ui-btn--${variant}`, size && `ui-btn--${size}`]
    .filter(Boolean).join(" ");
  return (
    <button className={cls} onClick={onClick} disabled={disabled} title={title} type={type}>
      {children}
    </button>
  );
}

export function Chip(
  { on, onClick, dot, title, children }: {
    on?: boolean; onClick?: () => void; dot?: string; title?: string; children: ReactNode;
  },
) {
  return (
    <button className="ui-chip" data-on={!!on} onClick={onClick} title={title}>
      {dot && <span className="ui-chip__dot" style={{ background: dot }} />}
      {children}
    </button>
  );
}

export function Segmented<T extends string>(
  { value, options, onChange }: {
    value: T; options: { value: T; label: string }[]; onChange: (v: T) => void;
  },
) {
  return (
    <div className="ui-seg">
      {options.map((o) => (
        <button key={o.value} data-on={o.value === value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return <label className="ui-field"><span>{label}</span>{children}</label>;
}

export function Select<T extends string>(
  { value, options, onChange }: {
    value: T; options: { value: T; label: string }[]; onChange: (v: T) => void;
  },
) {
  return (
    <select className="ui-select" value={value}
      onChange={(e) => onChange(e.target.value as T)}>
      {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  );
}

export function Badge(
  { children, tone }: { children: ReactNode; tone?: "good" | "warn" | "bad" },
) {
  return <span className={`ui-badge ${tone ? `ui-badge--${tone}` : ""}`}>{children}</span>;
}

export function Toolbar({ children }: Div) {
  return <div className="ui-toolbar">{children}</div>;
}
export function ToolbarLabel({ children }: Div) {
  return <span className="ui-toolbar__label">{children}</span>;
}
export function ToolbarSep() { return <span className="ui-toolbar__sep" />; }

export function Stat({ k, v }: { k: ReactNode; v: ReactNode }) {
  return <div className="ui-stat"><div className="ui-stat__v">{v}</div><div className="ui-stat__k">{k}</div></div>;
}

export function Spinner() { return <span className="ui-spin" />; }
export function Empty({ children }: Div) { return <div className="ui-empty">{children}</div>; }

/* ─── layout ─────────────────────────────────────────────────────────────── */

export function Page({ children }: Div) { return <div className="page">{children}</div>; }

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

/* ─── form controls ──────────────────────────────────────────────────────── */

type InputProps = {
  value: string | number;
  onChange: (v: string) => void;
  type?: "text" | "number" | "password";
  placeholder?: string;
  min?: number; max?: number; step?: number;
  disabled?: boolean; style?: CSSProperties; onEnter?: () => void;
};
export function Input(p: InputProps) {
  return (
    <input
      className="ui-input" type={p.type ?? "text"} value={p.value}
      placeholder={p.placeholder} min={p.min} max={p.max} step={p.step}
      disabled={p.disabled} style={p.style}
      onChange={(e) => p.onChange(e.target.value)}
      onKeyDown={(e) => { if (e.key === "Enter") p.onEnter?.(); }}
    />
  );
}

export function NumberField(
  { label, value, onChange, min, max, step, disabled, hint }: {
    label: string; value: number | string; onChange: (v: string) => void;
    min?: number; max?: number; step?: number; disabled?: boolean; hint?: string;
  },
) {
  return (
    <Field label={label}>
      <Input type="number" value={value} onChange={onChange}
        min={min} max={max} step={step} disabled={disabled} />
      {hint && <span className="ui-field__hint">{hint}</span>}
    </Field>
  );
}

export function Textarea(
  { value, onChange, rows = 6, placeholder, style }: {
    value: string; onChange: (v: string) => void; rows?: number;
    placeholder?: string; style?: CSSProperties;
  },
) {
  return (
    <textarea className="ui-textarea" value={value} rows={rows} placeholder={placeholder}
      style={style} onChange={(e) => onChange(e.target.value)} />
  );
}

export function Checkbox(
  { checked, onChange, children, disabled }: {
    checked: boolean; onChange: (v: boolean) => void; children: ReactNode; disabled?: boolean;
  },
) {
  return (
    <label className="ui-check">
      <input type="checkbox" checked={checked} disabled={disabled}
        onChange={(e) => onChange(e.target.checked)} />
      <span>{children}</span>
    </label>
  );
}

/* ─── data display ───────────────────────────────────────────────────────── */

export type Column<T> = {
  header: ReactNode;
  cell: (row: T, i: number) => ReactNode;
  align?: "left" | "right" | "center";
  width?: string | number;
};
export function Table<T>(
  { columns, rows, empty, rowKey, onRowClick, isRowClickable }: {
    columns: Column<T>[]; rows: T[]; empty?: ReactNode;
    rowKey?: (row: T, i: number) => string | number;
    onRowClick?: (row: T, i: number) => void;
    isRowClickable?: (row: T, i: number) => boolean;
  },
) {
  if (!rows.length) return <Empty>{empty ?? "nothing here yet"}</Empty>;
  return (
    <div className="ui-table-wrap">
      <table className="ui-table">
        <thead>
          <tr>{columns.map((c, i) => (
            <th key={i} style={{ textAlign: c.align, width: c.width }}>{c.header}</th>
          ))}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const clickable = !!onRowClick && (isRowClickable?.(r, i) ?? true);
            return (
              <tr key={rowKey ? rowKey(r, i) : i}
                className={clickable ? "ui-table__row--action" : undefined}
                onClick={clickable ? () => onRowClick(r, i) : undefined}>
                {columns.map((c, j) => (
                  <td key={j} style={{ textAlign: c.align }}>{c.cell(r, i)}</td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function DefList({ items }: { items: [ReactNode, ReactNode][] }) {
  return (
    <dl className="ui-deflist">
      {items.map(([k, v], i) => (
        <div key={i}><dt>{k}</dt><dd>{v}</dd></div>
      ))}
    </dl>
  );
}

export function Tabs<T extends string>(
  { value, tabs, onChange }: {
    value: T; tabs: { id: T; label: ReactNode }[]; onChange: (id: T) => void;
  },
) {
  return (
    <div className="ui-tabs">
      {tabs.map((t) => (
        <button key={t.id} className="ui-tab" data-on={t.id === value}
          onClick={() => onChange(t.id)}>{t.label}</button>
      ))}
    </div>
  );
}

export function ProgressBar({ value, max = 100, label }: { value: number | null; max?: number; label?: ReactNode }) {
  const pct = value == null ? null : Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className="ui-progress">
      <div className={`ui-progress__track${pct == null ? " is-indeterminate" : ""}`}>
        <div className="ui-progress__fill" style={pct == null ? undefined : { width: `${pct}%` }} />
      </div>
      {label && <span className="ui-progress__label mono">{label}</span>}
    </div>
  );
}

export function ConnBadge({ ok, labels }: { ok: boolean; labels?: [string, string] }) {
  const [on, off] = labels ?? ["connected", "offline"];
  return <Badge tone={ok ? "good" : "bad"}>{ok ? on : off}</Badge>;
}

export function LogTail({ text, style }: { text: string; style?: CSSProperties }) {
  return <pre className="ui-logtail" style={style}>{text || "(no output)"}</pre>;
}

/* ─── figures ────────────────────────────────────────────────────────────── */

export type GalleryItem = { src: string; href?: string; label?: string; onClick?: () => void };
/** Responsive thumbnail grid on paper-white cells — image galleries (vis PNGs,
 *  cutouts, reconstructions). Each cell links out (new tab) or fires onClick. */
export function Gallery({ items, thumb = 150, empty }: { items: GalleryItem[]; thumb?: number; empty?: ReactNode }) {
  if (!items.length) return <Empty>{empty ?? "nothing rendered yet"}</Empty>;
  return (
    <div className="ui-gallery" style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${thumb}px, 1fr))` }}>
      {items.map((it, i) => {
        const inner = <>
          <img src={it.src} loading="lazy" alt={it.label ?? ""} />
          {it.label && <span className="ui-gallery__cap mono">{it.label}</span>}
        </>;
        return it.href
          ? <a key={i} className="ui-gallery__cell" href={it.href} target="_blank" rel="noreferrer">{inner}</a>
          : <button key={i} className="ui-gallery__cell" onClick={it.onClick} type="button">{inner}</button>;
      })}
    </div>
  );
}

/** A server-rendered matplotlib PNG on a paper-white inset (readable in both
 *  themes), with an optional chip toolbar and a reload nonce so `?fresh=1`
 *  re-renders. `srcFor(active)` builds the URL for the selected toolbar key. */
export function PngFigure(
  { srcFor, toolbar, active, onActive, alt, minHeight = 220 }: {
    srcFor: (active?: string) => string;
    toolbar?: { key: string; label: string }[];
    active?: string; onActive?: (key: string) => void;
    alt?: string; minHeight?: number;
  },
) {
  const src = srcFor(active);
  return (
    <div className="ui-figure">
      {toolbar && toolbar.length > 0 && (
        <div className="ui-figure__bar">
          {toolbar.map((t) => (
            <button key={t.key} className="ui-chip" data-on={t.key === active}
              onClick={() => onActive?.(t.key)}>{t.label}</button>
          ))}
        </div>
      )}
      <div className="ui-figure__paper" style={{ minHeight }}>
        <img src={src} alt={alt ?? ""} loading="lazy" />
      </div>
    </div>
  );
}
