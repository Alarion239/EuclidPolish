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
