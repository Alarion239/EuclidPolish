/* Display primitives: Badge, Chip, Stat, Kpi, DefList, Callout, EmptyState
   (+ compat Empty), Skeleton, Spinner, ProgressBar, Kbd, ConnBadge,
   CopyButton. */
import {
  forwardRef, useEffect, useId, useRef, useState, type CSSProperties, type MouseEvent, type ReactNode,
} from "react";
import { Link } from "react-router-dom";
import { copyText } from "./download";
import { Icon, type IconName } from "./icons";
import { Tooltip } from "./overlays";
import { cx } from "./slot";

export type Tone = "neutral" | "good" | "warn" | "bad" | "info" | "accent";

/* ─── Badge / Chip ────────────────────────────────────────────────────────── */

export function Badge(
  { children, tone, dot = false, title, className, size = "md" }: {
    children: ReactNode; tone?: Tone; dot?: boolean; title?: string; className?: string;
    size?: "sm" | "md";
  },
) {
  return (
    <span title={title}
      className={cx("ui-badge", tone && tone !== "neutral" && `ui-badge--${tone}`, size === "sm" && "ui-badge--sm", className)}>
      {dot && <span className="ui-badge__dot" aria-hidden="true" />}
      {children}
    </span>
  );
}

/** Toggle / filter pill (aria-pressed). `onRemove` adds an × (removable tag). */
export const Chip = forwardRef<HTMLButtonElement, {
  on?: boolean; onClick?: (e: MouseEvent<HTMLButtonElement>) => void; dot?: string; title?: string;
  children: ReactNode; disabled?: boolean; onRemove?: () => void; className?: string;
  "aria-label"?: string;
}>(function Chip({ on, onClick, dot, title, children, disabled, onRemove, className, "aria-label": ariaLabel }, ref) {
  const chip = (
    <button ref={ref} type="button" className={cx("ui-chip", className)} data-on={!!on}
      aria-pressed={onClick ? !!on : undefined} onClick={onClick} title={title} disabled={disabled}
      aria-label={ariaLabel}>
      {dot && <span className="ui-chip__dot" style={{ background: dot }} aria-hidden="true" />}
      {children}
    </button>
  );
  if (!onRemove) return chip;
  return (
    <span className="ui-chipgroup">
      {chip}
      <button type="button" className="ui-chip__remove" onClick={onRemove}
        aria-label={`Remove ${typeof children === "string" ? children : ""}`.trim()}>
        <Icon name="close" size={11} />
      </button>
    </span>
  );
});

/* ─── Stat / Kpi / DefList ────────────────────────────────────────────────── */

/** Compact value-over-key figure (the old `Stat k v`). */
export function Stat(
  { k, v, hint, tone, sub, className }: {
    k: ReactNode; v: ReactNode; hint?: ReactNode; tone?: Tone; sub?: ReactNode; className?: string;
  },
) {
  const key = <div className="ui-stat__k">{k}</div>;
  return (
    <div className={cx("ui-stat", tone && tone !== "neutral" && `ui-stat--${tone}`, className)}>
      <div className="ui-stat__v">{v}</div>
      {hint ? <Tooltip content={hint}><div className="ui-stat__k ui-stat__k--hint" tabIndex={0}>{k}</div></Tooltip> : key}
      {sub != null && <div className="ui-stat__sub">{sub}</div>}
    </div>
  );
}

/** Dashboard tile: label, big value (+unit), delta, footnote. It is an action
 *  when one of these is set:
 *  - `to`: an SPA route, rendered as a router <Link> (no page reload, so the
 *    in-memory jobs store, query cache and inspector survive);
 *  - `href`: an external or download URL, a plain <a>;
 *  - `onClick` alone: a <button> (with `to`/`href` it runs on click too).
 *  `loading` shows a skeleton value. */
export function Kpi(
  { label, value, unit, delta, deltaTone, tone, hint, footer, onClick, to, href, loading = false, icon,
    className }: {
    label: ReactNode; value: ReactNode; unit?: ReactNode; delta?: ReactNode; deltaTone?: Tone;
    tone?: Tone; hint?: ReactNode; footer?: ReactNode; onClick?: () => void;
    /** SPA route (router link). */
    to?: string;
    /** External or download URL (plain link; a full page load for an app path). */
    href?: string;
    loading?: boolean; icon?: IconName; className?: string;
  },
) {
  const hintId = useId();
  const action = to != null || href != null || onClick != null;
  const described = action && hint != null ? hintId : undefined;
  // Spans throughout: a link or button tile may only hold phrasing content.
  const body = (
    <>
      <span className="ui-kpi__label">
        {icon && <Icon name={icon} size={14} />}
        <span>{label}</span>
        {hint != null && (action
          // Inside a link/button no focusable may nest: the icon is hover-only
          // and the hint is the tile's accessible description instead.
          ? (
            <Tooltip content={hint}>
              <span className="ui-kpi__hint" aria-hidden="true"><Icon name="help" size={12} /></span>
            </Tooltip>
          )
          : (
            <Tooltip content={hint}>
              <button type="button" className="ui-kpi__hint" aria-label="About this figure">
                <Icon name="help" size={12} />
              </button>
            </Tooltip>
          ))}
      </span>
      <span className="ui-kpi__value">
        {loading ? <Skeleton width="60%" height={26} /> : <>{value}{unit != null && <span className="ui-kpi__unit">{unit}</span>}</>}
      </span>
      {delta != null && !loading && (
        <span className={cx("ui-kpi__delta", deltaTone && `ui-kpi__delta--${deltaTone}`)}>{delta}</span>
      )}
      {footer != null && <span className="ui-kpi__foot">{footer}</span>}
      {described && <span id={hintId} hidden>{hint}</span>}
    </>
  );
  const cls = cx("ui-kpi", tone && tone !== "neutral" && `ui-kpi--${tone}`, action && "ui-kpi--action", className);
  if (to != null) return <Link className={cls} to={to} onClick={onClick} aria-describedby={described}>{body}</Link>;
  if (href != null) return <a className={cls} href={href} onClick={onClick} aria-describedby={described}>{body}</a>;
  if (onClick) {
    return <button type="button" className={cls} onClick={onClick} aria-describedby={described}>{body}</button>;
  }
  return <div className={cls}>{body}</div>;
}

export function DefList(
  { items, className, dense = false }: {
    items: ([ReactNode, ReactNode] | null | false | undefined)[]; className?: string; dense?: boolean;
  },
) {
  return (
    <dl className={cx("ui-deflist", dense && "ui-deflist--dense", className)}>
      {items.filter((it): it is [ReactNode, ReactNode] => !!it).map(([k, v], i) => (
        <div key={i}><dt>{k}</dt><dd>{v}</dd></div>
      ))}
    </dl>
  );
}

/* ─── Callout / EmptyState / Skeleton / Spinner ───────────────────────────── */

const TONE_ICON: Record<string, IconName> = { info: "info", good: "success", warn: "warn", bad: "error", neutral: "info", accent: "info" };

/** Inline message block. `bad`/`warn` are announced (role=alert); others are
 *  a polite status. */
export function Callout(
  { tone = "info", title, children, action, icon = true, className, onDismiss }: {
    tone?: Tone; title?: ReactNode; children?: ReactNode; action?: ReactNode; icon?: boolean;
    className?: string; onDismiss?: () => void;
  },
) {
  const urgent = tone === "bad" || tone === "warn";
  return (
    <div className={cx("ui-callout", `ui-callout--${tone}`, className)} role={urgent ? "alert" : "status"}>
      {icon && <Icon name={TONE_ICON[tone] ?? "info"} className="ui-callout__icon" />}
      <div className="ui-callout__body">
        {title != null && <div className="ui-callout__title">{title}</div>}
        {children != null && <div className="ui-callout__text">{children}</div>}
      </div>
      {action != null && <div className="ui-callout__action">{action}</div>}
      {onDismiss && (
        <button type="button" className="ui-iconbtn ui-iconbtn--ghost ui-iconbtn--sm" aria-label="Dismiss" onClick={onDismiss}>
          <Icon name="close" size={14} />
        </button>
      )}
    </div>
  );
}

/** A designed empty / not-yet-available state: icon, title, explanation and
 *  an optional action (e.g. "Run evaluation"). */
export function EmptyState(
  { title, children, action, icon, compact = false, className }: {
    title: ReactNode; children?: ReactNode; action?: ReactNode; icon?: IconName;
    compact?: boolean; className?: string;
  },
) {
  return (
    <div className={cx("ui-emptystate", compact && "ui-emptystate--compact", className)}>
      {icon && <Icon name={icon} size={compact ? 18 : 26} className="ui-emptystate__icon" />}
      <div className="ui-emptystate__title">{title}</div>
      {children != null && <div className="ui-emptystate__text">{children}</div>}
      {action != null && <div className="ui-emptystate__action">{action}</div>}
    </div>
  );
}

/** Compat: the old one-line muted empty/loading block. */
export function Empty({ children, className, style }: { children?: ReactNode; className?: string; style?: CSSProperties }) {
  return <div className={cx("ui-empty", className)} style={style}>{children}</div>;
}

/** Shimmering placeholder. `lines` stacks text-like bars. Hidden from AT;
 *  wrap a loading region in `aria-busy` or pair with a status label. */
export function Skeleton(
  { width, height = 14, lines, radius, className, style }: {
    width?: number | string; height?: number | string; lines?: number; radius?: number | string;
    className?: string; style?: CSSProperties;
  },
) {
  if (lines && lines > 1) {
    return (
      <span className={cx("ui-skeleton-lines", className)} style={style} aria-hidden="true">
        {Array.from({ length: lines }, (_, i) => (
          <span key={i} className="ui-skeleton"
            style={{ height, width: i === lines - 1 ? "62%" : "100%", borderRadius: radius }} />
        ))}
      </span>
    );
  }
  return <span className={cx("ui-skeleton", className)} aria-hidden="true"
    style={{ width: width ?? "100%", height, borderRadius: radius, ...style }} />;
}

/** Activity spinner. With `label` it is a polite status for screen readers. */
export function Spinner({ label, size = "md" }: { label?: string; size?: "sm" | "md" | "lg" }) {
  return (
    <span className={cx("ui-spin", size !== "md" && `ui-spin--${size}`)}
      role={label ? "status" : undefined} aria-label={label} />
  );
}

/* ─── ProgressBar ─────────────────────────────────────────────────────────── */

/** Determinate (`value`/`max`) or indeterminate (`value = null`) bar. */
export function ProgressBar(
  { value, max = 100, label, tone, "aria-label": ariaLabel, className }: {
    value: number | null; max?: number; label?: ReactNode; tone?: Tone; "aria-label"?: string;
    className?: string;
  },
) {
  const pct = value == null || !(max > 0) ? null : Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className={cx("ui-progress", tone && `ui-progress--${tone}`, className)}>
      <div className={cx("ui-progress__track", pct == null && "is-indeterminate")} role="progressbar"
        aria-label={ariaLabel ?? (typeof label === "string" ? label : "Progress")}
        aria-valuemin={pct == null ? undefined : 0} aria-valuemax={pct == null ? undefined : 100}
        aria-valuenow={pct == null ? undefined : Math.round(pct)}>
        <div className="ui-progress__fill" style={pct == null ? undefined : { width: `${pct}%` }} />
      </div>
      {label && <span className="ui-progress__label mono">{label}</span>}
    </div>
  );
}

/* ─── Kbd / ConnBadge / CopyButton ────────────────────────────────────────── */

const IS_MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || "");

const KEY_GLYPH: Record<string, [string, string]> = {
  mod: ["⌘", "Ctrl"], meta: ["⌘", "Win"], cmd: ["⌘", "⌘"], ctrl: ["⌃", "Ctrl"],
  alt: ["⌥", "Alt"], option: ["⌥", "Alt"], shift: ["⇧", "Shift"], enter: ["↵", "Enter"],
  escape: ["Esc", "Esc"], esc: ["Esc", "Esc"], up: ["↑", "↑"], down: ["↓", "↓"],
  left: ["←", "←"], right: ["→", "→"], arrowup: ["↑", "↑"], arrowdown: ["↓", "↓"],
  arrowleft: ["←", "←"], arrowright: ["→", "→"], space: ["Space", "Space"], tab: ["Tab", "Tab"],
  backspace: ["⌫", "Backspace"],
};

/** Split a combo ("mod+k", "shift+?") into display keys for this platform. */
export function comboKeys(combo: string, mac = IS_MAC): string[] {
  return combo.split(/\+(?!$)/).map((k) => {
    const g = KEY_GLYPH[k.toLowerCase()];
    if (g) return mac ? g[0] : g[1];
    return k.length === 1 ? k.toUpperCase() : k;
  });
}

/** Keyboard key(s): `<Kbd>Esc</Kbd>` or `<Kbd keys="mod+k" />`. */
export function Kbd({ children, keys }: { children?: ReactNode; keys?: string }) {
  if (keys != null) {
    return (
      <span className="ui-kbdgroup" aria-label={keys}>
        {comboKeys(keys).map((k, i) => <kbd key={i} className="ui-kbd">{k}</kbd>)}
      </span>
    );
  }
  return <kbd className="ui-kbd">{children}</kbd>;
}

export function ConnBadge({ ok, labels }: { ok: boolean; labels?: [string, string] }) {
  const [on, off] = labels ?? ["connected", "offline"];
  return <Badge tone={ok ? "good" : "bad"} dot>{ok ? on : off}</Badge>;
}

/** Copies `value` (or its thunk's result); flips to a check for 1.5 s. */
export function CopyButton(
  { value, label = "Copy", size = "sm", showLabel = false, className }: {
    value: string | (() => string); label?: string; size?: "sm" | "md"; showLabel?: boolean;
    className?: string;
  },
) {
  const [state, setState] = useState<"idle" | "done" | "fail">("idle");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);
  const onClick = async () => {
    const ok = await copyText(typeof value === "function" ? value() : value);
    setState(ok ? "done" : "fail");
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setState("idle"), 1500);
  };
  const text = state === "done" ? "Copied" : state === "fail" ? "Copy failed" : label;
  return (
    <Tooltip content={showLabel ? null : text}>
      <button type="button" onClick={onClick} aria-label={showLabel ? undefined : label} data-state={state}
        className={cx(showLabel ? "ui-btn ui-btn--ghost" : "ui-iconbtn ui-iconbtn--ghost",
          size === "sm" && (showLabel ? "ui-btn--sm" : "ui-iconbtn--sm"), "ui-copy", className)}>
        <Icon name={state === "done" ? "check" : "copy"} size={14} />
        {showLabel && <span className="ui-btn__label">{text}</span>}
        <span className="sr-only" aria-live="polite">{state === "idle" ? "" : text}</span>
      </button>
    </Tooltip>
  );
}
