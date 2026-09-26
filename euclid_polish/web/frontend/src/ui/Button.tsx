/* Button and IconButton. Always `type="button"` unless told otherwise, a
   visible focus ring, `loading` (spinner + aria-busy, click-guarded), icons,
   and link rendering (`href`, or `asChild` around a router <Link>). */
import {
  forwardRef, type AnchorHTMLAttributes, type ButtonHTMLAttributes, type MouseEvent, type ReactElement,
  type ReactNode, type Ref,
} from "react";
import { Icon, type IconName } from "./icons";
import { Tooltip, type Side } from "./overlays";
import { Slot, cx } from "./slot";

export type ButtonVariant = "default" | "primary" | "ghost" | "danger" | "subtle";
export type ButtonSize = "sm" | "md" | "lg";

type Common = {
  variant?: ButtonVariant;
  size?: ButtonSize;
  /** Shows a spinner, sets aria-busy and ignores clicks (the width is kept). */
  loading?: boolean;
  /** Leading icon (an icon name or any node). */
  icon?: IconName | ReactNode;
  iconRight?: IconName | ReactNode;
  /** Render the single child (e.g. a router <Link>) with the button styles.
   *  The ref, `loading` (aria-busy + click guard, which also blocks a Link's
   *  navigation) and every other prop are forwarded onto the child. */
  asChild?: boolean;
  /** Render an <a> with the button styles (ref and props forwarded; the
   *  button-only attributes — type, form*, name, value — are dropped). */
  href?: string;
  download?: boolean | string;
  target?: string;
  rel?: string;
};

export type ButtonProps = Common & Omit<ButtonHTMLAttributes<HTMLButtonElement>, "onClick"> & {
  onClick?: (e: MouseEvent<HTMLButtonElement | HTMLAnchorElement>) => void;
};

const renderIcon = (icon: IconName | ReactNode | undefined) =>
  typeof icon === "string" ? <Icon name={icon as IconName} /> : icon;

export function buttonClass(
  variant: ButtonVariant = "default", size: ButtonSize = "md", extra?: string,
): string {
  return cx("ui-btn", variant !== "default" && `ui-btn--${variant}`, size !== "md" && `ui-btn--${size}`, extra);
}

/** Attributes that only mean something on a <button>. */
type ButtonOnly = "form" | "formAction" | "formEncType" | "formMethod" | "formNoValidate" | "formTarget"
  | "name" | "value";

/** The ref's element is the rendered one: a <button>, the <a> in `href` mode,
 *  or the child in `asChild` mode (typed HTMLButtonElement for compatibility). */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "default", size = "md", loading = false, icon, iconRight, asChild, href, download,
    target, rel, className, children, type = "button", disabled, onClick, ...rest },
  ref,
) {
  const cls = buttonClass(variant, size, className);
  // A click while loading (or disabled, for the non-<button> modes) does nothing.
  const guarded = (off: boolean) => (e: MouseEvent<HTMLButtonElement | HTMLAnchorElement>) => {
    if (off) { e.preventDefault(); return; }
    onClick?.(e);
  };
  const inner = (
    <>
      {loading ? <span className="ui-spin ui-btn__spin" aria-hidden="true" /> : renderIcon(icon)}
      {children != null && children !== false && <span className="ui-btn__label">{children}</span>}
      {renderIcon(iconRight)}
    </>
  );
  if (asChild) {
    const off = !!(disabled || loading);
    return (
      <Slot ref={ref as Ref<HTMLElement>} className={cls} {...(rest as object)}
        aria-busy={loading || undefined} data-loading={loading || undefined}
        {...(disabled ? { "aria-disabled": true } : {})}
        onClick={guarded(off) as never}>
        {children as ReactElement}
      </Slot>
    );
  }
  if (href != null) {
    const off = !!(disabled || loading);
    const anchorProps = omitButtonOnly(rest);
    return (
      <a ref={ref as unknown as Ref<HTMLAnchorElement>} {...anchorProps}
        className={cls} href={off ? undefined : href} download={download} target={target}
        rel={rel ?? (target === "_blank" ? "noreferrer" : undefined)}
        aria-disabled={off || anchorProps["aria-disabled"] || undefined} aria-busy={loading || undefined}
        data-loading={loading || undefined}
        onClick={guarded(off)}>
        {inner}
      </a>
    );
  }
  return (
    <button ref={ref} type={type} className={cls} disabled={disabled}
      aria-busy={loading || undefined} data-loading={loading || undefined}
      onClick={guarded(loading)}
      {...rest}>
      {inner}
    </button>
  );
});

function omitButtonOnly(
  rest: Omit<ButtonHTMLAttributes<HTMLButtonElement>, "onClick">,
): AnchorHTMLAttributes<HTMLAnchorElement> {
  const out: Record<string, unknown> = { ...rest };
  const drop: ButtonOnly[] = ["form", "formAction", "formEncType", "formMethod", "formNoValidate",
    "formTarget", "name", "value"];
  for (const k of drop) delete out[k];
  return out as AnchorHTMLAttributes<HTMLAnchorElement>;
}

export type IconButtonProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, "onClick" | "children"> & {
  /** Accessible name; also the tooltip unless `tooltip={false}`. */
  label: string;
  icon: IconName | ReactNode;
  variant?: "default" | "ghost" | "primary" | "danger";
  size?: "sm" | "md";
  onClick?: (e: MouseEvent<HTMLButtonElement>) => void;
  /** Toggle buttons: sets aria-pressed. */
  pressed?: boolean;
  tooltip?: ReactNode | false;
  tooltipSide?: Side;
  loading?: boolean;
};

/** Square icon-only button with a required label (aria-label + tooltip). */
export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  { label, icon, variant = "ghost", size = "md", onClick, pressed, tooltip, tooltipSide = "top",
    loading = false, className, type = "button", ...rest },
  ref,
) {
  const btn = (
    <button ref={ref} type={type} aria-label={label} aria-pressed={pressed}
      aria-busy={loading || undefined} data-on={pressed || undefined}
      className={cx("ui-iconbtn", `ui-iconbtn--${variant}`, size === "sm" && "ui-iconbtn--sm", className)}
      onClick={(e) => { if (!loading) onClick?.(e); }} {...rest}>
      {loading ? <span className="ui-spin" aria-hidden="true" /> : renderIcon(icon)}
    </button>
  );
  if (tooltip === false) return btn;
  return <Tooltip content={tooltip ?? label} side={tooltipSide}>{btn}</Tooltip>;
});
