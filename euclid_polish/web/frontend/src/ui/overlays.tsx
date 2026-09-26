/* Overlay primitives on Radix: Tooltip, Popover, Dialog, Menu (dropdown) and
   ContextMenu. Every overlay renders in a portal on the token surfaces, traps
   / restores focus the Radix way, and closes on Escape. */
import * as RContext from "@radix-ui/react-context-menu";
import * as RDialog from "@radix-ui/react-dialog";
import * as RMenu from "@radix-ui/react-dropdown-menu";
import * as RPopover from "@radix-ui/react-popover";
import * as RTooltip from "@radix-ui/react-tooltip";
import {
  createContext, useContext, type CSSProperties, type ReactElement, type ReactNode,
} from "react";
import { FieldContext } from "./fieldContext";
import { Icon } from "./icons";
import { cx } from "./slot";

/* ─── Tooltip ─────────────────────────────────────────────────────────────── */

const HasTooltipProvider = createContext(false);

/** One shared tooltip provider (skip-delay between neighbouring tooltips).
 *  `UiProvider` mounts it; a Tooltip rendered outside it brings its own. */
export function TooltipProvider({ children, delay = 350 }: { children: ReactNode; delay?: number }) {
  return (
    <RTooltip.Provider delayDuration={delay} skipDelayDuration={250}>
      <HasTooltipProvider.Provider value={true}>{children}</HasTooltipProvider.Provider>
    </RTooltip.Provider>
  );
}

export type Side = "top" | "right" | "bottom" | "left";
export type Align = "start" | "center" | "end";

/** Hover/focus tooltip around ONE focusable child element (it receives the
 *  trigger props, so it must forward refs — every kit control does). Empty
 *  `content` renders the child alone. */
export function Tooltip(
  { content, children, side = "top", align = "center", delay, open, onOpenChange }: {
    content: ReactNode; children: ReactElement; side?: Side; align?: Align; delay?: number;
    open?: boolean; onOpenChange?: (open: boolean) => void;
  },
) {
  const hasProvider = useContext(HasTooltipProvider);
  if (content == null || content === false || content === "") return children;
  const tip = (
    <RTooltip.Root delayDuration={delay} open={open} onOpenChange={onOpenChange}>
      <RTooltip.Trigger asChild>{children}</RTooltip.Trigger>
      <RTooltip.Portal>
        <RTooltip.Content className="ui-tooltip" side={side} align={align} sideOffset={6} collisionPadding={8}>
          {content}
          <RTooltip.Arrow className="ui-tooltip__arrow" width={10} height={5} />
        </RTooltip.Content>
      </RTooltip.Portal>
    </RTooltip.Root>
  );
  return hasProvider ? tip : <TooltipProvider>{tip}</TooltipProvider>;
}

/* ─── Popover ─────────────────────────────────────────────────────────────── */

/** Click-to-open floating panel anchored to `trigger` (one ref-forwarding
 *  element). Controlled with `open`/`onOpenChange` or uncontrolled. */
export function Popover(
  { trigger, children, side = "bottom", align = "start", open, onOpenChange, label, className,
    style, modal = false, width }: {
    trigger: ReactElement; children: ReactNode; side?: Side; align?: Align;
    open?: boolean; onOpenChange?: (open: boolean) => void;
    /** Accessible name of the panel (a dialog). */
    label?: string; className?: string; style?: CSSProperties; modal?: boolean;
    width?: number | string;
  },
) {
  return (
    <RPopover.Root open={open} onOpenChange={onOpenChange} modal={modal}>
      <RPopover.Trigger asChild>{trigger}</RPopover.Trigger>
      <RPopover.Portal>
        <RPopover.Content className={cx("ui-popover", className)} side={side} align={align}
          sideOffset={6} collisionPadding={8} aria-label={label}
          style={width != null ? { ...style, width } : style}>
          {/* Controls in the panel are not the surrounding Field's control. */}
          <FieldContext.Provider value={null}>{children}</FieldContext.Provider>
        </RPopover.Content>
      </RPopover.Portal>
    </RPopover.Root>
  );
}
export const PopoverClose = RPopover.Close;

/* ─── Dialog ──────────────────────────────────────────────────────────────── */

export type DialogSize = "sm" | "md" | "lg" | "xl";

/** Modal dialog with a title bar, scrollable body and optional footer. Pass
 *  `trigger` for an uncontrolled dialog, or `open`/`onOpenChange`. The
 *  `description` is the dialog's accessible description (aria-describedby);
 *  `role="alertdialog"` is for confirmations that interrupt (confirm()). */
export function Dialog(
  { open, onOpenChange, trigger, title, description, children, footer, size = "md",
    className, closeLabel = "Close", onOpenAutoFocus, role = "dialog" }: {
    open?: boolean; onOpenChange?: (open: boolean) => void; trigger?: ReactElement;
    title: ReactNode; description?: ReactNode; children?: ReactNode; footer?: ReactNode;
    size?: DialogSize; className?: string; closeLabel?: string;
    onOpenAutoFocus?: (e: Event) => void;
    role?: "dialog" | "alertdialog";
  },
) {
  const hasDesc = description != null && description !== false && description !== "";
  return (
    <RDialog.Root open={open} onOpenChange={onOpenChange}>
      {trigger && <RDialog.Trigger asChild>{trigger}</RDialog.Trigger>}
      <RDialog.Portal>
        <RDialog.Overlay className="ui-dialog__overlay" />
        <RDialog.Content className={cx("ui-dialog", `ui-dialog--${size}`, className)}
          onOpenAutoFocus={onOpenAutoFocus} role={role}
          {...(hasDesc ? {} : { "aria-describedby": undefined })}>
          {/* Controls in the dialog are not the surrounding Field's control. */}
          <FieldContext.Provider value={null}>
            <header className="ui-dialog__head">
              <RDialog.Title className="ui-dialog__title">{title}</RDialog.Title>
              <RDialog.Close asChild>
                <button type="button" className="ui-iconbtn ui-iconbtn--ghost ui-iconbtn--sm" aria-label={closeLabel}>
                  <Icon name="close" />
                </button>
              </RDialog.Close>
            </header>
            {hasDesc && (
              // a <div> so a rich description (paragraphs, lists) nests validly
              <RDialog.Description asChild><div className="ui-dialog__desc">{description}</div></RDialog.Description>
            )}
            {children != null && <div className="ui-dialog__body">{children}</div>}
            {footer && <footer className="ui-dialog__foot">{footer}</footer>}
          </FieldContext.Provider>
        </RDialog.Content>
      </RDialog.Portal>
    </RDialog.Root>
  );
}
export const DialogClose = RDialog.Close;

/* ─── Menu / ContextMenu ──────────────────────────────────────────────────── */

type ItemBase = { id?: string; label: ReactNode; disabled?: boolean; icon?: ReactNode };
export type MenuItem =
  | (ItemBase & {
      type?: "item"; onSelect?: () => void; shortcut?: string; tone?: "danger";
      /** Keep the menu open after selecting. */
      keepOpen?: boolean;
    })
  | (ItemBase & {
      type: "checkbox"; checked: boolean; onCheckedChange: (checked: boolean) => void;
      /** Default true: toggling several checkboxes in a row is the common case. */
      keepOpen?: boolean;
    })
  | { type: "separator"; id?: string }
  | { type: "label"; id?: string; label: ReactNode }
  | (ItemBase & { type: "sub"; items: MenuItem[] });

type Parts = typeof RMenu | typeof RContext;

function itemKey(it: MenuItem, i: number): string {
  return it.id ?? `${it.type ?? "item"}-${i}`;
}

function renderItems(P: Parts, items: MenuItem[]): ReactNode {
  return items.map((it, i) => {
    const key = itemKey(it, i);
    switch (it.type) {
      case "separator":
        return <P.Separator key={key} className="ui-menu__sep" />;
      case "label":
        return <P.Label key={key} className="ui-menu__label">{it.label}</P.Label>;
      case "checkbox":
        return (
          <P.CheckboxItem key={key} className="ui-menu__item" checked={it.checked}
            disabled={it.disabled} onCheckedChange={(v) => it.onCheckedChange(v === true)}
            onSelect={(e) => { if (it.keepOpen !== false) e.preventDefault(); }}>
            <span className="ui-menu__check">
              <P.ItemIndicator><Icon name="check" size={14} /></P.ItemIndicator>
            </span>
            <span className="ui-menu__text">{it.label}</span>
          </P.CheckboxItem>
        );
      case "sub":
        return (
          <P.Sub key={key}>
            <P.SubTrigger className="ui-menu__item" disabled={it.disabled}>
              {it.icon && <span className="ui-menu__icon">{it.icon}</span>}
              <span className="ui-menu__text">{it.label}</span>
              <Icon name="chevronRight" size={14} className="ui-menu__chev" />
            </P.SubTrigger>
            <P.Portal>
              <P.SubContent className="ui-menu" sideOffset={4} collisionPadding={8}>
                {renderItems(P, it.items)}
              </P.SubContent>
            </P.Portal>
          </P.Sub>
        );
      default:
        return (
          <P.Item key={key} className={cx("ui-menu__item", it.tone === "danger" && "ui-menu__item--danger")}
            disabled={it.disabled}
            onSelect={(e) => { if (it.keepOpen) e.preventDefault(); it.onSelect?.(); }}>
            {it.icon && <span className="ui-menu__icon">{it.icon}</span>}
            <span className="ui-menu__text">{it.label}</span>
            {it.shortcut && <span className="ui-menu__kbd">{it.shortcut}</span>}
          </P.Item>
        );
    }
  });
}

/** Dropdown menu from a declarative item list, opened by `trigger`. */
export function Menu(
  { trigger, items, align = "start", side = "bottom", label, open, onOpenChange }: {
    trigger: ReactElement; items: MenuItem[]; align?: Align; side?: Side;
    /** Accessible name of the menu. */
    label?: string; open?: boolean; onOpenChange?: (open: boolean) => void;
  },
) {
  return (
    <RMenu.Root open={open} onOpenChange={onOpenChange} modal={false}>
      <RMenu.Trigger asChild>{trigger}</RMenu.Trigger>
      <RMenu.Portal>
        <RMenu.Content className="ui-menu" align={align} side={side} sideOffset={6}
          collisionPadding={8} aria-label={label}>
          {renderItems(RMenu, items)}
        </RMenu.Content>
      </RMenu.Portal>
    </RMenu.Root>
  );
}

/** Right-click (or long-press / context-menu key) menu over `children`. */
export function ContextMenu(
  { items, children, label, disabled }: {
    items: MenuItem[]; children: ReactElement; label?: string; disabled?: boolean;
  },
) {
  return (
    <RContext.Root modal={false}>
      <RContext.Trigger asChild disabled={disabled}>{children}</RContext.Trigger>
      <RContext.Portal>
        <RContext.Content className="ui-menu" collisionPadding={8} aria-label={label}>
          {renderItems(RContext, items)}
        </RContext.Content>
      </RContext.Portal>
    </RContext.Root>
  );
}
