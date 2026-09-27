/* Toolbar: the one look for a tab's control bar (the per-workspace
   .ens-bar / .rl-bar / .ops-bar / .dt-bar all drew the same box): a calm
   rounded strip at the top of the page that scrolls away with it (it never
   sticks over the images), wrapping onto more lines when narrow.

     <Toolbar label="Curves controls">
       <ToolbarGroup label="Show"><Segmented … /></ToolbarGroup>
       <ToolbarSeparator />
       <ToolbarGroup label="Smooth" hideLabel><Select … /></ToolbarGroup>
       <ToolbarSpacer />
       <Button size="sm">Export</Button>
     </Toolbar>

   Labels are sentence case in the UI face, as authored. */
import { useId, type CSSProperties, type ReactNode } from "react";
import { cx } from "./slot";

/** A tab's control bar (role=toolbar, named by `label`). `plain` drops the
 *  box (no border, background or padding) for a bar inside a card. */
export function Toolbar(
  { label, children, plain = false, className, style }: {
    label: string; children?: ReactNode; plain?: boolean; className?: string; style?: CSSProperties;
  },
) {
  return (
    <div role="toolbar" aria-label={label} className={cx("ui-toolbar", plain && "ui-toolbar--plain", className)}
      style={style}>
      {children}
    </div>
  );
}

/** A labelled cluster of controls: the visible label (sentence case) names
 *  the group; `hideLabel` keeps it for screen readers only. */
export function ToolbarGroup(
  { label, children, hideLabel = false, className }: {
    label: ReactNode; children?: ReactNode; hideLabel?: boolean; className?: string;
  },
) {
  const id = useId();
  return (
    <div role="group" aria-labelledby={id} className={cx("ui-toolbar__group", className)}>
      <span id={id} className={hideLabel ? "sr-only" : "ui-toolbar__label"}>{label}</span>
      {children}
    </div>
  );
}

/** Free text in the bar (a count, a status): dim sentence-case text. */
export function ToolbarText({ children, className }: { children?: ReactNode; className?: string }) {
  return <span className={cx("ui-toolbar__text", className)}>{children}</span>;
}

/** Pushes what follows to the right end of the line. */
export function ToolbarSpacer() {
  return <span className="ui-toolbar__spacer" aria-hidden="true" />;
}

/** A thin vertical rule between groups. */
export function ToolbarSeparator() {
  return <span className="ui-toolbar__sep" role="separator" aria-orientation="vertical" />;
}
