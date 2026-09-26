/* A small inline-SVG icon set (stroke icons on a 16-unit grid, currentColor).
   No icon font or package: add a path here when the shell or a workspace needs
   a new glyph. Icons are decorative (aria-hidden) — label the control. */
import type { CSSProperties } from "react";

const PATHS = {
  check: "M3 8.5l3.2 3L13 4.5",
  close: "M4 4l8 8M12 4l-8 8",
  plus: "M8 3v10M3 8h10",
  minus: "M3 8h10",
  chevronDown: "M4 6l4 4 4-4",
  chevronUp: "M4 10l4-4 4 4",
  chevronRight: "M6 4l4 4-4 4",
  chevronLeft: "M10 4l-4 4 4 4",
  arrowUp: "M8 13V3M4 7l4-4 4 4",
  arrowDown: "M8 3v10M4 9l4 4 4-4",
  search: "M7 12A5 5 0 1 0 7 2a5 5 0 0 0 0 10zM10.8 10.8L14 14",
  copy: "M5.5 5.5h7v8h-7zM3.5 10.5v-8h7",
  download: "M8 2.5v8M4.5 7L8 10.5 11.5 7M3 13.5h10",
  reset: "M3 8a5 5 0 1 0 1.5-3.6M3 2.5V5h2.5",
  more: "M2.6 8a0.9 0.9 0 1 0 1.8 0a0.9 0.9 0 1 0-1.8 0M7.1 8a0.9 0.9 0 1 0 1.8 0a0.9 0.9 0 1 0-1.8 0M11.6 8a0.9 0.9 0 1 0 1.8 0a0.9 0.9 0 1 0-1.8 0",
  info: "M8 14A6 6 0 1 0 8 2a6 6 0 0 0 0 12zM8 7.2V11M8 5h.01",
  warn: "M8 2.5l6 11H2zM8 6.5v3.2M8 11.8h.01",
  error: "M8 14A6 6 0 1 0 8 2a6 6 0 0 0 0 12zM6 6l4 4M10 6l-4 4",
  success: "M8 14A6 6 0 1 0 8 2a6 6 0 0 0 0 12zM5.5 8.2l1.8 1.7 3.2-3.6",
  columns: "M2.5 3.5h11v9h-11zM6.2 3.5v9M9.8 3.5v9",
  filter: "M2.5 3.5h11L9.2 8.6v4l-2.4 1.2V8.6z",
  external: "M9.5 2.5h4v4M13.5 2.5L8 8M12 9.5v4H2.5V4h4",
  help: "M8 14A6 6 0 1 0 8 2a6 6 0 0 0 0 12zM6.3 6.2a1.8 1.8 0 1 1 2.4 1.7c-.5.2-.7.6-.7 1.1v.3M8 11.3h.01",
  zoomIn: "M7 12A5 5 0 1 0 7 2a5 5 0 0 0 0 10zM10.8 10.8L14 14M5 7h4M7 5v4",
  zoomOut: "M7 12A5 5 0 1 0 7 2a5 5 0 0 0 0 10zM10.8 10.8L14 14M5 7h4",
  image: "M2.5 3.5h11v9h-11zM2.5 10.5l3-3 3 3 2-2 3 3M10.3 6.3h.01",
  table: "M2.5 3.5h11v9h-11zM2.5 6.5h11M2.5 9.5h11M6.5 6.5v6",
  /* shell: workspaces */
  home: "M2.5 7.5L8 3l5.5 4.5M4 6.5V13h3V9.5h2V13h3V6.5",
  globe: "M8 14A6 6 0 1 0 8 2a6 6 0 0 0 0 12zM2 8h12M8 2c1.7 1.7 2.5 3.7 2.5 6S9.7 12.3 8 14c-1.7-1.7-2.5-3.7-2.5-6S6.3 3.7 8 2z",
  layers: "M8 2.5l5.5 3L8 8.5l-5.5-3zM2.5 8.5L8 11.5l5.5-3M2.5 11L8 14l5.5-3",
  wave: "M1.5 8.5h2.8l1.7-4.5 3 9 1.7-4.5h3.8",
  database: "M3 4c0-.8 2.2-1.5 5-1.5s5 .7 5 1.5-2.2 1.5-5 1.5S3 4.8 3 4zM3 4v8c0 .8 2.2 1.5 5 1.5s5-.7 5-1.5V4M3 8c0 .8 2.2 1.5 5 1.5s5-.7 5-1.5",
  fileSearch: "M8.5 2H3.5v12h4M8.5 2l3 3v2M8.5 2v3h3M10.8 13.2a2 2 0 1 0 0-4 2 2 0 0 0 0 4zM12.2 12.6L14 14.4",
  server: "M2.5 2.5h11v4.5h-11zM2.5 9h11v4.5h-11zM5 4.75h.01M5 11.25h.01",
  settings: "M8 10.2A2.2 2.2 0 1 0 8 5.8a2.2 2.2 0 0 0 0 4.4zM8 1.8v1.7M8 12.5v1.7M1.8 8h1.7M12.5 8h1.7M3.6 3.6l1.2 1.2M11.2 11.2l1.2 1.2M3.6 12.4l1.2-1.2M11.2 4.8l1.2-1.2",
  /* shell: chrome */
  menu: "M2.5 4h11M2.5 8h11M2.5 12h11",
  sidebar: "M2.5 3h11v10h-11zM6 3v10",
  panelRight: "M2.5 3h11v10h-11zM10 3v10",
  sun: "M8 10.8A2.8 2.8 0 1 0 8 5.2a2.8 2.8 0 0 0 0 5.6zM8 1.5V3M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1 1M11.6 11.6l1 1M3.4 12.6l1-1M11.6 4.4l1-1",
  moon: "M13 9.6A5.5 5.5 0 0 1 6.4 3 5.5 5.5 0 1 0 13 9.6z",
  monitor: "M2 3h12v8H2zM6 14h4M8 11v3",
  contrast: "M8 14A6 6 0 1 0 8 2a6 6 0 0 0 0 12zM8 2v12M8 5l4.6-1.4M8 8h6M8 11l4.6 1.4",
  activity: "M1.5 8.5h3l2-5 3 9 2-4h3",
  command: "M5.5 5.5V4a1.5 1.5 0 1 0-1.5 1.5h1.5zm0 0h5m-5 0v5m5-5V4A1.5 1.5 0 1 1 12 5.5h-1.5zm0 0v5m0 0H12a1.5 1.5 0 1 1-1.5 1.5v-1.5zm0 0h-5m0 0V12A1.5 1.5 0 1 1 4 10.5h1.5z",
  keyboard: "M1.5 4.5h13v7h-13zM4 7h.01M6.5 7h.01M9 7h.01M11.5 7h.01M5 9.5h6",
  pin: "M6 2.5h4M7 2.5v4L4.5 9h7L9 6.5v-4M8 9v4.5",
  link: "M6.5 9.5l3-3M7 4.5l1-1a2.5 2.5 0 0 1 3.5 3.5l-1 1M9 11.5l-1 1a2.5 2.5 0 0 1-3.5-3.5l1-1",
  stop: "M4.5 4.5h7v7h-7z",
} as const;

export type IconName = keyof typeof PATHS;
export const ICON_NAMES = Object.keys(PATHS) as IconName[];

export function Icon(
  { name, size = 16, className, style, strokeWidth = 1.6 }:
  { name: IconName; size?: number; className?: string; style?: CSSProperties; strokeWidth?: number },
) {
  return (
    <svg className={className ? `ui-icon ${className}` : "ui-icon"} style={style}
      width={size} height={size} viewBox="0 0 16 16" fill="none" stroke="currentColor"
      strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round"
      aria-hidden="true" focusable="false">
      <path d={PATHS[name]} />
    </svg>
  );
}
