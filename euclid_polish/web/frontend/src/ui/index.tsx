/* The UI kit (v2, on Radix) — every shared control, overlay and display
   primitive. Import from "../ui" (never from the part modules), keep
   primitives dumb, and let pages own data. Documented in src/FOUNDATION.md §9.

   C8 names: Button, IconButton, Tooltip, Popover, Dialog, confirm, Menu,
   ContextMenu, Tabs, Segmented, Switch, Checkbox, Slider, RangeSlider,
   NumberField, Input, Select, Field, Card, CardHead, CardBody, Section, Badge,
   Chip, Stat, Kpi, DefList, Callout, EmptyState, Skeleton, ProgressBar,
   LogView, JsonTree, CopyButton, Kbd, DataTable, toast.
   Compat names kept for the pre-rework pages: Page, PageHead, Empty, Spinner,
   Table/Column, LogTail, Gallery, PngFigure, ConnBadge, Textarea,
   JobProgressView. */
import "./ui.css";
import "./pages-compat.css";

export { Button, IconButton, buttonClass } from "./Button";
export type { ButtonProps, ButtonSize, ButtonVariant, IconButtonProps } from "./Button";

export {
  ContextMenu, Dialog, DialogClose, Menu, Popover, PopoverClose, Tooltip, TooltipProvider,
} from "./overlays";
export type { Align, DialogSize, MenuItem, Side } from "./overlays";

export { ConfirmHost, confirm, resetConfirm } from "./confirm";
export type { ConfirmOptions } from "./confirm";

export { Toaster, toast } from "./toast";
export { UiProvider } from "./UiProvider";

export {
  Checkbox, Field, Input, MultiSelect, NumberField, RangeSlider, Segmented, Select, Slider, Switch,
  Tabs, Textarea,
} from "./controls";
export type { InputProps, SegmentedOption, SelectOption, TabItem } from "./controls";
export { useFieldAria } from "./fieldContext";

export { Card, CardBody, CardHead, Page, PageHead, Section } from "./layout";

export {
  Badge, Callout, Chip, ConnBadge, CopyButton, DefList, Empty, EmptyState, Kbd, Kpi, ProgressBar,
  Skeleton, Spinner, Stat, comboKeys,
} from "./display";
export type { Tone } from "./display";

export { LogTail, LogView, findMatches } from "./LogView";
export type { LogMatch } from "./LogView";
export { JsonTree, jsonPath } from "./JsonTree";

export { DataTable } from "./DataTable";
export type { DataTableProps } from "./DataTable";
export type { DataColumn, SortSpec, SortState } from "./tableModel";
export { Table } from "./Table";
export type { Column } from "./Table";

export { Gallery, PngFigure } from "./figures";
export type { GalleryItem } from "./figures";

export { JobProgress, JobProgressView } from "./JobProgress";

export { Icon, ICON_NAMES } from "./icons";
export type { IconName } from "./icons";
export { copyText, downloadBlob, downloadText, safeFileName } from "./download";
export { fromSliderPos, toSliderPos } from "./scale";
export type { SliderScale } from "./scale";
export { cx } from "./slot";
