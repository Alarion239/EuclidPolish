/* Form controls: Field, Input, Textarea, NumberField, Select, MultiSelect,
   Checkbox, Switch, Segmented, Tabs, Slider, RangeSlider.

   Value-based change handlers (`onChange(value)`, not the DOM event) keep
   pages terse; every control forwards its ref, takes aria-* props and shows
   the shared focus ring. */
import * as RSlider from "@radix-ui/react-slider";
import * as RSwitch from "@radix-ui/react-switch";
import * as RTabs from "@radix-ui/react-tabs";
import * as RToggle from "@radix-ui/react-toggle-group";
import { Command } from "cmdk";
import {
  cloneElement, forwardRef, isValidElement, useEffect, useId, useMemo, useRef, useState,
  type CSSProperties, type InputHTMLAttributes, type MouseEvent, type ReactElement, type ReactNode,
  type TextareaHTMLAttributes,
} from "react";
import { Link } from "react-router-dom";
import { FieldContext, useField, useFieldAria, type FieldCtx } from "./fieldContext";
import { Icon } from "./icons";
import { Popover } from "./overlays";
import { effectiveScale, fromSliderPos, toSliderPos, LOG_STEPS, type SliderScale } from "./scale";
import { composeRefs, cx } from "./slot";

/* ─── Field ───────────────────────────────────────────────────────────────── */

type AriaInvalid = InputHTMLAttributes<HTMLElement>["aria-invalid"];

const RAW_CONTROLS = new Set(["input", "select", "textarea"]);
type RawControlProps = { "aria-describedby"?: string; "aria-invalid"?: AriaInvalid };

/** A labelled control. The `<label>` wraps the label text and the control, so
 *  any child (kit control or raw element) is associated; `hint` adds a "?"
 *  popover next to the label. `description` (inline help) and `error` show
 *  below, OUTSIDE the label: they describe the control (aria-describedby),
 *  `error` also marks it aria-invalid, and neither is part of its name. Kit
 *  inputs and selects pick this up from context; a single raw
 *  <input>/<select>/<textarea> child is wired directly. */
export function Field(
  { label, children, hint, description, error, inline = false, className, style }: {
    label: ReactNode; children: ReactNode; hint?: ReactNode;
    /** Inline help text under the control (the control's description). */
    description?: ReactNode;
    error?: ReactNode;
    /** Label beside the control instead of above it. */
    inline?: boolean; className?: string; style?: CSSProperties;
  },
) {
  const id = useId();
  const hasError = error != null && error !== false;
  const hasDesc = description != null && description !== false && description !== "";
  const labelId = `${id}-label`, descId = `${id}-desc`, errorId = `${id}-error`;
  const labelText = typeof label === "string" ? label : undefined;
  const describedBy = cx(hasDesc && descId, hasError && errorId) || undefined;
  const ctx = useMemo<FieldCtx>(
    () => ({ labelId, labelText, describedBy, invalid: hasError }),
    [labelId, labelText, describedBy, hasError],
  );
  let control = children;
  if (describedBy && isValidElement<RawControlProps>(children) && typeof children.type === "string"
    && RAW_CONTROLS.has(children.type)) {
    const own = children.props;
    control = cloneElement(children as ReactElement<RawControlProps>, {
      "aria-describedby": cx(own["aria-describedby"], describedBy),
      "aria-invalid": own["aria-invalid"] ?? (hasError || undefined),
    });
  }
  // The root is a <span> (valid wherever the old <label> root was). The
  // structure inside the label never depends on `error`, so an error
  // appearing while the user types does not remount (and blur) the control.
  return (
    <span className={cx("ui-field", inline && "ui-field--inline", className)} style={style}>
      <label className="ui-field__main">
        <span className="ui-field__label">
          <span id={labelId}>{label}</span>
          {hint != null && <FieldHint label={label}>{hint}</FieldHint>}
        </span>
        <FieldContext.Provider value={ctx}>{control}</FieldContext.Provider>
      </label>
      {hasDesc && <span id={descId} className="ui-field__hint">{description}</span>}
      {hasError && <span id={errorId} className="ui-field__error" role="alert">{error}</span>}
    </span>
  );
}

function FieldHint({ label, children }: { label: ReactNode; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const name = typeof label === "string" ? `About ${label}` : "About this field";
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    // A click on the hint must not also activate (focus / toggle) the control
    // the surrounding <label> labels. A NATIVE listener on the trigger cancels
    // the label activation before the event reaches the label (React's
    // delegated handlers run too late for that); we toggle the popover here,
    // and Radix skips its own toggle because the event is defaultPrevented.
    const onClick = (e: Event) => { e.preventDefault(); setOpen((o) => !o); };
    el.addEventListener("click", onClick);
    return () => el.removeEventListener("click", onClick);
  }, []);
  return (
    <Popover open={open} onOpenChange={setOpen} side="top" label={name} className="ui-field__hintpop"
      trigger={
        // A span (not a <button>): a button is labelable and would steal the
        // label's association from the real control.
        <span ref={ref} role="button" tabIndex={0} aria-label={name} className="ui-field__hintbtn"
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setOpen((o) => !o); }
          }}>
          <Icon name="help" size={13} />
        </span>
      }>
      <div className="ui-field__hinttext">{children}</div>
    </Popover>
  );
}

/* ─── Input / Textarea ────────────────────────────────────────────────────── */

type NativeInput = Omit<InputHTMLAttributes<HTMLInputElement>, "onChange" | "value" | "size" | "type">;
export type InputProps = NativeInput & {
  value: string | number;
  onChange: (value: string) => void;
  type?: "text" | "number" | "password" | "search" | "email" | "url" | "date" | "time" | "datetime-local";
  /** Called on Enter (IME composition excluded). */
  onEnter?: () => void;
  size?: "sm" | "md";
  /** Leading icon inside the field (e.g. "search"). */
  icon?: "search" | "filter";
  /** Shows a clear (×) button while non-empty. */
  clearable?: boolean;
};

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  { value, onChange, type = "text", onEnter, size = "md", icon, clearable, className, style, onKeyDown,
    "aria-describedby": describedBy, "aria-invalid": invalid, ...rest },
  ref,
) {
  const inner = useRef<HTMLInputElement>(null);
  const aria = useFieldAria(describedBy, invalid);
  const input = (
    <input ref={composeRefs(ref, inner)} type={type} value={value}
      className={cx("ui-input", size === "sm" && "ui-input--sm", !icon && !clearable && className)}
      style={!icon && !clearable ? style : undefined}
      onChange={(e) => onChange(e.target.value)}
      onKeyDown={(e) => {
        onKeyDown?.(e);
        if (!e.defaultPrevented && e.key === "Enter" && !e.nativeEvent.isComposing) onEnter?.();
      }}
      {...rest} {...aria} />
  );
  if (!icon && !clearable) return input;
  return (
    <span className={cx("ui-inputwrap", icon && "ui-inputwrap--icon", clearable && "ui-inputwrap--clear", className)}
      style={style}>
      {icon && <Icon name={icon} size={14} className="ui-inputwrap__icon" />}
      {input}
      {clearable && String(value) !== "" && (
        <button type="button" className="ui-inputwrap__clear" aria-label="Clear"
          onClick={() => { onChange(""); inner.current?.focus(); }}>
          <Icon name="close" size={12} />
        </button>
      )}
    </span>
  );
});

type NativeTextarea = Omit<TextareaHTMLAttributes<HTMLTextAreaElement>, "onChange" | "value">;
export const Textarea = forwardRef<HTMLTextAreaElement, NativeTextarea & {
  value: string; onChange: (value: string) => void;
}>(function Textarea(
  { value, onChange, rows = 6, className, "aria-describedby": describedBy, "aria-invalid": invalid, ...rest },
  ref,
) {
  const aria = useFieldAria(describedBy, invalid);
  return (
    <textarea ref={ref} className={cx("ui-textarea", className)} value={value} rows={rows}
      onChange={(e) => onChange(e.target.value)} {...rest} {...aria} />
  );
});

/* ─── NumberField ─────────────────────────────────────────────────────────── */

/** Labelled number input. `onChange` receives the raw string (so partial input
 *  like "1e" or "-" survives while typing); parse on submit. */
export function NumberField(
  { label, value, onChange, min, max, step, disabled, hint, unit, placeholder, onEnter, size,
    "aria-label": ariaLabel, style }: {
    label?: ReactNode; value: number | string; onChange: (value: string) => void;
    min?: number; max?: number; step?: number | "any"; disabled?: boolean;
    /** Inline hint text under the input. */
    hint?: ReactNode; unit?: string; placeholder?: string; onEnter?: () => void;
    size?: "sm" | "md"; "aria-label"?: string; style?: CSSProperties;
  },
) {
  const invalid = value !== "" && value != null && !Number.isFinite(Number(value));
  const input = (
    <span className={cx("ui-number", unit && "ui-number--unit")}>
      <Input type="number" value={value} onChange={onChange} min={min} max={max} step={step}
        disabled={disabled} placeholder={placeholder} onEnter={onEnter} size={size}
        aria-invalid={invalid || undefined} aria-label={label == null ? ariaLabel : undefined} style={style} />
      {unit && <span className="ui-number__unit">{unit}</span>}
    </span>
  );
  if (label == null) return input;
  // The hint is the input's description (under the field, not in its name).
  return <Field label={label} description={hint}>{input}</Field>;
}

/* ─── Select / MultiSelect ────────────────────────────────────────────────── */

export type SelectOption<T extends string = string> = {
  value: T; label: string; disabled?: boolean;
  /** Secondary text (searchable lists show it; also matched by the search). */
  hint?: string;
};

type SelectBase<T extends string> = {
  options: SelectOption<T>[];
  disabled?: boolean;
  placeholder?: string;
  size?: "sm" | "md";
  "aria-label"?: string;
  id?: string;
  className?: string;
  style?: CSSProperties;
};

/** Single choice. Native <select> by default; `searchable` opens a filterable
 *  list (for long option lists such as members or tiles). */
export function Select<T extends string>(
  { value, onChange, options, searchable = false, disabled, placeholder, size = "md", className, style,
    id, "aria-label": ariaLabel }: SelectBase<T> & {
    value: T; onChange: (value: T) => void; searchable?: boolean;
  },
) {
  const aria = useFieldAria();
  if (!searchable) {
    const known = options.some((o) => o.value === value);
    return (
      <select id={id} className={cx("ui-select", size === "sm" && "ui-select--sm", className)} style={style}
        value={value} disabled={disabled} aria-label={ariaLabel} {...aria}
        onChange={(e) => onChange(e.target.value as T)}>
        {!known && <option value={value} disabled>{placeholder ?? (value || "—")}</option>}
        {options.map((o) => (
          <option key={o.value} value={o.value} disabled={o.disabled}>{o.label}</option>
        ))}
      </select>
    );
  }
  const current = options.find((o) => o.value === value);
  return (
    <ListPicker id={id} options={options} selected={[value]} multiple={false} disabled={disabled}
      size={size} className={className} style={style} ariaLabel={ariaLabel}
      summary={current ? current.label : (placeholder ?? "Select…")} placeholderShown={!current}
      onPick={(v) => onChange(v)} />
  );
}

/** Many choices from a searchable checklist; the trigger summarises the pick. */
export function MultiSelect<T extends string>(
  { value, onChange, options, disabled, placeholder, size = "md", className, style, id,
    "aria-label": ariaLabel, maxSummary = 2 }: SelectBase<T> & {
    value: readonly T[]; onChange: (value: T[]) => void;
    /** Labels listed in the trigger before collapsing to "+n". */
    maxSummary?: number;
  },
) {
  const labels = options.filter((o) => value.includes(o.value)).map((o) => o.label);
  const summary = labels.length === 0 ? (placeholder ?? "None")
    : labels.length <= maxSummary ? labels.join(", ")
      : `${labels.slice(0, maxSummary).join(", ")} +${labels.length - maxSummary}`;
  return (
    <ListPicker id={id} options={options} selected={value} multiple disabled={disabled} size={size}
      className={className} style={style} ariaLabel={ariaLabel} summary={summary}
      placeholderShown={labels.length === 0}
      onPick={(v) => onChange(value.includes(v) ? value.filter((x) => x !== v) : [...value, v])}
      onAll={(all) => onChange(all ? options.filter((o) => !o.disabled).map((o) => o.value) : [])} />
  );
}

function ListPicker<T extends string>(
  { id, options, selected, multiple, disabled, size, className, style, ariaLabel, summary,
    placeholderShown, onPick, onAll }: {
    id?: string; options: SelectOption<T>[]; selected: readonly T[]; multiple: boolean;
    disabled?: boolean; size: "sm" | "md"; className?: string; style?: CSSProperties;
    ariaLabel?: string; summary: string; placeholderShown: boolean;
    onPick: (value: T) => void; onAll?: (all: boolean) => void;
  },
) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  useEffect(() => { if (!open) setQuery(""); }, [open]);
  const allOn = options.length > 0 && options.every((o) => o.disabled || selected.includes(o.value));
  /* The trigger is named "<label> <current value>" (aria-labelledby over the
     label and the value text), so the value is announced too: a native
     <label> or a plain aria-label would replace the button's text. The label
     is `aria-label` if given, else the surrounding Field's label. */
  const field = useField();
  const aria = useFieldAria();
  const uid = useId();
  const nameId = `${uid}-name`, valueId = `${uid}-value`;
  const labelledBy = ariaLabel != null ? `${nameId} ${valueId}` : field ? `${field.labelId} ${valueId}` : undefined;
  const panelName = ariaLabel ?? field?.labelText ?? "Options";
  return (
    <Popover open={open} onOpenChange={setOpen} align="start" className="ui-picker" label={panelName}
      trigger={
        <button id={id} type="button" disabled={disabled} aria-haspopup="listbox" aria-labelledby={labelledBy}
          {...aria}
          className={cx("ui-select", "ui-select--button", size === "sm" && "ui-select--sm", className)}
          style={style} data-placeholder={placeholderShown || undefined}>
          {ariaLabel != null && <span id={nameId} hidden>{ariaLabel}</span>}
          <span id={valueId} className="ui-select__value">{summary}</span>
          <Icon name="chevronDown" size={14} className="ui-select__chev" />
        </button>
      }>
      <Command className="ui-picker__cmd" loop label={panelName}>
        <div className="ui-picker__search">
          <Icon name="search" size={14} />
          <Command.Input value={query} onValueChange={setQuery} placeholder="Search…" autoFocus />
        </div>
        {multiple && onAll && (
          <div className="ui-picker__bulk">
            <button type="button" onClick={() => onAll(!allOn)}>{allOn ? "Clear all" : "Select all"}</button>
            <span className="muted">{selected.length} selected</span>
          </div>
        )}
        <Command.List className="ui-picker__list">
          <Command.Empty className="ui-picker__empty">No match</Command.Empty>
          {options.map((o) => {
            const on = selected.includes(o.value);
            return (
              <Command.Item key={o.value} value={o.value} keywords={[o.label, o.hint ?? ""]}
                disabled={o.disabled} data-on={on || undefined} className="ui-picker__item"
                onSelect={() => { onPick(o.value); if (!multiple) setOpen(false); }}>
                <span className="ui-picker__check" aria-hidden="true">
                  {on && <Icon name="check" size={13} />}
                </span>
                <span className="ui-picker__label">{o.label}</span>
                {o.hint && <span className="ui-picker__hint">{o.hint}</span>}
              </Command.Item>
            );
          })}
        </Command.List>
      </Command>
    </Popover>
  );
}

/* ─── Checkbox / Switch ───────────────────────────────────────────────────── */

export const Checkbox = forwardRef<HTMLInputElement, {
  checked: boolean; onChange: (checked: boolean) => void; children?: ReactNode;
  disabled?: boolean; indeterminate?: boolean; title?: string; className?: string;
  "aria-label"?: string; id?: string; name?: string;
  onClick?: (e: MouseEvent<HTMLInputElement>) => void;
}>(function Checkbox(
  { checked, onChange, children, disabled, indeterminate = false, title, className, id, name, onClick,
    "aria-label": ariaLabel },
  ref,
) {
  const inner = useRef<HTMLInputElement>(null);
  useEffect(() => { if (inner.current) inner.current.indeterminate = indeterminate; }, [indeterminate]);
  const box = (
    <input ref={composeRefs(ref, inner)} type="checkbox" id={id} name={name} checked={checked}
      disabled={disabled} aria-label={ariaLabel}
      onClick={onClick} onChange={(e) => onChange(e.target.checked)} />
  );
  if (children == null) {
    return <span className={cx("ui-check", "ui-check--bare", className)} title={title}>{box}</span>;
  }
  return (
    <label className={cx("ui-check", disabled && "is-disabled", className)} title={title}>
      {box}
      <span>{children}</span>
    </label>
  );
});

export const Switch = forwardRef<HTMLButtonElement, {
  checked: boolean; onChange: (checked: boolean) => void; children?: ReactNode;
  disabled?: boolean; size?: "sm" | "md"; "aria-label"?: string; title?: string; id?: string;
}>(function Switch({ checked, onChange, children, disabled, size = "md", title, id, "aria-label": ariaLabel }, ref) {
  const autoId = useId();
  const sw = (
    <RSwitch.Root ref={ref} id={id ?? autoId} checked={checked} onCheckedChange={onChange}
      disabled={disabled} aria-label={children == null ? ariaLabel : undefined}
      className={cx("ui-switch", size === "sm" && "ui-switch--sm")} title={title}>
      <RSwitch.Thumb className="ui-switch__thumb" />
    </RSwitch.Root>
  );
  if (children == null) return sw;
  return (
    <span className={cx("ui-switchrow", disabled && "is-disabled")}>
      {sw}
      <label htmlFor={id ?? autoId}>{children}</label>
    </span>
  );
});

/* ─── Segmented / Tabs ────────────────────────────────────────────────────── */

export type SegmentedOption<T extends string> = {
  value: T; label: ReactNode; disabled?: boolean; title?: string;
};

/** A small exclusive choice (radio-like toggle group with arrow-key focus). */
export function Segmented<T extends string>(
  { value, options, onChange, size = "md", "aria-label": ariaLabel, disabled, className }: {
    value: T; options: SegmentedOption<T>[]; onChange: (value: T) => void;
    size?: "sm" | "md"; "aria-label"?: string; disabled?: boolean; className?: string;
  },
) {
  return (
    <RToggle.Root type="single" value={value} disabled={disabled} aria-label={ariaLabel}
      className={cx("ui-seg", size === "sm" && "ui-seg--sm", className)}
      onValueChange={(v) => { if (v) onChange(v as T); }}>
      {options.map((o) => (
        <RToggle.Item key={o.value} value={o.value} disabled={o.disabled} title={o.title}
          data-on={o.value === value} className="ui-seg__item">
          {o.label}
        </RToggle.Item>
      ))}
    </RToggle.Root>
  );
}

export type TabItem<T extends string> = {
  id: T; label: ReactNode; disabled?: boolean;
  /** Count/status shown after the label. */
  badge?: ReactNode;
  /** Router path: the strip becomes navigation links (router-linked tabs). */
  to?: string;
};

/** Tab strip. With `onChange`, ARIA tabs (arrow keys, Home/End); pass the
 *  active panel as `children` to render it in a linked tabpanel. With `to`
 *  on the items, a <nav> of router links marking the current one. */
export function Tabs<T extends string>(
  { value, tabs, onChange, children, variant = "pill", "aria-label": ariaLabel, className }: {
    value: T; tabs: TabItem<T>[]; onChange?: (id: T) => void; children?: ReactNode;
    variant?: "pill" | "line"; "aria-label"?: string; className?: string;
  },
) {
  const cls = cx("ui-tabs", variant === "line" && "ui-tabs--line", className);
  if (tabs.some((t) => t.to != null)) {
    return (
      <>
        <nav className={cls} aria-label={ariaLabel ?? "Tabs"}>
          {tabs.map((t) => (
            t.disabled || t.to == null
              ? <span key={t.id} className="ui-tab" aria-disabled="true">{t.label}</span>
              : (
                <Link key={t.id} to={t.to} className="ui-tab" data-on={t.id === value}
                  aria-current={t.id === value ? "page" : undefined}
                  onClick={() => onChange?.(t.id)}>
                  {t.label}{t.badge != null && <span className="ui-tab__badge">{t.badge}</span>}
                </Link>
              )
          ))}
        </nav>
        {children}
      </>
    );
  }
  return (
    <RTabs.Root value={value} onValueChange={(v) => onChange?.(v as T)} activationMode="automatic">
      <RTabs.List className={cls} aria-label={ariaLabel}>
        {tabs.map((t) => (
          <RTabs.Trigger key={t.id} value={t.id} disabled={t.disabled} className="ui-tab"
            data-on={t.id === value}
            // Radix points every tab at its panel; with no panel rendered here
            // (the page draws the content itself) that reference would dangle.
            {...(children == null ? { "aria-controls": undefined } : {})}>
            {t.label}{t.badge != null && <span className="ui-tab__badge">{t.badge}</span>}
          </RTabs.Trigger>
        ))}
      </RTabs.List>
      {children != null && <RTabs.Content value={value} className="ui-tabs__panel">{children}</RTabs.Content>}
    </RTabs.Root>
  );
}

/* ─── Slider / RangeSlider ────────────────────────────────────────────────── */

type SliderCommon = {
  min: number; max: number; step?: number;
  /** "log" maps the track through log10 (min, max > 0; otherwise linear + a warning). */
  scale?: SliderScale;
  disabled?: boolean;
  "aria-label"?: string;
  /** Formats the value for aria-valuetext and the optional value label. */
  format?: (v: number) => string;
  /** Show the formatted value beside the track. */
  showValue?: boolean;
  className?: string;
  style?: CSSProperties;
};

function useTrack(o: { min: number; max: number; step?: number; scale?: SliderScale }) {
  return useMemo(() => {
    const log = effectiveScale(o) === "log";
    return {
      min: log ? 0 : o.min, max: log ? LOG_STEPS : o.max,
      step: log ? 1 : (o.step ?? ((o.max - o.min) / 100 || 1)),
      toPos: (v: number) => toSliderPos(v, o),
      fromPos: (p: number) => fromSliderPos(p, o),
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [o.min, o.max, o.step, o.scale]);
}

const defaultFormat = (v: number) => (Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-2)
  ? v.toExponential(2).replace("e+", "e") : String(Number(v.toPrecision(4))));

export function Slider(
  { value, onChange, onCommit, format = defaultFormat, showValue = false, disabled, className, style,
    "aria-label": ariaLabel, ...o }: SliderCommon & {
    value: number; onChange: (value: number) => void;
    /** Fired once when the drag / key press ends (expensive updates go here). */
    onCommit?: (value: number) => void;
  },
) {
  const t = useTrack(o);
  return (
    <span className={cx("ui-slider-wrap", className)} style={style}>
      <RSlider.Root className="ui-slider" min={t.min} max={t.max} step={t.step} disabled={disabled}
        value={[t.toPos(value)]}
        onValueChange={([p]) => onChange(t.fromPos(p))}
        onValueCommit={([p]) => onCommit?.(t.fromPos(p))}>
        <RSlider.Track className="ui-slider__track"><RSlider.Range className="ui-slider__range" /></RSlider.Track>
        <RSlider.Thumb className="ui-slider__thumb" aria-label={ariaLabel} aria-valuetext={format(value)} />
      </RSlider.Root>
      {showValue && <span className="ui-slider__value mono">{format(value)}</span>}
    </span>
  );
}

export function RangeSlider(
  { value, onChange, onCommit, format = defaultFormat, showValue = false, disabled, className, style,
    "aria-label": ariaLabel, minGap = 0, ...o }: SliderCommon & {
    value: [number, number]; onChange: (value: [number, number]) => void;
    onCommit?: (value: [number, number]) => void;
    /** Minimum distance between the thumbs, in track steps. */
    minGap?: number;
  },
) {
  const t = useTrack(o);
  const label = ariaLabel ?? "Range";
  const toVals = (ps: number[]) => [t.fromPos(ps[0]), t.fromPos(ps[1])] as [number, number];
  return (
    <span className={cx("ui-slider-wrap", className)} style={style}>
      {showValue && <span className="ui-slider__value mono">{format(value[0])}</span>}
      <RSlider.Root className="ui-slider" min={t.min} max={t.max} step={t.step} disabled={disabled}
        minStepsBetweenThumbs={minGap} value={[t.toPos(value[0]), t.toPos(value[1])]}
        onValueChange={(ps) => onChange(toVals(ps))}
        onValueCommit={(ps) => onCommit?.(toVals(ps))}>
        <RSlider.Track className="ui-slider__track"><RSlider.Range className="ui-slider__range" /></RSlider.Track>
        <RSlider.Thumb className="ui-slider__thumb" aria-label={`${label} minimum`} aria-valuetext={format(value[0])} />
        <RSlider.Thumb className="ui-slider__thumb" aria-label={`${label} maximum`} aria-valuetext={format(value[1])} />
      </RSlider.Root>
      {showValue && <span className="ui-slider__value mono">{format(value[1])}</span>}
    </span>
  );
}
