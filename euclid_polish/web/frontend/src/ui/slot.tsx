/* `asChild` support: render the single child element instead of our own tag,
   merging our props onto it (className concatenated, style merged, event
   handlers chained — the child's runs first — and refs composed). A tiny
   stand-in for @radix-ui/react-slot so the kit adds no dependency. */
import {
  Children, cloneElement, forwardRef, isValidElement,
  type HTMLAttributes, type ReactNode, type Ref,
} from "react";

type AnyProps = Record<string, unknown>;

export function composeRefs<T>(...refs: (Ref<T> | undefined)[]): (node: T | null) => void {
  return (node) => {
    for (const r of refs) {
      if (typeof r === "function") r(node);
      else if (r && typeof r === "object") (r as { current: T | null }).current = node;
    }
  };
}

export function mergeProps(ours: AnyProps, theirs: AnyProps): AnyProps {
  const out: AnyProps = { ...ours, ...theirs };
  for (const k of Object.keys(ours)) {
    const a = ours[k], b = theirs[k];
    if (/^on[A-Z]/.test(k) && typeof a === "function" && typeof b === "function") {
      out[k] = (...args: unknown[]) => {
        (b as (...x: unknown[]) => void)(...args);
        const ev = args[0] as { defaultPrevented?: boolean } | undefined;
        if (!ev?.defaultPrevented) (a as (...x: unknown[]) => void)(...args);
      };
    } else if (k === "className" && a && b) {
      out[k] = `${a as string} ${b as string}`;
    } else if (k === "style" && a && b) {
      out[k] = { ...(a as object), ...(b as object) };
    }
  }
  return out;
}

export const Slot = forwardRef<HTMLElement, HTMLAttributes<HTMLElement> & { children?: ReactNode }>(
  function Slot({ children, ...props }, ref) {
    const child = Children.only(children);
    if (!isValidElement(child)) return null;
    const childRef = (child as unknown as { ref?: Ref<HTMLElement> }).ref;
    return cloneElement(child, {
      ...mergeProps(props as AnyProps, child.props as AnyProps),
      ref: composeRefs(ref, childRef),
    } as AnyProps);
  },
);

/** "a b" from truthy class parts. */
export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}
